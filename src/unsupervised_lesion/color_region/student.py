from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn

from .teacher import make_region_crop, region_statistics
from ..data import load_audit, validate_fit_rows
from ..io import (ARTIFACTS, DINO_ASSET_ROOT, PROJECT_ROOT, atomic_json,
                  json_hash, now, sha256, within_project)
from ..segmentation import (LesionConfig, color_regions, lesion_score,
                           robust_features, unlabeled_lung_probability)


VERSION = "v16_selective_color_region_dino_student"
SEED = 73
HANDCRAFTED_NAMES = ("mean_l", "mean_a", "mean_b", "std_l", "std_a", "std_b",
                     "log_area_fraction", "center_x", "center_y", "width", "height")


def select_training_rows(rows: list[dict], maximum_images: int, seed: int = SEED) -> list[dict]:
    if maximum_images < 1:
        raise ValueError("maximum_images must be positive")
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row.get("source_group") or f"ungrouped:{row['id']}", []).append(row)
    rng = random.Random(seed)
    keys = sorted(grouped); rng.shuffle(keys)
    for values in grouped.values():
        values.sort(key=lambda item: (item["pixel_sha256"], item["id"]))
    selected = []
    offset = 0
    while len(selected) < maximum_images:
        added = False
        for key in keys:
            if offset < len(grouped[key]):
                selected.append(grouped[key][offset]); added = True
                if len(selected) == maximum_images:
                    break
        if not added:
            break
        offset += 1
    return selected


def scale_rgb(rgb: np.ndarray, maximum: int = 768) -> tuple[np.ndarray, float]:
    scale = min(1.0, maximum / max(rgb.shape[:2]))
    if scale == 1.0:
        return np.asarray(rgb, np.uint8), scale
    return cv2.resize(np.asarray(rgb, np.uint8),
                      (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)),
                      interpolation=cv2.INTER_AREA), scale


def handcrafted(region: dict, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    x0, y0, x1, y1 = region["box"]
    return np.asarray([
        *region["mean_lab"], *region["std_lab"], np.log(max(region["area_fraction"], 1e-8)),
        ((x0 + x1) / 2) / max(width, 1), ((y0 + y1) / 2) / max(height, 1),
        (x1 - x0) / max(width, 1), (y1 - y0) / max(height, 1),
    ], np.float32)


def make_clean_region_crop(rgb: np.ndarray, region: dict,
                           padding_fraction: float = 0.55) -> Image.Image:
    x0, y0, x1, y1 = region["box"]
    padding = max(8, round(max(x1 - x0, y1 - y0) * padding_fraction))
    left, top = max(0, x0 - padding), max(0, y0 - padding)
    right, bottom = min(rgb.shape[1], x1 + padding), min(rgb.shape[0], y1 + padding)
    return Image.fromarray(np.asarray(rgb[top:bottom, left:right], dtype=np.uint8))


def selective_regions(rgb: np.ndarray, lung: np.ndarray, labels: np.ndarray,
                      left: LesionConfig, right: LesionConfig,
                      maximum_per_class: int = 12) -> list[tuple[dict, int, float]]:
    features = robust_features(rgb, lung)
    raw = ((lesion_score(features, left) >= left.threshold)
           & (features["dark"] >= left.minimum_dark)
           & (lesion_score(features, right) >= right.threshold)
           & (features["dark"] >= right.minimum_dark) & lung)
    maximum = int(labels.max())
    totals = np.bincount(labels[lung], minlength=maximum + 1)
    positives = np.bincount(labels[raw], minlength=maximum + 1)
    support = positives / np.maximum(totals, 1)
    stats = region_statistics(rgb, lung, labels)
    positive = [(item, 1, float(support[item["region_id"]])) for item in stats
                if support[item["region_id"]] >= 0.85]
    negative = [(item, 0, float(support[item["region_id"]])) for item in stats
                if support[item["region_id"]] <= 0.02]
    positive.sort(key=lambda value: (-value[2], -value[0]["area"], value[0]["region_id"]))
    negative.sort(key=lambda value: (value[2], -value[0]["area"], value[0]["region_id"]))
    return positive[:maximum_per_class] + negative[:maximum_per_class]


def _dino_provenance() -> tuple[dict, Path]:
    provenance_path = DINO_ASSET_ROOT / "provenance.json"
    record = json.loads(provenance_path.read_text(encoding="utf-8"))
    if record.get("repository") != "facebook/dinov2-with-registers-small":
        raise ValueError("Unexpected DINO repository")
    directory = Path(record["local_directory"])
    if not directory.is_dir():
        directory = (DINO_ASSET_ROOT / "cache"
                     / "models--facebook--dinov2-with-registers-small/snapshots" / record["revision"])
    directory = directory.resolve(); directory.relative_to(PROJECT_ROOT.resolve())
    for name, digest in record["files_sha256"].items():
        candidate = directory / name
        if not candidate.is_file() or sha256(candidate) != digest:
            raise ValueError(f"Pinned DINO artifact changed: {name}")
    return record, directory


class DinoCropEncoder:
    def __init__(self, directory: Path):
        from transformers import Dinov2WithRegistersModel
        if not torch.cuda.is_available():
            raise RuntimeError("DINO crop encoding requires CUDA")
        self.model = Dinov2WithRegistersModel.from_pretrained(
            directory, local_files_only=True, use_safetensors=True,
            attn_implementation="sdpa").eval().to("cuda")
        self.model.requires_grad_(False)
        self.registers = int(self.model.config.num_register_tokens)
        self.mean = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None]

    @torch.inference_mode()
    def encode(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        output = []
        for start in range(0, len(images), batch_size):
            arrays = []
            for image in images[start:start + batch_size]:
                resized = image.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
                arrays.append(np.asarray(resized, np.uint8).transpose(2, 0, 1))
            tensor = torch.from_numpy(np.stack(arrays).copy()).float().to("cuda") / 255.0
            hidden = self.model(pixel_values=(tensor - self.mean) / self.std,
                                return_dict=True).last_hidden_state
            cls = hidden[:, 0]
            patches = hidden[:, 1 + self.registers:].mean(dim=1)
            output.append(torch.cat([cls, patches], dim=1).float().cpu().numpy())
        return np.concatenate(output).astype(np.float32) if output else np.empty((0, 768), np.float32)

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()


class RegionMLP(nn.Module):
    def __init__(self, inputs: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, 128), nn.SiLU(), nn.Dropout(.20),
                                 nn.Linear(128, 32), nn.SiLU(), nn.Dropout(.10),
                                 nn.Linear(32, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(1)


def _balanced_accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
    positive = target == 1; negative = ~positive
    recall = float((prediction[positive] == 1).mean()) if positive.any() else 0.0
    specificity = float((prediction[negative] == 0).mean()) if negative.any() else 0.0
    return (recall + specificity) / 2


def fit_student(maximum_images: int = 128, crop_mode: str = "dimmed") -> dict:
    from .evaluate import load_candidate
    audit = load_audit()
    rows = select_training_rows(audit["fit_rows"], maximum_images)
    validate_fit_rows(rows, audit)
    base = load_candidate()
    left = LesionConfig(**base["ensemble_left"]["config"])
    right = LesionConfig(**base["ensemble_right"]["config"])
    dino_record, dino_directory = _dino_provenance()
    if crop_mode not in {"dimmed", "clean_context"}:
        raise ValueError("Unknown region crop mode")
    version = VERSION if crop_mode == "dimmed" else "v17_clean_color_region_dino_student"
    protocol = {
        "version": version, "seed": SEED, "maximum_images": maximum_images,
        "maximum_side": 768, "region_segments": 96, "maximum_per_class_per_image": 12,
        "crop_mode": crop_mode, "crop_padding_fraction": 0.55,
        "positive_support": 0.85, "negative_support": 0.02,
        "source_candidate_fingerprint": base["candidate_fingerprint"],
        "audit_fingerprint": audit["audit_fingerprint"], "dino_revision": dino_record["revision"],
        "epochs": 25, "marked_images_used": 0, "reference_pixels_used_for_fit": 0,
    }
    fingerprint = json_hash(protocol)
    folder = ARTIFACTS / "color_region_student" / fingerprint
    cache = folder / "embedding_cache"; cache.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "protocol.json", protocol)
    encoder = None
    try:
        for index, row in enumerate(rows, start=1):
            array_path = cache / f"{row['id']}.npz"
            record_path = cache / f"{row['id']}.json"
            if array_path.is_file() and record_path.is_file():
                print(f"v16 embedding cache: {index}/{len(rows)} cached", flush=True); continue
            if encoder is None:
                encoder = DinoCropEncoder(dino_directory)
            rgb_native = np.asarray(ImageOps.exif_transpose(
                Image.open(within_project(row["path"]))).convert("RGB"))
            rgb, scale = scale_rgb(rgb_native, protocol["maximum_side"])
            lung = unlabeled_lung_probability(rgb) >= base["foreground_alpha_threshold"]
            labels = color_regions(rgb, lung, protocol["region_segments"])
            selected = selective_regions(rgb, lung, labels, left, right,
                                         protocol["maximum_per_class_per_image"])
            crops = [(make_region_crop(rgb, labels, item[0]) if crop_mode == "dimmed"
                      else make_clean_region_crop(rgb, item[0])) for item in selected]
            embeddings = encoder.encode(crops)
            features = np.stack([handcrafted(item[0], rgb.shape[:2]) for item in selected])
            targets = np.asarray([item[1] for item in selected], np.int8)
            supports = np.asarray([item[2] for item in selected], np.float32)
            region_ids = np.asarray([item[0]["region_id"] for item in selected], np.int32)
            np.savez_compressed(array_path, embeddings=embeddings.astype(np.float16),
                                handcrafted=features, targets=targets, supports=supports,
                                region_ids=region_ids)
            atomic_json(record_path, {"id": row["id"],
                        "source_group": row.get("source_group") or f"ungrouped:{row['id']}",
                        "pixel_sha256": row["pixel_sha256"], "file_sha256": row["file_sha256"],
                        "scale": scale, "samples": len(selected),
                        "positive": int(targets.sum()), "negative": int((targets == 0).sum()),
                        "protocol_fingerprint": fingerprint, "reference_pixels_used": 0})
            print(f"v16 embedding cache: {index}/{len(rows)} samples={len(selected)}", flush=True)
    finally:
        if encoder is not None:
            encoder.close()

    arrays, records = [], []
    for row in rows:
        record = json.loads((cache / f"{row['id']}.json").read_text())
        record["source_group"] = record.get("source_group") or f"ungrouped:{record['id']}"
        if record["protocol_fingerprint"] != fingerprint or record["reference_pixels_used"] != 0:
            raise ValueError("Invalid v16 embedding cache provenance")
        with np.load(cache / f"{row['id']}.npz", allow_pickle=False) as values:
            arrays.append({name: values[name] for name in values.files})
        records.append(record)
    x = np.concatenate([np.column_stack((item["embeddings"].astype(np.float32), item["handcrafted"]))
                        for item in arrays])
    y = np.concatenate([item["targets"] for item in arrays]).astype(np.float32)
    sample_groups = np.concatenate([np.repeat(record["source_group"], record["samples"])
                                    for record in records])
    groups = sorted(set(sample_groups.tolist())); rng = random.Random(SEED); rng.shuffle(groups)
    validation_groups = set(groups[:max(1, round(len(groups) * .20))])
    validation = np.isin(sample_groups, list(validation_groups)); training = ~validation
    if len(np.unique(y[training])) < 2 or len(np.unique(y[validation])) < 2:
        raise ValueError("Both pseudo classes required in train and validation")
    mean = x[training].mean(axis=0); std = np.maximum(x[training].std(axis=0), 1e-6)
    x = (x - mean) / std
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = RegionMLP(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-4)
    positives = float(y[training].sum()); negatives = float(training.sum() - positives)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / max(positives, 1), device=device))
    train_x = torch.from_numpy(x[training]).float(); train_y = torch.from_numpy(y[training]).float()
    val_x = torch.from_numpy(x[validation]).float().to(device); val_y = y[validation].astype(np.int8)
    generator = torch.Generator().manual_seed(SEED)
    history, best = [], -1.0
    checkpoint_path = folder / "best.pt"
    for epoch in range(1, protocol["epochs"] + 1):
        network.train(); order = torch.randperm(len(train_x), generator=generator)
        losses = []
        for start in range(0, len(order), 256):
            indices = order[start:start + 256]
            batch_x = train_x[indices].to(device); batch_y = train_y[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(network(batch_x), batch_y); loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        network.eval()
        with torch.inference_mode():
            probability = network(val_x).sigmoid().cpu().numpy()
        prediction = (probability >= .5).astype(np.int8)
        balanced = _balanced_accuracy(val_y, prediction)
        epoch_row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                     "pseudo_validation_balanced_accuracy": balanced,
                     "pseudo_validation_accuracy": float((prediction == val_y).mean())}
        history.append(epoch_row)
        payload = {"version": version, "epoch": epoch, "inputs": x.shape[1],
                   "model": network.state_dict(), "mean": mean, "std": std,
                   "protocol": protocol, "protocol_fingerprint": fingerprint,
                   "history": history, "handcrafted_names": HANDCRAFTED_NAMES}
        torch.save(payload, folder / "last.pt")
        if balanced > best:
            best = balanced; torch.save(payload, checkpoint_path)
        atomic_json(folder / "history.json", history)
        print(f"v16 epoch {epoch}/{protocol['epochs']} pseudo_balanced={balanced:.4f}", flush=True)
    artifact = {"status": "fit_complete", "created_at": now(), "version": version,
                "artifact_fingerprint": fingerprint, "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
                "checkpoint_sha256": sha256(checkpoint_path), "fit_images": len(rows),
                "fit_regions": len(y), "fit_positive": int(y.sum()), "fit_negative": int((y == 0).sum()),
                "training_groups": sorted(set(groups) - validation_groups),
                "pseudo_validation_groups": sorted(validation_groups),
                "best_pseudo_validation_balanced_accuracy": best,
                "audit_fingerprint": audit["audit_fingerprint"],
                "marked_images_used": 0, "reference_pixels_used_for_fit": 0,
                "dino_repository": dino_record["repository"], "dino_revision": dino_record["revision"]}
    atomic_json(folder / "artifact.json", artifact)
    atomic_json(ARTIFACTS / "color_region_student/current.json",
                {"path": str((folder / "artifact.json").relative_to(PROJECT_ROOT)),
                 "sha256": sha256(folder / "artifact.json"), "artifact_fingerprint": fingerprint})
    return artifact


def load_student() -> tuple[dict, dict]:
    pointer = json.loads((ARTIFACTS / "color_region_student/current.json").read_text())
    artifact_path = within_project(pointer["path"])
    if sha256(artifact_path) != pointer["sha256"]:
        raise ValueError("Student artifact changed")
    artifact = json.loads(artifact_path.read_text())
    checkpoint_path = within_project(artifact["checkpoint"])
    if sha256(checkpoint_path) != artifact["checkpoint_sha256"]:
        raise ValueError("Student checkpoint changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["protocol_fingerprint"] != artifact["artifact_fingerprint"]:
        raise ValueError("Student protocol mismatch")
    return artifact, checkpoint


def student_region_probabilities(encoder: DinoCropEncoder, network: RegionMLP,
                                 checkpoint: dict, rgb: np.ndarray, lung: np.ndarray,
                                 labels: np.ndarray) -> tuple[list[int], list[float]]:
    regions = region_statistics(rgb, lung, labels)
    crop_mode = checkpoint["protocol"].get("crop_mode", "dimmed")
    crops = [(make_region_crop(rgb, labels, region) if crop_mode == "dimmed"
              else make_clean_region_crop(rgb, region)) for region in regions]
    embedding = encoder.encode(crops)
    color_shape = np.stack([handcrafted(region, rgb.shape[:2]) for region in regions])
    x = np.column_stack((embedding, color_shape))
    x = (x - checkpoint["mean"]) / checkpoint["std"]
    device = next(network.parameters()).device
    with torch.inference_mode():
        probability = network(torch.from_numpy(x).float().to(device)).sigmoid().cpu().numpy()
    return [item["region_id"] for item in regions], probability.astype(float).tolist()
