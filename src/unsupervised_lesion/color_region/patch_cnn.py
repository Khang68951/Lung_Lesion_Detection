from __future__ import annotations

import json
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn

from .student import (make_clean_region_crop, scale_rgb,
                                   select_training_rows, selective_regions)
from ..data import load_audit, validate_fit_rows
from ..io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256, within_project
from ..segmentation import LesionConfig, color_regions, unlabeled_lung_probability


VERSION = "v18_raw_rgb_color_region_patch_cnn"
SEED = 73


class PatchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        inputs = 3
        for outputs in (16, 32, 64, 128):
            layers.extend([nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                           nn.BatchNorm2d(outputs), nn.SiLU(), nn.MaxPool2d(2)])
            inputs = outputs
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(128, 32), nn.SiLU(), nn.Dropout(.15), nn.Linear(32, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(value)).squeeze(1)


def _tensor(crops: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(crops.transpose(0, 3, 1, 2).copy()).float() / 255.0


def _augment(value: torch.Tensor) -> torch.Tensor:
    if torch.rand(()) < .5:
        value = torch.flip(value, (3,))
    if torch.rand(()) < .25:
        value = torch.flip(value, (2,))
    brightness = .92 + .16 * torch.rand((len(value), 1, 1, 1), device=value.device)
    contrast = .92 + .16 * torch.rand((len(value), 1, 1, 1), device=value.device)
    mean = value.mean(dim=(2, 3), keepdim=True)
    return ((value - mean) * contrast + mean).mul(brightness).clamp(0, 1)


def _balanced(target: np.ndarray, predicted: np.ndarray) -> float:
    positive = target == 1; negative = ~positive
    return .5 * (float((predicted[positive] == 1).mean())
                 + float((predicted[negative] == 0).mean()))


def fit_patch_cnn(maximum_images: int = 128) -> dict:
    from .evaluate import load_candidate
    audit = load_audit(); rows = select_training_rows(audit["fit_rows"], maximum_images)
    validate_fit_rows(rows, audit)
    base = load_candidate()
    left = LesionConfig(**base["ensemble_left"]["config"])
    right = LesionConfig(**base["ensemble_right"]["config"])
    protocol = {"version": VERSION, "seed": SEED, "maximum_images": maximum_images,
                "maximum_side": 768, "region_segments": 96, "maximum_per_class_per_image": 12,
                "positive_support": .85, "negative_support": .02, "crop_size": 96,
                "crop_mode": "clean_context", "epochs": 30,
                "source_candidate_fingerprint": base["candidate_fingerprint"],
                "audit_fingerprint": audit["audit_fingerprint"],
                "marked_images_used": 0, "reference_pixels_used_for_fit": 0}
    fingerprint = json_hash(protocol); folder = ARTIFACTS / "color_patch_cnn" / fingerprint
    cache = folder / "patch_cache"; cache.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "protocol.json", protocol)
    for index, row in enumerate(rows, start=1):
        array_path = cache / f"{row['id']}.npz"; record_path = cache / f"{row['id']}.json"
        if array_path.is_file() and record_path.is_file():
            print(f"v18 patch cache: {index}/{len(rows)} cached", flush=True); continue
        rgb_native = np.asarray(ImageOps.exif_transpose(
            Image.open(within_project(row["path"]))).convert("RGB"))
        rgb, scale = scale_rgb(rgb_native, protocol["maximum_side"])
        lung = unlabeled_lung_probability(rgb) >= base["foreground_alpha_threshold"]
        labels = color_regions(rgb, lung, protocol["region_segments"])
        selected = selective_regions(rgb, lung, labels, left, right,
                                     protocol["maximum_per_class_per_image"])
        crops = np.stack([np.asarray(make_clean_region_crop(rgb, item[0]).resize(
            (protocol["crop_size"], protocol["crop_size"]), Image.Resampling.LANCZOS), np.uint8)
                          for item in selected])
        targets = np.asarray([item[1] for item in selected], np.int8)
        np.savez_compressed(array_path, crops=crops, targets=targets,
                            supports=np.asarray([item[2] for item in selected], np.float32))
        atomic_json(record_path, {"id": row["id"],
                    "source_group": row.get("source_group") or f"ungrouped:{row['id']}",
                    "pixel_sha256": row["pixel_sha256"], "file_sha256": row["file_sha256"],
                    "scale": scale, "samples": len(selected), "positive": int(targets.sum()),
                    "negative": int((targets == 0).sum()), "protocol_fingerprint": fingerprint,
                    "reference_pixels_used": 0})
        print(f"v18 patch cache: {index}/{len(rows)} samples={len(selected)}", flush=True)
    arrays, records = [], []
    for row in rows:
        record = json.loads((cache / f"{row['id']}.json").read_text())
        if record["protocol_fingerprint"] != fingerprint or record["reference_pixels_used"]:
            raise ValueError("Invalid v18 cache provenance")
        with np.load(cache / f"{row['id']}.npz", allow_pickle=False) as values:
            arrays.append({name: values[name] for name in values.files})
        records.append(record)
    crops = np.concatenate([item["crops"] for item in arrays])
    target = np.concatenate([item["targets"] for item in arrays]).astype(np.float32)
    sample_groups = np.concatenate([np.repeat(record["source_group"], record["samples"])
                                    for record in records])
    groups = sorted(set(sample_groups.tolist())); rng = random.Random(SEED); rng.shuffle(groups)
    validation_groups = set(groups[:max(1, round(len(groups) * .20))])
    validation = np.isin(sample_groups, list(validation_groups)); training = ~validation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED); np.random.seed(SEED)
    network = PatchCNN().to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=5e-4, weight_decay=1e-4)
    positives = float(target[training].sum()); negatives = float(training.sum() - positives)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / max(positives, 1), device=device))
    train_x = _tensor(crops[training]); train_y = torch.from_numpy(target[training]).float()
    val_x = _tensor(crops[validation]); val_y = target[validation].astype(np.int8)
    generator = torch.Generator().manual_seed(SEED); history = []; best = -1.0
    for epoch in range(1, protocol["epochs"] + 1):
        network.train(); order = torch.randperm(len(train_x), generator=generator); losses = []
        for start in range(0, len(order), 128):
            indices = order[start:start + 128]
            batch_x = _augment(train_x[indices].to(device)); batch_y = train_y[indices].to(device)
            optimizer.zero_grad(set_to_none=True); loss = criterion(network(batch_x), batch_y)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        network.eval(); probabilities = []
        with torch.inference_mode():
            for start in range(0, len(val_x), 256):
                probabilities.append(network(val_x[start:start + 256].to(device)).sigmoid().cpu().numpy())
        probability = np.concatenate(probabilities); predicted = (probability >= .5).astype(np.int8)
        balanced = _balanced(val_y, predicted)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "pseudo_validation_balanced_accuracy": balanced,
                        "pseudo_validation_accuracy": float((predicted == val_y).mean())})
        payload = {"version": VERSION, "epoch": epoch, "model": network.state_dict(),
                   "protocol": protocol, "protocol_fingerprint": fingerprint, "history": history}
        torch.save(payload, folder / "last.pt")
        if balanced > best:
            best = balanced; torch.save(payload, folder / "best.pt")
        atomic_json(folder / "history.json", history)
        print(f"v18 epoch {epoch}/{protocol['epochs']} pseudo_balanced={balanced:.4f}", flush=True)
    checkpoint = folder / "best.pt"
    artifact = {"status": "fit_complete", "created_at": now(), "version": VERSION,
                "artifact_fingerprint": fingerprint,
                "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
                "checkpoint_sha256": sha256(checkpoint), "fit_images": len(rows),
                "fit_regions": len(target), "fit_positive": int(target.sum()),
                "fit_negative": int((target == 0).sum()),
                "training_groups": sorted(set(groups) - validation_groups),
                "pseudo_validation_groups": sorted(validation_groups),
                "best_pseudo_validation_balanced_accuracy": best,
                "audit_fingerprint": audit["audit_fingerprint"],
                "marked_images_used": 0, "reference_pixels_used_for_fit": 0}
    atomic_json(folder / "artifact.json", artifact)
    atomic_json(ARTIFACTS / "color_patch_cnn/current.json",
                {"path": str((folder / "artifact.json").relative_to(PROJECT_ROOT)),
                 "sha256": sha256(folder / "artifact.json"), "artifact_fingerprint": fingerprint})
    return artifact


def load_patch_cnn() -> tuple[dict, dict]:
    pointer = json.loads((ARTIFACTS / "color_patch_cnn/current.json").read_text())
    path = within_project(pointer["path"])
    if sha256(path) != pointer["sha256"]:
        raise ValueError("Patch CNN artifact changed")
    artifact = json.loads(path.read_text()); checkpoint_path = within_project(artifact["checkpoint"])
    if sha256(checkpoint_path) != artifact["checkpoint_sha256"]:
        raise ValueError("Patch CNN checkpoint changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return artifact, checkpoint


def predict_regions(network: PatchCNN, rgb: np.ndarray, lung: np.ndarray,
                    labels: np.ndarray, crop_size: int = 96) -> tuple[list[int], list[float]]:
    from .teacher import region_statistics
    regions = region_statistics(rgb, lung, labels)
    crops = np.stack([np.asarray(make_clean_region_crop(rgb, region).resize(
        (crop_size, crop_size), Image.Resampling.LANCZOS), np.uint8) for region in regions])
    tensor = _tensor(crops); device = next(network.parameters()).device; output = []
    network.eval()
    with torch.inference_mode():
        for start in range(0, len(tensor), 256):
            output.append(network(tensor[start:start + 256].to(device)).sigmoid().cpu().numpy())
    return [region["region_id"] for region in regions], np.concatenate(output).astype(float).tolist()
