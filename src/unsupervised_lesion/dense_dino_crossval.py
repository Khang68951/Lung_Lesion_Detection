from __future__ import annotations

import json
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .binary_crossval import _loss
from .color_region.student import _dino_provenance
from .crossval_segformer import _score_native, development_folds
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256
from .reference_v2 import DevelopmentReferenceStore
from .segformer_segmentation import _rank
from .segmentation import unlabeled_lung_probability
from .supervised_segmentation import _letterbox, _prepare, _restore


VERSION = "v40_frozen_dense_dinov2_colour_decoder_reference_v2"
SEED = 4004
EPOCHS = 60


def _colour_channels(rgb: np.ndarray) -> np.ndarray:
    small = cv2.resize(rgb, (96, 96), interpolation=cv2.INTER_AREA)
    rgb_value = small.astype(np.float32) / 255.0
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] /= 255.0
    lab[:, :, 1:] = (lab[:, :, 1:] - 128.0) / 128.0
    return np.concatenate([rgb_value, lab], axis=2).transpose(2, 0, 1).copy()


def patch_grid(hidden: torch.Tensor, registers: int) -> torch.Tensor:
    patches = hidden[:, 1 + registers:]
    side = int(round(patches.shape[1] ** .5))
    if side * side != patches.shape[1]:
        raise ValueError(f"DINO patch token count is not square: {patches.shape[1]}")
    patches = F.normalize(patches.float(), dim=-1)
    return patches.transpose(1, 2).reshape(patches.shape[0], patches.shape[2], side, side)


class DenseColourDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(384, 64, 1), nn.GroupNorm(8, 64), nn.GELU())
        self.decode = nn.Sequential(
            nn.Conv2d(70, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 1, 1))

    def forward(self, feature: torch.Tensor, colour: torch.Tensor) -> torch.Tensor:
        semantic = F.interpolate(self.project(feature), size=(96, 96), mode="bilinear",
                                 align_corners=False)
        return self.decode(torch.cat([semantic, colour], dim=1))


def _feature_cache(store: DevelopmentReferenceStore, items: list[dict]) -> tuple[dict, object]:
    provenance, directory = _dino_provenance()
    protocol = {
        "version": VERSION, "encoder": provenance["repository"],
        "encoder_revision": provenance["revision"], "encoder_files_sha256": provenance["files_sha256"],
        "reference_fingerprint": store.manifest["fingerprint"], "input_size": 448,
        "hidden_layers": [-4, -3, -2, -1], "hidden_reduction": "mean",
        "token_policy": "discard_cls_and_registers_l2_normalize_patches",
        "labels_used_during_encoding": 0, "sealed_test_images_opened": 0,
        "source_images": [{"id": row["id"], "image_sha256": row["sha256"]["image"]}
                          for row in store.rows],
    }
    fingerprint = json_hash(protocol); root = ARTIFACTS / "dense_dino_features" / fingerprint
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / "protocol.json", protocol)
    expected = {item["id"] for item in items}
    complete = {path.stem for path in root.glob("*.npz")}
    if complete != expected:
        from transformers import Dinov2WithRegistersModel
        if not torch.cuda.is_available():
            raise RuntimeError("V40 DINO feature extraction requires CUDA")
        model = Dinov2WithRegistersModel.from_pretrained(
            directory, local_files_only=True, use_safetensors=True,
            attn_implementation="sdpa").eval().to("cuda")
        model.requires_grad_(False)
        mean = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None]
        std = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None]
        by_id = {item["id"]: item for item in items}
        with torch.inference_mode():
            for identifier in sorted(expected - complete):
                rgb448, _ = _letterbox_size(by_id[identifier]["payload"]["rgb"], 448)
                value = torch.from_numpy(rgb448.transpose(2, 0, 1).copy())[None].float().to("cuda") / 255
                with torch.autocast("cuda"):
                    output = model(pixel_values=(value - mean) / std,
                                   output_hidden_states=True, return_dict=True)
                    hidden = torch.stack(output.hidden_states[-4:]).mean(0)
                grid = patch_grid(hidden, int(model.config.num_register_tokens))[0]
                np.savez_compressed(root / f"{identifier}.npz", feature=grid.cpu().numpy().astype(np.float16))
        del model
        torch.cuda.empty_cache()
    files = [{"id": item["id"], "path": str((root / f"{item['id']}.npz").relative_to(PROJECT_ROOT)),
              "sha256": sha256(root / f"{item['id']}.npz")} for item in items]
    artifact = {"status": "done", "created_at": now(), "version": VERSION,
                "feature_fingerprint": fingerprint, "images": len(files),
                "labels_used_during_encoding": 0, "test_payload_opened": False, "files": files}
    atomic_json(root / "artifact.json", artifact)
    return artifact, root


def _letterbox_size(rgb: np.ndarray, size: int) -> tuple[np.ndarray, tuple]:
    height, width = rgb.shape[:2]; scale = min(size / height, size / width)
    resized_height, resized_width = max(1, round(height * scale)), max(1, round(width * scale))
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), np.uint8)
    top, left = (size - resized_height) // 2, (size - resized_width) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas, (height, width, resized_height, resized_width, top, left)


def _augment(item: dict, rng: random.Random) -> tuple[torch.Tensor, ...]:
    feature = item["feature"].copy(); colour = item["colour"].copy()
    target = item["lesion"].astype(np.float32).copy()
    valid = (item["lung"] & item["valid"] & ~item["ignore_lung"] &
             ~item["ignore_lesion"]).astype(np.float32)
    if rng.random() < .5:
        feature = np.flip(feature, 2).copy(); colour = np.flip(colour, 2).copy()
        target = np.flip(target, 1).copy(); valid = np.flip(valid, 1).copy()
    if rng.random() < .5:
        feature = np.flip(feature, 1).copy(); colour = np.flip(colour, 1).copy()
        target = np.flip(target, 0).copy(); valid = np.flip(valid, 0).copy()
    turns = rng.randrange(4)
    if turns:
        feature = np.rot90(feature, turns, axes=(1, 2)).copy()
        colour = np.rot90(colour, turns, axes=(1, 2)).copy()
        target = np.rot90(target, turns).copy(); valid = np.rot90(valid, turns).copy()
    return (torch.from_numpy(feature).float(), torch.from_numpy(colour).float(),
            torch.from_numpy(target), torch.from_numpy(valid))


def _predict(model: nn.Module, items: list[dict], device: torch.device) -> list[np.ndarray]:
    model.eval(); result = []
    with torch.inference_mode():
        for item in items:
            feature = torch.from_numpy(item["feature"])[None].float().to(device)
            colour = torch.from_numpy(item["colour"])[None].float().to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(feature, colour)
            probability = F.interpolate(logits, size=item["rgb"].shape[:2], mode="bilinear",
                                        align_corners=False).sigmoid()[0, 0]
            result.append(_restore(probability.float().cpu().numpy(), item["geometry"]))
    return result


def fit_dense_dino_crossval(epochs: int = EPOCHS) -> dict:
    if epochs != EPOCHS:
        raise ValueError(f"V40 predeclared exactly {EPOCHS} epochs; received {epochs}")
    store = DevelopmentReferenceStore(); metadata = store.rows
    items = [_prepare(store.load(row)) for row in metadata]
    feature_artifact, feature_root = _feature_cache(store, items)
    for item in items:
        with np.load(feature_root / f"{item['id']}.npz", allow_pickle=False) as values:
            item["feature"] = values["feature"].astype(np.float32)
        item["colour"] = _colour_channels(item["rgb"])
    folds = development_folds(metadata)
    lung_probabilities = {item["id"]: unlabeled_lung_probability(item["payload"]["rgb"])
                          for item in items}
    protocol = {
        "version": VERSION, "seed": SEED, "epochs_per_fold": EPOCHS, "batch_size": 2,
        "feature_fingerprint": feature_artifact["feature_fingerprint"],
        "reference_fingerprint": store.manifest["fingerprint"], "fold_groups": folds,
        "decoder": "dino384_to64_plus_rgb_lab6_to64_to32_to1_at96",
        "optimizer": {"name": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-3},
        "scheduler": "cosine_to_1e-6", "augmentation": "paired_hv_flip_rot90",
        "loss_domain": "valid_reference_lung_only", "encoder_trainable": False,
        "development_images": len(items), "development_groups": len({x["group"] for x in items}),
        "epoch_lesion_thresholds": [.25, .40, .55, .70],
        "oof_lung_thresholds": [round(value, 2) for value in np.arange(.50, .951, .05)],
        "oof_lesion_thresholds": [round(value, 3) for value in np.arange(.10, .901, .025)],
        "sealed_test_images_opened": 0,
    }
    fingerprint = json_hash(protocol); root = ARTIFACTS / "dense_dino_crossval" / fingerprint
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / "protocol.json", protocol)
    oof = root / "oof"; oof.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); artifacts = []
    for fold_index, validation_groups in enumerate(folds):
        training = [item for item in items if item["group"] not in validation_groups]
        validation = [item for item in items if item["group"] in validation_groups]
        folder = root / f"fold_{fold_index}"; folder.mkdir(exist_ok=True)
        torch.manual_seed(SEED + fold_index); np.random.seed(SEED + fold_index)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED + fold_index)
        model = DenseColourDecoder().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        rng = random.Random(SEED + fold_index); history = []; best_rank = (-1., -1., -1., -999.)
        for epoch in range(1, EPOCHS + 1):
            model.train(); order = list(range(len(training))); rng.shuffle(order); losses = []
            for start in range(0, len(order), 2):
                batch = [_augment(training[index], rng) for index in order[start:start + 2]]
                feature = torch.stack([x[0] for x in batch]).to(device)
                colour = torch.stack([x[1] for x in batch]).to(device)
                target = torch.stack([x[2] for x in batch]).to(device)
                valid = torch.stack([x[3] for x in batch]).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(feature, colour)
                    loss, components = _loss(logits, target, valid)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
            scheduler.step(); probabilities = _predict(model, validation, device); choices = []
            for threshold in protocol["epoch_lesion_thresholds"]:
                summary, _ = _score_native(validation, probabilities,
                    [lung_probabilities[x["id"]] for x in validation], .50, threshold)
                choices.append({"lesion_threshold": threshold, "summary": summary})
            winner = max(choices, key=lambda x: _rank(x["summary"])); rank = _rank(winner["summary"])
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "winner": winner,
                            "last_batch_components": components,
                            "learning_rate": optimizer.param_groups[0]["lr"]})
            checkpoint = {"version": VERSION, "fold": fold_index, "epoch": epoch,
                          "model": model.state_dict(), "protocol_fingerprint": fingerprint,
                          "winner": winner, "rank": rank}
            torch.save(checkpoint, folder / "last.pt")
            if rank > best_rank:
                best_rank = rank; torch.save(checkpoint, folder / "best.pt")
            atomic_json(folder / "history.json", history)
            print(f"{VERSION} fold {fold_index + 1}/4 epoch {epoch}/{EPOCHS} "
                  f"loss={np.mean(losses):.4f} floor={rank[0]:.4f} dice={rank[1]:.4f}", flush=True)
        path = folder / "best.pt"; best = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(best["model"]); probabilities = _predict(model, validation, device)
        for item, probability in zip(validation, probabilities):
            np.savez_compressed(oof / f"{item['id']}.npz", lesion=probability,
                                lung=lung_probabilities[item["id"]], fold=fold_index)
        artifact = {"fold": fold_index,
                    "training_groups": sorted({x["group"] for x in training}),
                    "validation_groups": validation_groups, "best_epoch": best["epoch"],
                    "checkpoint": str(path.relative_to(PROJECT_ROOT)),
                    "checkpoint_sha256": sha256(path)}
        artifacts.append(artifact); atomic_json(folder / "artifact.json", artifact)
    probabilities = []; lungs = []
    for item in items:
        with np.load(oof / f"{item['id']}.npz", allow_pickle=False) as values:
            probabilities.append(values["lesion"]); lungs.append(values["lung"])
    candidates = []
    for lung_threshold in protocol["oof_lung_thresholds"]:
        for lesion_threshold in protocol["oof_lesion_thresholds"]:
            summary, _ = _score_native(items, probabilities, lungs, lung_threshold, lesion_threshold)
            candidates.append({"lung_threshold": lung_threshold, "lesion_threshold": lesion_threshold,
                               "summary": summary})
    winner = max(candidates, key=lambda x: _rank(x["summary"])); metric = winner["summary"]["pooled_lesion"]
    checks = {"lesion_dice_at_least_0_80": metric["dice"] >= .80,
              "precision_at_least_0_80": metric["precision"] >= .80,
              "recall_at_least_0_80": metric["recall"] >= .80,
              "specificity_at_least_0_80": metric["specificity"] >= .80,
              "lung_dice_at_least_0_90": winner["summary"]["pooled_lung"]["dice"] >= .90}
    report = {"status": "done", "created_at": now(), "version": VERSION, "protocol": protocol,
              "feature_artifact": feature_artifact, "fold_artifacts": artifacts,
              "oof_winner": winner, "oof_gate": {"passed": all(checks.values()), "checks": checks},
              "candidate_count": len(candidates), "test_payload_opened": False}
    atomic_json(root / "report.json", report)
    return report
