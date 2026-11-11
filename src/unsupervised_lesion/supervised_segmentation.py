from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

from .evaluate import _measure, summarize
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256, within_project
from .reference import ReferenceStore


VERSION = "v19_supervised_calibration_deeplab"
SEED = 1904
IMAGE_SIZE = 384
EPOCHS = 60


def split_calibration_groups(rows: list[dict], validation_groups: int = 3) -> tuple[list[str], list[str]]:
    groups = sorted({row["group"] for row in rows},
                    key=lambda value: hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest())
    if len(groups) <= validation_groups:
        raise ValueError("Not enough calibration groups for a protected internal split")
    return sorted(groups[validation_groups:]), sorted(groups[:validation_groups])


def build_network(pretrained_backbone: bool = True) -> nn.Module:
    # Keep every downloaded generic weight under the workflow, not a global cache.
    torch.hub.set_dir(str(ARTIFACTS / "torch_hub"))
    backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained_backbone else None
    return deeplabv3_mobilenet_v3_large(
        weights=None, weights_backbone=backbone, num_classes=2, aux_loss=False)


@dataclass
class Geometry:
    height: int
    width: int
    resized_height: int
    resized_width: int
    top: int
    left: int


def _letterbox(array: np.ndarray, interpolation: int, fill: int = 0) -> tuple[np.ndarray, Geometry]:
    height, width = array.shape[:2]
    scale = min(IMAGE_SIZE / height, IMAGE_SIZE / width)
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))
    resized = cv2.resize(array, (resized_width, resized_height), interpolation=interpolation)
    shape = (IMAGE_SIZE, IMAGE_SIZE) + (() if array.ndim == 2 else (array.shape[2],))
    canvas = np.full(shape, fill, dtype=array.dtype)
    top = (IMAGE_SIZE - resized_height) // 2
    left = (IMAGE_SIZE - resized_width) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas, Geometry(height, width, resized_height, resized_width, top, left)


def _restore(probability: np.ndarray, geometry: Geometry) -> np.ndarray:
    cropped = probability[geometry.top:geometry.top + geometry.resized_height,
                          geometry.left:geometry.left + geometry.resized_width]
    return cv2.resize(cropped, (geometry.width, geometry.height), interpolation=cv2.INTER_LINEAR)


def _prepare(payload: dict) -> dict:
    rgb, geometry = _letterbox(payload["rgb"], cv2.INTER_AREA)
    planes = {}
    for key in ("lung", "lesion", "valid", "ignore_lung", "ignore_lesion"):
        planes[key], _ = _letterbox(payload[key].astype(np.uint8), cv2.INTER_NEAREST)
        planes[key] = planes[key] > 0
    return {"id": payload["row"]["id"], "group": payload["row"]["group"],
            "rgb": rgb, "geometry": geometry, "payload": payload, **planes}


def _image_tensor(rgb: np.ndarray) -> torch.Tensor:
    value = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (value - mean) / std


def _augment(item: dict, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = item["rgb"].copy()
    target = np.stack([item["lung"], item["lesion"]]).astype(np.float32)
    valid = np.stack([item["valid"] & ~item["ignore_lung"],
                      item["valid"] & ~item["ignore_lesion"] & item["lung"]]).astype(np.float32)
    if rng.random() < .5:
        rgb = np.flip(rgb, 1).copy(); target = np.flip(target, 2).copy(); valid = np.flip(valid, 2).copy()
    if rng.random() < .5:
        rgb = np.flip(rgb, 0).copy(); target = np.flip(target, 1).copy(); valid = np.flip(valid, 1).copy()
    rotations = rng.randrange(4)
    if rotations:
        rgb = np.rot90(rgb, rotations).copy()
        target = np.rot90(target, rotations, axes=(1, 2)).copy()
        valid = np.rot90(valid, rotations, axes=(1, 2)).copy()
    rgb_float = rgb.astype(np.float32) / 255.0
    brightness = rng.uniform(.85, 1.15); contrast = rng.uniform(.85, 1.15)
    channel = np.asarray([rng.uniform(.93, 1.07) for _ in range(3)], np.float32)
    mean = rgb_float.mean(axis=(0, 1), keepdims=True)
    rgb_float = np.clip(((rgb_float - mean) * contrast + mean) * brightness * channel, 0, 1)
    rgb = np.round(rgb_float * 255).astype(np.uint8)
    return _image_tensor(rgb), torch.from_numpy(target), torch.from_numpy(valid)


def _masked_bce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * valid).sum() / valid.sum().clamp_min(1)


def _soft_dice(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid() * valid; target = target * valid
    numerator = 2 * (probability * target).sum(dim=(1, 2)) + 1
    denominator = probability.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1
    return 1 - (numerator / denominator).mean()


def _loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, dict]:
    lung_bce = _masked_bce(logits[:, 0], target[:, 0], valid[:, 0])
    lesion_bce = _masked_bce(logits[:, 1], target[:, 1], valid[:, 1])
    lung_dice = _soft_dice(logits[:, 0], target[:, 0], valid[:, 0])
    lesion_dice = _soft_dice(logits[:, 1], target[:, 1], valid[:, 1])
    negative = valid[:, 1] * (1 - target[:, 1])
    false_positive = (logits[:, 1].sigmoid() * negative).sum() / negative.sum().clamp_min(1)
    total = .35 * lung_bce + .65 * lung_dice + .8 * lesion_bce + 1.2 * lesion_dice + .25 * false_positive
    return total, {"lung_bce": float(lung_bce.detach()), "lung_dice_loss": float(lung_dice.detach()),
                   "lesion_bce": float(lesion_bce.detach()),
                   "lesion_dice_loss": float(lesion_dice.detach()),
                   "false_positive_penalty": float(false_positive.detach())}


def _predict(network: nn.Module, items: list[dict], device: torch.device) -> list[np.ndarray]:
    network.eval(); probabilities = []
    with torch.inference_mode():
        for item in items:
            image = _image_tensor(item["rgb"])[None].to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = network(image)["out"]
            probabilities.append(logits.sigmoid()[0].float().cpu().numpy())
    return probabilities


def _score(items: list[dict], probabilities: list[np.ndarray], lung_threshold: float,
           lesion_threshold: float) -> tuple[dict, list[dict]]:
    rows = []
    for item, probability in zip(items, probabilities):
        lung_probability = _restore(probability[0], item["geometry"])
        lesion_probability = _restore(probability[1], item["geometry"])
        lung = lung_probability >= lung_threshold
        lesion = (lesion_probability >= lesion_threshold) & lung
        rows.append(_measure(item["payload"], lung, lesion))
    return summarize(rows), rows


def _rank(summary: dict) -> tuple[float, float, float, float]:
    lesion = summary["pooled_lesion"]
    floor = min(lesion["precision"] or 0, lesion["recall"] or 0, lesion["specificity"] or 0)
    return (floor, lesion["dice"] or 0, summary["pooled_lung"]["dice"] or 0,
            -(summary["ratio_mae_percentage_points"] or 999))


def fit_supervised(epochs: int = EPOCHS) -> dict:
    store = ReferenceStore("calibration")
    train_groups, validation_groups = split_calibration_groups(store.rows)
    train_set = [_prepare(store.load(row)) for row in store.rows if row["group"] in train_groups]
    validation_set = [_prepare(store.load(row)) for row in store.rows if row["group"] in validation_groups]
    protocol = {"version": VERSION, "seed": SEED, "epochs": epochs, "image_size": IMAGE_SIZE,
                "backbone": "torchvision MobileNetV3-Large ImageNet1K-v2",
                "head": "DeepLabV3 two independent logits: lung, lesion",
                "fit_reference_role": "calibration_only", "training_groups": train_groups,
                "internal_validation_groups": validation_groups,
                "training_ids": [item["id"] for item in train_set],
                "internal_validation_ids": [item["id"] for item in validation_set],
                "confirmation_images_used_for_fit_or_selection": 0,
                "sealed_test_images_opened": 0,
                "threshold_grid": {"lung": [.35, .45, .55, .65],
                                   "lesion": [.25, .35, .45, .55, .65, .75]}}
    fingerprint = json_hash(protocol); folder = ARTIFACTS / "supervised_segmentation" / fingerprint
    folder.mkdir(parents=True, exist_ok=True); atomic_json(folder / "protocol.json", protocol)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = build_network(True).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1.5e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []; best_rank = (-1., -1., -1., -999.); rng = random.Random(SEED)
    for epoch in range(1, epochs + 1):
        network.train(); order = list(range(len(train_set))); rng.shuffle(order); losses = []
        for start in range(0, len(order), 2):
            samples = [_augment(train_set[index], rng) for index in order[start:start + 2]]
            image = torch.stack([sample[0] for sample in samples]).to(device)
            target = torch.stack([sample[1] for sample in samples]).to(device)
            valid = torch.stack([sample[2] for sample in samples]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = network(image)["out"]
                loss, components = _loss(logits, target, valid)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(network.parameters(), 2.0)
            scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
        scheduler.step()
        probability = _predict(network, validation_set, device)
        summary, _ = _score(validation_set, probability, .45, .45)
        rank = _rank(summary)
        epoch_row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                     "learning_rate": optimizer.param_groups[0]["lr"],
                     "internal_validation": summary, "last_batch_components": components}
        history.append(epoch_row)
        payload = {"version": VERSION, "epoch": epoch, "model": network.state_dict(),
                   "protocol": protocol, "protocol_fingerprint": fingerprint,
                   "validation_rank_at_0_45": rank}
        torch.save(payload, folder / "last.pt")
        if rank > best_rank:
            best_rank = rank; torch.save(payload, folder / "best.pt")
        atomic_json(folder / "history.json", history)
        print(f"v19 epoch {epoch}/{epochs} loss={np.mean(losses):.4f} "
              f"val_floor={rank[0]:.4f} val_dice={rank[1]:.4f} lung={rank[2]:.4f}", flush=True)
    checkpoint = torch.load(folder / "best.pt", map_location=device, weights_only=False)
    network.load_state_dict(checkpoint["model"])
    probability = _predict(network, validation_set, device)
    candidates = []
    for lung_threshold in protocol["threshold_grid"]["lung"]:
        for lesion_threshold in protocol["threshold_grid"]["lesion"]:
            summary, _ = _score(validation_set, probability, lung_threshold, lesion_threshold)
            candidates.append({"lung_threshold": lung_threshold,
                               "lesion_threshold": lesion_threshold, "summary": summary})
    winner = max(candidates, key=lambda item: _rank(item["summary"]))
    checkpoint_path = folder / "best.pt"
    artifact = {"status": "fit_complete", "created_at": now(), "version": VERSION,
                "artifact_fingerprint": fingerprint,
                "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
                "checkpoint_sha256": sha256(checkpoint_path),
                "best_epoch": checkpoint["epoch"], "device": str(device),
                "training_images": len(train_set), "internal_validation_images": len(validation_set),
                "training_groups": train_groups, "internal_validation_groups": validation_groups,
                "confirmation_images_used_for_fit_or_selection": 0,
                "sealed_test_images_opened": 0, "threshold_selection": winner}
    atomic_json(folder / "artifact.json", artifact)
    candidate = {"status": "frozen_after_internal_validation", "created_at": now(),
                 "version": VERSION, "selection_role": "calibration_group_holdout_only",
                 "lesion_postprocess": "supervised_deeplab_v19",
                 "artifact_path": str((folder / "artifact.json").relative_to(PROJECT_ROOT)),
                 "artifact_sha256": sha256(folder / "artifact.json"),
                 "checkpoint": artifact["checkpoint"],
                 "checkpoint_sha256": artifact["checkpoint_sha256"],
                 "lung_threshold": winner["lung_threshold"],
                 "lesion_threshold": winner["lesion_threshold"],
                 "internal_validation_summary": winner["summary"],
                 "confirmation_images_used_for_selection": 0, "test_payload_opened": False}
    candidate["candidate_fingerprint"] = json_hash(candidate)
    candidate_folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
    atomic_json(candidate_folder / "candidate.json", candidate)
    atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                {"path": str((candidate_folder / "candidate.json").relative_to(PROJECT_ROOT)),
                 "sha256": sha256(candidate_folder / "candidate.json"),
                 "candidate_fingerprint": candidate["candidate_fingerprint"]})
    atomic_json(ARTIFACTS / "supervised_segmentation/current.json",
                {"path": artifact["checkpoint"], "artifact_path": candidate["artifact_path"],
                 "artifact_fingerprint": fingerprint})
    return {"artifact": artifact, "candidate": candidate, "candidate_count": len(candidates)}


def load_supervised_candidate(candidate: dict, device: torch.device | None = None) -> nn.Module:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = within_project(candidate["checkpoint"])
    if sha256(checkpoint_path) != candidate["checkpoint_sha256"]:
        raise ValueError("Supervised checkpoint checksum changed")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    network = build_network(False).to(device)
    network.load_state_dict(checkpoint["model"]); network.eval()
    return network


def evaluate_store(candidate: dict, store: ReferenceStore) -> tuple[dict, list[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = load_supervised_candidate(candidate, device)
    items = [_prepare(store.load(row)) for row in store.rows]
    probability = _predict(network, items, device)
    return _score(items, probability, candidate["lung_threshold"], candidate["lesion_threshold"])
