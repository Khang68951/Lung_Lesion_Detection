from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import json
import random

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

from .binary_crossval import _loss
from .crossval_segformer import development_folds
from .evaluate import _measure, summarize
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256
from .lung_consensus_v54 import V53_ROOT, vote_consensus
from .reference_v2 import DevelopmentReferenceStore
from .segformer_segmentation import _image_tensor


VERSION = "v62_supervised_v54_side_crop_group_oof"
SEED = 6205
IMAGE_SIZE = 384
PRETRAINED = "nvidia/segformer-b1-finetuned-ade-512-512"


@dataclass(frozen=True)
class CropGeometry:
    native_height: int
    native_width: int
    y0: int
    y1: int
    x0: int
    x1: int
    image_size: int


def v54_side_masks(path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        shape = tuple(values["mask_shape"])
        metadata = json.loads(str(values["metadata_json"]))
        raw = [np.unpackbits(value, count=shape[0] * shape[1]).reshape(shape).astype(bool)
               for value in values["packed_masks"]]
    output = {}
    for side in ("left", "right"):
        planes = []
        for variant in range(12):
            selected = [mask for mask, record in zip(raw, metadata)
                        if record["variant"] == variant and record["side"] == side
                        and record["output"] == 2]
            if len(selected) != 1:
                raise ValueError(f"Incomplete V53 {side} mask variant {variant}: {path}")
            planes.append(selected[0])
        output[side] = vote_consensus(planes, 3)
    return output


def crop_geometry(mask: np.ndarray, image_size: int = IMAGE_SIZE,
                  margin_fraction: float = .04) -> CropGeometry:
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("V62 side crop requires a non-empty 2D mask")
    height, width = mask.shape
    ys, xs = np.where(mask)
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())
    margin = max(4, round(max(ymax - ymin + 1, xmax - xmin + 1)
                          * margin_fraction))
    y0 = max(0, ymin - margin); y1 = min(height, ymax + 1 + margin)
    x0 = max(0, xmin - margin); x1 = min(width, xmax + 1 + margin)
    return CropGeometry(height, width, y0, y1, x0, x1, image_size)


def restore_crop(probability: np.ndarray, geometry: CropGeometry) -> np.ndarray:
    probability = np.asarray(probability, np.float32)
    if probability.shape != (geometry.image_size, geometry.image_size):
        raise ValueError("Invalid V62 crop probability geometry")
    restored = np.zeros((geometry.native_height, geometry.native_width), np.float32)
    patch = cv2.resize(probability, (geometry.x1 - geometry.x0, geometry.y1 - geometry.y0),
                       interpolation=cv2.INTER_LINEAR)
    restored[geometry.y0:geometry.y1, geometry.x0:geometry.x1] = patch
    return restored


def prepare_side_item(payload: dict, side_mask: np.ndarray, side: str,
                      image_size: int = IMAGE_SIZE) -> dict:
    geometry = crop_geometry(side_mask, image_size)
    sl = np.s_[geometry.y0:geometry.y1, geometry.x0:geometry.x1]
    rgb = payload["rgb"][sl].copy()
    local_side = side_mask[sl]
    # Removing the opposite lobe and central organs makes colour/transition learning
    # spend all spatial capacity on one V54 anatomical lobe.
    rgb[~local_side] = 0
    target = (payload["lesion"] & payload["lung"])[sl]
    valid = (side_mask & payload["lung"] & payload["valid"]
             & ~payload["ignore_lung"] & ~payload["ignore_lesion"])[sl]
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    target = cv2.resize(target.astype(np.uint8), (image_size, image_size),
                        interpolation=cv2.INTER_NEAREST) > 0
    valid = cv2.resize(valid.astype(np.uint8), (image_size, image_size),
                       interpolation=cv2.INTER_NEAREST) > 0
    return {"id": payload["row"]["id"], "group": payload["row"]["group"],
            "side": side, "rgb": rgb, "target": target, "valid": valid,
            "geometry": geometry}


def _augment(item: dict, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = item["rgb"].copy(); target = item["target"].copy(); valid = item["valid"].copy()
    if rng.random() < .5:
        rgb = np.flip(rgb, 1).copy(); target = np.flip(target, 1).copy(); valid = np.flip(valid, 1).copy()
    if rng.random() < .25:
        rgb = np.flip(rgb, 0).copy(); target = np.flip(target, 0).copy(); valid = np.flip(valid, 0).copy()
    value = rgb.astype(np.float32) / 255
    tissue = valid[..., None]
    pixels = value[valid]
    mean = pixels.mean(0, keepdims=True).reshape(1, 1, 3) if len(pixels) else value.mean((0, 1), keepdims=True)
    value = np.clip((value - mean) * rng.uniform(.85, 1.15) + mean, 0, 1)
    value = np.clip(value * rng.uniform(.88, 1.12)
                    * np.asarray([rng.uniform(.94, 1.06) for _ in range(3)]), 0, 1)
    value[~tissue.repeat(3, axis=2)] = 0
    return (_image_tensor(np.round(value * 255).astype(np.uint8)),
            torch.from_numpy(target.astype(np.float32)),
            torch.from_numpy(valid.astype(np.float32)))


def _predict_sides(model, items: list[dict], device: torch.device) -> dict[str, np.ndarray]:
    model.eval(); merged = {}
    with torch.inference_mode():
        for item in items:
            logits = model(pixel_values=_image_tensor(item["rgb"])[None].to(device)).logits
            probability = torch.nn.functional.interpolate(
                logits, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
            ).sigmoid()[0, 0].float().cpu().numpy()
            restored = restore_crop(probability, item["geometry"])
            identifier = item["id"]
            merged[identifier] = np.maximum(merged.get(identifier, np.zeros_like(restored)), restored)
    return merged


def fit_lobe_crop_v62(epochs: int = 25) -> dict:
    if epochs < 1:
        raise ValueError("V62 epochs must be positive")
    store = DevelopmentReferenceStore(); rows = list(store.rows)
    payloads = {row["id"]: store.load(row) for row in rows}
    masks = {}; side_items = []
    for row in rows:
        source = V53_ROOT / f"{row['id']}.npz"
        sides = v54_side_masks(source); masks[row["id"]] = sides["left"] | sides["right"]
        side_items.extend(prepare_side_item(payloads[row["id"]], sides[side], side)
                          for side in ("left", "right"))
    folds = development_folds(rows)
    protocol = {
        "version": VERSION, "seed": SEED, "epochs_per_fold": epochs,
        "image_size": IMAGE_SIZE, "pretrained": PRETRAINED,
        "initialization": "public ADE20K SegFormer-B1 encoder; fresh binary head",
        "learning_rate": 6e-5,
        "batch_size": 2, "fold_groups": folds, "development_images": len(rows),
        "side_crops": len(side_items), "crop_margin_fraction": .04,
        "input": "one V54 left/right side per crop; non-side pixels black",
        "loss_domain": "V54 side intersect reference lung and valid reference pixels",
        "checkpoint_selection": "fixed final epoch; no held-out epoch selection",
        "lesion_thresholds": [round(float(x), 3) for x in np.arange(.05, .951, .01)],
        "fit_policy": "user-authorized supervised development fitting on 2026-11-05",
        "reference_fingerprint": store.manifest["fingerprint"],
        "marked_fit_images": len(rows), "sealed_test_images_opened": 0,
    }
    fingerprint = json_hash(protocol); root = ARTIFACTS / "lobe_crop_crossval_v62" / fingerprint
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / "protocol.json", protocol)
    report_path = root / "report.json"
    if report_path.is_file():
        return json.loads(report_path.read_text())
    oof = root / "oof"; oof.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_artifacts = []
    for fold_index, validation_groups in enumerate(folds):
        training = [item for item in side_items if item["group"] not in validation_groups]
        validation = [item for item in side_items if item["group"] in validation_groups]
        folder = root / f"fold_{fold_index}"; folder.mkdir(exist_ok=True)
        artifact_path = folder / "artifact.json"
        if artifact_path.is_file() and all((oof / f"{item['id']}.npz").is_file() for item in validation):
            fold_artifacts.append(json.loads(artifact_path.read_text())); continue
        torch.manual_seed(SEED + fold_index); np.random.seed(SEED + fold_index)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED + fold_index)
        model = SegformerForSemanticSegmentation.from_pretrained(
            PRETRAINED, num_labels=1, ignore_mismatched_sizes=True, use_safetensors=True,
            cache_dir=ARTIFACTS / "model_cache", id2label={0: "lesion"},
            label2id={"lesion": 0}).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=5e-7)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        rng = random.Random(SEED + fold_index); start_epoch = 1; history = []
        last_path = folder / "last.pt"
        if last_path.is_file():
            state = torch.load(last_path, map_location=device, weights_only=False)
            if state.get("protocol_fingerprint") != fingerprint:
                raise ValueError("V62 resume protocol changed")
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"])
            rng.setstate(state["python_rng_state"]); history = state["history"]
            start_epoch = int(state["epoch"]) + 1
        for epoch in range(start_epoch, epochs + 1):
            model.train(); order = list(range(len(training))); rng.shuffle(order); losses = []
            for start in range(0, len(order), protocol["batch_size"]):
                samples = [_augment(training[index], rng)
                           for index in order[start:start + protocol["batch_size"]]]
                images = torch.stack([x[0] for x in samples]).to(device)
                targets = torch.stack([x[1] for x in samples]).to(device)
                valid = torch.stack([x[2] for x in samples]).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    loss, components = _loss(model(pixel_values=images).logits, targets, valid)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
            scheduler.step(); history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                "last_batch_components": components, "learning_rate": optimizer.param_groups[0]["lr"]})
            torch.save({"version": VERSION, "fold": fold_index, "epoch": epoch,
                "protocol_fingerprint": fingerprint, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "python_rng_state": rng.getstate(), "history": history},
                last_path)
            atomic_json(folder / "history.json", history)
            print(f"{VERSION} fold {fold_index + 1}/4 epoch {epoch}/{epochs} "
                  f"loss={np.mean(losses):.4f}", flush=True)
        merged = _predict_sides(model, validation, device)
        for identifier, probability in merged.items():
            np.savez_compressed(oof / f"{identifier}.npz", lesion=probability,
                                lung=masks[identifier].astype(np.uint8), fold=fold_index)
        artifact = {"fold": fold_index,
            "training_groups": sorted({item["group"] for item in training}),
            "validation_groups": validation_groups, "final_epoch": epochs,
            "checkpoint": str(last_path.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": sha256(last_path),
            "initialized_from": PRETRAINED}
        atomic_json(artifact_path, artifact); fold_artifacts.append(artifact)
        del model, optimizer, scheduler, scaler
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    frozen = []
    for row in rows:
        with np.load(oof / f"{row['id']}.npz", allow_pickle=False) as values:
            frozen.append((values["lung"].astype(bool), values["lesion"].astype(np.float32)))
    candidates = []
    for threshold in protocol["lesion_thresholds"]:
        measured = [_measure(payloads[row["id"]], lung, (score >= threshold) & lung)
                    for row, (lung, score) in zip(rows, frozen)]
        candidates.append({"lesion_threshold": threshold, "summary": summarize(measured)})
    winner = max(candidates, key=lambda x: (
        min(x["summary"]["pooled_lesion"][k] or 0 for k in ("precision", "recall", "specificity")),
        x["summary"]["pooled_lesion"]["dice"] or 0))
    dice_winner = max(candidates, key=lambda x: x["summary"]["pooled_lesion"]["dice"] or 0)
    metric = winner["summary"]["pooled_lesion"]
    checks = {"lesion_dice_at_least_0_80": metric["dice"] >= .80,
              "precision_at_least_0_80": metric["precision"] >= .80,
              "recall_at_least_0_80": metric["recall"] >= .80,
              "specificity_at_least_0_80": metric["specificity"] >= .80,
              "lung_dice_at_least_0_95": winner["summary"]["pooled_lung"]["dice"] >= .95}
    report = {"status": "done", "created_at": now(), "version": VERSION,
              "protocol": protocol, "protocol_fingerprint": fingerprint,
              "fold_artifacts": fold_artifacts, "oof_balanced_winner": winner,
              "oof_dice_winner": dice_winner,
              "oof_gate": {"passed": all(checks.values()), "checks": checks},
              "candidate_count": len(candidates), "test_payload_opened": False}
    atomic_json(report_path, report)
    if report["oof_gate"]["passed"]:
        atomic_json(ARTIFACTS / "test_unlock.json", {"created_at": now(),
            "confirmation_gate_passed": True, "gate_source": VERSION,
            "crossval_fingerprint": fingerprint})
    return report
