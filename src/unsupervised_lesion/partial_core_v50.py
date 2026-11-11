from __future__ import annotations

import json
import math
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from .binary_crossval import _predict, prepare_at_size
from .crossval_segformer import _score_native
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256, within_project
from .reference_v2 import DevelopmentReferenceStore
from .segformer_segmentation import _rank
from .segmentation import robust_features, unlabeled_lung_probability
from .unmarked_pseudo_v47 import (_augment, build_pseudo_cache, letterbox_rgb)


VERSION = "v50_unmarked_partial_core_segformer_b1"
SEED = 5004
IMAGE_SIZE = 384
EPOCHS = 6
POSITIVE_MARGIN = .75
NEGATIVE_MARGIN = -.50
GLOBAL_THRESHOLD = .1880188137292862
LOCAL_THRESHOLD = .3467429578304291


def partial_cores(features: dict[str, np.ndarray], lung: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    global_margin = features["dark"] - GLOBAL_THRESHOLD
    local_margin = features["dark"] + features["local_dark"] - LOCAL_THRESHOLD
    positive = (global_margin >= POSITIVE_MARGIN) & (local_margin >= POSITIVE_MARGIN) & lung
    negative = (global_margin <= NEGATIVE_MARGIN) & (local_margin <= NEGATIVE_MARGIN) & lung
    return positive, negative


def build_partial_core_cache() -> tuple[dict, object]:
    source, source_root = build_pseudo_cache()
    protocol = {
        "version": VERSION, "source_cache_fingerprint": source["cache_fingerprint"],
        "source_artifact_sha256": sha256(source_root / "artifact.json"),
        "fit_images": len(source["records"]), "image_size": IMAGE_SIZE,
        "global_threshold": GLOBAL_THRESHOLD, "local_threshold": LOCAL_THRESHOLD,
        "positive_margin": POSITIVE_MARGIN, "negative_margin": NEGATIVE_MARGIN,
        "ambiguous_policy": "ignore", "reference_images_used": 0,
        "reference_planes_used": 0,
    }
    fingerprint = json_hash(protocol); root = ARTIFACTS / "partial_core_v50" / fingerprint
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / "protocol.json", protocol)
    artifact_path = root / "artifact.json"
    if artifact_path.is_file():
        artifact = json.loads(artifact_path.read_text())
        if (artifact.get("status") != "done" or artifact.get("protocol") != protocol
                or len(artifact.get("records", [])) != len(source["records"])
                or artifact.get("test_payload_opened") is not False):
            raise ValueError("Existing V50 partial-core cache failed integrity checks")
        return artifact, root
    records = []
    for index, record in enumerate(source["records"], start=1):
        path = root / f"{record['id']}.npz"
        if not path.is_file():
            source_path = within_project(record["source_path"])
            if sha256(source_path) != record["source_sha256"]:
                raise ValueError(f"V50 source changed: {record['id']}")
            bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Cannot decode V50 source: {record['id']}")
            rgb = letterbox_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            pseudo_path = within_project(record["pseudo_path"])
            if sha256(pseudo_path) != record["pseudo_sha256"]:
                raise ValueError(f"V50 source pseudo changed: {record['id']}")
            with np.load(pseudo_path, allow_pickle=False) as values:
                lung = values["lung"].astype(bool)
            positive, negative = partial_cores(robust_features(rgb, lung), lung)
            np.savez_compressed(path, target=positive.astype(np.uint8),
                                valid=(positive | negative).astype(np.uint8),
                                lung=lung.astype(np.uint8))
        with np.load(path, allow_pickle=False) as values:
            target = values["target"].astype(bool); valid = values["valid"].astype(bool)
            lung = values["lung"].astype(bool)
            if target.shape != (IMAGE_SIZE, IMAGE_SIZE) or np.any(target & ~valid) or np.any(valid & ~lung):
                raise ValueError(f"Invalid V50 core geometry: {record['id']}")
        records.append({"id": record["id"], "source_path": record["source_path"],
                        "source_sha256": record["source_sha256"],
                        "core_path": str(path.relative_to(PROJECT_ROOT)), "core_sha256": sha256(path),
                        "lung_pixels": int(lung.sum()), "positive_pixels": int(target.sum()),
                        "negative_pixels": int((valid & ~target).sum())})
        if index % 100 == 0:
            print(f"{VERSION} core {index}/{len(source['records'])}", flush=True)
    lung_pixels = sum(row["lung_pixels"] for row in records)
    artifact = {"status": "done", "created_at": now(), "version": VERSION,
                "cache_fingerprint": fingerprint, "protocol": protocol, "records": records,
                "positive_fraction": sum(row["positive_pixels"] for row in records) / lung_pixels,
                "negative_fraction": sum(row["negative_pixels"] for row in records) / lung_pixels,
                "ignored_fraction": 1 - sum(row["positive_pixels"] + row["negative_pixels"]
                                             for row in records) / lung_pixels,
                "reference_images_used": 0, "reference_planes_used": 0,
                "test_payload_opened": False}
    atomic_json(artifact_path, artifact)
    return artifact, root


def _partial_loss(logits: torch.Tensor, target: torch.Tensor,
                  valid: torch.Tensor) -> tuple[torch.Tensor, dict]:
    logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    positive = target * valid; negative = (1 - target) * valid
    pos_weight = (negative.sum() / positive.sum().clamp_min(1)).clamp(1, 10)
    bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none",
                                                 pos_weight=pos_weight)
    bce = (bce_map * valid).sum() / valid.sum().clamp_min(1)
    probability = logits.sigmoid() * valid
    dice = 1 - ((2 * (probability * positive).sum(dim=(1, 2)) + 1) /
                (probability.sum(dim=(1, 2)) + positive.sum(dim=(1, 2)) + 1)).mean()
    false_positive = (probability * negative).sum() / negative.sum().clamp_min(1)
    total = .70 * bce + 1.00 * dice + .40 * false_positive
    return total, {"bce": float(bce.detach()), "positive_dice_loss": float(dice.detach()),
                   "confident_negative_fp": float(false_positive.detach()),
                   "batch_pos_weight": float(pos_weight.detach())}


def fit_partial_core_v50() -> dict:
    core_artifact, core_root = build_partial_core_cache()
    protocol = {"version": VERSION, "seed": SEED, "epochs": EPOCHS, "batch_size": 2,
                "image_size": IMAGE_SIZE, "pretrained": "nvidia/segformer-b1-finetuned-ade-512-512",
                "optimizer": {"name": "AdamW", "learning_rate": 6e-5, "weight_decay": 1e-3},
                "scheduler": "cosine_to_5e-7", "loss": "balanced_bce_core_dice_confident_negative_fp",
                "augmentation": "paired_hv_flip_rot90_mild_colour_jitter",
                "checkpoint_selection": "fixed_final_complete_epoch",
                "core_cache_fingerprint": core_artifact["cache_fingerprint"],
                "core_artifact_sha256": sha256(core_root / "artifact.json"),
                "fit_images": len(core_artifact["records"]), "marked_fit_images": 0,
                "marked_fit_planes": 0, "sealed_test_images_opened": 0}
    fingerprint = json_hash(protocol); root = ARTIFACTS / "partial_core_segformer_v50" / fingerprint
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / "protocol.json", protocol)
    report_path = root / "report.json"
    if report_path.is_file():
        return json.loads(report_path.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("V50 training requires the authorized GPU")
    model = SegformerForSemanticSegmentation.from_pretrained(
        protocol["pretrained"], num_labels=1, ignore_mismatched_sizes=True, use_safetensors=True,
        cache_dir=ARTIFACTS / "model_cache", id2label={0: "lesion"}, label2id={"lesion": 0}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=1e-3)
    scaler = torch.amp.GradScaler("cuda"); rng = random.Random(SEED); history = []
    records = core_artifact["records"]; completed = 0
    for epoch in range(1, EPOCHS + 1):
        path = root / f"epoch_{epoch}.pt"
        if not path.is_file():
            break
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint.get("epoch") != epoch or checkpoint.get("protocol_fingerprint") != fingerprint:
            raise ValueError(f"Invalid V50 checkpoint: {path.name}")
        completed = epoch
    if completed:
        checkpoint = torch.load(root / f"epoch_{completed}.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"]); rng.setstate(checkpoint["rng_state"])
        history = json.loads((root / "history.json").read_text())
        if len(history) != completed:
            raise ValueError("V50 history/checkpoint mismatch")
        print(f"{VERSION} resume after epoch {completed}/{EPOCHS}", flush=True)
    base_lr, eta_min = 6e-5, 5e-7
    for epoch in range(completed + 1, EPOCHS + 1):
        learning_rate = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * (epoch - 1) / EPOCHS)) / 2
        for group in optimizer.param_groups: group["lr"] = learning_rate
        model.train(); order = list(range(len(records))); rng.shuffle(order); losses = []
        for start in range(0, len(order), 2):
            batch = []
            for item_index in order[start:start + 2]:
                record = records[item_index]; source_path = within_project(record["source_path"])
                if sha256(source_path) != record["source_sha256"]:
                    raise ValueError(f"V50 source changed during fit: {record['id']}")
                bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
                rgb = letterbox_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                core_path = within_project(record["core_path"])
                if sha256(core_path) != record["core_sha256"]:
                    raise ValueError(f"V50 core changed during fit: {record['id']}")
                with np.load(core_path, allow_pickle=False) as values:
                    batch.append(_augment(rgb, values["target"].astype(bool),
                                          values["valid"].astype(bool), rng))
            image = torch.stack([sample[0] for sample in batch]).to(device)
            target = torch.stack([sample[1] for sample in batch]).to(device)
            valid = torch.stack([sample[2] for sample in batch]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                logits = model(pixel_values=image).logits; loss, components = _partial_loss(logits, target, valid)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
        next_lr = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * epoch / EPOCHS)) / 2
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "last_batch_components": components, "learning_rate": next_lr})
        torch.save({"version": VERSION, "epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "rng_state": rng.getstate(), "config": model.config.to_dict(),
                    "protocol_fingerprint": fingerprint}, root / f"epoch_{epoch}.pt")
        atomic_json(root / "history.json", history)
        print(f"{VERSION} epoch {epoch}/{EPOCHS} loss={np.mean(losses):.4f}", flush=True)
    checkpoint_path = root / f"epoch_{EPOCHS}.pt"; checkpoint_sha = sha256(checkpoint_path)
    store = DevelopmentReferenceStore(); payloads = [store.load(row) for row in store.rows]
    items = [prepare_at_size(payload, IMAGE_SIZE) for payload in payloads]
    probabilities = _predict(model, items, device)
    lungs = [unlabeled_lung_probability(payload["rgb"]) for payload in payloads]
    predictions = root / "development_predictions"; predictions.mkdir(exist_ok=True)
    for item, lesion, lung in zip(items, probabilities, lungs):
        np.savez_compressed(predictions / f"{item['id']}.npz", lesion=lesion, lung=lung)
    candidates = []
    for lung_threshold in [round(value, 2) for value in np.arange(.30, .901, .05)]:
        for lesion_threshold in [round(value, 3) for value in np.arange(.10, .901, .025)]:
            summary, _ = _score_native(items, probabilities, lungs, lung_threshold, lesion_threshold)
            candidates.append({"lung_threshold": lung_threshold, "lesion_threshold": lesion_threshold,
                               "summary": summary})
    winner = max(candidates, key=lambda value: _rank(value["summary"]))
    lesion = winner["summary"]["pooled_lesion"]
    checks = {"lesion_dice_at_least_0_80": lesion["dice"] >= .80,
              "precision_at_least_0_80": lesion["precision"] >= .80,
              "recall_at_least_0_80": lesion["recall"] >= .80,
              "specificity_at_least_0_80": lesion["specificity"] >= .80,
              "lung_dice_at_least_0_90": winner["summary"]["pooled_lung"]["dice"] >= .90}
    report = {"status": "done", "created_at": now(), "version": VERSION,
              "protocol": protocol, "core_artifact": core_artifact,
              "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
              "checkpoint_sha256": checkpoint_sha, "history": history,
              "recovery": {"resumed_after_epoch": completed}, "development_winner": winner,
              "development_gate": {"passed": all(checks.values()), "checks": checks},
              "candidate_count": len(candidates), "test_payload_opened": False}
    atomic_json(report_path, report)
    return report
