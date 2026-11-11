from __future__ import annotations

import json
import math
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from .binary_crossval import _loss, _predict, prepare_at_size
from .crossval_segformer import _score_native
from .data import load_audit, validate_fit_rows
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256, within_project
from .reference_v2 import DevelopmentReferenceStore
from .segmentation import (LesionConfig, color_regions, lesion_mask,
                           region_consensus_from_seed, robust_features,
                           unlabeled_lung_probability)
from .segformer_segmentation import _image_tensor, _rank


VERSION = "v47_unmarked_pseudo_segformer_b1"
SEED = 4704
IMAGE_SIZE = 384
EPOCHS = 5
V11_FINGERPRINT = "3060f1fec194671f606e94bb234a79a22b20927239abd5570e915a7ac658728d"


def _candidate() -> dict:
    path = ARTIFACTS / "frozen_candidate" / V11_FINGERPRINT / "candidate.json"
    value = json.loads(path.read_text())
    if value.get("version") != "v11b_coarse_regions_refined" or json_hash(
            {key: item for key, item in value.items() if key != "candidate_fingerprint"}) != V11_FINGERPRINT:
        raise ValueError("Frozen V11b pseudo teacher changed")
    return value


def letterbox_rgb(rgb: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = min(size / height, size / width)
    resized_height, resized_width = max(1, round(height * scale)), max(1, round(width * scale))
    interpolation = cv2.INTER_AREA if scale <= 1 else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.zeros((size, size, 3), np.uint8)
    top, left = (size - resized_height) // 2, (size - resized_width) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas


def pseudo_mask(rgb: np.ndarray, candidate: dict) -> tuple[np.ndarray, np.ndarray]:
    lung = unlabeled_lung_probability(rgb) >= candidate["foreground_alpha_threshold"]
    features = robust_features(rgb, lung)
    left = lesion_mask(features, lung, LesionConfig(**candidate["ensemble_left"]["config"]))
    right = lesion_mask(features, lung, LesionConfig(**candidate["ensemble_right"]["config"]))
    seed = left | right
    labels = color_regions(rgb, lung, candidate["lesion_region_segments"])
    lesion = region_consensus_from_seed(labels, lung, seed, candidate["lesion_region_support"])
    return lung, lesion


def build_pseudo_cache() -> tuple[dict, object]:
    audit = load_audit(); rows = list(audit["fit_rows"]); validate_fit_rows(rows, audit)
    if len(rows) != 1764:
        raise ValueError(f"Expected 1764 unmarked identities, found {len(rows)}")
    candidate = _candidate()
    protocol = {
        "version": VERSION, "audit_fingerprint": audit["audit_fingerprint"],
        "fit_images": len(rows), "image_size": IMAGE_SIZE,
        "pseudo_teacher_fingerprint": candidate["candidate_fingerprint"],
        "lung_threshold": candidate["foreground_alpha_threshold"],
        "lesion_rule": "v11b_union_slic_region_consensus",
        "region_segments": candidate["lesion_region_segments"],
        "region_support": candidate["lesion_region_support"],
        "source_rows_fingerprint": json_hash([{key: row[key] for key in
            ("id", "pixel_sha256", "file_sha256")} for row in rows]),
        "reference_images_used": 0, "reference_planes_used": 0,
    }
    fingerprint = json_hash(protocol); root = ARTIFACTS / "unmarked_pseudo_v47" / fingerprint
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / "protocol.json", protocol)
    artifact_path = root / "artifact.json"
    if artifact_path.exists():
        artifact = json.loads(artifact_path.read_text())
        if (artifact.get("status") != "done" or artifact.get("cache_fingerprint") != fingerprint
                or artifact.get("protocol") != protocol or len(artifact.get("records", [])) != len(rows)
                or artifact.get("reference_images_used") != 0
                or artifact.get("reference_planes_used") != 0
                or artifact.get("test_payload_opened") is not False):
            raise ValueError("Existing V47 pseudo artifact failed integrity checks")
        return artifact, root
    records = []
    for index, row in enumerate(rows, start=1):
        path = root / f"{row['id']}.npz"
        if not path.exists():
            bgr = cv2.imread(str(within_project(row["path"])), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Cannot decode unmarked image: {row['id']}")
            rgb = letterbox_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            lung, lesion = pseudo_mask(rgb, candidate)
            np.savez_compressed(path, lung=lung.astype(np.uint8), lesion=lesion.astype(np.uint8))
        with np.load(path, allow_pickle=False) as values:
            lung_pixels = int(values["lung"].sum()); lesion_pixels = int(values["lesion"].sum())
            if values["lung"].shape != (IMAGE_SIZE, IMAGE_SIZE) or np.any(values["lesion"] > values["lung"]):
                raise ValueError(f"Invalid V47 pseudo geometry: {row['id']}")
        records.append({"id": row["id"], "source_path": row["path"],
                        "source_sha256": row["file_sha256"],
                        "pseudo_path": str(path.relative_to(PROJECT_ROOT)),
                        "pseudo_sha256": sha256(path), "lung_pixels": lung_pixels,
                        "lesion_pixels": lesion_pixels})
        if index % 100 == 0:
            print(f"{VERSION} pseudo {index}/{len(rows)}", flush=True)
    artifact = {"status": "done", "created_at": now(), "version": VERSION,
                "cache_fingerprint": fingerprint, "protocol": protocol, "records": records,
                "mean_pseudo_lesion_lung_ratio": float(np.mean([
                    row["lesion_pixels"] / max(1, row["lung_pixels"]) for row in records])),
                "empty_lung_images": sum(row["lung_pixels"] == 0 for row in records),
                "empty_lesion_images": sum(row["lesion_pixels"] == 0 for row in records),
                "reference_images_used": 0, "reference_planes_used": 0,
                "test_payload_opened": False}
    atomic_json(artifact_path, artifact)
    return artifact, root


def _augment(rgb: np.ndarray, target: np.ndarray, valid: np.ndarray,
             rng: random.Random) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rng.random() < .5:
        rgb, target, valid = np.flip(rgb, 1).copy(), np.flip(target, 1).copy(), np.flip(valid, 1).copy()
    if rng.random() < .5:
        rgb, target, valid = np.flip(rgb, 0).copy(), np.flip(target, 0).copy(), np.flip(valid, 0).copy()
    turns = rng.randrange(4)
    if turns:
        rgb = np.rot90(rgb, turns).copy(); target = np.rot90(target, turns).copy()
        valid = np.rot90(valid, turns).copy()
    value = rgb.astype(np.float32) / 255.; mean = value.mean(axis=(0, 1), keepdims=True)
    value = np.clip(((value - mean) * rng.uniform(.85, 1.15) + mean) * rng.uniform(.85, 1.15)
                    * np.asarray([rng.uniform(.93, 1.07) for _ in range(3)]), 0, 1)
    return (_image_tensor(np.round(value * 255).astype(np.uint8)),
            torch.from_numpy(target.astype(np.float32)), torch.from_numpy(valid.astype(np.float32)))


def _advance_augment_rng(rng: random.Random) -> None:
    """Consume exactly the random draws made by one call to ``_augment``."""
    rng.random(); rng.random(); rng.randrange(4)
    rng.uniform(.85, 1.15); rng.uniform(.85, 1.15)
    for _ in range(3):
        rng.uniform(.93, 1.07)


def fit_unmarked_pseudo_v47() -> dict:
    pseudo_artifact, pseudo_root = build_pseudo_cache()
    protocol = {
        "version": VERSION, "seed": SEED, "epochs": EPOCHS, "batch_size": 2,
        "image_size": IMAGE_SIZE, "pretrained": "nvidia/segformer-b1-finetuned-ade-512-512",
        "optimizer": {"name": "AdamW", "learning_rate": 6e-5, "weight_decay": 1e-3},
        "scheduler": "cosine_to_5e-7", "loss": "bce_dice_false_positive_inside_pseudo_lung",
        "augmentation": "paired_hv_flip_rot90_mild_colour_jitter",
        "checkpoint_selection": "fixed_final_complete_epoch",
        "pseudo_cache_fingerprint": pseudo_artifact["cache_fingerprint"],
        "pseudo_artifact_sha256": sha256(pseudo_root / "artifact.json"),
        "fit_images": len(pseudo_artifact["records"]), "marked_fit_images": 0,
        "marked_fit_planes": 0, "sealed_test_images_opened": 0,
    }
    fingerprint = json_hash(protocol); root = ARTIFACTS / "unmarked_pseudo_segformer" / fingerprint
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / "protocol.json", protocol)
    report_path = root / "report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("V47 training requires the authorized GPU")
    model = SegformerForSemanticSegmentation.from_pretrained(
        protocol["pretrained"], num_labels=1, ignore_mismatched_sizes=True, use_safetensors=True,
        cache_dir=ARTIFACTS / "model_cache", id2label={0: "lesion"}, label2id={"lesion": 0}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=1e-3)
    scaler = torch.amp.GradScaler("cuda"); rng = random.Random(SEED); history = []
    records = pseudo_artifact["records"]
    completed_epochs = 0
    optimizer_state_recovered = True
    checkpoint_paths = [root / f"epoch_{epoch}.pt" for epoch in range(1, EPOCHS + 1)]
    for epoch, path in enumerate(checkpoint_paths, start=1):
        if not path.exists():
            break
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint.get("version") != VERSION or checkpoint.get("epoch") != epoch or \
                checkpoint.get("protocol_fingerprint") != fingerprint:
            raise ValueError(f"Invalid V47 recovery checkpoint: {path.name}")
        completed_epochs = epoch
    if completed_epochs:
        checkpoint = torch.load(checkpoint_paths[completed_epochs - 1], map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint["model"])
        history_path = root / "history.json"
        history = json.loads(history_path.read_text())
        if len(history) != completed_epochs or [row["epoch"] for row in history] != list(
                range(1, completed_epochs + 1)):
            raise ValueError("V47 history does not match recovery checkpoints")
        if all(key in checkpoint for key in ("optimizer", "scaler", "rng_state")):
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            rng.setstate(checkpoint["rng_state"])
        else:
            optimizer_state_recovered = False
            # Recreate the exact Python RNG position used by completed epochs. The
            # first two legacy checkpoints contain weights only.
            for _epoch in range(1, completed_epochs + 1):
                replay_order = list(range(len(records))); rng.shuffle(replay_order)
                for _ in replay_order:
                    _advance_augment_rng(rng)
        print(f"{VERSION} resume after epoch {completed_epochs}/{EPOCHS}", flush=True)

    base_lr, eta_min = 6e-5, 5e-7
    for epoch in range(completed_epochs + 1, EPOCHS + 1):
        # Epoch N uses the cosine value after N-1 completed scheduler steps.
        learning_rate = eta_min + (base_lr - eta_min) * (1 + math.cos(
            math.pi * (epoch - 1) / EPOCHS)) / 2
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train(); order = list(range(len(records))); rng.shuffle(order); losses = []
        for start in range(0, len(order), 2):
            batch = []
            for index in order[start:start + 2]:
                record = records[index]
                bgr = cv2.imread(str(within_project(record["source_path"])), cv2.IMREAD_COLOR)
                if bgr is None or sha256(within_project(record["source_path"])) != record["source_sha256"]:
                    raise ValueError(f"V47 source changed: {record['id']}")
                rgb = letterbox_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                pseudo_path = within_project(record["pseudo_path"])
                if sha256(pseudo_path) != record["pseudo_sha256"]:
                    raise ValueError(f"V47 pseudo cache changed: {record['id']}")
                with np.load(pseudo_path, allow_pickle=False) as values:
                    batch.append(_augment(rgb, values["lesion"].astype(bool),
                                          values["lung"].astype(bool), rng))
            image = torch.stack([sample[0] for sample in batch]).to(device)
            target = torch.stack([sample[1] for sample in batch]).to(device)
            valid = torch.stack([sample[2] for sample in batch]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                logits = model(pixel_values=image).logits; loss, components = _loss(logits, target, valid)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
        next_learning_rate = eta_min + (base_lr - eta_min) * (1 + math.cos(
            math.pi * epoch / EPOCHS)) / 2
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "last_batch_components": components,
                        "learning_rate": next_learning_rate})
        checkpoint = {"version": VERSION, "epoch": epoch, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                      "rng_state": rng.getstate(), "config": model.config.to_dict(),
                      "protocol_fingerprint": fingerprint}
        torch.save(checkpoint, root / f"epoch_{epoch}.pt")
        atomic_json(root / "history.json", history)
        print(f"{VERSION} epoch {epoch}/{EPOCHS} loss={np.mean(losses):.4f}", flush=True)
    checkpoint_path = root / f"epoch_{EPOCHS}.pt"
    checkpoint_sha = sha256(checkpoint_path)
    # Reference planes are first opened here, after all fitting is complete and hashed.
    store = DevelopmentReferenceStore(); payloads = [store.load(row) for row in store.rows]
    items = [prepare_at_size(payload, IMAGE_SIZE) for payload in payloads]
    probabilities = _predict(model, items, device)
    lungs = [unlabeled_lung_probability(payload["rgb"]) for payload in payloads]
    oof = root / "development_predictions"; oof.mkdir(exist_ok=True)
    for item, probability, lung in zip(items, probabilities, lungs):
        np.savez_compressed(oof / f"{item['id']}.npz", lesion=probability, lung=lung)
    candidates = []
    for lung_threshold in [round(value, 2) for value in np.arange(.30, .901, .05)]:
        for lesion_threshold in [round(value, 3) for value in np.arange(.10, .901, .025)]:
            summary, _ = _score_native(items, probabilities, lungs, lung_threshold, lesion_threshold)
            candidates.append({"lung_threshold": lung_threshold, "lesion_threshold": lesion_threshold,
                               "summary": summary})
    winner = max(candidates, key=lambda value: _rank(value["summary"]))
    metric = winner["summary"]["pooled_lesion"]
    checks = {"lesion_dice_at_least_0_80": metric["dice"] >= .80,
              "precision_at_least_0_80": metric["precision"] >= .80,
              "recall_at_least_0_80": metric["recall"] >= .80,
              "specificity_at_least_0_80": metric["specificity"] >= .80,
              "lung_dice_at_least_0_90": winner["summary"]["pooled_lung"]["dice"] >= .90}
    report = {"status": "done", "created_at": now(), "version": VERSION,
              "protocol": protocol, "pseudo_artifact": pseudo_artifact,
              "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
              "checkpoint_sha256": checkpoint_sha, "history": history,
              "recovery": {"resumed_after_epoch": completed_epochs,
                           "optimizer_state_recovered": optimizer_state_recovered},
              "development_winner": winner,
              "development_gate": {"passed": all(checks.values()), "checks": checks},
              "candidate_count": len(candidates), "test_payload_opened": False}
    atomic_json(report_path, report)
    return report
