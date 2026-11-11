from __future__ import annotations

import hashlib
import json
import random

import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

from .evaluate import _measure, summarize
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256
from .reference import ReferenceStore
from .segformer_segmentation import (_augment, _load, _loss, _predict, _rank, _restore)
from .segmentation import unlabeled_lung_probability
from .supervised_segmentation import _prepare


VERSION = "v25_development_group_crossval"
SEED = 2504


def development_folds(rows: list[dict], folds: int = 4) -> list[list[str]]:
    groups = sorted({row["group"] for row in rows},
                    key=lambda value: hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest())
    result = [[] for _ in range(folds)]
    for index, group in enumerate(groups): result[index % folds].append(group)
    return [sorted(fold) for fold in result]


def _score_native(items: list[dict], lesion_probabilities: list[np.ndarray],
                  lung_probabilities: list[np.ndarray], lung_threshold: float,
                  lesion_threshold: float) -> tuple[dict, list[dict]]:
    rows = []
    for item, lesion_probability, lung_probability in zip(items, lesion_probabilities, lung_probabilities):
        lung = lung_probability >= lung_threshold; lesion = (lesion_probability >= lesion_threshold) & lung
        rows.append(_measure(item["payload"], lung, lesion))
    return summarize(rows), rows


def fit_crossval(epochs: int = 35) -> dict:
    stores = [ReferenceStore("calibration"), ReferenceStore("confirmation")]
    metadata = stores[0].rows + stores[1].rows
    payload_by_id = {row["id"]: store.load(row) for store in stores for row in store.rows}
    items = [_prepare(payload_by_id[row["id"]]) for row in metadata]
    item_by_id = {item["id"]: item for item in items}; folds = development_folds(metadata)
    protocol = {"version": VERSION, "seed": SEED, "epochs_per_fold": epochs,
                "pretrained": "nvidia/segformer-b1-finetuned-ade-512-512",
                "development_roles": ["former_calibration", "former_confirmation"],
                "development_images": len(items), "development_groups": len({row["group"] for row in metadata}),
                "fold_groups": folds, "sealed_test_images_opened": 0,
                "epoch_lesion_thresholds": [.25, .40, .55, .70],
                "oof_lung_thresholds": [round(value, 2) for value in np.arange(.50, .951, .05)],
                "oof_lesion_thresholds": [round(value, 3) for value in np.arange(.20, .601, .025)]}
    fingerprint = json_hash(protocol); root = ARTIFACTS / "crossval_segformer" / fingerprint
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / "protocol.json", protocol)
    lung_probabilities = {item["id"]: unlabeled_lung_probability(item["payload"]["rgb"])
                          for item in items}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_artifacts = []; oof = root / "oof"; oof.mkdir(exist_ok=True)
    for fold_index, validation_groups in enumerate(folds):
        fold_folder = root / f"fold_{fold_index}"; fold_folder.mkdir(exist_ok=True)
        training = [item for item in items if item["group"] not in validation_groups]
        validation = [item for item in items if item["group"] in validation_groups]
        torch.manual_seed(SEED + fold_index); np.random.seed(SEED + fold_index)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED + fold_index)
        model = SegformerForSemanticSegmentation.from_pretrained(
            protocol["pretrained"], num_labels=3, ignore_mismatched_sizes=True, use_safetensors=True,
            cache_dir=ARTIFACTS / "model_cache", id2label={0: "background", 1: "normal_lung", 2: "lesion"},
            label2id={"background": 0, "normal_lung": 1, "lesion": 2}).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-7)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        rng = random.Random(SEED + fold_index); history = []; best_rank = (-1., -1., -1., -999.)
        for epoch in range(1, epochs + 1):
            model.train(); order = list(range(len(training))); rng.shuffle(order); losses = []
            for start in range(0, len(order), 2):
                samples = [_augment(training[index], rng) for index in order[start:start + 2]]
                image = torch.stack([sample[0] for sample in samples]).to(device)
                target = torch.stack([sample[1] for sample in samples]).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(pixel_values=image).logits; loss, components = _loss(logits, target)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
            scheduler.step(); boxed = _predict(model, validation, device)
            native = [_restore(probability[2], item["geometry"])
                      for item, probability in zip(validation, boxed)]
            candidates = []
            for threshold in protocol["epoch_lesion_thresholds"]:
                summary, _ = _score_native(validation, native,
                    [lung_probabilities[item["id"]] for item in validation], .90, threshold)
                candidates.append({"lesion_threshold": threshold, "summary": summary})
            winner = max(candidates, key=lambda value: _rank(value["summary"])); rank = _rank(winner["summary"])
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "winner": winner,
                            "last_batch_components": components,
                            "learning_rate": optimizer.param_groups[0]["lr"]})
            checkpoint = {"version": VERSION, "fold": fold_index, "epoch": epoch,
                          "model": model.state_dict(), "config": model.config.to_dict(),
                          "protocol_fingerprint": fingerprint, "winner": winner, "rank": rank}
            torch.save(checkpoint, fold_folder / "last.pt")
            if rank > best_rank: best_rank = rank; torch.save(checkpoint, fold_folder / "best.pt")
            atomic_json(fold_folder / "history.json", history)
            print(f"v25 fold {fold_index + 1}/4 epoch {epoch}/{epochs} loss={np.mean(losses):.4f} "
                  f"floor={rank[0]:.4f} dice={rank[1]:.4f} lung={rank[2]:.4f}", flush=True)
        best_path = fold_folder / "best.pt"; best = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best["model"]); boxed = _predict(model, validation, device)
        for item, probability in zip(validation, boxed):
            np.savez_compressed(oof / f"{item['id']}.npz",
                                lesion=_restore(probability[2], item["geometry"]),
                                lung=lung_probabilities[item["id"]], fold=fold_index)
        artifact = {"fold": fold_index, "training_groups": sorted({item["group"] for item in training}),
                    "validation_groups": validation_groups, "training_images": len(training),
                    "validation_images": len(validation), "best_epoch": best["epoch"],
                    "checkpoint": str(best_path.relative_to(PROJECT_ROOT)),
                    "checkpoint_sha256": sha256(best_path)}
        atomic_json(fold_folder / "artifact.json", artifact); fold_artifacts.append(artifact)
    oof_items = [item_by_id[row["id"]] for row in metadata]; oof_lesion = []; oof_lung = []
    for item in oof_items:
        with np.load(oof / f"{item['id']}.npz", allow_pickle=False) as values:
            oof_lesion.append(values["lesion"]); oof_lung.append(values["lung"])
    candidates = []
    for lung_threshold in protocol["oof_lung_thresholds"]:
        for lesion_threshold in protocol["oof_lesion_thresholds"]:
            summary, _ = _score_native(oof_items, oof_lesion, oof_lung, lung_threshold, lesion_threshold)
            candidates.append({"lung_threshold": lung_threshold, "lesion_threshold": lesion_threshold,
                               "summary": summary})
    winner = max(candidates, key=lambda value: _rank(value["summary"])); metrics = winner["summary"]["pooled_lesion"]
    checks = {"lesion_dice_at_least_0_80": metrics["dice"] >= .80,
              "precision_at_least_0_80": metrics["precision"] >= .80,
              "recall_at_least_0_80": metrics["recall"] >= .80,
              "specificity_at_least_0_80": metrics["specificity"] >= .80,
              "lung_dice_at_least_0_90": winner["summary"]["pooled_lung"]["dice"] >= .90}
    report = {"status": "done", "created_at": now(), "version": VERSION, "protocol": protocol,
              "fold_artifacts": fold_artifacts, "oof_winner": winner,
              "oof_gate": {"passed": all(checks.values()), "checks": checks},
              "candidate_count": len(candidates), "test_payload_opened": False}
    atomic_json(root / "report.json", report)
    if report["oof_gate"]["passed"]:
        candidate = {"status": "frozen_after_oof_validation", "created_at": now(), "version": VERSION,
                     "selection_role": "four_fold_group_oof", "lesion_postprocess": "crossval_ensemble_v25",
                     "fold_checkpoints": [{"path": fold["checkpoint"], "sha256": fold["checkpoint_sha256"]}
                                          for fold in fold_artifacts],
                     "lung_threshold": winner["lung_threshold"],
                     "lesion_threshold": winner["lesion_threshold"], "oof_summary": winner["summary"],
                     "oof_gate": report["oof_gate"], "test_payload_opened": False}
        candidate["candidate_fingerprint"] = json_hash(candidate)
        folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
        atomic_json(folder / "candidate.json", candidate)
        atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                    {"path": str((folder / "candidate.json").relative_to(PROJECT_ROOT)),
                     "sha256": sha256(folder / "candidate.json"),
                     "candidate_fingerprint": candidate["candidate_fingerprint"]})
        atomic_json(ARTIFACTS / "test_unlock.json",
                    {"created_at": now(), "confirmation_gate_passed": True,
                     "gate_source": "four_fold_group_oof", "candidate_fingerprint": candidate["candidate_fingerprint"],
                     "crossval_fingerprint": fingerprint})
        report["candidate"] = candidate; atomic_json(root / "report.json", report)
    return report


def evaluate_crossval_store(candidate: dict, store: ReferenceStore) -> tuple[dict, list[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = [_prepare(store.load(row)) for row in store.rows]
    accumulated = [np.zeros(item["payload"]["rgb"].shape[:2], np.float32) for item in items]
    for checkpoint in candidate["fold_checkpoints"]:
        model = _load({"checkpoint": checkpoint["path"], "checkpoint_sha256": checkpoint["sha256"]}, device)
        probabilities = _predict(model, items, device)
        for index, (item, probability) in enumerate(zip(items, probabilities)):
            accumulated[index] += _restore(probability[2], item["geometry"]) / len(candidate["fold_checkpoints"])
    lungs = [unlabeled_lung_probability(item["payload"]["rgb"]) for item in items]
    return _score_native(items, accumulated, lungs, candidate["lung_threshold"], candidate["lesion_threshold"])
