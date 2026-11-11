from __future__ import annotations

import json

import cv2
import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from .crossval_segformer import _score_native
from .evaluate import _measure, summarize
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256, within_project
from .reference import ReferenceStore
from .segformer_segmentation import _load, _predict, _restore, _rank
from .segmentation import unlabeled_lung_probability
from .supervised_color import _sample, color_features
from .supervised_segmentation import _prepare


VERSION = "v26_group_heldout_semantic_color_stack"
SEED = 2604


def meta_features(rgb: np.ndarray, lung: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    base = color_features(rgb, lung)
    local5 = cv2.GaussianBlur(semantic, (0, 0), 5.0)
    local15 = cv2.GaussianBlur(semantic, (0, 0), 15.0)
    extra = np.stack([semantic, local5, local15, semantic - local5, semantic - local15], axis=2)
    return np.concatenate([base, extra.astype(np.float32)], axis=2)


def _predict_meta(model, features: np.ndarray, lung: np.ndarray) -> np.ndarray:
    output = np.zeros(lung.shape, np.float32); indices = np.flatnonzero(lung.ravel())
    flat = features.reshape(-1, features.shape[2]); values = []
    for start in range(0, len(indices), 200_000):
        values.append(model.predict_proba(flat[indices[start:start + 200_000]])[:, 1])
    if values: output.ravel()[indices] = np.concatenate(values).astype(np.float32)
    return output


def _crossval_root() -> tuple[dict, object]:
    matches = []
    for path in (ARTIFACTS / "crossval_segformer").glob("*/report.json"):
        report = json.loads(path.read_text())
        if report.get("version") == "v25_development_group_crossval": matches.append((report, path.parent))
    if len(matches) != 1: raise ValueError(f"Expected one v25 report, found {len(matches)}")
    return matches[0]


def fit_stacked_meta() -> dict:
    crossval, crossval_root = _crossval_root(); folds = crossval["protocol"]["fold_groups"]
    stores = [ReferenceStore("calibration"), ReferenceStore("confirmation")]
    payloads = [store.load(row) for store in stores for row in store.rows]
    by_id = {payload["row"]["id"]: payload for payload in payloads}
    base_scores = {}; lung_scores = {}; sampled = {}; rng = np.random.default_rng(SEED)
    for payload in payloads:
        identifier = payload["row"]["id"]
        with np.load(crossval_root / "oof" / f"{identifier}.npz", allow_pickle=False) as values:
            semantic = values["lesion"].astype(np.float32); lung_probability = values["lung"].astype(np.float32)
        base_scores[identifier] = semantic; lung_scores[identifier] = lung_probability
        lung = lung_probability >= .50; features = meta_features(payload["rgb"], lung, semantic)
        valid = payload["lung"] & ~payload["ignore_lung"] & ~payload["ignore_lesion"]
        sampled[identifier] = _sample(features, payload["lesion"] & payload["lung"], valid, rng, 6000)
    protocol = {"version": VERSION, "seed": SEED, "source_crossval": str(crossval_root.relative_to(PROJECT_ROOT)),
                "source_fold_groups": folds, "meta_classifier": "HistGradientBoostingClassifier",
                "maximum_samples_per_class_per_image": 6000, "sealed_test_images_opened": 0,
                "lung_thresholds": [round(value, 2) for value in np.arange(.50, .951, .05)],
                "lesion_thresholds": [round(value, 3) for value in np.arange(.10, .901, .025)]}
    fingerprint = json_hash(protocol); root = ARTIFACTS / "stacked_meta" / fingerprint
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / "protocol.json", protocol)
    oof_folder = root / "oof"; oof_folder.mkdir(exist_ok=True); models = []
    for fold_index, validation_groups in enumerate(folds):
        train_ids = [payload["row"]["id"] for payload in payloads
                     if payload["row"]["group"] not in validation_groups]
        validation = [payload for payload in payloads if payload["row"]["group"] in validation_groups]
        x = np.concatenate([sampled[identifier][0] for identifier in train_ids])
        y = np.concatenate([sampled[identifier][1] for identifier in train_ids])
        model = HistGradientBoostingClassifier(learning_rate=.07, max_iter=250, max_leaf_nodes=31,
            min_samples_leaf=45, l2_regularization=1.5, validation_fraction=None,
            random_state=SEED + fold_index).fit(x, y)
        path = root / f"meta_fold_{fold_index}.joblib"; joblib.dump(model, path)
        models.append({"fold": fold_index, "path": str(path.relative_to(PROJECT_ROOT)),
                       "sha256": sha256(path), "training_groups": sorted({by_id[i]["row"]["group"] for i in train_ids}),
                       "validation_groups": validation_groups, "training_samples": len(y)})
        for payload in validation:
            identifier = payload["row"]["id"]; lung = lung_scores[identifier] >= .50
            probability = _predict_meta(model, meta_features(payload["rgb"], lung, base_scores[identifier]), lung)
            np.savez_compressed(oof_folder / f"{identifier}.npz", lesion=probability,
                                lung=lung_scores[identifier], fold=fold_index)
        print(f"v26 meta fold {fold_index + 1}/4 fit_samples={len(y)} val_images={len(validation)}", flush=True)
    stacked = []; lungs = []
    for payload in payloads:
        with np.load(oof_folder / f"{payload['row']['id']}.npz", allow_pickle=False) as values:
            stacked.append(values["lesion"]); lungs.append(values["lung"])
    items = [{"payload": payload} for payload in payloads]; candidates = []
    for lung_threshold in protocol["lung_thresholds"]:
        for lesion_threshold in protocol["lesion_thresholds"]:
            summary, _ = _score_native(items, stacked, lungs, lung_threshold, lesion_threshold)
            candidates.append({"lung_threshold": lung_threshold, "lesion_threshold": lesion_threshold,
                               "summary": summary})
    winner = max(candidates, key=lambda value: _rank(value["summary"])); metric = winner["summary"]["pooled_lesion"]
    checks = {"lesion_dice_at_least_0_80": metric["dice"] >= .80,
              "precision_at_least_0_80": metric["precision"] >= .80,
              "recall_at_least_0_80": metric["recall"] >= .80,
              "specificity_at_least_0_80": metric["specificity"] >= .80,
              "lung_dice_at_least_0_90": winner["summary"]["pooled_lung"]["dice"] >= .90}
    report = {"status": "done", "created_at": now(), "version": VERSION, "protocol": protocol,
              "meta_models": models, "oof_winner": winner,
              "oof_gate": {"passed": all(checks.values()), "checks": checks},
              "candidate_count": len(candidates), "test_payload_opened": False}
    atomic_json(root / "report.json", report)
    if report["oof_gate"]["passed"]:
        candidate = {"status": "frozen_after_oof_validation", "created_at": now(), "version": VERSION,
            "selection_role": "four_fold_group_oof_stacked", "lesion_postprocess": "stacked_meta_v26",
            "base_checkpoints": [{"path": fold["checkpoint"], "sha256": fold["checkpoint_sha256"]}
                                 for fold in crossval["fold_artifacts"]],
            "meta_models": [{"path": model["path"], "sha256": model["sha256"]} for model in models],
            "lung_threshold": winner["lung_threshold"], "lesion_threshold": winner["lesion_threshold"],
            "oof_summary": winner["summary"], "oof_gate": report["oof_gate"], "test_payload_opened": False}
        candidate["candidate_fingerprint"] = json_hash(candidate)
        folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
        atomic_json(folder / "candidate.json", candidate)
        atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                    {"path": str((folder / "candidate.json").relative_to(PROJECT_ROOT)),
                     "sha256": sha256(folder / "candidate.json"),
                     "candidate_fingerprint": candidate["candidate_fingerprint"]})
        atomic_json(ARTIFACTS / "test_unlock.json", {"created_at": now(),
            "confirmation_gate_passed": True, "gate_source": "four_fold_group_oof_stacked",
            "candidate_fingerprint": candidate["candidate_fingerprint"], "stack_fingerprint": fingerprint})
        report["candidate"] = candidate; atomic_json(root / "report.json", report)
    return report


def evaluate_stacked_store(candidate: dict, store: ReferenceStore) -> tuple[dict, list[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = [_prepare(store.load(row)) for row in store.rows]
    semantic = [np.zeros(item["payload"]["rgb"].shape[:2], np.float32) for item in items]
    for checkpoint in candidate["base_checkpoints"]:
        probabilities = _predict(_load({"checkpoint": checkpoint["path"],
            "checkpoint_sha256": checkpoint["sha256"]}, device), items, device)
        for index, (item, probability) in enumerate(zip(items, probabilities)):
            semantic[index] += _restore(probability[2], item["geometry"]) / len(candidate["base_checkpoints"])
    lungs = [unlabeled_lung_probability(item["payload"]["rgb"]) for item in items]
    stacked = [np.zeros_like(score) for score in semantic]
    for record in candidate["meta_models"]:
        path = within_project(record["path"])
        if sha256(path) != record["sha256"]: raise ValueError("Meta model checksum changed")
        model = joblib.load(path)
        for index, item in enumerate(items):
            roi = lungs[index] >= .50
            features = meta_features(item["payload"]["rgb"], roi, semantic[index])
            stacked[index] += _predict_meta(model, features, roi) / len(candidate["meta_models"])
    return _score_native(items, stacked, lungs, candidate["lung_threshold"], candidate["lesion_threshold"])
