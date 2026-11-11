from __future__ import annotations

import json

import cv2
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .crossval_segformer import _score_native
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256
from .reference_v2 import DevelopmentReferenceStore
from .segformer_segmentation import _rank
from .supervised_color import _sample, color_features
from .supervised_segmentation import _prepare


VERSION = "v42_dual_semantic_group_heldout_pixel_meta_reference_v2"
SEED = 4204


def dual_meta_features(rgb: np.ndarray, lung: np.ndarray, v37: np.ndarray,
                       v40: np.ndarray, region: np.ndarray) -> np.ndarray:
    blend = .5 * v37 + .5 * v40
    semantic = [v37, v40, blend, region, v37 - v40, region - blend]
    for score in (v37, v40, blend):
        semantic.extend([cv2.GaussianBlur(score, (0, 0), 5.0),
                         cv2.GaussianBlur(score, (0, 0), 15.0)])
    extra = np.stack(semantic, axis=2).astype(np.float32)
    return np.concatenate([color_features(rgb, lung), extra], axis=2)


def _predict(model, features: np.ndarray, lung: np.ndarray) -> np.ndarray:
    result = np.zeros(lung.shape, np.float32); indices = np.flatnonzero(lung.ravel())
    flat = features.reshape(-1, features.shape[2]); chunks = []
    for start in range(0, len(indices), 200_000):
        chunks.append(model.predict_proba(flat[indices[start:start + 200_000]])[:, 1])
    if chunks:
        result.ravel()[indices] = np.concatenate(chunks).astype(np.float32)
    return result


def _root(parent: str, version: str):
    paths = []
    for path in (ARTIFACTS / parent).glob("*/report.json"):
        if json.loads(path.read_text()).get("version") == version:
            paths.append(path.parent)
    if len(paths) != 1:
        raise ValueError(f"Expected one {version} run, found {len(paths)}")
    return paths[0]


def fit_dual_meta() -> dict:
    v37_root = _root("binary_crossval", "v37_binary_segformer_b1_relative_lab_reference_v2")
    v40_root = _root("dense_dino_crossval", "v40_frozen_dense_dinov2_colour_decoder_reference_v2")
    v41_root = _root("region_consensus_v41", "v41_cached_colour_region_consensus_reference_v2")
    v37_report = json.loads((v37_root / "report.json").read_text())
    folds = v37_report["protocol"]["fold_groups"]
    store = DevelopmentReferenceStore(); payloads = [store.load(row) for row in store.rows]
    data = {}; sampled = {}; rng = np.random.default_rng(SEED)
    for payload in payloads:
        identifier = payload["row"]["id"]
        with np.load(v37_root / "oof" / f"{identifier}.npz", allow_pickle=False) as values:
            p37 = values["lesion"].astype(np.float32); lung_probability = values["lung"].astype(np.float32)
        with np.load(v40_root / "oof" / f"{identifier}.npz", allow_pickle=False) as values:
            p40 = values["lesion"].astype(np.float32)
        with np.load(v41_root / "winner_scores" / f"{identifier}.npz", allow_pickle=False) as values:
            region = values["lesion"].astype(np.float32)
        lung = lung_probability >= .50
        features = dual_meta_features(payload["rgb"], lung, p37, p40, region)
        valid = payload["lung"] & payload["valid"] & ~payload["ignore_lung"] & ~payload["ignore_lesion"]
        sampled[identifier] = _sample(features, payload["lesion"] & payload["lung"], valid, rng, 6000)
        data[identifier] = {"payload": payload, "features": features, "lung": lung_probability}
    protocol = {
        "version": VERSION, "seed": SEED, "reference_fingerprint": store.manifest["fingerprint"],
        "v37_root": str(v37_root.relative_to(PROJECT_ROOT)),
        "v40_root": str(v40_root.relative_to(PROJECT_ROOT)),
        "v41_root": str(v41_root.relative_to(PROJECT_ROOT)), "fold_groups": folds,
        "maximum_samples_per_class_per_image": 6000,
        "classifier": {"name": "HistGradientBoostingClassifier", "learning_rate": .05,
                       "max_iter": 300, "max_leaf_nodes": 15, "min_samples_leaf": 75,
                       "l2_regularization": 3.0, "validation_fraction": None},
        "lung_thresholds": [round(x, 2) for x in np.arange(.50, .951, .05)],
        "lesion_thresholds": [round(x, 3) for x in np.arange(.10, .901, .025)],
        "sealed_test_images_opened": 0,
    }
    fingerprint = json_hash(protocol); root = ARTIFACTS / "dual_meta_v42" / fingerprint
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / "protocol.json", protocol)
    oof = root / "oof"; oof.mkdir(exist_ok=True); models = []
    for fold_index, validation_groups in enumerate(folds):
        train_ids = [p["row"]["id"] for p in payloads if p["row"]["group"] not in validation_groups]
        validation = [p for p in payloads if p["row"]["group"] in validation_groups]
        x = np.concatenate([sampled[i][0] for i in train_ids]); y = np.concatenate([sampled[i][1] for i in train_ids])
        model = HistGradientBoostingClassifier(
            learning_rate=.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=75,
            l2_regularization=3.0, validation_fraction=None,
            random_state=SEED + fold_index).fit(x, y)
        path = root / f"meta_fold_{fold_index}.joblib"; joblib.dump(model, path)
        models.append({"fold": fold_index, "path": str(path.relative_to(PROJECT_ROOT)),
                       "sha256": sha256(path), "validation_groups": validation_groups,
                       "training_groups": sorted({data[i]["payload"]["row"]["group"] for i in train_ids}),
                       "training_samples": int(len(y))})
        for payload in validation:
            identifier = payload["row"]["id"]; record = data[identifier]
            probability = _predict(model, record["features"], record["lung"] >= .50)
            np.savez_compressed(oof / f"{identifier}.npz", lesion=probability,
                                lung=record["lung"], fold=fold_index)
        print(f"{VERSION} fold {fold_index + 1}/4 samples={len(y)} images={len(validation)}", flush=True)
    items = [_prepare(payload) for payload in payloads]; probabilities = []; lungs = []
    for item in items:
        with np.load(oof / f"{item['id']}.npz", allow_pickle=False) as values:
            probabilities.append(values["lesion"]); lungs.append(values["lung"])
    candidates = []
    for lung_threshold in protocol["lung_thresholds"]:
        for lesion_threshold in protocol["lesion_thresholds"]:
            summary, _ = _score_native(items, probabilities, lungs, lung_threshold, lesion_threshold)
            candidates.append({"lung_threshold": lung_threshold, "lesion_threshold": lesion_threshold,
                               "summary": summary})
    winner = max(candidates, key=lambda x: _rank(x["summary"])); metrics = winner["summary"]["pooled_lesion"]
    checks = {"lesion_dice_at_least_0_80": metrics["dice"] >= .80,
              "precision_at_least_0_80": metrics["precision"] >= .80,
              "recall_at_least_0_80": metrics["recall"] >= .80,
              "specificity_at_least_0_80": metrics["specificity"] >= .80,
              "lung_dice_at_least_0_90": winner["summary"]["pooled_lung"]["dice"] >= .90}
    report = {"status": "done", "created_at": now(), "version": VERSION, "protocol": protocol,
              "meta_models": models, "oof_winner": winner,
              "oof_gate": {"passed": all(checks.values()), "checks": checks},
              "candidate_count": len(candidates), "test_payload_opened": False}
    atomic_json(root / "report.json", report)
    return report
