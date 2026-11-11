from __future__ import annotations

import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .crossval_segformer import _score_native
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256
from .reference_v2 import DevelopmentReferenceStore
from .segformer_segmentation import _rank
from .style_ensemble_v46 import BASE_WEIGHTS, PARENTS, STYLE_VERSION, _root
from .supervised_color import _probability, _sample, color_features
from .supervised_segmentation import _prepare


VERSION = "v69_strong_colour_semantic_group_oof"
SEED = 6904


def strong_meta_features(rgb: np.ndarray, lung: np.ndarray, old: np.ndarray,
                         v64: np.ndarray, v67: np.ndarray) -> np.ndarray:
    ensemble = (.65 * old + .10 * v64 + .25 * v67).astype(np.float32)
    smooth5 = cv2.GaussianBlur(ensemble, (0, 0), 5.0)
    smooth15 = cv2.GaussianBlur(ensemble, (0, 0), 15.0)
    height, width = lung.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    xx = (xx / max(width - 1, 1) - .5)[..., None]
    yy = (yy / max(height - 1, 1) - .5)[..., None]
    inside = ensemble[lung]
    quantiles = (np.quantile(inside, [.25, .50, .75, .90]).astype(np.float32)
                 if inside.size else np.zeros(4, np.float32))
    image_planes = np.broadcast_to(quantiles, (height, width, 4))
    semantic = np.stack([old, v64, v67, ensemble, smooth5, smooth15,
                         ensemble - smooth5, ensemble - smooth15], axis=2)
    return np.concatenate([color_features(rgb, lung), semantic, xx, yy, image_planes],
                          axis=2).astype(np.float32)


def _v64_report() -> tuple[dict, Path]:
    matches = []
    for path in (ARTIFACTS / "multiview_crossval_v64").glob("*/report.json"):
        report = json.loads(path.read_text())
        if report.get("version") == "v64_supervised_v54_multiview_group_oof":
            matches.append((report, path.parent))
    if len(matches) != 1:
        raise ValueError(f"Expected one V64 report, found {len(matches)}")
    return matches[0]


def fit_strong_meta_v69() -> dict:
    v64_report, _ = _v64_report()
    folds = v64_report["protocol"]["fold_groups"]
    tta_root = ARTIFACTS / "tta_multiview_v68"
    roots = {version: _root(PARENTS[version], version) for version in PARENTS}
    store = DevelopmentReferenceStore()
    items = [_prepare(store.load(row)) for row in store.rows]
    by_id = {item["id"]: item for item in items}
    planes: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rng = np.random.default_rng(SEED)
    for item in items:
        identifier = item["id"]
        base = None
        for version, weight in BASE_WEIGHTS.items():
            with np.load(roots[version] / "oof" / f"{identifier}.npz",
                         allow_pickle=False) as values:
                value = values["lesion"].astype(np.float32)
            base = value * weight if base is None else base + value * weight
        with np.load(roots[STYLE_VERSION] / "oof" / f"{identifier}.npz",
                     allow_pickle=False) as values:
            style = values["lesion"].astype(np.float32)
        old = (.75 * base + .25 * style).astype(np.float32)
        with np.load(tta_root / "v64" / f"{identifier}.npz", allow_pickle=False) as values:
            v64 = values["average"].astype(np.float32)
            lung = values["lung"].astype(bool)
        with np.load(tta_root / "v67" / f"{identifier}.npz", allow_pickle=False) as values:
            v67 = values["average"].astype(np.float32)
        planes[identifier] = old, v64, v67, lung
        features = strong_meta_features(item["payload"]["rgb"], lung, old, v64, v67)
        payload = item["payload"]
        valid = (payload["valid"] & ~payload["ignore_lung"] &
                 ~payload["ignore_lesion"] & payload["lung"])
        samples[identifier] = _sample(features, payload["lesion"] & payload["lung"],
                                      valid, rng, 10_000)

    protocol = {
        "version": VERSION, "seed": SEED, "fold_groups": folds,
        "maximum_samples_per_class_per_image": 10_000,
        "classifier": "HistGradientBoostingClassifier",
        "classifier_parameters": {"learning_rate": .05, "max_iter": 400,
            "max_leaf_nodes": 63, "min_samples_leaf": 35, "l2_regularization": 2.0},
        "sources": {version: sha256(root / "report.json") for version, root in roots.items()},
        "v64_report_sha256": sha256(next((ARTIFACTS / "multiview_crossval_v64").glob("*/report.json"))),
        "tta_root": str(tta_root.relative_to(PROJECT_ROOT)),
        "lesion_thresholds": [round(float(x), 3) for x in np.arange(.20, .801, .01)],
        "sealed_test_images_opened": 0,
    }
    fingerprint = json_hash(protocol)
    root = ARTIFACTS / "strong_meta_v69" / fingerprint
    oof = root / "oof"
    oof.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "protocol.json", protocol)
    models = []
    for fold_index, validation_groups in enumerate(folds):
        train_ids = [item["id"] for item in items
                     if item["payload"]["row"]["group"] not in validation_groups]
        validation = [item for item in items
                      if item["payload"]["row"]["group"] in validation_groups]
        x = np.concatenate([samples[identifier][0] for identifier in train_ids])
        y = np.concatenate([samples[identifier][1] for identifier in train_ids])
        classifier = HistGradientBoostingClassifier(
            learning_rate=.05, max_iter=400, max_leaf_nodes=63,
            min_samples_leaf=35, l2_regularization=2.0,
            validation_fraction=None, random_state=SEED + fold_index).fit(x, y)
        model_path = root / f"meta_fold_{fold_index}.joblib"
        joblib.dump(classifier, model_path)
        models.append({"fold": fold_index, "path": str(model_path.relative_to(PROJECT_ROOT)),
                       "sha256": sha256(model_path), "training_samples": int(len(y)),
                       "validation_groups": validation_groups})
        for item in validation:
            identifier = item["id"]
            old, v64, v67, lung = planes[identifier]
            features = strong_meta_features(item["payload"]["rgb"], lung, old, v64, v67)
            probability = _probability(classifier, features, lung)
            np.savez_compressed(oof / f"{identifier}.npz", lesion=probability,
                                lung=lung.astype(np.uint8), fold=fold_index)
        print(f"V69 fold {fold_index + 1}/{len(folds)} complete", flush=True)

    probabilities, lungs = [], []
    for item in items:
        with np.load(oof / f"{item['id']}.npz", allow_pickle=False) as values:
            probabilities.append(values["lesion"].astype(np.float32))
            lungs.append(values["lung"].astype(np.float32))
    candidates = []
    for threshold in protocol["lesion_thresholds"]:
        summary, _ = _score_native(items, probabilities, lungs, .50, threshold)
        candidates.append({"lung_threshold": .50, "lesion_threshold": threshold,
                           "summary": summary})
    winner = max(candidates, key=lambda value: _rank(value["summary"]))
    metric = winner["summary"]["pooled_lesion"]
    checks = {"lesion_dice_at_least_0_80": metric["dice"] >= .80,
              "precision_at_least_0_80": metric["precision"] >= .80,
              "recall_at_least_0_80": metric["recall"] >= .80,
              "specificity_at_least_0_80": metric["specificity"] >= .80,
              "lung_dice_at_least_0_90": winner["summary"]["pooled_lung"]["dice"] >= .90}
    report = {"status": "done", "created_at": now(), "version": VERSION,
              "protocol": protocol, "models": models, "winner": winner,
              "oof_gate": {"passed": all(checks.values()), "checks": checks},
              "candidate_count": len(candidates), "test_payload_opened": False,
              "results": candidates}
    atomic_json(root / "report.json", report)
    return report
