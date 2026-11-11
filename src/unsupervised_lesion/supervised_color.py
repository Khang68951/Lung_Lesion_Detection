from __future__ import annotations

import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from .evaluate import _measure, summarize
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256, within_project
from .reference import ReferenceStore
from .supervised_segmentation import (_prepare, _predict, _restore, load_supervised_candidate,
                                      split_calibration_groups)
from .segmentation import color_regions


VERSION = "v20_supervised_color_texture"
SEED = 2004


def color_features(rgb: np.ndarray, lung: np.ndarray) -> np.ndarray:
    value = rgb.astype(np.float32) / 255.0
    total = value.sum(axis=2, keepdims=True) + 1e-4
    chromaticity = value / total
    lab_raw = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = np.stack([lab_raw[..., 0] / 255.0,
                    (lab_raw[..., 1] - 128.0) / 128.0,
                    (lab_raw[..., 2] - 128.0) / 128.0], axis=2)
    hsv_raw = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv = np.stack([hsv_raw[..., 0] / 180.0, hsv_raw[..., 1] / 255.0,
                    hsv_raw[..., 2] / 255.0], axis=2)
    features = [value, chromaticity, lab, hsv]
    for sigma in (5.0, 15.0):
        local = cv2.GaussianBlur(lab, (0, 0), sigma)
        features.append(lab - local)
        local_l2 = cv2.GaussianBlur(lab[..., 0] ** 2, (0, 0), sigma)
        features.append(np.sqrt(np.maximum(local_l2 - local[..., 0] ** 2, 0))[..., None])
    gradient_x = cv2.Sobel(lab[..., 0], cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(lab[..., 0], cv2.CV_32F, 0, 1, ksize=3)
    features.append(np.hypot(gradient_x, gradient_y)[..., None])
    if lung.any():
        median = np.median(lab[lung], axis=0)
        scale = np.median(np.abs(lab[lung] - median), axis=0) + .02
    else:
        median = np.zeros(3, np.float32); scale = np.ones(3, np.float32)
    features.append((lab - median) / scale)
    return np.nan_to_num(np.concatenate(features, axis=2), copy=False).astype(np.float32)


def _sample(features: np.ndarray, target: np.ndarray, valid: np.ndarray,
            rng: np.random.Generator, per_class: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    flat_valid = np.flatnonzero(valid.ravel())
    flat_target = target.ravel()
    positive = flat_valid[flat_target[flat_valid]]
    negative = flat_valid[~flat_target[flat_valid]]
    positive = rng.choice(positive, min(per_class, len(positive)), replace=False)
    negative = rng.choice(negative, min(per_class, len(negative)), replace=False)
    indices = np.concatenate([positive, negative]); rng.shuffle(indices)
    return features.reshape(-1, features.shape[2])[indices], flat_target[indices].astype(np.uint8)


def _probability(classifier: HistGradientBoostingClassifier, features: np.ndarray,
                 lung: np.ndarray) -> np.ndarray:
    output = np.zeros(lung.shape, np.float32); indices = np.flatnonzero(lung.ravel())
    flat = features.reshape(-1, features.shape[2]); chunks = []
    for start in range(0, len(indices), 200_000):
        chunks.append(classifier.predict_proba(flat[indices[start:start + 200_000]])[:, 1])
    if chunks: output.ravel()[indices] = np.concatenate(chunks).astype(np.float32)
    return output


def aggregate_regions(rgb: np.ndarray, lung: np.ndarray, probability: np.ndarray,
                      segments: int, mode: str, threshold: float,
                      pixel_threshold: float = .40) -> np.ndarray:
    labels = color_regions(rgb, lung, segments); output = np.zeros_like(lung)
    for region_id in np.unique(labels[lung]):
        region = (labels == region_id) & lung
        if not region.any(): continue
        score = (float(probability[region].mean()) if mode == "mean" else
                 float((probability[region] >= pixel_threshold).mean()))
        if score >= threshold: output[region] = True
    return output


def _load_v19() -> dict:
    pointer = json.loads((ARTIFACTS / "frozen_candidate/current.json").read_text())
    path = within_project(pointer["path"])
    if sha256(path) != pointer["sha256"]: raise ValueError("Candidate pointer checksum changed")
    candidate = json.loads(path.read_text())
    if candidate.get("version") != "v19_supervised_calibration_deeplab":
        raise ValueError("v20 requires the frozen v19 lung model")
    return candidate


def fit_supervised_color() -> dict:
    base = _load_v19(); store = ReferenceStore("calibration")
    train_groups, validation_groups = split_calibration_groups(store.rows)
    train_items = [_prepare(store.load(row)) for row in store.rows if row["group"] in train_groups]
    validation_items = [_prepare(store.load(row)) for row in store.rows if row["group"] in validation_groups]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = load_supervised_candidate(base, device)
    train_probabilities = _predict(network, train_items, device)
    validation_probabilities = _predict(network, validation_items, device)
    rng = np.random.default_rng(SEED); samples = []; targets = []
    for item, probability in zip(train_items, train_probabilities):
        lung = probability[0] >= base["lung_threshold"]
        features = color_features(item["rgb"], lung)
        valid = item["valid"] & ~item["ignore_lesion"] & item["lung"]
        x, y = _sample(features, item["lesion"], valid, rng)
        samples.append(x); targets.append(y)
    x = np.concatenate(samples); y = np.concatenate(targets)
    classifier = HistGradientBoostingClassifier(
        learning_rate=.08, max_iter=300, max_leaf_nodes=31, min_samples_leaf=35,
        l2_regularization=1.0, validation_fraction=None, random_state=SEED)
    classifier.fit(x, y)
    protocol = {"version": VERSION, "seed": SEED, "base_candidate_fingerprint": base["candidate_fingerprint"],
                "classifier": "sklearn HistGradientBoostingClassifier", "training_samples": len(y),
                "positive_samples": int(y.sum()), "negative_samples": int((y == 0).sum()),
                "training_groups": train_groups, "internal_validation_groups": validation_groups,
                "confirmation_images_used_for_fit_or_selection": 0, "sealed_test_images_opened": 0,
                "thresholds": [round(value, 2) for value in np.arange(.10, .91, .05)]}
    fingerprint = json_hash(protocol); folder = ARTIFACTS / "supervised_color" / fingerprint
    folder.mkdir(parents=True, exist_ok=True); atomic_json(folder / "protocol.json", protocol)
    model_path = folder / "model.joblib"; joblib.dump(classifier, model_path)
    cache = folder / "internal_validation_probabilities"; cache.mkdir(exist_ok=True)
    restored = []
    for item, probability in zip(validation_items, validation_probabilities):
        lung = probability[0] >= base["lung_threshold"]
        lesion_probability = _probability(classifier, color_features(item["rgb"], lung), lung)
        native_lung = _restore(probability[0], item["geometry"])
        native_lesion = _restore(lesion_probability, item["geometry"])
        np.savez_compressed(cache / f"{item['id']}.npz", lung=native_lung, lesion=native_lesion)
        restored.append((native_lung, native_lesion))
    candidates = []
    for threshold in protocol["thresholds"]:
        rows = []
        for item, (lung_probability, lesion_probability) in zip(validation_items, restored):
            lung = lung_probability >= base["lung_threshold"]
            lesion = (lesion_probability >= threshold) & lung
            rows.append(_measure(item["payload"], lung, lesion))
        summary = summarize(rows); metrics = summary["pooled_lesion"]
        rank = (min(metrics["precision"] or 0, metrics["recall"] or 0, metrics["specificity"] or 0),
                metrics["dice"] or 0, -(summary["ratio_mae_percentage_points"] or 999))
        candidates.append({"lesion_threshold": threshold, "summary": summary, "rank": rank})
    winner = max(candidates, key=lambda item: tuple(item["rank"]))
    artifact = {"status": "fit_complete", "created_at": now(), "version": VERSION,
                "artifact_fingerprint": fingerprint, "model": str(model_path.relative_to(PROJECT_ROOT)),
                "model_sha256": sha256(model_path), "protocol": protocol,
                "internal_validation_winner": winner}
    atomic_json(folder / "artifact.json", artifact)
    candidate = {"status": "frozen_after_internal_validation", "created_at": now(), "version": VERSION,
                 "selection_role": "calibration_group_holdout_only",
                 "lesion_postprocess": "supervised_color_v20",
                 "base_candidate_fingerprint": base["candidate_fingerprint"],
                 "lung_checkpoint": base["checkpoint"], "lung_checkpoint_sha256": base["checkpoint_sha256"],
                 "lung_threshold": base["lung_threshold"], "color_model": artifact["model"],
                 "color_model_sha256": artifact["model_sha256"],
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
    return {"artifact": artifact, "candidate": candidate, "candidate_count": len(candidates)}


def calibrate_color_regions() -> dict:
    base = _load_current_v20(); model_path = within_project(base["color_model"])
    folder = model_path.parent; protocol = json.loads((folder / "protocol.json").read_text())
    validation_groups = set(protocol["internal_validation_groups"]); store = ReferenceStore("calibration")
    items = [store.load(row) for row in store.rows if row["group"] in validation_groups]
    cache = folder / "internal_validation_probabilities"; candidates = []
    configurations = []
    for segments in (80, 160, 240):
        for threshold in (.20, .30, .40, .50, .60, .70):
            configurations.append((segments, "mean", threshold))
        for threshold in (.25, .40, .55, .70):
            configurations.append((segments, "support", threshold))
    for segments, mode, threshold in configurations:
        rows = []
        for payload in items:
            with np.load(cache / f"{payload['row']['id']}.npz", allow_pickle=False) as values:
                lung_probability = values["lung"]; lesion_probability = values["lesion"]
            lung = lung_probability >= base["lung_threshold"]
            lesion = aggregate_regions(payload["rgb"], lung, lesion_probability, segments, mode,
                                       threshold, base["lesion_threshold"])
            rows.append(_measure(payload, lung, lesion))
        summary = summarize(rows); metrics = summary["pooled_lesion"]
        rank = (min(metrics["precision"] or 0, metrics["recall"] or 0, metrics["specificity"] or 0),
                metrics["dice"] or 0, -(summary["ratio_mae_percentage_points"] or 999))
        candidates.append({"segments": segments, "mode": mode, "threshold": threshold,
                           "summary": summary, "rank": rank})
    winner = max(candidates, key=lambda item: tuple(item["rank"]))
    direct = base["internal_validation_summary"]; direct_metrics = direct["pooled_lesion"]
    direct_rank = (min(direct_metrics["precision"], direct_metrics["recall"], direct_metrics["specificity"]),
                   direct_metrics["dice"], -(direct["ratio_mae_percentage_points"] or 999))
    report = {"status": "done", "created_at": now(), "version": "v20b_cached_color_regions",
              "base_candidate_fingerprint": base["candidate_fingerprint"],
              "candidate_count": len(candidates), "winner": winner,
              "direct_rank": direct_rank, "replacement": tuple(winner["rank"]) > direct_rank,
              "confirmation_images_opened": 0, "test_payload_opened": False}
    fingerprint = json_hash({key: value for key, value in report.items() if key != "created_at"})
    run_folder = ARTIFACTS.parent / "runs" / f"calibration_supervised_color_regions_{fingerprint[:16]}"
    atomic_json(run_folder / "report.json", {**report, "candidates": candidates})
    if report["replacement"]:
        candidate = {**{key: value for key, value in base.items()
                        if key not in {"created_at", "candidate_fingerprint", "internal_validation_summary"}},
                     "created_at": now(), "version": "v20b_supervised_color_regions",
                     "lesion_postprocess": "supervised_color_region_v20b",
                     "region_segments": winner["segments"], "region_mode": winner["mode"],
                     "region_threshold": winner["threshold"],
                     "internal_validation_summary": winner["summary"]}
        candidate["candidate_fingerprint"] = json_hash(candidate)
        candidate_folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
        atomic_json(candidate_folder / "candidate.json", candidate)
        atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                    {"path": str((candidate_folder / "candidate.json").relative_to(PROJECT_ROOT)),
                     "sha256": sha256(candidate_folder / "candidate.json"),
                     "candidate_fingerprint": candidate["candidate_fingerprint"]})
        report["candidate"] = candidate
    return report


def _load_current_v20() -> dict:
    pointer = json.loads((ARTIFACTS / "frozen_candidate/current.json").read_text())
    path = within_project(pointer["path"])
    if sha256(path) != pointer["sha256"]: raise ValueError("Candidate pointer checksum changed")
    candidate = json.loads(path.read_text())
    if candidate.get("version") != VERSION: raise ValueError("v20b requires direct v20")
    return candidate


def evaluate_color_store(candidate: dict, store: ReferenceStore) -> tuple[dict, list[dict]]:
    base = {"checkpoint": candidate["lung_checkpoint"],
            "checkpoint_sha256": candidate["lung_checkpoint_sha256"]}
    model_path = within_project(candidate["color_model"])
    if sha256(model_path) != candidate["color_model_sha256"]: raise ValueError("Color model changed")
    classifier = joblib.load(model_path); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = load_supervised_candidate(base, device)
    items = [_prepare(store.load(row)) for row in store.rows]; probabilities = _predict(network, items, device)
    rows = []
    for item, probability in zip(items, probabilities):
        lung_box = probability[0] >= candidate["lung_threshold"]
        lesion_box = _probability(classifier, color_features(item["rgb"], lung_box), lung_box)
        lung = _restore(probability[0], item["geometry"]) >= candidate["lung_threshold"]
        lesion_probability = _restore(lesion_box, item["geometry"])
        if candidate.get("lesion_postprocess") == "supervised_color_region_v20b":
            lesion = aggregate_regions(item["payload"]["rgb"], lung, lesion_probability,
                                       candidate["region_segments"], candidate["region_mode"],
                                       candidate["region_threshold"], candidate["lesion_threshold"])
        else:
            lesion = (lesion_probability >= candidate["lesion_threshold"]) & lung
        rows.append(_measure(item["payload"], lung, lesion))
    return summarize(rows), rows
