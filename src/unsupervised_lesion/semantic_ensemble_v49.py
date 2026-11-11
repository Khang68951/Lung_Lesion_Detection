from __future__ import annotations

import json

import numpy as np

from .evaluate import _measure, summarize
from .io import ARTIFACTS, atomic_json, json_hash, now, sha256
from .reference_v2 import DevelopmentReferenceStore
from .segformer_segmentation import _rank


VERSION = "v49_v47_medsiglip_cached_ensemble"
V47_ROOT = ARTIFACTS / "unmarked_pseudo_segformer" / "87d285122fe6009a0a193c45e3211ed9969feafaab02ca1af8a5d612556689f4"
V48_ROOT = ARTIFACTS / "medsiglip_v48_full" / "e9bb7c860dacb80113ec479a3bd96b6115a332e8dff5a4841095bfc8836c8b78"
V47_REPORT_SHA256 = "1b6a1a414294c6bda31a7459629ccda183dfdb97e4c7fc2981ee5a7f3461aaa3"
V48_REPORT_SHA256 = "c3c6e6d205818a132ee3add234b2e70d61e140fe6c70fe661f94de0eb962b29b"


def transform_medsiglip(score: np.ndarray) -> np.ndarray:
    value = np.asarray(score, np.float32)
    return 1.0 / (1.0 + np.exp(-(value - .309) / .006))


def blend_probability(v47: np.ndarray, medsiglip: np.ndarray, weight: float) -> np.ndarray:
    if not 0 <= weight <= 1 or v47.shape != medsiglip.shape:
        raise ValueError("Invalid V49 blend")
    return (1 - weight) * np.asarray(v47, np.float32) + weight * transform_medsiglip(medsiglip)


def calibrate_semantic_ensemble_v49() -> dict:
    if sha256(V47_ROOT / "report.json") != V47_REPORT_SHA256:
        raise ValueError("V47 report changed")
    if sha256(V48_ROOT / "report.json") != V48_REPORT_SHA256:
        raise ValueError("V48 report changed")
    v47_report = json.loads((V47_ROOT / "report.json").read_text())
    v48_report = json.loads((V48_ROOT / "report.json").read_text())
    if v47_report.get("test_payload_opened") is not False or v48_report.get("test_payload_opened") is not False:
        raise ValueError("A source candidate opened sealed test")
    protocol = {
        "version": VERSION, "v47_report_sha256": V47_REPORT_SHA256,
        "v48_report_sha256": V48_REPORT_SHA256,
        "medsiglip_transform": "sigmoid((score - 0.309) / 0.006)",
        "medsiglip_weights": [0., .25, .50, .75, 1.],
        "lung_thresholds": [round(value, 2) for value in np.arange(.30, .901, .05)],
        "lesion_thresholds": [round(value, 3) for value in np.arange(.10, .901, .025)],
        "model_weights_updated": False, "marked_fit_images": 0, "marked_fit_planes": 0,
        "sealed_test_images_opened": 0,
    }
    fingerprint = json_hash(protocol); root = ARTIFACTS / "semantic_ensemble_v49" / fingerprint
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / "protocol.json", protocol)
    report_path = root / "report.json"
    if report_path.is_file():
        return json.loads(report_path.read_text())
    store = DevelopmentReferenceStore(); payloads = [store.load(row) for row in store.rows]
    v47, medsiglip, lungs, source_hashes = [], [], [], {}
    for row in store.rows:
        v47_path = V47_ROOT / "development_predictions" / f"{row['id']}.npz"
        width_paths = [V48_ROOT / f"{row['id']}_w{width}.npz" for width in (96, 144)]
        with np.load(v47_path, allow_pickle=False) as values:
            v47.append(values["lesion"]); lungs.append(values["lung"])
        planes = []
        for path in width_paths:
            with np.load(path, allow_pickle=False) as values:
                planes.append(values["score"])
        medsiglip.append(np.mean(planes, axis=0))
        source_hashes[row["id"]] = {"v47": sha256(v47_path), "medsiglip_96": sha256(width_paths[0]),
                                      "medsiglip_144": sha256(width_paths[1])}
    candidates = []
    for weight in protocol["medsiglip_weights"]:
        probabilities = [blend_probability(first, second, weight)
                         for first, second in zip(v47, medsiglip)]
        for lung_threshold in protocol["lung_thresholds"]:
            lung_masks = [probability >= lung_threshold for probability in lungs]
            for lesion_threshold in protocol["lesion_thresholds"]:
                measured = [_measure(payload, lung, (probability >= lesion_threshold) & lung)
                            for payload, lung, probability in zip(payloads, lung_masks, probabilities)]
                candidates.append({"medsiglip_weight": weight, "lung_threshold": lung_threshold,
                                   "lesion_threshold": lesion_threshold,
                                   "summary": summarize(measured)})
    winner = max(candidates, key=lambda value: _rank(value["summary"]))
    lesion = winner["summary"]["pooled_lesion"]
    checks = {"lesion_dice_at_least_0_80": lesion["dice"] >= .80,
              "precision_at_least_0_80": lesion["precision"] >= .80,
              "recall_at_least_0_80": lesion["recall"] >= .80,
              "specificity_at_least_0_80": lesion["specificity"] >= .80,
              "lung_dice_at_least_0_90": winner["summary"]["pooled_lung"]["dice"] >= .90}
    report = {"status": "done", "created_at": now(), "version": VERSION,
              "protocol": protocol, "protocol_fingerprint": fingerprint,
              "source_files_sha256": source_hashes, "development_winner": winner,
              "development_gate": {"passed": all(checks.values()), "checks": checks},
              "candidate_count": len(candidates), "model_weights_updated": False,
              "marked_fit_images": 0, "marked_fit_planes": 0, "test_payload_opened": False}
    atomic_json(report_path, report)
    return report
