from __future__ import annotations

import json

import numpy as np
from skimage.segmentation import slic

from .crossval_segformer import _score_native
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now
from .reference_v2 import DevelopmentReferenceStore
from .segformer_segmentation import _rank
from .supervised_segmentation import _prepare


VERSION = "v41_cached_colour_region_consensus_reference_v2"


def region_mean_probability(probability: np.ndarray, labels: np.ndarray,
                            mask: np.ndarray) -> np.ndarray:
    inside = mask & (labels >= 0)
    output = np.zeros(probability.shape, np.float32)
    if not inside.any():
        return output
    indices = labels[inside].astype(np.int64)
    count = np.bincount(indices)
    total = np.bincount(indices, weights=probability[inside], minlength=len(count))
    means = total / np.maximum(count, 1)
    output[inside] = means[indices]
    return output


def _run_root(parent: str, version: str):
    matches = []
    for path in (ARTIFACTS / parent).glob("*/report.json"):
        report = json.loads(path.read_text())
        if report.get("version") == version:
            matches.append(path.parent)
    if len(matches) != 1:
        raise ValueError(f"Expected one {version} run, found {len(matches)}")
    return matches[0]


def calibrate_region_consensus() -> dict:
    v37 = _run_root("binary_crossval", "v37_binary_segformer_b1_relative_lab_reference_v2")
    v40 = _run_root("dense_dino_crossval", "v40_frozen_dense_dinov2_colour_decoder_reference_v2")
    store = DevelopmentReferenceStore(); items = [_prepare(store.load(row)) for row in store.rows]
    base = []; lungs = []
    for item in items:
        with np.load(v37 / "oof" / f"{item['id']}.npz", allow_pickle=False) as values:
            first = values["lesion"]; lungs.append(values["lung"])
        with np.load(v40 / "oof" / f"{item['id']}.npz", allow_pickle=False) as values:
            second = values["lesion"]
        base.append((.5 * first + .5 * second).astype(np.float32))
    protocol = {
        "version": VERSION, "reference_fingerprint": store.manifest["fingerprint"],
        "v37_root": str(v37.relative_to(PROJECT_ROOT)), "v40_root": str(v40.relative_to(PROJECT_ROOT)),
        "weights": {"v37": .5, "v40": .5}, "lung_threshold": .50,
        "requested_regions": [160, 240, 320], "compactness": 12,
        "region_weights": [.50, 1.0],
        "lesion_thresholds": [round(x, 3) for x in np.arange(.35, .651, .025)],
        "slic_inputs": "native_clean_rgb_and_predicted_lung_only",
        "reference_planes_used_by_slic": 0, "sealed_test_images_opened": 0,
    }
    fingerprint = json_hash(protocol); root = ARTIFACTS / "region_consensus_v41" / fingerprint
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / "protocol.json", protocol)
    candidates = []
    score_cache = {}
    for region_count in protocol["requested_regions"]:
        regional = []
        for item, probability, lung_probability in zip(items, base, lungs):
            predicted_lung = lung_probability >= .50
            labels = slic(item["payload"]["rgb"], n_segments=region_count, compactness=12,
                          start_label=0, mask=predicted_lung, channel_axis=-1,
                          enforce_connectivity=True)
            regional.append(region_mean_probability(probability, labels, predicted_lung))
        for region_weight in protocol["region_weights"]:
            scores = [(1 - region_weight) * raw + region_weight * mean
                      for raw, mean in zip(base, regional)]
            score_cache[(region_count, region_weight)] = scores
            for threshold in protocol["lesion_thresholds"]:
                summary, _ = _score_native(items, scores, lungs, .50, threshold)
                candidates.append({"requested_regions": region_count,
                                   "region_weight": region_weight,
                                   "lung_threshold": .50, "lesion_threshold": threshold,
                                   "summary": summary})
    winner = max(candidates, key=lambda value: _rank(value["summary"]))
    metrics = winner["summary"]["pooled_lesion"]
    checks = {"lesion_dice_at_least_0_80": metrics["dice"] >= .80,
              "precision_at_least_0_80": metrics["precision"] >= .80,
              "recall_at_least_0_80": metrics["recall"] >= .80,
              "specificity_at_least_0_80": metrics["specificity"] >= .80,
              "lung_dice_at_least_0_90": winner["summary"]["pooled_lung"]["dice"] >= .90}
    selected = score_cache[(winner["requested_regions"], winner["region_weight"])]
    planes = root / "winner_scores"; planes.mkdir(exist_ok=True)
    for item, score, lung in zip(items, selected, lungs):
        np.savez_compressed(planes / f"{item['id']}.npz", lesion=score, lung=lung)
    report = {"status": "done", "created_at": now(), "version": VERSION,
              "protocol": protocol, "candidate_count": len(candidates), "winner": winner,
              "oof_gate": {"passed": all(checks.values()), "checks": checks},
              "winner_score_planes": str(planes.relative_to(PROJECT_ROOT)),
              "test_payload_opened": False, "results": candidates}
    atomic_json(root / "report.json", report)
    return report
