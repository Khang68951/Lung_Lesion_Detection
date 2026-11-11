from __future__ import annotations

import json

import numpy as np

from .crossval_segformer import _score_native
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256
from .reference_v2 import DevelopmentReferenceStore
from .segformer_segmentation import _rank
from .supervised_segmentation import _prepare


VERSION = "v46_cached_v45_complement_reference_v2"
BASE_WEIGHTS = {
    "v37_binary_segformer_b1_relative_lab_reference_v2": .375,
    "v40_frozen_dense_dinov2_colour_decoder_reference_v2": .1875,
    "v43_binary_segformer_b1_native512_reference_v2": .1875,
    "v44_frozen_dense_sam2_multiscale_colour_decoder_reference_v2": .25,
}
STYLE_VERSION = "v45_binary_segformer_b1_native512_unmarked_lab_style_reference_v2"
PARENTS = {
    "v37_binary_segformer_b1_relative_lab_reference_v2": "binary_crossval",
    "v40_frozen_dense_dinov2_colour_decoder_reference_v2": "dense_dino_crossval",
    "v43_binary_segformer_b1_native512_reference_v2": "binary_crossval",
    "v44_frozen_dense_sam2_multiscale_colour_decoder_reference_v2": "dense_sam2_crossval",
    STYLE_VERSION: "binary_crossval",
}


def blend_scores(base: np.ndarray, style: np.ndarray, style_weight: float) -> np.ndarray:
    if not 0 <= style_weight <= 1 or base.shape != style.shape:
        raise ValueError("Invalid V46 blend inputs")
    return ((1 - style_weight) * base + style_weight * style).astype(np.float32)


def _root(parent: str, version: str):
    matches = []
    for path in (ARTIFACTS / parent).glob("*/report.json"):
        if json.loads(path.read_text()).get("version") == version:
            matches.append(path.parent)
    if len(matches) != 1:
        raise ValueError(f"Expected one completed {version} run, found {len(matches)}")
    return matches[0]


def calibrate_style_ensemble_v46() -> dict:
    roots = {version: _root(parent, version) for version, parent in PARENTS.items()}
    store = DevelopmentReferenceStore()
    items = [_prepare(store.load(row)) for row in store.rows]
    base_probabilities, style_probabilities, lung_probabilities = [], [], []
    for item in items:
        identifier = item["id"]
        base = None
        for version, weight in BASE_WEIGHTS.items():
            with np.load(roots[version] / "oof" / f"{identifier}.npz", allow_pickle=False) as values:
                value = values["lesion"].astype(np.float32)
            base = value * weight if base is None else base + value * weight
        with np.load(roots[STYLE_VERSION] / "oof" / f"{identifier}.npz", allow_pickle=False) as values:
            style_probabilities.append(values["lesion"].astype(np.float32))
            lung_probabilities.append(values["lung"].astype(np.float32))
        base_probabilities.append(base.astype(np.float32))
    protocol = {
        "version": VERSION, "reference_fingerprint": store.manifest["fingerprint"],
        "base_weights": BASE_WEIGHTS, "style_version": STYLE_VERSION,
        "style_weights": [0., .25, .50, .75, 1.], "lung_threshold": .50,
        "lesion_thresholds": [round(value, 3) for value in np.arange(.35, .651, .025)],
        "candidate_count": 65,
        "source_reports": {
            version: {"path": str((root / "report.json").relative_to(PROJECT_ROOT)),
                      "sha256": sha256(root / "report.json")}
            for version, root in roots.items()},
        "sealed_test_images_opened": 0,
    }
    candidates = []
    for style_weight in protocol["style_weights"]:
        scores = [blend_scores(base, style, style_weight)
                  for base, style in zip(base_probabilities, style_probabilities)]
        for threshold in protocol["lesion_thresholds"]:
            summary, _ = _score_native(items, scores, lung_probabilities, .50, threshold)
            candidates.append({"style_weight": style_weight, "lung_threshold": .50,
                               "lesion_threshold": threshold, "summary": summary})
    if len(candidates) != protocol["candidate_count"]:
        raise ValueError("V46 locked candidate count changed")
    winner = max(candidates, key=lambda value: _rank(value["summary"]))
    metric = winner["summary"]["pooled_lesion"]
    checks = {
        "lesion_dice_at_least_0_80": metric["dice"] >= .80,
        "precision_at_least_0_80": metric["precision"] >= .80,
        "recall_at_least_0_80": metric["recall"] >= .80,
        "specificity_at_least_0_80": metric["specificity"] >= .80,
        "lung_dice_at_least_0_90": winner["summary"]["pooled_lung"]["dice"] >= .90,
    }
    report = {"status": "done", "created_at": now(), "version": VERSION,
              "protocol": protocol, "winner": winner,
              "oof_gate": {"passed": all(checks.values()), "checks": checks},
              "candidate_count": len(candidates), "test_payload_opened": False,
              "results": candidates}
    fingerprint = json_hash(protocol)
    root = ARTIFACTS / "style_ensemble_v46" / fingerprint
    atomic_json(root / "report.json", report)
    if report["oof_gate"]["passed"]:
        atomic_json(ARTIFACTS / "test_unlock.json", {
            "created_at": now(), "confirmation_gate_passed": True,
            "gate_source": VERSION, "ensemble_fingerprint": fingerprint})
    return report
