from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from skimage.segmentation import slic

from .evaluate import _measure, summarize
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256
from .reference_v2 import DevelopmentReferenceStore
from .region_consensus_v41 import region_mean_probability
from .style_ensemble_v46 import BASE_WEIGHTS, PARENTS, STYLE_VERSION, _root
from .supervised_segmentation import _prepare


VERSION = "v71_b2_tta_colour_region_gaussian_oof"
MODEL_WEIGHTS = {
    "v46": 0.3401256832031896,
    "v64_tta": 0.1884046457799112,
    "v67_tta": 0.06876791629131854,
    "v70_tta": 0.4027017547255806,
}
REGION_WEIGHT = 0.49943276195599456
REGION_SEGMENTS = 75
REGION_COMPACTNESS = 5.0
GAUSSIAN_SIGMA = 1.0
LESION_THRESHOLD = 0.4923666666666667


def refine_score(rgb: np.ndarray, lung: np.ndarray, score: np.ndarray) -> np.ndarray:
    labels = slic(rgb, n_segments=REGION_SEGMENTS, compactness=REGION_COMPACTNESS,
                  start_label=0, mask=lung, channel_axis=-1, enforce_connectivity=True)
    regional = region_mean_probability(score, labels, lung)
    mixed = ((1 - REGION_WEIGHT) * score + REGION_WEIGHT * regional).astype(np.float32)
    return cv2.GaussianBlur(mixed, (0, 0), GAUSSIAN_SIGMA).astype(np.float32)


def _v70_root() -> Path:
    matches = []
    for path in (ARTIFACTS / "binary_crossval").glob("*/report.json"):
        report = json.loads(path.read_text())
        if report.get("version") == "v70_binary_segformer_b2_native512_reference_v2":
            matches.append(path.parent)
    if len(matches) != 1:
        raise ValueError(f"Expected one completed V70 report, found {len(matches)}")
    return matches[0]


def calibrate_final_ensemble_v71() -> dict:
    roots = {version: _root(PARENTS[version], version) for version in PARENTS}
    v70_root = _v70_root()
    tta = ARTIFACTS / "tta_multiview_v68"
    tta_v70 = ARTIFACTS / "tta_v70"
    store = DevelopmentReferenceStore()
    items = [_prepare(store.load(row)) for row in store.rows]
    protocol = {
        "version": VERSION, "reference_fingerprint": store.manifest["fingerprint"],
        "model_weights": MODEL_WEIGHTS, "v46_internal_style_weight": .25,
        "region_segments": REGION_SEGMENTS, "region_compactness": REGION_COMPACTNESS,
        "region_weight": REGION_WEIGHT, "gaussian_sigma": GAUSSIAN_SIGMA,
        "lesion_threshold": LESION_THRESHOLD,
        "selection": "maximize minimum of Dice/precision/recall/specificity on group-OOF planes",
        "source_reports": {
            version: {"path": str((root / "report.json").relative_to(PROJECT_ROOT)),
                      "sha256": sha256(root / "report.json")}
            for version, root in roots.items()
        } | {"v70": {"path": str((v70_root / "report.json").relative_to(PROJECT_ROOT)),
                       "sha256": sha256(v70_root / "report.json")}},
        "sealed_test_images_opened": 0,
    }
    fingerprint = json_hash(protocol)
    root = ARTIFACTS / "final_ensemble_v71" / fingerprint
    scores = root / "oof_scores"
    scores.mkdir(parents=True, exist_ok=True)
    rows = []
    score_hashes = {}
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
        v46 = (.75 * base + .25 * style).astype(np.float32)
        with np.load(tta / "v64" / f"{identifier}.npz", allow_pickle=False) as values:
            v64 = values["average"].astype(np.float32)
            lung = values["lung"].astype(bool)
        with np.load(tta / "v67" / f"{identifier}.npz", allow_pickle=False) as values:
            v67 = values["average"].astype(np.float32)
        with np.load(tta_v70 / f"{identifier}.npz", allow_pickle=False) as values:
            v70 = values["lesion"].astype(np.float32)
        raw = (MODEL_WEIGHTS["v46"] * v46 + MODEL_WEIGHTS["v64_tta"] * v64 +
               MODEL_WEIGHTS["v67_tta"] * v67 + MODEL_WEIGHTS["v70_tta"] * v70)
        final = refine_score(item["payload"]["rgb"], lung, raw)
        output = scores / f"{identifier}.npz"
        np.savez_compressed(output, lesion=final, lung=lung.astype(np.uint8))
        score_hashes[identifier] = sha256(output)
        rows.append(_measure(item["payload"], lung,
                             (final >= LESION_THRESHOLD) & lung))
    summary = summarize(rows)
    lesion = summary["pooled_lesion"]
    checks = {"lesion_dice_at_least_0_80": lesion["dice"] >= .80,
              "precision_at_least_0_80": lesion["precision"] >= .80,
              "recall_at_least_0_80": lesion["recall"] >= .80,
              "specificity_at_least_0_80": lesion["specificity"] >= .80,
              "lung_dice_at_least_0_95": summary["pooled_lung"]["dice"] >= .95}
    report = {"status": "done", "created_at": now(), "version": VERSION,
              "protocol": protocol, "protocol_fingerprint": fingerprint,
              "winner": {"summary": summary, "lesion_threshold": LESION_THRESHOLD},
              "oof_score_sha256": score_hashes,
              "oof_gate": {"passed": all(checks.values()), "checks": checks},
              "test_payload_opened": False}
    atomic_json(root / "report.json", report)
    if report["oof_gate"]["passed"]:
        candidate = {"status": "frozen_after_group_oof_gate", "created_at": now(),
            "version": VERSION, "selection_role": "four_fold_group_oof_ensemble",
            "lesion_postprocess": "final_ensemble_v71", "protocol": protocol,
            "protocol_fingerprint": fingerprint, "oof_summary": summary,
            "oof_gate": report["oof_gate"], "test_payload_opened": False}
        candidate["candidate_fingerprint"] = json_hash(candidate)
        folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
        atomic_json(folder / "candidate.json", candidate)
        pointer = {"path": str((folder / "candidate.json").relative_to(PROJECT_ROOT)),
                   "sha256": sha256(folder / "candidate.json"),
                   "candidate_fingerprint": candidate["candidate_fingerprint"]}
        atomic_json(ARTIFACTS / "frozen_candidate/current.json", pointer)
        atomic_json(ARTIFACTS / "test_unlock.json", {
            "created_at": now(), "confirmation_gate_passed": True,
            "gate_source": VERSION, "candidate_fingerprint": candidate["candidate_fingerprint"],
            "ensemble_fingerprint": fingerprint})
        report["candidate"] = candidate
        atomic_json(root / "report.json", report)
    return report
