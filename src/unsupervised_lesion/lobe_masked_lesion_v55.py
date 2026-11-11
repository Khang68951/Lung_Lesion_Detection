from __future__ import annotations

import json

import numpy as np

from .evaluate import _measure, summarize
from .io import ARTIFACTS, atomic_json, json_hash, now, sha256
from .lung_consensus_v54 import V53_ROOT, vote_consensus
from .reference_v2 import DevelopmentReferenceStore


VERSION = "v55_v51_lesion_inside_v54_lobes"
V54_ROOT = ARTIFACTS / "lung_consensus_v54" / "5cc655f9e6ac21f9fe28480022266fa4296ce0aa31a5d7f06bf7aef042a24e17"
V54_REPORT_SHA256 = "a3748f9381851d6d8bab55d019c7dd4b6fe19a017ce9ef0b54083015baee8b6d"
V51_ROOT = ARTIFACTS / "sam_lobe_v51" / "360b8d649d3939c22c38fe17959c54f89edd66a4a1f83e8a12773bf854dd2e90"
V51_REPORT_SHA256 = "775198eb7bd49b1089d30fadc861c5ed47eb29275221151b8bc5dc2793714012"


def _balanced_key(summary: dict) -> tuple:
    lesion = summary["pooled_lesion"]
    return (min(lesion[key] or 0 for key in ("precision", "recall", "specificity")),
            lesion["dice"] or 0, lesion["balanced_accuracy"] or 0,
            -summary["ratio_mae_percentage_points"])


def calibrate_lobe_masked_lesion_v55() -> dict:
    if sha256(V54_ROOT / "report.json") != V54_REPORT_SHA256:
        raise ValueError("Pinned V54 report changed")
    if sha256(V51_ROOT / "report.json") != V51_REPORT_SHA256:
        raise ValueError("Pinned V51 report changed")
    protocol = {
        "version": VERSION, "v54_report_sha256": V54_REPORT_SHA256,
        "v51_report_sha256": V51_REPORT_SHA256,
        "lung_mask": "V54 SAM output 2, at least 3 of 12 prompt variants",
        "lesion_scores": "blend of frozen V47 pixel and V51 SAM-region probability",
        "region_weights": [round(value, 2) for value in np.arange(0, 1.01, .1)],
        "lesion_thresholds": [round(value, 3) for value in np.arange(.10, .701, .01)],
        "selection": "maximize worst precision/recall/specificity, then Dice",
        "marked_fit_images": 0, "marked_fit_planes": 0,
        "sealed_test_images_opened": 0,
    }
    fingerprint = json_hash(protocol)
    root = ARTIFACTS / "lobe_masked_lesion_v55" / fingerprint
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "protocol.json", protocol)
    report_path = root / "report.json"
    if report_path.is_file():
        return json.loads(report_path.read_text())

    store = DevelopmentReferenceStore()
    rows = list(store.rows)
    frozen, source_hashes = [], {}
    for row in rows:
        v53_path = V53_ROOT / f"{row['id']}.npz"
        v51_path = V51_ROOT / f"{row['id']}.npz"
        with np.load(v53_path, allow_pickle=False) as values:
            shape = tuple(values["mask_shape"])
            metadata = json.loads(str(values["metadata_json"]))
            raw = [np.unpackbits(value, count=shape[0] * shape[1]).reshape(shape).astype(bool)
                   for value in values["packed_masks"]]
        planes = []
        for variant in range(12):
            pair = [mask for mask, record in zip(raw, metadata)
                    if record["variant"] == variant and record["output"] == 2]
            if len(pair) != 2:
                raise ValueError(f"Incomplete V54 source pair: {row['id']} v{variant}")
            planes.append(np.logical_or.reduce(pair))
        lung = vote_consensus(planes, 3)
        with np.load(v51_path, allow_pickle=False) as values:
            pixel = values["v47"].copy(); region = values["region"].copy()
        frozen.append((lung, pixel, region))
        source_hashes[row["id"]] = {"v53": sha256(v53_path), "v51": sha256(v51_path)}

    payloads = [store.load(row) for row in rows]
    candidates = []
    for weight in protocol["region_weights"]:
        scores = [(1 - weight) * pixel + weight * region for _, pixel, region in frozen]
        for threshold in protocol["lesion_thresholds"]:
            measured = [_measure(payload, lung, (score >= threshold) & lung)
                        for payload, (lung, _, _), score in zip(payloads, frozen, scores)]
            candidates.append({"region_weight": weight, "lesion_threshold": threshold,
                               "summary": summarize(measured)})
    balanced = max(candidates, key=lambda value: _balanced_key(value["summary"]))
    dice_winner = max(candidates, key=lambda value: value["summary"]["pooled_lesion"]["dice"])
    lesion = balanced["summary"]["pooled_lesion"]
    checks = {"lesion_dice_at_least_0_80": lesion["dice"] >= .80,
              "precision_at_least_0_80": lesion["precision"] >= .80,
              "recall_at_least_0_80": lesion["recall"] >= .80,
              "specificity_at_least_0_80": lesion["specificity"] >= .80,
              "lung_dice_at_least_0_95": balanced["summary"]["pooled_lung"]["dice"] >= .95}
    report = {"status": "done", "created_at": now(), "version": VERSION,
              "protocol": protocol, "protocol_fingerprint": fingerprint,
              "source_files_sha256": source_hashes,
              "development_balanced_winner": balanced,
              "development_dice_winner": dice_winner,
              "development_gate": {"passed": all(checks.values()), "checks": checks},
              "candidate_count": len(candidates), "model_weights_updated": False,
              "marked_fit_images": 0, "marked_fit_planes": 0,
              "test_payload_opened": False}
    atomic_json(report_path, report)
    return report
