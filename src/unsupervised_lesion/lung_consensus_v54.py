from __future__ import annotations

import json

import numpy as np

from .evaluate import _measure, summarize
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256
from .reference_v2 import DevelopmentReferenceStore


VERSION = "v54_sam2_lobe_vote_consensus"
V53_ROOT = ARTIFACTS / "lung_lobe_v53" / "87d627851b3c8c86c73be106535607831c97fa4ae7bed8e26331f9f13672a0cb"
V53_REPORT_SHA256 = "b3ced5fb214162d498c9cecdb2e9ec9fb438d1e1d0eec5ebfb29b6aa1073db88"
OUTPUT_SETS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))


def vote_consensus(masks: list[np.ndarray], minimum_votes: int) -> np.ndarray:
    if not masks or minimum_votes < 1 or minimum_votes > len(masks):
        raise ValueError("Invalid V54 vote configuration")
    shape = np.asarray(masks[0]).shape
    if any(np.asarray(mask).shape != shape for mask in masks):
        raise ValueError("V54 masks have inconsistent geometry")
    votes = np.zeros(shape, np.uint8)
    for mask in masks:
        votes += np.asarray(mask, bool)
    return votes >= minimum_votes


def _selection_key(summary: dict) -> tuple:
    lung = summary["pooled_lung"]
    return (lung["dice"], min(lung[key] for key in ("precision", "recall", "specificity")),
            lung["balanced_accuracy"])


def calibrate_lung_consensus_v54() -> dict:
    if sha256(V53_ROOT / "report.json") != V53_REPORT_SHA256:
        raise ValueError("Pinned V53 report changed")
    protocol = {
        "version": VERSION, "v53_report_sha256": V53_REPORT_SHA256,
        "source": "frozen V53 RGB-only SAM2 masks",
        "sam_output_sets": [list(value) for value in OUTPUT_SETS],
        "vote_thresholds": "all valid integer thresholds for each output set",
        "selection": "pooled lung Dice, then worst precision/recall/specificity",
        "stability_check": "leave-one-source-group-out hyperparameter selection",
        "marked_fit_images": 0, "marked_fit_planes": 0,
        "sealed_test_images_opened": 0,
    }
    fingerprint = json_hash(protocol)
    root = ARTIFACTS / "lung_consensus_v54" / fingerprint
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "protocol.json", protocol)
    report_path = root / "report.json"
    if report_path.is_file():
        return json.loads(report_path.read_text())

    store = DevelopmentReferenceStore()
    rows = list(store.rows)
    # Read and validate all frozen prediction planes before opening any reference payload.
    frozen, source_hashes = [], {}
    for row in rows:
        path = V53_ROOT / f"{row['id']}.npz"
        with np.load(path, allow_pickle=False) as values:
            shape = tuple(values["mask_shape"])
            metadata = json.loads(str(values["metadata_json"]))
            raw = [np.unpackbits(value, count=shape[0] * shape[1]).reshape(shape).astype(bool)
                   for value in values["packed_masks"]]
        by_variant_output = {}
        for variant in range(12):
            for output in range(3):
                pair = [mask for mask, record in zip(raw, metadata)
                        if record["variant"] == variant and record["output"] == output]
                if len(pair) != 2:
                    raise ValueError(f"Incomplete V53 lobe pair: {row['id']} v{variant} o{output}")
                by_variant_output[(variant, output)] = np.logical_or.reduce(pair)
        frozen.append(by_variant_output)
        source_hashes[row["id"]] = sha256(path)

    payloads = [store.load(row) for row in rows]
    candidates = []
    for outputs in OUTPUT_SETS:
        maximum_votes = 12 * len(outputs)
        for minimum_votes in range(1, maximum_votes + 1):
            measured = []
            for payload, planes in zip(payloads, frozen):
                masks = [planes[(variant, output)] for variant in range(12) for output in outputs]
                lung = vote_consensus(masks, minimum_votes)
                measured.append(_measure(payload, lung, np.zeros_like(lung)))
            candidates.append({"outputs": list(outputs), "minimum_votes": minimum_votes,
                               "measured": measured, "summary": summarize(measured)})
    winner = max(candidates, key=lambda value: _selection_key(value["summary"]))

    groups = sorted({row["group"] for row in rows})
    logo_measured, logo_choices, per_group = [], {}, {}
    for group in groups:
        fit_indices = [index for index, row in enumerate(rows) if row["group"] != group]
        holdout_indices = [index for index, row in enumerate(rows) if row["group"] == group]
        selected = max(candidates, key=lambda value: _selection_key(
            summarize([value["measured"][index] for index in fit_indices])))
        held = [selected["measured"][index] for index in holdout_indices]
        logo_measured.extend(held)
        group_summary = summarize(held)
        per_group[group] = group_summary["pooled_lung"]
        logo_choices[group] = {"outputs": selected["outputs"],
                               "minimum_votes": selected["minimum_votes"]}
    logo_summary = summarize(logo_measured)
    lung = winner["summary"]["pooled_lung"]
    stable = logo_summary["pooled_lung"]
    checks = {
        "development_lung_dice_at_least_0_95": lung["dice"] >= .95,
        "development_precision_at_least_0_90": lung["precision"] >= .90,
        "development_recall_at_least_0_90": lung["recall"] >= .90,
        "development_specificity_at_least_0_90": lung["specificity"] >= .90,
        "logo_lung_dice_at_least_0_95": stable["dice"] >= .95,
        "logo_precision_at_least_0_90": stable["precision"] >= .90,
        "logo_recall_at_least_0_90": stable["recall"] >= .90,
        "logo_specificity_at_least_0_90": stable["specificity"] >= .90,
    }
    top = sorted(candidates, key=lambda value: _selection_key(value["summary"]), reverse=True)[:10]
    report = {
        "status": "done", "created_at": now(), "version": VERSION,
        "protocol": protocol, "protocol_fingerprint": fingerprint,
        "source_files_sha256": source_hashes,
        "development_winner": {key: winner[key] for key in ("outputs", "minimum_votes", "summary")},
        "logo_selection_summary": logo_summary, "logo_choices": logo_choices,
        "logo_per_group_lung": per_group,
        "top_candidates": [{"outputs": value["outputs"], "minimum_votes": value["minimum_votes"],
                            "pooled_lung": value["summary"]["pooled_lung"]} for value in top],
        "development_gate": {"passed": all(checks.values()), "checks": checks},
        "candidate_count": len(candidates), "model_weights_updated": False,
        "marked_fit_images": 0, "marked_fit_planes": 0, "test_payload_opened": False,
    }
    atomic_json(report_path, report)
    return report
