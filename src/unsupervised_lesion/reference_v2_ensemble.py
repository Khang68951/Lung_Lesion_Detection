from __future__ import annotations

import json

import numpy as np

from .crossval_segformer import _score_native
from .io import ARTIFACTS, atomic_json, json_hash, now
from .reference_v2 import DevelopmentReferenceStore
from .segformer_segmentation import _rank
from .supervised_segmentation import _prepare


MULTI = ARTIFACTS / "crossval_segformer/5eb0ebaf5f4f39d520dce17b9297a226c86d02aa207b5784ad326c011197d2bc/oof"
OLD_BINARY = ARTIFACTS / "binary_crossval/7351e8d6d9776ac9975812fd9c506114fcdb37e0340a527736d5c9981eeea426/oof"
V2_BINARY = ARTIFACTS / "binary_crossval/f19a75070f257481df905b3fb5a125a4784dd9ecf35255a326b71efd290f39bb/oof"


def calibrate_reference_v2_ensemble() -> dict:
    store = DevelopmentReferenceStore(); items = [_prepare(store.load(row)) for row in store.rows]
    multi = []; old_binary = []; v2_binary = []; lungs = []
    for item in items:
        identifier = item["id"]
        with np.load(MULTI / f"{identifier}.npz", allow_pickle=False) as values:
            multi.append(values["lesion"].astype(np.float32))
        with np.load(OLD_BINARY / f"{identifier}.npz", allow_pickle=False) as values:
            old_binary.append(values["lesion"].astype(np.float32))
        with np.load(V2_BINARY / f"{identifier}.npz", allow_pickle=False) as values:
            v2_binary.append(values["lesion"].astype(np.float32)); lungs.append(values["lung"].astype(np.float32))
    weights = [(a / 4, b / 4, (4 - a - b) / 4)
               for a in range(5) for b in range(5 - a)]
    candidates = []
    for multi_weight, old_weight, v2_weight in weights:
        scores = [multi_weight * a + old_weight * b + v2_weight * c
                  for a, b, c in zip(multi, old_binary, v2_binary)]
        for lung_threshold in [round(value, 2) for value in np.arange(.50, .951, .05)]:
            for lesion_threshold in [round(value, 3) for value in np.arange(.10, .901, .025)]:
                summary, _ = _score_native(items, scores, lungs, lung_threshold, lesion_threshold)
                candidates.append({"weights": {"historical_multiclass": multi_weight,
                                                "historical_binary": old_weight,
                                                "reference_v2_binary": v2_weight},
                                   "lung_threshold": lung_threshold,
                                   "lesion_threshold": lesion_threshold, "summary": summary})
    winner = max(candidates, key=lambda value: _rank(value["summary"])); metric = winner["summary"]["pooled_lesion"]
    checks = {"lesion_dice_at_least_0_80": metric["dice"] >= .80,
              "precision_at_least_0_80": metric["precision"] >= .80,
              "recall_at_least_0_80": metric["recall"] >= .80,
              "specificity_at_least_0_80": metric["specificity"] >= .80,
              "lung_dice_at_least_0_90": winner["summary"]["pooled_lung"]["dice"] >= .90}
    protocol = {"version": "v36_cached_cross_reference_oof_ensemble",
                "reference_fingerprint": store.manifest["fingerprint"],
                "source_oof": {"historical_multiclass": str(MULTI), "historical_binary": str(OLD_BINARY),
                               "reference_v2_binary": str(V2_BINARY)},
                "weights": weights, "test_payload_opened": False}
    report = {"status": "done", "created_at": now(), "protocol": protocol,
              "candidate_count": len(candidates), "winner": winner,
              "oof_gate": {"passed": all(checks.values()), "checks": checks},
              "test_payload_opened": False}
    root = ARTIFACTS / "reference_v2_ensemble" / json_hash(protocol)
    atomic_json(root / "report.json", report)
    return report

