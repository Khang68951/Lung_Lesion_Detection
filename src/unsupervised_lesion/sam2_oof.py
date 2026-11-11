from __future__ import annotations

import json

import numpy as np
import torch

from .binary_crossval import _report_root, _predict as predict_binary
from .crossval_segformer import _score_native
from .evaluate import _measure, summarize
from .io import (ARTIFACTS, PROJECT_ROOT, SAM2_CHECKPOINT, atomic_json,
                 json_hash, now, sha256)
from .reference import ReferenceStore
from .segformer_segmentation import _load, _predict as predict_multi, _restore, _rank
from .segmentation import (conservative_sam_fusion, sam2_boundary_refiner,
                           unlabeled_lung_probability)
from .supervised_segmentation import _prepare


VERSION = "v29_oof_sam2_boundary"


def calibrate_sam2_oof() -> dict:
    multi, multi_root = _report_root("crossval_segformer", "v25_development_group_crossval")
    binary, binary_root = _report_root("binary_crossval", "v27_binary_lesion_group_crossval")
    stores = [ReferenceStore("calibration"), ReferenceStore("confirmation")]
    payloads = [store.load(row) for store in stores for row in store.rows]
    protocol = {"version": VERSION, "source": "v28 immutable OOF ensemble",
                "binary_weight": .75, "lung_threshold": .50, "lesion_threshold": .50,
                "core_threshold": .75, "padding_fractions": [.02, .05],
                "erosion_fractions": [0., .01], "sam_dilation_fractions": [0., .01, .02],
                "sam2_checkpoint": str(SAM2_CHECKPOINT.relative_to(PROJECT_ROOT)),
                "sam2_training": "official generic SAM2.1; no project reference masks",
                "test_payload_opened": False}
    fingerprint = json_hash(protocol); root = ARTIFACTS / "sam2_oof" / fingerprint
    cache = root / "cache"; cache.mkdir(parents=True, exist_ok=True); atomic_json(root / "protocol.json", protocol)
    refiner = sam2_boundary_refiner(); prepared = []
    for index, payload in enumerate(payloads, start=1):
        identifier = payload["row"]["id"]
        with np.load(multi_root / "oof" / f"{identifier}.npz", allow_pickle=False) as values:
            multi_score = values["lesion"].astype(np.float32)
        with np.load(binary_root / "oof" / f"{identifier}.npz", allow_pickle=False) as values:
            binary_score = values["lesion"].astype(np.float32); lung_probability = values["lung"].astype(np.float32)
        score = .25 * multi_score + .75 * binary_score; lung = lung_probability >= .50
        seed = (score >= .50) & lung; core = (score >= .75) & lung; proposals = {}
        first = True; metadata = {}
        for padding in protocol["padding_fractions"]:
            for erosion in protocol["erosion_fractions"]:
                key = f"p{padding:.2f}_e{erosion:.2f}"
                proposal, record, _ = refiner.refine(payload["rgb"], lung, seed, core,
                                                     padding, erosion, encode_image=first)
                first = False; proposals[key] = proposal; metadata[key] = record
        np.savez_compressed(cache / f"{identifier}.npz", seed=seed, lung=lung, score=score,
                            **{f"proposal_{key}": value for key, value in proposals.items()})
        atomic_json(cache / f"{identifier}.json", {"id": identifier, "group": payload["row"]["group"],
                    "prompt_source": "v28 OOF only", "reference_pixels_used_for_proposal": 0,
                    "proposals": metadata})
        prepared.append(payload); print(f"v29 SAM2 cache {index}/{len(payloads)}", flush=True)
    candidates = []
    for padding in protocol["padding_fractions"]:
        for erosion in protocol["erosion_fractions"]:
            key = f"p{padding:.2f}_e{erosion:.2f}"
            for composition, dilation in [("proposal", None)] + [
                    ("seed_intersection_dilated_proposal", value)
                    for value in protocol["sam_dilation_fractions"]]:
                rows = []
                for payload in prepared:
                    with np.load(cache / f"{payload['row']['id']}.npz", allow_pickle=False) as values:
                        seed = values["seed"]; lung = values["lung"]; proposal = values[f"proposal_{key}"]
                    prediction = proposal if composition == "proposal" else conservative_sam_fusion(seed, proposal, dilation)
                    rows.append(_measure(payload, lung, prediction))
                summary = summarize(rows); candidates.append({"padding_fraction": padding,
                    "erosion_fraction": erosion, "composition": composition,
                    "sam_dilation_fraction": dilation, "summary": summary})
    winner = max(candidates, key=lambda value: _rank(value["summary"])); metric = winner["summary"]["pooled_lesion"]
    checks = {"lesion_dice_at_least_0_80": metric["dice"] >= .80,
              "precision_at_least_0_80": metric["precision"] >= .80,
              "recall_at_least_0_80": metric["recall"] >= .80,
              "specificity_at_least_0_80": metric["specificity"] >= .80,
              "lung_dice_at_least_0_90": winner["summary"]["pooled_lung"]["dice"] >= .90}
    report = {"status": "done", "created_at": now(), "version": VERSION, "protocol": protocol,
              "winner": winner, "oof_gate": {"passed": all(checks.values()), "checks": checks},
              "candidate_count": len(candidates), "results": candidates, "test_payload_opened": False}
    atomic_json(root / "report.json", report)
    if report["oof_gate"]["passed"]:
        candidate = {"status": "frozen_after_oof_validation", "created_at": now(), "version": VERSION,
            "selection_role": "four_fold_group_oof_sam2", "lesion_postprocess": "sam2_oof_v29",
            "multiclass_checkpoints": [{"path": fold["checkpoint"], "sha256": fold["checkpoint_sha256"]}
                                       for fold in multi["fold_artifacts"]],
            "binary_checkpoints": [{"path": fold["checkpoint"], "sha256": fold["checkpoint_sha256"]}
                                   for fold in binary["fold_artifacts"]],
            "binary_weight": .75, "lung_threshold": .50, "lesion_threshold": .50,
            "core_threshold": .75, "padding_fraction": winner["padding_fraction"],
            "erosion_fraction": winner["erosion_fraction"], "composition": winner["composition"],
            "sam_dilation_fraction": winner["sam_dilation_fraction"],
            "sam2_weight_sha256": sha256(refiner.checkpoint), "oof_summary": winner["summary"],
            "oof_gate": report["oof_gate"], "test_payload_opened": False}
        candidate["candidate_fingerprint"] = json_hash(candidate); folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
        atomic_json(folder / "candidate.json", candidate)
        atomic_json(ARTIFACTS / "frozen_candidate/current.json", {"path": str((folder / "candidate.json").relative_to(PROJECT_ROOT)),
            "sha256": sha256(folder / "candidate.json"), "candidate_fingerprint": candidate["candidate_fingerprint"]})
        atomic_json(ARTIFACTS / "test_unlock.json", {"created_at": now(), "confirmation_gate_passed": True,
            "gate_source": "four_fold_group_oof_sam2", "candidate_fingerprint": candidate["candidate_fingerprint"],
            "sam2_fingerprint": fingerprint})
        report["candidate"] = candidate; atomic_json(root / "report.json", report)
    return report


def evaluate_sam2_store(candidate: dict, store: ReferenceStore) -> tuple[dict, list[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); items = [_prepare(store.load(row)) for row in store.rows]
    multi = [np.zeros(item["payload"]["rgb"].shape[:2], np.float32) for item in items]
    binary = [np.zeros_like(value) for value in multi]
    for checkpoint in candidate["multiclass_checkpoints"]:
        probabilities = predict_multi(_load({"checkpoint": checkpoint["path"], "checkpoint_sha256": checkpoint["sha256"]}, device), items, device)
        for i, (item, probability) in enumerate(zip(items, probabilities)):
            multi[i] += _restore(probability[2], item["geometry"]) / len(candidate["multiclass_checkpoints"])
    for checkpoint in candidate["binary_checkpoints"]:
        probabilities = predict_binary(_load({"checkpoint": checkpoint["path"], "checkpoint_sha256": checkpoint["sha256"]}, device), items, device)
        for i, probability in enumerate(probabilities): binary[i] += probability / len(candidate["binary_checkpoints"])
    refiner = sam2_boundary_refiner(); rows = []
    for item, a, b in zip(items, multi, binary):
        score = (1 - candidate["binary_weight"]) * a + candidate["binary_weight"] * b
        lung = unlabeled_lung_probability(item["payload"]["rgb"]) >= candidate["lung_threshold"]
        seed = (score >= candidate["lesion_threshold"]) & lung; core = (score >= candidate["core_threshold"]) & lung
        proposal, _, _ = refiner.refine(item["payload"]["rgb"], lung, seed, core,
                                        candidate["padding_fraction"], candidate["erosion_fraction"])
        prediction = (proposal if candidate["composition"] == "proposal" else
                      conservative_sam_fusion(seed, proposal, candidate["sam_dilation_fraction"]))
        rows.append(_measure(item["payload"], lung, prediction))
    return summarize(rows), rows
