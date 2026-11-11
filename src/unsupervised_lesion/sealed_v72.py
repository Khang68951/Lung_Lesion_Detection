from __future__ import annotations

import gc
import json

import numpy as np
import torch

from .evaluate import _measure, load_candidate, summarize
from .final_ensemble_v71 import refine_score
from .final_ensemble_v72 import LESION_THRESHOLD, MODEL_WEIGHTS, VERSION
from .io import ARTIFACTS, atomic_json, now, sha256
from .reference import ReferenceStore
from .sealed_v71 import (_binary_average, _dino_average, _multiview_average, _report,
                         _sam_cache, _sam_decoder_average, _test_rgb, _v70_average)
from .style_ensemble_v46 import BASE_WEIGHTS, PARENTS


def predict_sealed_v72() -> dict:
    candidate = load_candidate()
    if candidate.get("version") != VERSION or not candidate.get("oof_gate", {}).get("passed"):
        raise ValueError("Current frozen candidate is not passing V72")
    _, payloads = _test_rgb(candidate)
    root = ARTIFACTS / "sealed_v72" / candidate["candidate_fingerprint"]
    predictions = root / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    report_path = root / "prediction_report.json"
    if report_path.is_file() and len(list(predictions.glob("*.npz"))) == len(payloads):
        return json.loads(report_path.read_text())
    _sam_cache(payloads, root)
    device = torch.device("cuda")
    reports = {version: _report(PARENTS[version], version)[0] for version in PARENTS}
    v37 = _binary_average(payloads, reports["v37_binary_segformer_b1_relative_lab_reference_v2"],
                          "relative_lab", 384, device)
    v40 = _dino_average(payloads,
        reports["v40_frozen_dense_dinov2_colour_decoder_reference_v2"], device)
    v43 = _binary_average(payloads, reports["v43_binary_segformer_b1_native512_reference_v2"],
                          "rgb", 512, device)
    v44 = _sam_decoder_average(payloads,
        reports["v44_frozen_dense_sam2_multiscale_colour_decoder_reference_v2"], root, device)
    v45 = _binary_average(payloads,
        reports["v45_binary_segformer_b1_native512_unmarked_lab_style_reference_v2"],
        "rgb", 512, device)
    v64_report, _ = _report("multiview_crossval_v64", "v64_supervised_v54_multiview_group_oof")
    v67_report, _ = _report("style_multiview_v67", "v67_supervised_v54_multiview_unmarked_style_group_oof")
    v70_report, _ = _report("binary_crossval", "v70_binary_segformer_b2_native512_reference_v2")
    v64 = _multiview_average(payloads, v64_report, root, device)
    v67 = _multiview_average(payloads, v67_report, root, device)
    v70 = _v70_average(payloads, v70_report, device)
    hashes = {}
    for payload, a, b, c, d, e, f, g, h in zip(
            payloads, v37, v40, v43, v44, v45, v64, v67, v70):
        identifier = payload["row"]["id"]
        with np.load(root / "sam" / f"{identifier}.npz", allow_pickle=False) as values:
            lung = values["lung"].astype(bool)
        raw = (MODEL_WEIGHTS["v37"] * a + MODEL_WEIGHTS["v40"] * b
               + MODEL_WEIGHTS["v43"] * c + MODEL_WEIGHTS["v44"] * d
               + MODEL_WEIGHTS["v45"] * e + MODEL_WEIGHTS["v64_tta"] * f
               + MODEL_WEIGHTS["v67_tta"] * g + MODEL_WEIGHTS["v70_tta"] * h)
        score = refine_score(payload["rgb"], lung, raw)
        lesion = (score >= LESION_THRESHOLD) & lung
        path = predictions / f"{identifier}.npz"
        np.savez_compressed(path, lung=lung.astype(np.uint8), lesion=lesion.astype(np.uint8),
                            score=score, lesion_ratio=np.asarray(lesion.sum() / max(lung.sum(), 1)))
        hashes[identifier] = sha256(path)
    del v37, v40, v43, v44, v45, v64, v67, v70
    gc.collect(); torch.cuda.empty_cache()
    report = {
        "status": "predictions_frozen", "created_at": now(), "version": VERSION,
        "candidate_fingerprint": candidate["candidate_fingerprint"],
        "images": len(payloads), "prediction_sha256": hashes,
        "reference_planes_opened": 0, "test_rgb_opened": len(payloads),
    }
    atomic_json(report_path, report)
    return report


def evaluate_sealed_v72() -> dict:
    candidate = load_candidate()
    root = ARTIFACTS / "sealed_v72" / candidate["candidate_fingerprint"]
    prediction_report = json.loads((root / "prediction_report.json").read_text())
    store = ReferenceStore("test", candidate)
    rows = []
    for row in store.rows:
        path = root / "predictions" / f"{row['id']}.npz"
        if sha256(path) != prediction_report["prediction_sha256"].get(row["id"]):
            raise ValueError(f"Sealed V72 prediction changed: {row['id']}")
        with np.load(path, allow_pickle=False) as values:
            lung = values["lung"].astype(bool)
            lesion = values["lesion"].astype(bool)
        rows.append(_measure(store.load(row), lung, lesion))
    summary = summarize(rows)
    metric = summary["pooled_lesion"]
    checks = {
        "lesion_dice_at_least_0_80": metric["dice"] >= .80,
        "precision_at_least_0_80": metric["precision"] >= .80,
        "recall_at_least_0_80": metric["recall"] >= .80,
        "specificity_at_least_0_80": metric["specificity"] >= .80,
        "lung_dice_at_least_0_90": summary["pooled_lung"]["dice"] >= .90,
    }
    report = {
        "status": "done", "created_at": now(), "version": VERSION,
        "candidate_fingerprint": candidate["candidate_fingerprint"],
        "summary": summary, "test_gate": {"passed": all(checks.values()), "checks": checks},
        "prediction_report_sha256": sha256(root / "prediction_report.json"),
        "test_payload_opened": True, "per_image": rows,
    }
    atomic_json(root / "evaluation_report.json", report)
    return report
