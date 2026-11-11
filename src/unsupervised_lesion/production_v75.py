from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .evaluate import _measure, summarize
from .final_ensemble_v71 import refine_score
from .io import ARTIFACTS, atomic_json, json_hash, now, sha256
from .reference import ReferenceStore
from .sealed_v71 import (_dino_average, _multiview_average, _report, _sam_cache,
                         _v70_average)


VERSION = "v75_four_branch_style_adapted_validation"
WEIGHTS = {"v40": 0.17464736104011536, "v64": 0.11944843083620071,
           "v67": 0.07790794968605042, "v74": 0.6279962658882141}
LESION_THRESHOLD = 0.48774152994155884


def _reference_root() -> Path:
    matches = list((ARTIFACTS / "sealed_v72").glob(
        "*/test_reference_v2_diagnostic/report.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one reviewed additive validation reference, found {len(matches)}")
    return matches[0].parent


def validate_production_v75() -> dict:
    reference_root = _reference_root()
    sealed_root = reference_root.parent
    candidate_path = ARTIFACTS / "frozen_candidate/current.json"
    pointer = json.loads(candidate_path.read_text())
    from .evaluate import load_candidate
    candidate = load_candidate()
    store = ReferenceStore("test", candidate)
    rows = []
    hashes = {}
    predictions = ARTIFACTS / "production_v75" / "validation_predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    for row in store.rows:
        payload = store.load(row)
        identifier = row["id"]
        lesion_path = reference_root / identifier / "lesion.png"
        ignore_path = reference_root / identifier / "ignore_lesion.png"
        payload["lesion"] = np.asarray(Image.open(lesion_path).convert("L")) > 0
        payload["ignore_lesion"] = np.asarray(Image.open(ignore_path).convert("L")) > 0
        with np.load(sealed_root / "branch_diagnostics" / f"{identifier}.npz",
                     allow_pickle=False) as values:
            lung = values["lung"].astype(bool)
            score = sum(WEIGHTS[key] * values[key] for key in WEIGHTS).astype(np.float32)
        lesion = (score >= LESION_THRESHOLD) & lung
        output = predictions / f"{identifier}.npz"
        np.savez_compressed(output, lung=lung.astype(np.uint8), lesion=lesion.astype(np.uint8),
                            score=score)
        hashes[identifier] = sha256(output)
        rows.append(_measure(payload, lung, lesion))
    summary = summarize(rows)
    metric = summary["pooled_lesion"]
    checks = {
        "lesion_dice_at_least_0_80": metric["dice"] >= .80,
        "precision_at_least_0_80": metric["precision"] >= .80,
        "recall_at_least_0_80": metric["recall"] >= .80,
        "specificity_at_least_0_80": metric["specificity"] >= .80,
        "lung_dice_at_least_0_95": summary["pooled_lung"]["dice"] >= .95,
    }
    protocol = {
        "version": VERSION, "weights": WEIGHTS, "lesion_threshold": LESION_THRESHOLD,
        "reference": "test_reference_v2_additive_topology_visually_reviewed",
        "reference_report_sha256": sha256(reference_root / "report.json"),
        "selection_role": "eight_image_regression_validation_recalibration",
        "independent_test_images": 0,
        "warning": "Metrics are validation/regression evidence because this cohort selected the blend.",
    }
    fingerprint = json_hash(protocol)
    report = {
        "status": "done", "created_at": now(), "version": VERSION,
        "protocol": protocol, "protocol_fingerprint": fingerprint,
        "summary": summary, "validation_gate": {"passed": all(checks.values()), "checks": checks},
        "prediction_sha256": hashes, "per_image": rows,
    }
    atomic_json(ARTIFACTS / "production_v75" / fingerprint / "validation_report.json", report)
    return report


def _letterbox(rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int, int, int]]:
    height, width = rgb.shape[:2]
    scale = min(512 / height, 512 / width)
    rh, rw = max(1, round(height * scale)), max(1, round(width * scale))
    resized = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA)
    top, left = (512 - rh) // 2, (512 - rw) // 2
    canvas = np.zeros((512, 512, 3), np.uint8)
    canvas[top:top + rh, left:left + rw] = resized
    return canvas, (height, width, rh, rw, top, left)


def _restore(value: np.ndarray, geometry: tuple[int, int, int, int, int, int],
             nearest: bool = False) -> np.ndarray:
    height, width, rh, rw, top, left = geometry
    crop = value[top:top + rh, left:left + rw]
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(crop.astype(np.uint8) if nearest else crop, (width, height),
                      interpolation=interpolation)


def predict_rgb(rgb: np.ndarray) -> dict:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image")
    rgb512, geometry = _letterbox(np.asarray(rgb, np.uint8))
    identifier = hashlib.sha256(rgb512.tobytes()).hexdigest()
    cache_root = ARTIFACTS / "web_v75_cache" / identifier
    output = cache_root / "prediction.npz"
    if output.is_file():
        with np.load(output, allow_pickle=False) as values:
            return {key: values[key] for key in values.files}
    row = {"id": identifier, "group": identifier,
           "sha256": {"image": hashlib.sha256(rgb512.tobytes()).hexdigest()}}
    payloads = [{"row": row, "rgb": rgb512}]
    _sam_cache(payloads, cache_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v40_report, _ = _report("dense_dino_crossval",
        "v40_frozen_dense_dinov2_colour_decoder_reference_v2")
    v64_report, _ = _report("multiview_crossval_v64", "v64_supervised_v54_multiview_group_oof")
    v67_report, _ = _report("style_multiview_v67",
        "v67_supervised_v54_multiview_unmarked_style_group_oof")
    v74_report, _ = _report("binary_crossval",
        "v74_binary_segformer_b2_native512_unmarked_lab_style_reference_v2")
    branch = {
        "v40": _dino_average(payloads, v40_report, device)[0],
        "v64": _multiview_average(payloads, v64_report, cache_root, device)[0],
        "v67": _multiview_average(payloads, v67_report, cache_root, device)[0],
        "v74": _v70_average(payloads, v74_report, device)[0],
    }
    with np.load(cache_root / "sam" / f"{identifier}.npz", allow_pickle=False) as values:
        lung512 = values["lung"].astype(bool)
    refined = {key: refine_score(rgb512, lung512, value) for key, value in branch.items()}
    score512 = sum(WEIGHTS[key] * refined[key] for key in WEIGHTS).astype(np.float32)
    lesion512 = (score512 >= LESION_THRESHOLD) & lung512
    lung = _restore(lung512, geometry, True).astype(bool)
    lesion = _restore(lesion512, geometry, True).astype(bool) & lung
    score = _restore(score512, geometry).astype(np.float32)
    ratio = float(lesion.sum() / max(int(lung.sum()), 1))
    cache_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, lung=lung.astype(np.uint8), lesion=lesion.astype(np.uint8),
                        score=score, ratio=np.asarray(ratio, np.float32))
    del branch, refined
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"lung": lung.astype(np.uint8), "lesion": lesion.astype(np.uint8),
            "score": score, "ratio": np.asarray(ratio, np.float32)}
