from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image

from .io import (ARTIFACTS, RUNS, PROJECT_ROOT, LUNG_UNET_CHECKPOINT,
                 SAM2_CHECKPOINT, atomic_json, json_hash, now, sha256,
                 within_project)
from .metrics import binary_metrics, pooled
from .model import load_model
from .data import load_audit, validate_fit_rows
from .reference import ReferenceStore
from .segmentation import (LesionConfig, foreground_mask, lesion_mask, rembg_alpha,
                           robust_features, unlabeled_lung_probability, color_regions,
                           region_lesion_mask, grabcut_lesion_mask, sam2_boundary_refiner,
                           conservative_sam_fusion, region_consensus_from_seed)
from .segmentation import embedding_tissue_regions
from .segmentation import erode_prediction, cap_mask_by_score, lesion_score
from .color_region.teacher import (INSTRUCTION as COLOR_REGION_INSTRUCTION,
                                   ZeroShotRegionTeacher, _qwen_provenance,
                                   make_region_crop, region_probability_map,
                                   region_statistics)
from .color_region.student import (DinoCropEncoder, RegionMLP, _dino_provenance,
                                   load_student, scale_rgb,
                                   student_region_probabilities)
from .color_region.patch_cnn import PatchCNN, load_patch_cnn, predict_regions


def _measure(payload: dict, predicted_lung: np.ndarray, predicted_lesion: np.ndarray) -> dict:
    reference_lung = payload["lung"]
    reference_lesion = payload["lesion"] & reference_lung
    lung_valid = payload["valid"] & ~payload["ignore_lung"]
    lesion_valid = lung_valid & ~payload["ignore_lesion"] & reference_lung
    strict_valid = payload["valid"] & ~payload["ignore_lung"] & ~payload["ignore_lesion"]
    lung_metrics = binary_metrics(predicted_lung, reference_lung, lung_valid)
    lesion_metrics = binary_metrics(predicted_lesion, reference_lesion, lesion_valid)
    strict_metrics = binary_metrics(predicted_lesion, reference_lesion, strict_valid)
    predicted_denominator = int((predicted_lung & payload["valid"]).sum())
    reference_denominator = int((reference_lung & payload["valid"]).sum())
    predicted_ratio = (float((predicted_lesion & payload["valid"]).sum()) / predicted_denominator
                       if predicted_denominator else None)
    reference_ratio = (float((reference_lesion & payload["valid"]).sum()) / reference_denominator
                       if reference_denominator else None)
    return {
        "id": payload["row"]["id"], "group": payload["row"]["group"],
        "lung": lung_metrics, "lesion_within_reference_lung": lesion_metrics,
        "lesion_strict_canvas": strict_metrics,
        "predicted_lung_pixels": predicted_denominator,
        "predicted_lesion_pixels": int((predicted_lesion & payload["valid"]).sum()),
        "reference_lung_pixels": reference_denominator,
        "reference_lesion_pixels": int((reference_lesion & payload["valid"]).sum()),
        "predicted_ratio": predicted_ratio, "reference_ratio": reference_ratio,
        "ratio_absolute_error": (abs(predicted_ratio - reference_ratio)
                                 if predicted_ratio is not None and reference_ratio is not None else None),
    }


def summarize(rows: list[dict]) -> dict:
    lesion = pooled([row["lesion_within_reference_lung"] for row in rows])
    lung = pooled([row["lung"] for row in rows])
    strict = pooled([row["lesion_strict_canvas"] for row in rows])
    ratio_errors = [row["ratio_absolute_error"] for row in rows if row["ratio_absolute_error"] is not None]
    macro = {}
    for name, key in (("lung", "lung"), ("lesion", "lesion_within_reference_lung"),
                      ("strict", "lesion_strict_canvas")):
        macro[name] = {metric: float(np.mean([row[key][metric] for row in rows
                                               if row[key][metric] is not None]))
                       for metric in ("dice", "iou", "precision", "recall", "specificity",
                                      "balanced_accuracy")}
    return {"images": len(rows), "groups": len({row["group"] for row in rows}),
            "pooled_lung": lung, "pooled_lesion": lesion, "pooled_strict_canvas": strict,
            "macro_image": macro,
            "ratio_mae_percentage_points": 100 * float(np.mean(ratio_errors)) if ratio_errors else None}


def _candidate_result(store: ReferenceStore, config: LesionConfig,
                      foreground_method: str = "classical",
                      alpha_threshold: int = 128,
                      region_support: float | None = None,
                      region_segments: int = 450) -> tuple[dict, list[dict]]:
    prepared = _prepare_store(store, foreground_method, alpha_threshold)
    if region_support is not None:
        for item in prepared:
            item["region_labels"] = color_regions(item["payload"]["rgb"], item["predicted_lung"], region_segments)
    return _score_prepared(prepared, config, region_support)


def _prepare_store(store: ReferenceStore, foreground_method: str = "classical",
                   alpha_threshold: int = 128) -> list[dict]:
    prepared = []
    for metadata in store.rows:
        payload = store.load(metadata)
        if foreground_method == "classical":
            predicted_lung = foreground_mask(payload["rgb"])
        elif foreground_method == "rembg_u2netp":
            predicted_lung = rembg_alpha(payload["rgb"]) >= alpha_threshold
        elif foreground_method == "rembg_isnet_general_use":
            predicted_lung = rembg_alpha(payload["rgb"], "isnet-general-use") >= alpha_threshold
        elif foreground_method == "rembg_u2net":
            predicted_lung = rembg_alpha(payload["rgb"], "u2net") >= alpha_threshold
        elif foreground_method == "unlabeled_unet_v1":
            predicted_lung = unlabeled_lung_probability(payload["rgb"]) >= alpha_threshold
        else:
            raise ValueError(f"Unknown foreground method: {foreground_method}")
        prepared.append({"payload": payload, "predicted_lung": predicted_lung,
                         "features": robust_features(payload["rgb"], predicted_lung)})
    return prepared


def _score_prepared(prepared: list[dict], config: LesionConfig,
                    region_support: float | None = None) -> tuple[dict, list[dict]]:
    rows = []
    for item in prepared:
        if region_support is None:
            predicted_lesion = lesion_mask(item["features"], item["predicted_lung"], config)
        else:
            predicted_lesion = region_lesion_mask(item["features"], item["predicted_lung"],
                                                  item["region_labels"], config, region_support)
        rows.append(_measure(item["payload"], item["predicted_lung"], predicted_lesion))
    return summarize(rows), rows


def calibrate() -> dict:
    model = load_model()
    store = ReferenceStore("calibration")
    prepared = _prepare_store(store)
    results = []
    for index, candidate in enumerate(model["candidates"], start=1):
        config = LesionConfig(**candidate["config"])
        summary, _ = _score_prepared(prepared, config)
        quality = summary["pooled_lesion"]
        floor = min(value for value in (quality["precision"], quality["recall"], quality["specificity"])
                    if value is not None)
        results.append({**candidate, "summary": summary,
                        "selection_key": [quality["dice"], floor,
                                          -summary["ratio_mae_percentage_points"]]})
        if index % 32 == 0:
            print(f"calibration candidates: {index}/{len(model['candidates'])}", flush=True)
    results.sort(key=lambda row: tuple(row["selection_key"]), reverse=True)
    winner = results[0]
    candidate = {
        "status": "frozen_after_calibration", "created_at": now(),
        "model_id": model["model_id"], "model_fingerprint": model["artifact_fingerprint"],
        "selection_role": "calibration_validation_only", "test_payload_opened": False,
        "family": winner["family"], "fit_quantile": winner["fit_quantile"],
        "config": winner["config"], "calibration_summary": winner["summary"],
    }
    candidate["candidate_fingerprint"] = json_hash(candidate)
    run_id = f"calibration_{uuid.uuid4().hex[:16]}"
    run = {"status": "done", "run_id": run_id, "created_at": now(),
           "model_id": model["model_id"], "candidate_count": len(results),
           "winner": candidate, "results": results,
           "test_payload_opened": False}
    atomic_json(RUNS / run_id / "report.json", run)
    folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
    atomic_json(folder / "candidate.json", candidate)
    atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                {"path": str((folder / "candidate.json").relative_to(ARTIFACTS.parent.parent)),
                 "sha256": sha256(folder / "candidate.json"),
                 "candidate_fingerprint": candidate["candidate_fingerprint"]})
    return run


def calibrate_rembg(model_name: str = "u2netp") -> dict:
    """Materially new v2: generic foreground model plus the frozen appearance fit."""
    model = load_model()
    store = ReferenceStore("calibration")
    payloads = []
    for index, metadata in enumerate(store.rows, start=1):
        payload = store.load(metadata)
        payloads.append({"payload": payload, "alpha": rembg_alpha(payload["rgb"], model_name)})
        print(f"rembg calibration foreground: {index}/{len(store.rows)}", flush=True)
    results = []
    alpha_thresholds = ((64, 96, 128, 160, 192) if model_name == "u2netp"
                        else (8, 16, 32, 48, 64, 96, 128))
    method_name = {"u2netp": "rembg_u2netp", "isnet-general-use": "rembg_isnet_general_use",
                   "u2net": "rembg_u2net"}[model_name]
    for alpha_threshold in alpha_thresholds:
        prepared = []
        for item in payloads:
            lung = item["alpha"] >= alpha_threshold
            prepared.append({"payload": item["payload"], "predicted_lung": lung,
                             "features": robust_features(item["payload"]["rgb"], lung)})
        for candidate in model["candidates"]:
            summary, _ = _score_prepared(prepared, LesionConfig(**candidate["config"]))
            quality = summary["pooled_lesion"]
            floor = min(value for value in (quality["precision"], quality["recall"], quality["specificity"])
                        if value is not None)
            lung_dice = summary["pooled_lung"]["dice"] or 0.0
            results.append({**candidate, "foreground_method": method_name,
                            "foreground_alpha_threshold": alpha_threshold,
                            "summary": summary,
                            "selection_key": [int(lung_dice >= 0.90), quality["dice"], floor,
                                              lung_dice, -summary["ratio_mae_percentage_points"]]})
        print(f"rembg calibration alpha threshold complete: {alpha_threshold}", flush=True)
    results.sort(key=lambda row: tuple(row["selection_key"]), reverse=True)
    winner = results[0]
    if model_name == "u2netp":
        weight = (ARTIFACTS.parent / "../Lung_Region_Contrast_Lab/weights/rembg/u2netp.onnx").resolve()
    else:
        filename = "isnet-general-use.onnx" if model_name == "isnet-general-use" else "u2net.onnx"
        weight = (ARTIFACTS.parent / f"../Legacy_Project/Test/models/{filename}").resolve()
    candidate = {
        "status": "frozen_after_calibration", "created_at": now(),
        "version": f"v3_{model_name.replace('-', '_')}",
        "model_id": model["model_id"], "model_fingerprint": model["artifact_fingerprint"],
        "selection_role": "calibration_validation_only", "test_payload_opened": False,
        "foreground_method": winner["foreground_method"],
        "foreground_alpha_threshold": winner["foreground_alpha_threshold"],
        "foreground_weight_sha256": sha256(weight),
        "foreground_weight_training": "Generic official u2netp; no project reference masks.",
        "family": winner["family"], "fit_quantile": winner["fit_quantile"],
        "config": winner["config"], "calibration_summary": winner["summary"],
    }
    candidate["candidate_fingerprint"] = json_hash(candidate)
    run_id = f"calibration_{model_name.replace('-', '_')}_{uuid.uuid4().hex[:16]}"
    run = {"status": "done", "run_id": run_id, "created_at": now(),
           "model_id": model["model_id"], "candidate_count": len(results),
           "winner": candidate, "results": results, "test_payload_opened": False}
    atomic_json(RUNS / run_id / "report.json", run)
    folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
    atomic_json(folder / "candidate.json", candidate)
    atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                {"path": str((folder / "candidate.json").relative_to(ARTIFACTS.parent.parent)),
                 "sha256": sha256(folder / "candidate.json"),
                 "candidate_fingerprint": candidate["candidate_fingerprint"]})
    return run


def calibrate_unlabeled_unet() -> dict:
    model = load_model()
    store = ReferenceStore("calibration")
    payloads = []
    for index, metadata in enumerate(store.rows, start=1):
        payload = store.load(metadata)
        payloads.append({"payload": payload,
                         "probability": unlabeled_lung_probability(payload["rgb"])})
        print(f"unlabeled U-Net calibration: {index}/{len(store.rows)}", flush=True)
    results = []
    thresholds = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
    for threshold in thresholds:
        prepared = []
        for item in payloads:
            lung = item["probability"] >= threshold
            prepared.append({"payload": item["payload"], "predicted_lung": lung,
                             "features": robust_features(item["payload"]["rgb"], lung)})
        for appearance in model["candidates"]:
            summary, _ = _score_prepared(prepared, LesionConfig(**appearance["config"]))
            quality = summary["pooled_lesion"]
            floor = min(value for value in (quality["precision"], quality["recall"], quality["specificity"])
                        if value is not None)
            lung_dice = summary["pooled_lung"]["dice"] or 0.0
            results.append({**appearance, "foreground_method": "unlabeled_unet_v1",
                            "foreground_alpha_threshold": threshold, "summary": summary,
                            "selection_key": [int(lung_dice >= 0.90), quality["dice"], floor,
                                              lung_dice, -summary["ratio_mae_percentage_points"]]})
        print(f"unlabeled U-Net threshold complete: {threshold}", flush=True)
    results.sort(key=lambda row: tuple(row["selection_key"]), reverse=True)
    winner = results[0]
    checkpoint = LUNG_UNET_CHECKPOINT
    candidate = {
        "status": "frozen_after_calibration", "created_at": now(), "version": "v4_unlabeled_unet",
        "model_id": model["model_id"], "model_fingerprint": model["artifact_fingerprint"],
        "selection_role": "calibration_validation_only", "test_payload_opened": False,
        "foreground_method": "unlabeled_unet_v1",
        "foreground_alpha_threshold": winner["foreground_alpha_threshold"],
        "foreground_weight_sha256": sha256(checkpoint),
        "foreground_weight_training": "Small U-Net fitted to Lab/GrabCut pseudo-masks on 1,764 unmarked images; no human/PDF masks.",
        "family": winner["family"], "fit_quantile": winner["fit_quantile"],
        "config": winner["config"], "calibration_summary": winner["summary"],
    }
    candidate["candidate_fingerprint"] = json_hash(candidate)
    run_id = f"calibration_unlabeled_unet_{uuid.uuid4().hex[:16]}"
    run = {"status": "done", "run_id": run_id, "created_at": now(),
           "model_id": model["model_id"], "candidate_count": len(results),
           "winner": candidate, "results": results, "test_payload_opened": False}
    atomic_json(RUNS / run_id / "report.json", run)
    folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
    atomic_json(folder / "candidate.json", candidate)
    atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                {"path": str((folder / "candidate.json").relative_to(ARTIFACTS.parent.parent)),
                 "sha256": sha256(folder / "candidate.json"),
                 "candidate_fingerprint": candidate["candidate_fingerprint"]})
    return run


def calibrate_region_unet() -> dict:
    model = load_model()
    store = ReferenceStore("calibration")
    payloads = []
    for index, metadata in enumerate(store.rows, start=1):
        payload = store.load(metadata)
        payloads.append({"payload": payload,
                         "probability": unlabeled_lung_probability(payload["rgb"])})
        print(f"region U-Net calibration: {index}/{len(store.rows)}", flush=True)
    results = []
    for threshold in (0.30, 0.40, 0.50):
        prepared = []
        for item in payloads:
            lung = item["probability"] >= threshold
            prepared.append({"payload": item["payload"], "predicted_lung": lung,
                             "features": robust_features(item["payload"]["rgb"], lung),
                             "region_labels": color_regions(item["payload"]["rgb"], lung, 450)})
        for support in (0.30, 0.50, 0.70):
            for appearance in model["candidates"]:
                summary, _ = _score_prepared(prepared, LesionConfig(**appearance["config"]), support)
                quality = summary["pooled_lesion"]
                floor = min(value for value in (quality["precision"], quality["recall"], quality["specificity"])
                            if value is not None)
                lung_dice = summary["pooled_lung"]["dice"] or 0.0
                results.append({**appearance, "foreground_method": "unlabeled_unet_v1",
                                "foreground_alpha_threshold": threshold,
                                "lesion_postprocess": "slic_region_support",
                                "lesion_region_segments": 450,
                                "lesion_region_support": support, "summary": summary,
                                "selection_key": [int(lung_dice >= 0.90), quality["dice"], floor,
                                                  lung_dice, -summary["ratio_mae_percentage_points"]]})
            print(f"region U-Net complete: threshold={threshold}, support={support}", flush=True)
    results.sort(key=lambda row: tuple(row["selection_key"]), reverse=True)
    winner = results[0]
    checkpoint = LUNG_UNET_CHECKPOINT
    candidate = {
        "status": "frozen_after_calibration", "created_at": now(), "version": "v6_region_unet",
        "model_id": model["model_id"], "model_fingerprint": model["artifact_fingerprint"],
        "selection_role": "calibration_validation_only", "test_payload_opened": False,
        "foreground_method": "unlabeled_unet_v1",
        "foreground_alpha_threshold": winner["foreground_alpha_threshold"],
        "foreground_weight_sha256": sha256(checkpoint),
        "foreground_weight_training": "Small U-Net fitted to Lab/GrabCut pseudo-masks on 1,764 unmarked images; no human/PDF masks.",
        "lesion_postprocess": winner["lesion_postprocess"],
        "lesion_region_segments": winner["lesion_region_segments"],
        "lesion_region_support": winner["lesion_region_support"],
        "family": winner["family"], "fit_quantile": winner["fit_quantile"],
        "config": winner["config"], "calibration_summary": winner["summary"],
    }
    candidate["candidate_fingerprint"] = json_hash(candidate)
    run_id = f"calibration_region_unet_{uuid.uuid4().hex[:16]}"
    run = {"status": "done", "run_id": run_id, "created_at": now(),
           "model_id": model["model_id"], "candidate_count": len(results),
           "winner": candidate, "results": results, "test_payload_opened": False}
    atomic_json(RUNS / run_id / "report.json", run)
    folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
    atomic_json(folder / "candidate.json", candidate)
    atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                {"path": str((folder / "candidate.json").relative_to(ARTIFACTS.parent.parent)),
                 "sha256": sha256(folder / "candidate.json"),
                 "candidate_fingerprint": candidate["candidate_fingerprint"]})
    return run


def _score_prepared_grabcut(prepared: list[dict], config: LesionConfig,
                            core_threshold: float, iterations: int = 2) -> tuple[dict, list[dict]]:
    rows = []
    for item in prepared:
        prediction = grabcut_lesion_mask(item["payload"]["rgb"], item["features"],
                                         item["predicted_lung"], config,
                                         core_threshold, iterations)
        rows.append(_measure(item["payload"], item["predicted_lung"], prediction))
    return summarize(rows), rows


def calibrate_grabcut_unet() -> dict:
    model = load_model()
    store = ReferenceStore("calibration")
    payloads = []
    for index, metadata in enumerate(store.rows, start=1):
        payload = store.load(metadata)
        payloads.append({"payload": payload,
                         "probability": unlabeled_lung_probability(payload["rgb"])})
        print(f"grabcut U-Net calibration: {index}/{len(store.rows)}", flush=True)
    # One canonical morphology choice per family/quantile keeps the search bounded.
    canonical = {}
    for candidate in model["candidates"]:
        if (candidate["fit_quantile"] in {0.30, 0.40, 0.50}
                and candidate["config"]["minimum_dark"] == -0.25
                and candidate["config"]["minimum_component_fraction"] == 0.0008):
            canonical[(candidate["family"], candidate["fit_quantile"])] = candidate
    results = []
    for threshold in (0.30, 0.50):
        prepared = []
        for item in payloads:
            lung = item["probability"] >= threshold
            prepared.append({"payload": item["payload"], "predicted_lung": lung,
                             "features": robust_features(item["payload"]["rgb"], lung)})
        for (family, low_quantile), appearance in sorted(canonical.items()):
            for high_quantile in (0.60, 0.70, 0.78, 0.84):
                core_threshold = model["distributions"][family]["quantiles"][str(high_quantile)]
                summary, _ = _score_prepared_grabcut(
                    prepared, LesionConfig(**appearance["config"]), core_threshold, 2)
                quality = summary["pooled_lesion"]
                floor = min(value for value in (quality["precision"], quality["recall"], quality["specificity"])
                            if value is not None)
                lung_dice = summary["pooled_lung"]["dice"] or 0.0
                results.append({**appearance, "foreground_method": "unlabeled_unet_v1",
                                "foreground_alpha_threshold": threshold,
                                "lesion_postprocess": "grabcut_seeded_gmm",
                                "lesion_core_quantile": high_quantile,
                                "lesion_core_threshold": core_threshold,
                                "lesion_grabcut_iterations": 2,
                                "summary": summary,
                                "selection_key": [int(lung_dice >= 0.90), quality["dice"], floor,
                                                  lung_dice, -summary["ratio_mae_percentage_points"]]})
            print(f"grabcut complete: lung={threshold}, family={family}, low={low_quantile}", flush=True)
    results.sort(key=lambda row: tuple(row["selection_key"]), reverse=True)
    winner = results[0]
    checkpoint = LUNG_UNET_CHECKPOINT
    candidate = {
        "status": "frozen_after_calibration", "created_at": now(), "version": "v7_grabcut_unet",
        "model_id": model["model_id"], "model_fingerprint": model["artifact_fingerprint"],
        "selection_role": "calibration_validation_only", "test_payload_opened": False,
        "foreground_method": "unlabeled_unet_v1",
        "foreground_alpha_threshold": winner["foreground_alpha_threshold"],
        "foreground_weight_sha256": sha256(checkpoint),
        "foreground_weight_training": "Small U-Net fitted to Lab/GrabCut pseudo-masks on 1,764 unmarked images; no human/PDF masks.",
        "lesion_postprocess": winner["lesion_postprocess"],
        "lesion_core_quantile": winner["lesion_core_quantile"],
        "lesion_core_threshold": winner["lesion_core_threshold"],
        "lesion_grabcut_iterations": winner["lesion_grabcut_iterations"],
        "family": winner["family"], "fit_quantile": winner["fit_quantile"],
        "config": winner["config"], "calibration_summary": winner["summary"],
    }
    candidate["candidate_fingerprint"] = json_hash(candidate)
    run_id = f"calibration_grabcut_unet_{uuid.uuid4().hex[:16]}"
    run = {"status": "done", "run_id": run_id, "created_at": now(),
           "model_id": model["model_id"], "candidate_count": len(results),
           "winner": candidate, "results": results, "test_payload_opened": False}
    atomic_json(RUNS / run_id / "report.json", run)
    folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
    atomic_json(folder / "candidate.json", candidate)
    atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                {"path": str((folder / "candidate.json").relative_to(ARTIFACTS.parent.parent)),
                 "sha256": sha256(folder / "candidate.json"),
                 "candidate_fingerprint": candidate["candidate_fingerprint"]})
    return run


def calibrate_ensemble_unet() -> dict:
    model = load_model()
    store = ReferenceStore("calibration")
    prepared = _prepare_store(store, "unlabeled_unet_v1", 0.30)
    canonical = []
    for candidate in model["candidates"]:
        if (candidate["fit_quantile"] in {0.40, 0.50, 0.60, 0.70}
                and candidate["config"]["minimum_dark"] == -0.25
                and candidate["config"]["minimum_component_fraction"] == 0.0008):
            canonical.append(candidate)
    canonical.sort(key=lambda row: (row["family"], row["fit_quantile"]))
    base_masks = []
    for index, candidate in enumerate(canonical):
        config = LesionConfig(**candidate["config"])
        masks = [lesion_mask(item["features"], item["predicted_lung"], config)
                 for item in prepared]
        base_masks.append(masks)
        print(f"ensemble base masks: {index + 1}/{len(canonical)}", flush=True)
    results = []
    for left in range(len(canonical)):
        for right in range(left + 1, len(canonical)):
            for operation in ("intersection", "union"):
                rows = []
                for item, first, second in zip(prepared, base_masks[left], base_masks[right]):
                    prediction = (first & second) if operation == "intersection" else (first | second)
                    rows.append(_measure(item["payload"], item["predicted_lung"], prediction))
                summary = summarize(rows)
                quality = summary["pooled_lesion"]
                floor = min(quality[key] for key in ("precision", "recall", "specificity")
                            if quality[key] is not None)
                results.append({
                    "left": canonical[left], "right": canonical[right],
                    "ensemble_operation": operation, "summary": summary,
                    "selection_key": [floor, quality["dice"],
                                      -summary["ratio_mae_percentage_points"]],
                })
        if (left + 1) % 6 == 0:
            print(f"ensemble pairs: {left + 1}/{len(canonical)}", flush=True)
    results.sort(key=lambda row: tuple(row["selection_key"]), reverse=True)
    winner = results[0]
    checkpoint = LUNG_UNET_CHECKPOINT
    candidate = {
        "status": "frozen_after_calibration", "created_at": now(), "version": "v8_balanced_ensemble",
        "model_id": model["model_id"], "model_fingerprint": model["artifact_fingerprint"],
        "selection_role": "calibration_validation_only", "test_payload_opened": False,
        "selection_objective": "maximize minimum pooled precision/recall/specificity, then Dice",
        "foreground_method": "unlabeled_unet_v1", "foreground_alpha_threshold": 0.30,
        "foreground_weight_sha256": sha256(checkpoint),
        "foreground_weight_training": "Small U-Net fitted to Lab/GrabCut pseudo-masks on 1,764 unmarked images; no human/PDF masks.",
        "lesion_postprocess": "boolean_ensemble",
        "ensemble_operation": winner["ensemble_operation"],
        "ensemble_left": winner["left"], "ensemble_right": winner["right"],
        "calibration_summary": winner["summary"],
    }
    candidate["candidate_fingerprint"] = json_hash(candidate)
    run_id = f"calibration_ensemble_unet_{uuid.uuid4().hex[:16]}"
    run = {"status": "done", "run_id": run_id, "created_at": now(),
           "model_id": model["model_id"], "candidate_count": len(results),
           "winner": candidate, "results": results, "test_payload_opened": False}
    atomic_json(RUNS / run_id / "report.json", run)
    folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
    atomic_json(folder / "candidate.json", candidate)
    atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                {"path": str((folder / "candidate.json").relative_to(ARTIFACTS.parent.parent)),
                 "sha256": sha256(folder / "candidate.json"),
                 "candidate_fingerprint": candidate["candidate_fingerprint"]})
    return run


def _score_prepared_ensemble(prepared: list[dict], left: LesionConfig,
                             right: LesionConfig, operation: str) -> tuple[dict, list[dict]]:
    if operation not in {"intersection", "union"}:
        raise ValueError("Unknown ensemble operation")
    rows = []
    for item in prepared:
        first = lesion_mask(item["features"], item["predicted_lung"], left)
        second = lesion_mask(item["features"], item["predicted_lung"], right)
        prediction = first & second if operation == "intersection" else first | second
        rows.append(_measure(item["payload"], item["predicted_lung"], prediction))
    return summarize(rows), rows


def _ensemble_masks(prepared: list[dict], left: LesionConfig,
                    right: LesionConfig, operation: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    probable, core = [], []
    for item in prepared:
        first = lesion_mask(item["features"], item["predicted_lung"], left)
        second = lesion_mask(item["features"], item["predicted_lung"], right)
        probable.append(first & second if operation == "intersection" else first | second)
        core.append(first & second)
    return probable, core


def _score_sam2_variant(prepared: list[dict], proposals: list[np.ndarray],
                        probable: list[np.ndarray], composition: str) -> tuple[dict, list[dict]]:
    rows = []
    for item, proposal, base in zip(prepared, proposals, probable):
        if composition == "sam_only":
            prediction = proposal
        elif composition == "sam_intersection_dilated_seed":
            radius = max(2, round(min(base.shape) * 0.01))
            import cv2
            dilated = cv2.dilate(base.astype(np.uint8),
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                           (2 * radius + 1, 2 * radius + 1))) > 0
            prediction = proposal & dilated
        else:
            raise ValueError(f"Unknown SAM2 composition: {composition}")
        rows.append(_measure(item["payload"], item["predicted_lung"], prediction))
    return summarize(rows), rows


def calibrate_sam2_unet() -> dict:
    """v9: generic SAM2 boundaries around frozen v8 reference-free seeds."""
    model = load_model()
    base = load_candidate()
    if base.get("version") != "v8_balanced_ensemble":
        raise ValueError("SAM2 refinement requires the frozen v8 balanced ensemble as its base")
    store = ReferenceStore("calibration")
    prepared = _prepare_store(store, base["foreground_method"], base["foreground_alpha_threshold"])
    left = LesionConfig(**base["ensemble_left"]["config"])
    right = LesionConfig(**base["ensemble_right"]["config"])
    probable, core = _ensemble_masks(prepared, left, right, base["ensemble_operation"])
    run_id = f"calibration_sam2_unet_{uuid.uuid4().hex[:16]}"
    cache = RUNS / run_id / "sam2_cache"
    refiner = sam2_boundary_refiner()
    settings = [(0.05, 0.0), (0.05, 0.01), (0.12, 0.0), (0.12, 0.01)]
    proposal_sets = {setting: [] for setting in settings}
    cache_records = []
    for image_index, (item, base_mask, core_mask) in enumerate(zip(prepared, probable, core), start=1):
        image_record = {"id": item["payload"]["row"]["id"], "settings": []}
        arrays = {}
        for setting_index, (padding, erosion) in enumerate(settings):
            proposal, metadata, raw = refiner.refine(
                item["payload"]["rgb"], item["predicted_lung"], base_mask, core_mask,
                padding, erosion, encode_image=(setting_index == 0))
            proposal_sets[(padding, erosion)].append(proposal)
            arrays[f"proposal_{setting_index}"] = proposal.astype(np.uint8)
            for raw_index, mask in enumerate(raw):
                arrays[f"raw_{setting_index}_{raw_index}"] = mask.astype(np.uint8)
            image_record["settings"].append({"padding_fraction": padding,
                                              "erosion_fraction": erosion,
                                              "metadata": metadata,
                                              "raw_mask_count": len(raw)})
        cache.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache / f"{image_record['id']}.npz", **arrays)
        atomic_json(cache / f"{image_record['id']}.json", image_record)
        cache_records.append(image_record)
        print(f"SAM2 calibration proposals: {image_index}/{len(prepared)}", flush=True)
    results = []
    for padding, erosion in settings:
        for composition in ("sam_only", "sam_intersection_dilated_seed"):
            summary, rows = _score_sam2_variant(
                prepared, proposal_sets[(padding, erosion)], probable, composition)
            quality = summary["pooled_lesion"]
            floor = min(quality[key] for key in ("precision", "recall", "specificity")
                        if quality[key] is not None)
            results.append({"padding_fraction": padding, "erosion_fraction": erosion,
                            "composition": composition, "summary": summary,
                            "per_image": rows,
                            "selection_key": [floor, quality["dice"],
                                              -summary["ratio_mae_percentage_points"]]})
    results.sort(key=lambda row: tuple(row["selection_key"]), reverse=True)
    winner = results[0]
    checkpoint = SAM2_CHECKPOINT
    candidate = {
        "status": "frozen_after_calibration", "created_at": now(), "version": "v9_sam2_boundary",
        "model_id": model["model_id"], "model_fingerprint": model["artifact_fingerprint"],
        "selection_role": "calibration_validation_only", "test_payload_opened": False,
        "selection_objective": "maximize minimum pooled precision/recall/specificity, then Dice",
        "foreground_method": base["foreground_method"],
        "foreground_alpha_threshold": base["foreground_alpha_threshold"],
        "foreground_weight_sha256": base["foreground_weight_sha256"],
        "foreground_weight_training": base["foreground_weight_training"],
        "lesion_postprocess": "sam2_boundary_from_unlabeled_ensemble",
        "sam2_weight_sha256": sha256(checkpoint),
        "sam2_training": "Official generic SAM2.1 Hiera Tiny; no project reference masks.",
        "sam2_padding_fraction": winner["padding_fraction"],
        "sam2_erosion_fraction": winner["erosion_fraction"],
        "sam2_composition": winner["composition"],
        "ensemble_operation": base["ensemble_operation"],
        "ensemble_left": base["ensemble_left"], "ensemble_right": base["ensemble_right"],
        "calibration_summary": winner["summary"],
    }
    candidate["candidate_fingerprint"] = json_hash(candidate)
    run = {"status": "done", "run_id": run_id, "created_at": now(),
           "model_id": model["model_id"], "candidate_count": len(results),
           "base_candidate_fingerprint": base["candidate_fingerprint"],
           "sam2_cache_records": cache_records, "winner": candidate,
           "results": results, "test_payload_opened": False}
    atomic_json(RUNS / run_id / "report.json", run)
    base_quality = base["calibration_summary"]["pooled_lesion"]
    new_quality = winner["summary"]["pooled_lesion"]
    base_floor = min(base_quality[key] for key in ("precision", "recall", "specificity"))
    new_floor = min(new_quality[key] for key in ("precision", "recall", "specificity"))
    run["replaces_current"] = bool(new_floor > base_floor and new_quality["dice"] > base_quality["dice"])
    atomic_json(RUNS / run_id / "report.json", run)
    if run["replaces_current"]:
        folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
        atomic_json(folder / "candidate.json", candidate)
        atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                    {"path": str((folder / "candidate.json").relative_to(ARTIFACTS.parent.parent)),
                     "sha256": sha256(folder / "candidate.json"),
                     "candidate_fingerprint": candidate["candidate_fingerprint"]})
    return run


def calibrate_sam2_fusion() -> dict:
    """v10: conservative v8 seed intersection with immutable cached v9 SAM masks."""
    model = load_model()
    base = load_candidate()
    if base.get("version") != "v8_balanced_ensemble":
        raise ValueError("Conservative SAM2 fusion requires v8 as the current base")
    completed = []
    for path in RUNS.glob("calibration_sam2_unet_*/report.json"):
        report = json.loads(path.read_text())
        if report.get("status") == "done" and report.get("winner", {}).get("version") == "v9_sam2_boundary":
            completed.append((report["created_at"], path, report))
    if not completed:
        raise FileNotFoundError("No completed v9 SAM2 calibration cache")
    _, source_path, source = max(completed, key=lambda row: row[0])
    source_run = source["run_id"]
    store = ReferenceStore("calibration")
    prepared = _prepare_store(store, base["foreground_method"], base["foreground_alpha_threshold"])
    probable, _ = _ensemble_masks(
        prepared, LesionConfig(**base["ensemble_left"]["config"]),
        LesionConfig(**base["ensemble_right"]["config"]), base["ensemble_operation"])
    setting_lookup = []
    first = source["sam2_cache_records"][0]
    for index, setting in enumerate(first["settings"]):
        setting_lookup.append((index, setting["padding_fraction"], setting["erosion_fraction"]))
    proposals = {index: [] for index, _, _ in setting_lookup}
    cache_root = source_path.parent / "sam2_cache"
    for item in prepared:
        image_id = item["payload"]["row"]["id"]
        with np.load(cache_root / f"{image_id}.npz") as arrays:
            for index, _, _ in setting_lookup:
                proposals[index].append(np.asarray(arrays[f"proposal_{index}"], bool))
    results = []
    for setting_index, padding, erosion in setting_lookup:
        for dilation in (0.0, 0.01, 0.02):
            rows = []
            for item, seed, proposal in zip(prepared, probable, proposals[setting_index]):
                prediction = conservative_sam_fusion(seed, proposal, dilation)
                rows.append(_measure(item["payload"], item["predicted_lung"], prediction))
            summary = summarize(rows)
            quality = summary["pooled_lesion"]
            floor = min(quality[key] for key in ("precision", "recall", "specificity")
                        if quality[key] is not None)
            results.append({"source_run_id": source_run, "setting_index": setting_index,
                            "padding_fraction": padding, "erosion_fraction": erosion,
                            "sam_dilation_fraction": dilation, "summary": summary,
                            "per_image": rows,
                            "selection_key": [floor, quality["dice"],
                                              -summary["ratio_mae_percentage_points"]]})
    results.sort(key=lambda row: tuple(row["selection_key"]), reverse=True)
    winner = results[0]
    checkpoint = SAM2_CHECKPOINT
    candidate = {
        "status": "frozen_after_calibration", "created_at": now(), "version": "v10_sam2_fusion",
        "model_id": model["model_id"], "model_fingerprint": model["artifact_fingerprint"],
        "selection_role": "calibration_validation_only", "test_payload_opened": False,
        "selection_objective": "maximize minimum pooled precision/recall/specificity, then Dice",
        "foreground_method": base["foreground_method"],
        "foreground_alpha_threshold": base["foreground_alpha_threshold"],
        "foreground_weight_sha256": base["foreground_weight_sha256"],
        "foreground_weight_training": base["foreground_weight_training"],
        "lesion_postprocess": "seed_intersection_dilated_sam2",
        "sam2_weight_sha256": sha256(checkpoint),
        "sam2_training": "Official generic SAM2.1 Hiera Tiny; no project reference masks.",
        "sam2_source_calibration_run_id": source_run,
        "sam2_padding_fraction": winner["padding_fraction"],
        "sam2_erosion_fraction": winner["erosion_fraction"],
        "sam2_dilation_fraction": winner["sam_dilation_fraction"],
        "ensemble_operation": base["ensemble_operation"],
        "ensemble_left": base["ensemble_left"], "ensemble_right": base["ensemble_right"],
        "calibration_summary": winner["summary"],
    }
    candidate["candidate_fingerprint"] = json_hash(candidate)
    run_id = f"calibration_sam2_fusion_{uuid.uuid4().hex[:16]}"
    base_quality = base["calibration_summary"]["pooled_lesion"]
    new_quality = winner["summary"]["pooled_lesion"]
    base_floor = min(base_quality[key] for key in ("precision", "recall", "specificity"))
    new_floor = min(new_quality[key] for key in ("precision", "recall", "specificity"))
    run = {"status": "done", "run_id": run_id, "created_at": now(),
           "model_id": model["model_id"], "candidate_count": len(results),
           "base_candidate_fingerprint": base["candidate_fingerprint"],
           "source_sam2_run_id": source_run, "winner": candidate, "results": results,
           "test_payload_opened": False,
           "replaces_current": bool(new_floor > base_floor and new_quality["dice"] > base_quality["dice"])}
    atomic_json(RUNS / run_id / "report.json", run)
    if run["replaces_current"]:
        folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
        atomic_json(folder / "candidate.json", candidate)
        atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                    {"path": str((folder / "candidate.json").relative_to(ARTIFACTS.parent.parent)),
                     "sha256": sha256(folder / "candidate.json"),
                     "candidate_fingerprint": candidate["candidate_fingerprint"]})
    return run


def calibrate_coarse_regions() -> dict:
    """v11: coarse RGB tissue regions aggregate the frozen balanced v8 seed."""
    model = load_model()
    base = load_candidate()
    if base.get("version") != "v8_balanced_ensemble":
        raise ValueError("Coarse-region consensus requires v8 as the current base")
    store = ReferenceStore("calibration")
    prepared = _prepare_store(store, base["foreground_method"], base["foreground_alpha_threshold"])
    probable, _ = _ensemble_masks(
        prepared, LesionConfig(**base["ensemble_left"]["config"]),
        LesionConfig(**base["ensemble_right"]["config"]), base["ensemble_operation"])
    segment_counts = (40, 80, 160)
    label_sets = {}
    for segments in segment_counts:
        label_sets[segments] = [color_regions(item["payload"]["rgb"], item["predicted_lung"], segments)
                                for item in prepared]
        print(f"coarse regions ready: {segments}", flush=True)
    results = []
    for segments in segment_counts:
        for support in (0.25, 0.40, 0.55, 0.70):
            rows = []
            for item, seed, labels in zip(prepared, probable, label_sets[segments]):
                prediction = region_consensus_from_seed(labels, item["predicted_lung"], seed, support)
                rows.append(_measure(item["payload"], item["predicted_lung"], prediction))
            summary = summarize(rows)
            quality = summary["pooled_lesion"]
            floor = min(quality[key] for key in ("precision", "recall", "specificity")
                        if quality[key] is not None)
            results.append({"region_segments": segments, "region_support": support,
                            "summary": summary, "per_image": rows,
                            "selection_key": [floor, quality["dice"],
                                              -summary["ratio_mae_percentage_points"]]})
    results.sort(key=lambda row: tuple(row["selection_key"]), reverse=True)
    winner = results[0]
    candidate = {
        "status": "frozen_after_calibration", "created_at": now(), "version": "v11_coarse_regions",
        "model_id": model["model_id"], "model_fingerprint": model["artifact_fingerprint"],
        "selection_role": "calibration_validation_only", "test_payload_opened": False,
        "selection_objective": "maximize minimum pooled precision/recall/specificity, then Dice",
        "foreground_method": base["foreground_method"],
        "foreground_alpha_threshold": base["foreground_alpha_threshold"],
        "foreground_weight_sha256": base["foreground_weight_sha256"],
        "foreground_weight_training": base["foreground_weight_training"],
        "lesion_postprocess": "coarse_region_consensus",
        "lesion_region_segments": winner["region_segments"],
        "lesion_region_support": winner["region_support"],
        "ensemble_operation": base["ensemble_operation"],
        "ensemble_left": base["ensemble_left"], "ensemble_right": base["ensemble_right"],
        "calibration_summary": winner["summary"],
    }
    candidate["candidate_fingerprint"] = json_hash(candidate)
    run_id = f"calibration_coarse_regions_{uuid.uuid4().hex[:16]}"
    base_quality = base["calibration_summary"]["pooled_lesion"]
    new_quality = winner["summary"]["pooled_lesion"]
    base_floor = min(base_quality[key] for key in ("precision", "recall", "specificity"))
    new_floor = min(new_quality[key] for key in ("precision", "recall", "specificity"))
    run = {"status": "done", "run_id": run_id, "created_at": now(),
           "model_id": model["model_id"], "candidate_count": len(results),
           "base_candidate_fingerprint": base["candidate_fingerprint"],
           "winner": candidate, "results": results, "test_payload_opened": False,
           "replaces_current": bool(new_floor > base_floor and new_quality["dice"] > base_quality["dice"])}
    atomic_json(RUNS / run_id / "report.json", run)
    if run["replaces_current"]:
        folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
        atomic_json(folder / "candidate.json", candidate)
        atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                    {"path": str((folder / "candidate.json").relative_to(ARTIFACTS.parent.parent)),
                     "sha256": sha256(folder / "candidate.json"),
                     "candidate_fingerprint": candidate["candidate_fingerprint"]})
    return run


def calibrate_coarse_regions_refined() -> dict:
    """v11b: calibration interpolation at the v11 precision/recall crossing."""
    model = load_model()
    base = load_candidate()
    if base.get("version") != "v8_balanced_ensemble":
        raise ValueError("Refined coarse-region calibration requires v8 as current")
    store = ReferenceStore("calibration")
    prepared = _prepare_store(store, base["foreground_method"], base["foreground_alpha_threshold"])
    probable, _ = _ensemble_masks(
        prepared, LesionConfig(**base["ensemble_left"]["config"]),
        LesionConfig(**base["ensemble_right"]["config"]), base["ensemble_operation"])
    labels = [color_regions(item["payload"]["rgb"], item["predicted_lung"], 160)
              for item in prepared]
    results = []
    for support in (0.43, 0.46, 0.49, 0.52):
        rows = []
        for item, seed, regions in zip(prepared, probable, labels):
            prediction = region_consensus_from_seed(regions, item["predicted_lung"], seed, support)
            rows.append(_measure(item["payload"], item["predicted_lung"], prediction))
        summary = summarize(rows)
        quality = summary["pooled_lesion"]
        floor = min(quality[key] for key in ("precision", "recall", "specificity")
                    if quality[key] is not None)
        results.append({"region_segments": 160, "region_support": support,
                        "summary": summary, "per_image": rows,
                        "selection_key": [floor, quality["dice"],
                                          -summary["ratio_mae_percentage_points"]]})
    results.sort(key=lambda row: tuple(row["selection_key"]), reverse=True)
    winner = results[0]
    candidate = {
        "status": "frozen_after_calibration", "created_at": now(),
        "version": "v11b_coarse_regions_refined", "model_id": model["model_id"],
        "model_fingerprint": model["artifact_fingerprint"],
        "selection_role": "calibration_validation_only", "test_payload_opened": False,
        "selection_objective": "maximize minimum pooled precision/recall/specificity, then Dice",
        "foreground_method": base["foreground_method"],
        "foreground_alpha_threshold": base["foreground_alpha_threshold"],
        "foreground_weight_sha256": base["foreground_weight_sha256"],
        "foreground_weight_training": base["foreground_weight_training"],
        "lesion_postprocess": "coarse_region_consensus",
        "lesion_region_segments": 160, "lesion_region_support": winner["region_support"],
        "ensemble_operation": base["ensemble_operation"],
        "ensemble_left": base["ensemble_left"], "ensemble_right": base["ensemble_right"],
        "calibration_summary": winner["summary"],
    }
    candidate["candidate_fingerprint"] = json_hash(candidate)
    run_id = f"calibration_coarse_regions_refined_{uuid.uuid4().hex[:16]}"
    base_quality = base["calibration_summary"]["pooled_lesion"]
    new_quality = winner["summary"]["pooled_lesion"]
    base_floor = min(base_quality[key] for key in ("precision", "recall", "specificity"))
    new_floor = min(new_quality[key] for key in ("precision", "recall", "specificity"))
    run = {"status": "done", "run_id": run_id, "created_at": now(),
           "model_id": model["model_id"], "candidate_count": len(results),
           "base_candidate_fingerprint": base["candidate_fingerprint"],
           "winner": candidate, "results": results, "test_payload_opened": False,
           "replaces_current": bool(new_floor > base_floor and new_quality["dice"] > base_quality["dice"])}
    atomic_json(RUNS / run_id / "report.json", run)
    if run["replaces_current"]:
        folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
        atomic_json(folder / "candidate.json", candidate)
        atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                    {"path": str((folder / "candidate.json").relative_to(ARTIFACTS.parent.parent)),
                     "sha256": sha256(folder / "candidate.json"),
                     "candidate_fingerprint": candidate["candidate_fingerprint"]})
    return run


def calibrate_sam2_embedding_regions() -> dict:
    """v12: per-image clustering of generic SAM2 encoder tissue embeddings."""
    model = load_model()
    base = load_candidate()
    if base.get("version") != "v11b_coarse_regions_refined":
        raise ValueError("SAM2 embedding clustering requires current v11b")
    store = ReferenceStore("calibration")
    prepared = _prepare_store(store, base["foreground_method"], base["foreground_alpha_threshold"])
    probable, _ = _ensemble_masks(
        prepared, LesionConfig(**base["ensemble_left"]["config"]),
        LesionConfig(**base["ensemble_right"]["config"]), base["ensemble_operation"])
    run_id = f"calibration_sam2_embedding_{uuid.uuid4().hex[:16]}"
    cache = RUNS / run_id / "embedding_cache"
    cache.mkdir(parents=True, exist_ok=True)
    refiner = sam2_boundary_refiner()
    settings = [(32, 0.25), (32, 0.50), (64, 0.25), (64, 0.50)]
    label_sets = {setting: [] for setting in settings}
    cache_records = []
    for index, item in enumerate(prepared, start=1):
        embedding = refiner.embedding(item["payload"]["rgb"])
        arrays = {"embedding": embedding.astype(np.float16)}
        record = {"id": item["payload"]["row"]["id"], "embedding_shape": list(embedding.shape),
                  "settings": []}
        for setting_index, (clusters, spatial) in enumerate(settings):
            labels = embedding_tissue_regions(embedding, item["predicted_lung"], clusters, spatial)
            label_sets[(clusters, spatial)].append(labels)
            arrays[f"labels_{setting_index}"] = labels.astype(np.int16)
            record["settings"].append({"cluster_count": clusters, "spatial_weight": spatial,
                                       "native_regions": int(labels.max())})
        np.savez_compressed(cache / f"{record['id']}.npz", **arrays)
        atomic_json(cache / f"{record['id']}.json", record)
        cache_records.append(record)
        print(f"SAM2 embedding regions: {index}/{len(prepared)}", flush=True)
    results = []
    for clusters, spatial in settings:
        for support in (0.30, 0.45, 0.60):
            rows = []
            for item, seed, labels in zip(prepared, probable, label_sets[(clusters, spatial)]):
                prediction = region_consensus_from_seed(labels, item["predicted_lung"], seed, support)
                rows.append(_measure(item["payload"], item["predicted_lung"], prediction))
            summary = summarize(rows)
            quality = summary["pooled_lesion"]
            floor = min(quality[key] for key in ("precision", "recall", "specificity")
                        if quality[key] is not None)
            results.append({"embedding_clusters": clusters, "embedding_spatial_weight": spatial,
                            "region_support": support, "summary": summary, "per_image": rows,
                            "selection_key": [floor, quality["dice"],
                                              -summary["ratio_mae_percentage_points"]]})
    results.sort(key=lambda row: tuple(row["selection_key"]), reverse=True)
    winner = results[0]
    checkpoint = SAM2_CHECKPOINT
    candidate = {
        "status": "frozen_after_calibration", "created_at": now(),
        "version": "v12_sam2_embedding_regions", "model_id": model["model_id"],
        "model_fingerprint": model["artifact_fingerprint"],
        "selection_role": "calibration_validation_only", "test_payload_opened": False,
        "selection_objective": "maximize minimum pooled precision/recall/specificity, then Dice",
        "foreground_method": base["foreground_method"],
        "foreground_alpha_threshold": base["foreground_alpha_threshold"],
        "foreground_weight_sha256": base["foreground_weight_sha256"],
        "foreground_weight_training": base["foreground_weight_training"],
        "lesion_postprocess": "sam2_embedding_region_consensus",
        "sam2_weight_sha256": sha256(checkpoint),
        "sam2_training": "Official generic SAM2.1 Hiera Tiny; no project reference masks.",
        "embedding_clusters": winner["embedding_clusters"],
        "embedding_spatial_weight": winner["embedding_spatial_weight"],
        "lesion_region_support": winner["region_support"],
        "ensemble_operation": base["ensemble_operation"],
        "ensemble_left": base["ensemble_left"], "ensemble_right": base["ensemble_right"],
        "calibration_summary": winner["summary"],
    }
    candidate["candidate_fingerprint"] = json_hash(candidate)
    base_quality = base["calibration_summary"]["pooled_lesion"]
    new_quality = winner["summary"]["pooled_lesion"]
    base_floor = min(base_quality[key] for key in ("precision", "recall", "specificity"))
    new_floor = min(new_quality[key] for key in ("precision", "recall", "specificity"))
    run = {"status": "done", "run_id": run_id, "created_at": now(),
           "model_id": model["model_id"], "candidate_count": len(results),
           "base_candidate_fingerprint": base["candidate_fingerprint"],
           "embedding_cache_records": cache_records, "winner": candidate,
           "results": results, "test_payload_opened": False,
           "replaces_current": bool(new_floor > base_floor and new_quality["dice"] > base_quality["dice"])}
    atomic_json(RUNS / run_id / "report.json", run)
    if run["replaces_current"]:
        folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
        atomic_json(folder / "candidate.json", candidate)
        atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                    {"path": str((folder / "candidate.json").relative_to(ARTIFACTS.parent.parent)),
                     "sha256": sha256(folder / "candidate.json"),
                     "candidate_fingerprint": candidate["candidate_fingerprint"]})
    return run


def calibrate_boundary_erosion() -> dict:
    """v13: small erosion of the frozen v11b region consensus."""
    model = load_model()
    base = load_candidate()
    if base.get("version") != "v11b_coarse_regions_refined":
        raise ValueError("Boundary erosion requires current v11b")
    store = ReferenceStore("calibration")
    prepared = _prepare_store(store, base["foreground_method"], base["foreground_alpha_threshold"])
    probable, _ = _ensemble_masks(
        prepared, LesionConfig(**base["ensemble_left"]["config"]),
        LesionConfig(**base["ensemble_right"]["config"]), base["ensemble_operation"])
    base_predictions = []
    for item, seed in zip(prepared, probable):
        labels = color_regions(item["payload"]["rgb"], item["predicted_lung"],
                               base["lesion_region_segments"])
        base_predictions.append(region_consensus_from_seed(
            labels, item["predicted_lung"], seed, base["lesion_region_support"]))
    results = []
    for radius in (0.002, 0.004, 0.006):
        rows = [_measure(item["payload"], item["predicted_lung"],
                         erode_prediction(prediction, radius))
                for item, prediction in zip(prepared, base_predictions)]
        summary = summarize(rows)
        quality = summary["pooled_lesion"]
        floor = min(quality[key] for key in ("precision", "recall", "specificity")
                    if quality[key] is not None)
        results.append({"erosion_fraction": radius, "summary": summary, "per_image": rows,
                        "selection_key": [floor, quality["dice"],
                                          -summary["ratio_mae_percentage_points"]]})
    results.sort(key=lambda row: tuple(row["selection_key"]), reverse=True)
    winner = results[0]
    candidate = {**{key: value for key, value in base.items()
                    if key not in {"created_at", "version", "candidate_fingerprint",
                                   "calibration_summary", "lesion_postprocess"}},
                 "created_at": now(), "version": "v13_boundary_erosion",
                 "lesion_postprocess": "coarse_region_consensus_eroded",
                 "lesion_erosion_fraction": winner["erosion_fraction"],
                 "calibration_summary": winner["summary"]}
    candidate["candidate_fingerprint"] = json_hash(candidate)
    run_id = f"calibration_boundary_erosion_{uuid.uuid4().hex[:16]}"
    base_quality = base["calibration_summary"]["pooled_lesion"]
    new_quality = winner["summary"]["pooled_lesion"]
    base_floor = min(base_quality[key] for key in ("precision", "recall", "specificity"))
    new_floor = min(new_quality[key] for key in ("precision", "recall", "specificity"))
    run = {"status": "done", "run_id": run_id, "created_at": now(),
           "model_id": model["model_id"], "candidate_count": len(results),
           "base_candidate_fingerprint": base["candidate_fingerprint"],
           "winner": candidate, "results": results, "test_payload_opened": False,
           "replaces_current": bool(new_floor > base_floor and new_quality["dice"] > base_quality["dice"])}
    atomic_json(RUNS / run_id / "report.json", run)
    if run["replaces_current"]:
        folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
        atomic_json(folder / "candidate.json", candidate)
        atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                    {"path": str((folder / "candidate.json").relative_to(ARTIFACTS.parent.parent)),
                     "sha256": sha256(folder / "candidate.json"),
                     "candidate_fingerprint": candidate["candidate_fingerprint"]})
    return run


def fit_ratio_prior(max_images: int | None = None) -> dict:
    """Fit image-level v8 seed-area statistics only on protected unmarked RGBs."""
    audit = load_audit()
    rows = list(audit["fit_rows"])
    if max_images is not None:
        rows = rows[:max_images]
    validate_fit_rows(rows, audit)
    candidate = load_candidate()
    if candidate.get("version") != "v11b_coarse_regions_refined":
        raise ValueError("Ratio-prior fit requires current v11b")
    left = LesionConfig(**candidate["ensemble_left"]["config"])
    right = LesionConfig(**candidate["ensemble_right"]["config"])
    fractions = []
    image_rows = []
    for index, row in enumerate(rows, start=1):
        rgb = np.asarray(Image.open(within_project(row["path"])).convert("RGB"))
        scale = min(1.0, 384 / max(rgb.shape[:2]))
        if scale < 1.0:
            rgb = cv2.resize(rgb, (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)),
                             interpolation=cv2.INTER_AREA)
        lung = unlabeled_lung_probability(rgb) >= candidate["foreground_alpha_threshold"]
        if int(lung.sum()) < 100:
            image_rows.append({"id": row["id"], "status": "no_lung"})
            continue
        features = robust_features(rgb, lung)
        first = lesion_mask(features, lung, left)
        second = lesion_mask(features, lung, right)
        prediction = first & second if candidate["ensemble_operation"] == "intersection" else first | second
        fraction = float(prediction.sum() / lung.sum())
        fractions.append(fraction)
        image_rows.append({"id": row["id"], "status": "used", "fraction": fraction,
                           "lung_pixels": int(lung.sum()), "lesion_pixels": int(prediction.sum())})
        if index % 100 == 0:
            print(f"ratio prior fit: {index}/{len(rows)}", flush=True)
    if not fractions:
        raise RuntimeError("No unmarked ratio-prior samples")
    quantiles = {str(value): float(np.quantile(fractions, value))
                 for value in (0.40, 0.50, 0.60, 0.70, 0.80)}
    artifact = {"status": "fit_complete", "created_at": now(),
                "method": "Image-level v8 appearance-union fraction within unlabeled U-Net lung",
                "source_candidate_fingerprint": candidate["candidate_fingerprint"],
                "fit_images_requested": len(rows), "fit_images_used": len(fractions),
                "fit_images_failed": len(rows) - len(fractions),
                "fit_identity_hash": json_hash([{key: row[key] for key in
                                                  ("id", "pixel_sha256", "file_sha256")}
                                                 for row in rows]),
                "data_audit_fingerprint": audit["audit_fingerprint"],
                "reference_pixels_used_for_fit": 0, "quantiles": quantiles,
                "minimum": float(np.min(fractions)), "maximum": float(np.max(fractions)),
                "mean": float(np.mean(fractions)), "image_rows": image_rows}
    artifact["artifact_fingerprint"] = json_hash(artifact)
    folder = ARTIFACTS / "ratio_prior" / artifact["artifact_fingerprint"]
    atomic_json(folder / "prior.json", artifact)
    atomic_json(ARTIFACTS / "ratio_prior/current.json",
                {"path": str((folder / "prior.json").relative_to(PROJECT_ROOT)),
                 "sha256": sha256(folder / "prior.json"),
                 "artifact_fingerprint": artifact["artifact_fingerprint"]})
    return artifact


def load_ratio_prior() -> dict:
    pointer = json.loads((ARTIFACTS / "ratio_prior/current.json").read_text())
    path = within_project(pointer["path"])
    if sha256(path) != pointer["sha256"]:
        raise ValueError("Ratio-prior checksum changed")
    artifact = json.loads(path.read_text())
    expected = artifact.pop("artifact_fingerprint")
    actual = json_hash(artifact)
    artifact["artifact_fingerprint"] = expected
    if expected != actual:
        raise ValueError("Ratio-prior fingerprint changed")
    return artifact


def calibrate_ratio_prior() -> dict:
    """v14: cap outlier lesion areas using unmarked image-level statistics."""
    model = load_model()
    base = load_candidate()
    prior = load_ratio_prior()
    if base.get("version") != "v11b_coarse_regions_refined":
        raise ValueError("Ratio-prior calibration requires current v11b")
    if prior["source_candidate_fingerprint"] != base["candidate_fingerprint"]:
        raise ValueError("Ratio prior was fit for a different candidate")
    store = ReferenceStore("calibration")
    prepared = _prepare_store(store, base["foreground_method"], base["foreground_alpha_threshold"])
    left = LesionConfig(**base["ensemble_left"]["config"])
    right = LesionConfig(**base["ensemble_right"]["config"])
    probable, _ = _ensemble_masks(prepared, left, right, base["ensemble_operation"])
    base_predictions, margins = [], []
    for item, seed in zip(prepared, probable):
        labels = color_regions(item["payload"]["rgb"], item["predicted_lung"],
                               base["lesion_region_segments"])
        base_predictions.append(region_consensus_from_seed(
            labels, item["predicted_lung"], seed, base["lesion_region_support"]))
        margins.append(np.maximum(lesion_score(item["features"], left) - left.threshold,
                                  lesion_score(item["features"], right) - right.threshold))
    results = []
    for quantile, cap in prior["quantiles"].items():
        rows = [_measure(item["payload"], item["predicted_lung"],
                         cap_mask_by_score(prediction, item["predicted_lung"], margin, cap))
                for item, prediction, margin in zip(prepared, base_predictions, margins)]
        summary = summarize(rows)
        quality = summary["pooled_lesion"]
        floor = min(quality[key] for key in ("precision", "recall", "specificity")
                    if quality[key] is not None)
        results.append({"prior_quantile": float(quantile), "maximum_fraction": cap,
                        "summary": summary, "per_image": rows,
                        "selection_key": [floor, quality["dice"],
                                          -summary["ratio_mae_percentage_points"]]})
    results.sort(key=lambda row: tuple(row["selection_key"]), reverse=True)
    winner = results[0]
    candidate = {**{key: value for key, value in base.items()
                    if key not in {"created_at", "version", "candidate_fingerprint",
                                   "calibration_summary", "lesion_postprocess"}},
                 "created_at": now(), "version": "v14_unmarked_area_prior",
                 "lesion_postprocess": "coarse_region_consensus_area_cap",
                 "ratio_prior_fingerprint": prior["artifact_fingerprint"],
                 "ratio_prior_quantile": winner["prior_quantile"],
                 "ratio_maximum_fraction": winner["maximum_fraction"],
                 "calibration_summary": winner["summary"]}
    candidate["candidate_fingerprint"] = json_hash(candidate)
    run_id = f"calibration_ratio_prior_{uuid.uuid4().hex[:16]}"
    base_quality = base["calibration_summary"]["pooled_lesion"]
    new_quality = winner["summary"]["pooled_lesion"]
    base_floor = min(base_quality[key] for key in ("precision", "recall", "specificity"))
    new_floor = min(new_quality[key] for key in ("precision", "recall", "specificity"))
    run = {"status": "done", "run_id": run_id, "created_at": now(),
           "model_id": model["model_id"], "candidate_count": len(results),
           "base_candidate_fingerprint": base["candidate_fingerprint"],
           "ratio_prior_fingerprint": prior["artifact_fingerprint"],
           "winner": candidate, "results": results, "test_payload_opened": False,
           "replaces_current": bool(new_floor > base_floor and new_quality["dice"] > base_quality["dice"])}
    atomic_json(RUNS / run_id / "report.json", run)
    if run["replaces_current"]:
        folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
        atomic_json(folder / "candidate.json", candidate)
        atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                    {"path": str((folder / "candidate.json").relative_to(ARTIFACTS.parent.parent)),
                     "sha256": sha256(folder / "candidate.json"),
                     "candidate_fingerprint": candidate["candidate_fingerprint"]})
    return run


def calibrate_color_region_qwen() -> dict:
    """v15b: cache zero-shot Qwen region scores before opening reference masks."""
    model = load_model()
    base = load_candidate()
    store = ReferenceStore("calibration")
    provenance, directory = _qwen_provenance()
    thresholds = [value / 10 for value in range(1, 10)]
    protocol = {
        "version": "v15b_direct_qwen_color_regions", "seed": 73,
        "region_segments": 48, "foreground_method": "unlabeled_unet_v1",
        "foreground_threshold": 0.5, "qwen_repository": provenance["repository"],
        "qwen_revision": provenance["revision"], "instruction": COLOR_REGION_INSTRUCTION,
        "views": ["original"], "thresholds": thresholds,
        "reference_masks_loaded_after_prediction_cache": True,
        "marked_images_used_for_fit": 0, "reference_pixels_used_for_fit": 0,
    }
    protocol_fingerprint = json_hash(protocol)
    run_id = f"calibration_color_region_qwen_{protocol_fingerprint[:16]}"
    run_dir = RUNS / run_id
    cache = run_dir / "prediction_cache"
    cache.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "protocol.json", protocol)
    report_path = run_dir / "report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())

    missing = [row for row in store.rows if not (cache / f"{row['id']}.npz").is_file()]
    teacher = ZeroShotRegionTeacher(directory) if missing else None
    try:
        for image_index, metadata in enumerate(store.rows, start=1):
            array_path = cache / f"{metadata['id']}.npz"
            record_path = cache / f"{metadata['id']}.json"
            if array_path.is_file() and record_path.is_file():
                print(f"v15b Qwen cache: {image_index}/{len(store.rows)} cached", flush=True)
                continue
            rgb = store.load_rgb(metadata)
            lung = unlabeled_lung_probability(rgb) >= protocol["foreground_threshold"]
            labels = color_regions(rgb, lung, protocol["region_segments"])
            regions = region_statistics(rgb, lung, labels)
            context = Image.fromarray(rgb)
            region_ids, lesion_probabilities, records = [], [], []
            for region_index, region in enumerate(regions, start=1):
                crop = make_region_crop(rgb, labels, region)
                prediction = teacher.classify(context, crop, False)
                region_ids.append(region["region_id"])
                lesion_probabilities.append(float(prediction["probabilities"][0]))
                records.append({"region": region, "teacher": prediction})
                if region_index % 10 == 0 or region_index == len(regions):
                    print(f"v15b Qwen image {image_index}/{len(store.rows)} region "
                          f"{region_index}/{len(regions)}", flush=True)
            probability = region_probability_map(labels, region_ids, lesion_probabilities)
            np.savez_compressed(array_path, lung=lung.astype(np.uint8),
                                labels=labels.astype(np.int32), probability=probability.astype(np.float32))
            atomic_json(record_path, {
                "id": metadata["id"], "identity": metadata["identity"],
                "image_sha256": metadata["sha256"]["image"],
                "protocol_fingerprint": protocol_fingerprint, "regions": records,
                "reference_planes_loaded": False,
            })
    finally:
        if teacher is not None:
            teacher.close()

    # Reference planes are opened only after every clean-RGB prediction is durable.
    cached = []
    for metadata in store.rows:
        array_path = cache / f"{metadata['id']}.npz"
        record_path = cache / f"{metadata['id']}.json"
        if not array_path.is_file() or not record_path.is_file():
            raise RuntimeError("Incomplete Qwen cache before reference scoring")
        record = json.loads(record_path.read_text())
        if record["protocol_fingerprint"] != protocol_fingerprint or record["reference_planes_loaded"]:
            raise ValueError("Invalid Qwen cache provenance")
        with np.load(array_path, allow_pickle=False) as values:
            cached.append({"payload": store.load(metadata),
                           "lung": values["lung"].astype(bool),
                           "probability": values["probability"].astype(np.float32)})

    results = []
    for threshold in thresholds:
        rows = [_measure(item["payload"], item["lung"],
                         (item["probability"] >= threshold) & item["lung"])
                for item in cached]
        summary = summarize(rows)
        quality = summary["pooled_lesion"]
        floor = min(quality[key] for key in ("precision", "recall", "specificity")
                    if quality[key] is not None)
        results.append({"lesion_probability_threshold": threshold, "summary": summary,
                        "per_image": rows,
                        "selection_key": [floor, quality["dice"],
                                          -summary["ratio_mae_percentage_points"]]})
    results.sort(key=lambda row: tuple(row["selection_key"]), reverse=True)
    winner = results[0]
    candidate = {
        "status": "frozen_after_calibration", "created_at": now(),
        "version": "v15b_direct_qwen_color_regions", "model_id": model["model_id"],
        "selection_role": "calibration_validation_only", "test_payload_opened": False,
        "selection_objective": "maximize minimum pooled precision/recall/specificity, then Dice",
        "foreground_method": protocol["foreground_method"],
        "foreground_alpha_threshold": protocol["foreground_threshold"],
        "lesion_postprocess": "direct_qwen_color_region_probability",
        "lesion_region_segments": protocol["region_segments"],
        "lesion_probability_threshold": winner["lesion_probability_threshold"],
        "qwen_repository": provenance["repository"], "qwen_revision": provenance["revision"],
        "qwen_protocol_fingerprint": protocol_fingerprint,
        "marked_images_used_for_fit": 0, "reference_pixels_used_for_fit": 0,
        "calibration_summary": winner["summary"],
    }
    candidate["candidate_fingerprint"] = json_hash(candidate)
    base_quality = base["calibration_summary"]["pooled_lesion"]
    new_quality = winner["summary"]["pooled_lesion"]
    base_floor = min(base_quality[key] for key in ("precision", "recall", "specificity"))
    new_floor = min(new_quality[key] for key in ("precision", "recall", "specificity"))
    run = {"status": "done", "run_id": run_id, "created_at": now(),
           "protocol": protocol, "protocol_fingerprint": protocol_fingerprint,
           "candidate_count": len(results), "base_candidate_fingerprint": base["candidate_fingerprint"],
           "winner": candidate, "results": results, "test_payload_opened": False,
           "prediction_cache_complete_before_reference_scoring": True,
           "replaces_current": bool(new_floor > base_floor and new_quality["dice"] > base_quality["dice"])}
    atomic_json(report_path, run)
    if run["replaces_current"]:
        folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
        atomic_json(folder / "candidate.json", candidate)
        atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                    {"path": str((folder / "candidate.json").relative_to(ARTIFACTS.parent.parent)),
                     "sha256": sha256(folder / "candidate.json"),
                     "candidate_fingerprint": candidate["candidate_fingerprint"]})
    return run


def calibrate_color_region_qwen_conditional() -> dict:
    """v15c: reinterpret immutable Qwen A/B/C logits without new inference."""
    model = load_model()
    base = load_candidate()
    source_fingerprint = "f506a91348b08187b99e782e42ed08389cd489510324db9bfef2100575b94379"
    source_run = RUNS / f"calibration_color_region_qwen_{source_fingerprint[:16]}"
    source_report = json.loads((source_run / "report.json").read_text())
    if source_report["protocol_fingerprint"] != source_fingerprint:
        raise ValueError("Unexpected v15b source protocol")
    store = ReferenceStore("calibration")
    cached = []
    for metadata in store.rows:
        array_path = source_run / "prediction_cache" / f"{metadata['id']}.npz"
        record_path = source_run / "prediction_cache" / f"{metadata['id']}.json"
        record = json.loads(record_path.read_text())
        if record["protocol_fingerprint"] != source_fingerprint or record["reference_planes_loaded"]:
            raise ValueError("Invalid v15b cached logits")
        with np.load(array_path, allow_pickle=False) as values:
            labels = values["labels"].astype(np.int32)
            lung = values["lung"].astype(bool)
        ids, conditional, decisive = [], [], []
        for item in record["regions"]:
            probabilities = item["teacher"]["probabilities"]
            mass = float(probabilities[0] + probabilities[1])
            ids.append(int(item["region"]["region_id"]))
            conditional.append(float(probabilities[0] / max(mass, 1e-8)))
            decisive.append(mass)
        cached.append({"payload": store.load(metadata), "lung": lung,
                       "conditional": region_probability_map(labels, ids, conditional),
                       "decisive": region_probability_map(labels, ids, decisive)})

    results = []
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.90):
        for minimum_mass in (0.0, 0.05, 0.10):
            rows = [_measure(item["payload"], item["lung"],
                             (item["conditional"] >= threshold)
                             & (item["decisive"] >= minimum_mass) & item["lung"])
                    for item in cached]
            summary = summarize(rows)
            quality = summary["pooled_lesion"]
            floor = min(quality[key] for key in ("precision", "recall", "specificity")
                        if quality[key] is not None)
            results.append({"conditional_ab_threshold": threshold,
                            "minimum_decisive_mass": minimum_mass,
                            "summary": summary, "per_image": rows,
                            "selection_key": [floor, quality["dice"],
                                              -summary["ratio_mae_percentage_points"]]})
    results.sort(key=lambda row: tuple(row["selection_key"]), reverse=True)
    winner = results[0]
    candidate = {
        "status": "frozen_after_calibration", "created_at": now(),
        "version": "v15c_qwen_conditional_ab_color_regions", "model_id": model["model_id"],
        "selection_role": "calibration_validation_only", "test_payload_opened": False,
        "selection_objective": "maximize minimum pooled precision/recall/specificity, then Dice",
        "foreground_method": "unlabeled_unet_v1", "foreground_alpha_threshold": 0.5,
        "lesion_postprocess": "qwen_conditional_ab_color_region_probability",
        "lesion_region_segments": 48,
        "conditional_ab_threshold": winner["conditional_ab_threshold"],
        "minimum_decisive_mass": winner["minimum_decisive_mass"],
        "source_qwen_protocol_fingerprint": source_fingerprint,
        "marked_images_used_for_fit": 0, "reference_pixels_used_for_fit": 0,
        "calibration_summary": winner["summary"],
    }
    candidate["candidate_fingerprint"] = json_hash(candidate)
    base_quality = base["calibration_summary"]["pooled_lesion"]
    new_quality = winner["summary"]["pooled_lesion"]
    base_floor = min(base_quality[key] for key in ("precision", "recall", "specificity"))
    new_floor = min(new_quality[key] for key in ("precision", "recall", "specificity"))
    run_id = f"calibration_color_region_qwen_conditional_{uuid.uuid4().hex[:16]}"
    run = {"status": "done", "run_id": run_id, "created_at": now(),
           "source_protocol_fingerprint": source_fingerprint,
           "candidate_count": len(results), "base_candidate_fingerprint": base["candidate_fingerprint"],
           "winner": candidate, "results": results, "test_payload_opened": False,
           "new_gpu_inference": False,
           "replaces_current": bool(new_floor > base_floor and new_quality["dice"] > base_quality["dice"])}
    atomic_json(RUNS / run_id / "report.json", run)
    if run["replaces_current"]:
        folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
        atomic_json(folder / "candidate.json", candidate)
        atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                    {"path": str((folder / "candidate.json").relative_to(ARTIFACTS.parent.parent)),
                     "sha256": sha256(folder / "candidate.json"),
                     "candidate_fingerprint": candidate["candidate_fingerprint"]})
    return run


def calibrate_color_region_student() -> dict:
    """v16: cache student region probabilities before opening reference masks."""
    model = load_model()
    base = load_candidate()
    artifact, checkpoint = load_student()
    dino_record, dino_directory = _dino_provenance()
    thresholds = [value / 10 for value in range(1, 10)]
    protocol = {"version": "v16_color_region_student_calibration",
                "student_artifact_fingerprint": artifact["artifact_fingerprint"],
                "student_checkpoint_sha256": artifact["checkpoint_sha256"],
                "dino_revision": dino_record["revision"],
                "maximum_side": checkpoint["protocol"]["maximum_side"],
                "region_segments": checkpoint["protocol"]["region_segments"],
                "foreground_threshold": 0.5, "thresholds": thresholds,
                "reference_masks_loaded_after_prediction_cache": True,
                "marked_images_used_for_fit": 0, "reference_pixels_used_for_fit": 0}
    fingerprint = json_hash(protocol)
    run_id = f"calibration_color_region_student_{fingerprint[:16]}"
    run_dir = RUNS / run_id; cache = run_dir / "prediction_cache"
    cache.mkdir(parents=True, exist_ok=True); atomic_json(run_dir / "protocol.json", protocol)
    report_path = run_dir / "report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    store = ReferenceStore("calibration")
    missing = [row for row in store.rows if not (cache / f"{row['id']}.npz").is_file()]
    encoder = network = None
    if missing:
        encoder = DinoCropEncoder(dino_directory)
        network = RegionMLP(checkpoint["inputs"]).to("cuda")
        network.load_state_dict(checkpoint["model"]); network.eval()
    try:
        for index, metadata in enumerate(store.rows, start=1):
            array_path = cache / f"{metadata['id']}.npz"
            record_path = cache / f"{metadata['id']}.json"
            if array_path.is_file() and record_path.is_file():
                print(f"v16 calibration cache: {index}/{len(store.rows)} cached", flush=True); continue
            rgb_native = store.load_rgb(metadata)
            rgb, scale = scale_rgb(rgb_native, protocol["maximum_side"])
            lung = unlabeled_lung_probability(rgb) >= protocol["foreground_threshold"]
            labels = color_regions(rgb, lung, protocol["region_segments"])
            region_ids, probabilities = student_region_probabilities(
                encoder, network, checkpoint, rgb, lung, labels)
            small_probability = region_probability_map(labels, region_ids, probabilities)
            native_probability = cv2.resize(small_probability, (rgb_native.shape[1], rgb_native.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
            native_lung = cv2.resize(lung.astype(np.uint8),
                                     (rgb_native.shape[1], rgb_native.shape[0]),
                                     interpolation=cv2.INTER_NEAREST) > 0
            np.savez_compressed(array_path, lung=native_lung.astype(np.uint8),
                                probability=native_probability.astype(np.float32))
            atomic_json(record_path, {"id": metadata["id"], "identity": metadata["identity"],
                        "image_sha256": metadata["sha256"]["image"], "scale": scale,
                        "regions": len(region_ids), "protocol_fingerprint": fingerprint,
                        "reference_planes_loaded": False})
            print(f"v16 calibration cache: {index}/{len(store.rows)} regions={len(region_ids)}", flush=True)
    finally:
        if encoder is not None:
            encoder.close()
        if network is not None:
            del network
            torch.cuda.empty_cache()

    cached = []
    for metadata in store.rows:
        record = json.loads((cache / f"{metadata['id']}.json").read_text())
        if record["protocol_fingerprint"] != fingerprint or record["reference_planes_loaded"]:
            raise ValueError("Invalid v16 calibration cache provenance")
        with np.load(cache / f"{metadata['id']}.npz", allow_pickle=False) as values:
            cached.append({"payload": store.load(metadata), "lung": values["lung"].astype(bool),
                           "probability": values["probability"].astype(np.float32)})
    results = []
    for threshold in thresholds:
        rows = [_measure(item["payload"], item["lung"],
                         (item["probability"] >= threshold) & item["lung"])
                for item in cached]
        summary = summarize(rows); quality = summary["pooled_lesion"]
        floor = min(quality[key] for key in ("precision", "recall", "specificity")
                    if quality[key] is not None)
        results.append({"student_probability_threshold": threshold, "summary": summary,
                        "per_image": rows,
                        "selection_key": [floor, quality["dice"],
                                          -summary["ratio_mae_percentage_points"]]})
    results.sort(key=lambda row: tuple(row["selection_key"]), reverse=True)
    winner = results[0]
    candidate_version = ("v17_clean_color_region_dino_student"
                         if artifact["version"].startswith("v17") else "v16_color_region_dino_student")
    candidate = {"status": "frozen_after_calibration", "created_at": now(),
                 "version": candidate_version, "model_id": model["model_id"],
                 "selection_role": "calibration_validation_only", "test_payload_opened": False,
                 "selection_objective": "maximize minimum pooled precision/recall/specificity, then Dice",
                 "foreground_method": "unlabeled_unet_v1", "foreground_alpha_threshold": 0.5,
                 "lesion_postprocess": "color_region_dino_student_probability",
                 "lesion_region_segments": protocol["region_segments"],
                 "student_probability_threshold": winner["student_probability_threshold"],
                 "student_artifact_fingerprint": artifact["artifact_fingerprint"],
                 "student_checkpoint_sha256": artifact["checkpoint_sha256"],
                 "student_crop_mode": checkpoint["protocol"].get("crop_mode", "dimmed"),
                 "dino_revision": dino_record["revision"],
                 "marked_images_used_for_fit": 0, "reference_pixels_used_for_fit": 0,
                 "calibration_summary": winner["summary"]}
    candidate["candidate_fingerprint"] = json_hash(candidate)
    base_quality = base["calibration_summary"]["pooled_lesion"]
    new_quality = winner["summary"]["pooled_lesion"]
    base_floor = min(base_quality[key] for key in ("precision", "recall", "specificity"))
    new_floor = min(new_quality[key] for key in ("precision", "recall", "specificity"))
    run = {"status": "done", "run_id": run_id, "created_at": now(), "protocol": protocol,
           "protocol_fingerprint": fingerprint, "candidate_count": len(results),
           "base_candidate_fingerprint": base["candidate_fingerprint"], "winner": candidate,
           "results": results, "test_payload_opened": False,
           "prediction_cache_complete_before_reference_scoring": True,
           "replaces_current": bool(new_floor > base_floor and new_quality["dice"] > base_quality["dice"])}
    atomic_json(report_path, run)
    if run["replaces_current"]:
        folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
        atomic_json(folder / "candidate.json", candidate)
        atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                    {"path": str((folder / "candidate.json").relative_to(ARTIFACTS.parent.parent)),
                     "sha256": sha256(folder / "candidate.json"),
                     "candidate_fingerprint": candidate["candidate_fingerprint"]})
    return run


def calibrate_color_region_student_fusion() -> dict:
    """v16b: fuse immutable student probabilities with the unchanged v11b mask."""
    model = load_model(); base = load_candidate()
    if base.get("version") != "v11b_coarse_regions_refined":
        raise ValueError("v16b requires current v11b")
    source_fingerprint = "e8383856d8c4571a2590d8a003955b1ff5cb14abbfdf5992829000fe39b34332"
    source_run = RUNS / f"calibration_color_region_student_{source_fingerprint[:16]}"
    source = json.loads((source_run / "report.json").read_text())
    if source["protocol_fingerprint"] != source_fingerprint:
        raise ValueError("Unexpected v16 probability source")
    store = ReferenceStore("calibration")
    prepared = _prepare_store(store, base["foreground_method"], base["foreground_alpha_threshold"])
    probable, _ = _ensemble_masks(
        prepared, LesionConfig(**base["ensemble_left"]["config"]),
        LesionConfig(**base["ensemble_right"]["config"]), base["ensemble_operation"])
    base_predictions = []
    for item, seed in zip(prepared, probable):
        labels = color_regions(item["payload"]["rgb"], item["predicted_lung"],
                               base["lesion_region_segments"])
        base_predictions.append(region_consensus_from_seed(
            labels, item["predicted_lung"], seed, base["lesion_region_support"]))
    probability_maps = []
    for metadata in store.rows:
        record = json.loads((source_run / "prediction_cache" / f"{metadata['id']}.json").read_text())
        if record["protocol_fingerprint"] != source_fingerprint or record["reference_planes_loaded"]:
            raise ValueError("Invalid v16 probability cache")
        with np.load(source_run / "prediction_cache" / f"{metadata['id']}.npz",
                     allow_pickle=False) as values:
            probability_maps.append(values["probability"].astype(np.float32))
    results = []
    for operation in ("intersection", "union"):
        for threshold in [value / 10 for value in range(1, 10)]:
            predictions = []
            for base_mask, probability, item in zip(base_predictions, probability_maps, prepared):
                student = (probability >= threshold) & item["predicted_lung"]
                predictions.append(base_mask & student if operation == "intersection" else base_mask | student)
            rows = [_measure(item["payload"], item["predicted_lung"], prediction)
                    for item, prediction in zip(prepared, predictions)]
            summary = summarize(rows); quality = summary["pooled_lesion"]
            floor = min(quality[key] for key in ("precision", "recall", "specificity")
                        if quality[key] is not None)
            results.append({"operation": operation, "student_probability_threshold": threshold,
                            "summary": summary, "per_image": rows,
                            "selection_key": [floor, quality["dice"],
                                              -summary["ratio_mae_percentage_points"]]})
    results.sort(key=lambda row: tuple(row["selection_key"]), reverse=True)
    winner = results[0]
    candidate = {**{key: value for key, value in base.items()
                    if key not in {"created_at", "version", "candidate_fingerprint",
                                   "calibration_summary", "lesion_postprocess"}},
                 "created_at": now(), "version": "v16b_color_region_student_fusion",
                 "lesion_postprocess": "v11b_boolean_student_probability",
                 "student_fusion_operation": winner["operation"],
                 "student_probability_threshold": winner["student_probability_threshold"],
                 "student_source_protocol_fingerprint": source_fingerprint,
                 "calibration_summary": winner["summary"]}
    candidate["candidate_fingerprint"] = json_hash(candidate)
    base_quality = base["calibration_summary"]["pooled_lesion"]
    new_quality = winner["summary"]["pooled_lesion"]
    base_floor = min(base_quality[key] for key in ("precision", "recall", "specificity"))
    new_floor = min(new_quality[key] for key in ("precision", "recall", "specificity"))
    run_id = f"calibration_color_region_student_fusion_{uuid.uuid4().hex[:16]}"
    run = {"status": "done", "run_id": run_id, "created_at": now(),
           "source_protocol_fingerprint": source_fingerprint, "candidate_count": len(results),
           "base_candidate_fingerprint": base["candidate_fingerprint"], "winner": candidate,
           "results": results, "test_payload_opened": False, "new_gpu_inference": False,
           "replaces_current": bool(new_floor > base_floor and new_quality["dice"] > base_quality["dice"])}
    atomic_json(RUNS / run_id / "report.json", run)
    if run["replaces_current"]:
        folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
        atomic_json(folder / "candidate.json", candidate)
        atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                    {"path": str((folder / "candidate.json").relative_to(ARTIFACTS.parent.parent)),
                     "sha256": sha256(folder / "candidate.json"),
                     "candidate_fingerprint": candidate["candidate_fingerprint"]})
    return run


def calibrate_color_patch_cnn() -> dict:
    """v18: cache raw-RGB patch CNN probabilities before opening references."""
    model = load_model(); base = load_candidate(); artifact, checkpoint = load_patch_cnn()
    thresholds = [value / 10 for value in range(1, 10)]
    protocol = {"version": "v18_color_patch_cnn_calibration",
                "artifact_fingerprint": artifact["artifact_fingerprint"],
                "checkpoint_sha256": artifact["checkpoint_sha256"],
                "maximum_side": checkpoint["protocol"]["maximum_side"],
                "region_segments": checkpoint["protocol"]["region_segments"],
                "crop_size": checkpoint["protocol"]["crop_size"],
                "foreground_threshold": 0.5, "thresholds": thresholds,
                "reference_masks_loaded_after_prediction_cache": True,
                "marked_images_used_for_fit": 0, "reference_pixels_used_for_fit": 0}
    fingerprint = json_hash(protocol); run_id = f"calibration_color_patch_cnn_{fingerprint[:16]}"
    run_dir = RUNS / run_id; cache = run_dir / "prediction_cache"
    cache.mkdir(parents=True, exist_ok=True); atomic_json(run_dir / "protocol.json", protocol)
    report_path = run_dir / "report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    store = ReferenceStore("calibration")
    missing = [row for row in store.rows if not (cache / f"{row['id']}.npz").is_file()]
    network = None
    if missing:
        if not torch.cuda.is_available():
            raise RuntimeError("v18 calibration requires CUDA")
        network = PatchCNN().to("cuda"); network.load_state_dict(checkpoint["model"]); network.eval()
    try:
        for index, metadata in enumerate(store.rows, start=1):
            array_path = cache / f"{metadata['id']}.npz"; record_path = cache / f"{metadata['id']}.json"
            if array_path.is_file() and record_path.is_file():
                print(f"v18 calibration cache: {index}/{len(store.rows)} cached", flush=True); continue
            rgb_native = store.load_rgb(metadata); rgb, scale = scale_rgb(rgb_native, protocol["maximum_side"])
            lung = unlabeled_lung_probability(rgb) >= protocol["foreground_threshold"]
            labels = color_regions(rgb, lung, protocol["region_segments"])
            ids, probabilities = predict_regions(network, rgb, lung, labels, protocol["crop_size"])
            small = region_probability_map(labels, ids, probabilities)
            native_probability = cv2.resize(small, (rgb_native.shape[1], rgb_native.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
            native_lung = cv2.resize(lung.astype(np.uint8),
                                     (rgb_native.shape[1], rgb_native.shape[0]),
                                     interpolation=cv2.INTER_NEAREST) > 0
            np.savez_compressed(array_path, lung=native_lung.astype(np.uint8),
                                probability=native_probability.astype(np.float32))
            atomic_json(record_path, {"id": metadata["id"], "identity": metadata["identity"],
                        "image_sha256": metadata["sha256"]["image"], "scale": scale,
                        "regions": len(ids), "protocol_fingerprint": fingerprint,
                        "reference_planes_loaded": False})
            print(f"v18 calibration cache: {index}/{len(store.rows)} regions={len(ids)}", flush=True)
    finally:
        if network is not None:
            del network; torch.cuda.empty_cache()
    cached = []
    for metadata in store.rows:
        record = json.loads((cache / f"{metadata['id']}.json").read_text())
        if record["protocol_fingerprint"] != fingerprint or record["reference_planes_loaded"]:
            raise ValueError("Invalid v18 calibration cache")
        with np.load(cache / f"{metadata['id']}.npz", allow_pickle=False) as values:
            cached.append({"payload": store.load(metadata), "lung": values["lung"].astype(bool),
                           "probability": values["probability"].astype(np.float32)})
    results = []
    for threshold in thresholds:
        rows = [_measure(item["payload"], item["lung"],
                         (item["probability"] >= threshold) & item["lung"]) for item in cached]
        summary = summarize(rows); quality = summary["pooled_lesion"]
        floor = min(quality[key] for key in ("precision", "recall", "specificity")
                    if quality[key] is not None)
        results.append({"patch_probability_threshold": threshold, "summary": summary,
                        "per_image": rows,
                        "selection_key": [floor, quality["dice"],
                                          -summary["ratio_mae_percentage_points"]]})
    results.sort(key=lambda row: tuple(row["selection_key"]), reverse=True); winner = results[0]
    candidate = {"status": "frozen_after_calibration", "created_at": now(),
                 "version": "v18_raw_rgb_color_region_patch_cnn", "model_id": model["model_id"],
                 "selection_role": "calibration_validation_only", "test_payload_opened": False,
                 "selection_objective": "maximize minimum pooled precision/recall/specificity, then Dice",
                 "foreground_method": "unlabeled_unet_v1", "foreground_alpha_threshold": 0.5,
                 "lesion_postprocess": "raw_rgb_color_region_patch_cnn_probability",
                 "lesion_region_segments": protocol["region_segments"],
                 "patch_probability_threshold": winner["patch_probability_threshold"],
                 "patch_cnn_artifact_fingerprint": artifact["artifact_fingerprint"],
                 "patch_cnn_checkpoint_sha256": artifact["checkpoint_sha256"],
                 "marked_images_used_for_fit": 0, "reference_pixels_used_for_fit": 0,
                 "calibration_summary": winner["summary"]}
    candidate["candidate_fingerprint"] = json_hash(candidate)
    base_quality = base["calibration_summary"]["pooled_lesion"]
    new_quality = winner["summary"]["pooled_lesion"]
    base_floor = min(base_quality[key] for key in ("precision", "recall", "specificity"))
    new_floor = min(new_quality[key] for key in ("precision", "recall", "specificity"))
    run = {"status": "done", "run_id": run_id, "created_at": now(), "protocol": protocol,
           "protocol_fingerprint": fingerprint, "candidate_count": len(results),
           "base_candidate_fingerprint": base["candidate_fingerprint"], "winner": candidate,
           "results": results, "test_payload_opened": False,
           "prediction_cache_complete_before_reference_scoring": True,
           "replaces_current": bool(new_floor > base_floor and new_quality["dice"] > base_quality["dice"])}
    atomic_json(report_path, run)
    if run["replaces_current"]:
        folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
        atomic_json(folder / "candidate.json", candidate)
        atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                    {"path": str((folder / "candidate.json").relative_to(ARTIFACTS.parent.parent)),
                     "sha256": sha256(folder / "candidate.json"),
                     "candidate_fingerprint": candidate["candidate_fingerprint"]})
    return run


def load_candidate() -> dict:
    pointer = json.loads((ARTIFACTS / "frozen_candidate/current.json").read_text())
    path = ARTIFACTS.parent.parent / pointer["path"]
    if sha256(path) != pointer["sha256"]:
        raise ValueError("Frozen candidate checksum changed")
    candidate = json.loads(path.read_text())
    expected = candidate.pop("candidate_fingerprint")
    actual = json_hash(candidate)
    candidate["candidate_fingerprint"] = expected
    if expected != actual:
        raise ValueError("Frozen candidate fingerprint changed")
    return candidate


def confirm() -> dict:
    candidate = load_candidate()
    store = ReferenceStore("confirmation")
    if candidate.get("lesion_postprocess") == "sam2_oof_v29":
        from .sam2_oof import evaluate_sam2_store
        summary, rows = evaluate_sam2_store(candidate, store)
    elif candidate.get("lesion_postprocess") == "oof_ensemble_v28":
        from .binary_crossval import evaluate_oof_ensemble_store
        summary, rows = evaluate_oof_ensemble_store(candidate, store)
    elif candidate.get("lesion_postprocess") in {"binary_crossval_v27", "binary_crossval_v30"}:
        from .binary_crossval import evaluate_binary_store
        summary, rows = evaluate_binary_store(candidate, store)
    elif candidate.get("lesion_postprocess") == "stacked_meta_v26":
        from .stacked_meta import evaluate_stacked_store
        summary, rows = evaluate_stacked_store(candidate, store)
    elif candidate.get("lesion_postprocess") == "crossval_ensemble_v25":
        from .crossval_segformer import evaluate_crossval_store
        summary, rows = evaluate_crossval_store(candidate, store)
    elif candidate.get("lesion_postprocess") == "dual_segformer_unlabeled_lung_v24":
        from .segformer_segmentation import evaluate_unlabeled_lung_ensemble_store
        summary, rows = evaluate_unlabeled_lung_ensemble_store(candidate, store)
    elif candidate.get("lesion_postprocess") == "dual_segformer_v23":
        from .segformer_segmentation import evaluate_dual_ensemble_store
        summary, rows = evaluate_dual_ensemble_store(candidate, store)
    elif candidate.get("lesion_postprocess") == "hybrid_unlabeled_lung_v21d":
        from .segformer_segmentation import evaluate_unlabeled_lung_store
        summary, rows = evaluate_unlabeled_lung_store(candidate, store)
    elif candidate.get("lesion_postprocess") in {"hybrid_v21b", "hybrid_union_v21c"}:
        from .segformer_segmentation import evaluate_hybrid_store
        summary, rows = evaluate_hybrid_store(candidate, store)
    elif candidate.get("lesion_postprocess") in {"segformer_v21", "segformer_v22"}:
        from .segformer_segmentation import evaluate_segformer_store
        summary, rows = evaluate_segformer_store(candidate, store)
    elif candidate.get("lesion_postprocess") in {"supervised_color_v20", "supervised_color_region_v20b"}:
        from .supervised_color import evaluate_color_store
        summary, rows = evaluate_color_store(candidate, store)
    elif candidate.get("lesion_postprocess") == "supervised_deeplab_v19":
        from .supervised_segmentation import evaluate_store
        summary, rows = evaluate_store(candidate, store)
    elif candidate.get("lesion_postprocess") == "coarse_region_consensus_eroded":
        prepared = _prepare_store(store, candidate["foreground_method"],
                                  candidate["foreground_alpha_threshold"])
        probable, _ = _ensemble_masks(
            prepared, LesionConfig(**candidate["ensemble_left"]["config"]),
            LesionConfig(**candidate["ensemble_right"]["config"]),
            candidate["ensemble_operation"])
        rows = []
        for item, seed in zip(prepared, probable):
            labels = color_regions(item["payload"]["rgb"], item["predicted_lung"],
                                   candidate["lesion_region_segments"])
            prediction = region_consensus_from_seed(labels, item["predicted_lung"], seed,
                                                    candidate["lesion_region_support"])
            prediction = erode_prediction(prediction, candidate["lesion_erosion_fraction"])
            rows.append(_measure(item["payload"], item["predicted_lung"], prediction))
        summary = summarize(rows)
    elif candidate.get("lesion_postprocess") == "sam2_embedding_region_consensus":
        prepared = _prepare_store(store, candidate["foreground_method"],
                                  candidate["foreground_alpha_threshold"])
        probable, _ = _ensemble_masks(
            prepared, LesionConfig(**candidate["ensemble_left"]["config"]),
            LesionConfig(**candidate["ensemble_right"]["config"]),
            candidate["ensemble_operation"])
        refiner = sam2_boundary_refiner()
        rows = []
        for item, seed in zip(prepared, probable):
            embedding = refiner.embedding(item["payload"]["rgb"])
            labels = embedding_tissue_regions(embedding, item["predicted_lung"],
                                              candidate["embedding_clusters"],
                                              candidate["embedding_spatial_weight"])
            prediction = region_consensus_from_seed(labels, item["predicted_lung"], seed,
                                                    candidate["lesion_region_support"])
            rows.append(_measure(item["payload"], item["predicted_lung"], prediction))
        summary = summarize(rows)
    elif candidate.get("lesion_postprocess") == "coarse_region_consensus":
        prepared = _prepare_store(store, candidate["foreground_method"],
                                  candidate["foreground_alpha_threshold"])
        probable, _ = _ensemble_masks(
            prepared, LesionConfig(**candidate["ensemble_left"]["config"]),
            LesionConfig(**candidate["ensemble_right"]["config"]),
            candidate["ensemble_operation"])
        rows = []
        for item, seed in zip(prepared, probable):
            labels = color_regions(item["payload"]["rgb"], item["predicted_lung"],
                                   candidate["lesion_region_segments"])
            prediction = region_consensus_from_seed(labels, item["predicted_lung"], seed,
                                                    candidate["lesion_region_support"])
            rows.append(_measure(item["payload"], item["predicted_lung"], prediction))
        summary = summarize(rows)
    elif candidate.get("lesion_postprocess") in {
            "sam2_boundary_from_unlabeled_ensemble", "seed_intersection_dilated_sam2"}:
        prepared = _prepare_store(store, candidate["foreground_method"],
                                  candidate["foreground_alpha_threshold"])
        probable, core = _ensemble_masks(
            prepared, LesionConfig(**candidate["ensemble_left"]["config"]),
            LesionConfig(**candidate["ensemble_right"]["config"]),
            candidate["ensemble_operation"])
        proposals = []
        refiner = sam2_boundary_refiner()
        for item, base_mask, core_mask in zip(prepared, probable, core):
            proposal, _, _ = refiner.refine(
                item["payload"]["rgb"], item["predicted_lung"], base_mask, core_mask,
                candidate["sam2_padding_fraction"], candidate["sam2_erosion_fraction"])
            proposals.append(proposal)
        if candidate["lesion_postprocess"] == "seed_intersection_dilated_sam2":
            rows = [_measure(item["payload"], item["predicted_lung"],
                             conservative_sam_fusion(base_mask, proposal,
                                                     candidate["sam2_dilation_fraction"]))
                    for item, base_mask, proposal in zip(prepared, probable, proposals)]
            summary = summarize(rows)
        else:
            summary, rows = _score_sam2_variant(
                prepared, proposals, probable, candidate["sam2_composition"])
    elif candidate.get("lesion_postprocess") == "boolean_ensemble":
        prepared = _prepare_store(store, candidate["foreground_method"],
                                  candidate["foreground_alpha_threshold"])
        summary, rows = _score_prepared_ensemble(
            prepared, LesionConfig(**candidate["ensemble_left"]["config"]),
            LesionConfig(**candidate["ensemble_right"]["config"]),
            candidate["ensemble_operation"])
    elif candidate.get("lesion_postprocess") == "grabcut_seeded_gmm":
        prepared = _prepare_store(store, candidate.get("foreground_method", "classical"),
                                  candidate.get("foreground_alpha_threshold", 128))
        summary, rows = _score_prepared_grabcut(
            prepared, LesionConfig(**candidate["config"]),
            candidate["lesion_core_threshold"], candidate["lesion_grabcut_iterations"])
    else:
        summary, rows = _candidate_result(store, LesionConfig(**candidate["config"]),
                                          candidate.get("foreground_method", "classical"),
                                          candidate.get("foreground_alpha_threshold", 128),
                                          candidate.get("lesion_region_support"),
                                          candidate.get("lesion_region_segments", 450))
    lesion = summary["pooled_lesion"]
    checks = {
        "lesion_dice_at_least_0_80": lesion["dice"] is not None and lesion["dice"] >= 0.80,
        "precision_at_least_0_80": lesion["precision"] is not None and lesion["precision"] >= 0.80,
        "recall_at_least_0_80": lesion["recall"] is not None and lesion["recall"] >= 0.80,
        "specificity_at_least_0_80": lesion["specificity"] is not None and lesion["specificity"] >= 0.80,
        "lung_dice_at_least_0_90": summary["pooled_lung"]["dice"] is not None and summary["pooled_lung"]["dice"] >= 0.90,
    }
    report = {"status": "done", "created_at": now(), "role": "confirmation_validation",
              "candidate": candidate, "summary": summary, "per_image": rows,
              "gate": {"passed": all(checks.values()), "checks": checks},
              "test_payload_opened": False}
    run_id = f"confirmation_{uuid.uuid4().hex[:16]}"
    report["run_id"] = run_id
    atomic_json(RUNS / run_id / "report.json", report)
    atomic_json(ARTIFACTS / "confirmation/current.json", report)
    if report["gate"]["passed"]:
        atomic_json(ARTIFACTS / "test_unlock.json",
                    {"created_at": now(), "confirmation_gate_passed": True,
                     "candidate_fingerprint": candidate["candidate_fingerprint"],
                     "confirmation_run_id": run_id})
    return report

