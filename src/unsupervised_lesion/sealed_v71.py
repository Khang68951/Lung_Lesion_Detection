from __future__ import annotations

import gc
import json
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from .binary_crossval import prepare_at_size, relative_lab_image
from .color_region.student import _dino_provenance
from .dense_dino_crossval import (DenseColourDecoder, _colour_channels as dino_colours,
                                  _letterbox_size, patch_grid)
from .dense_sam2_crossval import (SAM2ColourDecoder, _colour_channels as sam_colours,
                                  CHECKPOINT, CHECKPOINT_SHA256)
from .evaluate import _measure, load_candidate, summarize
from .final_ensemble_v71 import (LESION_THRESHOLD, MODEL_WEIGHTS, VERSION, refine_score)
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, now, sha256, within_project
from .lobe_crop_crossval_v62 import crop_geometry, restore_crop
from .lung_consensus_v54 import vote_consensus
from .lung_lobe_v53 import _predictor, lobe_prompt_variants
from .reference import ReferenceStore
from .segformer_segmentation import _image_tensor, _load as load_segformer
from .segmentation import unlabeled_lung_probability
from .style_ensemble_v46 import BASE_WEIGHTS, PARENTS, STYLE_VERSION, _root
from .supervised_segmentation import _prepare


def _fake_payload(row: dict, rgb: np.ndarray) -> dict:
    shape = rgb.shape[:2]
    zeros = np.zeros(shape, bool)
    return {"row": row, "rgb": rgb, "lung": zeros, "lesion": zeros,
            "valid": np.ones(shape, bool), "ignore_lung": zeros,
            "ignore_lesion": zeros}


def _test_rgb(candidate: dict) -> tuple[ReferenceStore, list[dict]]:
    store = ReferenceStore("test", candidate)
    payloads = [{"row": row, "rgb": store.load_rgb(row)} for row in store.rows]
    if any(payload["rgb"].shape[:2] != (512, 512) for payload in payloads):
        raise ValueError("V71 sealed inference expects frozen 512x512 canvases")
    return store, payloads


def _report(parent: str, version: str) -> tuple[dict, object]:
    matches = []
    for path in (ARTIFACTS / parent).glob("*/report.json"):
        value = json.loads(path.read_text())
        if value.get("version") == version:
            matches.append((value, path.parent))
    if len(matches) != 1:
        raise ValueError(f"Expected one completed {version} report, found {len(matches)}")
    return matches[0]


def _sam_cache(payloads: list[dict], root) -> None:
    missing = [p for p in payloads if not (root / "sam" / f"{p['row']['id']}.npz").is_file()]
    if not missing:
        return
    (root / "sam").mkdir(parents=True, exist_ok=True)
    predictor, model, _ = _predictor()
    try:
        for index, payload in enumerate(missing, 1):
            rgb = payload["rgb"]
            probability = unlabeled_lung_probability(rgb)
            variants = lobe_prompt_variants(probability)
            if len(variants) != 12:
                raise ValueError(f"Incomplete V53 prompt variants: {payload['row']['id']}")
            predictor.set_image(rgb)
            features = predictor._features
            embed64 = features["image_embed"][0].detach().cpu().numpy().astype(np.float16)
            high256 = features["high_res_feats"][0][0].detach().cpu().numpy().astype(np.float16)
            high128 = features["high_res_feats"][1][0].detach().cpu().numpy().astype(np.float16)
            sides = {"left": [], "right": []}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for variant in variants:
                    for prompt in variant["prompts"]:
                        predicted, _, _ = predictor.predict(
                            point_coords=prompt["points"], point_labels=prompt["point_labels"],
                            box=prompt["box"], multimask_output=True)
                        sides[prompt["side"]].append(np.asarray(predicted[2], bool))
            left = vote_consensus(sides["left"], 3)
            right = vote_consensus(sides["right"], 3)
            np.savez_compressed(root / "sam" / f"{payload['row']['id']}.npz",
                lung=(left | right).astype(np.uint8), left=left.astype(np.uint8),
                right=right.astype(np.uint8), embed64=embed64,
                high128=high128, high256=high256,
                source_image_sha256=np.asarray(payload["row"]["sha256"]["image"]))
            print(f"V71 sealed SAM/lung {index}/{len(missing)}", flush=True)
    finally:
        del predictor, model
        gc.collect(); torch.cuda.empty_cache()


def _binary_average(payloads: list[dict], report: dict, input_mode: str,
                    image_size: int, device: torch.device) -> list[np.ndarray]:
    items = [prepare_at_size(_fake_payload(p["row"], p["rgb"]), image_size)
             for p in payloads]
    if input_mode == "relative_lab":
        for item, payload in zip(items, payloads):
            item["rgb"] = relative_lab_image(
                item["rgb"], unlabeled_lung_probability(payload["rgb"]) >= .50)
    output = [np.zeros((512, 512), np.float32) for _ in items]
    for artifact in report["fold_artifacts"]:
        model = load_segformer({"checkpoint": artifact["checkpoint"],
                                "checkpoint_sha256": artifact["checkpoint_sha256"]}, device)
        model.eval()
        with torch.inference_mode():
            for index, item in enumerate(items):
                logits = model(pixel_values=_image_tensor(item["rgb"])[None].to(device)).logits
                value = F.interpolate(logits, size=item["rgb"].shape[:2], mode="bilinear",
                                      align_corners=False).sigmoid()[0, 0].float().cpu().numpy()
                from .supervised_segmentation import _restore
                output[index] += _restore(value, item["geometry"]) / len(report["fold_artifacts"])
        del model; gc.collect(); torch.cuda.empty_cache()
    return output


def _dino_average(payloads: list[dict], report: dict, device: torch.device) -> list[np.ndarray]:
    provenance, directory = _dino_provenance()
    from transformers import Dinov2WithRegistersModel
    encoder = Dinov2WithRegistersModel.from_pretrained(
        directory, local_files_only=True, use_safetensors=True,
        attn_implementation="sdpa").eval().to(device)
    encoder.requires_grad_(False)
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    features = []
    with torch.inference_mode():
        for payload in payloads:
            rgb448, _ = _letterbox_size(payload["rgb"], 448)
            value = torch.from_numpy(rgb448.transpose(2, 0, 1).copy())[None].float().to(device) / 255
            with torch.autocast("cuda"):
                encoded = encoder(pixel_values=(value - mean) / std,
                                  output_hidden_states=True, return_dict=True)
                hidden = torch.stack(encoded.hidden_states[-4:]).mean(0)
            features.append(patch_grid(hidden, int(encoder.config.num_register_tokens))[0]
                            .float().cpu().numpy())
    del encoder; gc.collect(); torch.cuda.empty_cache()
    items = [_prepare(_fake_payload(p["row"], p["rgb"])) for p in payloads]
    for item, feature in zip(items, features):
        item["feature"] = feature
        item["colour"] = dino_colours(item["rgb"])
    output = [np.zeros((512, 512), np.float32) for _ in items]
    from .dense_dino_crossval import _predict
    for artifact in report["fold_artifacts"]:
        path = within_project(artifact["checkpoint"])
        if sha256(path) != artifact["checkpoint_sha256"]:
            raise ValueError("V40 checkpoint changed")
        state = torch.load(path, map_location=device, weights_only=False)
        model = DenseColourDecoder().to(device)
        model.load_state_dict(state["model"])
        for index, value in enumerate(_predict(model, items, device)):
            output[index] += value / len(report["fold_artifacts"])
        del model; gc.collect(); torch.cuda.empty_cache()
    return output


def _sam_decoder_average(payloads: list[dict], report: dict, cache_root,
                         device: torch.device) -> list[np.ndarray]:
    items = [_prepare(_fake_payload(p["row"], p["rgb"])) for p in payloads]
    for item in items:
        with np.load(cache_root / "sam" / f"{item['id']}.npz", allow_pickle=False) as values:
            for key in ("embed64", "high128", "high256"):
                item[key] = values[key].astype(np.float32)
        item["colour"] = sam_colours(item["rgb"])
    output = [np.zeros((512, 512), np.float32) for _ in items]
    from .dense_sam2_crossval import _predict
    for artifact in report["fold_artifacts"]:
        path = within_project(artifact["checkpoint"])
        if sha256(path) != artifact["checkpoint_sha256"]:
            raise ValueError("V44 checkpoint changed")
        state = torch.load(path, map_location=device, weights_only=False)
        model = SAM2ColourDecoder().to(device)
        model.load_state_dict(state["model"])
        for index, value in enumerate(_predict(model, items, device)):
            output[index] += value / len(report["fold_artifacts"])
        del model; gc.collect(); torch.cuda.empty_cache()
    return output


def _new_b1(device: torch.device):
    return SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b1-finetuned-ade-512-512", num_labels=1,
        ignore_mismatched_sizes=True, use_safetensors=True,
        cache_dir=ARTIFACTS / "model_cache", id2label={0: "lesion"},
        label2id={"lesion": 0}).to(device)


def _tta(model, rgb: np.ndarray, device: torch.device) -> np.ndarray:
    values = []
    model.eval()
    with torch.inference_mode():
        for fy, fx in ((False, False), (False, True), (True, False), (True, True)):
            augmented = rgb
            if fy: augmented = np.flip(augmented, 0)
            if fx: augmented = np.flip(augmented, 1)
            augmented = np.ascontiguousarray(augmented)
            with torch.autocast("cuda"):
                logits = model(pixel_values=_image_tensor(augmented)[None].to(device)).logits
                value = F.interpolate(logits, size=augmented.shape[:2], mode="bilinear",
                                      align_corners=False).sigmoid()[0, 0].float().cpu().numpy()
            if fx: value = np.flip(value, 1)
            if fy: value = np.flip(value, 0)
            values.append(value.copy())
    return np.mean(values, axis=0).astype(np.float32)


def _multiview_average(payloads: list[dict], report: dict, cache_root,
                       device: torch.device) -> list[np.ndarray]:
    outputs = [np.zeros((512, 512), np.float32) for _ in payloads]
    supports = []
    views = []
    for payload in payloads:
        with np.load(cache_root / "sam" / f"{payload['row']['id']}.npz",
                     allow_pickle=False) as values:
            left = values["left"].astype(bool); right = values["right"].astype(bool)
        lung = left | right
        item_views = []
        for name, support in (("whole", lung), ("left", left), ("right", right)):
            geometry = crop_geometry(support, 512)
            sl = np.s_[geometry.y0:geometry.y1, geometry.x0:geometry.x1]
            crop = payload["rgb"][sl].copy()
            crop[~support[sl]] = 0
            crop = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
            item_views.append((name, crop, geometry, support))
        views.append(item_views); supports.append(lung)
    for artifact in report["fold_artifacts"]:
        path = within_project(artifact["checkpoint"])
        if sha256(path) != artifact["checkpoint_sha256"]:
            raise ValueError("Multiview checkpoint changed")
        state = torch.load(path, map_location=device, weights_only=False)
        model = _new_b1(device); model.load_state_dict(state["model"])
        for index, item_views in enumerate(views):
            total = np.zeros((512, 512), np.float32)
            count = np.zeros((512, 512), np.float32)
            for _, crop, geometry, support in item_views:
                restored = restore_crop(_tta(model, crop, device), geometry)
                total[support] += restored[support]; count[support] += 1
            outputs[index] += total / np.maximum(count, 1) / len(report["fold_artifacts"])
        del model; gc.collect(); torch.cuda.empty_cache()
    return outputs


def _v70_average(payloads: list[dict], report: dict,
                  device: torch.device) -> list[np.ndarray]:
    output = [np.zeros((512, 512), np.float32) for _ in payloads]
    for artifact in report["fold_artifacts"]:
        model = load_segformer({"checkpoint": artifact["checkpoint"],
                                "checkpoint_sha256": artifact["checkpoint_sha256"]}, device)
        for index, payload in enumerate(payloads):
            output[index] += _tta(model, payload["rgb"], device) / len(report["fold_artifacts"])
        del model; gc.collect(); torch.cuda.empty_cache()
    return output


def predict_sealed_v71() -> dict:
    candidate = load_candidate()
    if candidate.get("version") != VERSION or not candidate.get("oof_gate", {}).get("passed"):
        raise ValueError("Current frozen candidate is not passing V71")
    store, payloads = _test_rgb(candidate)
    root = ARTIFACTS / "sealed_v71" / candidate["candidate_fingerprint"]
    predictions = root / "predictions"; predictions.mkdir(parents=True, exist_ok=True)
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
    base = [BASE_WEIGHTS["v37_binary_segformer_b1_relative_lab_reference_v2"] * a +
            BASE_WEIGHTS["v40_frozen_dense_dinov2_colour_decoder_reference_v2"] * b +
            BASE_WEIGHTS["v43_binary_segformer_b1_native512_reference_v2"] * c +
            BASE_WEIGHTS["v44_frozen_dense_sam2_multiscale_colour_decoder_reference_v2"] * d
            for a, b, c, d in zip(v37, v40, v43, v44)]
    v46 = [(.75 * a + .25 * b).astype(np.float32) for a, b in zip(base, v45)]
    v64_report, _ = _report("multiview_crossval_v64", "v64_supervised_v54_multiview_group_oof")
    v67_report, _ = _report("style_multiview_v67", "v67_supervised_v54_multiview_unmarked_style_group_oof")
    v70_report, _ = _report("binary_crossval", "v70_binary_segformer_b2_native512_reference_v2")
    v64 = _multiview_average(payloads, v64_report, root, device)
    v67 = _multiview_average(payloads, v67_report, root, device)
    v70 = _v70_average(payloads, v70_report, device)
    hashes = {}
    for payload, a, b, c, d in zip(payloads, v46, v64, v67, v70):
        identifier = payload["row"]["id"]
        with np.load(root / "sam" / f"{identifier}.npz", allow_pickle=False) as values:
            lung = values["lung"].astype(bool)
        raw = (MODEL_WEIGHTS["v46"] * a + MODEL_WEIGHTS["v64_tta"] * b +
               MODEL_WEIGHTS["v67_tta"] * c + MODEL_WEIGHTS["v70_tta"] * d)
        score = refine_score(payload["rgb"], lung, raw)
        lesion = (score >= LESION_THRESHOLD) & lung
        path = predictions / f"{identifier}.npz"
        np.savez_compressed(path, lung=lung.astype(np.uint8), lesion=lesion.astype(np.uint8),
                            score=score, lesion_ratio=np.asarray(lesion.sum() / max(lung.sum(), 1)))
        hashes[identifier] = sha256(path)
    report = {"status": "predictions_frozen", "created_at": now(), "version": VERSION,
              "candidate_fingerprint": candidate["candidate_fingerprint"],
              "images": len(payloads), "prediction_sha256": hashes,
              "reference_planes_opened": 0, "test_rgb_opened": len(payloads)}
    atomic_json(report_path, report)
    return report


def evaluate_sealed_v71() -> dict:
    candidate = load_candidate(); root = ARTIFACTS / "sealed_v71" / candidate["candidate_fingerprint"]
    prediction_report = json.loads((root / "prediction_report.json").read_text())
    store = ReferenceStore("test", candidate); rows = []
    for row in store.rows:
        path = root / "predictions" / f"{row['id']}.npz"
        if sha256(path) != prediction_report["prediction_sha256"].get(row["id"]):
            raise ValueError(f"Sealed V71 prediction changed: {row['id']}")
        with np.load(path, allow_pickle=False) as values:
            lung = values["lung"].astype(bool); lesion = values["lesion"].astype(bool)
        rows.append(_measure(store.load(row), lung, lesion))
    summary = summarize(rows); metric = summary["pooled_lesion"]
    checks = {"lesion_dice_at_least_0_80": metric["dice"] >= .80,
              "precision_at_least_0_80": metric["precision"] >= .80,
              "recall_at_least_0_80": metric["recall"] >= .80,
              "specificity_at_least_0_80": metric["specificity"] >= .80,
              "lung_dice_at_least_0_90": summary["pooled_lung"]["dice"] >= .90}
    report = {"status": "done", "created_at": now(), "version": VERSION,
              "candidate_fingerprint": candidate["candidate_fingerprint"],
              "summary": summary, "test_gate": {"passed": all(checks.values()), "checks": checks},
              "prediction_report_sha256": sha256(root / "prediction_report.json"),
              "test_payload_opened": True, "per_image": rows}
    atomic_json(root / "evaluation_report.json", report)
    return report
