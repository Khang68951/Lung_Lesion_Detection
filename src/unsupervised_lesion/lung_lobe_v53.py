from __future__ import annotations

import gc
import json
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from .evaluate import _measure, summarize
from .io import (ARTIFACTS, PROJECT_ROOT, SAM2_CHECKPOINT, SAM2_VENDOR,
                 atomic_json, json_hash, now, sha256, within_project)
from .reference_v2 import DevelopmentReferenceStore
from .sam_lobe_v51 import SAM_SHA256, V47_REPORT_SHA256, V47_ROOT


VERSION = "v53_sam2_anatomical_lobe_prompts"
TOP_FRACTIONS = (0.05, 0.15, 0.25, 0.35)
OVERLAPS = (0.02, 0.10, 0.18)


def _peaks(mask: np.ndarray, count: int = 3) -> np.ndarray:
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    points = []
    for _ in range(count):
        _, peak, _, point = cv2.minMaxLoc(distance)
        if peak <= 0:
            break
        points.append(point)
        cv2.circle(distance, point, max(6, round(min(mask.shape) * .06)), 0, -1)
    return np.asarray(points, np.float32)


def lobe_prompt_variants(lung_probability: np.ndarray) -> list[dict]:
    """Derive paired lobe prompts without annotations or image-specific tuning."""
    probability = np.asarray(lung_probability, np.float32)
    coarse = probability >= .85
    ys, xs = np.where(coarse)
    if not len(xs):
        return []
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    width, height = x1 - x0 + 1, y1 - y0 + 1
    middle = (x0 + x1) / 2
    variants = []
    for top_fraction in TOP_FRACTIONS:
        top = y0 + round(height * top_fraction)
        for overlap in OVERLAPS:
            margin = round(width * overlap)
            prompts = []
            for side in ("left", "right"):
                allowed = np.zeros_like(coarse)
                if side == "left":
                    boundary = min(coarse.shape[1] - 1, round(middle) + margin)
                    allowed[top:y1 + 1, x0:boundary + 1] = True
                else:
                    boundary = max(0, round(middle) - margin)
                    allowed[top:y1 + 1, boundary:x1 + 1] = True
                seed = coarse & allowed
                points = _peaks(seed)
                if not len(points):
                    break
                sy, sx = np.where(seed)
                pad = max(4, round(max(width, height) * .03))
                box = np.asarray([max(0, int(sx.min()) - pad), max(0, int(sy.min()) - pad),
                                  min(coarse.shape[1] - 1, int(sx.max()) + pad),
                                  min(coarse.shape[0] - 1, int(sy.max()) + pad)], np.float32)
                prompts.append({"side": side, "box": box, "points": points,
                                "point_labels": np.ones(len(points), np.int32)})
            if len(prompts) == 2:
                variants.append({"top_fraction": top_fraction, "overlap": overlap,
                                 "prompts": prompts})
    return variants


def _predictor():
    if not torch.cuda.is_available():
        raise RuntimeError("V53 SAM2 lobe prompts require the authorized GPU")
    if str(SAM2_VENDOR) not in sys.path:
        sys.path.insert(0, str(SAM2_VENDOR))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if sha256(SAM2_CHECKPOINT) != SAM_SHA256:
        raise ValueError("Pinned SAM2 checkpoint changed")
    model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", str(SAM2_CHECKPOINT),
                       device="cuda", apply_postprocessing=False)
    model.requires_grad_(False)
    return SAM2ImagePredictor(model), model, SAM2_CHECKPOINT


def calibrate_lung_lobe_v53() -> dict:
    if sha256(V47_ROOT / "report.json") != V47_REPORT_SHA256:
        raise ValueError("V47 report changed")
    protocol = {
        "version": VERSION, "sam_checkpoint_sha256": SAM_SHA256,
        "v47_report_sha256": V47_REPORT_SHA256,
        "coarse_prompt_threshold": .85, "top_fractions": list(TOP_FRACTIONS),
        "side_overlaps": list(OVERLAPS), "positive_points_per_lobe": 3,
        "sam_outputs_per_prompt": 3, "base_thresholds": [.30, .50, .70, .85, .95],
        "fusion_modes": ["sam", "intersect_base", "sam_plus_very_high_confidence"],
        "marked_fit_images": 0, "marked_fit_planes": 0,
        "sealed_test_images_opened": 0,
    }
    fingerprint = json_hash(protocol)
    root = ARTIFACTS / "lung_lobe_v53" / fingerprint
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "protocol.json", protocol)
    report_path = root / "report.json"
    if report_path.is_file():
        return json.loads(report_path.read_text())

    store = DevelopmentReferenceStore()
    rows = list(store.rows)
    missing = [row for row in rows if not (root / f"{row['id']}.npz").is_file()]
    predictor = model = checkpoint = None
    try:
        if missing:
            predictor, model, checkpoint = _predictor()
        for index, row in enumerate(rows, start=1):
            output = root / f"{row['id']}.npz"
            if output.is_file():
                print(f"{VERSION} cache {index}/{len(rows)}", flush=True)
                continue
            image_path = within_project(row["files"]["image"])
            if sha256(image_path) != row["sha256"]["image"]:
                raise ValueError(f"V53 clean RGB changed: {row['id']}")
            rgb = np.asarray(Image.open(image_path).convert("RGB"))
            v47_path = V47_ROOT / "development_predictions" / f"{row['id']}.npz"
            with np.load(v47_path, allow_pickle=False) as values:
                probability = values["lung"].astype(np.float32)
            variants = lobe_prompt_variants(probability)
            predictor.set_image(rgb)
            masks, scores, metadata = [], [], []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for variant_index, variant in enumerate(variants):
                    for prompt in variant["prompts"]:
                        predicted, quality, _ = predictor.predict(
                            point_coords=prompt["points"], point_labels=prompt["point_labels"],
                            box=prompt["box"], multimask_output=True)
                        for output_index, (mask, score) in enumerate(zip(predicted, quality)):
                            masks.append(np.asarray(mask, bool)); scores.append(float(score))
                            metadata.append({"variant": variant_index, "side": prompt["side"],
                                             "output": output_index,
                                             "top_fraction": variant["top_fraction"],
                                             "overlap": variant["overlap"]})
            packed = np.stack([np.packbits(mask.ravel()) for mask in masks])
            np.savez_compressed(output, lung_probability=probability, packed_masks=packed,
                                mask_shape=np.asarray(probability.shape, np.int32),
                                scores=np.asarray(scores, np.float32),
                                metadata_json=np.asarray(json.dumps(metadata)),
                                source_image_sha256=np.asarray(row["sha256"]["image"]),
                                source_v47_sha256=np.asarray(sha256(v47_path)))
            print(f"{VERSION} image {index}/{len(rows)} masks={len(masks)}", flush=True)
    finally:
        if predictor is not None:
            del predictor, model
            gc.collect(); torch.cuda.empty_cache()

    # Labels are opened only after all RGB-only SAM planes have been frozen.
    payloads = [store.load(row) for row in rows]
    frozen = []
    source_hashes = {}
    for row in rows:
        path = root / f"{row['id']}.npz"
        with np.load(path, allow_pickle=False) as values:
            shape = tuple(values["mask_shape"])
            raw = [np.unpackbits(value, count=shape[0] * shape[1]).reshape(shape).astype(bool)
                   for value in values["packed_masks"]]
            frozen.append((values["lung_probability"].copy(), raw,
                           json.loads(str(values["metadata_json"]))))
        source_hashes[row["id"]] = sha256(path)

    candidates = []
    for variant in range(len(TOP_FRACTIONS) * len(OVERLAPS)):
        for output_index in range(3):
            sam_masks = []
            for probability, raw, metadata in frozen:
                chosen = [mask for mask, record in zip(raw, metadata)
                          if record["variant"] == variant and record["output"] == output_index]
                sam_masks.append(np.logical_or.reduce(chosen) if chosen else np.zeros_like(probability, bool))
            for base_threshold in protocol["base_thresholds"]:
                for fusion in protocol["fusion_modes"]:
                    lungs = []
                    for sam, (probability, _, _) in zip(sam_masks, frozen):
                        if fusion == "sam":
                            lung = sam
                        elif fusion == "intersect_base":
                            lung = sam & (probability >= base_threshold)
                        else:
                            lung = sam | (probability >= .995)
                        lungs.append(lung)
                    summary = summarize([_measure(payload, lung, np.zeros_like(lung))
                                         for payload, lung in zip(payloads, lungs)])
                    candidates.append({"variant": variant, "output_index": output_index,
                                       "base_threshold": base_threshold, "fusion": fusion,
                                       "summary": summary})
    winner = max(candidates, key=lambda value: (
        value["summary"]["pooled_lung"]["dice"],
        min(value["summary"]["pooled_lung"][key]
            for key in ("precision", "recall", "specificity"))))
    lung = winner["summary"]["pooled_lung"]
    checks = {"lung_dice_at_least_0_90": lung["dice"] >= .90,
              "lung_dice_at_least_0_95": lung["dice"] >= .95,
              "lung_precision_at_least_0_90": lung["precision"] >= .90,
              "lung_recall_at_least_0_90": lung["recall"] >= .90,
              "lung_specificity_at_least_0_90": lung["specificity"] >= .90}
    report = {"status": "done", "created_at": now(), "version": VERSION,
              "protocol": protocol, "protocol_fingerprint": fingerprint,
              "sam_checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)) if checkpoint else
                                str(SAM2_CHECKPOINT.relative_to(PROJECT_ROOT)),
              "source_files_sha256": source_hashes, "development_winner": winner,
              "development_gate": {"passed_minimum": checks["lung_dice_at_least_0_90"],
                                   "passed_stretch": all(checks.values()), "checks": checks},
              "candidate_count": len(candidates), "model_weights_updated": False,
              "marked_fit_images": 0, "marked_fit_planes": 0,
              "test_payload_opened": False}
    atomic_json(report_path, report)
    return report
