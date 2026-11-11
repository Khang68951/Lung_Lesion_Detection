from __future__ import annotations

import gc
import json
import sys

import numpy as np
import torch
from PIL import Image

from .evaluate import _measure, summarize
from .io import (ARTIFACTS, PROJECT_ROOT, SAM2_CHECKPOINT, SAM2_VENDOR,
                 atomic_json, json_hash, now, sha256, within_project)
from .reference_v2 import DevelopmentReferenceStore
from .segformer_segmentation import _rank


VERSION = "v51_sam2_automatic_lobe_consensus"
SAM_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
V47_ROOT = ARTIFACTS / "unmarked_pseudo_segformer" / "87d285122fe6009a0a193c45e3211ed9969feafaab02ca1af8a5d612556689f4"
V47_REPORT_SHA256 = "1b6a1a414294c6bda31a7459629ccda183dfdb97e4c7fc2981ee5a7f3461aaa3"


def proposal_consensus(proposals: list[dict], lung: np.ndarray,
                       lesion_probability: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[dict]]:
    lung = np.asarray(lung, bool); lesion_probability = np.asarray(lesion_probability, np.float32)
    if lung.shape != lesion_probability.shape:
        raise ValueError("V51 lung/probability geometry differs")
    lung_pixels = max(1, int(lung.sum())); retained, metadata = [], []
    for proposal in proposals:
        mask = np.asarray(proposal["segmentation"], bool)
        if mask.shape != lung.shape or not mask.any():
            continue
        clipped = mask & lung; clipped_pixels = int(clipped.sum())
        containment = clipped_pixels / int(mask.sum()); area_fraction = clipped_pixels / lung_pixels
        if containment < .80 or not .015 <= area_fraction <= .55:
            continue
        mean_probability = float(lesion_probability[clipped].mean())
        retained.append(clipped)
        metadata.append({"predicted_iou": float(proposal["predicted_iou"]),
                         "stability_score": float(proposal["stability_score"]),
                         "containment": float(containment), "area_fraction": float(area_fraction),
                         "mean_v47_probability": mean_probability})
    region = lesion_probability.copy(); covered = np.zeros(lung.shape, bool)
    if retained:
        region[lung] = 0
        for mask, record in zip(retained, metadata):
            region[mask] = np.maximum(region[mask], record["mean_v47_probability"])
            covered |= mask
        region[lung & ~covered] = lesion_probability[lung & ~covered]
    region[~lung] = 0
    return region, retained, metadata


def _generator():
    if not torch.cuda.is_available():
        raise RuntimeError("V51 SAM2 automatic masks require the authorized GPU")
    if str(SAM2_VENDOR) not in sys.path:
        sys.path.insert(0, str(SAM2_VENDOR))
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    if sha256(SAM2_CHECKPOINT) != SAM_SHA256:
        raise ValueError("Pinned SAM2 checkpoint changed")
    model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", str(SAM2_CHECKPOINT),
                       device="cuda", apply_postprocessing=False)
    return SAM2AutomaticMaskGenerator(
        model, points_per_side=16, points_per_batch=64, pred_iou_thresh=.75,
        stability_score_thresh=.85, crop_n_layers=0, min_mask_region_area=0,
        output_mode="binary_mask", use_m2m=False, multimask_output=True), model


def calibrate_sam_lobe_v51() -> dict:
    if sha256(V47_ROOT / "report.json") != V47_REPORT_SHA256:
        raise ValueError("V47 report changed")
    protocol = {"version": VERSION, "sam_checkpoint_sha256": SAM_SHA256,
                "v47_report_sha256": V47_REPORT_SHA256, "points_per_side": 16,
                "points_per_batch": 64, "pred_iou_threshold": .75,
                "stability_threshold": .85, "crop_layers": 0,
                "minimum_lung_containment": .80, "minimum_lung_area_fraction": .015,
                "maximum_lung_area_fraction": .55, "proposal_score": "mean_v47_probability",
                "overlap_reducer": "maximum", "uncovered_policy": "retain_v47_probability",
                "region_weights": [0., .25, .50, .75, 1.],
                "lung_thresholds": [round(value, 2) for value in np.arange(.30, .901, .05)],
                "lesion_thresholds": [round(value, 3) for value in np.arange(.10, .901, .025)],
                "model_weights_updated": False, "marked_fit_images": 0,
                "marked_fit_planes": 0, "sealed_test_images_opened": 0}
    fingerprint = json_hash(protocol); root = ARTIFACTS / "sam_lobe_v51" / fingerprint
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / "protocol.json", protocol)
    report_path = root / "report.json"
    if report_path.is_file():
        return json.loads(report_path.read_text())
    store = DevelopmentReferenceStore(); rows = list(store.rows)
    missing = [row for row in rows if not (root / f"{row['id']}.npz").is_file()]
    generator = model = None
    try:
        if missing:
            generator, model = _generator()
        for index, row in enumerate(rows, start=1):
            path = root / f"{row['id']}.npz"
            if path.is_file():
                print(f"{VERSION} cache {index}/{len(rows)}", flush=True); continue
            image_path = within_project(row["files"]["image"])
            if sha256(image_path) != row["sha256"]["image"]:
                raise ValueError(f"V51 clean RGB changed: {row['id']}")
            rgb = np.asarray(Image.open(image_path).convert("RGB"))
            v47_path = V47_ROOT / "development_predictions" / f"{row['id']}.npz"
            with np.load(v47_path, allow_pickle=False) as values:
                lesion = values["lesion"].astype(np.float32); lung_probability = values["lung"].astype(np.float32)
            lung = lung_probability >= .30
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                proposals = generator.generate(rgb)
            region, retained, metadata = proposal_consensus(proposals, lung, lesion)
            if retained:
                packed = np.stack([np.packbits(mask.ravel()) for mask in retained])
            else:
                packed = np.empty((0, (lung.size + 7) // 8), np.uint8)
            np.savez_compressed(path, region=region, v47=lesion, lung=lung_probability,
                                packed_masks=packed, mask_shape=np.asarray(lung.shape, np.int32),
                                metadata_json=np.asarray(json.dumps(metadata)),
                                source_image_sha256=np.asarray(row["sha256"]["image"]),
                                source_v47_sha256=np.asarray(sha256(v47_path)))
            print(f"{VERSION} image {index}/{len(rows)} proposals={len(proposals)} retained={len(retained)}",
                  flush=True)
    finally:
        if generator is not None:
            del generator, model
            gc.collect(); torch.cuda.empty_cache()
    # Reference masks are first loaded after every RGB-only SAM/V47 plane is frozen.
    payloads = [store.load(row) for row in rows]
    v47_values, region_values, lungs, source_hashes = [], [], [], {}
    for row in rows:
        path = root / f"{row['id']}.npz"
        with np.load(path, allow_pickle=False) as values:
            v47_values.append(values["v47"]); region_values.append(values["region"]); lungs.append(values["lung"])
        source_hashes[row["id"]] = sha256(path)
    candidates = []
    for weight in protocol["region_weights"]:
        scores = [(1 - weight) * pixel + weight * region
                  for pixel, region in zip(v47_values, region_values)]
        for lung_threshold in protocol["lung_thresholds"]:
            lung_masks = [probability >= lung_threshold for probability in lungs]
            for lesion_threshold in protocol["lesion_thresholds"]:
                measured = [_measure(payload, lung, (score >= lesion_threshold) & lung)
                            for payload, lung, score in zip(payloads, lung_masks, scores)]
                candidates.append({"region_weight": weight, "lung_threshold": lung_threshold,
                                   "lesion_threshold": lesion_threshold,
                                   "summary": summarize(measured)})
    winner = max(candidates, key=lambda value: _rank(value["summary"]))
    lesion = winner["summary"]["pooled_lesion"]
    checks = {"lesion_dice_at_least_0_80": lesion["dice"] >= .80,
              "precision_at_least_0_80": lesion["precision"] >= .80,
              "recall_at_least_0_80": lesion["recall"] >= .80,
              "specificity_at_least_0_80": lesion["specificity"] >= .80,
              "lung_dice_at_least_0_90": winner["summary"]["pooled_lung"]["dice"] >= .90}
    report = {"status": "done", "created_at": now(), "version": VERSION,
              "protocol": protocol, "protocol_fingerprint": fingerprint,
              "sam_checkpoint": str(SAM2_CHECKPOINT.relative_to(PROJECT_ROOT)),
              "source_files_sha256": source_hashes, "development_winner": winner,
              "development_gate": {"passed": all(checks.values()), "checks": checks},
              "candidate_count": len(candidates), "model_weights_updated": False,
              "marked_fit_images": 0, "marked_fit_planes": 0, "test_payload_opened": False}
    atomic_json(report_path, report)
    return report
