from __future__ import annotations

import gc
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from skimage.color import rgb2lab
from transformers import SegformerForSemanticSegmentation

from .color_region.student import DinoCropEncoder, _dino_provenance, select_training_rows
from .data import load_audit, validate_fit_rows
from .evaluate import _measure, summarize
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256, within_project
from .reference_v2 import DevelopmentReferenceStore
from .sam_lobe_v51 import (_generator, proposal_consensus, SAM_SHA256, V47_REPORT_SHA256,
                           V47_ROOT)
from .segformer_segmentation import _image_tensor, _rank
from .segmentation import unlabeled_lung_probability
from .unmarked_pseudo_v47 import letterbox_rgb


VERSION = "v52_unmarked_sam_compartment_dino_student"
SEED = 5204
MAXIMUM_IMAGES = 256
POSITIVE_THRESHOLD = .70
NEGATIVE_THRESHOLD = .20
V51_ROOT = ARTIFACTS / "sam_lobe_v51" / "360b8d649d3939c22c38fe17959c54f89edd66a4a1f83e8a12773bf854dd2e90"
V51_REPORT_SHA256 = "775198eb7bd49b1089d30fadc861c5ed47eb29275221151b8bc5dc2793714012"


def _proposal_crop(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("Cannot crop an empty V52 proposal")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad = max(6, round(max(x1 - x0, y1 - y0) * .20))
    x0, x1 = max(0, x0 - pad), min(rgb.shape[1], x1 + pad)
    y0, y1 = max(0, y0 - pad), min(rgb.shape[0], y1 + pad)
    crop = rgb[y0:y1, x0:x1].copy(); selected = mask[y0:y1, x0:x1]
    dimmed = np.clip(crop.astype(np.float32) * .15, 0, 255).astype(np.uint8)
    dimmed[selected] = crop[selected]
    return Image.fromarray(dimmed)


def _statistics(rgb: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    lab = rgb2lab(rgb); height, width = rgb.shape[:2]; rows = []
    for mask in masks:
        ys, xs = np.where(mask); values = lab[mask]
        rows.append([*values.mean(0), *values.std(0), np.log(max(mask.mean(), 1e-8)),
                     ((xs.min() + xs.max() + 1) / 2) / width,
                     ((ys.min() + ys.max() + 1) / 2) / height,
                     (xs.max() - xs.min() + 1) / width, (ys.max() - ys.min() + 1) / height])
    return np.asarray(rows, np.float32) if rows else np.empty((0, 11), np.float32)


def _unpack_masks(packed: np.ndarray, shape: tuple[int, int]) -> list[np.ndarray]:
    if len(packed) == 0:
        return []
    values = np.unpackbits(packed, axis=1)[:, :int(np.prod(shape))]
    return [row.reshape(shape).astype(bool) for row in values]


def _v47_model(device: torch.device):
    checkpoint_path = V47_ROOT / "epoch_5.pt"
    if sha256(checkpoint_path) != "e7bafc79a8a2b03f3552a9767480d46b6b72e43aabfc38547cd21192a125408f":
        raise ValueError("V47 final checkpoint changed")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b1-finetuned-ade-512-512", num_labels=1,
        ignore_mismatched_sizes=True, use_safetensors=True, cache_dir=ARTIFACTS / "model_cache",
        id2label={0: "lesion"}, label2id={"lesion": 0}).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"]); model.requires_grad_(False)
    return model


@torch.inference_mode()
def _v47_probability(model, rgb: np.ndarray) -> np.ndarray:
    logits = model(pixel_values=_image_tensor(rgb)[None].to("cuda")).logits
    return F.interpolate(logits, size=rgb.shape[:2], mode="bilinear",
                         align_corners=False).sigmoid()[0, 0].float().cpu().numpy()


def _student_probability(features: np.ndarray, mean: np.ndarray, scale: np.ndarray,
                         coefficient: np.ndarray, intercept: float) -> np.ndarray:
    logits = ((features - mean) / scale) @ coefficient + intercept
    return (1 / (1 + np.exp(-np.clip(logits, -30, 30)))).astype(np.float32)


def fit_sam_dino_student_v52() -> dict:
    if sha256(V47_ROOT / "report.json") != V47_REPORT_SHA256:
        raise ValueError("V47 report changed")
    if sha256(V51_ROOT / "report.json") != V51_REPORT_SHA256:
        raise ValueError("V51 report changed")
    audit = load_audit(); rows = select_training_rows(audit["fit_rows"], MAXIMUM_IMAGES, SEED)
    validate_fit_rows(rows, audit)
    dino_record, dino_directory = _dino_provenance()
    protocol = {"version": VERSION, "seed": SEED, "maximum_images": MAXIMUM_IMAGES,
                "image_size": 512, "sam_checkpoint_sha256": SAM_SHA256,
                "v47_report_sha256": V47_REPORT_SHA256,
                "v51_report_sha256": V51_REPORT_SHA256,
                "dino_revision": dino_record["revision"], "positive_threshold": POSITIVE_THRESHOLD,
                "negative_threshold": NEGATIVE_THRESHOLD, "uncertain_policy": "ignore",
                "proposal_protocol": "exact_v51", "crop_mode": "dimmed_proposal_with_20pct_context",
                "features": "frozen_dino_cls_plus_patch_mean_and_11_colour_shape_without_v47",
                "classifier": {"name": "LogisticRegression", "C": 1., "class_weight": "balanced",
                               "penalty": "l2", "max_iter": 1000},
                "fit_rows_fingerprint": json_hash([{"id": row["id"], "sha256": row["file_sha256"]}
                                                    for row in rows]),
                "student_weights": [0., .25, .50, .75, 1.],
                "lung_thresholds": [round(value, 2) for value in np.arange(.30, .901, .05)],
                "lesion_thresholds": [round(value, 3) for value in np.arange(.10, .901, .025)],
                "marked_fit_images": 0, "marked_fit_planes": 0, "sealed_test_images_opened": 0}
    fingerprint = json_hash(protocol); root = ARTIFACTS / "sam_dino_student_v52" / fingerprint
    train_cache = root / "unmarked_cache"; dev_cache = root / "development_cache"
    train_cache.mkdir(parents=True, exist_ok=True); dev_cache.mkdir(exist_ok=True)
    atomic_json(root / "protocol.json", protocol)
    report_path = root / "report.json"
    if report_path.is_file():
        return json.loads(report_path.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("V52 requires the authorized GPU")
    training_missing = [row for row in rows if not (train_cache / f"{row['id']}.npz").is_file()]
    encoder = generator = sam_model = v47_model = None
    try:
        if training_missing:
            encoder = DinoCropEncoder(dino_directory); generator, sam_model = _generator()
            v47_model = _v47_model(device)
        for index, row in enumerate(rows, start=1):
            path = train_cache / f"{row['id']}.npz"
            if path.is_file():
                if index % 25 == 0: print(f"{VERSION} unmarked {index}/{len(rows)} cached", flush=True)
                continue
            source = within_project(row["path"])
            if sha256(source) != row["file_sha256"]:
                raise ValueError(f"V52 source changed: {row['id']}")
            bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
            rgb = letterbox_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 512)
            lesion = _v47_probability(v47_model, rgb)
            lung_probability = unlabeled_lung_probability(rgb); lung = lung_probability >= .30
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                proposals = generator.generate(rgb)
            _, masks, metadata = proposal_consensus(proposals, lung, lesion)
            crops = [_proposal_crop(rgb, mask) for mask in masks]
            embeddings = encoder.encode(crops) if crops else np.empty((0, 768), np.float32)
            features = np.concatenate([embeddings, _statistics(rgb, masks)], axis=1)
            means = np.asarray([record["mean_v47_probability"] for record in metadata], np.float32)
            targets = np.full(len(means), -1, np.int8)
            targets[means <= NEGATIVE_THRESHOLD] = 0; targets[means >= POSITIVE_THRESHOLD] = 1
            np.savez_compressed(path, features=features.astype(np.float16), targets=targets,
                                teacher_means=means, source_sha256=np.asarray(row["file_sha256"]),
                                proposals=np.asarray(len(proposals)), retained=np.asarray(len(masks)))
            print(f"{VERSION} unmarked {index}/{len(rows)} retained={len(masks)} labeled={(targets >= 0).sum()}",
                  flush=True)
        arrays, cache_hashes = [], {}
        for row in rows:
            path = train_cache / f"{row['id']}.npz"; cache_hashes[row["id"]] = sha256(path)
            with np.load(path, allow_pickle=False) as values:
                selected = values["targets"] >= 0
                if selected.any(): arrays.append((values["features"][selected].astype(np.float32),
                                                  values["targets"][selected].astype(np.int8)))
        features = np.concatenate([item[0] for item in arrays]); targets = np.concatenate([item[1] for item in arrays])
        if set(np.unique(targets)) != {0, 1}:
            raise ValueError("V52 pseudo training lacks both classes")
        mean = features.mean(0); scale = np.maximum(features.std(0), 1e-5)
        classifier = LogisticRegression(C=1., class_weight="balanced", penalty="l2", max_iter=1000,
                                        random_state=SEED).fit((features - mean) / scale, targets)
        coefficient = classifier.coef_[0].astype(np.float32); intercept = float(classifier.intercept_[0])
        classifier_path = root / "classifier.npz"
        np.savez_compressed(classifier_path, mean=mean, scale=scale, coefficient=coefficient,
                            intercept=np.asarray(intercept), negative=np.asarray((targets == 0).sum()),
                            positive=np.asarray((targets == 1).sum()))
        # DINO may not have been needed for a complete training cache, but it is
        # required for any missing frozen development score.
        store = DevelopmentReferenceStore(); dev_rows = list(store.rows)
        if any(not (dev_cache / f"{row['id']}.npz").is_file() for row in dev_rows) and encoder is None:
            encoder = DinoCropEncoder(dino_directory)
        for index, row in enumerate(dev_rows, start=1):
            path = dev_cache / f"{row['id']}.npz"
            if path.is_file(): continue
            v51_path = V51_ROOT / f"{row['id']}.npz"
            with np.load(v51_path, allow_pickle=False) as values:
                packed = values["packed_masks"]; shape = tuple(values["mask_shape"])
                pixel = values["v47"].astype(np.float32); lung_probability = values["lung"].astype(np.float32)
            masks = _unpack_masks(packed, shape)
            image_path = within_project(row["files"]["image"])
            if sha256(image_path) != row["sha256"]["image"]:
                raise ValueError(f"V52 development RGB changed: {row['id']}")
            rgb = np.asarray(Image.open(image_path).convert("RGB"))
            crops = [_proposal_crop(rgb, mask) for mask in masks]
            embeddings = encoder.encode(crops) if crops else np.empty((0, 768), np.float32)
            proposal_features = np.concatenate([embeddings, _statistics(rgb, masks)], axis=1)
            probabilities = _student_probability(proposal_features, mean, scale, coefficient, intercept)
            student = pixel.copy(); covered = np.zeros(shape, bool)
            if masks:
                student[:] = 0
                for mask, probability in zip(masks, probabilities):
                    student[mask] = np.maximum(student[mask], probability); covered |= mask
                student[~covered] = pixel[~covered]
            student[lung_probability < .30] = 0
            np.savez_compressed(path, student=student, v47=pixel, lung=lung_probability,
                                proposal_probabilities=probabilities, source_v51_sha256=np.asarray(sha256(v51_path)))
            print(f"{VERSION} development {index}/{len(dev_rows)} proposals={len(masks)}", flush=True)
    finally:
        for value in (encoder, generator, sam_model, v47_model):
            if value is not None: del value
        gc.collect(); torch.cuda.empty_cache()
    # Open references only after the student classifier and all development
    # probability planes are immutable.
    store = DevelopmentReferenceStore(); dev_rows = list(store.rows); payloads = [store.load(row) for row in dev_rows]
    pixels, students, lungs, dev_hashes = [], [], [], {}
    for row in dev_rows:
        path = dev_cache / f"{row['id']}.npz"; dev_hashes[row["id"]] = sha256(path)
        with np.load(path, allow_pickle=False) as values:
            pixels.append(values["v47"]); students.append(values["student"]); lungs.append(values["lung"])
    candidates = []
    for weight in protocol["student_weights"]:
        scores = [(1 - weight) * pixel + weight * student for pixel, student in zip(pixels, students)]
        for lung_threshold in protocol["lung_thresholds"]:
            lung_masks = [value >= lung_threshold for value in lungs]
            for lesion_threshold in protocol["lesion_thresholds"]:
                summary = summarize([_measure(payload, lung, (score >= lesion_threshold) & lung)
                                     for payload, lung, score in zip(payloads, lung_masks, scores)])
                candidates.append({"student_weight": weight, "lung_threshold": lung_threshold,
                                   "lesion_threshold": lesion_threshold, "summary": summary})
    winner = max(candidates, key=lambda value: _rank(value["summary"])); lesion = winner["summary"]["pooled_lesion"]
    checks = {"lesion_dice_at_least_0_80": lesion["dice"] >= .80,
              "precision_at_least_0_80": lesion["precision"] >= .80,
              "recall_at_least_0_80": lesion["recall"] >= .80,
              "specificity_at_least_0_80": lesion["specificity"] >= .80,
              "lung_dice_at_least_0_90": winner["summary"]["pooled_lung"]["dice"] >= .90}
    report = {"status": "done", "created_at": now(), "version": VERSION,
              "protocol": protocol, "protocol_fingerprint": fingerprint,
              "training": {"examples": int(len(targets)), "positive": int((targets == 1).sum()),
                           "negative": int((targets == 0).sum()), "training_accuracy": float(classifier.score(
                               (features - mean) / scale, targets)), "cache_sha256": cache_hashes},
              "classifier": str(classifier_path.relative_to(PROJECT_ROOT)),
              "classifier_sha256": sha256(classifier_path), "development_cache_sha256": dev_hashes,
              "development_winner": winner,
              "development_gate": {"passed": all(checks.values()), "checks": checks},
              "candidate_count": len(candidates), "marked_fit_images": 0,
              "marked_fit_planes": 0, "test_payload_opened": False}
    atomic_json(report_path, report)
    return report
