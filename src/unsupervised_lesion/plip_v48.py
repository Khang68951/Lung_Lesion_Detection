from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import SiglipModel, SiglipProcessor

from .evaluate import _measure, summarize
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256, within_project
from .reference_v2 import DevelopmentReferenceStore
from .segmentation import unlabeled_lung_probability
from .segformer_segmentation import _rank


VERSION = "v48_frozen_medsiglip_dense_pilot"
REPOSITORY = "fokan/MedSigLIP"
UPSTREAM_REPOSITORY = "google/medsiglip-448"
REVISION = "9cea28a1a1195f665105faa6e8544c112fd960a4"
EXPECTED_MODEL_SHA256 = "3e1722b8100cca8b87cfe16e42140737aa6d312854e9e6bc24dff6db70e21d72"
PATCH_WIDTHS = (96, 144)
STRIDE_FRACTION = 0.5
PILOT_IMAGES = 5
PROMPTS = {
    "lesion": (
        "a gross pathology photograph of abnormal diseased pig lung tissue",
        "dark purple consolidated hemorrhagic or necrotic lung lesion",
        "pneumonia lesion in gross porcine lung tissue",
    ),
    "normal": (
        "a gross pathology photograph of healthy normal pink pig lung tissue",
        "normal aerated pink porcine lung parenchyma",
        "healthy unaffected lung lobe tissue",
    ),
    "artifact": (
        "blood vessel airway cut edge or non lung artifact in a specimen photograph",
        "background cloth glove label or surgical artifact",
    ),
}


def _starts(length: int, width: int, stride: int) -> list[int]:
    if length <= width:
        return [0]
    values = list(range(0, length - width + 1, stride))
    if values[-1] != length - width:
        values.append(length - width)
    return values


def _model_files(directory: Path) -> dict[str, str]:
    names = ("config.json", "preprocessor_config.json", "model.safetensors",
             "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json")
    result = {}
    for name in names:
        path = directory / name
        if path.is_file():
            result[name] = sha256(path)
    if "config.json" not in result or "model.safetensors" not in result:
        raise FileNotFoundError("Incomplete pinned MedSigLIP snapshot")
    if result["model.safetensors"] != EXPECTED_MODEL_SHA256:
        raise ValueError("MedSigLIP mirror differs from official upstream LFS hash")
    return result


class FrozenMedSiglip:
    def __init__(self, directory: Path):
        if not torch.cuda.is_available():
            raise RuntimeError("V48 MedSigLIP pilot requires the authorized GPU")
        self.processor = SiglipProcessor.from_pretrained(directory, local_files_only=True)
        self.model = SiglipModel.from_pretrained(
            directory, local_files_only=True, torch_dtype=torch.float16).eval().to("cuda")
        self.model.requires_grad_(False)
        flat = [prompt for label in ("lesion", "normal", "artifact") for prompt in PROMPTS[label]]
        encoded = self.processor(text=flat, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            features = self.model.get_text_features(**encoded)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            features = torch.nn.functional.normalize(features.float(), dim=1)
        offsets = np.cumsum([0] + [len(PROMPTS[label]) for label in ("lesion", "normal", "artifact")])
        classes = []
        for start, stop in zip(offsets[:-1], offsets[1:]):
            classes.append(torch.nn.functional.normalize(features[start:stop].mean(0), dim=0))
        self.text_features = torch.stack(classes)
        self.logit_scale = float(self.model.logit_scale.exp().detach().clamp(max=100))

    @torch.inference_mode()
    def patch_probabilities(self, patches: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        output = []
        for start in range(0, len(patches), batch_size):
            inputs = self.processor(images=patches[start:start + batch_size],
                                    return_tensors="pt").to("cuda")
            with torch.autocast("cuda"):
                features = self.model.get_image_features(**inputs)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            features = torch.nn.functional.normalize(features.float(), dim=1)
            probability = (self.logit_scale * features @ self.text_features.T).softmax(1)
            output.append(probability[:, 0].cpu().numpy())
        return np.concatenate(output).astype(np.float32)

    def close(self) -> None:
        del self.model, self.processor, self.text_features
        gc.collect(); torch.cuda.empty_cache()


def dense_score(rgb: np.ndarray, encoder: FrozenMedSiglip, width: int) -> np.ndarray:
    height, image_width = rgb.shape[:2]
    stride = max(1, round(width * STRIDE_FRACTION))
    windows = [(x, y, min(width, image_width), min(width, height))
               for y in _starts(height, min(width, height), stride)
               for x in _starts(image_width, min(width, image_width), stride)]
    patches = [Image.fromarray(rgb[y:y + h, x:x + w]) for x, y, w, h in windows]
    probabilities = encoder.patch_probabilities(patches)
    numerator = np.zeros((height, image_width), np.float32)
    denominator = np.zeros((height, image_width), np.float32)
    kernels: dict[tuple[int, int], np.ndarray] = {}
    for (x, y, w, h), probability in zip(windows, probabilities):
        key = (h, w)
        if key not in kernels:
            wy = np.hanning(h) if h > 2 else np.ones(h)
            wx = np.hanning(w) if w > 2 else np.ones(w)
            kernels[key] = np.maximum(np.outer(wy, wx), .05).astype(np.float32)
        weight = kernels[key]
        numerator[y:y + h, x:x + w] += float(probability) * weight
        denominator[y:y + h, x:x + w] += weight
    return numerator / np.maximum(denominator, 1e-6)


def _clean_rgb(row: dict) -> np.ndarray:
    path = within_project(row["files"]["image"])
    if sha256(path) != row["sha256"]["image"]:
        raise ValueError(f"V48 clean RGB changed: {row['id']}")
    return np.asarray(Image.open(path).convert("RGB"))


def _load_or_encode_clean(rows: list[dict], root: Path, directory: Path) -> dict[int, list[np.ndarray]]:
    scores: dict[int, list[np.ndarray]] = {width: [] for width in PATCH_WIDTHS}
    encoder = None
    try:
        for index, row in enumerate(rows, start=1):
            rgb = None
            for width in PATCH_WIDTHS:
                path = root / f"{row['id']}_w{width}.npz"
                if path.is_file():
                    with np.load(path, allow_pickle=False) as values:
                        score = values["score"]
                else:
                    if encoder is None:
                        encoder = FrozenMedSiglip(directory)
                    if rgb is None:
                        rgb = _clean_rgb(row)
                    score = dense_score(rgb, encoder, width)
                    np.savez_compressed(path, score=score)
                scores[width].append(score)
            print(f"{VERSION} image {index}/{len(rows)}", flush=True)
    finally:
        if encoder is not None:
            encoder.close()
    return scores


def run_medsiglip_v48_pilot() -> dict:
    protocol = {
        "version": VERSION, "repository": REPOSITORY,
        "upstream_repository": UPSTREAM_REPOSITORY, "revision": REVISION,
        "expected_upstream_model_sha256": EXPECTED_MODEL_SHA256,
        "patch_widths": list(PATCH_WIDTHS), "stride_fraction": STRIDE_FRACTION,
        "pilot_images": PILOT_IMAGES, "prompts": PROMPTS,
        "weights_updated": False, "project_marked_fit_images": 0,
        "project_marked_fit_planes": 0, "sealed_test_images_opened": 0,
        "continuation_auc_floor": .60,
    }
    fingerprint = json_hash(protocol); root = ARTIFACTS / "medsiglip_v48" / fingerprint
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / "protocol.json", protocol)
    report_path = root / "pilot.json"
    if report_path.is_file():
        return __import__("json").loads(report_path.read_text())
    cache = ARTIFACTS / "huggingface"
    directory = Path(snapshot_download(repo_id=REPOSITORY, revision=REVISION,
                                       cache_dir=cache, local_files_only=False))
    directory.resolve().relative_to(PROJECT_ROOT.resolve())
    files = _model_files(directory)
    store = DevelopmentReferenceStore(); rows = store.rows[:PILOT_IMAGES]
    # Freeze every RGB-only score before opening any reference plane.
    scores = _load_or_encode_clean(rows, root, directory)
    payloads = [store.load(row) for row in rows]
    targets, valid_masks = [], []
    for payload in payloads:
        valid = payload["lung"] & payload["valid"] & ~payload["ignore_lesion"]
        valid_masks.append(valid); targets.append(payload["lesion"][valid])
    pooled_target = np.concatenate(targets)
    auc_by_width = {}
    for width in PATCH_WIDTHS:
        values = np.concatenate([score[valid] for score, valid in zip(scores[width], valid_masks)])
        auc_by_width[str(width)] = float(roc_auc_score(pooled_target, values))
    ensemble = [np.mean([scores[width][index] for width in PATCH_WIDTHS], axis=0)
                for index in range(len(payloads))]
    pooled_ensemble = np.concatenate([score[valid] for score, valid in zip(ensemble, valid_masks)])
    ensemble_auc = float(roc_auc_score(pooled_target, pooled_ensemble))
    per_image = []
    for payload, target, valid, score in zip(payloads, targets, valid_masks, ensemble):
        per_image.append({"id": payload["row"]["id"],
                          "auc": float(roc_auc_score(target, score[valid]))
                          if np.unique(target).size == 2 else None})
    report = {
        "status": "done", "created_at": now(), "version": VERSION,
        "protocol": protocol, "protocol_fingerprint": fingerprint,
        "model_files_sha256": files, "pilot_ids": [row["id"] for row in rows],
        "auc_by_width": auc_by_width, "ensemble_auc": ensemble_auc,
        "per_image": per_image, "continue_to_full_validation": ensemble_auc > .60,
        "reference_use": "validation_auc_only_after_frozen_inference",
        "project_marked_fit_images": 0, "project_marked_fit_planes": 0,
        "test_payload_opened": False,
    }
    atomic_json(report_path, report)
    return report


def run_medsiglip_v48_full() -> dict:
    pilot = run_medsiglip_v48_pilot()
    if not pilot["continue_to_full_validation"]:
        raise RuntimeError("V48 pilot AUC did not pass the predeclared continuation floor")
    protocol = dict(pilot["protocol"])
    protocol.update({
        "stage": "full_development_validation", "development_images": 31,
        "lung_thresholds": [round(value, 2) for value in np.arange(.30, .901, .05)],
        "lesion_thresholds": [round(value, 3) for value in np.arange(.280, .341, .001)],
        "threshold_range_basis": "locked_after_pilot_score_scale_before_full_validation",
    })
    fingerprint = json_hash(protocol); root = ARTIFACTS / "medsiglip_v48_full" / fingerprint
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / "protocol.json", protocol)
    report_path = root / "report.json"
    if report_path.is_file():
        return __import__("json").loads(report_path.read_text())
    directory = Path(snapshot_download(repo_id=REPOSITORY, revision=REVISION,
                                       cache_dir=ARTIFACTS / "huggingface",
                                       local_files_only=True))
    store = DevelopmentReferenceStore(); rows = list(store.rows)
    # Scores are generated from clean RGB paths only. Reference masks are opened
    # only after every score plane exists on disk.
    scores_by_width = _load_or_encode_clean(rows, root, directory)
    score_maps = [np.mean([scores_by_width[width][index] for width in PATCH_WIDTHS], axis=0)
                  for index in range(len(rows))]
    score_hashes = {row["id"]: {str(width): sha256(root / f"{row['id']}_w{width}.npz")
                                for width in PATCH_WIDTHS} for row in rows}
    payloads = [store.load(row) for row in rows]
    lungs = [unlabeled_lung_probability(payload["rgb"]) for payload in payloads]
    candidates = []
    for lung_threshold in protocol["lung_thresholds"]:
        for lesion_threshold in protocol["lesion_thresholds"]:
            measured = [_measure(payload, lung_probability >= lung_threshold,
                                 (score >= lesion_threshold) & (lung_probability >= lung_threshold))
                        for payload, score, lung_probability in zip(payloads, score_maps, lungs)]
            candidates.append({"lung_threshold": lung_threshold,
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
              "pilot_report_sha256": sha256(ARTIFACTS / "medsiglip_v48"
                                               / pilot["protocol_fingerprint"] / "pilot.json"),
              "model_files_sha256": pilot["model_files_sha256"],
              "score_files_sha256": score_hashes, "development_winner": winner,
              "development_gate": {"passed": all(checks.values()), "checks": checks},
              "candidate_count": len(candidates), "weights_updated": False,
              "project_marked_fit_images": 0, "project_marked_fit_planes": 0,
              "test_payload_opened": False}
    atomic_json(report_path, report)
    return report
