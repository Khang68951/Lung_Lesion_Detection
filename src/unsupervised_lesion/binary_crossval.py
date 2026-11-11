from __future__ import annotations

import json
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from .crossval_segformer import development_folds, _score_native
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256
from .reference import ReferenceStore
from .reference_v2 import DevelopmentReferenceStore
from .segformer_segmentation import _image_tensor, _load, _restore, _rank
from .segmentation import unlabeled_lung_probability
from .supervised_segmentation import Geometry, _prepare


VERSION = "v27_binary_lesion_group_crossval"
SEED = 2704


def relative_lab_image(rgb: np.ndarray, predicted_lung: np.ndarray) -> np.ndarray:
    """Normalize within leakage-safe predicted lung while retaining relative colour."""
    tissue = predicted_lung.astype(bool)
    if tissue.shape != rgb.shape[:2] or int(tissue.sum()) < 64:
        return rgb.copy()
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    normalized = np.zeros_like(lab); normalized[:, :, 1:] = 128
    low, high = np.percentile(lab[:, :, 0][tissue], [2, 98])
    luminance_scale = max(8.0, float(high - low))
    normalized[:, :, 0] = np.clip(20 + (lab[:, :, 0] - low) * 215 / luminance_scale, 0, 255)
    for channel in (1, 2):
        values = lab[:, :, channel][tissue]
        lower, median, upper = np.percentile(values, [10, 50, 90])
        chroma_scale = max(8.0, float(upper - lower))
        normalized[:, :, channel] = np.clip(128 + (lab[:, :, channel] - median) * 96 / chroma_scale,
                                             32, 224)
    normalized[~tissue] = [0, 128, 128]
    return cv2.cvtColor(np.round(normalized).astype(np.uint8), cv2.COLOR_LAB2RGB)


def prepare_at_size(payload: dict, image_size: int) -> dict:
    if image_size == 384:
        return _prepare(payload)
    def letterbox(array: np.ndarray, interpolation: int) -> tuple[np.ndarray, Geometry]:
        height, width = array.shape[:2]; scale = min(image_size / height, image_size / width)
        resized_height, resized_width = max(1, round(height * scale)), max(1, round(width * scale))
        resized = cv2.resize(array, (resized_width, resized_height), interpolation=interpolation)
        shape = (image_size, image_size) + (() if array.ndim == 2 else (array.shape[2],))
        canvas = np.zeros(shape, dtype=array.dtype)
        top, left = (image_size - resized_height) // 2, (image_size - resized_width) // 2
        canvas[top:top + resized_height, left:left + resized_width] = resized
        return canvas, Geometry(height, width, resized_height, resized_width, top, left)
    rgb, geometry = letterbox(payload["rgb"], cv2.INTER_AREA); planes = {}
    for key in ("lung", "lesion", "valid", "ignore_lung", "ignore_lesion"):
        planes[key], _ = letterbox(payload[key].astype(np.uint8), cv2.INTER_NEAREST)
        planes[key] = planes[key] > 0
    return {"id": payload["row"]["id"], "group": payload["row"]["group"],
            "rgb": rgb, "geometry": geometry, "payload": payload, **planes}


def _augment(item: dict, rng: random.Random,
             style_records: list[dict] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = item["rgb"].copy(); target = item["lesion"].astype(np.float32)
    if style_records is not None:
        from .style_bank import maybe_apply_unmarked_style
        rgb = maybe_apply_unmarked_style(rgb, style_records, rng, .50)
    valid = (item["lung"] & item["valid"] & ~item["ignore_lung"] & ~item["ignore_lesion"]).astype(np.float32)
    if rng.random() < .5: rgb = np.flip(rgb, 1).copy(); target = np.flip(target, 1).copy(); valid = np.flip(valid, 1).copy()
    if rng.random() < .5: rgb = np.flip(rgb, 0).copy(); target = np.flip(target, 0).copy(); valid = np.flip(valid, 0).copy()
    rotation = rng.randrange(4)
    if rotation:
        rgb = np.rot90(rgb, rotation).copy(); target = np.rot90(target, rotation).copy(); valid = np.rot90(valid, rotation).copy()
    value = rgb.astype(np.float32) / 255.0; mean = value.mean(axis=(0, 1), keepdims=True)
    value = np.clip(((value - mean) * rng.uniform(.80, 1.20) + mean) * rng.uniform(.82, 1.18)
                    * np.asarray([rng.uniform(.90, 1.10) for _ in range(3)]), 0, 1)
    return (_image_tensor(np.round(value * 255).astype(np.uint8)), torch.from_numpy(target),
            torch.from_numpy(valid))


def _loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, dict]:
    logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce_map * valid).sum() / valid.sum().clamp_min(1)
    probability = logits.sigmoid() * valid; masked_target = target * valid
    dice = 1 - ((2 * (probability * masked_target).sum(dim=(1, 2)) + 1) /
                (probability.sum(dim=(1, 2)) + masked_target.sum(dim=(1, 2)) + 1)).mean()
    negative = valid * (1 - target)
    false_positive = (probability * negative).sum() / negative.sum().clamp_min(1)
    total = .65 * bce + 1.35 * dice + .25 * false_positive
    return total, {"bce": float(bce.detach()), "dice_loss": float(dice.detach()),
                   "false_positive_penalty": float(false_positive.detach())}


def _predict(model, items: list[dict], device: torch.device) -> list[np.ndarray]:
    model.eval(); result = []
    with torch.inference_mode():
        for item in items:
            logits = model(pixel_values=_image_tensor(item["rgb"])[None].to(device)).logits
            logits = F.interpolate(logits, size=item["rgb"].shape[:2], mode="bilinear", align_corners=False)
            result.append(_restore(logits.sigmoid()[0, 0].float().cpu().numpy(), item["geometry"]))
    return result


def fit_binary_crossval(epochs: int = 35,
                        pretrained: str = "nvidia/segformer-b1-finetuned-ade-512-512",
                        version: str = VERSION, batch_size: int = 2,
                        postprocess: str = "binary_crossval_v27",
                        development_store: DevelopmentReferenceStore | None = None,
                        input_mode: str = "rgb",
                        initialization: dict | None = None,
                        image_size: int = 384,
                        style_artifact: dict | None = None) -> dict:
    stores = ([development_store] if development_store is not None else
              [ReferenceStore("calibration"), ReferenceStore("confirmation")])
    metadata = (development_store.rows if development_store is not None else
                stores[0].rows + stores[1].rows)
    payloads = {row["id"]: store.load(row) for store in stores for row in store.rows}
    items = [prepare_at_size(payloads[row["id"]], image_size) for row in metadata]
    folds = development_folds(metadata)
    lung_probabilities = {item["id"]: unlabeled_lung_probability(item["payload"]["rgb"]) for item in items}
    if input_mode == "relative_lab":
        for item in items:
            item["rgb"] = relative_lab_image(item["rgb"], lung_probabilities[item["id"]] >= .50)
    elif input_mode != "rgb":
        raise ValueError(f"Unknown binary cross-validation input mode: {input_mode}")
    protocol = {"version": version, "seed": SEED, "epochs_per_fold": epochs,
                "pretrained": pretrained, "batch_size": batch_size, "fold_groups": folds,
                "development_images": len(items), "development_groups": len({item["group"] for item in items}),
                "loss_domain": "valid_reference_lung_only", "sealed_test_images_opened": 0,
                "epoch_lesion_thresholds": [.25, .40, .55, .70],
                "oof_lung_thresholds": [round(value, 2) for value in np.arange(.50, .951, .05)],
                "oof_lesion_thresholds": [round(value, 3) for value in np.arange(.10, .901, .025)]}
    if development_store is not None:
        protocol["reference_version"] = development_store.manifest["version"]
        protocol["reference_fingerprint"] = development_store.manifest["fingerprint"]
    if input_mode != "rgb": protocol["input_mode"] = input_mode
    if image_size != 384: protocol["image_size"] = image_size
    if initialization is not None:
        protocol["initialization"] = initialization
    style_records = None
    if style_artifact is not None:
        style_records = style_artifact["records"]
        protocol["style_bank"] = {"version": style_artifact["version"],
            "artifact_fingerprint": style_artifact["artifact_fingerprint"],
            "images": len(style_records), "transfer_probability": .50,
            "policy": "robust_lab_median_p10_p90_span_nonblack_pixels_training_only"}
    fingerprint = json_hash(protocol); root = ARTIFACTS / "binary_crossval" / fingerprint
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / "protocol.json", protocol)
    oof = root / "oof"; oof.mkdir(exist_ok=True); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts = []
    for fold_index, validation_groups in enumerate(folds):
        training = [item for item in items if item["group"] not in validation_groups]
        validation = [item for item in items if item["group"] in validation_groups]
        fold_folder = root / f"fold_{fold_index}"; fold_folder.mkdir(exist_ok=True)
        torch.manual_seed(SEED + fold_index); np.random.seed(SEED + fold_index)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED + fold_index)
        model = SegformerForSemanticSegmentation.from_pretrained(protocol["pretrained"], num_labels=1,
            ignore_mismatched_sizes=True, use_safetensors=True, cache_dir=ARTIFACTS / "model_cache",
            id2label={0: "lesion"}, label2id={"lesion": 0}).to(device)
        if initialization is not None:
            from .masked_pretraining import load_domain_weights
            transfer = load_domain_weights(model, initialization["checkpoint"],
                                           initialization["checkpoint_sha256"])
            if transfer["checkpoint_sha256"] != initialization["checkpoint_sha256"]:
                raise ValueError("V39 transfer provenance mismatch")
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-7)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda"); rng = random.Random(SEED + fold_index)
        best_rank = (-1., -1., -1., -999.); history = []
        for epoch in range(1, epochs + 1):
            model.train(); order = list(range(len(training))); rng.shuffle(order); losses = []
            for start in range(0, len(order), batch_size):
                samples = [_augment(training[index], rng, style_records)
                           for index in order[start:start + batch_size]]
                image = torch.stack([sample[0] for sample in samples]).to(device)
                target = torch.stack([sample[1] for sample in samples]).to(device)
                valid = torch.stack([sample[2] for sample in samples]).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(pixel_values=image).logits; loss, components = _loss(logits, target, valid)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
            scheduler.step(); probabilities = _predict(model, validation, device); choices = []
            for threshold in protocol["epoch_lesion_thresholds"]:
                summary, _ = _score_native(validation, probabilities,
                    [lung_probabilities[item["id"]] for item in validation], .50, threshold)
                choices.append({"lesion_threshold": threshold, "summary": summary})
            winner = max(choices, key=lambda value: _rank(value["summary"])); rank = _rank(winner["summary"])
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "winner": winner,
                            "last_batch_components": components, "learning_rate": optimizer.param_groups[0]["lr"]})
            checkpoint = {"version": version, "fold": fold_index, "epoch": epoch,
                          "model": model.state_dict(), "config": model.config.to_dict(),
                          "protocol_fingerprint": fingerprint, "winner": winner, "rank": rank}
            torch.save(checkpoint, fold_folder / "last.pt")
            if rank > best_rank: best_rank = rank; torch.save(checkpoint, fold_folder / "best.pt")
            atomic_json(fold_folder / "history.json", history)
            print(f"{version} fold {fold_index + 1}/4 epoch {epoch}/{epochs} loss={np.mean(losses):.4f} "
                  f"floor={rank[0]:.4f} dice={rank[1]:.4f}", flush=True)
        path = fold_folder / "best.pt"; best = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(best["model"]); probabilities = _predict(model, validation, device)
        for item, probability in zip(validation, probabilities):
            np.savez_compressed(oof / f"{item['id']}.npz", lesion=probability,
                                lung=lung_probabilities[item["id"]], fold=fold_index)
        artifact = {"fold": fold_index, "training_groups": sorted({item["group"] for item in training}),
                    "validation_groups": validation_groups, "best_epoch": best["epoch"],
                    "checkpoint": str(path.relative_to(PROJECT_ROOT)), "checkpoint_sha256": sha256(path)}
        artifacts.append(artifact); atomic_json(fold_folder / "artifact.json", artifact)
    lesion_probabilities = []; lungs = []
    for item in items:
        with np.load(oof / f"{item['id']}.npz", allow_pickle=False) as values:
            lesion_probabilities.append(values["lesion"]); lungs.append(values["lung"])
    candidates = []
    for lung_threshold in protocol["oof_lung_thresholds"]:
        for lesion_threshold in protocol["oof_lesion_thresholds"]:
            summary, _ = _score_native(items, lesion_probabilities, lungs, lung_threshold, lesion_threshold)
            candidates.append({"lung_threshold": lung_threshold, "lesion_threshold": lesion_threshold,
                               "summary": summary})
    winner = max(candidates, key=lambda value: _rank(value["summary"])); metric = winner["summary"]["pooled_lesion"]
    checks = {"lesion_dice_at_least_0_80": metric["dice"] >= .80,
              "precision_at_least_0_80": metric["precision"] >= .80,
              "recall_at_least_0_80": metric["recall"] >= .80,
              "specificity_at_least_0_80": metric["specificity"] >= .80,
              "lung_dice_at_least_0_90": winner["summary"]["pooled_lung"]["dice"] >= .90}
    report = {"status": "done", "created_at": now(), "version": version, "protocol": protocol,
              "fold_artifacts": artifacts, "oof_winner": winner,
              "oof_gate": {"passed": all(checks.values()), "checks": checks},
              "candidate_count": len(candidates), "test_payload_opened": False}
    atomic_json(root / "report.json", report)
    if report["oof_gate"]["passed"]:
        candidate = {"status": "frozen_after_oof_validation", "created_at": now(), "version": version,
            "selection_role": "four_fold_group_oof_binary", "lesion_postprocess": postprocess,
            "fold_checkpoints": [{"path": fold["checkpoint"], "sha256": fold["checkpoint_sha256"]}
                                 for fold in artifacts], "lung_threshold": winner["lung_threshold"],
            "lesion_threshold": winner["lesion_threshold"], "oof_summary": winner["summary"],
            "oof_gate": report["oof_gate"], "test_payload_opened": False}
        candidate["candidate_fingerprint"] = json_hash(candidate); folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
        atomic_json(folder / "candidate.json", candidate)
        atomic_json(ARTIFACTS / "frozen_candidate/current.json", {"path": str((folder / "candidate.json").relative_to(PROJECT_ROOT)),
            "sha256": sha256(folder / "candidate.json"), "candidate_fingerprint": candidate["candidate_fingerprint"]})
        atomic_json(ARTIFACTS / "test_unlock.json", {"created_at": now(), "confirmation_gate_passed": True,
            "gate_source": "four_fold_group_oof_binary", "candidate_fingerprint": candidate["candidate_fingerprint"],
            "crossval_fingerprint": fingerprint})
        report["candidate"] = candidate; atomic_json(root / "report.json", report)
    return report


def fit_binary_b2_crossval(epochs: int = 25) -> dict:
    return fit_binary_crossval(epochs, "nvidia/segformer-b2-finetuned-ade-512-512",
                               "v30_binary_segformer_b2_group_crossval", 1,
                               "binary_crossval_v30")


def fit_binary_v2_crossval(epochs: int = 35) -> dict:
    return fit_binary_crossval(epochs, "nvidia/segformer-b1-finetuned-ade-512-512",
                               "v35_binary_segformer_b1_reference_v2_group_crossval", 2,
                               "binary_crossval_v35_reference_v2", DevelopmentReferenceStore())


def fit_binary_relative_lab_crossval(epochs: int = 35) -> dict:
    return fit_binary_crossval(epochs, "nvidia/segformer-b1-finetuned-ade-512-512",
                               "v37_binary_segformer_b1_relative_lab_reference_v2", 2,
                               "binary_crossval_v37_relative_lab", DevelopmentReferenceStore(),
                               "relative_lab")


def fit_binary_v39_crossval(epochs: int = 35) -> dict:
    from .masked_pretraining import find_completed_pretraining
    artifact = find_completed_pretraining()
    initialization = {"version": artifact["version"],
                      "checkpoint": artifact["checkpoint"],
                      "checkpoint_sha256": artifact["checkpoint_sha256"],
                      "artifact_fingerprint": artifact["artifact_fingerprint"],
                      "classifier_policy": "fresh_one_logit_classifier"}
    return fit_binary_crossval(epochs, "nvidia/segformer-b1-finetuned-ade-512-512",
                               "v39_binary_segformer_b1_unlabeled_pretrained_reference_v2", 2,
                               "binary_crossval_v39_unlabeled_pretrained",
                               DevelopmentReferenceStore(), "rgb", initialization)


def fit_binary_v43_crossval(epochs: int = 35) -> dict:
    return fit_binary_crossval(epochs, "nvidia/segformer-b1-finetuned-ade-512-512",
                               "v43_binary_segformer_b1_native512_reference_v2", 1,
                               "binary_crossval_v43_native512", DevelopmentReferenceStore(),
                               "rgb", None, 512)


def fit_binary_v45_crossval(epochs: int = 35) -> dict:
    from .style_bank import build_style_bank
    return fit_binary_crossval(epochs, "nvidia/segformer-b1-finetuned-ade-512-512",
                               "v45_binary_segformer_b1_native512_unmarked_lab_style_reference_v2", 1,
                               "binary_crossval_v45_unmarked_lab_style", DevelopmentReferenceStore(),
                               "rgb", None, 512, build_style_bank())


def fit_binary_v70_crossval(epochs: int = 30) -> dict:
    """Higher-capacity native-resolution reference-v2 lesion baseline."""
    return fit_binary_crossval(epochs, "nvidia/segformer-b2-finetuned-ade-512-512",
                               "v70_binary_segformer_b2_native512_reference_v2", 1,
                               "binary_crossval_v70_b2_native512",
                               DevelopmentReferenceStore(), "rgb", None, 512)


def fit_binary_v74_crossval(epochs: int = 40) -> dict:
    """B2 native-resolution model with unmarked-image Lab style transfer."""
    from .style_bank import build_style_bank
    return fit_binary_crossval(epochs, "nvidia/segformer-b2-finetuned-ade-512-512",
                               "v74_binary_segformer_b2_native512_unmarked_lab_style_reference_v2", 1,
                               "binary_crossval_v74_b2_native512_unmarked_lab_style",
                               DevelopmentReferenceStore(), "rgb", None, 512, build_style_bank())


def evaluate_binary_store(candidate: dict, store: ReferenceStore) -> tuple[dict, list[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = [_prepare(store.load(row)) for row in store.rows]
    accumulated = [np.zeros(item["payload"]["rgb"].shape[:2], np.float32) for item in items]
    for checkpoint in candidate["fold_checkpoints"]:
        probabilities = _predict(_load({"checkpoint": checkpoint["path"],
            "checkpoint_sha256": checkpoint["sha256"]}, device), items, device)
        for index, probability in enumerate(probabilities): accumulated[index] += probability / len(candidate["fold_checkpoints"])
    lungs = [unlabeled_lung_probability(item["payload"]["rgb"]) for item in items]
    return _score_native(items, accumulated, lungs, candidate["lung_threshold"], candidate["lesion_threshold"])


def _report_root(parent: str, version: str) -> tuple[dict, object]:
    matches = []
    for path in (ARTIFACTS / parent).glob("*/report.json"):
        value = json.loads(path.read_text())
        if value.get("version") == version: matches.append((value, path.parent))
    if len(matches) != 1: raise ValueError(f"Expected one {version} report, found {len(matches)}")
    return matches[0]


def calibrate_binary_multiclass_ensemble() -> dict:
    multi, multi_root = _report_root("crossval_segformer", "v25_development_group_crossval")
    binary, binary_root = _report_root("binary_crossval", VERSION)
    stores = [ReferenceStore("calibration"), ReferenceStore("confirmation")]
    payloads = [store.load(row) for store in stores for row in store.rows]
    items = [{"payload": payload} for payload in payloads]; multi_scores = []; binary_scores = []; lungs = []
    for payload in payloads:
        identifier = payload["row"]["id"]
        with np.load(multi_root / "oof" / f"{identifier}.npz", allow_pickle=False) as values:
            multi_scores.append(values["lesion"])
        with np.load(binary_root / "oof" / f"{identifier}.npz", allow_pickle=False) as values:
            binary_scores.append(values["lesion"]); lungs.append(values["lung"])
    candidates = []
    for weight in (0., .25, .50, .75, 1.):
        scores = [(1 - weight) * first + weight * second for first, second in zip(multi_scores, binary_scores)]
        for threshold in [round(value, 3) for value in np.arange(.10, .901, .025)]:
            summary, _ = _score_native(items, scores, lungs, .50, threshold)
            candidates.append({"binary_weight": weight, "lung_threshold": .50,
                               "lesion_threshold": threshold, "summary": summary})
    winner = max(candidates, key=lambda value: _rank(value["summary"])); metric = winner["summary"]["pooled_lesion"]
    checks = {"lesion_dice_at_least_0_80": metric["dice"] >= .80,
              "precision_at_least_0_80": metric["precision"] >= .80,
              "recall_at_least_0_80": metric["recall"] >= .80,
              "specificity_at_least_0_80": metric["specificity"] >= .80,
              "lung_dice_at_least_0_90": winner["summary"]["pooled_lung"]["dice"] >= .90}
    report = {"status": "done", "created_at": now(), "version": "v28_binary_multiclass_oof_ensemble",
              "winner": winner, "oof_gate": {"passed": all(checks.values()), "checks": checks},
              "candidate_count": len(candidates), "test_payload_opened": False, "results": candidates}
    fingerprint = json_hash({"version": report["version"], "source_multi": str(multi_root),
                             "source_binary": str(binary_root), "weights": [0, .25, .5, .75, 1]})
    root = ARTIFACTS / "oof_ensemble" / fingerprint; atomic_json(root / "report.json", report)
    if report["oof_gate"]["passed"]:
        candidate = {"status": "frozen_after_oof_validation", "created_at": now(), "version": report["version"],
            "selection_role": "four_fold_group_oof_ensemble", "lesion_postprocess": "oof_ensemble_v28",
            "multiclass_checkpoints": [{"path": fold["checkpoint"], "sha256": fold["checkpoint_sha256"]}
                                       for fold in multi["fold_artifacts"]],
            "binary_checkpoints": [{"path": fold["checkpoint"], "sha256": fold["checkpoint_sha256"]}
                                   for fold in binary["fold_artifacts"]],
            "binary_weight": winner["binary_weight"], "lung_threshold": .50,
            "lesion_threshold": winner["lesion_threshold"], "oof_summary": winner["summary"],
            "oof_gate": report["oof_gate"], "test_payload_opened": False}
        candidate["candidate_fingerprint"] = json_hash(candidate); folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
        atomic_json(folder / "candidate.json", candidate)
        atomic_json(ARTIFACTS / "frozen_candidate/current.json", {"path": str((folder / "candidate.json").relative_to(PROJECT_ROOT)),
            "sha256": sha256(folder / "candidate.json"), "candidate_fingerprint": candidate["candidate_fingerprint"]})
        atomic_json(ARTIFACTS / "test_unlock.json", {"created_at": now(), "confirmation_gate_passed": True,
            "gate_source": "four_fold_group_oof_ensemble", "candidate_fingerprint": candidate["candidate_fingerprint"],
            "ensemble_fingerprint": fingerprint})
        report["candidate"] = candidate; atomic_json(root / "report.json", report)
    return report


def evaluate_oof_ensemble_store(candidate: dict, store: ReferenceStore) -> tuple[dict, list[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); items = [_prepare(store.load(row)) for row in store.rows]
    multi = [np.zeros(item["payload"]["rgb"].shape[:2], np.float32) for item in items]
    binary = [np.zeros_like(value) for value in multi]
    from .segformer_segmentation import _predict as predict_multi
    for checkpoint in candidate["multiclass_checkpoints"]:
        probabilities = predict_multi(_load({"checkpoint": checkpoint["path"], "checkpoint_sha256": checkpoint["sha256"]}, device), items, device)
        for index, (item, probability) in enumerate(zip(items, probabilities)):
            multi[index] += _restore(probability[2], item["geometry"]) / len(candidate["multiclass_checkpoints"])
    for checkpoint in candidate["binary_checkpoints"]:
        probabilities = _predict(_load({"checkpoint": checkpoint["path"], "checkpoint_sha256": checkpoint["sha256"]}, device), items, device)
        for index, probability in enumerate(probabilities): binary[index] += probability / len(candidate["binary_checkpoints"])
    weight = candidate["binary_weight"]; scores = [(1 - weight) * a + weight * b for a, b in zip(multi, binary)]
    lungs = [unlabeled_lung_probability(item["payload"]["rgb"]) for item in items]
    return _score_native(items, scores, lungs, candidate["lung_threshold"], candidate["lesion_threshold"])
