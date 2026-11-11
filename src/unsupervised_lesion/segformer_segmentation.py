from __future__ import annotations

import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from .evaluate import _measure, summarize
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256, within_project
from .reference import ReferenceStore
from .supervised_segmentation import (_image_tensor, _prepare, _restore,
                                      split_calibration_groups, load_supervised_candidate,
                                      _predict as predict_v19)
from .segmentation import unlabeled_lung_probability


VERSION = "v21_pretrained_segformer_b1"
SEED = 2104
EPOCHS = 50


def _augment(item: dict, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    rgb = item["rgb"].copy()
    target = np.zeros(item["lung"].shape, np.int64)
    target[item["lung"]] = 1; target[item["lesion"] & item["lung"]] = 2
    valid = item["valid"] & ~item["ignore_lung"] & ~item["ignore_lesion"]
    target[~valid] = -100
    if rng.random() < .5: rgb = np.flip(rgb, 1).copy(); target = np.flip(target, 1).copy()
    if rng.random() < .5: rgb = np.flip(rgb, 0).copy(); target = np.flip(target, 0).copy()
    rotation = rng.randrange(4)
    if rotation:
        rgb = np.rot90(rgb, rotation).copy(); target = np.rot90(target, rotation).copy()
    value = rgb.astype(np.float32) / 255.0
    mean = value.mean(axis=(0, 1), keepdims=True)
    value = np.clip(((value - mean) * rng.uniform(.85, 1.15) + mean) * rng.uniform(.85, 1.15)
                    * np.asarray([rng.uniform(.93, 1.07) for _ in range(3)]), 0, 1)
    return _image_tensor(np.round(value * 255).astype(np.uint8)), torch.from_numpy(target)


def _dice_loss(probability: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    probability = probability * valid; target = target * valid
    return 1 - ((2 * (probability * target).sum(dim=(1, 2)) + 1) /
                (probability.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1)).mean()


def _loss(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
    logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    weights = torch.tensor([.35, 1.0, 1.35], device=logits.device)
    ce = F.cross_entropy(logits, target, weight=weights, ignore_index=-100)
    probability = logits.softmax(1); valid = (target != -100).float()
    lung_target = (target > 0).float(); lesion_target = (target == 2).float()
    lung_dice = _dice_loss(1 - probability[:, 0], lung_target, valid)
    lesion_dice = _dice_loss(probability[:, 2], lesion_target, valid)
    normal = (target == 1).float()
    false_positive = (probability[:, 2] * normal).sum() / normal.sum().clamp_min(1)
    total = .65 * ce + .55 * lung_dice + 1.15 * lesion_dice + .20 * false_positive
    return total, {"cross_entropy": float(ce.detach()), "lung_dice_loss": float(lung_dice.detach()),
                   "lesion_dice_loss": float(lesion_dice.detach()),
                   "false_positive_penalty": float(false_positive.detach())}


def _predict(model, items: list[dict], device: torch.device) -> list[np.ndarray]:
    model.eval(); output = []
    with torch.inference_mode():
        for item in items:
            logits = model(pixel_values=_image_tensor(item["rgb"])[None].to(device)).logits
            logits = F.interpolate(logits, size=item["rgb"].shape[:2], mode="bilinear", align_corners=False)
            output.append(logits.softmax(1)[0].float().cpu().numpy())
    return output


def _score(items: list[dict], probabilities: list[np.ndarray], lung_threshold: float,
           lesion_threshold: float) -> tuple[dict, list[dict]]:
    rows = []
    for item, probability in zip(items, probabilities):
        lung_probability = _restore(1 - probability[0], item["geometry"])
        lesion_probability = _restore(probability[2], item["geometry"])
        lung = lung_probability >= lung_threshold
        lesion = (lesion_probability >= lesion_threshold) & lung
        rows.append(_measure(item["payload"], lung, lesion))
    return summarize(rows), rows


def _rank(summary: dict) -> tuple:
    metrics = summary["pooled_lesion"]
    return (min(metrics["precision"] or 0, metrics["recall"] or 0, metrics["specificity"] or 0),
            metrics["dice"] or 0, summary["pooled_lung"]["dice"] or 0,
            -(summary["ratio_mae_percentage_points"] or 999))


def fit_segformer(epochs: int = EPOCHS, pretrained: str = "nvidia/mit-b1",
                  version: str = VERSION, postprocess: str = "segformer_v21") -> dict:
    store = ReferenceStore("calibration"); train_groups, validation_groups = split_calibration_groups(store.rows)
    train_set = [_prepare(store.load(row)) for row in store.rows if row["group"] in train_groups]
    validation_set = [_prepare(store.load(row)) for row in store.rows if row["group"] in validation_groups]
    protocol = {"version": version, "seed": SEED, "epochs": epochs,
                "pretrained": pretrained, "classes": ["background", "normal_lung", "lesion"],
                "training_groups": train_groups, "internal_validation_groups": validation_groups,
                "training_ids": [item["id"] for item in train_set],
                "internal_validation_ids": [item["id"] for item in validation_set],
                "confirmation_images_used_for_fit_or_selection": 0, "sealed_test_images_opened": 0,
                "lung_thresholds": [.35, .45, .55], "lesion_thresholds": [.25, .40, .55, .70]}
    fingerprint = json_hash(protocol); folder = ARTIFACTS / "segformer" / fingerprint
    folder.mkdir(parents=True, exist_ok=True); atomic_json(folder / "protocol.json", protocol)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained, num_labels=3, ignore_mismatched_sizes=True,
        use_safetensors=True,
        cache_dir=ARTIFACTS / "model_cache", id2label={0: "background", 1: "normal_lung", 2: "lesion"},
        label2id={"background": 0, "normal_lung": 1, "lesion": 2}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []; best_rank = (-1., -1., -1., -999.); rng = random.Random(SEED)
    for epoch in range(1, epochs + 1):
        model.train(); order = list(range(len(train_set))); rng.shuffle(order); losses = []
        for start in range(0, len(order), 2):
            samples = [_augment(train_set[index], rng) for index in order[start:start + 2]]
            image = torch.stack([sample[0] for sample in samples]).to(device)
            target = torch.stack([sample[1] for sample in samples]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(pixel_values=image).logits; loss, components = _loss(logits, target)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
        scheduler.step(); probabilities = _predict(model, validation_set, device); epoch_candidates = []
        for lung_threshold in protocol["lung_thresholds"]:
            for lesion_threshold in protocol["lesion_thresholds"]:
                summary, _ = _score(validation_set, probabilities, lung_threshold, lesion_threshold)
                epoch_candidates.append({"lung_threshold": lung_threshold,
                                         "lesion_threshold": lesion_threshold, "summary": summary})
        winner = max(epoch_candidates, key=lambda item: _rank(item["summary"])); rank = _rank(winner["summary"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "learning_rate": optimizer.param_groups[0]["lr"], "winner": winner,
                        "last_batch_components": components})
        payload = {"version": version, "epoch": epoch, "model": model.state_dict(),
                   "config": model.config.to_dict(), "protocol": protocol,
                   "protocol_fingerprint": fingerprint, "winner": winner, "rank": rank}
        torch.save(payload, folder / "last.pt")
        if rank > best_rank: best_rank = rank; torch.save(payload, folder / "best.pt")
        atomic_json(folder / "history.json", history)
        print(f"{version} epoch {epoch}/{epochs} loss={np.mean(losses):.4f} floor={rank[0]:.4f} "
              f"dice={rank[1]:.4f} lung={rank[2]:.4f}", flush=True)
    checkpoint_path = folder / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False); winner = checkpoint["winner"]
    artifact = {"status": "fit_complete", "created_at": now(), "version": version,
                "artifact_fingerprint": fingerprint, "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
                "checkpoint_sha256": sha256(checkpoint_path), "best_epoch": checkpoint["epoch"],
                "training_images": len(train_set), "internal_validation_images": len(validation_set),
                "training_groups": train_groups, "internal_validation_groups": validation_groups,
                "confirmation_images_used_for_fit_or_selection": 0, "sealed_test_images_opened": 0,
                "internal_validation_winner": winner}
    atomic_json(folder / "artifact.json", artifact)
    candidate = {"status": "frozen_after_internal_validation", "created_at": now(), "version": version,
                 "selection_role": "calibration_group_holdout_only", "lesion_postprocess": postprocess,
                 "checkpoint": artifact["checkpoint"], "checkpoint_sha256": artifact["checkpoint_sha256"],
                 "lung_threshold": winner["lung_threshold"], "lesion_threshold": winner["lesion_threshold"],
                 "internal_validation_summary": winner["summary"],
                 "confirmation_images_used_for_selection": 0, "test_payload_opened": False}
    candidate["candidate_fingerprint"] = json_hash(candidate)
    candidate_folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
    atomic_json(candidate_folder / "candidate.json", candidate)
    atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                {"path": str((candidate_folder / "candidate.json").relative_to(PROJECT_ROOT)),
                 "sha256": sha256(candidate_folder / "candidate.json"),
                 "candidate_fingerprint": candidate["candidate_fingerprint"]})
    return {"artifact": artifact, "candidate": candidate}


def fit_dense_segformer(epochs: int = EPOCHS) -> dict:
    return fit_segformer(epochs, "nvidia/segformer-b1-finetuned-ade-512-512",
                         "v22_dense_pretrained_segformer_b1", "segformer_v22")


def _load(candidate: dict, device: torch.device):
    path = within_project(candidate["checkpoint"])
    if sha256(path) != candidate["checkpoint_sha256"]: raise ValueError("SegFormer checkpoint changed")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = SegformerForSemanticSegmentation(SegformerConfig.from_dict(checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval(); return model


def evaluate_segformer_store(candidate: dict, store: ReferenceStore) -> tuple[dict, list[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = _load(candidate, device)
    items = [_prepare(store.load(row)) for row in store.rows]; probabilities = _predict(model, items, device)
    return _score(items, probabilities, candidate["lung_threshold"], candidate["lesion_threshold"])


def _current_v21() -> dict:
    pointer = json.loads((ARTIFACTS / "frozen_candidate/current.json").read_text())
    path = within_project(pointer["path"])
    if sha256(path) != pointer["sha256"]: raise ValueError("Candidate pointer checksum changed")
    candidate = json.loads(path.read_text())
    if candidate.get("version") != VERSION: raise ValueError("v21b requires direct v21")
    return candidate


def _find_v19() -> dict:
    matches = []
    for path in (ARTIFACTS / "frozen_candidate").glob("*/candidate.json"):
        candidate = json.loads(path.read_text())
        if candidate.get("version") == "v19_supervised_calibration_deeplab": matches.append(candidate)
    if len(matches) != 1: raise ValueError(f"Expected one v19 candidate, found {len(matches)}")
    return matches[0]


def calibrate_hybrid() -> dict:
    base = _current_v21(); v19 = _find_v19(); store = ReferenceStore("calibration")
    _, validation_groups = split_calibration_groups(store.rows)
    items = [_prepare(store.load(row)) for row in store.rows if row["group"] in validation_groups]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segformer = _load(base, device); segment_probabilities = _predict(segformer, items, device)
    lung_model = load_supervised_candidate(v19, device); lung_probabilities = predict_v19(lung_model, items, device)
    candidates = []
    for threshold in [round(value, 3) for value in np.arange(.20, .601, .025)]:
        rows = []
        for item, lesion_probability, lung_probability in zip(items, segment_probabilities, lung_probabilities):
            lung = _restore(lung_probability[0], item["geometry"]) >= v19["lung_threshold"]
            lesion_score = _restore(lesion_probability[2], item["geometry"])
            rows.append(_measure(item["payload"], lung, (lesion_score >= threshold) & lung))
        summary = summarize(rows); candidates.append({"lesion_threshold": threshold, "summary": summary})
    winner = max(candidates, key=lambda item: _rank(item["summary"])); metrics = winner["summary"]["pooled_lesion"]
    gate_like = (metrics["dice"] >= .80 and metrics["precision"] >= .80 and metrics["recall"] >= .80
                 and metrics["specificity"] >= .80 and winner["summary"]["pooled_lung"]["dice"] >= .90)
    candidate = {"status": "frozen_after_internal_validation", "created_at": now(),
                 "version": "v21b_segformer_lesion_v19_lung",
                 "selection_role": "calibration_group_holdout_only", "lesion_postprocess": "hybrid_v21b",
                 "checkpoint": base["checkpoint"], "checkpoint_sha256": base["checkpoint_sha256"],
                 "lung_checkpoint": v19["checkpoint"], "lung_checkpoint_sha256": v19["checkpoint_sha256"],
                 "lung_threshold": v19["lung_threshold"], "lesion_threshold": winner["lesion_threshold"],
                 "internal_validation_summary": winner["summary"],
                 "internal_gate_passed": gate_like, "confirmation_images_used_for_selection": 0,
                 "test_payload_opened": False}
    candidate["candidate_fingerprint"] = json_hash(candidate)
    candidate_folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
    atomic_json(candidate_folder / "candidate.json", candidate)
    atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                {"path": str((candidate_folder / "candidate.json").relative_to(PROJECT_ROOT)),
                 "sha256": sha256(candidate_folder / "candidate.json"),
                 "candidate_fingerprint": candidate["candidate_fingerprint"]})
    report = {"status": "done", "created_at": now(), "version": candidate["version"],
              "candidate": candidate, "candidate_count": len(candidates), "results": candidates,
              "confirmation_images_opened": 0, "test_payload_opened": False}
    run = ARTIFACTS.parent / "runs" / f"calibration_hybrid_{candidate['candidate_fingerprint'][:16]}"
    atomic_json(run / "report.json", report); return report


def evaluate_hybrid_store(candidate: dict, store: ReferenceStore) -> tuple[dict, list[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segformer = _load(candidate, device)
    lung_base = {"checkpoint": candidate["lung_checkpoint"],
                 "checkpoint_sha256": candidate["lung_checkpoint_sha256"]}
    lung_model = load_supervised_candidate(lung_base, device)
    items = [_prepare(store.load(row)) for row in store.rows]
    lesion_probabilities = _predict(segformer, items, device); lung_probabilities = predict_v19(lung_model, items, device)
    rows = []
    for item, lesion_probability, lung_probability in zip(items, lesion_probabilities, lung_probabilities):
        lung = _restore(lung_probability[0], item["geometry"]) >= candidate["lung_threshold"]
        lesion_score = _restore(lesion_probability[2], item["geometry"])
        lesion = lesion_score >= candidate["lesion_threshold"]
        if candidate.get("lesion_postprocess") == "hybrid_union_v21c": lung = lung | lesion
        else: lesion = lesion & lung
        rows.append(_measure(item["payload"], lung, lesion))
    return summarize(rows), rows


def calibrate_union_lung() -> dict:
    pointer = json.loads((ARTIFACTS / "frozen_candidate/current.json").read_text())
    base = json.loads(within_project(pointer["path"]).read_text())
    if base.get("version") != "v21b_segformer_lesion_v19_lung": raise ValueError("v21c requires v21b")
    store = ReferenceStore("calibration"); _, validation_groups = split_calibration_groups(store.rows)
    items = [_prepare(store.load(row)) for row in store.rows if row["group"] in validation_groups]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segformer = _load(base, device); lung_base = {"checkpoint": base["lung_checkpoint"],
                                                  "checkpoint_sha256": base["lung_checkpoint_sha256"]}
    lung_model = load_supervised_candidate(lung_base, device)
    lesion_probabilities = _predict(segformer, items, device); lung_probabilities = predict_v19(lung_model, items, device)
    candidates = []
    for threshold in [round(value, 3) for value in np.arange(.20, .601, .025)]:
        rows = []
        for item, lesion_probability, lung_probability in zip(items, lesion_probabilities, lung_probabilities):
            lesion = _restore(lesion_probability[2], item["geometry"]) >= threshold
            lung = (_restore(lung_probability[0], item["geometry"]) >= base["lung_threshold"]) | lesion
            rows.append(_measure(item["payload"], lung, lesion))
        summary = summarize(rows); candidates.append({"lesion_threshold": threshold, "summary": summary})
    winner = max(candidates, key=lambda item: _rank(item["summary"])); metrics = winner["summary"]["pooled_lesion"]
    gate_like = (metrics["dice"] >= .80 and metrics["precision"] >= .80 and metrics["recall"] >= .80
                 and metrics["specificity"] >= .80 and winner["summary"]["pooled_lung"]["dice"] >= .90)
    candidate = {**{key: value for key, value in base.items()
                    if key not in {"created_at", "version", "candidate_fingerprint", "lesion_postprocess",
                                   "lesion_threshold", "internal_validation_summary", "internal_gate_passed"}},
                 "created_at": now(), "version": "v21c_segformer_lesion_union_v19_lung",
                 "lesion_postprocess": "hybrid_union_v21c", "lesion_threshold": winner["lesion_threshold"],
                 "internal_validation_summary": winner["summary"], "internal_gate_passed": gate_like}
    candidate["candidate_fingerprint"] = json_hash(candidate)
    folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
    atomic_json(folder / "candidate.json", candidate)
    atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                {"path": str((folder / "candidate.json").relative_to(PROJECT_ROOT)),
                 "sha256": sha256(folder / "candidate.json"),
                 "candidate_fingerprint": candidate["candidate_fingerprint"]})
    report = {"status": "done", "created_at": now(), "version": candidate["version"],
              "candidate": candidate, "candidate_count": len(candidates), "results": candidates,
              "confirmation_images_opened": 0, "test_payload_opened": False}
    run = ARTIFACTS.parent / "runs" / f"calibration_union_{candidate['candidate_fingerprint'][:16]}"
    atomic_json(run / "report.json", report); return report


def calibrate_unlabeled_lung() -> dict:
    pointer = json.loads((ARTIFACTS / "frozen_candidate/current.json").read_text())
    base = json.loads(within_project(pointer["path"]).read_text())
    if base.get("version") != "v21c_segformer_lesion_union_v19_lung": raise ValueError("v21d requires v21c")
    v11_matches = []
    for path in (ARTIFACTS / "frozen_candidate").glob("*/candidate.json"):
        value = json.loads(path.read_text())
        if value.get("version") == "v11b_coarse_regions_refined": v11_matches.append(value)
    if len(v11_matches) != 1: raise ValueError(f"Expected one v11b candidate, found {len(v11_matches)}")
    v11 = v11_matches[0]; store = ReferenceStore("calibration"); _, validation_groups = split_calibration_groups(store.rows)
    items = [_prepare(store.load(row)) for row in store.rows if row["group"] in validation_groups]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); segformer = _load(base, device)
    lesion_probabilities = _predict(segformer, items, device)
    lungs = [unlabeled_lung_probability(item["payload"]["rgb"]) >= v11["foreground_alpha_threshold"]
             for item in items]
    candidates = []
    for threshold in [round(value, 3) for value in np.arange(.20, .601, .025)]:
        rows = []
        for item, lesion_probability, lung in zip(items, lesion_probabilities, lungs):
            lesion = (_restore(lesion_probability[2], item["geometry"]) >= threshold) & lung
            rows.append(_measure(item["payload"], lung, lesion))
        summary = summarize(rows); candidates.append({"lesion_threshold": threshold, "summary": summary})
    winner = max(candidates, key=lambda item: _rank(item["summary"])); metrics = winner["summary"]["pooled_lesion"]
    gate_like = (metrics["dice"] >= .80 and metrics["precision"] >= .80 and metrics["recall"] >= .80
                 and metrics["specificity"] >= .80 and winner["summary"]["pooled_lung"]["dice"] >= .90)
    candidate = {"status": "frozen_after_internal_validation", "created_at": now(),
                 "version": "v21d_segformer_lesion_unlabeled_lung", "selection_role": "calibration_group_holdout_only",
                 "lesion_postprocess": "hybrid_unlabeled_lung_v21d", "checkpoint": base["checkpoint"],
                 "checkpoint_sha256": base["checkpoint_sha256"],
                 "foreground_method": v11["foreground_method"],
                 "foreground_alpha_threshold": v11["foreground_alpha_threshold"],
                 "foreground_weight_sha256": v11["foreground_weight_sha256"],
                 "lesion_threshold": winner["lesion_threshold"], "internal_validation_summary": winner["summary"],
                 "internal_gate_passed": gate_like, "confirmation_images_used_for_selection": 0,
                 "test_payload_opened": False}
    candidate["candidate_fingerprint"] = json_hash(candidate); folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
    atomic_json(folder / "candidate.json", candidate)
    atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                {"path": str((folder / "candidate.json").relative_to(PROJECT_ROOT)),
                 "sha256": sha256(folder / "candidate.json"),
                 "candidate_fingerprint": candidate["candidate_fingerprint"]})
    report = {"status": "done", "created_at": now(), "version": candidate["version"],
              "candidate": candidate, "candidate_count": len(candidates), "results": candidates,
              "confirmation_images_opened": 0, "test_payload_opened": False}
    run = ARTIFACTS.parent / "runs" / f"calibration_v21d_{candidate['candidate_fingerprint'][:16]}"
    atomic_json(run / "report.json", report); return report


def evaluate_unlabeled_lung_store(candidate: dict, store: ReferenceStore) -> tuple[dict, list[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = _load(candidate, device)
    items = [_prepare(store.load(row)) for row in store.rows]; probabilities = _predict(model, items, device); rows = []
    for item, probability in zip(items, probabilities):
        lung = unlabeled_lung_probability(item["payload"]["rgb"]) >= candidate["foreground_alpha_threshold"]
        lesion = (_restore(probability[2], item["geometry"]) >= candidate["lesion_threshold"]) & lung
        rows.append(_measure(item["payload"], lung, lesion))
    return summarize(rows), rows


def _find_candidate_version(version: str) -> dict:
    matches = []
    for path in (ARTIFACTS / "frozen_candidate").glob("*/candidate.json"):
        value = json.loads(path.read_text())
        if value.get("version") == version: matches.append(value)
    if len(matches) != 1: raise ValueError(f"Expected one {version} candidate, found {len(matches)}")
    return matches[0]


def calibrate_dual_ensemble() -> dict:
    v21 = _find_candidate_version(VERSION); v22 = _find_candidate_version("v22_dense_pretrained_segformer_b1")
    v19 = _find_v19(); store = ReferenceStore("calibration"); _, validation_groups = split_calibration_groups(store.rows)
    items = [_prepare(store.load(row)) for row in store.rows if row["group"] in validation_groups]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probabilities21 = _predict(_load(v21, device), items, device)
    probabilities22 = _predict(_load(v22, device), items, device)
    lung_probabilities = predict_v19(load_supervised_candidate(v19, device), items, device)
    native = []
    for item, p21, p22, lung in zip(items, probabilities21, probabilities22, lung_probabilities):
        native.append((_restore(p21[2], item["geometry"]), _restore(p22[2], item["geometry"]),
                       _restore(lung[0], item["geometry"])))
    candidates = []
    for weight in (0., .25, .50, .75, 1.):
        scores = [(1 - weight) * p21 + weight * p22 for p21, p22, _ in native]
        for threshold in [round(value, 3) for value in np.arange(.20, .601, .025)]:
            for recovery in (.60, .75, .90):
                rows = []
                for item, score, (_, _, lung_probability) in zip(items, scores, native):
                    lung = (lung_probability >= v19["lung_threshold"]) | (score >= recovery)
                    lesion = (score >= threshold) & lung
                    rows.append(_measure(item["payload"], lung, lesion))
                summary = summarize(rows)
                candidates.append({"v22_weight": weight, "lesion_threshold": threshold,
                                   "lung_recovery_threshold": recovery, "summary": summary})
    winner = max(candidates, key=lambda item: _rank(item["summary"])); metrics = winner["summary"]["pooled_lesion"]
    gate_like = (metrics["dice"] >= .80 and metrics["precision"] >= .80 and metrics["recall"] >= .80
                 and metrics["specificity"] >= .80 and winner["summary"]["pooled_lung"]["dice"] >= .90)
    candidate = {"status": "frozen_after_internal_validation", "created_at": now(),
                 "version": "v23_dual_segformer_conservative_lung", "selection_role": "calibration_group_holdout_only",
                 "lesion_postprocess": "dual_segformer_v23", "v21_checkpoint": v21["checkpoint"],
                 "v21_checkpoint_sha256": v21["checkpoint_sha256"], "v22_checkpoint": v22["checkpoint"],
                 "v22_checkpoint_sha256": v22["checkpoint_sha256"],
                 "lung_checkpoint": v19["checkpoint"], "lung_checkpoint_sha256": v19["checkpoint_sha256"],
                 "lung_threshold": v19["lung_threshold"], "v22_weight": winner["v22_weight"],
                 "lesion_threshold": winner["lesion_threshold"],
                 "lung_recovery_threshold": winner["lung_recovery_threshold"],
                 "internal_validation_summary": winner["summary"], "internal_gate_passed": gate_like,
                 "confirmation_images_used_for_selection": 0, "test_payload_opened": False}
    candidate["candidate_fingerprint"] = json_hash(candidate); folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
    atomic_json(folder / "candidate.json", candidate)
    atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                {"path": str((folder / "candidate.json").relative_to(PROJECT_ROOT)),
                 "sha256": sha256(folder / "candidate.json"),
                 "candidate_fingerprint": candidate["candidate_fingerprint"]})
    report = {"status": "done", "created_at": now(), "version": candidate["version"],
              "candidate": candidate, "candidate_count": len(candidates), "results": candidates,
              "confirmation_images_opened": 0, "test_payload_opened": False}
    run = ARTIFACTS.parent / "runs" / f"calibration_v23_{candidate['candidate_fingerprint'][:16]}"
    atomic_json(run / "report.json", report); return report


def evaluate_dual_ensemble_store(candidate: dict, store: ReferenceStore) -> tuple[dict, list[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = {"checkpoint": candidate["v21_checkpoint"], "checkpoint_sha256": candidate["v21_checkpoint_sha256"]}
    b = {"checkpoint": candidate["v22_checkpoint"], "checkpoint_sha256": candidate["v22_checkpoint_sha256"]}
    lung_base = {"checkpoint": candidate["lung_checkpoint"],
                 "checkpoint_sha256": candidate["lung_checkpoint_sha256"]}
    items = [_prepare(store.load(row)) for row in store.rows]
    p21 = _predict(_load(a, device), items, device); p22 = _predict(_load(b, device), items, device)
    lung_probabilities = predict_v19(load_supervised_candidate(lung_base, device), items, device); rows = []
    weight = candidate["v22_weight"]
    for item, first, second, lung_probability in zip(items, p21, p22, lung_probabilities):
        score = ((1 - weight) * _restore(first[2], item["geometry"]) +
                 weight * _restore(second[2], item["geometry"]))
        lung = ((_restore(lung_probability[0], item["geometry"]) >= candidate["lung_threshold"]) |
                (score >= candidate["lung_recovery_threshold"]))
        lesion = (score >= candidate["lesion_threshold"]) & lung
        rows.append(_measure(item["payload"], lung, lesion))
    return summarize(rows), rows


def calibrate_unlabeled_lung_ensemble() -> dict:
    v21 = _find_candidate_version(VERSION); v22 = _find_candidate_version("v22_dense_pretrained_segformer_b1")
    store = ReferenceStore("calibration"); _, validation_groups = split_calibration_groups(store.rows)
    items = [_prepare(store.load(row)) for row in store.rows if row["group"] in validation_groups]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p21 = _predict(_load(v21, device), items, device); p22 = _predict(_load(v22, device), items, device)
    native = [(_restore(a[2], item["geometry"]), _restore(b[2], item["geometry"]),
               unlabeled_lung_probability(item["payload"]["rgb"]))
              for item, a, b in zip(items, p21, p22)]
    lung_thresholds = [round(value, 2) for value in np.arange(.10, .901, .05)]
    eligible_lung = []
    for threshold in lung_thresholds:
        rows = [_measure(item["payload"], lung_probability >= threshold, np.zeros_like(lung_probability, bool))
                for item, (_, _, lung_probability) in zip(items, native)]
        if summarize(rows)["pooled_lung"]["dice"] >= .90: eligible_lung.append(threshold)
    candidates = []
    for weight in (0., .25, .50, .75, 1.):
        scores = [(1 - weight) * first + weight * second for first, second, _ in native]
        for lesion_threshold in [round(value, 3) for value in np.arange(.20, .601, .025)]:
            for lung_threshold in eligible_lung:
                rows = []
                for item, score, (_, _, lung_probability) in zip(items, scores, native):
                    lung = lung_probability >= lung_threshold; lesion = (score >= lesion_threshold) & lung
                    rows.append(_measure(item["payload"], lung, lesion))
                summary = summarize(rows); candidates.append({"v22_weight": weight,
                    "lesion_threshold": lesion_threshold, "lung_threshold": lung_threshold,
                    "summary": summary})
    if not candidates: raise RuntimeError("No unlabeled lung threshold reached Dice 0.90")
    winner = max(candidates, key=lambda item: _rank(item["summary"])); metrics = winner["summary"]["pooled_lesion"]
    gate_like = all((metrics["dice"] >= .80, metrics["precision"] >= .80, metrics["recall"] >= .80,
                     metrics["specificity"] >= .80, winner["summary"]["pooled_lung"]["dice"] >= .90))
    v11 = _find_candidate_version("v11b_coarse_regions_refined")
    candidate = {"status": "frozen_after_internal_validation", "created_at": now(),
                 "version": "v24_dual_segformer_unlabeled_lung", "selection_role": "calibration_group_holdout_only",
                 "lesion_postprocess": "dual_segformer_unlabeled_lung_v24",
                 "v21_checkpoint": v21["checkpoint"], "v21_checkpoint_sha256": v21["checkpoint_sha256"],
                 "v22_checkpoint": v22["checkpoint"], "v22_checkpoint_sha256": v22["checkpoint_sha256"],
                 "foreground_weight_sha256": v11["foreground_weight_sha256"],
                 "lung_threshold": winner["lung_threshold"], "v22_weight": winner["v22_weight"],
                 "lesion_threshold": winner["lesion_threshold"], "internal_validation_summary": winner["summary"],
                 "internal_gate_passed": gate_like, "confirmation_images_used_for_selection": 0,
                 "test_payload_opened": False}
    candidate["candidate_fingerprint"] = json_hash(candidate); folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
    atomic_json(folder / "candidate.json", candidate)
    atomic_json(ARTIFACTS / "frozen_candidate/current.json",
                {"path": str((folder / "candidate.json").relative_to(PROJECT_ROOT)),
                 "sha256": sha256(folder / "candidate.json"),
                 "candidate_fingerprint": candidate["candidate_fingerprint"]})
    report = {"status": "done", "created_at": now(), "version": candidate["version"],
              "candidate": candidate, "candidate_count": len(candidates),
              "eligible_lung_thresholds": eligible_lung, "results": candidates,
              "confirmation_images_opened": 0, "test_payload_opened": False}
    run = ARTIFACTS.parent / "runs" / f"calibration_v24_{candidate['candidate_fingerprint'][:16]}"
    atomic_json(run / "report.json", report); return report


def evaluate_unlabeled_lung_ensemble_store(candidate: dict, store: ReferenceStore) -> tuple[dict, list[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = {"checkpoint": candidate["v21_checkpoint"], "checkpoint_sha256": candidate["v21_checkpoint_sha256"]}
    b = {"checkpoint": candidate["v22_checkpoint"], "checkpoint_sha256": candidate["v22_checkpoint_sha256"]}
    items = [_prepare(store.load(row)) for row in store.rows]
    p21 = _predict(_load(a, device), items, device); p22 = _predict(_load(b, device), items, device); rows = []
    weight = candidate["v22_weight"]
    for item, first, second in zip(items, p21, p22):
        score = ((1 - weight) * _restore(first[2], item["geometry"]) +
                 weight * _restore(second[2], item["geometry"]))
        lung = unlabeled_lung_probability(item["payload"]["rgb"]) >= candidate["lung_threshold"]
        rows.append(_measure(item["payload"], lung, (score >= candidate["lesion_threshold"]) & lung))
    return summarize(rows), rows
