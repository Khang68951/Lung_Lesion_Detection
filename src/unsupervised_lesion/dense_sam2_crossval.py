from __future__ import annotations

import json
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .binary_crossval import _loss
from .crossval_segformer import _score_native, development_folds
from .io import (ARTIFACTS, PROJECT_ROOT, SAM2_CHECKPOINT, SAM2_VENDOR,
                 atomic_json, json_hash, now, sha256)
from .reference_v2 import DevelopmentReferenceStore
from .segformer_segmentation import _rank
from .segmentation import unlabeled_lung_probability
from .supervised_segmentation import _prepare, _restore


VERSION = "v44_frozen_dense_sam2_multiscale_colour_decoder_reference_v2"
SEED = 4404
EPOCHS = 60
CHECKPOINT = SAM2_CHECKPOINT
CHECKPOINT_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
VENDOR_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"


def _colour_channels(rgb: np.ndarray) -> np.ndarray:
    small = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
    rgb_value = small.astype(np.float32) / 255.0
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] /= 255.0; lab[:, :, 1:] = (lab[:, :, 1:] - 128.0) / 128.0
    return np.concatenate([rgb_value, lab], axis=2).transpose(2, 0, 1).copy()


class SAM2ColourDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.coarse = nn.Sequential(nn.Conv2d(256, 64, 1), nn.GroupNorm(8, 64), nn.GELU())
        self.middle = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                    nn.GroupNorm(8, 64), nn.GELU())
        self.fine = nn.Sequential(nn.Conv2d(96, 32, 3, padding=1),
                                  nn.GroupNorm(8, 32), nn.GELU())
        self.classifier = nn.Sequential(nn.Conv2d(38, 32, 3, padding=1),
                                        nn.GroupNorm(8, 32), nn.GELU(), nn.Conv2d(32, 1, 1))

    def forward(self, embed64: torch.Tensor, high128: torch.Tensor,
                high256: torch.Tensor, colour: torch.Tensor) -> torch.Tensor:
        coarse = F.interpolate(self.coarse(embed64), size=(128, 128), mode="bilinear",
                               align_corners=False)
        middle = self.middle(torch.cat([coarse, high128], dim=1))
        middle = F.interpolate(middle, size=(256, 256), mode="bilinear", align_corners=False)
        fine = self.fine(torch.cat([middle, high256], dim=1))
        return self.classifier(torch.cat([fine, colour], dim=1))


def _cache_features(store: DevelopmentReferenceStore, payloads: list[dict]) -> tuple[dict, object]:
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("Pinned SAM2.1 Hiera Tiny checkpoint changed")
    protocol = {
        "version": VERSION, "sam2_repository": "facebook/sam2.1-hiera-tiny",
        "sam2_revision": "de431c4043854a71d8101e17995dfe596bf101a5",
        "sam2_checkpoint": str(CHECKPOINT.relative_to(PROJECT_ROOT)),
        "sam2_checkpoint_sha256": CHECKPOINT_SHA256, "sam2_vendor_commit": VENDOR_COMMIT,
        "feature_shapes": {"embed64": [256, 64, 64], "high128": [64, 128, 128],
                           "high256": [32, 256, 256]},
        "feature_dtype": "float16", "reference_fingerprint": store.manifest["fingerprint"],
        "labels_used_during_encoding": 0, "sealed_test_images_opened": 0,
        "source_images": [{"id": row["id"], "image_sha256": row["sha256"]["image"]}
                          for row in store.rows],
    }
    fingerprint = json_hash(protocol); root = ARTIFACTS / "dense_sam2_features" / fingerprint
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / "protocol.json", protocol)
    expected = {p["row"]["id"] for p in payloads}; complete = {p.stem for p in root.glob("*.npz")}
    if complete != expected:
        if not torch.cuda.is_available():
            raise RuntimeError("V44 SAM2 feature extraction requires CUDA")
        if str(SAM2_VENDOR) not in sys.path: sys.path.insert(0, str(SAM2_VENDOR))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", str(CHECKPOINT),
                           device="cuda", apply_postprocessing=False).eval()
        model.requires_grad_(False); predictor = SAM2ImagePredictor(model)
        by_id = {p["row"]["id"]: p for p in payloads}
        with torch.inference_mode():
            for index, identifier in enumerate(sorted(expected - complete), start=1):
                predictor.set_image(np.asarray(by_id[identifier]["rgb"], np.uint8).copy())
                feature = predictor._features
                embed64 = feature["image_embed"][0].detach().cpu().numpy().astype(np.float16)
                high256 = feature["high_res_feats"][0][0].detach().cpu().numpy().astype(np.float16)
                high128 = feature["high_res_feats"][1][0].detach().cpu().numpy().astype(np.float16)
                if (embed64.shape != (256, 64, 64) or high128.shape != (64, 128, 128)
                        or high256.shape != (32, 256, 256)):
                    raise ValueError(f"Unexpected SAM2 feature shapes for {identifier}")
                np.savez_compressed(root / f"{identifier}.npz", embed64=embed64,
                                    high128=high128, high256=high256)
                print(f"{VERSION} feature {index}/{len(expected - complete)}", flush=True)
        del predictor, model; torch.cuda.empty_cache()
    files = [{"id": p["row"]["id"],
              "path": str((root / f"{p['row']['id']}.npz").relative_to(PROJECT_ROOT)),
              "sha256": sha256(root / f"{p['row']['id']}.npz")} for p in payloads]
    artifact = {"status": "done", "created_at": now(), "version": VERSION,
                "feature_fingerprint": fingerprint, "images": len(files),
                "labels_used_during_encoding": 0, "test_payload_opened": False, "files": files}
    atomic_json(root / "artifact.json", artifact)
    return artifact, root


def _augment(item: dict, rng: random.Random) -> tuple[torch.Tensor, ...]:
    names = ("embed64", "high128", "high256", "colour")
    arrays = {name: item[name].copy() for name in names}
    target = item["lesion"].astype(np.float32).copy()
    valid = (item["lung"] & item["valid"] & ~item["ignore_lung"] &
             ~item["ignore_lesion"]).astype(np.float32)
    if rng.random() < .5:
        arrays = {k: np.flip(v, 2).copy() for k, v in arrays.items()}
        target = np.flip(target, 1).copy(); valid = np.flip(valid, 1).copy()
    if rng.random() < .5:
        arrays = {k: np.flip(v, 1).copy() for k, v in arrays.items()}
        target = np.flip(target, 0).copy(); valid = np.flip(valid, 0).copy()
    turns = rng.randrange(4)
    if turns:
        arrays = {k: np.rot90(v, turns, axes=(1, 2)).copy() for k, v in arrays.items()}
        target = np.rot90(target, turns).copy(); valid = np.rot90(valid, turns).copy()
    return tuple(torch.from_numpy(arrays[k]).float() for k in names) + (
        torch.from_numpy(target), torch.from_numpy(valid))


def _predict(model: nn.Module, items: list[dict], device: torch.device) -> list[np.ndarray]:
    model.eval(); result = []
    with torch.inference_mode():
        for item in items:
            tensors = [torch.from_numpy(item[k])[None].float().to(device)
                       for k in ("embed64", "high128", "high256", "colour")]
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(*tensors)
            probability = F.interpolate(logits, size=item["rgb"].shape[:2], mode="bilinear",
                                        align_corners=False).sigmoid()[0, 0]
            result.append(_restore(probability.float().cpu().numpy(), item["geometry"]))
    return result


def fit_dense_sam2_crossval(epochs: int = EPOCHS) -> dict:
    if epochs != EPOCHS:
        raise ValueError(f"V44 predeclared exactly {EPOCHS} epochs; received {epochs}")
    store = DevelopmentReferenceStore(); payloads = [store.load(row) for row in store.rows]
    feature_artifact, feature_root = _cache_features(store, payloads)
    items = [_prepare(payload) for payload in payloads]
    for item in items:
        with np.load(feature_root / f"{item['id']}.npz", allow_pickle=False) as values:
            for key in ("embed64", "high128", "high256"):
                item[key] = values[key].astype(np.float32)
        item["colour"] = _colour_channels(item["rgb"])
    folds = development_folds(store.rows)
    lung_probabilities = {item["id"]: unlabeled_lung_probability(item["payload"]["rgb"])
                          for item in items}
    protocol = {
        "version": VERSION, "seed": SEED, "epochs_per_fold": EPOCHS, "batch_size": 2,
        "feature_fingerprint": feature_artifact["feature_fingerprint"],
        "reference_fingerprint": store.manifest["fingerprint"], "fold_groups": folds,
        "decoder": "sam2_64to128to256_multiscale_plus_rgb_lab6",
        "optimizer": {"name": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-3},
        "scheduler": "cosine_to_1e-6", "gradient_clip": 1.0,
        "augmentation": "paired_hv_flip_rot90", "loss_domain": "valid_reference_lung_only",
        "encoder_trainable": False, "development_images": len(items),
        "development_groups": len({x["group"] for x in items}),
        "epoch_lesion_thresholds": [.25, .40, .55, .70],
        "oof_lung_thresholds": [round(x, 2) for x in np.arange(.50, .951, .05)],
        "oof_lesion_thresholds": [round(x, 3) for x in np.arange(.10, .901, .025)],
        "sealed_test_images_opened": 0,
    }
    fingerprint = json_hash(protocol); root = ARTIFACTS / "dense_sam2_crossval" / fingerprint
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / "protocol.json", protocol)
    oof = root / "oof"; oof.mkdir(exist_ok=True); device = torch.device("cuda"); artifacts = []
    for fold_index, validation_groups in enumerate(folds):
        training = [x for x in items if x["group"] not in validation_groups]
        validation = [x for x in items if x["group"] in validation_groups]
        folder = root / f"fold_{fold_index}"; folder.mkdir(exist_ok=True)
        torch.manual_seed(SEED + fold_index); np.random.seed(SEED + fold_index)
        torch.cuda.manual_seed_all(SEED + fold_index); model = SAM2ColourDecoder().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda"); rng = random.Random(SEED + fold_index)
        history = []; best_rank = (-1., -1., -1., -999.)
        for epoch in range(1, EPOCHS + 1):
            model.train(); order = list(range(len(training))); rng.shuffle(order); losses = []
            for start in range(0, len(order), 2):
                batch = [_augment(training[i], rng) for i in order[start:start + 2]]
                values = [torch.stack([sample[i] for sample in batch]).to(device) for i in range(6)]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda"):
                    logits = model(*values[:4]); loss, components = _loss(logits, values[4], values[5])
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
            scheduler.step(); probabilities = _predict(model, validation, device); choices = []
            for threshold in protocol["epoch_lesion_thresholds"]:
                summary, _ = _score_native(validation, probabilities,
                    [lung_probabilities[x["id"]] for x in validation], .50, threshold)
                choices.append({"lesion_threshold": threshold, "summary": summary})
            winner = max(choices, key=lambda x: _rank(x["summary"])); rank = _rank(winner["summary"])
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "winner": winner,
                            "last_batch_components": components,
                            "learning_rate": optimizer.param_groups[0]["lr"]})
            checkpoint = {"version": VERSION, "fold": fold_index, "epoch": epoch,
                          "model": model.state_dict(), "protocol_fingerprint": fingerprint,
                          "winner": winner, "rank": rank}
            torch.save(checkpoint, folder / "last.pt")
            if rank > best_rank: best_rank = rank; torch.save(checkpoint, folder / "best.pt")
            atomic_json(folder / "history.json", history)
            print(f"{VERSION} fold {fold_index + 1}/4 epoch {epoch}/{EPOCHS} "
                  f"loss={np.mean(losses):.4f} floor={rank[0]:.4f} dice={rank[1]:.4f}", flush=True)
        path = folder / "best.pt"; best = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(best["model"]); probabilities = _predict(model, validation, device)
        for item, probability in zip(validation, probabilities):
            np.savez_compressed(oof / f"{item['id']}.npz", lesion=probability,
                                lung=lung_probabilities[item["id"]], fold=fold_index)
        artifact = {"fold": fold_index, "training_groups": sorted({x["group"] for x in training}),
                    "validation_groups": validation_groups, "best_epoch": best["epoch"],
                    "checkpoint": str(path.relative_to(PROJECT_ROOT)), "checkpoint_sha256": sha256(path)}
        artifacts.append(artifact); atomic_json(folder / "artifact.json", artifact)
    probabilities = []; lungs = []
    for item in items:
        with np.load(oof / f"{item['id']}.npz", allow_pickle=False) as values:
            probabilities.append(values["lesion"]); lungs.append(values["lung"])
    candidates = []
    for lung_threshold in protocol["oof_lung_thresholds"]:
        for lesion_threshold in protocol["oof_lesion_thresholds"]:
            summary, _ = _score_native(items, probabilities, lungs, lung_threshold, lesion_threshold)
            candidates.append({"lung_threshold": lung_threshold, "lesion_threshold": lesion_threshold,
                               "summary": summary})
    winner = max(candidates, key=lambda x: _rank(x["summary"])); metric = winner["summary"]["pooled_lesion"]
    checks = {"lesion_dice_at_least_0_80": metric["dice"] >= .80,
              "precision_at_least_0_80": metric["precision"] >= .80,
              "recall_at_least_0_80": metric["recall"] >= .80,
              "specificity_at_least_0_80": metric["specificity"] >= .80,
              "lung_dice_at_least_0_90": winner["summary"]["pooled_lung"]["dice"] >= .90}
    report = {"status": "done", "created_at": now(), "version": VERSION, "protocol": protocol,
              "feature_artifact": feature_artifact, "fold_artifacts": artifacts,
              "oof_winner": winner, "oof_gate": {"passed": all(checks.values()), "checks": checks},
              "candidate_count": len(candidates), "test_payload_opened": False}
    atomic_json(root / "report.json", report)
    return report
