from __future__ import annotations

import gc
import json
import random

import numpy as np
import segmentation_models_pytorch as smp
import torch

from .binary_crossval import _loss
from .crossval_segformer import development_folds
from .evaluate import _measure, summarize
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256
from .lobe_crop_crossval_v62 import _augment, prepare_side_item, restore_crop, v54_side_masks
from .lung_consensus_v54 import V53_ROOT
from .reference_v2 import DevelopmentReferenceStore


VERSION = "v65_supervised_resnet34_unetpp_multiview_group_oof"
SEED = 6505
IMAGE_SIZE = 512


def build_unetpp() -> torch.nn.Module:
    return smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights="imagenet",
                            in_channels=3, classes=1, activation=None)


def _predict(model, items: list[dict], supports: dict[tuple[str, str], np.ndarray],
             device: torch.device) -> dict[str, dict[str, np.ndarray]]:
    model.eval(); sums = {}; counts = {}; maxima = {}; whole = {}
    with torch.inference_mode():
        for item in items:
            probability = model(_image(item["rgb"])[None].to(device)).sigmoid()[0, 0].float().cpu().numpy()
            restored = restore_crop(probability, item["geometry"])
            identifier, view = item["id"], item["side"]; support = supports[(identifier, view)]
            restored[~support] = 0
            if identifier not in sums:
                sums[identifier] = np.zeros_like(restored); counts[identifier] = np.zeros_like(restored)
                maxima[identifier] = np.zeros_like(restored)
            sums[identifier][support] += restored[support]; counts[identifier][support] += 1
            maxima[identifier] = np.maximum(maxima[identifier], restored)
            if view == "whole": whole[identifier] = restored
    return {identifier: {"average": sums[identifier] / np.maximum(counts[identifier], 1),
                         "maximum": maxima[identifier], "whole": whole[identifier]}
            for identifier in sums}


def _image(rgb: np.ndarray) -> torch.Tensor:
    value = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    return (value - mean) / std


def fit_unetpp_v65(epochs: int = 20) -> dict:
    if epochs < 1: raise ValueError("V65 epochs must be positive")
    store = DevelopmentReferenceStore(); rows = list(store.rows)
    payloads = {row["id"]: store.load(row) for row in rows}
    items = []; lungs = {}; supports = {}; source_hashes = {}
    for row in rows:
        path = V53_ROOT / f"{row['id']}.npz"; sides = v54_side_masks(path)
        whole = sides["left"] | sides["right"]; lungs[row["id"]] = whole
        for view, mask in (("whole", whole), ("left", sides["left"]), ("right", sides["right"])):
            items.append(prepare_side_item(payloads[row["id"]], mask, view, IMAGE_SIZE))
            supports[(row["id"], view)] = mask
        source_hashes[row["id"]] = sha256(path)
    folds = development_folds(rows)
    protocol = {"version": VERSION, "seed": SEED, "architecture": "smp.UnetPlusPlus",
        "segmentation_models_pytorch": smp.__version__, "encoder": "resnet34",
        "encoder_weights": "ImageNet", "epochs_per_fold": epochs, "image_size": IMAGE_SIZE,
        "batch_size": 1, "learning_rate": 1e-4, "weight_decay": 1e-4,
        "views": ["whole", "left", "right"], "view_fusion": ["average", "maximum", "whole"],
        "fold_groups": folds, "development_images": len(rows), "training_views": len(items),
        "checkpoint_selection": "fixed final epoch; no held-out epoch selection",
        "lesion_thresholds": [round(float(x), 3) for x in np.arange(.05, .951, .01)],
        "fit_policy": "user-authorized supervised development fitting on 2026-11-05",
        "reference_fingerprint": store.manifest["fingerprint"],
        "marked_fit_images": len(rows), "sealed_test_images_opened": 0}
    fingerprint = json_hash(protocol); root = ARTIFACTS / "unetpp_crossval_v65" / fingerprint
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / "protocol.json", protocol)
    report_path = root / "report.json"
    if report_path.is_file(): return json.loads(report_path.read_text())
    oof = root / "oof"; oof.mkdir(exist_ok=True); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts = []
    for fold_index, validation_groups in enumerate(folds):
        training = [item for item in items if item["group"] not in validation_groups]
        validation = [item for item in items if item["group"] in validation_groups]
        validation_ids = sorted({item["id"] for item in validation})
        folder = root / f"fold_{fold_index}"; folder.mkdir(exist_ok=True); artifact_path = folder / "artifact.json"
        if artifact_path.is_file() and all((oof / f"{x}.npz").is_file() for x in validation_ids):
            artifacts.append(json.loads(artifact_path.read_text())); continue
        torch.manual_seed(SEED + fold_index); np.random.seed(SEED + fold_index)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED + fold_index)
        model = build_unetpp().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"],
                                      weight_decay=protocol["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        rng = random.Random(SEED + fold_index); history = []; start_epoch = 1; last = folder / "last.pt"
        if last.is_file():
            state = torch.load(last, map_location=device, weights_only=False)
            if state.get("protocol_fingerprint") != fingerprint: raise ValueError("V65 resume protocol changed")
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"])
            rng.setstate(state["python_rng_state"]); history = state["history"]; start_epoch = state["epoch"] + 1
        for epoch in range(start_epoch, epochs + 1):
            model.train(); order = list(range(len(training))); rng.shuffle(order); losses = []
            for index in order:
                image, target, valid = _augment(training[index], rng)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    loss, components = _loss(model(image[None].to(device)), target[None].to(device),
                                             valid[None].to(device))
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
            scheduler.step(); history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                "last_batch_components": components, "learning_rate": optimizer.param_groups[0]["lr"]})
            torch.save({"version": VERSION, "fold": fold_index, "epoch": epoch,
                "protocol_fingerprint": fingerprint, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "python_rng_state": rng.getstate(), "history": history}, last)
            atomic_json(folder / "history.json", history)
            print(f"{VERSION} fold {fold_index + 1}/4 epoch {epoch}/{epochs} loss={np.mean(losses):.4f}", flush=True)
        predictions = _predict(model, validation, supports, device)
        for identifier, modes in predictions.items():
            np.savez_compressed(oof / f"{identifier}.npz", lung=lungs[identifier].astype(np.uint8),
                                average=modes["average"], maximum=modes["maximum"], whole=modes["whole"],
                                fold=fold_index)
        artifact = {"fold": fold_index, "training_groups": sorted({x["group"] for x in training}),
            "validation_groups": validation_groups, "final_epoch": epochs,
            "checkpoint": str(last.relative_to(PROJECT_ROOT)), "checkpoint_sha256": sha256(last)}
        atomic_json(artifact_path, artifact); artifacts.append(artifact)
        del model, optimizer, scheduler, scaler
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    frozen = []
    for row in rows:
        with np.load(oof / f"{row['id']}.npz", allow_pickle=False) as values:
            frozen.append({key: values[key].copy() for key in ("average", "maximum", "whole")} |
                          {"lung": values["lung"].astype(bool)})
    candidates = []
    for mode in protocol["view_fusion"]:
        for threshold in protocol["lesion_thresholds"]:
            measured = [_measure(payloads[row["id"]], item["lung"],
                                 (item[mode] >= threshold) & item["lung"])
                        for row, item in zip(rows, frozen)]
            candidates.append({"mode": mode, "lesion_threshold": threshold, "summary": summarize(measured)})
    key = lambda x: (min(x["summary"]["pooled_lesion"][m] or 0 for m in ("precision", "recall", "specificity")),
                     x["summary"]["pooled_lesion"]["dice"] or 0)
    winner = max(candidates, key=key); dice_winner = max(candidates, key=lambda x: x["summary"]["pooled_lesion"]["dice"] or 0)
    metric = winner["summary"]["pooled_lesion"]
    checks = {"lesion_dice_at_least_0_80": metric["dice"] >= .80,
              "precision_at_least_0_80": metric["precision"] >= .80,
              "recall_at_least_0_80": metric["recall"] >= .80,
              "specificity_at_least_0_80": metric["specificity"] >= .80,
              "lung_dice_at_least_0_95": winner["summary"]["pooled_lung"]["dice"] >= .95}
    report = {"status": "done", "created_at": now(), "version": VERSION, "protocol": protocol,
        "protocol_fingerprint": fingerprint, "source_v53_sha256": source_hashes,
        "fold_artifacts": artifacts, "oof_balanced_winner": winner, "oof_dice_winner": dice_winner,
        "oof_gate": {"passed": all(checks.values()), "checks": checks},
        "candidate_count": len(candidates), "test_payload_opened": False}
    atomic_json(report_path, report)
    if report["oof_gate"]["passed"]:
        atomic_json(ARTIFACTS / "test_unlock.json", {"created_at": now(), "confirmation_gate_passed": True,
            "gate_source": VERSION, "crossval_fingerprint": fingerprint})
    return report
