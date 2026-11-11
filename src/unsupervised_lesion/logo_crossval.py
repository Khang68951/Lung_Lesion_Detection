from __future__ import annotations

import random

import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

from .binary_crossval import _augment, _loss, _predict, relative_lab_image
from .crossval_segformer import _score_native
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256
from .reference_v2 import DevelopmentReferenceStore
from .segformer_segmentation import _rank
from .segmentation import unlabeled_lung_probability
from .supervised_segmentation import _prepare


VERSION = "v38_binary_segformer_b1_relative_lab_reference_v2_logo"
SEED = 3804


def fit_logo_crossval(epochs: int = 25, batch_size: int = 2) -> dict:
    store = DevelopmentReferenceStore(); metadata = store.rows
    payloads = {row["id"]: store.load(row) for row in metadata}
    items = [_prepare(payloads[row["id"]]) for row in metadata]
    lung_probabilities = {item["id"]: unlabeled_lung_probability(item["payload"]["rgb"]) for item in items}
    for item in items:
        item["rgb"] = relative_lab_image(item["rgb"], lung_probabilities[item["id"]] >= .50)
    groups = sorted({item["group"] for item in items})
    protocol = {"version": VERSION, "seed": SEED, "epochs_per_group": epochs,
                "checkpoint_epoch_policy": "fixed_final_epoch_no_held_group_selection",
                "pretrained": "nvidia/segformer-b1-finetuned-ade-512-512", "batch_size": batch_size,
                "held_out_groups": groups, "development_images": len(items),
                "development_groups": len(groups), "input_mode": "relative_lab",
                "reference_version": store.manifest["version"],
                "reference_fingerprint": store.manifest["fingerprint"],
                "loss_domain": "valid_reference_lung_only", "sealed_test_images_opened": 0,
                "oof_lung_thresholds": [round(value, 2) for value in np.arange(.50, .951, .05)],
                "oof_lesion_thresholds": [round(value, 3) for value in np.arange(.10, .901, .025)]}
    fingerprint = json_hash(protocol); root = ARTIFACTS / "logo_crossval" / fingerprint
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / "protocol.json", protocol)
    report_path = root / "report.json"
    if report_path.exists():
        import json
        return json.loads(report_path.read_text())
    oof = root / "oof"; oof.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); artifacts = []
    for fold_index, held_group in enumerate(groups):
        training = [item for item in items if item["group"] != held_group]
        validation = [item for item in items if item["group"] == held_group]
        folder = root / f"group_{fold_index:02d}"; folder.mkdir(exist_ok=True)
        torch.manual_seed(SEED + fold_index); np.random.seed(SEED + fold_index)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED + fold_index)
        model = SegformerForSemanticSegmentation.from_pretrained(protocol["pretrained"], num_labels=1,
            ignore_mismatched_sizes=True, use_safetensors=True, cache_dir=ARTIFACTS / "model_cache",
            id2label={0: "lesion"}, label2id={"lesion": 0}).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-7)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        rng = random.Random(SEED + fold_index); start_epoch = 0; history = []
        state_path = folder / "last.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=device, weights_only=False)
            if state["protocol_fingerprint"] != fingerprint or state["held_group"] != held_group:
                raise ValueError("LOGO resume state does not match protocol")
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"])
            rng.setstate(state["python_rng"]); torch.set_rng_state(state["torch_rng"].cpu())
            start_epoch = state["epoch"]; history = state["history"]
        for epoch_index in range(start_epoch, epochs):
            model.train(); order = list(range(len(training))); rng.shuffle(order); losses = []
            for start in range(0, len(order), batch_size):
                samples = [_augment(training[index], rng) for index in order[start:start + batch_size]]
                image = torch.stack([sample[0] for sample in samples]).to(device)
                target = torch.stack([sample[1] for sample in samples]).to(device)
                valid = torch.stack([sample[2] for sample in samples]).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(pixel_values=image).logits; loss, components = _loss(logits, target, valid)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
            scheduler.step(); history.append({"epoch": epoch_index + 1, "train_loss": float(np.mean(losses)),
                                              "last_batch_components": components,
                                              "learning_rate": optimizer.param_groups[0]["lr"]})
            state = {"protocol_fingerprint": fingerprint, "held_group": held_group,
                     "epoch": epoch_index + 1, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "scaler": scaler.state_dict(), "python_rng": rng.getstate(),
                     "torch_rng": torch.get_rng_state(), "history": history}
            torch.save(state, state_path); atomic_json(folder / "history.json", history)
            print(f"{VERSION} group {fold_index + 1}/{len(groups)} epoch {epoch_index + 1}/{epochs} "
                  f"loss={np.mean(losses):.4f}", flush=True)
        probabilities = _predict(model, validation, device)
        for item, probability in zip(validation, probabilities):
            np.savez_compressed(oof / f"{item['id']}.npz", lesion=probability,
                                lung=lung_probabilities[item["id"]], held_group=held_group)
        artifact = {"fold": fold_index, "held_group": held_group,
                    "training_groups": sorted({item["group"] for item in training}),
                    "validation_images": [item["id"] for item in validation],
                    "checkpoint": str(state_path.relative_to(PROJECT_ROOT)),
                    "checkpoint_sha256": sha256(state_path)}
        artifacts.append(artifact); atomic_json(folder / "artifact.json", artifact)
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
    report = {"status": "done", "created_at": now(), "version": VERSION,
              "protocol": protocol, "group_artifacts": artifacts, "oof_winner": winner,
              "oof_gate": {"passed": all(checks.values()), "checks": checks},
              "candidate_count": len(candidates), "test_payload_opened": False,
              "note": "A gate pass still does not unlock test until test-reference cleanup is versioned."}
    atomic_json(report_path, report); return report

