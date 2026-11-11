from __future__ import annotations

import gc
import json
import random

import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

from .binary_crossval import _loss
from .crossval_segformer import development_folds
from .evaluate import _measure, summarize
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256
from .lobe_crop_crossval_v62 import _augment, prepare_side_item, v54_side_masks
from .lung_consensus_v54 import V53_ROOT
from .multiview_crossval_v64 import _predict_views
from .reference_v2 import DevelopmentReferenceStore
from .style_bank import build_style_bank, maybe_apply_unmarked_style


VERSION = "v67_supervised_v54_multiview_unmarked_style_group_oof"
V64_VERSION = "v64_supervised_v54_multiview_group_oof"
PRETRAINED = "nvidia/segformer-b1-finetuned-ade-512-512"
SEED = 6705
IMAGE_SIZE = 512


def _v64_report() -> dict:
    matches = []
    for path in (ARTIFACTS / "multiview_crossval_v64").glob("*/report.json"):
        report = json.loads(path.read_text())
        if report.get("version") == V64_VERSION and report.get("status") == "done":
            matches.append(report)
    if len(matches) != 1: raise ValueError(f"Expected one completed V64 report, found {len(matches)}")
    return matches[0]


def _style_augment(item: dict, records: list[dict], rng: random.Random):
    styled = dict(item)
    styled["rgb"] = maybe_apply_unmarked_style(item["rgb"], records, rng, .75)
    return _augment(styled, rng)


def fit_style_multiview_v67(epochs: int = 5) -> dict:
    if epochs < 1: raise ValueError("V67 epochs must be positive")
    store = DevelopmentReferenceStore(); rows = list(store.rows)
    payloads = {row["id"]: store.load(row) for row in rows}; items = []; lungs = {}; supports = {}
    for row in rows:
        sides = v54_side_masks(V53_ROOT / f"{row['id']}.npz"); whole = sides["left"] | sides["right"]
        lungs[row["id"]] = whole
        for view, mask in (("whole", whole), ("left", sides["left"]), ("right", sides["right"])):
            items.append(prepare_side_item(payloads[row["id"]], mask, view, IMAGE_SIZE))
            supports[(row["id"], view)] = mask
    folds = development_folds(rows); v64 = _v64_report(); style = build_style_bank(); records = style["records"]
    if [x["validation_groups"] for x in v64["fold_artifacts"]] != folds: raise ValueError("V64/V67 folds differ")
    protocol = {"version": VERSION, "seed": SEED, "epochs_per_fold": epochs,
        "image_size": IMAGE_SIZE, "batch_size": 1, "pretrained": PRETRAINED,
        "initialization": V64_VERSION, "learning_rate": 1e-5,
        "views": ["whole", "left", "right"], "style_probability": .75,
        "style_bank_fingerprint": style["artifact_fingerprint"], "unmarked_style_images": len(records),
        "style_policy": "robust Lab median and p10-p90 span; training views only",
        "fold_groups": folds, "checkpoint_selection": "fixed final epoch",
        "lesion_thresholds": [round(float(x), 3) for x in np.arange(.05, .951, .01)],
        "reference_fingerprint": store.manifest["fingerprint"],
        "fit_policy": "user-authorized supervised development plus unmarked style fitting",
        "marked_fit_images": len(rows), "sealed_test_images_opened": 0}
    fingerprint = json_hash(protocol); root = ARTIFACTS / "style_multiview_v67" / fingerprint
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / "protocol.json", protocol)
    report_path = root / "report.json"
    if report_path.is_file(): return json.loads(report_path.read_text())
    oof = root / "oof"; oof.mkdir(exist_ok=True); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts = []
    for fold_index, validation_groups in enumerate(folds):
        training = [x for x in items if x["group"] not in validation_groups]
        validation = [x for x in items if x["group"] in validation_groups]
        validation_ids = sorted({x["id"] for x in validation}); folder = root / f"fold_{fold_index}"
        folder.mkdir(exist_ok=True); artifact_path = folder / "artifact.json"
        if artifact_path.is_file() and all((oof / f"{x}.npz").is_file() for x in validation_ids):
            artifacts.append(json.loads(artifact_path.read_text())); continue
        torch.manual_seed(SEED + fold_index); np.random.seed(SEED + fold_index)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED + fold_index)
        model = SegformerForSemanticSegmentation.from_pretrained(PRETRAINED, num_labels=1,
            ignore_mismatched_sizes=True, use_safetensors=True, cache_dir=ARTIFACTS / "model_cache",
            id2label={0: "lesion"}, label2id={"lesion": 0}).to(device)
        source = PROJECT_ROOT / v64["fold_artifacts"][fold_index]["checkpoint"]
        if sha256(source) != v64["fold_artifacts"][fold_index]["checkpoint_sha256"]:
            raise ValueError("Pinned V64 checkpoint changed")
        source_state = torch.load(source, map_location=device, weights_only=False)
        model.load_state_dict(source_state["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-7)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        rng = random.Random(SEED + fold_index); history = []; start_epoch = 1; last = folder / "last.pt"
        if last.is_file():
            state = torch.load(last, map_location=device, weights_only=False)
            if state.get("protocol_fingerprint") != fingerprint: raise ValueError("V67 resume protocol changed")
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"])
            rng.setstate(state["python_rng_state"]); history = state["history"]; start_epoch = state["epoch"] + 1
        for epoch in range(start_epoch, epochs + 1):
            model.train(); order = list(range(len(training))); rng.shuffle(order); losses = []
            for index in order:
                image, target, valid = _style_augment(training[index], records, rng)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    loss, components = _loss(model(pixel_values=image[None].to(device)).logits,
                                             target[None].to(device), valid[None].to(device))
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1); scaler.step(optimizer); scaler.update()
                losses.append(float(loss.detach()))
            scheduler.step(); history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                "last_batch_components": components, "learning_rate": optimizer.param_groups[0]["lr"]})
            torch.save({"version": VERSION, "fold": fold_index, "epoch": epoch,
                "protocol_fingerprint": fingerprint, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "python_rng_state": rng.getstate(), "history": history}, last)
            atomic_json(folder / "history.json", history)
            print(f"{VERSION} fold {fold_index + 1}/4 epoch {epoch}/{epochs} loss={np.mean(losses):.4f}", flush=True)
        predictions = _predict_views(model, validation, supports, device)
        for identifier, modes in predictions.items():
            np.savez_compressed(oof / f"{identifier}.npz", lung=lungs[identifier].astype(np.uint8),
                                average=modes["average"], maximum=modes["maximum"], whole=modes["whole"], fold=fold_index)
        artifact = {"fold": fold_index, "training_groups": sorted({x["group"] for x in training}),
            "validation_groups": validation_groups, "final_epoch": epochs,
            "checkpoint": str(last.relative_to(PROJECT_ROOT)), "checkpoint_sha256": sha256(last),
            "initialized_from": str(source.relative_to(PROJECT_ROOT)), "initialized_from_sha256": sha256(source)}
        atomic_json(artifact_path, artifact); artifacts.append(artifact)
        del model, optimizer, scheduler, scaler, source_state
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    frozen = []
    for row in rows:
        with np.load(oof / f"{row['id']}.npz", allow_pickle=False) as values:
            frozen.append({k: values[k].copy() for k in ("average", "maximum", "whole")} |
                          {"lung": values["lung"].astype(bool)})
    candidates = []
    for mode in ("average", "maximum", "whole"):
        for threshold in protocol["lesion_thresholds"]:
            measured = [_measure(payloads[row["id"]], item["lung"],
                                 (item[mode] >= threshold) & item["lung"])
                        for row, item in zip(rows, frozen)]
            candidates.append({"mode": mode, "lesion_threshold": threshold, "summary": summarize(measured)})
    key = lambda x: (min(x["summary"]["pooled_lesion"][m] or 0 for m in ("precision", "recall", "specificity")),
                     x["summary"]["pooled_lesion"]["dice"] or 0)
    winner = max(candidates, key=key); dice_winner = max(candidates, key=lambda x:x["summary"]["pooled_lesion"]["dice"] or 0)
    metric = winner["summary"]["pooled_lesion"]
    checks = {"lesion_dice_at_least_0_80": metric["dice"] >= .80,
              "precision_at_least_0_80": metric["precision"] >= .80,
              "recall_at_least_0_80": metric["recall"] >= .80,
              "specificity_at_least_0_80": metric["specificity"] >= .80,
              "lung_dice_at_least_0_95": winner["summary"]["pooled_lung"]["dice"] >= .95}
    report = {"status":"done", "created_at":now(), "version":VERSION, "protocol":protocol,
        "protocol_fingerprint":fingerprint, "fold_artifacts":artifacts,
        "oof_balanced_winner":winner, "oof_dice_winner":dice_winner,
        "oof_gate":{"passed":all(checks.values()),"checks":checks},
        "candidate_count":len(candidates),"test_payload_opened":False}
    atomic_json(report_path,report)
    if report["oof_gate"]["passed"]:
        atomic_json(ARTIFACTS / "test_unlock.json", {"created_at":now(),"confirmation_gate_passed":True,
            "gate_source":VERSION,"crossval_fingerprint":fingerprint})
    return report
