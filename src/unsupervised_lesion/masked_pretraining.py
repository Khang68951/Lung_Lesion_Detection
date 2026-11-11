from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from .data import load_audit, validate_fit_rows
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256, within_project
from .supervised_segmentation import _image_tensor, _letterbox


VERSION = "v39_unlabeled_masked_reconstruction_segformer_b1"
SEED = 3904
IMAGE_SIZE = 384
PATCH_SIZE = 32
MASK_RATIO = 0.50
EPOCHS = 3
BATCH_SIZE = 2
PRETRAINED = "nvidia/segformer-b1-finetuned-ade-512-512"
CLASSIFIER_KEYS = {"decode_head.classifier.weight", "decode_head.classifier.bias"}


def _state_fingerprint(state: dict[str, torch.Tensor], exclude_classifier: bool = False) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        if exclude_classifier and name in CLASSIFIER_KEYS:
            continue
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _spatial_augment(rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < .5:
        rgb = np.flip(rgb, 1).copy()
    if rng.random() < .5:
        rgb = np.flip(rgb, 0).copy()
    turns = rng.randrange(4)
    return np.rot90(rgb, turns).copy() if turns else rgb


def masked_reconstruction_example(rgb: np.ndarray, rng: random.Random
                                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized corrupted input, RGB target and an exact patch mask."""
    if rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Expected {(IMAGE_SIZE, IMAGE_SIZE, 3)}, got {rgb.shape}")
    rgb = _spatial_augment(rgb, rng)
    patches_per_side = IMAGE_SIZE // PATCH_SIZE
    patch_count = patches_per_side * patches_per_side
    selected = rng.sample(range(patch_count), int(round(patch_count * MASK_RATIO)))
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.float32)
    for index in selected:
        row, column = divmod(index, patches_per_side)
        y0, x0 = row * PATCH_SIZE, column * PATCH_SIZE
        mask[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE] = 1
    corrupted = rgb.copy()
    corrupted[mask > 0] = np.asarray([124, 116, 104], np.uint8)
    target = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255.0
    return _image_tensor(corrupted), target, torch.from_numpy(mask)


def _load_rgb(row: dict) -> np.ndarray:
    image = cv2.imread(str(within_project(row["path"])), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode fit image: {row['id']}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return _letterbox(rgb, cv2.INTER_AREA)[0]


def _loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    reconstruction = F.interpolate(logits, size=target.shape[-2:], mode="bilinear",
                                   align_corners=False).sigmoid()
    pixel_loss = F.smooth_l1_loss(reconstruction, target, reduction="none").mean(1)
    return (pixel_loss * mask).sum() / mask.sum().clamp_min(1)


def _checkpoint_payload(model, optimizer, scheduler, scaler, rng, epoch: int,
                        fingerprint: str, history: list[dict]) -> dict:
    payload = {
        "version": VERSION,
        "epoch": epoch,
        "model": model.state_dict(),
        "config": model.config.to_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "python_rng_state": rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "protocol_fingerprint": fingerprint,
        "history": history,
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def _restore_training_state(checkpoint: dict, model, optimizer, scheduler, scaler,
                            rng: random.Random) -> int:
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    rng.setstate(checkpoint["python_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    return int(checkpoint["epoch"])


def fit_masked_reconstruction(epochs: int = EPOCHS) -> dict:
    if epochs != EPOCHS:
        raise ValueError(f"V39 predeclared exactly {EPOCHS} epochs; received {epochs}")
    audit = load_audit()
    rows = list(audit["fit_rows"])
    validate_fit_rows(rows, audit)
    if len(rows) != 1764:
        raise ValueError(f"Expected all 1764 audited fit identities, found {len(rows)}")

    fit_manifest = {
        "version": VERSION,
        "audit_fingerprint": audit["audit_fingerprint"],
        "fit_images": len(rows),
        "protected_images": 0,
        "rows": [{key: row[key] for key in ("id", "path", "pixel_sha256", "file_sha256")}
                 for row in rows],
    }
    fit_manifest["fingerprint"] = json_hash(fit_manifest)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED, num_labels=3, ignore_mismatched_sizes=True, use_safetensors=True,
        cache_dir=ARTIFACTS / "model_cache",
        id2label={0: "red", 1: "green", 2: "blue"},
        label2id={"red": 0, "green": 1, "blue": 2})
    source_fingerprint = _state_fingerprint(model.state_dict(), exclude_classifier=True)
    protocol = {
        "version": VERSION, "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "pretrained": PRETRAINED, "source_non_classifier_state_fingerprint": source_fingerprint,
        "image_size": IMAGE_SIZE, "letterbox_fill": 0, "patch_size": PATCH_SIZE,
        "mask_ratio": MASK_RATIO, "masked_patches_per_image": 72,
        "mask_fill_rgb": [124, 116, 104], "spatial_augmentation": "paired_hv_flip_rot90",
        "objective": "masked_pixel_smooth_l1_sigmoid_rgb", "learning_rate": 3e-5,
        "weight_decay": 1e-3, "scheduler": "cosine_to_5e-7", "gradient_clip": 1.0,
        "fit_manifest_fingerprint": fit_manifest["fingerprint"],
        "audit_fingerprint": audit["audit_fingerprint"], "fit_images": len(rows),
        "reference_images_used": 0, "reference_planes_used": 0, "sealed_test_images_opened": 0,
    }
    fingerprint = json_hash(protocol)
    root = ARTIFACTS / "masked_pretraining" / fingerprint
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "fit_manifest.json", fit_manifest)
    atomic_json(root / "protocol.json", protocol)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=5e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = random.Random(SEED)
    last_path = root / "last.pt"
    history: list[dict] = []
    completed_epoch = 0
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("protocol_fingerprint") != fingerprint:
            raise ValueError("Existing V39 checkpoint protocol mismatch")
        history = checkpoint.get("history", [])
        completed_epoch = _restore_training_state(
            checkpoint, model, optimizer, scheduler, scaler, rng)
        print(f"{VERSION} resuming after complete epoch {completed_epoch}/{EPOCHS}", flush=True)

    for epoch in range(completed_epoch + 1, EPOCHS + 1):
        model.train()
        order = list(range(len(rows))); rng.shuffle(order)
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            samples = [masked_reconstruction_example(_load_rgb(rows[index]), rng)
                       for index in order[start:start + BATCH_SIZE]]
            image = torch.stack([sample[0] for sample in samples]).to(device)
            target = torch.stack([sample[1] for sample in samples]).to(device)
            mask = torch.stack([sample[2] for sample in samples]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = _loss(model(pixel_values=image).logits, target, mask)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "batches": len(losses), "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(record)
        checkpoint = _checkpoint_payload(model, optimizer, scheduler, scaler, rng,
                                         epoch, fingerprint, history)
        torch.save(checkpoint, last_path)
        atomic_json(root / "history.json", history)
        print(f"{VERSION} epoch {epoch}/{EPOCHS} loss={record['train_loss']:.6f}", flush=True)

    checkpoint_sha = sha256(last_path)
    artifact = {
        "status": "fit_complete", "created_at": now(), "version": VERSION,
        "artifact_fingerprint": fingerprint, "protocol": protocol,
        "fit_manifest": str((root / "fit_manifest.json").relative_to(PROJECT_ROOT)),
        "fit_manifest_sha256": sha256(root / "fit_manifest.json"),
        "checkpoint": str(last_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": checkpoint_sha, "completed_epochs": EPOCHS,
        "learned_non_classifier_state_fingerprint": _state_fingerprint(
            model.state_dict(), exclude_classifier=True),
        "reference_images_used": 0, "reference_planes_used": 0,
        "test_payload_opened": False, "history": history,
    }
    atomic_json(root / "artifact.json", artifact)
    return artifact


def find_completed_pretraining() -> dict:
    matches = []
    for path in (ARTIFACTS / "masked_pretraining").glob("*/artifact.json"):
        artifact = json.loads(path.read_text())
        if artifact.get("version") == VERSION and artifact.get("status") == "fit_complete":
            matches.append(artifact)
    if len(matches) != 1:
        raise ValueError(f"Expected one completed {VERSION} artifact, found {len(matches)}")
    checkpoint = within_project(matches[0]["checkpoint"])
    if sha256(checkpoint) != matches[0]["checkpoint_sha256"]:
        raise ValueError("V39 pretraining checkpoint changed")
    return matches[0]


def load_domain_weights(model: torch.nn.Module, checkpoint_path: str | Path,
                        checkpoint_sha256: str) -> dict:
    path = within_project(checkpoint_path)
    if sha256(path) != checkpoint_sha256:
        raise ValueError("V39 pretraining checkpoint checksum mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("version") != VERSION or checkpoint.get("epoch") != EPOCHS:
        raise ValueError("V39 pretraining checkpoint is not complete")
    destination = model.state_dict(); transferred = {}
    for name, tensor in checkpoint["model"].items():
        if name not in CLASSIFIER_KEYS and name in destination and destination[name].shape == tensor.shape:
            transferred[name] = tensor
    if len(transferred) < 100:
        raise ValueError(f"Only {len(transferred)} domain-pretrained tensors were compatible")
    missing, unexpected = model.load_state_dict(transferred, strict=False)
    if unexpected or not CLASSIFIER_KEYS.issubset(set(missing)):
        raise ValueError("Unexpected V39 transfer state")
    return {"transferred_tensors": len(transferred),
            "excluded_classifier_keys": sorted(CLASSIFIER_KEYS),
            "checkpoint_sha256": checkpoint_sha256}
