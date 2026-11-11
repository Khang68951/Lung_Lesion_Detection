from __future__ import annotations

import json
import random
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .data import load_audit, validate_fit_rows
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256, within_project
from .segmentation import LesionConfig, foreground_mask, lesion_score, robust_features


FAMILIES = [
    {"name": "global_dark", "dark_weight": 1.0, "local_dark_weight": 0.0,
     "a_weight": 0.0, "b_weight": 0.0, "saturation_weight": 0.0},
    {"name": "global_local", "dark_weight": 1.0, "local_dark_weight": 0.5,
     "a_weight": 0.0, "b_weight": 0.0, "saturation_weight": 0.0},
    {"name": "local_strong", "dark_weight": 1.0, "local_dark_weight": 1.0,
     "a_weight": 0.0, "b_weight": 0.0, "saturation_weight": 0.0},
    {"name": "purple_dark", "dark_weight": 1.0, "local_dark_weight": 0.5,
     "a_weight": 0.2, "b_weight": -0.1, "saturation_weight": 0.0},
    {"name": "saturated_dark", "dark_weight": 1.0, "local_dark_weight": 0.5,
     "a_weight": 0.15, "b_weight": -0.05, "saturation_weight": 0.15},
    {"name": "local_purple", "dark_weight": 0.75, "local_dark_weight": 1.0,
     "a_weight": 0.2, "b_weight": -0.1, "saturation_weight": 0.1},
    {"name": "absolute_dark", "dark_weight": 0.0, "local_dark_weight": 0.0,
     "a_weight": 0.0, "b_weight": 0.0, "saturation_weight": 0.0,
     "absolute_dark_weight": 1.0},
    {"name": "absolute_purple", "dark_weight": 0.0, "local_dark_weight": 0.0,
     "a_weight": 0.0, "b_weight": 0.0, "saturation_weight": 0.0,
     "absolute_dark_weight": 1.0, "absolute_red_weight": 0.25,
     "absolute_purple_weight": 0.20},
    {"name": "relative_absolute", "dark_weight": 0.75, "local_dark_weight": 0.35,
     "a_weight": 0.1, "b_weight": -0.05, "saturation_weight": 0.0,
     "absolute_dark_weight": 0.5, "absolute_red_weight": 0.1,
     "absolute_purple_weight": 0.1},
]
QUANTILES = [0.30, 0.40, 0.50, 0.60, 0.70, 0.78, 0.84, 0.88, 0.92, 0.95]


def _resize_for_fit(rgb: np.ndarray, maximum: int = 384) -> np.ndarray:
    height, width = rgb.shape[:2]
    factor = min(1.0, maximum / max(height, width))
    if factor == 1.0:
        return rgb
    return cv2.resize(rgb, (round(width * factor), round(height * factor)), interpolation=cv2.INTER_AREA)


def fit_appearance(max_images: int | None = None, seed: int = 73) -> dict:
    audit = load_audit()
    rows = list(audit["fit_rows"])
    if max_images is not None:
        rows = rows[:max_images]
    validate_fit_rows(rows, audit)
    rng = np.random.default_rng(seed)
    samples = {family["name"]: [] for family in FAMILIES}
    image_rows = []
    for index, row in enumerate(rows, start=1):
        path = within_project(row["path"])
        rgb = np.asarray(Image.open(path).convert("RGB"))
        rgb = _resize_for_fit(rgb)
        lung = foreground_mask(rgb)
        if int(lung.sum()) < max(100, round(lung.size * 0.01)):
            image_rows.append({"id": row["id"], "status": "no_foreground", "pixels": int(lung.sum())})
            continue
        features = robust_features(rgb, lung)
        indices = np.flatnonzero(lung)
        if len(indices) > 1024:
            indices = rng.choice(indices, 1024, replace=False)
        for family in FAMILIES:
            config = LesionConfig(**{key: value for key, value in family.items() if key != "name"})
            values = lesion_score(features, config).ravel()[indices]
            samples[family["name"]].append(values.astype(np.float32))
        image_rows.append({"id": row["id"], "status": "used", "pixels": int(lung.sum()),
                           "sampled_pixels": int(len(indices))})
        if index % 100 == 0:
            print(f"fit appearance: {index}/{len(rows)}", flush=True)

    distributions = {}
    candidates = []
    for family in FAMILIES:
        name = family["name"]
        values = np.concatenate(samples[name]) if samples[name] else np.empty(0, np.float32)
        if not len(values):
            raise RuntimeError(f"No foreground samples for {name}")
        thresholds = {str(q): float(np.quantile(values, q)) for q in QUANTILES}
        distributions[name] = {
            "sample_pixels": int(len(values)),
            "mean": float(values.mean()), "std": float(values.std()),
            "quantiles": thresholds,
        }
        for q in QUANTILES:
            for minimum_dark in (-0.25, 0.0):
                for minimum_component in (0.0002, 0.0008):
                    config = LesionConfig(
                        threshold=thresholds[str(q)],
                        minimum_dark=minimum_dark,
                        minimum_component_fraction=minimum_component,
                        **{key: value for key, value in family.items() if key != "name"},
                    )
                    candidates.append({"family": name, "fit_quantile": q, "config": config.to_dict()})
    model_id = f"appearance_{uuid.uuid4().hex[:16]}"
    artifact = {
        "status": "fit_complete", "model_id": model_id, "created_at": now(),
        "method": "Per-image robust Lab/HSV/local-dark features; candidate thresholds are quantiles fitted only on unmarked RGB identities.",
        "seed": seed, "fit_images_requested": len(rows),
        "fit_images_used": sum(row["status"] == "used" for row in image_rows),
        "fit_images_failed": sum(row["status"] != "used" for row in image_rows),
        "fit_identity_hash": json_hash([{key: row[key] for key in ("id", "pixel_sha256", "file_sha256")} for row in rows]),
        "data_audit_fingerprint": audit["audit_fingerprint"],
        "data_audit_sha256": sha256(ARTIFACTS / "data_audit.json"),
        "reference_pixels_used_for_fit": 0,
        "distributions": distributions,
        "candidates": candidates,
        "image_audit": image_rows,
        "limitations": ["Unmarked images are not assumed healthy.",
                        "Dark-cluster lesion semantics are a declared rule, not learned disease labels.",
                        "Marked references select among frozen candidates only after fit."],
    }
    artifact["artifact_fingerprint"] = json_hash(artifact)
    folder = ARTIFACTS / "appearance_model" / model_id
    atomic_json(folder / "model.json", artifact)
    atomic_json(ARTIFACTS / "appearance_model/current.json",
                {"model_id": model_id, "path": str((folder / "model.json").relative_to(PROJECT_ROOT)),
                 "sha256": sha256(folder / "model.json")})
    return artifact


def load_model(path: Path | None = None) -> dict:
    if path is None:
        pointer = json.loads((ARTIFACTS / "appearance_model/current.json").read_text())
        path = within_project(pointer["path"])
        if sha256(path) != pointer["sha256"]:
            raise ValueError("Current appearance model checksum changed")
    model = json.loads(Path(path).read_text())
    expected = model.pop("artifact_fingerprint")
    actual = json_hash(model)
    model["artifact_fingerprint"] = expected
    if expected != actual:
        raise ValueError("Appearance model fingerprint changed")
    return model

