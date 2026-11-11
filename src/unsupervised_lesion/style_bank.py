from __future__ import annotations

import json
import random

import cv2
import numpy as np

from .data import load_audit, validate_fit_rows
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256, within_project


VERSION = "v45_unmarked_robust_lab_style_bank"


def _stats(rgb: np.ndarray) -> tuple[list[float], list[float]]:
    mask = rgb.astype(np.int16).sum(axis=2) > 30
    if int(mask.sum()) < 64:
        mask[:] = True
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    median = []; span = []
    for channel in range(3):
        values = lab[:, :, channel][mask]
        p10, p50, p90 = np.percentile(values, [10, 50, 90])
        median.append(float(p50)); span.append(float(max(8.0, p90 - p10)))
    return median, span


def build_style_bank() -> dict:
    audit = load_audit(); rows = list(audit["fit_rows"]); validate_fit_rows(rows, audit)
    if len(rows) != 1764:
        raise ValueError(f"Expected 1764 unmarked identities, found {len(rows)}")
    protocol = {"version": VERSION, "audit_fingerprint": audit["audit_fingerprint"],
                "fit_images": len(rows), "resize_maximum_side": 256,
                "pixel_policy": "rgb_sum_greater_than_30", "statistics": "lab_median_p10_p90_span",
                "reference_images_used": 0, "reference_planes_used": 0,
                "source_rows_fingerprint": json_hash([{k: row[k] for k in
                    ("id", "pixel_sha256", "file_sha256")} for row in rows])}
    fingerprint = json_hash(protocol); root = ARTIFACTS / "style_bank" / fingerprint
    artifact_path = root / "artifact.json"
    if artifact_path.exists():
        artifact = json.loads(artifact_path.read_text())
        if artifact.get("artifact_fingerprint") != fingerprint:
            raise ValueError("Existing V45 style bank fingerprint mismatch")
        return artifact
    records = []
    for index, row in enumerate(rows, start=1):
        bgr = cv2.imread(str(within_project(row["path"])), cv2.IMREAD_COLOR)
        if bgr is None: raise ValueError(f"Cannot decode style image: {row['id']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); height, width = rgb.shape[:2]
        scale = min(1.0, 256 / max(height, width))
        if scale < 1:
            rgb = cv2.resize(rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        median, span = _stats(rgb)
        records.append({"id": row["id"], "pixel_sha256": row["pixel_sha256"],
                        "file_sha256": row["file_sha256"], "lab_median": median,
                        "lab_p10_p90_span": span})
        if index % 200 == 0: print(f"{VERSION} {index}/{len(rows)}", flush=True)
    artifact = {"status": "done", "created_at": now(), "version": VERSION,
                "artifact_fingerprint": fingerprint, "protocol": protocol,
                "records": records, "reference_images_used": 0, "reference_planes_used": 0,
                "test_payload_opened": False}
    atomic_json(artifact_path, artifact); artifact["artifact_path"] = str(artifact_path.relative_to(PROJECT_ROOT))
    artifact["artifact_sha256"] = sha256(artifact_path)
    atomic_json(root / "pointer.json", {"artifact_path": artifact["artifact_path"],
                                        "artifact_sha256": artifact["artifact_sha256"]})
    return artifact


def apply_unmarked_style(rgb: np.ndarray, donor: dict) -> np.ndarray:
    tissue = rgb.astype(np.int16).sum(axis=2) > 30
    if int(tissue.sum()) < 64:
        return rgb.copy()
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    source_median, source_span = _stats(rgb)
    donor_median = donor["lab_median"]; donor_span = donor["lab_p10_p90_span"]
    transformed = lab.copy()
    for channel in range(3):
        transformed[:, :, channel] = ((lab[:, :, channel] - source_median[channel])
                                      * donor_span[channel] / max(source_span[channel], 8.0)
                                      + donor_median[channel])
    transformed = np.clip(np.round(transformed), 0, 255).astype(np.uint8)
    output = cv2.cvtColor(transformed, cv2.COLOR_LAB2RGB); output[~tissue] = rgb[~tissue]
    return output


def maybe_apply_unmarked_style(rgb: np.ndarray, records: list[dict],
                               rng: random.Random, probability: float = .50) -> np.ndarray:
    if not records or rng.random() >= probability:
        return rgb.copy()
    return apply_unmarked_style(rgb, records[rng.randrange(len(records))])
