from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_ROOT = PROJECT_ROOT / "Unsupervised_Lesion_Workflow"
RUNS = WORKFLOW_ROOT / "runs"
ARTIFACTS = WORKFLOW_ROOT / "artifacts"
SAM2_VENDOR = WORKFLOW_ROOT / "vendor" / "sam2"
SAM2_CHECKPOINT = WORKFLOW_ROOT / "models" / "sam2.1_hiera_tiny.pt"
LUNG_UNET_CHECKPOINT = WORKFLOW_ROOT / "models" / "unmarked_lung_unet_best.pt"
DINO_ASSET_ROOT = WORKFLOW_ROOT / "models" / "dinov2_with_registers_small"


def within_project(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    candidate.relative_to(PROJECT_ROOT.resolve())
    return candidate


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()


def atomic_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

