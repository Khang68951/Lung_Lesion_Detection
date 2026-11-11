from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import REFERENCE_MANIFEST, reference_rows, verify_reference_manifest
from .io import ARTIFACTS, PROJECT_ROOT, json_hash, sha256, within_project


def _reference_path(value: str) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        if raw.exists():
            return within_project(raw)
        # The frozen manifest predates the repository relocation and contains an
        # absolute project-root /outputs path. Preserve only its suffix.
        if "outputs" in raw.parts:
            suffix = Path(*raw.parts[raw.parts.index("outputs"):])
            relocated = PROJECT_ROOT / "Legacy_Project" / suffix
            if relocated.exists():
                return within_project(relocated)
        raise FileNotFoundError(f"Frozen absolute reference cannot be relocated: {value}")
    direct = PROJECT_ROOT / raw
    legacy = PROJECT_ROOT / "Legacy_Project" / raw
    if direct.exists():
        return within_project(direct)
    if legacy.exists():
        return within_project(legacy)
    raise FileNotFoundError(f"Frozen reference path cannot be relocated: {value}")


class ReferenceStore:
    def __init__(self, role: str, frozen_candidate: dict | None = None):
        if role == "test":
            self._verify_test_unlock(frozen_candidate)
        self.role = role
        self.manifest = json.loads(REFERENCE_MANIFEST.read_text())
        verify_reference_manifest(self.manifest)
        self.rows = reference_rows(self.manifest, role)

    @staticmethod
    def _verify_test_unlock(candidate: dict | None) -> None:
        path = ARTIFACTS / "test_unlock.json"
        if not path.exists() or candidate is None:
            raise PermissionError("Sealed test payload requires a frozen passing confirmation")
        unlock = json.loads(path.read_text())
        if not unlock.get("confirmation_gate_passed"):
            raise PermissionError("Confirmation gate did not pass")
        if unlock.get("candidate_fingerprint") != candidate.get("candidate_fingerprint"):
            raise PermissionError("Test unlock does not match frozen candidate")

    def load(self, row: dict) -> dict:
        masks = {}
        for name in ("lung", "lesion", "valid", "ignore_lung", "ignore_lesion"):
            path = _reference_path(row["files"][name])
            if sha256(path) != row["sha256"][name]:
                raise ValueError(f"Frozen {name} reference changed for {row['id']}")
            masks[name] = np.asarray(Image.open(path).convert("L")) > 0
        image_path = _reference_path(row["files"]["image"])
        if sha256(image_path) != row["sha256"]["image"]:
            raise ValueError(f"Frozen image changed for {row['id']}")
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        if any(mask.shape != rgb.shape[:2] for mask in masks.values()):
            raise ValueError(f"Reference dimensions differ for {row['id']}")
        return {"row": row, "rgb": rgb, **masks}

    def load_rgb(self, row: dict) -> np.ndarray:
        """Load and verify clean RGB only, without resolving any reference plane."""
        image_path = _reference_path(row["files"]["image"])
        if sha256(image_path) != row["sha256"]["image"]:
            raise ValueError(f"Frozen image changed for {row['id']}")
        return np.asarray(Image.open(image_path).convert("RGB"))

