from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .io import ARTIFACTS, json_hash, sha256, within_project


MANIFEST = ARTIFACTS / "references_v2/manifest.json"


def verify_manifest(manifest: dict) -> None:
    expected = manifest.get("fingerprint")
    actual = json_hash({key: value for key, value in manifest.items() if key != "fingerprint"})
    if expected != actual:
        raise ValueError("Versioned development reference fingerprint changed")
    if manifest.get("test_payload_opened") is not False or manifest.get("roles") != ["calibration", "confirmation"]:
        raise ValueError("Versioned reference must remain development-only")


class DevelopmentReferenceStore:
    def __init__(self, manifest_path: Path = MANIFEST):
        self.manifest_path = within_project(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        verify_manifest(self.manifest)
        self.rows = self.manifest["records"]

    def load(self, row: dict) -> dict:
        masks = {}
        for name in ("lung", "lesion", "valid", "ignore_lung", "ignore_lesion"):
            path = within_project(row["files"][name])
            if sha256(path) != row["sha256"][name]:
                raise ValueError(f"Versioned {name} changed for {row['id']}")
            masks[name] = np.asarray(Image.open(path).convert("L")) > 0
        image_path = within_project(row["files"]["image"])
        if sha256(image_path) != row["sha256"]["image"]:
            raise ValueError(f"Versioned image changed for {row['id']}")
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        if any(mask.shape != rgb.shape[:2] for mask in masks.values()):
            raise ValueError(f"Versioned reference dimensions differ for {row['id']}")
        return {"row": row, "rgb": rgb, **masks}

