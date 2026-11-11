from __future__ import annotations

import json
import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable

from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256, within_project


DATASET_MANIFEST = PROJECT_ROOT / "Dataset/manifest.json"
REFERENCE_MANIFEST = PROJECT_ROOT / "Legacy_Project/outputs/pdf_retraining_20261020/dataset.json"
AUDIT_PATH = ARTIFACTS / "data_audit.json"


def _audit_fingerprint(audit: dict) -> str:
    return json_hash({key: value for key, value in audit.items()
                      if key not in {"audit_fingerprint", "created_at"}})


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_reference_manifest(manifest: dict) -> None:
    payload = {key: value for key, value in manifest.items() if key != "fingerprint"}
    # Preserve the historical manifest's original canonicalization (default JSON
    # separators, sorted keys) rather than rewriting its fingerprint convention.
    historical = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if historical != manifest.get("fingerprint"):
        raise ValueError("Frozen reference manifest fingerprint changed")


def reference_rows(manifest: dict, role: str) -> list[dict]:
    """Return metadata only. Payload access is handled by ReferenceStore."""
    verify_reference_manifest(manifest)
    split = {"calibration": "train", "confirmation": "val", "test": "test"}[role]
    # Historical rows contain only lung labels and are not lesion references.
    return [row for row in manifest["records"]
            if row["split"] == split and not row["id"].startswith("historical_")]


def build_audit() -> dict:
    source = _read_json(DATASET_MANIFEST)
    references = _read_json(REFERENCE_MANIFEST)
    verify_reference_manifest(references)
    records = source["records"]
    marked = [row for row in records if row.get("marked_images")]
    protected_rgb = {row["pixel_sha256"] for row in marked}

    # One observation per exact RGB identity. Marked aliases always win exclusion.
    by_rgb: dict[str, list[dict]] = {}
    for row in records:
        by_rgb.setdefault(row["pixel_sha256"], []).append(row)
    fit_rows = []
    excluded = []
    for identity, aliases in sorted(by_rgb.items()):
        reasons = []
        if identity in protected_rgb or any(row.get("marked_images") for row in aliases):
            reasons.append("marked_or_exact_rgb_alias")
        if any(row.get("legacy_excluded_training_alias") for row in aliases):
            reasons.append("conservative_pdf_source_alias")
        if reasons:
            excluded.append({"pixel_sha256": identity, "ids": [row["id"] for row in aliases],
                             "reasons": reasons})
            continue
        # Prefer the old-dataset occurrence to a PDF occurrence where both exist.
        row = sorted(aliases, key=lambda item: (item["source_kind"] != "old_dataset", item["id"]))[0]
        fit_rows.append({"id": row["id"], "path": row["path"],
                         "pixel_sha256": identity, "file_sha256": row["sha256"],
                         "source_kind": row["source_kind"],
                         "source_group": row.get("legacy_source_group"),
                         "aliases": sorted(alias["id"] for alias in aliases)})

    roles = {role: reference_rows(references, role)
             for role in ("calibration", "confirmation", "test")}
    role_ids = {role: [row["id"] for row in rows] for role, rows in roles.items()}
    if set(role_ids["calibration"]) & set(role_ids["confirmation"] + role_ids["test"]):
        raise ValueError("Reference IDs overlap roles")

    audit = {
        "status": "done",
        "created_at": now(),
        "dataset_manifest": str(DATASET_MANIFEST.relative_to(PROJECT_ROOT)),
        "dataset_manifest_sha256": sha256(DATASET_MANIFEST),
        "reference_manifest": str(REFERENCE_MANIFEST.relative_to(PROJECT_ROOT)),
        "reference_manifest_sha256": sha256(REFERENCE_MANIFEST),
        "source_records": len(records),
        "source_unique_rgb": len(by_rgb),
        "reference_only_unique_rgb": len(protected_rgb),
        "fit_unique_rgb": len(fit_rows),
        "fit_source_kinds": dict(Counter(row["source_kind"] for row in fit_rows)),
        "reference_roles": {role: {"images": len(rows),
                                     "groups": len({row["group"] for row in rows}),
                                     "ids": role_ids[role]}
                            for role, rows in roles.items()},
        "fit_rows": fit_rows,
        "excluded_reference_aliases": excluded,
        "test_payload_opened": False,
        "policy": {
            "marked_and_exact_rgb_aliases_never_fit": True,
            "conservative_pdf_source_aliases_never_fit": True,
            "calibration_is_historical_train_repurposed_as_validation": True,
            "confirmation_is_historical_validation": True,
            "test_requires_unlock": True,
            "unmarked_does_not_mean_healthy": True,
        },
    }
    audit["audit_fingerprint"] = _audit_fingerprint(audit)
    atomic_json(AUDIT_PATH, audit)
    return audit


def load_audit() -> dict:
    audit = _read_json(AUDIT_PATH) if AUDIT_PATH.exists() else build_audit()
    expected = audit["audit_fingerprint"]
    actual = _audit_fingerprint(audit)
    if expected != actual:
        raise ValueError("Data audit fingerprint changed")
    return audit


def validate_fit_rows(rows: Iterable[dict], audit: dict) -> None:
    protected = {row["pixel_sha256"] for row in audit["excluded_reference_aliases"]}
    seen = set()
    for row in rows:
        if row["pixel_sha256"] in protected:
            raise ValueError(f"Reference identity entered fit: {row['id']}")
        if row["pixel_sha256"] in seen:
            raise ValueError(f"Duplicate RGB identity entered fit: {row['id']}")
        seen.add(row["pixel_sha256"])
        path = within_project(row["path"])
        if sha256(path) != row["file_sha256"]:
            raise ValueError(f"Fit source changed: {row['id']}")

