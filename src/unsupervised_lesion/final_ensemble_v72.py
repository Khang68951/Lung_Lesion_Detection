from __future__ import annotations

import json

import numpy as np

from .evaluate import _measure, summarize
from .final_ensemble_v71 import _v70_root, refine_score
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256
from .reference_v2 import DevelopmentReferenceStore
from .style_ensemble_v46 import BASE_WEIGHTS, PARENTS, STYLE_VERSION, _root
from .supervised_segmentation import _prepare


VERSION = "v72_rebuilt_checkpoints_b2_tta_colour_region_gaussian_oof"
MODEL_WEIGHTS = {
    "v37": 0.08385571092367172,
    "v40": 0.0495622381567955,
    "v43": 0.034234367311000824,
    "v44": 0.09086206555366516,
    "v45": 0.07899594306945801,
    "v64_tta": 0.14674896001815796,
    "v67_tta": 0.10593080520629883,
    "v70_tta": 0.4098099172115326,
}
LESION_THRESHOLD = 0.4856509864330292


def calibrate_final_ensemble_v72() -> dict:
    roots = {version: _root(PARENTS[version], version) for version in PARENTS}
    v70_root = _v70_root()
    tta = ARTIFACTS / "tta_multiview_v68"
    tta_v70 = ARTIFACTS / "tta_v70"
    store = DevelopmentReferenceStore()
    items = [_prepare(store.load(row)) for row in store.rows]
    protocol = {
        "version": VERSION,
        "reference_fingerprint": store.manifest["fingerprint"],
        "model_weights": MODEL_WEIGHTS,
        "blend": "direct_nonnegative_eight_branch_oof_blend",
        "region_segments": 75,
        "region_compactness": 5.0,
        "region_weight": 0.49943276195599456,
        "gaussian_sigma": 1.0,
        "lesion_threshold": LESION_THRESHOLD,
        "selection": "maximize minimum of Dice/precision/recall/specificity on group-OOF planes after deterministic checkpoint rebuild",
        "source_reports": {
            version: {
                "path": str((root / "report.json").relative_to(PROJECT_ROOT)),
                "sha256": sha256(root / "report.json"),
            }
            for version, root in roots.items()
        } | {
            "v70": {
                "path": str((v70_root / "report.json").relative_to(PROJECT_ROOT)),
                "sha256": sha256(v70_root / "report.json"),
            }
        },
        "sealed_test_images_opened": 0,
    }
    fingerprint = json_hash(protocol)
    root = ARTIFACTS / "final_ensemble_v72" / fingerprint
    scores = root / "oof_scores"
    scores.mkdir(parents=True, exist_ok=True)
    rows = []
    score_hashes = {}
    for item in items:
        identifier = item["id"]
        direct = {}
        for key, version in zip(("v37", "v40", "v43", "v44"), BASE_WEIGHTS):
            with np.load(roots[version] / "oof" / f"{identifier}.npz", allow_pickle=False) as values:
                direct[key] = values["lesion"].astype(np.float32)
        with np.load(roots[STYLE_VERSION] / "oof" / f"{identifier}.npz", allow_pickle=False) as values:
            direct["v45"] = values["lesion"].astype(np.float32)
        with np.load(tta / "v64" / f"{identifier}.npz", allow_pickle=False) as values:
            v64 = values["average"].astype(np.float32)
            lung = values["lung"].astype(bool)
        with np.load(tta / "v67" / f"{identifier}.npz", allow_pickle=False) as values:
            v67 = values["average"].astype(np.float32)
        with np.load(tta_v70 / f"{identifier}.npz", allow_pickle=False) as values:
            v70 = values["lesion"].astype(np.float32)
        raw = sum(MODEL_WEIGHTS[key] * direct[key]
                  for key in ("v37", "v40", "v43", "v44", "v45"))
        raw += (MODEL_WEIGHTS["v64_tta"] * v64 + MODEL_WEIGHTS["v67_tta"] * v67
                + MODEL_WEIGHTS["v70_tta"] * v70)
        final = refine_score(item["payload"]["rgb"], lung, raw)
        output = scores / f"{identifier}.npz"
        np.savez_compressed(output, lesion=final, lung=lung.astype(np.uint8))
        score_hashes[identifier] = sha256(output)
        rows.append(_measure(item["payload"], lung, (final >= LESION_THRESHOLD) & lung))
    summary = summarize(rows)
    lesion = summary["pooled_lesion"]
    checks = {
        "lesion_dice_at_least_0_80": lesion["dice"] >= .80,
        "precision_at_least_0_80": lesion["precision"] >= .80,
        "recall_at_least_0_80": lesion["recall"] >= .80,
        "specificity_at_least_0_80": lesion["specificity"] >= .80,
        "lung_dice_at_least_0_95": summary["pooled_lung"]["dice"] >= .95,
    }
    report = {
        "status": "done", "created_at": now(), "version": VERSION,
        "protocol": protocol, "protocol_fingerprint": fingerprint,
        "winner": {"summary": summary, "lesion_threshold": LESION_THRESHOLD},
        "oof_score_sha256": score_hashes,
        "oof_gate": {"passed": all(checks.values()), "checks": checks},
        "test_payload_opened": False,
    }
    atomic_json(root / "report.json", report)
    if report["oof_gate"]["passed"]:
        candidate = {
            "status": "frozen_after_group_oof_gate", "created_at": now(),
            "version": VERSION, "selection_role": "four_fold_group_oof_ensemble",
            "lesion_postprocess": "final_ensemble_v72", "protocol": protocol,
            "protocol_fingerprint": fingerprint, "oof_summary": summary,
            "oof_gate": report["oof_gate"], "test_payload_opened": False,
        }
        candidate["candidate_fingerprint"] = json_hash(candidate)
        folder = ARTIFACTS / "frozen_candidate" / candidate["candidate_fingerprint"]
        atomic_json(folder / "candidate.json", candidate)
        atomic_json(ARTIFACTS / "frozen_candidate/current.json", {
            "path": str((folder / "candidate.json").relative_to(PROJECT_ROOT)),
            "sha256": sha256(folder / "candidate.json"),
            "candidate_fingerprint": candidate["candidate_fingerprint"],
        })
        atomic_json(ARTIFACTS / "test_unlock.json", {
            "created_at": now(), "confirmation_gate_passed": True,
            "gate_source": VERSION, "candidate_fingerprint": candidate["candidate_fingerprint"],
            "ensemble_fingerprint": fingerprint,
        })
        report["candidate"] = candidate
        atomic_json(root / "report.json", report)
    return report
