from __future__ import annotations

import gc
import json
import random
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw
from skimage.color import rgb2lab
from sklearn.cluster import KMeans

from ..data import load_audit, validate_fit_rows
from ..io import ARTIFACTS, PROJECT_ROOT, RUNS, atomic_json, json_hash, now, sha256, within_project
from ..segmentation import color_regions, unlabeled_lung_probability


VERSION = "v15_color_region_zero_shot_teacher_pilot"
SEED = 73
INSTRUCTION = """You inspect clean RGB photographs of gross pig lungs. The first
image is the clean whole-lung context. The second image is one target tissue region
from that same unannotated photograph; pixels outside the target region are dimmed.
Classify only the natural-color target tissue. A = visibly abnormal lesion-like
lung tissue; B = visibly normal/non-lesion lung tissue; C = mixed, ambiguous,
artifact, background, cut edge, vessel-only, or cannot decide. Dark or purple
color alone is not proof of lesion: use texture and context too. Do not estimate
severity and do not make a disease diagnosis. Return exactly one letter: A, B or C."""


def _qwen_provenance() -> tuple[dict, Path]:
    path = PROJECT_ROOT / "Legacy_Project/outputs/teacher_weights_20261021/provenance.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("repository") != "Qwen/Qwen3-VL-4B-Instruct":
        raise ValueError("Unexpected Qwen repository")
    if record.get("revision") != "ebb281ec70b05090aa6165b016eac8ec08e71b17":
        raise ValueError("Unexpected Qwen revision")
    directory = Path(record["directory"])
    if not directory.is_dir():
        directory = (PROJECT_ROOT / "Legacy_Project/outputs/teacher_weights_20261021/cache"
                     / "models--Qwen--Qwen3-VL-4B-Instruct/snapshots" / record["revision"])
    directory = directory.resolve()
    directory.relative_to(PROJECT_ROOT.resolve())
    if not directory.is_dir():
        raise FileNotFoundError("Pinned local Qwen snapshot is missing")
    for name, digest in record.get("files_sha256", {}).items():
        candidate = directory / name
        if not candidate.is_file() or sha256(candidate) != digest:
            raise ValueError(f"Pinned Qwen artifact changed: {name}")
    return record, directory


def select_pilot_rows(rows: list[dict], maximum_images: int, seed: int = SEED) -> list[dict]:
    """Choose at most one identity per acquisition group without lesion information."""
    if maximum_images < 1:
        raise ValueError("maximum_images must be positive")
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        key = row.get("source_group") or f"ungrouped:{row['id']}"
        grouped.setdefault(key, []).append(row)
    rng = random.Random(seed)
    keys = sorted(grouped)
    rng.shuffle(keys)
    selected = []
    for key in keys[:maximum_images]:
        options = sorted(grouped[key], key=lambda item: (item["pixel_sha256"], item["id"]))
        selected.append(options[0])
    return selected


def region_statistics(rgb: np.ndarray, lung: np.ndarray, labels: np.ndarray) -> list[dict]:
    lab = rgb2lab(np.asarray(rgb, dtype=np.uint8))
    result = []
    lung_pixels = max(1, int(lung.sum()))
    for region_id in np.unique(labels[lung]):
        if region_id <= 0:
            continue
        mask = (labels == region_id) & lung
        area = int(mask.sum())
        if area < max(24, round(lung_pixels * 0.0015)):
            continue
        ys, xs = np.where(mask)
        values = lab[mask]
        result.append({
            "region_id": int(region_id), "area": area,
            "area_fraction": float(area / lung_pixels),
            "mean_lab": values.mean(axis=0).tolist(),
            "std_lab": values.std(axis=0).tolist(),
            "box": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
        })
    return result


def select_diverse_regions(regions: list[dict], count: int, seed: int = SEED) -> list[dict]:
    if count < 1 or not regions:
        return []
    if len(regions) <= count:
        return sorted(regions, key=lambda item: item["region_id"])
    matrix = np.asarray([item["mean_lab"] + item["std_lab"] for item in regions], np.float64)
    scale = np.maximum(np.std(matrix, axis=0), 1.0)
    matrix = (matrix - np.mean(matrix, axis=0)) / scale
    weights = np.asarray([item["area"] for item in regions], np.float64)
    model = KMeans(n_clusters=count, random_state=seed, n_init=20, max_iter=300)
    model.fit(matrix, sample_weight=weights)
    chosen = []
    used: set[int] = set()
    for center in model.cluster_centers_:
        order = np.argsort(np.sum((matrix - center) ** 2, axis=1))
        index = next(int(value) for value in order if int(value) not in used)
        used.add(index)
        chosen.append(regions[index])
    return sorted(chosen, key=lambda item: item["region_id"])


def make_region_crop(rgb: np.ndarray, labels: np.ndarray, region: dict,
                     padding_fraction: float = 0.55) -> Image.Image:
    x0, y0, x1, y1 = region["box"]
    width, height = x1 - x0, y1 - y0
    padding = max(8, round(max(width, height) * padding_fraction))
    left, top = max(0, x0 - padding), max(0, y0 - padding)
    right, bottom = min(rgb.shape[1], x1 + padding), min(rgb.shape[0], y1 + padding)
    crop = np.asarray(rgb[top:bottom, left:right], dtype=np.uint8).copy()
    target = labels[top:bottom, left:right] == region["region_id"]
    dimmed = np.clip(crop.astype(np.float32) * 0.16, 0, 255).astype(np.uint8)
    dimmed[target] = crop[target]
    return Image.fromarray(dimmed)


def pilot_gate(records: list[dict], minimum_each_class: int = 4,
               minimum_coverage: float = 0.25) -> dict:
    labels = [item["accepted_label"] for item in records]
    counts = {name: labels.count(name) for name in ("lesion", "normal", "uncertain")}
    accepted = counts["lesion"] + counts["normal"]
    coverage = accepted / max(1, len(records))
    passed = (counts["lesion"] >= minimum_each_class
              and counts["normal"] >= minimum_each_class
              and coverage >= minimum_coverage)
    return {"passed": bool(passed), "counts": counts, "coverage": float(coverage),
            "minimum_each_class": minimum_each_class, "minimum_coverage": minimum_coverage}


def region_probability_map(labels: np.ndarray, region_ids: list[int],
                           probabilities: list[float]) -> np.ndarray:
    if len(region_ids) != len(probabilities) or labels.ndim != 2:
        raise ValueError("Region probability geometry mismatch")
    lookup = np.zeros(int(labels.max()) + 1, np.float32)
    for region_id, probability in zip(region_ids, probabilities):
        if not 0 < int(region_id) < len(lookup) or not 0.0 <= float(probability) <= 1.0:
            raise ValueError("Invalid region probability")
        lookup[int(region_id)] = float(probability)
    return lookup[labels]


class ZeroShotRegionTeacher:
    def __init__(self, directory: Path):
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen color-region teacher requires CUDA")
        torch.manual_seed(SEED)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        self.processor = AutoProcessor.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False,
            min_pixels=96 * 28 * 28, max_pixels=448 * 28 * 28)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            directory, quantization_config=quantization, torch_dtype=torch.float16,
            device_map={"": 0}, attn_implementation="sdpa", trust_remote_code=False,
            local_files_only=True).eval()
        token_ids = [self.processor.tokenizer.encode(value, add_special_tokens=False)
                     for value in ("A", "B", "C")]
        if any(len(value) != 1 for value in token_ids):
            raise ValueError("A/B/C must each be a single Qwen token")
        self.answer_ids = [value[0] for value in token_ids]

    @torch.inference_mode()
    def classify(self, context: Image.Image, crop: Image.Image, flipped: bool = False) -> dict:
        context = context.copy(); crop = crop.copy()
        context.thumbnail((640, 640), Image.Resampling.LANCZOS)
        crop.thumbnail((320, 320), Image.Resampling.LANCZOS)
        if flipped:
            context, crop = ImageOps.mirror(context), ImageOps.mirror(crop)
        content = [
            {"type": "text", "text": INSTRUCTION},
            {"type": "text", "text": "WHOLE-LUNG CONTEXT (clean, unannotated):"},
            {"type": "image", "image": context},
            {"type": "text", "text": "TARGET COLOR/TISSUE REGION (outside dimmed): A, B or C?"},
            {"type": "image", "image": crop},
        ]
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")
        started = time.monotonic()
        logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1]
        probability = logits[self.answer_ids].float().softmax(0).cpu().numpy()
        index = int(np.argmax(probability))
        return {"probabilities": probability.tolist(), "raw_label": ("lesion", "normal", "uncertain")[index],
                "confidence": float(probability[index]), "seconds": time.monotonic() - started,
                "input_tokens": int(inputs["input_ids"].shape[1])}

    def close(self) -> None:
        del self.model, self.processor
        gc.collect()
        torch.cuda.empty_cache()


def _accepted(first: dict, flipped: dict, threshold: float) -> str:
    if (first["raw_label"] == flipped["raw_label"]
            and first["raw_label"] != "uncertain"
            and min(first["confidence"], flipped["confidence"]) >= threshold):
        return first["raw_label"]
    return "uncertain"


def _save_preview(path: Path, records: list[dict]) -> None:
    tiles = []
    for record in records:
        image = Image.open(within_project(record["crop_path"])).convert("RGB")
        image.thumbnail((180, 150), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (190, 185), "white")
        tile.paste(image, ((190 - image.width) // 2, 2))
        ImageDraw.Draw(tile).text((5, 156), f"{record['image_id']} R{record['region_id']}\n{record['accepted_label']}", fill="black")
        tiles.append(tile)
    columns = 6
    canvas = Image.new("RGB", (columns * 190, ((len(tiles) + columns - 1) // columns) * 185), "white")
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % columns) * 190, (index // columns) * 185))
    canvas.save(path, quality=92)


def finalize_pilot(run_dir: Path) -> dict:
    """Finalize a fully cached teacher run without repeating GPU inference."""
    run_dir = Path(run_dir).resolve()
    run_dir.relative_to(RUNS.resolve())
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    records = json.loads((run_dir / "records.json").read_text(encoding="utf-8"))
    expected = protocol["maximum_images"] * protocol["regions_per_image"]
    if len(records) != expected:
        raise ValueError(f"Pilot cache incomplete: {len(records)}/{expected}")
    provenance, _ = _qwen_provenance()
    gate = pilot_gate(records)
    preview = run_dir / "preview.jpg"
    _save_preview(preview, records)
    report = {
        "status": "done", "created_at": now(), "run_id": run_dir.name, "version": VERSION,
        "protocol": protocol, "protocol_fingerprint": json_hash(protocol),
        "qwen_repository": provenance["repository"], "qwen_revision": provenance["revision"],
        "qwen_provenance_sha256": sha256(PROJECT_ROOT / "Legacy_Project/outputs/teacher_weights_20261021/provenance.json"),
        "images": protocol["maximum_images"], "regions": len(records), "gate": gate,
        "reference_pixels_used_for_fit": 0, "marked_images_used": 0,
        "test_payload_opened": False, "student_training_allowed": gate["passed"],
        "records_path": str((run_dir / "records.json").relative_to(PROJECT_ROOT)),
        "preview_path": str(preview.relative_to(PROJECT_ROOT)),
    }
    atomic_json(run_dir / "report.json", report)
    pointer = {"path": str((run_dir / "report.json").relative_to(PROJECT_ROOT)),
               "sha256": sha256(run_dir / "report.json"), "run_id": run_dir.name,
               "student_training_allowed": gate["passed"]}
    atomic_json(ARTIFACTS / "color_region_teacher/current_pilot.json", pointer)
    return report


def run_pilot(maximum_images: int = 12, regions_per_image: int = 4,
              confidence: float = 0.60) -> dict:
    audit = load_audit()
    rows = select_pilot_rows(audit["fit_rows"], maximum_images)
    validate_fit_rows(rows, audit)
    protocol = {"version": VERSION, "seed": SEED, "maximum_images": maximum_images,
                "regions_per_image": regions_per_image, "confidence": confidence,
                "instruction": INSTRUCTION, "audit_fingerprint": audit["audit_fingerprint"],
                "reference_pixels_used_for_fit": 0, "marked_images_used": 0}
    run_id = f"color_region_teacher_pilot_{uuid.uuid4().hex[:16]}"
    run_dir = RUNS / run_id
    crops_dir = run_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(run_dir / "protocol.json", protocol)

    prepared = []
    for index, row in enumerate(rows, start=1):
        path = within_project(row["path"])
        rgb = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
        lung = unlabeled_lung_probability(rgb) >= 0.5
        labels = color_regions(rgb, lung, n_segments=80)
        regions = select_diverse_regions(region_statistics(rgb, lung, labels), regions_per_image)
        context_path = crops_dir / f"{row['id']}_context.jpg"
        Image.fromarray(rgb).save(context_path, quality=92)
        for region in regions:
            crop = make_region_crop(rgb, labels, region)
            crop_path = crops_dir / f"{row['id']}_region_{region['region_id']:04d}.png"
            crop.save(crop_path)
            prepared.append({"row": row, "region": region, "context_path": context_path,
                             "crop_path": crop_path})
        print(f"v15 pilot geometry: {index}/{len(rows)} images, {len(prepared)} regions", flush=True)

    provenance, directory = _qwen_provenance()
    teacher = ZeroShotRegionTeacher(directory)
    records = []
    try:
        for index, item in enumerate(prepared, start=1):
            context = Image.open(item["context_path"]).convert("RGB")
            crop = Image.open(item["crop_path"]).convert("RGB")
            first = teacher.classify(context, crop, False)
            flipped = teacher.classify(context, crop, True)
            region = item["region"]
            record = {
                "image_id": item["row"]["id"], "source_group": item["row"].get("source_group"),
                "pixel_sha256": item["row"]["pixel_sha256"], "region_id": region["region_id"],
                "area_fraction": region["area_fraction"], "mean_lab": region["mean_lab"],
                "std_lab": region["std_lab"], "box": region["box"],
                "context_path": str(item["context_path"].relative_to(PROJECT_ROOT)),
                "crop_path": str(item["crop_path"].relative_to(PROJECT_ROOT)),
                "original": first, "horizontal_flip": flipped,
                "accepted_label": _accepted(first, flipped, confidence),
            }
            records.append(record)
            atomic_json(run_dir / "records.json", records)
            print(f"v15 Qwen region: {index}/{len(prepared)} {record['accepted_label']}", flush=True)
    finally:
        teacher.close()

    return finalize_pilot(run_dir)
