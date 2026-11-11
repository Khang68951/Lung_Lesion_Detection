from __future__ import annotations

import gc
import json
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from sklearn.cluster import KMeans
from skimage.color import rgb2lab

from .color_region.teacher import ZeroShotRegionTeacher, _qwen_provenance
from .evaluate import _measure, summarize
from .io import ARTIFACTS, PROJECT_ROOT, atomic_json, json_hash, now, sha256, within_project
from .lung_consensus_v54 import V53_ROOT, vote_consensus
from .reference_v2 import DevelopmentReferenceStore


VERSION = "v57_relative_qwen_color_clusters_pilot"
LETTERS = "ABCDEFGH"
MAXIMUM_IMAGES = 8
CLUSTERS = 8


def cluster_score_plane(labels: np.ndarray, scores: list[float]) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 2 or len(scores) != CLUSTERS or labels.min() < -1 or labels.max() >= CLUSTERS:
        raise ValueError("Invalid V57 cluster geometry")
    lookup = np.asarray(scores, np.float32)
    output = np.zeros(labels.shape, np.float32)
    inside = labels >= 0
    output[inside] = lookup[labels[inside]]
    return output


def _color_clusters(rgb: np.ndarray, lung: np.ndarray) -> np.ndarray:
    lab = rgb2lab(np.asarray(rgb, np.uint8)).astype(np.float32)
    pixels = lab[lung]
    if len(pixels) < CLUSTERS:
        raise ValueError("V57 lung has too few pixels")
    step = max(1, len(pixels) // 50000)
    model = KMeans(n_clusters=CLUSTERS, random_state=57, n_init=10, max_iter=300)
    model.fit(pixels[::step])
    labels = np.full(lung.shape, -1, np.int16)
    labels[lung] = model.predict(pixels).astype(np.int16)
    return labels


def make_cluster_collage(rgb: np.ndarray, labels: np.ndarray) -> Image.Image:
    tiles = []
    for cluster, letter in enumerate(LETTERS):
        target = labels == cluster
        dimmed = np.clip(np.asarray(rgb, np.float32) * .14, 0, 255).astype(np.uint8)
        dimmed[target] = rgb[target]
        tile = Image.fromarray(dimmed)
        tile.thumbnail((256, 256), Image.Resampling.LANCZOS)
        framed = Image.new("RGB", (264, 288), "black")
        framed.paste(tile, ((264 - tile.width) // 2, 24))
        ImageDraw.Draw(framed).text((8, 4), f"TARGET {letter}", fill="white")
        tiles.append(framed)
    collage = Image.new("RGB", (4 * 264, 2 * 288), "black")
    for index, tile in enumerate(tiles):
        collage.paste(tile, ((index % 4) * 264, (index // 4) * 288))
    return collage


@torch.inference_mode()
def _rank_clusters(teacher: ZeroShotRegionTeacher, context: Image.Image,
                   collage: Image.Image, question: str) -> dict:
    context = context.copy(); collage = collage.copy()
    context.thumbnail((640, 640), Image.Resampling.LANCZOS)
    collage.thumbnail((896, 512), Image.Resampling.LANCZOS)
    instruction = (
        "You inspect a clean gross pig-lung photograph. Eight mutually exclusive color clusters "
        "inside predicted lung lobes are shown as TARGET A through TARGET H; non-target pixels are "
        "dimmed. Compare natural color, local texture, and abrupt transitions. Ignore background, "
        "cut edges, glare, vessels, labels, airway, heart, and liver. " + question +
        " Reply with exactly one letter A, B, C, D, E, F, G, or H.")
    content = [{"type": "text", "text": instruction},
               {"type": "text", "text": "CLEAN WHOLE-LUNG CONTEXT:"},
               {"type": "image", "image": context},
               {"type": "text", "text": "EIGHT TARGET CLUSTERS:"},
               {"type": "image", "image": collage}]
    inputs = teacher.processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")
    ids = [teacher.processor.tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(value) != 1 for value in ids):
        raise ValueError("V57 answer letters must be single tokens")
    started = time.monotonic()
    logits = teacher.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1]
    probability = logits[[value[0] for value in ids]].float().softmax(0).cpu().numpy()
    return {"probabilities": probability.tolist(), "winner": LETTERS[int(np.argmax(probability))],
            "seconds": time.monotonic() - started, "input_tokens": int(inputs["input_ids"].shape[1])}


def _pilot_rows(rows: list[dict]) -> list[dict]:
    selected, seen = [], set()
    for row in rows:
        if row["group"] in seen:
            continue
        selected.append(row); seen.add(row["group"])
        if len(selected) == MAXIMUM_IMAGES:
            break
    return selected


def run_relative_qwen_v57_pilot() -> dict:
    provenance, directory = _qwen_provenance()
    protocol = {"version": VERSION, "qwen_repository": provenance["repository"],
                "qwen_revision": provenance["revision"], "images": MAXIMUM_IMAGES,
                "clusters_per_image": CLUSTERS, "kmeans_seed": 57,
                "queries": ["most lesion-like/abnormal", "most normal/healthy"],
                "score_modes": ["abnormal", "inverse_normal", "contrast", "rank_contrast"],
                "marked_fit_images": 0, "marked_fit_planes": 0,
                "sealed_test_images_opened": 0}
    fingerprint = json_hash(protocol)
    root = ARTIFACTS / "relative_qwen_v57" / fingerprint
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "protocol.json", protocol)
    report_path = root / "report.json"
    if report_path.is_file():
        return json.loads(report_path.read_text())

    store = DevelopmentReferenceStore(); rows = _pilot_rows(list(store.rows))
    missing = [row for row in rows if not (root / f"{row['id']}.npz").is_file()]
    teacher = None
    try:
        if missing:
            teacher = ZeroShotRegionTeacher(directory)
        for index, row in enumerate(rows, start=1):
            output = root / f"{row['id']}.npz"
            if output.is_file():
                print(f"{VERSION} cache {index}/{len(rows)}", flush=True); continue
            image_path = within_project(row["files"]["image"])
            if sha256(image_path) != row["sha256"]["image"]:
                raise ValueError(f"V57 clean RGB changed: {row['id']}")
            rgb = np.asarray(ImageOps.exif_transpose(Image.open(image_path)).convert("RGB"))
            v53_path = V53_ROOT / f"{row['id']}.npz"
            with np.load(v53_path, allow_pickle=False) as values:
                shape = tuple(values["mask_shape"]); metadata = json.loads(str(values["metadata_json"]))
                raw = [np.unpackbits(value, count=shape[0] * shape[1]).reshape(shape).astype(bool)
                       for value in values["packed_masks"]]
            planes = []
            for variant in range(12):
                pair = [mask for mask, record in zip(raw, metadata)
                        if record["variant"] == variant and record["output"] == 2]
                planes.append(np.logical_or.reduce(pair))
            lung = vote_consensus(planes, 3)
            labels = _color_clusters(rgb, lung)
            collage = make_cluster_collage(rgb, labels)
            abnormal = _rank_clusters(teacher, Image.fromarray(rgb), collage,
                                      "Select the cluster most likely to contain lesion-like abnormal lung tissue.")
            normal = _rank_clusters(teacher, Image.fromarray(rgb), collage,
                                    "Select the cluster most likely to contain normal healthy lung tissue.")
            np.savez_compressed(output, labels=labels, lung=lung.astype(np.uint8),
                                abnormal=np.asarray(abnormal["probabilities"], np.float32),
                                normal=np.asarray(normal["probabilities"], np.float32),
                                abnormal_json=np.asarray(json.dumps(abnormal)),
                                normal_json=np.asarray(json.dumps(normal)),
                                source_image_sha256=np.asarray(row["sha256"]["image"]),
                                source_v53_sha256=np.asarray(sha256(v53_path)))
            collage.save(root / f"{row['id']}_collage.jpg", quality=90)
            print(f"{VERSION} image {index}/{len(rows)} abnormal={abnormal['winner']} normal={normal['winner']}",
                  flush=True)
    finally:
        if teacher is not None:
            teacher.close(); del teacher
            gc.collect(); torch.cuda.empty_cache()

    # Open references only after every Qwen score vector is frozen.
    payloads = [store.load(row) for row in rows]
    frozen, source_hashes = [], {}
    for row in rows:
        path = root / f"{row['id']}.npz"
        with np.load(path, allow_pickle=False) as values:
            labels = values["labels"].copy(); lung = values["lung"].astype(bool)
            abnormal = values["abnormal"].copy(); normal = values["normal"].copy()
        contrast = abnormal / np.maximum(abnormal + normal, 1e-8)
        order = np.argsort(np.argsort(contrast))
        rank = (order.astype(np.float32) + .5) / CLUSTERS
        frozen.append({"lung": lung, "abnormal": cluster_score_plane(labels, abnormal),
                       "inverse_normal": cluster_score_plane(labels, 1 - normal),
                       "contrast": cluster_score_plane(labels, contrast),
                       "rank_contrast": cluster_score_plane(labels, rank)})
        source_hashes[row["id"]] = sha256(path)
    candidates = []
    for mode in protocol["score_modes"]:
        for threshold in np.arange(.05, .951, .025):
            measured = [_measure(payload, item["lung"],
                                 (item[mode] >= threshold) & item["lung"])
                        for payload, item in zip(payloads, frozen)]
            candidates.append({"mode": mode, "threshold": round(float(threshold), 3),
                               "summary": summarize(measured)})
    winner = max(candidates, key=lambda value: (
        min(value["summary"]["pooled_lesion"][key] or 0
            for key in ("precision", "recall", "specificity")),
        value["summary"]["pooled_lesion"]["dice"] or 0))
    dice_winner = max(candidates, key=lambda value: value["summary"]["pooled_lesion"]["dice"] or 0)
    report = {"status": "done", "created_at": now(), "version": VERSION,
              "protocol": protocol, "protocol_fingerprint": fingerprint,
              "source_files_sha256": source_hashes, "balanced_winner": winner,
              "dice_winner": dice_winner,
              "pilot_gate": {"passed": dice_winner["summary"]["pooled_lesion"]["dice"] >= .70,
                             "minimum_dice": .70},
              "candidate_count": len(candidates), "model_weights_updated": False,
              "marked_fit_images": 0, "marked_fit_planes": 0,
              "test_payload_opened": False}
    atomic_json(report_path, report)
    return report
