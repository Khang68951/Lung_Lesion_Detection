from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy import ndimage
from skimage.color import rgb2lab, rgb2hsv
from skimage.segmentation import slic
from PIL import Image

from .io import LUNG_UNET_CHECKPOINT, SAM2_CHECKPOINT, SAM2_VENDOR


_REMBG_SESSIONS = {}
_UNLABELED_LUNG_MODEL = None
_SAM2_REFINER = None


def rembg_alpha(rgb: np.ndarray, model: str = "u2netp") -> np.ndarray:
    """Generic pretrained foreground alpha; never trained on project references."""
    global _REMBG_SESSIONS
    import os
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[3]
    if model == "u2netp":
        weight_root = project_root / "Lung_Region_Contrast_Lab/weights/rembg"
        model_path = weight_root / "u2netp.onnx"
    elif model in {"u2net", "isnet-general-use"}:
        weight_root = project_root / "Legacy_Project/Test/models"
        filename = "isnet-general-use.onnx" if model == "isnet-general-use" else "u2net.onnx"
        model_path = weight_root / filename
    else:
        raise ValueError(f"Unsupported generic foreground model: {model}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing generic foreground weight: {model_path}")
    os.environ["U2NET_HOME"] = str(weight_root)
    import onnxruntime
    onnxruntime.disable_telemetry_events()
    from rembg import new_session, remove
    if model not in _REMBG_SESSIONS:
        _REMBG_SESSIONS[model] = new_session(model, providers=["CPUExecutionProvider"])
    return np.asarray(remove(Image.fromarray(np.asarray(rgb, dtype=np.uint8)),
                             session=_REMBG_SESSIONS[model], only_mask=True).convert("L"))


def unlabeled_lung_probability(rgb: np.ndarray) -> np.ndarray:
    """U-Net fitted only to Lab/GrabCut pseudo-masks from 1,764 unmarked images."""
    global _UNLABELED_LUNG_MODEL
    import torch
    from torch import nn
    from torch.nn import functional as functional

    class ConvBlock(nn.Module):
        def __init__(self, inputs, outputs):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(inputs, outputs, 3, padding=1, bias=False), nn.GroupNorm(4, outputs), nn.SiLU(),
                nn.Conv2d(outputs, outputs, 3, padding=1, bias=False), nn.GroupNorm(4, outputs), nn.SiLU())
        def forward(self, value):
            return self.net(value)

    class SmallUNet(nn.Module):
        def __init__(self, base=16):
            super().__init__()
            self.enc1, self.enc2 = ConvBlock(3, base), ConvBlock(base, base * 2)
            self.enc3, self.enc4 = ConvBlock(base * 2, base * 4), ConvBlock(base * 4, base * 8)
            self.center = ConvBlock(base * 8, base * 16)
            self.dec4, self.dec3 = ConvBlock(base * 24, base * 8), ConvBlock(base * 12, base * 4)
            self.dec2, self.dec1 = ConvBlock(base * 6, base * 2), ConvBlock(base * 3, base)
            self.head = nn.Conv2d(base, 1, 1)
        def forward(self, value):
            a = self.enc1(value); b = self.enc2(functional.max_pool2d(a, 2))
            c = self.enc3(functional.max_pool2d(b, 2)); d = self.enc4(functional.max_pool2d(c, 2))
            value = self.center(functional.max_pool2d(d, 2))
            for skip, block in ((d, self.dec4), (c, self.dec3), (b, self.dec2), (a, self.dec1)):
                value = functional.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                value = block(torch.cat([value, skip], dim=1))
            return self.head(value)

    if _UNLABELED_LUNG_MODEL is None:
        state = torch.load(LUNG_UNET_CHECKPOINT, map_location="cpu", weights_only=True)
        if state.get("model_settings") != {"base_channels": 16, "image_size": 256}:
            raise ValueError("Unexpected unlabeled lung checkpoint architecture")
        _UNLABELED_LUNG_MODEL = SmallUNet(16).eval()
        _UNLABELED_LUNG_MODEL.load_state_dict(state["model"])
        torch.set_num_threads(2)
    native_shape = rgb.shape[:2]
    resized = cv2.resize(np.asarray(rgb, dtype=np.uint8), (256, 256), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy()).float()[None] / 255.0
    with torch.inference_mode():
        probability = _UNLABELED_LUNG_MODEL(tensor).sigmoid()[0, 0].numpy()
    return cv2.resize(probability, (native_shape[1], native_shape[0]), interpolation=cv2.INTER_LINEAR)


def _scaled(rgb: np.ndarray, maximum: int = 768) -> tuple[np.ndarray, float]:
    height, width = rgb.shape[:2]
    scale = min(1.0, maximum / max(height, width))
    if scale == 1.0:
        return rgb, 1.0
    return cv2.resize(rgb, (round(width * scale), round(height * scale)),
                      interpolation=cv2.INTER_AREA), scale


def _largest(mask: np.ndarray, minimum_fraction: float = 0.001) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    keep = labels == largest
    # Retain other substantial tissue pieces close to the largest component.
    grown = cv2.dilate(keep.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
    threshold = max(64, round(mask.size * minimum_fraction))
    for index, area in enumerate(areas, start=1):
        component = labels == index
        if index != largest and area >= threshold and np.any(component & grown):
            keep |= component
    return keep


def foreground_mask(rgb: np.ndarray) -> np.ndarray:
    """Classical, image-only gross tissue segmentation against the pale cloth."""
    small, scale = _scaled(np.asarray(rgb, dtype=np.uint8))
    lab = rgb2lab(small)
    hsv = rgb2hsv(small)
    h, w = small.shape[:2]
    border_width = max(2, round(min(h, w) * 0.035))
    border = np.zeros((h, w), bool)
    border[:border_width] = border[-border_width:] = True
    border[:, :border_width] = border[:, -border_width:] = True
    background = np.median(lab[border], axis=0)
    distance = np.sqrt(np.sum((lab - background) ** 2, axis=2))
    # Tissue is darker/redder/more saturated than the cloth. Black letterbox is
    # excluded here and restored only as background.
    value = hsv[..., 2]
    candidate = ((distance >= 11.0) & ((hsv[..., 1] >= 0.10) | (lab[..., 0] <= background[0] - 7.0)))
    candidate &= value >= 0.035
    radius = max(2, round(min(h, w) * 0.006))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    candidate = _largest(candidate)
    candidate = ndimage.binary_fill_holes(candidate)
    if scale != 1.0:
        candidate = cv2.resize(candidate.astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
                               interpolation=cv2.INTER_NEAREST) > 0
    return candidate


def robust_features(rgb: np.ndarray, lung: np.ndarray) -> dict[str, np.ndarray]:
    rgb = np.asarray(rgb, dtype=np.uint8)
    lung = np.asarray(lung, dtype=bool)
    lab = rgb2lab(rgb)
    hsv = rgb2hsv(rgb)
    if not lung.any():
        empty = np.zeros(lung.shape, np.float32)
        return {name: empty.copy() for name in ("dark", "a", "b", "saturation", "local_dark",
                                                 "absolute_dark", "absolute_red", "absolute_purple")}
    values = lab[lung]
    median = np.median(values, axis=0)
    scale = np.maximum(np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0),
                       np.array([4.0, 3.0, 3.0]))
    dark = (median[0] - lab[..., 0]) / scale[0]
    a = (lab[..., 1] - median[1]) / scale[1]
    b = (lab[..., 2] - median[2]) / scale[2]
    saturation_values = hsv[..., 1][lung]
    sat_median = np.median(saturation_values)
    sat_scale = max(np.percentile(saturation_values, 75) - np.percentile(saturation_values, 25), 0.05)
    saturation = (hsv[..., 1] - sat_median) / sat_scale
    maximum = max(rgb.shape[:2])
    reduced = lab[..., 0]
    if maximum > 768:
        factor = 768 / maximum
        reduced = cv2.resize(reduced, (round(rgb.shape[1] * factor), round(rgb.shape[0] * factor)),
                             interpolation=cv2.INTER_AREA)
    sigma = max(5.0, min(reduced.shape) * 0.045)
    local_mean = cv2.GaussianBlur(reduced.astype(np.float32), (0, 0), sigma)
    local_dark = (local_mean - reduced) / scale[0]
    if local_dark.shape != lung.shape:
        local_dark = cv2.resize(local_dark, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    absolute_dark = (60.0 - lab[..., 0]) / 20.0
    absolute_red = (lab[..., 1] - 15.0) / 15.0
    absolute_purple = (10.0 - lab[..., 2]) / 15.0
    return {"dark": dark.astype(np.float32), "a": a.astype(np.float32),
            "b": b.astype(np.float32), "saturation": saturation.astype(np.float32),
            "local_dark": local_dark.astype(np.float32),
            "absolute_dark": absolute_dark.astype(np.float32),
            "absolute_red": absolute_red.astype(np.float32),
            "absolute_purple": absolute_purple.astype(np.float32)}


@dataclass(frozen=True)
class LesionConfig:
    threshold: float = 1.0
    dark_weight: float = 1.0
    local_dark_weight: float = 0.5
    a_weight: float = 0.2
    b_weight: float = -0.1
    saturation_weight: float = 0.1
    absolute_dark_weight: float = 0.0
    absolute_red_weight: float = 0.0
    absolute_purple_weight: float = 0.0
    minimum_dark: float = 0.0
    minimum_component_fraction: float = 0.0005
    closing_fraction: float = 0.003

    def to_dict(self) -> dict:
        return asdict(self)


def lesion_score(features: dict[str, np.ndarray], config: LesionConfig) -> np.ndarray:
    return (config.dark_weight * features["dark"]
            + config.local_dark_weight * features["local_dark"]
            + config.a_weight * features["a"]
            + config.b_weight * features["b"]
            + config.saturation_weight * features["saturation"]
            + config.absolute_dark_weight * features["absolute_dark"]
            + config.absolute_red_weight * features["absolute_red"]
            + config.absolute_purple_weight * features["absolute_purple"])


def lesion_mask(features: dict[str, np.ndarray], lung: np.ndarray,
                config: LesionConfig) -> np.ndarray:
    score = lesion_score(features, config)
    candidate = (score >= config.threshold) & (features["dark"] >= config.minimum_dark) & lung
    radius = max(1, round(min(lung.shape) * config.closing_fraction))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    minimum = max(8, round(int(lung.sum()) * config.minimum_component_fraction))
    output = np.zeros_like(candidate)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= minimum:
            output[labels == index] = True
    return output & lung


def color_regions(rgb: np.ndarray, lung: np.ndarray, n_segments: int = 450) -> np.ndarray:
    if not lung.any():
        return np.zeros(lung.shape, np.int32)
    return slic(np.asarray(rgb, dtype=np.uint8), n_segments=n_segments, compactness=6.0,
                sigma=1.0, mask=np.asarray(lung, dtype=bool), start_label=1,
                enforce_connectivity=True, convert2lab=True).astype(np.int32)


def region_lesion_mask(features: dict[str, np.ndarray], lung: np.ndarray,
                       labels: np.ndarray, config: LesionConfig,
                       support: float) -> np.ndarray:
    if not 0 < support <= 1 or labels.shape != lung.shape:
        raise ValueError("Invalid region support or geometry")
    raw = ((lesion_score(features, config) >= config.threshold)
           & (features["dark"] >= config.minimum_dark) & lung)
    maximum = int(labels.max())
    totals = np.bincount(labels[lung], minlength=maximum + 1)
    positives = np.bincount(labels[raw], minlength=maximum + 1)
    fractions = positives / np.maximum(totals, 1)
    selected = fractions >= support
    selected[0] = False
    candidate = selected[labels] & lung
    radius = max(1, round(min(lung.shape) * config.closing_fraction))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    count, components, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    minimum = max(8, round(int(lung.sum()) * config.minimum_component_fraction))
    output = np.zeros_like(candidate)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= minimum:
            output[components == index] = True
    return output & lung


def grabcut_lesion_mask(rgb: np.ndarray, features: dict[str, np.ndarray],
                        lung: np.ndarray, config: LesionConfig,
                        core_threshold: float, iterations: int = 2) -> np.ndarray:
    """Per-image unsupervised color GMM initialized by fitted low/high scores."""
    score = lesion_score(features, config)
    probable = (score >= config.threshold) & (features["dark"] >= config.minimum_dark) & lung
    core = (score >= core_threshold) & (features["dark"] >= config.minimum_dark) & lung
    if not probable.any() or not core.any() or np.all(probable == lung):
        return lesion_mask(features, lung, config)
    height, width = lung.shape
    scale = min(1.0, 320 / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small_rgb = cv2.resize(np.asarray(rgb, dtype=np.uint8), size, interpolation=cv2.INTER_AREA)
    small_lung = cv2.resize(lung.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    small_probable = cv2.resize(probable.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    small_core = cv2.resize(core.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    mask = np.full(small_lung.shape, cv2.GC_BGD, np.uint8)
    mask[small_lung] = cv2.GC_PR_BGD
    mask[small_probable] = cv2.GC_PR_FGD
    mask[small_core] = cv2.GC_FGD
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small_rgb, mask, None, background_model, foreground_model,
                    iterations, cv2.GC_INIT_WITH_MASK)
        result = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)) & small_lung
    except cv2.error:
        result = small_probable
    result = cv2.resize(result.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    result &= lung
    count, components, stats, _ = cv2.connectedComponentsWithStats(result.astype(np.uint8), 8)
    minimum = max(8, round(int(lung.sum()) * config.minimum_component_fraction))
    output = np.zeros_like(result)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= minimum:
            output[components == index] = True
    return output


def _prompt_components(probable: np.ndarray, core: np.ndarray, lung: np.ndarray,
                       padding_fraction: float, erosion_fraction: float,
                       maximum_components: int = 5) -> list[dict]:
    """Create reference-free SAM prompts from appearance-mask components."""
    probable = np.asarray(probable, bool) & np.asarray(lung, bool)
    core = np.asarray(core, bool) & probable
    count, labels, stats, _ = cv2.connectedComponentsWithStats(probable.astype(np.uint8), 8)
    minimum = max(16, round(int(lung.sum()) * 0.001))
    indices = [index for index in range(1, count)
               if stats[index, cv2.CC_STAT_AREA] >= minimum]
    indices.sort(key=lambda index: int(stats[index, cv2.CC_STAT_AREA]), reverse=True)
    prompts = []
    for index in indices[:maximum_components]:
        component = labels == index
        x = int(stats[index, cv2.CC_STAT_LEFT]); y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH]); height = int(stats[index, cv2.CC_STAT_HEIGHT])
        pad = max(3, round(max(width, height) * padding_fraction))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1 = min(lung.shape[1] - 1, x + width - 1 + pad)
        y1 = min(lung.shape[0] - 1, y + height - 1 + pad)

        positive = core & component
        radius = round(min(lung.shape) * erosion_fraction)
        if radius > 0 and positive.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            eroded = cv2.erode(positive.astype(np.uint8), kernel) > 0
            if eroded.any():
                positive = eroded
        if not positive.any():
            positive = component
        distance = cv2.distanceTransform(positive.astype(np.uint8), cv2.DIST_L2, 5)
        points = []
        labels_out = []
        for _ in range(2):
            _, peak, _, location = cv2.minMaxLoc(distance)
            if peak <= 0:
                break
            points.append(location); labels_out.append(1)
            cv2.circle(distance, location, max(3, round(min(width, height) * 0.2)), 0, -1)

        box_region = np.zeros_like(lung, bool)
        box_region[y0:y1 + 1, x0:x1 + 1] = True
        dilation = max(2, round(min(lung.shape) * 0.008))
        dilated = cv2.dilate(component.astype(np.uint8),
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (2 * dilation + 1, 2 * dilation + 1))) > 0
        negative = box_region & lung & ~dilated
        negative_distance = cv2.distanceTransform(negative.astype(np.uint8), cv2.DIST_L2, 5)
        for _ in range(4):
            _, peak, _, location = cv2.minMaxLoc(negative_distance)
            if peak <= 0:
                break
            points.append(location); labels_out.append(0)
            cv2.circle(negative_distance, location, max(4, round(min(width, height) * 0.2)), 0, -1)
        if points:
            prompts.append({"component": component, "box": np.array([x0, y0, x1, y1], np.float32),
                            "points": np.asarray(points, np.float32),
                            "point_labels": np.asarray(labels_out, np.int32),
                            "area": int(component.sum())})
    return prompts


class SAM2BoundaryRefiner:
    """Generic SAM2 boundary proposals driven only by unlabeled appearance seeds."""

    def __init__(self, maximum_side: int = 512):
        import torch
        if str(SAM2_VENDOR) not in sys.path:
            sys.path.insert(0, str(SAM2_VENDOR))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.checkpoint = SAM2_CHECKPOINT
        self.maximum_side = maximum_side
        torch.set_num_threads(2)
        model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", str(SAM2_CHECKPOINT),
                           device="cpu", apply_postprocessing=False)
        self.predictor = SAM2ImagePredictor(model)

    def embedding(self, rgb: np.ndarray) -> np.ndarray:
        """Return the generic 64x64 SAM2 image embedding for one RGB image."""
        import torch
        height, width = rgb.shape[:2]
        scale = min(1.0, self.maximum_side / max(height, width))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        small_rgb = cv2.resize(np.asarray(rgb, np.uint8), size, interpolation=cv2.INTER_AREA)
        with torch.inference_mode():
            self.predictor.set_image(small_rgb)
        tensor = self.predictor._features["image_embed"][0]
        return tensor.detach().cpu().float().permute(1, 2, 0).numpy()

    def refine(self, rgb: np.ndarray, lung: np.ndarray, probable: np.ndarray,
               core: np.ndarray, padding_fraction: float,
               erosion_fraction: float, encode_image: bool = True) -> tuple[np.ndarray, dict, list[np.ndarray]]:
        import torch
        height, width = lung.shape
        scale = min(1.0, self.maximum_side / max(height, width))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        small_rgb = cv2.resize(np.asarray(rgb, np.uint8), size, interpolation=cv2.INTER_AREA)
        small_lung = cv2.resize(lung.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        small_probable = cv2.resize(probable.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        small_core = cv2.resize(core.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        prompts = _prompt_components(small_probable, small_core, small_lung,
                                     padding_fraction, erosion_fraction, 5)
        if encode_image:
            with torch.inference_mode():
                self.predictor.set_image(small_rgb)
        elif not getattr(self.predictor, "_is_image_set", False):
            raise RuntimeError("SAM2 image embedding is not initialized")
        selected = np.zeros(small_lung.shape, bool)
        raw_masks = []
        prompt_records = []
        for prompt in prompts:
            with torch.inference_mode():
                masks, scores, _ = self.predictor.predict(
                    point_coords=prompt["points"], point_labels=prompt["point_labels"],
                    box=prompt["box"], multimask_output=True)
            component = prompt["component"]
            dilation = max(2, round(min(small_lung.shape) * 0.012))
            nearby = cv2.dilate(component.astype(np.uint8),
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                          (2 * dilation + 1, 2 * dilation + 1))) > 0
            x0, y0, x1, y1 = prompt["box"].astype(int)
            box_mask = np.zeros_like(small_lung)
            box_mask[y0:y1 + 1, x0:x1 + 1] = True
            objectives = []
            clipped_masks = []
            for mask, score in zip(masks, scores):
                clipped = np.asarray(mask, bool) & small_lung & box_mask
                coverage = float((clipped & component).sum() / max(component.sum(), 1))
                locality = float((clipped & nearby).sum() / max(clipped.sum(), 1))
                objective = 0.45 * coverage + 0.35 * locality + 0.20 * float(np.clip(score, 0, 1))
                clipped_masks.append(clipped); objectives.append(objective)
                raw_masks.append(clipped)
            chosen = int(np.argmax(objectives))
            selected |= clipped_masks[chosen]
            prompt_records.append({
                "box": [int(value) for value in prompt["box"]],
                "points": prompt["points"].round(3).tolist(),
                "point_labels": prompt["point_labels"].tolist(),
                "component_area": prompt["area"],
                "sam_scores": [float(value) for value in scores],
                "selection_objectives": [float(value) for value in objectives],
                "selected_index": chosen,
            })
        native = cv2.resize(selected.astype(np.uint8), (width, height),
                            interpolation=cv2.INTER_NEAREST) > 0
        metadata = {"scale": scale, "encoded_shape": list(small_rgb.shape[:2]),
                    "prompt_count": len(prompts), "prompts": prompt_records}
        return native & lung, metadata, raw_masks


def sam2_boundary_refiner() -> SAM2BoundaryRefiner:
    global _SAM2_REFINER
    if _SAM2_REFINER is None:
        _SAM2_REFINER = SAM2BoundaryRefiner()
    return _SAM2_REFINER


def conservative_sam_fusion(probable: np.ndarray, proposal: np.ndarray,
                            dilation_fraction: float) -> np.ndarray:
    """Keep only appearance seeds supported by a nearby generic SAM boundary."""
    if dilation_fraction < 0:
        raise ValueError("SAM dilation fraction must be non-negative")
    probable = np.asarray(probable, bool)
    proposal = np.asarray(proposal, bool)
    if probable.shape != proposal.shape:
        raise ValueError("SAM proposal and appearance seed geometry differ")
    radius = round(min(probable.shape) * dilation_fraction)
    if radius > 0:
        proposal = cv2.dilate(proposal.astype(np.uint8),
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (2 * radius + 1, 2 * radius + 1))) > 0
    return probable & proposal


def region_consensus_from_seed(labels: np.ndarray, lung: np.ndarray,
                               seed: np.ndarray, support: float) -> np.ndarray:
    """Promote a coarse RGB tissue region when enough seed pixels support it."""
    if not 0 < support <= 1:
        raise ValueError("Region support must be in (0, 1]")
    labels = np.asarray(labels, np.int32)
    lung = np.asarray(lung, bool)
    seed = np.asarray(seed, bool) & lung
    if labels.shape != lung.shape or seed.shape != lung.shape:
        raise ValueError("Region consensus geometry differs")
    maximum = int(labels.max())
    totals = np.bincount(labels[lung], minlength=maximum + 1)
    positives = np.bincount(labels[seed], minlength=maximum + 1)
    selected = positives / np.maximum(totals, 1) >= support
    selected[0] = False
    return selected[labels] & lung


def embedding_tissue_regions(embedding: np.ndarray, lung: np.ndarray,
                             cluster_count: int, spatial_weight: float,
                             seed: int = 73) -> np.ndarray:
    """Cluster generic embeddings per image; no reference or cross-image fit."""
    from sklearn.cluster import MiniBatchKMeans
    embedding = np.asarray(embedding, np.float32)
    lung = np.asarray(lung, bool)
    if embedding.ndim != 3 or cluster_count < 2 or spatial_weight < 0:
        raise ValueError("Invalid embedding-region configuration")
    height, width = embedding.shape[:2]
    small_lung = cv2.resize(lung.astype(np.uint8), (width, height),
                            interpolation=cv2.INTER_NEAREST) > 0
    locations = np.flatnonzero(small_lung)
    if len(locations) < 2:
        return np.zeros(lung.shape, np.int32)
    vectors = embedding.reshape(-1, embedding.shape[2])[locations]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-6)
    yy, xx = np.divmod(locations, width)
    coordinates = np.column_stack((yy / max(height - 1, 1), xx / max(width - 1, 1))).astype(np.float32)
    vectors = np.column_stack((vectors, coordinates * spatial_weight)).astype(np.float32)
    clusters = min(cluster_count, len(locations))
    estimator = MiniBatchKMeans(n_clusters=clusters, random_state=seed, n_init=3,
                                batch_size=min(2048, len(locations)), max_iter=100,
                                reassignment_ratio=0.0)
    assigned = estimator.fit_predict(vectors)
    small_labels = np.zeros((height, width), np.int32)
    small_labels.ravel()[locations] = assigned.astype(np.int32) + 1
    labels = cv2.resize(small_labels, (lung.shape[1], lung.shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(np.int32)
    labels[~lung] = 0
    missing = lung & (labels == 0)
    if missing.any() and (labels > 0).any():
        nearest = ndimage.distance_transform_edt(labels == 0, return_distances=False,
                                                 return_indices=True)
        labels[missing] = labels[tuple(axis[missing] for axis in nearest)]
    return labels


def erode_prediction(mask: np.ndarray, radius_fraction: float) -> np.ndarray:
    if radius_fraction < 0:
        raise ValueError("Erosion fraction must be non-negative")
    mask = np.asarray(mask, bool)
    radius = round(min(mask.shape) * radius_fraction)
    if radius <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * radius + 1, 2 * radius + 1))
    return cv2.erode(mask.astype(np.uint8), kernel) > 0


def cap_mask_by_score(mask: np.ndarray, lung: np.ndarray, score: np.ndarray,
                      maximum_fraction: float) -> np.ndarray:
    """Remove the lowest-score positives until an unlabeled area prior is met."""
    if not 0 < maximum_fraction <= 1:
        raise ValueError("Area cap must be in (0, 1]")
    mask = np.asarray(mask, bool) & np.asarray(lung, bool)
    score = np.asarray(score, np.float32)
    if score.shape != mask.shape:
        raise ValueError("Area-cap score geometry differs")
    limit = max(1, round(int(lung.sum()) * maximum_fraction))
    indices = np.flatnonzero(mask)
    if len(indices) <= limit:
        return mask
    values = score.ravel()[indices]
    selected = np.argpartition(values, -limit)[-limit:]
    output = np.zeros(mask.size, bool)
    output[indices[selected]] = True
    return output.reshape(mask.shape)

