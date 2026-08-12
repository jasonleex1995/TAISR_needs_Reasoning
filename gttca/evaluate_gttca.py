"""GTTCA evaluator for ReasonText benchmark.

Single SR-output directory in, per-text-region JSONL/CSV out. Pipeline:
  1. Scale GT polygon (HR coords) to SR image space.
  2. Crop axis-aligned bbox + polygon mask (background fill outside polygon).
  3. Apply text_rotation / text_flipped to get an upright crop.
  4. Upscale (LANCZOS) if short side < min_short_side.
  5. Run PaddleOCR-VL-1.5 on the oriented crop.
  6. Compute exact-match (strict + relaxed) against GT text.

Run via micromamba env `rac-eval-paddle` on a node with one CUDA GPU visible.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import math
import os
import re
import traceback
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


LOGGER = logging.getLogger("gttca")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MASK_BACKGROUND_COLORS: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "gray": (127, 127, 127),
    "white": (255, 255, 255),
}
DEFAULT_PROMPT = "OCR:"


# ---------- data classes ----------


@dataclass(frozen=True)
class TextAnnotation:
    text: str
    coordinates: tuple[tuple[float, float], ...]
    rotation: int = 0
    flipped: bool = False
    difficulty: str = ""
    reason_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImageAnnotationInfo:
    image_name: str
    source_key: str
    annotations: tuple[TextAnnotation, ...]
    hr_image: str = ""
    lr_image: str = ""


@dataclass
class CropArtifacts:
    bbox_xyxy: tuple[int, int, int, int]
    masked_crop: Image.Image


# ---------- annotation loading (ReasonText-meta.json) ----------


def parse_annotation_coordinates(annotation: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    raw = annotation.get("hr_coords")
    if raw is None:
        raw = annotation.get("coordinates")
    if raw is None:
        raise KeyError("Annotation must provide either 'hr_coords' or 'coordinates'.")
    return tuple((float(x), float(y)) for x, y in raw)


def parse_reason_types(annotation: Mapping[str, Any]) -> tuple[str, ...]:
    raw = annotation.get("reason_types") or []
    if isinstance(raw, str):
        raw = [raw]
    return tuple(str(item) for item in raw if str(item))


def load_meta(meta_path: Path) -> dict[str, ImageAnnotationInfo]:
    """Parse ReasonText-meta.json into {image_filename -> ImageAnnotationInfo}.

    Tolerates both ReasonText (`hr_coords`/`text_rotation`/`text_flipped`) and
    older RealText (`coordinates`/`rotation`/`flipped`) field names.
    """
    with meta_path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)
    if not isinstance(raw_data, dict):
        raise ValueError(f"meta JSON must be an object: {meta_path}")

    parsed: dict[str, ImageAnnotationInfo] = {}
    for source_key, payload in raw_data.items():
        if not isinstance(payload, Mapping):
            raise ValueError(f"meta entry must be a JSON object: {source_key}")

        hr_image = str(payload.get("hr_image", "") or "")
        candidates = []
        if hr_image:
            candidates.append(Path(hr_image).name)
        candidates.append(Path(source_key).name)
        # ReasonText keys are bare stems (e.g. "DRealSR__Canon_10__256_01") -> need ext
        # so prefer the hr_image-derived name when present.
        image_name = next((c for c in candidates if c), "")
        if not image_name:
            raise ValueError(f"Could not resolve image name from meta entry: {source_key}")

        annotations = []
        for ann in payload.get("annotations", []):
            annotations.append(
                TextAnnotation(
                    text=str(ann.get("text", "")),
                    coordinates=parse_annotation_coordinates(ann),
                    rotation=int(ann.get("text_rotation", ann.get("rotation", 0)) or 0),
                    flipped=bool(ann.get("text_flipped", ann.get("flipped", False))),
                    difficulty=str(ann.get("text_difficulty", ann.get("difficulty", "")) or ""),
                    reason_types=parse_reason_types(ann),
                )
            )

        if image_name in parsed:
            raise ValueError(f"Duplicate image name in meta: {image_name}")
        parsed[image_name] = ImageAnnotationInfo(
            image_name=image_name,
            source_key=source_key,
            annotations=tuple(annotations),
            hr_image=hr_image,
            lr_image=str(payload.get("lr_image", "") or ""),
        )
    return parsed


# ---------- image discovery ----------


def collect_image_map(directory: Path) -> dict[str, Path]:
    """List image files in `directory` keyed by stem (filename without ext).

    Skips dotfiles (e.g. `.DS_Store`, `.ipynb_checkpoints`). Stem-based key lets
    LR `.jpg` and SR `.png` be matched even when extensions differ.
    """
    mapping: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = path.stem
        if stem in mapping:
            raise ValueError(f"Duplicate stem in {directory}: {stem} -> {mapping[stem]} and {path}")
        mapping[stem] = path
    return mapping


# ---------- polygon -> crop pipeline ----------


def scale_polygon(
    coordinates: Sequence[tuple[float, float]],
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    points = np.array(coordinates, dtype=np.float32)
    scale_x = target_width / max(source_width, 1)
    scale_y = target_height / max(source_height, 1)
    points[:, 0] *= scale_x
    points[:, 1] *= scale_y
    points[:, 0] = np.clip(points[:, 0], 0, max(target_width - 1, 0))
    points[:, 1] = np.clip(points[:, 1], 0, max(target_height - 1, 0))
    return points


def compute_polygon_bbox(
    polygon: np.ndarray,
    image_width: int,
    image_height: int,
    padding_px: int,
) -> tuple[int, int, int, int]:
    min_x = max(int(math.floor(float(np.min(polygon[:, 0])))) - padding_px, 0)
    min_y = max(int(math.floor(float(np.min(polygon[:, 1])))) - padding_px, 0)
    max_x = min(int(math.ceil(float(np.max(polygon[:, 0])))) + padding_px + 1, image_width)
    max_y = min(int(math.ceil(float(np.max(polygon[:, 1])))) + padding_px + 1, image_height)
    if max_x <= min_x or max_y <= min_y:
        raise ValueError(f"Degenerate polygon bbox: {(min_x, min_y, max_x, max_y)}")
    return min_x, min_y, max_x, max_y


def build_polygon_crop(
    image: Image.Image,
    polygon: np.ndarray,
    padding_px: int,
    background_color: tuple[int, int, int],
) -> CropArtifacts:
    rgb = np.asarray(image.convert("RGB"))
    image_height, image_width = rgb.shape[:2]
    bbox = compute_polygon_bbox(polygon, image_width=image_width, image_height=image_height, padding_px=padding_px)
    x0, y0, x1, y1 = bbox

    bbox_crop = rgb[y0:y1, x0:x1].copy()
    if bbox_crop.size == 0:
        raise ValueError(f"Empty bbox crop for polygon bbox {bbox}")

    local_polygon = polygon.copy()
    local_polygon[:, 0] -= x0
    local_polygon[:, 1] -= y0
    local_polygon_int = np.round(local_polygon).astype(np.int32)

    mask = np.zeros((bbox_crop.shape[0], bbox_crop.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [local_polygon_int], 255)

    masked = np.full_like(bbox_crop, background_color)
    masked[mask == 255] = bbox_crop[mask == 255]
    return CropArtifacts(bbox_xyxy=bbox, masked_crop=Image.fromarray(masked))


def orient_crop_to_upright(
    image: Image.Image,
    rotation: int,
    flipped: bool,
    rotation_direction: str,
    orientation_order: str,
    fillcolor: tuple[int, int, int],
    apply_rotation: bool,
    apply_flip: bool,
) -> Image.Image:
    def rot_step(img: Image.Image) -> Image.Image:
        if not apply_rotation:
            return img
        normalized = rotation % 360
        if normalized == 0:
            return img
        angle = normalized if rotation_direction == "ccw" else -normalized
        return img.rotate(
            angle,
            expand=True,
            resample=getattr(Image.Resampling, "BICUBIC", Image.BICUBIC),
            fillcolor=fillcolor,
        )

    def flip_step(img: Image.Image) -> Image.Image:
        if not apply_flip or not flipped:
            return img
        return ImageOps.mirror(img)

    if orientation_order == "rotate_then_flip":
        return flip_step(rot_step(image))
    return rot_step(flip_step(image))


def resize_for_ocr(
    image: Image.Image,
    min_short_side: int,
    max_long_side: int,
    max_upscale_factor: float,
) -> Image.Image:
    width, height = image.size
    short_side = min(width, height)
    long_side = max(width, height)
    if short_side <= 0 or short_side >= min_short_side:
        return image
    scale = min(
        min_short_side / short_side,
        max_long_side / max(long_side, 1),
        max_upscale_factor,
    )
    if scale <= 1.0:
        return image
    new_size = (max(int(round(width * scale)), 1), max(int(round(height * scale)), 1))
    return image.resize(new_size, resample=getattr(Image.Resampling, "LANCZOS", Image.LANCZOS))


# ---------- OCR engine: PaddleOCR-VL-1.5 ----------


def patch_create_causal_mask_compatibility() -> None:
    """PaddleOCR-VL-1.5's modeling code calls ``create_causal_mask`` with the
    legacy ``input_embeds`` kwarg; newer transformers renamed it to
    ``inputs_embeds``. Bridge the two."""
    from transformers import masking_utils

    current = masking_utils.create_causal_mask
    if getattr(current, "_paddleocr_vl_compat", False):
        return
    parameters = inspect.signature(current).parameters
    if "inputs_embeds" in parameters or "input_embeds" not in parameters:
        return

    def shim(*args: Any, **kwargs: Any):
        if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return current(*args, **kwargs)

    shim._paddleocr_vl_compat = True
    masking_utils.create_causal_mask = shim


def resolve_torch_dtype(dtype: str) -> torch.dtype:
    name = dtype.lower()
    if name in {"auto", "fp32", "float32"}:
        return torch.float32
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype}")


def resolve_preferred_loader(model_path: Path) -> str:
    config_path = model_path / "config.json"
    try:
        payload = json.loads(config_path.read_text())
    except Exception:
        return "causal_lm"
    auto_map = payload.get("auto_map", {}) or {}
    if "AutoModelForImageTextToText" in auto_map:
        return "image_text_to_text"
    return "causal_lm"


def move_batch_to_device(batch: Any, device: str) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    return batch


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    stripped = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", stripped)
    stripped = re.sub(r"\n?```$", "", stripped)
    return stripped


def maybe_parse_json(text: str) -> Any | None:
    candidate = strip_code_fences(text).strip()
    if not candidate or candidate[0] not in "[{":
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _walk_for_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        return "\n".join(p for p in (_walk_for_text(item) for item in payload) if p)
    if isinstance(payload, dict):
        for key in ("text", "value", "result", "content", "prediction", "label"):
            if key in payload:
                extracted = _walk_for_text(payload[key])
                if extracted:
                    return extracted
        return "\n".join(p for p in (_walk_for_text(v) for v in payload.values()) if p)
    return str(payload)


def extract_prediction_text(raw: str) -> str:
    candidate = strip_code_fences(raw)
    parsed = maybe_parse_json(candidate)
    if parsed is not None:
        candidate = _walk_for_text(parsed)
    candidate = re.sub(r"^\s*(OCR|Result|Prediction|Answer)\s*:\s*", "", candidate, flags=re.IGNORECASE)
    return candidate.strip("\r\n")


class PaddleOCRVLEngine:
    def __init__(
        self,
        *,
        model_path: Path,
        device: str,
        dtype: str,
        prompt: str,
        max_new_tokens: int,
        max_pixels: int,
        trust_remote_code: bool,
        attn_implementation: str | None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            AutoModelForImageTextToText = None

        patch_create_causal_mask_compatibility()
        self.model_path = str(model_path)
        self.device = device
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.max_pixels = max_pixels

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
        model_kwargs: dict[str, Any] = {
            "torch_dtype": resolve_torch_dtype(dtype),
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = None
        if resolve_preferred_loader(Path(self.model_path)) == "image_text_to_text" and AutoModelForImageTextToText is not None:
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(self.model_path, **model_kwargs)
            except Exception as exc:
                LOGGER.warning(
                    "AutoModelForImageTextToText failed for %s (%s); falling back to AutoModelForCausalLM.",
                    self.model_path,
                    exc,
                )
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        self.model = self.model.to(self.device).eval()

    def recognize(self, image: Image.Image) -> tuple[str, str]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        template_kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "images_kwargs": {
                "size": {
                    "shortest_edge": getattr(self.processor.image_processor, "min_pixels", 28 * 28),
                    "longest_edge": self.max_pixels,
                }
            },
        }
        try:
            inputs = self.processor.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("images_kwargs", None)
            inputs = self.processor.apply_chat_template(messages, **template_kwargs)

        inputs = move_batch_to_device(inputs, self.device)
        prompt_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated_tokens = outputs[0][prompt_length:].detach().cpu()
        try:
            raw = self.processor.decode(generated_tokens, skip_special_tokens=True)
        except TypeError:
            raw = self.processor.decode(generated_tokens)
        return raw, extract_prediction_text(raw)


# ---------- exact-match metrics ----------


def normalize_text_relaxed(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("　", " ")
    normalized = normalized.casefold()
    return re.sub(r"\s+", "", normalized)


def compute_exact_match(label: str, prediction: str) -> dict[str, int]:
    return {
        "exact_match_strict": int(label == prediction),
        "exact_match_ignore_case_whitespace": int(
            normalize_text_relaxed(label) == normalize_text_relaxed(prediction)
        ),
    }


# ---------- I/O helpers ----------


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_ready(dict(row)), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json_ready(v) for k, v in row.items()})


def apply_environment_overrides(weights_config: Mapping[str, Any]) -> dict[str, str]:
    env_mapping = weights_config.get("env") or {}
    if not isinstance(env_mapping, Mapping):
        raise ValueError("weights_config['env'] must be a JSON object.")
    applied: dict[str, str] = {}
    for k, v in env_mapping.items():
        if v in (None, ""):
            continue
        os.environ.setdefault(str(k), str(v))
        applied[str(k)] = os.environ[str(k)]
    return applied


def load_weights_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"weights config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"weights config must be a JSON object: {path}")
    return payload


def resolve_paddleocr_path(cli_path: Path | None, weights_config: Mapping[str, Any]) -> Path:
    if cli_path is not None:
        return cli_path.expanduser()
    configured = weights_config.get("paddleocr_model_path")
    if not configured:
        raise ValueError("PaddleOCR model path missing: pass --paddleocr-model-path or set 'paddleocr_model_path' in weights_config.")
    return Path(str(configured)).expanduser()


# ---------- per-region evaluation ----------


def evaluate_one_image(
    *,
    model_name: str,
    image_stem: str,
    sr_path: Path,
    hr_path: Path,
    annotation_info: ImageAnnotationInfo,
    engine: PaddleOCRVLEngine,
    args: argparse.Namespace,
    mask_background_rgb: tuple[int, int, int],
) -> list[dict[str, Any]]:
    with Image.open(sr_path) as sr_handle:
        sr_image = sr_handle.convert("RGB")
    with Image.open(hr_path) as hr_handle:
        # We only need HR dimensions (annotations are in HR coord space).
        hr_w, hr_h = hr_handle.size

    rows: list[dict[str, Any]] = []
    for ann_idx, annotation in enumerate(annotation_info.annotations):
        if args.limit_text_regions is not None and ann_idx >= args.limit_text_regions:
            break

        record_id = f"{model_name}::{image_stem}::ann::{ann_idx:04d}"
        row: dict[str, Any] = {
            "record_id": record_id,
            "model_name": model_name,
            "image_stem": image_stem,
            "image_path": str(sr_path),
            "annotation_index": ann_idx,
            "source_key": annotation_info.source_key,
            "hr_image": annotation_info.hr_image,
            "lr_image": annotation_info.lr_image,
            "label": annotation.text,
            "prediction": "",
            "prediction_raw": "",
            "text_rotation": annotation.rotation,
            "text_flipped": int(annotation.flipped),
            "text_difficulty": annotation.difficulty,
            "reason_types": list(annotation.reason_types),
            "reason_types_joined": "|".join(annotation.reason_types),
            "polygon_num_points": len(annotation.coordinates),
            "hr_coords": [[float(x), float(y)] for x, y in annotation.coordinates],
            "exact_match_strict": 0,
            "exact_match_ignore_case_whitespace": 0,
            "status": "ok",
            "error_type": "",
            "error_message": "",
        }
        try:
            scaled = scale_polygon(
                annotation.coordinates,
                source_width=hr_w,
                source_height=hr_h,
                target_width=sr_image.width,
                target_height=sr_image.height,
            )
            crop = build_polygon_crop(
                image=sr_image,
                polygon=scaled,
                padding_px=args.padding_px,
                background_color=mask_background_rgb,
            )
            oriented = orient_crop_to_upright(
                image=crop.masked_crop,
                rotation=annotation.rotation,
                flipped=annotation.flipped,
                rotation_direction=args.rotation_direction,
                orientation_order=args.orientation_order,
                fillcolor=mask_background_rgb,
                apply_rotation=args.apply_rotation,
                apply_flip=args.apply_flip,
            )
            ocr_input = resize_for_ocr(
                oriented,
                min_short_side=args.min_short_side,
                max_long_side=args.max_long_side,
                max_upscale_factor=args.max_upscale_factor,
            )
            raw_pred, parsed_pred = engine.recognize(ocr_input)
            row.update(
                {
                    "prediction": parsed_pred,
                    "prediction_raw": raw_pred,
                    "polygon_scaled": scaled.round(3).tolist(),
                    "bbox_xyxy": list(crop.bbox_xyxy),
                    "ocr_input_width": ocr_input.width,
                    "ocr_input_height": ocr_input.height,
                }
            )
            row.update(compute_exact_match(annotation.text, parsed_pred))
        except Exception as exc:
            row.update(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=5),
                }
            )
        rows.append(row)
    return rows


# ---------- CLI ----------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sr-dir", type=Path, required=True,
                        help="Directory containing SR output images (one image per text-instance source).")
    parser.add_argument("--hr-dir", type=Path, required=True,
                        help="Directory containing HR reference images (used only to read native HR dims).")
    parser.add_argument("--meta-json", type=Path, required=True,
                        help="ReasonText-meta.json (annotations keyed by image stem or hr_image).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write text_regions.jsonl, .csv, and run_config.json.")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Label written into every output row. Default: parent dir of --sr-dir.")
    parser.add_argument("--weights-config", type=Path, required=True,
                        help="JSON config: paddleocr_model_path + optional 'env' overrides.")

    parser.add_argument("--paddleocr-model-path", type=Path, default=None,
                        help="Override for PaddleOCR-VL-1.5 path; default reads weights_config.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float32")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--attn-implementation", default=None)
    parser.set_defaults(trust_remote_code=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")

    parser.add_argument("--mask-background", choices=sorted(MASK_BACKGROUND_COLORS), default="black")
    parser.add_argument("--padding-px", type=int, default=8)
    parser.add_argument("--rotation-direction", choices=["ccw", "cw"], default="ccw")
    parser.add_argument("--orientation-order", choices=["rotate_then_flip", "flip_then_rotate"],
                        default="rotate_then_flip")
    parser.set_defaults(apply_rotation=True, apply_flip=True)
    parser.add_argument("--no-apply-rotation", dest="apply_rotation", action="store_false")
    parser.add_argument("--no-apply-flip", dest="apply_flip", action="store_false")
    parser.add_argument("--min-short-side", type=int, default=64)
    parser.add_argument("--max-long-side", type=int, default=1536)
    parser.add_argument("--max-upscale-factor", type=float, default=4.0)

    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--limit-text-regions", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if --output-dir/text_regions.jsonl already exists (non-empty).")
    parser.add_argument("--verbose", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    args = build_parser().parse_args()
    args.sr_dir = args.sr_dir.expanduser()
    args.hr_dir = args.hr_dir.expanduser()
    args.meta_json = args.meta_json.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.weights_config = args.weights_config.expanduser()
    if args.paddleocr_model_path is not None:
        args.paddleocr_model_path = args.paddleocr_model_path.expanduser()
    if args.model_name is None:
        # `.../exp01/ReasonText/B3_FaithDiff/images` -> "B3_FaithDiff"
        # `.../exp01/ReasonText/B3_FaithDiff` -> "B3_FaithDiff"
        candidate = args.sr_dir
        if candidate.name == "images":
            candidate = candidate.parent
        args.model_name = candidate.name
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    text_regions_path = args.output_dir / "text_regions.jsonl"
    if args.skip_existing and text_regions_path.is_file() and text_regions_path.stat().st_size > 0:
        LOGGER.info("Skipping (already exists): %s", text_regions_path)
        return

    if not args.sr_dir.is_dir():
        raise FileNotFoundError(f"SR directory does not exist: {args.sr_dir}")
    if not args.hr_dir.is_dir():
        raise FileNotFoundError(f"HR directory does not exist: {args.hr_dir}")
    if not args.meta_json.is_file():
        raise FileNotFoundError(f"meta JSON does not exist: {args.meta_json}")

    weights_config = load_weights_config(args.weights_config)
    applied_env = apply_environment_overrides(weights_config)
    args.paddleocr_model_path = resolve_paddleocr_path(args.paddleocr_model_path, weights_config)
    if not args.paddleocr_model_path.is_dir():
        raise FileNotFoundError(f"PaddleOCR model path does not exist: {args.paddleocr_model_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_background_rgb = MASK_BACKGROUND_COLORS[args.mask_background]

    annotations = load_meta(args.meta_json)
    hr_images = collect_image_map(args.hr_dir)
    sr_images = collect_image_map(args.sr_dir)

    # Match by stem, since HR/SR may use different extensions.
    ann_by_stem: dict[str, ImageAnnotationInfo] = {
        Path(info.image_name).stem: info for info in annotations.values()
    }
    image_stems = sorted(ann_by_stem.keys())
    if args.limit_images is not None:
        image_stems = image_stems[: args.limit_images]

    run_config = vars(args).copy()
    run_config["weights_config_path"] = str(args.weights_config)
    run_config["applied_env_overrides"] = applied_env
    run_config["resolved_model_name"] = args.model_name
    run_config["resolved_output_dir"] = str(args.output_dir)
    run_config["text_regions_jsonl_path"] = str(text_regions_path)
    run_config["num_hr_images_found"] = len(hr_images)
    run_config["num_sr_images_found"] = len(sr_images)
    run_config["num_annotated_images"] = len(annotations)
    write_json(args.output_dir / "run_config.json", run_config)

    LOGGER.info(
        "Loaded %d annotations, %d HR images, %d SR images. Loading PaddleOCR-VL on %s (%s)...",
        len(annotations), len(hr_images), len(sr_images), args.device, args.dtype,
    )
    engine = PaddleOCRVLEngine(
        model_path=args.paddleocr_model_path,
        device=args.device,
        dtype=args.dtype,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_pixels=args.max_pixels,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )

    all_rows: list[dict[str, Any]] = []
    n_sr_missing = 0
    n_hr_missing = 0
    iterator = tqdm(image_stems, desc=f"{args.model_name}", dynamic_ncols=True) if tqdm is not None else image_stems
    for stem in iterator:
        annotation_info = ann_by_stem.get(stem)
        if annotation_info is None:
            continue
        sr_path = sr_images.get(stem)
        hr_path = hr_images.get(stem)
        if sr_path is None:
            n_sr_missing += 1
            LOGGER.warning("Missing SR image for %s (skip)", stem)
            continue
        if hr_path is None:
            n_hr_missing += 1
            LOGGER.warning("Missing HR image for %s (skip)", stem)
            continue
        rows = evaluate_one_image(
            model_name=args.model_name,
            image_stem=stem,
            sr_path=sr_path,
            hr_path=hr_path,
            annotation_info=annotation_info,
            engine=engine,
            args=args,
            mask_background_rgb=mask_background_rgb,
        )
        all_rows.extend(rows)

    write_jsonl(text_regions_path, all_rows)
    write_csv(args.output_dir / "text_regions.csv", all_rows)

    n_ok = sum(1 for r in all_rows if r["status"] == "ok")
    n_err = len(all_rows) - n_ok
    n_strict = sum(int(r.get("exact_match_strict", 0)) for r in all_rows)
    n_relaxed = sum(int(r.get("exact_match_ignore_case_whitespace", 0)) for r in all_rows)
    LOGGER.info(
        "Done. %d records (%d ok, %d error). exact_strict=%d (%.2f%%), exact_relaxed=%d (%.2f%%). "
        "SR missing=%d, HR missing=%d.",
        len(all_rows), n_ok, n_err,
        n_strict, 100.0 * n_strict / max(len(all_rows), 1),
        n_relaxed, 100.0 * n_relaxed / max(len(all_rows), 1),
        n_sr_missing, n_hr_missing,
    )


if __name__ == "__main__":
    main()
