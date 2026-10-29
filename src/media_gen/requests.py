"""Validate OpenAI-shaped image and video requests into worker parameters."""

from __future__ import annotations

import re
import secrets
from pathlib import Path
from typing import Any

from media_gen.catalog import MediaModel
from media_gen.errors import MediaError, invalid

SIZE = re.compile(r"^(\d{2,5})x(\d{2,5})$")
MAX_PROMPT_CHARS = 8000
MAX_SEED = 2**32 - 1
OUTPUT_FORMATS = ("png", "jpeg", "webp")
# OpenAI image options with no local equivalent, accepted and ignored.
IGNORED_IMAGE_FIELDS = {"quality", "style", "user", "moderation"}
IMAGE_FIELDS = {
    "model",
    "prompt",
    "n",
    "size",
    "response_format",
    "output_format",
    "output_compression",
    "background",
    "seed",
    "steps",
    "guidance",
    *IGNORED_IMAGE_FIELDS,
}
VIDEO_FIELDS = {"model", "prompt", "seconds", "size", "seed", "input_reference"}


def resolve_model(models: dict[str, MediaModel], name: Any, kind: str, models_dir: Path) -> MediaModel:
    if not isinstance(name, str) or not name:
        raise invalid("`model` is required", "model", code="missing_required_parameter")
    model = models.get(name)
    if model is None or model.kind != kind:
        raise MediaError(404, "model_not_found", f"no {kind} model named {name!r}", param="model")
    if not model.is_installed(models_dir):
        raise MediaError(
            404,
            "model_not_installed",
            f"{name} is not installed (run `media-gen pull {name}`)",
            param="model",
        )
    return model


def _prompt(body: dict[str, Any]) -> str:
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise invalid("`prompt` must be a non-empty string", "prompt", code="missing_required_parameter")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise invalid(f"`prompt` is longer than {MAX_PROMPT_CHARS} characters", "prompt")
    return prompt


def _int(body: dict[str, Any], name: str, default: int | None, low: int, high: int) -> int | None:
    value = body.get(name, default)
    if value is None:
        return None
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise invalid(f"`{name}` must be an integer from {low} to {high}", name)
    return value


def _size(body: dict[str, Any], model: MediaModel, *, snap: bool = False) -> tuple[int, int]:
    """The requested size; with ``snap``, each side rounded to the model's multiple."""
    raw = body.get("size", "auto")
    if raw in (None, "auto"):
        return model.default_size
    match = SIZE.match(raw) if isinstance(raw, str) else None
    if not match:
        raise invalid("`size` must be `auto` or WIDTHxHEIGHT, such as 1024x1024", "size")
    width, height = int(match[1]), int(match[2])
    step = model.size_multiple
    if snap:
        width, height = (max(step, round(side / step) * step) for side in (width, height))
    if width % step or height % step:
        raise invalid(f"{model.id} needs a width and height that are multiples of {step}", "size")
    if min(width, height) < 256 or width * height > model.max_pixels:
        raise invalid(
            f"{model.id} supports sizes from 256 px per side up to {model.max_pixels:,} pixels", "size"
        )
    return width, height


def _reject_unknown(body: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise invalid(f"unsupported parameter: {unknown[0]}", unknown[0], code="unknown_parameter")


def _seed(body: dict[str, Any]) -> int:
    seed = _int(body, "seed", None, 0, MAX_SEED)
    return secrets.randbelow(MAX_SEED) if seed is None else seed


def image_params(body: dict[str, Any], model: MediaModel) -> dict[str, Any]:
    _reject_unknown(body, IMAGE_FIELDS)
    if body.get("response_format") not in (None, "b64_json"):
        raise invalid("only `response_format: b64_json` is supported", "response_format")
    if body.get("background") not in (None, "auto", "opaque"):
        raise invalid("transparent backgrounds are not supported", "background")
    output_format = body.get("output_format", "png")
    if output_format not in OUTPUT_FORMATS:
        raise invalid("`output_format` must be png, jpeg or webp", "output_format")
    compression = _int(body, "output_compression", None, 0, 100)
    if compression is not None and output_format == "png":
        raise invalid("`output_compression` applies to jpeg and webp only", "output_compression")
    guidance = body.get("guidance")
    if guidance is not None and (
        isinstance(guidance, bool) or not isinstance(guidance, int | float) or not 0 <= guidance <= 20
    ):
        raise invalid("`guidance` must be a number from 0 to 20", "guidance")
    width, height = _size(body, model)
    return {
        "prompt": _prompt(body),
        "n": _int(body, "n", 1, 1, 4),
        "width": width,
        "height": height,
        "steps": _int(body, "steps", model.default_steps, 1, 100),
        "guidance": guidance,
        "seed": _seed(body),
        "output_format": output_format,
        "output_compression": compression,
    }


def video_params(body: dict[str, Any], model: MediaModel) -> dict[str, Any]:
    _reject_unknown(body, VIDEO_FIELDS)
    seconds = _int(body, "seconds", model.default_seconds, 1, model.max_seconds)
    # Snap to the model's spatial stride (LTX: 64, Wan: 16).
    width, height = _size(body, model, snap=True)
    return {
        "prompt": _prompt(body),
        "seconds": seconds,
        "width": width,
        "height": height,
        "fps": model.video_fps,
        # LTX: 8k+1 at 24 fps. Wan I2V: 4k+1 at 16 fps.
        "num_frames": round(seconds * model.video_fps / model.temporal_multiple) * model.temporal_multiple
        + 1,
        "seed": _seed(body),
    }
