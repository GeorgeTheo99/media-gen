"""Text-to-image through mflux (MLX)."""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import mlx.core as mx
from mflux.models.common.config.model_config import ModelConfig


def _model_class(family: str):
    if family == "flux3":
        from mflux.models.flux3.variants.txt2img.flux3_klein import Flux3Klein

        return Flux3Klein
    if family == "qwen21":
        from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

        return QwenImage21
    raise ValueError(f"unsupported mflux family {family!r}")


class _Progress:
    """mflux in-loop callback; `report` is swapped per image."""

    def __init__(self) -> None:
        self.report: Callable[[int, int], None] = lambda step, total: None

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps) -> None:
        self.report(t + 1, config.num_inference_steps)


def encode(image, output_format: str, compression: int | None) -> bytes:
    buffer = io.BytesIO()
    if output_format == "png":
        image.save(buffer, format="PNG")
    else:
        quality = 90 if compression is None else compression
        image.convert("RGB").save(buffer, format=output_format.upper(), quality=quality)
    return buffer.getvalue()


class MfluxBackend:
    def __init__(self, spec: dict[str, Any]):
        engine = spec["engine"]
        cls = _model_class(engine["family"])
        self.model = cls(
            model_config=ModelConfig.from_name(engine["mflux_model"]),
            quantize=engine.get("quantize"),
            model_path=spec["paths"][0],
        )
        self.progress = _Progress()
        self.model.callbacks.register(self.progress)

    def run(self, params: dict[str, Any], progress: Callable[[float], None]) -> dict[str, Any]:
        n = params["n"]
        images = []
        mx.reset_peak_memory()
        for index in range(n):
            self.progress.report = lambda step, total, i=index: progress(100 * (i + step / total) / n)
            options = {} if params.get("guidance") is None else {"guidance": params["guidance"]}
            generated = self.model.generate_image(
                seed=params["seed"] + index,
                prompt=params["prompt"],
                num_inference_steps=params["steps"],
                width=params["width"],
                height=params["height"],
                **options,
            )
            images.append(encode(generated.image, params["output_format"], params.get("output_compression")))
            mx.clear_cache()
        return {"images": images, "seed": params["seed"], "peak_memory_bytes": mx.get_peak_memory()}
