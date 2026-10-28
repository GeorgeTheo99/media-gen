"""Text- and image-to-video with synchronized audio: LTX-2.5 distilled through ltx-2-mlx."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import ltx_pipelines_mlx.utils.samplers as samplers
import mlx.core as mx
from ltx_pipelines_mlx.distilled import DistilledPipeline

from media_gen.worker import BackendError

# Share of the progress bar for each denoising stage: half resolution, then the full-size refine.
STAGES = ((0, 70), (70, 95))
# The VAE decode's memory budget. The library's default (half of unified memory) skips tiling on a big Mac,
# so a 10 s clip peaked at 150 GiB; a tiled decode fits under this instead.
DECODE_BUDGET_GIB = 24


class LtxBackend:
    def __init__(self, spec: dict[str, Any]):
        # Read when the decode runs; this is the worker's own process.
        os.environ.setdefault("LTX2_VAE_DECODE_BUDGET_GB", str(DECODE_BUDGET_GIB))
        (self.model_path,) = spec["paths"]
        # Without its bundled text encoder the library would quietly download a Gemma instead.
        needed = ("transformer-distilled.safetensors", "text_encoder.safetensors", "text_encoder_config.json")
        missing = [name for name in needed if not (Path(self.model_path) / name).is_file()]
        if missing:
            raise ValueError(f"{self.model_path} is an incomplete LTX MLX pack: missing {', '.join(missing)}")

    def run(self, params: dict[str, Any], progress: Callable[[float], None]) -> dict[str, Any]:
        output = Path(params["output_path"])
        # Without ffmpeg the library cannot mux audio, so ask for video only.
        has_audio = shutil.which("ffmpeg") is not None
        stage = 0
        # The library resets MLX's peak before its tiled decode, so each step's is kept too.
        peak = 0
        library_tqdm = samplers.tqdm

        def tracked(iterable, *args, **kwargs):
            nonlocal stage, peak
            start, end = STAGES[min(stage, len(STAGES) - 1)]
            stage += 1
            steps = list(iterable)
            for index, item in enumerate(library_tqdm(steps, *args, **kwargs)):
                yield item
                peak = max(peak, mx.get_peak_memory())
                progress(start + (end - start) * (index + 1) / len(steps))

        # The pack bundles its own Gemma text encoder, so nothing else is read.
        pipe = DistilledPipeline(model_dir=self.model_path, low_memory=True)
        pipe.verbose = False
        pipe.generate_audio = has_audio
        samplers.tqdm = tracked
        mx.reset_peak_memory()
        try:
            pipe.generate_and_save(
                prompt=params["prompt"],
                output_path=str(output),
                height=params["height"],
                width=params["width"],
                num_frames=params["num_frames"],
                frame_rate=params["fps"],
                seed=params["seed"],
                image=params.get("image_path"),
            )
        finally:
            samplers.tqdm = library_tqdm
            peak = max(peak, mx.get_peak_memory())
            del pipe
            mx.clear_cache()
        if not output.is_file() or output.stat().st_size == 0:
            raise BackendError("generation_failed", "the video pipeline finished without writing a video")
        return {"has_audio": has_audio, "peak_memory_bytes": peak}
