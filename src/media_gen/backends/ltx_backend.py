"""Text- and image-to-video with synchronized audio: LTX-2.3 through mlx-video-with-audio."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx_video.generate_av as av

from media_gen.worker import BackendError

# The unified pack's fixed distilled schedule: 8 stage-one plus 3 stage-two steps.
DENOISE_STEPS = len(av.DEFAULT_STAGE_1_SIGMAS) - 1 + len(av.DEFAULT_STAGE_2_SIGMAS) - 1


class LtxBackend:
    def __init__(self, spec: dict[str, Any]):
        self.model_path, self.text_encoder_path = spec["paths"]
        if not av.is_unified_mlx_model(Path(self.model_path)):
            raise ValueError(f"{self.model_path} is not a unified LTX MLX pack")
        av.validate_text_encoder_config(Path(self.text_encoder_path))

    def run(self, params: dict[str, Any], progress: Callable[[float], None]) -> dict[str, Any]:
        output = Path(params["output_path"])
        # Without ffmpeg the library cannot mux audio, so ask for video only.
        has_audio = shutil.which("ffmpeg") is not None
        steps = 0
        library_tqdm = av.tqdm

        def tracked(iterable, *args, **kwargs):
            nonlocal steps
            for item in library_tqdm(iterable, *args, **kwargs):
                yield item
                steps += 1
                progress(min(95, 95 * steps / DENOISE_STEPS))

        av.tqdm = tracked
        mx.reset_peak_memory()
        try:
            av.generate_video_with_audio(
                model_repo=self.model_path,
                text_encoder_repo=self.text_encoder_path,
                prompt=params["prompt"],
                width=params["width"],
                height=params["height"],
                num_frames=params["num_frames"],
                fps=params["fps"],
                seed=params["seed"],
                image=params.get("image_path"),
                output_path=str(output),
                no_audio=not has_audio,
                verbose=False,
            )
        finally:
            av.tqdm = library_tqdm
            peak = mx.get_peak_memory()
            mx.clear_cache()
        if not output.is_file() or output.stat().st_size == 0:
            raise BackendError("generation_failed", "the video pipeline finished without writing a video")
        return {"has_audio": has_audio, "peak_memory_bytes": peak}
