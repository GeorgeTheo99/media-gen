"""Pinned MLX-Video Wan 2.2 I2V adapter, with offline tokenizer and verified MP4 output."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from media_gen.worker import BackendError


class WanBackend:
    def __init__(self, spec: dict[str, Any]) -> None:
        self.model_path, tokenizer_path = map(Path, spec["paths"])
        self.engine = spec["engine"]
        self.ffmpeg = shutil.which("ffmpeg")
        self.ffprobe = shutil.which("ffprobe")
        if not self.ffmpeg or not self.ffprobe:
            raise BackendError("missing_dependency", "Wan video generation requires ffmpeg and ffprobe")

        # This pinned engine hardcodes google/umt5-xxl. Give it a private, pinned HF cache
        # before importing Hugging Face libraries; never resolve a moving remote revision.
        # Keep the cache outside the snapshot it links to (no recursive directory graph).
        cache = tokenizer_path.parent / f".{tokenizer_path.name}.wan-hub"
        repo = cache / "models--google--umt5-xxl"
        snapshot = repo / "snapshots" / tokenizer_path.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists():
            snapshot.symlink_to(tokenizer_path, target_is_directory=True)
        (repo / "refs").mkdir(exist_ok=True)
        (repo / "refs" / "main").write_text(tokenizer_path.name)
        os.environ["HF_HUB_CACHE"] = str(cache)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        from mlx_video.models.wan_2.generate import generate_video

        self.generate = generate_video

    def _finish_video(self, raw: Path, output: Path, params: dict[str, Any]) -> None:
        if not raw.is_file() or raw.stat().st_size == 0:
            raise BackendError("generation_failed", "Wan did not write a video")
        probe = subprocess.run(
            [
                self.ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames,r_frame_rate",
                "-of",
                "json",
                str(raw),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        streams = json.loads(probe.stdout).get("streams", [])
        if len(streams) != 1:
            raise BackendError("generation_failed", "Wan output has no readable video stream")
        stream = streams[0]
        frames = int(stream.get("nb_frames", 0))
        fps_num, fps_den = map(int, stream["r_frame_rate"].split("/"))
        if (
            stream["width"] != params["width"]
            or stream["height"] != params["height"]
            or fps_num != params["fps"] * fps_den
            or frames not in (params["num_frames"], params["num_frames"] + 3)
        ):
            raise BackendError(
                "generation_failed", "Wan output dimensions, frame rate or duration are incorrect"
            )
        # Upstream's non-causal VAE decode emits 4*T frames for 4*T-3 requested.
        # Retain the requested prefix, rather than silently returning a longer clip.
        if frames != params["num_frames"]:
            subprocess.run(
                [
                    self.ffmpeg,
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(raw),
                    "-map",
                    "0:v:0",
                    "-frames:v",
                    str(params["num_frames"]),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output),
                ],
                check=True,
                capture_output=True,
                timeout=300,
            )
            raw.unlink()
        else:
            raw.replace(output)
        if not output.is_file() or output.stat().st_size == 0:
            raise BackendError("generation_failed", "Wan did not write a playable video")

    def run(self, params: dict[str, Any], progress) -> dict[str, Any]:
        import mlx.core as mx

        if not params.get("image_path"):
            raise BackendError("invalid_request", "Wan 2.2 I2V requires a reference image")
        output = Path(params["output_path"])
        raw = output.with_name("wan-untrimmed.mp4")
        progress(5)
        try:
            mx.reset_peak_memory()
            self.generate(
                model_dir=str(self.model_path),
                prompt=params["prompt"],
                image=params["image_path"],
                width=params["width"],
                height=params["height"],
                num_frames=params["num_frames"],
                steps=self.engine["steps"],
                guide_scale=(3.5, 3.5),
                shift=5.0,
                seed=params["seed"],
                output_path=str(raw),
                tiling="aggressive",
                # Avoid compiled closures retaining both experts during VAE decoding.
                no_compile=True,
                trim_first_frames=0,
            )
            progress(95)
            self._finish_video(raw, output, params)
        finally:
            raw.unlink(missing_ok=True)
        progress(100)
        return {"seed": params["seed"], "peak_memory_bytes": mx.get_peak_memory()}
