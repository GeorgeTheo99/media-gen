"""A dependency-free backend for tests: solid-colour images and a stub video file."""

from __future__ import annotations

import io
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from media_gen.worker import BackendError


class FakeBackend:
    def __init__(self, spec: dict[str, Any]):
        engine = spec.get("engine", {})
        if engine.get("fail_load"):
            raise RuntimeError("fake load failure")
        time.sleep(engine.get("load_seconds", 0))
        self.pid = os.getpid()

    def run(self, params: dict[str, Any], progress: Callable[[float], None]) -> dict[str, Any]:
        behaviour = params.get("prompt", "")
        if behaviour == "crash":
            os._exit(3)
        if behaviour == "fail":
            raise BackendError("bad_prompt", "fake failure")
        delay = float(behaviour.removeprefix("sleep:")) if behaviour.startswith("sleep:") else 0
        for step in range(4):
            time.sleep(delay / 4)
            progress((step + 1) * 25)
        if "output_path" in params:
            Path(params["output_path"]).write_bytes(b"\x00\x00\x00\x18ftypmp42fake-video")
            return {"has_audio": False, "worker_pid": self.pid}
        # An edit's green channel counts its references, each read as the real backend would.
        references = [Image.open(path).size for path in params.get("image_paths", [])]
        images = []
        for i in range(params["n"]):
            buffer = io.BytesIO()
            colour = ((params["seed"] + i) * 37) % 256
            image = Image.new("RGB", (params["width"], params["height"]), (colour, len(references), 0))
            image.save(buffer, format=params["output_format"].upper().replace("JPG", "JPEG"))
            images.append(buffer.getvalue())
        return {"images": images, "seed": params["seed"], "worker_pid": self.pid}
