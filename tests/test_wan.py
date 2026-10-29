"""Wan contracts without downloading weights or running an MLX inference."""

import io
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from media_gen.api import create_app
from media_gen.backends.wan_backend import WanBackend
from media_gen.catalog import MODELS
from media_gen.errors import MediaError
from media_gen.requests import video_params
from media_gen.worker import BackendError

from .conftest import FAKE_MODELS, wait_for_video


def test_wan_manifest_and_frame_contract():
    model = MODELS["wan2.2-i2v-a14b"]
    manifests = model.resolved_manifests()
    assert len(manifests) == 2
    assert 42_000_000_000 < model.download_bytes() < 44_000_000_000
    assert model.requires_reference
    for seconds, frames in [(1, 17), (3, 49), (5, 81)]:
        params = video_params({"prompt": "A bird flies", "seconds": seconds}, model)
        assert params["fps"] == 16
        assert params["num_frames"] == frames
    with pytest.raises(MediaError):
        video_params({"prompt": "A bird flies", "seconds": 6}, model)
    ltx = video_params({"prompt": "A bird flies", "seconds": 3}, MODELS["ltx-2.5"])
    assert ltx["fps"] == 24 and ltx["num_frames"] == 73


def test_i2v_requires_reference_before_enqueue(settings, plenty_of_memory):
    models = dict(FAKE_MODELS)
    models["fake-wan"] = replace(
        models["fake-video"],
        id="fake-wan",
        requires_reference=True,
        video_fps=16,
        temporal_multiple=4,
    )
    with TestClient(create_app(settings, models=models, memory=plenty_of_memory)) as client:
        response = client.post("/v1/videos", json={"model": "fake-wan", "prompt": "A bird flies"})
        assert response.status_code == 400
        assert response.json()["error"]["param"] == "input_reference"
        assert client.get("/v1/videos").json()["data"] == []
        assert list(settings.outputs_dir.iterdir()) == []
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), "blue").save(buf, format="PNG")
        response = client.post(
            "/v1/videos",
            data={"model": "fake-wan", "prompt": "A bird flies", "seconds": "3"},
            files={"input_reference": ("bird.png", buf.getvalue(), "image/png")},
        )
        assert response.status_code == 200
        job_id = response.json()["id"]
        assert wait_for_video(client, job_id)["status"] == "completed"
        record = client.app.state.service.store.get(job_id)
        assert record.params["num_frames"] == 49
        assert record.params["fps"] == 16


def test_backend_uses_pinned_offline_tokenizer(tmp_path, monkeypatch):
    tokenizer = tmp_path / ("a" * 40)
    tokenizer.mkdir()
    fake_generate = Mock()
    monkeypatch.setitem(
        sys.modules, "mlx_video.models.wan_2.generate", SimpleNamespace(generate_video=fake_generate)
    )
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    # Register environment changes with pytest so constructor assignments are restored.
    for key in ("HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "before")
    backend = WanBackend({"paths": [str(tmp_path / "model"), str(tokenizer)], "engine": {"steps": 40}})
    import os

    cache = Path(os.environ["HF_HUB_CACHE"])
    assert cache.parent == tokenizer.parent
    assert not cache.is_relative_to(tokenizer)
    repo = cache / "models--google--umt5-xxl"
    assert (repo / "refs/main").read_text() == tokenizer.name
    assert (repo / "snapshots" / tokenizer.name).resolve() == tokenizer
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert backend.generate is fake_generate


def test_backend_requires_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(BackendError, match="ffmpeg and ffprobe"):
        WanBackend({"paths": [str(tmp_path), str(tmp_path)], "engine": {}})


def test_run_contract_and_cleanup(tmp_path, monkeypatch):
    mx = SimpleNamespace(reset_peak_memory=Mock(), get_peak_memory=lambda: 123)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    backend = WanBackend.__new__(WanBackend)
    backend.model_path = tmp_path / "model"
    backend.engine = {"steps": 40}
    backend.generate = Mock()
    backend._finish_video = Mock()
    params = {
        "prompt": "A bird flies",
        "image_path": "bird.png",
        "output_path": str(tmp_path / "video.mp4"),
        "seed": 42,
        "width": 832,
        "height": 480,
        "num_frames": 81,
        "fps": 16,
    }
    updates = []
    result = backend.run(params, updates.append)
    assert updates == [5, 95, 100]
    kwargs = backend.generate.call_args.kwargs
    assert kwargs["image"] == "bird.png"
    assert kwargs["steps"] == 40
    assert kwargs["guide_scale"] == (3.5, 3.5)
    assert kwargs["no_compile"] is True and kwargs["trim_first_frames"] == 0
    assert kwargs["num_frames"] == 81
    assert result == {"seed": 42, "peak_memory_bytes": 123}
    with pytest.raises(BackendError, match="requires a reference"):
        backend.run({**params, "image_path": None}, Mock())

    raw = tmp_path / "wan-untrimmed.mp4"
    raw.write_bytes(b"incomplete")
    backend.generate.side_effect = RuntimeError("inference failed")
    with pytest.raises(RuntimeError, match="inference failed"):
        backend.run(params, Mock())
    assert not raw.exists()


@pytest.mark.parametrize("overrides", [{"nb_frames": "10"}, {"width": 512}, {"r_frame_rate": "24/1"}])
def test_reject_incorrect_video(tmp_path, monkeypatch, overrides):
    backend = WanBackend.__new__(WanBackend)
    backend.ffprobe = "ffprobe"
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"video")
    stream = {"width": 256, "height": 256, "nb_frames": "20", "r_frame_rate": "16/1", **overrides}
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=json.dumps({"streams": [stream]}))
    )
    with pytest.raises(BackendError, match="incorrect"):
        backend._finish_video(
            raw, tmp_path / "output.mp4", {"width": 256, "height": 256, "fps": 16, "num_frames": 17}
        )


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="requires ffmpeg/ffprobe"
)
@pytest.mark.parametrize("source_frames", [17, 20])
def test_real_mp4_frame_contract(tmp_path, source_frames):
    backend = WanBackend.__new__(WanBackend)
    backend.ffmpeg, backend.ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    raw, output = tmp_path / "raw.mp4", tmp_path / "output.mp4"
    subprocess.run(
        [
            backend.ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=256x256:r=16",
            "-frames:v",
            str(source_frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(raw),
        ],
        check=True,
    )
    backend._finish_video(raw, output, {"width": 256, "height": 256, "fps": 16, "num_frames": 17})
    assert not raw.exists()
    probe = subprocess.run(
        [
            backend.ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=nb_frames,width,height,r_frame_rate",
            "-of",
            "json",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream == {"width": 256, "height": 256, "r_frame_rate": "16/1", "nb_frames": "17"}
