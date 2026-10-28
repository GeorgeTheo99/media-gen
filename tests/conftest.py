from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from media_gen.api import create_app
from media_gen.catalog import MediaModel
from media_gen.memory import MemoryGuard
from media_gen.settings import Settings

GIB = 1 << 30


def fake_model(model_id: str, kind: str = "image", **overrides) -> MediaModel:
    base = MediaModel(
        id=model_id,
        kind=kind,  # type: ignore[arg-type]
        backend="fake",
        display_name=model_id,
        license="test",
        manifests=(),
        memory_gib=1,
        default_size=(256, 256),
        size_multiple=16 if kind == "image" else 64,
        default_steps=2,
        max_seconds=10 if kind == "video" else 0,
        default_seconds=2 if kind == "video" else 0,
    )
    return replace(base, **overrides)


FAKE_MODELS = {
    m.id: m
    for m in (
        fake_model("fake-image"),
        fake_model("fake-image-2"),
        fake_model("fake-broken", engine={"fail_load": True}),
        fake_model("fake-video", kind="video"),
    )
}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never reach a real host: every transport must be injected."""

    def refuse(self, request):
        raise AssertionError(f"test tried to reach {request.url}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(home=tmp_path / "home", idle_unload_seconds=60, max_queued_jobs=4)


@pytest.fixture
def plenty_of_memory(settings: Settings) -> MemoryGuard:
    return MemoryGuard(settings, available=lambda: 1024 * GIB)


@pytest.fixture
def make_client(settings: Settings, plenty_of_memory: MemoryGuard) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def make(settings: Settings = settings, models=FAKE_MODELS, memory=plenty_of_memory) -> TestClient:
        client = TestClient(create_app(settings, models=models, memory=memory))
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for client in reversed(clients):
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


def wait_for_video(client: TestClient, video_id: str, timeout: float = 30) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        video = client.get(f"/v1/videos/{video_id}").json()
        if video["status"] in ("completed", "failed"):
            return video
        time.sleep(0.1)
    raise AssertionError(f"video {video_id} did not finish: {video}")
