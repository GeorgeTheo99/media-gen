from __future__ import annotations

import asyncio
import time
from dataclasses import replace

import httpx
import pytest

from media_gen.errors import InsufficientMemory
from media_gen.memory import MemoryGuard

GIB = 1 << 30


class FakeMac:
    """Free memory plus an oMLX that frees a model's size when it unloads it."""

    def __init__(self, free_gib: int, models: list[dict]):
        self.free = free_gib * GIB
        self.models = {m["id"]: m for m in models}
        self.unloaded: list[str] = []

    def available(self) -> int:
        return self.free

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer omlx-key"
        if request.url.path == "/v1/models/status":
            return httpx.Response(200, json={"models": list(self.models.values())})
        model_id = request.url.path.split("/")[3]
        model = self.models[model_id]
        if model.get("refuse"):
            return httpx.Response(409)
        model["loaded"] = False
        self.unloaded.append(model_id)
        self.free += model["actual_size"]
        return httpx.Response(200, json={"status": "ok"})


def omlx_model(model_id: str, gib: int, idle_seconds: int, **extra) -> dict:
    return {
        "id": model_id,
        "loaded": True,
        "is_loading": False,
        "pinned": False,
        "actual_size": gib * GIB,
        "last_access": time.time() - idle_seconds,
        **extra,
    }


@pytest.fixture
def omlx_settings(settings):
    return replace(settings, omlx_url="http://omlx.test", omlx_api_key="omlx-key", memory_reserve_gib=8)


def guard(settings, mac: FakeMac) -> MemoryGuard:
    return MemoryGuard(
        settings, available=mac.available, transport=httpx.MockTransport(mac.handler), settle_seconds=0.2
    )


def test_enough_memory_touches_nothing(omlx_settings):
    mac = FakeMac(100, [omlx_model("big", 300, 3600)])
    asyncio.run(guard(omlx_settings, mac).make_room(40))
    assert mac.unloaded == []


def test_unloads_least_recently_used_idle_models_until_it_fits(omlx_settings):
    mac = FakeMac(
        10,
        [
            omlx_model("recent-idle", 50, 900),
            omlx_model("oldest", 30, 7200),
            omlx_model("busy", 200, 10),
            omlx_model("pinned", 200, 7200, pinned=True),
            omlx_model("loading", 200, 7200, is_loading=True),
            omlx_model("unused", 200, 7200, loaded=False),
        ],
    )
    asyncio.run(guard(omlx_settings, mac).make_room(80))  # needs 88 GiB
    assert mac.unloaded == ["oldest", "recent-idle"]


def test_refused_unload_is_skipped(omlx_settings):
    mac = FakeMac(10, [omlx_model("stuck", 100, 7200, refuse=True), omlx_model("ok", 100, 3600)])
    asyncio.run(guard(omlx_settings, mac).make_room(40))
    assert mac.unloaded == ["ok"]


def test_raises_when_only_busy_models_remain(omlx_settings):
    mac = FakeMac(10, [omlx_model("busy", 300, 5)])
    with pytest.raises(InsufficientMemory) as caught:
        asyncio.run(guard(omlx_settings, mac).make_room(40))
    assert caught.value.status == 503
    assert caught.value.retry_after
    assert mac.unloaded == []


def test_without_omlx_it_only_checks(settings):
    with pytest.raises(InsufficientMemory):
        asyncio.run(MemoryGuard(settings, available=lambda: 4 * GIB).make_room(40))


def test_unreachable_omlx_is_insufficient_memory(omlx_settings):
    def down(request):
        raise httpx.ConnectError("refused")

    memory = MemoryGuard(omlx_settings, available=lambda: GIB, transport=httpx.MockTransport(down))
    with pytest.raises(InsufficientMemory):
        asyncio.run(memory.make_room(40))


def test_insufficient_memory_reaches_the_client_as_503(make_client, settings):
    client = make_client(memory=MemoryGuard(settings, available=lambda: 0))
    response = client.post("/v1/images/generations", json={"model": "fake-image", "prompt": "x"})
    assert response.status_code == 503
    assert response.headers["retry-after"] == "60"
    assert response.json()["error"]["code"] == "insufficient_memory"
