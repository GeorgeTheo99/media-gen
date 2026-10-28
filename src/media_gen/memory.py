"""Make room in unified memory before loading a model.

oMLX on the same Mac may hold hundreds of GiB of language models, and macOS
kills a process rather than swap a GPU working set. Before a load, media-gen
checks free memory and, when oMLX is configured, unloads its least recently
used idle models until the new model fits. Models that are pinned, loading
or used within `omlx_busy_seconds` are never touched.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from urllib.parse import quote

import httpx
import psutil

from media_gen.errors import InsufficientMemory
from media_gen.settings import Settings

log = logging.getLogger(__name__)

GIB = 1 << 30


def available_bytes() -> int:
    return psutil.virtual_memory().available


class MemoryGuard:
    def __init__(
        self,
        settings: Settings,
        available: Callable[[], int] = available_bytes,
        transport: httpx.AsyncBaseTransport | None = None,
        settle_seconds: float = 20,
    ):
        self._settings = settings
        self._available = available
        self._transport = transport
        self._settle_seconds = settle_seconds

    def _client(self) -> httpx.AsyncClient:
        headers = {}
        if self._settings.omlx_api_key:
            headers["Authorization"] = f"Bearer {self._settings.omlx_api_key}"
        return httpx.AsyncClient(
            base_url=self._settings.omlx_url or "",
            headers=headers,
            timeout=30,
            transport=self._transport,
        )

    async def _idle_omlx_models(self, client: httpx.AsyncClient) -> list[dict]:
        response = await client.get("/v1/models/status")
        response.raise_for_status()
        cutoff = time.time() - self._settings.omlx_busy_seconds
        idle = [
            m
            for m in response.json().get("models", [])
            if m.get("loaded")
            and not m.get("is_loading")
            and not m.get("pinned")
            and (m.get("last_access") or 0) < cutoff
        ]
        return sorted(idle, key=lambda m: m.get("last_access") or 0)

    async def _wait_for(self, need: int) -> bool:
        deadline = time.monotonic() + self._settle_seconds
        while time.monotonic() < deadline:
            if self._available() >= need:
                return True
            await asyncio.sleep(0.5)
        return self._available() >= need

    async def make_room(self, model_gib: int) -> None:
        need = (model_gib + self._settings.memory_reserve_gib) * GIB
        if self._available() >= need:
            return
        shortfall = f"needs {need / GIB:.0f} GiB free, {self._available() / GIB:.0f} GiB available"
        if not self._settings.omlx_url:
            raise InsufficientMemory(f"not enough memory to load the model: {shortfall}")
        try:
            async with self._client() as client:
                for model in await self._idle_omlx_models(client):
                    log.info("unloading idle oMLX model %s to make room", model["id"])
                    response = await client.post(f"/v1/models/{quote(model['id'], safe='')}/unload")
                    if response.status_code not in (200, 202):
                        log.warning("oMLX refused to unload %s: HTTP %s", model["id"], response.status_code)
                        continue
                    if await self._wait_for(need):
                        return
        except httpx.HTTPError as exc:
            log.warning("could not reach oMLX to free memory: %s", exc)
        raise InsufficientMemory(
            f"not enough memory to load the model ({shortfall}) and no idle oMLX model is left to unload"
        )
