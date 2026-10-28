"""One generation process holding one loaded model.

The HTTP server never imports MLX. Each model runs in a spawned child, so
switching or unloading a model releases all of its memory, a cancelled job
stops at once, and a crash (or the kernel's memory killer) cannot take the
service down with it.

Protocol over a Pipe, parent to child: {"type": "run", "params": {...}} and
{"type": "stop"}. Child to parent: {"type": "ready"}, {"type": "progress",
"value": 0-100}, {"type": "done", "result": {...}} and {"type": "error",
"code": ..., "message": ...}.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import multiprocessing
import os
import traceback
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

from media_gen.errors import GenerationFailed

log = logging.getLogger(__name__)

BACKENDS = {
    "mflux": "media_gen.backends.mflux_backend:MfluxBackend",
    "ltx": "media_gen.backends.ltx_backend:LtxBackend",
    "fake": "media_gen.backends.fake:FakeBackend",
}


class BackendError(Exception):
    """Raised by a backend for a failure the client should see verbatim."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _load_backend(spec: dict[str, Any]):
    module_name, class_name = BACKENDS[spec["backend"]].split(":")
    return getattr(importlib.import_module(module_name), class_name)(spec)


def child_main(conn: Connection, spec: dict[str, Any]) -> None:
    # Weights are pinned and already on disk: never let a library fetch more.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        backend = _load_backend(spec)
    except Exception as exc:
        traceback.print_exc()
        conn.send({"type": "error", "code": "model_load_failed", "message": f"{type(exc).__name__}: {exc}"})
        return
    conn.send({"type": "ready"})
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if message["type"] == "stop":
            return
        try:
            result = backend.run(
                message["params"], lambda value: conn.send({"type": "progress", "value": int(value)})
            )
            conn.send({"type": "done", "result": result})
        except BackendError as exc:
            conn.send({"type": "error", "code": exc.code, "message": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            conn.send(
                {"type": "error", "code": "generation_failed", "message": f"{type(exc).__name__}: {exc}"}
            )


class WorkerDied(GenerationFailed):
    def __init__(self, exitcode: int | None):
        hint = " (killed: most likely out of memory)" if exitcode == -9 else ""
        super().__init__("worker_crashed", f"the generation process exited with code {exitcode}{hint}")
        self.exitcode = exitcode


class WorkerProcess:
    """Parent-side handle. Not thread-safe: the scheduler drives it from one task."""

    def __init__(self, model_id: str, spec: dict[str, Any]):
        self.model_id = model_id
        self._spec = spec
        ctx = multiprocessing.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._process = ctx.Process(
            target=child_main, args=(child, spec), name=f"media-gen-{model_id}", daemon=True
        )
        self._child = child

    @property
    def alive(self) -> bool:
        return self._process.is_alive()

    @property
    def pid(self) -> int | None:
        return self._process.pid

    def _recv_blocking(self) -> dict[str, Any]:
        while True:
            if self._conn.poll(0.25):
                try:
                    return self._conn.recv()
                except (EOFError, OSError):
                    self._process.join(5)
                    raise WorkerDied(self._process.exitcode) from None
            if not self._process.is_alive():
                if self._conn.poll(0):
                    continue
                raise WorkerDied(self._process.exitcode)

    async def _recv(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._recv_blocking)

    async def start(self) -> None:
        self._process.start()
        self._child.close()
        message = await self._recv()
        if message["type"] == "error":
            await self.stop()
            raise GenerationFailed(message["code"], message["message"])
        if message["type"] != "ready":
            raise GenerationFailed("worker_protocol", f"unexpected message {message['type']!r}")

    async def run(self, params: dict[str, Any], on_progress: Callable[[int], None]) -> dict[str, Any]:
        self._conn.send({"type": "run", "params": params})
        while True:
            message = await self._recv()
            kind = message["type"]
            if kind == "progress":
                on_progress(message["value"])
            elif kind == "done":
                return message["result"]
            elif kind == "error":
                raise GenerationFailed(message["code"], message["message"])

    def kill(self) -> None:
        if self._process.is_alive():
            self._process.kill()

    async def stop(self) -> None:
        if self._process.is_alive():
            with contextlib.suppress(BrokenPipeError, OSError):
                self._conn.send({"type": "stop"})
            await asyncio.to_thread(self._process.join, 15)
            if self._process.is_alive():
                log.warning("worker %s did not stop; killing it", self.model_id)
                self._process.kill()
                await asyncio.to_thread(self._process.join, 5)
        self._conn.close()
