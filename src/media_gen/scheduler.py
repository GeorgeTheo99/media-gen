"""A single-GPU FIFO queue: one job runs at a time, in one worker process."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from media_gen.catalog import MediaModel
from media_gen.errors import GenerationFailed, MediaError
from media_gen.memory import MemoryGuard
from media_gen.settings import Settings
from media_gen.worker import WorkerDied, WorkerProcess

log = logging.getLogger(__name__)


class JobCancelled(MediaError):
    def __init__(self) -> None:
        super().__init__(409, "cancelled", "the job was cancelled")


class ShuttingDown(MediaError):
    """The service stopped before the job finished; a stored video resumes at the next start."""

    def __init__(self) -> None:
        super().__init__(503, "shutting_down", "the service is shutting down", retry_after=30)


@dataclass(eq=False)
class Job:
    id: str
    model: MediaModel
    params: dict[str, Any]
    on_start: Callable[[], None] = lambda: None
    on_progress: Callable[[int], None] = lambda value: None
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_running_loop().create_future())
    cancelled: bool = False


def worker_spec(model: MediaModel, models_dir: Path) -> dict[str, Any]:
    return {
        "model": model.id,
        "backend": model.backend,
        "engine": model.engine,
        "paths": [str(m.directory(models_dir)) for m in model.resolved_manifests()],
    }


class Scheduler:
    def __init__(
        self,
        settings: Settings,
        memory: MemoryGuard,
        spec_for: Callable[[MediaModel], dict[str, Any]] | None = None,
    ):
        self._settings = settings
        self._memory = memory
        self._spec_for = spec_for or (lambda model: worker_spec(model, settings.models_dir))
        self._queue: deque[Job] = deque()
        self._wakeup = asyncio.Event()
        self._current: Job | None = None
        self._worker: WorkerProcess | None = None
        self._state = "idle"
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="media-gen-scheduler")

    async def stop(self) -> None:
        # Captured first: cancelling the loop clears the current job.
        pending = [*self._queue, *([self._current] if self._current else [])]
        if self._current and self._worker:
            self._worker.kill()  # its job is abandoned either way; do not wait for it
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        for job in pending:
            if not job.future.done():
                job.future.set_exception(ShuttingDown())
        self._queue.clear()
        await self._release_worker()

    def status(self) -> dict[str, Any]:
        return {
            "state": self._state,
            "loaded_model": self._worker.model_id if self._worker else None,
            "running": self._current.id if self._current else None,
            "queued": len(self._queue),
        }

    def submit(self, job: Job) -> None:
        if len(self._queue) >= self._settings.max_queued_jobs:
            raise MediaError(
                429, "queue_full", "too many queued generations; try again later", retry_after=30
            )
        self._queue.append(job)
        self._wakeup.set()

    def cancel(self, job_id: str) -> bool:
        for job in self._queue:
            if job.id == job_id:
                self._queue.remove(job)
                job.cancelled = True
                if not job.future.done():
                    job.future.set_exception(JobCancelled())
                return True
        if self._current and self._current.id == job_id:
            self._current.cancelled = True
            if self._worker:
                self._worker.kill()
            return True
        return False

    async def _release_worker(self) -> None:
        if self._worker:
            await self._worker.stop()
        self._worker = None

    async def _worker_for(self, model: MediaModel) -> WorkerProcess:
        if self._worker and self._worker.model_id == model.id and self._worker.alive:
            return self._worker
        await self._release_worker()
        self._state = "making_room"
        await self._memory.make_room(model.memory_gib)
        self._state = "loading"
        log.info("loading %s", model.id)
        started = time.monotonic()
        worker = WorkerProcess(model.id, self._spec_for(model))
        await worker.start()
        log.info("loaded %s in %.1fs", model.id, time.monotonic() - started)
        self._worker = worker
        return worker

    async def _run(self, job: Job) -> dict[str, Any]:
        worker = await self._worker_for(job.model)
        if job.cancelled:
            raise JobCancelled()
        self._state = "generating"
        job.on_start()
        started = time.monotonic()
        try:
            result = await worker.run(job.id, job.params, job.on_progress)
        except WorkerDied:
            self._worker = None
            await worker.stop()
            if job.cancelled:
                raise JobCancelled() from None
            raise
        peak = result.get("peak_memory_bytes")
        log.info(
            "%s %s done in %.1fs%s",
            job.model.id,
            job.id,
            time.monotonic() - started,
            f", peak {peak / 2**30:.1f} GiB" if peak else "",
        )
        return result

    async def _loop(self) -> None:
        while True:
            if not self._queue:
                self._state = "idle" if self._worker is None else "ready"
                self._wakeup.clear()
                try:
                    await asyncio.wait_for(self._wakeup.wait(), timeout=self._settings.idle_unload_seconds)
                except TimeoutError:
                    if self._worker:
                        log.info(
                            "unloading %s after %ss idle",
                            self._worker.model_id,
                            self._settings.idle_unload_seconds,
                        )
                        await self._release_worker()
                continue
            job = self._queue.popleft()
            if job.future.done():
                continue
            self._current = job
            try:
                result = await self._run(job)
                if not job.future.done():
                    job.future.set_result(result)
            except MediaError as exc:
                if not job.future.done():
                    job.future.set_exception(exc)
            except Exception as exc:
                log.exception("job %s failed", job.id)
                if not job.future.done():
                    job.future.set_exception(
                        GenerationFailed("internal_error", f"{type(exc).__name__}: {exc}")
                    )
            finally:
                self._current = None
