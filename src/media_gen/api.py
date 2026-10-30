"""HTTP API: OpenAI-compatible image generation and video jobs."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hmac
import io
import logging
import secrets
import shutil
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

from media_gen import __version__
from media_gen.catalog import MODELS, MediaModel
from media_gen.custom import ProfileStore
from media_gen.errors import MediaError, invalid
from media_gen.memory import MemoryGuard
from media_gen.requests import image_params, resolve_model, video_params
from media_gen.scheduler import Job, Scheduler, ShuttingDown
from media_gen.settings import Settings
from media_gen.store import VideoRecord, VideoStore

log = logging.getLogger(__name__)

MAX_REFERENCE_BYTES = 20 * 1024 * 1024
# The reference image plus the form's other fields and multipart framing.
MAX_VIDEO_FORM_BYTES = MAX_REFERENCE_BYTES + 1024 * 1024
RETENTION_SWEEP_SECONDS = 600
DISCONNECT_POLL_SECONDS = 1.0


def _error_response(exc: MediaError) -> JSONResponse:
    headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
    return JSONResponse(exc.body(), status_code=exc.status, headers=headers)


class Service:
    """Shared state for one running app."""

    def __init__(self, settings: Settings, models: dict[str, MediaModel], scheduler: Scheduler):
        self.settings = settings
        self.models = models
        self.scheduler = scheduler
        self.store = VideoStore(settings.database)
        self.custom = ProfileStore(settings.home)
        self._tasks: set[asyncio.Task] = set()

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def served(self) -> dict[str, MediaModel]:
        """The curated models and the owner's enabled custom profiles (read on every call)."""
        custom = {profile_id: model for profile_id, (_, model) in self.custom.served(self.models).items()}
        return {**custom, **self.models}

    def video_dir(self, video_id: str) -> Path:
        return self.settings.outputs_dir / video_id

    # Video jobs -------------------------------------------------------------

    def enqueue_video(self, record: VideoRecord, model: MediaModel) -> None:
        directory = self.video_dir(record.id)
        directory.mkdir(parents=True, exist_ok=True)
        params = {**record.params, "output_path": str(directory / "video.mp4")}
        last = {"progress": record.progress}

        def on_start() -> None:
            self.store.update(record.id, status="in_progress", started_at=int(time.time()))

        def on_progress(value: int) -> None:
            if value != last["progress"]:
                last["progress"] = value
                self.store.update(record.id, progress=value)

        job = Job(id=record.id, model=model, params=params, on_start=on_start, on_progress=on_progress)
        self.scheduler.submit(job)
        self._spawn(self._track_video(job))

    def _finished(self) -> dict[str, int]:
        now = int(time.time())
        return {"completed_at": now, "expires_at": now + self.settings.video_retention_hours * 3600}

    async def _track_video(self, job: Job) -> None:
        try:
            await job.future
        except ShuttingDown:
            return  # left queued or in progress: recover() resumes or fails it at the next start
        except MediaError as exc:
            if self.store.get(job.id) is not None:
                self.store.update(
                    job.id,
                    status="failed",
                    error_code=exc.code,
                    error_message=exc.message,
                    **self._finished(),
                )
            return
        self.store.update(
            job.id,
            status="completed",
            progress=100,
            output_path=job.params["output_path"],
            **self._finished(),
        )

    def recover(self) -> None:
        """Fail videos a restart interrupted and requeue the ones still waiting."""
        for record in self.store.by_status("in_progress"):
            self.store.update(
                record.id,
                status="failed",
                error_code="interrupted",
                error_message="the service restarted while this video was generating",
                **self._finished(),
            )
        for record in self.store.by_status("queued"):
            try:
                model = self.served().get(record.model)
                if model is None:
                    raise MediaError(404, "model_not_found", f"{record.model} is no longer served")
                self.enqueue_video(record, model)
            except MediaError as exc:
                self.store.update(
                    record.id,
                    status="failed",
                    error_code=exc.code,
                    error_message=exc.message,
                    **self._finished(),
                )

    def remove_video(self, video_id: str) -> None:
        shutil.rmtree(self.video_dir(video_id), ignore_errors=True)
        self.store.delete(video_id)

    def sweep_expired(self) -> None:
        for record in self.store.expired():
            self.remove_video(record.id)

    async def retention_loop(self) -> None:
        while True:
            self.sweep_expired()
            await asyncio.sleep(RETENTION_SWEEP_SECONDS)

    async def drain(self) -> None:
        """Let the video trackers settle after the scheduler stopped, before the store closes."""
        await asyncio.gather(*self._tasks, return_exceptions=True)


def create_app(
    settings: Settings,
    models: dict[str, MediaModel] | None = None,
    memory: MemoryGuard | None = None,
    scheduler: Scheduler | None = None,
) -> FastAPI:
    models = MODELS if models is None else models

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings.outputs_dir.mkdir(parents=True, exist_ok=True)
        sched = scheduler or Scheduler(settings, memory or MemoryGuard(settings))
        service = Service(settings, models, sched)
        app.state.service = service
        sched.start()
        service.recover()
        retention = asyncio.create_task(service.retention_loop())
        try:
            yield
        finally:
            retention.cancel()
            await sched.stop()
            await service.drain()
            service.store.close()

    app = FastAPI(title="media-gen", version=__version__, lifespan=lifespan)

    @app.exception_handler(MediaError)
    async def media_error(request: Request, exc: MediaError) -> JSONResponse:
        return _error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(invalid("the request body is not valid"))

    def service(request: Request) -> Service:
        return request.app.state.service

    def authorize(request: Request) -> None:
        if settings.api_key is None:
            return
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(token.encode(), settings.api_key.encode()):
            raise MediaError(401, "invalid_api_key", "a valid bearer key is required")

    async def json_body(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except ValueError:
            raise invalid("the request body must be a JSON object") from None
        if not isinstance(body, dict):
            raise invalid("the request body must be a JSON object")
        return body

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "scheduler": service(request).scheduler.status()}

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def list_models(request: Request) -> dict[str, Any]:
        models_dir = settings.models_dir
        svc = service(request)
        rows = [
            {
                "id": m.id,
                "object": "model",
                "created": 0,
                "owned_by": "media-gen",
                "kind": m.kind,
                "display_name": m.display_name,
                "license": m.license,
                "installed": m.is_installed(models_dir),
                "memory_gib": m.memory_gib,
                "download_bytes": m.download_bytes(),
            }
            for m in svc.models.values()
        ]
        # The owner's enabled profiles, told apart by `custom` and the curated model they build on.
        for profile, model in svc.custom.served(svc.models).values():
            rows.append({
                "id": model.id,
                "object": "model",
                "created": 0,
                "owned_by": "media-gen",
                "kind": model.kind,
                "display_name": model.display_name,
                "license": model.license,
                "installed": model.is_installed(models_dir),
                "memory_gib": model.memory_gib,
                "download_bytes": 0,
                "custom": True,
                "base_model": profile.base,
                "source": profile.source,
            })
        return {"object": "list", "data": rows}

    @app.post("/v1/images/generations", dependencies=[Depends(authorize)])
    async def generate_images(request: Request) -> dict[str, Any]:
        body = await json_body(request)
        svc = service(request)
        model = resolve_model(svc.served(), body.get("model"), "image", settings.models_dir)
        params = image_params(body, model)
        job = Job(id=f"img_{secrets.token_hex(12)}", model=model, params=params)
        svc.scheduler.submit(job)
        while True:
            done, _ = await asyncio.wait({job.future}, timeout=DISCONNECT_POLL_SECONDS)
            if done:
                break
            if await request.is_disconnected():
                # Nobody will read the image: free the GPU for the next job.
                svc.scheduler.cancel(job.id)
                raise MediaError(499, "client_disconnected", "the client disconnected")
        result = job.future.result()
        return {
            "created": int(time.time()),
            "data": [{"b64_json": base64.b64encode(image).decode()} for image in result["images"]],
            "output_format": params["output_format"],
            "size": f"{params['width']}x{params['height']}",
            "seed": result["seed"],
        }

    async def video_request(request: Request) -> tuple[dict[str, Any], bytes | None]:
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("multipart/form-data"):
            length = request.headers.get("content-length", "")
            if length.isdigit() and int(length) > MAX_VIDEO_FORM_BYTES:
                raise MediaError(413, "request_too_large", "the request body is larger than 21 MiB")
            form = await request.form(max_files=1, max_fields=16)
            body: dict[str, Any] = {}
            reference = None
            for key, value in form.multi_items():
                if key == "input_reference":
                    if isinstance(value, str):
                        raise invalid("`input_reference` must be an uploaded image file", "input_reference")
                    reference = await value.read(MAX_REFERENCE_BYTES + 1)
                else:
                    body[key] = value
            return body, reference
        body = await json_body(request)
        if "input_reference" in body:
            raise invalid("send `input_reference` as a multipart file upload", "input_reference")
        return body, None

    @app.post("/v1/videos", dependencies=[Depends(authorize)])
    async def create_video(request: Request) -> dict[str, Any]:
        svc = service(request)
        body, reference = await video_request(request)
        model = resolve_model(svc.served(), body.get("model"), "video", settings.models_dir)
        params = video_params({k: v for k, v in body.items() if k != "input_reference"}, model)
        if model.requires_reference and reference is None:
            raise invalid("this model requires an uploaded `input_reference` image", "input_reference")
        video_id = f"video_{secrets.token_hex(12)}"
        directory = svc.video_dir(video_id)
        directory.mkdir(parents=True)
        try:
            if reference is not None:
                if len(reference) > MAX_REFERENCE_BYTES:
                    raise invalid("`input_reference` is larger than 20 MiB", "input_reference")
                try:
                    with Image.open(io.BytesIO(reference)) as image:
                        image.convert("RGB").save(directory / "input.png")
                except (UnidentifiedImageError, OSError):
                    raise invalid("`input_reference` is not a readable image", "input_reference") from None
                params["image_path"] = str(directory / "input.png")
            record = VideoRecord(
                id=video_id,
                model=model.id,
                status="queued",
                progress=0,
                seconds=params["seconds"],
                size=f"{params['width']}x{params['height']}",
                params=params,
                created_at=int(time.time()),
            )
            svc.store.insert(record)
            try:
                svc.enqueue_video(record, model)
            except MediaError:
                svc.store.delete(video_id)
                raise
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return record.to_api()

    def _video(svc: Service, video_id: str) -> VideoRecord:
        record = svc.store.get(video_id)
        if record is None:
            raise MediaError(404, "video_not_found", f"no video with id {video_id!r}")
        return record

    @app.get("/v1/videos", dependencies=[Depends(authorize)])
    async def list_videos(
        request: Request, limit: int = 20, after: str | None = None, order: str = "desc"
    ) -> dict[str, Any]:
        if not 1 <= limit <= 100:
            raise invalid("`limit` must be from 1 to 100", "limit")
        if order not in ("asc", "desc"):
            raise invalid("`order` must be asc or desc", "order")
        records, has_more = service(request).store.list(limit, after, order)
        data = [r.to_api() for r in records]
        return {
            "object": "list",
            "data": data,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
            "has_more": has_more,
        }

    @app.get("/v1/videos/{video_id}", dependencies=[Depends(authorize)])
    async def get_video(request: Request, video_id: str) -> dict[str, Any]:
        return _video(service(request), video_id).to_api()

    @app.get("/v1/videos/{video_id}/content", dependencies=[Depends(authorize)])
    async def video_content(request: Request, video_id: str, variant: str = "video") -> FileResponse:
        record = _video(service(request), video_id)
        if variant != "video":
            raise invalid("only `variant=video` is available", "variant")
        if record.status != "completed" or not record.output_path or not Path(record.output_path).is_file():
            raise MediaError(409, "video_not_ready", f"video {video_id} is {record.status}")
        return FileResponse(record.output_path, media_type="video/mp4", filename=f"{video_id}.mp4")

    @app.delete("/v1/videos/{video_id}", dependencies=[Depends(authorize)])
    async def delete_video(request: Request, video_id: str) -> dict[str, Any]:
        svc = service(request)
        _video(svc, video_id)
        svc.scheduler.cancel(video_id)
        svc.remove_video(video_id)
        return {"id": video_id, "object": "video.deleted", "deleted": True}

    return app
