from __future__ import annotations

import time
from dataclasses import replace

from tests.conftest import wait_for_video


def test_idle_worker_is_unloaded(make_client, settings):
    client = make_client(settings=replace(settings, idle_unload_seconds=1))
    assert (
        client.post("/v1/images/generations", json={"model": "fake-image", "prompt": "x"}).status_code == 200
    )
    worker = client.app.state.service.scheduler._worker
    deadline = time.monotonic() + 10
    while client.get("/health").json()["scheduler"]["loaded_model"] is not None:
        assert time.monotonic() < deadline
        time.sleep(0.1)
    assert not worker.alive


def test_deleting_a_queued_video_never_runs_it(client):
    running = client.post("/v1/videos", json={"model": "fake-video", "prompt": "sleep:1"}).json()
    queued = client.post("/v1/videos", json={"model": "fake-video", "prompt": "fail"}).json()
    assert client.get("/health").json()["scheduler"]["queued"] == 1
    client.delete(f"/v1/videos/{queued['id']}")
    assert client.get("/health").json()["scheduler"]["queued"] == 0
    assert wait_for_video(client, running["id"])["status"] == "completed"
