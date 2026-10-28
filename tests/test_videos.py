from __future__ import annotations

import io
import time

from PIL import Image

from media_gen.store import VideoRecord, VideoStore
from tests.conftest import wait_for_video


def create(client, **body):
    return client.post("/v1/videos", json={"model": "fake-video", "prompt": "waves", **body})


def test_video_job_lifecycle(client):
    response = create(client, seconds="3", size="512x256", seed=1)
    assert response.status_code == 200, response.text
    video = response.json()
    assert video["object"] == "video"
    assert video["status"] == "queued"
    assert video["seconds"] == "3"
    assert video["size"] == "512x256"
    assert video["id"].startswith("video_")

    done = wait_for_video(client, video["id"])
    assert done["status"] == "completed"
    assert done["progress"] == 100
    assert done["expires_at"] > done["completed_at"]
    content = client.get(f"/v1/videos/{video['id']}/content")
    assert content.status_code == 200
    assert content.headers["content-type"] == "video/mp4"
    assert content.content.startswith(b"\x00\x00\x00\x18ftyp")
    record = client.app.state.service.store.get(video["id"])
    assert record.params["num_frames"] == 73  # 3 s at 24 fps, rounded to 8k + 1


def test_content_is_unavailable_until_completed(client):
    video = create(client, prompt="sleep:2").json()
    response = client.get(f"/v1/videos/{video['id']}/content")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "video_not_ready"
    assert client.get(f"/v1/videos/{video['id']}/content?variant=thumbnail").status_code == 400


def test_failed_video_reports_the_error(client):
    video = create(client, prompt="fail").json()
    done = wait_for_video(client, video["id"])
    assert done["status"] == "failed"
    assert done["error"] == {"code": "bad_prompt", "message": "fake failure"}


def test_multipart_with_input_reference(client):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "blue").save(buffer, format="JPEG")
    response = client.post(
        "/v1/videos",
        data={"model": "fake-video", "prompt": "animate", "seconds": "2"},
        files={"input_reference": ("frame.jpg", buffer.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 200, response.text
    video_id = response.json()["id"]
    params = client.app.state.service.store.get(video_id).params
    assert params["image_path"].endswith(f"{video_id}/input.png")
    assert wait_for_video(client, video_id)["status"] == "completed"

    bad = client.post(
        "/v1/videos",
        data={"model": "fake-video", "prompt": "animate"},
        files={"input_reference": ("frame.jpg", b"not an image", "image/jpeg")},
    )
    assert bad.status_code == 400
    assert bad.json()["error"]["param"] == "input_reference"


def test_validation(client):
    assert create(client, seconds="11").json()["error"]["param"] == "seconds"
    assert create(client, size="100x100").json()["error"]["param"] == "size"
    assert create(client, size="4096x4096").json()["error"]["param"] == "size"
    # OpenAI's video sizes snap to the model's multiple of 64.
    assert create(client, size="720x480").json()["size"] == "704x512"
    assert create(client, input_reference="https://example.com/a.png").status_code == 400
    assert create(client, model="fake-image").status_code == 404


def test_list_paginates_newest_first(client):
    ids = [create(client).json()["id"] for _ in range(3)]
    for video_id in ids:
        wait_for_video(client, video_id)
    page = client.get("/v1/videos?limit=2").json()
    assert page["has_more"] is True
    rest = client.get(f"/v1/videos?limit=2&after={page['last_id']}").json()
    assert rest["has_more"] is False
    assert {v["id"] for v in page["data"] + rest["data"]} == set(ids)
    assert client.get("/v1/videos?order=sideways").status_code == 400


def test_delete_cancels_a_running_video(client):
    video = create(client, prompt="sleep:20").json()
    deadline = time.monotonic() + 15
    while client.get(f"/v1/videos/{video['id']}").json()["status"] != "in_progress":
        assert time.monotonic() < deadline
        time.sleep(0.1)
    response = client.delete(f"/v1/videos/{video['id']}")
    assert response.json() == {"id": video["id"], "object": "video.deleted", "deleted": True}
    assert client.get(f"/v1/videos/{video['id']}").status_code == 404
    # The killed worker is replaced on the next job.
    assert wait_for_video(client, create(client).json()["id"])["status"] == "completed"


def test_restart_fails_interrupted_and_requeues_waiting(make_client, settings):
    store = VideoStore(settings.database)
    params = {
        "prompt": "waves",
        "seconds": 1,
        "width": 256,
        "height": 256,
        "fps": 24,
        "num_frames": 25,
        "seed": 1,
    }
    for video_id, status in (("video_running", "in_progress"), ("video_waiting", "queued")):
        store.insert(
            VideoRecord(
                id=video_id,
                model="fake-video",
                status=status,
                progress=0,
                seconds=1,
                size="256x256",
                params=params,
                created_at=int(time.time()),
            )
        )
    store.close()
    client = make_client()
    assert client.get("/v1/videos/video_running").json()["error"]["code"] == "interrupted"
    assert wait_for_video(client, "video_waiting")["status"] == "completed"


def test_retention_removes_expired_videos(client):
    video_id = create(client).json()["id"]
    wait_for_video(client, video_id)
    service = client.app.state.service
    service.store.update(video_id, expires_at=int(time.time()) - 1)
    service.sweep_expired()
    assert client.get(f"/v1/videos/{video_id}").status_code == 404
    assert not service.video_dir(video_id).exists()


def test_shutdown_keeps_waiting_videos_for_the_next_start(settings, plenty_of_memory):
    from fastapi.testclient import TestClient

    from media_gen.api import create_app
    from tests.conftest import FAKE_MODELS

    with TestClient(create_app(settings, models=FAKE_MODELS, memory=plenty_of_memory)) as first:
        running = create(first, prompt="sleep:30").json()["id"]
        waiting = create(first).json()["id"]
        deadline = time.monotonic() + 15
        while first.get(f"/v1/videos/{running}").json()["status"] != "in_progress":
            assert time.monotonic() < deadline
            time.sleep(0.1)
    store = VideoStore(settings.database)
    assert store.get(running).status == "in_progress"
    assert store.get(waiting).status == "queued"
    store.close()

    with TestClient(create_app(settings, models=FAKE_MODELS, memory=plenty_of_memory)) as second:
        assert second.get(f"/v1/videos/{running}").json()["error"]["code"] == "interrupted"
        assert wait_for_video(second, waiting)["status"] == "completed"
