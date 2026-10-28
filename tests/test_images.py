from __future__ import annotations

import base64
import io
import threading
import time
from dataclasses import replace

from PIL import Image

from media_gen.catalog import MODELS


def generate(client, **body):
    return client.post(
        "/v1/images/generations", json={"model": "fake-image", "prompt": "a red square", **body}
    )


def test_generates_openai_shaped_png(client):
    response = generate(client, size="512x256", seed=7)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["output_format"] == "png"
    assert body["size"] == "512x256"
    assert body["seed"] == 7
    [item] = body["data"]
    image = Image.open(io.BytesIO(base64.b64decode(item["b64_json"])))
    assert image.format == "PNG"
    assert image.size == (512, 256)


def test_n_images_use_consecutive_seeds(client):
    body = generate(client, n=3, seed=10, output_format="jpeg", output_compression=80).json()
    colours = [
        Image.open(io.BytesIO(base64.b64decode(d["b64_json"]))).convert("RGB").getpixel((0, 0))[0]
        for d in body["data"]
    ]
    assert len(colours) == 3
    assert colours[0] < colours[1] < colours[2]


def test_rejects_bad_requests_with_openai_errors(client):
    cases = {
        "size": ("1000x1000", "size"),
        "response_format": ("url", "response_format"),
        "n": (5, "n"),
        "background": ("transparent", "background"),
        "nonsense": (1, "nonsense"),
    }
    for field, (value, param) in cases.items():
        response = generate(client, **{field: value})
        assert response.status_code == 400, field
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["param"] == param
    assert generate(client, prompt=" ").json()["error"]["code"] == "missing_required_parameter"
    assert client.post("/v1/images/generations", content=b"[1]").status_code == 400


def test_ignores_openai_only_options(client):
    assert generate(client, quality="high", user="someone").status_code == 200


def test_unknown_and_wrong_kind_models_are_not_found(client):
    assert generate(client, model="nope").status_code == 404
    response = generate(client, model="fake-video")
    assert response.json()["error"]["code"] == "model_not_found"


def test_uninstalled_curated_model_is_reported(make_client):
    client = make_client(models={"z-image-turbo": MODELS["z-image-turbo"]})
    response = client.post("/v1/images/generations", json={"model": "z-image-turbo", "prompt": "x"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_installed"
    listed = client.get("/v1/models").json()["data"]
    assert listed[0]["installed"] is False
    assert listed[0]["kind"] == "image"


def test_api_key_is_required_when_configured(make_client, settings):
    client = make_client(settings=replace(settings, api_key="secret"))
    assert client.get("/health").status_code == 200
    assert generate(client).status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = client.post(
        "/v1/images/generations",
        json={"model": "fake-image", "prompt": "x"},
        headers={"Authorization": "Bearer secret"},
    )
    assert ok.status_code == 200


def test_worker_is_reused_then_replaced_on_model_switch(client):
    first = generate(client).json()
    assert client.get("/health").json()["scheduler"]["loaded_model"] == "fake-image"
    pid = client.app.state.service.scheduler._worker.pid
    generate(client)
    assert client.app.state.service.scheduler._worker.pid == pid
    generate(client, model="fake-image-2")
    assert client.app.state.service.scheduler._worker.pid != pid
    assert first["data"]


def test_crash_fails_the_request_and_the_next_one_recovers(client):
    response = generate(client, prompt="crash")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "worker_crashed"
    assert generate(client).status_code == 200


def test_backend_and_load_errors_are_reported(client):
    assert generate(client, prompt="fail").json()["error"]["code"] == "bad_prompt"
    response = generate(client, model="fake-broken")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "model_load_failed"


def test_queue_is_bounded(client):
    results = []

    def slow():
        results.append(generate(client, prompt="sleep:1.5").status_code)

    threads = [threading.Thread(target=slow) for _ in range(6)]
    for t in threads:
        t.start()
        time.sleep(0.05)
    for t in threads:
        t.join()
    assert results.count(200) == 5  # one running plus max_queued_jobs=4
    assert results.count(429) == 1
