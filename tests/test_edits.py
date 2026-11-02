from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from media_gen.catalog import MODELS

from .conftest import FAKE_MODELS, fake_model

EDIT_MODELS = {**FAKE_MODELS, "fake-editor": fake_model("fake-editor", max_references=2)}


def png(size: tuple[int, int] = (320, 256)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (0, 128, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def edit(client, images: list[bytes], **fields):
    return client.post(
        "/v1/images/edits",
        data={"model": "fake-editor", "prompt": "make it blue", **fields},
        files=[("image[]", (f"ref{i}.png", data, "image/png")) for i, data in enumerate(images)],
    )


@pytest.fixture
def client(make_client):
    return make_client(models=EDIT_MODELS)


def decoded(response) -> Image.Image:
    [item] = response.json()["data"]
    return Image.open(io.BytesIO(base64.b64decode(item["b64_json"]))).convert("RGB")


def test_edit_passes_every_reference_to_the_backend(client, settings):
    response = edit(client, [png(), png()], size="512x256", seed=3, n="1")
    assert response.status_code == 200, response.text
    assert response.json()["seed"] == 3
    image = decoded(response)
    assert image.size == (512, 256)
    assert image.getpixel((0, 0))[1] == 2
    # The references are gone once the image is made.
    assert not any(settings.edits_dir.iterdir())


def test_auto_size_follows_the_first_reference_within_the_pixel_cap(client):
    small = edit(client, [png((330, 270))])
    assert small.json()["size"] == "320x256"
    large = edit(client, [png((4000, 2000))])
    width, height = map(int, large.json()["size"].split("x"))
    assert width * height <= 1024 * 1024
    assert abs(width / height - 2) < 0.05


def test_single_image_field_is_accepted(client):
    response = client.post(
        "/v1/images/edits",
        data={"model": "fake-editor", "prompt": "x"},
        files={"image": ("ref.png", png(), "image/png")},
    )
    assert response.status_code == 200, response.text


def test_refuses_models_without_references_and_too_many_images(client):
    plain = edit(client, [png()], model="fake-image")
    assert plain.status_code == 400
    assert plain.json()["error"]["param"] == "model"
    too_many = edit(client, [png(), png(), png()])
    assert too_many.status_code == 400
    assert "at most 2" in too_many.json()["error"]["message"]


def test_refuses_missing_and_unreadable_images(client):
    missing = client.post("/v1/images/edits", data={"model": "fake-editor", "prompt": "x"},
                          files={"other": ("a.txt", b"x", "text/plain")})
    assert missing.status_code == 400
    assert missing.json()["error"]["param"] == "other"
    none = client.post("/v1/images/edits", data={"model": "fake-editor", "prompt": "x"},
                       files={"unused": ("", b"", "application/octet-stream")})
    assert none.status_code == 400
    unreadable = edit(client, [b"not an image"])
    assert unreadable.status_code == 400
    assert unreadable.json()["error"]["param"] == "image"
    as_json = client.post("/v1/images/edits", json={"model": "fake-editor", "prompt": "x"})
    assert as_json.status_code == 400
    assert as_json.json()["error"]["param"] == "image"


def test_output_and_references_share_the_pixel_budget(make_client):
    editor = fake_model("fake-editor", max_references=3, max_pixels=1_000_000)
    client = make_client(models={"fake-editor": editor})
    assert edit(client, [png((512, 512)), png((512, 512))], size="512x512").status_code == 200
    over = edit(client, [png((512, 512)), png((512, 512)), png((512, 512))], size="512x512")
    assert over.status_code == 400
    assert over.json()["error"]["param"] == "size"


def test_prompt_and_fields_are_validated_like_generations(client):
    assert edit(client, [png()], prompt=" ").json()["error"]["code"] == "missing_required_parameter"
    assert edit(client, [png()], size="1000x1000").json()["error"]["param"] == "size"
    assert edit(client, [png()], nonsense="1").json()["error"]["param"] == "nonsense"


def test_flux_klein_takes_references():
    assert MODELS["flux3-image-klein"].max_references == 3
    assert MODELS["qwen-image-2.1"].max_references == 0
