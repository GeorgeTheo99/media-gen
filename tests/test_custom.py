from __future__ import annotations

import json
import math
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from media_gen import custom
from media_gen.catalog import LTX_LORA, QWEN_LORA
from media_gen.cli import main as cli_main
from media_gen.custom import CustomError, ProfileStore, check_adapter, verify_adapter
from tests.conftest import FAKE_MODELS, fake_model, wait_for_video

FORMAT = replace(LTX_LORA, weights="weights.safetensors", fusion_gib=2)
BASE = fake_model("fake-video", kind="video", adapters=FORMAT)
MODELS = {**FAKE_MODELS, BASE.id: BASE}
SIZES = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "I64": 8}
Q = "diffusion_model.transformer_blocks.0.attn1.to_q"
FF = "diffusion_model.transformer_blocks.0.ff.net.0.proj"


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]], *, pad: int = 0) -> Path:
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for name, (dtype, shape) in tensors.items():
        end = offset + math.prod(shape) * SIZES[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}
        offset = end
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset + pad))
    return path


def lora(path: Path, rank: int = 4, **overrides) -> Path:
    tensors = {
        f"{Q}.lora_A.weight": ("BF16", [rank, 64]),
        f"{Q}.lora_B.weight": ("BF16", [64, rank]),
        f"{FF}.lora_A.weight": ("BF16", [rank, 64]),
        f"{FF}.lora_B.weight": ("BF16", [256, rank]),
    }
    for name, value in overrides.items():
        if value is None:
            tensors.pop(name)
        else:
            tensors[name] = value
    return write_safetensors(path, tensors)


@pytest.fixture
def weights(tmp_path, monkeypatch) -> Path:
    """The base model's weights: one 8-bit quantized layer (64 in-features packed into 16) and one bf16."""
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "quantize_config.json").write_text(json.dumps({"quantization": {"bits": 8, "group_size": 64}}))
    path = write_safetensors(pack / "weights.safetensors", {
        "transformer.transformer_blocks.0.attn1.to_q.weight": ("U32", [64, 16]),
        "transformer.transformer_blocks.0.attn1.to_q.scales": ("BF16", [64, 1]),
        "transformer.transformer_blocks.0.attn1.to_q.biases": ("BF16", [64, 1]),
        "transformer.transformer_blocks.0.ff.proj_in.weight": ("BF16", [256, 64]),
    })
    monkeypatch.setattr(custom, "target_weights", lambda base, models_dir: path)
    return path


@pytest.fixture
def store(settings) -> ProfileStore:
    return ProfileStore(settings.home)


def add(store, settings, source: Path, name: str = "style", **kwargs):
    return store.import_adapter(
        source, name=name, base=kwargs.pop("base", BASE), models_dir=settings.models_dir,
        license=kwargs.pop("license", "Apache-2.0"), provenance=kwargs.pop("provenance", "my own training"),
        **kwargs,
    )


def test_import_copies_and_checks_a_disabled_profile(store, settings, weights, tmp_path):
    source = lora(tmp_path / "Style LoRA.safetensors")
    profile = add(store, settings, source, strength=0.8)
    assert (profile.id, profile.base, profile.enabled) == ("custom-style", "fake-video", False)
    assert (profile.rank, profile.layers, profile.strength) == (4, 2, 0.8)
    assert profile.filename == "Style LoRA.safetensors"
    assert profile.display_name == "style (custom fake-video)"
    copy = store.adapter_path(profile)
    assert copy.read_bytes() == source.read_bytes()
    assert copy.stat().st_mode & 0o777 == 0o444
    assert copy.parent == settings.home / "custom" / "adapters"
    assert not list(copy.parent.glob(".import-*"))
    assert store.load() == {"custom-style": profile}
    # Disabled: not served.
    assert store.served(MODELS) == {}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({f"{Q}.alpha": ("BF16", [])}, "not a LoRA weight"),
        ({f"{Q}.lora_B.weight": None}, "only one half"),
        ({"diffusion_model.transformer_blocks.9.attn1.to_q.lora_A.weight": ("BF16", [4, 64]),
          "diffusion_model.transformer_blocks.9.attn1.to_q.lora_B.weight": ("BF16", [64, 4])}, "not a layer"),
        # The quantized layer has 64 in-features, not its 16 packed columns.
        ({f"{Q}.lora_A.weight": ("BF16", [4, 16])}, "made for another model"),
        ({f"{FF}.lora_B.weight": ("BF16", [128, 4])}, "made for another model"),
        ({f"{Q}.lora_B.weight": ("BF16", [64, 8])}, "not a LoRA pair"),
        ({f"{Q}.lora_A.weight": ("I64", [4, 64])}, "F16, BF16 or F32"),
        ({"transformer_blocks.0.attn1.to_q.lora_A.weight": ("BF16", [4, 64])}, "two tensors patch"),
    ],
)
def test_incompatible_adapters_are_refused(store, settings, weights, tmp_path, overrides, message):
    with pytest.raises(CustomError, match=message):
        add(store, settings, lora(tmp_path / "bad.safetensors", **overrides))
    assert store.load() == {}
    assert not any((settings.home / "custom" / "adapters").iterdir())


def test_malformed_files_are_refused(store, settings, weights, tmp_path):
    good = lora(tmp_path / "good.safetensors")
    truncated = tmp_path / "truncated.safetensors"
    truncated.write_bytes(good.read_bytes()[:-10])
    padded = tmp_path / "padded.safetensors"
    padded.write_bytes(good.read_bytes() + b"\0")
    pickle = tmp_path / "adapter.pt"
    pickle.write_bytes(b"\x80\x04")
    for path, message in [
        (truncated, "file size does not match"),
        (padded, "file size does not match"),
        (pickle, "only .safetensors"),
        (tmp_path / "missing.safetensors", "cannot open"),
        (tmp_path, "only .safetensors"),
    ]:
        with pytest.raises(CustomError, match=message):
            add(store, settings, path)
    garbage = tmp_path / "garbage.safetensors"
    garbage.write_bytes(b"\xff" * 64)
    # Named as the owner's file, not the copy being checked.
    with pytest.raises(CustomError, match="^garbage.safetensors is not a safetensors file"):
        add(store, settings, garbage)


def test_import_arguments_are_checked(store, settings, weights, tmp_path):
    source = lora(tmp_path / "a.safetensors")
    for kwargs, message in [
        ({"name": "Bad Name"}, "lowercase"),
        ({"strength": 0}, "strength"),
        ({"strength": 2.5}, "strength"),
        ({"license": "  "}, "license"),
        ({"provenance": "x" * 301}, "source"),
        ({"base": FAKE_MODELS["fake-image"]}, "does not accept"),
    ]:
        with pytest.raises(CustomError, match=message):
            add(store, settings, source, **kwargs)
    add(store, settings, source)
    with pytest.raises(CustomError, match="already exists"):
        add(store, settings, source)


def test_check_adapter_reads_the_engine_names(weights, tmp_path):
    # PEFT naming without the ComfyUI prefix, as the engine also reads it.
    path = write_safetensors(tmp_path / "peft.safetensors", {
        "transformer_blocks.0.ff.net.0.proj.lora_A.weight": ("F32", [8, 64]),
        "transformer_blocks.0.ff.net.0.proj.lora_B.weight": ("F32", [256, 8]),
    })
    assert check_adapter(path, BASE, weights) == (8, 1)


IMAGE = fake_model("fake-image", adapters=QWEN_LORA)
BLOCK = "transformer_blocks.0"


@pytest.fixture
def sharded(tmp_path) -> Path:
    """A diffusers transformer in two shards: attention, and an image MLP with separate gate and up layers."""
    transformer = tmp_path / "qwen" / "transformer"
    transformer.mkdir(parents=True)
    shards = {
        "model-1.safetensors": {f"{BLOCK}.attn.to_q.weight": ("BF16", [64, 64])},
        "model-2.safetensors": {
            f"{BLOCK}.img_mlp.gate_layer.weight": ("BF16", [128, 64]),
            f"{BLOCK}.img_mlp.proj.weight": ("BF16", [128, 64]),
            f"{BLOCK}.img_mlp.out.weight": ("BF16", [64, 128]),
        },
    }
    weight_map = {}
    for shard, tensors in shards.items():
        write_safetensors(transformer / shard, tensors)
        weight_map.update(dict.fromkeys(tensors, shard))
    index = transformer / "diffusion_pytorch_model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": weight_map}))
    return index


def qwen_lora(path: Path, **overrides) -> Path:
    """An ai-toolkit (ComfyUI naming) adapter with a fused gate/up pair, as published for Qwen-Image 2.1."""
    tensors = {
        f"diffusion_model.{BLOCK}.attn.to_q.lora_A.weight": ("BF16", [4, 64]),
        f"diffusion_model.{BLOCK}.attn.to_q.lora_B.weight": ("BF16", [64, 4]),
        f"diffusion_model.{BLOCK}.img_mlp.gate_up.lora_A.weight": ("BF16", [4, 64]),
        f"diffusion_model.{BLOCK}.img_mlp.gate_up.lora_B.weight": ("BF16", [256, 4]),
        f"diffusion_model.{BLOCK}.img_mlp.out.lora_A.weight": ("BF16", [4, 128]),
        f"diffusion_model.{BLOCK}.img_mlp.out.lora_B.weight": ("BF16", [64, 4]),
    }
    for name, value in overrides.items():
        if value is None:
            tensors.pop(name)
        else:
            tensors[name] = value
    return write_safetensors(path, tensors)


def test_qwen_adapters_are_checked_against_the_sharded_transformer(sharded, tmp_path):
    assert check_adapter(qwen_lora(tmp_path / "a.safetensors"), IMAGE, sharded) == (4, 3)
    # diffusers naming, with the gate and up layers as separate pairs.
    separate = write_safetensors(tmp_path / "b.safetensors", {
        f"transformer.{BLOCK}.img_mlp.gate_layer.lora_A.weight": ("F32", [8, 64]),
        f"transformer.{BLOCK}.img_mlp.gate_layer.lora_B.weight": ("F32", [128, 8]),
        f"transformer.{BLOCK}.img_mlp.proj.lora_A.weight": ("F32", [8, 64]),
        f"transformer.{BLOCK}.img_mlp.proj.lora_B.weight": ("F32", [128, 8]),
    })
    assert check_adapter(separate, IMAGE, sharded) == (8, 2)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        # Names the engine skips: no prefix, the text stream, an alpha tensor.
        ({f"{BLOCK}.attn.to_k.lora_A.weight": ("BF16", [4, 64])}, "not a LoRA weight the engine applies"),
        ({f"diffusion_model.{BLOCK}.txt_mlp.out.lora_A.weight": ("BF16", [4, 128])}, "engine applies"),
        ({f"diffusion_model.{BLOCK}.attn.to_q.alpha": ("BF16", [])}, "engine applies"),
        ({f"diffusion_model.{BLOCK}.img_mlp.gate_up.lora_B.weight": ("BF16", [128, 4])}, "another model"),
        ({f"transformer.{BLOCK}.img_mlp.proj.lora_A.weight": ("BF16", [4, 64]),
          f"transformer.{BLOCK}.img_mlp.proj.lora_B.weight": ("BF16", [128, 4])}, "two tensors patch"),
        ({f"diffusion_model.{BLOCK}.attn.to_k.lora_A.weight": ("BF16", [4, 64]),
          f"diffusion_model.{BLOCK}.attn.to_k.lora_B.weight": ("BF16", [64, 4])}, "not a layer"),
    ],
)
def test_qwen_adapters_the_engine_would_not_apply_in_full_are_refused(sharded, tmp_path, overrides, message):
    with pytest.raises(CustomError, match=message):
        check_adapter(qwen_lora(tmp_path / "a.safetensors", **overrides), IMAGE, sharded)


def test_an_unreadable_checkpoint_index_is_refused(sharded, tmp_path):
    sharded.write_text("{}")
    with pytest.raises(CustomError, match="not a readable checkpoint index"):
        check_adapter(qwen_lora(tmp_path / "a.safetensors"), IMAGE, sharded)


def test_enabled_profile_is_served_beside_its_base(make_client, store, settings, weights, tmp_path):
    profile = add(store, settings, lora(tmp_path / "a.safetensors"), display_name="Watercolor")
    client = make_client(models=MODELS)
    ids = [row["id"] for row in client.get("/v1/models").json()["data"]]
    assert "custom-style" not in ids
    refused = client.post("/v1/videos", json={"model": "custom-style", "prompt": "waves"})
    assert refused.status_code == 404

    store.set_enabled(profile.id, True)
    rows = {row["id"]: row for row in client.get("/v1/models").json()["data"]}
    assert rows["custom-style"]["custom"] is True
    assert rows["custom-style"]["base_model"] == "fake-video"
    assert rows["custom-style"]["display_name"] == "Watercolor"
    assert rows["custom-style"]["source"] == "my own training"
    assert rows["custom-style"]["memory_gib"] == BASE.memory_gib + FORMAT.fusion_gib + 1
    assert "custom" not in rows["fake-video"]

    video = client.post("/v1/videos", json={"model": "custom-style", "prompt": "waves"}).json()
    assert video["model"] == "custom-style"
    assert wait_for_video(client, video["id"])["status"] == "completed"
    served = client.app.state.service.served()
    assert served["custom-style"].engine["loras"] == [{
        "path": str(store.adapter_path(profile)), "sha256": profile.adapter_sha256,
        "bytes": profile.adapter_bytes, "strength": 1.0,
    }]
    # The base model is unchanged.
    assert served["fake-video"] is BASE
    assert "loras" not in served["fake-video"].engine

    store.set_enabled(profile.id, False)
    assert client.post("/v1/videos", json={"model": "custom-style", "prompt": "waves"}).status_code == 404


def test_a_broken_store_leaves_curated_models_served(make_client, store, settings):
    store.root.mkdir(parents=True)
    store.path.write_text("{not json")
    client = make_client(models=MODELS)
    assert {row["id"] for row in client.get("/v1/models").json()["data"]} == set(MODELS)
    assert client.post("/v1/videos", json={"model": "fake-video", "prompt": "waves"}).status_code == 200
    with pytest.raises(CustomError, match="unreadable"):
        store.load()


@pytest.mark.parametrize(
    "change", [{"enabled": "false"}, {"adapter_bytes": "5"}, {"id": 5}, {"strength": 9}, {"rank": True}])
def test_a_hand_edited_store_serves_nothing_unchecked(make_client, store, settings, weights, tmp_path,
                                                      change):
    store.set_enabled(add(store, settings, lora(tmp_path / "a.safetensors")).id, True)
    data = json.loads(store.path.read_text())
    data["profiles"][0].update(change)
    store.path.write_text(json.dumps(data))
    with pytest.raises(CustomError, match="invalid profile"):
        store.load()
    client = make_client(models=MODELS)
    assert {row["id"] for row in client.get("/v1/models").json()["data"]} == set(MODELS)
    assert client.post("/v1/videos", json={"model": "fake-video", "prompt": "waves"}).status_code == 200


def test_a_damaged_shared_copy_is_replaced_on_import(store, settings, weights, tmp_path):
    source = lora(tmp_path / "a.safetensors")
    first = add(store, settings, source, name="one")
    path = store.adapter_path(first)
    path.chmod(0o644)
    path.write_bytes(b"damaged")
    add(store, settings, source, name="two")
    assert path.read_bytes() == source.read_bytes()


def test_a_refused_import_leaves_no_file(store, settings, weights, tmp_path):
    with pytest.raises(CustomError, match="file name"):
        add(store, settings, lora(tmp_path / "a\u200b.safetensors"))
    assert not (settings.home / "custom").exists()


def test_an_unexpected_quantization_fails_closed(store, settings, weights, tmp_path):
    config = {"quantization": {"bits": 8, "group_size": 32}}
    (weights.parent / "quantize_config.json").write_text(json.dumps(config))
    with pytest.raises(CustomError, match="quantization"):
        add(store, settings, lora(tmp_path / "a.safetensors"))


def test_remove_keeps_a_file_another_profile_uses(store, settings, weights, tmp_path):
    source = lora(tmp_path / "a.safetensors")
    first = add(store, settings, source, name="one")
    second = add(store, settings, source, name="two", strength=0.5)
    assert first.adapter_sha256 == second.adapter_sha256
    store.remove("custom-one")
    assert store.adapter_path(second).is_file()
    store.remove("custom-two")
    assert not store.adapter_path(second).exists()
    assert store.load() == {}
    with pytest.raises(CustomError, match="no custom profile"):
        store.remove("custom-two")


def test_verify_adapter_detects_a_changed_file(store, settings, weights, tmp_path):
    profile = add(store, settings, lora(tmp_path / "a.safetensors"))
    path = store.adapter_path(profile)
    spec = {"path": str(path), "sha256": profile.adapter_sha256, "bytes": profile.adapter_bytes}
    verify_adapter(spec)
    path.chmod(0o644)
    path.write_bytes(path.read_bytes()[:-1] + b"\1")
    with pytest.raises(CustomError, match="changed since it was imported"):
        verify_adapter(spec)
    path.unlink()
    with pytest.raises(CustomError, match="adapter was removed"):
        verify_adapter(spec)


def test_a_new_adapter_under_the_same_id_gets_a_new_worker(make_client, store, settings, weights, tmp_path):
    client = make_client(models=MODELS)
    scheduler = client.app.state.service.scheduler
    store.set_enabled(add(store, settings, lora(tmp_path / "a.safetensors")).id, True)
    video = client.post("/v1/videos", json={"model": "custom-style", "prompt": "waves"}).json()
    wait_for_video(client, video["id"])
    first = scheduler._worker
    store.remove("custom-style")
    store.set_enabled(add(store, settings, lora(tmp_path / "b.safetensors", rank=8)).id, True)
    video = client.post("/v1/videos", json={"model": "custom-style", "prompt": "waves"}).json()
    wait_for_video(client, video["id"])
    assert scheduler._worker is not first
    shas = [worker.spec["engine"]["loras"][0]["sha256"] for worker in (first, scheduler._worker)]
    assert shas[0] != shas[1]


def test_cli(settings, weights, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MEDIA_GEN_HOME", str(settings.home))
    monkeypatch.setattr("media_gen.cli.MODELS", MODELS)
    source = lora(tmp_path / "a.safetensors")
    args = ["custom", "import", str(source), "--base", "fake-video", "--name", "style",
            "--license", "CC-BY-4.0", "--source", "https://example.com/style"]
    assert cli_main(args) == 0
    assert "custom-style: imported, disabled" in capsys.readouterr().out
    assert cli_main(args) == 1
    assert "already exists" in capsys.readouterr().err
    assert cli_main(["custom", "enable", "custom-style"]) == 0
    capsys.readouterr()
    assert cli_main(["custom", "list", "--json"]) == 0
    (row,) = json.loads(capsys.readouterr().out)
    assert (row["id"], row["enabled"], row["installed"], row["kind"]) == ("custom-style", True, True, "video")
    assert row["license"] == "CC-BY-4.0"
    assert cli_main(["custom", "remove", "custom-style"]) == 0
    assert cli_main(["custom", "disable", "custom-style"]) == 1
    assert cli_main(["custom", "import", str(source), "--base", "fake-image", "--name", "x",
                     "--license", "x", "--source", "x"]) == 1
    assert "does not accept" in capsys.readouterr().err
