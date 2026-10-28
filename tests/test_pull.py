from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import httpx
import pytest

from media_gen.catalog import MODELS, Manifest, ManifestFile, manifest
from media_gen.pull import PullError, RateLimiter, _require_https, pull_manifest

BIG = b"weights" * 50_000
SMALL = b'{"hidden_size": 8}\n'


def git_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def make_manifest(big: bytes = BIG) -> Manifest:
    return Manifest(
        name="test",
        repo_id="org/repo",
        revision="a" * 40,
        files=(
            ManifestFile("config.json", len(SMALL), git_sha1=git_sha1(SMALL)),
            ManifestFile("sub/model.safetensors", len(BIG), sha256=hashlib.sha256(big).hexdigest()),
        ),
        digest="d" * 64,
    )


class Hub:
    def __init__(self, ignore_range: bool = False):
        self.files = {"config.json": SMALL, "sub/model.safetensors": BIG}
        self.requests: list[httpx.Request] = []
        self.ignore_range = ignore_range

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        prefix = f"/org/repo/resolve/{'a' * 40}/"
        assert request.url.path.startswith(prefix)
        data = self.files[request.url.path.removeprefix(prefix)]
        if (header := request.headers.get("range")) and not self.ignore_range:
            start = int(header.removeprefix("bytes=").rstrip("-"))
            return httpx.Response(206, content=data[start:])
        return httpx.Response(200, content=data)


def client_for(hub: Hub) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(hub.handler), base_url="https://huggingface.co")


def test_downloads_verifies_and_marks_complete(tmp_path):
    hub = Hub()
    m = make_manifest()
    pull_manifest(m, tmp_path, RateLimiter(1e12), client_for(hub), log=lambda s: None)
    root = m.directory(tmp_path)
    assert (root / "sub/model.safetensors").read_bytes() == BIG
    assert m.is_installed(tmp_path)
    assert json.loads((root / ".media-gen-complete").read_text())["digest"] == m.digest
    # A second pull downloads nothing.
    pull_manifest(m, tmp_path, RateLimiter(1e12), client_for(hub), log=lambda s: None)
    assert len(hub.requests) == 2


def test_resumes_a_partial_download(tmp_path):
    hub = Hub()
    m = make_manifest()
    root = m.directory(tmp_path)
    (root / "sub").mkdir(parents=True)
    (root / "sub/model.safetensors.part").write_bytes(BIG[:1000])
    pull_manifest(m, tmp_path, RateLimiter(1e12), client_for(hub), log=lambda s: None)
    ranged = [r for r in hub.requests if "range" in r.headers]
    assert ranged[0].headers["range"] == "bytes=1000-"
    assert (root / "sub/model.safetensors").read_bytes() == BIG


def test_restarts_when_the_server_ignores_the_range(tmp_path):
    hub = Hub(ignore_range=True)
    m = make_manifest()
    root = m.directory(tmp_path)
    (root / "sub").mkdir(parents=True)
    (root / "sub/model.safetensors.part").write_bytes(BIG[:1000])
    pull_manifest(m, tmp_path, RateLimiter(1e12), client_for(hub), log=lambda s: None)
    assert (root / "sub/model.safetensors").read_bytes() == BIG


def test_hash_mismatch_fails_and_discards(tmp_path):
    m = make_manifest(big=b"something else entirely")
    with pytest.raises(PullError):
        pull_manifest(m, tmp_path, RateLimiter(1e12), client_for(Hub()), log=lambda s: None)
    root = m.directory(tmp_path)
    assert not m.is_installed(tmp_path)
    assert not (root / "sub/model.safetensors").exists()


def test_refuses_plain_http():
    with pytest.raises(PullError):
        _require_https(httpx.Request("GET", "http://cdn.example/file"))


def test_rate_limiter_paces(monkeypatch):
    slept = []
    monkeypatch.setattr("media_gen.pull.time.sleep", slept.append)
    limiter = RateLimiter(1000)
    limiter.consume(1000)
    limiter.consume(500)
    assert slept and 0.4 < slept[-1] <= 0.5


def test_shipped_manifests_are_valid():
    for model in MODELS.values():
        for m in model.resolved_manifests():
            assert len(m.revision) == 40
            assert m.files
            assert model.download_bytes() > 1e9
    assert manifest("z-image-turbo").repo_id == "Tongyi-MAI/Z-Image-Turbo"


def test_remove_keeps_shared_repositories(tmp_path, monkeypatch, capsys):
    from media_gen import catalog, cli
    from media_gen.catalog import COMPLETE_MARKER, MediaModel

    shared = make_manifest()
    alone = Manifest("alone", "org/alone", "b" * 40, shared.files, "e" * 64)
    manifests = {"shared": replace(shared, name="shared"), "alone": alone}
    monkeypatch.setattr(catalog, "manifest", manifests.__getitem__)
    models = {
        "one": MediaModel("one", "image", "fake", "One", "test", ("alone", "shared"), 1),
        "two": MediaModel("two", "image", "fake", "Two", "test", ("shared",), 1),
    }
    monkeypatch.setattr(cli, "MODELS", models)
    home = tmp_path / "home"
    for m in manifests.values():
        root = m.directory(home / "models")
        root.mkdir(parents=True)
        (root / COMPLETE_MARKER).write_text(json.dumps({"digest": m.digest}))
    monkeypatch.setenv("MEDIA_GEN_HOME", str(home))

    assert cli.main(["remove", "one"]) == 0
    assert not alone.directory(home / "models").exists()
    assert not (home / "models" / "org--alone").exists()
    assert manifests["shared"].directory(home / "models").exists()
    assert cli.main(["remove", "nope"]) == 2
