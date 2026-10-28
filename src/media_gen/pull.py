"""Download pinned model files: HTTPS only, resumable, rate-limited, hash-verified."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

import httpx

from media_gen.catalog import COMPLETE_MARKER, Manifest, ManifestFile, MediaModel

CHUNK = 1 << 20


class PullError(RuntimeError):
    pass


class RateLimiter:
    """A token bucket shared by every download in one pull."""

    def __init__(self, bytes_per_second: float):
        self.rate = bytes_per_second
        self.allowance = bytes_per_second
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def consume(self, amount: int) -> None:
        with self.lock:
            now = time.monotonic()
            self.allowance = min(self.rate, self.allowance + (now - self.last) * self.rate)
            self.last = now
            self.allowance -= amount
            wait = -self.allowance / self.rate if self.allowance < 0 else 0.0
        if wait:
            time.sleep(wait)


def _hasher(spec: ManifestFile):
    if spec.sha256 is not None:
        return hashlib.sha256()
    h = hashlib.sha1()
    h.update(b"blob %d\0" % spec.size)
    return h


def _expected(spec: ManifestFile) -> str:
    return spec.sha256 or spec.git_sha1 or ""


def _hash_file(path: Path, spec: ManifestFile):
    h = _hasher(spec)
    with path.open("rb") as f:
        while chunk := f.read(CHUNK * 8):
            h.update(chunk)
    return h


def _require_https(request: httpx.Request) -> None:
    if request.url.scheme != "https":
        raise PullError(f"refusing non-HTTPS download URL: {request.url.host}")


def hf_token() -> str | None:
    """The Hugging Face token gated models need: HF_TOKEN, else the one `hf auth login` saved."""
    if token := os.environ.get("HF_TOKEN", "").strip():
        return token
    hf_home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    path = Path(os.environ.get("HF_TOKEN_PATH") or hf_home / "token")
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


def make_client() -> httpx.Client:
    headers = {}
    if token := hf_token():
        # httpx drops Authorization when a redirect leaves huggingface.co.
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(
        headers=headers,
        follow_redirects=True,
        timeout=httpx.Timeout(60, connect=30),
        event_hooks={"request": [_require_https]},
    )


def _download(client: httpx.Client, url: str, spec: ManifestFile, target: Path, limiter: RateLimiter) -> None:
    part = target.with_name(target.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > spec.size:
        part.unlink()
        offset = 0
    h = _hash_file(part, spec) if offset else _hasher(spec)
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with client.stream("GET", url, headers=headers) as response:
        if offset and response.status_code == 200:
            # The server ignored the range: start over.
            offset, h = 0, _hasher(spec)
        elif response.status_code in (401, 403):
            raise PullError(
                f"{spec.path}: HTTP {response.status_code}. This model is gated: accept its terms at "
                f"{url.split('/resolve/')[0]} while signed in, then sign in here with `hf auth login`"
            )
        elif response.status_code not in (200, 206):
            raise PullError(f"{spec.path}: HTTP {response.status_code}")
        with part.open("ab" if offset else "wb") as out:
            for chunk in response.iter_bytes(CHUNK):
                limiter.consume(len(chunk))
                out.write(chunk)
                h.update(chunk)
    size = part.stat().st_size
    if size != spec.size:
        raise PullError(f"{spec.path}: got {size} bytes, expected {spec.size}")
    if h.hexdigest() != _expected(spec):
        part.unlink()
        raise PullError(f"{spec.path}: hash mismatch (the partial download was removed)")
    os.replace(part, target)


def pull_manifest(
    manifest: Manifest,
    models_dir: Path,
    limiter: RateLimiter,
    client: httpx.Client,
    log: Callable[[str], None] = print,
) -> None:
    if manifest.is_installed(models_dir):
        log(f"{manifest.repo_id}@{manifest.revision[:12]}: already installed")
        return
    root = manifest.directory(models_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PullError(f"{manifest.repo_id} is being downloaded by another process") from exc
        done = 0
        for spec in manifest.files:
            target = root / spec.path
            target.parent.mkdir(parents=True, exist_ok=True)
            if not (
                target.exists()
                and target.stat().st_size == spec.size
                and _hash_file(target, spec).hexdigest() == _expected(spec)
            ):
                log(f"{manifest.repo_id}: {spec.path} ({spec.size / 1e9:.2f} GB)")
                url = (
                    f"https://huggingface.co/{manifest.repo_id}/resolve/"
                    f"{manifest.revision}/{quote(spec.path)}"
                )
                _download(client, url, spec, target, limiter)
            done += spec.size
            log(f"{manifest.repo_id}: {done / manifest.total_bytes:.0%}")
        marker = root / COMPLETE_MARKER
        marker.write_text(json.dumps({"digest": manifest.digest, "repo_id": manifest.repo_id}) + "\n")


def pull(
    model: MediaModel,
    models_dir: Path,
    limit_mbps: int,
    client: httpx.Client | None = None,
    log: Callable[[str], None] = print,
) -> None:
    if limit_mbps <= 0:
        raise PullError("the download limit must be positive")
    limiter = RateLimiter(limit_mbps * 1_000_000 / 8)
    owned = client is None
    client = client or make_client()
    try:
        for manifest in model.resolved_manifests():
            pull_manifest(manifest, models_dir, limiter, client, log)
    finally:
        if owned:
            client.close()
