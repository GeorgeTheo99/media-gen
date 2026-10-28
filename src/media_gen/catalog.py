"""The curated models and the pinned files each one is built from."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any, Literal

COMPLETE_MARKER = ".media-gen-complete"


@dataclass(frozen=True)
class ManifestFile:
    path: str
    size: int
    sha256: str | None = None
    git_sha1: str | None = None


@dataclass(frozen=True)
class Manifest:
    name: str
    repo_id: str
    revision: str
    files: tuple[ManifestFile, ...]
    digest: str

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    def directory(self, models_dir: Path) -> Path:
        return models_dir / self.repo_id.replace("/", "--") / self.revision

    def is_installed(self, models_dir: Path) -> bool:
        marker = self.directory(models_dir) / COMPLETE_MARKER
        try:
            return json.loads(marker.read_text()).get("digest") == self.digest
        except (OSError, ValueError):
            return False


@cache
def manifest(name: str) -> Manifest:
    raw = resources.files("media_gen.manifests").joinpath(f"{name}.json").read_bytes()
    data = json.loads(raw)
    if data.get("schema") != 1:
        raise ValueError(f"manifest {name}: unsupported schema {data.get('schema')!r}")
    files = tuple(ManifestFile(**f) for f in data["files"])
    for f in files:
        if (f.sha256 is None) == (f.git_sha1 is None):
            raise ValueError(f"manifest {name}: {f.path} needs exactly one of sha256 or git_sha1")
        if f.path.startswith("/") or ".." in Path(f.path).parts:
            raise ValueError(f"manifest {name}: unsafe path {f.path!r}")
    return Manifest(
        name=name,
        repo_id=data["repo_id"],
        revision=data["revision"],
        files=files,
        digest=hashlib.sha256(raw).hexdigest(),
    )


@dataclass(frozen=True)
class MediaModel:
    id: str
    kind: Literal["image", "video"]
    backend: Literal["mflux", "ltx"]
    display_name: str
    license: str
    # Pinned repositories, in the order the backend expects them.
    manifests: tuple[str, ...]
    # Unified memory to load the model and run its largest allowed request, with headroom over
    # the measured MLX peak (M3 Ultra): FLUX 3 Klein 20.9 GiB at 2048x2048, Qwen-Image 2.1
    # 66.1 GiB at 1664x1664, LTX-2.5 29.2 GiB for 15 s at 1280x704 (with its tiled decode).
    memory_gib: int
    # Backend-specific load options, passed through to the worker.
    engine: dict[str, Any] = field(default_factory=dict)
    default_size: tuple[int, int] = (1024, 1024)
    max_pixels: int = 2048 * 2048
    size_multiple: int = 16
    default_steps: int | None = None
    max_seconds: int = 0
    default_seconds: int = 0

    def resolved_manifests(self) -> list[Manifest]:
        return [manifest(name) for name in self.manifests]

    def is_installed(self, models_dir: Path) -> bool:
        return all(m.is_installed(models_dir) for m in self.resolved_manifests())

    def download_bytes(self) -> int:
        return sum(m.total_bytes for m in self.resolved_manifests())


MODELS: dict[str, MediaModel] = {
    m.id: m
    for m in (
        MediaModel(
            id="flux3-image-klein",
            kind="image",
            backend="mflux",
            display_name="FLUX 3 Image Klein",
            license="flux-kommunity",
            manifests=("flux3-image-klein",),
            memory_gib=28,
            engine={"family": "flux3", "mflux_model": "flux3-image-klein"},
            default_steps=16,
        ),
        MediaModel(
            id="qwen-image-2.1",
            kind="image",
            backend="mflux",
            display_name="Qwen-Image 2.1",
            license="qwen-research",
            manifests=("qwen-image-2.1",),
            memory_gib=74,
            engine={"family": "qwen21", "mflux_model": "qwen-image-2.1"},
            default_size=(1328, 1328),
            # Its native 2K takes over 10 minutes here; 1664x1664 takes about 7.
            max_pixels=1664 * 1664,
            default_steps=40,
        ),
        MediaModel(
            id="ltx-2.5",
            kind="video",
            backend="ltx",
            display_name="LTX-2.5 Distilled (MLX Q8, with audio)",
            license="ltx-2-community",
            manifests=("ltx-2.5-mlx-q8",),
            memory_gib=40,
            default_size=(1280, 704),
            max_pixels=1280 * 768,
            size_multiple=64,
            # 5 s takes about 3 minutes here, 15 s about 13.
            max_seconds=15,
            default_seconds=5,
        ),
    )
}
