"""Custom profiles: a curated base model plus a LoRA adapter the owner supplies, on this install only.

Profiles live in ``MEDIA_GEN_HOME/custom``, outside the curated model files and every release:
``profiles.json`` and the adapters it names, copied in as ``adapters/<sha256>.safetensors`` (read-only).
An imported profile is disabled until the owner enables it, is served under its own ``custom-<name>``
id, and never changes its base model, which stays the default.

Only a safetensors file whose every tensor is half of a LoRA pair the engine applies to one of the base
model's weights, with matching shapes, is accepted: the engine silently skips keys it does not know, so
an adapter for another model or in another format would otherwise "work" and change nothing.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import math
import os
import re
import secrets
import stat
import struct
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from media_gen.catalog import MediaModel

log = logging.getLogger(__name__)

PREFIX = "custom-"
NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,38}[a-z0-9])?$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SCHEMA = 1
MAX_ADAPTER_BYTES = 16 << 30
MAX_HEADER_BYTES = 100 << 20
MAX_RANK = 1024
MAX_STRENGTH = 2.0
MAX_DISPLAY_NAME = 80
MAX_TEXT = 300
GIB = 1 << 30
CHUNK = 8 << 20
ADAPTER_DTYPES = {"F16": 2, "BF16": 2, "F32": 4}
PAIR = (".lora_A.weight", ".lora_B.weight")


class CustomError(ValueError):
    """A refused import or change, explained for the owner."""


@dataclass(frozen=True)
class Profile:
    id: str
    base: str
    display_name: str
    license: str
    source: str
    strength: float
    # The adapter file: its SHA-256 (also its name in adapters/), size, and the imported file's name.
    adapter_sha256: str
    adapter_bytes: int
    filename: str
    rank: int
    layers: int
    enabled: bool
    imported_at: int

    def check(self) -> None:
        """Every field has its type and range: a hand-edited store must not serve something unchecked."""
        texts = (self.id, self.base, self.display_name, self.license, self.source, self.filename)
        counts = (self.adapter_bytes, self.rank, self.layers, self.imported_at)
        if not (
            all(isinstance(value, str) for value in texts)
            and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in counts)
            and isinstance(self.enabled, bool)
            and isinstance(self.strength, int | float) and not isinstance(self.strength, bool)
            and 0 < self.strength <= MAX_STRENGTH
            and self.id.startswith(PREFIX) and NAME.match(self.id.removeprefix(PREFIX))
            and SHA256.match(self.adapter_sha256)
        ):
            raise ValueError(f"profile {self.id!r} has a field of the wrong type or out of range")

    def row(self, base: MediaModel | None, models_dir: Path) -> dict[str, Any]:
        """For `media-gen custom list --json` and Model Gateway."""
        return {
            **asdict(self),
            "kind": base.kind if base else None,
            "installed": bool(base and base.is_installed(models_dir)),
        }


# Safetensors ------------------------------------------------------------------


def read_header(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    """A safetensors file's tensor table (without ``__metadata__``) and the byte offset its data starts at."""
    try:
        with path.open("rb") as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                raise CustomError(f"{path.name} is not a safetensors file")
            (length,) = struct.unpack("<Q", prefix)
            if not 2 <= length <= MAX_HEADER_BYTES:
                raise CustomError(f"{path.name} is not a safetensors file (bad header length)")
            raw = f.read(length)
    except OSError as exc:
        raise CustomError(f"cannot read {path.name}: {exc.strerror or exc}") from None
    if len(raw) != length:
        raise CustomError(f"{path.name} is truncated")
    try:
        header = json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        raise CustomError(f"{path.name} is not a safetensors file (bad header)") from None
    if not isinstance(header, dict):
        raise CustomError(f"{path.name} is not a safetensors file (bad header)")
    header.pop("__metadata__", None)
    return header, 8 + length


def _check_layout(name: str, header: dict[str, Any], data_start: int, size: int) -> None:
    """Every tensor is a supported float type whose bytes tile the data section exactly."""
    spans = []
    for key, info in header.items():
        shape = info.get("shape") if isinstance(info, dict) else None
        offsets = info.get("data_offsets") if isinstance(info, dict) else None
        dtype = info.get("dtype") if isinstance(info, dict) else None
        if (
            not isinstance(shape, list)
            or not all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(o, int) and not isinstance(o, bool) for o in offsets)
        ):
            raise CustomError(f"{name}: tensor {key!r} has a malformed entry")
        if dtype not in ADAPTER_DTYPES:
            raise CustomError(f"{name}: tensor {key!r} is {dtype}; adapters must be F16, BF16 or F32")
        begin, end = offsets
        if end - begin != math.prod(shape) * ADAPTER_DTYPES[dtype]:
            raise CustomError(f"{name}: tensor {key!r} has the wrong byte length for its shape")
        spans.append((begin, end))
    position = 0
    for begin, end in sorted(spans):
        if begin != position:
            raise CustomError(f"{name}: tensor data overlaps or has gaps")
        position = end
    if data_start + position != size:
        raise CustomError(f"{name}: the file size does not match its tensors")


# Compatibility ----------------------------------------------------------------


def target_weights(base: MediaModel, models_dir: Path) -> Path:
    if base.adapters is None:
        raise CustomError(f"{base.id} does not accept LoRA adapters")
    (pack,) = base.resolved_manifests()
    return pack.directory(models_dir) / base.adapters.weights


def _quantization(pack_dir: Path) -> tuple[int, int]:
    """The pack's quantization bits and group size; (0, 0) when unknown."""
    try:
        config = json.loads((pack_dir / "quantize_config.json").read_text())["quantization"]
        bits, group_size = config["bits"], config["group_size"]
    except (OSError, ValueError, KeyError, TypeError):
        return 0, 0
    if bits in (2, 3, 4, 5, 6, 8) and isinstance(group_size, int) and group_size > 0:
        return bits, group_size
    return 0, 0


def check_adapter(adapter: Path, base: MediaModel, weights: Path) -> tuple[int, int]:
    """Refuse an adapter the engine would not apply in full to ``base``; return its rank and layer count."""
    fmt = base.adapters
    if fmt is None:
        raise CustomError(f"{base.id} does not accept LoRA adapters")
    header, data_start = read_header(adapter)
    _check_layout(adapter.name, header, data_start, adapter.stat().st_size)
    if not header:
        raise CustomError(f"{adapter.name} has no tensors")
    model, _ = read_header(weights)
    bits, group_size = _quantization(weights.parent)

    pairs: dict[str, dict[str, list[int]]] = {}
    for key, info in header.items():
        renamed = key
        for old, new in fmt.renames:
            if old in renamed:
                renamed = renamed.replace(old, new)
        half = next((suffix for suffix in PAIR if renamed.endswith(suffix)), None)
        if half is None:
            raise CustomError(
                f"{adapter.name}: {key!r} is not a LoRA weight. Only adapters with lora_A/lora_B pairs "
                f"(PEFT or ComfyUI naming, no alpha or other tensors) are supported"
            )
        target = renamed.removesuffix(half)
        if half in pairs.setdefault(target, {}):
            raise CustomError(f"{adapter.name}: two tensors patch {target!r}")
        pairs[target][half] = info["shape"]

    ranks = set()
    for target, halves in pairs.items():
        if len(halves) != 2:
            raise CustomError(f"{adapter.name}: {target!r} has only one half of its LoRA pair")
        weight = model.get(f"{fmt.prefix}{target}.weight")
        if weight is None:
            raise CustomError(
                f"{adapter.name}: {target!r} is not a layer of {base.id}; this adapter was made for "
                f"another model or uses naming the engine does not read"
            )
        shape = weight.get("shape") or []
        if len(shape) != 2:
            raise CustomError(f"{adapter.name}: {target!r} is not a linear layer of {base.id}")
        out_features, in_features = shape
        scales = model.get(f"{fmt.prefix}{target}.scales")
        if scales is not None:
            in_features = in_features * 32 // bits if bits else 0
            # One bit width for the whole pack: its scales must agree, else the check fails closed.
            if not in_features or (scales.get("shape") or [0, 0])[-1] * group_size != in_features:
                raise CustomError(f"{base.id}: its quantization is not the one recorded, so adapters cannot "
                                  f"be checked")
        a, b = halves[PAIR[0]], halves[PAIR[1]]
        if len(a) != 2 or len(b) != 2 or a[0] != b[1] or not 1 <= a[0] <= MAX_RANK:
            raise CustomError(f"{adapter.name}: {target!r} is not a LoRA pair of rank 1-{MAX_RANK}")
        if a[1] != in_features or b[0] != out_features:
            raise CustomError(
                f"{adapter.name}: {target!r} is {b[0]}x{a[1]} but {base.id}'s layer is "
                f"{out_features}x{in_features}; this adapter was made for another model"
            )
        ranks.add(a[0])
    return max(ranks), len(pairs)


# Serving ----------------------------------------------------------------------


def served_model(profile: Profile, base: MediaModel, adapter: Path) -> MediaModel:
    """The base model with the adapter applied, under the profile's id and name."""
    fusion_gib = base.adapters.fusion_gib if base.adapters else 0
    lora = {
        "path": str(adapter),
        "sha256": profile.adapter_sha256,
        "bytes": profile.adapter_bytes,
        "strength": profile.strength,
    }
    return replace(
        base,
        id=profile.id,
        display_name=profile.display_name,
        license=profile.license,
        memory_gib=base.memory_gib + fusion_gib + math.ceil(profile.adapter_bytes / GIB),
        engine={**base.engine, "loras": [lora]},
        adapters=None,
    )


def verify_adapter(lora: dict[str, Any]) -> None:
    """The adapter file is still the one imported (run in the worker before the engine reads it)."""
    path = Path(lora["path"])
    digest = hashlib.sha256()
    try:
        with path.open("rb") as f:
            while chunk := f.read(CHUNK):
                digest.update(chunk)
        size = path.stat().st_size
    except FileNotFoundError:
        raise CustomError("this profile's adapter was removed") from None
    except OSError as exc:
        raise CustomError(f"the adapter {path.name} cannot be read: {exc.strerror or exc}") from None
    if size != lora["bytes"] or digest.hexdigest() != lora["sha256"]:
        raise CustomError(f"the adapter {path.name} changed since it was imported; remove and import it")


# The store --------------------------------------------------------------------


def _intact(path: Path, sha256: str, size: int) -> bool:
    try:
        verify_adapter({"path": str(path), "sha256": sha256, "bytes": size})
    except CustomError:
        return False
    return True


def _text(value: str, what: str, limit: int = MAX_TEXT) -> str:
    value = " ".join(value.split())
    if not value or len(value) > limit or not value.isprintable():
        raise CustomError(f"{what} must be printable text of 1-{limit} characters")
    return value


class ProfileStore:
    def __init__(self, home: Path):
        self.root = home / "custom"
        self.path = self.root / "profiles.json"
        self.adapters = self.root / "adapters"

    def adapter_path(self, profile: Profile) -> Path:
        return self.adapters / f"{profile.adapter_sha256}.safetensors"

    def load(self) -> dict[str, Profile]:
        """Every profile; an absent store is empty. Raises CustomError when the file is not readable."""
        try:
            data = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            raise CustomError(f"{self.path} is unreadable: {exc}") from None
        rows = data.get("profiles") if isinstance(data, dict) and data.get("schema") == SCHEMA else None
        if not isinstance(rows, list):
            raise CustomError(f"{self.path} has an unsupported format")
        profiles = {}
        try:
            for row in rows:
                profile = Profile(**row)
                if not isinstance(profile.adapter_sha256, str):
                    raise ValueError(f"profile {profile.id!r} has no adapter hash")
                profile.check()
                profiles[profile.id] = profile
        except (TypeError, ValueError) as exc:
            raise CustomError(f"{self.path} has an invalid profile: {exc}") from None
        return profiles

    def served(self, curated: dict[str, MediaModel]) -> dict[str, tuple[Profile, MediaModel]]:
        """The enabled profiles whose base model is curated, with their models. A broken store serves none."""
        try:
            profiles = self.load()
        except CustomError as exc:
            log.warning("custom profiles are not served: %s", exc)
            return {}
        served = {}
        for profile in profiles.values():
            base = curated.get(profile.base)
            if profile.enabled and base is not None and base.adapters is not None:
                served[profile.id] = (profile, served_model(profile, base, self.adapter_path(profile)))
        return served

    @contextlib.contextmanager
    def _locked(self) -> Iterator[dict[str, Profile]]:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.adapters.mkdir(exist_ok=True, mode=0o700)
        with (self.root / ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield self.load()

    def _save(self, profiles: dict[str, Profile]) -> None:
        body = json.dumps({"schema": SCHEMA, "profiles": [asdict(p) for p in profiles.values()]}, indent=2)
        part = self.path.with_name(f".profiles-{secrets.token_hex(4)}.json")
        try:
            with part.open("x") as f:
                f.write(body + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(part, self.path)
        finally:
            part.unlink(missing_ok=True)

    def import_adapter(
        self,
        source: Path,
        *,
        name: str,
        base: MediaModel,
        models_dir: Path,
        license: str,
        provenance: str,
        display_name: str | None = None,
        strength: float = 1.0,
    ) -> Profile:
        """Copy, check and record an adapter as a new, disabled profile."""
        if not NAME.match(name):
            raise CustomError("the name must be 1-40 lowercase letters, digits or hyphens")
        if base.adapters is None:
            raise CustomError(f"{base.id} does not accept LoRA adapters")
        if not base.is_installed(models_dir):
            raise CustomError(f"{base.id} is not installed")
        if not (isinstance(strength, int | float) and 0 < strength <= MAX_STRENGTH):
            raise CustomError(f"the strength must be above 0 and at most {MAX_STRENGTH}")
        if source.suffix != ".safetensors":
            raise CustomError("only .safetensors adapters are supported")
        profile_id = PREFIX + name
        filename = _text(source.name, "the file name")
        display = _text(display_name or f"{name} (custom {base.display_name.split(' (')[0]})",
                        "the display name", MAX_DISPLAY_NAME)
        license_text, provenance_text = _text(license, "the license"), _text(provenance, "the source")
        with self._locked() as profiles:
            if profile_id in profiles:
                raise CustomError(f"{profile_id} already exists; remove it first")
            part = self.adapters / f".import-{secrets.token_hex(8)}.part"
            try:
                digest, size = self._copy(source, part)
                rank, layers = check_adapter(part, base, target_weights(base, models_dir))
                final = self.adapters / f"{digest}.safetensors"
                shared = any(p.adapter_sha256 == digest for p in profiles.values())
                # Another profile's copy of the same file is reused only while it is intact.
                if not (shared and _intact(final, digest, size)):
                    os.chmod(part, 0o444)
                    os.replace(part, final)
                profile = Profile(
                    id=profile_id, base=base.id, display_name=display, license=license_text,
                    source=provenance_text, strength=float(strength), adapter_sha256=digest,
                    adapter_bytes=size, filename=filename, rank=rank, layers=layers, enabled=False,
                    imported_at=int(time.time()),
                )
                try:
                    self._save({**profiles, profile_id: profile})
                except BaseException:
                    if not shared:
                        final.unlink(missing_ok=True)
                    raise
            finally:
                part.unlink(missing_ok=True)
        return profile

    @staticmethod
    def _copy(source: Path, target: Path) -> tuple[str, int]:
        try:
            fd = os.open(source, os.O_RDONLY)
        except OSError as exc:
            raise CustomError(f"cannot open {source}: {exc.strerror or exc}") from None
        digest, size = hashlib.sha256(), 0
        with os.fdopen(fd, "rb") as src:
            info = os.fstat(src.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise CustomError(f"{source} is not a regular file")
            if info.st_size > MAX_ADAPTER_BYTES:
                raise CustomError(f"{source.name} is larger than {MAX_ADAPTER_BYTES // GIB} GiB")
            with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as out:
                while chunk := src.read(CHUNK):
                    size += len(chunk)
                    if size > MAX_ADAPTER_BYTES:
                        raise CustomError(f"{source.name} is larger than {MAX_ADAPTER_BYTES // GIB} GiB")
                    digest.update(chunk)
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
        return digest.hexdigest(), size

    def set_enabled(self, profile_id: str, enabled: bool) -> Profile:
        with self._locked() as profiles:
            if profile_id not in profiles:
                raise CustomError(f"no custom profile {profile_id!r}")
            profile = replace(profiles[profile_id], enabled=enabled)
            self._save({**profiles, profile_id: profile})
        return profile

    def remove(self, profile_id: str) -> None:
        """Forget a profile, and delete its adapter unless another profile uses the same file."""
        with self._locked() as profiles:
            profile = profiles.pop(profile_id, None)
            if profile is None:
                raise CustomError(f"no custom profile {profile_id!r}")
            self._save(profiles)
            if not any(p.adapter_sha256 == profile.adapter_sha256 for p in profiles.values()):
                self.adapter_path(profile).unlink(missing_ok=True)
