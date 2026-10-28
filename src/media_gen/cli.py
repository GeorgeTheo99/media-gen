"""`media-gen serve | models | pull | remove`."""

from __future__ import annotations

import argparse
import json
import sys

from media_gen.catalog import MODELS
from media_gen.settings import Settings


def _models(settings: Settings, as_json: bool) -> int:
    rows = [
        {
            "id": m.id,
            "kind": m.kind,
            "installed": m.is_installed(settings.models_dir),
            "download_gb": round(m.download_bytes() / 1e9, 1),
            "memory_gib": m.memory_gib,
            "license": m.license,
        }
        for m in MODELS.values()
    ]
    if as_json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            state = "installed" if r["installed"] else f"{r['download_gb']} GB to download"
            print(f"{r['id']:<20} {r['kind']:<6} {state:<22} needs {r['memory_gib']} GiB  {r['license']}")
    return 0


def _pull(settings: Settings, names: list[str], limit_mbps: int | None) -> int:
    from media_gen.pull import PullError, pull

    unknown = [n for n in names if n not in MODELS]
    if unknown:
        print(f"unknown model: {', '.join(unknown)} (see `media-gen models`)", file=sys.stderr)
        return 2
    for name in names:
        try:
            pull(
                MODELS[name],
                settings.models_dir,
                limit_mbps or settings.download_limit_mbps,
                log=lambda line: print(line, flush=True),
            )
        except PullError as exc:
            print(f"{name}: {exc}", file=sys.stderr)
            return 1
    return 0


def _remove(settings: Settings, names: list[str]) -> int:
    """Delete models' files, keeping any repository another installed model still uses."""
    import shutil

    unknown = [n for n in names if n not in MODELS]
    if unknown:
        print(f"unknown model: {', '.join(unknown)}", file=sys.stderr)
        return 2
    keep = {
        manifest.name
        for model in MODELS.values()
        if model.id not in names and model.is_installed(settings.models_dir)
        for manifest in model.resolved_manifests()
    }
    for name in names:
        for manifest in MODELS[name].resolved_manifests():
            directory = manifest.directory(settings.models_dir)
            if manifest.name in keep or not directory.exists():
                continue
            shutil.rmtree(directory)
            parent = directory.parent
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        print(f"{name}: removed", flush=True)
    return 0


def _prune(settings: Settings) -> int:
    """Delete the files of models no longer curated (a release retired them), keeping every current one."""
    import shutil

    from media_gen.catalog import COMPLETE_MARKER

    current = {
        manifest.directory(settings.models_dir)
        for model in MODELS.values()
        for manifest in model.resolved_manifests()
    }
    if not settings.models_dir.is_dir():
        return 0
    for repository in sorted(settings.models_dir.iterdir()):
        if not repository.is_dir() or repository.is_symlink():
            continue
        for revision in sorted(repository.iterdir()):
            # Only what `pull` wrote: a finished download's marker, or its lock while one ran.
            ours = (revision / COMPLETE_MARKER).is_file() or (revision / ".lock").is_file()
            if revision in current or not revision.is_dir() or revision.is_symlink() or not ours:
                continue
            shutil.rmtree(revision)
            print(f"{repository.name}@{revision.name[:12]}: removed (no longer curated)", flush=True)
        if not any(repository.iterdir()):
            repository.rmdir()
    return 0


def _serve(settings: Settings) -> int:
    import logging

    import uvicorn

    from media_gen.api import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="media-gen")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="run the HTTP service")
    models = sub.add_parser("models", help="list the curated models")
    models.add_argument("--json", action="store_true")
    pull = sub.add_parser("pull", help="download and verify models")
    pull.add_argument("models", nargs="+")
    pull.add_argument("--limit-mbps", type=int, help="download cap in megabits per second")
    remove = sub.add_parser("remove", help="delete models' files")
    remove.add_argument("models", nargs="+")
    sub.add_parser("prune", help="delete the files of models this release no longer curates")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    if args.command == "models":
        return _models(settings, args.json)
    if args.command == "pull":
        return _pull(settings, args.models, args.limit_mbps)
    if args.command == "remove":
        return _remove(settings, args.models)
    if args.command == "prune":
        return _prune(settings)
    return _serve(settings)


if __name__ == "__main__":
    sys.exit(main())
