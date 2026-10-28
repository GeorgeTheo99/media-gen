"""Write a pinned download manifest for one Hugging Face repository.

    uv run python scripts/pin_manifest.py Tongyi-MAI/Z-Image-Turbo \
        --revision <commit> --include 'transformer/*' --include 'vae/*' ... \
        --out src/media_gen/manifests/z-image-turbo.json

The manifest records every selected file's size and hash at one commit: the
LFS SHA-256 for large files, the Git blob SHA-1 for the rest. `media-gen pull`
downloads exactly these files and verifies each one.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
from pathlib import Path

import httpx

COMMIT = re.compile(r"^[0-9a-f]{40}$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("repo_id")
    parser.add_argument("--revision", required=True, help="full 40-character commit")
    parser.add_argument("--include", action="append", required=True, help="fnmatch pattern (repeatable)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not COMMIT.match(args.revision):
        parser.error("--revision must be a full commit hash")

    response = httpx.get(
        f"https://huggingface.co/api/models/{args.repo_id}/revision/{args.revision}",
        params={"blobs": "true"},
        # Gated repositories list their files only to an account that accepted their terms.
        headers={"Authorization": f"Bearer {token}"} if (token := os.environ.get("HF_TOKEN")) else {},
        timeout=30,
    )
    response.raise_for_status()
    info = response.json()
    if info["sha"] != args.revision:
        raise SystemExit(f"Hugging Face resolved {args.revision} to {info['sha']}")

    files = []
    for sibling in sorted(info["siblings"], key=lambda s: s["rfilename"]):
        path = sibling["rfilename"]
        if not any(fnmatch.fnmatch(path, pattern) for pattern in args.include):
            continue
        lfs = sibling.get("lfs")
        if lfs:
            files.append({"path": path, "size": lfs["size"], "sha256": lfs["sha256"]})
        else:
            files.append({"path": path, "size": sibling["size"], "git_sha1": sibling["blobId"]})
    if not files:
        raise SystemExit("no files matched --include")

    manifest = {"schema": 1, "repo_id": args.repo_id, "revision": args.revision, "files": files}
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(f["size"] for f in files)
    print(f"{args.out}: {len(files)} files, {total / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
