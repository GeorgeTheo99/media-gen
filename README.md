# media-gen

Local image and video generation for Apple silicon, behind OpenAI-compatible
endpoints. Built for [Model Gateway](https://github.com/GeorgeTheo99/model-gateway),
which installs it with `model-gateway media add MODEL` and routes to it, but it
runs on its own too.

| Model | Kind | Download | Memory | Speed (M3 Ultra) |
|---|---|---|---|---|
| `flux2-klein-4b` | image | 16 GB | 24 GiB | ~8 s at 1024² |
| `z-image-turbo` | image (strong text rendering) | 33 GB | 40 GiB | ~40 s at 1024² |
| `ltx-2.3-distilled` | video with synchronized audio | 49 GB | 48 GiB | ~95 s for 5 s at 768×512 |

Engines: [mflux](https://github.com/filipstrand/mflux) for images and
[mlx-video-with-audio](https://github.com/james-see/mlx-video-with-audio) for
LTX-2.3, both pinned. Each model is a set of Hugging Face repositories pinned at
a commit with every file's size and hash (`src/media_gen/manifests/`).

## Run

```bash
uv sync --extra engines
uv run media-gen models                          # curated models and what is installed
uv run media-gen pull flux2-klein-4b             # HTTPS only, resumable, rate-limited, verified
uv run media-gen serve                           # 127.0.0.1:9112
```

```bash
curl -s localhost:9112/v1/images/generations -H 'content-type: application/json' \
  -d '{"model": "flux2-klein-4b", "prompt": "a lighthouse at dusk", "size": "1024x1024"}'

curl -s localhost:9112/v1/videos -H 'content-type: application/json' \
  -d '{"model": "ltx-2.3-distilled", "prompt": "waves on a beach at sunset", "seconds": "5"}'
# poll GET /v1/videos/{id} until "completed", then GET /v1/videos/{id}/content
```

## API

- `POST /v1/images/generations`: `model`, `prompt`, `n` (1-4), `size` (`auto`
  or `WxH`, multiples of 16), `output_format` (`png`, `jpeg`, `webp`),
  `output_compression`, plus `seed`, `steps` and `guidance`. Returns
  `b64_json` only. `quality`, `style`, `user` and `moderation` are accepted and
  ignored; anything else unknown is a 400.
- `POST /v1/videos` (JSON, or multipart with an `input_reference` image for
  image-to-video): `model`, `prompt`, `seconds` (1-12), `size` (snapped to
  multiples of 64, so `1280x720` becomes `1280x704`; up to 1280×768 pixels), `seed`. Then `GET /v1/videos`, `GET /v1/videos/{id}`,
  `GET /v1/videos/{id}/content` and `DELETE /v1/videos/{id}` (cancels a queued
  or running video). Videos expire after `MEDIA_GEN_VIDEO_RETENTION_HOURS` (24).
- `GET /v1/models`, `GET /health`.

Errors use the OpenAI envelope. A busy queue answers 429 and a model that
cannot fit in memory 503 `insufficient_memory`, both with `Retry-After`.

## How it runs

One job runs at a time, in a spawned worker process holding one model. The
HTTP server never imports MLX, so unloading or switching a model frees all of
its memory, a cancelled job stops at once, and a crash (or the kernel's memory
killer) fails only that job. An idle model unloads after
`MEDIA_GEN_IDLE_UNLOAD_SECONDS` (600). Video jobs are stored in SQLite and
survive restarts: queued ones run again, interrupted ones fail.

Before a load, media-gen checks free memory. With `MEDIA_GEN_OMLX_URL` (and
`MEDIA_GEN_OMLX_API_KEY_FILE`) it unloads oMLX's least recently used idle
models, never pinned, loading or recently used ones
(`MEDIA_GEN_OMLX_BUSY_SECONDS`, 300), until the model fits.

## Settings

| Variable | Default |
|---|---|
| `MEDIA_GEN_HOME` | `~/Library/Application Support/media-gen` (models, outputs, jobs) |
| `MEDIA_GEN_HOST`, `MEDIA_GEN_PORT` | `127.0.0.1`, `9112` |
| `MEDIA_GEN_API_KEY_FILE` | unset: no key required |
| `MEDIA_GEN_MEMORY_RESERVE_GIB` | `16` |
| `MEDIA_GEN_MAX_QUEUED_JOBS` | `16` |
| `MEDIA_GEN_DOWNLOAD_LIMIT_MBPS` | `200` |

Videos with audio need `ffmpeg` on `PATH`; without it, videos are silent.

## Develop

```bash
uv sync                      # no engines needed: tests use a fake backend in real worker processes
uv run pytest -q
uv run ruff check src tests scripts
```

Pin a new model with `scripts/pin_manifest.py REPO --revision COMMIT --include PATTERN ... --out src/media_gen/manifests/NAME.json`.

## Licenses

media-gen is Apache-2.0. Models keep their own licenses: FLUX.2 [klein] 4B and
Z-Image Turbo are Apache-2.0; LTX-2.3 is under the LTX-2 Community License and
its Gemma 3 text encoder under the Gemma terms.
