# Pictor — development guide

Python 3.13 / `uv` project. FastAPI server exposing an OpenAI-compatible API in front of a
local image model. **Positioning**: a gateway that puts any local image model inside an
OpenAI conversation, on Metal (Mac) and CUDA (NVIDIA) from one codebase. Clients are
external products and are never named here; the wire contract is `PROTOCOL.md`.

Today one backend / one model: Qwen-Image-2.1 through `diffusers.QwenImage21Pipeline`.

## Layout

```
src/pictor/
  config.py        Settings (pydantic-settings, PICTOR_* env / .env)
  cli.py           serve | doctor | generate
  app.py           FastAPI factory + lifespan (creates AppState)
  api/             chat.py (/v1/chat/completions), images.py (Images API), misc.py (models, health),
                   schemas.py (GenerationOptions = the `generation` request object), sse.py, state.py
  orchestrator/    conversation.py (OpenAI messages -> Turns with PIL images)
                   router.py (Turns -> Plan: chat | generate | edit; params from inline tokens/overrides)
                   replies.py (status strings the server writes itself)
  engine/          queue.py (one worker thread, FIFO, idle unload), jobs.py (Job, events bridged to asyncio)
  backends/        base.py (ImageBackend, GenerationRequest/Result, Capabilities)
                   torch_backend.py (diffusers QwenImage21Pipeline, MPS verified, CUDA untested)
  util/images.py   data URLs, sizing (multiples of 32, aspect presets)
scripts/           macOS LaunchAgent (com.yv17labs.pictor)
tests/             pytest, FakeBackend in conftest.py — no model needed
```

## Rules

- **Backends are dumb**: they take a `GenerationRequest` and return a `GenerationResult`.
  All decisions (mode, prompt rewriting, sizes, fallbacks) live in `orchestrator/router.py`.
  A backend announces what it can do through `Capabilities`; the router degrades gracefully.
- **One worker thread, one model in memory**. Backends are driven from `engine/queue.py` only.
  Never call a backend from the event loop.
- **Progress + cancellation are mandatory** in a backend: call `on_progress` per step and
  raise `GenerationCancelled` when `should_cancel()` is true.
- **Protocol changes go to PROTOCOL.md first.** Extension fields (`progress`, `images`,
  `generation`) must stay ignorable by standard OpenAI clients. The extension object is named
  `generation` on purpose: it does not depend on the project name.
- **The server stores nothing**: images travel inline; the client re-sends what it wants edited.
- Strings the server says come from `orchestrator/replies.py`. Code, comments and docs in English.
- Known leaks to fix when a second model lands: `RGBA_PROMPT` (router.py, Qwen's transparency
  prompt) and `MULTIPLE = 32` (util/images.py) belong to the backend, not the orchestrator.

## Roadmap (agreed 2026-09-20)

1. Model registry: several models declared in config, all listed on `/v1/models`, one loaded
   at a time, swap on demand.
2. CUDA validation on a rented GPU (bf16 on 48 GB; int8 + cpu_offload on 24 GB), `torchao` in a
   `cuda` extra, a Dockerfile.
3. Optional markdown fallback (`![](data:…)` in `content`) for clients that do not read `images`.

## Commands

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
uv run pictor doctor
uv run pictor serve
```

Smoke test with the real model: `uv run pictor generate "a red fox" --steps 8 --width 512 --height 512`,
then an edit: `uv run pictor generate "make it red" --image <png> --steps 8`.

## Memory notes (M4 Pro 48 GB, bf16)

Weights ~33 GB; the machine has room for nothing else while generating. ~10 s/step at 1024²,
~20 s to load. Idle unload after 15 min (`PICTOR_IDLE_UNLOAD_S`); reload ~20 s.
