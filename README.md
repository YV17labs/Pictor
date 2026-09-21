# Pictor

OpenAI-compatible gateway for **local image models**. Any client that speaks the OpenAI chat
API can generate and edit images through it: write the prompt in a chat window, attach
images, iterate on the result. Runs on Metal (MPS) on Apple Silicon and on CUDA on NVIDIA
machines with the same code and the same weights.

The model shipped today is **[Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)**,
through Qwen's native diffusers pipeline, with every feature of the model:

- text-to-image up to 2K, strong text rendering;
- instruction editing of one or more attached images (up to 10 references), identity preserved;
- local edits guided by coloured outlines or masks drawn on the reference;
- multi-reference composition (put this object into that scene);
- transparent (RGBA) output.

Backends are pluggable (`backends/base.py`): a model announces what it can do through
`Capabilities` and the server degrades gracefully. One model is loaded at a time.

## Requirements

| | Mac | NVIDIA |
|---|---|---|
| Hardware | Apple Silicon, **48 GB** unified memory (weights are ~33 GB in bf16) | 48 GB VRAM for bf16; 24 GB with `PICTOR_QUANT=int8` + `PICTOR_CPU_OFFLOAD=true` (untested) |
| Python | 3.13 (`uv` manages it) | 3.13 |
| Disk | 33 GB Hugging Face checkpoint (downloaded on first start) | 33 GB |

## Install and run

```bash
uv sync --extra dev
cp .env.example .env            # adjust if you like
uv run pictor doctor            # checks torch, diffusers, the weights
uv run pictor serve             # http://127.0.0.1:8091 — first start downloads the weights
```

To have the server start at login on macOS without a terminal:

```bash
scripts/install-launchagent.sh          # --uninstall to remove
```

In your client, set the API base URL to `http://127.0.0.1:8091/v1` and pick the model
`qwen-image-2.1`. `GET /v1/models` describes the model's capabilities and defaults (sizes,
aspect ratios, steps, guidance, transparency) so a client can build its settings panel from it.

## How to talk to it

Prompts go to the model **as written** — Qwen-Image-2.1 understands English best, keep text to
render in double quotes. A few inline tokens are understood: `seed 42`, `30 steps`, `16:9` /
`portrait` / `square`, `transparent background`, `2K`.

- No image attached → generation. « A neon shop sign that reads "QWEN", rainy night »
- Image(s) attached → editing. « Repaint the car in glossy Ferrari red, keep everything else »
- Outlines or a mask drawn on the image → « In the red circle, change the hair to black.
  Remove the watch in the blue circle. »
- Several images → composition. « Put the sticker from the second image on the hood of the car
  in the first image »
- « transparent background » → RGBA PNG.
- Say « this image » / « the previous image » without attaching anything to edit the last
  result again (the client re-sends the conversation; the server keeps nothing).

## Timings (M4 Pro, 48 GB)

| | |
|---|---|
| Model load (after 15 min idle or at start) | ~20 s |
| 1024×1024, 20 steps | ~3 min 40 (~10 s/step) |
| Edit at 768, 12 steps | ~1 min 20 (~6 s/step) |
| 40 steps (Qwen's recommendation) | twice the above |

768 px halves the time of 1024; 12 steps nearly halves it again. 2K is several times slower
(not measured).

## API

See [PROTOCOL.md](PROTOCOL.md). Short version:

- `POST /v1/chat/completions` — the conversational route; streams `progress` and `images` deltas.
- `POST /v1/images/generations`, `POST /v1/images/edits` — OpenAI Images API.
- `GET /v1/models`, `GET /health`.

Images travel inline (base64) in an `images` array on the assistant message (OpenRouter
convention); the server never writes an image to disk. A client has to know that field to
display the picture — a plain OpenAI client only shows the text status lines.

```bash
curl -N localhost:8091/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen-image-2.1", "stream": true,
  "messages": [{"role": "user", "content": "a red fox in the snow, 16:9, seed 7"}]}'
```

## Configuration

Environment variables / `.env` entries prefixed `PICTOR_` (see `.env.example`): device,
CUDA quantization and offload, generation defaults, idle unload.

## Development

```bash
uv run pytest            # fake backend, no model needed
uv run ruff check src tests
uv run pictor generate "a fox" --steps 8 --width 512 --height 512 --out fox.png   # real model, no server
uv run pictor generate "make it red" --image photo.png --steps 12 --out red.png    # edit
```

## License

MIT for this code. Qwen-Image-2.1 weights are under the Qwen Research License.
