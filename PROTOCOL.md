# Pictor wire protocol

Pictor exposes an **OpenAI-compatible HTTP API** so that any OpenAI client can talk to it
without a dedicated SDK. Everything specific to image generation is carried in *extension
fields* that standard clients ignore.

Default listen address: `http://127.0.0.1:8091`. All routes are under `/v1`.

Changes: **0.3.0** — the extension object is named `generation` (was `specterforge`),
`owned_by` is `pictor`.

## GET /v1/models

```json
{
  "object": "list",
  "data": [{
    "id": "qwen-image-2.1",
    "object": "model",
    "owned_by": "pictor",
    "generation": {
      "kind": "image",
      "backend": "torch",
      "device": "mps",
      "loaded": true,
      "capabilities": {
        "text_to_image": true,
        "reference_edit": true,
        "max_reference_images": 10,
        "rgba": true,
        "max_side": 2048
      },
      "defaults": {
        "steps": 20, "max_steps": 100,
        "base_size": 1024, "base_sizes": [768, 1024, 1536, 2048],
        "aspect_ratios": ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "21:9"],
        "guidance": 1.0
      }
    }
  }]
}
```

**A client recognises an image model by `generation.kind == "image"`** on a model entry
(any other model, or a server without this field, is a text LLM). `capabilities` says what
the model can do; `defaults` are the values a settings UI should start from and the allowed
choices.

## POST /v1/chat/completions

The conversational route. The server reads the whole conversation: the last user text is
the prompt (passed to the model as written), attached images are the references (an
attached image means "edit"; none means "generate"; a short question gets a help answer),
and previously generated images are reused when the user says "this image" without
attaching anything.

### Request

Standard OpenAI chat body. Fields the server reads:

| Field | Notes |
|---|---|
| `model` | Any id from `/v1/models`. |
| `messages[]` | `role` ∈ `system` / `user` / `assistant`. `content` is a string or an array of parts: `{"type":"text","text":…}` or `{"type":"image_url","image_url":{"url":…}}`. URLs are `data:` URLs (base64) or plain `http(s)` URLs. Assistant messages may carry `image_url` parts too (previously generated images), which is how iterative editing works: **the server keeps nothing**, the client re-sends what it wants edited. |
| `stream` | `true` for SSE. |
| `generation` | Optional per-request overrides, see below. |

Everything else (`temperature`, `tools`, `max_tokens`, …) is accepted and ignored.

```json
"generation": {
  "mode": "auto | chat | generate | edit",
  "base_size": 1024,          // target side length; area ≈ base_size², see defaults.base_sizes
  "aspect_ratio": "16:9",     // preset from defaults.aspect_ratios; omit/null = auto
  "width": 1024, "height": 1024,   // explicit size (overrides base_size + aspect_ratio)
  "steps": 40,
  "seed": 42,                 // omit for random
  "guidance": 1.0,            // > 1 enables true CFG (needs negative_prompt), doubles the time
  "negative_prompt": "",
  "transparent": false        // RGBA output
}
```

Every field is optional; omitted fields fall back to the server defaults. A client should
only send this object when the selected model has `kind == "image"`. Text-LLM sampling
fields (`temperature`, `top_p`, `max_tokens`, penalties, `tools`, `system` messages) are
accepted and ignored by the image server, so a client may hide them in image mode.

### Streaming response (`text/event-stream`)

Each event is `data: <json>\n\n`; the stream ends with `data: [DONE]\n\n`.
Chunks follow the OpenAI shape:

```json
{"id":"chatcmpl-…","object":"chat.completion.chunk","created":…,"model":"qwen-image-2.1",
 "choices":[{"index":0,"delta":{…},"finish_reason":null}]}
```

`delta` may contain, in addition to the standard `role` and `content`:

**`progress`** (extension) — transient, never persisted by the client:

```json
"progress": {"stage": "generating", "step": 12, "total": 40, "percent": 30, "message": "Generating 12/40"}
```

`stage` ∈ `planning` / `queued` / `loading` / `encoding` / `generating` / `decoding` / `encoding_png`.
The same progress chunk may be re-sent every 10 s as a heartbeat while the worker is silent
(model load); clients should treat identical consecutive progress chunks as no-ops.

**`images`** (OpenRouter convention) — one entry per produced image:

```json
"images": [{
  "type": "image_url",
  "image_url": {"url": "data:image/png;base64,…"},
  "generation": {"seed": 42, "width": 1024, "height": 1024, "steps": 40,
                 "prompt": "<prompt sent to the model>", "duration_ms": 61234,
                 "backend": "torch", "mode": "text_to_image | reference_edit"}
}]
```

The final chunk carries `finish_reason: "stop"`. An error after the stream has
started is emitted as `data: {"error": {"message": …, "type": …, "code": …}}`
followed by `[DONE]`; before the stream starts it is a normal HTTP 4xx/5xx JSON
error `{"error": {...}}`.

**Cancellation**: the client closes the connection. The server cancels the running
job at the next denoising step.

### Non-streaming response

`choices[0].message` = `{"role":"assistant","content":"…","images":[…]}` with the
same `images` entries as above.

## POST /v1/images/generations (OpenAI Images API)

```json
{"prompt": "…", "n": 1, "size": "1024x1024", "response_format": "b64_json",
 "seed": 42, "steps": 40, "negative_prompt": "", "guidance": 1.0, "transparent": false}
```

→ `{"created": …, "data": [{"b64_json": "…", "revised_prompt": "…", "seed": 42, "width": …, "height": …}]}`
(`b64_json` only: the server stores nothing, so there is no `url` form).

## POST /v1/images/edits (multipart/form-data)

Fields: `image` (repeatable, up to `max_reference_images`, in reading order), `prompt`,
`size`, `seed`, `steps`, `guidance`, `negative_prompt`. Same response as generations.

## GET /health

`{"status":"ok","backend":"torch","device":"mps","model":"qwen-image-2.1","loaded":true,"busy":false,"queue":0,
"capabilities":{…},"memory":{"allocated_gb":32.5,"driver_gb":36.0}}`
