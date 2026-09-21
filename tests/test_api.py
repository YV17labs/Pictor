import base64
import io
import json

import httpx
import pytest
from PIL import Image

from pictor.util.images import encode_png, to_data_url


@pytest.fixture
async def client(app):
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


def _events(text: str) -> list:
    out = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            payload = line[6:]
            out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


async def test_models_and_health(client):
    r = await client.get("/v1/models")
    assert r.status_code == 200
    m = r.json()["data"][0]
    assert m["id"] == "qwen-image-2.1" and m["generation"]["capabilities"]["reference_edit"] is True
    h = (await client.get("/health")).json()
    assert h["status"] == "ok" and h["backend"] == "fake"


async def test_chat_stream_generates_image(client, backend):
    body = {"model": "qwen-image-2.1", "stream": True,
            "messages": [{"role": "user", "content": "a lighthouse at night, seed 3"}]}
    r = await client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    ev = _events(r.text)
    assert ev[-1] == "[DONE]"
    deltas = [e["choices"][0]["delta"] for e in ev[:-1]]
    assert deltas[0]["role"] == "assistant"
    stages = [d["progress"]["stage"] for d in deltas if "progress" in d]
    assert stages[0] == "planning" and "generating" in stages and stages[-1] == "encoding_png"
    imgs = [d["images"][0] for d in deltas if "images" in d]
    assert len(imgs) == 1
    url = imgs[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    png = base64.b64decode(url.split(",", 1)[1])
    assert Image.open(io.BytesIO(png)).size == (64, 64)
    assert imgs[0]["generation"]["seed"] == 3 and "file" not in imgs[0]["generation"]
    assert ev[-2]["choices"][0]["finish_reason"] == "stop"
    text = "".join(d.get("content") or "" for d in deltas)
    assert "Generating" in text and "Done in" in text
    assert backend.requests[-1].steps == 3


async def test_chat_stream_question_is_text_only(client):
    body = {"stream": True, "messages": [{"role": "user", "content": "Hello, what can you do?"}]}
    ev = _events((await client.post("/v1/chat/completions", json=body)).text)
    deltas = [e["choices"][0]["delta"] for e in ev[:-1]]
    assert not any("images" in d for d in deltas)
    assert "Pictor" in "".join(d.get("content") or "" for d in deltas)


async def test_chat_with_attached_image_edits(client, backend):
    ref = to_data_url(encode_png(Image.new("RGB", (128, 64), "green")))
    body = {"stream": False, "messages": [
        {"role": "user", "content": [{"type": "text", "text": "make the sky orange"},
                                     {"type": "image_url", "image_url": {"url": ref}}]}]}
    r = await client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    msg = r.json()["choices"][0]["message"]
    assert msg["images"][0]["generation"]["mode"] == "reference_edit"
    req = backend.requests[-1]
    assert len(req.references) == 1 and req.width > req.height


async def test_assistant_images_are_reused(client, backend):
    prev = to_data_url(encode_png(Image.new("RGB", (64, 128), "green")))
    body = {"stream": False, "messages": [
        {"role": "user", "content": "a cat"},
        {"role": "assistant", "content": [{"type": "text", "text": "Done."},
                                          {"type": "image_url", "image_url": {"url": prev}}]},
        {"role": "user", "content": "change the background of this image to a beach"}]}
    r = await client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    req = backend.requests[-1]
    assert len(req.references) == 1 and req.height > req.width


async def test_images_generations_and_edits(client):
    r = await client.post("/v1/images/generations", json={"prompt": "a fox", "size": "64x32", "seed": 9})
    assert r.status_code == 200
    d = r.json()["data"][0]
    assert d["seed"] == 9 and Image.open(io.BytesIO(base64.b64decode(d["b64_json"]))).size == (64, 32)

    png = encode_png(Image.new("RGB", (64, 64), "green"))
    r = await client.post("/v1/images/edits", data={"prompt": "make it blue"},
                          files=[("image", ("a.png", png, "image/png"))])
    assert r.status_code == 200
    assert Image.open(io.BytesIO(base64.b64decode(r.json()["data"][0]["b64_json"]))).size == (64, 64)
    # nothing is written anywhere: no url form
    r = await client.post("/v1/images/generations", json={"prompt": "a fox", "response_format": "url"})
    assert r.status_code == 422


async def test_generation_options_override(client, backend):
    body = {"stream": False, "messages": [{"role": "user", "content": "hello"}],
            "generation": {"mode": "generate", "width": 96, "height": 64, "steps": 2, "seed": 5}}
    r = await client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    req = backend.requests[-1]
    assert (req.width, req.height, req.steps, req.seed) == (96, 64, 2, 5)


async def test_validation_error_shape(client):
    r = await client.post("/v1/chat/completions", json={"stream": True})
    assert r.status_code == 422 and "error" in r.json()


async def test_models_expose_kind_and_defaults(client):
    m = (await client.get("/v1/models")).json()["data"][0]["generation"]
    assert m["kind"] == "image"
    assert m["defaults"]["steps"] == 3 and m["defaults"]["base_sizes"] == [256]  # capped by max_side in tests
    assert "16:9" in m["defaults"]["aspect_ratios"]


async def test_base_size_and_ratio_override(client, backend):
    body = {"stream": False, "messages": [{"role": "user", "content": "a lighthouse"}],
            "generation": {"base_size": 256, "aspect_ratio": "16:9", "steps": 1}}
    r = await client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    req = backend.requests[-1]
    assert req.width > req.height and abs(req.width / req.height - 16 / 9) < 0.1
    assert 0.8 < (req.width * req.height) / (256 * 256) < 1.2  # area ~= base_size²
