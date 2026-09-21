"""Chunk builders for the OpenAI streaming format."""

from __future__ import annotations

import json
import time
import uuid


def new_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def chunk(cid: str, model: str, delta: dict, finish: str | None = None, usage: dict | None = None) -> str:
    body = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage is not None:
        body["usage"] = usage
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"


def error_event(message: str, etype: str = "server_error", code: str | None = None) -> str:
    return f"data: {json.dumps({'error': {'message': message, 'type': etype, 'code': code}}, ensure_ascii=False)}\n\n"


DONE = "data: [DONE]\n\n"


def progress_delta(stage: str, step: int, total: int, message: str) -> dict:
    pct = int(100 * step / total) if total else 0
    return {"progress": {"stage": stage, "step": step, "total": total, "percent": pct, "message": message}}


def image_delta(data_url: str, meta: dict) -> dict:
    return {"images": [{"type": "image_url", "image_url": {"url": data_url}, "generation": meta}]}


def completion(cid: str, model: str, content: str, images: list[dict]) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if images:
        message["images"] = images
    return {
        "id": cid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
