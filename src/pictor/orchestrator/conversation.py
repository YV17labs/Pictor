"""Turn OpenAI `messages` into a compact conversation with decoded images."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
from PIL import Image

from pictor.util.images import decode_data_url, open_image

log = logging.getLogger(__name__)


@dataclass
class Turn:
    role: str
    text: str = ""
    images: list[Image.Image] = field(default_factory=list)


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
    return "\n".join(p for p in parts if p)


def _image_urls(content: Any, message: dict) -> list[str]:
    urls: list[str] = []
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("image_url", "input_image"):
                iu = part.get("image_url")
                url = iu.get("url") if isinstance(iu, dict) else iu
                if url:
                    urls.append(str(url))
    # OpenRouter-style assistant images (non-streaming shape echoed back by clients)
    for img in message.get("images") or []:
        if isinstance(img, dict):
            iu = img.get("image_url")
            url = iu.get("url") if isinstance(iu, dict) else iu
            if url:
                urls.append(str(url))
    return urls


async def _load_url(url: str, max_bytes: int, client: httpx.AsyncClient | None) -> Image.Image | None:
    try:
        if url.startswith("data:"):
            data, _ = decode_data_url(url)
            if len(data) > max_bytes:
                raise ValueError("image too large")
            return open_image(data)
        if url.startswith(("http://", "https://")) and client is not None:
            r = await client.get(url, timeout=30)
            r.raise_for_status()
            if len(r.content) > max_bytes:
                raise ValueError("image too large")
            return open_image(r.content)
    except Exception as e:  # noqa: BLE001
        log.warning("could not load image %s: %s", url[:60], e)
    return None


async def parse_messages(
    messages: list[dict],
    max_bytes: int = 40 * 1024 * 1024,
    client: httpx.AsyncClient | None = None,
) -> list[Turn]:
    turns: list[Turn] = []
    for m in messages:
        role = str(m.get("role", "user"))
        if role not in ("system", "user", "assistant"):
            continue  # tool messages carry nothing useful for an image model
        content = m.get("content")
        turn = Turn(role=role, text=_extract_text(content).strip())
        for url in _image_urls(content, m):
            img = await _load_url(url, max_bytes, client)
            if img is not None:
                turn.images.append(img)
        if turn.text or turn.images:
            turns.append(turn)
    return turns


def last_user(turns: list[Turn]) -> Turn | None:
    for t in reversed(turns):
        if t.role == "user":
            return t
    return None


def latest_images(turns: list[Turn], before: Turn | None = None, limit: int = 1) -> list[Image.Image]:
    """Most recent images in the conversation (any role), newest first, excluding `before`."""
    found: list[Image.Image] = []
    for t in reversed(turns):
        if t is before:
            continue
        for img in reversed(t.images):
            found.append(img)
            if len(found) >= limit:
                return found
    return found
