"""Image helpers: data URLs, sizing rules shared by every backend."""

from __future__ import annotations

import base64
import io
import math
import re
from fractions import Fraction

from PIL import Image

MULTIPLE = 32  # Qwen-Image-2.1 latent = 16 px, diffusers rounds to 32

_DATA_URL = re.compile(r"^data:(?P<mime>[\w/+.-]+)?(?:;charset=[\w-]+)?(?P<b64>;base64)?,(?P<data>.*)$", re.S)

ASPECT_PRESETS: dict[str, tuple[int, int]] = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "3:2": (3, 2),
    "2:3": (2, 3),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "21:9": (21, 9),
}


def decode_data_url(url: str) -> tuple[bytes, str]:
    """Return (bytes, mime) for a data: URL. Raises ValueError otherwise."""
    m = _DATA_URL.match(url.strip())
    if not m:
        raise ValueError("not a data URL")
    mime = m.group("mime") or "application/octet-stream"
    data = m.group("data")
    if m.group("b64"):
        return base64.b64decode(data), mime
    from urllib.parse import unquote_to_bytes

    return unquote_to_bytes(data), mime


def open_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    return img


def encode_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def to_data_url(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def round_to_multiple(value: int, multiple: int = MULTIPLE) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def clamp_size(width: int, height: int, max_side: int, multiple: int = MULTIPLE) -> tuple[int, int]:
    """Round to the latent multiple and scale down so the longest side <= max_side."""
    scale = min(1.0, max_side / max(width, height))
    w = round_to_multiple(int(width * scale), multiple)
    h = round_to_multiple(int(height * scale), multiple)
    return min(w, max_side), min(h, max_side)


def size_for_ratio(ratio: str | float | Fraction, base: int = 1024, max_side: int = 2048) -> tuple[int, int]:
    """Width/height with area ~= base*base and the requested aspect ratio."""
    if isinstance(ratio, str):
        if ratio in ASPECT_PRESETS:
            rw, rh = ASPECT_PRESETS[ratio]
            r = rw / rh
        else:
            a, b = ratio.replace("x", ":").split(":")
            r = float(a) / float(b)
    else:
        r = float(ratio)
    area = base * base
    w = math.sqrt(area * r)
    h = w / r
    return clamp_size(int(w), int(h), max_side)


def size_like(img: Image.Image, base: int = 1024, max_side: int = 2048) -> tuple[int, int]:
    """Output size that keeps the aspect ratio of a reference image."""
    return size_for_ratio(Fraction(img.width, img.height), base=base, max_side=max_side)


def parse_size(size: str | None, default: tuple[int, int]) -> tuple[int, int]:
    """OpenAI-style 'WxH' or an aspect preset like '16:9'."""
    if not size or size == "auto":
        return default
    s = size.lower().strip()
    if s in ASPECT_PRESETS:
        return size_for_ratio(s, base=max(default))
    if "x" in s:
        w, h = s.split("x", 1)
        return int(w), int(h)
    raise ValueError(f"unsupported size {size!r}")


def fit_reference(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale a reference image so its longest side <= max_side (keeps mode)."""
    if max(img.size) <= max_side:
        return img
    scale = max_side / max(img.size)
    return img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)


def flatten_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return img.convert("RGB")
