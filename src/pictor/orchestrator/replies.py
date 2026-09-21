"""Sentences the assistant writes itself (status lines). Prompts are never touched."""

from __future__ import annotations

_T: dict[str, str] = {
    "help": (
        "I am Pictor, a local image server. Current model: Qwen-Image-2.1 (native pipeline).\n\n"
        "- **Create**: describe the picture, text to render in double quotes.\n"
        "- **Edit**: attach one or more images (up to {n}) and say what to change. Outline areas in colour "
        "or paint a mask for a local edit, and name the colours.\n"
        "- **Options** inline: \"seed 42\", \"30 steps\", \"transparent background\", or the Image panel.\n\n"
        "Backend: {backend} ({device})."
    ),
    "image_only": "Got the image. What should I do with it? For example: \"replace the background with a sunset beach\".",
    "intro_generate": "Generating: “{prompt}”\n{w}×{h}, {steps} steps, seed {seed}.",
    "intro_edit": "Editing the image ({n} reference{s}): “{prompt}”\n{w}×{h}, {steps} steps, seed {seed}.",
    "done": "Done in {sec} s (seed {seed}). Tell me what to change, or ask for a variation.",
    "cancelled": "Generation cancelled.",
    "error": "Generation failed: {error}",
    "note_references_truncated": "Note: only the first {n} reference images were used.",
    "note_rgba_not_produced": "Note: the model did not produce an alpha channel for this image.",
    "progress_planning": "Reading the request…",
    "progress_queued": "Queued…",
    "progress_loading": "Loading the model…",
    "progress_encoding": "Encoding the prompt…",
    "progress_generating": "Generating {step}/{total}",
    "progress_decoding": "Decoding the image…",
    "progress_encoding_png": "Encoding the PNG…",
}


def t(key: str, **kw) -> str:
    s = _T.get(key) or key
    try:
        return s.format(**kw)
    except (KeyError, IndexError):
        return s
