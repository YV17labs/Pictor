"""The image backend: Qwen's native diffusers pipeline (see torch_backend.py)."""

from __future__ import annotations

from pictor.backends.base import (
    Capabilities,
    GenerationCancelled,
    GenerationRequest,
    GenerationResult,
    ImageBackend,
    ProgressFn,
)
from pictor.config import Settings


def create_backend(settings: Settings) -> ImageBackend:
    from pictor.backends.torch_backend import TorchBackend

    return TorchBackend(settings)


__all__ = [
    "Capabilities",
    "GenerationCancelled",
    "GenerationRequest",
    "GenerationResult",
    "ImageBackend",
    "ProgressFn",
    "create_backend",
]
