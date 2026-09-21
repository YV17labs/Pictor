from __future__ import annotations

import time

import pytest
from PIL import Image

from pictor.app import create_app
from pictor.backends.base import (
    CancelFn,
    Capabilities,
    GenerationCancelled,
    GenerationRequest,
    GenerationResult,
    ImageBackend,
    ProgressFn,
)
from pictor.config import Settings


class FakeBackend(ImageBackend):
    """Deterministic stand-in: emits progress per step and returns a flat-colour image."""

    name = "fake"

    def __init__(self, caps: Capabilities | None = None, delay: float = 0.0) -> None:
        self._caps = caps or Capabilities()
        self._loaded = False
        self.delay = delay
        self.requests: list[GenerationRequest] = []

    @property
    def device(self) -> str:
        return "fake"

    @property
    def capabilities(self) -> Capabilities:
        return self._caps

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self, on_progress: ProgressFn | None = None) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def generate(self, request: GenerationRequest, on_progress: ProgressFn, should_cancel: CancelFn) -> GenerationResult:
        self.requests.append(request)
        on_progress("encoding", 0, 1)
        for i in range(request.steps):
            if should_cancel():
                raise GenerationCancelled()
            if self.delay:
                time.sleep(self.delay)
            on_progress("generating", i + 1, request.steps)
        img = Image.new("RGBA" if request.transparent else "RGB", (request.width, request.height), (200, 30, 30, 255))
        return GenerationResult(
            image=img, seed=request.seed, width=request.width, height=request.height, steps=request.steps,
            prompt=request.prompt, duration_ms=5, backend=self.name,
            mode="reference_edit" if request.references else "text_to_image",
        )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        default_steps=3, default_width=64, default_height=64,
        max_side=256, preload=False, _env_file=None,
    )


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def app(settings, backend):
    return create_app(settings, backend=backend, preload=False)
