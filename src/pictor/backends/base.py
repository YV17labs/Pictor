"""Backend contract. The single implementation is torch_backend.py (diffusers).

Kept as an interface so a second implementation (another framework, a remote GPU box)
can be dropped in without touching the API or the router.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from PIL import Image

# on_progress(stage, step, total). Stages: loading, encoding, generating, decoding.
ProgressFn = Callable[[str, int, int], None]
# should_cancel() -> True when the caller wants the job aborted at the next safe point.
CancelFn = Callable[[], bool]


class GenerationCancelled(Exception):
    """Raised inside a backend when should_cancel() turned true."""


@dataclass(frozen=True)
class Capabilities:
    text_to_image: bool = True
    reference_edit: bool = True  # instruction editing conditioned on reference images
    max_reference_images: int = 10
    rgba: bool = True
    max_side: int = 2048

    def as_dict(self) -> dict:
        return {
            "text_to_image": self.text_to_image,
            "reference_edit": self.reference_edit,
            "max_reference_images": self.max_reference_images,
            "rgba": self.rgba,
            "max_side": self.max_side,
        }


@dataclass
class GenerationRequest:
    prompt: str
    width: int
    height: int
    steps: int
    seed: int
    guidance: float = 1.0
    negative_prompt: str | None = None
    references: list[Image.Image] = field(default_factory=list)  # condition images, in reading order
    transparent: bool = False  # RGBA output

    @property
    def is_edit(self) -> bool:
        return bool(self.references)


@dataclass
class GenerationResult:
    image: Image.Image
    seed: int
    width: int
    height: int
    steps: int
    prompt: str
    duration_ms: int
    backend: str
    mode: str  # text_to_image | reference_edit
    notes: list[str] = field(default_factory=list)  # human-readable caveats (keys into replies.py)


class ImageBackend(ABC):
    name: str = "base"

    @property
    @abstractmethod
    def device(self) -> str: ...

    @property
    @abstractmethod
    def capabilities(self) -> Capabilities: ...

    @property
    @abstractmethod
    def loaded(self) -> bool: ...

    @abstractmethod
    def load(self, on_progress: ProgressFn | None = None) -> None:
        """Load weights into memory. Called from the engine worker thread."""

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
        on_progress: ProgressFn,
        should_cancel: CancelFn,
    ) -> GenerationResult:
        """Run one job. Must call on_progress per denoising step and honour should_cancel."""

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "device": self.device,
            "loaded": self.loaded,
            "capabilities": self.capabilities.as_dict(),
        }
