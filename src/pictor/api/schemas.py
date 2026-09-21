"""Request models. Lenient on purpose: OpenAI clients send many fields we ignore."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class GenerationOptions(BaseModel):
    """Per-request overrides (see PROTOCOL.md); every field optional, null = server default."""

    model_config = ConfigDict(extra="ignore")

    mode: Literal["auto", "chat", "generate", "edit"] = "auto"
    base_size: int | None = Field(default=None, ge=256, le=4096)  # target side length: 768, 1024, 1536, 2048
    aspect_ratio: str | None = None  # "1:1", "16:9", ... ; null = auto (reference ratio or default)
    width: int | None = Field(default=None, ge=64, le=4096)
    height: int | None = Field(default=None, ge=64, le=4096)
    steps: int | None = Field(default=None, ge=1, le=200)
    seed: int | None = Field(default=None, ge=0)
    guidance: float | None = Field(default=None, ge=0.0, le=20.0)
    negative_prompt: str | None = None
    transparent: bool | None = None

    def as_overrides(self) -> dict[str, Any]:
        d = self.model_dump(exclude_none=True)
        if d.get("mode") == "auto":
            d.pop("mode")
        return d


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[dict[str, Any]]
    stream: bool = False
    generation: GenerationOptions | None = None


class ImageGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt: str
    model: str | None = None
    n: int = Field(default=1, ge=1, le=4)
    size: str | None = "1024x1024"
    response_format: Literal["b64_json"] = "b64_json"  # nothing is stored server-side, so no "url"
    seed: int | None = Field(default=None, ge=0)
    steps: int | None = Field(default=None, ge=1, le=200)
    guidance: float | None = None
    negative_prompt: str | None = None
    transparent: bool = False
