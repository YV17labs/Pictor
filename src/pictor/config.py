"""Runtime configuration (environment / .env driven)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PICTOR_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8091
    model_id: str = "Qwen/Qwen-Image-2.1"
    model_name: str = "qwen-image-2.1"  # id exposed on /v1/models
    preload: bool = True
    keep_loaded: bool = True
    # Unload the weights after this many seconds without a job (0 = never). Reload takes ~20 s.
    idle_unload_s: int = 900

    # Torch / diffusers backend
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    quant: Literal["none", "int8", "fp8"] = "none"  # weight-only quantization of the transformer (torchao, CUDA)
    cpu_offload: bool = False  # CUDA: keep idle sub-models in host RAM (24 GB cards)

    # Generation defaults
    default_width: int = 1024
    default_height: int = 1024
    default_steps: int = 20
    default_guidance: float = 1.0
    max_side: int = 2048
    max_steps: int = 100
    max_reference_images: int = Field(default=10, ge=1, le=10)
    max_upload_bytes: int = 40 * 1024 * 1024


def load_settings() -> Settings:
    return Settings()
