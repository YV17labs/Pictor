"""Qwen's native pipeline: diffusers `QwenImage21Pipeline` (PyTorch).

Runs on Metal (MPS) on Apple Silicon and on CUDA on NVIDIA machines with the same code
and the same weights: text-to-image, instruction editing with up to 10 reference images
(including coloured outlines / masks drawn on them), identity preservation, RGBA output.

Memory (1024², bf16): ~33 GB of weights. On a 24 GB CUDA card use `PICTOR_CPU_OFFLOAD`
and/or `PICTOR_QUANT=int8`. MPS is verified on an M4 Pro 48 GB; CUDA is untested here.
"""

from __future__ import annotations

import gc
import logging
import threading
import time

from PIL import Image

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
from pictor.util.images import fit_reference

log = logging.getLogger(__name__)


class TorchBackend(ImageBackend):
    name = "torch"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipe = None
        self._device = self._pick_device(settings.device)
        self._lock = threading.Lock()

    # -- metadata ---------------------------------------------------------------------
    @staticmethod
    def _pick_device(requested: str) -> str:
        if requested != "auto":
            return requested
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @property
    def device(self) -> str:
        return self._device

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            text_to_image=True,
            reference_edit=True,
            max_reference_images=self._settings.max_reference_images,
            rgba=True,
            max_side=self._settings.max_side,
        )

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def describe(self) -> dict:
        d = super().describe()
        d["quantization"] = self._settings.quant
        d["cpu_offload"] = self._settings.cpu_offload
        d["memory"] = self._memory_stats()
        return d

    def _memory_stats(self) -> dict:
        try:
            import torch

            if self._device == "cuda":
                return {"allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 1),
                        "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
            if self._device == "mps":
                return {"allocated_gb": round(torch.mps.current_allocated_memory() / 1e9, 1),
                        "driver_gb": round(torch.mps.driver_allocated_memory() / 1e9, 1)}
        except Exception:  # noqa: BLE001
            pass
        return {}

    # -- lifecycle --------------------------------------------------------------------
    def load(self, on_progress: ProgressFn | None = None) -> None:
        with self._lock:
            if self._pipe is not None:
                return
            import torch
            from diffusers import QwenImage21Pipeline

            if on_progress:
                on_progress("loading", 0, 1)
            t0 = time.time()
            dtype = torch.bfloat16 if self._device != "cpu" else torch.float32
            pipe = QwenImage21Pipeline.from_pretrained(self._settings.model_id, dtype=dtype)
            self._apply_quantization(pipe)
            if self._settings.cpu_offload and self._device == "cuda":
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(self._device)
            self._pipe = pipe
            log.info("QwenImage21Pipeline ready on %s in %.1fs", self._device, time.time() - t0)
            if on_progress:
                on_progress("loading", 1, 1)

    def _apply_quantization(self, pipe) -> None:  # noqa: ANN001
        quant = self._settings.quant
        if quant == "none":
            return
        try:
            from torchao.quantization import quantize_

            if quant == "int8":
                from torchao.quantization import Int8WeightOnlyConfig as Cfg
            else:
                from torchao.quantization import Float8WeightOnlyConfig as Cfg
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("PICTOR_QUANT needs torchao: uv pip install torchao") from e
        log.info("quantizing transformer weights with torchao %s", quant)
        quantize_(pipe.transformer, Cfg())

    def unload(self) -> None:
        with self._lock:
            self._pipe = None
            gc.collect()
            try:
                import torch

                if self._device == "cuda":
                    torch.cuda.empty_cache()
                elif self._device == "mps":
                    torch.mps.empty_cache()
            except Exception:  # pragma: no cover
                pass

    # -- generation -------------------------------------------------------------------
    def generate(self, request: GenerationRequest, on_progress: ProgressFn, should_cancel: CancelFn) -> GenerationResult:
        if self._pipe is None:
            self.load(on_progress)
        if should_cancel():
            raise GenerationCancelled()
        import torch

        refs = [fit_reference(img, self._settings.max_side) for img in request.references]
        refs = refs[: self._settings.max_reference_images]
        mode = "reference_edit" if refs else "text_to_image"
        notes: list[str] = []
        if len(request.references) > len(refs):
            notes.append("references_truncated")

        total = request.steps

        def _on_step(pipe, step: int, timestep, kwargs):  # noqa: ANN001
            on_progress("generating", step + 1, total)
            if should_cancel():
                raise GenerationCancelled()
            return {}

        gen_device = "cpu" if self._device == "mps" else self._device
        generator = torch.Generator(gen_device).manual_seed(request.seed)

        on_progress("encoding", 0, 1)
        t0 = time.time()
        try:
            out = self._pipe(
                prompt=request.prompt,
                image=refs or None,
                negative_prompt=request.negative_prompt or None,
                true_cfg_scale=request.guidance,
                height=request.height,
                width=request.width,
                num_inference_steps=request.steps,
                generator=generator,
                callback_on_step_end=_on_step,
                output_resolution=max(request.width, request.height),
            )
        finally:
            if self._device == "cuda":
                torch.cuda.empty_cache()
            elif self._device == "mps":
                torch.mps.empty_cache()
        on_progress("decoding", 1, 1)

        pil: Image.Image = out.images[0]
        if request.transparent and pil.mode != "RGBA":
            notes.append("rgba_not_produced")
        elif not request.transparent and pil.mode == "RGBA":
            pil = pil.convert("RGB")  # the pipeline always decodes 4 channels; alpha is ~255 for opaque outputs
        log.info("%s %dx%d in %.1fs — %s", mode, pil.width, pil.height, time.time() - t0, self._memory_stats())
        return GenerationResult(
            image=pil, seed=request.seed, width=pil.width, height=pil.height, steps=request.steps,
            prompt=request.prompt, duration_ms=int((time.time() - t0) * 1000), backend=self.name,
            mode=mode, notes=notes,
        )
