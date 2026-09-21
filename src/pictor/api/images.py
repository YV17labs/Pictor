"""OpenAI Images API: POST /v1/images/generations and /v1/images/edits."""

from __future__ import annotations

import base64
import random
import time

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from pictor.api.schemas import ImageGenerationRequest
from pictor.api.state import AppState
from pictor.backends.base import GenerationRequest
from pictor.orchestrator.router import RGBA_PROMPT
from pictor.util.images import encode_png, fit_reference, open_image, parse_size, size_like

router = APIRouter()


def _state(request: Request) -> AppState:
    return request.app.state.pictor


async def _execute(state: AppState, gen: GenerationRequest) -> dict:
    """Run one job and return an OpenAI Images entry. Nothing is written to disk: b64_json only."""
    job = state.engine.submit(gen)
    async for ev in job.events():
        if ev.kind == "result" and ev.result is not None:
            return {
                "b64_json": base64.b64encode(encode_png(ev.result.image)).decode("ascii"),
                "revised_prompt": ev.result.prompt, "seed": ev.result.seed,
                "width": ev.result.width, "height": ev.result.height,
            }
        if ev.kind == "error":
            raise HTTPException(status_code=500, detail={"error": {"message": ev.error, "type": "server_error"}})
        if ev.kind == "cancelled":
            raise HTTPException(status_code=499, detail={"error": {"message": "cancelled", "type": "cancelled"}})
    raise HTTPException(status_code=500, detail={"error": {"message": "no result", "type": "server_error"}})


@router.post("/v1/images/generations")
async def generations(request: Request, body: ImageGenerationRequest):
    state = _state(request)
    s = state.settings
    caps = state.backend.capabilities
    try:
        width, height = parse_size(body.size, (s.default_width, s.default_height))
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": {"message": str(e), "type": "invalid_request_error"}}) from e
    prompt = body.prompt
    if body.transparent and caps.rgba:
        prompt = RGBA_PROMPT.format(prompt=prompt.rstrip(". "))
    data = []
    base_seed = body.seed if body.seed is not None else random.randint(0, 2**31 - 1)
    for i in range(body.n):
        gen = GenerationRequest(
            prompt=prompt, width=min(width, caps.max_side), height=min(height, caps.max_side),
            steps=min(body.steps or s.default_steps, s.max_steps), seed=base_seed + i,
            guidance=body.guidance if body.guidance is not None else s.default_guidance,
            negative_prompt=body.negative_prompt, transparent=body.transparent and caps.rgba,
        )
        data.append(await _execute(state, gen))
    return JSONResponse({"created": int(time.time()), "data": data})


@router.post("/v1/images/edits")
async def edits(
    request: Request,
    prompt: str = Form(...),
    image: list[UploadFile] = File(...),
    size: str | None = Form(default=None),
    seed: int | None = Form(default=None),
    steps: int | None = Form(default=None),
    guidance: float | None = Form(default=None),
    negative_prompt: str | None = Form(default=None),
):
    state = _state(request)
    s = state.settings
    caps = state.backend.capabilities
    if not caps.reference_edit:
        raise HTTPException(status_code=400, detail={"error": {"message": "backend cannot edit images", "type": "invalid_request_error"}})
    refs = []
    for up in image[: max(1, caps.max_reference_images)]:
        raw = await up.read()
        if len(raw) > s.max_upload_bytes:
            raise HTTPException(status_code=413, detail={"error": {"message": "image too large", "type": "invalid_request_error"}})
        try:
            refs.append(fit_reference(open_image(raw), caps.max_side))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail={"error": {"message": f"bad image: {e}", "type": "invalid_request_error"}}) from e
    if size:
        width, height = parse_size(size, (s.default_width, s.default_height))
    else:
        width, height = size_like(refs[-1], base=max(s.default_width, s.default_height), max_side=caps.max_side)
    gen = GenerationRequest(
        prompt=prompt, width=width, height=height, steps=min(steps or s.default_steps, s.max_steps),
        seed=seed if seed is not None else random.randint(0, 2**31 - 1),
        guidance=guidance if guidance is not None else s.default_guidance, negative_prompt=negative_prompt,
        references=refs,
    )
    return JSONResponse({"created": int(time.time()), "data": [await _execute(state, gen)]})
