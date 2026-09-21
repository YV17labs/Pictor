"""GET /v1/models, /health."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from pictor.api.state import AppState

router = APIRouter()


def _state(request: Request) -> AppState:
    return request.app.state.pictor


def _describe(state: AppState) -> dict:
    """Backend description + generation defaults, so clients can build a settings UI."""
    s = state.settings
    d = state.backend.describe()
    d["kind"] = "image"
    d["defaults"] = {
        "steps": s.default_steps,
        "max_steps": s.max_steps,
        "base_size": max(s.default_width, s.default_height),
        "base_sizes": [b for b in (768, 1024, 1536, 2048) if b <= s.max_side] or [s.max_side],
        "aspect_ratios": ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "21:9"],
        "guidance": s.default_guidance,
    }
    return d


@router.get("/v1/models")
async def models(request: Request):
    state = _state(request)
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": state.settings.model_name,
            "object": "model",
            "created": 0,
            "owned_by": "pictor",
            "generation": _describe(state),
        }],
    })


@router.get("/v1/models/{model_id}")
async def model(request: Request, model_id: str):
    state = _state(request)
    if model_id != state.settings.model_name:
        raise HTTPException(status_code=404, detail={"error": {"message": "model not found", "type": "invalid_request_error"}})
    return JSONResponse({"id": model_id, "object": "model", "created": 0, "owned_by": "pictor",
                         "generation": _describe(state)})


@router.get("/health")
async def health(request: Request):
    state = _state(request)
    return JSONResponse({
        "status": "ok" if not state.engine.load_error else "degraded",
        "backend": state.backend.name,
        "device": state.backend.device,
        "model": state.settings.model_name,
        "loaded": state.backend.loaded,
        "loading": state.engine.loading,
        "load_error": state.engine.load_error,
        "busy": state.engine.busy,
        "queue": state.engine.queue_length,
        "capabilities": state.backend.capabilities.as_dict(),
        "memory": state.backend.describe().get("memory", {}),
    })
