"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from pictor import __version__
from pictor.api import chat, images, misc
from pictor.api.state import AppState
from pictor.backends import create_backend
from pictor.backends.base import ImageBackend
from pictor.config import Settings, load_settings
from pictor.engine.queue import Engine
from pictor.orchestrator.router import Router

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, backend: ImageBackend | None = None, preload: bool | None = None) -> FastAPI:
    settings = settings or load_settings()
    backend = backend or create_backend(settings)
    do_preload = settings.preload if preload is None else preload

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = AppState(
            settings=settings,
            backend=backend,
            engine=Engine(backend, settings),
            router=Router(settings),
            http=httpx.AsyncClient(follow_redirects=True),
        )
        app.state.pictor = state
        state.engine.start(preload=do_preload)
        log.info("Pictor %s — %s on %s, model=%s (images are returned inline, nothing is stored)",
                 __version__, backend.name, backend.device, settings.model_id)
        try:
            yield
        finally:
            state.engine.stop()
            await state.http.aclose()

    app = FastAPI(title="Pictor", version=__version__, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(misc.router)
    app.include_router(chat.router)
    app.include_router(images.router)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) and "error" in exc.detail else {
            "error": {"message": str(exc.detail), "type": "invalid_request_error", "code": None}
        }
        return JSONResponse(detail, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse({"error": {"message": str(exc.errors()[:3]), "type": "invalid_request_error", "code": None}},
                            status_code=422)

    return app
