"""Objects shared by the routes, created once in the app lifespan."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from pictor.backends.base import ImageBackend
from pictor.config import Settings
from pictor.engine.queue import Engine
from pictor.orchestrator.router import Router


@dataclass
class AppState:
    settings: Settings
    backend: ImageBackend
    engine: Engine
    router: Router
    http: httpx.AsyncClient
