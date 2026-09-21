"""Job objects shared between the HTTP layer (asyncio) and the worker thread."""

from __future__ import annotations

import asyncio
import enum
import threading
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from pictor.backends.base import GenerationRequest, GenerationResult

TIMEOUT = object()


class JobStatus(enum.StrEnum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


@dataclass(frozen=True)
class JobEvent:
    kind: str  # progress | result | error | cancelled
    stage: str | None = None
    step: int = 0
    total: int = 0
    result: GenerationResult | None = None
    error: str | None = None

    @property
    def percent(self) -> int:
        if self.total <= 0:
            return 0
        return max(0, min(100, int(100 * self.step / self.total)))


@dataclass
class Job:
    request: GenerationRequest
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: float = field(default_factory=time.time)
    status: JobStatus = JobStatus.queued
    loop: asyncio.AbstractEventLoop | None = None
    _queue: asyncio.Queue[JobEvent | None] = field(default_factory=asyncio.Queue)
    _cancel: threading.Event = field(default_factory=threading.Event)
    _last_emit: float = 0.0

    # -- worker side (any thread) -----------------------------------------------------
    def emit(self, event: JobEvent) -> None:
        if self.loop is None:
            return
        # Progress can be chatty; the queue is unbounded so put_nowait never blocks.
        self.loop.call_soon_threadsafe(self._queue.put_nowait, event)

    def emit_progress(self, stage: str, step: int, total: int) -> None:
        self.emit(JobEvent(kind="progress", stage=stage, step=step, total=total))

    def finish(self, event: JobEvent | None) -> None:
        if event is not None:
            self.emit(event)
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._queue.put_nowait, None)

    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    # -- HTTP side (event loop) ---------------------------------------------------------
    def cancel(self) -> None:
        self._cancel.set()

    async def events(self) -> AsyncIterator[JobEvent]:
        while True:
            ev = await self._queue.get()
            if ev is None:
                return
            yield ev

    async def wait_event(self, timeout: float) -> JobEvent | None | object:
        """Next event, `None` when the job is over, `TIMEOUT` when nothing arrived in time.

        Cancelling `Queue.get()` is safe (unlike cancelling an async generator's `__anext__`),
        which is why the streaming route uses this instead of `events()`.
        """
        try:
            return await asyncio.wait_for(self._queue.get(), timeout)
        except TimeoutError:
            return TIMEOUT
