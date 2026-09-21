"""Single-worker engine: one model in memory, one job at a time, FIFO queue, idle unload."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
import traceback

from pictor.backends.base import GenerationCancelled, GenerationRequest, ImageBackend
from pictor.config import Settings
from pictor.engine.jobs import Job, JobEvent, JobStatus

log = logging.getLogger(__name__)


class Engine:
    def __init__(self, backend: ImageBackend, settings: Settings) -> None:
        self.backend = backend
        self.settings = settings
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._pending: list[Job] = []
        self._current: Job | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="pictor-worker", daemon=True)
        self._idle_thread = threading.Thread(target=self._idle_watch, name="pictor-idle", daemon=True)
        self._started = False
        self._stopping = False
        self._unload_pending = False
        self.load_error: str | None = None
        self.loading = False
        self.last_activity = time.time()

    # -- lifecycle --------------------------------------------------------------------
    def start(self, preload: bool) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()
        if preload:
            self._queue.put(_LoadMarker())  # type: ignore[arg-type]
        if self.settings.idle_unload_s > 0:
            self._idle_thread.start()

    def stop(self) -> None:
        self._stopping = True
        self._queue.put(None)

    def _idle_watch(self) -> None:
        """Ask the worker to drop the weights after `idle_unload_s` seconds without a job."""
        while not self._stopping:
            time.sleep(5)
            idle = time.time() - self.last_activity
            if (
                self.backend.loaded
                and not self.busy
                and not self._unload_pending
                and self.queue_length == 0
                and idle >= self.settings.idle_unload_s
            ):
                self._unload_pending = True
                self._queue.put(_UnloadMarker())  # type: ignore[arg-type]

    # -- submission ---------------------------------------------------------------------
    def submit(self, request: GenerationRequest) -> Job:
        job = Job(request=request, loop=asyncio.get_running_loop())
        with self._lock:
            self._pending.append(job)
        self._queue.put(job)
        return job

    def queue_position(self, job: Job) -> int:
        with self._lock:
            try:
                return self._pending.index(job)
            except ValueError:
                return 0

    @property
    def busy(self) -> bool:
        return self._current is not None

    @property
    def queue_length(self) -> int:
        with self._lock:
            return len(self._pending)

    # -- worker ---------------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            if isinstance(item, _LoadMarker):
                self._preload()
                continue
            if isinstance(item, _UnloadMarker):
                self._unload_pending = False
                if self.queue_length == 0 and time.time() - self.last_activity >= self.settings.idle_unload_s:
                    log.info("idle for %ds, unloading the model", self.settings.idle_unload_s)
                    self.backend.unload()
                continue
            job: Job = item
            self.last_activity = time.time()
            with self._lock:
                if job in self._pending:
                    self._pending.remove(job)
            self._execute(job)

    def _preload(self) -> None:
        self.loading = True
        try:
            self.backend.load()
            self.load_error = None
        except Exception as e:  # pragma: no cover - depends on the machine
            self.load_error = f"{type(e).__name__}: {e}"
            log.error("model preload failed: %s\n%s", e, traceback.format_exc())
        finally:
            self.loading = False

    def _execute(self, job: Job) -> None:
        if job.cancel_requested():
            job.status = JobStatus.cancelled
            job.finish(JobEvent(kind="cancelled"))
            return
        self._current = job
        job.status = JobStatus.running
        try:
            if not self.backend.loaded:
                self.loading = True
                job.emit_progress("loading", 0, 1)
                try:
                    self.backend.load(job.emit_progress)
                finally:
                    self.loading = False
            result = self.backend.generate(job.request, job.emit_progress, job.cancel_requested)
            job.status = JobStatus.done
            job.finish(JobEvent(kind="result", result=result))
        except GenerationCancelled:
            job.status = JobStatus.cancelled
            job.finish(JobEvent(kind="cancelled"))
        except Exception as e:
            job.status = JobStatus.failed
            log.error("job %s failed: %s\n%s", job.id, e, traceback.format_exc())
            job.finish(JobEvent(kind="error", error=f"{type(e).__name__}: {e}"))
        finally:
            self._current = None
            self.last_activity = time.time()
            if not self.settings.keep_loaded:
                self.backend.unload()


class _LoadMarker:
    """Queue sentinel asking the worker to load the model ahead of the first job."""


class _UnloadMarker:
    """Queue sentinel asking the worker to drop the weights after an idle period."""
