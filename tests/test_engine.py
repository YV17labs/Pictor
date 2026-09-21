import asyncio

from pictor.backends.base import GenerationRequest
from pictor.config import Settings
from pictor.engine.queue import Engine
from tests.conftest import FakeBackend


def _req(steps=5):
    return GenerationRequest(prompt="x", width=32, height=32, steps=steps, seed=1)


async def test_jobs_run_in_order_and_cancel():
    backend = FakeBackend(delay=0.02)
    engine = Engine(backend, Settings(_env_file=None))
    engine.start(preload=False)
    try:
        a = engine.submit(_req(10))
        b = engine.submit(_req(10))
        c = engine.submit(_req(10))
        assert engine.queue_position(c) >= 1
        c.cancel()

        async def collect(job):
            kinds = []
            async for ev in job.events():
                kinds.append(ev.kind)
            return kinds

        ka, kb, kc = await asyncio.gather(collect(a), collect(b), collect(c))
        assert ka[-1] == "result" and kb[-1] == "result"
        assert kc == ["cancelled"]
        assert ka.count("progress") >= 10
    finally:
        engine.stop()


async def test_running_job_cancels_at_next_step():
    backend = FakeBackend(delay=0.05)
    engine = Engine(backend, Settings(_env_file=None))
    engine.start(preload=False)
    try:
        job = engine.submit(_req(50))
        kinds = []
        async for ev in job.events():
            kinds.append(ev.kind)
            if ev.kind == "progress" and ev.step == 2:
                job.cancel()
        assert kinds[-1] == "cancelled" and len(kinds) < 20
    finally:
        engine.stop()
