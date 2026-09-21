"""POST /v1/chat/completions — the conversational front door: an image model driven from any OpenAI chat client."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from pictor.api import sse
from pictor.api.schemas import ChatCompletionRequest
from pictor.api.state import AppState
from pictor.engine.jobs import TIMEOUT, Job, JobEvent, JobStatus
from pictor.orchestrator.conversation import parse_messages
from pictor.orchestrator.replies import t
from pictor.orchestrator.router import Plan
from pictor.util.images import encode_png, to_data_url

log = logging.getLogger(__name__)
router = APIRouter()

HEARTBEAT_S = 10.0


def _state(request: Request) -> AppState:
    return request.app.state.pictor


def _progress_message(stage: str, step: int, total: int) -> str:
    return t(f"progress_{stage}", step=step, total=total) if f"progress_{stage}" in _PROGRESS_KEYS else stage


_PROGRESS_KEYS = {"progress_planning", "progress_queued", "progress_loading", "progress_encoding",
                  "progress_generating", "progress_decoding", "progress_encoding_png"}


async def _plan(state: AppState, body: ChatCompletionRequest) -> Plan:
    turns = await parse_messages(body.messages, max_bytes=state.settings.max_upload_bytes, client=state.http)
    overrides = body.generation.as_overrides() if body.generation else {}
    return await state.router.plan(
        turns, state.backend.capabilities, overrides,
        backend_name=state.backend.name, device=state.backend.device,
    )


def _notes_text(notes: list[str], max_refs: int) -> str:
    return "\n".join(t(f"note_{n}", n=max_refs) for n in notes)


async def _run_job(state: AppState, plan: Plan) -> AsyncIterator[JobEvent | tuple[str, int, int]]:
    """Yield job events, interleaved with heartbeat progress while the worker is silent."""
    assert plan.request is not None
    job: Job = state.engine.submit(plan.request)
    pos = state.engine.queue_position(job)
    if pos > 0 or state.engine.busy:
        yield ("queued", pos, 0)
    last: tuple[str, int, int] = ("queued", 0, 0)
    try:
        while True:
            ev = await job.wait_event(HEARTBEAT_S)
            if ev is TIMEOUT:
                yield last  # keep the stream alive during long silent phases (model load)
                continue
            if ev is None:
                return
            assert isinstance(ev, JobEvent)
            if ev.kind == "progress":
                last = (ev.stage or "generating", ev.step, ev.total)
            yield ev
    finally:
        if job.status in (JobStatus.queued, JobStatus.running):
            job.cancel()


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    state = _state(request)
    model = body.model or state.settings.model_name
    if not body.messages:
        raise HTTPException(status_code=400, detail={"error": {"message": "messages is required", "type": "invalid_request_error"}})

    if body.stream:
        return StreamingResponse(_stream(state, body, model), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    plan = await _plan(state, body)
    cid = sse.new_id()
    if plan.action == "chat":
        return JSONResponse(sse.completion(cid, model, plan.reply, []))
    assert plan.request is not None
    content = [plan.reply]
    images: list[dict] = []
    async for ev in _run_job(state, plan):
        if isinstance(ev, tuple):
            continue
        if ev.kind == "result" and ev.result is not None:
            png = encode_png(ev.result.image)
            images.append({"type": "image_url", "image_url": {"url": to_data_url(png)}, "generation": _meta(ev.result)})
            content.append(_done_text(ev.result, plan, state))
        elif ev.kind == "error":
            return JSONResponse({"error": {"message": ev.error, "type": "server_error", "code": "generation_failed"}}, status_code=500)
        elif ev.kind == "cancelled":
            content.append(t("cancelled"))
    return JSONResponse(sse.completion(cid, model, "\n\n".join(c for c in content if c), images))


def _meta(result) -> dict:  # noqa: ANN001
    return {
        "seed": result.seed, "width": result.width, "height": result.height, "steps": result.steps,
        "prompt": result.prompt, "duration_ms": result.duration_ms, "backend": result.backend, "mode": result.mode,
    }


def _done_text(result, plan: Plan, state: AppState) -> str:  # noqa: ANN001
    notes = list(dict.fromkeys(plan.notes + result.notes))
    text = t("done", sec=f"{result.duration_ms / 1000:.0f}", seed=result.seed)
    extra = _notes_text(notes, state.settings.max_reference_images)
    return f"{text}\n\n{extra}" if extra else text


async def _stream(state: AppState, body: ChatCompletionRequest, model: str) -> AsyncIterator[str]:
    cid = sse.new_id()
    yield sse.chunk(cid, model, {"role": "assistant", "content": ""})
    yield sse.chunk(cid, model, sse.progress_delta("planning", 0, 0, _progress_message("planning", 0, 0)))
    try:
        plan = await _plan(state, body)
    except Exception as e:  # noqa: BLE001
        log.exception("planning failed")
        yield sse.error_event(f"{type(e).__name__}: {e}", code="planning_failed")
        yield sse.DONE
        return

    if plan.action == "chat":
        yield sse.chunk(cid, model, {"content": plan.reply})
        yield sse.chunk(cid, model, {}, finish="stop")
        yield sse.DONE
        return

    assert plan.request is not None
    yield sse.chunk(cid, model, {"content": plan.reply + "\n\n"})
    finished = False
    try:
        async for ev in _run_job(state, plan):
            if isinstance(ev, tuple):
                stage, step, total = ev
                yield sse.chunk(cid, model, sse.progress_delta(stage, step, total, _progress_message(stage, step, total)))
                continue
            if ev.kind == "progress":
                stage = ev.stage or "generating"
                yield sse.chunk(cid, model, sse.progress_delta(stage, ev.step, ev.total, _progress_message(stage, ev.step, ev.total)))
            elif ev.kind == "result" and ev.result is not None:
                yield sse.chunk(cid, model, sse.progress_delta("encoding_png", 1, 1, _progress_message("encoding_png", 1, 1)))
                png = await asyncio.to_thread(encode_png, ev.result.image)
                yield sse.chunk(cid, model, sse.image_delta(to_data_url(png), _meta(ev.result)))
                yield sse.chunk(cid, model, {"content": _done_text(ev.result, plan, state)})
                finished = True
            elif ev.kind == "cancelled":
                yield sse.chunk(cid, model, {"content": t("cancelled")})
                finished = True
            elif ev.kind == "error":
                yield sse.chunk(cid, model, {"content": t("error", error=ev.error)})
                yield sse.error_event(ev.error or "generation failed", code="generation_failed")
                finished = True
    except asyncio.CancelledError:
        raise
    if finished:
        yield sse.chunk(cid, model, {}, finish="stop", usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    yield sse.DONE
