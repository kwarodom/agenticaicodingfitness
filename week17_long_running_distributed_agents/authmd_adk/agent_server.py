#!/usr/bin/env python3
"""PART B — the long-running work-order service (FastAPI + uvicorn).

One persistent service holding NO work-order state in memory — every session
lives in the durable DatabaseSessionService (SQLite locally, Cloud SQL in prod).
Restart it, scale it to zero over a quiet weekend, resume days later from a
webhook: the agent wakes with a fresh, least-privilege token every time because
state stores the GRANT, not the token.

Endpoints
  POST /work_order                       → create a new work-order session
  POST /chat                             → send a message to the coordinator
  POST /webhooks/request_approval        → start the user-claimed approval (emails OTP)
  POST /webhooks/approved                → human confirmed the OTP → resume + apply
  GET  /status/{user_id}/{session_id}    → current step + grants (no LLM call)
  GET  /healthz

Run it (needs Part A's app_server.py running too):
    .venv/bin/python week17/authmd_adk/app_server.py    # terminal 1
    .venv/bin/python week17/authmd_adk/agent_server.py   # terminal 2
    .venv/bin/python week17/authmd_adk/run_full_demo.py  # terminal 3
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # package-local imports

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google.genai import types
from pydantic import BaseModel

from agent import build_runner, build_session_service
from config import APP_BASE, APP_NAME, DEFAULT_USER_ID, make_model, model_label
from resume_handler import WorkOrderResumeHandler
from work_order import WorkOrderStep, initial_state

STATIC_DIR = Path(__file__).resolve().parent / "static"

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = make_model()
    session_service = build_session_service()
    runner = build_runner(session_service=session_service, model=model)
    _state["session_service"] = session_service
    _state["runner"] = runner
    _state["resume"] = WorkOrderResumeHandler(runner, DEFAULT_USER_ID, session_service)
    print(f"✅ work-order service up — model={model_label(model)}  app={APP_NAME}")
    yield
    _state.clear()


app = FastAPI(title="Energy work order — long-running ADK agent (auth.md × ADK, Part B)",
              lifespan=lifespan)

# Dev-only: lets the UI run even if opened from a different origin.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


class WorkOrderRequest(BaseModel):
    session_id: str
    site_id: str = "site-bkk-01"
    user_id: str = DEFAULT_USER_ID


class ChatRequest(BaseModel):
    session_id: str
    message: str
    user_id: str = DEFAULT_USER_ID


class WebhookRequest(BaseModel):
    session_id: str
    user_id: str = DEFAULT_USER_ID


class ApprovedRequest(BaseModel):
    session_id: str
    otp: str
    user_id: str = DEFAULT_USER_ID


async def _run_turn(user_id: str, session_id: str, message: str) -> str:
    final_text = ""
    async for event in _state["runner"].run_async(
        user_id=user_id, session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part.from_text(text=message)]),
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    final_text = part.text
    return final_text


async def _snapshot(user_id: str, session_id: str) -> dict:
    session = await _state["session_service"].get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id)
    if session is None:
        raise HTTPException(404, f"no work-order session {session_id!r}")
    s = session.state
    step = s.get("current_step")
    grants = s.get("auth_grants", {})
    return {
        "session_id": session_id,
        "current_step": step,
        "paused_waiting_for": WorkOrderStep.PAUSED_STEPS.get(step),
        "pending_signals": s.get("pending_signals", []),
        "site_id": s.get("site_id"),
        "analysis": s.get("analysis", {}),
        "recommended_setpoint_c": s.get("recommended_setpoint_c"),
        "applied_setpoint_c": s.get("applied_setpoint_c"),
        # surface the grants (durable) but NEVER the live token
        "grants": {k: {"flow": v["flow"], "scopes": v["scopes"],
                       "claimed": bool(v.get("claim_ref"))}
                   for k, v in grants.items()},
        "complete": WorkOrderStep.is_terminal(step),
    }


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.post("/work_order")
async def work_order(req: WorkOrderRequest) -> dict:
    await _state["session_service"].create_session(
        app_name=APP_NAME, user_id=req.user_id, session_id=req.session_id,
        state=initial_state(req.site_id))
    return await _snapshot(req.user_id, req.session_id)


@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    reply = await _run_turn(req.user_id, req.session_id, req.message)
    return {"reply": reply, **await _snapshot(req.user_id, req.session_id)}


@app.post("/webhooks/request_approval")
async def request_approval(req: WebhookRequest) -> dict:
    """Park-time hook: start the user-claimed flow → app emails the human an OTP."""
    reply = await _state["resume"].request_approval(req.session_id)
    return {"reply": reply, **await _snapshot(req.user_id, req.session_id)}


@app.post("/webhooks/approved")
async def approved(req: ApprovedRequest) -> dict:
    """Human confirmed the OTP → complete the claim, re-mint, resume to APPROVED."""
    reply = await _state["resume"].approved(req.session_id, req.otp)
    return {"reply": reply, **await _snapshot(req.user_id, req.session_id)}


@app.get("/status/{user_id}/{session_id}")
async def status(user_id: str, session_id: str) -> dict:
    return await _snapshot(user_id, session_id)


# ── web UI + thin proxies to Part A (so the browser stays single-origin) ─────
@app.get("/")
async def ui() -> FileResponse:
    """The visualization single-page app."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/app/audit")
async def app_audit() -> dict:
    """Proxy Part A's auth audit log so the UI can show tokens being minted."""
    async with httpx.AsyncClient(base_url=APP_BASE, timeout=10) as c:
        return (await c.get("/_demo/audit")).json()


@app.get("/app/inbox/{email}")
async def app_inbox(email: str) -> dict:
    """Proxy Part A's demo 'inbox' so the UI can show the human's OTP."""
    async with httpx.AsyncClient(base_url=APP_BASE, timeout=10) as c:
        r = await c.get(f"/_demo/inbox/{email}")
        if r.status_code != 200:
            return {"email": email, "otp": None}
        return r.json()


if __name__ == "__main__":
    import uvicorn
    from config import AGENT_HOST, AGENT_PORT
    uvicorn.run(app, host=AGENT_HOST, port=AGENT_PORT)
