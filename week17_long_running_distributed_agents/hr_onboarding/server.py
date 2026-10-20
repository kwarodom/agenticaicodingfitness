#!/usr/bin/env python3
"""The long-running onboarding service (FastAPI + uvicorn).

This is the "long-running background process": one persistent service that holds
NO onboarding state in memory — every session lives in the durable
DatabaseSessionService (SQLite locally, Cloud SQL in prod). It can be restarted,
scaled to zero over a quiet weekend, and resumed days later from a webhook
without losing a single onboarding.

Endpoints
  POST /onboard                         → create a new onboarding session
  POST /chat                            → send a message to the coordinator
  POST /webhooks/document_signed        → resume: contract signed
  POST /webhooks/hardware_delivered     → resume: laptop delivered
  GET  /status/{user_id}/{session_id}   → current step + details (no LLM call)
  GET  /healthz

Run it (foreground):
    .venv/bin/python week17/hr_onboarding/server.py
Run it (background process):
    nohup .venv/bin/python week17/hr_onboarding/server.py > /tmp/onboarding.log 2>&1 &
Then drive it with:  .venv/bin/python week17/hr_onboarding/client.py
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # package-local imports

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google.genai import types
from pydantic import BaseModel

from agent import build_runner, build_session_service
from config import APP_NAME, DEFAULT_USER_ID, make_model, model_label
from onboarding_steps import OnboardingStep, initial_state
from resume_handler import OnboardingResumeHandler

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Shared, process-wide objects (built once at startup).
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = make_model()
    session_service = build_session_service()
    runner = build_runner(session_service=session_service, model=model)
    _state["session_service"] = session_service
    _state["runner"] = runner
    _state["resume"] = OnboardingResumeHandler(runner, DEFAULT_USER_ID)
    print(f"✅ onboarding service up — model={model_label(model)}  app={APP_NAME}")
    yield
    _state.clear()


app = FastAPI(title="HR Onboarding — long-running agent", lifespan=lifespan)

# Dev-only: lets the visualizer run even if opened from another origin.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/")
async def ui() -> FileResponse:
    """The live state-machine visualizer (single-page app)."""
    return FileResponse(STATIC_DIR / "index.html")


# ── request/response models ──────────────────────────────────────────────────
class OnboardRequest(BaseModel):
    session_id: str
    user_id: str = DEFAULT_USER_ID


class ChatRequest(BaseModel):
    session_id: str
    message: str
    user_id: str = DEFAULT_USER_ID


class WebhookRequest(BaseModel):
    session_id: str
    user_id: str = DEFAULT_USER_ID


# ── helpers ──────────────────────────────────────────────────────────────────
async def _run_turn(user_id: str, session_id: str, message: str) -> str:
    runner = _state["runner"]
    final_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
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
        raise HTTPException(404, f"no onboarding session {session_id!r}")
    s = session.state
    step = s.get("current_step")
    return {
        "session_id": session_id,
        "current_step": step,
        "paused_waiting_for": OnboardingStep.PAUSED_STEPS.get(step),
        "pending_signals": s.get("pending_signals", []),
        "new_hire_details": s.get("new_hire_details", {}),
        "complete": OnboardingStep.is_terminal(step),
    }


# ── endpoints ────────────────────────────────────────────────────────────────
@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.post("/onboard")
async def onboard(req: OnboardRequest) -> dict:
    """Create a durable onboarding session seeded at START."""
    await _state["session_service"].create_session(
        app_name=APP_NAME, user_id=req.user_id, session_id=req.session_id,
        state=initial_state())
    return await _snapshot(req.user_id, req.session_id)


@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    """Send a message to the coordinator and run one agent turn."""
    reply = await _run_turn(req.user_id, req.session_id, req.message)
    snap = await _snapshot(req.user_id, req.session_id)
    return {"reply": reply, **snap}


@app.post("/webhooks/document_signed")
async def document_signed(req: WebhookRequest) -> dict:
    """External system reports the contract was signed → resume the agent."""
    reply = await _state["resume"].document_signed(req.session_id)
    snap = await _snapshot(req.user_id, req.session_id)
    return {"reply": reply, **snap}


@app.post("/webhooks/hardware_delivered")
async def hardware_delivered(req: WebhookRequest) -> dict:
    """External system reports the laptop was delivered → resume the agent."""
    reply = await _state["resume"].hardware_delivered(req.session_id)
    snap = await _snapshot(req.user_id, req.session_id)
    return {"reply": reply, **snap}


@app.get("/status/{user_id}/{session_id}")
async def status(user_id: str, session_id: str) -> dict:
    """Read the durable checkpoint without invoking the model."""
    return await _snapshot(user_id, session_id)


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("ONBOARDING_PORT", "8077"))
    uvicorn.run(app, host="127.0.0.1", port=port)
