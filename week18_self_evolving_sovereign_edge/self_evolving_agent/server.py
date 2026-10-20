#!/usr/bin/env python3
"""The self-evolving agent — live visualizer service (FastAPI + uvicorn).

A long-running process that holds ONE SelfEvolvingAgent whose memory lives on
disk (memory/). Through the browser UI you:

  1. Run a task            → watch which memory the agent injects, and its cost
  2. Consolidate           → watch the subconscious loop distil that run into
                             semantic facts (MEMORY.md/USER.md) + a SKILL.md
  3. Run the SAME task     → watch it load the new skill and finish in fewer
                             turns at lower cost  ← compound returns, live
  4. Reset                 → wipe memory back to an amnesiac slate and repeat

If the `claude` CLI is installed the runs are REAL LLM agent turns; otherwise the
agent uses a deterministic offline simulation (the UI shows which mode is live).

Endpoints
  GET  /                       the visualizer (static/index.html)
  GET  /healthz
  GET  /api/state              full snapshot: episodic runs, semantic, procedural
  GET  /api/preview?prompt=…   what memory WOULD be injected for a prompt
  POST /api/run                {prompt, label} → run one task
  POST /api/consolidate        {session_id}    → run the subconscious loop
  POST /api/reset              wipe memory (amnesiac slate)

Run:
    .venv/bin/python week18/self_evolving_agent/server.py        # → http://127.0.0.1:8088
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # import the package

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from self_evolving_agent import config
from self_evolving_agent.core.agent import SelfEvolvingAgent

# NOTE: we deliberately do NOT inject ANTHROPIC_API_KEY. Live runs go through the
# Claude Agent SDK, which drives the local `claude` CLI using your Claude Code
# subscription auth — no API key needed (and an unrelated key would be rejected).
STATIC_DIR = Path(__file__).resolve().parent / "static"
_lock = threading.Lock()                 # serialise agent access (one SQLite conn)
_agent: SelfEvolvingAgent | None = None

# Mode: 'auto' (live if the claude CLI is available), 'live', or 'sim'.
# Live = real LLM agent turns + real meta-cognitive consolidation (authentic, but
# real-model turn counts vary). Sim = deterministic offline turns that cleanly
# demonstrate the guaranteed compound-returns curve. Toggle it live in the UI.
_mode = os.environ.get("SELF_EVOLVING_MODE", "auto").lower()


def agent() -> SelfEvolvingAgent:
    global _agent
    if _agent is None:
        _agent = SelfEvolvingAgent()
    _apply_mode()
    return _agent


def _apply_mode() -> None:
    if _agent is None:
        return
    if _mode == "sim":
        _agent.live = False
    elif _mode == "live":
        _agent.live = config.sdk_available()
    else:                                # auto
        _agent.live = config.sdk_available()


app = FastAPI(title="Self-Evolving Agent — live visualizer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


# ── request models ────────────────────────────────────────────────────────────
class RunReq(BaseModel):
    prompt: str
    label: str = ""


class ConsolidateReq(BaseModel):
    session_id: str


class ModeReq(BaseModel):
    mode: str                            # 'live' | 'sim' | 'auto'


# ── state snapshot ────────────────────────────────────────────────────────────
def _snapshot() -> dict:
    a = agent()
    runs = a.db.list_sessions()
    return {
        "mode": "live" if a.live else "simulated",
        "mode_setting": _mode,
        "live_available": config.sdk_available(),
        "model": a.model,
        "runs": [{"label": r["label"], "turns": r["turns"] or 0,
                  "cost": round(r["cost"] or 0, 4),
                  "skills_loaded": (r["skills_loaded"] or "").split(", ") if r["skills_loaded"] else [],
                  "session_id": r["session_id"]} for r in runs],
        "semantic": {
            "memory_md": a.semantic.memory_md.read_text(errors="replace"),
            "user_md": a.semantic.user_md.read_text(errors="replace"),
            "has_facts": a.semantic.has_learned_facts(),
        },
        "skills": a.skills.list_skills(),
    }


# Sync endpoints → FastAPI runs them in a worker thread, so the agent's internal
# asyncio.run() (live SDK turns) works without clashing with the server loop.
@app.get("/api/state")
def state() -> dict:
    with _lock:
        return _snapshot()


@app.get("/api/preview")
def preview(prompt: str) -> dict:
    with _lock:
        a = agent()
        rep = a.memory_report(prompt)
        rep["system_prompt"] = a.build_system_prompt(prompt)
        return rep


@app.post("/api/run")
def run(req: RunReq) -> dict:
    with _lock:
        a = agent()
        result = a.run_task(req.prompt, label=req.label or "run")
        result["state"] = _snapshot()
        return result


@app.post("/api/consolidate")
def consolidate(req: ConsolidateReq) -> dict:
    with _lock:
        a = agent()
        learned = a.consolidate(req.session_id)
        return {"learned": learned, "state": _snapshot()}


@app.post("/api/mode")
def set_mode(req: ModeReq) -> dict:
    global _mode
    with _lock:
        if req.mode in ("live", "sim", "auto"):
            _mode = req.mode
        _apply_mode()
        return _snapshot()


@app.post("/api/reset")
def reset() -> dict:
    global _agent
    with _lock:
        if _agent is not None:
            _agent.close()
            _agent = None
        removed = config.wipe_memory()
        return {"removed": removed, "state": _snapshot()}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("SELF_EVOLVING_PORT", "8088"))
    a = agent()
    print(f"\n  🧠  Self-evolving agent — live visualizer")
    print(f"      mode: {'LIVE (real LLM via claude CLI)' if a.live else 'SIMULATED (offline)'}"
          f"   model: {a.model}")
    print(f"      open  →  http://127.0.0.1:{port}\n", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
