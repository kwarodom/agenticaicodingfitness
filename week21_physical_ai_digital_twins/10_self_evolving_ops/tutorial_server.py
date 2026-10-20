#!/usr/bin/env python3
"""Interactive, explainable tutorial for **the self-evolving operator**.

A small control plane that serves a clickable web guide (static/guide.html) and,
for each chapter, lets you read the CONCEPT, view the demo SOURCE, and RUN it.

Two modes, auto-detected (see config.py):
  • REAL — a live OpenAI-compatible endpoint (Ollama / vLLM / llama.cpp on this
    laptop, or a DGX you point DGX_BASE_URL at). Only Ch 5 makes an LLM call.
  • SIM  — no endpoint reachable → a faithful simulator runs instead, so every
    concept is learnable with no GPU. Real commands are always shown.

Either way cloud cost is $0.00.

Launch (auto-picks a free port if 8209 is taken):

    .venv/bin/python week21/10_self_evolving_ops/tutorial_server.py
    # → http://127.0.0.1:8209
"""
from __future__ import annotations

import asyncio
import os
import shutil
import socket
import sys
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

PKG = Path(__file__).resolve().parent                 # …/week21/10_self_evolving_ops
ROOT = PKG.parents[1]                                 # …/agenticaicodingfitness
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable
DEMOS = PKG / "demos"

GUIDE_PORT = int(os.environ.get("TWIN_GUIDE_PORT", "8209"))


def _port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _pick_free_port(preferred: int, span: int = 40) -> int:
    for p in range(preferred, preferred + span):
        if not _port_busy(p):
            return p
    return preferred


STEPS = [
    {"id":"intro","group":"Foundations","kind":"concept",
     "title":"Ch 1 · The self-evolving operator","level":"beginner",
     "desc":"Week 21 · Tutorial 10 of 12 · Phase 5: The AGENTS — learn to control. A fixed "
     "policy — even App 9's RL-trained one — meets a building that DRIFTS: seasons turn, "
     "coils foul, tenants change. The self-evolving operator keeps learning after deployment: "
     "Week 18's tripartite memory + Week 23's data flywheel, wrapped in a safe-autonomy "
     "ladder so learning can never hurt the building.\n\n"
     "In this tutorial:\n"
     "  • Ch 2 · Tripartite memory for ops — EPISODIC control episodes → CONSOLIDATION → SEMANTIC building facts ('ballroom thermal lag ≈ 45 min') + PROCEDURAL playbooks (SKILL.md for buildings).\n"
     "  • Ch 3 · The override flywheel — every operator override is a labeled preference (chosen ≻ rejected, DPO-shaped); curate → customize → evaluate → promote until corrections hit zero.\n"
     "  • Ch 4 · The safe-autonomy ladder — OFFLINE → SHADOW → ADVISORY → SUPERVISED AUTONOMY, with a numeric gate per rung and hard guardrails at the top.\n"
     "  • Ch 5 · A guardrailed week — 7 simulated days: a frozen sensor trips a watchdog into Guideline 36 fallback, an override flows into memory, one LLM call writes the postmortem.\n\n"
     "Why it matters:\n"
     "  • Buildings drift; policies don't. Without memory + a flywheel, App 9's policy is at its best on deployment day and worse every day after.\n"
     "  • Operator overrides are the most expensive labels a building produces — the flywheel turns them into training data instead of log noise.\n"
     "  • Constrain the ACTION SPACE, not just the reward: clamps and masking beat penalty terms for safety.\n"
     "  • Honest calibration: fully-autonomous RL in commercial BMS is still rare — advisory or tightly-bounded autonomy is the deployed norm (DeepMind/Google, BrainBox AI, PassiveLogic).\n\n"
     "Where it fits:\n"
     "  • App 10 of Week 21 (Phase 5 · AGENTS). Prerequisite: App 09 (RL building controls — the policy this app deploys safely). Feeds App 12 (the capstone's agent fleet). Bridges Week 18 (self-evolving agents) and Week 23 (data flywheel) into building operations.\n\n"
     "How to run:\n"
     "  • Click Run per chapter. Ch 2–4 are pure stdlib — offline, $0, no endpoint needed. Ch 5 makes one short LLM call: REAL via 🔌 Connection (Ollama / DGX / cloud), else a faithful SIM ($0)."},
    {"id":"step01","group":"Memory","kind":"run","demo":"step01_tripartite_memory.py",
     "title":"Ch 2 · Tripartite memory for ops","level":"beginner",
     "desc":"A fixed policy meets a building that drifts. Week 18's memory model applied to "
     "ops: replay 3 morning-startup episodes, run the consolidation loop, and watch it write "
     "2 semantic facts and 1 SKILL.md-style playbook — memory stores printed before/after."},
    {"id":"step02","group":"Flywheel","kind":"run","demo":"step02_override_flywheel.py",
     "title":"Ch 3 · The override flywheel","level":"intermediate",
     "desc":"Every operator override is a labeled preference (chosen ≻ rejected — DPO's exact "
     "shape). Week 23's flywheel on control logs: 5 repeated mornings, corrections fall 4 → 0, "
     "with a compound-returns table (corrections, K·h, kWh, operator-minutes)."},
    {"id":"step03","group":"Ladder","kind":"run","demo":"step03_autonomy_ladder.py",
     "title":"Ch 4 · The safe-autonomy ladder","level":"advanced",
     "desc":"OFFLINE → SHADOW → ADVISORY → SUPERVISED AUTONOMY: walk one policy up with a "
     "numeric gate per rung, the hard-guardrail set, the constrain-the-action-space principle, "
     "and an honest look at who runs learned control on real buildings today."},
    {"id":"step04","group":"Operate","kind":"run","demo":"step04_guardrailed_week.py",
     "title":"Ch 5 · A guardrailed week","level":"advanced",
     "desc":"7 simulated days at rung 4. Day 3: a sensor freezes in-range → stuck-sensor "
     "watchdog → automatic G36 fallback + alert. Day 5: an override flows into episodic "
     "memory. Ends with one LLM call writing the incident postmortem."},
    {"id":"outro","group":"Operate","kind":"concept",
     "title":"Appendix · Where this sits in the twin stack","level":"all levels",
     "desc":"This app is the AGENTS layer of Digital Twin = SCENE + STATE + SIMULATION + "
     "AGENTS — the part that learns and acts.\n\n"
     "Under it: SCENE (Apps 02–04, the OpenUSD stage), STATE (App 05, live telemetry the "
     "agent's episodes are made of), SIMULATION (Apps 06–08, where rung 1 of the ladder "
     "lives — Sinergym/BOPTEST run over these models). Beside it: App 09 trained the policy "
     "this app deploys safely; App 11 boosts the humans the ADVISORY rung reports to. "
     "App 12's capstone wires this operator into the full hotel twin.\n\n"
     "Earlier weeks: Week 18 built the tripartite memory + consolidation loop for coding "
     "agents; Week 23 (App 11 there) built the data flywheel — this app is both of them, "
     "pointed at a building, behind guardrails."},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="The self-evolving operator — interactive tutorial")
_run_lock = asyncio.Lock()

# the model the user picked in the UI; injected into demo runs via DGX_MODEL.
SELECTED = {"model": config.MODEL}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(PKG / "static" / "guide.html",
                        headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/api/steps")
async def steps() -> dict:
    def public(s):
        return {k: s.get(k) for k in ("id", "group", "title", "desc", "kind", "level")} | \
               {"demo": s.get("demo")}
    real = config.MODE == "real"
    import sim as opssim
    models = config.list_local_models() if real else opssim.installed_models()
    if SELECTED["model"] not in models:        # keep the selection valid
        SELECTED["model"] = (models[0] if models else config.MODEL)
    return {"steps": [public(s) for s in STEPS], "mode": config.MODE,
            "conn": config.CONN, "conn_human": config.conn_human(),
            "model": SELECTED["model"], "base_url": config.BASE_URL, "models": models}


class ModelRequest(BaseModel):
    model: str


@app.post("/api/select_model")
async def select_model(req: ModelRequest) -> dict:
    SELECTED["model"] = req.model
    return {"ok": True, "model": req.model}


class ConnRequest(BaseModel):
    conn: str = "local"
    url: str | None = None
    key: str | None = None
    auth: str | None = None


@app.post("/api/connect")
async def connect(req: ConnRequest) -> dict:
    """Re-point the connection at runtime (local / tunnel / cloud) and re-detect."""
    config.apply_connection(req.model_dump())
    models = config.list_local_models()
    SELECTED["model"] = config.MODEL if config.MODEL in models else (models[0] if models else config.MODEL)
    return {"ok": True, "conn": config.CONN, "mode": config.MODE,
            "base_url": config.safe_base_url(), "endpoint_up": config.endpoint_up(),
            "model": SELECTED["model"], "models": models}


@app.get("/api/source/{step_id}")
async def source(step_id: str) -> dict:
    step = STEP_BY_ID.get(step_id)
    if not step or not step.get("demo"):
        return {"source": "(no source for this step)"}
    path = DEMOS / step["demo"]
    if not path.exists():
        return {"source": f"(missing file: {step['demo']})"}
    return {"source": path.read_text(), "filename": step["demo"]}


def _stream_demo(demo: str, timeout: float):
    async def gen():
        start = time.time()
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "DGX_MODEL": SELECTED["model"]}
        proc = await asyncio.create_subprocess_exec(
            PY, str(DEMOS / demo), cwd=str(PKG), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            while True:
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(),
                        timeout=max(1, start + timeout - time.time()))
                except asyncio.TimeoutError:
                    proc.kill()
                    yield (f"\n⏱  step exceeded {timeout:.0f}s — killed.\n"
                           f"__EXIT__ 124 {time.time()-start:.1f}\n")
                    return
                if not line:
                    break
                yield line.decode(errors="replace")
            await proc.wait()
            yield f"__EXIT__ {proc.returncode} {time.time()-start:.1f}\n"
        finally:
            if proc.returncode is None:
                proc.kill()
    return gen()


class RunRequest(BaseModel):
    step_id: str


@app.post("/api/run")
async def run_step(req: RunRequest):
    step = STEP_BY_ID.get(req.step_id)
    if step is None or step.get("kind") != "run":
        async def err():
            yield f"step {req.step_id!r} is not runnable\n__EXIT__ 1 0\n"
        return StreamingResponse(err(), media_type="text/plain")

    # REAL inference + multi-step demos get more headroom.
    timeout = 360.0 if config.MODE == "real" else 120.0

    async def body():
        if _run_lock.locked():
            yield "⚠  another demo is already running — wait for it to finish.\n__EXIT__ 1 0\n"
            return
        async with _run_lock:
            yield f"$ {Path(PY).name} demos/{step['demo']}\n\n"
            async for chunk in _stream_demo(step["demo"], timeout):
                yield chunk
    return StreamingResponse(body(), media_type="text/plain")


@app.post("/api/cleanup")
async def cleanup() -> dict:
    removed = []
    sb = PKG / ".sandbox"
    if sb.exists():
        shutil.rmtree(sb); removed.append(".sandbox/")
    for pyc in PKG.rglob("__pycache__"):
        shutil.rmtree(pyc, ignore_errors=True)
    removed.append("__pycache__")
    return {"messages": [f"removed: {removed}"]}


if __name__ == "__main__":
    import uvicorn

    port = _pick_free_port(GUIDE_PORT)
    banner = ["", "  ▣  The self-evolving operator — agents that learn your building safely"]
    if config.MODE == "real":
        banner += [f"      ✓ REAL endpoint: {config.MODEL} @ {config.BASE_URL}",
                   "        Ch 5's LLM call runs for real, on your endpoint — cloud cost $0.00."]
    else:
        banner += ["      ◈ SIM mode — no endpoint reachable; Ch 5 streams a canned answer.",
                   "        every concept is learnable with no GPU. Go REAL anytime:",
                   "        ollama serve   (or set DGX_BASE_URL / use 🔌 Connection)"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set TWIN_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
