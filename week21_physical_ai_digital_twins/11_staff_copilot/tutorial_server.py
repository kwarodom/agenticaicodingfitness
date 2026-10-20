#!/usr/bin/env python3
"""Interactive, explainable tutorial for **the staff copilot** — a twin that boosts the humans.

A small control plane that serves a clickable web guide (static/guide.html) and,
for each chapter, lets you read the CONCEPT, view the demo SOURCE, and RUN it.

Two modes, auto-detected (see config.py):
  • REAL — a live OpenAI-compatible endpoint (Ollama / vLLM / NIM on this laptop,
    or a DGX you point DGX_BASE_URL at). Ch 2's final synthesis runs for real.
  • SIM  — no endpoint reachable → the built-in twin simulator answers instead,
    so every concept is learnable with no GPU. Real commands are always shown.

Either way cloud cost is $0.00.

Launch (auto-picks a free port if 8210 is taken):

    .venv/bin/python week21/11_staff_copilot/tutorial_server.py
    # → http://127.0.0.1:8210
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

PKG = Path(__file__).resolve().parent                 # …/week21/11_staff_copilot
ROOT = PKG.parents[1]                                 # …/agenticaicodingfitness
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable
DEMOS = PKG / "demos"

GUIDE_PORT = int(os.environ.get("TWIN_GUIDE_PORT", "8210"))


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
     "title":"Ch 1 · The staff copilot — boost the humans","level":"beginner",
     "desc":"Week 21 · App 11 of 12 · Phase 5: AGENTS. The productivity module: the same twin "
     "the agent fleet controls (App 10) also makes every technician, engineer and duty manager "
     "faster — an LLM copilot grounded in the twin's four stores, work orders with spatial "
     "context, cuOpt technician dispatch, and Metropolis eyes on the building.\n\n"
     "In this tutorial:\n"
     "  • Ch 2 · Copilot over the twin — one duty-manager question walked through all four tool stores (semantics graph · historian · CMMS · O&M docs) to a grounded answer.\n"
     "  • Ch 3 · Work-order intelligence — spatial context turns a complaint into a work order with location, access path, likely parts, a lockout/tagout note and two batched PMs.\n"
     "  • Ch 4 · cuOpt dispatch — 8 work orders × 3 technicians with skills and time windows: naive assignment vs an optimized route, minutes saved printed.\n"
     "  • Ch 5 · Eyes on the building — Metropolis/VSS occupancy analytics: detect a queue and an after-hours room, answer an NL video question, push the correction into the energy model.\n\n"
     "Why it matters:\n"
     "  • Autonomous agents (App 10) are the long game; boosting the humans is the payback the building gets on day one — from the SAME four stores.\n"
     "  • 'All VAVs fed by AHU-3 on floor 5' is one graph query instead of a BMS spelunking session — the Week 15 GraphRAG pattern grounded in a building.\n"
     "  • Good work orders move the metrics vendors chase: first-time-fix rate, MTTR, truck rolls.\n"
     "  • Occupancy from cameras corrects the energy model's schedules — vision closes the loop back to SIMULATION (Apps 6–8).\n\n"
     "Where it fits:\n"
     "  • Phase 5 (AGENTS) of the twin stack. Prerequisite: App 5 (the graph + TSDB STATE layer this copilot queries). Sibling: App 10 (agents act on the twin; this app makes humans faster with it). Everything lands in the capstone, App 12.\n\n"
     "How to run:\n"
     "  • Click Run per chapter. Ch 2 synthesizes its final answer on a live endpoint via 🔌 Connection (REAL) or the built-in simulator (SIM, $0). Ch 3–5 run on the built-in twin stores everywhere — no GPU needed."},
    {"id":"step01","group":"Copilot","kind":"run","demo":"step01_copilot_tools.py",
     "title":"Ch 2 · Copilot over the twin","level":"beginner",
     "desc":"An LLM agent whose TOOLS are the twin's four stores — semantics graph "
     "(Brick/RealEstateCore/ASHRAE 223P), time-series historian, CMMS, and O&M-doc RAG. One "
     "duty-manager question ('why is the west ballroom warm and who should fix it?') walks "
     "through all four, tool call by tool call, to a grounded answer."},
    {"id":"step02","group":"Work orders","kind":"run","demo":"step02_work_orders.py",
     "title":"Ch 3 · Work-order intelligence","level":"intermediate",
     "desc":"Spatial context makes work orders good: asset → location → access path → what "
     "else is nearby. An NL complaint becomes a draft work order with likely parts, a "
     "lockout/tagout note, and two nearby PMs batched into the same visit."},
    {"id":"step03","group":"Dispatch","kind":"run","demo":"step03_cuopt_dispatch.py",
     "title":"Ch 4 · cuOpt dispatch","level":"advanced",
     "desc":"NVIDIA cuOpt is GPU-accelerated optimization — VRP/TSP with time windows and "
     "priorities plus LP/MILP, open-sourced Apache-2.0 at GTC March 2025, also served as a "
     "NIM. 8 work orders × 3 technicians: naive assignment vs an optimized route, with the "
     "minutes-saved table (pure-Python greedy + 2-opt, honestly labeled as a stand-in)."},
    {"id":"step04","group":"Vision","kind":"run","demo":"step04_metropolis_vss.py",
     "title":"Ch 5 · Eyes on the building — Metropolis & VSS","level":"advanced",
     "desc":"DeepStream + TAO + Metropolis microservices count people; the VSS AI Blueprint "
     "puts a VLM over the cameras so staff ask video questions in English. Camera placement "
     "is simulated in the twin first, and occupancy analytics correct the energy model's "
     "schedules — closing the loop to Apps 6–8."},
    {"id":"outro","group":"Vision","kind":"concept",
     "title":"Appendix · Where this sits in the twin stack","level":"all levels",
     "desc":"The staff copilot is the human half of the AGENTS layer — Digital Twin = SCENE "
     "+ STATE + SIMULATION + AGENTS.\n\n"
     "SCENE (Apps 2–4) gives it floors, rooms and access paths; STATE (App 5) gives it the "
     "semantics graph + historian it queries, joined to the CMMS; SIMULATION (Apps 6–8) "
     "receives the occupancy corrections Metropolis produces and answers the copilot's "
     "what-ifs; AGENTS (Apps 9–11): App 10's fleet controls the building, App 11 (THIS) "
     "boosts the humans running it. Ch 2 is the Week 15 GraphRAG pattern grounded in a "
     "building. Everything meets in the capstone, App 12."},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="The staff copilot — interactive tutorial")
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
    import sim as twinsim
    models = config.list_local_models() if real else twinsim.installed_models()
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
    banner = ["", "  ▣  The staff copilot — a twin that boosts the humans"]
    if config.MODE == "real":
        banner += [f"      ✓ REAL endpoint: {config.MODEL} @ {config.BASE_URL}",
                   "        Ch 2's synthesis runs for real, fully on-device — cloud cost $0.00."]
    else:
        banner += ["      ◈ SIM mode — no endpoint reachable; the built-in twin simulator answers.",
                   "        every concept is learnable with no GPU. Go REAL anytime:",
                   "        ollama run qwen3.6:35b-a3b-q8_0   (or set DGX_BASE_URL)"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set TWIN_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
