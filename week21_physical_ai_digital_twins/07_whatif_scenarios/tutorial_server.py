#!/usr/bin/env python3
"""Interactive, explainable tutorial for **What-if engineering — scenarios you can trust**.

A small control plane that serves a clickable web guide (static/guide.html) and,
for each chapter, lets you read the CONCEPT, view the demo SOURCE, and RUN it.

Two modes, auto-detected (see config.py):
  • REAL — a live OpenAI-compatible endpoint (Ollama / vLLM / llama.cpp on this
    laptop, or a DGX you point DGX_BASE_URL at). Only Ch 5's planner call uses it.
  • SIM  — no endpoint reachable → a faithful simulator runs instead, so every
    concept is learnable with no GPU. Real commands are always shown.

Either way cloud cost is $0.00.

Launch (auto-picks a free port if 8206 is taken):

    .venv/bin/python week21/07_whatif_scenarios/tutorial_server.py
    # → http://127.0.0.1:8206
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

PKG = Path(__file__).resolve().parent                 # …/week21/07_whatif_scenarios
ROOT = PKG.parents[1]                                 # …/agenticaicodingfitness
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable
DEMOS = PKG / "demos"

GUIDE_PORT = int(os.environ.get("TWIN_GUIDE_PORT", "8206"))


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
     "title":"Ch 1 · What-if engineering — scenarios you can trust","level":"beginner",
     "desc":"Week 21 · Tutorial 07 of 12 · Phase 4: the SIMULATION. App 6 gave the twin an "
     "energy model; this app turns that model into a what-if MACHINE you can trust — sweep "
     "the scenario grid, gate the model against real meters (ASHRAE Guideline 14), plug "
     "models from different tools into one loop (FMI), and put the whole thing behind a "
     "BMS-shaped API agents can drive (Alfalfa).\n\n"
     "In this tutorial:\n"
     "  • Ch 2 · Parametric sweeps — chiller COP × setpoint × window film = 18 ranked annual scenarios over an RC model of a Bangkok hotel, and which tool runs sweeps at which scale (eppy · OpenStudio Measures · jEPlus · besos).\n"
     "  • Ch 3 · The calibration gate — ASHRAE Guideline 14's NMBE and CV(RMSE) criteria against 12 months of metered data; an uncalibrated and a tuned model face the gate, one fails.\n"
     "  • Ch 4 · FMI/FMU co-simulation — the neutral standard (fmi-standard.org) that lets EnergyPlus, Modelica and your controller keep their own solvers and exchange values via do_step(); walked as a wire trace.\n"
     "  • Ch 5 · Alfalfa — NREL's virtual building behind a BMS-shaped REST API with Haystack-tagged points; run the pre-cool what-if via the API, read the KPI deltas, then ask the model for the next 3 scenarios.\n\n"
     "Why it matters:\n"
     "  • A twin that can't answer 'what if?' is a 3D dashboard — SIMULATION is the layer that earns the word 'twin'.\n"
     "  • An uncalibrated model's what-if answer is an opinion, not a prediction; Guideline 14 is the pass/fail line between the two.\n"
     "  • FMI is a durable format-level skill (like USD and IFC) — it outlives any vendor app, including Omniverse, which has no first-party FMI runtime.\n"
     "  • Alfalfa is where agents (Apps 09–10) earn the right to touch the real BMS: same API shape, zero risk.\n\n"
     "Where it fits:\n"
     "  • App 07 of Week 21 (Phase 4 · the SIMULATION, with Apps 06 and 08). Prerequisite: App 06 (the energy model being swept and calibrated). Feeds App 08 (surrogates trained on these parametric runs) and App 12 (the capstone's what-if console).\n\n"
     "How to run:\n"
     "  • Click Run per chapter. Ch 2–4 are pure stdlib — offline, $0, no endpoint needed (eppy/fmpy optional for REAL flavor). Ch 5 makes one short LLM call: REAL via 🔌 Connection (Ollama / DGX / cloud), else a faithful SIM ($0)."},
    {"id":"step01","group":"Sweep","kind":"run","demo":"step01_parametric_sweeps.py",
     "title":"Ch 2 · Parametric sweeps","level":"beginner",
     "desc":"Sweep chiller COP {4.5, 5.5, 6.5} × cooling setpoint {23, 24, 25 °C} × window "
     "film {no, yes} over the built-in RC model of a Bangkok hotel — 18 annual runs ranked "
     "by kWh, THB and comfort-violation hours. Then the tool ladder: eppy for tens of runs, "
     "OpenStudio Measures for shareable transformations, jEPlus for thousands, besos to search."},
    {"id":"step02","group":"Trust","kind":"run","demo":"step02_calibration_gate.py",
     "title":"Ch 3 · The calibration gate (ASHRAE G14)","level":"intermediate",
     "desc":"Compute NMBE and CV(RMSE) of an uncalibrated vs a tuned model against 12 months "
     "of metered kWh and run the Guideline 14 gate (monthly ±5 % / 15 %; hourly ±10 % / 30 %). "
     "The uncalibrated model fails on bias despite tracking the shape — the gate needs both."},
    {"id":"step03","group":"Co-sim","kind":"run","demo":"step03_fmi_cosim.py",
     "title":"Ch 4 · FMI/FMU co-simulation","level":"advanced",
     "desc":"FMI = the Modelica Association's standard for packaging models; an FMU is a zip "
     "of modelDescription.xml + binaries. Walk the Co-Simulation wire protocol — instantiate → "
     "setup_experiment → do_step — between a building FMU and a controller, hour by hour."},
    {"id":"step04","group":"Alfalfa","kind":"run","demo":"step04_alfalfa_whatif.py",
     "title":"Ch 5 · Alfalfa — what-if behind a BMS API","level":"advanced",
     "desc":"Drive an in-process Alfalfa-like API (list points → read → write setpoint → "
     "advance clock) through the what-if 'pre-cool the ballroom 2 h before the peak tariff "
     "window', read the KPI deltas, then one short LLM call proposes the next 3 scenarios."},
    {"id":"outro","group":"Alfalfa","kind":"concept",
     "title":"Appendix · Where this sits in the twin stack","level":"all levels",
     "desc":"This app is the trust layer of SIMULATION — the third layer of "
     "Digital Twin = SCENE + STATE + SIMULATION + AGENTS.\n\n"
     "SCENE (Apps 02–04) gave the building a body in OpenUSD; STATE (App 05) made it live; "
     "SIMULATION (App 06 built the physics, App 07 — THIS — made its what-ifs trustworthy and "
     "drivable, App 08 makes them answer in milliseconds with PhysicsNeMo surrogates); AGENTS "
     "(Apps 09–11) then learn on exactly this machinery — Sinergym and BOPTEST (App 09) are "
     "the RL-shaped siblings of Alfalfa, and the capstone (App 12) puts this what-if console "
     "in front of the hotel's operators.\n\n"
     "The durable skills from this app are format-level, not vendor-level: ASHRAE Guideline 14 "
     "for trust, FMI for interoperability, a BMS-shaped API for agent access."},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="What-if scenarios — interactive tutorial")
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
    banner = ["", "  ▣  What-if engineering — scenarios you can trust"]
    if config.MODE == "real":
        banner += [f"      ✓ REAL endpoint: {config.MODEL} @ {config.BASE_URL}",
                   "        Ch 5's planner call runs for real, on your endpoint — cloud cost $0.00."]
    else:
        banner += ["      ◈ SIM mode — no endpoint reachable; Ch 5 streams a canned answer.",
                   "        every concept is learnable with no GPU. Go REAL anytime:",
                   "        ollama serve   (or set DGX_BASE_URL / use 🔌 Connection)"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set TWIN_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
