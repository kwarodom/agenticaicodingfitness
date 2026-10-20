#!/usr/bin/env python3
"""Interactive, explainable tutorial for **Physics-ML surrogates — millisecond what-if**.

A small control plane that serves a clickable web guide (static/guide.html) and,
for each chapter, lets you read the CONCEPT, view the demo SOURCE, and RUN it.

Two modes, auto-detected (see config.py):
  • REAL — a live OpenAI-compatible endpoint (Ollama / vLLM / llama.cpp on this
    laptop, or a DGX you point DGX_BASE_URL at). Only Ch 5 makes an LLM call.
  • SIM  — no endpoint reachable → a faithful simulator runs instead, so every
    concept is learnable with no GPU. Real commands are always shown.

Either way cloud cost is $0.00.

Launch (auto-picks a free port if 8207 is taken):

    .venv/bin/python week21/08_physics_surrogates/tutorial_server.py
    # → http://127.0.0.1:8207
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

PKG = Path(__file__).resolve().parent                 # …/week21/08_physics_surrogates
ROOT = PKG.parents[1]                                 # …/agenticaicodingfitness
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable
DEMOS = PKG / "demos"

GUIDE_PORT = int(os.environ.get("TWIN_GUIDE_PORT", "8207"))


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
     "title":"Ch 1 · Physics-ML surrogates — millisecond what-if","level":"beginner",
     "desc":"Week 21 · Tutorial 08 of 12 · Phase 4: the SIMULATION layer. A twin viewport "
     "needs answers in MILLISECONDS while a facilities manager drags a slider — but a full "
     "CFD run takes minutes to hours, and even an annual EnergyPlus run takes minutes. The "
     "pattern this app teaches: SOLVER for truth, SURROGATE for interaction — train a fast "
     "ML model on parametric solver runs, serve it in the twin viewport, and fall back to "
     "physics outside its training envelope.\n\n"
     "In this tutorial:\n"
     "  • Ch 2 · Why surrogates — the latency ladder (CFD hours → EnergyPlus minutes → RC seconds → surrogate ms) and the canonical proof: data-center twins (Cadence Reality DC on Omniverse APIs; the AI-factory Blueprint with Schneider Electric, ETAP, Vertiv), commonly quoted at 100–1000× interactive speedups.\n"
     "  • Ch 3 · PhysicsNeMo — NVIDIA's open-source physics-ML framework (renamed from Modulus, GTC 2025): FNO/AFNO, MeshGraphNet, DoMINO, diffusion, physics-informed losses — and the honest scope: it is NOT a building-energy simulator; you generate the data and own the ML lifecycle.\n"
     "  • Ch 4 · Train a surrogate LIVE — 200 parametric runs of an RC hotel model, gradient descent in pure Python (watch the loss fall), a µs-vs-ms benchmark — then BREAK it outside the envelope and install the fallback-to-physics guard.\n"
     "  • Ch 5 · Earth-2 forecasts — AI weather (FourCastNet, CorrDiff, earth2studio, Earth-2 NIMs) as boundary conditions: a do-nothing vs pre-cool decision table for tomorrow's heat spike, plus one short LLM summary for the facilities manager.\n\n"
     "Why it matters:\n"
     "  • Milliseconds is a hard requirement of interaction — no solver meets it, so every serious twin pairs a solver (truth) with a surrogate (speed).\n"
     "  • The data-center twin market proved the pattern commercially; a hotel's ballroom airflow and chiller-plant room are the same physics.\n"
     "  • A surrogate is a cache of physics — and caches go stale: the envelope guard is the difference between a fast tool and a confident liar.\n"
     "  • Forecasts turn what-if into what-NEXT: pre-cool before the spike, evaluated in ms against every forecast update.\n\n"
     "Where it fits:\n"
     "  • App 08 of Week 21 (Phase 4 · SIMULATION). Prereq: App 07 — a surrogate of an UNCALIBRATED model is just a fast wrong answer. Feeds App 09 (RL needs thousands of fast environment steps) and App 12 (the capstone's what-if console).\n\n"
     "How to run:\n"
     "  • Click Run per chapter. Ch 2–4 are pure stdlib — offline, $0, no endpoint needed. Ch 5 makes one short LLM call: REAL via 🔌 Connection (Ollama / DGX / cloud), else a faithful SIM ($0)."},
    {"id":"step01","group":"Why","kind":"run","demo":"step01_why_surrogates.py",
     "title":"Ch 2 · Why surrogates — the latency ladder","level":"beginner",
     "desc":"CFD takes minutes–hours, an annual EnergyPlus run minutes — but the twin viewport "
     "needs milliseconds while a manager drags a slider. The latency ladder, the pattern "
     "(SOLVER for truth, SURROGATE for interaction), and the shipping proof: data-center "
     "twins — Cadence Reality DC and the AI-factory Blueprint (Schneider Electric, ETAP, Vertiv)."},
    {"id":"step02","group":"PhysicsNeMo","kind":"run","demo":"step02_physicsnemo.py",
     "title":"Ch 3 · PhysicsNeMo — the surrogate factory","level":"intermediate",
     "desc":"NVIDIA's open-source physics-ML framework (ex-Modulus, Apache-2.0): FNO/AFNO, "
     "MeshGraphNet, DoMINO, physics-informed losses. The workflow with realistic stage "
     "timings — and the honest framing: not a building simulator; you generate the training "
     "data and own the ML lifecycle. It wins for FIELD surrogates, not schedules."},
    {"id":"step03","group":"Train","kind":"run","demo":"step03_train_surrogate.py",
     "title":"Ch 4 · Train a surrogate live — then break it","level":"advanced",
     "desc":"200 parametric runs of an RC hotel model, gradient descent in pure Python (loss "
     "printed live), ~1% holdout MAE at ~150× the speed — then the DANGER demo: query at "
     "40 °C / 120% occupancy and watch it be wrong, confidently. The fix: know the envelope, "
     "fall back to physics outside it."},
    {"id":"step04","group":"Earth-2","kind":"run","demo":"step04_earth2_forecast.py",
     "title":"Ch 5 · Earth-2 forecasts — pre-cool before the spike","level":"advanced",
     "desc":"Earth-2 (FourCastNet, CorrDiff, earth2studio) supplies tomorrow's boundary "
     "conditions; the guarded surrogate sweeps do-nothing vs pre-cool in milliseconds and "
     "prints the decision table (peak kW · kWh · THB). One short LLM call summarizes the "
     "recommendation for the facilities manager."},
    {"id":"outro","group":"Earth-2","kind":"concept",
     "title":"Appendix · Where this sits in the twin stack","level":"all levels",
     "desc":"This app completes the SIMULATION layer of Digital Twin = SCENE + STATE + "
     "SIMULATION + AGENTS.\n\n"
     "Within Phase 4: App 06 built the truth (EnergyPlus + reduced-order models), App 07 "
     "made it credible (the Guideline 14 calibration gate, what-if engineering), and App 08 "
     "(THIS) made it FAST — a trained surrogate serving millisecond what-if in the twin "
     "viewport, with Earth-2 forecasts in and a fallback-to-physics guard.\n\n"
     "Downstream: App 09 trains RL controllers against fast simulation (thousands of env "
     "steps per second is a surrogate's home turf), App 10 wires the self-evolving operator "
     "on top, and App 12's capstone what-if console is exactly this app's loop — solver "
     "offline, surrogate online, physics on guard.\n\n"
     "NVIDIA rails used here: PhysicsNeMo (train the surrogate on the DGX) · Earth-2 "
     "(forecast boundary conditions) · Omniverse (the viewport the milliseconds are for)."},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="Physics-ML surrogates — interactive tutorial")
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
    banner = ["", "  ▣  Physics-ML surrogates — millisecond what-if for the twin"]
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
