#!/usr/bin/env python3
"""Interactive, explainable tutorial for **RL on the building — gyms over the sim**.

A small control plane that serves a clickable web guide (static/guide.html) and,
for each chapter, lets you read the CONCEPT, view the demo SOURCE, and RUN it.

Two modes, auto-detected (see config.py):
  • REAL — a live OpenAI-compatible endpoint (Ollama / vLLM / NIM on this laptop,
    or a DGX you point DGX_BASE_URL at) answers the LLM chapter for real. The gym
    demos additionally go REAL if `sinergym` / `CityLearn` are installed.
  • SIM  — nothing installed/reachable → a faithful built-in RC-zone gym runs
    instead, so every concept is learnable with no GPU. Real commands are shown.

Either way cloud cost is $0.00.

Launch (auto-picks a free port if 8208 is taken):

    .venv/bin/python week21/09_rl_building_controls/tutorial_server.py
    # → http://127.0.0.1:8208
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

PKG = Path(__file__).resolve().parent                 # …/week21/09_rl_building_controls
ROOT = PKG.parents[1]                                 # …/agenticaicodingfitness
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable
DEMOS = PKG / "demos"

GUIDE_PORT = int(os.environ.get("TWIN_GUIDE_PORT", "8208"))


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
     "title":"Ch 1 · RL on the building — gyms over the sim","level":"beginner",
     "desc":"Week 21 · App 09 of 12 · Phase 5 (AGENTS). An RL agent can't practice on the "
     "real building — a bad week of exploration is a bad week for the guests. So the field "
     "wraps building SIMULATIONS in gym environments: Sinergym (one building's HVAC over "
     "EnergyPlus), CityLearn (a district's storage and demand response), and BOPTEST (a "
     "fixed, referee-grade benchmark over REST). This app builds the same MDP on a tiny "
     "pure-stdlib RC hotel zone and learns a policy live.\n\n"
     "In this tutorial:\n"
     "  • Ch 2 · Building control as an MDP — Sinergym's obs/action/reward design, run live on our RC zone.\n"
     "  • Ch 3 · Learn a policy LIVE — rule-based baseline vs a tiny honest policy search; learning curve + scorecard.\n"
     "  • Ch 4 · CityLearn — district demand response: batteries flatten the peak; grid KPIs vs baseline.\n"
     "  • Ch 5 · BOPTEST — the REST wire-trace and the fixed KPI scorecard that makes controller claims comparable.\n\n"
     "Why it matters:\n"
     "  • The gym API (reset/step/reward) is THE interface between building physics (Apps 6–8) and learning agents (App 10).\n"
     "  • Obs/action/reward design decides what the agent CAN learn — energy vs comfort is a weighted trade, not magic.\n"
     "  • Honest calibration: RL typically saves 5–20% vs rule-based control at equal-or-better comfort, and tuned MPC still often wins on BOPTEST — knowing this keeps your claims credible.\n"
     "  • Fixed scenarios + fixed KPIs (BOPTEST) are what turn 'our controller is better' into evidence.\n\n"
     "Where it fits:\n"
     "  • Prerequisite: App 6 (EnergyPlus — the sim these gyms wrap). Feeds App 10 (the safe-autonomy ladder that deploys the policy) and the App 12 capstone. Phase 5 · AGENTS in the Week 21 twin stack.\n\n"
     "How to run:\n"
     "  • Click Run per chapter — everything works in SIM with pure stdlib, $0. Optional REAL upgrades: `pip install sinergym` / `pip install CityLearn`; an Ollama/DGX endpoint (🔌 Connection) makes Ch 5's verdict a live inference."},
    {"id":"step01","group":"The MDP","kind":"run","demo":"step01_sinergym_mdp.py",
     "title":"Ch 2 · Building control as an MDP — Sinergym","level":"beginner",
     "desc":"Sinergym wraps EnergyPlus in a gym: obs ≈17 floats, action = [heating sp, "
     "cooling sp], LinearReward trading energy vs comfort. We build the exact same MDP on "
     "our RC hotel zone and step it live, printing obs/action/reward."},
    {"id":"step02","group":"Learn","kind":"run","demo":"step02_learn_policy.py",
     "title":"Ch 3 · Learn a policy LIVE — baseline vs search","level":"intermediate",
     "desc":"A fixed setback schedule vs an honest ~30-line hill-climbing policy search, "
     "trained across simulated weeks. ASCII learning curve, then the scorecard: kWh, "
     "comfort K·h, reward — ~16% savings at equal comfort, inside the published 5–20% band."},
    {"id":"step03","group":"District","kind":"run","demo":"step03_citylearn_district.py",
     "title":"Ch 4 · CityLearn — district demand response","level":"advanced",
     "desc":"District scale: pre-simulated loads, agents dispatch batteries/storage/EVs — "
     "the MARL testbed for buildings. A 3-building mini-district flattens its peak 23% and "
     "prints CityLearn-style KPIs, including the honest ones that get slightly worse."},
    {"id":"step04","group":"Benchmark","kind":"run","demo":"step04_boptest_kpis.py",
     "title":"Ch 5 · BOPTEST — fair benchmarking over REST","level":"advanced",
     "desc":"Modelica emulators + baseline controllers in Docker, driven via REST: "
     "initialize → advance ×672 → /kpi. The fixed KPI scorecard for baseline vs the Ch 3 "
     "policy — and the honest verdict that tuned MPC still often beats RL."},
    {"id":"outro","group":"Benchmark","kind":"concept",
     "title":"Appendix · Where this sits in the twin stack","level":"all levels",
     "desc":"This app is the start of the AGENTS layer of Digital Twin = SCENE + STATE + "
     "SIMULATION + AGENTS: it turns the SIMULATION layer (Apps 6–8) into environments "
     "where control policies can learn safely, and hands the learned policy to App 10.\n\n"
     "Where this sits in Week 21: Apps 2–4 SCENE (OpenUSD ← BIM) · App 5 STATE (live "
     "telemetry) · Apps 6–8 SIMULATION (EnergyPlus · calibration · surrogates) · "
     "App 9 (THIS) gyms + RL · App 10 safe self-evolving ops · App 11 staff copilot · "
     "App 12 capstone. Remember the bridge: a policy that wins in the gym is NOT yet safe "
     "on the building — App 10's safe-autonomy ladder is how it earns real setpoints."},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="RL on the building — interactive tutorial")
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
    import sim as rcsim
    models = config.list_local_models() if real else rcsim.installed_models()
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
    banner = ["", "  ▣  RL on the building — gyms over the simulation"]
    if config.MODE == "real":
        banner += [f"      ✓ REAL endpoint: {config.MODEL} @ {config.BASE_URL}",
                   "        Ch 5's MPC-vs-RL verdict runs as a real inference — cloud cost $0.00."]
    else:
        banner += ["      ◈ SIM mode — no endpoint reachable; the RC-zone gym still runs everything.",
                   "        every concept is learnable with no GPU. Go REAL anytime:",
                   "        ollama serve   (or set DGX_BASE_URL / use 🔌 Connection)"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set TWIN_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
