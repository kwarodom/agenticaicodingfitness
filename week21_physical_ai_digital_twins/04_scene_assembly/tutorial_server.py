#!/usr/bin/env python3
"""Interactive, explainable tutorial for **assembling the building twin scene**.

A small control plane that serves a clickable web guide (static/guide.html) and,
for each chapter, lets you read the CONCEPT, view the demo SOURCE, and RUN it.

Two modes, auto-detected (see config.py):
  • REAL — a live OpenAI-compatible endpoint (Ollama / vLLM / llama.cpp on this
    laptop, or a DGX you point DGX_BASE_URL at) powers the Ch 5 copilot; and
    `pip install usd-core` upgrades the USD exports to the real pxr API.
  • SIM  — nothing installed/reachable → a faithful built-in mini-stage runs
    instead, so every concept is learnable with no GPU. Real commands are shown.

Either way cloud cost is $0.00.

Launch (auto-picks a free port if 8203 is taken):

    .venv/bin/python week21/04_scene_assembly/tutorial_server.py
    # → http://127.0.0.1:8203
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

PKG = Path(__file__).resolve().parent                 # …/week21/04_scene_assembly
ROOT = PKG.parents[1]                                 # …/agenticaicodingfitness
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable
DEMOS = PKG / "demos"

GUIDE_PORT = int(os.environ.get("TWIN_GUIDE_PORT", "8203"))


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
     "title":"Ch 1 · Assembling the building twin scene","level":"beginner",
     "desc":"Week 21 · App 04 of 12 · Phase 2: the SCENE. Mirrors NVIDIA's learning path "
     "'Assembling Digital Twins With Omniverse and OpenUSD' — their factory becomes the "
     "AltoTech Grand Bangkok hotel. Assembly is what turns a pile of converted BIM assets "
     "(App 3) into ONE composed stage: validated assets, one central materials library, "
     "discipline layers, a payload per floor, instanced furniture — ready for live data (App 5).\n\n"
     "In this tutorial:\n"
     "  • Ch 2 · Project structure & asset hygiene — the twin folder convention, SimReady-style metadata, and the validator gate (units, up-axis, materials, GlobalId, poly budgets).\n"
     "  • Ch 3 · Materials & precision placement — ONE central materials library (change the marble once → 412 surfaces update), placeholder rebinds, centimetre AHU placement, and the custom attributes App 5 binds to.\n"
     "  • Ch 4 · Optimize for scale — instanceable furniture (4,080 chairs share 3 meshes), a payload per floor (open F12 only), and the stage-walk → CSV inventory export.\n"
     "  • Ch 5 · Assemble Grand Bangkok end-to-end — five discipline layers, two detailed floors, sensors as first-class prims, waypoints, and the ready-for-App-5 checklist.\n\n"
     "Why it matters:\n"
     "  • A stage that takes 90 s to open never gets opened — payloads and instancing are what make building-scale USD usable day-to-day.\n"
     "  • The GlobalIds and custom attributes authored here are the join keys the ENTIRE rest of the week binds to — miss them and there is no twin, only a model.\n"
     "  • Everything runs offline on a built-in mini-stage ($0); the exported .usda opens in USD Composer / usdview for real.\n\n"
     "Where it fits:\n"
     "  • Prerequisites: App 2 (OpenUSD grammar) and App 3 (BIM → USD). Feeds App 5 (live binding) and the App 12 capstone. Phase 2 · SCENE of Digital Twin = SCENE + STATE + SIMULATION + AGENTS.\n\n"
     "How to run:\n"
     "  • Click Run per chapter — all four run offline, stdlib-only, $0. `pip install usd-core` upgrades exports to the real pxr API; an LLM endpoint via 🔌 Connection makes Ch 5's closing copilot question REAL."},
    {"id":"step01","group":"Getting started","kind":"run","demo":"step01_asset_hygiene.py",
     "title":"Ch 2 · Project structure & asset hygiene","level":"beginner",
     "desc":"The twin project folder convention (assets/ materials/ layers/ config/), "
     "SimReady-style metadata (physics flags, semantic labels, pivot/units conventions), "
     "and the validator gate — units, up-axis, materials, GlobalId, polygon budgets — "
     "with a printed report table and the fix for every failure."},
    {"id":"step02","group":"Manage assets","kind":"run","demo":"step02_materials_placement.py",
     "title":"Ch 3 · Materials & precision placement","level":"intermediate",
     "desc":"One centralized materials library beats 400 copies — change once, updates "
     "everywhere (MDL vs UsdPreviewSurface in one line). Then place an AHU on its pad to "
     "the centimetre and author the custom attributes (zone, serviced-by, BACnet ref) "
     "that App 5 binds live data to."},
    {"id":"step03","group":"Optimize","kind":"run","demo":"step03_optimize_scale.py",
     "title":"Ch 4 · Optimize for scale","level":"intermediate",
     "desc":"Instancing best practices (4,080 chairs share 3 prototype meshes; point "
     "instancers for very high counts), a payload per floor so you compose only the "
     "working set, a naive-vs-optimized table, and the stage-walk → CSV inventory export."},
    {"id":"step04","group":"Assemble","kind":"run","demo":"step04_assemble_hotel.py",
     "title":"Ch 5 · Assemble Grand Bangkok end-to-end","level":"advanced",
     "desc":"Five discipline layers, 22 payloaded floors (2 loaded), instanced guest "
     "rooms, sensors as first-class prims with relationships to their equipment, "
     "navigation waypoints — composed tree, inventory summary, and the ready-for-App-5 "
     "checklist, closed by a twin-copilot question."},
    {"id":"outro","group":"Assemble","kind":"concept",
     "title":"Appendix · Where this sits in the twin stack","level":"all levels",
     "desc":"Assembly closes the SCENE layer of Digital Twin = SCENE + STATE + SIMULATION "
     "+ AGENTS: App 2 taught the USD grammar, App 3 converted BIM into assets, App 4 "
     "(THIS) composed them into one validated, optimized, identified stage.\n\n"
     "Where it goes next: App 5 binds live BACnet/MQTT telemetry onto the GlobalIds and "
     "alto:bacnetRef attributes you authored here (STATE) · Apps 6–8 give the stage "
     "physics so it can answer what-if (SIMULATION) · Apps 9–11 put agents on it "
     "(AGENTS) · App 12 assembles the whole living hotel.\n\n"
     "Source being transposed: NVIDIA's 'Assembling Digital Twins With Omniverse and "
     "OpenUSD' learning path — the same asset-library → validate → materials → assemble → "
     "optimize moves, factory swapped for hotel."},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="Scene assembly — interactive tutorial")
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
    import sim as dgxsim
    models = config.list_local_models() if real else dgxsim.installed_models()
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
    banner = ["", "  ◳  Scene assembly — assembling the building twin scene"]
    if config.MODE == "real":
        banner += [f"      ✓ REAL endpoint: {config.MODEL} @ {config.BASE_URL}",
                   "        Ch 5's copilot runs for real, on-device — cloud cost $0.00."]
    else:
        banner += ["      ◈ SIM mode — no endpoint reachable; the built-in mini-stage",
                   "        teaches every concept with no GPU, $0. Go REAL anytime:",
                   "        pip install usd-core   ·   ollama serve  (or set DGX_BASE_URL)"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set TWIN_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
