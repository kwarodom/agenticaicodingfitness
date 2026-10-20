#!/usr/bin/env python3
"""Interactive, explainable tutorial — **Capstone: AltoTech Grand Bangkok, the living twin**.

A small control plane that serves a clickable web guide (static/guide.html) and,
for each chapter, lets you read the CONCEPT, view the demo SOURCE, and RUN it.

Two modes, auto-detected (see config.py):
  • REAL — a live OpenAI-compatible endpoint (Ollama / vLLM / NIM on this laptop,
    or a DGX you point DGX_BASE_URL at) powers the agent-reasoning moments.
  • SIM  — no endpoint reachable → canned agent answers stream instead, so every
    concept is learnable with no GPU. Real commands are always shown.

Either way cloud cost is $0.00.

Launch (auto-picks a free port if 8211 is taken):

    .venv/bin/python week21/12_capstone_twin_hotel/tutorial_server.py
    # → http://127.0.0.1:8211
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

PKG = Path(__file__).resolve().parent                 # …/week21/12_capstone_twin_hotel
ROOT = PKG.parents[1]                                 # …/agenticaicodingfitness
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable
DEMOS = PKG / "demos"

GUIDE_PORT = int(os.environ.get("TWIN_GUIDE_PORT", "8211"))


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
     "title":"Ch 1 · The living twin — everything, combined","level":"beginner",
     "desc":"Week 21 · Tutorial 12 of 12 · Phase 6: the CAPSTONE. The Week 23 capstone gave "
     "AltoTech Grand Bangkok an agent FLEET; this capstone gives the fleet a BODY — a digital "
     "twin built from every app this week: Digital Twin = SCENE (OpenUSD) + STATE (live data) + "
     "SIMULATION (physics) + AGENTS (control & copilot).\n\n"
     "In this tutorial:\n"
     "  • Ch 2 · Assemble the twin — the SCENE from BIM: discipline layers, a payload per floor, "
     "instanced furniture, GlobalIds + COBie attrs, then the twin-readiness validator (Apps 2–4).\n"
     "  • Ch 3 · Go live — bind BACnet-ish telemetry to session-layer opinions, join graph + TSDB + "
     "scene on GlobalId, light up the floor-5 heat-map and one anomaly alert prim (App 5).\n"
     "  • Ch 4 · The what-if console — pass the ASHRAE Guideline 14 gate, sweep, fit a guarded "
     "millisecond surrogate, price a pre-cool decision in baht (Apps 6–8).\n"
     "  • Ch 5 · The learning operator — one policy up the autonomy ladder with hard guardrails; "
     "an override becomes a fact, a frozen sensor trips the G36 fallback (Apps 9–10).\n"
     "  • Ch 6 · A day in the life — 06:00 brief, 09:30 VIP arrival, 14:00 chiller alarm end to "
     "end, plus the hospital / office / factory / smart-city variant briefs (App 11 + everything).\n\n"
     "Why it matters:\n"
     "  • A 3D model is not a twin. A BIM file is not a twin. This app shows the four layers "
     "working as ONE system on one building — the same pattern for a hospital, office, factory "
     "or district.\n"
     "  • Every number on screen traces to a mechanism you learned in Apps 1–11; nothing is a "
     "screenshot.\n\n"
     "Where it fits:\n"
     "  • Prerequisite: all of Apps 01–11 (each chapter names the apps it condenses). Continues "
     "Week 14/15 (graphs), Week 18 (self-evolving), Week 23 (the fleet).\n\n"
     "How to run:\n"
     "  • Click Run per chapter: REAL agent-reasoning moments against an endpoint via 🔌 "
     "Connection, or SIM with no GPU ($0). Every chapter is deterministic in SIM."},
    {"id":"step01","group":"Scene","kind":"run","demo":"step01_assemble_twin.py",
     "title":"Ch 2 · Assemble the twin","level":"beginner",
     "desc":"The SCENE from BIM to a validated stage: discipline layers composed in strength "
     "order, a payload per floor (30 floors, load one), 4,200 chairs as 6 instanced prototypes, "
     "every entity carrying its IFC GlobalId + COBie attrs — then the twin-readiness validator "
     "gates go-live. Its two WARNs are real handover problems: missing DATA, not geometry."},
    {"id":"step02","group":"State","kind":"run","demo":"step02_go_live.py",
     "title":"Ch 3 · Go live","level":"intermediate",
     "desc":"A mini bus samples every BACnet-ish point; the binding service writes each value as "
     "a SESSION-layer opinion on the prim with the same GlobalId; the canonical spatial-temporal "
     "query joins GRAPH + TSDB + SCENE in one pass. Payoff: a floor-5 heat-map from live prim "
     "attrs and an anomaly alert prim — room 1203, still misbehaving since Week 23."},
    {"id":"step03","group":"Simulation","kind":"run","demo":"step03_whatif_console.py",
     "title":"Ch 4 · The what-if console","level":"intermediate",
     "desc":"Fit the 5R1C zone model to the meter and pass the ASHRAE Guideline 14 gate — a "
     "twin's answers are only credible AFTER calibration — then sweep setpoints, fit a "
     "millisecond surrogate with an honest training-envelope guard, and make a real call: "
     "pre-cool ahead of tomorrow's forecast, priced in baht and comfort."},
    {"id":"step04","group":"Agents","kind":"run","demo":"step04_learning_operator.py",
     "title":"Ch 5 · The learning operator","level":"advanced",
     "desc":"A policy earns autonomy one rung at a time (offline → shadow → advisory → "
     "supervised), every proposal passing hard guardrails. A simulated week: an operator "
     "OVERRIDE becomes a consolidated fact, a frozen sensor trips the watchdog into the "
     "Guideline 36 fallback, and the LLM writes the postmortem — REAL if an endpoint is up."},
    {"id":"step05","group":"Agents","kind":"run","demo":"step05_a_day_in_the_life.py",
     "title":"Ch 6 · A day in the life","level":"advanced",
     "desc":"The finale — three moments of one operating day, each crossing every layer: the "
     "06:00 morning brief, the 09:30 VIP arrival (pre-condition inside guardrails), the 14:00 "
     "chiller alarm end to end (detect → diagnose → dispatch → verify → remember). Closes with "
     "the hospital / office / factory / smart-city variant briefs."},
    {"id":"outro","group":"Agents","kind":"concept",
     "title":"Appendix · The whole course in one building","level":"all levels",
     "desc":"Digital Twin = SCENE + STATE + SIMULATION + AGENTS — and which app taught each "
     "layer:\n\n"
     "  SCENE       Apps 02 OpenUSD · 03 BIM→USD · 04 assembly — layers, payloads, instancing, "
     "GlobalIds.\n"
     "  STATE       App 05 — session-layer telemetry, graph + TSDB + scene joined on GlobalId.\n"
     "  SIMULATION  Apps 06 EnergyPlus/5R1C · 07 G14-calibrated what-ifs · 08 surrogates + "
     "Earth-2.\n"
     "  AGENTS      Apps 09 RL · 10 self-evolving operator + autonomy ladder · 11 staff "
     "copilot + cuOpt + Metropolis.\n"
     "  MAP         App 01 — the three computers (DGX train · OVX simulate · AGX act).\n\n"
     "Continuity: the graphs are Week 14/15, the memory loop is Week 18, the fleet and the "
     "room-1203 lore are Week 23. Variants: hospital (redundancy/IAQ/compliance), office "
     "(tenant comfort/leases), factory (Mega blueprint/robot fleets), smart-city district "
     "(CityLearn-style demand response)."},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="Capstone: the living twin — interactive tutorial")
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
    banner = ["", "  ◳  Capstone: AltoTech Grand Bangkok — the living twin (Week 21, App 12)"]
    if config.MODE == "real":
        banner += [f"      ✓ REAL endpoint: {config.MODEL} @ {config.BASE_URL}",
                   "        agent-reasoning moments run for real — cloud cost $0.00."]
    else:
        banner += ["      ◈ SIM mode — no endpoint reachable; canned agent answers stream",
                   "        instead. Every chapter still runs, deterministic, $0. Go REAL",
                   "        anytime: ollama run gemma4:12b   (or set DGX_BASE_URL)"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set TWIN_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
