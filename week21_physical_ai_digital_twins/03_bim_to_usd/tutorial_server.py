#!/usr/bin/env python3
"""Interactive, explainable tutorial for **BIM → USD: from model to twin-ready scene**.

A small control plane that serves a clickable web guide (static/guide.html) and,
for each chapter, lets you read the CONCEPT, view the demo SOURCE, and RUN it.

Two modes, auto-detected (see config.py):
  • REAL — a live OpenAI-compatible endpoint (Ollama / vLLM / llama.cpp on this
    laptop, or a DGX you point DGX_BASE_URL at). Only Ch 3 makes an LLM call;
    Ch 5 additionally goes REAL if `usd-core` is installed (no GPU needed).
  • SIM  — no endpoint reachable → a faithful simulator runs instead, so every
    concept is learnable with no GPU. Real commands are always shown.

Either way cloud cost is $0.00.

Launch (auto-picks a free port if 8202 is taken):

    .venv/bin/python week21/03_bim_to_usd/tutorial_server.py
    # → http://127.0.0.1:8202
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

PKG = Path(__file__).resolve().parent                 # …/week21/03_bim_to_usd
ROOT = PKG.parents[1]                                 # …/agenticaicodingfitness
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable
DEMOS = PKG / "demos"

GUIDE_PORT = int(os.environ.get("TWIN_GUIDE_PORT", "8202"))


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
     "title":"Ch 1 · BIM → USD: from model to twin-ready scene","level":"beginner",
     "desc":"Week 21 · Tutorial 03 of 12 · Phase 2: the SCENE. A BIM file is not a twin — it is "
     "the raw material for one. This app teaches the honest craft of getting a building model "
     "INTO OpenUSD without losing the two things a twin lives on: element identity (the IFC "
     "GlobalId) and handover data (COBie). Geometry is the easy part; every pipeline keeps it.\n\n"
     "In this tutorial:\n"
     "  • Ch 2 · BIM LOD, properly — LOD 100–500 as Level of DEVELOPMENT (reliability), not detail; the BIMForum spec; why LOD 500 is a field-verification attestation, not more polygons.\n"
     "  • Ch 3 · What survives BIM → USD — the IFC spatial tree → USD prim hierarchy map, the ALWAYS / USUALLY / NEVER survival table, and THE rule: preserve the GlobalId on every prim.\n"
     "  • Ch 4 · Pipelines in 2026 — the post-connector era (Launcher EOL Oct 1 2025): five real paths ranked, plus a simulated preservation scorecard of the same hotel wing through three of them.\n"
     "  • Ch 5 · A mini IFC → USD converter, live — walk a mock-IFC hotel wing, emit a USD stage with hierarchy + GlobalIds + namespaced Psets, then lint it with a pass/fail twin-readiness checklist.\n\n"
     "Why it matters:\n"
     "  • LOD = Level of DEVELOPMENT, not detail — a pretty manufacturer mesh in a concept model is still LOD 200; reliability, not polygons, is what a twin can bind data to.\n"
     "  • The twin-readiness bar: as-built LOD 300 geometry (350 for maintained MEP) + LOD-500-grade DATA (the COBie handover: manufacturer, model, serial, install date, warranty, zone). Twins fail on missing data far more often than missing geometry.\n"
     "  • The IFC GlobalId is the join key: App 5 binds live telemetry, CMMS records and the semantics graph to geometry through it. A converter that drops GUIDs produces a pretty mesh no data can ever bind to.\n\n"
     "Where it fits:\n"
     "  • App 03 of Week 21 (Phase 2 · the SCENE). Prerequisite: App 02 (OpenUSD foundations — stage, prim, attribute, layers). Feeds App 04 (assembling the converted stage into the full building scene) and App 05 (binding live telemetry to these exact GlobalIds).\n\n"
     "How to run:\n"
     "  • Click Run per chapter. Ch 2 and Ch 4 are pure stdlib — offline, $0, no endpoint needed. Ch 3 makes one short LLM call: REAL via 🔌 Connection (Ollama / DGX / cloud), else a faithful SIM. Ch 5 goes REAL with `pip install usd-core` (no GPU), else emits faithful USD-flavored text."},
    {"id":"step01","group":"LOD","kind":"run","demo":"step01_lod_levels.py",
     "title":"Ch 2 · BIM LOD, properly","level":"beginner",
     "desc":"LOD = Level of DEVELOPMENT — how much you can rely on an element — not detail. "
     "Walks the BIMForum LOD 100–500 table with twin relevance per level, the two levels people "
     "get wrong (350 = interfaces, 500 = an attestation with NO geometry), and drills the "
     "twin-readiness bar: LOD 300 geometry + LOD-500-grade COBie data."},
    {"id":"step02","group":"Convert","kind":"run","demo":"step02_what_survives.py",
     "title":"Ch 3 · What survives BIM → USD","level":"intermediate",
     "desc":"The IFC spatial tree maps 1:1 onto a USD prim hierarchy — but USD is scene "
     "description, not BIM authoring. The ALWAYS / USUALLY / NEVER survival table (meshes and "
     "transforms always; Psets and materials sometimes; parametric intelligence never), plus one "
     "short LLM call on why the GlobalId join key is load-bearing."},
    {"id":"step03","group":"Pipelines","kind":"run","demo":"step03_pipelines.py",
     "title":"Ch 4 · Pipelines in 2026 — the post-connector era","level":"intermediate",
     "desc":"NVIDIA deprecated the Omniverse Launcher (EOL Oct 1 2025) and its first-party "
     "connectors — so this ranks the five real paths instead, IFC4 + IfcOpenShell first. Then a "
     "simulated conversion of the same hotel wing through paths 1, 4 and 5, with a preservation "
     "scorecard: geometry survives everywhere; Psets and GUIDs separate a scene from a twin."},
    {"id":"step04","group":"Build","kind":"run","demo":"step04_mini_converter.py",
     "title":"Ch 5 · A mini IFC → USD converter, live","level":"advanced",
     "desc":"Build the whole idea in ~40 lines: walk a mock-IFC hotel wing (2 storeys, spaces, "
     "AHU + VAVs + walls), emit a USD stage preserving hierarchy, GlobalIds and namespaced "
     "Psets/COBie — then LINT it pass/fail: GUID coverage, COBie completeness per asset class, "
     "orphaned spaces. REAL with usd-core installed, else faithful USD-flavored text."},
    {"id":"outro","group":"Build","kind":"concept",
     "title":"Appendix · Where this sits in the twin stack","level":"all levels",
     "desc":"This app is the SCENE layer of the twin stack — the second of Week 21's four:\n\n"
     "Digital Twin = SCENE (OpenUSD stage from BIM — Apps 02–04, THIS app converts) + STATE "
     "(live telemetry bound to prims — App 05, joining on the GlobalIds this app preserved) + "
     "SIMULATION (calibrated physics + surrogates — Apps 06–08) + AGENTS (RL control, "
     "self-evolving ops, staff copilot — Apps 09–11), all combined in the capstone (App 12).\n\n"
     "The converted stage flows forward: App 04 assembles it into the full building scene "
     "(discipline layers, payloads per floor, instancing); App 05 binds BMS/IoT telemetry to "
     "the exact ifc:GlobalId attrs this converter wrote. Durable skills over vendor apps: "
     "IFC and OpenUSD (AOUSD) outlive any connector."},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="BIM → USD — interactive tutorial")
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
    banner = ["", "  ▣  BIM → USD — from model to twin-ready scene"]
    if config.MODE == "real":
        banner += [f"      ✓ REAL endpoint: {config.MODEL} @ {config.BASE_URL}",
                   "        Ch 3's LLM call runs for real, on your endpoint — cloud cost $0.00."]
    else:
        banner += ["      ◈ SIM mode — no endpoint reachable; Ch 3 streams a canned answer.",
                   "        every concept is learnable with no GPU. Go REAL anytime:",
                   "        ollama serve   (or set DGX_BASE_URL / use 🔌 Connection)",
                   "        pip install usd-core   → Ch 5 writes a genuine USD stage (no GPU)"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set TWIN_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
