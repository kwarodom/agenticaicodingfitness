#!/usr/bin/env python3
"""Interactive, explainable tutorial for **OpenUSD foundations** (Week 21 · App 02).

A small control plane that serves a clickable web guide (static/guide.html) and,
for each chapter, lets you read the CONCEPT, view the demo SOURCE, and RUN it.

Two modes per demo, auto-detected at the top of each demo (NOT the LLM endpoint):
  • REAL — `pip install usd-core` is present → the demos author genuine OpenUSD
    stages (.usda files) in .sandbox/ with the pxr API. No GPU, no Omniverse.
  • SIM  — usd-core absent → sim.py's mini-USD teaching engine runs instead,
    mimicking the same API shape and printing the same results.

Either way cloud cost is $0.00 — these chapters never call an LLM.

Launch (auto-picks a free port if 8201 is taken):

    .venv/bin/python week21/02_openusd_foundations/tutorial_server.py
    # → http://127.0.0.1:8201
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

PKG = Path(__file__).resolve().parent                 # …/week21/02_openusd_foundations
ROOT = PKG.parents[1]                                 # …/agenticaicodingfitness
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable
DEMOS = PKG / "demos"

GUIDE_PORT = int(os.environ.get("TWIN_GUIDE_PORT", "8201"))


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
     "title":"Ch 1 · OpenUSD — the scene language of the twin","level":"beginner",
     "desc":"Week 21 · App 02 of 12 · Phase 2: the SCENE. OpenUSD (Universal Scene "
     "Description) is the open, layered scene format this whole week builds on — the "
     "geometry-and-semantics backbone of the digital twin. This tutorial teaches USD from "
     "zero with runnable code: REAL if `pip install usd-core` is present (no GPU, no "
     "Omniverse needed), else a faithful built-in mini-USD engine in sim.py.\n\n"
     "In this tutorial:\n"
     "  • Ch 2 · Stage · prim · attribute · relationship — the scene graph: Xform/Scope/Mesh prims at BIM-shaped paths, a custom ifc:GlobalId attribute, a sensor→equipment relationship, kind/purpose metadata, and the two classic gotchas (metersPerUnit & upAxis).\n"
     "  • Ch 3 · Layers & composition — one file per discipline sublayered into one building; layer order = opinion strength; a stronger layer overrides WITHOUT destroying the weaker opinion; the session layer for live values.\n"
     "  • Ch 4 · References, payloads, variants, instancing — compose assets in, lazy-load floors on demand (the #1 scale lever), switch lod variants, and instance the 4,000 hotel chairs.\n"
     "  • Ch 5 · Compose a mini hotel — site + building + payloaded floors + instanced furniture + a Sensors scope with GlobalIds; flattened export vs live composition.\n\n"
     "Why it matters:\n"
     "  • Digital Twin = SCENE + STATE + SIMULATION + AGENTS — USD is the SCENE layer everything else binds to.\n"
     "  • Non-destructive layered opinions are exactly how BIM disciplines federate — USD does natively what coordination meetings do manually.\n"
     "  • Payloads + instancing are why a 40-storey hotel opens in a viewer at all.\n"
     "  • The skills are format-level and vendor-neutral (openusd.org, AOUSD) — they outlive any single viewer or vendor app.\n\n"
     "Where it fits:\n"
     "  • Prerequisite: App 01 (the Physical AI map). Feeds App 03 (BIM→USD: LOD, GUID preservation), App 04 (scene assembly at scale), App 05 (live telemetry on USD attributes) and the App 12 capstone.\n\n"
     "How to run:\n"
     "  • Click Run per chapter. These chapters never call an LLM — ignore the connection pill; the mode that matters is printed at the top of each demo: REAL if usd-core is importable, else SIM. Go REAL anytime: pip install usd-core ($0, ~30 MB, no GPU)."},
    {"id":"step01","group":"Foundations","kind":"run","demo":"step01_stage_prims.py",
     "title":"Ch 2 · Stage · prim · attribute · relationship","level":"beginner",
     "desc":"A stage is the composed scene; prims are its tree (Xform/Scope/Mesh); attributes "
     "carry values (displayColor, a custom ifc:GlobalId — the BIM join key); relationships "
     "carry meaning (a sensor pointing at the FCU it instruments). Plus kind/purpose metadata "
     "and the two unit gotchas — metersPerUnit and upAxis — that mangle every first Revit import."},
    {"id":"step02","group":"Compose","kind":"run","demo":"step02_layers_composition.py",
     "title":"Ch 3 · Layers & composition","level":"intermediate",
     "desc":"One .usda per discipline — architecture, MEP, ops — sublayered into building.usda. "
     "Layer order = opinion strength: an ops override wins over architecture's wall color "
     "without touching its file; remove the override and the original shines through. The "
     "session layer holds live runtime values that never pollute the source files."},
    {"id":"step03","group":"Compose","kind":"run","demo":"step03_refs_payloads_variants.py",
     "title":"Ch 4 · References, payloads, variants, instancing","level":"intermediate",
     "desc":"Reference composes an asset in; payload makes it lazy-loadable — each floor its "
     "own file, so viewers open instantly and load floors on demand (the #1 scale lever); an "
     "lod variant set switches proxy/full; instanceable=true makes the 4,000th chair nearly "
     "free — with the naive-vs-instanced memory table to prove it."},
    {"id":"step04","group":"Assemble","kind":"run","demo":"step04_mini_hotel.py",
     "title":"Ch 5 · Compose a mini hotel","level":"advanced",
     "desc":"Assemble everything: site + building + two payloaded floors + instanced chairs + "
     "a Sensors scope carrying IFC GlobalIds — then print the composed tree (what usdview "
     "shows) and contrast a flattened handoff export with the living, layered composition."},
    {"id":"outro","group":"Assemble","kind":"concept",
     "title":"Appendix · Where this sits in the twin stack","level":"all levels",
     "desc":"OpenUSD is the SCENE layer of Digital Twin = SCENE + STATE + SIMULATION + "
     "AGENTS — the geometry and semantics every other layer binds to (telemetry binds to "
     "prims in App 05, physics results render onto them in Apps 06–08, agents act through "
     "them in Apps 09–11).\n\n"
     "Where this sits in Week 21: App 01 the Physical AI map · App 02 (THIS) the USD scene "
     "language · App 03 BIM→USD (LOD, GUID preservation) · App 04 scene assembly at scale · "
     "App 05 live state on USD attributes · Apps 06–08 simulation · Apps 09–11 agents · "
     "App 12 the capstone twin hotel."},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="OpenUSD foundations — interactive tutorial")
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
    banner = ["", "  ▣  OpenUSD foundations — the scene language of the digital twin"]
    try:
        import pxr  # noqa: F401
        banner += ["      ✓ REAL USD: usd-core (pxr) installed — demos author genuine",
                   "        .usda stages in .sandbox/. No GPU, no Omniverse, $0."]
    except ImportError:
        banner += ["      ◈ SIM — usd-core not installed; the mini-USD engine in sim.py",
                   "        teaches the same API. Go REAL anytime (no GPU, ~30 MB):",
                   "        pip install usd-core"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set TWIN_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
