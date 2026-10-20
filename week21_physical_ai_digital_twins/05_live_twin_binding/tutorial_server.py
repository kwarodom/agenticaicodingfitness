#!/usr/bin/env python3
"""Interactive, explainable tutorial for **the living twin — binding real-time data**.

A small control plane that serves a clickable web guide (static/guide.html) and,
for each chapter, lets you read the CONCEPT, view the demo SOURCE, and RUN it.

Two modes, auto-detected (see config.py):
  • REAL — a live OpenAI-compatible endpoint (Ollama / vLLM / llama.cpp on this
    laptop, or a DGX you point DGX_BASE_URL at). Only Ch 5 makes an LLM call.
  • SIM  — no endpoint reachable → a faithful simulator runs instead, so every
    concept is learnable with no GPU. Real commands are always shown.

Either way cloud cost is $0.00.

Launch (auto-picks a free port if 8204 is taken):

    .venv/bin/python week21/05_live_twin_binding/tutorial_server.py
    # → http://127.0.0.1:8204
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

PKG = Path(__file__).resolve().parent                 # …/week21/05_live_twin_binding
ROOT = PKG.parents[1]                                 # …/agenticaicodingfitness
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable
DEMOS = PKG / "demos"

GUIDE_PORT = int(os.environ.get("TWIN_GUIDE_PORT", "8204"))


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
     "title":"Ch 1 · The living twin — binding real-time data to the scene","level":"beginner",
     "desc":"Week 21 · Tutorial 05 of 12 · Phase 3: the STATE. Apps 02–04 built a SCENE — but a "
     "3D model is not a twin. It becomes one the moment live telemetry is BOUND to the geometry: "
     "a reading born on a BACnet field bus ends its journey as a USD attribute on the right prim, "
     "with a semantics graph and a time-series database standing beside the stage.\n\n"
     "In this tutorial:\n"
     "  • Ch 2 · The protocol chain — field bus to prim: BACnet/Modbus → gateway → MQTT → binding service → USD attribute, and the ID join (object ↔ point ↔ tag ↔ IFC GlobalId ↔ prim) that makes it possible.\n"
     "  • Ch 3 · Live layers & Fabric — NVIDIA's iot-samples pattern: live values land on a SESSION layer so authored asset files stay pristine; Fabric (USDRT) absorbs 10 Hz bursts and flushes at checkpoints.\n"
     "  • Ch 4 · The three-store architecture — USD = geometry, graph (Brick/RealEstateCore) = meaning, TSDB = history, joined on the GlobalId; the canonical query 'all VAVs fed by AHU-3 on floor 5, last 24 h' as one traversal.\n"
     "  • Ch 5 · From data to the viewport — heat-map displayColor opinions, an alert prim at a stuck damper, and one short LLM call turning the Ch 4 question into a query plan.\n\n"
     "Why it matters:\n"
     "  • Live data that edits your authored BIM layers destroys the asset investment — the session layer / Fabric split is THE production pattern (App 2's opinion-strength lesson, applied).\n"
     "  • Azure Digital Twins has meaning but no geometry; Omniverse has geometry but no standard semantics — a real building twin needs the three-store join.\n"
     "  • Every later chapter of Week 21 — energy calibration, RL rewards, the staff copilot — reads THIS state layer; get the binding wrong and everything above it lies.\n"
     "  • The GlobalId you preserved in App 03 is the join key that pays off here.\n\n"
     "Where it fits:\n"
     "  • App 05 of Week 21 — the whole of Phase 3 (STATE). Prerequisite: App 04 (an assembled scene). Feeds App 10 (self-evolving ops), App 11 (staff copilot) and the capstone App 12. Continues Week 14/15's Neo4j/GraphRAG work. Next: App 06 (energy simulation).\n\n"
     "How to run:\n"
     "  • Click Run per chapter. Ch 2–4 are pure stdlib — offline, $0, no endpoint needed (Ch 3 goes REAL if usd-core is installed). Ch 5 makes one short LLM call: REAL via 🔌 Connection (Ollama / DGX / cloud), else a faithful SIM ($0)."},
    {"id":"step01","group":"Bind","kind":"run","demo":"step01_protocol_chain.py",
     "title":"Ch 2 · The protocol chain — field bus to prim","level":"beginner",
     "desc":"Watch one reading morph at every hop: BACnet/IP object → gateway normalization → "
     "MQTT topic → binding service → USD attribute. The heart of it is the ID join — field "
     "object ↔ point ↔ equipment tag ↔ IFC GlobalId ↔ prim path — running on a mini in-process bus."},
    {"id":"step02","group":"Layers","kind":"run","demo":"step02_live_layers.py",
     "title":"Ch 3 · Live layers & Fabric","level":"intermediate",
     "desc":"NVIDIA's iot-samples pattern: sensor values become iot:* attributes on a SESSION/LIVE "
     "layer, so telemetry never dirties authored asset files. Streams 20 ticks — 1 Hz to the "
     "session layer, then a 10 Hz burst that Fabric (USDRT) absorbs and flushes at one checkpoint."},
    {"id":"step03","group":"Stores","kind":"run","demo":"step03_three_stores.py",
     "title":"Ch 4 · The three-store architecture","level":"advanced",
     "desc":"USD = geometry, Brick/RealEstateCore graph = meaning, TSDB = history — joined on the "
     "GlobalId. Runs the canonical query 'all VAVs fed by AHU-3 on floor 5, with last-24 h zone "
     "temps' as one graph traversal + TSDB fetch + USD highlight, vs the BMS-spelunking alternative."},
    {"id":"step04","group":"Viewport","kind":"run","demo":"step04_data_to_viewport.py",
     "title":"Ch 5 · From data to the viewport","level":"advanced",
     "desc":"The payoff: heat-map room colors from live temp deviation (session-layer displayColor "
     "opinions), an alert prim at the stuck-damper VAV that explains Ch 4's hot afternoons, and one "
     "short LLM call translating the operator's question into a graph + TSDB + USD query plan."},
    {"id":"outro","group":"Viewport","kind":"concept",
     "title":"Appendix · Where this sits in the twin stack","level":"all levels",
     "desc":"This app IS the STATE layer of the twin equation:\n\n"
     "Digital Twin = SCENE (OpenUSD stage from BIM — Apps 02–04) + STATE (live telemetry bound "
     "to prims + Brick graph + TSDB — THIS app) + SIMULATION (calibrated physics + surrogates — "
     "Apps 06–08) + AGENTS (RL control, self-evolving ops, staff copilot — Apps 09–11), all "
     "combined in the capstone (App 12).\n\n"
     "Everything above STATE consumes it: App 06 calibrates against the TSDB, App 10's operator "
     "agents act on live attrs, App 11's copilot runs exactly Ch 4's three-store query — and the "
     "graph store is where Week 14/15's Neo4j/GraphRAG work plugs in."},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="The living twin — interactive tutorial")
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
    banner = ["", "  ▣  The living twin — binding real-time data to the scene"]
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
