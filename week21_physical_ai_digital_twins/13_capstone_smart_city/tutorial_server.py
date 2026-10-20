#!/usr/bin/env python3
"""Interactive, explainable tutorial — **Capstone II: the Sovereign Smart City**.

A small control plane that serves a clickable web guide (static/guide.html) and,
for each chapter, lets you read the CONCEPT, view the demo SOURCE, and RUN it.

Two modes, auto-detected (see config.py):
  • REAL — a live OpenAI-compatible endpoint (Ollama / vLLM / NIM on this laptop,
    or a DGX you point DGX_BASE_URL at) powers the agent-reasoning moments.
  • SIM  — no endpoint reachable → canned agent answers stream instead, so every
    concept is learnable with no GPU. Real commands are always shown.

Either way cloud cost is $0.00.

Launch (auto-picks a free port if 8211 is taken):

    .venv/bin/python week21/13_capstone_smart_city/tutorial_server.py
    # → http://127.0.0.1:8212
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

GUIDE_PORT = int(os.environ.get("TWIN_GUIDE_PORT", "8212"))


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
     "title":"Ch 1 \u00b7 A twin of twins \u2014 the sovereign smart city","level":"beginner",
     "desc":"Week 21 \u00b7 Capstone II (App 13) \u00b7 Phase 6. Capstone I gave one building a living "
     "twin; this capstone scales the same four layers to a CITY \u2014 Krung Alto: 4 districts, "
     "6 road corridors, 24 cameras, 4 substations, 5 grid-interactive buildings (one of them "
     "the App 12 hotel, referenced in as a nested twin). Grounded in NVIDIA's real Smart City "
     "Blueprint (VSS 3.0): RTVI CV detection, VLM alert verification, Kafka events, VSS "
     "agents \u2014 run sovereignly on GPUs the city owns.\n\n"
     "In this tutorial:\n"
     "  \u2022 Ch 2 \u00b7 Assemble the city twin \u2014 districts as payloads, street furniture instanced, "
     "building twins REFERENCED in (a twin of twins), readiness validator.\n"
     "  \u2022 Ch 3 \u00b7 Eyes on the city \u2014 the VSS Smart City Blueprint pipeline: RTDETR/GDINO "
     "stream budgets per GPU, Kafka wire trace, VLM verification killing false positives, "
     "NL questions to the VSS agent.\n"
     "  \u2022 Ch 4 \u00b7 City what-if \u2014 price a corridor reroute before retiming a signal; flatten "
     "the district evening peak CityLearn-style (hospital hard-excluded).\n"
     "  \u2022 Ch 5 \u00b7 Sovereign & self-evolving \u2014 the data-residency audit (video never leaves "
     "town), autonomy rungs earned PER ACTION TYPE, the flywheel turning incidents into "
     "playbooks.\n"
     "  \u2022 Ch 6 \u00b7 A night in the city \u2014 sideswipe at Asoke, substation trip, morning brief: "
     "every layer, end to end.\n\n"
     "Why it matters:\n"
     "  \u2022 Cities are where sovereignty stops being a preference: camera video legally and "
     "ethically cannot leave town \u2014 the whole stack must run on city-owned GPUs.\n"
     "  \u2022 The blast radius of a bad action is bigger than a building's: autonomy per action "
     "type (alerts rung 4, signal retiming rung 3, substation switching rung 1 forever).\n\n"
     "Where it fits:\n"
     "  \u2022 Prerequisite: Apps 01\u201312 (this reuses everything; App 12's hotel twin is "
     "literally referenced into the city stage). The smart-city variant brief from App 12, "
     "built for real.\n\n"
     "How to run:\n"
     "  \u2022 Click Run per chapter: REAL agent-reasoning moments against an endpoint via 🔌 "
     "Connection, or SIM with no GPU ($0). Every chapter is deterministic in SIM."},
    {"id":"step01","group":"Scene","kind":"run","demo":"step01_assemble_city_twin.py",
     "title":"Ch 2 \u00b7 Assemble the city twin","level":"beginner",
     "desc":"The SCENE rules at city scale: 4 districts as payloads (load Sukhumvit, not all "
     "four), 2,900 streetlights as 3 instanced prototypes, and Capstone I's hotel stage "
     "composed in as a REFERENCE \u2014 a twin of twins. The readiness validator gates go-live; "
     "its WARNs are, once again, missing data, not geometry."},
    {"id":"step02","group":"State","kind":"run","demo":"step02_vss_camera_network.py",
     "title":"Ch 3 \u00b7 Eyes on the city \u2014 VSS blueprint","level":"intermediate",
     "desc":"NVIDIA's Smart City Blueprint (VSS 3.0), simulated faithfully: RTVI CV "
     "(RTDETR 30 streams/GPU vs GDINO 6\u201312), Kafka event flow, the Alert Verification VLM "
     "rejecting 2 of 5 raw events before any human sees them, and a natural-language "
     "question answered by the VSS agent. Real endpoint map and docker compose shown."},
    {"id":"step03","group":"Simulation","kind":"run","demo":"step03_district_whatif.py",
     "title":"Ch 4 \u00b7 City what-if","level":"intermediate",
     "desc":"Price interventions in the twin before they touch the street: close Asoke "
     "Interchange and verify the diversion holds v/c under jam threshold; then flatten the "
     "35 MW district evening peak with CityLearn-style dispatch \u2014 the App 12 hotel sheds "
     "0.5 MW, the hospital is hard-excluded by guardrail."},
    {"id":"step04","group":"Agents","kind":"run","demo":"step04_sovereign_self_evolving.py",
     "title":"Ch 5 \u00b7 Sovereign & self-evolving","level":"advanced",
     "desc":"The two audits before agents touch a city: sovereignty (six data classes \u2014 "
     "camera video never leaves the city DGX; local vs local_shared vs remote-NIM modes "
     "honestly labeled) and autonomy (rungs earned per ACTION TYPE, substation switching at "
     "rung 1 forever). Then the flywheel distils tonight's incidents into playbooks."},
    {"id":"step05","group":"Agents","kind":"run","demo":"step05_a_night_in_the_city.py",
     "title":"Ch 6 \u00b7 A night in the city","level":"advanced",
     "desc":"The finale: 23:15 sideswipe (CV detect 2 s \u2192 VLM verify 9 s \u2192 advisory reroute "
     "\u2192 supervised dispatch + hospital notify \u2192 auto-restore), 01:00 substation partial trip "
     "(agent observes at rung 1; DR fleet incl. the hotel rides it through), 06:00 morning "
     "brief + consolidation. Twin = SCENE + STATE + SIMULATION + AGENTS, city-sized."},
    {"id":"outro","group":"Agents","kind":"concept",
     "title":"Appendix \u00b7 From a room to a city","level":"all levels",
     "desc":"The same four layers at every scale:\n\n"
     "  ROOM      a zone model + a sensor (App 6)\n"
     "  BUILDING  Capstone I \u2014 the living hotel (App 12)\n"
     "  DISTRICT  corridors + substations + a DR fleet (this app, Chs 4\u20135)\n"
     "  CITY      Capstone II \u2014 a twin of twins with eyes (VSS), a street-and-grid\n"
     "            simulation, and a fleet whose autonomy is earned per action.\n\n"
     "Blueprint reality check: the VSS Smart City Blueprint ships as docker compose with "
     "documented per-GPU stream budgets (H100/L40S/RTX PRO 6000) \u2014 the quickstart is a real "
     "afternoon of work; the sovereign twin AROUND it is this course. Variants: campus "
     "(office park = mini-district), airport (VSS + dispatch heavy), industrial estate "
     "(Mega blueprint adjacency)."},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="Capstone II: the sovereign smart city — interactive tutorial")
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
    banner = ["", "  ◳  Capstone II: Krung Alto — the sovereign smart city (Week 21, App 13)"]
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
