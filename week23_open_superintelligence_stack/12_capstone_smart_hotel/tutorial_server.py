#!/usr/bin/env python3
"""Interactive, explainable tutorial for **Sovereign AI on a DGX**.

A small control plane that serves a clickable web guide (static/guide.html) and,
for each chapter, lets you read the CONCEPT, view the demo SOURCE, and RUN it.

Two modes, auto-detected (see config.py):
  • REAL — a live OpenAI-compatible endpoint (Ollama / vLLM / llama.cpp on this
    laptop, or a DGX you point DGX_BASE_URL at). Genuine on-device inference.
  • SIM  — no endpoint reachable → a faithful DGX Spark simulator runs instead,
    so every concept is learnable with no GPU. Real commands are always shown.

Either way cloud cost is $0.00.

Launch (auto-picks a free port if 8092 is taken):

    .venv/bin/python week19/sovereign_dgx/tutorial_server.py
    # → http://127.0.0.1:8092
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

PKG = Path(__file__).resolve().parent                 # …/week19/sovereign_dgx
ROOT = PKG.parents[1]                                 # …/agenticaicodingfitness
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable
DEMOS = PKG / "demos"

GUIDE_PORT = int(os.environ.get("HOTEL_GUIDE_PORT", "8111"))


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
    {"id":"intro","group":"Capstone","kind":"concept",
     "title":"Ch 1 · The sovereign autonomous hotel","level":"beginner",
     "desc":"Week 23 · Tutorial 12 of 12 · Phase: Capstone — combine everything. AltoTech Grand "
     "Bangkok (from Weeks 11/14/15) is now run by a self-improving, sovereign agent fleet that "
     "wires the whole Week 23 stack into ONE running system on your DGX. The tools genuinely "
     "mutate a real hotel state, so it is production-shaped — Agent = Model + Harness, end to end.\n\n"
     "The stack, wired together:\n"
     "  • MODEL    — Nemotron 3 (Super orchestrates, Nano runs cheap sub-agents)   [App 01]\n"
     "  • SERVE    — NIM / Dynamo, one OpenAI-compatible endpoint on the DGX        [Apps 02, 03]\n"
     "  • SKILLS   — NeMo Retriever RAG over the hotel SOPs                         [App 04]\n"
     "  • AI-Q     — Intent Router → Deep Agent → specialist sub-agents             [App 05]\n"
     "  • NEMOCLAW — the specialized Energy / Maintenance / Guest agents            [App 06]\n"
     "  • OPENSHELL— a signed policy gates every action (setpoint bounds, VIP)      [App 07]\n"
     "  • RELAY    — observe every call, right-size the model, Phoenix traces       [App 08]\n"
     "  • ECONOMICS— cost / energy / goodput per task                              [App 09]\n"
     "  • FLYWHEEL — verifiable rewards + curate → distill a cheaper model          [Apps 10, 11]\n\n"
     "In this tutorial:\n"
     "  • Ch 2 · Morning ops brief — AI-Q Deep Agent: run the morning brief where the Router "
     "escalates to the Deep Agent and fans work to the specialists.\n"
     "  • Ch 3 · CRITICAL alarm, room 1203 — safe triage: run the room-1203 CRITICAL alarm "
     "end-to-end, SOP-checked and policy-gated.\n"
     "  • Ch 4 · VIP request — guardrails stop unsafe action: run a VIP request and watch the "
     "signed policy block the unsafe setpoint change.\n"
     "  • Ch 5 · Observe & optimize — Relay · Phoenix · economics: observe the span tree, latency, "
     "cost and right-sizing, then read the inference economics.\n"
     "  • Ch 6 · Self-improve — Data Flywheel + verifiable rewards: run the flywheel — score "
     "decisions with verifiable rewards and distill a cheaper Nano student.\n\n"
     "Where it fits:\n"
     "  Prerequisite: ideally all of 01–11 (this uses every one). It's the finale — after this you "
     "can deploy the pattern to a real building.\n\n"
     "How to run:\n"
     "  Click Run per chapter: REAL against a Nemotron NIM/DGX endpoint via 🔌 Connection, or SIM "
     "with no GPU ($0). The room-1203 alarm runs end-to-end either way."},
    {"id":"step01","group":"Operate","kind":"run","demo":"step01_morning_brief.py",
     "title":"Ch 2 · Morning ops brief — AI-Q Deep Agent","level":"beginner",
     "desc":"The Intent Router escalates to the Deep Agent (Nemotron Super), which plans a "
     "morning sweep and fans work out to the NemoClaw specialists. Every action passes the "
     "OpenShell policy and is observed by NeMo Relay. [Apps 5, 6, 7, 8]"},
    {"id":"step02","group":"Operate","kind":"run","demo":"step02_critical_alarm.py",
     "title":"Ch 3 · CRITICAL alarm, room 1203 — safe triage","level":"intermediate",
     "desc":"The room-1203 thread, end-to-end: the Maintenance specialist reads telemetry, checks "
     "the SOP via the NeMo Retriever RAG skill, and dispatches a CRITICAL work order — every call "
     "gated by the signed OpenShell policy, right-sized to Super by the router. [Apps 6, 4, 7, 8]"},
    {"id":"step03","group":"Operate","kind":"run","demo":"step03_guest_vip.py",
     "title":"Ch 4 · VIP request — guardrails stop unsafe action","level":"intermediate",
     "desc":"A VIP asks for a cooler room; the Guest specialist tries to change the setpoint, but "
     "the signed policy protects VIP-occupied rooms and routes it to a human concierge. 'What the "
     "agent may DO', enforced. [App 7]"},
    {"id":"step04","group":"Improve","kind":"run","demo":"step04_observe_optimize.py",
     "title":"Ch 5 · Observe & optimize — Relay · Phoenix · economics","level":"intermediate",
     "desc":"Read the run back like Phoenix Agent Insights: the span tree, latency and cost per "
     "turn, and how the Router right-sized each request (Nano for cheap work, Super for hard "
     "reasoning). Then the inference economics — tokens, $/M-token, energy. [Apps 8, 9]"},
    {"id":"step05","group":"Improve","kind":"run","demo":"step05_flywheel.py",
     "title":"Ch 6 · Self-improve — Data Flywheel + verifiable rewards","level":"advanced",
     "desc":"Score each decision with a VERIFIABLE reward (energy saved, SOP-correct triage, VIP "
     "protected — objective, not a vibe). Curate the clean, high-reward traces to GRPO-distill a "
     "cheaper Nano student that matches the Super teacher. [Apps 11, 10]"},
    {"id":"outro","group":"Improve","kind":"concept",
     "title":"Appendix · Deploy it for real","level":"all levels",
     "desc":"This is one integrated system, not six demos. To run it against your own building:\n\n"
     "  • point the 🔌 Connection at a Nemotron NIM / DGX (local / tunnel / cloud);\n"
     "  • replace hotel/world.py tools with real BMS/PMS calls (egress on the allowlist);\n"
     "  • sign hotel/policy.py with your org key; export Relay telemetry to your Phoenix;\n"
     "  • feed curated traces into a real NeMo Customizer/Gym run.\n\n"
     "The harness stays identical — swap the SIM brain for the endpoint and it is production-shaped: "
     "autonomous, self-improving, and sovereign — nothing leaves the box."},
    {"id":"refs","group":"Go further","kind":"concept",
     "title":"Appendix · References & real-world applications","level":"all levels",
     "desc":"Curated references:\n  • Everything this app combines — the Sources section of week23/README.md, plus\n    each app's own appendix (01–11).\n  • The continuation — week21/ gives this fleet a BODY: a digital twin of the\n    same hotel (Capstone I) and a city around it (Capstone II).\n  • The lineage — Weeks 11 (patterns), 14/15 (graph memory), 17 (A2A), 18\n    (self-evolving, sovereign edge), 19 (DGX).\n\nReal-world applications:\n  • Hotel & building AI operations — the fictional AltoTech Grand Bangkok mirrors\n    a real product category: autonomous energy/comfort optimization for hotels\n    and commercial buildings (AltoTech's actual domain), where agent fleets\n    watch telemetry, dispatch work and answer staff.\n  • Commercial building copilots — Willow, JCI OpenBlue and Siemens Building X\n    assistants are the market's versions of the morning-brief + triage patterns\n    shown here.\n  • Data-center ops — the same fleet shape (energy desk + maintenance desk +\n    guarded actions + flywheel) runs AI factories; NVIDIA's DSX blueprint makes\n    the twin the facility's operating system.\n  • The generalization test — swap the SOP corpus, the telemetry schema and the\n    policy file, and the identical harness runs a hospital, campus or plant —\n    that is the Agent = Model + Harness thesis, cashed out."},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="Sovereign Autonomous Hotel — Week 23 capstone")
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



# ── 🖥️ DGX console — run commands ON the DGX over SSH (Tailscale) ─────────────
import dgxsh  # noqa: E402

_dgx_lock = asyncio.Lock()


@app.get("/api/dgx/status")
async def dgx_status() -> dict:
    return dgxsh.status()


class DgxConfig(BaseModel):
    host: str | None = None
    user: str | None = None
    port: str | None = None
    key: str | None = None


@app.post("/api/dgx/config")
async def dgx_config(req: DgxConfig) -> dict:
    dgxsh.apply_config(req.model_dump())
    return dgxsh.status()


class DgxRun(BaseModel):
    command: str


@app.post("/api/dgx/run")
async def dgx_run(req: DgxRun):
    """Stream one command's output from the DGX, live (600 s cap)."""
    cmd = (req.command or "").strip()

    async def body():
        if not cmd:
            yield "type a command first\n__EXIT__ 1 0\n"
            return
        if _dgx_lock.locked():
            yield "⚠  another DGX command is running — wait for it to finish.\n__EXIT__ 1 0\n"
            return
        async with _dgx_lock:
            yield f"🖥️  {dgxsh.target()} $ {cmd}\n\n"
            start = time.time()
            proc = await asyncio.create_subprocess_exec(
                *dgxsh._ssh_argv(cmd),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            try:
                while True:
                    try:
                        line = await asyncio.wait_for(
                            proc.stdout.readline(),
                            timeout=max(1, start + 600 - time.time()))
                    except asyncio.TimeoutError:
                        proc.kill()
                        yield f"\n⏱  exceeded 600s — killed.\n__EXIT__ 124 {time.time()-start:.1f}\n"
                        return
                    if not line:
                        break
                    yield line.decode(errors="replace")
                await proc.wait()
                yield f"__EXIT__ {proc.returncode} {time.time()-start:.1f}\n"
            finally:
                if proc.returncode is None:
                    proc.kill()
    return StreamingResponse(body(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    port = _pick_free_port(GUIDE_PORT)
    banner = ["", "  ▣  Sovereign Autonomous Hotel — the Week 23 capstone (AltoTech Grand Bangkok)"]
    if config.MODE == "real":
        banner += [f"      ✓ REAL endpoint: {config.MODEL} @ {config.BASE_URL}",
                   "        the agent fleet reasons on-device — cloud cost $0.00."]
    else:
        banner += ["      ◈ SIM mode — no endpoint reachable; a deterministic brain drives the",
                   "        real tools/policy/relay/flywheel, so it all runs with no GPU. Go REAL",
                   "        anytime: point 🔌 Connection at a Nemotron NIM/DGX endpoint."]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set HOTEL_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
