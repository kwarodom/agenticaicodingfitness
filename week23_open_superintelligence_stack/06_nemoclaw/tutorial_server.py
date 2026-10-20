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

GUIDE_PORT = int(os.environ.get("CLAW_GUIDE_PORT", "8105"))


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
     "title":"Ch 1 · What is NemoClaw? From model to specialist","level":"beginner",
     "desc":"Week 23 · Tutorial 06 of 12 · Phase: Build real agents.\n"
     "NVIDIA NemoClaw is how you build specialized agents: assemble a base model + persona + "
     "skills + tools + a signed policy, run each one safely inside an OpenShell sandbox, and "
     "orchestrate a whole fleet of them. It turns a general open model into a domain expert you own.\n\n"
     "In this tutorial:\n"
     "  • Ch 2 · Define a specialized agent — author the SPEC (role, base Nemotron model, system "
     "prompt, allowed skills & tools, policy) instead of hand-writing code.\n"
     "  • Ch 3 · Equip it — attach skills & tools — pull capabilities from the Agent Skills catalog "
     "and wire concrete tools so the persona can actually DO things.\n"
     "  • Ch 4 · Run it safely in OpenShell — execute a task in a sandbox where a signed policy and "
     "an egress allowlist gate every tool call.\n"
     "  • Ch 5 · A fleet of specialists — a supervisor routes each task to the right expert, turning "
     "single agents into a multi-agent system.\n\n"
     "Why it matters for sovereign agents:\n"
     "  • Turns a general model into a domain expert you own and can trust with a real job.\n"
     "  • Composes from the reusable Agent Skills catalog instead of re-prompting from scratch.\n"
     "  • Runs sandboxed under a signed policy — capability without containment is a liability.\n"
     "  • Scales cleanly to a supervised multi-agent fleet, all fully on-device.\n\n"
     "Where it fits:\n"
     "Prerequisites: 04_agent_skills and 05_aiq_research_lab (skills + orchestration are what you "
     "assemble here). Feeds 07_guardrails_openshell (guard the specialists you build) and the "
     "capstone. Next: 07_guardrails_openshell.\n\n"
     "How to run:\n"
     "Click Run per chapter: REAL against a Nemotron NIM/DGX endpoint via 🔌 Connection, or SIM "
     "with no GPU ($0)."},
    {"id":"step01","group":"Author","kind":"run","demo":"step01_define_agent.py",
     "title":"Ch 2 · Define a specialized agent","level":"beginner",
     "desc":"A NemoClaw agent is a SPEC, not code you hand-write. This chapter builds and prints "
     "one: role/persona, the base model (Nemotron Nano/Super), a system prompt, the allowed "
     "skills & tools, and the policy. Author the spec once → you have a reusable domain expert."},
    {"id":"step02","group":"Equip","kind":"run","demo":"step02_equip_skills.py",
     "title":"Ch 3 · Equip it — attach skills & tools","level":"intermediate",
     "desc":"A bare persona can't DO anything. Here the agent pulls skills from the catalog "
     "(NeMo Retriever RAG, cuOpt optimization, etc. — App 4) and wires concrete tools. Each "
     "attachment adds a capability the specialist can invoke on demand — it gains hands."},
    {"id":"step03","group":"Run","kind":"run","demo":"step03_run_openshell.py",
     "title":"Ch 4 · Run it safely in OpenShell","level":"intermediate",
     "desc":"An equipped specialist is powerful — so it runs inside an OpenShell sandbox (App 7) "
     "with a SIGNED policy and an egress allowlist. This chapter runs a task: the agent requests "
     "a tool, the policy gateway checks it, the sandbox executes it, and a result returns — "
     "every action gated. Capability without containment is a liability."},
    {"id":"step04","group":"Fleet","kind":"run","demo":"step04_fleet.py",
     "title":"Ch 5 · A fleet of specialists","level":"advanced",
     "desc":"One specialist is useful; a FLEET is a system. A supervisor routes a task to the right "
     "NemoClaw expert — HVAC, finance, code — each its own persona + skills + policy. This is the "
     "multi-agent pattern from Week 9/16, now built from authored specialists instead of prompts."},
    {"id":"outro","group":"Fleet","kind":"concept",
     "title":"Appendix · Ship specialized agents","level":"all levels",
     "desc":"NemoClaw is the 'build specialized agents' layer of Agent = Model + Harness: it authors "
     "the harness (persona + skills + tools + policy) around an open Nemotron model and runs it "
     "safely in OpenShell. Get it at github.com/NVIDIA/NemoClaw and browse blueprints at "
     "build.nvidia.com/blueprints.\n\n"
     "Where this sits in Week 23: App 1 Nemotron (model) · App 2 NIM (serve) · App 7 OpenShell "
     "(the sandbox it runs in) · App 4 Agent Skills (the capabilities it pulls) · App 6 (THIS) "
     "NemoClaw (build the specialist). A fleet of these = the Week 9/16 multi-agent system."},
    {"id":"refs","group":"Go further","kind":"concept",
     "title":"Appendix · References & real-world applications","level":"all levels",
     "desc":"Curated references:\n  • NemoClaw — nvidia.com/en-us/ai/nemoclaw/ and github.com/NVIDIA/NemoClaw\n  • OpenShell (the safety wrapper it ships with) — developer.nvidia.com/blog/run-autonomous-self-evolving-agents-more-safely-with-nvidia-openshell/\n  • The specialist-fleet pattern — Week 9/10 multi-agent material and Week 23\n    App 12's fleet.\n\nReal-world applications:\n  • Ops-center desks as agents — an energy desk, a maintenance desk, a guest desk:\n    narrow specialists with their own skills, tools and signed policy beat one\n    do-everything agent on both accuracy and auditability.\n  • Composition before fine-tuning — production teams exhaust persona + skills +\n    tools (cheap, reversible, evaluable) before touching weights; App 11's\n    flywheel is the escalation path when evals prove a real gap.\n  • Shippable agent bundles — MSPs and ISVs deliver 'a specialist in a box'\n    (model ref + persona + skills + signed policy) to customer sites, the way\n    they used to ship appliances.\n  • The capstone cast — Energy · Maintenance · Guest in App 12 are NemoClaw\n    specialists; Week 21's building/city operators are their physical-AI kin."},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="NVIDIA NemoClaw — interactive tutorial")
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
    banner = ["", "  ▣  NVIDIA NemoClaw — build specialized agents on your DGX"]
    if config.MODE == "real":
        banner += [f"      ✓ REAL base model: {config.MODEL} @ {config.BASE_URL}",
                   "        specialists run for real, fully on-device — cloud cost $0.00."]
    else:
        banner += ["      ◈ SIM mode — no endpoint reachable, simulating the base model.",
                   "        author + run specialized agents with no GPU. Go REAL anytime:",
                   "        ollama run qwen3.6:35b-a3b-q8_0   (or set DGX_BASE_URL)"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set CLAW_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
