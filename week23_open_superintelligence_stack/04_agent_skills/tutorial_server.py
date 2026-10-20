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

GUIDE_PORT = int(os.environ.get("SKILLS_GUIDE_PORT", "8103"))


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
     "title":"Ch 1 · What is an Agent Skill?","level":"beginner",
     "desc":"Week 23 · Tutorial 04 of 12 · Phase: Build real agents. NVIDIA Agent Skills are "
     "portable, framework-agnostic capabilities that connect frontier agents (Claude, GPT, "
     "Gemini, Nemotron) to your business. Like SKILL.md and MCP, they are standardized so one "
     "skill loads into any agent without rebuilding it.\n\n"
     "In this tutorial:\n"
     "  • Ch 2 · The skills catalog — browse NVIDIA's skills (AI-Q, NeMo Retriever, cuOpt,\n"
     "    VSS…) and see which business system each connects a frontier agent to.\n"
     "  • Ch 3 · Load a skill into a frontier agent — the agent discovers a SKILL.md, reads\n"
     "    its metadata, then invokes the skill's tools on demand.\n"
     "  • Ch 4 · Connect to your business — a NeMo Retriever RAG skill answers from YOUR docs,\n"
     "    grounding the agent in sovereign data that never leaves the perimeter.\n"
     "  • Ch 5 · Skills + MCP + A2A — the same skill rides MCP (agent→tools) and A2A\n"
     "    (agent→agent), so it works across agents and frameworks unchanged.\n\n"
     "Why it matters for sovereign agents:\n"
     "  • Progressive disclosure — metadata first, tools loaded on demand: small context.\n"
     "  • Write once, works everywhere — one skill runs across Claude, GPT, and Nemotron.\n"
     "  • Connects a general agent to YOUR sovereign data & tools — your expertise, on-device.\n"
     "  • Standard, open catalog: github.com/NVIDIA/skills.\n\n"
     "Where it fits:\n"
     "  Prerequisite: 01_nemotron_models. This is the first harness piece — skills are the\n"
     "  building block the next apps equip. Feeds 05_aiq_research_lab (AI-Q wires skills as\n"
     "  tools) and 06_nemoclaw (specialists built from skills). Next: 05_aiq_research_lab.\n\n"
     "How to run:\n"
     "  Click Run per chapter: REAL against a Nemotron/frontier endpoint via 🔌 Connection, "
     "or SIM with no GPU ($0)."},
    {"id":"step01","group":"Catalog","kind":"run","demo":"step01_skills_catalog.py",
     "title":"Ch 2 · The skills catalog","level":"beginner",
     "desc":"Browse the NVIDIA skills catalog — AI-Q (deep research), NeMo Retriever (doc "
     "intelligence), NeMo Evaluator, NeMo Curator, cuOpt, cuDF, VSS (video), Voice Chat, "
     "TensorRT-LLM and more — and see what business system each one connects a frontier "
     "agent to. Get them at github.com/NVIDIA/skills."},
    {"id":"step02","group":"Load","kind":"run","demo":"step02_load_skill.py",
     "title":"Ch 3 · Load a skill into a frontier agent","level":"intermediate",
     "desc":"A frontier agent (Claude / GPT / Nemotron) DISCOVERS a skill, reads its SKILL.md "
     "metadata (name, description, when-to-use), then INVOKES its tools and gets a result. "
     "Progressive disclosure keeps the agent's context small until the skill is actually needed."},
    {"id":"step03","group":"Connect","kind":"run","demo":"step03_connect_business.py",
     "title":"Ch 4 · Connect to your business","level":"intermediate",
     "desc":"A skill wrapping NeMo Retriever RAG over YOUR documents (or an internal API): the "
     "frontier agent queries the skill, retrieves grounded chunks, and answers from your "
     "sovereign data — which never leaves the perimeter. Same agent, now expert in YOUR business."},
    {"id":"step04","group":"Connect","kind":"run","demo":"step04_skills_mcp_a2a.py",
     "title":"Ch 5 · Skills + MCP + A2A — framework-agnostic","level":"advanced",
     "desc":"The SAME skill works across agents and frameworks. Skills ride MCP for tools "
     "(agent→tools) and A2A for agent-to-agent delegation (agent→agent). Write the skill once; "
     "Claude, GPT, and Nemotron all load it. Ties Week 7 (MCP/Skills) to Week 17 (A2A)."},
    {"id":"outro","group":"Connect","kind":"concept",
     "title":"Appendix · Skills across the stack","level":"all levels",
     "desc":"Skills are the portable, standard way to connect a frontier agent to your business — "
     "reused across the Week 23 apps: AI-Q (deep research), NemoClaw, and NeMo Relay all load "
     "the same catalog of skills. Get them at github.com/NVIDIA/skills.\n\n"
     "Where this sits in Week 23: Skills are the CONNECTIVE tissue — they let the frontier "
     "agent (the Model) reach your sovereign tools & data (the Harness) without rebuilding it."},
    {"id":"refs","group":"Go further","kind":"concept",
     "title":"Appendix · References & real-world applications","level":"all levels",
     "desc":"Curated references:\n  • The catalog — github.com/NVIDIA/skills\n  • The Skills format (Claude ecosystem origin) — docs.claude.com/en/docs/agents-and-tools/agent-skills\n  • This course's own skill-craft — Week 7 (mcp-and-skills) and the\n    skill-authoring plugin skill.\n  • MCP + A2A companions — modelcontextprotocol.io and Week 17's A2A material.\n\nReal-world applications:\n  • Write once, run on any frontier agent — the same SKILL.md powers Claude Code,\n    internal Nemotron agents, and GPT-based assistants; enterprises version their\n    domain know-how (SOPs, runbooks, style guides) as skills in git.\n  • GPU superpowers without integration work — the cuOpt (routing) and cuDF\n    (dataframes) skills give any agent solver-grade capabilities in one load.\n  • Business-system bridges — skills wrapping telemetry, CMMS and document\n    stores are exactly how AltoTech-style building agents connect (Weeks 15/21\n    do this against a real hotel domain).\n  • Skills + MCP + A2A — package know-how, serve tools, connect agents: the\n    three-way interop stack this course has assembled since Week 7."},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="NVIDIA Agent Skills — interactive tutorial")
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
    banner = ["", "  ▣  NVIDIA Agent Skills — connect frontier agents to your business"]
    if config.MODE == "real":
        banner += [f"      ✓ REAL endpoint: {config.MODEL} @ {config.BASE_URL}",
                   "        skills load into a real agent, fully on-device — cloud cost $0.00."]
    else:
        banner += ["      ◈ SIM mode — no endpoint reachable, simulating skill loading.",
                   "        every concept is learnable with no GPU. Go REAL anytime:",
                   "        ollama run nemotron-3-nano   (or set DGX_BASE_URL)"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set SKILLS_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
