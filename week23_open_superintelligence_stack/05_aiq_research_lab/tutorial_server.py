#!/usr/bin/env python3
"""Interactive, explainable tutorial for the **NVIDIA AI-Q Open Agent Blueprint**.

A small control plane that serves a clickable web guide (static/guide.html) and,
for each chapter, lets you read the CONCEPT, view the demo SOURCE, and RUN it.

AI-Q is the HARNESS in "Agent = Model + Harness": an open, customizable deep-research
multi-agent system (Intent Router → Deep Agent → researcher sub-agents → tools) built on
open Nemotron models via the NeMo Agent Toolkit.

Two modes, auto-detected (see config.py):
  • REAL — a live OpenAI-compatible endpoint (Ollama / vLLM / a DGX you point
    DGX_BASE_URL at) serving Nemotron. The router + sub-agents call it for real.
  • SIM  — no endpoint reachable → the AI-Q agents are simulated instead, so every
    chapter is learnable with no GPU. Real commands/prompts are always shown.

Either way cloud cost is $0.00.

Launch (auto-picks a free port if 8104 is taken):

    AIQ_GUIDE_PORT=8104 .venv/bin/python week23/05_aiq_research_lab/tutorial_server.py
    # → http://127.0.0.1:8104
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

GUIDE_PORT = int(os.environ.get("AIQ_GUIDE_PORT", "8104"))


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
     "title":"Ch 1 · What is AI-Q? A research lab for any domain","level":"beginner",
     "desc":"Week 23 · Tutorial 05 of 12 · Phase: Build real agents. The NVIDIA AI-Q Open Agent "
     "Blueprint is a customizable deep-research multi-agent lab: an Intent Router decides whether a "
     "query needs a quick answer or a full investigation, then a Deep Agent orchestrates planning and "
     "researcher sub-agents that reach the world through tools wired up by the NeMo Agent Toolkit.\n\n"
     "In this tutorial:\n"
     "  • Intent Router — route or escalate: a cheap Nemotron Nano classifies each query as shallow lookup or deep research.\n"
     "  • Deep Agent — orchestrate & plan: the escalated task is decomposed into an explicit, written research plan.\n"
     "  • Researcher sub-agents fan out: parallel Nemotron Super agents call tools to gather evidence and return findings.\n"
     "  • Tools & data via NeMo Agent Toolkit: the tool bus connecting docs, web search, RAG, MCP, and sandbox skills.\n\n"
     "Why it matters for sovereign agents:\n"
     "  • Route simple queries cheaply, escalate only hard ones to deep research — no big agent for \"who is X?\".\n"
     "  • Orchestrate specialist sub-agents instead of one model holding the whole task in context.\n"
     "  • Plug in your own tools and data through the NeMo Agent Toolkit — docs, web, RAG, MCP.\n"
     "  • Run it ~50% cheaper on open Nemotron, more observable and customizable than a closed stack.\n\n"
     "Where it fits:\n"
     "Prerequisite: 04_agent_skills (skills become the tools these agents call). Feeds 06_nemoclaw "
     "(build the specialists) and 08_nemo_relay (which observes these agents), plus the capstone. "
     "Next: 06_nemoclaw.\n\n"
     "How to run:\n"
     "Click Run per chapter: REAL against a Nemotron NIM/DGX endpoint via 🔌 Connection, or SIM with "
     "no GPU ($0)."},
    {"id":"step01","group":"Route","kind":"run","demo":"step01_intent_router.py",
     "title":"Ch 2 · Intent Router — route or escalate","level":"beginner",
     "desc":"The front door of the lab. A cheap, fast Nemotron 3 Nano running on LangGraph reads the "
     "user's query and classifies it: a simple lookup goes to Shallow Research (Nano + NAT + Dynamo "
     "priority routing), while a hard, multi-step question is escalated to the LangChain Deep Agent. "
     "The NAT Optimizer keeps the router's own prompts and model choice tuned. Routing cheaply is "
     "how AI-Q hits ~50% lower cost — you don't wake the big agent for 'who is X?'."},
    {"id":"step02","group":"Deep Agent","kind":"run","demo":"step02_deep_agent_plan.py",
     "title":"Ch 3 · Deep Agent — orchestrate & plan","level":"intermediate",
     "desc":"When the router escalates, the LangChain Deep Agent takes over. Its Orchestration layer "
     "(GPT 5.2 or an open Nemotron, wired via NAT) hands the task to a Planning Sub-Agent that "
     "decomposes it into an explicit, ordered research plan written to the filesystem To-Do. "
     "Decompose first, delegate second — a written plan is what makes a long research run "
     "observable and resumable instead of one giant opaque prompt."},
    {"id":"step03","group":"Deep Agent","kind":"run","demo":"step03_researchers_fanout.py",
     "title":"Ch 4 · Researcher sub-agents fan out","level":"intermediate",
     "desc":"The orchestrator dispatches the plan's sub-tasks to parallel Researcher Sub-Agents — each "
     "a Nemotron 3 Super instance that calls tools (web search, NeMo Retriever RAG) to gather evidence "
     "and returns findings to Memory. Fan-out is the multi-agent pattern: many focused Super agents "
     "working concurrently beat one model trying to hold the whole task in context."},
    {"id":"step04","group":"Tools","kind":"run","demo":"step04_tools_toolkit.py",
     "title":"Ch 5 · Tools & data via NeMo Agent Toolkit","level":"advanced",
     "desc":"How the agents actually touch the world. The NeMo Agent Toolkit (Week 16) is the tool bus "
     "that connects Documents, Tavily web search, NeMo Retriever RAG, MCP servers, and the AI Data "
     "Platform — plus Sandbox skills for Data Analysis and Image Processing, and a filesystem for "
     "To-Do / Memory / Files. Swap in your own tools and data and you have a custom research lab for "
     "your domain, with every tool call observable."},
    {"id":"outro","group":"Tools","kind":"concept",
     "title":"Appendix · Why AI-Q — accuracy, customize, observe, 50% cost","level":"all levels",
     "desc":"Why the open blueprint wins: Best Accuracy · Easier to Customize · More Observable · ~50% "
     "Lower Cost with open Nemotron models instead of a closed frontier model doing everything.\n\n"
     "AI-Q is the HARNESS layer of Agent = Model + Harness — it orchestrates the open MODELS you "
     "learned in the Nemotron app. Get it at build.nvidia.com/blueprints.\n\n"
     "Where this sits in Week 23: Nemotron (model) · NIM (serve) · AI-Q (THIS — orchestrate) · "
     "Data Flywheel (improve) · Dynamo (scale) · NeMo Gym (RL) · OpenShell (guard)."},
    {"id":"refs","group":"Go further","kind":"concept",
     "title":"Appendix · References & real-world applications","level":"all levels",
     "desc":"Curated references:\n  • AI-Q blueprint — build.nvidia.com (search: AI-Q research assistant)\n  • NeMo Agent Toolkit (the tool layer) — github.com/NVIDIA/NeMo-Agent-Toolkit —\n    Week 16 of this course teaches it end-to-end.\n  • Deep-research pattern lineage — this course's deep-research skill and the\n    orchestrator-workers pattern (Week 9/11).\n\nReal-world applications:\n  • Private-corpus deep research — due diligence over deal rooms, patent\n    landscaping, pharma literature review, competitive intel: domains where the\n    corpus cannot leave the building, so hosted deep-research products are\n    disqualified and an on-prem AI-Q-style lab wins.\n  • Cost-rational assistants — the Intent Router's cheap-first triage is the\n    production answer to 'our copilot bill exploded'; NVIDIA's ~50% figure is\n    routing math on a mixed workload, and your mileage depends on your mix.\n  • Regulated report generation — plan → research → synthesize with citations,\n    where every claim must trace to an internal source document.\n  • The capstone's brain — App 12 runs this exact router → deep-agent →\n    specialists shape against a hotel; Week 21 gives it a digital-twin body."},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="NVIDIA AI-Q — interactive tutorial")
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
    banner = ["", "  ▣  NVIDIA AI-Q — an open deep-research lab for any domain"]
    if config.MODE == "real":
        banner += [f"      ✓ REAL endpoint: {config.MODEL} @ {config.BASE_URL}",
                   "        router + sub-agents call your Nemotron endpoint — cloud cost $0.00."]
    else:
        banner += ["      ◈ SIM mode — no endpoint reachable, simulating the AI-Q agents.",
                   "        every chapter is learnable with no GPU. Go REAL anytime:",
                   "        ollama run nemotron-3-nano   (or set DGX_BASE_URL)"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set AIQ_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
