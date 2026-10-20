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

GUIDE_PORT = int(os.environ.get("DYNAMO_GUIDE_PORT", "8102"))


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
     "title":"Ch 1 · What is NVIDIA Dynamo?","level":"beginner",
     "desc":"Week 23 · Tutorial 03 of 12 · Phase: Serve it sovereignly (at scale). NVIDIA Dynamo "
     "is serve-at-scale for an always-on agent: a distributed inference framework combining "
     "disaggregated prefill/decode, KV-cache-aware routing, and an SLO Planner to keep a "
     "long-running agent fast and economical across 1 or many DGX Sparks.\n\n"
     "In this tutorial:\n"
     "  • Ch 2 · What is Dynamo? — the four pieces, and why one NIM isn't enough for an agent sending millions of prompts forever.\n"
     "  • Ch 3 · Disaggregated + cache-aware — split compute-bound prefill from memory-bound decode; route to the worker that cached the prefix.\n"
     "  • Ch 4 · SLO Planner — hold latency — declare TTFT/ITL targets; autoscale pools so latency stays flat under a load ramp.\n"
     "  • Ch 5 · Token economics — cost per 1M tokens and tokens/s per GPU and per megawatt, showing how these techniques compound.\n\n"
     "Why it matters for sovereign agents:\n"
     "  • A single NIM serves one model well but can't autoscale a fleet to meet demand.\n"
     "  • Disaggregating prefill and decode onto right-sized GPU pools uses hardware far more efficiently.\n"
     "  • KV-cache-aware routing reuses cached prefixes instead of recomputing them, cutting wasted compute.\n"
     "  • The SLO Planner holds latency under load; better token economics decide whether an always-on agent is viable.\n\n"
     "Where it fits:\n"
     "  • Prerequisite: 02_nim_microservices (serve one model first). Feeds 09_inference_economics (cost/throughput) and the capstone. Next: 04_agent_skills (start building agents).\n\n"
     "How to run:\n"
     "  • Click Run per chapter: REAL against a DGX endpoint via 🔌 Connection, or SIM with no GPU ($0). Multi-node chapters simulate the 2-Spark scaling if you have one."},
    {"id":"step01","group":"Serve","kind":"run","demo":"step01_what_is_dynamo.py",
     "title":"Ch 2 · What is Dynamo?","level":"beginner",
     "desc":"The four pieces of Dynamo and why a single NIM isn't enough for an always-on "
     "agent that sends millions of prompts, forever."},
    {"id":"step02","group":"Disaggregate","kind":"run","demo":"step02_disaggregated.py",
     "title":"Ch 3 · Disaggregated + cache-aware","level":"intermediate",
     "desc":"Prefill (compute-bound) and decode (memory-bound) on separate pools, with "
     "cache-aware routing — ~3x throughput at ~1/3 the cost/token vs a naive worker."},
    {"id":"step03","group":"Scale","kind":"run","demo":"step03_slo_planner.py",
     "title":"Ch 4 · SLO Planner — hold latency","level":"advanced",
     "desc":"Declare TTFT/ITL objectives; the Planner autoscales pools under a load ramp so "
     "latency stays flat while a naive worker breaches the SLO."},
    {"id":"step04","group":"Economics","kind":"run","demo":"step04_token_economics.py",
     "title":"Ch 5 · Token economics","level":"advanced",
     "desc":"Cost per 1M tokens, tokens/s per GPU and per megawatt — how disaggregation + "
     "caching + autoscaling compound into a far cheaper always-on agent."},
    {"id":"outro","group":"Economics","kind":"concept",
     "title":"Appendix · The serve-at-scale stack","level":"all levels",
     "desc":"Dynamo is the SCALE layer of Agent = Model + Harness — the cost/token that "
     "decides whether an always-on sovereign agent is viable.\n\n"
     "Where this sits in Week 23: App 1 Nemotron (model) · App 2 NIM (serve) · "
     "App 11 Data Flywheel (improve) · App 3 (THIS) Dynamo (scale) · App 10 NeMo Gym (RL) · "
     "App 7 OpenShell (guard)."},
    {"id":"refs","group":"Go further","kind":"concept",
     "title":"Appendix · References & real-world applications","level":"all levels",
     "desc":"Curated references:\n  • NVIDIA Dynamo — developer.nvidia.com/dynamo\n  • Disaggregated serving design docs — docs.dynamo.nvidia.com/dynamo/design-docs/disaggregated-serving\n  • The research lineage — DistServe (arxiv.org/abs/2401.09670) and Splitwise\n    (arxiv.org/abs/2311.18677) pioneered prefill/decode disaggregation; Dynamo\n    productizes it.\n  • Open source — github.com/ai-dynamo/dynamo\n\nReal-world applications:\n  • Frontier-scale serving — the prefill/decode split and KV-aware routing are how\n    large inference providers keep per-token cost survivable at millions of\n    concurrent requests; NVIDIA named launch partners across major AI labs at\n    GTC 2025.\n  • Agent workloads specifically — a fleet re-sending a 4k-token system prompt\n    every turn is the best case for cache-aware routing: the prefix computes once.\n  • AI-factory capacity planning — the SLO Planner pattern (declare TTFT/ITL,\n    autoscale pools) is the operational model behind the DSX blueprint's\n    'twin as operating system' story (Week 21 App 1).\n  • Honest scope — a single-model, low-QPS internal tool does not need Dynamo;\n    one NIM or vLLM is the right size until the fleet grows."},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="NVIDIA Dynamo — interactive tutorial")
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
    banner = ["", "  ▣  NVIDIA Dynamo — serving long-running agents at scale"]
    if config.MODE == "real":
        banner += [f"      ✓ REAL endpoint: {config.MODEL} @ {config.BASE_URL}",
                   "        demos run for real, fully on-device — cloud cost $0.00."]
    else:
        banner += ["      ◈ SIM mode — no endpoint reachable, simulating a DGX Spark.",
                   "        every concept is learnable with no GPU. Go REAL anytime:",
                   "        ollama run qwen3.6:35b-a3b-q8_0   (or set DGX_BASE_URL)"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set DYNAMO_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
