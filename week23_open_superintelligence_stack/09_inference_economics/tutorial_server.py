#!/usr/bin/env python3
"""Interactive, explainable tutorial for **AI Performance & Evaluation**.

The economics of intelligence: tokens as the unit of work, cost per million
tokens, throughput per GPU and per Megawatt, and the modern shift from raw
tokens to GOODPUT (cost per successful task) — which makes evaluation part of
performance. A small control plane that serves a clickable web guide
(static/guide.html) and, per chapter, lets you read the CONCEPT, view the demo
SOURCE, and RUN it.

Two modes, auto-detected (see config.py):
  • REAL — a live OpenAI-compatible endpoint. Demos MEASURE real tok/s and
    derive the dollars-per-million-token economics from it.
  • SIM  — no endpoint reachable → illustrative constants stand in, so every
    formula is learnable with no GPU. Real commands are always shown.

Either way cloud cost is $0.00.

Launch (auto-picks a free port if 8108 is taken):

    ECON_GUIDE_PORT=8108 .venv/bin/python week23/09_inference_economics/tutorial_server.py
    # → http://127.0.0.1:8108
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

GUIDE_PORT = int(os.environ.get("ECON_GUIDE_PORT", "8108"))


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
     "title":"Ch 1 · Tokens — the unit of AI work","level":"beginner",
     "desc":"Week 23 · Tutorial 09 of 12 · Phase: Observe, measure & optimize. This is AI "
     "Performance & Evaluation — tokens are the unit of work, so we measure cost per million "
     "tokens and throughput per GPU and per Megawatt. But only USEFUL tokens count, so goodput "
     "and evaluation decide the real value of a deployment.\n\n"
     "In this tutorial:\n"
     "  • Ch 2 · Cost per million tokens — compute $/Mtok two ways: open-on-DGX vs a hosted cloud API.\n"
     "  • Ch 3 · Throughput — per GPU and per Megawatt — tokens/s over silicon and over power.\n"
     "  • Ch 4 · From tokens to goodput — cost per SUCCESSFUL task, not cost per raw token.\n"
     "  • Ch 5 · Evaluate — score task success — run a golden set through an LLM-as-judge.\n\n"
     "Why it matters for sovereign agents:\n"
     "  • Tokens are the unit, but cheap tokens can buy WORSE outcomes if the model fails more often.\n"
     "  • Power (MW) is the real ceiling at scale — you run out of megawatts before you run out of GPU budget.\n"
     "  • Goodput reframes cost as dollars per SUCCESSFUL task, the number that actually predicts value.\n"
     "  • Evaluation (LLM-judge + golden set) gates quality, and open-on-DGX runs ≈50% cheaper (AI-Q).\n\n"
     "Where it fits:\n"
     "Prerequisites: 03_dynamo_serving (throughput/economics) and 08_nemo_relay (the cost/latency "
     "it observes). Feeds 10_nemo_gym_rl (evaluation gates what RL optimizes) and the capstone. "
     "Next: 10_nemo_gym_rl.\n\n"
     "How to run:\n"
     "Click Run per chapter: REAL (measures live tok/s and derives the economics) via 🔌 Connection, "
     "or SIM with illustrative numbers, no GPU ($0)."},
    {"id":"step01","group":"Cost","kind":"run","demo":"step01_cost_per_mtok.py",
     "title":"Ch 2 · Cost per million tokens","level":"beginner",
     "desc":"Cost/Mtok = infra cost / tokens. We compute it two ways — an open model on YOUR "
     "DGX (amortized infra only) vs a hosted cloud API (usage-billed) — landing on the AI-Q "
     "claim that open-on-DGX runs at roughly HALF the cost. In SIM the numbers are illustrative "
     "constants; in REAL we measure tok/s from a live generation and derive the $/Mtok."},
    {"id":"step02","group":"Throughput","kind":"run","demo":"step02_throughput.py",
     "title":"Ch 3 · Throughput — per GPU and per Megawatt","level":"intermediate",
     "desc":"Throughput has two denominators: tokens/s ÷ # of GPUs (how well you use silicon) "
     "and tokens/s ÷ Megawatts (how well you use POWER). At data-center scale POWER is the "
     "binding constraint, not chip count — you run out of megawatts before you run out of "
     "money for GPUs. That is why tokens-per-watt is the currency of an AI factory (App 3 · Dynamo)."},
    {"id":"step03","group":"Goodput","kind":"run","demo":"step03_goodput.py",
     "title":"Ch 4 · From tokens to goodput","level":"intermediate",
     "desc":"Raw tokens are cheap — only USEFUL tokens matter. A reasoning agent emits long "
     "chains of thought, so a 'cheaper per token' model can be more expensive per RESULT if it "
     "fails more often. The modern metric is cost per SUCCESSFUL task (goodput), not cost per "
     "token: a slower, pricier-per-token model that gets it right first try can win outright."},
    {"id":"step04","group":"Evaluate","kind":"run","demo":"step04_evaluate.py",
     "title":"Ch 5 · Evaluate — score task success","level":"advanced",
     "desc":"You can only optimize goodput if you can MEASURE success. This runs a golden set "
     "through an LLM-as-judge: task → agent answer → judge verdict (pass/fail) → aggregate to a "
     "score, the exact pattern from Weeks 10 & 15 that gates quality in CI. Without evaluation, "
     "chasing cheaper tokens silently buys you worse outcomes — perf without correctness is a trap."},
    {"id":"outro","group":"Evaluate","kind":"concept",
     "title":"Appendix · The economics of intelligence","level":"all levels",
     "desc":"The whole session in one line: performance × correctness = value. Cheap tokens are "
     "worthless if they're wrong; correct answers are unaffordable if they're slow. You need both.\n\n"
     "The sovereign play: open models + your own DGX (≈50% lower cost, AI-Q) + evaluation you "
     "control. Power (MW) is the real ceiling at scale.\n\n"
     "Where this sits in Week 23: App 1 Nemotron (model) · App 2 NIM (serve) · App 11 Data "
     "Flywheel (improve) · App 3 Dynamo (scale — the perf/$ & per-MW engine) · App 10 NeMo Gym "
     "(RL + eval — the correctness engine) · App 7 OpenShell (guard). This app is the lens that "
     "connects Dynamo's performance to Gym's evaluation."},
    {"id":"refs","group":"Go further","kind":"concept",
     "title":"Appendix · References & real-world applications","level":"all levels",
     "desc":"Curated references:\n  • Independent per-token benchmarks — artificialanalysis.ai (price + speed +\n    quality across providers, updated continuously)\n  • NVIDIA on tokens/megawatt & AI factories — blogs.nvidia.com (AI factory\n    series; the DSX blueprint prices data centers in tokens per MW)\n  • Goodput lineage — this course's Week 10/15 eval material (LLM-judge +\n    golden sets make 'successful task' measurable).\n\nReal-world applications:\n  • Capacity planning — AI data centers are now sized in tokens/second/MW;\n    that lens decides GPU purchases, power contracts, and whether an always-on\n    agent fleet is viable at all.\n  • The cheap-model trap — a model at a third of the per-token price that needs\n    two retries and a judge pass costs MORE per successful task: goodput math\n    catches what sticker price hides.\n  • Per-tenant budgets — SaaS copilots meter tokens per customer with circuit\n    breakers (Week 10's cost governance) to keep one tenant from eating the\n    margin.\n  • Make vs buy — the on-prem $/M-token you computed in Ch 3 vs cloud sticker\n    price, at your utilization, is the whole sovereign-hardware business case."},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="AI Performance & Evaluation — interactive tutorial")
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
    banner = ["", "  ▣  AI Performance & Evaluation — the economics of intelligence"]
    if config.MODE == "real":
        banner += [f"      ✓ REAL endpoint: {config.MODEL} @ {config.BASE_URL}",
                   "        demos MEASURE real tok/s and derive $/Mtok — cloud cost $0.00."]
    else:
        banner += ["      ◈ SIM mode — no endpoint reachable, using illustrative constants.",
                   "        every formula is learnable with no GPU. Go REAL anytime:",
                   "        ollama run qwen3.6:35b-a3b-q8_0   (or set DGX_BASE_URL)"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set ECON_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
