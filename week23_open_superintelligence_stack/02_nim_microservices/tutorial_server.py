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

GUIDE_PORT = int(os.environ.get("NIM_GUIDE_PORT", "8101"))


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
     "title":"Ch 1 · What is a NIM?","level":"beginner",
     "desc":"Week 23 · Tutorial 02 of 12 · Phase: Serve it sovereignly. NVIDIA NIM is one "
     "signed container that packages a MODEL + an auto-selected optimized ENGINE "
     "(TensorRT-LLM / vLLM / SGLang) + an OpenAI-compatible API — the RUNTIME layer of "
     "Agent = Model + Harness. One image, and you have a production endpoint on your own DGX.\n\n"
     "In this tutorial:\n"
     "  • Ch 2 — Deploy a NIM (one command): pull + run a Nemotron NIM so a model, its "
     "optimized engine, and an OpenAI API come up together on :8000.\n"
     "  • Ch 3 — NIM vs raw vLLM vs Ollama: compare the trade-offs (who owns optimization, "
     "security, support) and see how to promote Ollama → NIM with no client change.\n"
     "  • Ch 4 — Call a NIM (same OpenAI API): hit the endpoint with curl, the OpenAI SDK, "
     "and streaming — a live (or simulated) generation against your DGX.\n"
     "  • Ch 5 — The catalog + your own custom NIM: browse build.nvidia.com's ready NIMs and "
     "wrap YOUR fine-tuned model (Week 19 LoRA) as a NIM — the sovereign AI-factory loop.\n\n"
     "Why it matters for sovereign agents:\n"
     "  • One docker run → a production endpoint; ops never tunes vLLM internals by hand.\n"
     "  • Runs on your hardware / AI factory — your data never leaves the perimeter.\n"
     "  • Same OpenAI API as the cloud — existing apps and agents work unchanged.\n"
     "  • Wrap your OWN fine-tuned model as a NIM to close the sovereign AI-factory loop.\n\n"
     "Where it fits:\n"
     "  Prerequisite: 01_nemotron_models (the model you're serving). This gives every later "
     "app an OpenAI-compatible endpoint to call. Next: 03_dynamo_serving (serve at scale).\n\n"
     "How to run:\n"
     "  Click Run per chapter: REAL against a NIM/Ollama/vLLM endpoint via 🔌 Connection, "
     "or SIM with no GPU ($0)."},
    {"id":"step01","group":"Deploy","kind":"run","demo":"step01_deploy_nim.py",
     "title":"Ch 2 · Deploy a NIM (one command)","level":"beginner",
     "desc":"Pull + run a Nemotron NIM: model + optimized engine + OpenAI API on :8000, "
     "in one container. The sovereign one-command path."},
    {"id":"step02","group":"Choose","kind":"run","demo":"step02_nim_vs_diy.py",
     "title":"Ch 3 · NIM vs raw vLLM vs Ollama","level":"intermediate",
     "desc":"Same OpenAI API, different trade-offs: who does the optimization, security, "
     "and support. When to pick each — and how to promote Ollama→NIM with no client change."},
    {"id":"step03","group":"Call","kind":"run","demo":"step03_call_nim.py",
     "title":"Ch 4 · Call a NIM — same OpenAI API","level":"intermediate",
     "desc":"curl, the OpenAI SDK, streaming — identical to the cloud, pointed at your DGX. "
     "Runs one live (or simulated) generation against the connected endpoint."},
    {"id":"step04","group":"Scale","kind":"run","demo":"step04_catalog_custom.py",
     "title":"Ch 5 · The catalog + your own custom NIM","level":"advanced",
     "desc":"build.nvidia.com's ready NIMs, and wrapping YOUR fine-tuned model (Week 19 LoRA) "
     "as a NIM — the sovereign AI-factory lifecycle."},
    {"id":"outro","group":"Scale","kind":"concept",
     "title":"Appendix · The sovereign runtime","level":"all levels",
     "desc":"NIM is the RUNTIME layer of Agent = Model + Harness. Get NIMs at build.nvidia.com; "
     "production use needs NVIDIA AI Enterprise (bundled with DGX).\n\n"
     "Where this sits in Week 23: App 1 Nemotron (model) · App 2 (THIS) NIM (serve) · "
     "App 11 Data Flywheel (improve) · App 3 Dynamo (scale) · App 10 NeMo Gym (RL) · App 7 OpenShell (guard)."},
    {"id":"refs","group":"Go further","kind":"concept",
     "title":"Appendix · References & real-world applications","level":"all levels",
     "desc":"Curated references:\n  • NIM microservices — nvidia.com/en-us/ai-data-science/products/nim-microservices/\n  • NIM docs — docs.nvidia.com/nim/\n  • The catalog (pull + run today) — build.nvidia.com\n  • Sovereign AI factories (why signed containers matter) — blogs.nvidia.com/blog/sovereign-ai-agents-factories/\n\nReal-world applications:\n  • Air-gapped approvals — a signed, versioned container is something a bank's or\n    ministry's security team can actually review and whitelist; 'pip install a\n    serving stack' is not. This is NIM's real product.\n  • Try-then-repatriate — teams prototype against the hosted NIM on\n    build.nvidia.com, then run the identical container on-prem with zero code\n    change (the same base_url swap this course has used since Week 18).\n  • ISV embedding — partner products (industrial-software and healthcare vendors)\n    ship NIMs as their inference backend rather than building serving in-house.\n  • Fleet standardization — one deployment unit across DGX, cloud GPUs, and\n    workstations keeps ops teams sane at enterprise scale."},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="NIM Microservices — interactive tutorial")
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
    banner = ["", "  ▣  NIM Microservices — sovereign inference in one container"]
    if config.MODE == "real":
        banner += [f"      ✓ REAL endpoint: {config.MODEL} @ {config.BASE_URL}",
                   "        demos run for real, fully on-device — cloud cost $0.00."]
    else:
        banner += ["      ◈ SIM mode — no endpoint reachable, simulating a DGX Spark.",
                   "        every concept is learnable with no GPU. Go REAL anytime:",
                   "        ollama run qwen3.6:35b-a3b-q8_0   (or set DGX_BASE_URL)"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set NIM_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
