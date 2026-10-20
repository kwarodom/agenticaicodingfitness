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

GUIDE_PORT = int(os.environ.get("DGX_GUIDE_PORT", "8092"))


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
    {
        "id": "intro", "group": "Foundations", "kind": "concept",
        "title": "Ch 1 · Sovereign AI on a DGX — why", "level": "beginner",
        "desc": "Sovereign AI = you own the whole supply chain: the hardware, the "
        "data, the model weights, and the software. Nothing touches an external "
        "server unless you choose it. A DGX makes this practical on a desk.\n\n"
        "Cloud API vs DGX-sovereign:\n"
        "  • Data        — leaves your perimeter      →  never leaves the box\n"
        "  • Latency     — 50–200 ms round-trip       →  on-device, no network hop\n"
        "  • Uptime      — internet required          →  fully offline / air-gappable\n"
        "  • Cost        — per-token API fees          →  one-time hardware, $0/token\n"
        "  • Compliance  — data-residency contracts    →  guaranteed by physics\n\n"
        "The DGX sovereign stack — every layer on-prem:\n"
        "  Hardware → DGX Spark (GB10, 128 GB) · DGX Station (GB300, 784 GB)\n"
        "  Models   → Qwen 3.6 · Llama 3.3 · Gemma 4 · Nemotron (open weights)\n"
        "  Runtime  → Ollama · vLLM · llama.cpp · TensorRT-LLM\n"
        "  Serving  → OpenAI-compatible API · LiteLLM · llama-swap\n"
        "  App      → agents, RAG, fine-tuned domain models (later apps)\n\n"
        "This app runs that stack for real on a live endpoint — or SIMULATES a DGX "
        "Spark if you have no GPU. The MODE pill (top-right) tells you which. Click "
        "through Chapters 2–9 and watch sovereign AI on a DGX actually work — $0.",
    },
    {"id": "step01", "group": "Foundations", "kind": "run", "demo": "step01_dgx_hello.py",
     "title": "Ch 2 · Your first sovereign inference", "level": "beginner",
     "desc": "A drop-in OpenAI call where only the base_url changed — pointed at "
     "your DGX. Watch the answer stream with measured tok/s and $0 cloud cost. In "
     "SIM mode a DGX Spark is simulated so you still see the mechanics."},
    {"id": "step02", "group": "Hardware", "kind": "run", "demo": "step02_dgx_hardware.py",
     "title": "Ch 3 · DGX hardware — what fits in 128 GB", "level": "beginner",
     "desc": "DGX Spark (GB10, 128 GB) vs DGX Station (GB300, 784 GB). Memory "
     "bandwidth — not TOPS — sets decode tok/s. See exactly which models fit a "
     "Spark and why a 35B MoE outruns a 32B dense model."},
    {"id": "step03", "group": "Runtimes", "kind": "run", "demo": "step03_ollama_on_dgx.py",
     "title": "Ch 4 · Ollama — the one-command path", "level": "beginner",
     "desc": "The fastest way to a sovereign endpoint on a DGX: install, pull, "
     "serve. Auto-GPU, model library, OpenAI API on :11434. Watch the GB10 "
     "telemetry before/after the model loads, then a real (or simulated) call."},
    {"id": "step04", "group": "Runtimes", "kind": "run", "demo": "step04_vllm_on_dgx.py",
     "title": "Ch 5 · vLLM — throughput serving", "level": "intermediate",
     "desc": "Production serving for many users via PagedAttention + continuous "
     "batching, in NVIDIA's Blackwell container. Same OpenAI API; far higher "
     "aggregate throughput. REAL mode measures TTFT + decode on the live endpoint."},
    {"id": "step05", "group": "Runtimes", "kind": "run", "demo": "step05_llamacpp_on_dgx.py",
     "title": "Ch 6 · llama.cpp — GGUF + native CUDA", "level": "intermediate",
     "desc": "The lightweight C/C++ engine behind Ollama, built with CUDA for "
     "Blackwell. One binary, any GGUF, full control — ideal for edge + air-gap. "
     "Covers the Q4_K_M sweet spot and the build/serve commands."},
    {"id": "step06", "group": "Runtimes", "kind": "run", "demo": "step06_runtime_bakeoff.py",
     "title": "Ch 7 · Bake-off — pick the right server", "level": "intermediate",
     "desc": "Ollama vs vLLM vs llama.cpp vs TensorRT-LLM: a decision table plus a "
     "live tok/s number (REAL) or representative DGX figures (SIM). Same API — "
     "different ops trade-offs."},
    {"id": "step07", "group": "Models", "kind": "run", "demo": "step07_model_management.py",
     "title": "Ch 8 · Model management & NVFP4", "level": "advanced",
     "desc": "A DGX is a fleet. List what's resident, do the VRAM math per format, "
     "and quantize a 70B to NVFP4 (Blackwell-native 4-bit float) so it fits in "
     "128 GB. The lever that brings frontier-class models on-desk."},
    {"id": "step08", "group": "Scale", "kind": "run", "demo": "step08_multi_spark.py",
     "title": "Ch 9 · Scaling out — two DGX Sparks", "level": "advanced",
     "desc": "Link two Sparks over ConnectX-7 200GbE → 256 GB coherent memory and "
     "tensor parallelism for a 235B model. Wiring, NCCL smoke test, TensorRT-LLM "
     "multi-node, and the memory math for what 2–3 Sparks unlock."},
    {"id": "step09", "group": "Trust", "kind": "run", "demo": "step09_sovereignty_audit.py",
     "title": "Ch 10 · Sovereignty & air-gap audit", "level": "advanced",
     "desc": "Turn 'sovereign' from a claim into a CI check: endpoint loopback? no "
     "cloud creds in env? local weights? round-trip on-box? Plus the hardening "
     "checklist from the playbooks."},
    {
        "id": "outro", "group": "Trust", "kind": "concept",
        "title": "Appendix · Quickstart & what's next", "level": "all levels",
        "desc": "One-command quickstart by hardware:\n"
        "  • DGX Spark / Ubuntu — curl -fsSL https://ollama.com/install.sh | sh && "
        "ollama run qwen3.6:35b-a3b-q8_0\n"
        "  • Mac (Apple Silicon) — brew install ollama && ollama run gemma4:12b\n"
        "  • Point this app at a real DGX — export DGX_BASE_URL=http://my-spark.local:11434/v1\n\n"
        "Model-by-memory cheat sheet (single DGX Spark, 128 GB):\n"
        "  8B dense Q4 → trivial   ·  35B MoE Q8 → ideal   ·  70B dense NVFP4 → fits\n"
        "  235B → needs two linked Sparks (TP=2)\n\n"
        "Where this sits in Week 19:\n"
        "  THIS app (sovereign_dgx) → run + serve + manage models on a DGX.\n"
        "  Next: dgx_finetune (adapt a model to YOUR domain on the DGX),\n"
        "        dgx_observability (trace agents with Phoenix + NeMo Agent Toolkit),\n"
        "        self_evolving_agent_v2 (the week18 agent, now driven by a DGX model).\n\n"
        "You went from one local 'hello' to a managed, multi-node, audited sovereign "
        "model fleet — and not a byte left the building.",
    },
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="Sovereign AI on a DGX — interactive tutorial")
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
    import dgxsim
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
    banner = ["", "  ▣  Sovereign AI on a DGX — interactive, explainable tutorial"]
    if config.MODE == "real":
        banner += [f"      ✓ REAL endpoint: {config.MODEL} @ {config.BASE_URL}",
                   "        demos run for real, fully on-device — cloud cost $0.00."]
    else:
        banner += ["      ◈ SIM mode — no endpoint reachable, simulating a DGX Spark.",
                   "        every concept is learnable with no GPU. Go REAL anytime:",
                   "        ollama run qwen3.6:35b-a3b-q8_0   (or set DGX_BASE_URL)"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set DGX_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
