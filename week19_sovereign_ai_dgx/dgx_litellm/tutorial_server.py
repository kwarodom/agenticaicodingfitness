#!/usr/bin/env python3
"""Interactive, explainable tutorial for **Serving models on a DGX with LiteLLM**.

A clickable web guide: for each chapter read the CONCEPT, view the demo SOURCE, and
RUN it. Auto-detects a real LiteLLM proxy (PROXY mode) / a bare backend (DIRECT) /
nothing (SIM). Cloud cost $0.

    .venv/bin/python week19/dgx_litellm/tutorial_server.py
    # → http://127.0.0.1:8096
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

PKG = Path(__file__).resolve().parent
ROOT = PKG.parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable
DEMOS = PKG / "demos"
GUIDE_PORT = int(os.environ.get("LITELLM_GUIDE_PORT", "8096"))


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
        "title": "Ch 1 · Why a gateway (LiteLLM) on a DGX", "level": "beginner",
        "desc": "App 1 gave you several ways to SERVE a model (Ollama, vLLM, "
        "llama.cpp, TRT-LLM). A real deployment runs SEVERAL of these at once — "
        "different models, on different Sparks. LiteLLM is the GATEWAY that puts one "
        "OpenAI-compatible URL + key in front of all of them.\n\n"
        "What the gateway adds on top of raw serving:\n"
        "  • One endpoint   — apps call one URL/key; backends can change underneath\n"
        "  • Routing        — load-balance one model across multiple Sparks\n"
        "  • Fallbacks      — a Spark/model fails → transparently try the next\n"
        "  • Hot-swap       — register more models than fit in VRAM; load on demand\n"
        "  • Virtual keys   — per-team allow-lists, rate limits, budgets (quotas)\n"
        "  • Observability  — every call logged/traced to Phoenix on the box\n\n"
        "Where it sits in the sovereign stack:\n"
        "  apps/agents → [ LiteLLM gateway ] → Ollama / vLLM / llama.cpp / TRT-LLM\n\n"
        "Everything stays on-prem — the gateway is just an OpenAI-compatible proxy you "
        "run yourself. This app auto-detects a real LiteLLM proxy; without one it "
        "makes real backend calls or simulates the router, so you learn it at $0. "
        "Click through Ch 2–8.",
    },
    {"id": "step01", "group": "Setup", "kind": "run", "demo": "step01_install_config.py",
     "title": "Ch 2 · Install LiteLLM & write the config", "level": "beginner",
     "desc": "Install the proxy and write a real config.yaml model_list mapping "
     "friendly aliases (dgx-fast/-smart/-tiny) to your DGX backends. Launch with "
     "`litellm --config`, then make one gateway call."},
    {"id": "step02", "group": "Setup", "kind": "run", "demo": "step02_unified_endpoint.py",
     "title": "Ch 3 · One endpoint, many models", "level": "beginner",
     "desc": "Call several models through the SAME base_url + key — only the model "
     "name changes. Apps stop caring about IPs, ports, or which runtime serves what."},
    {"id": "step03", "group": "Scale", "kind": "run", "demo": "step03_routing.py",
     "title": "Ch 4 · Routing & load-balancing across Sparks", "level": "intermediate",
     "desc": "Give one alias multiple deployments and watch the router spread load — "
     "simple-shuffle vs usage-based vs latency-based — with per-deployment caps."},
    {"id": "step04", "group": "Scale", "kind": "run", "demo": "step04_fallbacks.py",
     "title": "Ch 5 · Fallbacks & reliability", "level": "intermediate",
     "desc": "Retries, cooldowns, and fallback chains keep the gateway answering when "
     "a Spark OOMs or a model reloads. Watch a request fall through to a healthy model."},
    {"id": "step05", "group": "Models", "kind": "run", "demo": "step05_hotswap.py",
     "title": "Ch 6 · Model management & hot-swap", "level": "intermediate",
     "desc": "128 GB can't hold every model. With llama-swap behind LiteLLM, register "
     "many models behind one URL and swap them in/out of VRAM on demand — plus "
     "aliases and wildcards."},
    {"id": "step06", "group": "Govern", "kind": "run", "demo": "step06_keys_budgets.py",
     "title": "Ch 7 · Virtual keys, budgets & rate limits", "level": "advanced",
     "desc": "Per-team virtual keys with model allow-lists, RPM/TPM limits, and "
     "budgets — which on a DGX act as QUOTAS protecting shared GPU capacity. Mint, "
     "scope, and revoke keys via the proxy API."},
    {"id": "step07", "group": "Observe", "kind": "run", "demo": "step07_observability.py",
     "title": "Ch 8 · Logging & observability callbacks", "level": "advanced",
     "desc": "Every call funnels through the gateway — the ideal place to observe. "
     "One callbacks block streams spans to Phoenix on the DGX (App 3) for fleet-wide, "
     "on-prem observability."},
    {
        "id": "outro", "group": "Observe", "kind": "concept",
        "title": "Appendix · The gateway in the sovereign stack", "level": "all levels",
        "desc": "The full picture, all on-prem:\n\n"
        "  apps / agents (App 4)\n"
        "       │  one OpenAI URL + virtual key\n"
        "  LiteLLM gateway (THIS app) — route · balance · fallback · keys · logs\n"
        "       │\n"
        "  backends (App 1): Ollama · vLLM · llama.cpp · TRT-LLM\n"
        "       │  models tuned to your domain (App 2)\n"
        "  observability: spans → Phoenix on the DGX (App 3)\n\n"
        "Quickstart:\n"
        "  pip install 'litellm[proxy]'\n"
        "  litellm --config litellm_config.yaml --port 4000\n"
        "  export LITELLM_BASE_URL=http://localhost:4000   # then point this app at it\n\n"
        "Where this sits in Week 19:\n"
        "  App 1 run/serve · App 2 fine-tune · App 3 observe+NAT · App 4 self-evolving\n"
        "  App 5 (THIS) — the SERVING/gateway layer that ties the runtimes together.\n\n"
        "LiteLLM is the single control point of a sovereign deployment: one URL to "
        "route, govern, and observe every local model call — and nothing leaves the box.",
    },
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="Serving on a DGX with LiteLLM — interactive tutorial")
_run_lock = asyncio.Lock()

SELECTED = {"model": config.MODEL}
_SIM_MODELS = ["qwen3.6:35b-a3b-q8_0", "llama3.3:70b", "gemma4:12b", "llama3.1:8b"]


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(PKG / "static" / "guide.html",
                        headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/api/steps")
async def steps() -> dict:
    def public(s):
        return {k: s.get(k) for k in ("id", "group", "title", "desc", "kind", "level")} | \
               {"demo": s.get("demo")}
    import litesim
    models = config.list_backend_models() or _SIM_MODELS
    if SELECTED["model"] not in models:
        SELECTED["model"] = models[0]
    return {"steps": [public(s) for s in STEPS], "mode": config.MODE,
            "situation": config.SITUATION, "conn": config.CONN,
            "conn_human": config.conn_human(), "model": SELECTED["model"], "models": models,
            "litellm_url": config.LITELLM_BASE_URL, "backend_url": config.BACKEND_URL,
            "aliases": litesim.aliases()}


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
    config.apply_connection(req.model_dump())
    models = config.list_backend_models() or _SIM_MODELS
    SELECTED["model"] = config.MODEL if config.MODEL in models else (models[0] if models else config.MODEL)
    return {"ok": True, "conn": config.CONN, "situation": config.SITUATION,
            "endpoint_up": config.backend_up(), "model": SELECTED["model"], "models": models}


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
    timeout = 300.0 if config.MODE == "real" else 120.0

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
    s = config.SITUATION
    banner = ["", "  ▣  Serving on a DGX with LiteLLM — interactive tutorial"]
    if s == "proxy":
        banner += [f"      ✓ REAL · PROXY — LiteLLM gateway @ {config.LITELLM_BASE_URL}"]
    elif s == "direct":
        banner += [f"      ✓ REAL · DIRECT — backend @ {config.BACKEND_URL} (router shown via sim)",
                   f"        run a proxy for full routing:  litellm --config litellm_config.yaml"]
    else:
        banner += ["      ◈ SIM — no proxy/backend; gateway simulated. Go real:",
                   "        pip install 'litellm[proxy]' && litellm --config litellm_config.yaml"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set LITELLM_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
