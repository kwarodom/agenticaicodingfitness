#!/usr/bin/env python3
"""Interactive, explainable tutorial for **Observability on a DGX** —
tracing a sovereign agent with Phoenix + OpenTelemetry, and running a NeMo Agent
Toolkit (NAT) workflow against the local DGX model.

    .venv/bin/python week19/dgx_observability/tutorial_server.py
    # → http://127.0.0.1:8094
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
GUIDE_PORT = int(os.environ.get("OBS_GUIDE_PORT", "8094"))


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
        "title": "Ch 1 · Why observe a sovereign agent", "level": "beginner",
        "desc": "An agent you can't see is an agent you can't trust, debug, or "
        "improve. Observability turns each run into an inspectable TRACE of SPANS. "
        "On a DGX the win is double: the traces — which contain your prompts and "
        "data — never leave the box either, because Phoenix runs on the same "
        "hardware.\n\n"
        "The three pillars of LLM observability:\n"
        "  • Tracing  — spans for every agent / LLM / tool step (OpenTelemetry)\n"
        "  • Metrics  — latency, tokens, tok/s, tool-call count, error rate\n"
        "  • Evals    — was the answer RIGHT? (LLM-as-judge, on your own model)\n\n"
        "Tools you'll use:\n"
        "  • Arize Phoenix — open-source, self-hosted LLM-observability UI\n"
        "  • OTel GenAI conventions — the gen_ai.* span attributes\n"
        "  • NeMo Agent Toolkit (NAT) — config-driven agents with built-in tracing\n\n"
        "This app runs a smart-hotel HVAC triage agent on your DGX model (or "
        "simulates it), traces every step, evaluates the trace, then rebuilds the "
        "same agent as a NAT YAML workflow with observability piped to Phoenix — "
        "all on-prem, $0. Click through Ch 2–8.",
    },
    {"id": "step01", "group": "Trace", "kind": "run", "demo": "step01_traced_agent.py",
     "title": "Ch 2 · Trace a sovereign agent", "level": "beginner",
     "desc": "Run the HVAC triage agent and capture every step as an OpenTelemetry "
     "span. See the exact span tree Phoenix would show — agent → LLM → tool, nested, "
     "with latency and token usage on each."},
    {"id": "step02", "group": "Trace", "kind": "run", "demo": "step02_phoenix.py",
     "title": "Ch 3 · Phoenix + OTel GenAI conventions", "level": "beginner",
     "desc": "Phoenix is open-source and self-hostable — ideal for sovereign AI. "
     "See the gen_ai.* span conventions and the one-time auto-instrumentation that "
     "streams your DGX agent's spans into a Phoenix on the same box."},
    {"id": "step03", "group": "Metrics", "kind": "run", "demo": "step03_metrics.py",
     "title": "Ch 4 · The metrics that matter on a DGX", "level": "intermediate",
     "desc": "Derive the numbers you alert on: latency waterfall, time-in-LLM vs "
     "tools, tokens per task, LLM-calls per task — and how to correlate them with "
     "GPU util to decide whether to quantize, batch, or fix a loop."},
    {"id": "step04", "group": "Evals", "kind": "run", "demo": "step04_eval_traces.py",
     "title": "Ch 5 · Evaluate traces (LLM-as-judge)", "level": "intermediate",
     "desc": "Latency says fast; evals say RIGHT. Attach evaluations to spans and "
     "flag a mis-triaged dispatch. In REAL mode the judge is your own DGX model — "
     "even the grading stays sovereign."},
    {"id": "step05", "group": "NeMo Agent Toolkit", "kind": "run", "demo": "step05_nat_intro.py",
     "title": "Ch 6 · NAT — register a tool", "level": "intermediate",
     "desc": "Stop hand-rolling loops. NAT is NVIDIA's config-driven agent framework. "
     "Build the first block — a tool — via FunctionBaseConfig + @register_function + "
     "FunctionInfo, pointed at your DGX model."},
    {"id": "step06", "group": "NeMo Agent Toolkit", "kind": "run", "demo": "step06_nat_workflow.py",
     "title": "Ch 7 · NAT — a YAML workflow on your DGX", "level": "advanced",
     "desc": "Compose tools + LLM + agent into a YAML workflow whose only "
     "sovereignty knob is base_url → your DGX. Supervisor → specialist with a "
     "human-in-the-loop approval gate, then run/serve/eval from the NAT CLI."},
    {"id": "step07", "group": "NeMo Agent Toolkit", "kind": "run", "demo": "step07_nat_observability.py",
     "title": "Ch 8 · NAT observability → Phoenix", "level": "advanced",
     "desc": "Add a telemetry block and NAT exports spans to Phoenix on your DGX. "
     "The complete sovereign, observable stack — model + fine-tune + agent + traces "
     "+ evals — none of it leaving the building. Plus a production checklist."},
    {
        "id": "outro", "group": "NeMo Agent Toolkit", "kind": "concept",
        "title": "Appendix · Production observability", "level": "all levels",
        "desc": "The sovereign, observable stack you built:\n\n"
        "  DGX hardware → served model (App 1) → fine-tuned weights (App 2)\n"
        "  → NAT agent workflow (this app) → OTel spans → Phoenix on the SAME box\n"
        "  → LLM-judge evals using your OWN model\n\n"
        "Golden rules:\n"
        "  • Trace everything; keep Phoenix on the DGX/VPN — traces contain prompts.\n"
        "  • Chart p50/p95 latency, tokens/run, tool-calls/run, eval FAIL-rate.\n"
        "  • Gate CI on latency AND eval scores; alert on loop blow-ups.\n"
        "  • Correlate agent metrics with GPU util to tune serving (quantize/batch).\n\n"
        "Where this sits in Week 19:\n"
        "  App 1 run/serve · App 2 fine-tune · App 3 (THIS) observe + NAT\n"
        "  App 4 self_evolving_agent_v2 — the Week 18 agent, driven by your DGX model\n\n"
        "You can now SEE, MEASURE, and JUDGE a sovereign agent — the prerequisite for "
        "trusting one in production.",
    },
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="Observability on a DGX — interactive tutorial")
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
    models = config.list_local_models() or _SIM_MODELS
    if SELECTED["model"] not in models:
        SELECTED["model"] = models[0]
    return {"steps": [public(s) for s in STEPS], "mode": config.MODE,
            "conn": config.CONN, "conn_human": config.conn_human(),
            "model": SELECTED["model"], "base_url": config.BASE_URL, "models": models,
            "phoenix": config.PHOENIX_ENDPOINT, "phoenix_up": config.phoenix_up()}


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
    models = config.list_local_models() or _SIM_MODELS
    SELECTED["model"] = config.MODEL if config.MODEL in models else (models[0] if models else config.MODEL)
    return {"ok": True, "conn": config.CONN, "endpoint_up": config.endpoint_up(),
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
    banner = ["", "  ▣  Observability on a DGX — Phoenix + NeMo Agent Toolkit"]
    if config.MODE == "real":
        banner += [f"      ✓ REAL agent endpoint: {config.MODEL} @ {config.BASE_URL}"]
    else:
        banner += ["      ◈ SIM mode — agent loop simulated; spans rendered locally."]
    banner += [f"      Phoenix: {'up ✓ ' + config.PHOENIX_ENDPOINT if config.phoenix_up() else 'not running (tree rendered locally)'}"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set OBS_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
