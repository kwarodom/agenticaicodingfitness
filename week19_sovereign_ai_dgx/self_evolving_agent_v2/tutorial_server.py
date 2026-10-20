#!/usr/bin/env python3
"""Interactive, explainable tutorial for **Self-Evolving Agent v2 — sovereign on a DGX**.

The Week 18 self-evolving agent, with a switchable brain (local DGX model ↔ Claude
↔ sim) and its tripartite memory living on the DGX. Read the CONCEPT, view the
SOURCE, RUN each step, and watch the agent get smarter — all on your hardware.

    .venv/bin/python week19/self_evolving_agent_v2/tutorial_server.py
    # → http://127.0.0.1:8095
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
GUIDE_PORT = int(os.environ.get("SEV_GUIDE_PORT", "8095"))


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
        "title": "Ch 1 · A sovereign self-evolving agent", "level": "beginner",
        "desc": "Week 18 built a self-evolving agent: one with episodic + semantic + "
        "procedural memory that CONSOLIDATES experience over time, so each run starts "
        "smarter than the last. v2 makes it sovereign — and adds a switchable brain.\n\n"
        "Two ideas drive this app:\n\n"
        "1) The SWITCHABLE BRAIN. The same agent + memory can think with a local "
        "model on the DGX, with Claude in the cloud, or with a scripted stub — flip "
        "BRAIN=local|claude|sim. The memory engine never changes. It also makes the "
        "sovereignty cost explicit: only BRAIN=local keeps prompts on the box.\n\n"
        "2) SOVEREIGN MEMORY. A cloud agent's memory lives in a vendor's database. "
        "Here the agent's tripartite memory is files on the DGX:\n"
        "   • Episodic   → .memory/episodes.jsonl  (raw log of what happened)\n"
        "   • Semantic   → .memory/MEMORY.md       (consolidated durable facts)\n"
        "   • Procedural → .memory/skills/         (reusable skills it writes itself)\n"
        "Your agent's accumulated decisions — often your most sensitive asset — never "
        "leave the building.\n\n"
        "Run Ch 2–7 in order: log episodes → consolidate → inspect the self-written "
        "skill → prove it evolved (cold vs warm) → audit sovereignty. With no model "
        "running it uses BRAIN=sim so you can learn the whole loop at $0.",
    },
    {"id": "step01", "group": "Brain", "kind": "run", "demo": "step01_brain_switch.py",
     "title": "Ch 2 · The switchable brain", "level": "beginner",
     "desc": "The same agent runs on a local DGX model, on Claude, or on a sim stub — "
     "one env var. See which brain is active, what it costs in sovereignty, and a "
     "thought from it. The memory engine is identical for all three."},
    {"id": "step02", "group": "Memory", "kind": "run", "demo": "step02_episodic.py",
     "title": "Ch 3 · Episodic memory", "level": "beginner",
     "desc": "Run two HVAC tasks; each is logged to .memory/episodes.jsonl on the "
     "DGX. Raw, timestamped experience — the auditable record the rest builds on."},
    {"id": "step03", "group": "Memory", "kind": "run", "demo": "step03_consolidate.py",
     "title": "Ch 4 · Consolidation (the 'sleep' loop)", "level": "intermediate",
     "desc": "The brain reads recent episodes and distills durable FACTS (→ MEMORY.md) "
     "and a reusable SKILL (→ skills/). Experience becomes reusable knowledge — the "
     "subconscious step that makes the agent evolve."},
    {"id": "step04", "group": "Memory", "kind": "run", "demo": "step04_skills.py",
     "title": "Ch 5 · Procedural memory (self-written skill)", "level": "intermediate",
     "desc": "Inspect the 'hvac-triage' skill the agent wrote for itself during "
     "consolidation. An agent improving its OWN procedures — files on the DGX you can "
     "review and version — is the core of self-evolving."},
    {"id": "step05", "group": "Evolve", "kind": "run", "demo": "step05_evolve.py",
     "title": "Ch 6 · Prove it evolved — cold vs warm", "level": "advanced",
     "desc": "Run a NEW task with recall OFF (cold, like run #1) vs ON (warm, recalls "
     "consolidated facts + skill). The warm agent answers with the house's specific "
     "policy — the payoff of accumulated, on-DGX memory."},
    {"id": "step06", "group": "Trust", "kind": "run", "demo": "step06_sovereign_audit.py",
     "title": "Ch 7 · Sovereign-memory audit", "level": "advanced",
     "desc": "Audit that BOTH the brain AND the memory are on the DGX. The contrast "
     "that defines v2: a cloud agent's mind + memory live off-box; this one's don't."},
    {
        "id": "outro", "group": "Trust", "kind": "concept",
        "title": "Appendix · Week 19, in one agent", "level": "all levels",
        "desc": "This capstone ties the whole week together:\n\n"
        "  • App 1 sovereign_dgx   → the model it thinks with runs on the DGX\n"
        "  • App 2 dgx_finetune    → those weights can be tuned to your domain\n"
        "  • App 3 dgx_observability → its agent loop is traceable (Phoenix + NAT)\n"
        "  • App 4 (THIS)          → it LEARNS over time, brain + memory on the DGX\n\n"
        "Going further (the Week 18 engine has more): garbage-collect stale skills, "
        "GEPA-style prompt evolution, semantic-search recall, and a live visualizer. "
        "Swap BRAIN between local and claude to feel the architecture's brain-agnosticism.\n\n"
        "Reset anytime with 🧹 (wipes .memory/) to watch the agent grow from amnesiac "
        "to expert again.\n\n"
        "A sovereign, self-evolving agent: a model you own, tuned to your domain, that "
        "you can watch, that gets better with use — and whose mind and memories never "
        "leave your building.",
    },
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="Self-Evolving Agent v2 — interactive tutorial")
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
    import memory
    models = config.list_local_models() or _SIM_MODELS
    if SELECTED["model"] not in models:
        SELECTED["model"] = models[0]
    return {"steps": [public(s) for s in STEPS], "brain": config.BRAIN,
            "conn": config.CONN, "conn_human": config.conn_human(),
            "model": SELECTED["model"], "base_url": config.BASE_URL, "models": models,
            "memory": memory.stats()}


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
    timeout = 300.0 if config.BRAIN != "sim" else 120.0

    async def body():
        if _run_lock.locked():
            yield "⚠  another demo is already running — wait for it to finish.\n__EXIT__ 1 0\n"
            return
        async with _run_lock:
            yield f"$ BRAIN={config.BRAIN} {Path(PY).name} demos/{step['demo']}\n\n"
            async for chunk in _stream_demo(step["demo"], timeout):
                yield chunk
    return StreamingResponse(body(), media_type="text/plain")


@app.post("/api/cleanup")
async def cleanup() -> dict:
    removed = config.wipe_memory()
    for pyc in PKG.rglob("__pycache__"):
        shutil.rmtree(pyc, ignore_errors=True)
    return {"messages": [f"reset memory — removed: {removed}. The agent is amnesiac again."]}


if __name__ == "__main__":
    import uvicorn

    port = _pick_free_port(GUIDE_PORT)
    sov = {"local": "✓ sovereign (DGX)", "claude": "⚠ cloud brain",
           "sim": "· offline sim"}.get(config.BRAIN, config.BRAIN)
    banner = ["", "  ▣  Self-Evolving Agent v2 — sovereign, on a DGX",
              f"      brain: {config.BRAIN}  [{sov}]   memory: on local disk (.memory/)"]
    if config.BRAIN == "local":
        banner += [f"      model: {config.MODEL} @ {config.BASE_URL}"]
    elif config.BRAIN == "sim":
        banner += ["      no model running → BRAIN=sim. Set BRAIN=local (Ollama) or",
                   "      BRAIN=claude (ANTHROPIC_API_KEY) to think with a real model."]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set SEV_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
