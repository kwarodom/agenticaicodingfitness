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

GUIDE_PORT = int(os.environ.get("GYM_GUIDE_PORT", "8109"))


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
     "title":"Ch 1 · What is verifiable-reward RL?","level":"beginner",
     "desc":"Week 23 · Tutorial 10 of 12 · Phase: Make it self-improve. NeMo Gym + NeMo RL is "
     "verifiable-reward RL: an environment = a TASK + a programmatic VERIFIER that returns a "
     "REWARD, and GRPO uses that reward to post-train a Nemotron policy. This is how Nemotron "
     "itself was forged — you run the same loop on your own box.\n\n"
     "In this tutorial:\n"
     "  • Ch 2 · What is verifiable-reward RL? — environment = task + verifier reward, why "
     "outcome-based rewards beat human preference, and the NeMo Gym vs NeMo RL split.\n"
     "  • Ch 3 · Define an environment — write a reward_fn that returns 1.0 on pass else 0.0, "
     "a deterministic verifier the model can't game.\n"
     "  • Ch 4 · GRPO training loop on the DGX — sample a group of rollouts, score each, "
     "compute group-relative advantage, update the policy; watch reward climb 0.30→0.80.\n"
     "  • Ch 5 · Multi-environment RL + evaluate — train across environments at once, then "
     "measure the trained policy's pass-rate (41%→68%) against the base model.\n\n"
     "Why it matters for sovereign agents:\n"
     "  • Outcome-based rewards (did the code/tool-call PASS?) beat human preference for "
     "coding, math, and tool-use — objective, cheap, hard to game.\n"
     "  • No reward model and no human labels — the verifier is just code that runs.\n"
     "  • The GRPO loop is simple and legible: rollout → reward → advantage → policy update.\n"
     "  • Improvements are trained IN, on-box — your DGX keeps the weights and the data.\n\n"
     "Where it fits:\n"
     "  Prerequisite: 09_inference_economics (evaluation defines the reward you optimize). "
     "Feeds 11_data_flywheel (RL is the 'customize' step of the flywheel). Next: "
     "11_data_flywheel. Together with 08 and 11 this is the 'self-evolving' engine.\n\n"
     "How to run:\n"
     "  Click Run per chapter: REAL against a Nemotron NIM/DGX endpoint via 🔌 Connection, or "
     "SIM with no GPU ($0)."},
    {"id":"step01","group":"Learn","kind":"run","demo":"step01_verifiable_reward.py",
     "title":"Ch 2 · What is verifiable-reward RL?","level":"beginner",
     "desc":"Environment = task + verifier reward. Why outcome-based rewards beat human "
     "preference for tool-use/coding/math, the NeMo Gym vs NeMo RL split, and the GRPO loop: "
     "rollout → reward → advantage → policy update."},
    {"id":"step02","group":"Define","kind":"run","demo":"step02_define_environment.py",
     "title":"Ch 3 · Define an environment","level":"intermediate",
     "desc":"A Python-ish NeMo Gym environment with a reward_fn (tool-use / calculator task) "
     "that returns 1.0 on pass else 0.0. Deterministic verifiers — the same answer always "
     "gets the same reward, so the signal can't be gamed."},
    {"id":"step03","group":"Train","kind":"run","demo":"step03_grpo_training.py",
     "title":"Ch 4 · GRPO training loop on the DGX","level":"advanced",
     "desc":"Sample a GROUP of rollouts per prompt, score each with the verifier, compute "
     "group-relative advantage, update the policy. A simulated training run over 20 rounds "
     "with reward climbing 0.30→0.80. 1-Spark vs 2-Spark (TP=2) scaling."},
    {"id":"step04","group":"Evaluate","kind":"run","demo":"step04_multienv_evaluate.py",
     "title":"Ch 5 · Multi-environment RL + evaluate","level":"advanced",
     "desc":"Train across several environments at once, then evaluate the trained policy's "
     "pass-rate (41%→68%) vs the base model. Runs the improved agent live, and ties back to "
     "the Data Flywheel (App 11) for promotion."},
    {"id":"outro","group":"Evaluate","kind":"concept",
     "title":"Appendix · The learning layer","level":"all levels",
     "desc":"NeMo Gym + NeMo RL is the LEARNING layer of Agent = Model + Harness: verifiable "
     "rewards turn agent outcomes into a better policy. NeMo RL runs GRPO on your DGX; NeMo Gym "
     "supplies the environments. Production use runs on NVIDIA AI Enterprise (bundled with DGX).\n\n"
     "Where this sits in Week 23: App 1 Nemotron (model) · App 2 NIM (serve) · "
     "App 11 Data Flywheel (improve) · App 3 Dynamo (scale) · App 10 (THIS) NeMo Gym (RL) · "
     "App 7 OpenShell (guard)."},
    {"id":"refs","group":"Go further","kind":"concept",
     "title":"Appendix · References & real-world applications","level":"all levels",
     "desc":"Curated references:\n  • NeMo RL — github.com/NVIDIA-NeMo/RL\n  • Agent RL techniques — developer.nvidia.com/blog/mastering-agentic-techniques-ai-agent-reinforcement-learning/\n  • GRPO's origin — the DeepSeekMath paper (arxiv.org/abs/2402.03300); DeepSeek-R1\n    (arxiv.org/abs/2501.12948) showed verifiable-reward RL at frontier scale.\n  • This course's eval spine — verifiable rewards are Week 10/15's 'measurable\n    success' idea, weaponized for training.\n\nReal-world applications:\n  • How reasoning models are made — R1, o-series and Nemotron reasoning variants\n    were all forged with RL against checkable rewards (math answers, passing\n    tests) rather than human preference alone.\n  • Coding agents — unit tests are nature's verifiable reward; SWE-bench-style\n    training loops (patch → run tests → reward) drive today's best repair agents.\n  • Enterprise verifiers — ticket resolved? invoice matched? SQL result equals\n    golden? Any business check you can automate becomes a training signal for\n    your domain agent.\n  • Honest limit — subjective quality (tone, judgment calls) still needs judges\n    and humans; verifiable-reward RL covers exactly what you can verify."},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="NeMo Gym + RL — interactive tutorial")
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
    banner = ["", "  ▣  NeMo Gym + NeMo RL — agents that learn from outcomes"]
    if config.MODE == "real":
        banner += [f"      ✓ REAL endpoint: {config.MODEL} @ {config.BASE_URL}",
                   "        demos run for real, fully on-device — cloud cost $0.00."]
    else:
        banner += ["      ◈ SIM mode — no endpoint reachable, simulating a DGX Spark.",
                   "        every concept is learnable with no GPU. Go REAL anytime:",
                   "        ollama run qwen3.6:35b-a3b-q8_0   (or set DGX_BASE_URL)"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set GYM_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
