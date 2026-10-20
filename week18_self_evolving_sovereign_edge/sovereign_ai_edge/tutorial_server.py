#!/usr/bin/env python3
"""Interactive, explainable tutorial for **Sovereign AI at the Edge**.

A small control plane that serves a clickable web guide (static/guide.html) and,
for each chapter of the tutorial, lets you:

  • read the CONCEPT (the why, distilled from the Week 18 PDF);
  • view the exact SOURCE of the demo that chapter runs;
  • RUN that demo for real — every call goes to a model running on THIS machine
    (Ollama's OpenAI-compatible endpoint by default) — and watch it stream live:
    LOCAL ✓ → PROMPT → REASON → ANSWER → tok/s, or ACT → OBSERVE for agents.

These are REAL local inferences, not mocks. Nothing here touches a cloud model —
that is the whole point of sovereign AI. Tokens are free (it's your hardware), so
a full click-through costs $0.00.

Launch (auto-picks a free port if 8091 is taken):

    .venv/bin/python week18/sovereign_ai_edge/tutorial_server.py
    # → http://127.0.0.1:8091
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

PKG = Path(__file__).resolve().parent                 # …/week18/sovereign_ai_edge
ROOT = PKG.parents[1]                                 # …/agenticaicodingfitness
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable
DEMOS = PKG / "demos"

GUIDE_PORT = int(os.environ.get("SOVEREIGN_GUIDE_PORT", "8091"))


def _port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _pick_free_port(preferred: int, span: int = 40) -> int:
    for p in range(preferred, preferred + span):
        if not _port_busy(p):
            return p
    return preferred


# ── the tutorial chapters ──────────────────────────────────────────────────────
# Each "run" step maps to a demos/stepNN_*.py file; concept steps just explain.
STEPS = [
    {
        "id": "intro", "group": "Foundations", "kind": "concept",
        "title": "Ch 1 · What is Sovereign AI?", "level": "beginner",
        "desc": "Sovereign AI means complete control over the AI supply chain: the "
        "hardware you compute on, the data you train and infer with, the models you "
        "run, and the software that ties them together. Nothing touches an external "
        "server unless you explicitly choose it.\n\n"
        "Edge AI takes it one step further: the compute lives at the SOURCE of the "
        "data — on a desk, a factory floor, a drone, a phone. That buys sub-5ms "
        "local latency, fully offline operation, and privacy by physical design.\n\n"
        "Cloud AI vs Sovereign Edge AI:\n"
        "  • Data privacy   — leaves your perimeter   →  never moves off-device\n"
        "  • Latency        — 50–200 ms round-trip    →  <5 ms local inference\n"
        "  • Uptime         — internet required       →  fully offline capable\n"
        "  • Cost           — per-token API fees       →  one-time hardware\n"
        "  • Compliance     — data-residency rules     →  guaranteed by physics\n"
        "  • Transparency   — black box                →  open weights, full audit\n\n"
        "The 5-layer sovereign stack — every layer on-prem, all open source today:\n"
        "  Hardware → DGX Spark, Jetson Orin, Raspberry Pi, Mac, phone\n"
        "  Models   → Gemma 4, Llama 4, Qwen 3.5, Mistral, DeepSeek\n"
        "  Runtime  → Ollama, vLLM, llama.cpp, LiteRT-LM, TensorRT-LLM\n"
        "  Serving  → LiteLLM, llama-swap, OpenAI-compatible proxy\n"
        "  App      → NeMo Agent Toolkit, LangChain, OpenClaw\n\n"
        "Everything in this tutorial runs that stack for real on this machine. Click "
        "through Chapters 2–10 and watch sovereign AI actually work — no cloud, $0.",
    },
    {
        "id": "step01", "group": "Foundations", "kind": "run", "demo": "step01_local_inference.py",
        "title": "Ch 2 · Your first sovereign inference", "level": "beginner",
        "desc": "The simplest proof of sovereignty: a real model running on THIS "
        "machine answers a prompt that never crossed the network. It's a drop-in "
        "OpenAI call — the only change from a cloud call is the base_url pointing at "
        "localhost. Watch LOCAL ✓ → PROMPT → REASON (the model thinks, on-device) → "
        "ANSWER stream out, with measured tok/s and a cloud cost of $0.0000.",
    },
    {
        "id": "step02", "group": "Hardware & models", "kind": "run", "demo": "step02_hardware_advisor.py",
        "title": "Ch 3 · Edge hardware advisor", "level": "beginner",
        "desc": "There's a spectrum of edge hardware from an $80 Raspberry Pi to a "
        "$4000 DGX Spark, and the deciding factor is memory bandwidth — LLM decode is "
        "memory-bound, not TOPS-bound. This demo encodes the PDF's decision matrices "
        "as real logic: give it a RAM budget and it returns the platform, the best "
        "Gemma 4 variant, the quantization, and the expected tok/s — then asks the "
        "local model to justify a pick.",
    },
    {
        "id": "step03", "group": "Hardware & models", "kind": "run", "demo": "step03_model_explorer.py",
        "title": "Ch 4 · Gemma 4 family + local bake-off", "level": "beginner",
        "desc": "Gemma 4 spans a sub-1.5 GB phone model to a 31B dense model, in two "
        "architectures: Dense (all params active, highest quality) and MoE (only "
        "3.8B active/token, ~6× faster decode). After the variant + benchmark tables, "
        "this runs the SAME prompt through EVERY Gemma you actually have pulled — a "
        "real on-device bake-off comparing answer quality and tokens/sec.",
    },
    {
        "id": "step04", "group": "Hardware & models", "kind": "run", "demo": "step04_quant_calculator.py",
        "title": "Ch 5 · Quantization & memory math", "level": "intermediate",
        "desc": "Raw BF16 weights are too big for the edge; quantization shrinks them "
        "to 4/8-bit with <5% quality loss. The math is simple: VRAM ≈ params × "
        "bytes/param × overhead. This demo prints the format table (BF16→FP8→NVFP4→"
        "Q4_K_M→2-bit), computes each Gemma size in each format, and then INSPECTS "
        "the running model to print its TRUE quantization — straight from the engine.",
    },
    {
        "id": "step05", "group": "Training", "kind": "run", "demo": "step05_finetune_recipe.py",
        "title": "Ch 6 · Fine-tuning on the edge (LoRA)", "level": "advanced",
        "desc": "Fine-tuning adapts a base model to YOUR domain without sending data "
        "to a cloud trainer (45–90 min for a 31B LoRA on a DGX Spark). Since you "
        "can't train 31B on a Mac, this demo (1) generates the real NeMo AutoModel "
        "YAML + SFT dataset into .sandbox/, and (2) demonstrates the EFFECT of domain "
        "adaptation locally — the same HVAC question with vs without a domain system "
        "prompt, showing what LoRA bakes into the weights.",
    },
    {
        "id": "step06", "group": "Serving", "kind": "run", "demo": "step06_serving_runtimes.py",
        "title": "Ch 7 · Inference & serving (the local API)", "level": "intermediate",
        "desc": "A model is useful once apps can call it. The sovereign stack exposes "
        "the SAME OpenAI-compatible API as the cloud, served from localhost — so every "
        "app works unchanged. Ollama (quick start), vLLM (throughput), LiteLLM (route "
        "+ hot-swap). This makes a real streaming call and measures the two numbers "
        "that matter: time-to-first-token and decode throughput.",
    },
    {
        "id": "step07", "group": "Agents", "kind": "run", "demo": "step07_tool_calling_agent.py",
        "title": "Ch 8 · A sovereign agent (function calling)", "level": "intermediate",
        "desc": "An agent ACTs on your systems, not just talks. Gemma 4 has native "
        "function calling: give it tool schemas and it emits structured JSON calls. "
        "The whole loop — REASON → ACT (call your tool) → OBSERVE → repeat — runs on "
        "your hardware. This wires two real Python functions (read a sensor, dispatch "
        "maintenance) and watches the local model diagnose a fault and take action.",
    },
    {
        "id": "step08", "group": "Real-world", "kind": "run", "demo": "step08_smart_hotel_mas.py",
        "title": "Ch 9 · Real-world: Smart-Hotel HVAC agent", "level": "advanced",
        "desc": "The payoff. A 500-room hotel runs Gemma 4 on an on-site DGX Spark; "
        "guest telemetry never leaves the property (compliance by physical design). "
        "One local agent clears the morning alarm queue: it lists alarms, pulls "
        "equipment status, and dispatches maintenance — CRITICAL for the "
        "guest-impacting room, routine for a dirty filter — all on-device.",
    },
    {
        "id": "step09", "group": "Real-world", "kind": "run", "demo": "step09_airgap_audit.py",
        "title": "Ch 10 · Security & air-gap audit", "level": "advanced",
        "desc": "'Sovereign' is a claim — this turns it into a checklist you can "
        "VERIFY. It audits the running setup live: the endpoint is loopback (not "
        "remote), no cloud credentials are set, the models are local weights (it "
        "flags any 'cloud' passthrough variant), and a real inference round-trips "
        "with zero external hops — then prints the hardening checklist. A real "
        "sovereignty audit you could run in CI.",
    },
    {
        "id": "outro", "group": "Real-world", "kind": "concept",
        "title": "Appendix · Quick-reference & deploy", "level": "all levels",
        "desc": "One-command quickstart by platform:\n"
        "  • DGX Spark / Ubuntu GPU — curl -fsSL https://ollama.com/install.sh | sh "
        "&& ollama run gemma4:26b\n"
        "  • Raspberry Pi 5 — …install.sh | sh && ollama run gemma4:2b\n"
        "  • Mac (Apple Silicon) — brew install ollama && ollama run gemma4:26b\n"
        "  • Phone (Android) — install 'Google AI Edge Gallery' → Agent Skills\n"
        "  • Jetson Orin — ./run.sh $(./autotag ollama); ollama run gemma4:4b\n"
        "  • Cross-platform — pip install litert-lm && litert-lm run gemma4-e2b\n\n"
        "Model selection by memory:\n"
        "  <2 GB → E2B (2-bit)   ·  8 GB → E4B (Q4_K_M)  ·  16 GB → 26B MoE (Q4_K_M)\n"
        "  24 GB → 31B (Q4_K_M)  ·  128 GB → 31B (NVFP4/BF16, 23–65 tok/s)\n\n"
        "Deploy checklist: pick the model that fits your RAM, quantize (NVFP4 on "
        "Blackwell, Q4_K_M elsewhere), serve via Ollama/vLLM behind localhost, add "
        "tools for the agentic layer, and run the Ch-10 audit before you trust it.\n\n"
        "You started with one local 'hello world' and ended with an audited, "
        "agentic, on-prem system — the same skills scale from a Raspberry Pi to a "
        "DGX Spark cluster, and not a byte left the building.",
    },
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="Sovereign AI at the Edge — interactive tutorial")
_run_lock = asyncio.Lock()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(PKG / "static" / "guide.html")


@app.get("/api/steps")
async def steps() -> dict:
    def public(s):
        return {k: s.get(k) for k in ("id", "group", "title", "desc", "kind", "level")} | \
               {"demo": s.get("demo")}
    up = config.endpoint_up()
    return {"steps": [public(s) for s in STEPS], "endpoint_ready": up,
            "model": config.MODEL if up else None,
            "base_url": config.BASE_URL,
            "models": config.list_local_models() if up else []}


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
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
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

    if not config.endpoint_up():
        async def noendpoint():
            yield (f"⚠  No local inference endpoint at {config.BASE_URL}\n"
                   "   Sovereign AI runs the model on YOUR machine. Start one:\n"
                   "     curl -fsSL https://ollama.com/install.sh | sh\n"
                   "     ollama run gemma4:12b   (or gemma4:2b on low-RAM devices)\n"
                   "__EXIT__ 1 0\n")
        return StreamingResponse(noendpoint(), media_type="text/plain")

    # Multi-round agent + multi-model bake-off do more work → allow longer.
    timeout = 480.0 if req.step_id in ("step03", "step08") else 300.0

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
    up = config.endpoint_up()
    banner = ["", "  ▣  Sovereign AI at the Edge — interactive, explainable tutorial"]
    if not up:
        banner += [f"      ⚠  no local endpoint at {config.BASE_URL} — demos show start hints.",
                   "         start one:  ollama run gemma4:12b"]
    else:
        banner += [f"      ✓ local model ready: {config.MODEL} @ {config.BASE_URL}",
                   "        demos run for real, fully on-device — cloud cost $0.00."]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} "
                   "(set SOVEREIGN_GUIDE_PORT to choose)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
