#!/usr/bin/env python3
"""Interactive, explainable tutorial for **Fine-tuning on a DGX**.

Serves a clickable web guide (static/guide.html). For each chapter you read the
CONCEPT, view the demo SOURCE, and RUN it. The training-loop demo is simulated
(you can't train a 70B on a laptop); dataset prep, recipe generation, and the
before/after eval run for real (eval uses a live model if one is reachable).

    .venv/bin/python week19/dgx_finetune/tutorial_server.py
    # → http://127.0.0.1:8093
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
GUIDE_PORT = int(os.environ.get("FT_GUIDE_PORT", "8093"))


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
        "title": "Ch 1 · Why fine-tune on a DGX", "level": "beginner",
        "desc": "Prompting and RAG steer a model from the OUTSIDE. Fine-tuning changes "
        "the WEIGHTS — it bakes your domain's vocabulary, format, and judgement into "
        "the model itself. On a DGX the whole cycle is sovereign: your proprietary "
        "training data never leaves the building.\n\n"
        "When to fine-tune (vs prompt / RAG):\n"
        "  • You need a consistent STYLE/FORMAT every time      → fine-tune\n"
        "  • The model must know niche domain facts/jargon       → fine-tune (or RAG)\n"
        "  • Facts change daily / need citations                → RAG, not fine-tune\n"
        "  • One-off behaviour tweak                            → just prompt\n\n"
        "The sovereign fine-tuning loop (this whole app):\n"
        "  data (on-prem) → recipe → train on DGX → eval → merge → GGUF/NVFP4 → serve\n\n"
        "This app EXECUTES a real fine-tune end to end with Unsloth:\n"
        "  💻 laptop: build dataset + script   →   🖥️ DGX (over SSH): push, train, serve\n"
        "  →   💻 laptop: evaluate base vs your tuned model\n\n"
        "Steps marked 🖥️ run real commands ON your DGX over SSH and stream the output "
        "here. First open the 🔌 DGX SSH panel (top-right) and enter your DGX host / "
        "user / key, the HuggingFace model to fine-tune, and (if gated) an HF token.\n\n"
        "Techniques: LoRA / QLoRA (4-bit base — fits big models on one Spark) via "
        "Unsloth (~2× faster, exports straight to GGUF for Ollama). Run STEPS 1→6 in order.",
    },
    {"id": "step01", "group": "Laptop", "kind": "run", "demo": "step01_dataset_prep.py",
     "title": "STEP 1 · 💻 Build dataset + training script", "level": "beginner",
     "desc": "Runs on your LAPTOP. Writes a real instruction dataset (train/val JSONL) "
     "and the actual Unsloth training script into .sandbox/. Your data stays local "
     "until you push it to YOUR DGX in STEP 2."},
    {"id": "step02", "group": "DGX (SSH)", "kind": "run", "demo": "step02_connect_push.py",
     "title": "STEP 2 · 💻→🖥️ Connect & push to the DGX", "level": "beginner",
     "desc": "REAL SSH. Tests the connection to your DGX (and that the GPU is visible), "
     "then scp's the dataset + script to the remote workdir. Configure SSH in the "
     "🔌 DGX SSH panel first."},
    {"id": "step03", "group": "DGX (SSH)", "kind": "run", "demo": "step03_prepare_dgx.py",
     "title": "STEP 3 · 🖥️ Prepare the DGX (container + GPU)", "level": "intermediate",
     "desc": "REAL SSH. Pulls NVIDIA's PyTorch container (correct CUDA for Blackwell) "
     "and reports free GPU memory so the real training starts fast."},
    {"id": "step04", "group": "DGX (SSH)", "kind": "run", "demo": "step04_train.py",
     "title": "STEP 4 · 🖥️ Run the REAL fine-tune (Unsloth)", "level": "advanced",
     "desc": "REAL SSH — the actual training. Installs Unsloth, loads YOUR model in "
     "4-bit, trains a LoRA on the HVAC dataset, exports a Q4_K_M GGUF. The live loss "
     "curve streams here. Can take a while (model download + steps)."},
    {"id": "step05", "group": "DGX (SSH)", "kind": "run", "demo": "step05_serve.py",
     "title": "STEP 5 · 🖥️ Serve the tuned model (Ollama)", "level": "advanced",
     "desc": "REAL SSH. Registers the GGUF as an Ollama model ('hvac-assistant') on the "
     "DGX and starts it — reachable on the same OpenAI API and selectable in this app's "
     "model dropdown."},
    {"id": "step06", "group": "Laptop", "kind": "run", "demo": "step06_evaluate.py",
     "title": "STEP 6 · 💻 Evaluate — base vs tuned", "level": "advanced",
     "desc": "Runs on your LAPTOP against the connection. Asks the same held-out "
     "questions to the BASE model and your TUNED 'hvac-assistant' and shows the "
     "difference — the real proof the fine-tune worked."},
    {
        "id": "outro", "group": "Laptop", "kind": "concept",
        "title": "Appendix · The full real loop", "level": "all levels",
        "desc": "What you actually executed:\n\n"
        "  💻 dataset+script → 🖥️ push (scp) → 🖥️ container → 🖥️ Unsloth train → 🖥️ GGUF\n"
        "  → 🖥️ ollama serve → 💻 eval base vs tuned\n\n"
        "…all on hardware you control; your data + weights never left the building.\n\n"
        "Knobs (🔌 DGX SSH panel): HF model, HF token (gated models), max-steps, and the "
        "SSH host/user/key. Smaller models train in minutes; big models take longer + "
        "need free VRAM (stop other models with `ollama stop`).\n\n"
        "Where this sits in Week 19:\n"
        "  App 1 run/serve · App 2 (THIS) fine-tune for real · App 3 observe · App 4 self-evolving\n\n"
        "Your domain-tuned model is now served on the DGX — pick 'hvac-assistant' in any "
        "app's model dropdown.",
    },
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="Fine-tuning on a DGX — interactive tutorial")
_run_lock = asyncio.Lock()

SELECTED = {"model": config.MODEL}
_SIM_MODELS = ["qwen3.6:35b-a3b-q8_0", "llama3.3:70b", "gemma4:12b", "llama3.1:8b"]

# ── persistence: remember connection + SSH across reloads AND restarts ────────
import json  # noqa: E402
SETTINGS_FILE = PKG / ".settings.json"
CONN_INPUTS = {"conn": "local", "url": "", "auth": ""}   # raw inputs for the UI to pre-fill


def _strip_userinfo(url: str) -> str:
    """Drop any user:pass@ credentials from a URL before persisting it."""
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        if p.username:
            netloc = p.hostname or ""
            if p.port:
                netloc += f":{p.port}"
            return p._replace(netloc=netloc).geturl()
    except Exception:
        pass
    return url


def _save_settings() -> None:
    # Persist ONLY non-secret metadata. Secrets — HF token, connection password /
    # ngrok basic-auth — are NOT written to disk; they live in memory for the
    # session and are re-entered after a server restart (explicit, not silent).
    ssh_keys = {"host": "FT_SSH_HOST", "user": "FT_SSH_USER", "port": "FT_SSH_PORT",
                "key": "FT_SSH_KEY", "workdir": "FT_WORKDIR", "hf_model": "FT_HF_MODEL"}
    data = {"conn_inputs": {"conn": CONN_INPUTS.get("conn", "local"),
                            "url": _strip_userinfo(CONN_INPUTS.get("url", ""))},
            "ssh": {k: os.environ.get(env, "") for k, env in ssh_keys.items()}}
    try:
        SETTINGS_FILE.write_text(json.dumps(data, indent=2))
        os.chmod(SETTINGS_FILE, 0o600)          # owner-only, defense in depth
    except Exception:
        pass


def _load_settings() -> None:
    if not SETTINGS_FILE.exists():
        return
    try:
        data = json.loads(SETTINGS_FILE.read_text())
    except Exception:
        return
    ci = data.get("conn_inputs")
    if ci and ci.get("conn") and ci.get("conn") != "local":
        CONN_INPUTS.update(ci)
        try:
            config.apply_connection(ci)
        except Exception:
            pass
    ssh = data.get("ssh") or {}
    if ssh.get("host") and ssh.get("user"):
        try:
            config.apply_ssh(ssh)
        except Exception:
            pass


_load_settings()


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
            "base_model": config.BASE_MODEL, "domain": config.DOMAIN,
            "ssh": config.ssh_status(), "conn_inputs": CONN_INPUTS}


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
    CONN_INPUTS.update({"conn": req.conn or "local", "url": (req.url or ""), "auth": (req.auth or "")})
    _save_settings()
    models = config.list_local_models() or _SIM_MODELS
    SELECTED["model"] = config.MODEL if config.MODEL in models else (models[0] if models else config.MODEL)
    return {"ok": True, "conn": config.CONN, "endpoint_up": config.endpoint_up(),
            "model": SELECTED["model"], "models": models}


class SSHRequest(BaseModel):
    host: str | None = None
    user: str | None = None
    port: str | None = None
    key: str | None = None
    workdir: str | None = None
    hf_model: str | None = None
    hf_token: str | None = None


@app.post("/api/ssh")
async def set_ssh(req: SSHRequest) -> dict:
    """Save the DGX SSH + fine-tune params (used by the 🖥️ steps). Persists to disk."""
    config.apply_ssh(req.model_dump())
    _save_settings()
    return {"ok": True, "ssh": config.ssh_status()}


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
    # 🖥️ DGX steps run real remote work — give them room.
    timeout = {"step02": 240.0, "step03": 1800.0, "step04": 5400.0,
               "step05": 900.0, "step06": 300.0}.get(req.step_id, 120.0)

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
    banner = ["", "  ▣  Fine-tuning on a DGX — interactive, explainable tutorial"]
    if config.MODE == "real":
        banner += [f"      ✓ REAL eval endpoint: {config.MODEL} @ {config.BASE_URL}",
                   "        (training loop is simulated unless run on a real DGX.)"]
    else:
        banner += ["      ◈ SIM mode — eval + training simulated, no GPU needed.",
                   "        go REAL for eval:  ollama run qwen3.6:35b-a3b-q8_0"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set FT_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
