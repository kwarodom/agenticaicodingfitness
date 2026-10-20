#!/usr/bin/env python3
"""Stack Navigator — the Week 23 hub: NVIDIA's Open Superintelligence Stack, live.

A small control plane that serves the navigator UI (static/guide.html) plus a
JSON API over the five-layer stack content (stack_content.json): hardware →
runtime → model → harness → flywheel, 16 nodes, one journey.

Two modes, auto-detected (see config.py):
  • REAL — a live OpenAI-compatible endpoint (Ollama / vLLM / NIM on this
    laptop, a DGX over a tunnel, or build.nvidia.com). Each node's demo prompt
    runs as genuine streamed inference.
  • SIM  — no endpoint reachable → each node streams its faithful canned answer
    token-by-token at plausible DGX-Spark tok/s. The real DGX Spark commands
    are part of the node content, so they are always shown either way.

Launch (auto-picks a free port if 8112 is taken):

    .venv/bin/python week23/00_stack_navigator/tutorial_server.py
    # → http://127.0.0.1:8112
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import sim  # noqa: E402

PKG = Path(__file__).resolve().parent                 # …/week23/00_stack_navigator
ROOT = PKG.parents[1]                                 # …/agenticaicodingfitness
STATIC = PKG / "static"

GUIDE_PORT = int(os.environ.get("NT_GUIDE_PORT", "8112"))


def _port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _pick_free_port(preferred: int, span: int = 40) -> int:
    for p in range(preferred, preferred + span):
        if not _port_busy(p):
            return p
    return preferred


# ── stack content, loaded once at startup ─────────────────────────────────────
STACK: dict = json.loads((PKG / "stack_content.json").read_text())

NODE_BY_ID: dict[str, dict] = {}
for _layer in STACK.get("layers", []):
    for _node in _layer.get("nodes", []):
        NODE_BY_ID[_node["id"]] = _node


def _build_apps() -> list[dict]:
    """The 12 Week 23 apps, titled from the stack nodes that map to them.

    Two node pairs share one app (NAT + AI-Q → 05; Guardrails + OpenShell → 07),
    so duplicate app mappings are folded into one entry with a joined title.
    """
    by_num: dict[str, dict] = {}
    for layer in STACK.get("layers", []):
        for node in layer.get("nodes", []):
            a = node.get("app")
            if not a:
                continue
            e = by_num.setdefault(a["num"], {
                "num": a["num"], "folder": a["folder"], "port": a["port"], "titles": []})
            if node["name"] not in e["titles"]:
                e["titles"].append(node["name"])
    return [{"num": e["num"], "folder": e["folder"], "port": e["port"],
             "title": " + ".join(e["titles"])}
            for e in sorted(by_num.values(), key=lambda x: x["num"])]


APPS = _build_apps()

app = FastAPI(title="Stack Navigator — the Week 23 hub")
_run_lock = threading.Lock()          # /api/run streams from a threadpool


# ── static UI ─────────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    guide = STATIC / "guide.html"
    if not guide.exists():
        return PlainTextResponse(
            "static/guide.html not found — the navigator UI has not been built yet.\n"
            "The JSON API is live: /api/stack /api/status /api/apps /api/probe /api/run\n",
            status_code=200)
    return FileResponse(guide, headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/static/{filename}")
async def static_file(filename: str):
    path = (STATIC / filename).resolve()
    if path.parent != STATIC.resolve() or not path.is_file():
        return PlainTextResponse("not found", status_code=404)
    return FileResponse(path, headers={"Cache-Control": "no-store, max-age=0"})


# ── stack content ─────────────────────────────────────────────────────────────
@app.get("/api/stack")
async def stack() -> dict:
    return STACK


# ── connection status + probing ───────────────────────────────────────────────
def _auth(url: str, api_key: str | None, conn: str) -> tuple[str, dict]:
    """(url, headers) with the same Bearer / basic-auth / x-api-key rules as
    config._open() — supports ngrok --basic-auth tunnels and Anthropic hosts."""
    headers: dict[str, str] = {}
    p = urlparse(url)
    user, pwd = p.username, p.password
    if user is None and api_key and ":" in api_key and conn != "local":
        user, pwd = api_key.split(":", 1)
    if user is not None:
        headers["Authorization"] = "Basic " + base64.b64encode(
            f"{user}:{pwd or ''}".encode()).decode()
        netloc = p.hostname or ""
        if p.port:
            netloc += f":{p.port}"
        url = p._replace(netloc=netloc).geturl()       # strip userinfo for urllib
    elif api_key and conn != "local":
        headers["Authorization"] = f"Bearer {api_key}"
    if (p.hostname or "").endswith("anthropic.com") and api_key and ":" not in api_key:
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    return url, headers


def _pick(models: list[str]) -> str | None:
    """First config._PREFERRED match (substring-aware), else first model."""
    for want in config._PREFERRED:
        for have in models:
            h = have.lower()
            if h == want or h.startswith(want) or want in h:
                return have
    return models[0] if models else None


def _probe_endpoint(conn: str, base_url: str | None, api_key: str | None) -> dict:
    """Probe one endpoint's /models within a ~2 s total budget. Never raises."""
    conn = (conn or config.CONN or "local").strip().lower()
    base = (base_url or "").strip()
    if not base:
        base = {
            "local":  "http://localhost:11434/v1",
            "tunnel": os.environ.get(
                "DGX_TUNNEL_URL", "http://your-spark.your-tailnet.ts.net:11434/v1"),
            "cloud":  os.environ.get(
                "DGX_CLOUD_URL", "https://integrate.api.nvidia.com/v1"),
        }.get(conn, "http://localhost:11434/v1")
    q = urlparse(base)
    if q.scheme and q.netloc and q.path in ("", "/"):
        base = base.rstrip("/") + "/v1"                # auto-append /v1
    native = base.rstrip("/").removesuffix("/v1") + "/api"

    def _openai_models() -> list[str]:
        url, headers = _auth(base.rstrip("/") + "/models", api_key, conn)
        with urlopen(Request(url, headers=headers), timeout=timeout()) as r:
            return [m["id"] for m in json.loads(r.read().decode()).get("data", [])]

    def _ollama_tags() -> list[str]:
        url, headers = _auth(native + "/tags", api_key, conn)
        with urlopen(Request(url, headers=headers), timeout=timeout()) as r:
            return [m["name"] for m in json.loads(r.read().decode()).get("models", [])]

    deadline = time.time() + 2.2

    def timeout() -> float:
        return max(0.5, min(2.0, deadline - time.time()))

    # cloud has no Ollama native API → OpenAI /models first there; else tags first.
    order = (_openai_models, _ollama_tags) if conn == "cloud" else (_ollama_tags, _openai_models)
    models, err = [], ""
    for fn in order:
        if time.time() > deadline:
            err = err or "probe budget (~2 s) exhausted"
            break
        try:
            models = fn()
            err = ""
            break
        except Exception as e:  # noqa: BLE001 — never raise to the client
            err = f"{type(e).__name__}: {e}"
    up = bool(models) or err == ""
    model = _pick(models)
    return {
        "mode": "real" if up else "sim",
        "conn": conn,
        "base_url": base,
        "model": model,
        "models": models,
        "detail": (f"✓ endpoint up · {len(models)} model(s)" if up
                   else f"⚠ not reachable — {err or 'no response'} · SIM stands in"),
    }


@app.get("/api/status")
async def status() -> dict:
    real = config.MODE == "real"
    models = await asyncio.to_thread(
        config.list_local_models) if real else sim.installed_models()
    model = _pick(models) or config.MODEL
    detail = (f"REAL · {config.conn_human()} · {len(models)} model(s) · {config.cost_note()}"
              if real else
              "SIM · no endpoint reachable — node answers stream from the simulator; "
              "real DGX Spark commands are always shown")
    return {"mode": config.MODE, "conn": config.CONN,
            "base_url": config.safe_base_url(), "model": model,
            "models": models, "detail": detail}


class ProbeRequest(BaseModel):
    conn: str = "local"
    base_url: str | None = None
    api_key: str | None = None


@app.post("/api/probe")
async def probe(req: ProbeRequest) -> dict:
    try:
        return await asyncio.to_thread(
            _probe_endpoint, req.conn, req.base_url, req.api_key)
    except Exception as e:  # noqa: BLE001 — the contract: never raises
        return {"mode": "sim", "conn": req.conn, "base_url": req.base_url or "",
                "model": None, "models": [],
                "detail": f"⚠ probe failed — {type(e).__name__}: {e}"}


# ── the 12 apps, with a liveness dot ──────────────────────────────────────────
@app.get("/api/apps")
async def apps() -> list[dict]:
    async def one(a: dict) -> dict:
        running = await asyncio.to_thread(_port_busy, a["port"])
        return {**a, "running": running}
    return list(await asyncio.gather(*(one(a) for a in APPS)))


# ── run a node: REAL streamed inference or SIM paced playback, as NDJSON ─────
class RunRequest(BaseModel):
    node_id: str
    prompt: str | None = None
    model: str | None = None
    # the 🔌 Connection panel's live choice — overrides the launch-time config
    conn: str | None = None
    base_url: str | None = None
    api_key: str | None = None


def _nd(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


def _real_chat_stream(model: str, prompt: str,
                      base: str | None = None, key: str | None = None,
                      conn: str | None = None):
    """Yield ('token'|'reason', text) from a streamed /chat/completions call."""
    url, headers = _auth((base or config.BASE_URL).rstrip("/") + "/chat/completions",
                         key if base else config.API_KEY,
                         conn or config.CONN)
    headers |= {"Content-Type": "application/json", "Accept": "text/event-stream"}
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.DEFAULT_MAX_TOKENS,
        "stream": True,
    }).encode()
    req = Request(url, data=payload, headers=headers, method="POST")
    with urlopen(req, timeout=120) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except Exception:
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            reason = delta.get("reasoning") or delta.get("reasoning_content") or ""
            text = delta.get("content") or ""
            if reason:
                yield "reason", reason
            if text:
                yield "token", text


@app.post("/api/run")
async def run_node(req: RunRequest):
    node = NODE_BY_ID.get(req.node_id)

    def gen():
        if node is None:
            yield _nd({"type": "error",
                       "detail": f"unknown node_id {req.node_id!r} — see /api/stack"})
            return
        if not _run_lock.acquire(blocking=False):
            yield _nd({"type": "error",
                       "detail": "another node is already running — wait for it to finish"})
            return
        try:
            prompt = (req.prompt or "").strip() or node.get("demo_prompt") or node["name"]
            # UI connection override: probe & connect in the browser must win over
            # whatever endpoint config.py resolved at server launch.
            base = key = None
            conn, real = config.CONN, config.MODE == "real"
            if (req.base_url or "").strip():
                base = req.base_url.strip()
                q = urlparse(base)
                if q.scheme and q.netloc and q.path in ("", "/"):
                    base = base.rstrip("/") + "/v1"    # auto-append /v1, like probe
                key = (req.api_key or "").strip() or None
                conn = (req.conn or "").strip().lower() or conn
                real = True
            model = (req.model or "").strip() or (
                config.MODEL if base else
                ((_pick(config.list_local_models()) or config.MODEL) if real else config.MODEL))
            cost = ("cloud usage billed" if conn == "cloud" else "on your box · $0.0000") \
                if base else config.cost_note()
            start = time.time()
            ntok = 0
            if real:
                yield _nd({"type": "meta", "mode": "real", "node_id": node["id"],
                           "node": node["name"], "model": model,
                           "conn": conn, "base_url": base or config.safe_base_url(),
                           "prompt": prompt, "cost_note": cost})
                deadline = start + 360.0
                try:
                    for kind, text in _real_chat_stream(model, prompt, base, key, conn):
                        ntok += 1
                        line = {"type": "token", "text": text}
                        if kind == "reason":
                            line["channel"] = "reasoning"
                        yield _nd(line)
                        if time.time() > deadline:
                            yield _nd({"type": "error",
                                       "detail": "run exceeded 360 s — stopped"})
                            return
                except Exception as e:  # noqa: BLE001 — errors become NDJSON lines
                    yield _nd({"type": "error",
                               "detail": f"{type(e).__name__}: {e} — check the 🔌 "
                                         "Connection (base_url / api key), or go SIM"})
                    return
            else:
                yield _nd({"type": "meta", "mode": "sim", "node_id": node["id"],
                           "node": node["name"], "model": model,
                           "conn": config.CONN, "base_url": config.safe_base_url(),
                           "prompt": prompt, "sim_tok_s": sim.tok_s(model),
                           "cost_note": "simulated · $0.0000"})
                for w in sim.stream_answer(node["id"], model):
                    ntok += 1
                    yield _nd({"type": "token", "text": w})
            dur = time.time() - start
            yield _nd({"type": "done", "mode": "real" if real else "sim",
                       "tokens": ntok, "seconds": round(dur, 2),
                       "tok_s": round(ntok / dur, 1) if dur > 0 else 0.0,
                       "cost_note": cost if real else "simulated · $0.0000"})
        finally:
            _run_lock.release()

    return StreamingResponse(gen(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    import uvicorn

    port = _pick_free_port(GUIDE_PORT)
    banner = ["", "  ▣  Stack Navigator — the Open Superintelligence Stack, bottom to top"]
    if config.MODE == "real":
        banner += [f"      ✓ REAL endpoint: {config.MODEL} @ {config.safe_base_url()}",
                   "        node demos run as live inference — sovereign when it's your box."]
    else:
        banner += ["      ◈ SIM mode — no endpoint reachable, simulating a DGX Spark.",
                   "        every layer is learnable with no GPU. Go REAL anytime:",
                   "        ollama run qwen3.6:35b-a3b-q8_0   (or set DGX_BASE_URL)"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set NT_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
