#!/usr/bin/env python3
"""Shared config for Week 19 **Fine-tuning on a DGX** (domain adaptation).

Fine-tuning bakes YOUR domain knowledge into open weights — without sending a
single training example to a cloud trainer. You can't train a 70B on a laptop, so
the heavy steps (the training loop) run in a faithful SIMULATOR (ftsim.py) that
streams a realistic loss curve, throughput, and checkpoints. The lighter steps
(dataset prep, recipe generation, the before/after EVAL) run for real — and the
eval uses a live model if one is reachable.

Two modes, auto-detected (same as App 1):
  • REAL — a live OpenAI-compatible endpoint (Ollama / vLLM on this box or a DGX).
  • SIM  — no endpoint → simulate. Real commands are always printed.

Cloud cost is always $0.

Env overrides::
    export DGX_BASE_URL=http://my-dgx-spark.local:11434/v1
    export DGX_MODEL=qwen3.6:35b-a3b-q8_0
    export DGX_MODE=sim | real
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


# ── connection switch: local DGX · tunnel (ngrok/cloudflared) · cloud provider ─
def _resolve_connection() -> tuple[str, str, str]:
    conn = os.environ.get("DGX_CONN", "").strip().lower()
    explicit = os.environ.get("DGX_BASE_URL") or os.environ.get("EDGE_BASE_URL")
    key = os.environ.get("DGX_API_KEY") or os.environ.get("EDGE_API_KEY")

    def _infer(url: str) -> str:
        h = (urlparse(url).hostname or "").lower()
        if h in ("localhost", "127.0.0.1", "::1") or h.endswith(".local") \
                or h.startswith(("192.168.", "10.", "172.")):
            return "local"
        if any(t in h for t in ("ngrok", "trycloudflare", "loca.lt", "ts.net", "tunnel")):
            return "tunnel"
        return "cloud"

    if explicit:
        return (conn or _infer(explicit)), explicit, (key or "dgx")
    if conn == "tunnel":
        return "tunnel", os.environ.get("DGX_TUNNEL_URL", ""), (key or os.environ.get("DGX_TUNNEL_KEY", "dgx"))
    if conn == "cloud":
        return "cloud", os.environ.get("DGX_CLOUD_URL", "https://ollama.com/v1"), (key or os.environ.get("DGX_CLOUD_KEY", ""))
    return "local", "http://localhost:11434/v1", (key or "dgx")


CONN, BASE_URL, API_KEY = _resolve_connection()


def conn_human() -> str:
    return {"local": "local DGX / localhost", "tunnel": "DGX over a tunnel",
            "cloud": "cloud provider"}.get(CONN, CONN)


def safe_base_url() -> str:
    """BASE_URL with any password masked, safe to print in the UI/logs."""
    p = urlparse(BASE_URL)
    if not p.username:
        return BASE_URL
    netloc = p.hostname or ""
    if p.port:
        netloc += f":{p.port}"
    return p._replace(netloc=f"{p.username}:***@{netloc}").geturl()


def _with_userinfo(url: str, auth: str) -> str:
    p = urlparse(url)
    netloc = p.hostname or ""
    if p.port:
        netloc += f":{p.port}"
    return p._replace(netloc=f"{auth}@{netloc}").geturl()


def apply_connection(p: dict) -> None:
    """Re-point the connection at runtime from a UI request, then re-detect."""
    global CONN, BASE_URL, API_KEY, MODE, MODEL
    conn = (p.get("conn") or "local").lower()
    for k in ("DGX_CONN", "DGX_BASE_URL", "DGX_TUNNEL_URL", "DGX_CLOUD_URL",
              "DGX_API_KEY", "EDGE_BASE_URL", "EDGE_API_KEY"):
        os.environ.pop(k, None)
    os.environ["DGX_CONN"] = conn
    def _norm(u):
        from urllib.parse import urlparse, urlunparse
        if not u:
            return u
        try:
            q = urlparse(u)
            if q.scheme and q.netloc and q.path in ("", "/"):
                q = q._replace(path="/v1")   # auto-append /v1 if the user omitted it
            return urlunparse(q)
        except Exception:
            return u
    url = _norm((p.get("url") or "").strip())
    key = (p.get("key") or "").strip()
    auth = (p.get("auth") or "").strip()
    if conn == "tunnel":
        if auth and url and "@" not in url.split("//", 1)[-1]:
            url = _with_userinfo(url, auth)
        if url:
            os.environ["DGX_TUNNEL_URL"] = url
        if key:
            os.environ["DGX_API_KEY"] = key
    elif conn == "cloud":
        if url:
            os.environ["DGX_CLOUD_URL"] = url
        if key:
            os.environ["DGX_API_KEY"] = key
    else:
        if url:
            os.environ["DGX_BASE_URL"] = url
    CONN, BASE_URL, API_KEY = _resolve_connection()
    MODE = mode()
    MODEL = pick_model()


def _open(url: str, timeout: float = 4):
    """urlopen that authenticates tunnel/cloud endpoints (Basic via URL userinfo or
    a DGX_API_KEY of the form "user:pass", e.g. ngrok --basic-auth; else Bearer)."""
    import base64
    headers, p = {}, urlparse(url)
    user, pwd = p.username, p.password
    if user is None and API_KEY and ":" in API_KEY and CONN != "local":
        user, pwd = API_KEY.split(":", 1)
    if user is not None:
        headers["Authorization"] = "Basic " + base64.b64encode(
            f"{user}:{pwd or ''}".encode()).decode()
        netloc = p.hostname or ""
        if p.port:
            netloc += f":{p.port}"
        url = p._replace(netloc=netloc).geturl()
    elif API_KEY and CONN != "local":
        headers["Authorization"] = f"Bearer {API_KEY}"
    if (p.hostname or "").endswith("anthropic.com") and API_KEY and ":" not in API_KEY:
        headers["x-api-key"] = API_KEY              # Anthropic uses x-api-key, not Bearer
        headers["anthropic-version"] = "2023-06-01"
    return urlopen(Request(url, headers=headers), timeout=timeout)

_PREFERRED = [
    "qwen3.6:35b-a3b-q8_0", "qwen3.6", "gemma4:12b", "gemma4", "llama3.1:8b", "qwen2.5",
]

DEFAULT_MAX_TOKENS = 700
PKG = Path(__file__).resolve().parent          # …/week19/dgx_finetune
SANDBOX = PKG / ".sandbox"                      # generated datasets / recipes go here

# The base model we'll pretend to (or really) fine-tune, and the domain we adapt to.
BASE_MODEL = os.environ.get("FT_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DOMAIN = "smart-hotel HVAC operations"          # ties back to the Week 18 hotel demo


def _native_base() -> str:
    return BASE_URL.rstrip("/").removesuffix("/v1") + "/api"


def list_local_models() -> list[str]:
    if not BASE_URL:
        return []
    import json
    try:
        with _open(_native_base() + "/tags", timeout=3) as r:
            return [m["name"] for m in json.loads(r.read().decode()).get("models", [])]
    except Exception:
        pass
    try:
        with _open(BASE_URL.rstrip("/") + "/models", timeout=3) as r:
            return [m["id"] for m in json.loads(r.read().decode()).get("data", [])]
    except Exception:
        return []


def endpoint_up() -> bool:
    if not BASE_URL:
        return False
    try:
        with _open(_native_base() + "/tags", timeout=4):
            return True
    except Exception:
        try:
            with _open(BASE_URL.rstrip("/") + "/models", timeout=4):
                return True
        except Exception:
            return False


def mode() -> str:
    forced = os.environ.get("DGX_MODE", "auto").lower()
    if forced in ("sim", "real"):
        return forced
    return "real" if endpoint_up() else "sim"


def pick_model(available: list[str] | None = None) -> str:
    pinned = os.environ.get("DGX_MODEL")
    if pinned:
        return pinned
    available = available if available is not None else list_local_models()
    for want in _PREFERRED:
        for have in available:
            if have == want or have.startswith(want):
                return have
    return available[0] if available else "qwen3.6:35b-a3b-q8_0"


MODE = mode()
MODEL = pick_model()


def ensure_sandbox() -> Path:
    SANDBOX.mkdir(parents=True, exist_ok=True)
    return SANDBOX


# ── DGX SSH target for REAL fine-tuning (set from the UI panel) ───────────────
def ssh_status() -> dict:
    host = os.environ.get("FT_SSH_HOST", "")
    user = os.environ.get("FT_SSH_USER", "")
    return {
        "configured": bool(host and user),
        "host": host, "user": user,
        "port": os.environ.get("FT_SSH_PORT", ""),
        "workdir": os.environ.get("FT_WORKDIR", "~/dgx_finetune_demo"),
        "hf_model": os.environ.get("FT_HF_MODEL", BASE_MODEL),
        "has_token": bool(os.environ.get("FT_HF_TOKEN")),
    }


def apply_ssh(p: dict) -> None:
    """Persist SSH + fine-tune params to env so demo subprocesses inherit them."""
    def setenv(k: str, v) -> None:
        v = (v or "").strip()
        if v:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)
    setenv("FT_SSH_HOST", p.get("host"))
    setenv("FT_SSH_USER", p.get("user"))
    setenv("FT_SSH_PORT", p.get("port"))
    setenv("FT_SSH_KEY", p.get("key"))
    setenv("FT_WORKDIR", p.get("workdir"))
    setenv("FT_HF_MODEL", p.get("hf_model"))
    setenv("FT_HF_TOKEN", p.get("hf_token"))
