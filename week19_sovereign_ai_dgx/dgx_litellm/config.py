#!/usr/bin/env python3
"""Config for Week 19 · App 5 — **Serving models on a DGX with LiteLLM**.

LiteLLM is a unified, OpenAI-compatible GATEWAY: one URL + one key format in
front of every backend you run on the DGX (Ollama, vLLM, llama.cpp, TensorRT-LLM,
NIM). It adds routing + load-balancing across models/Sparks, fallbacks, model
hot-swap, virtual keys with budgets/rate-limits, and logging callbacks — the
"Serving" layer of the sovereign stack, all on-prem.

Three runtime situations, auto-detected:
  • PROXY  — a real LiteLLM proxy answers at LITELLM_BASE_URL (default :4000).
    Calls go through it; routing is real.
  • DIRECT — no proxy, but a backend (Ollama/vLLM) is reachable. We illustrate the
    router/fallback logic with the simulator, but the GENERATION is a real local call.
  • SIM    — nothing reachable → fully simulated, so you learn it with no GPU, $0.

`mode()` collapses PROXY/DIRECT into "real" (generation is real) vs "sim".
Cloud cost is always $0.

Env overrides::
    export LITELLM_BASE_URL=http://localhost:4000   # the LiteLLM proxy
    export LITELLM_KEY=sk-1234                       # a virtual key
    export DGX_BASE_URL=http://localhost:11434/v1    # the backend it proxies
    export DGX_MODE=sim | real
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

# ── the LiteLLM proxy (the gateway) ───────────────────────────────────────────
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
LITELLM_KEY = os.environ.get("LITELLM_KEY", "sk-dgx-1234")


# ── connection switch for the BACKEND the proxy fronts ────────────────────────
# local DGX on the LAN · tunnel (ngrok/cloudflared) · cloud provider (Ollama Cloud).
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


# ── the backend it sits in front of (what App 1 served) ───────────────────────
CONN, BACKEND_URL, BACKEND_KEY = _resolve_connection()


def conn_human() -> str:
    return {"local": "local DGX / localhost", "tunnel": "DGX over a tunnel",
            "cloud": "cloud provider"}.get(CONN, CONN)


def safe_backend_url() -> str:
    """BACKEND_URL with any password masked, safe to print in the UI/logs."""
    p = urlparse(BACKEND_URL)
    if not p.username:
        return BACKEND_URL
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
    """Re-point the BACKEND connection at runtime from a UI request, then re-detect."""
    global CONN, BACKEND_URL, BACKEND_KEY, MODE, SITUATION, MODEL
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
    CONN, BACKEND_URL, BACKEND_KEY = _resolve_connection()
    MODE = mode()
    SITUATION = situation()
    MODEL = pick_model()


def _open(url: str, timeout: float = 4):
    """urlopen that authenticates tunnel/cloud backends (Basic via URL userinfo or
    a DGX_API_KEY of the form "user:pass", e.g. ngrok --basic-auth; else Bearer)."""
    import base64
    headers, p = {}, urlparse(url)
    user, pwd = p.username, p.password
    if user is None and BACKEND_KEY and ":" in BACKEND_KEY and CONN != "local":
        user, pwd = BACKEND_KEY.split(":", 1)
    if user is not None:
        headers["Authorization"] = "Basic " + base64.b64encode(
            f"{user}:{pwd or ''}".encode()).decode()
        netloc = p.hostname or ""
        if p.port:
            netloc += f":{p.port}"
        url = p._replace(netloc=netloc).geturl()
    elif BACKEND_KEY and CONN != "local":
        headers["Authorization"] = f"Bearer {BACKEND_KEY}"
    if (p.hostname or "").endswith("anthropic.com") and BACKEND_KEY and ":" not in BACKEND_KEY:
        headers["x-api-key"] = BACKEND_KEY          # Anthropic uses x-api-key, not Bearer
        headers["anthropic-version"] = "2023-06-01"
    return urlopen(Request(url, headers=headers), timeout=timeout)
_PREFERRED = ["qwen3.6:35b-a3b-q8_0", "qwen3.6", "gemma4:12b", "gemma4", "llama3.1:8b"]

DEFAULT_MAX_TOKENS = 400
PKG = Path(__file__).resolve().parent
SANDBOX = PKG / ".sandbox"


def _native_base() -> str:
    return BACKEND_URL.rstrip("/").removesuffix("/v1") + "/api"


def proxy_up() -> bool:
    """True if a real LiteLLM proxy answers."""
    for path in ("/health/liveliness", "/v1/models", "/"):
        try:
            with urlopen(LITELLM_BASE_URL.rstrip("/") + path, timeout=2):
                return True
        except Exception:
            continue
    return False


def backend_up() -> bool:
    if not BACKEND_URL:
        return False
    try:
        with _open(_native_base() + "/tags", timeout=4):
            return True
    except Exception:
        try:
            with _open(BACKEND_URL.rstrip("/") + "/models", timeout=4):
                return True
        except Exception:
            return False


def list_backend_models() -> list[str]:
    if not BACKEND_URL:
        return []
    import json
    try:
        with _open(_native_base() + "/tags", timeout=3) as r:
            return [m["name"] for m in json.loads(r.read().decode()).get("models", [])]
    except Exception:
        pass
    try:
        with _open(BACKEND_URL.rstrip("/") + "/models", timeout=3) as r:
            return [m["id"] for m in json.loads(r.read().decode()).get("data", [])]
    except Exception:
        return []


def pick_model(available: list[str] | None = None) -> str:
    pinned = os.environ.get("DGX_MODEL")
    if pinned:
        return pinned
    available = available if available is not None else list_backend_models()
    for want in _PREFERRED:
        for have in available:
            if have == want or have.startswith(want):
                return have
    return available[0] if available else "qwen3.6:35b-a3b-q8_0"


def mode() -> str:
    forced = os.environ.get("DGX_MODE", "auto").lower()
    if forced in ("sim", "real"):
        return forced
    return "real" if (proxy_up() or backend_up()) else "sim"


def situation() -> str:
    """proxy | direct | sim — finer-grained than mode() for the UI."""
    if os.environ.get("DGX_MODE", "").lower() == "sim":
        return "sim"
    if proxy_up():
        return "proxy"
    if backend_up():
        return "direct"
    return "sim"


MODE = mode()
SITUATION = situation()
MODEL = pick_model()


def ensure_sandbox() -> Path:
    SANDBOX.mkdir(parents=True, exist_ok=True)
    return SANDBOX
