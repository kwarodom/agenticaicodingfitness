#!/usr/bin/env python3
"""Config for Week 19 **Self-Evolving Agent v2 — sovereign, on a DGX**.

Week 18 built a self-evolving agent (episodic + semantic + procedural memory that
consolidates over time). v2 changes one profound thing: the agent now runs on a
DGX — and so does its MEMORY. Your agent's accumulated decisions and domain
knowledge (often your most sensitive asset) never leave the building.

The headline feature is the **switchable brain**: the SAME agent + memory engine
can be driven by a local sovereign model (DGX / Ollama) or by Claude (cloud) —
flip one env var. This proves the memory architecture is brain-agnostic.

    export BRAIN=local     # the DGX model (OpenAI-compatible) — sovereign default
    export BRAIN=claude    # Anthropic Claude (needs ANTHROPIC_API_KEY)
    export BRAIN=auto       # local if an endpoint is up, else claude, else sim
    export BRAIN=sim        # no model at all — scripted, for $0 offline learning

Other overrides:  DGX_BASE_URL · DGX_MODEL · CLAUDE_MODEL
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

# ── connection switch (for BRAIN=local): local DGX · tunnel · cloud provider ──
# Independent of the BRAIN switch: BRAIN picks WHO thinks (local model vs Claude),
# DGX_CONN picks HOW you reach the local model (LAN / tunnel / cloud Ollama).
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
    """Re-point the local-brain connection at runtime, then re-detect."""
    global CONN, BASE_URL, API_KEY, BRAIN, MODEL
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
    BRAIN = brain()
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
_PREFERRED = ["qwen3.6:35b-a3b-q8_0", "qwen3.6", "gemma4:12b", "gemma4", "llama3.1:8b"]

# ── the cloud brain (for the switchable-brain demo) ───────────────────────────
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

DEFAULT_MAX_TOKENS = 512

# ── memory lives on-prem (a throwaway dir for the tutorial) ───────────────────
PKG = Path(__file__).resolve().parent
MEMORY_DIR = PKG / ".memory"
EPISODES = MEMORY_DIR / "episodes.jsonl"        # episodic: append-only event log
SEMANTIC = MEMORY_DIR / "MEMORY.md"             # semantic: durable consolidated facts
SKILLS_DIR = MEMORY_DIR / "skills"              # procedural: reusable skills


def _native_base() -> str:
    return BASE_URL.rstrip("/").removesuffix("/v1") + "/api"


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


def _anthropic_ready() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except Exception:
        return False


def brain() -> str:
    """Resolve which brain drives the agent: local | claude | sim."""
    want = os.environ.get("BRAIN", "auto").lower()
    if want in ("local", "claude", "sim"):
        return want
    # auto
    if endpoint_up():
        return "local"
    if _anthropic_ready():
        return "claude"
    return "sim"


BRAIN = brain()
MODEL = pick_model()


def ensure_memory() -> Path:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    return MEMORY_DIR


def wipe_memory() -> list[str]:
    import shutil
    removed = []
    if MEMORY_DIR.exists():
        shutil.rmtree(MEMORY_DIR, ignore_errors=True)
        removed.append(".memory/")
    return removed or ["nothing (already clean)"]
