#!/usr/bin/env python3
"""Shared configuration for the Week 19 **Sovereign AI on DGX** demos.

This tutorial runs in one of two modes, auto-detected at import:

  • REAL — an OpenAI-compatible endpoint is reachable. Either a model running on
    THIS laptop (Ollama by default at http://localhost:11434/v1) or a real DGX
    Spark / Station you point at with DGX_BASE_URL. Every call is a genuine local
    inference — nothing touches a cloud model.

  • SIM  — no endpoint is reachable. The demos fall back to a faithful SIMULATOR
    (dgxsim.py) so you can learn every DGX concept on a plane with no GPU: mock
    nvidia-smi telemetry, a realistic model registry, and token-by-token streaming
    at plausible DGX-Spark tok/s. The exact commands you'd run on real hardware
    are always printed, so SIM is a dry-run of the real thing.

Either way, **cloud cost is $0** — that is the whole point of *sovereign* AI: the
compute lives where the data is.

Override with environment variables::

    export DGX_BASE_URL=http://my-dgx-spark.local:11434/v1   # a real DGX endpoint
    export DGX_MODEL=qwen3.6:35b-a3b-q8_0                     # pin a specific model
    export DGX_MODE=sim          # force the simulator even if an endpoint is up
    export DGX_MODE=real         # never simulate (error out instead)
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

# ── the connection switch — where does the model actually live? ───────────────
# Three ways to reach a sovereign model, chosen with DGX_CONN:
#   local  → a DGX on your LAN / this laptop  (http://localhost:11434/v1)
#   tunnel → a DGX on another network, exposed over a tunnel (ngrok / cloudflared /
#            tailscale). Set DGX_TUNNEL_URL (+ DGX_API_KEY if the tunnel needs auth).
#   cloud  → a hosted OpenAI-compatible provider (Ollama Cloud, etc). Set DGX_CLOUD_URL
#            (default https://ollama.com/v1) + DGX_API_KEY.
# An explicit DGX_BASE_URL always wins and the label is inferred from its host.
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
    """Inject 'user:pass' into a URL's host (for ngrok --basic-auth tunnels)."""
    p = urlparse(url)
    netloc = p.hostname or ""
    if p.port:
        netloc += f":{p.port}"
    return p._replace(netloc=f"{auth}@{netloc}").geturl()


def apply_connection(p: dict) -> None:
    """Re-point the connection at RUNTIME from a UI request, then re-detect.

    Sets the env vars (so demo subprocesses inherit the new connection) and
    recomputes CONN/BASE_URL/API_KEY/MODE/MODEL in this process.
    """
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
    auth = (p.get("auth") or "").strip()          # "user:pass" for tunnel basic-auth
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
    else:                                          # local
        if url:
            os.environ["DGX_BASE_URL"] = url
    CONN, BASE_URL, API_KEY = _resolve_connection()
    MODE = mode()
    MODEL = pick_model()


def _open(url: str, timeout: float = 4):
    """urlopen that authenticates tunnel/cloud endpoints.

    Supports HTTP Basic via creds in the URL (https://user:pass@host) or a
    DGX_API_KEY of the form "user:pass" (e.g. ngrok --basic-auth), else Bearer.
    """
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
        url = p._replace(netloc=netloc).geturl()       # strip userinfo for urllib
    elif API_KEY and CONN != "local":
        headers["Authorization"] = f"Bearer {API_KEY}"
    if (p.hostname or "").endswith("anthropic.com") and API_KEY and ":" not in API_KEY:
        headers["x-api-key"] = API_KEY              # Anthropic uses x-api-key, not Bearer
        headers["anthropic-version"] = "2023-06-01"
    return urlopen(Request(url, headers=headers), timeout=timeout)

# Models we'd reach for on a DGX Spark, best-first. Auto-detected at import.
_PREFERRED = [
    "qwen3.6:35b-a3b-q8_0", "qwen3.6:35b-a3b-nvfp4", "qwen3.6:35b", "qwen3.6",
    "gemma4:12b", "gemma4", "llama3.3:70b", "llama3.1:8b", "qwen2.5", "phi3.5",
]

# ── cost / latency rails (LOCAL tokens are free, but keep demos snappy) ───────
DEFAULT_MAX_TOKENS = 1024
FAST_MAX_TOKENS = 320

# ── paths ─────────────────────────────────────────────────────────────────────
PKG = Path(__file__).resolve().parent          # …/week19/sovereign_dgx
SANDBOX = PKG / ".sandbox"                      # scratch space for generated artifacts

# ── DGX hardware facts (accurate, used by the hardware + sizing demos) ────────
# DGX Spark = the desk-side dev box; DGX Station = the workgroup beast.
DGX_SPECS = {
    "DGX Spark": {
        "chip": "NVIDIA GB10 Grace Blackwell Superchip",
        "memory_gb": 128,
        "memory_type": "LPDDR5X unified (CPU+GPU coherent)",
        "bandwidth_gbs": 273,
        "fp4_tops": 1000,            # ~1 PFLOP sparse FP4
        "cpu": "20-core Arm (10× Cortex-X925 + 10× A725)",
        "nic": "ConnectX-7 200GbE (link two Sparks)",
        "power_w": 240,
        "fits_params_b": 200,        # up to ~200B locally (quantized)
        "note": "Two linked Sparks → up to ~405B params.",
    },
    "DGX Station": {
        "chip": "NVIDIA GB300 Grace Blackwell Ultra",
        "memory_gb": 784,
        "memory_type": "HBM3e + LPDDR5X coherent",
        "bandwidth_gbs": 8000,       # HBM3e on the Ultra GPU
        "fp4_tops": 20000,
        "cpu": "72-core Arm Grace",
        "nic": "ConnectX-8 800GbE",
        "power_w": 1800,
        "fits_params_b": 670,
        "note": "Workgroup-class; runs 670B-class models in one box.",
    },
}


def _native_base() -> str:
    """The Ollama *native* API root (…/api), derived from the OpenAI base_url."""
    return BASE_URL.rstrip("/").removesuffix("/v1") + "/api"


def list_local_models() -> list[str]:
    """Return model names available on the live endpoint (empty if it's down)."""
    if not BASE_URL:
        return []
    import json
    # Ollama native /api/tags first; fall back to the OpenAI /v1/models shape.
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
    """True if a real OpenAI-compatible / Ollama endpoint answers (local/tunnel/cloud)."""
    if not BASE_URL:
        return False
    try:
        with _open(_native_base() + "/tags", timeout=4):
            return True
    except Exception:
        # vLLM/llama.cpp/cloud expose /v1/models instead of /api/tags — try that too.
        try:
            with _open(BASE_URL.rstrip("/") + "/models", timeout=4):
                return True
        except Exception:
            return False


def mode() -> str:
    """'real' if we should hit a live endpoint, else 'sim'. Honors DGX_MODE."""
    forced = os.environ.get("DGX_MODE", "auto").lower()
    if forced == "sim":
        return "sim"
    if forced == "real":
        return "real"
    return "real" if endpoint_up() else "sim"


def pick_model(available: list[str] | None = None) -> str:
    """Choose a model: env override → first preferred match → first available →
    a sensible DGX default name for the simulator."""
    pinned = os.environ.get("DGX_MODEL")
    if pinned:
        return pinned
    available = available if available is not None else list_local_models()
    for want in _PREFERRED:
        for have in available:
            if have == want or have.startswith(want):
                return have
    if available:
        return available[0]
    return "qwen3.6:35b-a3b-q8_0"   # the simulator's default DGX Spark model


# Resolved once at import so every demo agrees.
MODE = mode()
MODEL = pick_model()


def ensure_sandbox() -> Path:
    """Create (and return) a throwaway working dir for generated artifacts."""
    SANDBOX.mkdir(parents=True, exist_ok=True)
    return SANDBOX
