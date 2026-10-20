#!/usr/bin/env python3
"""Shared configuration for the Week 18 **Sovereign AI at the Edge** demos.

One place for the local inference endpoint, the models to use, and a couple of
tiny helpers. Every demo in ``demos/`` makes REAL calls to a model running on
THIS machine — by default Ollama's OpenAI-compatible endpoint on
``http://localhost:11434/v1``. Nothing in this tutorial talks to a cloud model:
that is the whole point of *sovereign* AI — the compute lives where the data is.

Override with environment variables if your setup differs::

    export EDGE_BASE_URL=http://localhost:11434/v1
    export EDGE_MODEL=gemma4:12b
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.request import urlopen

# ── the sovereign endpoint (local, OpenAI-compatible) ─────────────────────────
# This is the headline of the whole tutorial: any app that speaks the OpenAI API
# works UNCHANGED against a model running on your own hardware. Swap the base_url
# and the data never leaves the building.
BASE_URL = os.environ.get("EDGE_BASE_URL", "http://localhost:11434/v1")
# Ollama ignores the key, but the OpenAI SDK requires a non-empty string.
API_KEY = os.environ.get("EDGE_API_KEY", "ollama")

# The model to drive demos with. Auto-detected at import if not pinned, so the
# tutorial just works with whatever Gemma 4 you have pulled.
_PREFERRED = ["gemma4:12b", "gemma4:latest", "gemma4", "gemma3", "llama3.2", "qwen2.5"]

# ── safety / cost rails (these are LOCAL tokens — free — but keep demos snappy) ─
DEFAULT_MAX_TOKENS = 1024      # thinking models need room to reason + answer
FAST_MAX_TOKENS = 320          # for short, direct answers

# ── paths ─────────────────────────────────────────────────────────────────────
PKG = Path(__file__).resolve().parent          # …/week18/sovereign_ai_edge
SANDBOX = PKG / ".sandbox"                      # scratch space for generated artifacts


def _native_base() -> str:
    """The Ollama *native* API root (…/api), derived from the OpenAI base_url."""
    return BASE_URL.rstrip("/").removesuffix("/v1") + "/api"


def list_local_models() -> list[str]:
    """Return model names available on the local endpoint (empty if it's down)."""
    try:
        with urlopen(_native_base() + "/tags", timeout=3) as r:
            import json
            data = json.loads(r.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def pick_model(available: list[str] | None = None) -> str:
    """Choose a model: env override → first preferred match → first available."""
    pinned = os.environ.get("EDGE_MODEL")
    if pinned:
        return pinned
    available = available if available is not None else list_local_models()
    for want in _PREFERRED:
        for have in available:
            if have == want or have.startswith(want):
                return have
    return available[0] if available else "gemma4:12b"


# Resolved once at import so every demo agrees on the model.
MODEL = pick_model()


def endpoint_up() -> bool:
    """True if the local inference endpoint answers — used for graceful fallback."""
    try:
        with urlopen(_native_base() + "/tags", timeout=3):
            return True
    except Exception:
        return False


def ensure_sandbox() -> Path:
    """Create (and return) a throwaway working dir for generated artifacts."""
    SANDBOX.mkdir(parents=True, exist_ok=True)
    return SANDBOX
