#!/usr/bin/env python3
"""Shared configuration for the auth.md × ADK example.

Three things modules here need:
  • where the AltoTech App (Part A) lives        → APP_BASE
  • where the durable ADK session store lives     → DB_URL
  • which MODEL the ADK agent (Part B) runs on     → make_model()

Plus the same tiny zero-dependency .env loader as the rest of week17, so the
repo-root ANTHROPIC_API_KEY / ALTO_LLM_API_KEY are picked up automatically.

The auth.md half (Part A + the authmd_client) is pure HTTP + crypto and needs
NO model and NO network beyond localhost — it runs fully offline. Only the ADK
agent (Part B) needs a model credential.
"""
from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "authmd_work_order"
DEFAULT_USER_ID = "energy_ops"

# ── Part A: where the agent-ready AltoTech Energy API is served ─────────────
# The app publishes its own discovery docs relative to this base. In production
# this is https://api.altotech.ai; locally it's a uvicorn process.
APP_HOST = os.environ.get("AUTHMD_APP_HOST", "127.0.0.1")
APP_PORT = int(os.environ.get("AUTHMD_APP_PORT", "8088"))
APP_BASE = os.environ.get("AUTHMD_APP_BASE", f"http://{APP_HOST}:{APP_PORT}")

# The long-running ADK agent service (Part B).
AGENT_HOST = os.environ.get("AUTHMD_AGENT_HOST", "127.0.0.1")
AGENT_PORT = int(os.environ.get("AUTHMD_AGENT_PORT", "8089"))
AGENT_BASE = os.environ.get("AUTHMD_AGENT_BASE", f"http://{AGENT_HOST}:{AGENT_PORT}")

# ── Durable session store (ADK DatabaseSessionService) ──────────────────────
# Local dev → SQLite beside this package; prod → swap for Cloud SQL. URL only.
DB_PATH = Path(__file__).resolve().parent / "work_order_sessions.db"
DB_URL = f"sqlite+aiosqlite:///{DB_PATH}"


# ── Minimal .env loader (no python-dotenv dependency) ───────────────────────
def load_env() -> None:
    """Populate os.environ from the repo-root .env for any key not already set.
    Safe to call repeatedly; never overrides an already-exported variable."""
    root = Path(__file__).resolve().parents[2]          # …/agenticaicodingfitness
    env_file = root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val


# ── Model selection (ADK is model-agnostic via LiteLlm) ─────────────────────
# Pick with AUTHMD_PROVIDER (default: the repo's AltoTech LiteLLM gateway):
#   alto       → OpenAI-compatible gateway (qwen3, llama3.3, …) — repo default.
#   anthropic  → Claude via LiteLlm (needs ANTHROPIC_API_KEY).
#   gemini     → the blog's model (needs GOOGLE_API_KEY or Vertex ADC).
ALTO_BASE_URL = os.environ.get("ALTO_LLM_BASE_URL", "https://alto-llm.altotech.ai")
ALTO_API_KEY = os.environ.get("ALTO_LLM_API_KEY", "")   # never hard-code a key: set it in .env


def make_model():
    """Return the model the ADK agents run on (a LiteLlm instance, or a model
    name string for Gemini)."""
    load_env()
    provider = os.environ.get("AUTHMD_PROVIDER", "alto").strip().lower()

    if provider == "gemini":
        return os.environ.get("AUTHMD_MODEL", "gemini-2.0-flash")

    from google.adk.models.lite_llm import LiteLlm

    if provider == "anthropic":
        return LiteLlm(model=os.environ.get("AUTHMD_MODEL", "anthropic/claude-sonnet-4-6"))

    if provider == "alto":
        name = os.environ.get("AUTHMD_MODEL", "qwen3")
        base = os.environ.get("ALTO_LLM_BASE_URL", ALTO_BASE_URL).rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        if not os.environ.get("ALTO_LLM_API_KEY", ALTO_API_KEY):
            raise RuntimeError(
                "AUTHMD_PROVIDER=alto needs ALTO_LLM_API_KEY (and ALTO_LLM_BASE_URL for your own "
                "OpenAI-compatible gateway) in .env — or set AUTHMD_PROVIDER=anthropic | gemini.")
        return LiteLlm(model=f"openai/{name}", api_base=base,
                       api_key=os.environ.get("ALTO_LLM_API_KEY", ALTO_API_KEY))

    raise RuntimeError(
        f"Unknown AUTHMD_PROVIDER={provider!r}. Use alto | anthropic | gemini "
        "— see week17/authmd_adk/README.md.")


def model_label(model) -> str:
    return getattr(model, "model", model) if not isinstance(model, str) else model
