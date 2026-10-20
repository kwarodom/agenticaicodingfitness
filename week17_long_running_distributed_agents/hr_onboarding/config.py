#!/usr/bin/env python3
"""Shared configuration for the HR onboarding agent.

Two things every module needs: which MODEL to run the agent on, and WHERE the
durable session store lives. Plus a tiny zero-dependency .env loader so the
repo-root ANTHROPIC_API_KEY is picked up without adding python-dotenv.
"""
from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "hr_onboarding"
DEFAULT_USER_ID = "hr_coordinator"

# ── Durable session store (ADK DatabaseSessionService) ──────────────────────
# Local dev → SQLite file beside this package; production → swap the URL for
# Cloud SQL (e.g. "postgresql+asyncpg://…/onboarding"). Only the URL changes.
DB_PATH = Path(__file__).resolve().parent / "onboarding_sessions.db"
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
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val


# ── Model selection ─────────────────────────────────────────────────────────
# ADK defaults to Gemini, but it's model-agnostic via LiteLlm, so the agents can
# run on whichever endpoint this environment actually has. Pick with
# ONBOARDING_PROVIDER (default: the repo's AltoTech LiteLLM gateway):
#
#   alto       → OpenAI-compatible gateway at ALTO_LLM_BASE_URL (open models:
#                qwen3, llama3.3, nemotron, …). Model via ONBOARDING_MODEL
#                (default 'qwen3'). This is the repo's standard LLM access.
#   anthropic  → Claude via LiteLlm (needs a valid ANTHROPIC_API_KEY).
#   gemini     → the blog's model (needs GOOGLE_API_KEY or Vertex ADC).
ALTO_BASE_URL = os.environ.get("ALTO_LLM_BASE_URL", "https://alto-llm.altotech.ai")
ALTO_API_KEY = os.environ.get("ALTO_LLM_API_KEY", "")   # never hard-code a key: set it in .env


def make_model():
    """Return the model the ADK agents run on (a LiteLlm instance or, for
    Gemini, a model-name string)."""
    load_env()
    provider = os.environ.get("ONBOARDING_PROVIDER", "alto").strip().lower()

    if provider == "gemini":
        return os.environ.get("ONBOARDING_MODEL", "gemini-2.0-flash")

    from google.adk.models.lite_llm import LiteLlm

    if provider == "anthropic":
        model = os.environ.get("ONBOARDING_MODEL", "anthropic/claude-sonnet-4-6")
        return LiteLlm(model=model)

    if provider == "alto":
        # OpenAI-compatible proxy → litellm 'openai/<model>' + api_base + api_key.
        name = os.environ.get("ONBOARDING_MODEL", "qwen3")
        base = os.environ.get("ALTO_LLM_BASE_URL", ALTO_BASE_URL).rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        if not os.environ.get("ALTO_LLM_API_KEY", ALTO_API_KEY):
            raise RuntimeError(
                "ONBOARDING_PROVIDER=alto needs ALTO_LLM_API_KEY (and ALTO_LLM_BASE_URL for your own "
                "OpenAI-compatible gateway) in .env — or set ONBOARDING_PROVIDER=anthropic | gemini.")
        return LiteLlm(model=f"openai/{name}",
                       api_base=base,
                       api_key=os.environ.get("ALTO_LLM_API_KEY", ALTO_API_KEY))

    raise RuntimeError(
        f"Unknown ONBOARDING_PROVIDER={provider!r}. Use alto | anthropic | gemini "
        "— see week17/hr_onboarding/README.md.")


def model_label(model) -> str:
    """Human-readable name for logs."""
    return getattr(model, "model", model) if not isinstance(model, str) else model
