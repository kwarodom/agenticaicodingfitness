#!/usr/bin/env python3
"""Shared configuration for the Week 18 agent-loop demos.

Keeps model ids and a couple of tiny helpers in one place so every demo runs
cheaply and consistently. The demos make REAL calls through the Claude Agent
SDK; they default to the fast/cheap model and small turn + budget caps so a full
run of the tutorial costs a few cents, not a few dollars.
"""
from __future__ import annotations

import os
from pathlib import Path

# Keep the agent loop CLEAN and matching the tutorial. Recent `claude` CLI
# versions can defer tools and make the model load them mid-loop via a
# `ToolSearch` step — useful in huge tool environments, but pure noise for a
# tutorial about the REASON → ACT → OBSERVE loop (it also makes the model guess
# tool params before the schema loads). We turn it off so every demo shows the
# loop exactly as the PDF describes it. (`setdefault` → you can still override.)
os.environ.setdefault("ENABLE_TOOL_SEARCH", "0")

# ── models ───────────────────────────────────────────────────────────────────
# Fast + cheap: the default for every demo (routine tutorial work).
MODEL_FAST = "claude-haiku-4-5-20251001"
# More capable: used where a demo needs stronger reasoning (e.g. a "smart" sub-agent).
MODEL_SMART = "claude-sonnet-4-6"

# ── safety rails applied across demos ────────────────────────────────────────
DEFAULT_MAX_TURNS = 8
DEFAULT_MAX_BUDGET_USD = 0.50

# ── paths ────────────────────────────────────────────────────────────────────
PKG = Path(__file__).resolve().parent          # …/week18/agent_loop
SANDBOX = PKG / ".sandbox"                      # scratch space demos may read/write
SESSION_FILE = PKG / ".sandbox" / "session.json"


_SAMPLE_FILES = {
    "app.py": (
        "def add(a, b):\n    return a + b\n\n"
        "def divide(a, b):\n    return a / b  # BUG: no zero-division guard\n\n"
        "def greet(name):\n    return f'Hello, {name}!'\n"
    ),
    "users.csv": (
        "name,plan,monthly_usd\n"
        "Acme Corp,enterprise,2400\n"
        "Globex,pro,290\n"
        "Initech,free,0\n"
    ),
    "notes.md": (
        "# Backlog\n\n"
        "- [ ] add zero-division guard to divide()\n"
        "- [ ] write tests for app.py\n"
        "- [ ] upgrade Initech from free plan\n"
    ),
}


def ensure_sandbox() -> Path:
    """Create (and return) a throwaway working directory for tool-using demos.

    Seeds a few small, realistic files the first time so demos that read/search
    files have something meaningful to work on. Safe to delete anytime.
    """
    SANDBOX.mkdir(parents=True, exist_ok=True)
    for name, body in _SAMPLE_FILES.items():
        f = SANDBOX / name
        if not f.exists():
            f.write_text(body)
    return SANDBOX
