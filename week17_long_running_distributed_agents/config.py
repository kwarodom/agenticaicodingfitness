#!/usr/bin/env python3
"""Shared settings + helpers for Week 17 — Long-Running Agents & Fleet Orchestration.

Everything the six checkpoints need in common lives here so each checkpoint stays
short and self-contained:

  - WorkOrderStep        the durable state machine every checkpoint advances
  - MODEL                the Gemini model name ADK agents are wired to
  - DB_PATH / DB_URL     where durable sessions are persisted (SQLite, offline)
  - MockLLM              a deterministic stand-in so demos run with NO API key
  - have_adk()/have_a2a() capability probes (checkpoints degrade gracefully)
  - banner()/step()      tiny print helpers for readable checkpoint output

Design rule (same as week15): nothing here requires google-adk, a2a-sdk, or any
network/credentials at import time. Heavy/optional imports stay lazy inside the
checkpoint that needs them, guarded with a clear `pip install ...` message.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

# ── Model the ADK agents are wired to ──────────────────────────────────────
# The repo's other ADK demo uses gemini-2.0-flash; the source blog shows the
# newer gemini-3.1-flash-lite. Swap freely — none of these checkpoints actually
# call Gemini (they are construction + simulation demos), so the string is just
# what the constructed Agent advertises.
MODEL = "gemini-2.0-flash"

# ── Durable session storage (SQLite, lives next to this file) ───────────────
# ADK's DatabaseSessionService takes a SQLAlchemy URL. The async FastAPI server
# in the blog uses "sqlite+aiosqlite:///sessions.db"; our offline checkpoints
# use plain sqlite3 against the same file so they run with zero dependencies.
DB_PATH = Path(__file__).resolve().parent / "week17_sessions.db"
DB_URL = f"sqlite+aiosqlite:///{DB_PATH}"          # what you'd pass to ADK
SYNC_DB_URL = f"sqlite:///{DB_PATH}"               # ADK sync variant

APP_NAME = "hotel_maintenance_fleet"


# ════════════════════════════════════════════════════════════════════════════
# The long-running workflow: a hotel maintenance work-order.
#
# A guest reports a broken HVAC unit in room R305. Diagnosing it is fast, but
# the replacement compressor must be ordered from an outside vendor and can take
# *days* to arrive. The agent must pause at AWAITING_PART — possibly for a long
# weekend — and resume EXACTLY where it left off when the part is delivered,
# without hallucinating that the repair already happened.
#
# This is the week15 smart-hotel domain, now stretched across real time. It is
# the canonical "long-running agent" shape from the ADK blog (an HR onboarding
# that pauses for a document signature), mapped onto our hotel.
# ════════════════════════════════════════════════════════════════════════════
class WorkOrderStep:
    """Explicit state machine. The agent's behaviour is driven by `current_step`
    in durable session state — NOT by replaying conversation history."""

    OPEN = "OPEN"                    # work-order created, fault not yet diagnosed
    DIAGNOSED = "DIAGNOSED"          # fault identified, part may be needed
    AWAITING_PART = "AWAITING_PART"  # paused — vendor is shipping a part (days)
    PART_DELIVERED = "PART_DELIVERED"  # part arrived (set by webhook/state_delta)
    REPAIRED = "REPAIRED"            # technician completed the fix
    CLOSED = "CLOSED"                # verified + guest notified

    ORDER = [OPEN, DIAGNOSED, AWAITING_PART, PART_DELIVERED, REPAIRED, CLOSED]

    @classmethod
    def is_terminal(cls, step: str) -> bool:
        return step == cls.CLOSED


# ── Deterministic offline LLM stand-in ──────────────────────────────────────
class MockLLM:
    """A tiny rule-based stand-in for a real model so every checkpoint runs
    offline, with no GOOGLE_API_KEY/ANTHROPIC_API_KEY and no network.

    Real ADK agents call Gemini; the A2A skill's lab uses exactly this pattern
    to keep the *protocol* (cards, lifecycle, delegation) the thing under test
    rather than the model. Call it like a function: `MockLLM()(prompt)`.
    """

    def __call__(self, prompt: str) -> str:
        p = prompt.lower()
        # routing used by the A2A orchestrator (CP5/CP6)
        if "part" in p or "compressor" in p or "vendor" in p:
            return "part-fulfillment"
        if "energy" in p or "occupancy" in p or "schedule" in p:
            return "energy-scheduling"
        return "no-match"


# ── Capability probes ───────────────────────────────────────────────────────
def _importable(name: str) -> bool:
    # find_spec on a dotted name imports the parent, which raises (not returns
    # None) when the parent is absent — so guard it.
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def have_adk() -> bool:
    """True if google-adk is importable (checkpoints show the real Agent wiring
    when present and fall back to a pure-Python simulation when not)."""
    return _importable("google.adk")


def have_a2a() -> bool:
    """True if an A2A SDK (a2a-sdk or python_a2a) is importable."""
    return _importable("a2a") or _importable("python_a2a")


# ── Print helpers ───────────────────────────────────────────────────────────
def banner(title: str) -> None:
    line = "═" * 76
    print(f"\n{line}\n  {title}\n{line}")


def step(msg: str) -> None:
    print(f"  → {msg}")
