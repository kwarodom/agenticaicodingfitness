#!/usr/bin/env python3
"""PART 6 · Sessions & 24/7 continuity  [INTERMEDIATE]

Every agent run creates a session — a persistent conversation with its own
context history. Capture the session id and you can RESUME later to pick up
exactly where you left off. This is the foundation of long-running and
perpetual-monitoring agents.

This demo runs the loop TWICE:
  Run 1 — start fresh, learn a fact, save the session id to disk.
  Run 2 — resume that session and ask a follow-up that only works if Claude
          still remembers Run 1. (No re-explaining — the context carried over.)

Run:  python demos/step06_sessions.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claude_agent_sdk import ClaudeAgentOptions  # noqa: E402

import config  # noqa: E402
from loopview import banner, require_cli, run_loop  # noqa: E402


def _save(session_id: str) -> None:
    config.ensure_sandbox()
    config.SESSION_FILE.write_text(json.dumps({"session_id": session_id}))


def _load() -> str | None:
    if config.SESSION_FILE.exists():
        return json.loads(config.SESSION_FILE.read_text()).get("session_id")
    return None


async def main() -> None:
    banner("PART 6", "Sessions & 24/7 continuity", "INTERMEDIATE")
    if not require_cli():
        return

    # ── Run 1: establish context, then persist the session id ───────────────
    print(">>> RUN 1 — fresh session. We tell Claude a secret project codename.\n")
    out1 = await run_loop(
        prompt="Remember this: our Q3 launch is internally codenamed 'Bluefin', "
        "targeting enterprise customers. Just acknowledge in one line.",
        options=ClaudeAgentOptions(
            max_turns=2,
            max_budget_usd=config.DEFAULT_MAX_BUDGET_USD,
            model=config.MODEL_FAST,
        ),
        title="run 1 — establish context",
    )

    if not out1.session_id:
        print("No session id captured — cannot demo resume.")
        return
    _save(out1.session_id)
    print(f"\n✔ saved session id to {config.SESSION_FILE.name} → {out1.session_id[:12]}…")

    # ── Run 2: resume and rely on memory from Run 1 ─────────────────────────
    sid = _load()
    print(f"\n>>> RUN 2 — resume session {sid[:12]}… and ask a follow-up.")
    print("    If resume works, Claude answers WITHOUT us repeating the codename.\n")
    await run_loop(
        prompt="What is our Q3 launch codename, and who is it targeting? "
        "Answer from what you already know.",
        options=ClaudeAgentOptions(
            resume=sid,
            max_turns=2,
            max_budget_usd=config.DEFAULT_MAX_BUDGET_USD,
            model=config.MODEL_FAST,
        ),
        title="run 2 — resumed session",
    )

    print("\nTakeaway: the second loop never had the codename in its prompt — it")
    print("recalled it from the resumed session. Persist the id (a file, Redis, a")
    print("DB row) and an agent can pause for days and wake up with full context.")
    print("A perpetual monitor is just this resume loop on a timer (see the PDF's")
    print("'24/7 perpetual monitoring loop').")


if __name__ == "__main__":
    asyncio.run(main())
