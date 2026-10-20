#!/usr/bin/env python3
"""PART 9 · Production patterns — bounded execution & cost control  [ADVANCED]

Unconstrained loops can run away — too many turns, runaway cost ("denial of
wallet"). Production loops always set hard caps and handle the stop reason.

This demo INTENTIONALLY sets a tiny budget so the loop hits its ceiling, proving
the guardrail fires and that you can recover gracefully. The key options:

    max_turns        — hard cap on tool-use round trips (prevents infinite loops)
    max_budget_usd   — hard cap on spend; loop stops with subtype error_max_budget_usd
    effort           — reasoning depth vs cost ("low" | "medium" | "high" | ...)

We give the loop a big task but a $0.02 budget. It will stop early — and we
detect that, explain it, and show how a real system would resume or split.

Run:  python demos/step09_production.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claude_agent_sdk import ClaudeAgentOptions  # noqa: E402

import config  # noqa: E402
from loopview import banner, require_cli, run_loop  # noqa: E402


async def main() -> None:
    banner("PART 9", "Production patterns — bounded execution & cost", "ADVANCED")
    if not require_cli():
        return

    print("Guardrails for this run (deliberately strict to trip the limit):")
    print("  max_turns      = 3")
    print("  max_budget_usd = 0.02     ← tiny on purpose")
    print("  effort         = low      ← cheapest reasoning for routine work\n")
    print("We hand it a deep task it CAN'T finish that cheaply. Watch the loop stop")
    print("itself at the budget ceiling instead of running away.\n")

    outcome = await run_loop(
        prompt="Read every file in this directory, then write a thorough, "
        "multi-section engineering audit covering bugs, style, test coverage, "
        "and a refactor plan with code examples for each file.",
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Glob", "Grep"],
            permission_mode="bypassPermissions",
            max_turns=3,
            max_budget_usd=0.02,
            effort="low",
            model=config.MODEL_FAST,
            cwd=str(config.ensure_sandbox()),
        ),
        title="bounded execution under a strict budget",
    )

    print("\n── how a production system reacts to the stop reason ──")
    reactions = {
        "error_max_budget_usd": "Budget hit → raise max_budget_usd, or split into "
        "smaller per-file tasks and aggregate. Alert if this recurs.",
        "error_max_turns": "Turn limit hit → resume the session with a higher "
        f"max_turns (resume='{outcome.session_id}').",
        "success": "Finished within budget — nothing to recover. Lower the cap "
        "further only if you want a tighter guarantee.",
    }
    print("  " + reactions.get(outcome.subtype or "", f"Stopped: {outcome.subtype}"))
    print(f"\n  (this run cost ${ (outcome.cost_usd or 0):.4f}; "
          f"stop reason = {outcome.subtype})")
    print("\nTakeaway: max_turns + max_budget_usd turn an open-ended loop into a")
    print("loop you can deploy. Always cap, always handle the stop subtype.")


if __name__ == "__main__":
    asyncio.run(main())
