#!/usr/bin/env python3
"""PART 4 · Built-in tools & permissions  [INTERMEDIATE]

The SDK ships with every tool that powers Claude Code — Read, Write, Edit, Bash,
Glob, Grep, WebSearch, and more. Three settings decide what actually runs:

    allowed_tools     — auto-approve these, no prompt
    disallowed_tools  — these NEVER run, regardless of anything else
    permission_mode   — what happens to everything else
                        ("default" | "acceptEdits" | "plan" | "bypassPermissions")

This demo gives the loop read + search tools to investigate the sandbox code and
find the bug — but DISALLOWS Bash, proving disallowed_tools wins. You'll see the
loop chain Glob → Grep → Read across several turns.

Run:  python demos/step03_builtin_tools.py
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
    banner("PART 4", "Built-in tools & permissions", "INTERMEDIATE")
    if not require_cli():
        return

    print("Permissions for this run:")
    print("  allowed_tools    = Read, Glob, Grep   (auto-approved)")
    print("  disallowed_tools = Bash, Write, Edit   (blocked even if asked)")
    print("  permission_mode  = default             (anything else would prompt)\n")
    print("So the loop can investigate the code but cannot run or modify it —")
    print("a safe, read-only posture. Watch it chain search tools across turns.\n")

    outcome = await run_loop(
        prompt="There is a bug in app.py. Find it using only read/search tools. "
        "Tell me the function name, the line, and how to fix it. Do not edit anything.",
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Glob", "Grep"],
            disallowed_tools=["Bash", "Write", "Edit"],
            permission_mode="default",
            max_turns=8,
            max_budget_usd=config.DEFAULT_MAX_BUDGET_USD,
            model=config.MODEL_FAST,
            cwd=str(config.ensure_sandbox()),
        ),
        title="read-only bug hunt",
    )

    print(f"\nTools the loop actually used: {outcome.tools_used or '(none)'}")
    print("Notice Bash never appears — disallowed_tools always wins.")


if __name__ == "__main__":
    asyncio.run(main())
