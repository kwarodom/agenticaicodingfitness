#!/usr/bin/env python3
"""PART 1–2 · Your very first agent loop  [BEGINNER]

The smallest real agent: give Claude a goal and a couple of tools, then let the
loop run until the goal is met. Without the loop, a model answers once and
stops. With it, Claude can list files, read them, and keep going on its own.

Watch the output: you will see SESSION → ACT → OBSERVE → … → RESULT. Each ACT is
one *turn*; the loop decides by itself when it is done.

Run:  python demos/step01_hello_agent.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import loopview/config

from claude_agent_sdk import ClaudeAgentOptions  # noqa: E402

import config  # noqa: E402
from loopview import banner, require_cli, run_loop  # noqa: E402


async def main() -> None:
    banner("PART 1–2", "Your very first agent loop", "BEGINNER")
    if not require_cli():
        return

    print("Idea: ask Claude what files are here. It must ACT (use a tool) to find")
    print("out — it cannot just guess — then OBSERVE the result and answer.\n")

    await run_loop(
        prompt="What files are in the current directory? List them, then tell me "
        "Count word print inside those files, find the file with highest number of the word print",
        options=ClaudeAgentOptions(
            allowed_tools=["Bash", "Glob"],
            permission_mode="bypassPermissions",  # auto-approve in this sandbox demo
            max_turns=4,
            max_budget_usd=config.DEFAULT_MAX_BUDGET_USD,
            model=config.MODEL_FAST,
            cwd=str(config.ensure_sandbox()),
        ),
        title="hello agent",
    )

    print("\nTakeaway: the loop turned a one-shot model into something that *acts*.")
    print("It chose a tool, saw the result, and only stopped when the goal was met.")


if __name__ == "__main__":
    asyncio.run(main())
