#!/usr/bin/env python3
"""PART 7 · Multi-agent orchestration  [ADVANCED]

When a task is large or has independent parts, the main loop can spawn
SUBAGENTS — separate Claude instances with their own fresh context, focused
prompt, and minimal tool set. Benefits: context isolation, parallelism,
specialisation, and cost control (cheaper models for routine subtasks).

In the Claude Agent SDK you declare subagents with ``AgentDefinition`` and pass
them via ``options.agents``. The main agent invokes them with the built-in
``Agent`` tool (so "Agent" must be in allowed_tools).

Here the main loop reviews the sandbox code by delegating to two specialists —
a code-reviewer and a security-scanner — then synthesises one report. Watch the
ACT lines call the ``Agent`` tool to hand work to each subagent.

Run:  python demos/step07_multi_agent.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions  # noqa: E402

import config  # noqa: E402
from loopview import banner, require_cli, run_loop  # noqa: E402


async def main() -> None:
    banner("PART 7", "Multi-agent orchestration", "ADVANCED")
    if not require_cli():
        return

    print("Main agent = orchestrator. Two specialist subagents do the work:")
    print("  • code-reviewer  — style, clarity, maintainability  (fast model)")
    print("  • bug-hunter     — correctness bugs & edge cases     (fast model)")
    print("Each runs in its own fresh context; the main loop synthesises.\n")

    agents = {
        "code-reviewer": AgentDefinition(
            description="Reviews code for style, naming, and maintainability. "
            "Use for general code-quality feedback.",
            prompt="You are a senior engineer doing a code-quality review. "
            "Check naming, clarity, error handling, and documentation. "
            "Cite file:line for each point. Be concise.",
            tools=["Read", "Grep", "Glob"],
            model="haiku",
        ),
        "bug-hunter": AgentDefinition(
            description="Finds correctness bugs and unhandled edge cases. "
            "Use to catch runtime errors before they ship.",
            prompt="You are a bug hunter. Find correctness bugs and unhandled "
            "edge cases (e.g. division by zero, None handling). For each: the "
            "file:line, why it breaks, and the fix. Be specific.",
            tools=["Read", "Grep", "Glob"],
            model="haiku",
        ),
    }

    await run_loop(
        prompt="Review the code in this directory. Use the code-reviewer subagent "
        "and the bug-hunter subagent, then synthesise their findings into a single "
        "prioritised report (most important issue first). Keep it short.",
        options=ClaudeAgentOptions(
            agents=agents,
            allowed_tools=["Agent", "Read", "Grep", "Glob"],
            permission_mode="bypassPermissions",
            max_turns=12,
            max_budget_usd=1.00,  # multi-agent runs cost a bit more
            model=config.MODEL_FAST,
            cwd=str(config.ensure_sandbox()),
        ),
        title="orchestrate two specialist subagents",
    )

    print("\nTakeaway: the orchestrator delegated to focused subagents (each with a")
    print("fresh context and its own tools) and merged the results. That is how you")
    print("scale a loop to big tasks without blowing one context window.")


if __name__ == "__main__":
    asyncio.run(main())
