#!/usr/bin/env python3
"""PART 3 · Turns, messages & result handling  [BEGINNER]

One *turn* is a full round trip inside the loop: Claude produces output (maybe a
tool call), the SDK runs the tool, and the result is fed back. A simple question
is 1–2 turns; a real task can be many. Only tool-use turns count toward
``max_turns`` — the final text-only reply does not.

This demo handles ALL the message types the loop emits, exactly as the tutorial
table lists them, and shows what each ResultMessage subtype means.

Run:  python demos/step02_turns_messages.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claude_agent_sdk import (  # noqa: E402
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
    UserMessage,
    ToolResultBlock,
    query,
)

import config  # noqa: E402
from loopview import banner, require_cli  # noqa: E402


async def main() -> None:
    banner("PART 3", "Turns, messages & result handling", "BEGINNER")
    if not require_cli():
        return

    print("The five message types (SystemMessage, AssistantMessage, UserMessage,")
    print("StreamEvent, ResultMessage) ARE the loop. Below we tag each one as it")
    print("arrives so you can see the round trips that make up the turns.\n")

    session_id = None
    turn = 0

    # This is the canonical 'handle every message type' pattern from the tutorial.
    async for msg in query(
        prompt="Read app.py, count how many functions (def ...) it defines, and "
        "name them. Then stop.",
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Grep", "Glob"],
            permission_mode="bypassPermissions",
            max_turns=6,
            max_budget_usd=config.DEFAULT_MAX_BUDGET_USD,
            model=config.MODEL_FAST,
            cwd=str(config.ensure_sandbox()),
        ),
    ):
        # 1. SystemMessage — capture the session id (needed to resume later).
        if isinstance(msg, SystemMessage) and msg.subtype == "init":
            session_id = msg.data.get("session_id")
            print(f"[SystemMessage/init]   session_id = {str(session_id)[:12]}…  "
                  "← save this to resume the session later")

        # 2. AssistantMessage — what Claude is doing this turn.
        elif isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    turn += 1
                    print(f"[AssistantMessage]     turn {turn}: ACT → {block.name}  "
                          f"{_brief(block.input)}")
                elif isinstance(block, TextBlock) and block.text.strip():
                    print(f"[AssistantMessage]     text → {_brief(block.text)}")

        # 3. UserMessage — the tool results returned to Claude.
        elif isinstance(msg, UserMessage):
            for block in (msg.content if isinstance(msg.content, list) else []):
                if isinstance(block, ToolResultBlock):
                    print(f"[UserMessage]          OBSERVE ← {_brief(_result_text(block))}")

        # 4. ResultMessage — the loop has stopped. Handle every subtype.
        elif isinstance(msg, ResultMessage):
            print()
            print("=" * 60)
            print(f"[ResultMessage]        subtype = {msg.subtype}")
            print(f"                       num_turns = {msg.num_turns} "
                  "(tool-use turns only)")
            print(f"                       cost = ${(msg.total_cost_usd or 0):.4f}")
            if msg.subtype == "success":
                print("                       → read msg.result and proceed:\n")
                print(_brief(msg.result, 600))
            elif msg.subtype == "error_max_turns":
                print(f"                       → resume the session: resume='{session_id}'")
            elif msg.subtype == "error_max_budget_usd":
                print("                       → raise budget or split into smaller tasks")
            else:
                print("                       → retry / check API status")
            print("=" * 60)

    print("\nTakeaway: SystemMessage gives you the session id, AssistantMessage shows")
    print("each ACT, UserMessage carries every OBSERVE, and ResultMessage tells you")
    print("how and why the loop stopped — plus what it cost.")


def _brief(value, n: int = 90) -> str:
    text = value if isinstance(value, str) else str(value)
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _result_text(block: ToolResultBlock) -> str:
    c = block.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(i.get("text", str(i)) if isinstance(i, dict) else str(i) for i in c)
    return str(c)


if __name__ == "__main__":
    asyncio.run(main())
