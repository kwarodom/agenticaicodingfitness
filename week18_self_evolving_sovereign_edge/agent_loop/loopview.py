#!/usr/bin/env python3
"""Make the agent loop VISIBLE.

A thin, dependency-free wrapper over ``claude_agent_sdk.query()`` that narrates
every message the loop emits as a labelled step:

    REASON   — Claude thinks / writes text
    ACT      — Claude calls a tool   (this is what advances a *turn*)
    OBSERVE  — the tool result is fed back into the loop
    RESULT   — the loop stops and the final answer is delivered

Run it and you literally watch the REASON → ACT → OBSERVE → repeat cycle from
the tutorial turn over, with a live turn counter and the real USD cost at the
end. Every demo in ``demos/`` builds on this so the loop is never a black box.

It deliberately prints PLAIN text (no ANSI colours) so it renders cleanly both
in a terminal and inside the tutorial web app's streaming output pane.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)

# Symbols chosen to read well as plain text in a browser <pre> pane.
S_SESSION = "●"
S_REASON = "  ·"
S_THINK = "  ~"
S_ACT = "  →"
S_OBSERVE = "  ←"
S_RESULT = "═"


def _p(line: str = "") -> None:
    print(line, flush=True)


def _short(value: Any, n: int = 220) -> str:
    text = value if isinstance(value, str) else str(value)
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _tool_result_text(block: ToolResultBlock) -> str:
    content = block.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text") or item.get("tool_name") or str(item))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(content)


@dataclass
class LoopOutcome:
    """What the loop produced — handy for chaining (e.g. resuming a session)."""

    session_id: str | None = None
    subtype: str | None = None
    result: str | None = None
    num_turns: int = 0
    cost_usd: float | None = None
    tools_used: list[str] = field(default_factory=list)


async def run_loop(
    prompt: str,
    options: ClaudeAgentOptions,
    *,
    title: str | None = None,
    show_text: bool = True,
) -> LoopOutcome:
    """Run one agent loop and narrate it as REASON → ACT → OBSERVE → RESULT.

    Returns a :class:`LoopOutcome` so callers can read the session id, cost, etc.
    """
    if title:
        _p(f"┌─ {title}")
        _p(f"│  prompt: {_short(prompt, 160)}")
        _p("└" + "─" * 60)

    out = LoopOutcome()
    turn = 0

    async for msg in query(prompt=prompt, options=options):
        # ── lifecycle / session boundary ───────────────────────────────────
        if isinstance(msg, SystemMessage):
            if msg.subtype == "init":
                out.session_id = msg.data.get("session_id")
                model = msg.data.get("model", "?")
                _p(f"{S_SESSION} SESSION started  ·  model={model}  ·  "
                   f"id={str(out.session_id)[:12]}…")
                _p("  the loop is now spinning: Claude will REASON, ACT, OBSERVE, repeat\n")
            elif msg.subtype == "compact_boundary":
                _p("  (context window compacted — older turns summarised)\n")

        # ── Claude's output: reasoning, text, and tool calls ───────────────
        elif isinstance(msg, AssistantMessage):
            advanced = False
            for block in msg.content:
                if isinstance(block, ThinkingBlock):
                    _p(f"{S_THINK} REASON (thinking): {_short(block.thinking, 160)}")
                elif isinstance(block, TextBlock):
                    if show_text and block.text.strip():
                        _p(f"{S_REASON} REASON (says): {_short(block.text, 200)}")
                elif isinstance(block, ToolUseBlock):
                    advanced = True
                    out.tools_used.append(block.name)
                    _p(f"{S_ACT} ACT  [turn {turn + 1}] calls tool: {block.name}")
                    if block.input:
                        _p(f"        input: {_short(block.input, 160)}")
            if advanced:
                turn += 1

        # ── tool results fed back into the loop ────────────────────────────
        elif isinstance(msg, UserMessage):
            blocks = msg.content if isinstance(msg.content, list) else []
            for block in blocks:
                if isinstance(block, ToolResultBlock):
                    flag = " (error)" if block.is_error else ""
                    _p(f"{S_OBSERVE} OBSERVE{flag}: {_short(_tool_result_text(block), 200)}")
                    _p("        ↑ result goes back to Claude — loop continues\n")

        # ── the loop stopped ───────────────────────────────────────────────
        elif isinstance(msg, ResultMessage):
            out.subtype = msg.subtype
            out.result = msg.result
            out.num_turns = msg.num_turns
            out.cost_usd = msg.total_cost_usd
            _p(S_RESULT * 62)
            if msg.subtype == "success":
                _p(f"{S_RESULT} RESULT (success) — the loop finished on its own")
                if msg.result:
                    _p("")
                    _p(_short(msg.result, 900))
            else:
                _p(f"{S_RESULT} RESULT (stopped early): {msg.subtype}")
                _p(_explain_stop(msg.subtype, out.session_id))
            _p("")
            _p(f"  tool-use turns: {msg.num_turns}   "
               f"tools called: {len(out.tools_used)}   "
               f"cost: ${(msg.total_cost_usd or 0):.4f}")
            _p(S_RESULT * 62)

    return out


def _explain_stop(subtype: str | None, session_id: str | None) -> str:
    table = {
        "error_max_turns": (
            "  Hit the turn limit. Recommended action: resume the session with a "
            f"higher max_turns (resume='{session_id}') or split the task."
        ),
        "error_max_budget_usd": (
            "  Hit the cost cap. Recommended action: raise max_budget_usd, or split "
            "into smaller, cheaper tasks."
        ),
        "error_during_execution": (
            "  An API/execution error occurred. Recommended action: retry with "
            "exponential backoff or check API status."
        ),
    }
    return table.get(subtype or "", f"  Stopped: {subtype}")


def require_cli() -> bool:
    """Print a friendly note (and return False) if the ``claude`` CLI is missing.

    These demos make REAL calls through the SDK, which drives the local
    ``claude`` CLI. The CLI uses your existing Claude Code / subscription auth —
    so no ANTHROPIC_API_KEY is required when you're already signed in.
    """
    import shutil

    if shutil.which("claude") is None:
        _p("⚠  The `claude` CLI was not found on PATH.")
        _p("   These demos call the model for real via the Claude Agent SDK,")
        _p("   which drives the `claude` CLI. Install it and sign in:")
        _p("     npm install -g @anthropic-ai/claude-code   # then: claude (sign in)")
        _p("   (No ANTHROPIC_API_KEY needed once you're signed in.)")
        return False
    return True


def banner(part: str, title: str, level: str) -> None:
    _p("━" * 64)
    _p(f"  {part} — {title}   [{level}]")
    _p("━" * 64)
    _p("")


if __name__ == "__main__":
    _p("loopview.py is a helper imported by the demos in demos/.")
    _p("Run a demo instead, e.g.:  python demos/step01_hello_agent.py")
    sys.exit(0)
