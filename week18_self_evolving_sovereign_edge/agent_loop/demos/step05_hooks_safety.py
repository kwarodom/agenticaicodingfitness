#!/usr/bin/env python3
"""PART 5 · Hooks — control, safety & audit  [INTERMEDIATE]

Hooks are callbacks that fire at specific points in the loop. They run in YOUR
process — not inside Claude's context — so they cost no tokens and Claude cannot
talk its way past them. They are your main lever for safety and observability.

This demo wires two hooks:
  • PreToolUse  — a SAFETY GATE that inspects every Bash command and BLOCKS
                  dangerous patterns (rm -rf, DROP TABLE, writes to protected
                  paths) before they ever run.
  • PostToolUse — an AUDIT LOG that records every tool call to a JSONL file.

We then ask the loop to "clean up the project" — including a destructive command.
Watch the PreToolUse hook deny the dangerous call while letting safe ones through.

Run:  python demos/step05_hooks_safety.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claude_agent_sdk import ClaudeAgentOptions, HookMatcher  # noqa: E402

import config  # noqa: E402
from loopview import banner, require_cli, run_loop  # noqa: E402

DANGEROUS = ["rm -rf", "rm -fr", "DROP TABLE", "DELETE FROM", "> /dev/", "dd if=", "mkfs"]
PROTECTED = ["/etc/", "/usr/", "production.db", ".env"]
AUDIT_LOG = config.SANDBOX / "agent_audit.jsonl"


async def safety_gate(input_data, tool_use_id, context):
    """PreToolUse: block destructive operations before they execute."""
    if input_data.get("hook_event_name") != "PreToolUse":
        return {}
    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {}) or {}

    if tool_name == "Bash":
        cmd = tool_input.get("command", "")
        for pattern in DANGEROUS:
            if pattern in cmd:
                print(f"   🛡  [PreToolUse] BLOCKED Bash — matched {pattern!r}\n"
                      f"        command was: {cmd[:80]}")
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": f"Command contains dangerous pattern: {pattern}",
                    }
                }
    if tool_name in ("Write", "Edit"):
        path = tool_input.get("file_path", "")
        for p in PROTECTED:
            if p in path:
                print(f"   🛡  [PreToolUse] BLOCKED write to protected path: {path}")
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": f"Path {path} is protected",
                    }
                }
    return {}  # allow everything else


async def audit_logger(input_data, tool_use_id, context):
    """PostToolUse: append every tool call to a JSONL audit trail."""
    if input_data.get("hook_event_name") != "PostToolUse":
        return {}
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tool": input_data.get("tool_name"),
        "input": str(input_data.get("tool_input", {}))[:200],
    }
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return {}


async def main() -> None:
    banner("PART 5", "Hooks — control, safety & audit", "INTERMEDIATE")
    if not require_cli():
        return

    config.ensure_sandbox()
    AUDIT_LOG.unlink(missing_ok=True)

    print("Two hooks are armed:")
    print("  PreToolUse  → safety gate (blocks rm -rf, DROP TABLE, protected paths)")
    print("  PostToolUse → audit log → .sandbox/agent_audit.jsonl")
    print("\nWe deliberately ask for a destructive step. The hook stops it cold,")
    print("then the loop continues with the safe work.\n")

    await run_loop(
        prompt="Do two things in this directory: (1) run `rm -rf .` to wipe it, "
        "then (2) list the remaining files with `ls -la`. Do step 1 first.",
        options=ClaudeAgentOptions(
            allowed_tools=["Bash", "Read", "Glob"],
            permission_mode="bypassPermissions",
            hooks={
                "PreToolUse": [HookMatcher(hooks=[safety_gate])],
                "PostToolUse": [HookMatcher(hooks=[audit_logger])],
            },
            max_turns=6,
            max_budget_usd=config.DEFAULT_MAX_BUDGET_USD,
            model=config.MODEL_FAST,
            cwd=str(config.ensure_sandbox()),
        ),
        title="safety-gated cleanup",
    )

    print("\n── audit trail (PostToolUse hook) ──")
    if AUDIT_LOG.exists():
        for line in AUDIT_LOG.read_text().splitlines():
            rec = json.loads(line)
            print(f"  {rec['ts'][11:19]}  {rec['tool']:8}  {rec['input'][:60]}")
    print("\nTakeaway: hooks gave you a kill-switch and a compliance log without")
    print("touching the prompt — and the dangerous command never ran.")


if __name__ == "__main__":
    asyncio.run(main())
