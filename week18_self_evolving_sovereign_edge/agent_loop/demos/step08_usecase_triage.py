#!/usr/bin/env python3
"""PART 8 · Real-world use case — Customer Support Triage Agent  [INTERMEDIATE]

The payoff. This is a production-shaped agent loop straight from the tutorial's
use cases: a SaaS startup gets 50–200 support tickets/day; reading, classifying,
and replying to L1s ate 3–4 founder-hours daily. This loop does it end-to-end.

It uses two CUSTOM tools wired to (fake) support systems:
  • list_new_tickets — fetch unassigned tickets from the last N hours
  • send_ticket_reply — post a reply and set status (resolved/escalated)

…and a system_prompt that encodes the triage workflow (L1 → reply & resolve,
L2/L3 → escalate). Watch one loop read every ticket, classify it, and either
resolve or escalate it — the whole queue cleared in one run.

Run:  python demos/step08_usecase_triage.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, tool  # noqa: E402

import config  # noqa: E402
from loopview import banner, require_cli, run_loop  # noqa: E402

# ── a fake support inbox (mutated by the tools as the loop works) ────────────
_TICKETS = [
    {"id": "T-101", "subject": "How do I export my data to CSV?",
     "body": "I need to download all my records as a spreadsheet. Where is that?",
     "status": "new"},
    {"id": "T-102", "subject": "Reset password link expired",
     "body": "The reset email link says expired. Can you send a new one?",
     "status": "new"},
    {"id": "T-103", "subject": "URGENT: production data missing after sync",
     "body": "Half our customer records vanished after this morning's sync. "
             "This is affecting live operations right now.",
     "status": "new"},
    {"id": "T-104", "subject": "Webhook returns 500 intermittently",
     "body": "About 1 in 10 webhook deliveries fail with HTTP 500 on your side.",
     "status": "new"},
]


@tool(
    "list_new_tickets",
    "List unassigned support tickets that still need triage. Returns each "
    "ticket's id, subject, and body. Call this first to see the queue.",
    {},
)
async def list_new_tickets(args):
    new = [t for t in _TICKETS if t["status"] == "new"]
    if not new:
        return {"content": [{"type": "text", "text": "No new tickets."}]}
    text = "\n\n".join(f"[{t['id']}] {t['subject']}\n{t['body']}" for t in new)
    return {"content": [{"type": "text", "text": text}]}


@tool(
    "send_ticket_reply",
    "Send a reply to a support ticket and set its status. "
    "For L1 (common, documented) issues set status='resolved'. "
    "For L2/L3 (engineering/critical) set status='escalated'. "
    "Args: ticket_id, reply (markdown), status (resolved|escalated), "
    "priority (low|normal|high|urgent).",
    {"ticket_id": str, "reply": str, "status": str, "priority": str},
)
async def send_ticket_reply(args):
    for t in _TICKETS:
        if t["id"] == args["ticket_id"]:
            t["status"] = args["status"]
            return {"content": [{"type": "text", "text":
                    f"{t['id']} → {args['status']} (priority={args.get('priority')}). "
                    f"Reply sent ({len(args['reply'])} chars)."}]}
    return {"content": [{"type": "text", "text": f"Unknown ticket {args['ticket_id']}"}]}


TRIAGE_PROMPT = """You are a customer-support specialist for a B2B SaaS product.

For each ticket:
1. Read the subject and body carefully.
2. Classify: L1 (common / documented), L2 (needs engineering), L3 (critical / data loss).
3. L1  → write a warm, helpful reply, status='resolved', priority='normal'.
4. L2  → write an acknowledgement, status='escalated', priority='high'.
5. L3  → write an urgent acknowledgement, status='escalated', priority='urgent'.

Tone: professional, empathetic, solution-focused. Process the WHOLE queue."""


async def main() -> None:
    banner("PART 8", "Real-world use case — Support Triage Agent", "INTERMEDIATE")
    if not require_cli():
        return

    inbox = create_sdk_mcp_server("support", tools=[list_new_tickets, send_ticket_reply])

    print("A real ops loop: 4 tickets in the queue (one is a critical data-loss")
    print("incident). The loop will read them, classify each, and resolve or")
    print("escalate appropriately — the kind of work that ate hours of founder time.\n")

    await run_loop(
        prompt="Process all new support tickets. Triage each one, then reply: "
        "resolve the L1s and escalate L2/L3 with the right priority.",
        options=ClaudeAgentOptions(
            mcp_servers={"support": inbox},
            allowed_tools=["mcp__support__list_new_tickets", "mcp__support__send_ticket_reply"],
            system_prompt=TRIAGE_PROMPT,
            permission_mode="bypassPermissions",
            max_turns=20,
            max_budget_usd=1.00,
            model=config.MODEL_FAST,
        ),
        title="clear the support queue",
        show_text=False,  # keep the focus on the ACT/OBSERVE loop, not long replies
    )

    print("\n── final inbox state ──")
    for t in _TICKETS:
        print(f"  {t['id']}  {t['status']:9}  {t['subject'][:48]}")
    print("\nTakeaway: one loop cleared the queue — resolving routine tickets and")
    print("escalating the critical incident — with custom tools + a workflow prompt.")
    print("Schedule this every 30 min (cron/queue) and L1 support runs itself.")


if __name__ == "__main__":
    asyncio.run(main())
