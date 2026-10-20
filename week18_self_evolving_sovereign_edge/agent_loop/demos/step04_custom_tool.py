#!/usr/bin/env python3
"""PART 4 · Defining a custom tool  [INTERMEDIATE]

Built-in tools touch files and the shell. Custom tools let the loop reach YOUR
systems — a CRM, billing, Slack, an internal API. In the Claude Agent SDK you:

    1. write an async handler and wrap it with @tool(name, description, schema)
    2. bundle one or more tools into an in-process server: create_sdk_mcp_server
    3. expose it via options.mcp_servers, and allow it by its MCP name
       ("mcp__<server>__<tool>")

The tool description is what Claude reads to decide WHEN to call it — write it
like documentation. Here we expose a fake "CRM" with two tools and ask the loop
to analyse churn risk. Watch it ACT on YOUR tools, OBSERVE your data, and reason.

Run:  python demos/step04_custom_tool.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claude_agent_sdk import (  # noqa: E402
    ClaudeAgentOptions,
    create_sdk_mcp_server,
    tool,
)

import config  # noqa: E402
from loopview import banner, require_cli, run_loop  # noqa: E402

# ── a tiny fake CRM the tools read from ──────────────────────────────────────
_CUSTOMERS = {
    "acme@corp.com": {"name": "Acme Corp", "plan": "enterprise", "mrr": 2400,
                      "open_tickets": 5, "churn_risk": 0.71},
    "globex@x.com": {"name": "Globex", "plan": "pro", "mrr": 290,
                     "open_tickets": 1, "churn_risk": 0.18},
    "initech@x.com": {"name": "Initech", "plan": "free", "mrr": 0,
                      "open_tickets": 0, "churn_risk": 0.42},
}


@tool(
    "list_customers",
    "List all customers with their plan and monthly revenue (MRR). "
    "Use this first to see who exists before looking up details.",
    {},
)
async def list_customers(args):
    lines = [f"- {c['name']} <{email}> | {c['plan']} | ${c['mrr']}/mo"
             for email, c in _CUSTOMERS.items()]
    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


@tool(
    "get_customer",
    "Fetch one customer's full record by email: name, plan, MRR, open support "
    "ticket count, and churn-risk score (0–1, higher = more likely to leave). "
    "Use this to assess an individual account.",
    {"email": str},
)
async def get_customer(args):
    c = _CUSTOMERS.get(args["email"].lower())
    if not c:
        return {"content": [{"type": "text", "text": f"No customer for {args['email']}"}]}
    text = (f"Name: {c['name']}\nPlan: {c['plan']}\nMRR: ${c['mrr']}\n"
            f"Open tickets: {c['open_tickets']}\nChurn risk: {c['churn_risk']:.0%}")
    return {"content": [{"type": "text", "text": text}]}


async def main() -> None:
    banner("PART 4", "Defining a custom tool", "INTERMEDIATE")
    if not require_cli():
        return

    crm = create_sdk_mcp_server("crm", tools=[list_customers, get_customer])

    print("We registered an in-process 'crm' server with two custom tools:")
    print("  mcp__crm__list_customers   and   mcp__crm__get_customer")
    print("Claude has never seen this data — it must ACT on your tools to get it.\n")

    await run_loop(
        prompt="Which customer is most at risk of churning, and why? "
        "Look across all customers, then recommend one concrete retention action.",
        options=ClaudeAgentOptions(
            mcp_servers={"crm": crm},
            allowed_tools=["mcp__crm__list_customers", "mcp__crm__get_customer"],
            permission_mode="bypassPermissions",
            max_turns=8,
            max_budget_usd=config.DEFAULT_MAX_BUDGET_USD,
            model=config.MODEL_FAST,
        ),
        title="churn analysis over a custom CRM tool",
    )

    print("\nTakeaway: a custom tool is just an async function + a good description.")
    print("The loop decided when to call it, fed real data back in, and reasoned over it.")


if __name__ == "__main__":
    asyncio.run(main())
