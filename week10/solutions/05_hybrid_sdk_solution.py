# %% [markdown]
# # Ex 5 SOLUTION — Hybrid + parallel subagents challenge
#
# Two parallel subagents (log_inspector + metric_correlator) whose outputs are
# merged by the main agent before drafting the answer.

# %%
from __future__ import annotations

import asyncio
import os
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()

SDK_AVAILABLE = bool(os.getenv("ANTHROPIC_API_KEY"))


class HybridState(TypedDict):
    ticket: str
    diagnosis: str | None
    answer: str | None


async def diagnose_with_parallel_subagents(state: HybridState) -> HybridState:
    if not SDK_AVAILABLE:
        return {"diagnosis": "[SDK skipped — no ANTHROPIC_API_KEY]"}

    from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions, query

    log_inspector = AgentDefinition(
        description="Log inspector — scans error logs for anomalies",
        prompt=(
            "You inspect server logs for error patterns. Given a ticket, "
            "hypothesize what log entries you would grep for, and what they "
            "would tell us. 60 words max."
        ),
        tools=["Read", "Grep"],
        model="sonnet",
    )

    metric_correlator = AgentDefinition(
        description="Metric correlator — identifies latency/error spikes",
        prompt=(
            "You correlate metrics (latency, error rate, saturation) around the "
            "reported incident time. Given a ticket, list the top 3 metrics to "
            "check and what each would indicate. 60 words max."
        ),
        tools=["Read"],
        model="sonnet",
    )

    options = ClaudeAgentOptions(
        system_prompt=(
            "You are a senior support engineer. When diagnosing, invoke BOTH "
            "log_inspector and metric_correlator in parallel, then synthesize a "
            "combined diagnosis."
        ),
        agents={
            "log_inspector": log_inspector,
            "metric_correlator": metric_correlator,
        },
        allowed_tools=["Agent"],
        setting_sources=["user", "project"],
        max_turns=8,
    )

    chunks: list[str] = []
    async for msg in query(
        prompt=f"Diagnose this incident using both subagents: {state['ticket']}",
        options=options,
    ):
        if hasattr(msg, "content"):
            chunks.append(str(msg.content))
    return {"diagnosis": "\n".join(chunks) if chunks else "[no diagnosis]"}


def draft_answer(state: HybridState) -> HybridState:
    return {
        "answer": (
            "Hello — our combined diagnosis:\n"
            f"{state['diagnosis']}\n\n"
            "We'll follow up within 1 business hour."
        )
    }


g = StateGraph(HybridState)
g.add_node("diagnose", diagnose_with_parallel_subagents)
g.add_node("draft", draft_answer)
g.add_edge(START, "diagnose")
g.add_edge("diagnose", "draft")
g.add_edge("draft", END)
app = g.compile()


async def main() -> None:
    result = await app.ainvoke({
        "ticket": "API is 500-ing since 7am and the dashboard shows no data.",
        "diagnosis": None,
        "answer": None,
    })
    print("\n=== DIAGNOSIS ===")
    print(result["diagnosis"])
    print("\n=== ANSWER ===")
    print(result["answer"])


if __name__ == "__main__":
    asyncio.run(main())
