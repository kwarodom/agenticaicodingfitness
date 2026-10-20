# %% [markdown]
# # Ex 5: Hybrid LangGraph node powered by Claude Agent SDK
#
# **Goal:** demonstrate the April 2026 Anthropic stack:
# 1. **Claude Agent SDK** inside a LangGraph node
# 2. an **Agent Skill** (structured specialized knowledge)
# 3. a **subagent** spawned for research-then-act
#
# Preview: the same agent definition shape runs on **Claude Managed Agents**
# (launched April 8, 2026: public beta, no waitlist).
#
# **Time:** 20 minutes.
#
# **Note:** requires `ANTHROPIC_API_KEY`. If you don't have one, watch along:
# the fallback prints a stub so the graph still runs.
#
# **⚠️ Operator warning.** If you have Claude Code installed with private MCP
# servers (internal DBs, CRM, HR, etc.), the SDK's subprocess may enumerate
# those tool names in its output even with `setting_sources=[]`. Do not
# live-demo this exercise on a laptop that has sensitive MCPs configured.
# Use a clean machine, a fresh user profile, or run from a terminal with
# `env -i` to strip inherited context.

# %%
from __future__ import annotations

import asyncio
import os
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()

SDK_AVAILABLE = bool(os.getenv("ANTHROPIC_API_KEY"))


# %% [markdown]
# ## State

# %%
class HybridState(TypedDict):
    ticket: str
    diagnosis: str | None
    answer: str | None


# %% [markdown]
# ## LangGraph node that calls Claude Agent SDK

# %%
async def diagnose_with_sdk(state: HybridState) -> HybridState:
    if not SDK_AVAILABLE:
        return {"diagnosis": "[SDK skipped: no ANTHROPIC_API_KEY; watch demo]"}

    from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions, query

    # A subagent: isolates context for focused research.
    researcher = AgentDefinition(
        description="Research specialist: investigates before acting",
        prompt=(
            "You investigate technical issues by breaking them into components. "
            "Return a numbered diagnosis in under 80 words."
        ),
        tools=["Read", "Grep"],
        model="sonnet",
    )

    # ISOLATION: pass setting_sources=[] (empty list) to opt out of all host
    # Claude Code config, including user-level MCP servers. If you omit this,
    # the SDK's subprocess picks up your local MCPs and can leak private data.
    # If you still see host data in the output, run from a terminal outside
    # your Claude Code workspace, or use `env -i` to strip inherited env vars.
    options = ClaudeAgentOptions(
        system_prompt="You are a senior support engineer.",
        agents={"researcher": researcher},
        allowed_tools=["Agent"],  # main agent can only spawn the researcher subagent
        setting_sources=[],       # opt out of user/project config (no MCPs, no Skills)
        max_turns=5,
    )

    # Collect only final human-readable TextBlock content from the main agent.
    # Skips ThinkingBlock / ToolUseBlock / ToolResultBlock reprs for clean output.
    texts: list[str] = []
    async for msg in query(
        prompt=f"Use the researcher subagent to diagnose: {state['ticket']}",
        options=options,
    ):
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for block in content:
                if type(block).__name__ == "TextBlock":
                    text = getattr(block, "text", None)
                    if text:
                        texts.append(text)
        elif isinstance(content, str) and content.strip():
            texts.append(content)
    return {"diagnosis": "\n\n".join(texts) if texts else "[no diagnosis]"}


def draft_answer(state: HybridState) -> HybridState:
    return {
        "answer": (
            "Hello, our diagnosis:\n"
            f"{state['diagnosis']}\n\n"
            "We'll follow up within 1 business hour with a fix."
        )
    }


# %% [markdown]
# ## Build graph

# %%
g = StateGraph(HybridState)
g.add_node("diagnose", diagnose_with_sdk)
g.add_node("draft", draft_answer)
g.add_edge(START, "diagnose")
g.add_edge("diagnose", "draft")
g.add_edge("draft", END)
app = g.compile()


async def main() -> None:
    result = await app.ainvoke({
        "ticket": "My API is returning 500 errors since 7am and the dashboard shows no data.",
        "diagnosis": None,
        "answer": None,
    })
    print("\n=== DIAGNOSIS ===")
    print(result["diagnosis"])
    print("\n=== ANSWER ===")
    print(result["answer"])


if __name__ == "__main__":
    asyncio.run(main())


# %% [markdown]
# ## Discussion points
#
# 1. **Why put the SDK in a LangGraph node?** Separation of concerns: LangGraph
#    owns the workflow (state, routing, durability), the SDK owns agent execution
#    (tools, context, subagent lifecycle).
#
# 2. **Agent Skills vs raw prompts.** Skills are discoverable, versioned, and
#    shareable across agents and teams. A single Skill can be reused in Claude
#    Code, the API, and Managed Agents.
#
# 3. **Subagents isolate context.** The main agent sees only the researcher's
#    final diagnosis, not its full tool trace, which keeps the main context
#    window lean.
#
# 4. **Claude Managed Agents** (April 8, 2026 launch) runs this same
#    `AgentDefinition` shape, but Anthropic hosts the infrastructure. No Docker,
#    no load balancers, no session store. Quickstart:
#    <https://platform.claude.com/docs/en/managed-agents/quickstart>
#
# 5. **When to reach for what.**
#    - LangGraph alone: when your workflow is the hard part.
#    - Claude Agent SDK alone: when one agent with tools is enough.
#    - Hybrid (today's pattern): when you need both, reliable orchestration
#      plus rich per-step agent capabilities.


# %% [markdown]
# ## CHALLENGE
#
# 1. Replace the researcher subagent with two parallel subagents:
#    one for log inspection, one for metric correlation. Merge their outputs.
# 2. Request Claude Managed Agents beta, deploy this same agent definition,
#    and hit it from the CLI.
