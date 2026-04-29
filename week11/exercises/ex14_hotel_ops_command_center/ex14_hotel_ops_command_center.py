"""
Exercise 14 — Hotel Operations Command Center (Expert, Full MAS).

NOTE: PDF specifies AutoGen GroupChat + CrewAI sub-crews. This
implementation uses AutoGen v0.4+ (autogen_agentchat) + raw Anthropic SDK
(in place of CrewAI), per project preference.

Top-level coordinator: AutoGen SelectorGroupChat with 4 department agents
(Energy, Maintenance, Guest Experience, F&B) plus an OpsDirector.
Each department agent's morning brief is pre-computed by a raw-SDK
sequential analysis (the "sub-crew" replacement).

Key concepts:
    - SelectorGroupChat as top-level orchestrator (LLM-driven turn taking)
    - Pre-computed analysis briefs as agent context (nested MAS)
    - Domain-specific agent personas
    - Bounded conversation via MaxMessageTermination
"""

import asyncio
import json
import os
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
)
from autogen_agentchat.ui import Console
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
sync_client = anthropic.Anthropic()


STATE = {
    "hotel": "AltoTech Grand Bangkok",
    "date": "2026-04-28 09:00 +07",
    "occupancy": "145/200 rooms",
    "energy_kw": 420,
    "target_kw": 380,
    "maintenance_tickets": {"open": 12, "critical": 2},
    "vip_arrivals_today": 5,
    "fb": {"breakfast_covers": 89, "bar_low_stock": 3},
}


# --- Mini analysis "sub-crew" via raw SDK ---


@dataclass
class MiniAgent:
    role: str
    goal: str
    backstory: str

    def system(self) -> str:
        return (
            f"You are a {self.role}. {self.backstory}\n"
            f"Your goal: {self.goal}"
        )

    def run(self, task: str, max_tokens: int = 400) -> str:
        resp = sync_client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=self.system(),
            messages=[{"role": "user", "content": task}],
        )
        return resp.content[0].text


def dept_brief(dept: str, focus: str) -> str:
    """Pre-compute a morning brief for a department (replaces sub-crew)."""
    analyst = MiniAgent(
        f"{dept} Analyst",
        f"Produce a 5-bullet morning brief for {dept}",
        f"Expert hotel {dept.lower()} operations specialist.",
    )
    return analyst.run(
        f"Hotel state: {json.dumps(STATE)}\n"
        f"Focus: {focus}\n"
        f"Write a 5-bullet-point {dept} brief for the GM morning meeting. "
        "Be concise, actionable."
    )


# --- AutoGen department agents ---


async def main() -> None:
    print("Generating department briefs via mini sub-crews...\n")
    energy_brief = dept_brief("Energy", "current kW vs target, quick wins")
    maint_brief = dept_brief(
        "Maintenance", "critical tickets and guest-facing issues"
    )
    guest_brief = dept_brief(
        "Guest Experience", "VIP arrivals and pending requests"
    )
    fb_brief = dept_brief(
        "Food & Beverage", "breakfast status and stock alerts"
    )

    model_client = AnthropicChatCompletionClient(
        model=MODEL, api_key=os.environ["ANTHROPIC_API_KEY"]
    )

    def make_dept_agent(name: str, dept: str, brief: str) -> AssistantAgent:
        return AssistantAgent(
            name=name,
            model_client=model_client,
            system_message=(
                f"You are the {dept} Manager AI.\n"
                f"Your morning brief:\n{brief}\n\n"
                "When called upon, report your top 3 priority actions. "
                "Be concise — max 4 sentences."
            ),
        )

    energy = make_dept_agent("EnergyMgr", "Energy", energy_brief)
    maint = make_dept_agent("MaintMgr", "Maintenance", maint_brief)
    guest = make_dept_agent("GuestMgr", "Guest Experience", guest_brief)
    fb = make_dept_agent("FBMgr", "Food & Beverage", fb_brief)

    ops_dir = AssistantAgent(
        name="OpsDirector",
        model_client=model_client,
        system_message=(
            "You are the Hotel Operations Director AI. "
            "Collect reports from EnergyMgr, MaintMgr, GuestMgr, FBMgr. "
            "After all 4 have reported, synthesise a 5-bullet GM summary "
            "and end your message with: BRIEFING COMPLETE."
        ),
    )

    team = SelectorGroupChat(
        participants=[ops_dir, energy, maint, guest, fb],
        model_client=model_client,
        termination_condition=(
            TextMentionTermination("BRIEFING COMPLETE")
            | MaxMessageTermination(max_messages=12)
        ),
    )

    print("\n=== Hotel Operations Command Center ===\n")
    initial = (
        "Good morning. Please provide the daily operations briefing. "
        "I need priority actions from all four departments."
    )
    await Console(team.run_stream(task=initial))
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
