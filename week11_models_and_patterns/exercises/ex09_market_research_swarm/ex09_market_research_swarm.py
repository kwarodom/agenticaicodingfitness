"""
Exercise 09 — Market Research Swarm (Intermediate).

NOTE: PDF specifies CrewAI; this implementation uses raw Anthropic SDK with
asyncio + a dataclass Agent abstraction, per project preference.

Four agents — Scout, FeatureAnalyst, PricingAnalyst, GTMStrategist. Scout
runs first; FeatureAnalyst and PricingAnalyst run in parallel on Scout's
output; Strategist synthesises all three. Demonstrates parallel-then-synthesis.

Key concepts:
    - Sequential prerequisite (Scout) feeding parallel analyses
    - asyncio.gather for the parallel layer
    - Synthesis agent reading multi-source context
    - Agent backstory as domain expertise injection
"""

import asyncio
import os
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

load_dotenv()
MODEL = "claude-haiku-4-5-20251001"
client = anthropic.AsyncAnthropic()


@dataclass
class Agent:
    role: str
    goal: str
    backstory: str

    def system(self) -> str:
        return (
            f"You are a {self.role}. {self.backstory}\n"
            f"Your goal: {self.goal}"
        )

    async def run(self, task: str, max_tokens: int = 700) -> str:
        resp = await client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=self.system(),
            messages=[{"role": "user", "content": task}],
        )
        return resp.content[0].text


scout = Agent(
    "Market Scout",
    "Identify top 5 hotel IoT energy platforms in SEA",
    "PropTech analyst covering ASEAN markets since 2018.",
)
feature_analyst = Agent(
    "Feature Analyst",
    "Compare technical capabilities across platforms",
    "IoT product manager, ex-Siemens Smart Infrastructure.",
)
pricing_analyst = Agent(
    "Pricing Analyst",
    "Decode pricing models and identify market gaps",
    "Commercial strategist for B2B SaaS/IoT subscriptions.",
)
strategist = Agent(
    "GTM Strategist",
    "Synthesise research into an actionable positioning brief",
    "Strategy consultant for ASEAN IoT scale-ups.",
)


SCOUT_TASK = (
    "List 5 competitors to CERO (AltoTech) in hotel energy management for "
    "SEA. Fields: name, country, product, USP. "
    "Output a 5-item bullet list with 4 fields each."
)


async def main() -> None:
    print("=" * 60)
    print("Market research swarm — Scout -> (Feature || Pricing) -> Strategy")
    print("=" * 60)

    print("\n[Scout] Identifying competitors...")
    scout_out = await scout.run(SCOUT_TASK)

    feature_task = (
        "Compare the 5 competitors across: AI optimisation, HVAC control, "
        "open API, mobile app, SEA localisation. Output a markdown table.\n\n"
        f"=== COMPETITORS ===\n{scout_out}"
    )
    pricing_task = (
        "Estimate pricing models for the 5 competitors: per-room/year, "
        "enterprise flat fee, or freemium. Note any gaps AltoTech could "
        "exploit. Output a 2-gap-opportunities pricing summary.\n\n"
        f"=== COMPETITORS ===\n{scout_out}"
    )

    print("[Feature || Pricing] Running in parallel...")
    matrix, pricing = await asyncio.gather(
        feature_analyst.run(feature_task),
        pricing_analyst.run(pricing_task),
    )

    strategy_task = (
        "Synthesise scout, feature, and pricing research into a 150-word "
        "GTM brief for AltoTech CERO. Include: top differentiator, pricing "
        "recommendation, one risk.\n\n"
        f"=== SCOUT ===\n{scout_out}\n\n"
        f"=== FEATURES ===\n{matrix}\n\n"
        f"=== PRICING ===\n{pricing}"
    )
    print("[Strategist] Synthesising...")
    brief = await strategist.run(strategy_task, max_tokens=600)

    print("\n=== GTM Strategy Brief ===")
    print(brief)

    out = os.path.join(os.path.dirname(__file__), "ex09_swarm_output.md")
    with open(out, "w") as f:
        f.write("# Scout — Competitors\n\n")
        f.write(scout_out)
        f.write("\n\n# Feature Analyst — Comparison Matrix\n\n")
        f.write(matrix)
        f.write("\n\n# Pricing Analyst — Pricing Summary\n\n")
        f.write(pricing)
        f.write("\n\n# Strategist — GTM Brief\n\n")
        f.write(brief)
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    asyncio.run(main())
