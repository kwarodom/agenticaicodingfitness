"""
Exercise 04 — 3-Agent Product Pipeline (Easy).

NOTE: PDF specifies CrewAI; this implementation uses raw Anthropic SDK with
a dataclass Agent abstraction instead, per project preference.

Three agents — Scout, Analyst, Strategist — chain sequentially. Scout
gathers competitor facts, Analyst builds a feature comparison matrix,
Strategist produces a positioning brief for AltoTech CERO.

Key concepts:
    - Passing upstream task outputs downstream (context chain)
    - Agent backstory shaping response style
    - Sequential process with 3+ tasks
    - Accessing individual task outputs
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
client = anthropic.Anthropic()


@dataclass
class Agent:
    role: str
    goal: str
    backstory: str

    def system_prompt(self) -> str:
        return (
            f"You are a {self.role}. {self.backstory}\n"
            f"Your goal: {self.goal}"
        )

    def run(self, task: str, max_tokens: int = 1500) -> str:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=self.system_prompt(),
            messages=[{"role": "user", "content": task}],
        )
        return resp.content[0].text


scout = Agent(
    role="Product Scout",
    goal="Identify competing IoT energy platforms and their headline features",
    backstory=(
        "Knows the smart-building landscape across SEA and knows how to "
        "read product pages critically."
    ),
)

analyst = Agent(
    role="Feature Analyst",
    goal="Map features onto a concise comparison matrix",
    backstory=(
        "Turns raw product info into clean comparison tables that "
        "executives can read in 60 seconds."
    ),
)

strategist = Agent(
    role="Positioning Strategist",
    goal="Write a differentiation brief for AltoTech's CERO platform",
    backstory=(
        "Strategy consultant who has repositioned 3 IoT startups for the "
        "ASEAN hospitality market."
    ),
)


SCOUT_TASK = (
    "List 4 competitors to AltoTech's CERO IoT platform "
    "(hotel energy management in SEA). "
    "For each: company, origin, key product, headline claim. "
    "Output as a bullet list with 4 fields each."
)

ANALYSIS_TEMPLATE = (
    "Using this competitor list, create a feature matrix comparing: "
    "AI optimisation, HVAC control, mobile app, API openness, SEA support. "
    "Rows = competitors, columns = features. "
    "Output as a markdown table with tick/cross/partial symbols.\n\n"
    "=== COMPETITOR LIST ===\n{scout}\n=== END ==="
)

STRATEGY_TEMPLATE = (
    "Using the competitor list and the feature matrix, write a 200-word "
    "positioning brief for AltoTech CERO. Identify 2 unique strengths and "
    "1 gap to address. Tone: confident, data-backed. "
    "Sections: Strengths / Gap / Tagline.\n\n"
    "=== SCOUT ===\n{scout}\n\n=== MATRIX ===\n{matrix}\n=== END ==="
)


if __name__ == "__main__":
    print("=" * 60)
    print("Sequential pipeline: Scout -> Analyst -> Strategist")
    print("=" * 60)

    print("\n[Scout] Identifying competitors...")
    scout_out = scout.run(SCOUT_TASK, max_tokens=800)

    print("[Analyst] Building feature matrix...")
    matrix = analyst.run(
        ANALYSIS_TEMPLATE.format(scout=scout_out), max_tokens=800
    )

    print("[Strategist] Writing positioning brief...")
    brief = strategist.run(
        STRATEGY_TEMPLATE.format(scout=scout_out, matrix=matrix),
        max_tokens=800,
    )

    print("\n=== Positioning Brief ===")
    print(brief)
    print("\n=== Feature Matrix ===")
    print(matrix)

    out = os.path.join(os.path.dirname(__file__), "ex04_pipeline_output.md")
    with open(out, "w") as f:
        f.write("# Scout — Competitors\n\n")
        f.write(scout_out)
        f.write("\n\n# Analyst — Feature Matrix\n\n")
        f.write(matrix)
        f.write("\n\n# Strategist — Positioning Brief\n\n")
        f.write(brief)
    print(f"\nSaved to: {out}")
