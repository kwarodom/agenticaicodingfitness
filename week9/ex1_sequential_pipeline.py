"""
Exercise 1 — Sequential Pipeline (Pattern A) without CrewAI.

Three role-specialised agents — Researcher, Writer, Editor — chained so each
consumes the previous one's output. Same coordination pattern as CrewAI's
Process.sequential, implemented directly on the Anthropic SDK so the
"organisational structuring" mechanic is visible end to end.

Maps to: Bellifemine et al. (2007) — organisational structuring.
"""

import os
import time
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


researcher = Agent(
    role="Senior Research Analyst",
    goal=(
        "Find and summarise the latest trends in building energy "
        "optimisation and smart HVAC systems."
    ),
    backstory=(
        "You are an expert energy analyst at AltoTech who tracks global "
        "trends in building efficiency, IoT sensors, and AI-driven HVAC "
        "optimisation."
    ),
)

writer = Agent(
    role="Technical Content Writer",
    goal="Write an engaging, informative blog post from a research brief.",
    backstory=(
        "You write for AltoTech's engineering blog. Your audience is "
        "building managers and facility engineers. You make complex topics "
        "accessible without losing technical accuracy."
    ),
)

editor = Agent(
    role="Chief Editor",
    goal=(
        "Review and polish articles for clarity, technical accuracy, and "
        "AltoTech brand voice."
    ),
    backstory=(
        "You are the senior editor at AltoTech, ensuring all published "
        "content meets high standards of technical accuracy and readability."
    ),
)


RESEARCH_TASK = (
    "Research the latest trends in AI-driven building energy optimisation. "
    "Focus on: (1) new HVAC control strategies, (2) IoT sensor advances, "
    "(3) real-world case studies with measurable energy savings. "
    "Return a structured brief with 5 key findings, each with a one-line "
    "rationale."
)

WRITING_TASK_TEMPLATE = (
    "Using the research brief below, write a 600-word blog post titled "
    '"This Week in Building Energy AI". Include an introduction, three '
    "main sections, and a conclusion with a call-to-action for AltoTech.\n\n"
    "=== RESEARCH BRIEF ===\n{brief}\n=== END BRIEF ==="
)

EDITING_TASK_TEMPLATE = (
    "Review the blog post below for: (1) technical accuracy, "
    "(2) grammar and clarity, (3) AltoTech brand voice consistency, "
    "(4) engaging headline. Return the final polished markdown only — "
    "no commentary.\n\n"
    "=== DRAFT ===\n{draft}\n=== END DRAFT ==="
)


def run_pipeline() -> str:
    print("=" * 60)
    print("Sequential MAS pipeline: Researcher -> Writer -> Editor")
    print("=" * 60)

    t0 = time.time()
    print("\n[1/3] Researcher working...")
    brief = researcher.run(RESEARCH_TASK, max_tokens=1200)
    print(f"      done in {time.time() - t0:.1f}s ({len(brief)} chars)")

    t1 = time.time()
    print("\n[2/3] Writer working...")
    draft = writer.run(
        WRITING_TASK_TEMPLATE.format(brief=brief), max_tokens=2000
    )
    print(f"      done in {time.time() - t1:.1f}s ({len(draft)} chars)")

    t2 = time.time()
    print("\n[3/3] Editor working...")
    final = editor.run(
        EDITING_TASK_TEMPLATE.format(draft=draft), max_tokens=2000
    )
    print(f"      done in {time.time() - t2:.1f}s ({len(final)} chars)")

    print(f"\nTotal pipeline time: {time.time() - t0:.1f}s")
    return final


if __name__ == "__main__":
    final_post = run_pipeline()

    out = os.path.join(os.path.dirname(__file__), "ex1_sequential_output.md")
    with open(out, "w") as f:
        f.write(final_post)

    print("\n" + "=" * 60)
    print("FINAL OUTPUT (first 500 chars):")
    print("=" * 60)
    print(final_post[:500])
    print("...")
    print(f"\nFull post saved to: {out}")
