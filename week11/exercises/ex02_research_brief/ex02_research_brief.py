"""
Exercise 02 — Research Brief (Beginner).

NOTE: PDF specifies CrewAI; this implementation uses raw Anthropic SDK with
a dataclass Agent abstraction instead, per project preference.

Two agents — Researcher and Writer — collaborate sequentially to produce an
executive brief on a topic. Output of the Researcher feeds the Writer.

Key concepts:
    - Agent roles, goals, backstories shape response style
    - Sequential pipeline — output of task N is input of task N+1
    - Specialisation: each agent focuses on one job
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
    role="Research Analyst",
    goal="Find key facts and statistics on the given topic",
    backstory=(
        "Meticulous analyst who surfaces accurate data and cites specifics."
    ),
)

writer = Agent(
    role="Technical Writer",
    goal="Transform research notes into a polished executive brief",
    backstory=(
        "B2B writer skilled at turning raw data into crisp, actionable "
        "summaries."
    ),
)


RESEARCH_TASK = (
    "Research: 'AI energy management for hotels in Southeast Asia'. "
    "Find: (1) market size, (2) energy savings %, "
    "(3) top 3 vendor names, (4) adoption barriers. "
    "Output 4 bullet points with specific numbers."
)

WRITING_TASK_TEMPLATE = (
    "Using the research below, write a 250-word executive brief titled "
    "'AI Energy Management: SEA Hotel Opportunity'. "
    "Sections: Overview · Key Findings · Recommendation. "
    "Format as markdown.\n\n"
    "=== RESEARCH ===\n{research}\n=== END RESEARCH ==="
)


if __name__ == "__main__":
    print("=" * 60)
    print("Sequential pipeline: Researcher -> Writer")
    print("=" * 60)

    t0 = time.time()
    print("\n[Research Analyst] Starting task...")
    research = researcher.run(RESEARCH_TASK, max_tokens=800)
    print(f"  done in {time.time() - t0:.1f}s")

    t1 = time.time()
    print("\n[Technical Writer] Writing brief based on research...")
    brief = writer.run(
        WRITING_TASK_TEMPLATE.format(research=research), max_tokens=800
    )
    print(f"  done in {time.time() - t1:.1f}s")

    print("\n=== Executive Brief ===")
    print(brief)

    out = os.path.join(os.path.dirname(__file__), "ex02_brief_output.md")
    with open(out, "w") as f:
        f.write("# Research Notes\n\n")
        f.write(research)
        f.write("\n\n---\n\n# Executive Brief\n\n")
        f.write(brief)
    print(f"\nSaved to: {out}")
