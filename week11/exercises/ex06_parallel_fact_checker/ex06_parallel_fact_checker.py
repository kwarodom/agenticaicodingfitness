"""
Exercise 06 — Parallel Fact-Checker (Easy).

Three specialist fact-checkers run concurrently via asyncio.gather. Each
agent evaluates a claim from a different analytical lens. A consensus
engine aggregates verdicts via confidence-weighted voting.

Key concepts:
    - asyncio.gather for concurrent Claude API calls
    - asyncio.to_thread to make sync SDK calls non-blocking
    - Confidence-weighted voting for consensus
    - Dataclass as structured agent output
"""

import asyncio
import os
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = "claude-haiku-4-5-20251001"


@dataclass
class Verdict:
    agent: str
    verdict: str  # TRUE / FALSE / UNCERTAIN
    confidence: float
    reason: str


async def fact_agent(name: str, lens: str, claim: str) -> Verdict:
    prompt = (
        f"You are {name}, specialising in {lens}.\n"
        f'Evaluate: "{claim}"\n'
        "Reply EXACTLY:\n"
        "VERDICT: TRUE|FALSE|UNCERTAIN\n"
        "CONFIDENCE: 0.0-1.0\n"
        "REASON: one sentence"
    )
    resp = await asyncio.to_thread(
        client.messages.create,
        model=MODEL,
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )
    lines = resp.content[0].text.strip().splitlines()
    try:
        verd = lines[0].split(": ", 1)[1].strip()
        conf = float(lines[1].split(": ", 1)[1].strip())
        rsn = lines[2].split(": ", 1)[1].strip()
    except (IndexError, ValueError):
        verd, conf, rsn = "UNCERTAIN", 0.5, resp.content[0].text[:80]
    return Verdict(name, verd, conf, rsn)


async def parallel_check(claim: str) -> None:
    agents = [
        ("DataVerifier", "statistics and quantitative claims"),
        ("SourceChecker", "source credibility and citation trails"),
        ("LogicAnalyst", "logical consistency and causal reasoning"),
    ]
    print(f"Claim: {claim!r}\n")
    results: list[Verdict] = await asyncio.gather(
        *[fact_agent(n, s, claim) for n, s in agents]
    )

    print("Individual verdicts:")
    for r in results:
        print(
            f"  {r.agent}: {r.verdict} "
            f"(conf={r.confidence:.2f}) — {r.reason}"
        )

    scores: dict[str, float] = {}
    for r in results:
        scores[r.verdict] = scores.get(r.verdict, 0) + r.confidence
    winner = max(scores, key=scores.__getitem__)
    total = sum(scores.values())
    print(
        f"\nCONSENSUS: {winner} "
        f"({scores[winner] / total:.0%} weight)\n"
    )


CLAIMS = [
    "AI can reduce hotel energy bills by up to 30%.",
    "Smart buildings represent 90% of global electricity consumption.",
]


if __name__ == "__main__":
    for c in CLAIMS:
        asyncio.run(parallel_check(c))
        print("-" * 60)
