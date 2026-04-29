"""
Exercise 3 — Parallel Swarm (Pattern C) with asyncio.

Five domain specialists audit the same building concurrently. Each writes
its findings into a shared results list — no sequential blocking. Total
wall-clock time should be ~max(individual call times), not their sum.

Maps to: distributed sensing + stigmergy (Wooldridge 2002 Ch.9).
    Each specialist has a partial view of the world.
    Shared state (the results list) is the modern blackboard.
"""

import asyncio
import json
import os
import time
from dotenv import load_dotenv
import anthropic

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
client = anthropic.AsyncAnthropic()

SPECIALISTS: dict[str, dict[str, str]] = {
    "energy": {
        "role": "Energy Efficiency Auditor",
        "prompt": (
            "Analyse building energy consumption patterns. Focus on chiller "
            "COP, lighting schedules, and peak demand management. "
            "Building: 50-floor office tower in Bangkok, 2,500 occupants."
        ),
    },
    "comfort": {
        "role": "Occupant Comfort Analyst",
        "prompt": (
            "Assess occupant comfort metrics. Focus on temperature "
            "uniformity, CO2 levels, humidity control, and lighting "
            "quality. Same building."
        ),
    },
    "maintenance": {
        "role": "Predictive Maintenance Engineer",
        "prompt": (
            "Evaluate equipment health and maintenance schedules. Focus on "
            "chiller vibration data, filter replacement cycles, and sensor "
            "calibration status. Same building."
        ),
    },
    "safety": {
        "role": "Safety Compliance Officer",
        "prompt": (
            "Review safety and compliance status. Focus on fire system "
            "status, emergency ventilation, air quality regulations, and "
            "occupancy limits. Same building."
        ),
    },
    "cost": {
        "role": "Cost Optimisation Analyst",
        "prompt": (
            "Analyse cost optimisation opportunities. Focus on demand "
            "response revenue, time-of-use rate optimisation, equipment "
            "lifecycle cost, and energy procurement strategy. Same "
            "building."
        ),
    },
}


async def run_specialist(name: str, spec: dict) -> dict:
    """Run a single specialist agent — independent of all other agents."""
    start = time.time()
    response = await client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=(
            f"You are a {spec['role']}. Provide a concise 3-bullet audit "
            "finding. Each bullet should be actionable and quantified."
        ),
        messages=[{"role": "user", "content": spec["prompt"]}],
    )
    return {
        "specialist": name,
        "role": spec["role"],
        "findings": response.content[0].text,
        "time_seconds": round(time.time() - start, 2),
    }


async def run_audit() -> list[dict]:
    """Fan out to all specialists concurrently, then return all results."""
    print("=" * 60)
    print(f"Parallel building audit — {len(SPECIALISTS)} specialists")
    print("=" * 60)

    overall_start = time.time()
    tasks = [
        run_specialist(name, spec) for name, spec in SPECIALISTS.items()
    ]
    results = await asyncio.gather(*tasks)
    overall_time = time.time() - overall_start

    for r in results:
        print(f"\n--- {r['role']} ({r['time_seconds']}s) ---")
        print(r["findings"])

    sequential_time = sum(r["time_seconds"] for r in results)
    slowest = max(r["time_seconds"] for r in results)

    print("\n" + "=" * 60)
    print(f"Parallel wall-clock     : {overall_time:.1f}s")
    print(f"Slowest single specialist: {slowest:.1f}s")
    print(f"Hypothetical sequential  : {sequential_time:.1f}s")
    print(f"Speedup vs sequential    : {sequential_time / overall_time:.1f}x")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = asyncio.run(run_audit())

    out = os.path.join(os.path.dirname(__file__), "ex3_audit_output.md")
    with open(out, "w") as f:
        f.write("# Exercise 3 — Parallel Building Audit\n\n")
        for r in results:
            f.write(f"## {r['role']} ({r['time_seconds']}s)\n\n")
            f.write(f"{r['findings']}\n\n---\n\n")

    out_json = os.path.join(
        os.path.dirname(__file__), "ex3_audit_output.json"
    )
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAudit report saved to : {out}")
    print(f"Raw JSON saved to     : {out_json}")
