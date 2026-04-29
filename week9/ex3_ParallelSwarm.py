# Exercise 3: Parallel Swarm Challenge
# Build an async parallel swarm of 5 specialist agents using Python's asyncio. This demonstrates distributed
# sensing and stigmergy — agents work independently on sub-tasks and write results to shared state, which an
# aggregator then synthesizes (Wooldridge, 2002, Ch.9).

import os
from dotenv import load_dotenv
load_dotenv()

#Step 1: Define the Specialist Agents
import asyncio
import time
import anthropic
client = anthropic.AsyncAnthropic()
MODEL = "claude-haiku-4-5-20251001"
SPECIALISTS = {
    "energy": {
        "role": "Energy Efficiency Auditor",
        "prompt": "Analyze building energy consumption patterns."
        " Focus on: chiller COP, lighting schedules,"
        " peak demand management. Building: 50-floor"
        " office tower in Bangkok, 2500 occupants.",
    },
    "comfort": {
        "role": "Occupant Comfort Analyst",
        "prompt": "Assess occupant comfort metrics. Focus on:"
        " temperature uniformity, CO2 levels, humidity"
        " control, lighting quality. Same building.",
    },
    "maintenance": {
        "role": "Predictive Maintenance Engineer",
        "prompt": "Evaluate equipment health and maintenance"
        " schedules. Focus on: chiller vibration data,"
        " filter replacement cycles, sensor calibration"
        " status. Same building.",
    },
    "safety": {
        "role": "Safety Compliance Officer",
        "prompt": "Review safety and compliance status. Focus on:"
        " fire system status, emergency ventilation,"
        " air quality regulations, occupancy limits.",
    },
    "cost": {
        "role": "Cost Optimization Analyst",
        "prompt": "Analyze cost optimization opportunities."
        " Focus on: demand response revenue, TOU rate"
        " optimization, equipment lifecycle cost,"
        " energy procurement strategy.",
    },
}

# Step 2: Async Agent Runner
async def run_specialist(name: str, spec: dict) -> dict:
    """Run a single specialist agent asynchronously."""
    start = time.time()
    response = await client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=f"You are a {spec['role']}. Provide a concise 3-bullet audit finding.",
        messages=[
            {"role": "user", "content": spec["prompt"]},
        ],
    )
    elapsed = time.time() - start
    return {
        "specialist": name,
        "role": spec["role"],
        "findings": response.content[0].text,
        "time_seconds": round(elapsed, 2),
    }

# Step 3: Parallel Orchestration
async def run_audit():
    """Fan-out to all specialists in parallel, then aggregate."""
    print("Starting parallel building audit...\n")
    overall_start = time.time()
    # Fan-out: all specialists run concurrently
    tasks = [
        run_specialist(name, spec)
        for name, spec in SPECIALISTS.items()
    ]
    results = await asyncio.gather(*tasks)
    overall_time = time.time() - overall_start
    # Print individual results
    for r in results:
        print(f"--- {r['role']} ({r['time_seconds']}s) ---")
        print(r["findings"])
        print()
    # Aggregate
    sequential_time = sum(r["time_seconds"] for r in results)
    print(f"Parallel time: {overall_time:.1f}s")
    print(f"Sequential would be: {sequential_time:.1f}s")
    print(f"Speedup: {sequential_time/overall_time:.1f}x")
    return results

# Run it
results = asyncio.run(run_audit())

# Expected Output
# All 5 specialists should complete in roughly the time of the slowest one (not the sum). Typical speedup is 3-5x
# compared to sequential execution. This demonstrates the core MAS advantage of parallel execution — one of
# the five reasons for MAS listed in Theory Part 1.
