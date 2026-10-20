#!/usr/bin/env python3
"""Lesson 06.2 — Parallel swarm (fan-out, then aggregate).

Run:  python week15/code/06_multi_agent/02_parallel_swarm.py

Five specialists audit a building AT THE SAME TIME using asyncio.gather, then
we report the speedup vs. doing them one-by-one. This is distributed sensing +
stigmergy: independent workers writing into shared results.

Mirrors the class repo's week9/ex3_ParallelSwarm.py. Uses Haiku on purpose —
cheap + fast is right for many parallel calls.
"""
import asyncio
import time
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import anthropic

client = anthropic.AsyncAnthropic()
MODEL = "claude-haiku-4-5-20251001"

SPECIALISTS = {
    "energy": "Energy Efficiency Auditor",
    "comfort": "Occupant Comfort Analyst",
    "maintenance": "Predictive Maintenance Engineer",
    "safety": "Safety Compliance Officer",
    "cost": "Cost Optimization Analyst",
}


async def run_specialist(name, role):
    start = time.time()
    resp = await client.messages.create(
        model=MODEL, max_tokens=200,
        system=f"You are a {role}. Give a concise 3-bullet audit finding.",
        messages=[{"role": "user", "content": "Audit a 50-floor office tower in Bangkok, 2500 occupants."}],
    )
    return {"name": name, "role": role,
            "findings": resp.content[0].text,
            "seconds": round(time.time() - start, 2)}


async def run_audit():
    print("Fanning out to 5 specialists in parallel...\n")
    overall_start = time.time()
    results = await asyncio.gather(*(run_specialist(n, r) for n, r in SPECIALISTS.items()))
    overall = time.time() - overall_start

    for r in results:
        print(f"--- {r['role']} ({r['seconds']}s) ---\n{r['findings']}\n")

    sequential = sum(r["seconds"] for r in results)
    print(f"Parallel time:        {overall:.1f}s")
    print(f"Sequential would be:  {sequential:.1f}s")
    print(f"Speedup:              {sequential / overall:.1f}x")


if __name__ == "__main__":
    asyncio.run(run_audit())
