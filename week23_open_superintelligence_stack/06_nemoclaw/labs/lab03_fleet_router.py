#!/usr/bin/env python3
"""LAB 03 · Fleet routing + right-sizing — does the supervisor pick the right expert?

demos/step04_fleet.py routes ONE task. Here you route a small batch, score the
model-as-router against a keyword baseline, and then do the fit-math that makes
fleets cheap: most specialists ride Nano-class models; only the hard desk gets
Super. Routing calls are REAL when an endpoint is up; the economics table is
illustrative fit-math (labeled as such), not a benchmark.

Run:  cd /Users/altodev/Desktop/agenticaicodingfitness && \
      .venv/bin/python week23/06_nemoclaw/labs/lab03_fleet_router.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import view    # noqa: E402  (reused for its thinking-model / direct-model helpers)

FLEET = {
    "hvac":    ("nemotron-3-nano",  "comfort/energy from telemetry"),
    "finance": ("nemotron-3-super", "budgets, invoices, forecasts"),
    "code":    ("nemotron-3-super", "reads/writes/reviews code"),
}

# (task, the specialist a human supervisor would pick)
TASKS = [
    ("Room 1203 keeps overheating and guests are complaining — investigate.", "hvac"),
    ("Q3 utility invoices are 12% over forecast — reconcile the budget.",     "finance"),
    ("The telemetry ingest script throws a KeyError on 'setpoint' — fix it.", "code"),
]


def keyword_route(task: str) -> str:
    t = task.lower()
    if any(k in t for k in ("room", "hvac", "temp", "cooling", "overheat")):
        return "hvac"
    if any(k in t for k in ("budget", "invoice", "forecast", "cost")):
        return "finance"
    return "code"


def _pick_router_model() -> str:
    """Routing is a terse call. Thinking models (qwen3.6, gemma4, nemotron-3)
    burn a small budget reasoning — same auto-swap view.classify() does."""
    if view.is_thinking_model(config.MODEL):
        alt = view.pick_direct_model()
        if alt:
            print(f"  ◆ auto-picked {alt} for terse routing — {config.MODEL} is a"
                  " thinking model (reasoning preamble would eat the budget).\n")
            return alt
    return config.MODEL


def model_route(client, model: str, task: str) -> tuple[str | None, float]:
    """One real routing call. Thinking models reason first — we give headroom and
    extract the LAST label mention (final answers come last)."""
    names = ", ".join(FLEET)
    t0 = time.time()
    r = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=200,
        messages=[{"role": "user", "content":
                   f"You are a supervisor over specialist agents [{names}]. Reply with the "
                   f"single specialist name best suited to this task.\nTask: {task}"}])
    msg = r.choices[0].message
    text = str(getattr(msg, "reasoning", "") or "") + " " + (msg.content or "")
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    hit = None
    for m in re.finditer("|".join(FLEET), text, re.I):
        hit = m.group(0).lower()
    return hit, time.time() - t0


EXPECTED = """[no endpoint — showing expected output for the model-router column]
  task 1 → model: hvac      (agrees with keyword + human)   ~1.8s
  task 2 → model: finance   (agrees with keyword + human)   ~1.6s
  task 3 → model: code      (agrees with keyword + human)   ~1.7s
  model-router score: 3/3 — worth it when tasks stop containing obvious keywords."""


def main() -> None:
    print("━" * 64)
    print("  ▣ LAB 03 — route a batch across the fleet, then right-size it")
    print("━" * 64 + "\n")
    print("The fleet (each a NemoClaw spec — persona + skills + signed policy):")
    for name, (model, role) in FLEET.items():
        print(f"  • {name:8s} [{model:16s}] {role}")
    print()

    print("◈ Round 1 — keyword baseline (local code, always runs):\n")
    kw_score = 0
    for task, want in TASKS:
        got = keyword_route(task)
        kw_score += (got == want)
        print(f"  {'✓' if got == want else '✗'} {got:8s} ← {task[:58]}…")
    print(f"\n  keyword score: {kw_score}/{len(TASKS)}\n")

    print("◈ Round 2 — the model as supervisor (REAL routing calls):\n")
    if config.MODE != "real":
        print("▣ SIM — no endpoint reachable. To go REAL: ollama pull qwen3.6:35b-a3b-q8_0\n")
        print(EXPECTED)
    else:
        print(f"▣ REAL · {config.MODEL} @ {config.safe_base_url()} · {config.cost_note()}\n")
        m_score = 0
        try:
            from openai import OpenAI
            # max_retries=0: one attempt per task — retries would triple the
            # worst-case wall time of a slow endpoint.
            client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY,
                            timeout=15.0, max_retries=0)
            router_model = _pick_router_model()
            for i, (task, want) in enumerate(TASKS, 1):
                got, dt = model_route(client, router_model, task)
                ok = got == want
                m_score += ok
                print(f"  {'✓' if ok else '✗'} task {i} → model: {got or '(no label)':8s} "
                      f"expected: {want:8s} {dt:5.1f}s")
            print(f"\n  model-router score: {m_score}/{len(TASKS)} "
                  f"(keyword baseline: {kw_score}/{len(TASKS)})")
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ endpoint call failed ({type(e).__name__}) — check the connection.\n")
            print(EXPECTED)

    print("\n◈ Round 3 — right-sizing fit-math (ILLUSTRATIVE, not a benchmark):")
    print("  runbook: on a GB10, a Nano-class MoE decodes ~3× faster than Super-class.")
    tasks_day, tok_task, super_tps = 1000, 600, 13.0     # ~13 tok/s Super-class, ×3 Nano
    all_super = tasks_day * tok_task / super_tps / 3600
    routed = (tasks_day * tok_task * (2 / 3) / (super_tps * 3) +
              tasks_day * tok_task * (1 / 3) / super_tps) / 3600
    print(f"  1000 tasks/day × 600 tok — all-Super ≈ {all_super:4.1f} GPU-h/day;")
    print(f"  routed (2/3 Nano, 1/3 Super)         ≈ {routed:4.1f} GPU-h/day "
          f"→ ~{(1 - routed / all_super) * 100:.0f}% saved.")
    print("\n✓ Takeaway — the supervisor routes; the specialists stay small. Ship")
    print("  specialists, not one do-everything agent. Next: ../07_guardrails_openshell/")


if __name__ == "__main__":
    main()
