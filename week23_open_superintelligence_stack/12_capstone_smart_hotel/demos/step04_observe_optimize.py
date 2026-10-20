#!/usr/bin/env python3
"""CH 5 · Observe & optimize — NeMo Relay, Phoenix trace, and the economics  [INTERMEDIATE]

NeMo Relay observed every call; here we read it back like Phoenix (Agent Insights):
the span tree, latency and cost per turn, and how the Router right-sized each request
(Nano for cheap work, Super for hard reasoning). Then the inference economics — tokens,
$ / M-token, and energy — because only USEFUL work counts.

Run:  python demos/step04_observe_optimize.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402
from hotel.runtime import build  # noqa: E402
from hotel.relay import MODELS, NANO, SUPER  # noqa: E402


def main() -> None:
    view.banner("CH 5", "Observe & optimize — Relay · Phoenix · economics", "INTERMEDIATE")
    view.mode_line()

    rt = build()
    # a mixed workload so the router has both easy and hard requests to place
    rt.orch.handle_event("Room 0902 is empty — reduce energy if safe.", "0902")   # → Nano
    rt.orch.handle_event("Room 1203 temperature alarm — triage and act.", "1203")  # → Super

    print("\n".join(rt.relay.render_trace("mixed workload")))

    t = rt.relay.totals()
    print("\n▣ INFERENCE ECONOMICS (App 9)")
    print(f"  router right-sizing: nano×{t['nano_calls']} (${MODELS[NANO]['mtok_usd']}/Mtok) "
          f"· super×{t['super_calls']} (${MODELS[SUPER]['mtok_usd']}/Mtok)")
    print(f"  ◆ {t['llm_tokens']} tokens · ${t['cost_usd']:.5f} · {t['latency_ms']}ms · ~{t['energy_wh']} Wh")
    all_super = t["llm_tokens"] / 1_000_000 * MODELS[SUPER]["mtok_usd"]
    print(f"  if every call used Super: ${all_super:.5f} → routing saved "
          f"${max(0, all_super - t['cost_usd']):.5f} at equal outcome.")
    print("\nTakeaway: observe everything (Phoenix), then optimize by sending each request to")
    print("the cheapest model that can do it — cost and Watts drop, correctness holds.")


if __name__ == "__main__":
    main()
