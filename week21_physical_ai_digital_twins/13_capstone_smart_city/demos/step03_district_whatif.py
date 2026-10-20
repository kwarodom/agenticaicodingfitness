#!/usr/bin/env python3
"""CH 4 · City what-if — reroute a corridor, flatten a district peak  [INTERMEDIATE]

The SIMULATION layer at city scale. Two engines, same discipline as Apps 6–8
(calibrate, then trust): a congestion model good enough to price a reroute before
you retime a single signal, and a CityLearn-style district energy dispatch (App 9
Ch 4) where buildings — including Capstone I's hotel — flatten the evening peak.

Run:  python demos/step03_district_whatif.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402
from city import grid_mobility as gm  # noqa: E402
from city import world  # noqa: E402


def bar(v: float, vmax: float, width: int = 26) -> str:
    return "█" * max(1, round(v / vmax * width))


def main() -> None:
    view.banner("CH 4", "City what-if — corridor reroute + district peak", "INTERMEDIATE")

    print("Evening baseline — the congestion map's atoms (v/c = volume/capacity):\n")
    print(f"  {'segment':<24}{'v/c':>6}{'km/h':>7}   level")
    print("  " + "─" * 52)
    for s in world.SEGMENTS:
        st = gm.segment_state(s["id"])
        print(f"  {st['name']:<24}{st['vc']:>6}{st['speed']:>7}   {st['level']}")
    print()

    print("WHAT-IF 1 · incident closes Asoke Interchange (S-06) → divert onto S-01/S-02:\n")
    print(f"  {'segment':<24}{'v/c':>6}{'km/h':>7}   level")
    print("  " + "─" * 52)
    for st in gm.reroute_whatif("S-06", ["S-01", "S-02"]):
        print(f"  {st['name']:<24}{st['vc']:>6}{st['speed']:>7}   {st['level']}")
    print("  The diversion holds both alternates under jam threshold (v/c < 1.0) —")
    print("  this is the number the duty engineer approves BEFORE signals are retimed.\n")

    print("WHAT-IF 2 · flatten the evening peak — CityLearn-style district dispatch:\n")
    d = gm.dispatch_peak()
    print(f"  {'hour':<7}{'base MW':>9}  {'':<27}{'dispatched':>11}")
    print("  " + "─" * 72)
    for h, b, s in zip(gm.HOURS, d["base"], d["shaved"]):
        print(f"  {h:<7}{b:>9}  {bar(b, 36):<27}{s:>11}  {bar(s, 36, 20)}")
    print(f"\n  district peak {d['peak_before']} → {d['peak_after']} MW · "
          f"flexible fleet: {d['flex_mw']:.1f} MW across {len(d['participants'])} buildings")
    for p in d["participants"]:
        print(f"    • {p}")
    print("  Old Town Hospital contributes 0 MW — life-safety loads are hard-excluded")
    print("  (a guardrail, not a preference — App 10's action-space lesson).\n")

    print("Honest scope: both engines are toys with the RIGHT SHAPE — real deployments")
    print("use calibrated traffic sims and building models that passed App 7's G14 gate.")
    print("An uncalibrated city twin's what-if is an opinion, at city scale.\n")

    print("Takeaway: the twin earns its keep when every intervention — reroute, retime,")
    print("DR call — is priced in simulation BEFORE it touches the street. Next: who is")
    print("allowed to act on these answers, and how that autonomy is earned.")


if __name__ == "__main__":
    main()
