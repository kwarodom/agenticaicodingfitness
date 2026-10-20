#!/usr/bin/env python3
"""PART 4 · Eyes on the building — Metropolis & VSS  [ADVANCED]

NVIDIA Metropolis is the vision-AI stack: DeepStream SDK (GPU video pipelines), TAO
(fine-tune detectors), and Metropolis microservices — for occupancy counting, queue/
space utilization, PPE/safety, slip-and-fall. The VSS (Video Search & Summarization)
AI Blueprint puts a VLM over the camera feeds so staff ask questions of video in plain
English. This demo runs a mock occupancy feed end-to-end — detection, an NL video
question, and the correction pushed back into the energy model's schedules.

Run:  python demos/step04_metropolis_vss.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

QUEUE_ZONE, QUEUE_N = "lobby", 15


def main() -> None:
    view.banner("PART 4", "Eyes on the building — Metropolis & VSS", "ADVANCED")
    print("▣ MODE: SIM — a mock occupancy feed stands in for DeepStream people-counting ($0).")
    print("  the real stack runs the same analytics on live RTSP streams, on-prem.\n")

    print("Two twin tie-ins that make cameras a TWIN feature, not a bolt-on:")
    print("  a) plan cameras IN the twin first — simulate placement/coverage/occlusion in the")
    print("     USD scene before buying hardware, and render synthetic training data from the")
    print("     same scene to fine-tune detectors with TAO.")
    print("  b) occupancy analytics feed the ENERGY MODEL's schedules — vision closes the")
    print("     loop back to SIMULATION (Apps 6–8).\n")

    print("Mock occupancy feed (people per frame):\n")
    print(f"  {'time':<8}{'camera':<9}{'zone':<24}{'count'}")
    print("  " + "─" * 48)
    feed = sim.occupancy_feed()
    for hhmm, cam, zone, n in feed:
        print(f"  {hhmm:<8}{cam:<9}{zone:<24}{n}")
    print()

    print("→ DETECT 1 · queue rule: lobby count > 15 in consecutive frames")
    over = [(t, n) for t, _, z, n in feed if z == QUEUE_ZONE and n > QUEUE_N]
    peak = max(over, key=lambda x: x[1])
    print(f"  ← QUEUE at reception {over[0][0]}–{over[-1][0]} (+10 min), "
          f"peak {peak[1]} people at {peak[0]} → page the duty manager, open desk 3\n")

    print("→ DETECT 2 · after-hours rule: occupied while the schedule says empty (> 21:00)")
    after = [(t, z, n) for t, _, z, n in feed if t > "21:00"]
    print(f"  ← {after[0][1]} occupied {after[0][0]}–{after[-1][0]} "
          f"({after[0][2]} people) — schedule says unoccupied after 18:00\n")

    q = "How long was the lobby queue over 15 people this morning, and when did it peak?"
    print(f'→ NL VIDEO QUESTION (the VSS pattern — a VLM over the feeds):\n  » "{q}"')
    mins = len(over) * 10
    print(f"  ← A (grounded in the feed): about {mins} minutes ({over[0][0]}–{over[-1][0]} "
          f"plus the frame after), peaking at {peak[1]} people at {peak[0]}.\n")

    print("→ PUSH · occupancy correction into the energy model (what-if via Apps 6–8):\n")
    base = sim.SCHEDULE_WHATIF["baseline_kwh_day"]
    print(f"  {'schedule correction':<72}{'Δ kWh/day':>10}")
    print("  " + "─" * 84)
    delta = 0.0
    for what, d in sim.SCHEDULE_WHATIF["corrections"]:
        delta += d
        print(f"  {what:<72}{d:>+10.1f}")
    print("  " + "─" * 84)
    print(f"  {'HVAC baseline ' + str(base) + ' kWh/day → corrected':<72}"
          f"{base + delta:>10.1f}  ({delta:+.1f}, {100 * delta / base:+.1f} %)")
    print("\n  Note the +9: cameras also catch rooms the schedule wrongly calls EMPTY. The")
    print("  win is a truthful model (and no 22:00 comfort complaint), not always fewer kWh.\n")

    print("# run it for real (GPU box / Jetson at the edge):")
    print("#   DeepStream pipeline for counting → TAO to fine-tune the detector on YOUR lobby")
    print("#   docker run --gpus all nvcr.io/nvidia/deepstream:<tag>")
    print("#   VSS AI Blueprint (VLM over streams): build.nvidia.com → video search & summarization\n")

    print("Takeaway: vision is the twin's highest-bandwidth live sensor — it corrects the")
    print("energy model, feeds the copilot, and files the work orders. All four chapters meet")
    print("in the capstone, App 12: the living building.")


if __name__ == "__main__":
    main()
