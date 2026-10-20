#!/usr/bin/env python3
"""CH 3 · Eyes on the city — the Smart City Blueprint (VSS 3.0) pipeline  [INTERMEDIATE]

NVIDIA's real blueprint, simulated faithfully: RTVI CV (RTDETR/GDINO) detects and
tracks on every stream, events flow through Kafka, the Alert Verification Service
(a VLM) re-watches each clip and rejects false positives, and a VSS agent answers
natural-language questions over the lot. Stream budgets are the documented per-GPU
numbers — the capacity math here is the real math.

Run:  python demos/step02_vss_camera_network.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402
from city import vss, world  # noqa: E402

PIPELINE = """\
  24 cams ─► [VIOS/NVStreamer] ─► [RTVI CV: RTDETR·GDINO] ─► kafka cv.events
                                                                   │
      ops map (:3002) ◄── kafka alerts.verified ◄── [Alert Verification VLM]
      VSS agent (:7777) — natural language over all of it (MCP: VST :8001 · VA :9901)
"""


def main() -> None:
    view.banner("CH 3", "Eyes on the city — the VSS Smart City Blueprint", "INTERMEDIATE")
    view.mode_line()

    print("The pipeline (every box is a real blueprint microservice):\n")
    print(PIPELINE)

    print("Capacity planning — fit 24 cameras on ONE GPU (documented stream budgets):\n")
    print(f"  {'GPU':<26}{'rtdetr/gpu':>11}{'gdino/gpu':>10}   fits our 19 rtdetr + 5 gdino?")
    print("  " + "─" * 78)
    for gpu, b in vss.STREAM_BUDGET.items():
        plan = vss.plan_deployment(gpu)
        ok = "✓ yes" if plan["ok"] else f"✗ no — fits {plan['fit']['rtdetr']}r/{plan['fit']['gdino']}g"
        print(f"  {gpu:<26}{b['rtdetr']:>11}{b['gdino']:>10}   {ok}")
    print("  GDINO (open-vocabulary) costs 2.5–5× the streams of RTDETR — spend it")
    print("  only where you need free-text queries; RTDETR everywhere else.\n")

    print("A night of raw CV events → VLM verification (the blueprint's core economics):\n")
    confirmed = 0
    for ev in vss.RAW_EVENTS:
        v = vss.verify(ev)
        mark = "✓ CONFIRMED" if v["verdict"] == "CONFIRMED" else "✗ REJECTED "
        confirmed += v["verdict"] == "CONFIRMED"
        print(f"  {ev['t']}  {ev['cam']}  {ev['type']:<22} {mark}  {v['reason']}")
    print(f"\n  {len(vss.RAW_EVENTS)} raw events → {confirmed} verified alerts. CV casts a "
          f"wide cheap net per-frame;\n  the VLM confirms per-alert — humans only ever see "
          f"the {confirmed}.\n")

    print("One event's trip through the wire (23:15 sideswipe at I-11):\n")
    for line in vss.kafka_trace(vss.RAW_EVENTS[1]):
        print(f"    {line}")
    print()

    print("Ask the VSS agent (natural language over the whole network):")
    view.generate(
        "You are the VSS agent for a city camera network. Tonight: 5 raw CV events, "
        "VLM verified 3, rejected 2 (bus dwell, lighting artifact); multi-camera "
        "sideswipe at Sukhumvit x Asoke 23:15 (CAM-13 + CAM-14). Answer the duty "
        "engineer's question: 'what did the cameras actually see tonight?' In 3 sentences, "
        "note that no plates/faces were extracted and video stayed on the city DGX.",
        max_tokens=200, title="VSS agent · what did the cameras see tonight?")
    print()
    print("run it for real (the actual blueprint):")
    print("  # docs.nvidia.com/vss/latest/smartcity-docs/Quickstart-Guide.html")
    print("  $ docker compose up   # after NGC login + env config (GPU per the table above)")
    print()
    print("Takeaway: detection is cheap, verification is the product — a VLM between")
    print("CV and humans is what makes 24 cameras operable by one engineer. Next:")
    print("what-if at city scale.")


if __name__ == "__main__":
    main()
