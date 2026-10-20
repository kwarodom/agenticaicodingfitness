#!/usr/bin/env python3
"""Lab 16-1 · The System One landscape in October 2026 — who ships what, and what the numbers do (not) mean.

Offline: reads landscape.json (from systemonemodels.org and the model cards, checked 2026-10-17).
  1. the open-weights models you can run yourself
  2. the hosted alternatives to Jev
  3. the scoreboards — and why one model can "win" and "lose" in the same week
  4. a decision table for AltoTech

Run: .venv/bin/python week24_typed_decisions_jev/16_local_system_one_models/labs/lab01_landscape.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "common"))
from jevkit import step, table  # noqa: E402

L = json.loads((HERE / "landscape.json").read_text(encoding="utf-8"))
print("━" * 72 + f"\n━━ Lab 16-1 · System One landscape (checked {L['checked']})\n" + "━" * 72)
print(f"◈ offline — no model, no key · source: {L['sources']['directory']}")

step(1, "open weights you can run on your own hardware")
table([[("● " if m["in_this_module"] else "  ") + m["model"], m["vendor"], m["params"], m["context"], m["licence"]]
       for m in L["open_weights"]], ["model (● = used in this module)", "vendor", "size", "context", "licence"])
for m in L["not_system_one"]:
    print(f"» also used: {m['model']} — {m['job']}. Why here: {m['why_here']}.")

step(2, "hosted decision models (Jev is no longer the only one)")
table([[m["model"], m["vendor"], m["price"], m["context"], m["images"]] for m in L["hosted"]],
      ["model", "vendor", "price", "context", "images"])

step(3, "scoreboards — read the column headers, not just the numbers")
table([[s["short"], s["Jev"], s["GLiNER2.5-Decide"] if s["GLiNER2.5-Decide"] is not None else "—",
        s["Laya"] if s["Laya"] is not None else "—"] for s in L["scores"]],
      ["board", "Jev", "GLiNER2.5-Decide", "Laya"])
print("» On the independent Jev Decision Index (broad, zero-shot) Jev scores 60 and Laya 4 (0 = chance).")
print("» On Fastino's own 17-domain suite GLiNER2.5-Decide beats Jev 60.2 to 57.6.")
print("» Fine-tuned on 1,001 Banking77 examples, Laya reaches 79.4 against Jev's 80.0.")
print("» All three are true. They measure different jobs: broad zero-shot, the vendor's chosen domains,")
print("  and one task after training. Only YOUR tickets, scored the same way for every model, answer YOUR question.")

step(4, "which one for what (AltoTech)")
table([
    ["many changing questions, long states, no GPUs to run", "Jev (hosted)", "best zero-shot, 64k context, $0.042/M"],
    ["text that must not leave the building (guest PII, contracts)", "Laya or GLiNER2.5-Decide", "local, Apache-2.0"],
    ["< 50 ms decisions in a control loop at the edge", "Laya (MLX / Core ML ports)", "no network round trip"],
    ["a narrow task with labelled history", "fine-tuned Laya or Decide", "a fine-tuned small model can close the gap"],
    ["you need offsets: room numbers, dates, equipment tags", "classic GLiNER → then decide", "decision models return no spans"],
    ["Thai text, local", "Laya Router / GLiNER2.5-multi-Decide", "never the English-only checkpoints"],
], ["need", "start with", "because"])
