#!/usr/bin/env python3
"""PART 1 · The Data Flywheel — a self-evolving agent loop  [BEGINNER]

An agent that ships is not an agent that's done. The NeMo Data Flywheel turns every
production interaction into fuel: observe traffic → curate it → customize a model →
evaluate it → promote the winner. NeMo Relay names the loop: observe → learn → optimize.

Run:  python demos/step01_the_loop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402

LOOP = """\
        ┌──────────────────────────────────────────────────────┐
        │                                                        │
        ▼                                                        │
   ① OBSERVE  ──►  ② CURATE  ──►  ③ CUSTOMIZE  ──►  ④ EVALUATE ──┘
   prod logs      NeMo Curator    NeMo Customizer   NeMo Evaluator
   (traffic)      dedup/filter/   LoRA/SFT/DPO/GRPO  LLM-judge A/B
                  PII-scrub/label  distill teacher→   promote if the
                                   student            student wins
"""


def main() -> None:
    view.banner("PART 1", "The Data Flywheel — a self-evolving loop", "BEGINNER")
    view.mode_line()

    print("The loop (NeMo Relay: observe → learn → optimize):\n")
    print(LOOP)
    print("Why this is the 'self-evolving' layer of Agent = Model + Harness:")
    print("  • ① Observe — every request/response/tool-call is logged as raw signal.")
    print("  • ② Curate — NeMo Curator dedups, quality-filters, scrubs PII, and labels")
    print("       the good traces (often with a big teacher model as the judge).")
    print("  • ③ Customize — NeMo Customizer fine-tunes a SMALL student (LoRA/SFT/DPO/GRPO)")
    print("       to match a big teacher — distillation on your DGX.")
    print("  • ④ Evaluate — NeMo Evaluator (LLM-judge) A/B-tests student vs teacher;")
    print("       if the student matches quality at lower cost, it's PROMOTED to serve.")
    print("  • …then the improved model produces new traffic, and the loop repeats.\n")

    print("The payoff: a model that gets cheaper AND better in production, automatically,")
    print("without sending your data off the box.\n")

    print("Takeaway: the flywheel converts sovereign traffic into a self-improving model.")
    print("Next: how Curator turns raw logs into training data.")


if __name__ == "__main__":
    main()
