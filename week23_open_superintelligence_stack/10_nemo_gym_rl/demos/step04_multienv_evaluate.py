#!/usr/bin/env python3
"""PART 4 · Multi-environment RL + evaluate  [ADVANCED]

Train across SEVERAL verifiable-reward environments at once (math + code + tool-use)
so the policy generalizes, then EVALUATE the trained policy's pass-rate against the
base model. Finishes by asking the improved agent to answer live (or simulated), and
ties back to the Data Flywheel (App 11) for promotion.

Run:  python demos/step04_multienv_evaluate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

# per-environment pass-rate: base model vs the GRPO-trained policy.
EVAL = [
    ("math_verify",  0.44, 0.71),
    ("code_unit",    0.38, 0.66),
    ("tool_call",    0.41, 0.72),
    ("json_schema",  0.52, 0.79),
    ("retrieval_qa", 0.30, 0.52),
]


def _bar(v: float, width: int = 24) -> str:
    fill = int(round(v * width))
    return "█" * fill + "·" * (width - fill)


def main() -> None:
    view.banner("PART 4", "Multi-environment RL + evaluate", "ADVANCED")
    view.mode_line()

    print("Train on a MIX of environments so one policy generalizes (not just one skill):\n")
    print(f"  {'Environment':<14}{'Verifier':<32}task type")
    print("  " + "─" * 70)
    for name, verifier, _reward, task in sim.ENVIRONMENTS[:3]:
        print(f"  {name:<14}{verifier:<32}{task}")
    print("\n  env=nemo_gym:math_verify,code_unit,tool_call   # comma-separated = multi-env\n")

    print("Evaluate the trained policy vs the base model (pass-rate on held-out tasks):\n")
    print(f"  {'Environment':<14}{'base':>6}{'trained':>9}   trained pass-rate")
    print("  " + "─" * 62)
    base_tot = trained_tot = 0.0
    for name, base, trained in EVAL:
        base_tot += base
        trained_tot += trained
        print(f"  {name:<14}{base:>6.0%}{trained:>9.0%}   {_bar(trained)}")
    b, t = base_tot / len(EVAL), trained_tot / len(EVAL)
    print("  " + "─" * 62)
    print(f"  {'OVERALL':<14}{b:>6.0%}{t:>9.0%}   ← {b:.0%} → {t:.0%} after RL\n")

    print("Ask the IMPROVED agent to act (live on the endpoint, or simulated):\n")
    view.generate("You are a room-booking agent. Book the 2pm slot in room Orion for 30 "
                  "minutes and state the single tool call you make.",
                  max_tokens=200, title="the GRPO-trained policy answering")

    print("\nClose the loop with the Data Flywheel (App 11):")
    print("  • Log real production traffic → mine hard/failed cases → new Gym tasks.")
    print("  • Re-run GRPO on those tasks → re-evaluate → PROMOTE the new policy if it")
    print("    beats the incumbent on the eval gate. Serve it as a NIM (App 2). Repeat.")

    print("\nTakeaway: multi-env GRPO lifted pass-rate ~41% → ~68%; evaluation is the gate")
    print("and the Data Flywheel is the engine that keeps the policy improving.")


if __name__ == "__main__":
    main()
