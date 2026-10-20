#!/usr/bin/env python3
"""STEP 6 · 💻 Evaluate — base vs your tuned model (REAL)  [ADVANCED]

The honest test of fine-tuning: behaviour on held-out questions. If the tuned
model ('hvac-assistant' from STEP 5) is reachable on the current connection, this
asks the SAME questions to the BASE model and to the TUNED model and shows the
difference — real calls over your connection. Otherwise it contrasts a generic vs
domain system prompt so you still see the effect.

Run:  python demos/step06_evaluate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import ftview  # noqa: E402

TUNED = "hvac-assistant"
DOMAIN_SYSTEM = ("You are HotelHVAC-Assistant, fine-tuned on smart-hotel HVAC SOPs. "
                 "Answer with SPECIFIC setpoints, thresholds (°C, Pa), and alarm "
                 "priorities (CRITICAL/ROUTINE). Be terse and operational.")
QUESTIONS = [
    "Filter ΔP is 260 Pa — what alarm priority and what action?",
    "A guest room is stuck at 26 °C with the fan-coil maxed. First two moves?",
]


def ask(model: str, system: str, q: str) -> str:
    client = ftview._client()
    r = client.chat.completions.create(
        model=model, max_tokens=200, temperature=0.3,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": q}])
    msg = r.choices[0].message
    return (msg.content or getattr(msg, "reasoning", "") or "").strip()


def main() -> None:
    print("━" * 64)
    print("  STEP 6 · 💻 Evaluate — base vs tuned (real)")
    print("━" * 64, "\n")
    ftview.mode_line()

    if ftview.is_sim():
        print("No live endpoint — showing the SIM contrast (generic vs domain prompt).")
        print("Connect to your DGX (where 'hvac-assistant' is served) for a REAL eval.\n")
        for i, q in enumerate(QUESTIONS, 1):
            print(f"── Q{i}: {q}")
            print("  BEFORE:"); ftview.generate([{"role": "system", "content": "You are a helpful assistant."},
                                                 {"role": "user", "content": q}], max_tokens=120, label="(generic)")
            print("  AFTER:"); ftview.generate([{"role": "system", "content": DOMAIN_SYSTEM},
                                                {"role": "user", "content": q}], max_tokens=120, label="(domain)")
            print()
        return

    models = config.list_local_models()
    tuned = TUNED if TUNED in models else None
    base = next((m for m in models if m != TUNED), config.MODEL)

    if tuned:
        print(f"Comparing BASE ({base}) vs TUNED ({tuned}) — real calls:\n")
        for i, q in enumerate(QUESTIONS, 1):
            print(f"── Q{i}: {q}")
            print(f"\n  BEFORE — {base}:")
            print("   ", ask(base, "You are a helpful assistant.", q).replace("\n", "\n    "))
            print(f"\n  AFTER  — {tuned} (your fine-tune):")
            print("   ", ask(tuned, "You are HotelHVAC-Assistant.", q).replace("\n", "\n    "))
            print()
        print("The TUNED model should answer with the house's specific setpoints/priorities")
        print("without needing a big system prompt — that's what the LoRA baked in.")
    else:
        print(f"'{TUNED}' not found on this connection (run STEPS 4–5 to create + serve it).")
        print(f"Showing the domain-prompt contrast on {base} instead:\n")
        for i, q in enumerate(QUESTIONS, 1):
            print(f"── Q{i}: {q}")
            print(f"\n  BEFORE — generic:\n    " + ask(base, "You are a helpful assistant.", q).replace("\n", "\n    "))
            print(f"\n  AFTER  — domain prompt:\n    " + ask(base, DOMAIN_SYSTEM, q).replace("\n", "\n    "))
            print()

    print("Gate it in CI: tuned must beat base on a golden set AND not regress on general Qs.")
    print("\nTakeaway: only the held-out eval proves the fine-tune worked — and here it")
    print("ran on YOUR model, on YOUR DGX, end to end.")


if __name__ == "__main__":
    main()
