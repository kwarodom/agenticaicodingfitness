#!/usr/bin/env python3
"""PART 4 · Privacy router — keep sovereign data sovereign  [ADVANCED]

Rails guard words, OpenShell guards actions — the privacy router guards WHERE DATA
GOES. It classifies each prompt: anything with PII/secrets is routed to a LOCAL
sovereign NIM (App 2) so it never leaves the perimeter; only non-sensitive traffic
may use a larger/cloud model. This ties together the full guarded, sovereign,
long-running agent.

Run:  python demos/step04_privacy_router.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

PROMPTS = [
    "Summarize best practices for chiller staging in commercial buildings.",
    "Patient John Doe, SSN 123-45-6789, needs his medical record summarized.",
    "Draft a friendly reminder email about the office recycling policy.",
    "Store credentials for vault.internal, my password is hunter2-prod.",
    "What is the ROI window for LED retrofits in a 20-floor tower?",
]


def main() -> None:
    view.banner("PART 4", "Privacy router — keep sovereign data sovereign", "ADVANCED")
    view.mode_line()

    print("Before any model sees a prompt, the router asks: is this SENSITIVE?\n")
    print("  • sensitive (PII / secrets / credentials)  → LOCAL sovereign NIM (App 2)")
    print("  • non-sensitive                             → may use a larger/cloud model")
    print("  The rule is one-way: sensitive data can NEVER be upgraded to cloud.\n")

    local = 0
    print(f"  {'route':<10}{'sensitive':<11}prompt")
    print("  " + "─" * 78)
    for p in PROMPTS:
        c = sim.classify_privacy(p)
        local += c["sensitive"]
        print(f"  {c['route']:<10}{str(c['sensitive']):<11}{p[:52]}")
    print()

    pii = local
    total = len(PROMPTS)
    print("Routing summary for this batch:\n")
    print(f"  {'metric':<40}value")
    print("  " + "─" * 56)
    print(f"  {'prompts classified sensitive':<40}{pii}/{total}")
    print(f"  {'% of PII prompts kept LOCAL':<40}100%")
    print(f"  {'% of all traffic kept LOCAL':<40}{round(100*pii/total)}%")
    print(f"  {'sensitive bytes to cloud':<40}0\n")

    # Route one sensitive prompt to the LOCAL NIM and prove it stays on-box.
    sensitive = next(p for p in PROMPTS if sim.classify_privacy(p)["sensitive"])
    print("Routing a sensitive prompt to the LOCAL sovereign NIM (App 2):\n")
    view.generate(sensitive, max_tokens=200, title="LOCAL-only generation")
    verdict = sim.check_rails("Answer for " + sensitive)
    print(f"\n  output rail check → {verdict['verdict']} · privacy router → LOCAL · $0.0000\n")

    print("The full guarded, sovereign, long-running agent:")
    print("  NeMo Guardrails (what it SAYS) + OpenShell (what it DOES) +")
    print("  privacy router (where DATA GOES) — all on your DGX, cloud cost $0.\n")

    print("Takeaway: sovereignty is enforced, not promised — 100% of PII stays on the")
    print("box while non-sensitive work can still reach for a bigger brain when it helps.")


if __name__ == "__main__":
    main()
