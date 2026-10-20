#!/usr/bin/env python3
"""PART 2 · Omniverse's 2026 shape — libraries + microservices, not an app  [BEGINNER]

If you last looked in 2022, forget the screenshots: Omniverse is NOT an app suite you
download and click around in. It is a collection of libraries and microservices you
BUILD with — the Kit SDK, OpenUSD at the core, Kit App Streaming, and USD NIMs.
The Launcher is gone (EOL Oct 1 2025) and the old first-party connectors with it.

Run:  python demos/step02_omniverse_2026.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

DIAGRAM = """\
   2022 mental model (dead):        2026 mental model (real):
   ┌──────────────────────┐         ┌────────────────────────────────────────┐
   │ Launcher → Create /  │   ═►    │  YOUR twin app  (built on Kit SDK)     │
   │ View + DCC connectors│         │  ── OpenUSD at the core ──             │
   └──────────────────────┘         │  Kit App Streaming · USD Code/Search   │
      an app you install            └────────────────────────────────────────┘
                                       a format + an SDK + microservices
"""


def main() -> None:
    view.banner("PART 2", "Omniverse's 2026 shape — libraries + microservices", "BEGINNER")
    print("▣ MODE: offline — this chapter is pure stdlib (no GPU, no endpoint, $0).\n")

    print(DIAGRAM)
    print("The pieces you actually build with:\n")
    print(f"  {'piece':<20}{'kind':<14}what it does")
    print("  " + "─" * 84)
    for name, kind, what in sim.OMNIVERSE_PIECES:
        print(f"  {name:<20}{kind:<14}{what}")
    print()

    print("What changed, and when:")
    print("  • Omniverse Launcher — EOL Oct 1 2025. There is no 'install Omniverse' anymore;")
    print("    you build a Kit app, or run NVIDIA's reference samples.")
    print("  • First-party DCC/BIM connectors — deprecated. Interchange is now NATIVE")
    print("    OpenUSD: your BIM/DCC tool exports USD itself, coordinated by AOUSD")
    print("    (Alliance for OpenUSD: Pixar, Adobe, Apple, Autodesk, NVIDIA — a Linux")
    print("    Foundation alliance, founded Aug 2023).")
    print("  • The old flagship apps (Create → USD Composer, View) are now reference")
    print("    SAMPLES — starting points for your own Kit app, not products to learn.\n")

    print("Why this is good news for you:")
    print("  • App-specific skills die with the app; FORMAT skills compound. USD files you")
    print("    author for the hotel twin outlive any viewer, vendor or version.")
    print("  • A twin built as 'your Kit app over your USD stage' is something you OWN —")
    print("    the same sovereignty argument as Week 18/19, applied to 3D.\n")

    print("Takeaway: learn the FORMAT (OpenUSD — App 02) and the SDK (Kit), not a deprecated")
    print("app. Next: Cosmos — the generative sibling of Omniverse's deterministic physics.")


if __name__ == "__main__":
    main()
