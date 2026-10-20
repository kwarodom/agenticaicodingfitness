#!/usr/bin/env python3
"""PART 6 · Sovereign-memory audit  [ADVANCED]

The v2 payoff: a self-evolving agent where BOTH the brain AND the memory live on
the DGX. A cloud agent's memory sits in a vendor's database; here your agent's
accumulated decisions and domain knowledge — often your most sensitive asset —
never leave the building. This audits exactly that.

Run:  python demos/step06_sovereign_audit.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import brain  # noqa: E402
import config  # noqa: E402
import memory  # noqa: E402
import sevview  # noqa: E402


def _check(label: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'WARN'}] {label:<32} {detail}")
    return ok


def main() -> None:
    sevview.banner("PART 6", "Sovereign-memory audit", "ADVANCED")
    sevview.brain_line()

    total = passed = 0

    brain_local = brain.name() == "local"
    total += 1; passed += _check(
        "brain is sovereign (on-DGX)", brain_local,
        f"BRAIN={brain.name()} → " + ("local model, on-box" if brain_local
        else "cloud/sim — set BRAIN=local for full sovereignty"))

    mem_local = config.MEMORY_DIR.is_dir()
    total += 1; passed += _check(
        "memory is on local disk", mem_local,
        f"{config.MEMORY_DIR}")

    leaked = [k for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if os.environ.get(k)]
    # only a problem for sovereignty if the brain is actually cloud
    creds_ok = brain_local and not (brain.name() == "claude")
    total += 1; passed += _check(
        "no cloud dependency for thinking", brain_local,
        "all inference on the DGX" if brain_local else f"cloud creds in use: {leaked or 'n/a'}")

    m = memory.stats()
    total += 1; passed += _check(
        "memory persists across runs", m["episodes"] > 0 or m["facts"] > 0,
        f"{m['episodes']} episodes · {m['facts']} facts · {m['skills']} skills on disk")

    print()
    print("The sovereignty contrast that defines v2:")
    print("  cloud self-evolving agent  → brain in a vendor API, memory in a vendor DB")
    print("                               (your accumulated know-how lives off-box)")
    print("  DGX self-evolving agent    → brain + memory both on YOUR hardware")
    print("                               (your know-how is an asset you physically own)")
    print()
    verdict = "FULLY SOVEREIGN ✓" if passed == total else "PARTIALLY SOVEREIGN"
    print(f"  audit: {passed}/{total} — {verdict}")
    if passed != total:
        print("  → set BRAIN=local with a live DGX/Ollama endpoint for a clean pass.")

    print("\nTakeaway: Week 19 in one agent — a model you run (App 1), tuned to your")
    print("domain (App 2), observable (App 3), that LEARNS over time, with both its")
    print("mind and its memory on the DGX. That's a sovereign, self-evolving agent.")


if __name__ == "__main__":
    main()
