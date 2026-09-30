#!/usr/bin/env python3
"""Exercise 01 · Claw basics: two speeds, three harnesses, one honest paragraph.

Fill in the three TODOs, save, then run:
    .venv/bin/python week26/01_what_is_a_claw/exercises/ex01_claw_basics.py

The checker is free and offline. Stuck? Compare with exercises/solutions/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from clawkit import banner, check  # noqa: E402

LAYERS = ("filesystem", "process", "network", "inference")

# ── TODO 1 ── which policy layers can you change on a RUNNING sandbox, and which are locked at creation?
#   Put each of the four LAYERS in exactly one of the two sets.
HOT = set()        # e.g. {"…", "…"}
LOCKED = set()

# ── TODO 2 ── pick a harness per job: "openclaw", "hermes" or "deepagents".
HARNESS = {
    "concierge": None,   # a Telegram concierge bot for hotel guests
    "refactor": None,    # a code-refactoring agent on a private repo
    "langfuse": None,    # traces must land in Langfuse with zero raw keys inside the sandbox
}

# ── TODO 3 ── explain to a hotel GM, in 2–4 sentences, why "the agent runs in a sandbox" is different
#   from "the agent is safe". Say what the sandbox LIMITS and what it does NOT fix.
GM_PARAGRAPH = """
"""


# ─────────────────────────── checker — no need to edit below ────────────────
def main() -> None:
    banner("Exercise 01 · claw basics", "offline checker · free · no Spark needed", status=False)
    good = True
    good &= check(HOT == {"network", "inference"} and LOCKED == {"filesystem", "process"},
                  "layers: network + inference are hot-reloadable · filesystem + process are locked at creation",
                  f"TODO 1: HOT={sorted(HOT) or '{}'} LOCKED={sorted(LOCKED) or '{}'} — which two can "
                  "`openshell policy update` / `inference set` change live?")
    good &= check(HARNESS == {"concierge": "openclaw", "refactor": "deepagents", "langfuse": "hermes"},
                  "harnesses: concierge → openclaw · refactor → deepagents · langfuse → hermes",
                  f"TODO 2: {HARNESS} — look at Section 4's 'good at' column")
    t = GM_PARAGRAPH.lower()
    limits = any(w in t for w in ("limit", "contain", "blast radius", "restrict", "can only", "cannot reach",
                                  "which files", "damage"))
    not_fix = any(w in t for w in ("not make", "doesn't make", "does not make", "not immune", "prompt injection",
                                   "still be fooled", "still make mistakes", "not correct", "can still"))
    good &= check(len(t.split()) >= 25 and limits and not_fix,
                  "your GM paragraph says what the sandbox limits AND what it does not fix",
                  "TODO 3: write ≥ 25 words that say what the sandbox limits (files, hosts, credentials…) "
                  "AND what it does not fix (wrong answers, prompt injection)")
    if not good:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)
    print("\n═ Done. Keep the GM paragraph — Module 07's threat model starts from it.")


if __name__ == "__main__":
    main()
