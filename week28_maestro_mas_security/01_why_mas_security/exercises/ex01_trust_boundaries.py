#!/usr/bin/env python3
"""Exercise 01 · Trust boundaries: who is a request really for, and where can someone else's text get in?

Fill in the three TODOs, save, then run:
    .venv/bin/python week28_maestro_mas_security/01_why_mas_security/exercises/ex01_trust_boundaries.py

The checker is free, offline and instant. Stuck? Compare with exercises/solutions/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from ensemble import principal_of  # noqa: E402
from maskit import banner, check, finish  # noqa: E402

# ── TODO 1 ── an agent read text from TWO origins before it acted: the duty manager's request and a web page.
#   Which origin should the action be treated as coming from? Pick one string from the set below.
#   (Lab 01-1 step 2 shows the trust levels.)
MIXED = {"staff:duty-manager", "web:reviews.example"}
ACTS_FOR = ""          # e.g. "staff:duty-manager" or "web:reviews.example"

# ── TODO 2 ── which trust boundaries did scenario A03 cross? (Lab 01-2, step 2.) Keep only the ones it crossed.
BOUNDARIES = {
    "guest → orchestrator",
    "orchestrator → concierge",
    "concierge → energy",
    "energy → set_room_setpoint",
    "maintenance → vendor agent (A2A)",
    "concierge → web page",
    "concierge → email",
}

# ── TODO 3 ── in 2–4 sentences for the hotel GM: why is securing each agent on its own not enough?
#   Say how an instruction gets from a guest to a dangerous tool, and what every hop must carry to stop it.
GM_PARAGRAPH = """
"""


# ─────────────────────────── checker — no need to edit below ────────────────
def main() -> None:
    banner("Exercise 01 · trust boundaries", "offline checker · free · instant", brain=False)
    good = True
    good &= check(ACTS_FOR == principal_of(MIXED) == "web:reviews.example",
                  "acts for: web:reviews.example — the least-trusted text an action read decides how far it is trusted",
                  f"TODO 1: ACTS_FOR={ACTS_FOR!r} — if a web page can steer the action, can it be trusted like staff?")
    want = {"guest → orchestrator", "orchestrator → concierge", "concierge → energy", "energy → set_room_setpoint"}
    good &= check(BOUNDARIES == want,
                  "A03 crossed 4 boundaries: guest → orchestrator → concierge → energy → set_room_setpoint",
                  f"TODO 2: you kept {len(BOUNDARIES)} — remove the boundaries A03 never touched (no vendor, web or email)")
    t = GM_PARAGRAPH.lower()
    travels = any(w in t for w in ("forward", "pass", "hand", "relay", "delegat", "travel", "hop", "another agent",
                                   "next agent", "from agent to agent"))
    carries = any(w in t for w in ("who asked", "who the request", "origin", "provenance", "on behalf", "principal",
                                   "who it is for", "who it's for", "the guest's", "originally"))
    good &= check(len(t.split()) >= 25 and travels and carries,
                  "your GM paragraph says how instructions travel between agents AND that each hop must carry who "
                  "the request is really for",
                  "TODO 3: write ≥ 25 words that say (a) the instruction is passed from agent to agent until it "
                  "reaches the one holding the tool, and (b) every hop must carry who originally asked")
    finish(good, "Done. Module 02 gives these boundaries a name: the seven MAESTRO layers.")


if __name__ == "__main__":
    main()
