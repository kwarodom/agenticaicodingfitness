#!/usr/bin/env python3
"""Solution · Exercise 03 — one good answer (yours may differ and still be right)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevkit import choice, noul  # noqa: E402

import ex03_fix_the_questions as ex  # noqa: E402

ex.good_outage = noul(
    "Does `message` report a failure or outage happening right now that needs immediate action?",
    true="Something has just failed or is failing now and needs action at once (tripped, burst, no power, flooding)",
    false="Minor or ongoing issues the sender is happy to have checked later, schedule questions, or thanks",
)
ex.good_team = choice("Which maintenance team should handle `message`?", {
    "hvac": "Chillers, AHUs, air conditioning, ventilation, filters, cooling or heating",
    "electrical": "Lights, breakers, power supply, sockets and electrical panels",
    "plumbing": "Pipes, leaks, water supply, drains and flooding",
    "not_a_request": "Thanks, feedback or anything that asks no team to do anything",
})
ex.good_callback = noul(
    "Does the sender of `message` ask to be phoned or called back?",
    true="The sender asks for a call, gives a number to call, or asks someone to ring them",
    false="No phone call is requested",
)

if __name__ == "__main__":
    ex.main()
