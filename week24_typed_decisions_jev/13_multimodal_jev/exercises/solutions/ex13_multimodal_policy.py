#!/usr/bin/env python3
"""Solution · Exercise 13 — one good answer (yours may differ and still be right)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevkit import noul  # noqa: E402

import ex13_multimodal_policy as ex  # noqa: E402

ex.HAZARD_QUESTION = noul(
    "Does `event` describe smoke, steam, haze, vapour, fire, sparks or leaking fluid that could be dangerous "
    "in a plant room? Treat steam and haze as possible smoke, because a camera cannot always tell them apart.")


def my_policy(signal: dict) -> dict:
    if not signal.get("perception_ok", True):
        return {"route": "human_review", "alert": False, "reasons": ["perception failed"], "execute": False}
    reasons, route, alert = [], signal.get("queue") or "front_desk", False
    if signal.get("safety", 0) >= 0.30:
        alert, route = True, "safety_review"
        reasons.append(f"safety {signal['safety']:.2f} ≥ 0.30")
    if signal.get("suspicious", 0) >= 0.50:
        reasons.append("suspicious payment request → finance review")
        if not alert:
            route = "finance_review"
    heard, trusted = signal.get("room_heard"), signal.get("room_trusted")
    if heard and trusted and heard != trusted:
        reasons.append(f"confirm room: heard {heard}, caller-ID {trusted}")
    return {"route": route, "alert": alert, "reasons": reasons or ["routine"], "execute": False}


ex.my_policy = my_policy

if __name__ == "__main__":
    ex.main()
