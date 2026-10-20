#!/usr/bin/env python3
"""Exercise 13 · One policy for every modality — write the hazard question and the rules yourself.

A photo, a voicemail and a CCTV clip all become TEXT; Jev judges the text; YOUR code decides.
Fill in the two TODOs, save, then run:
    .venv/bin/python week24_typed_decisions_jev/13_multimodal_jev/exercises/ex13_multimodal_policy.py

The checker runs 7 offline tests first (free, no API call). Only when they pass does it ask
Jev your hazard question about three real video-model event descriptions from Lab 13-4.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from jevkit import ask, banner, check, guarded, noul, table, validate  # noqa: E402,F401

# ── TODO 1 ── a NOUL about `event` (a sentence a video model wrote).
#   It must count smoke, steam AND haze as possible hazards — the video model in Lab 13-4
#   called the same cloud "steam" on one run and "vapor or smoke" on another.
#   noul("Does `event` describe …?")
HAZARD_QUESTION = None


# ── TODO 2 ── the policy. `signal` is a dict:
#   {"safety": 0..1 (Jev), "suspicious": 0..1 (Jev, optional), "queue": str,
#    "perception_ok": bool, "room_heard": str|None, "room_trusted": str|None}
#   Return {"route": str, "alert": bool, "reasons": [str, ...], "execute": False} with these rules:
#     1. perception_ok is False        → route "human_review", alert False (nothing else matters)
#     2. safety ≥ 0.30                 → alert True, route "safety_review"   (low bar: misses cost more)
#     3. suspicious ≥ 0.50             → route "finance_review" (unless rule 2 already made it safety_review),
#                                        and add a reason containing "finance"
#     4. room_heard and room_trusted both set and different → add a reason containing "confirm room"
#                                        (the route stays whatever rules 2–3 chose)
#     5. otherwise route = signal["queue"]; reasons = ["routine"] if you added none
#   execute is ALWAYS False.
def my_policy(signal: dict) -> dict:
    return None


# ─────────────────────────── checker — no need to edit below ────────────────
TESTS = [
    ({"safety": 0.9, "queue": "water_leak", "perception_ok": False}, "human_review", False, None),
    ({"safety": 0.35, "queue": "water_leak", "perception_ok": True}, "safety_review", True, None),
    ({"safety": 0.05, "queue": "housekeeping", "perception_ok": True}, "housekeeping", False, "routine"),
    ({"safety": 0.02, "suspicious": 0.97, "queue": "finance", "perception_ok": True}, "finance_review", False, "finance"),
    ({"safety": 0.93, "suspicious": 0.9, "queue": "hvac", "perception_ok": True}, "safety_review", True, "finance"),
    ({"safety": 0.93, "queue": "hvac", "perception_ok": True, "room_heard": "1223", "room_trusted": "1203"},
     "safety_review", True, "confirm room"),
    ({"safety": 0.1, "queue": "plumbing", "perception_ok": True, "room_heard": "1507", "room_trusted": "1507"},
     "plumbing", False, "routine"),
]
EVENTS = [  # real descriptions gemini-3.8-flash wrote for media/cctv_plantroom.mp4 (Lab 13-4)
    "Steam begins to leak from the blue industrial equipment on the left",
    "Vapor or smoke starts venting from the blue compressor unit on the left",
    "Normal operating conditions in the plant room",
]


def main() -> None:
    banner("Exercise 13 · one policy for every modality")
    ok = True
    q = HAZARD_QUESTION or {}
    text = str(q.get("instructions", "")).lower()
    ok &= check(q.get("type") == "noul" and "`event`" in text and all(w in text for w in ("smoke", "steam", "haze")),
                "HAZARD_QUESTION is a noul about `event` that names smoke, steam and haze",
                "TODO 1: HAZARD_QUESTION should be noul(\"Does `event` …\") mentioning smoke, steam and haze")
    for n, (sig, route, alert, reason) in enumerate(TESTS, 1):
        try:
            out = my_policy(dict(sig))
        except Exception as e:  # noqa: BLE001
            out = None
            print(f"   test {n} raised {type(e).__name__}: {e}")
        good = (isinstance(out, dict) and out.get("route") == route and out.get("alert") is alert
                and out.get("execute") is False and isinstance(out.get("reasons"), list)
                and (reason is None or any(reason in r for r in out["reasons"])))
        ok &= check(good, f"test {n}: {route}{' + alert' if alert else ''}",
                    f"TODO 2: test {n} expected route={route} alert={alert}"
                    f"{' reason~' + repr(reason) if reason else ''} · got {out}")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again — no API call was made.")
        sys.exit(1)

    print("\n▣ asking Jev your hazard question about three real video-model descriptions …")
    qs = guarded({"hazard": HAZARD_QUESTION})
    rows = []
    for ev in EVENTS:
        a = validate(ask({"event": ev}, qs, quiet=True), qs)
        p = a["hazard"]["noul"]
        d = my_policy({"safety": p, "queue": "plant_room", "perception_ok": True})
        rows.append([ev[:60], f"{p:.2f}", d["route"]])
    table(rows, ["video-model event", "hazard", "your policy"])
    print("═ Both the 'steam' and the 'smoke' wording should reach safety_review — the words differ, the rule doesn't.")


if __name__ == "__main__":
    main()
