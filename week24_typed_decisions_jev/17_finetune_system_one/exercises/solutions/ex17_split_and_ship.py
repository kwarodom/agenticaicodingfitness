#!/usr/bin/env python3
"""Exercise 17 · SOLUTION · Write the two functions that keep a fine-tune honest: a leak-proof split and a ship gate.

Fill in the two TODOs, save, then run:
    .venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/exercises/ex17_split_and_ship.py

The checker runs 12 offline tests (free, instant, no model). When they pass it grades the
2026-10-17 fine-tunes with YOUR gate and compares the verdicts with Lab 17-7's.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ftkit as F  # noqa: E402  (first: may re-run this file inside week24_typed_decisions_jev/.venv-local)
from jevkit import banner, check  # noqa: E402


# ── TODO 1 ── group_split(tickets, val_groups) → (train, val)
#   `tickets` is a list of dicts that each have a "sid" (the situation the ticket was generated from).
#   `val_groups` is a set of sids that belong to validation.
#   Every ticket whose sid is in val_groups goes to val, every other ticket to train — so no situation
#   is ever on both sides. Keep the original order inside each list. Do not modify the input.
#   If a sid in val_groups matches NO ticket, raise ValueError (a typo would silently shrink val).
def group_split(tickets: list[dict], val_groups: set) -> tuple[list[dict], list[dict]]:
    unknown = set(val_groups) - {t["sid"] for t in tickets}
    if unknown:
        raise ValueError(f"no ticket has situation id(s) {sorted(unknown)}")
    return ([t for t in tickets if t["sid"] not in val_groups], [t for t in tickets if t["sid"] in val_groups])


# ── TODO 2 ── ship(metrics, criteria) → ("SHIP" | "HOLD", {criterion id: "PASS" | "FAIL" | "MISSING"})
#   `criteria` is the list from criteria.json: each has "id", "metric" and a "min" and/or a "max".
#   For each criterion read metrics[criterion["metric"]]:
#     • missing or None                      → "MISSING"
#     • below "min" (if given) or above "max" (if given) → "FAIL"      (equal to the bound is a PASS)
#     • otherwise                            → "PASS"
#   The verdict is "SHIP" only if every criterion is "PASS" (MISSING holds a release, like FAIL),
#   and "HOLD" for an empty criteria list (nothing checked is not the same as everything passed).
def ship(metrics: dict, criteria: list[dict]) -> tuple[str, dict]:
    rows = {}
    for c in criteria:
        v = metrics.get(c["metric"])
        if v is None:
            rows[c["id"]] = "MISSING"
        elif (c.get("min") is not None and v < c["min"]) or (c.get("max") is not None and v > c["max"]):
            rows[c["id"]] = "FAIL"
        else:
            rows[c["id"]] = "PASS"
    return ("SHIP" if rows and all(s == "PASS" for s in rows.values()) else "HOLD"), rows


# ─────────────────────────── checker — no need to edit below ────────────────
T = [{"id": i, "sid": s} for i, s in enumerate(["a", "b", "a", "c", "b", "c", "a"])]
CRIT = [{"id": "C1", "metric": "gain", "min": 10}, {"id": "C2", "metric": "p", "max": 0.05},
        {"id": "C3", "metric": "recall", "min": 1.0}]


def main() -> None:
    banner("Exercise 17 · leak-proof split + ship gate")
    ok = True
    try:
        tr, va = group_split(T, {"b"}) or (None, None)
    except Exception as e:  # noqa: BLE001
        tr, va = None, None
        print(f"   group_split raised {type(e).__name__}: {e}")
    ok &= check(va is not None and [t["id"] for t in va] == [1, 4], "val holds every ticket of situation b, in order",
                f"TODO 1: val should be ids [1, 4] · got {va and [t['id'] for t in va]}")
    ok &= check(tr is not None and [t["id"] for t in tr] == [0, 2, 3, 5, 6], "train holds everything else, in order",
                f"TODO 1: train should be ids [0, 2, 3, 5, 6] · got {tr and [t['id'] for t in tr]}")
    ok &= check(tr is not None and not ({t["sid"] for t in tr} & {t["sid"] for t in va or []}),
                "no situation on both sides", "TODO 1: a situation appears in train AND val — that is the leak")
    try:
        group_split(T, {"b", "zz"})
        raised = False
    except ValueError:
        raised = True
    except Exception:  # noqa: BLE001
        raised = False
    ok &= check(raised, "an unknown situation id raises ValueError", "TODO 1: {'b', 'zz'} should raise ValueError")
    ok &= check(len(T) == 7 and T[1]["sid"] == "b", "input list left untouched", "TODO 1: do not modify `tickets`")

    cases = [({"gain": 12, "p": 0.01, "recall": 1.0}, "SHIP", {"C1": "PASS", "C2": "PASS", "C3": "PASS"}),
             ({"gain": 10, "p": 0.05, "recall": 1.0}, "SHIP", {"C1": "PASS", "C2": "PASS", "C3": "PASS"}),
             ({"gain": 9.9, "p": 0.01, "recall": 1.0}, "HOLD", {"C1": "FAIL", "C2": "PASS", "C3": "PASS"}),
             ({"gain": 30, "p": 0.2, "recall": 0.93}, "HOLD", {"C1": "PASS", "C2": "FAIL", "C3": "FAIL"}),
             ({"gain": 30, "p": 0.01}, "HOLD", {"C1": "PASS", "C2": "PASS", "C3": "MISSING"}),
             ({"gain": None, "p": 0.01, "recall": 1.0}, "HOLD", {"C1": "MISSING", "C2": "PASS", "C3": "PASS"})]
    for n, (m, want_v, want_rows) in enumerate(cases, 1):
        try:
            got = ship(m, CRIT)
        except Exception as e:  # noqa: BLE001
            got = f"raised {type(e).__name__}"
        ok &= check(got == (want_v, want_rows), f"gate case {n} → {want_v}",
                    f"TODO 2: case {n} {m} expected {(want_v, want_rows)} · got {got}")
    try:
        empty = ship({"gain": 99}, [])
    except Exception as e:  # noqa: BLE001
        empty = f"raised {type(e).__name__}"
    ok &= check(empty == ("HOLD", {}), "no criteria → HOLD", f"TODO 2: ship(m, []) should be ('HOLD', {{}}) · got {empty}")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        return

    print("\n✓ all offline tests pass — grading the 2026-10-17 fine-tunes with YOUR gate")
    crit = json.loads((F.HERE / "criteria.json").read_text(encoding="utf-8"))["criteria"]
    res_path = F.RUNS / "lab06_results.json"
    results = json.loads((res_path if res_path.is_file() else F.RECORDED / "lab06_results.json").read_text())
    for eng in ("laya_ft", "gliner_lora", "gliner_full"):
        m = results["engines"].get(eng)
        if not m:
            continue
        mine, ref = ship(m, crit)[0], F.ship_decision(m, crit)[0]
        fails = [c for c, s in ship(m, crit)[1].items() if s != "PASS"]
        print(f"   {eng:<12} your gate: {mine:<4}  Lab 17-7: {ref:<4}  {'✓' if mine == ref else '✕'}"
              + (f"   not passing: {', '.join(fails)}" if fails else ""))
    print("» Same criteria, same numbers, same verdict — whoever runs it. That is what makes it a gate.")


if __name__ == "__main__":
    main()
