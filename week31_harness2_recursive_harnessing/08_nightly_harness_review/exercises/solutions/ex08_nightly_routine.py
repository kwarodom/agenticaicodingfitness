#!/usr/bin/env python3
"""Solution 08 · Wire the nightly routine's four rules: selection, the envelope, the firewall, the ship hour.

Four TODOs. Run it: each check prints ✓ or ✕, and the script exits 1 until every check is ✓.

    .venv/bin/python week31_harness2_recursive_harnessing/08_nightly_harness_review/exercises/ex08_nightly_routine.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import harness as H                        # noqa: E402
import reviewkit as R                      # noqa: E402
from bench import promote                  # noqa: E402
from ops_sim import execute            # noqa: E402
from h2out import check, finish            # noqa: E402

# TODO 1 — the plain check's selection rule (EXAMPLE design): put the three weak signals in priority order, and
#          set how many tasks one night may harness (the EXAMPLE value in the tutorial).
PRIORITY = ["rejected_proposal", "low_rating", "helper_failed"]
N = 3


def select(wakes):
    weak = [w for w in wakes if w.signal in PRIORITY]
    weak.sort(key=lambda w: (PRIORITY.index(w.signal), 0 if w.human_reason else 1, w.wake_id))
    return weak[:N]


# TODO 2 — the send_to_user envelope (EXAMPLE design). `draft` is a harness tree or None (a KEEP_BASE night).
def envelope_for(draft):
    return {"channel": "operator", "outcome": "done", "notify": draft is not None,
            "draft": H.definition_hash(draft) if draft else None}


# TODO 3 — the selector's input: build it from the request, the sources and the anonymised deliverables ONLY.
#          (t.visible() gives request + sources; r.text() is a deliverable as a human reads it.)
def selector_input(t, runs):
    v = t.visible()
    return {"request": v["request"], "sources": v["sources"],
            "candidates": [{"id": f"c{i}", "deliverable": r.text()} for i, r in enumerate(runs)]}


# TODO 4 — three versions were approved today at 09:00, 13:00 and 17:00 (drafts[0..2], each containing the one
#          before). Choose the (hour, tree) events that ship them so no cold prefix lands in work hours (08–20)
#          and today's wakes see at most two versions.
v1 = H.base_harness()
night = R.nightly_review(R.day(R.BUSY_DAY), v1)
drafts = [promote(v1, night.records[:i])[0] for i in (1, 2, 3)]
SHIP = [(2, drafts[2])]          # the next deployment boundary, all three approvals bundled


# ── checker (do not edit below) ────────────────────────────────────────────────
good = True
busy = R.day(R.BUSY_DAY)
got = [w.task.id for w in select(busy)]
good &= check(got == ["F1-01", "F2-00", "F3-00"],
              f"selection: {', '.join(got)} (rejected proposal first, then the two low ratings; the helper waits)",
              f"selection {got}: weak signals in the EXAMPLE priority order, N = 3")
e0, e1 = envelope_for(None), envelope_for(night.draft)
good &= check(e0.get("notify") is False and e1.get("notify") is True and e0.get("outcome") == e1.get("outcome") == "done",
              "envelope: outcome=done always; notify=false on a KEEP_BASE night, true when a draft exists",
              f"envelope: KEEP_BASE night → {e0}, draft night → {e1}")
wake = R.day(["F1-01"])[0]
runs = [execute(v1, wake.task, seed=s) for s in range(3)]
si = selector_input(wake.task, runs)
blob = json.dumps(si, ensure_ascii=False, default=str)
leaks = [f for f in R.FORBIDDEN if f in R._keys(si)] + (["the reviewer's words"] if wake.human_reason in blob else [])
def strings(o):
    if isinstance(o, str):
        return {o}
    items = o.values() if isinstance(o, dict) else o if isinstance(o, (list, tuple)) else []
    return set().union(set(), *(strings(x) for x in items))


has_work = wake.task.request in strings(si) and all(r.text() in strings(si) for r in runs) and "work_orders" in R._keys(si)
good &= check(not leaks and has_work,
              "firewall: request, sources and deliverables in; no reason, signal, approval, rating or truth",
              f"firewall: leaks {leaks or 'none'} · request+sources+deliverables present: {has_work}")
sch = R.schedule(SHIP, v1)
cold = R.cold_prefixes(sch, H.prefix_hash(v1))
in_hours = [h for h in cold if 8 <= h < 20]
shipped_last = H.prefix_hash(dict(SHIP[-1][1])) == H.prefix_hash(drafts[2]) if SHIP else False
good &= check(not in_hours and len(set(sch)) <= 2 and shipped_last,
              f"ship plan: cold prefixes at {', '.join(f'{h:02d}:00' for h in cold) or 'none'} · "
              f"{len(set(sch))} version(s) seen today · the last approval ships",
              f"ship plan: cold prefixes in work hours {in_hours} · versions seen {len(set(sch))} · last approval shipped: {shipped_last}")
finish(good, "Done. Module 09 asks whether a version like this one should be adopted at all, and decides it before looking.")
