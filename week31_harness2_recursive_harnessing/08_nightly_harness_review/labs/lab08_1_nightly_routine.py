#!/usr/bin/env python3
"""Lab 08-1 · The nightly harness review routine: plain check, weak-signal selection, a draft version, a human gate.

Two simulated days of HVAC Expert wakes (EXAMPLE design for this course). A quiet day
ends at the plain check: no model call, silence. A busy day selects up to N = 3 weak-signal wakes, runs one
Propose–Probe–Compose step per wake in Shadow, merges the supported edits into a DRAFT version, writes the review
packet to .runs/, and sends the draft through a named-reviewer gate. In 🦙 LIVE mode a local model plays the
editor for the first selected task (2 calls); everything else stays SIM.

    .venv/bin/python week31_harness2_recursive_harnessing/08_nightly_harness_review/labs/lab08_1_nightly_routine.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harness as H                                                               # noqa: E402
import reviewkit as R                                                             # noqa: E402
from h2out import banner, brain_mode, brain_model, dim, edit, note, ok, result, runs_dir, show_text, step, table, warn  # noqa: E402

banner("Lab 08-1 · The nightly harness review routine", "plain check → select ≤ 3 → Shadow steps → draft → human gate")
dim("EXAMPLE design for this course (not deployed) on the SIM agent; every hotel, alarm and person is fictional")
v1 = H.base_harness()

step(1, "Night 1 — a quiet day: the plain check finds nothing, so the routine stays silent")
quiet_day = R.day(R.QUIET_DAY)
table([[w.wake_id, w.task.id, w.task.family, w.signal] for w in quiet_day], ["wake", "task", "family", "signal"])
n1 = R.nightly_review(quiet_day, v1)
note(f"weak-signal wakes: {len(n1.weak)} · model calls: {n1.model_calls} · draft: {n1.draft is not None}")
print("│ send_to_user " + json.dumps(n1.envelope))
ok("no weak signal → no model call, no draft, notify=false: the routine ends in silence")

step(2, "Night 2 — a busy day: the plain check (code, no model) lists the weak signals")
busy = R.day(R.BUSY_DAY)
weak = R.plain_check(busy)
chosen = R.select_tasks(busy, n=R.N_DEFAULT)
table([[w.wake_id, w.task.id, w.signal, (w.human_reason or "—")[:46], "SELECTED" if w in chosen else
        ("weak, waits (N = 3)" if w in weak else "quiet")] for w in busy],
      ["wake", "task", "signal", "human reason (content)", "tonight"])
note(f"{len(weak)} weak signals, N = {R.N_DEFAULT}: priority rejected_proposal > low_rating > helper_failed"
     f"; a recorded reason breaks ties")

step(3, "Shadow: one Propose–Probe–Compose step per selected wake (read-only tools, probes in the sandbox)")
live = brain_mode() == "ollama"
prop = {chosen[0].task.id: R.live_editor} if live else None
if live:
    dim(f"🦙 LIVE: {brain_model()} writes the INV and DOM candidates for {chosen[0].task.id} (2 calls); the rest stay SIM")
n2 = R.nightly_review(busy, v1, proposer_for=prop)
for rec in n2.records:
    print()
    rec.show()
for e in R.LIVE_LOG:                                  # LIVE only: what the local model actually wrote
    edit(f"LIVE {e['scope']} edit for {e['task']} (credited to {e['credited']}): {e['note']}")
    for p, lines in e["lines"].items():
        for ln in lines[:4]:
            print(f"│ + {p}: {ln[:110]}")
note(f"model calls tonight (executions + editor sessions, SIM count): {n2.model_calls}")

step(4, "The draft version: the reviewer's merge (bench.promote) over tonight's records")
if n2.draft is None:
    warn("no draft tonight: " + "; ".join(n2.promote_notes))
else:
    for n in n2.promote_notes:
        print(f"→ merge: {n}")
    p = R.packet(v1, n2, version="v2", parent="v1")
    show_text("HARNESS.json (state: draft)", p["HARNESS.json"])
    table([[c["path"], c["component"], c["change"], f"{c['delta_tokens']:+d}"] for c in p["changes"]],
          ["path", "comp.", "change", "≈tokens"])
    a0, a1 = p["always_on"]
    note(f"always-on {a0} → {a1} tokens ({a1 - a0:+d}) · prefix hash {p['prefix_hash'][0]} → {p['prefix_hash'][1]}"
         + (" · CHANGED: this version may ship only at a deployment boundary" if p['prefix_hash'][0] != p['prefix_hash'][1] else ""))
    show_text(f"EXPERIENCE entry for {p['experience'][0]['task']} (one of {len(p['experience'])})", p["experience"][0]["md"])
    out = R.write_packet(p, n2.draft, runs_dir(__file__) / "packet_v2")
    ok(f"packet written: {out.relative_to(out.parents[2])}/ (PACKET.md, HARNESS.json, harness/, EXPERIENCE_*.md)")

    step(5, "The gate: a named human publishes the draft to Shadow, or nothing ships")
    reg = R.Registry(v1)
    reg.add_draft("v2", n2.draft, parent="v1")
    tries = [("routine:nightly-harness-review", p["definition_hash"][1], "the routine approves its own draft"),
             ("agent owner, HVAC Expert (fictional)", p["definition_hash"][0], "reviewer read v1's hash, not the draft's"),
             ("agent owner, HVAC Expert (fictional)", p["definition_hash"][1], "reviewer read this draft")]
    for who, h, what in tries:
        code, msg = reg.publish("v2", who, h)
        print(f"{'✓' if code == 200 else '✕'} publish v2 · {what}: {code} — {msg}")
    edit("audit: " + " | ".join(reg.audit))

step(6, "The send_to_user envelope for night 2 (operator channel)")
print("│ " + json.dumps({k: (v if k != "evidence" else v[:3] + [f"… {len(v) - 3} more"]) for k, v in n2.envelope.items()}, ensure_ascii=False))
ok("notify=true only because a draft exists; a KEEP_BASE night sends nothing")
result(f"night 1: silent, 0 model calls · night 2: {len(n2.selected)} wakes harnessed, "
       f"{'draft v2 published to Shadow by a named reviewer' if n2.draft else 'KEEP_BASE'}")
