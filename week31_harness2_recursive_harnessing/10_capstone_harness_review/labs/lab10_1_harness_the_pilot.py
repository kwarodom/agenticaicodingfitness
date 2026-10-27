#!/usr/bin/env python3
"""Lab 10-1 · Harness the pilot family: register first, a Shadow week, a capped draft, the reviewer's change.

The capstone applies Modules 08 and 09 to the course's EXAMPLE pilot family: HVAC Expert · chiller
fault review · Thai hotel tenant class. It registers the capstone plan BEFORE running, runs four Shadow nights of
the routine, merges the week under an always-on cap, records one reviewer change with its own probe on the refine
half, and sends the result through the named-reviewer gate. Artefacts go to .runs/capstone/. No model is called.

    .venv/bin/python week31_harness2_recursive_harnessing/10_capstone_harness_review/labs/lab10_1_harness_the_pilot.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capkit as K                                                               # noqa: E402
from capkit import H, R                                                          # noqa: E402
from h2out import banner, dim, edit, fail, note, ok, result, runs_dir, step, table, warn  # noqa: E402

banner("Lab 10-1 · Harness the pilot family", f"{K.PILOT_KEY} · register → Shadow week → capped draft → gate",
       brain=False)
dim("EXAMPLE design for this course, on the SIM agent; every hotel, alarm and person is fictional")
out = runs_dir(__file__) / "capstone"
out.mkdir(parents=True, exist_ok=True)

step(1, "Register the capstone plan before anything runs")
status, h = K.registration(out)
if status == "MOVED":
    fail(f"prereg_capstone.json holds plan {h}; this lab's plan is {K.B.digest(K.PLAN)} — a threshold moved after a run")
    sys.exit(1)
ok(f"capstone registration {status}: .runs/capstone/prereg_capstone.json · hash {h} · {K.lab09_agreement()}")
table([[c, r] for c, r in K.PLAN["criteria"].items()], ["criterion", "rule (thresholds in the file)"], maxw=100)
note(f"always-on cap = v1's {K.CAP} tokens (always-on flat or lower) · k = 1 and no library, as Lab 09-2 decided")

step(2, "The Shadow week: four nights of the routine on the pilot agent's wakes")
nights = K.shadow_week()
rows = []
for i, n in enumerate(nights, 1):
    outcome = ", ".join(f"{r.task.id} {'KEEP_BASE' if r.keep_base else 'COMPOSED'}" for r in n.records) or "—"
    rows.append([i, ", ".join(w.task.id for w in n.wakes), len(n.weak), outcome, n.model_calls, n.envelope["notify"]])
table(rows, ["night", "wakes", "weak", "harnessed tonight", "model calls", "notify"], maxw=48)
note("F1-02 and F1-05 carry no human signal, so the routine never picks them; night 4 is silent")

step(3, "The weekly merge, under the registered cap")
draft, notes = K.weekly_merge(nights)
uncapped, _ = K.weekly_merge(nights, cap=None)
for n in notes:
    print(f"→ merge: {n}")
table([["uncapped merge", H.always_on_tokens(uncapped), ", ".join(H.changed_components(K.V1, uncapped))],
       ["capped merge (the draft)", H.always_on_tokens(draft), ", ".join(H.changed_components(K.V1, draft))]],
      ["merge", "always-on", "components changed"])
warn("the cap refused the trigger line and both guidance lines: the skill and the units script stay; the always-on "
     "CMMS rule is gone, so only the skill (when it fires) still asks for the maintenance check")

step(4, "The reviewer's change: the refused CMMS line becomes a script, probed on the refine half")
v2 = K.move_cmms_to_script(draft)
edit("add scripts/open_work_orders.py (h2kit.SCRIPT_OPEN_WO) — returns the equipment's open work orders in a fixed shape")
ids = ["F1-01", "F1-03"]
rows = [[name, H.always_on_tokens(t), f"{K.cmms_rate(t, ids):.0%}", H.validate_tree(t) or "valid"]
        for name, t in (("v1", K.V1), ("capped draft", draft), ("draft + script (v2)", v2))]
table(rows, ["tree", "always-on", "cmms checked (F1-01, F1-03 × 10 seeds)", "validate_tree"])
ok("the maintenance check is back, in code, at zero always-on cost — the probe used refine tasks only")

step(5, "Packet and gate: the reviewer's change makes a new hash, so the reviewer must re-read")
week = R.Night([w for n in nights for w in n.wakes], [w for n in nights for w in n.weak],
               [w for n in nights for w in n.selected], [r for n in nights for r in n.records], v2,
               notes + ["reviewer change: add scripts/open_work_orders.py (refused CMMS line moved to C7)"],
               sum(n.model_calls for n in nights))
p = R.packet(K.V1, week, version="v2", parent="v1")
R.write_packet(p, v2, out / "packet_v2")
H.write_tree(v2, out / "v2")
reg = R.Registry(K.V1)
reg.add_draft("v2", v2, parent="v1")
for read, what in ((H.definition_hash(draft), "the routine's draft (before the change)"), (H.definition_hash(v2), "v2 as changed")):
    code, msg = reg.publish("v2", K.REVIEWER, read)
    print(f"{'✓' if code == 200 else '✕'} publish after reading {what}: {code} — {msg}")
a0, a1 = p["always_on"]
note(f"v2 · definition {p['definition_hash'][1]} · prefix {p['prefix_hash'][0]} → {p['prefix_hash'][1]} "
     f"({'same: no cold prefix' if p['prefix_hash'][0] == p['prefix_hash'][1] else 'CHANGED'}) · always-on {a0} → {a1}")
ok("packet and tree written: .runs/capstone/packet_v2/ and .runs/capstone/v2/")
result(f"v2 for {K.PILOT_KEY}: {', '.join(p['components'])} changed, always-on {a1 - a0:+d}, in Shadow, "
       f"approved_by {K.REVIEWER} — Lab 10-2 defends it")
