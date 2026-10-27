#!/usr/bin/env python3
"""Lab 10-2 · Defend the version: the integrity suite, the registered criteria, and a one-page review packet.

Rebuilds Lab 10-1's v2 deterministically, checks the capstone registration, runs the seven integrity tests,
grades v2 against the registered criteria on the frozen F1 test half (measured after the runs by bench.judge),
applies the stop rule, and writes .runs/capstone/REVIEW.md — one page a named reviewer can sign. No model is called.

    .venv/bin/python week31_harness2_recursive_harnessing/10_capstone_harness_review/labs/lab10_2_defend_the_version.py
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capkit as K                                                               # noqa: E402
from capkit import B, H                                                          # noqa: E402
from h2out import banner, dim, fail, note, ok, result, runs_dir, step, table  # noqa: E402

banner("Lab 10-2 · Defend the version", "integrity suite · registered criteria · stop rule · REVIEW.md", brain=False)
dim("EXAMPLE design for this course, on the SIM agent; scores are measured after the runs, never inside the loop")
out = runs_dir(__file__) / "capstone"
out.mkdir(parents=True, exist_ok=True)

step(1, "The version and its registration")
v2, draft, notes, nights = K.build_version()
status, h = K.registration(out)
if status == "MOVED":
    fail("prereg_capstone.json holds a different plan: a threshold moved after a run")
    sys.exit(1)
reg = json.loads((out / "prereg_capstone.json").read_text(encoding="utf-8"))
assert reg["hash"] == B.digest(reg["plan"]) == B.digest(K.PLAN)
T = reg["plan"]["thresholds"]
saved = out / "v2"
same = saved.exists() and H.read_tree(saved) == v2
ok(f"registration {status} · hash {h} · registered {reg['registered_at']} · v2 {H.definition_hash(v2)}"
   + (" · matches Lab 10-1's saved tree" if same else " · (run Lab 10-1 to save the tree and packet)"))

step(2, "The integrity suite (seven tests, each a code check)")
suite = K.integrity_suite(v2, nights)
table([[name, "PASS" if okv else "FAIL", why] for name, okv, why in suite], ["test", "result", "evidence (SIM)"], maxw=112)

step(3, "The registered criteria on the frozen F1 test half (3 seeds × 3 trials)")
base_v, cand_v = K.f1_view(K.V1), K.f1_view(v2)
bm = [B.family_view(v)["F1"]["mean"] for v in base_v]
cm = [B.family_view(v)["F1"]["mean"] for v in cand_v]
gain = round(statistics.mean(cm) - statistics.mean(bm), 1)
res = {}
res["Quality"] = (gain >= T["quality_min_gain"] and all(c > b for c, b in zip(cm, bm)),
                  f"F1 v1 {bm} → v2 {cm} · gain {gain:+} · up at every seed: {all(c > b for c, b in zip(cm, bm))}")
unsafe2 = {g: sum(v["unsafe"][g] for v in cand_v) for g in T["safety_graders"]}
unsafe1 = {g: sum(v["unsafe"][g] for v in base_v) for g in T["safety_graders"]}
res["Safety"] = (not any(unsafe2.values()), "v2 / v1 failures (v1 not graded): "
                 + " · ".join(f"{g} {unsafe2[g]} / {unsafe1[g]}" for g in unsafe2))
honesty = next(s for s in suite if s[0].startswith("KEEP_BASE"))
res["Honesty"] = (honesty[1], honesty[2])
night = max(B.night_tokens(n.records) for n in nights if n.records)
aot = H.always_on_tokens(v2)
res["Cost"] = (aot <= T["always_on_cap"] and night <= T["cost_max_night_factor"] * B.NIGHT_ESTIMATE,
               f"always-on {aot} ≤ cap {T['always_on_cap']} · busiest night ≈ {night / 1e6:.1f}M input tokens (EXAMPLE) "
               f"≤ {T['cost_max_night_factor'] * B.NIGHT_ESTIMATE / 1e6:.0f}M")
res["Integrity"] = (all(okv for _, okv, _ in suite), f"{sum(okv for _, okv, _ in suite)}/{len(suite)} tests pass")
table([[c, "PASS" if okv else "FAIL", why] for c, (okv, why) in res.items()], ["criterion", "result", "measured (SIM)"], maxw=112)
p3 = (statistics.mean(B.family_view(v)["F1"]["pass3"] for v in base_v), statistics.mean(B.family_view(v)["F1"]["pass3"] for v in cand_v))
note(f"F1 pass^3 (diagnostic for this family): v1 {p3[0]:.2f} → v2 {p3[1]:.2f}")
uncapped, _ = K.weekly_merge(nights, cap=None)
um = [B.family_view(v)["F1"]["mean"] for v in K.f1_view(uncapped)]
note(f"the price of the cap: the uncapped merge scores {statistics.mean(um):.1f} at {H.always_on_tokens(uncapped)} always-on "
     f"tokens; v2 scores {statistics.mean(cm):.1f} at {aot}")

step(4, "The stop rule")
sd = max(statistics.stdev(bm), statistics.stdev(cm))
steps = [r.keep_base for n in nights for r in n.records]
kb = sum(steps) / len(steps)
stop = B.stop_rule(gain, sd, kb)
print(f"│ F1 gain {gain:+} vs SD over seeds {sd:.1f} · KEEP_BASE over the week's {len(steps)} steps: {kb:.0%} → "
      + ("STOP" if stop else "do not stop"))

step(5, "The one-page review packet")
passed = all(okv for okv, _ in res.values())
decision = "keep v2 in Shadow and run the two-week Shadow gate" if passed else "hold v2 in Shadow: no promotion to Assist"
review = K.write_review(out / "REVIEW.md", {
    "title": f"v2 · {K.PILOT_KEY} · sim-cloud",
    "Version": [f"- version v2 · parent v1 · definition hash {H.definition_hash(v2)} · prefix hash {H.prefix_hash(v2)} "
                f"({'unchanged' if H.prefix_hash(v2) == H.prefix_hash(K.V1) else 'changed'})",
                f"- created_by {K.R.ROUTINE} · approved_by {K.REVIEWER} · state shadow",
                f"- registration {h} (written before the Shadow week)"],
    "What changed": [f"- {c['change']} {c['path']} ({c['component']}, {c['delta_tokens']:+d} tokens)" for c in H.diff(K.V1, v2)]
                    + [f"- always-on {H.always_on_tokens(K.V1)} → {aot} tokens"],
    "Evidence": [f"- night {i}: " + "; ".join(f"{r.task.id} {r.task.signal} → "
                                                + ("KEEP_BASE" if r.keep_base else ", ".join(f"{x['component']} {x['verdict']}" for x in r.rows))
                                                for r in n.records) for i, n in enumerate(nights, 1) if n.records]
                + [f"- reviewer change: scripts/open_work_orders.py; CMMS check on F1-01, F1-03 (refine, 10 seeds): "
                   f"{K.cmms_rate(draft, ['F1-01', 'F1-03']):.0%} → {K.cmms_rate(v2, ['F1-01', 'F1-03']):.0%}"],
    "Integrity suite": [f"- {'PASS' if okv else 'FAIL'} {name}: {why}" for name, okv, why in suite],
    "Registered criteria (F1 test half, after the runs)": [f"- {'PASS' if okv else 'FAIL'} {c}: {why}" for c, (okv, why) in res.items()],
    "Decision": [f"- {decision}" + ("" if passed else " — Safety and Integrity fail (approval_routed); fix with refine-half "
                                     "evidence and re-test under a new registration"), f"- stop rule: {'STOP' if stop else 'do not stop'} (gain {gain:+} vs SD {sd:.1f}, KEEP_BASE {kb:.0%})",
                 "- rollback plan: if v2 reaches Serve and a trigger fires, move the Serve pointer back to v1 (one pointer change, one re-warm)"],
    "Not claimed": ["- no number here measures Harness², a real model or a production agent; the agent, editors and judge are SIM",
                    "- v1 fails approval_routed on the test half: a finding about today's harness, escalated separately",
                    "- the workspace fact inside skills/chiller_fault_review (3.0 K) must move to the knowledge graph before Write-back"],
})
ok(f"wrote .runs/capstone/REVIEW.md ({len(review.read_text(encoding='utf-8').splitlines())} lines) — measurement belongs "
   "here, at the gate; the tree and the EXPERIENCE entries stay free of it")

if not passed:
    step(6, "A failed criterion is a finding: where the evidence for a fix must come from")
    shadow = K.refine_evidence(v2)
    print(f"│ v2 on the REFINE half (F1, 10 mirrored seeds): approval_routed failures {shadow['fails']} of "
          f"{shadow['proposals']} ActionProposals · tasks: {', '.join(shadow['tasks']) or '—'}")
    note("the test half found the failure; only refine-half (Shadow) evidence may justify the change, and the re-test "
         "is a new registration — every look at the test half spends a little of its value")
    note("an instruction that is repeatedly ignored is an 'ignored invariant': its cheapest effective "
         "component is a plugin that enforces, not more prose — the capstone checklist asks you to build it")
result(("DEFENDED: " if passed else "NOT YET: ") + f"{sum(okv for okv, _ in res.values())}/{len(res)} registered criteria pass · {decision}")
