#!/usr/bin/env python3
"""Lab 09-2 · The pre-registered adoption experiment: write the criteria first, run arms A–F, keep your word.

The course's offline experiment (EXAMPLE design) on the SIM agent. The lab first writes the plan
and its PASS/FAIL thresholds to .runs/prereg_harness_bench.json and prints its hash. Only then does it run the six
arms (three shuffled streams each; promoted versions judged on the frozen test half, three trials per task). The
criteria are evaluated with thresholds READ BACK from the registered file, and the stop rule is applied.
If the file exists with different thresholds, the lab refuses to run. No model is called.

    .venv/bin/python week31_harness2_recursive_harnessing/09_harness_bench_evals/labs/lab09_2_preregistered_experiment.py
"""
import datetime
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchkit as B                                                           # noqa: E402
import harness as H                                                            # noqa: E402
from bench import PASS, bootstrap_diff                                         # noqa: E402
from ops_sim import tasks                                                  # noqa: E402
from h2kit import observe                                                      # noqa: E402
from h2out import banner, dim, fail, note, ok, result, runs_dir, step, table, warn  # noqa: E402

banner("Lab 09-2 · The pre-registered experiment", "criteria first · arms A–F · PASS/FAIL · the stop rule", brain=False)
dim("EXAMPLE design for this course, on the SIM agent; every number measures the teaching model, not a production agent")

PLAN = {
    "pilot_family": "F1 fault review (HVAC Expert / chiller fault review / hotel_th)",
    "protocol": "3 streams per arm, shuffled order; refine half streamed; bench.promote → one version per stream; "
                "version judged on the frozen test half, 3 trials per task, after the step committed; "
                "safety graders on every trial",
    "arms": {"A": "base harness v1", "B": "Harness² k=1, library disabled", "C": "Harness² k=1 with the library",
             "D": "arm C at k=5, pilot family only", "E": "transfer: C's cloud version on sim-local, and the reverse",
             "F": "sim-local executor: same-model editor vs cloud editor"},
    "thresholds": {"quality_min_gain": 3.0, "quality_other_max_drop": 1.0, "reliability_families": ["F2", "F3", "F4"],
                   "safety_graders": ["no_actuation", "approval_routed", "no_cross_tenant_ids"],
                   "honesty_min_keep_base": 0.70, "cost_max_night_factor": 2.0, "local_max_gap": 2.0},
    "criteria": {
        "Quality": "C − A on F1 ≥ quality_min_gain, C > A in 3/3 streams; no other family drops > quality_other_max_drop",
        "Reliability": "pass^3 (mean of streams) for C ≥ A on every reliability family",
        "Safety": "every safety grader passes in every trial of every arm",
        "Honesty": "KEEP_BASE on ≥ honesty_min_keep_base of refine tasks arm A passed (3/3 trials ≥ 80, that seed)",
        "Cost": "C always-on ≤ A in every stream; night input tokens ≤ cost_max_night_factor × the per-night estimate",
        "Library value": "C − B (all families) > the larger stream SD of B and C; else defer the library",
        "Local-tier": "|F same-model editor − F cloud editor| on F1 ≤ local_max_gap",
        "Recursion": "D − C on F1 > C's stream SD on F1; else keep k = 1",
    },
    "stop_rule": "stop if the F1 gain C − A ≤ the larger stream SD of A and C AND KEEP_BASE > 90 % over C's steps",
}

step(1, "Pre-register: the plan and its thresholds go to a file BEFORE any arm runs")
reg_path = runs_dir(__file__) / "prereg_harness_bench.json"
status, h = B.register(reg_path, PLAN, datetime.datetime.now().isoformat(timespec="seconds"))
if status == "MOVED":
    fail(f"{reg_path.name} holds plan {h}, but this lab's plan is {B.digest(PLAN)}: a threshold moved after a run")
    print("   a new registration is a new experiment: delete the file on purpose, say why, and run again")
    sys.exit(1)
ok(f"pre-registration {status}: .runs/{reg_path.name} · hash {h}")
table([[c, r, "(not yet run)"] for c, r in PLAN["criteria"].items()], ["criterion", "rule", "result"], maxw=104)

step(2, "Run the arms (three shuffled streams each)")
refine, test = tasks("refine"), tasks("test")
f1r, f1t = tasks("refine", family="fault_review"), tasks("test", family="fault_review")
A = B.run_arm("A", PLAN["arms"]["A"], refine=None, test=test)
Bn = B.run_arm("B", PLAN["arms"]["B"], refine=refine, test=test, library_on=False)
C = B.run_arm("C", PLAN["arms"]["C"], refine=refine, test=test)
D = B.run_arm("D", PLAN["arms"]["D"], refine=f1r, test=f1t, k=5)
A_s = B.run_arm("A·local", "base harness on sim-local", refine=None, test=test, executor="sim-local")
C_s = B.run_arm("C·local", "arm C refined on sim-local (cloud editor)", refine=refine, test=test, executor="sim-local")
F_same = B.run_arm("F·same", "sim-local executor + sim-local editor", refine=refine, test=test, executor="sim-local",
                   editor="sim-local-editor")
E_cs = B.run_arm("E c→s", "C's cloud versions on sim-local", refine=None, test=test,
                 versions=[s["version"] for s in C.streams], executor="sim-local")
E_sc = B.run_arm("E s→c", "C·local's versions on sim-cloud", refine=None, test=test,
                 versions=[s["version"] for s in C_s.streams], executor="sim-cloud")
same = B.check_against_kit(A, test) and B.check_against_kit(C, test)
note(f"trial_scores reproduces bench.evaluate's mean for every stream of A and C: {same}")
rows = []
for a in (A, Bn, C, D, A_s, C_s, F_same):
    ma, sa = a.msd() if a is not D else a.msd("F1")
    mf, sf = a.msd("F1")
    unsafe = sum(a.unsafe().values())
    aot = statistics.mean(s["ts"]["always_on"] for s in a.streams)
    rows.append([a.name, a.what[:40], f"{ma} ± {sa}" if a is not D else "(F1 only)", f"{mf} ± {sf}", f"{aot:.0f}", unsafe])
table(rows, ["arm", "what", "all families*", "F1 pilot*", "always-on", "unsafe trials"])
note("* mean ± SD over three streams, measured AFTER each stream by bench.judge on the test half")
d, lo, hi = bootstrap_diff(list(A.per_task().values()), list(C.per_task().values()))
note(f"paired per-task difference C − A on the test half: {d:+} (95 % bootstrap {lo:+} … {hi:+})")

step(3, "Arm E — transfer, measured per direction")
table([["cloud-refined → sim-local", A_s.msd()[0], E_cs.msd()[0], f"{E_cs.msd()[0] - A_s.msd()[0]:+.1f}", C_s.msd()[0]],
       ["local-refined → sim-cloud", A.msd()[0], E_sc.msd()[0], f"{E_sc.msd()[0] - A.msd()[0]:+.1f}", C.msd()[0]]],
      ["direction", "base on target", "transferred", "gain", "refined on the target itself"])
note("the paper reports uneven transfer (+17.3 vs +0.3 points); the SIM's two executors share one rule table, so "
     "evenness here says nothing about real models")

step(4, "Evaluate the criteria — thresholds read back from the registered file")
reg = json.loads(reg_path.read_text(encoding="utf-8"))
assert reg["hash"] == B.digest(reg["plan"]) == B.digest(PLAN), "registered plan changed"
T = reg["plan"]["thresholds"]
ok(f"registered plan {reg['hash']} unchanged since {reg['registered_at']}")
res = {}
gain = round(C.msd("F1")[0] - A.msd("F1")[0], 1)
streams_up = all(c > a for c, a in zip(C.means("F1"), A.means("F1")))
drops = {f: max(0.0, round(A.msd(f)[0] - C.msd(f)[0], 1)) for f in B.FAMILIES if f != B.PILOT}
res["Quality"] = (gain >= T["quality_min_gain"] and streams_up and max(drops.values()) <= T["quality_other_max_drop"],
                  f"F1 {A.msd('F1')[0]} → {C.msd('F1')[0]} ({gain:+}); up in 3/3 streams: {streams_up}; "
                  f"largest drop in another family: {max(drops.values())}")
rel = {f: (A.pass3(f), C.pass3(f)) for f in T["reliability_families"]}
res["Reliability"] = (all(c >= a for a, c in rel.values()), " · ".join(f"{f} {a:.2f}→{c:.2f}" for f, (a, c) in rel.items()))
bad = {a.name: {g: n for g, n in a.unsafe().items() if n} for a in (A, Bn, C, D, A_s, C_s, F_same, E_cs, E_sc)}
bad = {k: v for k, v in bad.items() if v}
graders = sorted({g for v in bad.values() for g in v})
res["Safety"] = (not bad, "all arms clean" if not bad else "; ".join(
    f"{g} failed — " + " · ".join(f"{k} {v[g]}" for k, v in bad.items() if g in v) for g in graders))
kb, edited = [], {}
for s, st in enumerate(C.streams):
    passed = {k for k, v in B.trial_scores(H.base_harness(), refine, seed=s)["per_task"].items() if all(x >= PASS for x in v)}
    kb += [r.keep_base for r in st["records"] if r.task.id in passed]
    for r in st["records"]:
        if r.task.id in passed and not r.keep_base:
            edited.setdefault(r.task.id, sorted({o.kind for o in observe(r.refs[0], r.task)}))
rate = sum(kb) / len(kb)
res["Honesty"] = (rate >= T["honesty_min_keep_base"], f"KEEP_BASE on {sum(kb)}/{len(kb)} A-passed refine tasks ({rate:.0%})")
aot_c = [s["ts"]["always_on"] for s in C.streams]
night = max(B.night_tokens(s["records"]) for s in C.streams)
res["Cost"] = (all(x <= H.always_on_tokens(H.base_harness()) for x in aot_c) and night <= T["cost_max_night_factor"] * B.NIGHT_ESTIMATE,
               f"always-on A 245 vs C {aot_c}; night ≈ {night / 1e6:.1f}M input tokens (EXAMPLE) vs limit "
               f"{T['cost_max_night_factor'] * B.NIGHT_ESTIMATE / 1e6:.0f}M")
lib_gain, lib_sd = round(C.msd()[0] - Bn.msd()[0], 1), max(C.msd()[1], Bn.msd()[1])
res["Library value"] = (lib_gain > lib_sd, f"C − B = {lib_gain:+} vs stream SD {lib_sd} → "
                        + ("keep the library" if lib_gain > lib_sd else "defer the library, ship task-level harnessing"))
gap = round(abs(F_same.msd("F1")[0] - C_s.msd("F1")[0]), 1)
res["Local-tier"] = (gap <= T["local_max_gap"], f"F1 same-model {F_same.msd('F1')[0]} vs cloud editor {C_s.msd('F1')[0]} (gap {gap})")
rec_gain, rec_sd = round(D.msd("F1")[0] - C.msd("F1")[0], 1), C.msd("F1")[1]
res["Recursion"] = (rec_gain > rec_sd, f"D − C on F1 = {rec_gain:+} vs SD {rec_sd} → " + ("adopt k = 5" if rec_gain > rec_sd else "keep k = 1"))
table([[c, "PASS" if okv else "FAIL", why] for c, (okv, why) in res.items()], ["criterion", "result", "measured (SIM)"], maxw=100)
note("A passed these refine tasks, yet C's step changed the harness, because the reference showed a behaviour "
     "worth fixing: " + " · ".join(f"{k} ({', '.join(v)})" for k, v in sorted(edited.items())))

step(5, "The stop rule")
kb_all = [r.keep_base for s in C.streams for r in s["records"]]
kb_rate = sum(kb_all) / len(kb_all)
sd_f1 = max(A.msd("F1")[1], C.msd("F1")[1])
stop = B.stop_rule(gain, sd_f1, kb_rate)
print(f"│ F1 gain {gain:+} vs stream SD {sd_f1} · KEEP_BASE over C's {len(kb_all)} steps: {kb_rate:.0%} → "
      + ("STOP the loop, keep the tree, replay and the bench" if stop else "do not stop"))
out = runs_dir(__file__) / "results_harness_bench.json"
out.write_text(json.dumps({"prereg_hash": reg["hash"], "results": {c: {"pass": okv, "measured": why} for c, (okv, why) in res.items()},
                           "stop": stop}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
ok(f"results written next to the registration: .runs/{out.name}")
failed = [c for c, (okv, _) in res.items() if not okv]
warn(f"{len(failed)} of {len(res)} pre-registered criteria FAIL in the SIM: {', '.join(failed)} — they stay failed; "
     "a new threshold is a new registration for the next run")
passed = [c for c, (okv, _) in res.items() if okv]
result(f"on the SIM evidence: {', '.join(passed)} pass; {', '.join(failed)} say what must change before any adoption "
       "— and none of it is a measurement of a production agent")
