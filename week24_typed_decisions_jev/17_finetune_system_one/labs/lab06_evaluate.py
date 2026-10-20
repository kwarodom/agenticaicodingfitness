#!/usr/bin/env python3
"""Lab 17-6 · The exam: did fine-tuning work? Held-out tickets, error bars, and what got worse.

  1. every engine on the 66 held-out tickets (48 hand-written + Module 16's 18), with 95 % intervals
  2. paired: which tickets each fine-tune FIXED and which it BROKE, and is the difference luck?
  3. the other two questions — safety recall and urgency — and each language separately
  4. confidence: calibration error, and a threshold fitted on VALIDATION, checked on held-out
  5. forgetting: a job it never trained on (review sentiment), before vs after
  6. speed: did the fine-tune get slower?
Results go to .runs/lab06_results.json for the ship/hold gate (Lab 17-7).

Needs your checkpoints from Labs 17-4/17-5 for live answers; otherwise replays the 2026-10-17 run.
Run: .venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab06_evaluate.py
"""
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import ftkit as F  # noqa: E402  (first: may re-run this lab inside week24_typed_decisions_jev/.venv-local)
from jevkit import banner, step, table  # noqa: E402

banner("Lab 17-6 · evaluate", "66 held-out tickets · error bars · fixed vs broke · calibration · forgetting")
F.L.banner_local()
held = F.split("test") + F.split("m16")
val = F.split("val")
PAIRS = {"laya_ft": "laya_router", "gliner_lora": "gliner_base", "gliner_full": "gliner_base"}
ENGINES = ["majority", "keywords", "laya_router", "laya_ft", "gliner_base", "gliner_lora", "gliner_full", "jev"]
results = {"held_out": len(held), "engines": {}}

# ── 1 · the scoreboard ────────────────────────────────────────────────────────
step(1, "every engine on the 66 held-out tickets (queue)")
P, SRC = {}, {}
for eng in ENGINES:
    p, src = F.predict(eng, held, majority="hvac")
    if p:
        P[eng], SRC[eng] = p, src
table([F.summary_row(e, held, P[e], SRC[e]) for e in P], F.SUMMARY_HEADERS)
missing = [e for e in ENGINES if e not in P]
if missing:
    print(f"» not available here (train them in Labs 17-4/17-5): {', '.join(missing)}")
print("» Read the 95 % column first. 66 tickets is a class-sized exam: intervals of ±10 points are normal,")
print("  so two engines whose intervals overlap heavily are not yet shown to be different.")

# ── 2 · paired: fixed vs broke ────────────────────────────────────────────────
step(2, "paired comparison — same tickets, before vs after")
rows = []
for ft, base in PAIRS.items():
    if ft in P and base in P:
        d = F.paired(held, P[base], P[ft])
        a0, n = F.accuracy(held, P[base])
        a1, _ = F.accuracy(held, P[ft])
        rows.append([f"{base} → {ft}", f"{a0}/{n} → {a1}/{n}", f"{(a1 - a0) / n * 100:+.0f} pts",
                     len(d["fixed"]), len(d["broke"]), f"{d['p']:.4f}"])
        results["engines"].setdefault(ft, {}).update(
            queue_gain_points=round((a1 - a0) / n * 100, 1), queue_mcnemar_p=d["p"], queue_broke=len(d["broke"]),
            broke=[t["id"] for t in d["broke"]], fixed=[t["id"] for t in d["fixed"]])
table(rows, ["comparison", "queue", "change", "fixed", "broke", "McNemar p"])
print("» McNemar looks only at tickets where the two disagree. p < 0.05: unlikely to be luck.")
for ft in PAIRS:
    if ft in results["engines"] and results["engines"][ft]["broke"]:
        print(f"» {ft} broke:")
        for tid in results["engines"][ft]["broke"]:
            t = next(x for x in held if x["id"] == tid)
            print(f"   ✗ {tid:<7} gold {t['queue']:<12} now {P[ft][tid]['queue']:<12} {t['text'][:58]}"
                  + (f"  [note: {t['note'][:40]}…]" if t.get("note") else ""))

# ── 3 · the other questions, per language ─────────────────────────────────────
step(3, "safety recall, urgency, and each language on its own")
rows = []
for eng in P:
    sr, ns = F.safety_recall(held, P[eng])
    fa = sum(1 for t in held if P[eng][t["id"]]["safety"] and not t["safety"])
    uok, un = F.accuracy(held, P[eng], "urgency")
    mae = statistics.mean(abs((P[eng][t["id"]]["urgency"] if P[eng][t["id"]]["urgency"] is not None else 1) - t["urgency"]) for t in held)
    en, ne = F.accuracy(held, P[eng], lang="en")
    th, nt = F.accuracy(held, P[eng], lang="th")
    rows.append([eng, f"{sr}/{ns}", fa, f"{uok}/{un}", f"{mae:.2f}", f"{en / ne:.0%}", f"{th / nt:.0%}"])
    results["engines"].setdefault(eng, {}).update(
        queue_acc=F.accuracy(held, P[eng])[0] / len(held), safety_recall=sr / ns, false_alarms=fa,
        urgency_acc=uok / un, en_acc=en / ne, th_acc=th / nt, th_minus_en_points=round((th / nt - en / ne) * 100, 1),
        median_ms=F.median_ms(P[eng]), source=SRC[eng])
table(rows, ["engine", "safety recall", "false alarms", "urgency", "urgency MAE", "queue EN", "queue TH"])
print("» Safety recall is the row a hotel cares about most: one missed fire outweighs ten tidy queues.")
for eng in ("laya_ft", "gliner_lora", "gliner_full", "jev"):
    if eng in P:
        miss = [t for t in held if t["safety"] and not P[eng][t["id"]]["safety"]]
        results["engines"][eng]["safety_missed"] = [t["id"] for t in miss]
        if miss:
            print(f"   {eng:<12} missed: " + " · ".join(f"{t['id']} “{t['text'][:34]}…”" for t in miss[:4]))
print("» Read the misses, not just the count: water creeping toward a switchboard room and an ice machine")
print("  leaking into a floor socket are 'water near electricity' in words no training situation used.")
print("  That is the next data to write — then retrain and re-grade under the SAME criteria.")

# ── 4 · confidence ────────────────────────────────────────────────────────────
step(4, "confidence: calibration, and a threshold fitted on VALIDATION, checked on held-out")
rows = []
for eng in ("laya_router", "laya_ft", "gliner_base", "gliner_lora", "gliner_full", "jev"):
    if eng not in P:
        continue
    pv, _ = F.predict(eng, val) if eng != "jev" else ({}, "")
    cut = F.threshold_for(val, pv, target_error=0.05) if pv else None
    cov, err = F.coverage_at(held, P[eng], cut)
    e = F.queue_ece(held, P[eng])
    rows.append([eng, f"{e:.2f}" if e is not None else "—", f"{cut:.3f}" if cut is not None else "—",
                 f"{cov:.0%}" if cut is not None else "—", f"{err:.0%}" if cut is not None else "—"])
    results["engines"].setdefault(eng, {}).update(queue_ece=e, cut_from_val=cut, held_coverage=cov, held_error_at_cut=err)
table(rows, ["engine", "queue ECE (held-out)", "cut for ≤5 % error (val)", "held-out answered", "error among them"])
print("» The cut is fitted where it is cheap to be wrong (val) and checked where it counts (held-out).")
print("  If 'error among them' is far above 5 %, val did not look like reality — keep the human in the loop.")
print("» GLiNER2.5-Decide's confidence is one score for the winning label, not a distribution: its ECE")
print("  is computed the same way but means less (Module 16).")

# ── 5 · forgetting ────────────────────────────────────────────────────────────
step(5, "forgetting: review sentiment — a job none of the fine-tunes trained on")
probe = F.split("probe")
PQ = {"sentiment": probe["question"]}
prec = F.Recorder("lab06_probe")
rows = []
for eng in ("laya_base", "laya_ft", "gliner_base", "gliner_lora", "gliner_full"):
    key = f"{eng}:{F.fingerprint(eng)}"
    if F.available(eng):
        m = F._load(eng)
        got = []
        for it in probe["items"]:
            if eng.startswith("laya"):
                got.append(m.predict({"message": it["text"]}, PQ)["answers"]["sentiment"]["choice"])
            else:
                got.append(m.classify_text(it["text"], F.L.to_gliner_schema(PQ))["sentiment"])
        prec.data[eng], prec.dirty, src = got, True, "LOCAL"
    elif eng in prec.data:
        got, src = prec.data[eng], "RECORDED"
    else:
        continue
    ok = sum(g == it["label"] for g, it in zip(got, probe["items"]))
    rows.append([eng, f"{ok}/{len(got)}", src])
    results["engines"].setdefault(eng, {})["probe_ok"] = ok
prec.save()
table(rows, ["engine", "sentiment correct (12)", "source"])
for ft, base in (("laya_ft", "laya_base"), ("gliner_lora", "gliner_base"), ("gliner_full", "gliner_base")):
    if ft in results["engines"] and "probe_ok" in results["engines"][ft] and "probe_ok" in results["engines"].get(base, {}):
        results["engines"][ft]["probe_drop"] = results["engines"][base]["probe_ok"] - results["engines"][ft]["probe_ok"]
print("» Compared with the SAME weights before training (laya_base, gliner_base). Twelve reviews can only")
print("  catch a big loss: ±1 is noise, ±3 is a signal. Note the LoRA row: while the adapter is ON it changes")
print("  every task the model runs, not just tickets. Switching it OFF gives the untouched base back — so")
print("  serve other tasks with the adapter disabled (Lab 17-5, step 5), never 'on for everything'.")

# ── 6 · speed ─────────────────────────────────────────────────────────────────
step(6, "speed — median time per ticket (3 questions), this machine")
rows = []
for ft, base in PAIRS.items():
    a, b = F.median_ms(P.get(base, {})), F.median_ms(P.get(ft, {}))
    if a and b:
        rows.append([f"{base} → {ft}", f"{a:.0f} ms → {b:.0f} ms", f"× {b / a:.2f}"])
        results["engines"][ft]["latency_ratio"] = round(b / a, 2)
table(rows, ["", "median", "ratio"])
print("» Same architecture, same size: a fine-tune should cost nothing at inference. A LoRA adapter that")
print("  is not merged adds a few small matrix multiplies; Laya's Router → one checkpoint can even be faster.")

F.RUNS.mkdir(exist_ok=True)
(F.RUNS / "lab06_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
if os.environ.get("JEV_RECORD") == "1":
    (F.RECORDED / "lab06_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
print("» saved .runs/lab06_results.json → Lab 17-7 grades it against the locked criteria.")
