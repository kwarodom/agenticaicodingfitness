#!/usr/bin/env python3
"""Lab 10-2 · Apply and measure: your defense configuration against the full red-team suite, with evidence per threat.

Reads YOUR data/capstone.yaml (deliverable a) and data/threat_model.yaml (deliverable b), then:
  1. the switches you turned on, the reason you gave for each, and the cost against the GM's budget
  2. the full suite (12 attacks, 8 benign tasks) on the SIM brain → the scorecard (deliverable c)
  3. for every threat, which switch backs each mitigation, and whether that switch ALONE stops the threat's
     scenario — the evidence a reviewer will ask for
  4. defense in depth for the critical-impact threats, and the attacks no threat covers yet
  5. on 🦙 LIVE: one spot check of `live_spot_check` on a real local model (≤ 6 model calls)

Everything acts on a simulated hotel. Options: --config PATH --tm PATH (default: this module's data/ files).

Run: .venv/bin/python week28_maestro_mas_security/10_capstone_secure_ensemble/labs/lab10_2_apply_and_measure.py
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capkit as ck  # noqa: E402
from maskit import bar, banner, brain_mode, brain_model, defend, dim, fail, note, ok, result, step, table, warn  # noqa: E402
from redteam import scorecard  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(ck.CONFIG))
ap.add_argument("--tm", default=str(ck.THREAT_MODEL))
args = ap.parse_args()

t0 = time.time()
banner("Lab 10-2 · apply and measure", "your capstone.yaml against the red-team suite · evidence per threat")
dim("the 20-scenario suite runs on the SIM brain; LIVE (if on) adds one spot check at the end")

step(1, f"your configuration · {Path(args.config).name}")
cfg = ck.load_config(args.config)
for e in cfg["errors"]:
    fail(e)
d, human = cfg["defenses"], cfg["human"]
total, rows = ck.cost(d)
table([[n, c, cfg["why"].get(n) or "— (no reason given)"] for n, c, _ in rows], ["switch", "cost", "why (yours)"], maxw=78)
note(f"approver behind HITL: {human}" + ("" if d.hitl else " (HITL is off, so the approver never gets asked)"))
print(f"◆ cost {bar(total, ck.BUDGET + 4)} {total} / {ck.BUDGET} points (the course's price list)")
(ok if total <= ck.BUDGET else fail)(f"cost {total} {'≤' if total <= ck.BUDGET else '>'} budget {ck.BUDGET}")

step(2, "the red-team suite on your configuration (SIM brain)")
outs = ck.suite(d, human)
sc = scorecard(outs, f"{len(d.enabled())} switches on · approver {human} · SIM brain")
for o in outs:
    if o.scenario.kind == "attack" and o.harms:
        b = ck.blockers(o.scenario.id, human)
        warn(f"{o.scenario.id} still harms · switches that would stop it alone: {', '.join(b)}")

step(3, f"evidence: does each mitigation's switch stop its threat's scenario on its own? · {Path(args.tm).name}")
tm, tm_errs = ck.load_tm(args.tm)
for e in tm_errs:
    fail(e)
ev = ck.evidence(tm, cfg) if tm.get("threats") and not tm_errs else []
if not tm_errs:
    ck.evidence_table(tm, ev)

step(4, "defense in depth: critical-impact threats (impact 5) need two independent controls")
threats = tm.get("threats", []) or [] if not tm_errs else []
for t in [t for t in threats if ck.is_critical_impact(t)]:
    b = sorted(ck.backed(ev, t["id"]))
    could = [n for n in ck.blockers(ck.scenarios_of(t)[0], human)] if ck.scenarios_of(t) else []
    msg = f"{t['id']} {t['title'][:58]} · backed by {', '.join(b) or 'nothing'}"
    if len(b) >= 2:
        defend(msg)
    else:
        fail(msg + (f" · candidates: {', '.join(could)}" if could else ""))
miss = ck.unmapped(tm)
if miss:
    warn(f"attacks no threat covers yet: {', '.join(miss)} — the suite found risks the model missed (MAESTRO step 6)")
else:
    ok("every attack scenario is covered by a threat")

step(5, "LIVE spot check on a real model (only when the runner is on 🦙 LIVE)")
if brain_mode() == "ollama":
    sid = cfg["live_spot_check"]
    s = ck.BY_ID[sid]
    note(f"{sid} · {s.title} · {brain_model()} · at most 6 model calls · a rerun can differ")
    o = ck.run_one(sid, d, human, brain="ollama", show=True)
    if s.kind == "attack":
        (defend if not o.harms else fail)(f"LIVE {sid}: " + ("no harm" if not o.harms else ", ".join(o.harms))
                                         + f" · {o.denials} denial(s) at the PEP")
    else:
        (ok if o.done else fail)(f"LIVE {sid}: " + ("task done" if o.done else "task NOT done — the model "
                                                    "answered differently, or a control blocked real work"))
else:
    dim("SIM brain: skipped. Flip the runner to 🦙 LIVE and rerun to try `live_spot_check` "
        f"({cfg['live_spot_check']}) on a local model; change it in capstone.yaml to try A02 or B01")

result(f"ASR {sc['succeeded']}/{sc['attacks']} · utility {sc['done']}/{sc['benign']} · cost {total}/{ck.BUDGET} · "
       f"{len(threats)} threats, {sum(1 for t in threats if t.get('status') == 'mitigated')} mitigated · "
       f"{time.time() - t0:.1f}s")
