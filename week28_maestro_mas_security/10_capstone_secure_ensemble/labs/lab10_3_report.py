#!/usr/bin/env python3
"""Lab 10-3 · Ship it: the threat diff, the CI gate, and the one-page MAESTRO report for the GM.

Deliverables (d) and (e). Reads your data/capstone.yaml and data/threat_model.yaml and:
  1. diffs your threat model against the design-time base (data/threat_model.base.yaml): what you added,
     removed or re-scored — the "threat diffing" idea from CSA's CI/CD article
  2. runs the CI gate: maestro.gate(fail_on="high") plus the suite's two numbers (ASR 0, utility 100 %)
  3. writes .runs/maestro_report.md: a summary for the GM, then the six MAESTRO steps as headings
     (layer table, ranked threats inherent → residual, controls and evidence, scorecard, open and accepted items,
     next steps)

    --ci        print only the gate and exit 1 if it fails (what a CI job runs on every pull request)
    --config PATH --tm PATH --out PATH    other inputs / output (default: this module's data/ and .runs/)

SIM brain only, offline, about one second. Everything describes a simulated hotel.

Run: .venv/bin/python week28_maestro_mas_security/10_capstone_secure_ensemble/labs/lab10_3_report.py
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capkit as ck  # noqa: E402
import maestro  # noqa: E402
from maskit import ROOT, banner, dim, fail, note, ok, result, step, table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(ck.CONFIG))
ap.add_argument("--tm", default=str(ck.THREAT_MODEL))
ap.add_argument("--out", default="")
ap.add_argument("--ci", action="store_true")
args = ap.parse_args()


def ci_gate() -> int:
    """The release gate: the threat model's gate AND the suite's two numbers. 0 = ship, 1 = block."""
    cfg = ck.load_config(args.config)
    tm, errs = ck.load_tm(args.tm)
    if errs or cfg["errors"] or not tm.get("threats"):
        for e in cfg["errors"] + errs:
            fail(e)
        return 1
    code, text = ck.gate_text(tm, "high")
    print(text)
    sc = ck.score(ck.suite(cfg["defenses"], cfg["human"]))
    suite_ok = sc["succeeded"] == 0 and sc["done"] == sc["benign"]
    print(("✓" if suite_ok else "✕") + f" suite: ASR {sc['succeeded']}/{sc['attacks']} · utility {sc['done']}/{sc['benign']}")
    return 1 if code or not suite_ok else 0


if args.ci:
    sys.exit(ci_gate())

t0 = time.time()
banner("Lab 10-3 · ship it", "threat diff · CI gate · the MAESTRO report for the GM", brain=False)

step(1, "threat diff: your threat model vs the design-time base")
base, _ = ck.load_tm(ck.BASE_TM)
mine, errs = ck.load_tm(args.tm)
for e in errs:
    fail(e)
dif = maestro.diff(base, mine) if mine.get("threats") and not errs else {"added": [], "removed": [], "changed": []}
byid = {t["id"]: t for t in mine.get("threats", []) or []}
rows = [[k, "added", byid[k]["title"], "{} {}".format(*maestro.effective(byid[k]))] for k in dif["added"]]
rows += [[k, "removed", "", ""] for k in dif["removed"]]
rows += [[k, f"{byid[k]['status']}", byid[k]["title"], "{} {}".format(*maestro.effective(byid[k]))] for k in dif["changed"]]
if rows:
    table(rows, ["id", "change", "threat", "effective risk now"], maxw=66)
else:
    dim("no change yet: your threat_model.yaml is still the design-time model")
note(f"{len(dif['added'])} added · {len(dif['removed'])} removed · {len(dif['changed'])} re-scored or re-statused")

step(2, "the CI gate: maestro.gate(fail_on='high') + ASR 0 + utility 100 %")
code = ci_gate()
(ok if code == 0 else fail)(f"CI gate exit code {code}: " + ("ship" if code == 0 else "this change would be blocked"))
dim("in CI: .venv/bin/python week28_maestro_mas_security/10_capstone_secure_ensemble/labs/lab10_3_report.py --ci   (exit 1 blocks the merge)")

step(3, "the MAESTRO report for the GM")
out = Path(args.out) if args.out else ck.report_path()
if errs or not mine.get("threats"):
    fail("the threat model has errors (step 1): fix them, then rerun to write the report")
    sys.exit(0)
ck.write_report(args.config, args.tm, out)
text = out.read_text(encoding="utf-8")
summary = text.split("## 1 ·")[0].strip().splitlines()
for ln in summary:
    if ln and not ln.startswith("<!--"):
        print("│ " + ln)
heads = [ln[3:] for ln in text.splitlines() if ln.startswith("## ")]
note(f"sections: {' · '.join(heads)}")
shown = out.resolve().relative_to(ROOT) if out.resolve().is_relative_to(ROOT) else out
ok(f"wrote {shown} ({len(text.splitlines())} lines, inputs fingerprint {ck.fingerprint(args.config, args.tm)})")
result(f"report written · gate {'passes' if code == 0 else 'fails'} · {time.time() - t0:.1f}s")
