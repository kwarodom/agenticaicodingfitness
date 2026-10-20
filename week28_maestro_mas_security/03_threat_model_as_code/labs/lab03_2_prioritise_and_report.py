#!/usr/bin/env python3
"""Lab 03-2 · Prioritise by risk, write the report, then mark threats mitigated only where the evidence says so.

Always the SIM brain (instant, offline, deterministic). MAESTRO steps 4–6: rank the threats, plan mitigations, implement
and monitor. The rule this lab enforces: a threat becomes `mitigated` only when the defense its mitigation
names actually BLOCKS the matching red-team scenario (redteam.run_scenario with just that defense on).
Threats with no scenario, or whose defense fails the test, stay `open` — whatever the YAML hoped.

Residual scoring (the course's rule): a verified block drops likelihood to 1; impact stays, because the
control makes the harm unlikely, not smaller. SIM results come from a teaching model of a gullible LLM.

Run: .venv/bin/python week28_maestro_mas_security/03_threat_model_as_code/labs/lab03_2_prioritise_and_report.py
"""
import copy
import sys
from collections import Counter
from pathlib import Path

import yaml

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE.parent / "common"))
sys.path.insert(0, str(MODULE))
import maestro  # noqa: E402
import tmkit  # noqa: E402
from maskit import banner, defend, dim, fail, note, ok, result, runs_dir, step, table, warn  # noqa: E402

banner("Lab 03-2 · prioritise, report, verify", "the evidence decides what counts as mitigated", brain=False)
dim("🧠 SIM brain for every scenario here, even in LIVE mode: evidence must be reproducible, and 12 scenarios "
    "would exceed the lab's LIVE budget")
tm = maestro.load(MODULE / "data" / "threat_model.yaml")
out = runs_dir(__file__)

step(1, "rank every threat by inherent risk (MAESTRO step 4)")
ranked = sorted(tm["threats"], key=lambda t: -maestro.risk(t["likelihood"], t["impact"])[0])
table([[i + 1, t["id"], t["title"], maestro.risk(t["likelihood"], t["impact"])[0],
        maestro.risk(t["likelihood"], t["impact"])[1], ", ".join(t["mitigations"])]
       for i, t in enumerate(ranked)], ["#", "id", "threat", "risk", "severity", "planned mitigations"], maxw=48)

step(2, "write the markdown report (before any mitigation)")
(out / "threat_report.md").write_text(maestro.report_md(tm) + "\n", encoding="utf-8")
ok(f"wrote {(out / 'threat_report.md').relative_to(MODULE.parent.parent)}")
print("\n".join("│ " + ln for ln in maestro.report_md(tm).splitlines()[:6]))
dim("… open the file for the full ranked table")

step(3, "verify each mitigation against its red-team scenario (MAESTRO step 6)")
after = copy.deepcopy(tm)
verdicts = Counter()
for t in after["threats"]:
    blocked, why = tmkit.verify(after, t)
    if blocked:
        t["status"] = "mitigated"
        t["residual_likelihood"], t["residual_impact"] = 1, t["impact"]
        defend(f"{t['id']} {why}")
        verdicts["mitigated"] += 1
    elif blocked is False:
        fail(f"{t['id']} {why} → stays open")
        verdicts["failed"] += 1
    else:
        warn(f"{t['id']} {why} → stays open")
        verdicts["unverifiable"] += 1
note(f"mitigated {verdicts['mitigated']} · defense failed its test {verdicts['failed']} · "
     f"nothing to test in SIM {verdicts['unverifiable']}")

step(4, "inherent vs residual heatmaps")
print("◆ inherent (every threat open)")
tmkit.show_heatmap(tm["threats"])
print("\n◆ residual (verified mitigations only; open threats keep their inherent score)")
tmkit.show_heatmap(after["threats"], residual=True)

step(5, "effective risk, before → after")
rows, before_sev, after_sev = [], Counter(), Counter()
for a, b in zip(tm["threats"], after["threats"]):
    s0, v0 = maestro.effective(a)
    s1, v1 = maestro.effective(b)
    before_sev[v0] += 1
    after_sev[v1] += 1
    rows.append([b["id"], b["title"], f"{s0} {v0}", f"{s1} {v1}", b["status"]])
table(rows, ["id", "threat", "before", "after", "status"], maxw=46)
sev_line = lambda c: " · ".join(f"{k} {c.get(k, 0)}" for k in ("critical", "high", "medium", "low"))  # noqa: E731
note(f"before: {sev_line(before_sev)}")
note(f"after:  {sev_line(after_sev)}")

step(6, "save the evidence-backed model + report, and gate it")
(out / "threat_model_mitigated.yaml").write_text(yaml.safe_dump(after, sort_keys=False, allow_unicode=True),
                                                  encoding="utf-8")
(out / "threat_report_mitigated.md").write_text(maestro.report_md(after) + "\n", encoding="utf-8")
ok("wrote .runs/threat_model_mitigated.yaml and .runs/threat_report_mitigated.md")
assert not maestro.validate(after)
print("◆ gate fail_on=critical")
crit = maestro.gate(after, fail_on="critical")
print("◆ gate fail_on=high")
high = maestro.gate(after, fail_on="high")

result(f"critical gate {'passes' if crit == 0 else 'fails'} · high gate {'passes' if high == 0 else 'fails'} — "
       "what is left open is the honest to-do list for Modules 04–08")
