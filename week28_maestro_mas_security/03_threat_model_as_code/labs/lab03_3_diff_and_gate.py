#!/usr/bin/env python3
"""Lab 03-3 · A design change, a threat diff, and a CI gate that fails, then passes after a verified mitigation.

Always the SIM brain (instant, offline, deterministic). The change: guests can now chat with the hotel on LINE
(data/line_channel_change.yaml — fictional ids, simulated hotel). v2 = v1 + the change. The gate judges the
PULL REQUEST, not the whole system: it looks only at the threats the diff says this change added or
re-scored. That is the threat-diffing idea from CSA's Feb 2026 CI/CD article; the code here is the
course's own (we do not install TITO).

Run: .venv/bin/python week28_maestro_mas_security/03_threat_model_as_code/labs/lab03_3_diff_and_gate.py
"""
import copy
import subprocess
import sys
from pathlib import Path

import yaml

MODULE = Path(__file__).resolve().parents[1]
ROOT = MODULE.parents[1]
sys.path.insert(0, str(MODULE.parent / "common"))
sys.path.insert(0, str(MODULE))
import maestro  # noqa: E402
import tmkit  # noqa: E402
from ensemble import Defenses, Ensemble  # noqa: E402
from maskit import attack, banner, defend, dim, note, ok, result, runs_dir, step, table  # noqa: E402

# REFERENCE — quoted verbatim from sources/maestro_cicd_2026-02-11.txt (Leath & Huang, CSA, 11 Feb 2026).
QUOTE_TITO = [
    "tito scan --repo ./ai-agent --maestro --mitre --attack-paths --report threat-model.md",
    "uses: Leathal1/TITO@v2",
    "diff-base: ${{ github.event.pull_request.base.sha }}",
    "fail-on: high",
]
UNBOUND, BOUND = "line:U7f3a9c", "guest:1203"          # a fictional LINE user id; the same user after binding

banner("Lab 03-3 · diff and gate a design change", "v1 + LINE guest chat = v2 · gate only what the change adds",
       brain=False)
dim("🧠 SIM brain for the checks here, even in LIVE mode: evidence for a gate must be reproducible")
out = runs_dir(__file__)
v1 = maestro.load(MODULE / "data" / "threat_model.yaml")
change = maestro.load(MODULE / "data" / "line_channel_change.yaml")

step(1, "the change: a LINE guest-chat channel")
c = change["components"][0]
note(f"new component {c['id']} on {c['layer']} ({c['kind']}, trust {c['trust']})")
table([[t["id"], t["title"], "/".join(t["layers"]), ", ".join(t["maestro"]),
        f"{t['likelihood']}×{t['impact']}={maestro.risk(t['likelihood'], t['impact'])[0]}",
        maestro.risk(t["likelihood"], t["impact"])[1]] for t in change["threats"]],
      ["id", "new threat", "layers", "MAESTRO", "L×I", "severity"], maxw=46)

step(2, "v2 = v1 + change · validate · diff v1 → v2")
v2 = tmkit.merge(v1, change)
errs = maestro.validate(v2)
ok(f"v2 validates ({len(v2['components'])} components, {len(v2['threats'])} threats)") if not errs else \
    [print("✕ " + e) for e in errs]
d = maestro.diff(v1, v2)
note(f"added {d['added']} · changed {d['changed'] or '—'} · removed {d['removed'] or '—'}")

step(3, "the gate on the change: fail_on=critical, only the new ids")
code_open = maestro.gate(v2, fail_on="critical", only=d["added"])
dim("v1's own open criticals (T01, T03, T06, T11) are NOT counted: the PR is judged on what it adds")
print("◆ for contrast, the same gate on the WHOLE model:")
whole = maestro.gate(v2, fail_on="critical")

step(4, "verify M-LINE-BIND in the simulator before claiming it")
for authz in (False, True):
    for principal, text in ((UNBOUND, "Room 1203 is too hot."), (BOUND, "Room 1203 is too hot."),
                            (BOUND, "Set room 1507 to 26.")):
        ens = Ensemble(Defenses(principal_authz=authz), brain="sim")
        run = ens.handle(text, principal=principal)
        acted = run.called("set_room_setpoint")
        line = f"principal_authz={'on ' if authz else 'off'} · {principal:<13} · {text:<22} → "
        if acted and principal == BOUND and text.startswith("Room 1203"):
            ok(line + "setpoint changed (legitimate: own room)")
        elif acted:
            attack(line + "setpoint changed")
        else:
            defend(line + (run.denied()[0].text.split(" · ", 1)[1] if run.denied() else "no action"))
dim("with the defense on, an unbound LINE id is external content; after binding it is guest:1203, own room only")
ens = Ensemble(Defenses(principal_authz=True), brain="sim")
blocked = not ens.handle("Room 1203 is too hot.", principal=UNBOUND).called("set_room_setpoint")
own_ok = Ensemble(Defenses(principal_authz=True), brain="sim").handle("Room 1203 is too hot.", principal=BOUND)\
    .called("set_room_setpoint")

step(5, "mark T16 mitigated (evidence above) and gate again")
v2m = copy.deepcopy(v2)
for t in v2m["threats"]:
    if t["id"] == "T16" and blocked and own_ok:
        t.update(status="mitigated", residual_likelihood=1, residual_impact=t["impact"])
        defend(f"T16 → mitigated · residual 1×{t['impact']}={maestro.effective(t)[0]} {maestro.effective(t)[1]}")
code_fixed = maestro.gate(v2m, fail_on="critical", only=d["added"])

step(6, "the same gate as a CI step (tmkit.py CLI, real exit codes)")
(out / "threat_model_v2.yaml").write_text(yaml.safe_dump(v2, sort_keys=False, allow_unicode=True), encoding="utf-8")
(out / "threat_model_v2_mitigated.yaml").write_text(yaml.safe_dump(v2m, sort_keys=False, allow_unicode=True),
                                                     encoding="utf-8")
codes = {}
for head in ("threat_model_v2.yaml", "threat_model_v2_mitigated.yaml"):
    rel = f"week28_maestro_mas_security/03_threat_model_as_code/.runs/{head}"
    print(f"→ tmkit.py --base week28_maestro_mas_security/03_threat_model_as_code/data/threat_model.yaml --head {rel} --fail-on critical")
    p = subprocess.run([sys.executable, str(MODULE / "tmkit.py"), "--base", str(MODULE / "data" / "threat_model.yaml"),
                        "--head", str(out / head), "--fail-on", "critical"], capture_output=True, text=True,
                       cwd=ROOT)
    print(p.stdout.rstrip())
    codes[head] = p.returncode
    note(f"exit code {p.returncode}")

step(7, "REFERENCE — how CSA's Feb 2026 article runs the same idea with TITO (not installed here)")
for q in QUOTE_TITO:
    print(f"│ {q}")
dim("quoted from sources/maestro_cicd_2026-02-11.txt; TITO scans code, our gate reads a YAML model you keep by hand")

result(f"gate on the change: {'FAIL' if code_open else 'pass'} → {'FAIL' if code_fixed else 'pass'} after a verified "
       f"mitigation · CLI exit codes {codes['threat_model_v2.yaml']} → {codes['threat_model_v2_mitigated.yaml']}")
