#!/usr/bin/env python3
"""Lab 03-1 · Load the ensemble's threat model, validate it, break it on purpose, and draw the inherent-risk heatmap.

Offline and instant. The threat model is data/threat_model.yaml: MAESTRO's six steps stored as code
(components → layer threats → cross-layer threats → likelihood × impact → mitigations → status). The 1–5
scale and its bands are the course's own; MAESTRO only says to assess likelihood and impact with a matrix.

Run: .venv/bin/python week28_maestro_mas_security/03_threat_model_as_code/labs/lab03_1_build_and_validate.py
"""
import copy
import sys
from collections import Counter
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE.parent / "common"))
sys.path.insert(0, str(MODULE))
import maestro  # noqa: E402
import tmkit  # noqa: E402
from maskit import banner, dim, note, ok, result, step, table, warn  # noqa: E402

banner("Lab 03-1 · build and validate the threat model", "offline · data/threat_model.yaml", brain=False)

step(1, "load the YAML")
tm = maestro.load(MODULE / "data" / "threat_model.yaml")
th = tm["threats"]
note(f"{tm['system']} · v{tm['version']} · {len(tm['components'])} components · "
     f"{len(tm['mitigations'])} mitigations · {len(th)} threats")

step(2, "validate: schema + every reference resolves (components, mitigations, MAESTRO ids)")
errs = maestro.validate(tm)
ok("0 errors — the model is well-formed") if not errs else [print("✕ " + e) for e in errs]

step(3, "break a COPY on purpose — what validation catches")
bad = copy.deepcopy(tm)
bad["threats"][0]["likelihood"] = 7                       # off the 1–5 scale
bad["threats"][3]["components"].append("spa_agent")       # a component nobody declared
bad["threats"][5]["maestro"].append("L8.T1")              # there is no Layer 8
bad["threats"][8]["status"] = "fixed"                     # not a known status
del bad["threats"][9]["title"]                            # a required field gone
bad["threats"][12]["status"] = "mitigated"                # claimed mitigated …
bad["threats"][12]["mitigations"] = []                    # … with nothing listed
for e in maestro.validate(bad):
    print("✕ " + e)
dim("validation is cheap, so a CI job can run it on every pull request before anyone reads the YAML")

step(4, "coverage: every MAESTRO layer, cross-layer threats, every component")
per_layer = Counter(lid for t in th for lid in t["layers"])
print("│ " + " · ".join(f"{lid} {per_layer.get(lid, 0)}" for lid in maestro.LAYER_ORDER))
xs = sorted({r for t in th for r in t.get("maestro", []) if r.startswith("X")})
note(f"cross-layer refs used: {', '.join(f'{x} {maestro.threat(x)['name']}' for x in xs)}")
touched = {c for t in th for c in t["components"]}
untouched = [c["id"] for c in tm["components"] if c["id"] not in touched]
(warn if untouched else ok)(f"components with no threat: {', '.join(untouched) or 'none'}")

step(5, "MAESTRO step 4 — Risk Assessment: L × I on the course's matrix")
rows = []
for t in sorted(th, key=lambda t: -maestro.risk(t["likelihood"], t["impact"])[0]):
    s, sev = maestro.risk(t["likelihood"], t["impact"])
    rows.append([t["id"], t["title"], "/".join(t["layers"]), ", ".join(t.get("maestro", [])),
                 f"{t['likelihood']}×{t['impact']}={s}", sev])
table(rows, ["id", "threat", "layers", "MAESTRO", "L×I", "severity"], maxw=50)
sev = Counter(maestro.risk(t["likelihood"], t["impact"])[1] for t in th)
note("severity: " + " · ".join(f"{k} {sev.get(k, 0)}" for k in ("critical", "high", "medium", "low")))

step(6, "inherent-risk heatmap (impact up, likelihood across)")
tmkit.show_heatmap(th)
dim("bands (course): score ≥15 critical · ≥10 high · ≥5 medium · else low")

result(f"{len(th)} threats on {len(per_layer)} layers · {len(xs)} cross-layer refs · "
       f"{sev.get('critical', 0)} critical before any mitigation")
