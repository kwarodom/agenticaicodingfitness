#!/usr/bin/env python3
"""Lab 02-1 · The MAESTRO catalogue as a layer stack: layers, threats, cross-layer threats, mitigations, steps.

Offline and instant. Every name below is parsed verbatim from the saved CSA page
(week28_maestro_mas_security/common/sources/maestro_framework_2025-02-06.txt) by maestro.catalog() — the same data the
runner's 🗺 MAESTRO map shows. Nothing here runs an agent.

Run: .venv/bin/python week28_maestro_mas_security/02_maestro_seven_layers/labs/lab02_1_seven_layers.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import maestro  # noqa: E402
from maskit import banner, bar, dim, note, result, step, table  # noqa: E402

banner("Lab 02-1 · the seven MAESTRO layers", "offline · parsed verbatim from the CSA page", brain=False)
cat = maestro.catalog()

step(1, "the layer stack, top (L7) to bottom (L1)")
most = max(len(L["threats"]) for L in cat["layers"])
for L in reversed(cat["layers"]):
    tag = "  ◀ vertical: cuts across every layer" if L["vertical"] else ""
    print(f"│ {L['id']} {L['name']:<31} {bar(len(L['threats']), most, 13)} {len(L['threats']):>2} threats{tag}")
total = sum(len(L["threats"]) for L in cat["layers"])
note(f"{total} layer-specific threats in the catalogue · L7 has the most ({most})")

step(2, "what each layer is (the source's own description, first sentence)")
rows = [[L["id"], L["name"], L["description"].split(". ")[0].rstrip(".") + "."] for L in reversed(cat["layers"])]
table(rows, ["layer", "name", "description"], maxw=74)

step(3, "the five cross-layer threats")
table([[x["id"], x["name"], x["description"]] for x in cat["cross_layer"]], ["id", "threat", "description"], maxw=74)

step(4, "mitigations: cross-layer and AI-specific")
m = cat["mitigations"]
table([["cross-layer", x["name"], x["description"]] for x in m["cross_layer"]] +
      [["AI-specific", x["name"], x["description"]] for x in m["ai_specific"]], ["kind", "mitigation", "description"],
      maxw=70)

step(5, "the six steps (Module 03 walks them end to end)")
for s in cat["steps"]:
    print(f"│ {s['n']}. {s['name']:<34} {s['description'].split('. ')[0].rstrip('.')}.")

step(6, "how to look a threat up by id")
for tid in ("L7.T2", "L2.T1", "X5"):
    t = maestro.threat(tid)
    print(f"│ maestro.threat({tid!r}) → {t['name']}")
dim("ids like L7.T2 are the course's handle: layer + position in that layer's list on the CSA page")

result(f"7 layers · {total} layer threats · {len(cat['cross_layer'])} cross-layer threats · "
       f"{len(m['cross_layer'])} + {len(m['ai_specific'])} mitigations · {len(cat['steps'])} steps · "
       f"{len(cat['patterns'])} patterns")
