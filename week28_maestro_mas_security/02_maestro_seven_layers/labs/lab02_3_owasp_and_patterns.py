#!/usr/bin/env python3
"""Lab 02-3 · OWASP ASI01–ASI10 on the MAESTRO layers, the eight architecture patterns, and four real harms.

Always the SIM brain (instant, offline, deterministic). Three views of the same system:
  1. the OWASP Top 10 for Agentic Applications (a RISK LIST) placed on MAESTRO's layers (an ARCHITECTURE
     LENS). The ASI→layer mapping is the course's own, not OWASP's or CSA's.
  2. MAESTRO's eight agentic architecture patterns, and which ones the Hotel Ops Ensemble has.
  3. four red-team scenarios against the NAIVE ensemble (every defense off): one ASI risk per layer,
     each ending in a concrete harm in the simulated hotel.

Run: .venv/bin/python week28_maestro_mas_security/02_maestro_seven_layers/labs/lab02_3_owasp_and_patterns.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import maestro  # noqa: E402
from ensemble import Defenses  # noqa: E402
from maskit import attack, banner, dim, note, result, step, table  # noqa: E402
from redteam import SCENARIOS, run_scenario  # noqa: E402

# Which MAESTRO patterns the Hotel Ops Ensemble exhibits (the course's reading of ensemble.py).
ENSEMBLE_PATTERNS = {
    "Single-Agent Pattern": ("no", "four agents, never one alone"),
    "Multi-Agent Pattern": ("yes", "four agents pass messages to each other"),
    "Unconstrained Conversational Autonomy": ("no", "free text in, but a fixed tool set per agent"),
    "Task-Oriented Agent Pattern": ("yes", "energy and maintenance exist to call tools"),
    "Hierarchical Agent Pattern": ("yes", "the orchestrator routes work to specialists"),
    "Distributed Agent Ecosystem": ("yes", "vendor found in an open directory, over A2A"),
    "Human-in-the-Loop Collaboration": ("yes", "duty manager approves critical actions"),
    "Self-Learning and Adaptive Agents": ("partial", "no training; shared memory shapes later turns"),
}
PICK = ["A06", "A05", "A04", "A08"]          # one ASI risk on L2, L3/L4, L6, L7

banner("Lab 02-3 · OWASP on MAESTRO, patterns, real harms", "risk list vs architecture lens", brain=False)
dim("🧠 SIM brain for every scenario here, even in LIVE mode: four scenarios would exceed the lab's LIVE budget")

step(1, "OWASP Top 10 for Agentic Applications (2026) on MAESTRO's layers — the course's mapping")
grid = []
for o in maestro.owasp():
    cells = ["●" if lid in o["layers"] else "·" for lid in maestro.LAYER_ORDER]
    grid.append([o["id"], o["name"]] + cells)
table(grid, ["id", "OWASP risk", *maestro.LAYER_ORDER], maxw=40)
per_layer = {lid: sum(1 for o in maestro.owasp() if lid in o["layers"]) for lid in maestro.LAYER_ORDER}
note("ASI risks per layer: " + " · ".join(f"{k} {v}" for k, v in per_layer.items()))
dim("the ● marks are the course's judgement (maestro.OWASP_TO_LAYERS), not part of OWASP or MAESTRO")

step(2, "MAESTRO's eight agentic architecture patterns — which ones does the ensemble have?")
rows = []
for p in maestro.patterns():
    f = p["fields"]
    threat = f.get("Threat") or " / ".join(k for k in f if k not in ("Description", "Example Threat Scenario",
                                                                      "Mitigation", "Mitigations"))
    has, why = ENSEMBLE_PATTERNS[p["name"]]
    rows.append([p["name"], threat, has, why])
table(rows, ["pattern (CSA)", "its threat (CSA)", "ensemble?", "why (course)"], maxw=46)
yes = [n for n, (h, _) in ENSEMBLE_PATTERNS.items() if h == "yes"]
note(f"{len(yes)} of 8 patterns present → their threats ALL apply at once: that is what makes a MAS different")

step(3, "four OWASP risks becoming real harms in the NAIVE ensemble (no defenses)")
by_id = {s.id: s for s in SCENARIOS}
harmed = 0
for sid in PICK:
    s = by_id[sid]
    print(f"\n◆ {s.id} · {s.title} · MAESTRO {'/'.join(s.layers)} · OWASP {'/'.join(s.owasp)}")
    for principal, text in s.steps:
        attack(f"{principal}: {text}")
    out = run_scenario(s, Defenses(), brain="sim", show=True)
    harmed += bool(out.harms)
    print(("✕ harm: " + ", ".join(out.harms)) if out.harms else "✓ no harm")

result(f"{harmed}/{len(PICK)} picked scenarios cause harm with every defense off (SIM brain: a teaching model "
       "of a gullible LLM, not a measurement)")
