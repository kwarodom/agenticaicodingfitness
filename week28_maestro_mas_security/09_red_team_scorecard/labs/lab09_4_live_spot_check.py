#!/usr/bin/env python3
"""Lab 09-4 · LIVE spot check: three scenarios on a real local model, next to the SIM brain's answer.

The SIM brain is a deterministic teaching model of a gullible LLM. Before you trust a scorecard, spot-check it on
the real thing. This lab runs A01, A02 and B01 on the naive ensemble twice: once on the SIM brain, once on the
LIVE brain (local Ollama, default gemma4:12b), with ONE shared budget of 6 model calls for the whole lab.

  • In SIM mode (the default) the LIVE half is skipped with a ◈ note: the lab still runs and exits 0.
  • In LIVE mode (the runner's 🦙 LIVE switch, or MAS_BRAIN=ollama) expect ≈ 5–12 s per model call on an M4 Max.

A real model's result can differ from run to run. Everything is simulated: a fake hotel, `.example` domains.

Run: .venv/bin/python week28_maestro_mas_security/09_red_team_scorecard/labs/lab09_4_live_spot_check.py
     MAS_BRAIN=ollama .venv/bin/python week28_maestro_mas_security/09_red_team_scorecard/labs/lab09_4_live_spot_check.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ensemble import Defenses  # noqa: E402
from maskit import banner, brain_mode, brain_model, dim, note, ollama_models, result, step, table, warn  # noqa: E402
from redteam import SCENARIOS  # noqa: E402
from scorekit import run_detailed  # noqa: E402

PICK = ["A01", "A02", "B01"]          # a direct injection, an indirect one, and a benign task as a control
BUDGET = 6                            # model calls for the WHOLE lab

banner("Lab 09-4 · LIVE spot check", "A01 · A02 · B01 on the naive ensemble · SIM vs a real local model")
chosen = [next(s for s in SCENARIOS if s.id == i) for i in PICK]


def verdict(o) -> str:
    if o.scenario.kind == "attack":
        return ("✕ harm: " + ", ".join(o.harms)) if o.harms else "✓ no harm"
    return "✓ done" if o.done else "✕ not done"


step(1, "the SIM brain's answer (deterministic, instant)")
sim = {s.id: run_detailed(s, Defenses(), brain="sim")[0] for s in chosen}
table([[s.id, s.title, verdict(sim[s.id])] for s in chosen], ["id", "scenario", "SIM brain"], maxw=58)

step(2, f"the LIVE brain's answer (≤ {BUDGET} model calls in total)")
if brain_mode() != "ollama":
    dim("SIM mode: the LIVE spot check is skipped. Switch the runner to 🦙 LIVE (or set MAS_BRAIN=ollama) to run it.")
    sys.exit(0)
model = brain_model()
if model not in ollama_models():
    warn(f"{model} is not in local Ollama — start Ollama or pull the model; skipping the LIVE half")
    sys.exit(0)
used, rows = 0, []
for s in chosen:
    if used >= BUDGET:
        rows.append([s.id, verdict(sim[s.id]), "◈ skipped: model-call budget spent", 0, "—"])
        continue
    print(f"\n◆ {s.id} · {s.title} · {s.steps[0][0]}: {s.steps[0][1]}")
    t0 = time.time()
    o, e = run_detailed(s, Defenses(), brain="ollama", llm_used=used, show=True)
    calls = e.llm_calls - used
    used = e.llm_calls
    cut = " · budget now spent" if used >= BUDGET else ""
    rows.append([s.id, verdict(sim[s.id]), verdict(o) + cut, calls, f"{time.time() - t0:.0f} s"])
print()
table(rows, ["id", "SIM brain", f"LIVE brain {model}", "model calls", "time"], maxw=58)
result(f"{used}/{BUDGET} model calls used · LIVE results can differ on a rerun")
note("where SIM and LIVE disagree, the SIM scorecard over- or under-states the risk for THIS model. Re-run the "
     "attacks that matter on the model you ship, and keep the controls that do not depend on the model saying no.")
