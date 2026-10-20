#!/usr/bin/env python3
"""Lab 09-1 · The full suite: twenty scenarios, two numbers, and an honest coverage map.

Offline, about a second. The lab pins the SIM brain (Lab 09-4 is the LIVE spot check).

  1. the whole suite (12 attacks, 8 benign tasks) on the naive ensemble and with Defenses.all()
  2. what the two numbers mean: attack success rate (ASR) and utility
  3. coverage: which MAESTRO layers and which OWASP ASI ids the suite exercises, and which it does not

The scenario tags (layers, ASI ids) are the course's own mapping, not part of MAESTRO or OWASP.
Everything is simulated: a fake hotel, fake guests, `.example` domains.

Run: .venv/bin/python week28_maestro_mas_security/09_red_team_scorecard/labs/lab09_1_full_suite.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import maestro  # noqa: E402
from ensemble import Defenses  # noqa: E402
from maskit import banner, dim, note, result, source_text, step, table, warn  # noqa: E402
from redteam import SCENARIOS, run_suite, scorecard  # noqa: E402

QUOTE_RED_TEAM = "Red Teaming:  Simulate attacks to find vulnerabilities."

banner("Lab 09-1 · the full suite", "12 attacks + 8 benign tasks · naive vs every defense · coverage", brain=False)
dim("this lab pins the 🧠 SIM brain: a deterministic teaching model, not a measurement of any real LLM")
assert " ".join(QUOTE_RED_TEAM.split()) in " ".join(source_text("maestro_framework_2025-02-06.txt").split())
note(f"MAESTRO, AI-specific mitigations: “{QUOTE_RED_TEAM}”")

step(1, "the naive ensemble")
t0 = time.time()
naive = scorecard(run_suite(Defenses()), "no defenses")
t_naive = time.time() - t0

step(2, "every defense on (Defenses.all(), spotlight = nonce)")
t0 = time.time()
full = scorecard(run_suite(Defenses.all()), "all 13 switches on")
t_all = time.time() - t0
result(f"ASR {naive['asr']:.0%} → {full['asr']:.0%} · utility {naive['utility']:.0%} → {full['utility']:.0%} · "
       f"{t_naive:.2f} s naive, {t_all:.2f} s defended for 20 scenarios")
note("ASR = attacks that caused ANY harm in the simulated hotel / attacks · utility = benign tasks done with no harm / "
     "benign tasks. Both are only as good as the scenarios behind them.")

step(3, "coverage by MAESTRO layer (the suite's tags are the course's mapping)")
atk = [s for s in SCENARIOS if s.kind == "attack"]
rows = []
for L in maestro.layers():
    ids = [s.id for s in atk if L["id"] in s.layers]
    rows.append([L["id"], L["name"], len(L["threats"]), ", ".join(ids) or "—"])
table(rows, ["layer", "name (MAESTRO)", "threats listed", "attack scenarios tagged with it"], maxw=48)

step(4, "coverage by OWASP Top 10 for Agentic Applications (ASI01–ASI10)")
rows, gaps = [], []
for o in maestro.owasp():
    ids = [s.id for s in atk if o["id"] in s.owasp]
    rows.append([o["id"], o["name"], ", ".join(ids) or "— none"])
    if not ids:
        gaps.append(f"{o['id']} {o['name']}")
table(rows, ["id", "risk (OWASP)", "attack scenarios"], maxw=48)
for g in gaps:
    warn(f"no scenario for {g}: nothing in the simulated hotel runs code, so the suite cannot test it")
note("a layer or ASI id with a scenario is TOUCHED, not covered: L4 alone lists "
     f"{len(maestro.layer('L4')['threats'])} threats ({', '.join(t['name'] for t in maestro.layer('L4')['threats'][:3])}, …)"
     ", and the suite's two L4 scenarios test none of the first three")
dim("a suite proves what it tests. 0 % ASR here means 'none of these 12 attacks worked', nothing more")
