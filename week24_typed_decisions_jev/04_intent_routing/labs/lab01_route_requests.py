#!/usr/bin/env python3
"""Lab 04-1 · Route Alto Copilot requests — topic + signals + a policy gate.

We reuse the reference implementation in week24/jev_lab/jev_lab.py:
  • QUESTIONS["intent"] — 1 choice (7 topics) + 5 nouls + 1 score, in ONE call
  • prepare()           — allowlists input fields (teaching labels are never sent)
  • policy()            — deterministic code: accepted() gate, dispatch, compute tier

Nine synthetic requests (English + Thai) → a routing table. Every route is a
PROPOSAL (execute: False); authorization is always checked separately.

Run: .venv/bin/python week24/04_intent_routing/labs/lab01_route_requests.py
"""
import sys
from pathlib import Path

WEEK24 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WEEK24 / "common"))
sys.path.insert(0, str(WEEK24 / "jev_lab"))
import jev_lab as ref  # noqa: E402  — the reference lab from the source guide
from jevkit import ask, banner, show, step, table, validate  # noqa: E402

QUESTIONS = ref.QUESTIONS["intent"]
SAMPLES = ref.SAMPLES["intent"]          # 9 rows, each with a teaching label in "expected"

banner("Lab 04-1 · intent routing", "topic + retrieval/live-data/MAS signals + complexity → policy")

step(1, "look at ONE request in full — 7 questions, 1 call")
first = SAMPLES[0]
state = ref.prepare("intent", first)                 # only `message` is sent — never `expected`
resp = ask(state, QUESTIONS)
show(resp, top=3)

step(2, "route all nine requests (9 calls)")
rows, agree = [], 0
for row in SAMPLES:
    state = ref.prepare("intent", row)
    a = validate(ask(state, QUESTIONS, quiet=True), QUESTIONS)
    rec = ref.policy("intent", state, a)             # deterministic code, no model involved
    label = row["expected"]["intent"]
    agree += a["intent"]["choice"] == label
    rows.append([row["id"], row["state"]["message"], label, a["intent"]["choice"],
                 f"{a['intent']['confidence']:.2f}", rec["route"], rec["dispatch"], rec["compute_tier"]])
table(rows, ["id", "message", "teaching label", "jev", "conf", "route", "dispatch", "tier"])
print(f"◆ Jev agreed with the teaching label on {agree}/{len(SAMPLES)} rows — read the disagreements, don't just count them")

step(3, "what the policy() did (read it in week24/jev_lab/jev_lab.py)")
print("│ route    = topic only if accepted(): confidence ≥ .75, peak ≥ .80, margin ≥ .20, not 'unknown'")
print("│ dispatch = single_agent if needs_mas ≤ .20 · mas_candidate if ≥ .80 · else review_dispatch")
print("│ tier     = reasoning_candidate if complexity ≥ 1.5 or its confidence < .75, else fast_candidate")
print("│ thresholds are ILLUSTRATIVE and uncalibrated — tune them on your own labeled traffic")
print("═ execute: False · authorization: must_be_checked_separately — a label never grants a permission.")
