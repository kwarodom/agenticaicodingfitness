#!/usr/bin/env python3
"""Lab 04-2 · From route to an approved prompt bundle (still no execution).

Jev decides WHICH specialist; it never writes the answer. The route selects a
pre-approved system prompt from a registry in code. Tools, model tier and data
access are resolved separately and re-checked at every call.

Three requests: a clear HVAC question, a coding request, and a vague one that
must fall through to "clarify" instead of guessing a prompt.

Run: .venv/bin/python week24/04_intent_routing/labs/lab02_prompt_bundle.py
"""
import json
import sys
from pathlib import Path

WEEK24 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WEEK24 / "common"))
sys.path.insert(0, str(WEEK24 / "jev_lab"))
import jev_lab as ref  # noqa: E402
from jevkit import ask, banner, step, validate  # noqa: E402

# The approved-prompt registry (from the source guide's prompt_preview.py).
PROMPTS = {
    "hvac": ("hvac_expert",
             "Separate observed symptoms from hypotheses. Use authorized building data. "
             "State missing evidence. Propose inspections only; do not change controls."),
    "energy_mv": ("mv_analyst",
                  "Use validated meter data and the approved baseline method. "
                  "Compute numbers in tools or code. State uncertainty and exclusions."),
    "facility_ops": ("facility_ops",
                     "Draft a maintenance proposal from the approved SOP and site evidence. "
                     "Keep a named-human approval step before a work order or action."),
    "sustainability": ("sustainability_reporter",
                       "Use approved emissions factors and source evidence. "
                       "Do not invent carbon reductions or claim certification."),
    "coding": ("coding_agent",
               "Work only in the authorized repository and sandbox. "
               "Inspect requirements, implement tests, and propose changes for review."),
    "general": ("general_assistant",
                "Answer the question directly. Use current sources when necessary. "
                "Do not access building or company data without a justified purpose."),
}

REQUESTS = [r for r in ref.SAMPLES["intent"] if r["id"] in ("i1", "i4", "i9")]

banner("Lab 04-2 · prompt bundle", "route → approved prompt from a registry in code")

for n, row in enumerate(REQUESTS, 1):
    step(n, f"“{row['state']['message']}”")
    state = ref.prepare("intent", row)
    a = validate(ask(state, ref.QUESTIONS["intent"], quiet=True), ref.QUESTIONS["intent"])
    rec = ref.policy("intent", state, a)
    route = rec["route"]
    print(f"» intent {a['intent']['choice']} (confidence {a['intent']['confidence']:.2f}) → route {route}")
    if route not in PROMPTS:
        print("→ no prompt selected automatically — ask the user to clarify, or send to review")
        continue
    agent, prompt = PROMPTS[route]
    bundle = {
        "agent_candidate": agent,
        "system_prompt_candidate": prompt,
        "compute_tier_candidate": rec["compute_tier"],
        "dispatch_candidate": rec["dispatch"],
        "specialists_suggested": rec["specialists_suggested"],
        "tools": "resolve separately from the authorized registry",
        "execute": False,
    }
    print(json.dumps(bundle, indent=2))

print("\n═ execute: False — no LLM, tool, inbox or building system was called.")
print("═ The prompt text comes from YOUR registry; Jev only picked the key.")
