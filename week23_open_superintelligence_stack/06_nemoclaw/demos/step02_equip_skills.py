#!/usr/bin/env python3
"""PART 2 · Equip it — attach skills & tools  [INTERMEDIATE]

A persona alone can't act. NemoClaw pulls SKILLS from the Agent Skills catalog
(App 4) and wires concrete TOOLS. Each attachment adds a capability the specialist
can invoke on demand. This demo shows the catalog, attaches two skills to the HVAC
specialist, and walks the discover → attach → invoke loop (a trace in SIM; a real
tool-call loop in REAL).

Run:  python demos/step02_equip_skills.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import view  # noqa: E402

# A slice of the Agent Skills catalog (App 4) — pluggable capabilities.
CATALOG = [
    ("nemo-retriever", "RAG over YOUR runbooks & manuals", "get_runbook"),
    ("cuopt",          "GPU-accelerated setpoint optimization", "optimize_setpoints"),
    ("nemo-guardrails", "policy + safety rails at the edge", "(applied as policy)"),
]

# The tools the two ATTACHED skills expose to the specialist.
TOOLS = [
    {"type": "function", "function": {
        "name": "get_runbook",
        "description": "Skill nemo-retriever: fetch the runbook step for an HVAC symptom.",
        "parameters": {"type": "object", "properties": {"symptom": {"type": "string"}},
                       "required": ["symptom"]}}},
    {"type": "function", "function": {
        "name": "optimize_setpoints",
        "description": "Skill cuopt: compute an energy-optimal setpoint for a room.",
        "parameters": {"type": "object", "properties": {"room": {"type": "string"}},
                       "required": ["room"]}}},
]
_RUNBOOK = {"overheating": "RB-07: check damper actuator, then verify chilled-water valve."}


def _impl(name, args):
    if name == "get_runbook":
        s = (args.get("symptom") or "").lower()
        step = next((v for k, v in _RUNBOOK.items() if k in s), "no runbook match")
        return json.dumps({"step": step, "source": "runbook-RAG"})
    if name == "optimize_setpoints":
        return json.dumps({"room": args.get("room"), "setpoint_c": 22.0, "kwh_saved": 3.1})
    return json.dumps({})


def main() -> None:
    view.banner("PART 2", "Equip it — attach skills & tools", "INTERMEDIATE")
    view.mode_line()

    print("Agent Skills catalog (App 4) — capabilities you can attach to the specialist:")
    for name, desc, tool in CATALOG:
        print(f"  • {name:16s} {desc:38s} → tool: {tool}")
    print("\nAttaching nemo-retriever + cuopt to the HVAC specialist → it gains hands.\n")

    if view.is_sim():
        print("SIM — the discover → attach → invoke loop (REAL mode executes it):")
        print("  → agent has skills [nemo-retriever, cuopt] attached")
        print("  ~ reason: room 1203 overheating → consult runbook via nemo-retriever")
        print("  → agent invokes get_runbook({'symptom':'overheating'})")
        print("  ← {'step':'RB-07: check damper actuator…','source':'runbook-RAG'}")
        print("  → agent invokes optimize_setpoints({'room':'1203'})")
        print("  ← {'setpoint_c':22.0,'kwh_saved':3.1}")
        print("  · answer: Per RB-07 check the damper; optimal setpoint 22°C saves ~3.1 kWh.")
    else:
        client = view._client()
        messages = [{"role": "system", "content": "You are an HVAC reliability engineer with two "
                     "attached skills: get_runbook (runbook RAG) and optimize_setpoints (cuopt). "
                     "For an overheating room, consult the runbook AND compute an optimal setpoint "
                     "before answering."},
                    {"role": "user", "content": "Room 1203 is overheating — advise."}]
        for _ in range(5):
            try:
                r = client.chat.completions.create(model=config.MODEL, messages=messages,
                                                   tools=TOOLS, max_tokens=400, temperature=0.2)
            except Exception as e:  # noqa: BLE001
                view._endpoint_error(e); return
            msg = r.choices[0].message
            calls = msg.tool_calls or []
            if not calls:
                print("  · answer:", (msg.content or "").strip()[:300]); break
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name,
                                                          "arguments": c.function.arguments}} for c in calls]})
            for c in calls:
                args = json.loads(c.function.arguments or "{}")
                res = _impl(c.function.name, args)
                print(f"  → ACT {c.function.name}({args})")
                print(f"  ← OBSERVE {res}")
                messages.append({"role": "tool", "tool_call_id": c.id, "content": res})

    print("\nTakeaway: skills are pulled from a catalog and attached — the specialist gains")
    print("capabilities without new code. Next: run it SAFELY inside an OpenShell sandbox.")


if __name__ == "__main__":
    main()
