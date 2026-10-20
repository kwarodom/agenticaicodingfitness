#!/usr/bin/env python3
"""Lab 02-2 · Decompose the Hotel Ops Ensemble into MAESTRO layers, then find the threats that apply.

Offline and instant. MAESTRO's first step is "System Decomposition": put every component on one of the
seven layers. This lab does that for the 17 components in ensemble.COMPONENTS with a reference mapping
(the course's own answer — Exercise 02 asks you to make yours), then walks each layer's threat landscape
from the CSA page and keeps the threats that touch a hotel component, with one line on how each would
show up at the AltoTech Grand.

Run: .venv/bin/python week28_maestro_mas_security/02_maestro_seven_layers/labs/lab02_2_decompose.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import maestro  # noqa: E402
from ensemble import COMPONENTS, ROLE_TOOLS, TOOLS  # noqa: E402
from maskit import banner, dim, note, ok, result, step, table  # noqa: E402

# The course's reference decomposition (NOT from MAESTRO: the framework names the layers, you place the parts).
REFERENCE_LAYER = {
    "brain": "L1",
    "shared_memory": "L2", "sop_corpus": "L2", "web_pages": "L2",
    "orchestrator": "L3", "message_bus": "L3", "tools": "L3",
    "email_gateway": "L4", "runtime": "L4",
    "audit_log": "L5",
    "pep": "L6", "human_approver": "L6",
    "concierge": "L7", "energy": "L7", "maintenance": "L7", "vendor": "L7", "agent_directory": "L7",
}

# Catalogue threats that apply to this hotel: id → (components it touches, how it would show up here).
HOTEL = {
    "L1.T1": (["brain"], "a guest message carries an extra instruction it obeys"),
    "L1.T3": (["brain"], "a downloaded model with a hidden trigger phrase"),
    "L1.T6": (["brain"], "talked into a task nobody gave it (e.g. bulk email)"),
    "L1.T7": (["brain"], "long requests stall every agent sharing one model"),
    "L2.T1": (["shared_memory"], "a guest's 'remember that…' note becomes house policy"),
    "L2.T2": (["shared_memory", "email_gateway"], "the guest list leaves through an email"),
    "L2.T4": (["sop_corpus"], "someone edits SOP-HVAC-07 after it was published"),
    "L2.T5": (["sop_corpus", "web_pages"], "a review page or SOP feeds instructions into retrieval"),
    "L3.T3": (["orchestrator", "message_bus"], "the orchestrator forwards raw guest text to a specialist"),
    "L3.T4": (["tools", "runtime"], "a poisoned package in the agent framework's dependencies"),
    "L3.T5": (["message_bus"], "delegations that multiply until the bus is saturated"),
    "L3.T6": (["tools"], "a tool reached via a delegation the PEP never sees"),
    "L4.T1": (["runtime"], "the agent container image is swapped for a tampered one"),
    "L4.T4": (["runtime"], "a request loop burns the runtime's whole action budget"),
    "L4.T6": (["runtime", "email_gateway"], "one compromised agent process reaches the others"),
    "L5.T4": (["audit_log"], "a harmful change that never triggers an alert"),
    "L5.T5": (["audit_log"], "guest names and emails copied into logs and traces"),
    "L5.T6": (["audit_log"], "an audit row deleted after the fact"),
    "L6.T2": (["pep"], "a request worded so the policy check misses it"),
    "L6.T4": (["pep", "audit_log"], "guest personal data handled outside what PDPA allows"),
    "L6.T6": (["pep", "human_approver"], "a denial or approval with no reason anyone can audit"),
    "L7.T1": (["vendor"], "a vendor agent that behaves badly once it is trusted"),
    "L7.T2": (["vendor", "agent_directory"], "a look-alike 'Acme HVAC Parts' Agent Card"),
    "L7.T3": (["concierge", "energy"], "a guest acts as if they were another room's guest"),
    "L7.T4": (["concierge", "energy", "maintenance"], "a legitimate tool used for an illegitimate goal"),
    "L7.T5": (["energy"], "'set the chiller to 3' rides along with a comfort request"),
    "L7.T9": (["energy", "maintenance"], "no one can prove which agent changed the chiller"),
    "L7.T10": (["agent_directory"], "a fake listing injected into the vendor directory"),
    "L7.T11": (["agent_directory"], "discovery returns the newest card, not the genuine one"),
    "L7.T13": (["vendor"], "a card that claims skills the vendor does not have"),
}

banner("Lab 02-2 · decompose the ensemble", "offline · reference mapping is the course's own", brain=False)

step(1, "the system inventory: 17 components (ensemble.COMPONENTS)")
table([[cid, kind, what] for cid, kind, what in COMPONENTS], ["component", "kind", "what it is"], maxw=64)
agents = [a for a in ROLE_TOOLS if a != "orchestrator"]
crit = sorted(t for t, (imp, _) in TOOLS.items() if imp == "critical")
dim(f"specialist agents: {', '.join(agents)} · critical tools: {', '.join(crit)}")

step(2, "MAESTRO step 1 — System Decomposition: one layer per component")
missing = [c for c, _, _ in COMPONENTS if c not in REFERENCE_LAYER]
counts = Counter(REFERENCE_LAYER.values())
for lid in reversed(maestro.LAYER_ORDER):
    names = [c for c, _, _ in COMPONENTS if REFERENCE_LAYER.get(c) == lid]
    print(f"│ {lid} {maestro.layer(lid)['name']:<31} {', '.join(names)}")
ok("every component has a layer") if not missing else print(f"✕ unmapped: {missing}")
note(f"L7 holds {counts['L7']} components — agents, the vendor and the directory are the ecosystem")

step(3, "MAESTRO step 2 — Layer-Specific Threat Modeling: which catalogue threats touch a hotel component")
applies_total, cat_total = 0, 0
for lid in reversed(maestro.LAYER_ORDER):
    L = maestro.layer(lid)
    hits = [t for t in L["threats"] if t["id"] in HOTEL]
    applies_total += len(hits)
    cat_total += len(L["threats"])
    print(f"\n◆ {lid} {L['name']} — {len(hits)} of {len(L['threats'])} catalogue threats apply")
    table([[t["id"], t["name"], ", ".join(HOTEL[t["id"]][0]), HOTEL[t["id"]][1]] for t in hits],
          ["id", "MAESTRO threat", "components", "how it would show up here"], maxw=58)
    skipped = [t["name"] for t in L["threats"] if t["id"] not in HOTEL]
    if skipped:
        dim("not modelled here: " + "; ".join(skipped))

step(4, "sanity: every threat points at components that exist, on a matching layer")
bad = [tid for tid, (comps, _) in HOTEL.items() if any(c not in REFERENCE_LAYER for c in comps)]
cross = [tid for tid, (comps, _) in HOTEL.items() if {REFERENCE_LAYER[c] for c in comps} - {tid.split(".")[0]}]
ok("all component names resolve") if not bad else print(f"✕ unknown components in {bad}")
note(f"{len(cross)} threats touch a component on ANOTHER layer: {', '.join(cross)}")
dim("that is MAESTRO's point about layers interacting — step 3 (cross-layer) starts from exactly these")

result(f"{len(REFERENCE_LAYER)} components on 7 layers · {applies_total} of {cat_total} catalogue threats apply "
       "to this hotel (the course's judgement) · the rest live in the runner's 🗺 MAESTRO map")
