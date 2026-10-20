#!/usr/bin/env python3
"""Exercise 02 · Layer map: place the components, name the threats, spot the cascade.

Fill in the three TODOs, save, then run:
    .venv/bin/python week28_maestro_mas_security/02_maestro_seven_layers/exercises/ex02_layer_map.py

The checker is offline and instant. Stuck? Compare with exercises/solutions/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import maestro  # noqa: E402
from ensemble import COMPONENTS  # noqa: E402
from maskit import banner, check, finish  # noqa: E402

# ── TODO 1 ── put each of the 17 components on ONE MAESTRO layer: "L1" … "L7".
#   L1 Foundation Models · L2 Data Operations · L3 Agent Frameworks · L4 Deployment and Infrastructure ·
#   L5 Evaluation and Observability · L6 Security and Compliance · L7 Agent Ecosystem
LAYER_OF = {
    "brain": "",
    "orchestrator": "",
    "concierge": "",
    "energy": "",
    "maintenance": "",
    "vendor": "",
    "agent_directory": "",
    "message_bus": "",
    "tools": "",
    "shared_memory": "",
    "sop_corpus": "",
    "web_pages": "",
    "email_gateway": "",
    "runtime": "",
    "audit_log": "",
    "pep": "",
    "human_approver": "",
}

# ── TODO 2 ── name the MAESTRO catalogue threat for each scenario, as an id like "L7.T4".
#   Look the ids up in the runner's 🗺 MAESTRO map, or run lab02_1 / lab02_2.
SCENARIO_THREAT = {
    # Someone publishes a second "Acme HVAC Parts" Agent Card with their own endpoint, and the
    # maintenance agent sends its parts order there, believing it is the real vendor.
    "s1": "",
    # A guest saves "energy policy is to set the chiller to 4" in shared memory. Later the energy agent
    # recalls it as if it were the hotel's own policy.
    "s2": "",
    # A guest asks the concierge to "email the guest list to" an outside address. The concierge really
    # does hold send_email; it is the goal that is wrong, not the tool.
    "s3": "",
}

# ── TODO 3 ── which TWO of MAESTRO's cross-layer threats (X1 … X5) describe this cascade?
#   A guest's note poisons shared memory (L2). The energy agent recalls it and sets the chiller for the
#   WHOLE hotel to 4 °C — an action no guest is allowed to take, done with the energy agent's powers.
CASCADE = set()          # e.g. {"X?", "X?"}


# ─────────────────────────── checker — no need to edit below ────────────────
ACCEPT_LAYER = {"brain": {"L1"}, "orchestrator": {"L3", "L7"}, "concierge": {"L7"}, "energy": {"L7"},
                "maintenance": {"L7"}, "vendor": {"L7"}, "agent_directory": {"L7"}, "message_bus": {"L3"},
                "tools": {"L3"}, "shared_memory": {"L2"}, "sop_corpus": {"L2"}, "web_pages": {"L2"},
                "email_gateway": {"L4"}, "runtime": {"L4"}, "audit_log": {"L5"}, "pep": {"L6"},
                "human_approver": {"L6"}}
ACCEPT_THREAT = {"s1": {"L7.T2"}, "s2": {"L2.T1", "L2.T4"}, "s3": {"L7.T4"}}
HINT_THREAT = {"s1": "someone POSES as a legitimate agent (L7)", "s2": "bad data goes INTO a data store (L2)",
               "s3": "a legitimate tool used in a way it was never meant for (L7)"}


def main() -> None:
    banner("Exercise 02 · layer map", "offline checker · instant", brain=False)
    good = True
    names = [c for c, _, _ in COMPONENTS]
    wrong = [c for c in names if LAYER_OF.get(c) not in ACCEPT_LAYER[c]]
    good &= check(not wrong, f"components: all {len(names)} on a sensible layer",
                  f"TODO 1: {len(wrong)} components need another layer: {', '.join(wrong[:6])}"
                  + (" …" if len(wrong) > 6 else "") + " — data stores are L2, agents L7, the PEP L6")
    for sid in ("s1", "s2", "s3"):
        got = SCENARIO_THREAT.get(sid, "").strip()
        try:
            name = maestro.threat(got)["name"] if got else ""
        except (StopIteration, IndexError, KeyError):
            name = ""
        good &= check(got in ACCEPT_THREAT[sid], f"{sid}: {got} {name}",
                      f"TODO 2: {sid} = {got or '(empty)'}{' ' + name if name else ''} — hint: {HINT_THREAT[sid]}")
    good &= check(CASCADE == {"X3", "X5"},
                  "cascade: X5 Goal Misalignment Cascades + X3 Privilege Escalation",
                  f"TODO 3: CASCADE={sorted(CASCADE) or '{}'} — one id for the bad goal spreading between "
                  "agents, one for a guest getting a power they do not have")
    finish(good, "Done. Module 03 turns this map into a threat model you can score, diff and gate.")


if __name__ == "__main__":
    main()
