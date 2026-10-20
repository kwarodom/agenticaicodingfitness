#!/usr/bin/env python3
"""PART 1 · Observe — capture every tool & LLM call  [BEGINNER]

A long-running agent (Hermes, on OpenShell) makes many tool + model calls per turn.
NeMo Relay sits underneath and records each step as a telemetry SPAN (kind =
tool / llm / agent). This demo walks ONE Hermes turn being observed, printing each
captured span as it happens — the raw material Agent Insights + the flywheel use.

Run:  python demos/step01_observe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import view  # noqa: E402
import relaylab  # noqa: E402

INSTRUMENT = """\
# instrument once — every scope becomes an OpenInference span exported to Phoenix
pip install nemo-relay arize-phoenix
python -m phoenix.server.main serve                 # UI + OTLP collector on :6006

import nemo_relay as R
cfg = R.OpenInferenceConfig(); cfg.endpoint = "http://localhost:6006/v1/traces"
cfg.set_resource_attribute("openinference.project.name", "hermes-agent")
R.OpenInferenceSubscriber(cfg).register("openinference")
with R.scope.scope("hermes-turn", R.ScopeType.Agent):   # each scope = a captured span
    with R.scope.scope("tool:terminal", R.ScopeType.Tool): ...
"""

# one Hermes turn, as the Relay observes it (each row = a captured span)
TURN = [
    ("hermes-agent", "relay", "span start: hermes-turn", "call"),
    ("hermes-agent", "tool:terminal", "ls ./repo && cat TODO.md", "call"),
    ("tool:terminal", "hermes-agent", "3 files · TODO.md (412 B)", "ret"),
    ("hermes-agent", "llm", "plan next action (gpt-4.4-mini)", "call"),
    ("llm", "hermes-agent", "call execute_code to run tests", "ret"),
    ("hermes-agent", "tool:execute_code", "pytest -q", "call"),
    ("tool:execute_code", "hermes-agent", "12 passed in 1.8s", "ret"),
]


def _real_turn() -> None:
    """REAL: emit an actual Relay span tree to Phoenix (one real DGX inference)."""
    print(f"{relaylab.status_line()}\n")
    print("Instrumenting one Hermes turn and exporting each scope as a span:\n")
    print(INSTRUMENT)

    def llm_call(prompt: str) -> str:
        return view.generate(prompt, max_tokens=200, title="llm:plan (real, recorded as a span)")["answer"]

    spans = relaylab.observed_turn(llm_call)
    print("\nSpans captured this turn (each is now live in Phoenix):\n")
    for i, s in enumerate(spans, 1):
        print(f"  span {i:>2} [{s['kind']:<5}] {s['name']}")
        if s.get("in"):
            print(f"           in : {str(s['in'])[:80]}")
        if s.get("out"):
            print(f"           out: {str(s['out'])[:80]}")
    print(f"\n{len(spans)} real spans exported. Open the trace tree in Agent Insights:")
    print(f"  → {relaylab.PHOENIX_URL}   (project '{relaylab.PROJECT}')")
    print("\nTakeaway: you can't optimize what you can't see. These are REAL OpenInference")
    print("spans — status, latency and cost — ready for the router (App 08.3) and flywheel.")


def _sim_turn() -> None:
    """SIM: narrate one turn as canned spans (no GPU / no Phoenix needed)."""
    print("Hermes is a long-running agent on OpenShell. NeMo Relay observes it:\n")
    print(INSTRUMENT)
    print("One Hermes turn, as the Relay captures it (each line = one span):\n")
    for i, (frm, to, msg, kind) in enumerate(TURN, 1):
        arrow = "←" if kind == "ret" else "→"
        knd = "tool" if "tool:" in frm + to else ("llm" if "llm" in frm + to else "agent")
        print(f"  span {i:>2} [{knd:<5}] {frm} {arrow} {to}")
        print(f"           {msg}")
    print("\n7 spans captured for one turn. That is the OBSERVE layer: nothing is guessed,")
    print("every tool and model call is recorded with timing + cost, ready to export.\n")

    print("A one-line natural-language summary of what was observed "
          f"({config.MODEL}):\n")
    view.generate("In two sentences, why must a long-running agent record every tool and "
                  "LLM call as telemetry before you can improve it?", max_tokens=200,
                  title="why observe every call")
    print("\nTakeaway: you can't optimize what you can't see. Next: read these spans in")
    print("Agent Insights (Phoenix) — the trace tree with status, latency and cost.")
    print(f"\n(Tip: {relaylab.status_line()})")


def main() -> None:
    view.banner("PART 1", "Observe — capture every tool & LLM call", "BEGINNER")
    view.mode_line()

    if relaylab.ready():
        _real_turn()
    else:
        _sim_turn()


if __name__ == "__main__":
    main()
