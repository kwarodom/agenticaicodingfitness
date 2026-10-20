#!/usr/bin/env python3
"""PART 6 · NeMo Agent Toolkit — a YAML workflow on your DGX  [ADVANCED]

NAT's superpower is separating agent LOGIC from agent INSTANTIATION: you compose
tools, LLMs, and agents in a YAML workflow. This demo writes a real NAT workflow
that points its LLM at your DGX (the local OpenAI-compatible endpoint) and wires
a supervisor → HVAC-specialist with human-in-the-loop approval before dispatch.

Run:  python demos/step06_nat_workflow.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import obsview  # noqa: E402

WORKFLOW = """\
# hvac_workflow.yaml — NeMo Agent Toolkit, pointed at YOUR DGX (sovereign)
llms:
  dgx_local:                         # the LLM all agents use
    _type: openai                    # OpenAI-compatible → works with Ollama/vLLM
    base_url: {base_url}             # ← your DGX endpoint, not the cloud
    model_name: {model}
    temperature: 0.2

functions:
  get_room_telemetry:
    _type: get_room_telemetry        # the tool we registered in step 5
  dispatch_work_order:
    _type: dispatch_work_order
  approval:
    _type: human_in_the_loop         # pause for an operator before acting

workflow:
  _type: react_agent                 # supervisor reasons + routes to tools/specialists
  llm_name: dgx_local
  tool_names: [get_room_telemetry, dispatch_work_order, approval]
  max_iterations: 6
  system_prompt: >
    You are the HVAC operations supervisor. Read telemetry, decide priority
    (occupied + far from setpoint = CRITICAL; empty + fouled filter = ROUTINE),
    and call `approval` before any CRITICAL dispatch.
"""

RUN = """\
# run / serve / evaluate the workflow with the NAT CLI (all on the DGX):
nat run    --config_file hvac_workflow.yaml --input "Clear the alarm queue"
nat serve  --config_file hvac_workflow.yaml        # OpenAI-compatible API
nat eval   --config_file hvac_workflow.yaml        # score against a dataset
"""


def main() -> None:
    obsview.banner("PART 6", "NeMo Agent Toolkit — a YAML workflow", "ADVANCED")
    obsview.mode_line()

    sb = config.ensure_sandbox()
    wf = WORKFLOW.format(base_url=config.BASE_URL, model=config.MODEL)
    (sb / "hvac_workflow.yaml").write_text(wf)
    print("Wrote a real NAT workflow → .sandbox/hvac_workflow.yaml\n")
    print(wf)
    print("Run it on the DGX:\n")
    print(RUN)

    print("Why this matters:")
    print("  • The `llms.dgx_local.base_url` is the ONLY sovereignty knob — point it at")
    print("    your DGX and every agent + sub-agent runs on-prem. No code change.")
    print("  • Logic (system_prompt, routing, HITL) lives in YAML → versioned, reviewable,")
    print("    swappable without touching Python.")
    print("  • `human_in_the_loop` injects an approval gate before a CRITICAL dispatch —")
    print("    the Week 10 HITL pattern, declared in config.")
    print("  • `nat run / serve / eval` give you the whole lifecycle from one file.")

    print("\nSimulating the supervisor clearing the queue (a real `nat run` does this")
    print("against your DGX model):\n")
    tr = obsview.traced_agent_run("Clear the alarm queue", project="nat-hvac-workflow")
    obsview.show_tree(tr)

    print("\nTakeaway: NAT turns the bespoke loop from Part 1 into a declarative,")
    print("sovereign workflow — same DGX model, now config-driven and serveable.")


if __name__ == "__main__":
    main()
