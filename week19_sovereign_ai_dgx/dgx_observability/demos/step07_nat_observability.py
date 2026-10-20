#!/usr/bin/env python3
"""PART 7 · NAT observability → Phoenix (the full picture)  [ADVANCED]

NeMo Agent Toolkit has observability built in: add a `telemetry` block to the
workflow YAML and NAT exports OpenTelemetry spans to a tracer of your choice —
including Phoenix running on your DGX. This demo writes the telemetry config,
shows the end-to-end sovereign loop, and prints a production checklist.

Run:  python demos/step07_nat_observability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import obsview  # noqa: E402

TELEMETRY = """\
# add to hvac_workflow.yaml — NAT exports spans to Phoenix on your DGX
general:
  telemetry:
    tracing:
      phoenix:
        _type: phoenix
        endpoint: {phoenix}/v1/traces
        project: hvac-workflow
    logging:
      console:
        _type: console
        level: INFO
"""


def main() -> None:
    obsview.banner("PART 7", "NAT observability → Phoenix", "ADVANCED")
    obsview.mode_line()

    sb = config.ensure_sandbox()
    tel = TELEMETRY.format(phoenix=config.PHOENIX_ENDPOINT)
    (sb / "telemetry_block.yaml").write_text(tel)
    print("Add this telemetry block to the workflow → .sandbox/telemetry_block.yaml\n")
    print(tel)

    print("The complete sovereign, observable stack you've built across Week 19:\n")
    print("   ┌─ DGX hardware (GB10, 128 GB) ─────────────────────────────┐")
    print("   │  model: served by Ollama/vLLM   (App 1)                   │")
    print("   │  weights: fine-tuned to your domain   (App 2)             │")
    print("   │  agent: NAT workflow, react_agent + HITL   (this app)     │")
    print("   │  traces: OpenTelemetry → Phoenix on the SAME box          │")
    print("   │  evals:  LLM-judge using your OWN model                   │")
    print("   └───────────────────────────────────────────────────────────┘")
    print("   ⛔ not a prompt, a weight, a trace, or an eval leaves the building.\n")

    print("Production observability checklist for a sovereign agent:")
    for item in [
        "Trace EVERY run (sample if volume is high); spans follow gen_ai.* conventions.",
        "Run Phoenix on the DGX/VPN only — traces contain prompts (PII). Never public.",
        "Chart p50/p95 latency, tokens/run, tool-call count, and FAIL-rate per project.",
        "Alert on: latency p95 regressions, loop blow-ups (LLM calls/run), eval FAIL spikes.",
        "Gate releases in CI: a new prompt/model must not regress latency OR eval scores.",
        "Correlate agent metrics with GPU util/tok-s to decide: quantize, batch, or fix loop.",
    ]:
        print(f"   • {item}")

    print("\nA last simulated run, now fully observable end-to-end:\n")
    tr = obsview.traced_agent_run("Final check: clear the queue", project="hvac-workflow")
    obsview.show_tree(tr)

    print("\nTakeaway: sovereign doesn't mean blind. With NAT + Phoenix on the DGX you")
    print("get cloud-grade observability — and it, too, never leaves your hardware.")


if __name__ == "__main__":
    main()
