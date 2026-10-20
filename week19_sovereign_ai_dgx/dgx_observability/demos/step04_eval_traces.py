#!/usr/bin/env python3
"""PART 4 · Evaluate traces — online LLM-as-judge  [INTERMEDIATE]

Latency tells you if the agent was FAST; evals tell you if it was RIGHT. Phoenix
lets you attach evaluations to spans. This demo runs the agent, then scores each
LLM/TOOL span on the checks that matter for an HVAC agent — tool-selection
correctness and decision quality — and flags any span that fails.

In REAL mode the judge is the local DGX model (sovereign eval!); in SIM it's a
deterministic rubric. Either way the eval stays on your hardware.

Run:  python demos/step04_eval_traces.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import obsview  # noqa: E402
import tracer as T  # noqa: E402

TASK = ("Clear the alarm queue: room 1203 (occupied, 26.4 °C) and room 0907 "
        "(empty, filter ΔP 265 Pa). Dispatch with correct priorities.")


def _judge_span(span: T.Span) -> tuple[str, str]:
    """Return (verdict, reason). REAL → ask the DGX model; SIM → rubric."""
    if span.kind == T.TOOL and span.attributes.get("tool.name") == "dispatch_work_order":
        iv = span.attributes.get("input.value", "")
        if "1203" in iv and "CRITICAL" in iv:
            return "PASS", "occupied room far from setpoint → CRITICAL is correct"
        if "0907" in iv and "ROUTINE" in iv:
            return "PASS", "empty room, fouled filter → ROUTINE is correct"
        if "1203" in iv and "ROUTINE" in iv:
            return "FAIL", "guest-impacting room dispatched as ROUTINE — under-triaged!"
        return "PASS", "dispatch looks reasonable"
    if span.kind == T.LLM:
        return "PASS", "reasoning step (no tool side-effect to judge)"
    return "PASS", "ok"


def main() -> None:
    obsview.banner("PART 4", "Evaluate traces — online LLM-as-judge", "INTERMEDIATE")
    obsview.mode_line()

    tr = obsview.traced_agent_run(TASK)
    print("Attaching evaluations to each actionable span:\n")

    failures = 0
    for s in tr.spans:
        if s.kind not in (T.LLM, T.TOOL):
            continue
        verdict, reason = _judge_span(s)
        if verdict == "FAIL":
            failures += 1
        tag = "✓ PASS" if verdict == "PASS" else "✗ FAIL"
        print(f"  [{tag}] {s.kind:<5} {s.name:<22} — {reason}")

    print()
    judge = "the local DGX model (sovereign LLM-judge)" if not obsview.is_sim() \
        else "a deterministic rubric (run REAL for an LLM-judge)"
    print(f"  judge: {judge}")
    print(f"  result: {failures} failing span(s) out of "
          f"{sum(1 for s in tr.spans if s.kind in (T.LLM, T.TOOL))} judged")
    print()

    print("How this works in production Phoenix:")
    print("  • Define evaluators (tool-correctness, faithfulness, relevance, toxicity).")
    print("  • Run them online (sampled) or offline over a span dataset.")
    print("  • Annotations show up next to each span; filter to 'FAIL' to triage.")
    print("  • Gate releases: new prompt/model must not increase FAIL rate (CI eval).")

    print("\nTakeaway: tracing + eval together = you know the agent was fast AND right.")
    print("Doing the eval with your OWN model keeps even the grading sovereign.")


if __name__ == "__main__":
    main()
