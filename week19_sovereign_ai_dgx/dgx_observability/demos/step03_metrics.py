#!/usr/bin/env python3
"""PART 3 · The metrics that matter on a DGX  [INTERMEDIATE]

Tracing gives you raw spans; observability is the metrics you derive. For a
sovereign agent the key numbers are latency (TTFT + total), throughput (tok/s),
token volume, and how those correlate with GPU utilization. This demo runs the
agent, then breaks the trace down into the metrics you'd chart in Phoenix.

Run:  python demos/step03_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import obsview  # noqa: E402
import tracer as T  # noqa: E402

TASK = "Triage room 1203 (occupied, 26.4 °C) and dispatch if guest-impacting."


def main() -> None:
    obsview.banner("PART 3", "The metrics that matter on a DGX", "INTERMEDIATE")
    obsview.mode_line()

    tr = obsview.traced_agent_run(TASK)
    t = tr.totals()

    print("Per-span latency waterfall (where the wall-clock went):")
    for s in tr.spans:
        if s.kind in (T.LLM, T.TOOL):
            bar = "█" * max(1, int(s.latency_ms / 20))
            print(f"  {s.kind:<5} {s.name:<22} {s.latency_ms:>6.0f} ms  {bar}")
    print()

    llm_ms = sum(s.latency_ms for s in tr.spans if s.kind == T.LLM)
    tool_ms = sum(s.latency_ms for s in tr.spans if s.kind == T.TOOL)
    print("Derived metrics (what you'd alert on):")
    print(f"  ◆ total latency        {t['latency_ms']:>7.0f} ms")
    print(f"  ◆ time in LLM calls    {llm_ms:>7.0f} ms  ({100*llm_ms/max(t['latency_ms'],1):.0f}%)")
    print(f"  ◆ time in tools        {tool_ms:>7.0f} ms")
    print(f"  ◆ tokens (in→out)      {t['input_tokens']}→{t['output_tokens']} = {t['total_tokens']}")
    print(f"  ◆ LLM calls per task   {t['llm_calls']}   (fewer = cheaper + faster)")
    print(f"  ◆ cloud cost           $0.0000  (it's your DGX)")
    print()

    print("Correlate with GPU telemetry (the sovereign-specific view):")
    print("  • tok/s low + GPU util low   → you're latency-bound (batch=1); try vLLM.")
    print("  • tok/s low + GPU util ~100% → memory-bandwidth bound; quantize (NVFP4).")
    print("  • spikes in LLM-call COUNT   → the agent is looping; check the prompt/tools.")
    print("  • Phoenix lets you chart p50/p95 latency + tokens over time, per project.")

    print("\nTakeaway: on-prem you don't watch a $ meter — you watch latency, tok/s, and")
    print("GPU util. Those three tell you whether to quantize, batch, or fix the loop.")


if __name__ == "__main__":
    main()
