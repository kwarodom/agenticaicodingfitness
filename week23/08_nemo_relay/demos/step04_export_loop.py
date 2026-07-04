#!/usr/bin/env python3
"""PART 4 · Learn — export telemetry & close the loop  [ADVANCED]

NeMo Relay speaks OpenTelemetry (OTel), so ONE instrumentation fans out to many
backends: Phoenix (Agent Insights), Datadog (ops dashboards + alerts), LangSmith
(LLM evals + datasets). Then the observed traces feed back into the Data Flywheel
(App 11): curate the failing / expensive turns → fine-tune → the agent gets better
and cheaper. Observe → Learn → Optimize becomes a closing loop.

Run:  python demos/step04_export_loop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import view  # noqa: E402
import relaylab  # noqa: E402

EXPORT = """\
# one instrumentation, many backends — the Relay fans OTel spans out
relay = Relay(service="hermes-agent", export=["phoenix", "datadog", "langsmith"])
relay.instrument()
# spans flow over OTLP; each backend reads the same trace:
#   phoenix   → Agent Insights (span tree, cost, latency)
#   datadog   → ops dashboards + latency/cost alerts
#   langsmith → LLM evals + build golden datasets from real turns
"""

EXPORTERS = [
    ("Phoenix",   "Agent Insights — trace tree, per-span cost & latency"),
    ("Datadog",   "ops dashboards + alerts on latency / cost spikes"),
    ("LangSmith", "LLM evals + golden datasets from real production turns"),
]

LOOP = [
    ("Relay traces",   "collect every Hermes turn (spans + cost + status)"),
    ("curate",         "pick failing / expensive / slow turns as examples"),
    ("Data Flywheel",  "fine-tune a smaller model on those curated turns (App 11)"),
    ("re-route",       "router now sends more traffic to the cheaper tuned model"),
    ("observe again",  "Relay measures the new cost/quality — loop repeats"),
]


def _live_export() -> None:
    """REAL: prove the OTLP export to Phoenix is live, with actual collected counts."""
    stats = relaylab.trace_stats()
    print(f"{relaylab.status_line()}\n")
    print("Exporter fan-out (Phoenix is LIVE on this box; the others are where you'd add them):\n")
    print(f"  NeMo Relay ──OTLP──▶ Phoenix ✓ LIVE   {stats['spans']} spans · "
          f"{stats['traces']} traces · {stats['total_latency_ms']:.0f} ms observed  ({relaylab.PHOENIX_URL})")
    print("  NeMo Relay ──OTLP──▶ Datadog          ops dashboards + alerts on latency / cost spikes")
    print("  NeMo Relay ──OTLP──▶ LangSmith        LLM evals + golden datasets from real turns")
    if stats["spans"] == 0:
        print("\n  (No spans yet — run step01/step02/step03 first to populate the live trace store.)")


def main() -> None:
    view.banner("PART 4", "Learn — export telemetry & close the loop", "ADVANCED")
    view.mode_line()

    if relaylab.ready():
        _live_export()
    else:
        print("NeMo Relay exports OTel once; every observability backend reads it:\n")
        print(EXPORT)
        print("Exporter fan-out:\n")
        for name, what in EXPORTERS:
            print(f"  NeMo Relay ──OTLP──▶ {name:<10}  {what}")
        print(f"\n(Tip: {relaylab.status_line()})")

    print("\nClosing the loop — observed traces feed the Data Flywheel (App 11):\n")
    for i, (stage, what) in enumerate(LOOP, 1):
        arrow = "   ↺" if i == len(LOOP) else "    "
        print(f"  {i}. {stage:<14} {what}")
        if i < len(LOOP):
            print("        │")
    print(f"     ╰──────────{arrow}  back to step 1 (self-improving agent)\n")

    print(f"A one-line takeaway on closing the loop ({config.MODEL}):\n")
    view.generate("In two sentences, why is exporting agent telemetry to Phoenix/Datadog/"
                  "LangSmith and feeding it into a fine-tuning flywheel a self-improving loop?",
                  max_tokens=200, title="why close the loop")
    print("\nTakeaway: observe → learn → optimize is a CYCLE, not a one-shot. NeMo Relay is")
    print("the bus that makes it turn. See the Appendix for where it sits in Week 23.")


if __name__ == "__main__":
    main()
