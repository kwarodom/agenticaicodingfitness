#!/usr/bin/env python3
"""PART 2 · Arize Phoenix + the OTel GenAI conventions  [BEGINNER]

Phoenix is an open-source, self-hostable LLM-observability UI — perfect for
sovereign deployments because it runs ON your DGX, so traces never leave either.
It ingests OpenTelemetry spans that follow the GenAI semantic conventions
(gen_ai.*). This demo shows the conventions, the one-time auto-instrumentation
setup, and (if Phoenix is running) confirms the live collector.

Run:  python demos/step02_phoenix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import obsview  # noqa: E402
import tracer as T  # noqa: E402


def main() -> None:
    obsview.banner("PART 2", "Arize Phoenix + OTel GenAI conventions", "BEGINNER")
    obsview.mode_line()

    print("Why Phoenix for sovereign AI: it's open-source and self-hosted, so it runs")
    print("on the SAME DGX as your model — traces (which contain prompts!) never leave.\n")

    print("The OTel GenAI semantic conventions every span should carry:")
    for k, v in [
        ("gen_ai.system", "the provider, e.g. 'openai' (your local endpoint speaks it)"),
        ("gen_ai.request.model", "which model served the call"),
        ("gen_ai.usage.input_tokens", "prompt tokens"),
        ("gen_ai.usage.output_tokens", "completion tokens"),
        ("openinference.span.kind", "LLM | TOOL | AGENT | CHAIN | RETRIEVER"),
        ("input.value / output.value", "the actual prompt + response (redact PII!)"),
    ]:
        print(f"  • {k:<30} {v}")
    print()

    tr = T.Tracer()
    print("One-time setup to stream your DGX agent's spans into real Phoenix:\n")
    print(tr.to_phoenix_hint())
    print()

    if config.phoenix_up():
        print(f"✓ Phoenix is LIVE at {config.PHOENIX_ENDPOINT} — open it to see traces.")
    else:
        print(f"ℹ Phoenix not running at {config.PHOENIX_ENDPOINT}. Start it with the")
        print("  command above; until then this app renders the span tree locally")
        print("  (same span shape, so it's a faithful preview of the Phoenix UI).")

    print("\nTakeaway: instrument once with OpenInference, point it at a Phoenix on your")
    print("DGX, and every agent run shows up in a searchable UI — fully on-prem.")


if __name__ == "__main__":
    main()
