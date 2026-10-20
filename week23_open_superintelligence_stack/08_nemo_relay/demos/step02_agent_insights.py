#!/usr/bin/env python3
"""PART 2 · Agent Insights with Phoenix — read the trace  [INTERMEDIATE]

Agent Insights is the Phoenix UI over the spans NeMo Relay exports. This demo
renders a Phoenix-style TRACE of one Hermes turn: the span table (hermes-turn →
llm → tool: terminal / execute_code) with Status ✓, Latency and Total Cost, then a
per-span DETAIL view (the assistant message; the raw tool result is omitted). This
is how you LEARN from a run — spot the slow span, the expensive span, the failure.

Run:  python demos/step02_agent_insights.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import view  # noqa: E402
import relaylab  # noqa: E402

OPEN_UI = """\
# Agent Insights = a Phoenix instance reading the Relay's OpenInference spans
python -m phoenix.server.main serve             # → http://localhost:6006
from phoenix.client import Client
df = Client(base_url="http://localhost:6006").spans.get_spans_dataframe(project_name="hermes-agent")
# each row is a span: name · kind · status · start/end (→ latency) · parent_id (→ tree)
"""

# a Phoenix-style span tree for one trace (name, kind, status, latency_ms, cost_usd, depth)
SPANS = [
    ("hermes-turn",          "agent", "✓", 2840, 0.0121, 0),
    ("llm · plan",           "llm",   "✓",  610, 0.0032, 1),
    ("tool · terminal",      "tool",  "✓",  140, 0.0000, 1),
    ("llm · decide",         "llm",   "✓",  520, 0.0028, 1),
    ("tool · execute_code",  "tool",  "✓", 1180, 0.0000, 2),
    ("llm · summarize",      "llm",   "✓",  390, 0.0061, 1),
]


def _tree() -> None:
    print("  Phoenix · Traces  →  trace  a1f9c2  (Hermes turn)")
    print("  ┌──────────────────────────────┬───────┬────────┬──────────┬──────────┐")
    print("  │ span                         │ kind  │ status │ latency  │ cost     │")
    print("  ├──────────────────────────────┼───────┼────────┼──────────┼──────────┤")
    for name, kind, status, lat, cost, depth in SPANS:
        indent = "  " * depth
        label = (indent + name)[:28].ljust(28)
        print(f"  │ {label} │ {kind:<5} │   {status}    │ {lat:>5} ms │ ${cost:0.4f} │")
    print("  └──────────────────────────────┴───────┴────────┴──────────┴──────────┘")
    total_ms = SPANS[0][3]
    total_cost = sum(s[4] for s in SPANS[1:])
    print(f"  Status ✓   ·   Total Latency {total_ms} ms   ·   Total Cost ${total_cost:0.4f}")


def _detail() -> None:
    print("\n  ── span detail · llm · summarize ─────────────────────────────")
    print("  attributes:")
    print('    llm.model_name   = "gpt-4.4-mini"')
    print("    llm.token_count.prompt      = 812")
    print("    llm.token_count.completion  = 143")
    print("  input.value (assistant message):")
    print('    "Tests pass (12/12). I fixed the null check in parse() and')
    print('     re-ran the suite. Opening a PR against main."')
    print("  output / tool result: (omitted — expand the tool span to view)")


def _real_tree() -> None:
    """REAL: render the actual spans Phoenix collected (real latency & status)."""
    spans = relaylab.fetch_spans()
    if not spans:
        print("  (no spans yet — recording one real Hermes turn first…)\n")
        relaylab.observed_turn(lambda p: view.generate(p, max_tokens=160)["answer"])
        spans = relaylab.fetch_spans()
    print(f"  Phoenix · project '{relaylab.PROJECT}'  →  {len(spans)} spans (slowest first)")
    print("  ┌──────────────────────────────┬───────┬────────┬────────────┐")
    print("  │ span                         │ kind  │ status │ latency    │")
    print("  ├──────────────────────────────┼───────┼────────┼────────────┤")
    for s in spans:
        label = s["name"][:28].ljust(28)
        print(f"  │ {label} │ {s['kind'][:5]:<5} │   {s['status']}    │ {s['latency_ms']:>7.1f} ms │")
    print("  └──────────────────────────────┴───────┴────────┴────────────┘")
    slowest = spans[0]
    print(f"  Slowest span: {slowest['name']} ({slowest['latency_ms']:.1f} ms). "
          f"Cost = $0.0000 (sovereign · on your DGX).")


def _real() -> None:
    print(f"{relaylab.status_line()}\n")
    print("NeMo Relay exports OpenInference spans; Phoenix (Agent Insights) reads them:\n")
    print(OPEN_UI)
    print("The REAL trace, read back from Phoenix (these are your actual spans):\n")
    _real_tree()
    print(f"\n  Open the interactive trace tree → {relaylab.PHOENIX_URL}  (project '{relaylab.PROJECT}')")
    print("\nReading the trace you SEE the slow span immediately — the same signal the router")
    print("(next) and the flywheel (App 11) act on.")

    print(f"\nA one-line takeaway from the trace ({config.MODEL}):\n")
    view.generate("In two sentences, what does a Phoenix trace tree (spans with status and "
                  "latency) let an engineer do that raw logs do not?",
                  max_tokens=200, title="what the trace tells you")


def _sim() -> None:
    print("NeMo Relay exports OTel spans; Phoenix (Agent Insights) reads them:\n")
    print(OPEN_UI)
    print("The trace tree for one Hermes turn, as Agent Insights shows it:\n")
    _tree()
    _detail()

    print("\nReading the trace, you can immediately see: execute_code is the slow span")
    print("(1180 ms) and summarize is the expensive one ($0.0061) — candidates to optimize.\n")

    print(f"A one-line takeaway from the trace ({config.MODEL}):\n")
    view.generate("In two sentences, what does a Phoenix trace tree (spans with status, "
                  "latency and cost) let an engineer do that raw logs do not?",
                  max_tokens=200, title="what the trace tells you")
    print(f"\n(Tip: {relaylab.status_line()})")


def main() -> None:
    view.banner("PART 2", "Agent Insights with Phoenix — read the trace", "INTERMEDIATE")
    view.mode_line()

    if relaylab.ready():
        _real()
    else:
        _sim()
    print("\nTakeaway: the trace turns a black-box turn into a readable tree — you SEE the")
    print("slow and costly spans. Next: OPTIMIZE — route each call to the right-sized model.")


if __name__ == "__main__":
    main()
