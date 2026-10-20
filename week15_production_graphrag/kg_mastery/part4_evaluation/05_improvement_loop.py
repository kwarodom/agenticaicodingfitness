#!/usr/bin/env python3
"""Part 4.4 — The GraphRAG Improvement Loop tracker (kg_mastery.pdf §4.4).

KEY INSIGHT: Quality is not a one-shot result, it is a TREND. You evaluate, read
the scores, diagnose the worst metric, make ONE targeted change, then re-measure.
This utility logs each iteration's four RAGAS scores plus the action you took to
a JSON file, then prints the progression so improvement (or regression) is
obvious at a glance.

THE 6-STEP LOOP:
  1. DESIGN    decide the schema / retrieval strategy for the question types
  2. BUILD     implement it (loaders, Cypher templates, prompts)
  3. EVALUATE  run RAGAS over the test set -> 4 scores
  4. DIAGNOSE  find the lowest metric and map it to a failure mode
  5. IMPROVE   make ONE change that targets that failure mode
  6. REPEAT    go back to EVALUATE; keep the change only if scores rise

FAILURE-MODE TAXONOMY (full table in FAILURE_MODES.md):
  - Cypher Syntax Error        -> generated query won't run
  - Hallucinated Entities      -> answer cites nodes that don't exist
  - Wrong Relationship Direction-> traverses (a)<-(b) instead of (a)->(b)
  - Missing Context            -> retrieval misses needed nodes (low recall)
  - Over-Retrieval             -> too many irrelevant rows (low precision)
  - Low Faithfulness           -> claims not grounded in retrieved context

Run:  python week15/kg_mastery/part4_evaluation/05_improvement_loop.py
"""
import json
import sys
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "improvement_log.json"

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# PDF §4.4 example progression: a baseline that improves over three iterations
# as each diagnosed failure mode is fixed.
SEED_ROWS = [
    {
        "iter": 0,
        "faithfulness": 0.61,
        "answer_relevancy": 0.70,
        "context_precision": 0.55,
        "context_recall": 0.60,
        "action": "Baseline: naive vector retrieval, no graph traversal",
    },
    {
        "iter": 1,
        "faithfulness": 0.74,
        "answer_relevancy": 0.78,
        "context_precision": 0.68,
        "context_recall": 0.72,
        "action": "Add graph traversal for relationships (fix Missing Context)",
    },
    {
        "iter": 2,
        "faithfulness": 0.85,
        "answer_relevancy": 0.84,
        "context_precision": 0.79,
        "context_recall": 0.80,
        "action": "Tighten Cypher to top-k relevant rows (fix Over-Retrieval)",
    },
    {
        "iter": 3,
        "faithfulness": 0.91,
        "answer_relevancy": 0.88,
        "context_precision": 0.83,
        "context_recall": 0.82,
        "action": "Add 'answer only from context' guardrail (fix Low Faithfulness)",
    },
]


class IterationLog:
    """Append-only log of evaluation iterations, persisted as JSON."""

    def __init__(self, path=LOG_PATH):
        self.path = Path(path)
        self.rows = self._load()

    def _load(self):
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def append(self, row):
        """Add one iteration record and persist."""
        for m in METRICS:
            if m not in row:
                raise ValueError(f"row missing metric '{m}'")
        self.rows.append(row)
        self.path.write_text(json.dumps(self.rows, indent=2))
        return row

    def seed_if_empty(self, seed_rows):
        if not self.rows:
            for r in seed_rows:
                self.append(r)

    def print_trend(self):
        if not self.rows:
            print("(no iterations logged yet)")
            return
        header = f"{'iter':>4} | " + " | ".join(f"{m[:9]:>9}" for m in METRICS) + " | action"
        print(header)
        print("-" * len(header))
        prev = None
        for r in self.rows:
            cells = []
            for m in METRICS:
                val = r[m]
                if prev is not None:
                    delta = val - prev[m]
                    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
                else:
                    arrow = " "
                cells.append(f"{val:>7.2f}{arrow}")
            print(f"{r['iter']:>4} | " + " | ".join(cells) + f" | {r['action']}")
            prev = r

        first, last = self.rows[0], self.rows[-1]
        print("\nNet change vs baseline:")
        for m in METRICS:
            print(f"  {m:<18} {first[m]:.2f} -> {last[m]:.2f}  ({last[m]-first[m]:+.2f})")


def main():
    log = IterationLog()
    log.seed_if_empty(SEED_ROWS)

    print("=== GraphRAG Improvement Loop ===")
    print("DESIGN -> BUILD -> EVALUATE -> DIAGNOSE -> IMPROVE -> REPEAT\n")

    print("Iteration trends:")
    log.print_trend()

    print(f"\nLog persisted at: {log.path}")
    print("Append a new iteration in code with:")
    print("  log.append({'iter': 4, 'faithfulness': ..., 'answer_relevancy': ...,")
    print("              'context_precision': ..., 'context_recall': ...,")
    print("              'action': 'what you changed'})")


if __name__ == "__main__":
    main()
