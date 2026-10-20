#!/usr/bin/env python3
"""Part 4.6 — CI/CD quality gate for GraphRAG (kg_mastery.pdf §4.6).

KEY INSIGHT: Evaluation only protects you if it BLOCKS bad changes from
shipping. This is a `graphrag_eval.py`-style gate: it runs your GraphRAG
function over a fixed test set, scores it with RAGAS, compares the mean of each
metric against a threshold, and exits non-zero if ANY metric regresses. Wire it
into CI (GitHub Actions YAML at the bottom) so a prompt tweak that tanks
faithfulness fails the build instead of reaching production.

CONTRACT: rag_function(question) -> {"answer": str, "contexts": list[str]}
That is the only thing you swap in — point it at your real pipeline (the
GraphCypherQAChain from part3, etc.). The dummy below lets this script run
end-to-end conceptually without a live pipeline.

Requires (for real runs):
  - ANTHROPIC_API_KEY in your environment / repo-root .env
  - pip install ragas datasets pandas langchain-anthropic

Run:  python week15/kg_mastery/part4_evaluation/03_cicd_gate.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import LLM_MODEL

# Minimum acceptable mean score per metric. A build fails if any metric is
# below its threshold. (Looser than the "good" targets in 01_ragas_eval.py:
# these are the floor you refuse to ship below.)
THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.70,
}


def run_evaluation(rag_function, testset_csv):
    """Run rag_function over every row of testset_csv and score with RAGAS.

    Returns a dict: {metric: mean_score, ..., "passed": bool}.
    """
    try:
        import pandas as pd
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from ragas.llms import LangchainLLMWrapper
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        print("⚠️  Missing dependency for the CI/CD gate.")
        print("   pip install ragas datasets pandas langchain-anthropic")
        sys.exit(1)

    df = pd.read_csv(testset_csv)

    questions, ground_truths, answers, contexts = [], [], [], []
    for _, row in df.iterrows():
        q = row["question"]
        out = rag_function(q)  # {"answer", "contexts"}
        questions.append(q)
        ground_truths.append(row["ground_truth"])
        answers.append(out["answer"])
        contexts.append(out["contexts"])

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "ground_truth": ground_truths,
            "contexts": contexts,
            "answer": answers,
        }
    )

    eval_llm = LangchainLLMWrapper(ChatAnthropic(model=LLM_MODEL, temperature=0))
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    results = evaluate(dataset, metrics=metrics, llm=eval_llm)

    rdf = results.to_pandas()
    scores = {m: float(rdf[m].mean()) for m in THRESHOLDS}
    scores["passed"] = all(scores[m] >= THRESHOLDS[m] for m in THRESHOLDS)
    return scores


# ── A dummy pipeline so this file runs without a live GraphRAG system. ───────
def dummy_rag_function(question):
    """Canned GraphRAG stand-in. Replace with your real pipeline."""
    return {
        "answer": "Rooms 204 and 305 had HVAC issues last week; "
        "technician Somchai resolved the room 305 alert.",
        "contexts": [
            "Room 204 device AC-204 triggered a HIGH temperature alert on 2026-05-18.",
            "Room 305 air conditioner AC-305 triggered an HVAC fault alert on 2026-05-20.",
            "Staff Somchai PERFORMED MaintenanceJob MJ-77 which RESOLVES ALT-305.",
        ],
    }


def _write_inline_testset():
    """Write a tiny gold test set to a temp CSV and return its path."""
    import csv

    rows = [
        {
            "question": "Which rooms had HVAC issues last week?",
            "ground_truth": "Rooms 204 and 305 had HVAC issues last week.",
        },
        {
            "question": "Who resolved the alert for room 305?",
            "ground_truth": "Technician Somchai resolved the alert for room 305.",
        },
    ]
    fd, path = tempfile.mkstemp(suffix=".csv", prefix="testset_")
    os.close(fd)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "ground_truth"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    testset_csv = _write_inline_testset()
    print(f"Using inline test set: {testset_csv}\n")
    try:
        scores = run_evaluation(dummy_rag_function, testset_csv)
    finally:
        os.unlink(testset_csv)

    print("=== Quality gate ===")
    for m, threshold in THRESHOLDS.items():
        status = "PASS" if scores[m] >= threshold else "FAIL"
        print(f"  {m:<18} {scores[m]:.3f}  (>= {threshold})  [{status}]")
    print(f"\nOverall: {'PASSED' if scores['passed'] else 'FAILED'}")

    sys.exit(0 if scores["passed"] else 1)


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set.")
        print("   export ANTHROPIC_API_KEY=sk-ant-...  (or add it to repo-root .env)")
        sys.exit(1)
    main()


# ─────────────────────────────────────────────────────────────────────────────
# GitHub Actions workflow — save as .github/workflows/graphrag_eval.yml
# ─────────────────────────────────────────────────────────────────────────────
#
# name: GraphRAG Evaluation
#
# on:
#   pull_request:
#     branches: [main]
#   push:
#     branches: [main]
#
# jobs:
#   evaluate:
#     runs-on: ubuntu-latest
#     steps:
#       - uses: actions/checkout@v4
#
#       - name: Set up Python
#         uses: actions/setup-python@v5
#         with:
#           python-version: "3.11"
#
#       - name: Install dependencies
#         run: |
#           pip install ragas datasets pandas langchain-anthropic
#
#       - name: Run GraphRAG quality gate
#         env:
#           ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
#         run: |
#           python graphrag_eval.py   # exits non-zero if any metric regresses
#
# The job fails the PR check whenever run_evaluation() returns passed=False,
# blocking merges that degrade answer quality.
