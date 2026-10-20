#!/usr/bin/env python3
"""Part 4.1 / 4.2 — Evaluating GraphRAG answers with RAGAS (kg_mastery.pdf §4.1/§4.2).

KEY INSIGHT: Building a GraphRAG system is half the job; the other half is
PROVING it works. RAGAS scores four independent dimensions of a RAG answer,
each on a 0..1 scale, using an LLM as the judge. You give it, per question:
  - question      the user's natural-language query
  - ground_truth  the known-correct answer (your gold label)
  - contexts      the chunks/rows the retriever actually returned
  - answer        what your GraphRAG pipeline generated

THE 4 METRICS (and what each one isolates):
  - faithfulness       Is every claim in `answer` supported by `contexts`?
                       Catches hallucination. Good: > 0.90
  - answer_relevancy   Does `answer` actually address the `question` asked
                       (not waffle)?                       Good: > 0.85
  - context_precision  Of the retrieved `contexts`, how much is relevant /
                       ranked high? Catches over-retrieval. Good: > 0.80
  - context_recall     Did retrieval pull ALL the context needed to support
                       the `ground_truth`? Catches misses. Good: > 0.75

INTERPRETING OUTPUT: results.to_pandas() gives one row per question plus the
per-metric scores. Low faithfulness with high relevancy = the model is making
things up confidently. Low context_recall = your retriever (Cypher / vector
search) is the bottleneck, not the generator. Fix retrieval before prompts.

Requires:
  - ANTHROPIC_API_KEY in your environment / repo-root .env
  - pip install ragas datasets pandas langchain-anthropic

Run:  python week15/kg_mastery/part4_evaluation/01_ragas_eval.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import LLM_MODEL


def _imports():
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from ragas.llms import LangchainLLMWrapper
        from langchain_anthropic import ChatAnthropic
        from datasets import Dataset
    except ImportError:
        print("⚠️  Missing dependency for RAGAS evaluation.")
        print("   pip install ragas datasets pandas langchain-anthropic")
        sys.exit(1)
    return (
        evaluate,
        [faithfulness, answer_relevancy, context_precision, context_recall],
        LangchainLLMWrapper,
        ChatAnthropic,
        Dataset,
    )


# Two hotel examples exactly mirroring the PDF walkthrough. In real use these
# come from your testset (see 02_testset_generation.py) + a live RAG run.
test_data = {
    "question": [
        "Which rooms had HVAC issues last week?",
        "Who resolved the alert for the air conditioner in room 305?",
    ],
    "ground_truth": [
        "Rooms 204 and 305 had HVAC issues last week.",
        "Technician Somchai resolved the air-conditioner alert in room 305.",
    ],
    "contexts": [
        [
            "Room 204 device AC-204 triggered a HIGH temperature alert on 2026-05-18.",
            "Room 305 air conditioner AC-305 triggered an HVAC fault alert on 2026-05-20.",
        ],
        [
            "MaintenanceJob MJ-77 (type=HVAC) RESOLVES alert ALT-305 for room 305.",
            "Staff Somchai (role=Technician) PERFORMED MaintenanceJob MJ-77.",
        ],
    ],
    "answer": [
        "Rooms 204 and 305 reported HVAC issues last week.",
        "The air-conditioner alert in room 305 was resolved by technician Somchai.",
    ],
}


def main():
    evaluate, metrics, LangchainLLMWrapper, ChatAnthropic, Dataset = _imports()

    # Use Claude as the judge LLM that scores each metric.
    eval_llm = LangchainLLMWrapper(ChatAnthropic(model=LLM_MODEL, temperature=0))

    dataset = Dataset.from_dict(test_data)

    print("Running RAGAS evaluation (LLM-as-judge over 4 metrics)...\n")
    results = evaluate(dataset, metrics=metrics, llm=eval_llm)

    df = results.to_pandas()
    metric_cols = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    print("\n=== RAGAS scores (0..1, higher is better) ===")
    print(df[metric_cols])

    print("\nGood-score thresholds:")
    print("  faithfulness     > 0.90")
    print("  answer_relevancy > 0.85")
    print("  context_precision> 0.80")
    print("  context_recall   > 0.75")


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set.")
        print("   export ANTHROPIC_API_KEY=sk-ant-...  (or add it to repo-root .env)")
        sys.exit(1)
    main()
