# Part 4 — Evaluating GraphRAG Systems

Runnable companions to **`kg_mastery.pdf` Part 4 (Evaluating GraphRAG Systems)**.
Where Parts 1–3 *build* a knowledge graph and a GraphRAG pipeline, Part 4 is
about *measuring whether it actually works* — and keeping it working. These
scripts use **RAGAS** (LLM-as-judge) to score answer quality, generate test
sets, gate CI/CD, monitor production, and drive an iterative improvement loop.

> These files measure quality; they do not build the graph. Point the eval
> harness at your own pipeline (e.g. the `GraphCypherQAChain` from Part 3).

## Files

| # | File | PDF § | What it is | Needs |
|---|------|-------|-----------|-------|
| 1 | `01_ragas_eval.py` | 4.1 / 4.2 | Score 2 hotel Q&A examples on the 4 RAGAS metrics | `ANTHROPIC_API_KEY`, `ragas datasets pandas langchain-anthropic` |
| 2 | `02_testset_generation.py` | 4.5 | Auto-generate a test set from `hotel_docs/` → `hotel_kg_testset.csv` | `ANTHROPIC_API_KEY`, + `langchain-community` |
| 3 | `03_cicd_gate.py` | 4.6 | CI/CD quality gate: fail the build if any metric regresses (with GitHub Actions YAML) | `ANTHROPIC_API_KEY` for real runs |
| 4 | `04_production_monitoring.py` | 4.7 | `@monitored_graph_query` decorator + metrics summary (latency, success/empty rate) | Neo4j optional |
| 5 | `05_improvement_loop.py` | 4.4 | Iteration tracker: logs the 4 scores + action per iteration, prints the trend | none (stdlib) |
| — | `FAILURE_MODES.md` | 4.3 | Failure-mode taxonomy (root cause / symptom / fix) + loop diagram | — |
| — | `hotel_docs/` | 4.5 | Two source `.txt` incident reports used to synthesize the test set | — |

## The 4 RAGAS metrics (and good-score targets)

| Metric | Catches | Good score |
|---|---|---|
| `faithfulness` | hallucination — claims not in the retrieved context | > 0.90 |
| `answer_relevancy` | answers that dodge the question | > 0.85 |
| `context_precision` | over-retrieval — irrelevant context | > 0.80 |
| `context_recall` | missing context — retrieval gaps | > 0.75 |

The CI gate (`03_cicd_gate.py`) uses *looser* floors (0.85 / 0.80 / 0.75 / 0.70)
as the line below which a change must not ship.

## Dependencies

```bash
pip install ragas datasets pandas langchain-anthropic
# script 2 also needs:
pip install langchain-community
```

Every script guards missing packages and `ANTHROPIC_API_KEY` with a clear
message instead of a traceback. Keys go in a repo-root `.env` (see top-level
README). RAGAS's test-set API has shifted across versions — `02_testset_generation.py`
targets the §4.5 API and prints your installed version if the imports differ.

## Suggested flow

1. `02_testset_generation.py` — build a gold test set from your docs.
2. `01_ragas_eval.py` — understand the 4 metrics on worked examples.
3. `03_cicd_gate.py` — wire scoring into CI so regressions fail the build.
4. `05_improvement_loop.py` — track scores across iterations; diagnose with `FAILURE_MODES.md`.
5. `04_production_monitoring.py` — watch the live system after it ships.
