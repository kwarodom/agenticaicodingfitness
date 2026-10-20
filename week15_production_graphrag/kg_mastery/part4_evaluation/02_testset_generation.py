#!/usr/bin/env python3
"""Part 4.5 — Automatic test-set generation with RAGAS (kg_mastery.pdf §4.5).

KEY INSIGHT: Hand-writing eval questions is slow and biased toward the cases you
already thought of. RAGAS can READ your source documents and synthesize a
diverse test set for you — each item carries a question, a reference answer, and
the contexts it was derived from. You then run YOUR GraphRAG pipeline over those
questions and score it (see 01_ragas_eval.py / 03_cicd_gate.py).

QUESTION DISTRIBUTIONS (the "evolution" types):
  - simple         single-fact lookups ("Who resolved alert ALT-305?")
  - reasoning      require inference across facts ("Why did room 204 use more
                   energy last week?")
  - multi_context  need MULTIPLE chunks stitched together — the sweet spot for
                   GraphRAG, where graph traversal beats flat retrieval.
We weight them 40 / 40 / 20 to bias toward the harder, graph-favoring cases.

TEST SET QUALITY: A good synthetic test set is (1) grounded — every question is
answerable from the docs, (2) diverse — mixes the evolution types above, and
(3) non-trivial — includes reasoning/multi-context items, not just lookups. Spot
check the generated CSV: drop leaking or unanswerable rows before using it as a
gold standard.

NOTE ON RAGAS VERSIONS: the testset API has changed across releases (the
`ragas.testset.evolutions` module and the `from_langchain` constructor were
reorganized in newer versions). This script targets the §4.5 API and falls back
to a clear message if your installed paths differ.

Requires:
  - ANTHROPIC_API_KEY in your environment / repo-root .env
  - pip install ragas datasets pandas langchain-anthropic langchain-community
  - source docs in ./hotel_docs/*.txt (two are shipped with this folder)

Run:  python week15/kg_mastery/part4_evaluation/02_testset_generation.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import LLM_MODEL

DOCS_DIR = Path(__file__).resolve().parent / "hotel_docs"
OUT_CSV = Path(__file__).resolve().parent / "hotel_kg_testset.csv"


def _imports():
    try:
        import ragas  # noqa: F401  (used only for the version hint below)
        from ragas.testset import TestsetGenerator
        from ragas.testset.evolutions import simple, reasoning, multi_context
        from langchain_anthropic import ChatAnthropic
        from langchain_community.document_loaders import DirectoryLoader
    except ImportError as e:
        # The import paths above match the §4.5 RAGAS API. Newer/older RAGAS
        # reorganized these modules — surface the installed version so the
        # student can adapt the import line.
        try:
            import ragas as _r

            ver = getattr(_r, "__version__", "unknown")
        except ImportError:
            ver = "not installed"
        print("⚠️  Could not import the RAGAS test-set generation API.")
        print(f"   ImportError: {e}")
        print(f"   Installed ragas version: {ver}")
        print("   This script targets the §4.5 API:")
        print("     from ragas.testset import TestsetGenerator")
        print("     from ragas.testset.evolutions import simple, reasoning, multi_context")
        print("   If your version differs, check the docs for that version's")
        print("   TestsetGenerator / synthesizer API, or:")
        print("   pip install ragas datasets pandas langchain-anthropic langchain-community")
        sys.exit(1)
    return (
        TestsetGenerator,
        simple,
        reasoning,
        multi_context,
        ChatAnthropic,
        DirectoryLoader,
    )


def main():
    (
        TestsetGenerator,
        simple,
        reasoning,
        multi_context,
        ChatAnthropic,
        DirectoryLoader,
    ) = _imports()

    if not DOCS_DIR.exists() or not any(DOCS_DIR.glob("*.txt")):
        print(f"⚠️  No source docs found in {DOCS_DIR}")
        print("   Add a few .txt files describing hotel incidents and re-run.")
        sys.exit(1)

    print(f"Loading documents from {DOCS_DIR} ...")
    loader = DirectoryLoader(str(DOCS_DIR), glob="*.txt")
    documents = loader.load()
    print(f"  loaded {len(documents)} document(s)")

    # Two roles: one LLM drafts the questions, one critiques/filters them.
    generator_llm = ChatAnthropic(model=LLM_MODEL, temperature=0)
    critic_llm = ChatAnthropic(model=LLM_MODEL, temperature=0)

    generator = TestsetGenerator.from_langchain(generator_llm, critic_llm)

    print("Generating synthetic test set (this calls the API)...")
    testset = generator.generate_with_langchain_docs(
        documents,
        test_size=10,
        distributions={simple: 0.4, reasoning: 0.4, multi_context: 0.2},
    )

    df = testset.to_pandas()
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} test cases to {OUT_CSV}")
    print("\n=== Test set (head) ===")
    print(df.head())


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set.")
        print("   export ANTHROPIC_API_KEY=sk-ant-...  (or add it to repo-root .env)")
        sys.exit(1)
    main()
