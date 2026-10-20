#!/usr/bin/env python3
"""Exercise 02 · Cosine, ranking and prefixes: the four functions behind every search.

Fill in the four TODOs, save, then run:
    week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/02_first_embeddings/exercises/ex02_cosine_and_rank.py

The checker is free and offline. Stuck? Compare with exercises/solutions/.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from embedkit import banner, check  # noqa: E402

PREFIX = {"search": "search result", "qa": "question answering", "fact": "fact checking", "code": "code retrieval"}

# ── TODO 1 ── cosine similarity of two vectors: dot product ÷ (length of a × length of b)
def cosine(a, b) -> float:
    return 0.0


# ── TODO 2 ── rank the rows of `docs` by cosine with `q`; return the best k as [(index, score), ...]
def rank(q, docs, k: int = 3) -> list:
    return []


# ── TODO 3 ── the model card's query prefix: "task: <task name> | query: <text>" (task names in PREFIX)
def query_text(text: str, task: str) -> str:
    return text


# ── TODO 4 ── the model card's document format: "title: <title> | text: <text>", "title: none" if no title
def doc_text(title, text: str) -> str:
    return text


# ─────────────────────────── checker — no need to edit below ────────────────
def main() -> None:
    banner("Exercise 02 · cosine, ranking and prefixes", "offline checker · numpy only · no model needed")
    good = True
    a, b, c = np.array([3.0, 4.0]), np.array([4.0, 3.0]), np.array([-3.0, -4.0])
    try:
        got = [round(cosine(a, b), 4), round(cosine(a, a), 4), round(cosine(a, c), 4)]
    except Exception as e:  # noqa: BLE001
        got = f"error: {e}"
    good &= check(got == [0.96, 1.0, -1.0], "cosine: (3,4)·(4,3) = 0.96 · self = 1 · opposite = −1",
                  f"TODO 1: cosine gave {got} — dot product divided by both lengths")
    q = np.array([1.0, 0.0])
    docs = np.array([[0.0, 1.0], [0.9, 0.1], [0.7, 0.7], [-1.0, 0.0]])
    try:
        r = rank(q, docs, k=3)
    except Exception as e:  # noqa: BLE001
        r = f"error: {e}"
    good &= check(isinstance(r, list) and [i for i, _ in r] == [1, 2, 0],
                  "rank: best three documents in order [1, 2, 0]",
                  f"TODO 2: rank returned {r} — list of (index, score), highest cosine first, k items")
    try:
        qs = query_text("water under the AHU", "search")
        qa = query_text("who resets the pool pump?", "qa")
    except Exception as e:  # noqa: BLE001
        qs = qa = f"error: {e}"
    good &= check(qs == "task: search result | query: water under the AHU"
                  and qa == "task: question answering | query: who resets the pool pump?",
                  "query_text: the model card's search and question-answering prefixes, exactly",
                  f"TODO 3: got {qs!r} / {qa!r}")
    try:
        d1, d2 = doc_text("Pool pump trip", "Reset it once."), doc_text(None, "Reset it once.")
    except Exception as e:  # noqa: BLE001
        d1 = d2 = f"error: {e}"
    good &= check(d1 == "title: Pool pump trip | text: Reset it once." and d2 == "title: none | text: Reset it once.",
                  "doc_text: 'title: … | text: …', and 'title: none' when there is no title",
                  f"TODO 4: got {d1!r} / {d2!r}")
    if not good:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)
    print("\n═ Done. Your rank() is what every retrieval lab this week does under the hood.")


if __name__ == "__main__":
    main()
