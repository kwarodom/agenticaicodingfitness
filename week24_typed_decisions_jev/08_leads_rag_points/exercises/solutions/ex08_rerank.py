#!/usr/bin/env python3
"""Solution · Exercise 08 — per-passage nouls in one call, selection in code."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevkit import noul  # noqa: E402

import ex08_rerank as ex  # noqa: E402


def passage_question(i: int) -> dict:
    return noul(f"Does `passages[{i}]` provide facts that help answer `query`, "
                f"rather than merely mentioning the topic?")


def select_top(scores: dict, k: int = 2, min_score: float = 0.5) -> list:
    kept = [(pid, s) for pid, s in scores.items() if s >= min_score]
    kept.sort(key=lambda kv: (-kv[1], kv[0]))          # score desc, then id asc
    return [pid for pid, _ in kept[:k]]


ex.passage_question = passage_question
ex.select_top = select_top

if __name__ == "__main__":
    ex.main()
