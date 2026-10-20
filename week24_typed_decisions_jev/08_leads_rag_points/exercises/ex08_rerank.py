#!/usr/bin/env python3
"""Exercise 08 · Re-rank passages with ONE call (speculative fan-out).

Instead of one call per passage, put ALL candidate passages in the state and ask
one noul per passage in the same request. Then code sorts and selects.

TODO 1 — passage_question(i): return a noul asking whether `passages[i]` helps
         answer `query`. Reference the passage with that exact backtick path.

TODO 2 — select_top(scores, k, min_score): scores = {"p1": 0.93, "p2": 0.10, …}
         Return up to k passage ids, highest score first, dropping any score
         below min_score. Break ties by id ("p1" before "p3").

Run: .venv/bin/python week24/08_leads_rag_points/exercises/ex08_rerank.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "common"))
from jevkit import ask, banner, check, guarded, noul, step, table, validate  # noqa: E402,F401

DATA = json.loads((HERE.parents[1] / "data" / "passages.json").read_text(encoding="utf-8"))


# ── TODO 1 ──
def passage_question(i: int) -> dict:
    return {}


# ── TODO 2 ──
def select_top(scores: dict, k: int = 2, min_score: float = 0.5) -> list:
    return []


# ─────────────────────────── checker — no need to edit below ────────────────
def main() -> None:
    banner("Exercise 08 · one-call re-rank")
    step(1, "offline checks — free")
    ok = True
    q = passage_question(3)
    ok &= check(isinstance(q, dict) and q.get("type") == "noul", "passage_question returns a noul",
                "TODO 1: passage_question(i) should return noul('…')")
    ins = str(q.get("instructions", ""))
    ok &= check("`passages[3]`" in ins and "`query`" in ins,
                "it points at `passages[3]` and `query`",
                "TODO 1: the instruction must contain `passages[i]` (with the real index) and `query` in backticks")
    tests = [
        (({"p1": .93, "p2": .10, "p3": .71}, 2, .5), ["p1", "p3"]),
        (({"p1": .6, "p2": .9, "p3": .6}, 3, .5), ["p2", "p1", "p3"]),
        (({"p1": .4, "p2": .3}, 2, .5), []),
        (({"p1": .99, "p2": .98, "p3": .97}, 1, .5), ["p1"]),
    ]
    for (scores, k, m), want in tests:
        got = select_top(scores, k, m)
        ok &= check(got == want, f"select_top({scores}, k={k}) → {got}",
                    f"TODO 2: select_top({scores}, k={k}, min={m}) should be {want}, got {got}")
    if not ok:
        print("\n⚠ fix the ✕ lines, save, run again — no API call was made.")
        sys.exit(1)

    step(2, "ONE live call: every passage in the state, one noul each")
    passages = DATA["passages"]
    state = {"query": DATA["query"], "passages": [p["text"] for p in passages]}
    questions = guarded({p["id"]: passage_question(i) for i, p in enumerate(passages)})
    r = ask(state, questions)
    a = validate(r, questions)
    scores = {pid: a[pid]["noul"] for pid in questions}
    table([[p["id"], p["text"][:56], f"{scores[p['id']]:.2f}"] for p in passages], ["id", "passage", "helps?"])
    print(f"◆ 1 call · {len(passages)} judgments · {r['usage']['input_tokens']} input tokens")

    step(3, "code selects the context")
    top = select_top(scores, k=2, min_score=0.5)
    for pid in top:
        p = next(x for x in passages if x["id"] == pid)
        print(f"   [{pid} · {p['source']}] {p['text']}")
    print("⚠ a re-ranker is not a safety filter: an injected passage can still look 'helpful'.")
    print("  Combine with lab 08-2's injection signal before feeding an answering model.")
    print("═ execute: False · document_access_expanded: False")


if __name__ == "__main__":
    main()
