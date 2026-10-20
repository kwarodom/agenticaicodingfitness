#!/usr/bin/env python3
"""Error Handling and Retry Logic. (smart_hotel_mas.pdf §"Error Handling and Retry Logic")

Production agents talk to flaky services (Neo4j, ChromaDB, the Anthropic API).
Two patterns keep the MAS resilient:

  1. with_retry  — a decorator that retries a callable with exponential backoff.
  2. GracefulDegradation — when the richest memory tier (L4) is down, fall back
     through the hierarchy L4 -> L3 -> L2 -> L1 so the agent always returns
     *something* rather than crashing.

Fallback hierarchy:  L4 (graph) -> L3 (semantic) -> L2 (episodic) -> L1 (working)
Each tier is queried inside its own try/except so one failure cascades to the next.
"""
import functools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # patterns/ -> workshop root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "checkpoints"))


def with_retry(max_retries=3, backoff=1.0, exceptions=(Exception,)):
    """Retry the wrapped function up to ``max_retries`` times with exp. backoff.

    On the Nth failure it waits ``backoff * (2 ** attempt)`` seconds, prints a
    diagnostic, and tries again. After exhausting retries the last exception is
    re-raised.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:  # noqa: BLE001
                    last_exc = e
                    wait = backoff * (2 ** attempt)
                    print(f"Attempt {attempt + 1} failed ({type(e).__name__}: {e}). "
                          f"Retrying in {wait}s...")
                    if attempt < max_retries - 1:
                        time.sleep(wait)
            # exhausted all retries
            raise last_exc
        return wrapper
    return decorator


class GracefulDegradation:
    """Query memory with automatic fallback down the tier hierarchy.

    Order: L4 knowledge graph -> L3 semantic -> L2 episodic -> L1 working.
    The first tier that responds wins; the rest are skipped.
    """

    def query_with_fallback(self, query, kg, semantic, episodic, working) -> dict:
        # ── L4: Knowledge Graph (richest, structured) ────────────────────────
        try:
            if kg is not None:
                data = kg.cypher("MATCH (e:Event) RETURN e LIMIT 5")
                return {"source": "L4_knowledge_graph", "data": data}
        except Exception as e:  # noqa: BLE001
            print(f"L4 unavailable ({type(e).__name__}); falling back to L3.")

        # ── L3: Semantic Memory (vector similarity) ──────────────────────────
        try:
            if semantic is not None:
                data = semantic.recall_similar(query)
                return {"source": "L3_semantic", "data": data}
        except Exception as e:  # noqa: BLE001
            print(f"L3 unavailable ({type(e).__name__}); falling back to L2.")

        # ── L2: Episodic Memory (recent event log) ───────────────────────────
        try:
            if episodic is not None:
                data = episodic.recall_recent(hours=24, limit=5)
                return {"source": "L2_episodic", "data": data}
        except Exception as e:  # noqa: BLE001
            print(f"L2 unavailable ({type(e).__name__}); falling back to L1.")

        # ── L1: Working Memory (always in-process, last resort) ──────────────
        try:
            if working is not None:
                return {"source": "L1_working", "data": working.get_all()}
        except Exception as e:  # noqa: BLE001
            print(f"L1 unavailable ({type(e).__name__}).")

        return {"source": "none", "data": None}


if __name__ == "__main__":
    # ── Demo 1: with_retry on a flaky function (fails twice, then succeeds) ──
    print("── with_retry demo (small backoff so the demo is quick) ──")

    _calls = {"n": 0}

    @with_retry(max_retries=3, backoff=0.05, exceptions=(RuntimeError,))
    def flaky_service():
        _calls["n"] += 1
        if _calls["n"] < 3:
            raise RuntimeError(f"transient failure #{_calls['n']}")
        return "✅ success on attempt 3"

    result = flaky_service()
    print(f"Result: {result}\n")

    # ── Demo 2: GracefulDegradation with everything down except L1 ───────────
    print("── GracefulDegradation demo (L4/L3/L2 absent, L1 wins) ──")
    from checkpoint2_memory import WorkingMemory

    wm = WorkingMemory()
    wm.set("current_batch", ["R101", "R102"])
    wm.set("status", "processing")

    gd = GracefulDegradation()
    out = gd.query_with_fallback(
        "high temperature anomaly",
        kg=None, semantic=None, episodic=None, working=wm,
    )
    print(f"Served from: {out['source']}")
    print(f"Data: {out['data']}")

    # Fallback hierarchy: L4 -> L3 -> L2 -> L1 (rich/structured -> always-on).
