#!/usr/bin/env python3
"""Exercise 19 · reference solution — build the graph behind GraphRAG: normalise, de-duplicate, traverse.

Fill in the three TODOs, save, then run:
    .venv/bin/python week25/19_multi_agent_rag_kg/exercises/ex19_graph_context.py

The checker is free and offline. Its triples describe the multi-agent chatbot as the
playbook's docker-compose files and backend code wire it (written by hand for this
exercise, with the messy casing and duplicates an LLM extractor produces). Stuck?
Compare with exercises/solutions/.
"""
import re
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from sparkkit import banner, check  # noqa: E402

TRIPLES = [
    ("Frontend", "sends chats to", "Backend"),
    ("backend", "runs", "Supervisor Agent"),
    ("Supervisor Agent", "is powered by", "gpt-oss-120b"),
    ("supervisor agent", "calls tool", "write_code"),
    ("write_code", "uses", "deepseek-coder"),
    ("Supervisor  Agent", "calls tool", "search_documents"),
    ("search_documents", "retrieves from", "Milvus"),
    ("Milvus", "stores vectors from", "qwen3-embedding"),
    ("Supervisor Agent", "calls tool", "explain_image"),
    ("explain_image", "uses", "qwen2.5-vl"),
    ("Backend", "stores chats in", "Postgres"),
    (" backend ", "Stores Chats In", "postgres"),
    ("FRONTEND", "sends chats to", "backend"),
]


# ── TODO 1 ── normalise one triple: every part lowercased, stripped, and inner runs of whitespace
#   collapsed to one space ("Supervisor  Agent" → "supervisor agent"). Return a 3-tuple.
def normalise(triple: tuple) -> tuple:
    return tuple(re.sub(r"\s+", " ", part).strip().lower() for part in triple)


# ── TODO 2 ── normalise every triple, drop duplicates (keep first-seen order), and return
#   (unique_triples, graph) where graph maps each entity → set of neighbour entities, BOTH directions.
def build_graph(triples: list) -> tuple:
    uniq = list(dict.fromkeys(normalise(t) for t in triples))
    graph = {}
    for s, _, o in uniq:
        graph.setdefault(s, set()).add(o)
        graph.setdefault(o, set()).add(s)
    return uniq, graph


# ── TODO 3 ── shortest path from entity a to entity b as a list [a, …, b] (breadth-first search).
#   Return None if either entity is missing or no path exists; return [a] when a == b.
def path(graph: dict, a: str, b: str):
    if a not in graph or b not in graph:
        return None
    prev, queue = {a: None}, deque([a])
    while queue:
        cur = queue.popleft()
        if cur == b:
            out = []
            while cur is not None:
                out.append(cur)
                cur = prev[cur]
            return out[::-1]
        for nxt in sorted(graph[cur]):
            if nxt not in prev:
                prev[nxt] = cur
                queue.append(nxt)
    return None


# ─────────────────────────── checker — no need to edit below ────────────────
def main() -> None:
    banner("Exercise 19 · the graph behind GraphRAG", "offline checker · free · no Spark needed", status=False)
    ok = True
    ok &= check(normalise(("Supervisor  Agent", " Calls Tool ", "WRITE_CODE")) == ("supervisor agent", "calls tool", "write_code"),
                "normalise: lowercase · strip · collapse inner spaces",
                f"TODO 1: got {normalise(('Supervisor  Agent', ' Calls Tool ', 'WRITE_CODE'))!r}")
    uniq, g = build_graph(TRIPLES)
    ok &= check(len(uniq) == 11 and uniq[0] == ("frontend", "sends chats to", "backend")
                and g.get("backend") == {"frontend", "supervisor agent", "postgres"},
                "build_graph: 13 raw → 11 unique · backend links frontend, supervisor agent, postgres",
                f"TODO 2: expected 11 unique triples (got {len(uniq)}) and backend's neighbours "
                f"{{frontend, supervisor agent, postgres}} (got {g.get('backend')!r})")
    p1 = path(g, "frontend", "milvus") if g else None
    ok &= check(p1 == ["frontend", "backend", "supervisor agent", "search_documents", "milvus"]
                and path(g, "deepseek-coder", "qwen3-embedding") is not None
                and len(path(g, "deepseek-coder", "qwen3-embedding") or []) == 6
                and path(g, "frontend", "redis") is None and path(g, "milvus", "milvus") == ["milvus"],
                "path: frontend → milvus in 4 hops · deepseek-coder → qwen3-embedding in 5 · unknown → None",
                f"TODO 3: path(frontend, milvus) should be 5 entities long (got {p1!r})")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)

    print("\n▣ your graph, applied: the context GraphRAG adds for 'How does a chat reach the vector database?'")
    hops = path(g, "frontend", "milvus")
    rel = {(s, o): p for s, p, o in uniq}
    for a, b in zip(hops, hops[1:]):
        print(f"│ ({a}) -[{rel.get((a, b)) or rel.get((b, a))}]-> ({b})")
    print("\n═ Plain vector search would retrieve chunks about Milvus OR about the frontend; the graph gives the "
          "chain that connects them. Compare with lab 02, whose LLM-extracted graph fell apart into islands.")


if __name__ == "__main__":
    main()
