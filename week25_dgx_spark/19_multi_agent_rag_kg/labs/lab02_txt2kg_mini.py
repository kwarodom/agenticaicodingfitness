#!/usr/bin/env python3
"""Lab 19-2 · txt2kg in miniature: extract triples with a local LLM, merge them into a graph.

The txt2kg playbook's pipeline, cut down to what fits in one screen of Python:

  chunk text → LLM extracts (subject, predicate, object) triples as JSON → parse
  (with txt2kg's fallback parser) → normalise + de-duplicate (txt2kg's mergeTriples)
  → graph → which entities link documents? what is 2 hops from X?

The input is real: the "Basic idea" section of three playbooks in this module
(multi-agent-chatbot, txt2kg, rag-ai-workbench), read from the clone at the repo root.
The system prompt is copied word for word from txt2kg's
frontend/app/api/ollama/route.ts. Three LLM calls, ≤ 300 tokens each, to the Spark's
Ollama, or to THIS laptop's Ollama as a labelled LAPTOP STAND-IN.

Writes .runs/triples.json and .runs/triples.cypher (paste into Neo4j Browser on the
txt2kg --neo4j stack, http://localhost:7474).

Run: .venv/bin/python week25/19_multi_agent_rag_kg/labs/lab02_txt2kg_mini.py [--laptop-model gemma4:12b]
"""
import argparse
import json
import re
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import ROOT, banner, chat_any, note, pick_laptop_model, result, step, table, warn  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
RUNS = HERE / ".runs"
PB = ROOT / "dgx-spark-playbooks" / "nvidia"
DOCS = ["playbook-multi-agent-chatbot", "playbook-txt2kg", "playbook-rag-ai-workbench"]
SPARK_MODEL = "llama3.1:8b"          # txt2kg's default for its Ollama stacks (playbook Step 3)

# Verbatim from dgx-spark-playbooks/nvidia/playbook-txt2kg/assets/frontend/app/api/ollama/route.ts
SYSTEM = """You are a knowledge graph builder that extracts structured information from text.
Extract subject-predicate-object triples from the following text.

Guidelines:
- Extract only factual triples present in the text
- Normalize entity names to their canonical form
- Return results in JSON format as an array of objects with "subject", "predicate", "object" fields
- Each triple should represent a clear relationship between two entities
- Focus on the most important relationships in the text"""


def basic_idea(folder: str) -> str:
    """The '## Basic idea' section of a playbook README, as plain prose."""
    text = (PB / folder / "README.md").read_text(encoding="utf-8")
    m = re.search(r"^## Basic idea\s*\n(.*?)(?=^## )", text, re.S | re.M)
    body = m.group(1) if m else ""
    body = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", body)          # [link](url) → link
    return re.sub(r"\*\*|`", "", body).strip()


def parse_triples(reply: str) -> list[dict]:
    """JSON objects first; else txt2kg's line fallback 'a - b - c' / 'a | b | c'. Works on truncated replies."""
    out = []
    for m in re.finditer(r"\{[^{}]*\}", reply):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if all(isinstance(d.get(k), str) and d.get(k).strip() for k in ("subject", "predicate", "object")):
            out.append({k: d[k] for k in ("subject", "predicate", "object")})
    if not out:
        for line in reply.splitlines():
            m = re.match(r"^[\s\-\*\d\.]*(.+?)\s*[\-\|]\s*(.+?)\s*[\-\|]\s*(.+)$", line)
            if m:
                out.append({"subject": m.group(1), "predicate": m.group(2), "object": m.group(3)})
    return out


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def bfs_path(adj: dict, a: str, b: str) -> list[str] | None:
    prev, q = {a: None}, deque([a])
    while q:
        cur = q.popleft()
        if cur == b:
            path = []
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return path[::-1]
        for nxt in adj.get(cur, {}):
            if nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    return None


def cypher_quote(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laptop-model", default="", help="laptop stand-in model (default: gemma4:12b if pulled)")
    args = ap.parse_args()

    banner("Lab 19-2 · txt2kg in miniature — triples from three playbooks, merged into one graph",
           "txt2kg's own prompt · JSON + fallback parser · normalise · de-duplicate · graph queries")
    if not PB.is_dir():
        warn(f"playbooks not found at {PB}.")
        print("$ git clone https://github.com/NVIDIA/dgx-spark-playbooks   # run at the repo root (gitignored)")
        result("Clone the playbooks, then rerun this lab.")
        return

    step(1, "the input: one chunk per playbook ('## Basic idea')")
    chunks = [(d.replace("playbook-", ""), basic_idea(d)) for d in DOCS]
    for name, text in chunks:
        print(f"│ {name:20s} {len(text):4d} chars · {text[:80]!r}…")

    step(2, "extract triples — one LLM call per chunk, txt2kg's system prompt")
    laptop_model = args.laptop_model or pick_laptop_model(["gemma4:12b", "nemotron-3-nano"])
    raw, sources = [], set()
    for name, text in chunks:
        r = chat_any("ollama", SPARK_MODEL, [{"role": "system", "content": SYSTEM},
                                             {"role": "user", "content": f"Extract triples from this text:\n\n{text}"}],
                     max_tokens=300, laptop_model=laptop_model)
        sources.add(r["source"])
        got = parse_triples(r.get("text") or "")
        closed = (r.get("text") or "").strip().strip("`").strip().endswith("]")
        cut = "" if closed or r["source"] == "reference" else " · reply cut at 300 tokens, kept the complete objects"
        print(f"◆ {name:20s} {len(got):2d} triples · {r['out_tokens']} tok in {r['total_ms'] / 1000:.1f}s "
              f"({r.get('model')}){cut}")
        raw += [{**t, "doc": name} for t in got]
    if not raw and sources == {"reference"}:
        warn("DRY: no Ollama endpoint (Spark or laptop) answered, so nothing was extracted.")
        result("Start Ollama on the Spark (txt2kg's ollama-compose, or Module 03) or on this laptop, then rerun.")
        return
    if not raw:
        warn("no triples parsed — the model's reply was not JSON. Try --laptop-model gemma4:12b.")
        result("Nothing to merge.")
        return

    step(3, "normalise + de-duplicate (txt2kg: lowercase, then mergeTriples on subject|predicate|object)")
    uniq, docs_of = {}, {}
    for t in raw:
        key = (norm(t["subject"]), norm(t["predicate"]), norm(t["object"]))
        uniq.setdefault(key, t["doc"])
        for ent in (key[0], key[2]):
            docs_of.setdefault(ent, set()).add(t["doc"])
    triples = list(uniq)
    print(f"◆ {len(raw)} raw triples → {len(triples)} unique · {len(docs_of)} entities")
    table([[s[:30], p[:26], o[:34], uniq[(s, p, o)]] for s, p, o in triples[:18]],
          ["subject", "predicate", "object", "from"])
    if len(triples) > 18:
        print(f"│ … {len(triples) - 18} more in .runs/triples.json")

    step(4, "the graph: hubs, entities that link documents, and a path")
    adj = {}
    for s, p, o in triples:
        adj.setdefault(s, {})[o] = p
        adj.setdefault(o, {})[s] = p
    hubs = sorted(adj, key=lambda e: -len(adj[e]))[:5]
    table([[h, len(adj[h]), ", ".join(sorted(docs_of[h]))] for h in hubs], ["entity", "links", "mentioned in"])
    bridges = [e for e in adj if len(docs_of[e]) > 1]
    print(f"◆ entities that appear in more than one playbook: {', '.join(bridges) or 'none this run'}")
    seen, islands = set(), 0
    for e in adj:
        if e not in seen:
            islands += 1
            seen |= {x for x in adj if bfs_path(adj, e, x)}
    print(f"◆ connected components: {islands} (1 would mean every fact is reachable from every other)")
    far = next(((a, b) for a in hubs for b in hubs if a < b and b not in adj[a] and bfs_path(adj, a, b)), None)
    if far:
        print(f"→ path {far[0]!r} → {far[1]!r}: " + " → ".join(bfs_path(adj, *far)))
    else:
        print("→ no multi-hop path between the top hubs: the extracted facts do not share entity names")
    top = hubs[0]
    ctx = [f"({top}) -[{p}]- ({n})" for n, p in adj[top].items()]
    print(f"· graph context for a question about {top!r} (what txt2kg's graph query adds to the prompt):")
    for line in ctx[:8]:
        print(f"│   {line}")

    RUNS.mkdir(exist_ok=True)
    (RUNS / "triples.json").write_text(json.dumps(
        [{"subject": s, "predicate": p, "object": o, "doc": uniq[(s, p, o)]} for s, p, o in triples], indent=1))
    (RUNS / "triples.cypher").write_text("".join(
        f"MERGE (a:Entity {{name: {cypher_quote(s)}}}) MERGE (b:Entity {{name: {cypher_quote(o)}}}) "
        f"MERGE (a)-[:RELATION {{type: {cypher_quote(p)}}}]->(b);\n" for s, p, o in triples))
    print("→ wrote week25/19_multi_agent_rag_kg/.runs/triples.json and triples.cypher")
    label = "LIVE on the Spark" if sources == {"spark"} else \
        "LAPTOP STAND-IN — the triples are real model output, not txt2kg on a Spark"
    note(label + ". Extraction is sampled: rerun and the triple count and wording change.")
    result("A knowledge graph is only as clean as its entity names: 'rag' and 'retrieval-augmented generation' "
           "are two nodes until you normalise them. That is why txt2kg asks for canonical names.")


if __name__ == "__main__":
    main()
