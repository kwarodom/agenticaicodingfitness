#!/usr/bin/env python3
"""Lab 02-2 · Task prefixes: does "task: search result | query:" really help? Measure it.

EmbeddingGemma 2 was trained with short text instructions in front of every input. The model card says
omitting them "still works but reduces precision". This lab runs the same 18 hotel questions against the same
18 documents three ways — no prefixes, the model's built-in prompts, and prompts plus real titles — and
reports recall@1 and MRR for each. Then it prints the exact prompt strings sentence-transformers adds.

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/02_first_embeddings/labs/lab02_2_task_prefixes.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import embedkit as ek  # noqa: E402

ek.banner("Lab 02-2 · task prefixes", "same queries, same documents, three ways of asking · measured here")
model = ek.load(("text",))
kb = ek.load_kb()
queries = [x["q"] for x in kb["queries"]]
gold = [x["gold"] for x in kb["queries"]]

ek.step(1, "the prompt strings this checkpoint ships with (config_sentence_transformers.json)")
show = ("SearchQuery", "Document", "QuestionAnswering", "Classification", "Clustering", "SentenceSimilarity")
ek.table([[k, repr(model.prompts[k])] for k in show if k in model.prompts], ["prompt_name", "prepended text"])

ek.step(2, "three ways to embed the same data")
plain_docs = [d["text"] for d in kb["docs"]]
titled_docs = [f"title: {d['title']} | text: {d['text']}" for d in kb["docs"]]
runs = {
    "A · no prefixes at all": (model.encode(queries), model.encode(plain_docs)),
    "B · prompts (SearchQuery / Document)": (model.encode(queries, prompt_name="SearchQuery"),
                                              model.encode(plain_docs, prompt_name="Document")),
    "C · prompts + real titles": (model.encode(queries, prompt_name="SearchQuery"), model.encode(titled_docs)),
}
rows = []
for name, (q, d) in runs.items():
    r1, m = ek.recall_at_1(q, d, gold), ek.mrr(q, d, gold)
    rows.append([name, f"{r1:.3f}", f"{m:.3f}", ek.bar(m, 1.0, 20)])
ek.table(rows, ["how", "recall@1", "MRR", ""])

ek.step(3, "where they disagree")
qa, da = runs["A · no prefixes at all"]
qc, dc = runs["C · prompts + real titles"]
for i, q in enumerate(queries):
    a = ek.top_k(qa[i], da, 1)[0][0]
    c = ek.top_k(qc[i], dc, 1)[0][0]
    if a != c or a != gold[i]:
        mark = lambda j: "✓" if j == gold[i] else "✕"   # noqa: E731
        print(f"  {q[:52]:52s} A→{mark(a)} {kb['docs'][a]['title'][:26]:26s}  C→{mark(c)} {kb['docs'][c]['title'][:26]}")
ek.result("small set, so small differences — but the habit matters: queries get a task prefix, documents get "
          "'title: … | text: …'. The model card says so and your own eval should confirm it.")
