#!/usr/bin/env python3
"""Lab 02-1 · First vectors: turn sentences into 768 numbers and rank them by meaning.

Loads EmbeddingGemma 2 text-only (270M parameters). The first run downloads the 1.5 GB checkpoint from
Hugging Face once; after that every lab runs offline from the cache. You will see what an embedding looks
like, check that it has length 1, reproduce the model card's quick start, and run your first semantic search
over the SYNTHETIC hotel knowledge base.

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/02_first_embeddings/labs/lab02_1_first_vectors.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import embedkit as ek  # noqa: E402

import numpy as np  # noqa: E402

ek.banner("Lab 02-1 · first vectors", "EmbeddingGemma 2 text-only · on this machine · offline after the first download")

ek.step(1, "load the text-only model (vision and audio encoders left out)")
with ek.timer() as t:
    model = ek.load(("text",))
ek.note(f"load took {t.s:.1f}s · {ek.params_m(model):.1f}M parameters · device {model.device}")

ek.step(2, "one sentence → one vector")
v = model.encode("The weather is lovely today.")
ek.table([["shape", str(v.shape)], ["dtype", str(v.dtype)], ["first 6 values", np.array2string(v[:6], precision=4)],
          ["L2 norm", f"{np.linalg.norm(v):.4f}"]], ["property", "value"])
ek.ok("768 numbers, length ≈ 1 — the model's last module normalises every vector")

ek.step(3, "the model card's quick start — similarity is a dot product of unit vectors")
sentences = ["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."]
emb = model.encode(sentences)
sim = model.similarity(emb, emb).numpy()
ek.table([[s[:30]] + [f"{x:.3f}" for x in row] for s, row in zip(sentences, sim)], ["", "lovely", "sunny", "stadium"])
ek.ok(f"'lovely' vs 'sunny' = {sim[0, 1]:.3f} > 'lovely' vs 'stadium' = {sim[0, 2]:.3f}")

ek.step(4, "first semantic search over the hotel knowledge base (SYNTHETIC)")
kb = ek.load_kb()
docs = [f"title: {d['title']} | text: {d['text']}" for d in kb["docs"]]
with ek.timer() as t:
    doc_emb = model.encode(docs)
ek.note(f"embedded {len(docs)} documents in {t.s:.2f}s")
for q in ("water on the floor under the air handler", "a guest wants more towels", "someone is stuck in the lift"):
    q_emb = model.encode(q, prompt_name="SearchQuery")
    hits = ek.top_k(q_emb, doc_emb, k=2)
    print(f"  {q!r}")
    for rank, (i, s) in enumerate(hits, 1):
        print(f"     {rank}. {s:.3f}  {kb['docs'][i]['title']}")
ek.result("no keyword overlap needed: 'lift' finds 'Lift entrapment', 'air handler' finds 'AHU condensate leak'.")
