#!/usr/bin/env python3
"""Lab 03-1 · Matryoshka: keep the first 512, 256 or 128 numbers — and measure what you lose.

EmbeddingGemma 2 was trained with Matryoshka Representation Learning (MRL): the most important information
sits in the leading dimensions, so a 768-d vector can be cut to 512, 256 or 128 numbers. This lab measures
recall@1 and MRR on the SYNTHETIC hotel knowledge base at each size, the storage per vector and per million
vectors, and reproduces the model card's warning: a cut vector is no longer unit length — re-normalise it
(sentence-transformers does it for you with truncate_dim + normalize_embeddings; a manual slice does not).

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/03_matryoshka_and_quantization/labs/lab03_1_matryoshka.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import embedkit as ek  # noqa: E402

import numpy as np  # noqa: E402

ek.banner("Lab 03-1 · Matryoshka truncation", "768 → 512 → 256 → 128 · quality, storage, the re-normalise trap")
model = ek.load(("text",))
kb = ek.load_kb()
gold = [x["gold"] for x in kb["queries"]]
q768 = model.encode([x["q"] for x in kb["queries"]], prompt_name="SearchQuery")
d768 = model.encode([f"title: {d['title']} | text: {d['text']}" for d in kb["docs"]])

ek.step(1, "quality and size at each Matryoshka dimension (cut + re-normalise)")
rows = []
for dim in ek.DIMS:
    q, d = ek.renorm(q768[:, :dim]), ek.renorm(d768[:, :dim])
    r1, m = ek.recall_at_1(q, d, gold), ek.mrr(q, d, gold)
    b = dim * 4
    rows.append([dim, f"{r1:.3f}", f"{m:.3f}", f"{b} B", f"{b * 1e6 / 2**30:.2f} GiB", f"{768 / dim:.1f}×",
                 ek.bar(m, 1.0, 16)])
ek.table(rows, ["dims", "recall@1", "MRR", "bytes/vector (fp32)", "1M vectors", "smaller", "MRR"])
ek.note("18 easy questions saturate at 1.000 — the model card's large benchmark shows the real slope:")
ek.note("REFERENCE (model card, MTEB multilingual v2): 768d 61.36 · 512d 61.17 · 256d 60.41 · 128d 57.89")

ek.step(2, "the trap — a hand-cut vector is shorter than 1")
v = model.encode("task: search result | query: water under the air handler", truncate_dim=128,
                 normalize_embeddings=True)
ek.table([["full 768-d vector", f"{np.linalg.norm(q768[0]):.4f}"],
          ["encode(truncate_dim=128, normalize_embeddings=True)", f"{np.linalg.norm(v):.4f}"],
          ["manual slice [:128]", f"{np.linalg.norm(q768[0][:128]):.4f}"],
          ["slice + ek.renorm()", f"{np.linalg.norm(ek.renorm(q768[0][:128])):.4f}"]], ["vector", "L2 norm"])
lengths = np.linalg.norm(d768[:, :128], axis=1)
ek.note(f"the 18 document vectors cut to 128-d have lengths {lengths.min():.3f} … {lengths.max():.3f} — not 1")
ek.note("norms of 1.00x rather than exactly 1: the model runs in bfloat16 on the GPU (≈ 3 significant digits)")

ek.step(3, "does skipping re-normalisation change the ranking? (dot product on cut vectors)")
changed = 0
for dim in (256, 128):
    raw = q768[:, :dim] @ d768[:, :dim].T
    fixed = ek.renorm(q768[:, :dim]) @ ek.renorm(d768[:, :dim]).T
    diff = int(np.sum(np.argsort(-raw, axis=1)[:, :3] != np.argsort(-fixed, axis=1)[:, :3]))
    changed += diff
    ek.note(f"{dim}-d: {diff} of {raw.shape[0] * 3} top-3 slots differ · scores off by up to "
            f"{float(np.max(np.abs(raw - fixed))):.3f}")
(ek.warn if changed else ek.ok)("re-normalise after cutting; a dot-product index (built for unit vectors) shows no error, "
                                 "just wrong scores — the model card calls this 'degrades ranking quality silently'")

ek.step(4, "a 128-d query cannot be scored against a 768-d index")
try:
    _ = ek.renorm(q768[0][:128]) @ ek.renorm(d768).T
    ek.warn("numpy accepted it?!")
except ValueError as e:
    ek.ok(f"numpy refuses: {str(e)[:70]}…  — queries and documents must share one dimension")
ek.result("pick the dimension per index: 256-d keeps almost all the quality at a third of the storage; test 128-d "
          "on your own data before you trust it (the model card says multimodal suffers most).")
