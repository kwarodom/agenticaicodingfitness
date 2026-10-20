#!/usr/bin/env python3
"""Lab 03-2 · int8 and binary vectors: store each number in 1 byte, or in 1 bit.

Matryoshka shortens the vector; quantization shrinks each number. sentence-transformers' quantize_embeddings
turns float32 vectors into int8 (4× smaller) or packed binary (32× smaller). Combine both and a 768-d float
vector of 3,072 bytes becomes a 16-byte fingerprint. This lab measures recall@1 for every combination on the
SYNTHETIC hotel knowledge base, so you can pick the smallest index that still answers correctly.

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/03_matryoshka_and_quantization/labs/lab03_2_int8_and_binary.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import embedkit as ek  # noqa: E402

import numpy as np  # noqa: E402
from sentence_transformers.util.quantization import quantize_embeddings  # noqa: E402

ek.banner("Lab 03-2 · int8 and binary vectors", "float32 → int8 → 1 bit · combined with Matryoshka · measured here")
model = ek.load(("text",))
kb = ek.load_kb()
gold = np.array([x["gold"] for x in kb["queries"]])
Q = model.encode([x["q"] for x in kb["queries"]], prompt_name="SearchQuery")
D = model.encode([f"title: {d['title']} | text: {d['text']}" for d in kb["docs"]])


def r1_float(q, d):
    return float(np.mean(np.argmax(q @ d.T, axis=1) == gold))


def r1_int8(q, d):
    # calibrate the int8 ranges on the documents (what you would do when you build the index)
    qi = quantize_embeddings(q, precision="int8", calibration_embeddings=d).astype(np.int32)
    di = quantize_embeddings(d, precision="int8", calibration_embeddings=d).astype(np.int32)
    return float(np.mean(np.argmax(qi @ di.T, axis=1) == gold))


def r1_binary(q, d):
    qb = quantize_embeddings(q, precision="ubinary")
    db = quantize_embeddings(d, precision="ubinary")
    ham = np.unpackbits(qb[:, None, :] ^ db[None, :, :], axis=2).sum(axis=2)   # Hamming distance
    return float(np.mean(np.argmin(ham, axis=1) == gold))


ek.step(1, "every combination of dimension × precision")
rows = []
for dim in ek.DIMS:
    q, d = ek.renorm(Q[:, :dim]), ek.renorm(D[:, :dim])
    for prec, fn, bytes_per in (("float32", r1_float, dim * 4), ("int8", r1_int8, dim), ("binary", r1_binary, dim // 8)):
        r = fn(q, d)
        rows.append([dim, prec, f"{bytes_per} B", f"{3072 / bytes_per:.0f}×", f"{r:.3f}", ek.bar(r, 1.0, 16)])
ek.table(rows, ["dims", "precision", "bytes/vector", "smaller than 768 fp32", "recall@1", ""])

ek.step(2, "what that means for an edge box")
for dim, prec, b in ((768, "float32", 3072), (256, "int8", 256), (128, "binary", 16)):
    ek.note(f"1,000,000 notes/photos/clips at {dim}-d {prec}: {b * 1e6 / 2**20:,.0f} MiB")
ek.result("18 documents is a toy: ties break easily at 1 bit. The pattern to keep: search a tiny binary index "
          "first, then re-score the top 50 with full vectors (rescoring) — measure it on YOUR data.")
