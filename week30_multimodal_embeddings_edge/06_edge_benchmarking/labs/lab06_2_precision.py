#!/usr/bin/env python3
"""Lab 06-2 · Precision: float32, bfloat16 — and why the model card says never float16.

Halving the bytes per weight halves the memory traffic, which is what an edge device is short of. But the
model card warns: EmbeddingGemma 2's activations exceed float16's range, so float16 "returns NaN or silently
degraded embeddings rather than raising an error". This lab loads the text model three ways on the CPU and
tests it twice: on 36 short texts (questions and notes) and on ONE long document (the whole knowledge base
repeated, ≈ 3,500 tokens). Short inputs pass; the long one is where float16 breaks.

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/06_edge_benchmarking/labs/lab06_2_precision.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import embedkit as ek  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ek.banner("Lab 06-2 · precision", "float32 vs bfloat16 vs float16 · CPU · short texts and one long document")
kb = ek.load_kb()
queries = [x["q"] for x in kb["queries"]]
gold = [x["gold"] for x in kb["queries"]]
docs = [f"title: {d['title']} | text: {d['text']}" for d in kb["docs"]]
long_doc = "title: Hotel operations manual | text: " + " ".join(d["text"] for d in kb["docs"]) * 6

vecs, rows = {}, []
for i, dtype in enumerate((torch.float32, torch.bfloat16, torch.float16), 1):
    name = str(dtype).replace("torch.", "")
    ek.step(i, name)
    m = ek.load(("text",), dev="cpu", dtype=dtype)
    if i == 1:
        ek.note(f"long document: {len(m.tokenizer(long_doc)['input_ids']):,} tokens")
    t0 = time.perf_counter()
    q = m.encode(queries, prompt_name="SearchQuery", convert_to_numpy=True).astype(np.float32)
    d = m.encode(docs, convert_to_numpy=True).astype(np.float32)
    secs = time.perf_counter() - t0
    t1 = time.perf_counter()
    L = m.encode([long_doc], convert_to_numpy=True).astype(np.float32)[0]
    long_secs = time.perf_counter() - t1
    nans = int(np.isnan(q).any(axis=1).sum() + np.isnan(d).any(axis=1).sum())
    long_nan = bool(np.isnan(L).any())
    vecs[name] = (q, d, L)
    r1 = ek.recall_at_1(np.nan_to_num(q), np.nan_to_num(d), gold)
    ek.note(f"{ek.param_mb(m):.0f} MB of weights · 36 short texts in {secs:.2f}s, NaN {nans}/36, recall@1 {r1:.3f} · "
            f"long document in {long_secs:.1f}s, NaN: {'YES' if long_nan else 'no'}")
    rows.append([name, f"{ek.param_mb(m):.0f} MB", f"{nans}/36", f"{r1:.3f}", "NaN" if long_nan else "ok"])
    ek._MODELS.clear()

ek.step(4, "agreement with float32")
ref_q, ref_d, ref_L = vecs["float32"]
for name in ("bfloat16", "float16"):
    q, d, L = vecs[name]
    cos = [ek.cosine(a, b) for a, b in zip(np.vstack([ref_q, ref_d]), np.vstack([q, d]))]
    cos = [c for c in cos if not np.isnan(c)]
    lc = ek.cosine(ref_L, L)
    row = next(r for r in rows if r[0] == name)
    row += [f"{np.mean(cos):.4f}" if cos else "n/a", "NaN" if np.isnan(lc) else f"{lc:.4f}"]
next(r for r in rows if r[0] == "float32").extend(["1.0000", "1.0000"])
ek.table(rows, ["dtype", "weights", "short NaN", "recall@1", "long doc", "short cos vs fp32", "long cos vs fp32"])
f16 = next(r for r in rows if r[0] == "float16")
if f16[4] == "NaN" or f16[2] != "0/36":
    ek.warn("float16 returned a NaN vector — no exception, no warning. Exactly the failure the model card describes; "
            "a test set of short texts would never have shown it.")
else:
    ek.warn("no NaNs this time — keep the rule anyway: longer or media inputs push activations higher")
ek.result("bfloat16 keeps float32's exponent range at half the bytes — use it where the hardware has it; use "
          "float32 on CPUs without it; never float16 for this model.")
