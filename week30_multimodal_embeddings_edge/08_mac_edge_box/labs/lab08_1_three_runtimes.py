#!/usr/bin/env python3
"""Lab 08-1 · Three runtimes, one Mac: PyTorch on the GPU, MLX, and ONNX Runtime — measured side by side.

The same EmbeddingGemma 2 text model, five ways on this Mac: PyTorch + MPS (bfloat16), MLX bf16, MLX 4-bit
(mlx-community builds, Apple's own framework), and ONNX Runtime on the CPU in q4 and fp16 (onnx-community).
A sixth attempt asks ONNX Runtime's CoreML provider — the route to the Neural Engine — to run it.
Each runtime runs in its own fresh process (MLX in week30_multimodal_embeddings_edge/.venv-mlx), so load time and memory are its own.
Every runtime embeds the same 18 hotel questions and notes; agreement is measured against PyTorch float32.

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/08_mac_edge_box/labs/lab08_1_three_runtimes.py
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import embedkit as ek  # noqa: E402
import mackit  # noqa: E402

import numpy as np  # noqa: E402

CHILD = r"""
import json, resource, statistics, sys, time
sys.path.insert(0, sys.argv[1])
import embedkit as ek, mackit
kind = sys.argv[2]
kb = ek.load_kb()
Q = ["task: search result | query: " + x["q"] for x in kb["queries"]]
D = [f"title: {d['title']} | text: {d['text']}" for d in kb["docs"]]
one = [Q[0]]
batch = (D * 2)[:32]
info = {}
if kind == "pytorch-mps":
    import torch
    t0 = time.perf_counter(); m = ek.load(("text",), dev="mps", dtype=torch.bfloat16, quiet=True); load_s = time.perf_counter() - t0
    enc = lambda xs: (lambda v: (torch.mps.synchronize(), v)[1])(m.encode(xs, convert_to_numpy=True))
    disk = 1489.0
elif kind.startswith("mlx-"):
    e = mackit.MLXEmbedder(kind.split("-", 1)[1]); load_s, disk, enc = e.load_s, e.disk_mb, e.encode
elif kind.startswith("onnx-"):
    dtype = kind.split("-")[1]
    prov = ("CoreMLExecutionProvider", "CPUExecutionProvider") if kind.endswith("coreml") else ("CPUExecutionProvider",)
    e = mackit.ONNXEmbedder(dtype, prov); load_s, disk, enc = e.load_s, e.disk_mb, e.encode
    info["provider"] = e.provider
enc(one); enc(batch)                                    # warm-up
def p50(xs, n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); enc(xs); ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)
q, d = enc(Q), enc(D)
out = {"kind": kind, "load_s": load_s, "disk_mb": disk, "one_ms": p50(one, 15), "batch_ms": p50(batch, 5),
       "recall": ek.recall_at_1(q, d, [x["gold"] for x in kb["queries"]]),
       "peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, "docs": [list(map(float, v)) for v in d], **info}
if kind.startswith("mlx-"):
    out["mlx_peak_mb"] = e.peak_mb()
print("JSON" + json.dumps(out))
"""

RUNTIMES = [("pytorch-mps", "PyTorch + MPS · bf16", mackit.EMBED_PY), ("mlx-bf16", "MLX · bf16", mackit.MLX_PY),
            ("mlx-4bit", "MLX · 4-bit", mackit.MLX_PY), ("onnx-q4", "ONNX Runtime · q4 · CPU", mackit.EMBED_PY),
            ("onnx-fp16", "ONNX Runtime · fp16 · CPU", mackit.EMBED_PY),
            ("onnx-q4-coreml", "ONNX Runtime · q4 · CoreML", mackit.EMBED_PY)]

ek.banner("Lab 08-1 · three runtimes, one Mac", "PyTorch MPS · MLX · ONNX Runtime · one fresh process each")
ek.note(f"{ek.machine()['cpu']} · {ek.machine()['ram_gb']} GB unified memory · power: {mackit.power_source()}")
results = {}
for i, (kind, label, py) in enumerate(RUNTIMES, 1):
    ek.step(i, label)
    if not Path(py).exists():
        ek.warn(f"{py} missing — skipped (see week30_multimodal_embeddings_edge/README.md)")
        continue
    p = subprocess.run([str(py), "-c", CHILD, str(ek.COMMON), kind], capture_output=True, text=True, timeout=900)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("JSON")), None)
    if not line:
        text = p.stderr + p.stdout
        import re
        cover = re.search(r"number of nodes in the graph: (\d+) number of nodes supported by CoreML: (\d+)", text)
        why = re.search(r"Status Message: \S+ operator\(\) (.+?not supported by the CoreML EP\.)", text)
        if cover:
            n, s = int(cover.group(1)), int(cover.group(2))
            ek.note(f"CoreML could take {s} of {n} graph nodes ({100 * s / n:.0f} %) — the rest would stay on the CPU")
        if why:
            ek.warn(f"{label} did not run: {why.group(1)}")
        else:
            err = [ln for ln in text.splitlines() if "Error" in ln or "error" in ln]
            ek.warn(f"{label} did not run: {(err[-1] if err else 'no output')[:200]}")
        continue
    r = json.loads(line[4:])
    results[kind] = r
    ek.ok(f"loaded in {r['load_s']:.1f}s · 1 query {r['one_ms']:.1f} ms · 32 notes {r['batch_ms']:.0f} ms · "
          f"recall@1 {r['recall']:.3f}" + (f" · ran on {r['provider']}" if "provider" in r else ""))

ek.step(len(RUNTIMES) + 1, "agreement with PyTorch float32 on the CPU (the reference)")
import torch  # noqa: E402

ref = ek.load(("text",), dev="cpu", dtype=torch.float32, quiet=True)
kb = ek.load_kb()
R = ref.encode([f"title: {d['title']} | text: {d['text']}" for d in kb["docs"]])
rows = []
labels = dict((k, l) for k, l, _ in RUNTIMES)
for kind, r in results.items():
    cos = [ek.cosine(a, b) for a, b in zip(R, np.array(r["docs"]))]
    peak = r.get("mlx_peak_mb")
    rows.append([labels[kind], f"{r['disk_mb']:.0f} MB", f"{r['load_s']:.1f}s", f"{r['one_ms']:.1f} ms",
                 f"{r['batch_ms'] / 32:.1f} ms", f"{r['recall']:.3f}", f"{min(cos):.4f}",
                 f"{r['peak_mb']:.0f} MB" + (f" (MLX {peak:.0f})" if peak else "")])
ek.table(rows, ["runtime", "weights on disk", "load", "1 query", "per note in ×32", "recall@1", "min cos vs fp32",
                "peak memory"])
(ek.runs(__file__) / "runtimes.json").write_text(json.dumps(
    {k: {x: v for x, v in r.items() if x != "docs"} for k, r in results.items()}, indent=2))
ek.note("weights on disk: PyTorch's checkpoint holds every encoder; MLX holds every encoder too; ONNX here is the "
        "text model only")
ek.result("one Mac, five working runtimes. Pick by job: MLX for Apple-native speed and all modalities, ONNX for a "
          "small portable text model, PyTorch for the reference. The Neural Engine is not a free lunch — check it.")
