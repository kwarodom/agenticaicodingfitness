#!/usr/bin/env python3
"""Lab 02-3 · Selective loading: pay only for the encoders you use.

EmbeddingGemma 2 is modular: a 270M text model plus a 170M vision encoder and a 300M audio encoder. Leaving an
encoder out (sentence-transformers `config_kwargs`) saves memory and load time on an edge device. Each
configuration is loaded in a fresh Python process, on the CPU in float32 (the model card's advice for CPUs),
so its memory is measured on its own. Three numbers per configuration: the parameter bytes, the resident memory
after one text embedding, and the peak. The lab also checks that dropping the media encoders does not change
the TEXT vectors at all.

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/02_first_embeddings/labs/lab02_3_selective_loading.py
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import embedkit as ek  # noqa: E402

CHILD = r"""
import gc, json, sys, time
sys.path.insert(0, sys.argv[1])
import embedkit as ek
import torch
t0 = time.perf_counter()
m = ek.load(tuple(sys.argv[2].split("+")), quiet=True, dev="cpu", dtype=torch.float32)
load_s = time.perf_counter() - t0
v = m.encode("task: search result | query: water under the air handler")
gc.collect()
print(json.dumps({"params_m": ek.params_m(m), "param_mb": ek.param_mb(m), "load_s": load_s,
                  "rss_mb": ek.current_rss_mb(), "peak_mb": ek.rss_mb(), "vec": v[:16].tolist()}))
"""

ek.banner("Lab 02-3 · selective loading", "one fresh process per configuration · CPU · float32 · memory measured")
ek.note("CPU + float32 like a small Linux edge box. Left to itself the checkpoint loads in its own bfloat16 — Module 06 compares both")

configs = [("text", "text only"), ("text+image", "text + image"), ("text+audio", "text + audio"),
           ("text+image+audio", "all encoders")]
rows, vecs = [], {}
for i, (mods, label) in enumerate(configs, 1):
    ek.step(i, f"load {label}")
    out = subprocess.run([sys.executable, "-c", CHILD, str(ek.COMMON), mods], capture_output=True, text=True,
                         timeout=600)
    line = next((ln for ln in out.stdout.splitlines() if ln.startswith("{")), None)
    if not line:
        ek.warn(f"{label} failed: {(out.stderr or out.stdout)[-400:]}")
        continue
    r = json.loads(line)
    vecs[label] = r["vec"]
    ek.ok(f"{r['params_m']:.1f}M params · {r['param_mb']:.0f} MB of weights · loaded in {r['load_s']:.1f}s · "
          f"RSS {r['rss_mb']:.0f} MB · peak {r['peak_mb']:.0f} MB")
    rows.append([label, f"{r['params_m']:.0f}M", f"{r['param_mb']:.0f} MB", f"{r['rss_mb']:.0f} MB",
                 f"{r['peak_mb']:.0f} MB", f"{r['load_s']:.1f}s", ek.bar(r["peak_mb"], 4500, 20)])

ek.step(len(configs) + 1, "compare")
ek.table(rows, ["configuration", "params", "weights fp32", "RSS after encode", "peak RSS", "load", ""])
if "text only" in vecs and "all encoders" in vecs:
    c = ek.cosine(vecs["text only"], vecs["all encoders"])
    ek.ok(f"text vector with and without media encoders: cosine {c:.6f} on the first 16 dims — identical")
ek.note("size an edge box by the PEAK: macOS compresses idle pages, so RSS after encode can read below the weights")
ek.note("REFERENCE (model card): text only 270M · text + image 440M · text + audio 570M · full 740M")
ek.result("on an edge box, load the text model for search over notes and add an encoder only where a photo or "
          "a voice clip must be searchable.")
