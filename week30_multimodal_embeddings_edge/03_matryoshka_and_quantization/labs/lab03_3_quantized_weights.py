#!/usr/bin/env python3
"""Lab 03-3 · Quantized weights: the GGUF route, and an int8 model you can measure today.

Vectors were half the story; the MODEL can shrink too. EmbeddingGemma 2's text model ships as GGUF for
llama.cpp and Ollama (ggml-org: Q8_0 310 MB, BF16 558 MB; Unsloth UD-Q4_K_XL 176 MB) — TEXT ONLY, no image or
audio encoder. A new architecture needs a runtime that knows it, so step 1 asks your Ollama and llama.cpp
whether they can load `gemma-embedding2`, and says what to upgrade if not. Steps 2–3 then quantize the text
model's linear layers to int8 with PyTorch on the CPU and measure size, speed and agreement with float32.

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/03_matryoshka_and_quantization/labs/lab03_3_quantized_weights.py
"""
import io
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import embedkit as ek  # noqa: E402

import numpy as np  # noqa: E402

ek.banner("Lab 03-3 · quantized weights", "GGUF runtimes checked · int8 PyTorch model measured · CPU")

ek.step(1, "can your GGUF runtimes load gemma-embedding2? (text-only GGUF)")
gguf_vecs = None
if ek.ollama_up():
    have = ek.ollama_models()
    if ek.GGUF_TAG in have:
        try:
            gguf_vecs = np.array(ek.ollama_embed(ek.GGUF_TAG, ["task: search result | query: test"]))
            ek.ok(f"Ollama loads {ek.GGUF_TAG} — {gguf_vecs.shape[1]}-d vectors")
        except Exception as e:  # noqa: BLE001
            body = getattr(e, "read", lambda: b"")().decode(errors="replace")
            msg = body or str(e)
            if "unknown model architecture" in msg:
                ek.warn("Ollama has the GGUF but cannot run it: unknown model architecture 'gemma-embedding2' — "
                        "upgrade Ollama (support arrived after 0.34.x)")
            else:
                ek.warn(f"Ollama embed failed: {msg[:160]}")
    else:
        ek.note(f"not pulled yet — to try it:  ollama pull {ek.GGUF_TAG}   (310 MB)")
    ver = subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip() if shutil.which("ollama") else ""
    ek.note(f"Ollama here: {ver or 'version unknown'}")
else:
    ek.note("Ollama is not running — skipped")
if shutil.which("llama-embedding"):
    v = subprocess.run(["llama-embedding", "--version"], capture_output=True, text=True)
    ek.note("llama.cpp here: " + ((v.stdout + v.stderr).strip().splitlines() or ["?"])[0])
ek.note("REFERENCE (Unsloth guide): llama-server --model embeddinggemma-2-UD-Q4_K_XL.gguf --embeddings --pooling mean")

ek.step(2, "int8 dynamic quantization of the text model — PyTorch, CPU")
import torch  # noqa: E402

fp32 = ek.load(("text",), dev="cpu", dtype=torch.float32, quiet=True)


def state_mb(m) -> float:
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    return buf.tell() / 2**20


size_fp32 = state_mb(fp32)
from sentence_transformers import SentenceTransformer  # noqa: E402

# a second, independent copy to quantize (ek.load caches, so build this one directly)
int8 = SentenceTransformer(ek.MODEL_ID, device="cpu", model_kwargs={"dtype": torch.float32},
                           config_kwargs={"vision_config": None, "audio_config": None})
engines = torch.backends.quantized.supported_engines
torch.backends.quantized.engine = "qnnpack" if "qnnpack" in engines else engines[0]   # ARM: qnnpack · x86: x86/fbgemm
ek.note(f"quantized engine: {torch.backends.quantized.engine} (available: {', '.join(engines)})")
int8 = torch.ao.quantization.quantize_dynamic(int8, {torch.nn.Linear}, dtype=torch.qint8)
size_int8 = state_mb(int8)
ek.table([["float32", f"{size_fp32:.0f} MB"], ["int8 (Linear layers)", f"{size_int8:.0f} MB"]],
         ["weights", "serialized size"])
ek.note("embedding tables stay float32 here — that is why it is not a clean 4× (GGUF Q8_0 quantizes them too)")

ek.step(3, "speed and agreement on the hotel knowledge base")
kb = ek.load_kb()
queries = [x["q"] for x in kb["queries"]]
gold = [x["gold"] for x in kb["queries"]]
docs = [f"title: {d['title']} | text: {d['text']}" for d in kb["docs"]]
rows, vecs = [], {}
for name, m in (("float32", fp32), ("int8", int8)):
    m.encode(queries[:2], prompt_name="SearchQuery")                         # warm-up
    t0 = time.perf_counter()
    q = m.encode(queries, prompt_name="SearchQuery")
    d = m.encode(docs)
    secs = time.perf_counter() - t0
    vecs[name] = (q, d)
    rows.append([name, f"{secs:.2f}s", f"{1000 * secs / (len(queries) + len(docs)):.0f} ms",
                 f"{ek.recall_at_1(q, d, gold):.3f}"])
ek.table(rows, ["weights", "36 texts", "per text", "recall@1"])
agree = [ek.cosine(a, b) for a, b in zip(vecs["float32"][1], vecs["int8"][1])]
ek.ok(f"int8 vs float32 document vectors: mean cosine {np.mean(agree):.4f} (min {np.min(agree):.4f})")
ek.result("quantized weights trade a little agreement for memory and, on some CPUs, speed — measure all three "
          "on the target device. GGUF is the portable way to ship it to llama.cpp, Ollama and LM Studio.")
