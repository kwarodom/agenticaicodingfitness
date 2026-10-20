# ▶ Embed Lab 03 — Smaller vectors, smaller models: Matryoshka, int8 and GGUF

> Part of Week 30 · Multimodal embeddings with EmbeddingGemma 2 on local, sovereign and edge hardware. Every lab runs on your own machine. Every number in this course was measured on the machine that ran it, or it is quoted with its source.

**What you'll actually do**
- Cut 768-number vectors down to 512, 256 or 128 numbers, and measure what you lose.
- Reproduce the model card's quiet trap: a cut vector is no longer length 1.
- Store each number in 1 byte (int8) or 1 bit (binary), and measure recall at every size.
- Shrink the **model** itself: the GGUF route for llama.cpp and Ollama, and an int8 PyTorch model you can measure today.

**Time** ~45 min · **Difficulty** intermediate · **Hardware** any laptop · the model from Lab 02

**Sources:** [model card — Matryoshka, truncation table](https://huggingface.co/google/embeddinggemma-2) · [ggml-org GGUF](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF) · [Unsloth run-locally guide](https://unsloth.ai/docs/models/embeddinggemma-2) · [sentence-transformers quantization](https://sbert.net) · [`week30_multimodal_embeddings_edge/RESEARCH.md`](../RESEARCH.md) §1

## 0 · Before you start

You need the model cached from Lab 02. One part of this module is optional: Ollama on `localhost:11434` (`ollama --version`). Lab 03-3 asks it whether it can run the text-only GGUF.

```bash
# on: laptop
ollama --version
```

✓ Checkpoint: you know which Ollama version you have (Lab 03-3 tells you whether it is new enough).

## 1 · Matryoshka: the important numbers come first

EmbeddingGemma 2 was trained with **Matryoshka Representation Learning**: like nesting dolls, the first 512, 256 or 128 numbers of a vector are themselves a usable, smaller vector. You cut the end off and keep going. The model card measures the cost on its big benchmark:

| dims | storage | MTEB multilingual (model card) |
|---|---|---|
| 768 | 1× | 61.36 |
| 512 | 1.5× smaller | 61.17 |
| 256 | 3× smaller | 60.41 |
| 128 | 6× smaller | 57.89 |

The quality is nearly the same down to 256. At 128 it drops, and the model card warns that multimodal quality drops most.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/03_matryoshka_and_quantization/labs/lab03_1_matryoshka.py
```

**Expected output** (captured on this Mac)

```
▣ STEP 1 · quality and size at each Matryoshka dimension (cut + re-normalise)
│ dims  recall@1  MRR    bytes/vector (fp32)  1M vectors  smaller  MRR
│ ────  ────────  ─────  ───────────────────  ──────────  ───────  ────────────────
│ 768   1.000     1.000  3072 B               2.86 GiB    1.0×     ████████████████
│ 512   1.000     1.000  2048 B               1.91 GiB    1.5×     ████████████████
│ 256   1.000     1.000  1024 B               0.95 GiB    3.0×     ████████████████
│ 128   1.000     1.000  512 B                0.48 GiB    6.0×     ████████████████
◆ 18 easy questions saturate at 1.000 — the model card's large benchmark shows the real slope:
◆ REFERENCE (model card, MTEB multilingual v2): 768d 61.36 · 512d 61.17 · 256d 60.41 · 128d 57.89
```

On 18 easy questions every size scores perfectly, so a small test cannot show the slope. That is a lesson in itself: **test on data as hard as your real queries** before you pick 128.

✓ Checkpoint: you can say how much storage 1 million vectors take at 768 and at 256 dimensions (2.86 GiB vs 0.95 GiB in float32).

## 2 · The trap: re-normalise after cutting

A unit vector cut in half is not a unit vector. Its length shrinks, and every dot product shrinks with it, by a different amount for each vector. sentence-transformers handles this for you: `encode(…, truncate_dim=128, normalize_embeddings=True)` cuts and then re-normalises. A hand-written `v[:128]` does not.

**Expected output** (captured on this Mac — Lab 03-1, steps 2–3)

```
▣ STEP 2 · the trap — a hand-cut vector is shorter than 1
│ vector                                               L2 norm
│ ───────────────────────────────────────────────────  ───────
│ full 768-d vector                                    1.0021
│ encode(truncate_dim=128, normalize_embeddings=True)  1.0013
│ manual slice [:128]                                  0.5985
│ slice + ek.renorm()                                  1.0000
◆ the 18 document vectors cut to 128-d have lengths 0.566 … 0.610 — not 1

▣ STEP 3 · does skipping re-normalisation change the ranking? (dot product on cut vectors)
◆ 256-d: 4 of 54 top-3 slots differ · scores off by up to 0.477
◆ 128-d: 27 of 54 top-3 slots differ · scores off by up to 0.626
```

At 128 dimensions, half of the top-3 results changed order. No error appeared. The model card calls this exactly that: it "degrades ranking quality silently". Two rules follow. Re-normalise every cut vector. And keep queries and documents at the same dimension, because a 128-d query cannot be scored against a 768-d index at all.

✓ Checkpoint: you can explain why a dot-product index gives wrong rankings, without any error, if you skip re-normalisation.

## 3 · int8 and binary: fewer bits per number

Matryoshka shortens the vector. **Quantization** shrinks each number in it: int8 uses 1 byte instead of 4, and binary keeps only the sign, 1 bit. Binary vectors are compared by **Hamming distance**, the count of bits that differ, which is very fast on any CPU.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/03_matryoshka_and_quantization/labs/lab03_2_int8_and_binary.py
```

**Expected output** (captured on this Mac)

```
│ dims  precision  bytes/vector  smaller than 768 fp32  recall@1
│ ────  ─────────  ────────────  ─────────────────────  ────────  ────────────────
│ 768   float32    3072 B        1×                     1.000     ████████████████
│ 768   int8       768 B         4×                     1.000     ████████████████
│ 768   binary     96 B          32×                    1.000     ████████████████
│ 512   float32    2048 B        2×                     1.000     ████████████████
│ 512   int8       512 B         6×                     1.000     ████████████████
│ 512   binary     64 B          48×                    1.000     ████████████████
│ 256   float32    1024 B        3×                     1.000     ████████████████
│ 256   int8       256 B         12×                    1.000     ████████████████
│ 256   binary     32 B          96×                    1.000     ████████████████
│ 128   float32    512 B         6×                     1.000     ████████████████
│ 128   int8       128 B         24×                    1.000     ████████████████
│ 128   binary     16 B          192×                   0.667     ███████████░░░░░

▣ STEP 2 · what that means for an edge box
◆ 1,000,000 notes/photos/clips at 768-d float32: 2,930 MiB
◆ 1,000,000 notes/photos/clips at 256-d int8: 244 MiB
◆ 1,000,000 notes/photos/clips at 128-d binary: 15 MiB
```

Only the most extreme combination broke: 128 dimensions × 1 bit = a 16-byte fingerprint, which got a third of the questions wrong. 256-d int8 kept full recall at 12× smaller. A million items in 244 MiB fit comfortably on a Raspberry Pi. The production pattern is **rescoring**: search the tiny binary index first, then re-score the top 50 hits with the full vectors.

✓ Checkpoint: you can pick an index format for a gateway with 512 MB free and 1 million items, and justify it.

## 4 · Smaller models: GGUF and int8 weights

The vectors were half the story; the **model** can shrink too. The text model ships as **GGUF**, the format of llama.cpp, Ollama and LM Studio:

| file | size | source |
|---|---|---|
| `embeddinggemma-2-UD-Q4_K_XL.gguf` | 176 MB | Unsloth |
| `embeddinggemma-2-GGUF:Q8_0` | 310 MB | ggml-org |
| `embeddinggemma-2-GGUF:BF16` | 558 MB | ggml-org |

**These GGUFs are text-only.** They have no image or audio encoder; for photos and voice you need the full model (Modules 04–07). A brand-new architecture also needs a runtime that knows it. Lab 03-3 asks yours first.

```bash
# on: laptop
ollama pull hf.co/ggml-org/embeddinggemma-2-GGUF:Q8_0
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/03_matryoshka_and_quantization/labs/lab03_3_quantized_weights.py
```

**Expected output** (captured on this Mac, Ollama 0.34.4 — too old for the new architecture)

```
▣ STEP 1 · can your GGUF runtimes load gemma-embedding2? (text-only GGUF)
⚠ Ollama has the GGUF but cannot run it: unknown model architecture 'gemma-embedding2' — upgrade Ollama (support arrived after 0.34.x)
◆ Ollama here: ollama version is 0.34.4
◆ llama.cpp here: version: 0.5.0 (build 11146, commit 7fe450e19)
◆ REFERENCE (Unsloth guide): llama-server --model embeddinggemma-2-UD-Q4_K_XL.gguf --embeddings --pooling mean

▣ STEP 2 · int8 dynamic quantization of the text model — PyTorch, CPU
◆ quantized engine: qnnpack (available: qnnpack)
│ weights               serialized size
│ ────────────────────  ───────────────
│ float32               1034 MB
│ int8 (Linear layers)  644 MB

▣ STEP 3 · speed and agreement on the hotel knowledge base
│ weights  36 texts  per text  recall@1
│ ───────  ────────  ────────  ────────
│ float32  0.44s     12 ms     1.000
│ int8     0.87s     24 ms     1.000
✓ int8 vs float32 document vectors: mean cosine 0.9700 (min 0.9617)
```

This is what edge work looks like on launch week. The GGUF downloaded, but Ollama 0.34.4 and llama.cpp 0.5.0 on this Mac did not yet know `gemma-embedding2`. Ollama's 0.40.0 release notes mention EmbeddingGemma 2. Upgrade, rerun, and the step turns ✓.

The int8 model is real and runs today. It is 38 % smaller, keeps full recall, and its vectors agree with float32 at a cosine of 0.97. On this ARM CPU it was **slower**, though, because the `qnnpack` int8 kernels lose to Apple's fast float32 maths. Measure speed on the real target device; do not assume quantized means fast.

✓ Checkpoint: you can name two reasons a GGUF might not run on a given box (a runtime too old for the architecture, or a GGUF that lacks the modality you need).

## Labs — run them here

**labs/lab03_1_matryoshka.py** — Recall, MRR and storage at 768/512/256/128 dimensions, plus the re-normalisation trap reproduced.

**labs/lab03_2_int8_and_binary.py** — Every combination of dimension and precision, from 3,072-byte float vectors down to 16-byte fingerprints.

**labs/lab03_3_quantized_weights.py** — Whether your Ollama and llama.cpp can run the text-only GGUF, then int8 PyTorch weights measured for size, speed and agreement.

## Try it yourself

`exercises/ex03_truncate_renorm.py` builds the storage layer of an edge vector index:

1. `truncate(v, dim)`: cut and re-normalise, and refuse any dimension the model was not trained for.
2. `storage_bytes(n, dim, precision)`
3. `pack_bits(v)` and `hamming(a, b)`: binary quantization by hand.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/03_matryoshka_and_quantization/exercises/ex03_truncate_renorm.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ truncate(v, 128): the first 128 numbers, re-normalised to length 1
✓ truncate(v, 300) raises ValueError — only 768/512/256/128 were trained
✓ storage_bytes: 1M × 768 fp32 = 3.07 GB · 256 int8 = 256 MB · 128 binary = 16 MB
✓ pack_bits: positive → 1, else 0, 8 per byte · hamming counts differing bits
```

<details><summary>Hint — hamming()</summary>

`np.bitwise_xor(a, b)` gives 1-bits exactly where a and b differ; `np.unpackbits(...).sum()` counts them.

</details>

✓ Checkpoint: all four checker lines are ✓.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `unknown model architecture: 'gemma-embedding2'` | Ollama or llama.cpp is older than the model | upgrade (Ollama 0.40.0 lists EmbeddingGemma 2), or stay on the PyTorch path |
| `Didn't find engine for operation quantized::linear_prepack NoQEngine` | no int8 engine selected | the lab picks `qnnpack` on ARM; on x86 it uses `x86`/`fbgemm` |
| `DeprecationWarning: … sentence_transformers.quantization` | old import path | import from `sentence_transformers.util.quantization` |

## Next

[Lab 04 — One space for text, photos, voice and video](../04_multimodal_search/TUTORIAL.md): load every encoder and let a sentence find a photo, a voicemail and a CCTV clip.
