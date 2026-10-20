# ▶ Embed Lab 06 — Benchmark like an edge engineer

> Part of Week 30 · Multimodal embeddings with EmbeddingGemma 2 on local, sovereign and edge hardware. Every lab runs on your own machine. Every number in this course was measured on the machine that ran it, or it is quoted with its source.

**What you'll actually do**
- Measure p50 and p95 latency for text, a batch of 32 texts, a photo, a voicemail and a CCTV clip, on the GPU and on the CPU.
- Test the model card's warning about float16 yourself, and find the input that makes it fail.
- Compute the **bandwidth floor** for every rung of the device ladder, then find out it is the wrong limit for an embedding model.
- Plan an index build from measured numbers, and label every estimate as one.

**Time** ~45 min · **Difficulty** advanced · **Hardware** any laptop · ~5 GB free RAM · a GPU (MPS or CUDA) optional

**Sources:** [model card — Numerical Precision](https://huggingface.co/google/embeddinggemma-2) · Tina Huang, [*Every Size Local AI In 24 Minutes*](https://youtu.be/rPGJhrunbxo) (capacity vs bandwidth) · vendor bandwidth specs in [`week30_multimodal_embeddings_edge/RESEARCH.md`](../RESEARCH.md) §4

## 0 · Before you start

Benchmarks lie when the machine is busy. Close heavy apps, plug in the laptop, and run each lab twice; keep the second run. Every number here is a **LAPTOP STAND-IN**: it describes this machine, not a Raspberry Pi or a phone.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python -c "import torch; print('MPS', torch.backends.mps.is_available(), '· CUDA', torch.cuda.is_available())"
```

✓ Checkpoint: you know whether your labs will measure a GPU, a CPU or both.

## 1 · Latency by modality: p50 and p95

Quote **percentiles**, never one lucky run. p50 is the typical case; p95 is what one user in twenty waits. Lab 06-1 warms the model up first (the first call compiles kernels and fills caches), then times each workload repeatedly.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/06_edge_benchmarking/labs/lab06_1_latency_by_modality.py
```

**Expected output** (captured on this Mac — Apple M4 Max, LAPTOP STAND-IN)

```
▣ STEP 1 · mps · bfloat16
│ work            runs  p50      p95      per item
│ ──────────────  ────  ───────  ───────  ─────────  ──────────────────
│ text ×1         20    22 ms    24 ms    22.2 ms    ░░░░░░░░░░░░░░░░░░
│ text ×32        8     135 ms   152 ms   4.2 ms     █░░░░░░░░░░░░░░░░░
│ photo           8     259 ms   264 ms   258.9 ms   ██░░░░░░░░░░░░░░░░
│ voicemail 12 s  6     105 ms   107 ms   105.3 ms   █░░░░░░░░░░░░░░░░░
│ CCTV 10 s       4     1247 ms  1251 ms  1247.3 ms  ███████░░░░░░░░░░░

▣ STEP 3 · GPU vs CPU
│ work            MPS p50  CPU p50  CPU / MPS
│ ──────────────  ───────  ───────  ─────────
│ text ×1         22 ms    34 ms    1.5×
│ text ×32        135 ms   495 ms   3.7×
│ photo           259 ms   920 ms   3.6×
│ voicemail 12 s  105 ms   1091 ms  10.4×
│ CCTV 10 s       1247 ms  4191 ms  3.4×
```

Read it like an edge engineer:

- **Batching** is the biggest lever. One text costs 22 ms; inside a batch of 32 it costs 4.2 ms.
- **Media costs tokens.** A photo is 280 tokens and 10 s of video is 1,400, which is why the CCTV clip is the slowest item.
- **The CPU is not hopeless.** 34 ms for a query is fine for search. Indexing photos and clips is where a GPU or NPU pays off.
- **p95 matters on shared hardware.** The CPU's text p95 was 152 ms against a 34 ms p50, because other work on the machine got in the way.

✓ Checkpoint: you can quote your machine's p50 and p95 for one text query, and the per-item cost inside a batch of 32.

## 2 · Precision: why the model card bans float16

Halving the bytes per weight halves the memory traffic. **bfloat16** keeps float32's exponent range at half the size. **float16** has more precision but a much smaller range. The model card warns that EmbeddingGemma 2's activations exceed float16's range, so it "returns NaN or silently degraded embeddings rather than raising an error".

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/06_edge_benchmarking/labs/lab06_2_precision.py
```

**Expected output** (captured on this Mac, CPU)

```
▣ STEP 3 · float16
✓ loaded google/embeddinggemma-2 · encoders text · 271.0M params · cpu · 1.6s
◆ 517 MB of weights · 36 short texts in 1.90s, NaN 0/36, recall@1 1.000 · long document in 63.7s, NaN: YES

▣ STEP 4 · agreement with float32
│ dtype     weights  short NaN  recall@1  long doc  short cos vs fp32  long cos vs fp32
│ ────────  ───────  ─────────  ────────  ────────  ─────────────────  ────────────────
│ float32   1034 MB  0/36       1.000     ok        1.0000             1.0000
│ bfloat16  517 MB   0/36       1.000     ok        1.0000             1.0000
│ float16   517 MB   0/36       1.000     NaN       1.0000             NaN
⚠ float16 returned a NaN vector — no exception, no warning. Exactly the failure the model card describes; a test set of short texts would never have shown it.
```

On 36 short texts, all three formats pass. The long document (3,502 tokens) is where float16 breaks: it returned a NaN vector with no exception and no warning, exactly as the model card says. Any search that touched that vector would silently return garbage. A test set of short texts would never have caught it, which is why you follow the vendor's rule (bfloat16 where the hardware has it, float32 elsewhere) and keep a long input in your regression tests. Half precision was also **much slower** on this CPU (67 s vs 2.0 s for the long document), because CPUs often lack fast bf16/fp16 maths. Precision is a GPU/NPU lever, not a free CPU win.

✓ Checkpoint: you can explain why short test texts passed while the long document returned NaN, and why that makes "it worked on my test set" dangerous.

## 3 · What limits an embedding model? Not the belt

Module 01's conveyor belt says: one forward pass must stream all the weights at least once, so it can never be faster than **weight bytes ÷ bandwidth**. That is the **bandwidth floor**. For an LLM writing one token at a time, the floor is close to reality. Lab 06-3 checks it for an embedding model.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/06_edge_benchmarking/labs/lab06_3_tier_projection.py
```

**Expected output** (captured on this Mac; other devices = ESTIMATE)

```
▣ STEP 1 · bandwidth floor per forward pass (ms) — weights cross the bus at least once
│ device                                 peak bandwidth  text Q4 GGUF  text Q8_0 GGUF  text bf16  full bf16
│ ─────────────────────────────────────  ──────────────  ────────────  ──────────────  ─────────  ─────────
│ Raspberry Pi 5 (CPU only)              17 GB/s         10.1          17.8            31.0       85.0
│ DGX Spark (GB10)                       273 GB/s        0.6           1.1             1.9        5.3
│ AMD Ryzen AI Max+ 395                  256 GB/s        0.7           1.2             2.1        5.6
│ Apple M4 Max 14-core (this Mac, peak)  410 GB/s        0.4           0.7             1.3        3.5
│ RTX 4090                               1008 GB/s       0.2           0.3             0.5        1.4
│ H100 SXM                               3350 GB/s       0.1           0.1             0.2        0.4
◆ ESTIMATE — a lower bound from vendor peak bandwidth, never a measurement of that device

▣ STEP 2 · this machine: measured time vs the bandwidth floor vs the work
│ work (MPS bf16)  tokens  measured p50  bandwidth floor  measured ÷ floor  work         achieved
│ ───────────────  ──────  ────────────  ───────────────  ────────────────  ───────────  ────────────
│ text ×1          14      22.2 ms       1.3 ms           17×               3.6 GFLOP    0.16 TFLOP/s
│ text ×32         1381    135.5 ms      1.3 ms           105×              359.1 GFLOP  2.65 TFLOP/s
```

The measured time is 17× to 105× the floor, so the belt is **not** what limits an embedding model. An encoder reads all its input tokens in **one** pass. The weights cross the bus once, while the maths grows with the token count. At batch 1, fixed overheads dominate (launching kernels, tokenising); at batch 32, the arithmetic does (359 GFLOP). So when you size an edge device for embeddings, look at its **compute** (CPU, GPU, NPU), and treat bandwidth as a floor only. The video's capacity-vs-bandwidth lesson still holds for the chat model that sits next to it.

✓ Checkpoint: you can explain why the same Raspberry Pi is bandwidth-limited for a chat model but compute-limited for an embedding model.

## 4 · Plan an index build from measured numbers

Step 3 of Lab 06-3 plans the job a hotel actually has: embed 50,000 notes overnight.

**Expected output** (captured on this Mac)

```
▣ STEP 3 · index plan for 50,000 hotel notes — from MEASURED per-item times on this machine
│ where                                             per note  50,000 notes
│ ────────────────────────────────────────────────  ────────  ──────────────────────
│ this machine · mps (measured)                     4.2 ms    3.5 min
│ this machine · cpu (measured)                     15.5 ms   12.9 min
│ Raspberry Pi 5 · bandwidth floor only (ESTIMATE)  —         ≥ 28 s, real: far more
```

Even on a CPU, a whole hotel's notes embed in minutes. On a smaller device it might take an hour, which is still a nightly batch job: re-embed what changed, and search all day. Before you promise a number for a Pi, a phone or a Jetson, run Lab 06-1 **on that device**. This lab only prints what it measured here, and labels everything else as an estimate.

✓ Checkpoint: you have a plan with a measured per-item time, and you can say which number in it is an estimate.

## Labs — run them here

**labs/lab06_1_latency_by_modality.py** — p50/p95 for text, batches, photos, voicemail and CCTV, on the GPU (bfloat16) and the CPU (float32).

**labs/lab06_2_precision.py** — float32 vs bfloat16 vs float16: memory, speed, NaNs and agreement, with the model card's warning tested honestly.

**labs/lab06_3_tier_projection.py** — Bandwidth floors for every rung, set against measured time and FLOPs, and an index plan built from measured numbers.

## Try it yourself

`exercises/ex06_benchmark_math.py` has four TODOs, the maths behind every benchmark table:

1. `percentile(values, p)` with linear interpolation.
2. `throughput(items, ms)`.
3. `bandwidth_floor_ms(weights_mb, gbps)`.
4. `limited_by(measured_ms, floor_ms)`: "bandwidth" within 2× of the floor, else "compute or overhead".

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/06_edge_benchmarking/exercises/ex06_benchmark_math.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ percentile: p50 22.3 ms · p95 48.75 ms (linear interpolation) · one value → itself
✓ throughput: 32 items in 135.5 ms = 236.2 items/s · 1 in 22.2 ms = 45.0/s
✓ bandwidth_floor_ms: 540 MB at 410 GB/s = 1.29 ms · 310 MB at 17 GB/s = 17.8 ms
✓ limited_by: far above the floor → compute or overhead · within 2× of it → bandwidth
```

<details><summary>Hint — percentile()</summary>

Sort the values. The position is `i = (n − 1) · p / 100`. Take the values at `floor(i)` and `ceil(i)` and interpolate by the fractional part of `i`. That is numpy's default.

</details>

✓ Checkpoint: all four checker lines are ✓.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| p95 far above p50 | other work on the machine, or thermal throttling | close apps, plug in, run twice and keep the second run |
| Lab 06-3 says "run Lab 06-1 first" | it reads `.runs/latency.json` | run Lab 06-1 |
| CPU bf16 slower than fp32 | no fast bf16 instructions on this CPU | expected — use float32 on such CPUs (model card) |
| MPS out of memory | other GPU apps, or all encoders in fp32 | use bfloat16 on MPS (the lab does) |

## Next

[Lab 07 — Capstone: Alto Edge Memory](../07_capstone_alto_edge_memory/TUTORIAL.md): one offline index of everything, an agent that searches it, and a grade based on evidence.
