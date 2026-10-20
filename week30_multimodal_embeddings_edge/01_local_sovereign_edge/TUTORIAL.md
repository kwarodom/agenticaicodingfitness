# ▶ Embed Lab 01 — Local, sovereign, edge: the device ladder

> Part of Week 30 · Multimodal embeddings with EmbeddingGemma 2 on local, sovereign and edge hardware. Every lab runs on your own machine. Every number in this course was measured on the machine that ran it, or it is quoted with its source.

**What you'll actually do**
- Learn the difference between **local**, **sovereign** and **edge** AI, and why they are not the same promise.
- Climb Tina Huang's device ladder, from a 32 KB Arduino to eight H100 GPUs, with her rule of thumb for what fits.
- Learn the one idea that explains the ladder: **capacity** decides what fits, **bandwidth** decides how fast it runs.
- Find out where an **embedding model** lands on the ladder. It sits several rungs below any chat model.
- Measure your own machine: its RAM, its CPU memory bandwidth and its GPU memory bandwidth.

**Time** ~35 min · **Difficulty** beginner · **Hardware** any laptop · no model download needed

**Sources:** Tina Huang, [*Every Size Local AI In 24 Minutes*](https://youtu.be/rPGJhrunbxo) (28 Sep 2026) · [EmbeddingGemma 2 launch blog](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/) · [model card](https://huggingface.co/google/embeddinggemma-2) · [Sector88 — What is sovereign AI?](https://www.sector88.co/blog/what-is-sovereign-ai) · [`week30_multimodal_embeddings_edge/RESEARCH.md`](../RESEARCH.md)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| The Week 30 Python | `week30_multimodal_embeddings_edge/.venv-embed/bin/python --version` → 3.12 | all labs run in it (torch, transformers 5.19, sentence-transformers 6.1) |
| The Embed Lab Runner | `.venv/bin/python week30_multimodal_embeddings_edge/00_embed_lab_runner/tutorial_server.py` | the course, ▶ run buttons and the ⌨ terminal at http://127.0.0.1:8130 |

If `.venv-embed` is missing, `week30_multimodal_embeddings_edge/README.md` has the two `uv` commands that create it.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python -c "import torch, transformers, sentence_transformers as s; print(torch.__version__, transformers.__version__, s.__version__)"
```

**Expected output** (captured on this Mac)

```
2.14.1 5.19.0 6.1.0
```

✓ Checkpoint: the command prints three versions, with transformers 5.19 or newer and sentence-transformers 6.1 or newer.

## 1 · Local, sovereign, edge — three different promises

People use these words as if they meant the same thing. They do not.

| word | the promise | the test |
|---|---|---|
| **local** | the model runs on hardware in front of you | pull the network cable; does it still answer? |
| **sovereign** | you control the hardware, the model version and the data, inside your own jurisdiction, and nothing phones home | who could read, change or switch off this system, and from which country? |
| **edge** | inference happens where the data is born: a phone, a gateway, a box in the plant room | does the data have to travel before it is understood? |

Sovereign AI, in one 2026 definition: "building and running artificial intelligence on infrastructure you own or directly control, inside your own legal jurisdiction, using models and data that never leave your perimeter" (Sector88). A vendor's cloud pinned to your country's region is **data residency**, not sovereignty. The vendor still runs it.

Edge AI is where sovereignty becomes physics. If the model lives on the device, the data never had to leave.

This week adds one point that most discussions miss: **embeddings are sensitive data**. A vector is a compressed copy of the meaning of a guest's voicemail, a CCTV frame or a maintenance note. Research shows that text can largely be rebuilt from its embedding (Morris et al., EMNLP 2023). Sending documents to a cloud embedding API is a data transfer, even if you only keep the vectors.

✓ Checkpoint: you can say why "our vendor hosts it in Thailand" is residency, and what would make it sovereign.

## 2 · The device ladder

In *Every Size Local AI In 24 Minutes*, Tina Huang runs AI on every size of hardware she could find. She groups it into five classes:

| class | her device | memory (as she quotes it) | what ran |
|---|---|---|---|
| chips (no OS) | Arduino Uno R4 | 32 KB | nothing — but it calls a bigger machine over Wi-Fi |
| chips | ESP32-S3 (~$20) | 8 MB | TinyStories 260K and 3M |
| mini computers | Raspberry Pi 5 | 8 GB | Whisper → Qwen3 1.7B → Piper, a full voice loop; Moondream on webcam frames |
| mini computers | iPhone 16 | 8 GB (apps get ~4–5 GB) | Gemma 4 (AI Edge Gallery), Stable Diffusion 1.5, Whisper Large, YOLO11n |
| personal computers | MacBook Pro | 32 GB | "pretty much any category": Qwen3 14B–32B, Qwen3 Coder 30B, Flux Dev |
| home servers | Mac Studio · AMD Ryzen AI Halo | 64 GB · 128 GB | 27–32B LLMs · Llama 3.3 70B, GPT-OSS 120B |
| GPUs | RTX 4090 · 8× H100 | 24 GB · 640 GB | fast image and video generation · ultra-large models |

Her rule of thumb for what fits: take the RAM, subtract 25 % for everything else, and divide by 0.6. The result is the largest model in **billions of parameters**. Her own example: 32 GB → 24 GB usable → 24 ÷ 0.6 = **40B**. The 0.6 GB per billion parameters roughly matches a 4-bit chat model plus overhead.

Lab 01-1 walks every rung with that rule.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/01_local_sovereign_edge/labs/lab01_1_device_ladder.py
```

**Expected output** (captured on this Mac)

```
▣ STEP 1 · her rule of thumb, rung by rung
│ class              device                        memory  usable (−25 %)  max params ≈ usable ÷ 0.6
│ ─────────────────  ────────────────────────────  ──────  ──────────────  ─────────────────────────
│ chip               Arduino Uno R4                32 KB   24 KB           —
│ chip               ESP32-S3                      8 MB    6 MB            9.8M
│ mini computer      Raspberry Pi 5                8 GB    6 GB            10.0B
│ mini computer      iPhone 16 (apps get ~4–5 GB)  4.5 GB  3.4 GB          5.6B
│ personal computer  MacBook Pro (the video's)     32 GB   24 GB           40.0B
│ home server        Mac Studio                    64 GB   48 GB           80.0B
│ home server        AMD Ryzen AI Halo             128 GB  96 GB           160.0B
│ GPU                RTX 4090                      24 GB   18 GB           30.0B
│ GPU                8× H100                       640 GB  480 GB          800.0B
◆ worked example from the video: 32 GB → 24 GB usable → 24 ÷ 0.6 = 40B parameters
✓ our function agrees: max_params_b(32) = 40.0
```

✓ Checkpoint: using the rule, you can say how many billions of parameters a 16 GB laptop fits (≈ 20B).

## 3 · Capacity vs bandwidth — the kitchen

The video's best idea is an analogy. Your computer is a restaurant kitchen:

| kitchen | computer | job |
|---|---|---|
| freezer downstairs | storage (SSD) | where the model lives when it is not running |
| prep counter | RAM | where the model sits while it runs. Its size is the **capacity**. |
| conveyor belt | memory bus | moves ingredients to the cooks. Its speed is the **bandwidth**. |
| head chef | CPU | can do anything, but there is only one |
| brigade of line cooks | GPU | many simple workers in parallel |
| onion-chopping machine | NPU | does one job, fast, on almost no power |

The Raspberry Pi 5 and the iPhone 16 both have 8 GB, the same counter. The iPhone is far faster because its belt is wider and it has a GPU and an NPU. Capacity decides **what fits**; bandwidth decides **how fast** it runs. A chat model writes one token at a time and must stream all its weights for every token, so for chat the belt is usually the bottleneck. Module 06 shows that an embedding model behaves differently.

Lab 01-2 measures your own belt.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/01_local_sovereign_edge/labs/lab01_2_measure_this_machine.py
```

**Expected output** (captured on this Mac)

```
▣ STEP 2 · bandwidth — CPU memory copy (numpy, one thread)
✓ CPU copy: 139 GB/s (best of 5, 1 GiB array)

▣ STEP 3 · bandwidth — GPU memory copy (PyTorch)
✓ GPU copy on mps: 314 GB/s (best of 10, 1 GiB tensor)

▣ STEP 4 · where this machine sits on the ladder
│ measure                        value     meaning
│ ─────────────────────────────  ────────  ──────────────────────────
│ capacity                       36 GB     personal computer
│ CPU copy bandwidth (measured)  139 GB/s  one core's view of the bus
│ GPU copy bandwidth (measured)  314 GB/s  closer to the vendor peak
◆ vendor peaks for comparison (RESEARCH.md §4): Raspberry Pi 5 ~17 GB/s · Apple M4 Max (14-core) 410 GB/s · AMD Ryzen AI Max+ 395 256 GB/s · RTX 4090 1,008 GB/s · H100 SXM 3,350 GB/s
```

The GPU copy reached 314 GB/s of the M4 Max's 410 GB/s peak. Measured speed is always below the peak on the spec sheet, and the gap is worth knowing before you promise a latency.

✓ Checkpoint: you know your machine's capacity, its measured CPU and GPU copy bandwidth, and its rung on the ladder.

## 4 · Where an embedding model lands

The video only covers **generative** models: chat, images, video, speech. It skips the model class that powers search, RAG and agent memory: the **embedding model**. An embedding model reads an input and returns a fixed list of numbers, a vector. It never writes text, so it can be small.

EmbeddingGemma 2 (Google DeepMind, released 6 October 2026) has 740M parameters in total. It is modular: a 270M text model, plus a 170M vision encoder and a 300M audio encoder that you only load if you need them. Lab 01-1's step 2 turns that into memory:

**Expected output** (captured on this Mac)

```
│ configuration           params  float32  bfloat16  int8 / Q8_0  4-bit / Q4
│ ──────────────────────  ──────  ───────  ────────  ───────────  ──────────
│ text only               270M    1080 MB  540 MB    270 MB       135 MB
│ text + image            440M    1760 MB  880 MB    440 MB       220 MB
│ text + audio            570M    2280 MB  1140 MB   570 MB       285 MB
│ all four (incl. video)  740M    2960 MB  1480 MB   740 MB       370 MB
◆ weights only — activations, the tokenizer and the runtime add more on top
◆ REFERENCE (launch blog): quantized on a Pixel 11 Pro, text-only uses ~191MB active RAM, full multimodal ~567MB
```

Google quotes about 191 MB of RAM for the quantized text model on a phone. That fits every rung from the Raspberry Pi up. The ESP32-S3 still cannot hold it, with only 6 MB usable. It can still use embeddings, though: it sends a sentence to a bigger machine on the same Wi-Fi and gets the vector back. That is the same trick the video's Arduino uses with the Mac Studio.

✓ Checkpoint: you can explain why a Raspberry Pi can run the hotel's search index all day even though it struggles with a 7B chat model.

## Labs — run them here

**labs/lab01_1_device_ladder.py** — Tina Huang's device ladder with her fit rule, and where EmbeddingGemma 2 lands on it.

**labs/lab01_2_measure_this_machine.py** — Your machine's RAM from the OS, plus CPU and GPU memory bandwidth measured with a 1 GiB copy.

## Try it yourself

`exercises/ex01_fit_formula.py` has four TODOs:

1. `usable_gb(ram)`: Tina's "subtract 25 %".
2. `max_params_b(ram)`: usable GB ÷ 0.6.
3. `weights_mb(params_m, dtype)`: parameters × bytes per parameter.
4. Label four deployments as `sovereign`, `residency only` or `not sovereign`.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/01_local_sovereign_edge/exercises/ex01_fit_formula.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ usable_gb(32) = 24
✓ max_params_b: 32 GB → 40B · 64 GB → 80B · 8 GB → 10B
✓ weights_mb: text bf16 540 MB · full fp32 2960 MB · text q4 135 MB
✓ sovereignty: cloud API ✕ · vendor region = residency only · own server, no egress ✓ · offline phone ✓
```

<details><summary>Hint — the vendor region</summary>

Ask who runs the machine and who could be compelled to hand over the data. If the answer is "the vendor", the data sits in your country but the control does not. That is residency.

</details>

✓ Checkpoint: all four checker lines are ✓.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `week30_multimodal_embeddings_edge/.venv-embed/bin/python: No such file or directory` | the Week 30 venv was not created | run the `uv` lines in `week30_multimodal_embeddings_edge/README.md` |
| Lab 01-2 says "no GPU backend here" | no Apple GPU (MPS) or NVIDIA GPU (CUDA) | fine. Your machine is CPU-only, like the Raspberry Pi in the video |
| the measured bandwidth varies between runs | other apps share the memory bus | close heavy apps; the lab already keeps the best of several runs |

## Next

[Lab 02 — Meet EmbeddingGemma 2](../02_first_embeddings/TUTORIAL.md): download the model once, turn sentences into vectors, and run your first search by meaning.
