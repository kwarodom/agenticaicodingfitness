# ▶ Embed Lab 08 — Your Mac as the edge box: MLX, ONNX and an always-on embedding server

> Part of Week 30 · Multimodal embeddings with EmbeddingGemma 2 on local, sovereign and edge hardware. Every lab runs on your own machine. Every number in this course was measured on the machine that ran it, or it is quoted with its source.

**What you'll actually do**
- Treat your Mac as the edge device, not a stand-in for one. In Tina Huang's tiers a MacBook is a "personal computer" and a Mac mini or Mac Studio is a "home server".
- Run EmbeddingGemma 2 on this Mac five ways: PyTorch on the Apple GPU, **MLX** (Apple's own ML framework) in bf16 and 4-bit, and **ONNX Runtime** in q4 and fp16. Measure them side by side.
- Try Apple's **Neural Engine** route through CoreML, and see why it does not run this model today.
- Run the full multimodal model in MLX: photos and voicemails, bf16 vs 4-bit, with accuracy checked per modality.
- Turn the Mac into the hotel's **always-on embedding server**: a local HTTP service that small devices can call, a 60-second load test, a power reading, and a launchd file that keeps it running.

**Time** ~60 min · **Difficulty** advanced · **Hardware** a Mac with Apple silicon (M1 or newer) · ~4 GB extra disk for the MLX and ONNX builds

**Sources:** [mlx-community/embeddinggemma-2-bf16](https://huggingface.co/mlx-community/embeddinggemma-2-bf16) · [mlx-community/embeddinggemma-2-4bit](https://huggingface.co/mlx-community/embeddinggemma-2-4bit) · [onnx-community/embeddinggemma-2-ONNX](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX) · [MLX](https://github.com/ml-explore/mlx) · [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) · [launch blog](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/) (lists MLX) · Tina Huang, [*Every Size Local AI In 24 Minutes*](https://youtu.be/rPGJhrunbxo) · [`week30_multimodal_embeddings_edge/RESEARCH.md`](../RESEARCH.md) §5

## 0 · Before you start

MLX gets its own environment, `week30_multimodal_embeddings_edge/.venv-mlx`. The MLX builds were converted with a specific `mlx-vlm` commit, and their model card says to install exactly that one. The labs start in `.venv-embed` like every other lab and switch themselves into `.venv-mlx` when they need MLX. You will see a `→ MLX lives in week30_multimodal_embeddings_edge/.venv-mlx` line when that happens.

```bash
# on: laptop
uv venv --python 3.12 week30_multimodal_embeddings_edge/.venv-mlx
uv pip install --python week30_multimodal_embeddings_edge/.venv-mlx/bin/python "git+https://github.com/Blaizzy/mlx-vlm.git@3d87e88402f307efbf68e568971aa887ee7d9ed0" "mlx>=0.32.3" "transformers>=5.18.0" torch torchvision torchcodec soundfile librosa pillow numpy fastapi uvicorn
uv pip install --python week30_multimodal_embeddings_edge/.venv-embed/bin/python onnxruntime
week30_multimodal_embeddings_edge/.venv-mlx/bin/python -c "import mlx.core as mx; print('mlx', mx.__version__, mx.default_device())"
```

**Expected output** (captured on this Mac)

```
mlx 0.32.3 Device(gpu, 0)
```

`torch` is in the MLX environment only because the model's processor needs it to turn photos and audio into model inputs. MLX does all the embedding. The labs download each build once: MLX bf16 is 1.49 GB, MLX 4-bit 1.10 GB, and the ONNX text model 175 MB (q4) plus 543 MB (fp16). ONNX goes into `week30_multimodal_embeddings_edge/.models/` (git-ignored), because ONNX Runtime refuses the Hugging Face cache's symlinked files.

✓ Checkpoint: `mlx` prints a version and `Device(gpu, 0)`.

## 1 · Why a Mac is a good edge box

A Mac with Apple silicon has one feature most edge devices lack: **unified memory**. The CPU, GPU and Neural Engine share one pool of RAM. In Module 01's kitchen, it is one big prep counter that the head chef, the line cooks and the onion machine can all reach. Nothing has to be copied to a separate graphics card, so a 36 GB Mac can hold the embedding model, the vector index and a local Gemma 4 at the same time.

| on this Mac | value | where it came from |
|---|---|---|
| chip · memory | Apple M4 Max · 36 GB unified | Lab 01-2 |
| GPU memory copy | 314 GB/s (of 410 GB/s peak) | Lab 01-2, Apple spec |
| ladder rung | personal computer (a Mac mini or Mac Studio = home server) | Tina Huang's tiers |

Three ways to run EmbeddingGemma 2 on it:

| runtime | what it is | builds used here |
|---|---|---|
| PyTorch + MPS | the reference stack; MPS is PyTorch's Apple-GPU backend | `google/embeddinggemma-2` (Labs 02–07) |
| **MLX** | Apple's open-source array framework, made for Apple silicon and unified memory | `mlx-community/embeddinggemma-2-bf16`, `-4bit` |
| **ONNX Runtime** | a portable inference engine; on a Mac it can hand work to CoreML and the Neural Engine | `onnx-community/embeddinggemma-2-ONNX` (q4, fp16 text model) |

✓ Checkpoint: you can say why unified memory matters when one Mac runs the embedder, the index and a chat model together.

## 2 · Three runtimes, one Mac

Lab 08-1 runs the same text job in each runtime, each in a fresh process so load time and memory are its own. The job: 18 questions and 18 notes, with the right prefixes typed in by hand, since only sentence-transformers adds them automatically. Then it compares every result with PyTorch float32 on the CPU.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/08_mac_edge_box/labs/lab08_1_three_runtimes.py
```

**Expected output** (captured on this Mac)

```
▣ STEP 6 · ONNX Runtime · q4 · CoreML
◆ CoreML could take 238 of 1066 graph nodes (22 %) — the rest would stay on the CPU
⚠ ONNX Runtime · q4 · CoreML did not run: Input (video_features) has a dynamic shape ({-1,512}) but the runtime shape ({0,512}) has zero elements. This is not supported by the CoreML EP.

▣ STEP 7 · agreement with PyTorch float32 on the CPU (the reference)
│ runtime                    weights on disk  load  1 query  per note in ×32  recall@1  min cos vs fp32  peak memory
│ ─────────────────────────  ───────────────  ────  ───────  ───────────────  ────────  ───────────────  ──────────────────
│ PyTorch + MPS · bf16       1489 MB          3.7s  22.5 ms  3.7 ms           1.000     0.9999           1232 MB
│ MLX · bf16                 1489 MB          0.8s  7.3 ms   2.1 ms           1.000     0.9998           2644 MB (MLX 2029)
│ MLX · 4-bit                1098 MB          0.9s  6.1 ms   2.1 ms           1.000     0.9598           2274 MB (MLX 1690)
│ ONNX Runtime · q4 · CPU    175 MB           1.0s  15.5 ms  26.8 ms          1.000     0.9703           1278 MB
│ ONNX Runtime · fp16 · CPU  543 MB           1.3s  17.2 ms  32.4 ms          1.000     1.0000           1500 MB
```

What the table says:

- **MLX is the fastest on this Mac.** One query takes 7.3 ms in MLX bf16 against 22.5 ms in PyTorch, three times faster, with the same answers (cosine 0.9998). MLX also loads in under a second.
- **4-bit MLX buys size, not much speed, here.** It is 26 % smaller and only slightly faster for one query (6.1 vs 7.3 ms), with the same 2.1 ms per note in a batch. Its vectors drift further, too (worst cosine 0.96). Small text models on a big GPU are not memory-bound (Module 06), so cutting bytes buys little speed.
- **ONNX q4 is the smallest by far.** At 175 MB it is text only and runs on the CPU, so it suits a box with little disk. It is quick for one query (15.5 ms) but slow in batches (26.8 ms per note).
- **All five ranked every question correctly** (recall@1 = 1.000). The differences are speed, size and drift, not answers on an easy set.

And the honest line, step 6: the **Neural Engine** route does not work for this model today. CoreML could take only 22 % of the graph, and it refuses the model's empty video input (a tensor with zero rows, which a text-only call passes). "The Mac has an NPU" does not mean "this model uses it". Check before you promise it.

✓ Checkpoint: you can name the fastest runtime on your Mac, the smallest, and the one to use as the reference.

## 3 · MLX with photos and voicemails

MLX runs the **whole** multimodal model, not just text. Lab 08-2 repeats Module 04's seven cross-modal questions in MLX bf16 and MLX 4-bit, and measures each build's drift from PyTorch float32 per modality. The 4-bit model card warns that audio is the modality quantization affects most.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/08_mac_edge_box/labs/lab08_2_mlx_multimodal.py
```

**Expected output** (captured on this Mac)

```
▣ STEP 4 · compare
│ build     weights  questions right  min cos text  min cos photo  min cos voice  per photo  per voicemail
│ ────────  ───────  ───────────────  ────────────  ─────────────  ─────────────  ─────────  ─────────────
│ MLX bf16  1489 MB  7/7              0.9999        0.9999         1.0000         151 ms     54 ms
│ MLX 4bit  1098 MB  7/7              0.9865        0.9831         0.9794         150 ms     59 ms
◆ REFERENCE (mlx-community 4-bit card): minimum cosine vs FP32 — text 0.980877 · image 0.990386 · audio 0.974435
```

Both builds answer all seven questions correctly, and MLX is faster here than PyTorch was in Module 04: 151 ms per photo against 247 ms, and 54 ms per voicemail against 111 ms. Compare the 4-bit row with the model card. Text and voice land close to the card's numbers. The **photo** score is lower here (0.983 vs the card's 0.990), because these are different photos from the card's test set. That is why you measure on your own data. For a voicemail index, choose bf16: it is the closest match to the original, and on this Mac 4-bit gives no speed benefit.

✓ Checkpoint: you can choose between MLX bf16 and 4-bit for a voicemail index and justify it with the voice column.

## 4 · The Mac mini as the hotel's embedding server

Tina Huang's home-server tier is "a dedicated always-on machine". Her Arduino, too small for any model, used a Mac Studio over Wi-Fi. `week30_multimodal_embeddings_edge/common/embed_service.py` applies that pattern to embeddings. It is a small FastAPI service on MLX 4-bit (the fastest single query) that pre-embeds the hotel notes and answers `/embed` and `/search` over plain JSON. A gateway, a Pi or a microcontroller sends words and gets back meaning.

Safety is built in. The service listens on **loopback** (`127.0.0.1`) by default. It refuses any network address unless `EMBED_TOKEN` is set to 24 characters or more, and then every request must carry that token. Anyone who can reach `/search` can read the knowledge base, so expose it on a trusted network only.

Lab 08-3 starts the service, plays the small device, runs a 60-second load test and checks power:

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/08_mac_edge_box/labs/lab08_3_always_on_service.py
```

**Expected output** (captured on this Mac, on AC power)

```
▣ STEP 2 · be the small device: plain JSON over HTTP
  POST /search {"q": "someone is stuck in the lift"}  →  [{'title': 'Lift entrapment', 'score': 0.832}, {'title': 'Pool pump trip', 'score': 0.651}]
│ dim  request  response (JSON)  round trip
│ ───  ───────  ───────────────  ──────────
│ 768  54 B     16,253 B         9.1 ms
│ 256  54 B     5,377 B          8.8 ms
│ 128  54 B     2,702 B          8.6 ms
◆ a microcontroller with 8 MB of RAM can parse a 128-d answer (≈ 2.7 KB of JSON); send /search instead and it never handles a vector at all

▣ STEP 3 · round-trip latency: 30 searches
✓ p50 9.0 ms · p95 9.8 ms · over loopback HTTP + JSON

▣ STEP 4 · sustained load: 60 s of 32-note batches — does the Mac slow down?
  window 1 (15 s):    409 notes/s  ████████████████████████
  window 2 (15 s):    387 notes/s  ███████████████████████░
  window 3 (15 s):    383 notes/s  ██████████████████████░░
✓ 23,264 notes in 60 s · last window vs first: -6.3 %

▣ STEP 5 · power — no sudo needed
◆ power source: AC — the battery is not discharging, so its draw cannot be read without sudo
◆ unplug and rerun for whole-Mac watts, or run Apple's tool yourself in the ⌨ terminal:
  $ sudo powermetrics --samplers cpu_power,gpu_power -i 1000 -n 10
```

A search takes 9 ms end to end, including HTTP and JSON. The Mac embedded about 390 notes per second for a full minute, slowing by 6 % as it warmed up. At that rate, the 50,000 hotel notes from Module 06 take about two minutes.

For power, the lab reads the battery's draw when the Mac is unplugged, which needs no admin password. On AC it says so honestly, and gives you Apple's `powermetrics` command to run yourself, since that one needs `sudo`.

The last step writes, but does **not** install, a launchd file that starts the service at login and restarts it if it stops:

```text
week30_multimodal_embeddings_edge/08_mac_edge_box/.runs/com.alto.edge-memory.plist   (RunAtLoad · KeepAlive · --host 127.0.0.1 · HF_HUB_OFFLINE=1)
```

```bash
# on: laptop
# only if you want it running permanently — this starts it at every login
cp week30_multimodal_embeddings_edge/08_mac_edge_box/.runs/com.alto.edge-memory.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.alto.edge-memory.plist
launchctl unload ~/Library/LaunchAgents/com.alto.edge-memory.plist   # stop it
```

✓ Checkpoint: you know the service's p50 latency, its sustained throughput, and the one environment variable that must be set before it may listen on the network.

## Labs — run them here

**labs/lab08_1_three_runtimes.py** — PyTorch MPS, MLX bf16/4-bit and ONNX Runtime q4/fp16 measured side by side on this Mac, plus the CoreML Neural Engine attempt.

**labs/lab08_2_mlx_multimodal.py** — Photos and voicemails in MLX bf16 and 4-bit, with the seven cross-modal questions and accuracy against PyTorch per modality.

**labs/lab08_3_always_on_service.py** — The Mac as a local embedding server: JSON round trips, a 60-second load test, power without sudo, and a launchd file.

## Try it yourself

`exercises/ex08_mac_edge_plan.py` turns the measurements into a deployment plan:

1. Pick a runtime for four jobs from the **measured** table.
2. `safe_bind(host, token)`: the service's network rule.
3. `joules_per_1000(watts, notes_per_s)`: energy from a power reading.
4. Fix a broken launchd file.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/08_mac_edge_box/exercises/ex08_mac_edge_plan.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ runtimes: voicemails → MLX bf16 · smallest text-only → ONNX q4 · fastest query → MLX 4-bit · reference → PyTorch
✓ safe_bind: loopback always · a network address only with a token of ≥ 24 characters
✓ joules_per_1000: 18 W at 400 notes/s = 45.0 J · 9.5 W at 120/s = 79.2 J
✓ launchd plist: starts at login, restarts if it dies, loopback only, offline
```

<details><summary>Hint — measure power yourself</summary>

Unplug the Mac and run Lab 08-3: step 5 reads the battery's draw and prints joules per 1,000 notes. Or, plugged in, run `sudo powermetrics --samplers cpu_power,gpu_power -i 1000 -n 10` in a second terminal while the load test runs, and plug the watts into your `joules_per_1000`.

</details>

✓ Checkpoint: all four checker lines are ✓.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `✕ week30_multimodal_embeddings_edge/.venv-mlx is missing` | the MLX environment was not created | run the Section 0 commands |
| `Could not import module 'EmbeddingGemma2Processor'` | the processor needs PyTorch, and `.venv-mlx` has none | install `torch torchvision` into `.venv-mlx` (Section 0 does) |
| `External data path escapes model directory` | ONNX Runtime refuses the Hugging Face cache's symlinked blobs | the labs download ONNX into `week30_multimodal_embeddings_edge/.models/` instead |
| CoreML step fails | the model's empty media inputs are unsupported, and only 22 % of the graph runs there | expected today; use MLX or ONNX on the CPU |
| `refusing to listen on a network address` | no `EMBED_TOKEN`, or one shorter than 24 characters | `export EMBED_TOKEN=$(openssl rand -hex 24)` — and only on a trusted network |
| the launchd service does not start | wrong paths in the plist, or the model is not cached | check `/tmp/alto-edge-memory.log`; run a lab once to cache the model |

## Next

You turned the Mac from a stand-in into the edge box. Back to the course: [Lab 01 — Local, sovereign, edge](../01_local_sovereign_edge/TUTORIAL.md), or rerun [Lab 07 — the capstone](../07_capstone_alto_edge_memory/TUTORIAL.md) with the agent's `search_memory` calling this service instead of loading the model itself.
