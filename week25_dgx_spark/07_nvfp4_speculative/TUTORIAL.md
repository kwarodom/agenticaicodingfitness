# ▶ Spark Lab 07 — NVFP4 quantization and speculative decoding

> Part of Week 25 · DGX Spark: fine-tune, serve, and build sandboxed agents. You type the commands, you see the real output. Every lab also runs in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's playbook, or a clearly marked EXAMPLE.

**What you'll actually do**
- Take NVFP4 apart: 4-bit E2M1 values, one FP8 scale per 16 values, and why that makes **4.5 bits** per weight.
- Quantize 65,536 numbers five ways in pure Python and measure the real error of each format.
- Quantize DeepSeek-R1-Distill-Llama-8B to NVFP4 with **NVIDIA Model Optimizer** on your Spark, then serve it with `trtllm-serve`.
- Measure size and quality: the checkpoint on disk, and six checkable questions against the NVFP4 and BF16 servers.
- Work out why **speculative decoding** speeds up a memory-bound Spark, predict the speed-up from the acceptance rate, then run EAGLE-3 and Draft-Target from the playbook.

**Time** ~55 min · **Difficulty** intermediate · **Hardware** 1 DGX Spark (or none: labs 01 and 04 are pure arithmetic, labs 02–03 run DRY)

**Official playbooks covered:** [NVFP4 Quantization](https://build.nvidia.com/spark/nvfp4-quantization) · [Speculative Decoding](https://build.nvidia.com/spark/speculative-decoding)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Module 01 done | `ssh -o BatchMode=yes spark-a true` | labs 02–03 drive the Spark over SSH |
| Docker without sudo on the Spark | `docker ps` on the Spark | every step here is a container |
| ~60 GB free disk (~160 GB for Section 6) | `df -h ~` on the Spark | the 8B model (~16 GB), its NVFP4 copy, and the TensorRT-LLM images. Section 6 adds gpt-oss-120b (~62 GB by the 4-bit arithmetic) or Llama 3.3 70B FP4 (~40 GB) |
| A Hugging Face login **on the Spark** | `hf auth login` once, on the Spark | the Llama FP4 models in Section 6 are gated. The labs never see your token |
| NGC access for `nvcr.io` images | `docker login nvcr.io` once, on the Spark, if a pull says *unauthorized* | the TensorRT-LLM containers |

```bash
# on: spark
docker ps
df -h ~
hf auth whoami
```

> 💡 Labs 01 and 04 need no Spark at all. If you are following in DRY mode, do them first: they explain every number the Spark labs print.

✓ Checkpoint: `docker ps` works on the Spark without `sudo`, and `df -h ~` shows enough space for the sections you plan to run.

## 1 · Why 4 bits, and why NVFP4 is not INT4

Module 01 showed that a Spark generates each token by reading every active weight from memory once, at 273 GB/s. Fewer bytes per weight means more tokens per second and bigger models per Spark. The playbook sums up NVFP4's promise: *"Cut memory use ~3.5× vs FP16 and ~1.8× vs FP8"* while keeping *"accuracy close to FP8 (usually <1% loss)"*.

The playbook describes NVFP4 as *"a 4-bit floating-point format for NVIDIA Blackwell GPUs"* that, *"unlike uniform INT4 quantization, … keeps floating-point semantics with a shared exponent and a compact mantissa"*. In detail (this layout is from NVIDIA's NVFP4 format description, not the playbook):

| Piece | Format | Shared by | Cost per value |
|---|---|---|---|
| each value | **E2M1**: 1 sign bit, 2 exponent bits, 1 mantissa bit | itself | 4 bits |
| block scale | **FP8 E4M3** | 16 values | 8 ÷ 16 = 0.5 bit |
| tensor scale | FP32 | the whole tensor | ≈ 0 |
| **total** | | | **4.5 bits** |

That 4.5 is why `sparkkit.BITS["nvfp4"]` is 4.5, and why 16 ÷ 4.5 = 3.56× (the playbook's "~3.5×") and 8 ÷ 4.5 = 1.78× ("~1.8×").

An E2M1 value can hold only eight magnitudes: **0, 0.5, 1, 1.5, 2, 3, 4, 6**. Quantizing a block works like this:

```text
block scale  = largest |value| in the 16  ÷ 6        → the largest value lands on ±6
stored scale = that, rounded to FP8 (E4M3)            → 8 bits per block
code         = nearest E2M1 value to (value ÷ scale)  → 4 bits per value
decoded      = code × scale
```

Two design choices matter. **Small blocks (16):** one outlier only spoils the precision of its own 15 neighbours. **An FP8 scale** instead of a power of two: MXFP4, the other 4-bit float format, uses 32-value blocks and power-of-two scales, so its scale fits each block more loosely. Lab 01 measures both effects.

```bash
# on: laptop
.venv/bin/python week25/07_nvfp4_speculative/labs/lab01_nvfp4_simulator.py
```

**Expected output** (captured on this Mac: pure Python with a fixed seed, so every machine prints the same numbers)

```
▣ STEP 1 · the E2M1 grid: every magnitude a 4-bit NVFP4 value can hold
│ 0  0.5  1  1.5  2  3  4  6   (and their negatives: 15 distinct values, ±0 share a code)

▣ STEP 2 · one 16-value block by hand
│ amax of block = 0.0913  →  block scale = amax / 6 = 0.015217
│ per-tensor FP32 scale = tensor amax 0.25 / (6 × 448) = 9.301e-05
│ block scale in FP8 units = 163.61 → rounded to E4M3 = 160  (FP8 has 3 mantissa bits)
│ value    ÷ scale  E2M1 code  decoded  abs error
│ ───────  ───────  ─────────  ───────  ─────────
│ -0.0051  -0.343   -0.5       -0.0074  0.0023
│ +0.0102  +0.685   +0.5       +0.0074  0.0028
│ -0.0186  -1.250   -1         -0.0149  0.0037
│ +0.0913  +6.135   +6         +0.0893  0.0020
│ +0.0222  +1.492   +1.5       +0.0223  0.0001
…
▣ STEP 3 · 65,536 weight-like numbers, five formats
│ format                     bits/value  rel. err (gauss)  rel. err (outliers)  SQNR gauss  SQNR outliers
│ ─────────────────────────  ──────────  ────────────────  ───────────────────  ──────────  ─────────────
│ FP8 E4M3 (per-tensor)      8.000        2.68 %            2.69 %               31.4 dB     31.4 dB
│ NVFP4 (16-blk FP8 scale)   4.500        9.51 %            9.43 %               20.4 dB     20.5 dB
│ MXFP4 (32-blk 2^k scale)   4.250       11.44 %           14.62 %               18.8 dB     16.7 dB
│ INT4 group-128 (FP16 sc.)  4.125       11.79 %           26.48 %               18.6 dB     11.5 dB
│ INT4 per-tensor            4.000       20.17 %           76.27 %               13.9 dB      2.4 dB

▣ STEP 4 · what 4.5 bits buys: an 8B model's weights
│ BF16 (no quantization)      16.1 GB  ████████████████████████████
│ FP8 E4M3 (per-tensor)        8.0 GB  ██████████████░░░░░░░░░░░░░░
│ NVFP4 (16-blk FP8 scale)     4.5 GB  ████████░░░░░░░░░░░░░░░░░░░░
│ BF16 ÷ NVFP4 = 3.56×   FP8 ÷ NVFP4 = 1.78×   (the playbook says ~3.5× and ~1.8×)
```

Read the "outliers" column. The test data is the same Gaussian with 0.5 % of values multiplied by 12, which is roughly what real weight rows look like. NVFP4's error does not move. INT4 with one scale per 128 values loses 7 dB, and a single INT4 scale for the whole tensor collapses. SQNR (signal-to-quantization-noise ratio) is a layer-level number: about 6 dB is one bit of precision. It is not model accuracy. Section 4 measures the model.

> 💡 The playbook adds that *"Blackwell Tensor Cores support mixed-precision execution across FP16, FP8, and FP4, so models can use FP4 for weights and activations while accumulating in higher precision"*. That is the 1 PFLOP FP4 figure from Module 01: the GB10 multiplies FP4 numbers natively, so NVFP4 saves memory *and* compute.

✓ Checkpoint: you can explain where the 0.5 extra bit comes from, and why NVFP4's error stays flat when outliers appear while INT4 per-tensor's does not.

## 2 · Quantize with Model Optimizer on your Spark

The playbook quantizes **`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`** inside NVIDIA's TensorRT-LLM container. The container clones **NVIDIA Model Optimizer** at tag `0.35.0` and runs its post-training quantization (PTQ) script. The script calibrates on sample text to measure each layer's value ranges, picks the scales, and exports a Hugging Face checkpoint.

| Setting | DGX Spark value (playbook, Step 4) |
|---|---|
| Container | `nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev` |
| GPU flag | `--gpus all` |
| Model Optimizer | `NVIDIA/Model-Optimizer` @ `0.35.0` |
| Output folder | `./output_models/saved_models_DeepSeek-R1-Distill-Llama-8B_nvfp4_hf/` |

This is the playbook's Step 5 command for DGX Spark, exactly as published. Run it from an empty working directory:

```bash
# on: spark
mkdir -p ~/w25/nvfp4/output_models && cd ~/w25/nvfp4
docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "./output_models:/workspace/output_models" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN=$HF_TOKEN \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c "
    git clone -b 0.35.0 --single-branch https://github.com/NVIDIA/Model-Optimizer.git /app/Model-Optimizer && \
    cd /app/Model-Optimizer && pip install -e '.[dev]' && \
    export ROOT_SAVE_PATH='/workspace/output_models' && \
    /app/Model-Optimizer/examples/llm_ptq/scripts/huggingface_example.sh \
    --model 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B' \
    --quant nvfp4 \
    --tp 1 \
    --export_fmt hf
  "
```

It takes 45–90 minutes (the playbook's estimate), longer than the Lab Runner lets a foreground lab live (900 s). So lab 02 starts the same command in the background. It makes three course changes, printed in the lab: `nohup … > ~/w25/logs/nvfp4_quant.log &` around the command, no `-it` (a background job has no terminal), and `-e HF_TOKEN` without `=$HF_TOKEN`, which passes the variable only if it is set. This model is public. For gated models, the token that `hf auth login` stored in `~/.cache/huggingface` reaches the container through the mounted cache.

```bash
# on: laptop
.venv/bin/python week25/07_nvfp4_speculative/labs/lab02_quantize_nvfp4.py --start   # launch (LIVE only)
.venv/bin/python week25/07_nvfp4_speculative/labs/lab02_quantize_nvfp4.py           # status, any time
```

**Expected output** (DRY mode, captured on this Mac. The outputs are EXAMPLE shapes, not a Spark's)

```
▣ STEP 2 · the quantization job (the playbook's Step 5, run in the background)
$ mkdir -p ~/w25/nvfp4/output_models ~/w25/logs && cd ~/w25/nvfp4 && \   [DRY]
  nohup docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  …
  echo "started quantization job (pid $!) → ~/w25/logs/nvfp4_quant.log"
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
started quantization job (pid <pid>) → ~/w25/logs/nvfp4_quant.log

▣ STEP 3 · status — the log tail and the exported checkpoint
$ ls ~/w25/nvfp4/output_models/saved_models_DeepSeek-R1-Distill-Llama-8B_nvfp4_hf/ 2>/dev/null || echo 'not exported yet'   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
config.json
generation_config.json
hf_quant_config.json
model-00001-of-00002.safetensors
…
```

The playbook's notes on this step, worth knowing before you panic: *"You can safely ignore `No module named 'mpi4py'`"* and *"`pynvml.NVMLError_NotSupported: Not Supported` can appear … and does not affect results."* Validate with the playbook's Step 7:

```bash
# on: spark
cd ~/w25/nvfp4
ls -la ./output_models/
find ./output_models/ \( -name "*.bin" -o -name "*.safetensors" -o -name "*.json" -o -name "*.jinja" \)
```

`hf_quant_config.json` is the file that tells a serving engine the checkpoint was produced by Model Optimizer and how it is quantized. Lab 02 prints it.

> 💡 **The playbook's other variant.** The repository also keeps a twin playbook (`nvidia/nvfp4-quantization/`) that quantizes the MoE model `Qwen/Qwen3.6-35B-A3B` inside the NGC **vLLM** container with Model Optimizer **recipes** (`hf_ptq.py --recipe …`). It offers two: `w4a16_nvfp4-fp8_attn-kv_fp8_cast` (NVFP4 *weights* only, activations stay 16-bit, *"recommended on DGX Spark"* for interactive use) and `nvfp4_experts_only-kv_fp8_cast` (W4A4: weights *and* activations in NVFP4, for *"higher-concurrency, compute-bound serving"*). Same format, two trade-offs: at one user a Spark is memory-bound, so 4-bit weights are what count.

✓ Checkpoint: `ls` on the output folder shows `.safetensors` files and `hf_quant_config.json`, or in DRY mode you can say which three things lab 02 changes about the playbook command and why.

## 3 · Serve the NVFP4 checkpoint

The playbook serves the result with **`trtllm-serve`** from the same container. This is its Step 8 (DGX Spark) command:

```bash
# on: spark
export MODEL_PATH="$HOME/w25/nvfp4/output_models/saved_models_DeepSeek-R1-Distill-Llama-8B_nvfp4_hf/"
docker run \
  -e HF_TOKEN=$HF_TOKEN \
  -v "$MODEL_PATH:/workspace/model" \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  trtllm-serve /workspace/model \
    --backend pytorch \
    --max_batch_size 4 \
    --port 8000
```

**Course change: port 8355, not 8000.** Port 8000 belongs to vLLM from Module 05. The TensorRT-LLM playbook uses 8355, and so does `sparkkit.PORTS["trtllm"]`. Lab 03 runs the command above with `--port 8355`, a container name (`--name w25-trtllm`, so `--stop` can find it), and `nohup` with a log. Then test it with the playbook's request, on the course port:

```bash
# on: spark
curl -X POST http://localhost:8355/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [{"role": "user", "content": "What is artificial intelligence?"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false
  }'
```

Or from the Lab Runner (it goes to `trtllm` on Spark A, :8355; without a Spark, the laptop stand-in answers and says so):

```spark
{"target": "trtllm", "which": "a", "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "messages": [{"role": "user", "content": "In two sentences: what does NVFP4 store for every block of 16 weights?"}], "max_tokens": 150}
```

DeepSeek-R1-Distill is a reasoning model: it writes a `<think>` section before answering, so give it room (`max_tokens` of several hundred) when you want a full answer.

Lab 03 starts the server and times one request:

```bash
# on: laptop
.venv/bin/python week25/07_nvfp4_speculative/labs/lab03_serve_and_measure.py --start nvfp4
.venv/bin/python week25/07_nvfp4_speculative/labs/lab03_serve_and_measure.py --label nvfp4
```

**Expected output** (captured on this Mac with no Spark: the timing is Ollama on THIS laptop, labelled, and says nothing about a Spark)

```
▣ STEP 2 · the endpoint — GET /v1/models on :8355
│ (no Spark host) → ○ down
◆ No TensorRT-LLM on the Spark, so the measurement below uses Ollama on THIS laptop (gemma3:4b). It shows how the measurement works; its speed says nothing about a Spark.

▣ STEP 3 · one timed request — the speculative-decoding playbook's prompt, temperature 0
→ POST http://localhost:11434/v1/completions · model=gemma3:4b · max_tokens=120 · LAPTOP STAND-IN (not Spark numbers)
· ANSWER  Here's how to solve the problem step-by-step:
          **Step 1: Calculate the initial speed**
          …
◆ LAPTOP STAND-IN (not Spark numbers) · gemma3:4b · 120 tokens in 12.5 s · 9.6 tok/s (end to end, incl. prefill)

▣ STEP 4 · your measurements so far (grouped by where they ran — never compare across groups)
│ where   label            model      tokens  seconds  tok/s  when
│ ──────  ───────────────  ─────────  ──────  ───────  ─────  ────────────────
│ laptop  laptop-stand-in  gemma3:4b  120     12.5     9.6    2026-09-29 11:37
```

On a Spark, compare your measured tok/s with Module 01's ceiling: an 8B model at 4.5 bits reads ~4.5 GB per token, so 273 GB/s ÷ 4.5 GB ≈ 60 tok/s is the upper bound for one stream. Your number will be below it. The gap is overhead, and embeddings that stay 16-bit.

✓ Checkpoint: `curl http://localhost:8355/v1/models` on the Spark lists the model, and lab 03 prints a LIVE tok/s. In DRY mode, lab 03 prints a clearly labelled laptop stand-in number instead.

## 4 · Measure it: size on disk, and quality

A quantized model is only useful if it is smaller **and** still right. Measure both.

**Size.** Lab 02's Step 4 runs `du -sh` on the NVFP4 folder and on the BF16 original in the Hugging Face cache, and prints the arithmetic next to them:

```
│ arithmetic: BF16 weights 16.1 GB · NVFP4 weights 4.5 GB (every weight at 4.5 bits)
│ the embeddings (128,256 × 4,096 ≈ 0.53 B params) and lm_head usually stay BF16, so expect the real checkpoint above 4.5 GB
```

(Printed by lab 02 on this Mac: arithmetic, not a measurement.) Your measured ratio will be below 3.56×, because quantization tools usually keep a few sensitive layers (the output head, often the embeddings) in 16 bits. `hf_quant_config.json` lists what was excluded.

**Quality.** Lab 01's SQNR is per layer. The model can still go wrong. The playbook is blunt: *"Quantization can change model quality. Run evaluations for your use case before deploying."* Lab 03's `--quality` mode is the smallest honest version: six questions, each with one checkable answer, at temperature 0. Run it against the NVFP4 server, then against the BF16 original served the same way (a course variant: the Hugging Face id in place of the exported folder):

```bash
# on: laptop
.venv/bin/python week25/07_nvfp4_speculative/labs/lab03_serve_and_measure.py --quality --label nvfp4
.venv/bin/python week25/07_nvfp4_speculative/labs/lab03_serve_and_measure.py --start bf16
.venv/bin/python week25/07_nvfp4_speculative/labs/lab03_serve_and_measure.py --quality --label bf16
```

**Expected output** (captured on this Mac with no Spark: the LAPTOP STAND-IN runs the same six questions on gemma3:4b, to show the harness)

```
▣ STEP 3 · quality — six questions with one right answer each, temperature 0
│    question                                              answer (thinking removed)
│ ─  ────────────────────────────────────────────────────  ─────────────────────────
│ ✓  What is 17 × 23? Answer with the number only.         391
│ ✓  What is the capital of Australia? Answer in one word  Canberra
│ ✓  Which planet is known as the Red Planet? Answer in o  Mars
│ ✓  What is the chemical symbol for gold? Answer with th  Au
│ ✓  A train travels 180 km in 3 hours. What is its speed  60
│ ✕  Spell the word 'spark' backwards. Answer with the wo  krasp
◆ LAPTOP STAND-IN (not the Spark's model) · gemma3:4b · score 5/6
```

(A 4B model misspelling "kraps" is a real result from this Mac. Letter-level tasks are hard for any tokenised model.) Six questions will catch a **broken** quantization (wrong scales, a layer that should have been excluded). They will not detect a 1 % accuracy drop. For that, run a real evaluation suite on your own task. Modules 13 and 20 build one.

| What you compare | Tool | Catches |
|---|---|---|
| per-layer error | lab 01 (SQNR) | a bad format choice |
| bytes on disk | lab 02 (`du -sh`) | layers left unquantized |
| six fixed questions | lab 03 `--quality` | a broken checkpoint |
| your task's eval set | Module 13 | the accuracy change that matters to you |

✓ Checkpoint: you have a `du -sh` ratio and two `--quality` scores (nvfp4, bf16) from the same Spark, or you can explain why the laptop stand-in score is not a result about NVFP4.

## 5 · Speculative decoding: why guessing ahead is free on a Spark

The speculative-decoding playbook's summary: *"a small, fast draft path … propose[s] several tokens ahead, then … the larger target model verif[ies] or correct[s] them in parallel."* Why would checking five tokens cost less than generating five?

Because single-stream decode on a Spark is **memory-bound**. One forward pass streams every weight from memory once, whatever the number of tokens it scores. Lab 04 puts numbers on it for Llama 3.3 70B at NVFP4:

```bash
# on: laptop
.venv/bin/python week25/07_nvfp4_speculative/labs/lab04_speculative_calculator.py
```

**Expected output** (captured on this Mac: arithmetic and a seeded simulation, the same everywhere)

```
▣ STEP 1 · why checking 6 tokens costs about the same as generating 1 (Llama 3.3 70B, NVFP4, one Spark)
│ tokens in one pass  weights read  time to read  math          time at peak  bound by
│ ──────────────────  ────────────  ────────────  ────────────  ────────────  ────────
│ 1 token             39.7 GB        145.5 ms       0.14 TFLOP    0.14 ms     memory
│ 6 tokens            39.7 GB        145.5 ms       0.85 TFLOP    0.85 ms     memory
│ 32 tokens           39.7 GB        145.5 ms       4.52 TFLOP    4.52 ms     memory

▣ STEP 2 · tokens per target step: E = (1 − α^(k+1)) / (1 − α)
│ acceptance  k=1   k=2   k=3   k=4   k=5   k=8
│ ──────────  ────  ────  ────  ────  ────  ────
│ α = 0.5     1.50  1.75  1.88  1.94  1.97  2.00
│ α = 0.6     1.60  1.96  2.18  2.31  2.38  2.47
│ α = 0.7     1.70  2.19  2.53  2.77  2.94  3.20
│ α = 0.8     1.80  2.44  2.95  3.36  3.69  4.33
│ α = 0.9     1.90  2.71  3.44  4.10  4.69  6.13

▣ STEP 3 · check the formula: simulate 20,000 steps per cell (seed 25)
│ case       formula  simulated  difference
│ ─────────  ───────  ─────────  ──────────
│ α=0.6 k=3  2.176    2.168      0.37 %
│ α=0.7 k=4  2.773    2.771      0.06 %
│ α=0.8 k=5  3.689    3.708      0.50 %
│ α=0.9 k=5  4.686    4.703      0.36 %
```

Where the formula comes from. The draft proposes **k** tokens. The target checks them left to right and accepts each with probability **α** (the *acceptance rate*), stopping at the first miss. Then it always adds one token of its own: the correction, or a bonus token if all k passed. The expected number of tokens per target pass is

```text
E = 1 + α + α² + … + α^k = (1 − α^(k+1)) / (1 − α)
```

This model assumes each token's acceptance is independent, which real text only roughly follows. It still predicts well enough to choose `k`.

**The draft is not free.** Each drafted token costs a draft pass. If one draft pass costs **c** target passes, the speed-up is `E ÷ (1 + k·c)`:

```
│ Draft-Target: 8B FP4 drafts for 70B FP4 · c = 0.113 · α = 0.7 · playbook max_draft_len = 4
│   k=2  E=2.19  speed-up 1.79×  ██████████████░░░░░░░░░░░░░░
│   k=4  E=2.77  speed-up 1.91×  ███████████████░░░░░░░░░░░░░  ← best · playbook
│   k=8  E=3.20  speed-up 1.68×  █████████████░░░░░░░░░░░░░░░

│ EAGLE-3 head on gpt-oss-120b (assumed c) · c = 0.050 · α = 0.7 · playbook max_draft_len = 5
│   k=5  E=2.94  speed-up 2.35×  ███████████████████░░░░░░░░░  ← playbook
│   k=6  E=3.06  speed-up 2.35×  ███████████████████░░░░░░░░░  ← best
```

Draft-Target's `c` is arithmetic (the 8B draft reads 8 ÷ 70.6 of the target's bytes). EAGLE-3's `c = 0.05` is an **assumption** for illustration. Change it with `--cost`, and the acceptance rate with `--alpha`.

| | EAGLE-3 | Draft-Target |
|---|---|---|
| Draft | a small head trained on the target's own hidden states | a separate small model with the same tokenizer |
| Playbook pair | `openai/gpt-oss-120b` + `nvidia/gpt-oss-120b-Eagle3-long-context` | `nvidia/Llama-3.3-70B-Instruct-FP4` + `nvidia/Llama-3.1-8B-Instruct-FP4` |
| `max_draft_len` | 5 | 4 |
| Playbook's pitch | *"features fused from multiple layers improve draft-token acceptance"* | *"an 8B draft model accelerates a 70B target model"* |

> 💡 Speculation helps most at **batch 1**, which is why both playbook configs set `max_batch_size: 1`. With many users, the batch already fills the idle compute that speculation would use.

✓ Checkpoint: from the Step 2 table you can say how many tokens per pass α = 0.8, k = 4 gives (3.36), and why raising k past ~6 barely helps at α = 0.7.

## 6 · Run EAGLE-3 and Draft-Target on your Spark

Both options use `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc12` and a TensorRT-LLM `extra-llm-api-config.yml` with a `speculative_config` block. The playbook: *"Run one option at a time on a free port."* This is Option 1, verbatim:

```bash
# on: spark
export TRTLLM_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc12"
docker run \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface/:/root/.cache/huggingface/" \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  "$TRTLLM_IMAGE" \
  bash -c '
    hf download openai/gpt-oss-120b && \
    hf download nvidia/gpt-oss-120b-Eagle3-long-context \
        --local-dir /opt/gpt-oss-120b-Eagle3/ && \
    cat > /tmp/extra-llm-api-config.yml <<EOF
enable_attention_dp: false
disable_overlap_scheduler: false
enable_autotuner: false
cuda_graph_config:
    max_batch_size: 1
speculative_config:
    decoding_type: Eagle
    max_draft_len: 5
    speculative_model_dir: /opt/gpt-oss-120b-Eagle3/

kv_cache_config:
    free_gpu_memory_fraction: 0.9
    enable_block_reuse: false
EOF
    export TIKTOKEN_ENCODINGS_BASE="/tmp/harmony-reqs" && \
    mkdir -p $TIKTOKEN_ENCODINGS_BASE && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
    trtllm-serve openai/gpt-oss-120b \
      --backend pytorch --tp_size 1 \
      --max_batch_size 1 \
      --extra_llm_api_options /tmp/extra-llm-api-config.yml'
```

The playbook's command has no `--port`, so it serves on `trtllm-serve`'s default, **8000**, and the playbook tests `http://localhost:8000/v1/completions`. Lab 03 appends `--port 8355` (the course's TensorRT-LLM port) and runs it in the background. Option 2 (Draft-Target) is in lab 03 verbatim, with the same changes. To see a speed-up you need a baseline. Lab 03's `baseline` server is a course variant: **the same command with the `speculative_config` block and the draft download removed**, so only speculation differs.

```bash
# on: laptop
.venv/bin/python week25/07_nvfp4_speculative/labs/lab03_serve_and_measure.py --start baseline
.venv/bin/python week25/07_nvfp4_speculative/labs/lab03_serve_and_measure.py --label baseline
.venv/bin/python week25/07_nvfp4_speculative/labs/lab03_serve_and_measure.py --start eagle3
.venv/bin/python week25/07_nvfp4_speculative/labs/lab03_serve_and_measure.py --label eagle3
.venv/bin/python week25/07_nvfp4_speculative/labs/lab03_serve_and_measure.py --stop
```

Each measurement sends the playbook's own EAGLE-3 test prompt (the train word problem) with `max_tokens: 300` at temperature 0, and saves tok/s under its label. Divide eagle3 by baseline for your speed-up, then let lab 04 back out the acceptance rate it implies:

```bash
# on: laptop
.venv/bin/python week25/07_nvfp4_speculative/labs/lab04_speculative_calculator.py --measured 1.8
```

**Expected output** (captured on this Mac for an example input of 1.8×, not a Spark measurement: put in your own)

```
▣ STEP 6 · back out α from YOUR measured speed-up (1.8× = eagle3 tok/s ÷ baseline tok/s, from lab 03)
│ Draft-Target: 8B FP4 drafts for 70B FP4      k=4  α ≈ 0.67
│ EAGLE-3 head on gpt-oss-120b (assumed c)     k=5  α ≈ 0.57
```

The playbook's next steps: *"Experiment with different `max_draft_len` values (1, 2, 3, 4, 8)"* and *"Monitor token acceptance rates and throughput improvements."* Edit `max_draft_len` in lab 03's heredoc to try them.

**Two Sparks.** Models too big for one Spark, such as `nvidia/Qwen3-235B-A22B-FP4`, run with tensor parallelism (`TP_SIZE=2`) plus an EAGLE-3 head (`nvidia/Qwen3-235B-A22B-Eagle3`, `max_draft_len: 3`) on port **8355**, launched with `mpirun` across both Sparks. The playbook's *Multi-node serving* tab has every step. It builds on Module 02's cabling and passwordless SSH. It also recommends *NVIDIA Sync Cluster Assistant* for that setup.

✓ Checkpoint: you have two tok/s numbers from the same Spark (baseline and eagle3) and a speed-up. Or, in DRY mode, you can predict the speed-up for α = 0.8 with the playbook's `max_draft_len: 5`.

## Labs — run them here

**labs/lab01_nvfp4_simulator.py** — Quantize 65,536 weight-like numbers to FP8, NVFP4, MXFP4 and INT4 in pure Python and measure the real error of each.

**labs/lab02_quantize_nvfp4.py** — Run the playbook's Model Optimizer NVFP4 quantization on your Spark in the background (`--start`), then check its log, files, `hf_quant_config.json` and size.

**labs/lab03_serve_and_measure.py** — Start `trtllm-serve` for NVFP4, BF16, EAGLE-3, Draft-Target or a baseline (`--start …`), then time a request or score six questions (`--quality`).

**labs/lab04_speculative_calculator.py** — Acceptance rate and draft length to tokens per step and speed-up, checked by simulation, plus α backed out of a measured speed-up.

Labs 01 and 04 run on the laptop. Labs 02 and 03 run LIVE on your Spark or DRY. Lab 03 falls back to a labelled laptop stand-in for its request.

## Try it yourself

**Exercise 07 — a 4-bit quantizer and a draft-length planner.** Open `week25/07_nvfp4_speculative/exercises/ex07_fp4_and_speculation.py`. It has four `TODO`s:

1. `e2m1_round(x)`: the nearest E2M1 value, with sign, saturating at ±6.
2. `quantize_block(block)`: scale = max |value| ÷ 6, then the decoded values.
3. `expected_tokens(alpha, k)`: the formula from Section 5.
4. `best_draft_len(alpha, c, kmax)`: the k with the best `E ÷ (1 + k·c)`.

```bash
# on: laptop
.venv/bin/python week25/07_nvfp4_speculative/exercises/ex07_fp4_and_speculation.py
```

**Expected output** (once all four TODOs are done. Captured on this Mac from the reference solution)

```
✓ e2m1_round: 2.6→3 · -0.3→-0.5 · 100→6 · 0.1→0 · -4.9→-4 · 5.2→6 · 1.2→1
✓ quantize_block: scale 0.06/6 = 0.01 → [0.01, -0.03, 0.005, 0.06, -0.02, 0, 0.03, -0.06]
✓ expected_tokens: α=0.8 k=4 → 3.36 · α=0.5 k=1 → 1.5 · α=1 k=5 → 6
✓ best_draft_len: α=0.7 c=0.113 → 4 · free drafts → kmax · weak draft → 1

▣ your quantizer, applied to 4,096 weight-like numbers (seed 7, 0.5 % outliers)
│ one scale per    16 values   SQNR  20.7 dB
│ one scale per    64 values   SQNR  17.4 dB
│ one scale per  4096 values   SQNR   6.7 dB

▣ your planner, applied (Draft-Target c = 0.113, from lab 04)
│ α = 0.6  best k = 3  E = 2.18 tokens per target pass
│ α = 0.7  best k = 4  E = 2.77 tokens per target pass
│ α = 0.8  best k = 5  E = 3.69 tokens per target pass
│ α = 0.9  best k = 9  E = 6.51 tokens per target pass
```

<details><summary>Hint — e2m1_round in one line</summary>

Take the magnitude, find the grid value closest to it (`min(E2M1_GRID, key=lambda g: abs(abs(x) - g))`), then put the sign back with `math.copysign`. Anything above 5 is closer to 6 than to 4, so saturation comes for free.

</details>

<details><summary>Hint — why does a block of 16 beat one scale for everything?</summary>

The scale is set by the block's largest value. With one scale for 4,096 values, a single outlier pushes every small value down to 0 or ±0.5. With 16, only its 15 neighbours pay.

</details>

<details><summary>Stretch — round the scale to FP8</summary>

Real NVFP4 stores the block scale as FP8 E4M3. Copy `minifloat_round` from lab 01 into your exercise and round `scale` (in units of a per-tensor scale) before using it. How much SQNR does it cost? Lab 01's NVFP4 row (20.5 dB) is the answer to check against.

</details>

✓ Checkpoint: all four checker lines are ✓.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named 'mpi4py'` or `pynvml.NVMLError_NotSupported` during quantization | Expected on DGX Spark, per the playbook. Check that the artifacts exist under `./output_models` |
| Quantization container exits with CUDA out of memory | The playbook: quantize a smaller model, or free memory. On unified memory also flush the page cache: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| `Cannot access gated repo` | Request access to the model on Hugging Face, then `hf auth login` on the Spark |
| `permission denied` removing `./output_models` | The container wrote files as root. The playbook: retry with `sudo rm -rf ./output_models` |
| `the input device is not a TTY` | You ran a playbook command with `-it` from a script or `nohup`. Drop `-it` (lab 02/03 already do) |
| Server does not respond on :8355 | Still loading, or it was started with the playbook's default port 8000. Check `tail ~/w25/logs/w25-trtllm.log` and `docker ps` |
| `bind: address already in use` | Another server holds the port (vLLM on 8000, or a previous `w25-trtllm`). Run lab 03 `--stop` first. The playbook: *"Run one option at a time"* |
| Speculative decoding is slower than the baseline | Low acceptance (creative prompts), or `max_batch_size` above 1. Lower `max_draft_len` and use lab 04 to see the break-even α |

## Next

Continue to [Lab 08 — LiteLLM: one gateway for every engine and both Sparks](../08_litellm_gateway/TUTORIAL.md): put one authenticated OpenAI endpoint in front of Ollama, vLLM, SGLang and the TensorRT-LLM server you just started, with fallbacks when a Spark is down.
