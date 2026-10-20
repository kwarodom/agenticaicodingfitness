# ▶ Spark Lab 05 — vLLM: high-throughput serving, tool calling, and two-Spark tensor parallel

> Part of Week 25 · DGX Spark: fine-tune, serve, and build sandboxed agents. You type the commands, you see the real output. Every lab also runs in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's playbook, or a clearly marked EXAMPLE.

**What you'll actually do**
- Learn why vLLM exists: PagedAttention, continuous batching, and an OpenAI-compatible API.
- Pick the right vLLM container for the job (upstream `vllm/vllm-openai`, or NGC `nvcr.io/nvidia/vllm`) and start your first server on port 8000.
- Size the three memory flags that matter on 128 GB of unified memory: `--gpu-memory-utilization`, `--max-model-len`, `--max-num-seqs`.
- Serve the playbook's **agent-ready** Qwen3.6-35B-A3B model and run a full tool-calling round trip.
- Measure continuous batching: total tok/s vs per-stream tok/s at 1, 2, 4 and 8 parallel requests.
- Run one 70B model across **two Sparks** with Ray and tensor parallelism, then (optional) NVIDIA's **Nemotron 3 Super 120B** in FP8, and serve a LoRA adapter next to its base model.

**Time** ~55 min · **Difficulty** intermediate · **Hardware** 1 Spark (2 Sparks for Section 6), or none: DRY mode + laptop stand-in

**Official playbooks covered:** [vLLM](https://build.nvidia.com/spark/vllm) (the current `playbook-vllm` and the older `vllm` version, which has the two-Spark 405B and agent-ready Qwen3.6 sections)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Module 01 done | `ssh -o BatchMode=yes spark-a true` | labs send commands to the Spark over SSH |
| Docker without sudo on the Spark | `docker ps` on the Spark | every vLLM path in the playbook is a container |
| ~60 GB free disk on the Spark | `df -h /` | the container image plus one or two models |
| A Hugging Face login **on the Spark** | `hf auth login` on the Spark, once | gated models (Llama) download with the Spark's own token |
| Optional: Ollama on your laptop | `curl -s localhost:11434/v1/models` | the laptop stand-in for the HTTP labs |

```bash
# on: spark
docker ps
hf auth whoami
df -h / | tail -1
```

> 🔐 The playbook passes `-e HF_TOKEN="$HF_TOKEN"` to the container. This course mounts the whole `~/.cache/huggingface` instead, so the token saved by `hf auth login` on the Spark is found and never typed into a command line. Both work.

✓ Checkpoint: `docker ps` runs without `sudo` on the Spark, and `hf auth whoami` prints your Hugging Face user.

## 1 · Why vLLM: PagedAttention and continuous batching

Ollama (Module 03) and llama.cpp (Module 04) are good for one person at a desk. vLLM is built for **many requests at once**. The playbook lists three ideas:

| Idea | What it does | Why you care on a Spark |
|---|---|---|
| **PagedAttention** | stores the KV cache in small pages, like virtual memory, instead of one big block per request | no memory is wasted on context a request never uses, so more users fit in 128 GB |
| **Continuous batching** | adds new requests to the batch that is already running, token by token | one read of the weights from memory serves every user in the batch (Section 5) |
| **OpenAI-compatible API** | `/v1/models`, `/v1/chat/completions`, `/v1/completions` | curl, the `openai` SDK, LiteLLM (Module 08) and NAT (Module 14) work unchanged |

Module 01 showed that one conversation's speed is capped by memory bandwidth: every token reads every active weight once. Continuous batching does not raise that cap for one user. It shares the same weight read across many users, so the **total** tokens per second grows with concurrency.

```text
  requests ──►  ┌─────────── vLLM (one container, port 8000) ────────────┐
  (any client)  │  scheduler: joins new requests into the running batch  │
                │  ┌──────────────┐   ┌───────────────────────────────┐  │
                │  │ model weights│   │ KV cache pages (PagedAttention)│  │
                │  └──────────────┘   └───────────────────────────────┘  │
                │     both live inside --gpu-memory-utilization × 128 GB │
                └────────────────────────────────────────────────────────┘
```

✓ Checkpoint: you can say which of the three ideas lets vLLM serve more users per Spark, and which one lets your existing OpenAI client talk to it.

## 2 · Pick the container and start your first server

The playbooks use several vLLM images. They are not interchangeable. Use the one the recipe names:

| Image (exact tag from the playbooks) | Used for |
|---|---|
| `vllm/vllm-openai:latest` | the current playbook's base configuration and the agent-ready Qwen3.6 recipe (single Spark) |
| `nvcr.io/nvidia/vllm:26.05-py3` | NVIDIA's NGC build, used for **two Sparks** (Ray + tensor parallel, Section 6) |
| `nvcr.io/nvidia/vllm:26.02-py3` | four or more Sparks through a QSFP switch (do not mix with the two-node tag) |
| `vllm/vllm-openai:gemma4-cu130` | the Gemma 4 family |
| `vllm/vllm-openai:v0.20.0` · `vllm/vllm-openai:cu130-nightly` | Nemotron Nano · Nemotron Super (Module 06) |

> 💡 The older playbook tells you to find the latest NGC build at <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm> (its example is `26.05.post1-py3`). The current playbook points to [vLLM Recipes for DGX Spark](https://recipes.vllm.ai/browse?panel=open&hw=dgx_spark_gb10), which lists a tested image and command per model.

This is the playbook's **base configuration**. The model is `nvidia/Llama-3.1-8B-Instruct-FP8` from the playbook's support matrix, and the context is 32K instead of the playbook's 131072 (Section 3 explains the trade):

```bash
# on: spark
docker pull vllm/vllm-openai:latest
docker run -d \
  --name vllm-server \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --entrypoint "" \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai:latest \
  vllm serve nvidia/Llama-3.1-8B-Instruct-FP8 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.8
```

Watch it start. The first run downloads the model, then loads it into memory:

```bash
# on: spark
docker logs -f vllm-server
```

**Expected output** (REFERENCE — quoted from the playbook: the lines that mean "ready")

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Or wait for the health endpoint, as the playbook does, then send its test request:

```bash
# on: spark
timeout 900 bash -c 'until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 10; done' \
  || { echo "Server failed to start within 900s"; docker logs vllm-server | tail -50; exit 1; }
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "nvidia/Llama-3.1-8B-Instruct-FP8", "messages": [{"role": "user", "content": "12*17"}], "max_tokens": 500}'
```

The playbook says the response should contain `"content": "204"` or a similar calculation. Try it from here. The block below goes to vLLM on your Spark, or to the labelled laptop stand-in:

```spark
{"target": "vllm", "which": "a", "model": "nvidia/Llama-3.1-8B-Instruct-FP8",
 "messages": [{"role": "user", "content": "12*17"}], "max_tokens": 150}
```

Lab 05-2 does all of this in one go. It only starts the container when you pass `--yes`:

```bash
# on: laptop
.venv/bin/python week25/05_vllm/labs/lab02_first_server.py          # add --yes to really start it
```

**Expected output** (captured on this Mac, DRY mode: no Spark configured, so step 4 used the laptop stand-in)

```
▣ STEP 3 · wait for /health (model loading can take several minutes)
$ timeout 600 bash -c 'until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 10; done'; docker logs vllm-server 2>&1 | grep -E 'Application startup complete|Uvicorn running' | tail -2   [DRY]
◈ REFERENCE — expected output from the NVIDIA playbook (not your machine):
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

▣ STEP 4 · the OpenAI API: GET /v1/models, then POST /v1/chat/completions
◆ no vLLM answering at (no Spark host) — the chat below falls back to the laptop stand-in
→ POST http://localhost:11434/v1/chat/completions · model=nemotron-3-nano:latest · Ollama on THIS laptop (stand-in, not the Spark)
· ANSWER  The product of 12 and 17 is **204**.
◆ LAPTOP STAND-IN (not Spark numbers) · nemotron-3-nano:latest · TTFT 9446 ms · 78 tok in 10.6s · 69.7 tok/s
✓ the answer contains 204, as the playbook expects
```

> ⚠ `-p 8000:8000` listens on every interface, including your tailnet (Module 01, Section 3). vLLM has no password by default. Module 08 puts a LiteLLM gateway with keys in front of it.

Stop it when you are done. Your downloaded model stays in the cache:

```bash
# on: spark
docker rm -f vllm-server 2>/dev/null || true
```

✓ Checkpoint: `curl http://localhost:8000/v1/models` on the Spark lists your model, and the `12*17` request returns 204.

## 3 · The flags that matter on 128 GB of unified memory

At start-up vLLM takes a fixed slice of memory, loads the weights into it, and turns **everything left** into KV cache pages. On a Spark that slice comes out of the same 128 GB the OS and your other processes use. Three flags decide the sum:

| Flag | Playbook value | What it controls |
|---|---|---|
| `--gpu-memory-utilization` | `0.8` (base) · `0.4` (agent-ready Qwen3.6) · `0.90` (Nemotron Super) | the slice vLLM may use for weights + KV cache |
| `--max-model-len` | `131072` (base) · `262144` (Qwen3.6) · `2048` (two-Spark 70B) | the longest prompt + output for one request |
| `--max-num-seqs` | `4` (Qwen3.6, Nemotron Super) · `8` (Nemotron Nano) · `1` (405B) | how many sequences run in one batch |
| `--kv-cache-dtype fp8` | Qwen3.6 and Nemotron Super recipes | halves the KV cache per token |

```text
KV room            = gpu-memory-utilization × 128 GB − weights − runtime (~4 GB, a course assumption)
KV per sequence    = 2 × layers × KV heads × head dim × max-model-len × bytes (2 for bf16, 1 for fp8)
full-length seqs   = KV room ÷ KV per sequence          → a safe --max-num-seqs
```

Lab 05-1 does that sum for models from the playbook's support matrix and prints a ready `docker run`:

```bash
# on: laptop
.venv/bin/python week25/05_vllm/labs/lab01_launch_builder.py
```

**Expected output** (arithmetic, the same on every machine)

```
▣ STEP 1 · what fits in --gpu-memory-utilization 0.8 × 128 GB, --max-model-len 32768, --kv-cache-dtype auto
│ #  model (HF handle)                    format  weights   KV room   KV / seq  fits at full context
│ ─  ───────────────────────────────────  ──────  ────────  ────────  ────────  ────────────────────────────────
│ 1  nvidia/Llama-3.1-8B-Instruct-FP8     fp8       8.0 GB   90.4 GB   4.29 GB    21.0 full-length seqs
│ 2  nvidia/Qwen3-14B-NVFP4               nvfp4     8.3 GB   90.1 GB   5.37 GB    16.8 full-length seqs
│ 3  nvidia/Qwen3-32B-NVFP4               nvfp4    18.4 GB   80.0 GB   8.59 GB     9.3 full-length seqs
│ 4  nvidia/Llama-3.3-70B-Instruct-NVFP4  nvfp4    39.7 GB   58.7 GB  10.74 GB     5.5 full-length seqs
│ 5  meta-llama/Llama-3.3-70B-Instruct    bf16    141.2 GB    0.0 GB  10.74 GB  ✕ weights alone exceed the slice

▣ STEP 2 · the three levers — Qwen3-32B NVFP4 at 32K context
│ util 0.5 · kv auto    4.8 × 32K seqs  █████░░░░░░░░░░░░░░░░░░░░░░░
│ util 0.5 · kv fp8     9.7 × 32K seqs  █████████░░░░░░░░░░░░░░░░░░░
│ util 0.8 · kv auto    9.3 × 32K seqs  █████████░░░░░░░░░░░░░░░░░░░
│ util 0.8 · kv fp8    18.6 × 32K seqs  █████████████████░░░░░░░░░░░
│ util 0.9 · kv auto   10.8 × 32K seqs  ██████████░░░░░░░░░░░░░░░░░░
│ util 0.9 · kv fp8    21.6 × 32K seqs  ████████████████████░░░░░░░░
│ util 0.8 · kv auto · ctx   8K →   37.2 full-length seqs
│ util 0.8 · kv auto · ctx 128K →    2.3 full-length seqs
```

Four lessons:

1. **Context is expensive.** Qwen3-32B at 0.8 fits 37 users at 8K context but only 2 at 128K. The playbook's default `MAX_MODEL_LEN=131072` is generous. Size it to your workload, as the playbook says.
2. **fp8 KV doubles your users** at the same memory. That is why both long-context agent recipes use `--kv-cache-dtype fp8`.
3. **Utilization is shared memory.** Raising `--gpu-memory-utilization` to 0.95, as the playbook suggests for a *dedicated* GPU, takes memory from the OS on a Spark. When the Nemotron playbook hits out-of-memory, its first fix is to *lower* it to 0.70.
4. **A 70B bf16 model does not fit one Spark at all.** Either use the NVFP4 checkpoint (row 4) or split it across two Sparks (Section 6).

> 💡 These are estimates. NVFP4 and FP8 checkpoints keep some layers at higher precision, and vLLM's own start-up log reports the real KV cache size and the maximum concurrency. Trust the log over the lab. Change the inputs with `--pick 4 --ctx 131072 --kv fp8`.

✓ Checkpoint: using lab 05-1, you can pick a `--max-model-len` and `--max-num-seqs` for Llama 3.3 70B NVFP4 that fits at `--gpu-memory-utilization 0.8`.

## 4 · Agent-ready: Qwen3.6-35B-A3B with tool calling

An agent needs three things from a model server: it must return **tool calls** in the OpenAI `tool_calls` field, it must keep **reasoning** apart from the answer, and it must handle **long multi-turn** contexts. The playbook's recommended agent-ready model for DGX Spark is `nvidia/Qwen3.6-35B-A3B-NVFP4`: 35B parameters, about 3B active per token (a Mixture-of-Experts, the Spark's sweet spot from Module 01).

**Where these flags come from (this module is the course's source of truth for them):**

| Source | What it gives |
|---|---|
| Older playbook `nvidia/vllm/README.md`, section "Run Agent Ready Qwen3.6 35B Model with vLLM" | **every flag in the command below**, the image `vllm/vllm-openai:latest`, and the `12*17` test |
| Current playbook `nvidia/playbook-vllm/README.md`, tab "Agent-ready Models" | only the model handle `nvidia/Qwen3.6-35B-A3B-NVFP4` and a link to the [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B?hardware=dgx_spark_gb10&features=tool_calling%2Creasoning) — no flags |
| recipes.vllm.ai | **nothing in this course.** When this module was written (2026-09-29) the recipe page did not show a DGX Spark command that could be quoted. Compare the recipe with the command below before you rely on either |

So the three agent flags, `--reasoning-parser qwen3`, `--tool-call-parser qwen3_xml` and `--enable-auto-tool-choice`, are quoted from the older playbook. Later modules (08, 14, 17, 20) reuse them from here.

This is that playbook's launch command, unchanged. The `vllm/vllm-openai` image's entrypoint is already `vllm serve`, so the model handle and flags are passed straight to the container:

```bash
# on: spark
docker pull vllm/vllm-openai:latest
export HF_TOKEN="your_huggingface_token"      # or rely on `hf auth login` and drop the -e line
docker run -it --gpus all -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --moe-backend marlin \
  --gpu-memory-utilization 0.4 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
  --load-format fastsafetensors \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice
```

The flags in three groups:

| Group | Flags | Why |
|---|---|---|
| Memory | `--gpu-memory-utilization 0.4` · `--max-model-len 262144` · `--max-num-seqs 4` · `--kv-cache-dtype fp8` | a long context for 4 agents at a time. The playbook does not explain the 0.4; one effect is that ~60% of the Spark stays free for other work |
| Speed | `--enable-chunked-prefill` · `--async-scheduling` · `--enable-prefix-caching` · `--speculative-config` (MTP, Module 07) · `--moe-backend marlin` · `--attention-backend flashinfer` | long agent prompts share a prefix turn after turn, so prefix caching saves work |
| Agent | `--reasoning-parser qwen3` · `--tool-call-parser qwen3_xml` · `--enable-auto-tool-choice` | turn the model's own syntax into OpenAI `reasoning` and `tool_calls` fields |

**The parser must match the model family.** From the playbooks: Qwen3.6 → `qwen3_xml`; Nemotron Nano and Super → `qwen3_coder`; Gemma 4 → `gemma4` (with `--reasoning-parser gemma4`). A wrong parser does not crash the server: tool calls come back as plain text in `content`, and your agent silently never runs a tool.

> ⚠ Reasoning models spend part of `max_tokens` thinking before they answer. The playbook's API test uses `max_tokens: 4096`. With a small budget you may see `finish_reason: length`, the thinking text under `reasoning`, and `content: null`.

Try a tool call from here. The runner sends the `tools` list along with the messages:

```spark
{"target": "vllm", "which": "a", "model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
 "messages": [{"role": "user", "content": "Guest in room 1204 says it feels warm. What is the temperature there right now?"}],
 "max_tokens": 150,
 "tools": [{"type": "function", "function": {"name": "get_room_temperature",
   "description": "Current air temperature of a hotel room, in degrees Celsius.",
   "parameters": {"type": "object", "properties": {"room": {"type": "string"}}, "required": ["room"]}}}]}
```

Lab 05-4 runs the whole agent loop once: the model asks for a tool, the lab runs it (a fake sensor, clearly local), sends the result back as a `role: tool` message, and the model answers:

```bash
# on: laptop
.venv/bin/python week25/05_vllm/labs/lab04_tool_calling.py
```

**Expected output** (captured on this Mac: no Spark, so a tool-capable laptop model stood in)

```
▣ STEP 2 · turn 1 — the model decides to call a tool
→ POST http://localhost:11434/v1/chat/completions · model=gemma4:12b · Ollama on THIS laptop (stand-in, not the Spark)
· ANSWER
→ tool_call get_room_temperature({"room":"1204"})
◆ LAPTOP STAND-IN (not Spark numbers) · gemma4:12b · TTFT 13340 ms · 20 tok in 13.3s · 1.5 tok/s
✓ 1 tool call(s) — parsed by the server into OpenAI `tool_calls`, not by this script

▣ STEP 3 · this script runs the tool locally and sends the result back as role=tool
│ get_room_temperature({"room":"1204"}) → {"room": "1204", "celsius": 26.5, "source": "fake sensor in lab04"}

▣ STEP 4 · turn 2 — the model answers from the tool result
→ POST http://localhost:11434/v1/chat/completions · model=gemma4:12b · Ollama on THIS laptop (stand-in, not the Spark)
· ANSWER  The current temperature in room 1204 is 26.5°C.
◆ LAPTOP STAND-IN (not Spark numbers) · gemma4:12b · TTFT 27577 ms · 24 tok in 27.6s · 0.9 tok/s
✓ the final answer quotes the tool's 26.5 °C — grounded in the tool result, not guessed
```

The slow laptop numbers include loading the model and other programs sharing the laptop. They say nothing about the Spark.

✓ Checkpoint: you can name the three flags that make vLLM return `tool_calls`, and lab 05-4 prints a `→ tool_call` line and a final answer quoting 26.5 °C.

## 5 · Continuous batching, measured

Lab 05-3 sends the same kind of request 1, 2, 4 and 8 at a time and measures two numbers:

- **total tok/s**: all tokens generated ÷ wall-clock time. How many users one box can serve.
- **per-stream tok/s**: one user's decode speed. How fast the words appear in *their* chat window.

```bash
# on: laptop
.venv/bin/python week25/05_vllm/labs/lab03_continuous_batching.py
```

**Expected output** (captured on this Mac against Ollama: LAPTOP STAND-IN, not a Spark, and not vLLM)

```
◆ endpoint: LAPTOP STAND-IN · Ollama on this Mac (not Spark numbers) · model gemma3:4b · max_tokens 64

▣ STEP 3 · total vs per-stream throughput
│ parallel  ok   total tok/s  per-stream tok/s  mean TTFT   vs 1   total
│ ────────  ───  ───────────  ────────────────  ──────────  ─────  ────────────────────
│ 1         1/1    90.4         95.4                 37 ms  ×1.00  ████████████████████
│ 2         2/2    80.9         88.4                401 ms  ×0.90  ██████████████████░░
│ 4         4/4    41.4         46.6               2695 ms  ×0.46  █████████░░░░░░░░░░░
│ 8         8/8    47.8         51.7               4661 ms  ×0.53  ███████████░░░░░░░░░
```

Read it honestly. On this laptop the total did **not** grow: from 2 requests on, each user got slower and the time to first token climbed from 37 ms to 4.7 s. The requests waited in a queue instead of sharing one batch. Other programs were using the same Ollama at the time, and a rerun gave different numbers, so the lab warns you when the solo request itself had to wait.

That is the contrast this lab exists to show. Run it against vLLM on your Spark (lab 05-2 first) and compare the **shape** of your curve with this one: with continuous batching, the total column should keep climbing while per-stream speed drops only slowly, until the batch runs out of compute or KV cache (`--max-num-seqs`). Never compare the laptop's numbers with the Spark's.

✓ Checkpoint: you ran lab 05-3 and can explain from your own table whether the server batched the requests or queued them.

## 6 · One model across two Sparks: Ray + tensor parallel

Llama 3.3 70B at bf16 needs 141 GB of weights, more than one Spark. **Tensor parallelism** (TP) splits every layer across GPUs: with `--tensor-parallel-size 2` each Spark holds half the weights and half the KV heads, and the two exchange partial results over the QSFP link on every layer of every token. vLLM runs the two halves as one **Ray** cluster.

```text
  Spark A (head)                      QSFP 200 Gb/s                  Spark B (worker)
 ┌───────────────────────────┐   all-reduce every layer    ┌───────────────────────────┐
 │ container node-NNNN       │◄───────────────────────────►│ container node-NNNN       │
 │ ray head · vllm serve     │                             │ ray worker                │
 │ half of every layer (TP 0)│                             │ half of every layer (TP 1)│
 │ API :8000 · Ray UI :8265  │                             │                           │
 └───────────────────────────┘                             └───────────────────────────┘
```

Lab 05-1's step 3 does the memory sum for the playbook's settings:

```
▣ STEP 3 · two Sparks, tensor parallel 2 — the playbook's Llama 3.3 70B (bf16)
│ setup           weights           vLLM slice         at --max-model-len 2048
│ ──────────────  ────────────────  ─────────────────  ───────────────────────
│ 1 Spark         141.2 GB          102.4 GB           ✕ does not fit
│ 2 Sparks, TP=2  70.6 GB per node  102.4 GB per node  83 × 2K seqs
│ TP=2 · ctx  8K →  20.7 full-length seqs
│ TP=2 · ctx 32K →   5.2 full-length seqs
│ nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 · ctx 256K · kv fp8
│   1 Spark         120.0 GB          ✕ does not fit
│   2 Sparks, TP=2   60.0 GB per node  0.54 GB KV per 256K seq → 71.5 full-length seqs
```

The last three lines are for the optional Nemotron run at the end of this section.

First finish Module 02 (QSFP cable, IPs, passwordless SSH between the Sparks), or run NVIDIA Sync's Cluster Assistant, which the playbook accepts as a replacement. Then follow the playbook on **both Sparks**.

**Step 1 — the cluster script and the NGC image** (on both Sparks). The script is pinned to a known-good vLLM commit and patched to install Ray inside the container, because the `26.05-py3` image ships without it:

```bash
# on: spark
wget https://raw.githubusercontent.com/vllm-project/vllm/51c1ee9b7c8acbba4899a8ebffd390685d171946/examples/ray_serving/run_cluster.sh
sed -i 's|^RAY_START_CMD="ray start|RAY_START_CMD="pip install -q --root-user-action=ignore '\''ray[default]>=2.9'\'' \&\& ray start|' run_cluster.sh
chmod +x run_cluster.sh
docker pull nvcr.io/nvidia/vllm:26.05-py3
```

Repeat the same four commands on Spark B (`ssh spark-b`).

**Step 2 — the Ray head, on Spark A.** Run it inside `tmux`: `run_cluster.sh` stops the container when its terminal closes. `enp1s0f1np1` is the playbook's example QSFP interface. Use the one that shows `(Up)` in `ibdev2netdev` (Module 02):

```bash
# on: spark
tmux new -s ray
export MN_IF_NAME=enp1s0f1np1
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
export VLLM_IMAGE=nvcr.io/nvidia/vllm:26.05-py3
echo "Using interface $MN_IF_NAME with IP $VLLM_HOST_IP"
bash run_cluster.sh $VLLM_IMAGE $VLLM_HOST_IP --head ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR=$VLLM_HOST_IP
```

**Step 3 — the Ray worker, on Spark B.** Replace `<NODE_1_IP_ADDRESS>` with the IP that Spark A just printed:

```bash
# on: spark-b
tmux new -s ray
export MN_IF_NAME=enp1s0f1np1
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
export HEAD_NODE_IP=<NODE_1_IP_ADDRESS>
export VLLM_IMAGE=nvcr.io/nvidia/vllm:26.05-py3
echo "Worker IP: $VLLM_HOST_IP, connecting to head node at: $HEAD_NODE_IP"
bash run_cluster.sh $VLLM_IMAGE $HEAD_NODE_IP --worker ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR=$HEAD_NODE_IP
```

**Step 4 — check the cluster, download, serve** (on Spark A, in a second terminal). The playbook says `ray status` should show 2 nodes with GPU resources. Llama 3.3 70B is gated: accept its license on Hugging Face first:

```bash
# on: spark
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
echo "Found container: $VLLM_CONTAINER"
docker exec $VLLM_CONTAINER ray status
docker exec -it $VLLM_CONTAINER /bin/bash -c '
  hf auth login
  hf download meta-llama/Llama-3.3-70B-Instruct'
docker exec -it $VLLM_CONTAINER /bin/bash -c '
  vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 2 --max-model-len 2048 \
    --gpu-memory-utilization 0.8 --distributed-executor-backend ray'
```

(`--gpu-memory-utilization 0.8` is from the older playbook; the current one leaves it at the default.) When the log says `Application startup complete.`, test it on Spark A with the playbook's request:

```bash
# on: spark
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.3-70B-Instruct", "prompt": "Write a haiku about a GPU", "max_tokens": 32, "temperature": 0.7}'
```

The Ray dashboard runs on port 8265 of Spark A: `ssh -N -L 8265:localhost:8265 spark-a`, then open `http://localhost:8265`.

| Want to go bigger? | From the playbooks |
|---|---|
| Llama 3.1 405B on two Sparks | `hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4` with `--max-model-len 64 --gpu-memory-utilization 0.9 --max-num-seqs 1 --max-num-batched-tokens 64`. The playbook warns it "has insufficient memory headroom for production use" |
| Four or more Sparks through a switch | image `nvcr.io/nvidia/vllm:26.02-py3`, the unpinned `run_cluster.sh` from vLLM's main branch, `MiniMaxAI/MiniMax-M2.5` with `--tensor-parallel-size 4` (= node count) |

> 💡 Two Sparks over TP add memory, not per-stream speed. Every token waits for an all-reduce over the link. Measure your own tok/s with lab 05-3 and compare it with row 4 of lab 05-1: the NVFP4 70B fits on **one** Spark. Module 07 quantizes models yourself.

### Optional: Nemotron 3 Super 120B in FP8 across both Sparks

> ⚠ **Course addition, run on two Sparks on 2026-10-03.** No NVIDIA Spark playbook serves this checkpoint on two Sparks. The flags come from the [FP8 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8) (written for 4× H100), changed where the table below says so. All output in this part was recorded on our two Sparks with `nvcr.io/nvidia/vllm:26.05-py3` (vLLM 0.20.1).

Module 06 serves Nemotron 3 Super as **NVFP4** (80 GB) on one Spark. The **FP8** checkpoint is 128.4 GB, too big for one Spark but comfortable on two: 57.57 GiB of weights per node. FP8 keeps more precision than NVFP4.

It also shows a different kind of model. Nemotron 3 Super is a Mamba-2/MoE hybrid: of its 88 layers only **8 are attention layers**, and only those keep a KV cache. One token costs 2 × 8 layers × 2 KV heads × 128 × 1 byte (fp8) = **4 KB** of KV. Llama 3.3 70B at bf16 costs 2 × 80 × 8 × 128 × 2 = 328 KB. The 40 Mamba layers keep a fixed-size state per sequence instead.

| Model card (4× H100) | Two Sparks | Why |
|---|---|---|
| `--tensor-parallel-size 4` | `--tensor-parallel-size 2` | one GPU per Spark |
| one machine, default backend | `--distributed-executor-backend ray` | the Ray cluster from Steps 1–3 |
| `--async-scheduling` | removed | vLLM 0.20.1 refuses it with Ray: `` `ray` does not support async scheduling yet `` |
| `--swap-space 0` | removed | the flag is gone in vLLM 0.20.1: `unrecognized arguments: --swap-space 0` |
| `--gpu-memory-utilization 0.9` | `0.8` | the 70B value above. On a Spark, GPU memory is system memory |
| `--served-model-name nvidia/nemotron-3-super` | `nemotron-3-super` | the name Module 06 uses, so the same ⚡ blocks work |

**Step 5 — the weights on both Sparks.** Each Spark loads its half from its **own** `~/.cache/huggingface`, so both need the full 128 GB. Download on Spark A:

```bash
# on: spark
hf download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8
```

Then either run the same command on Spark B, or copy it over the QSFP link, which is much faster than a second download. Copy the whole `hub` folder, not only the model's folder: recent `huggingface_hub` versions keep the data in a shared `hub/blobs/` store and the model folder only links into it. `rsync` skips files Spark B already has:

```bash
# on: spark
rsync -a --info=progress2 ~/.cache/huggingface/hub/ <SPARK_B_QSFP_IP>:.cache/huggingface/hub/
```

**Step 6 — serve it.** Keep the Ray head and worker from Steps 2–3 running (`ray status`: 2 nodes, 2 GPUs). Stop the 70B server first if it is still running. `--enable-expert-parallel` spreads the MoE experts over the two GPUs instead of slicing each expert. `HF_HUB_OFFLINE=1` makes both nodes load from their local cache:

```bash
# on: spark
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec -e HF_HUB_OFFLINE=1 $VLLM_CONTAINER vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \
  --served-model-name nemotron-3-super \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --distributed-executor-backend ray \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --trust-remote-code \
  --gpu-memory-utilization 0.8 \
  --max-cudagraph-capture-size 128 \
  --enable-chunked-prefill \
  --mamba-ssm-cache-dtype float32 \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

**Expected output** (RECORDED — Spark A + Spark B, 2026-10-03; log prefixes and timestamps trimmed; `ip=192.168.100.96` is Spark B's half)

```
(RayWorkerWrapper pid=406, ip=192.168.100.96) Model loading took 57.57 GiB memory and 283.078022 seconds
(RayWorkerWrapper pid=2786) Model loading took 57.57 GiB memory and 284.023674 seconds
GPU KV cache size: 15,948,274 tokens
Maximum concurrency for 262,144 tokens per request: 60.84x
init engine (profile, create kv cache, warmup model) took 62.15 s (compilation: 14.25 s)
(APIServer pid=1608) INFO:     Application startup complete.
```

About 6 minutes from start to ready, most of it reading 120 GB from disk. 15.9 million tokens of KV cache is room for **60 conversations of 262K tokens at once**. Lab 05-1's arithmetic said 71.5: its 60 GB weight estimate is lower than the 61.8 GB vLLM really loaded, and it leaves out the Mamba state. Trust the log.

**Step 7 — talk to it.** It is a reasoning model: it thinks before it answers, so give it a large `max_tokens`. With 400 it used every token to think and returned `content: null`:

```bash
# on: spark
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "nemotron-3-super", "messages": [{"role": "user", "content": "Write a haiku about a GPU"}], "max_tokens": 2000, "temperature": 0.7}' \
  | python3 -c 'import sys,json; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"]); print(r["usage"])'
```

**Expected output** (RECORDED — Spark A + Spark B, 2026-10-03; your haiku will differ at temperature 0.7)

```
Silicon heart beats
Pixels dance in parallel
Dreams render fast now
{'prompt_tokens': 23, 'total_tokens': 454, 'completion_tokens': 431, 'prompt_tokens_details': None}
```

```spark
{"target": "vllm", "which": "a", "model": "nemotron-3-super",
 "messages": [{"role": "user", "content": "In three sentences: why does tensor parallelism need a fast link between the GPUs?"}], "max_tokens": 2000}
```

Tool calling works with the same request as Section 4. Asked "What's the weather in Bangkok right now?" with a `get_weather` tool, it returned `get_weather({"city": "Bangkok"})` with `finish_reason: tool_calls` in 2.5 s. What we measured, from Spark B's side of the LAN:

| Requests at once | Total tok/s | Per stream tok/s |
|---|---|---|
| 1 (a haiku, thinking on: 740 tokens in 40.9 s) | 18.1 | 18.1 |
| 1 (thinking off) | 16.6 | 16.6 |
| 4 (thinking off) | 34.6 | 9.8 |

(RECORDED — 2026-10-03. "Thinking off" sends `"chat_template_kwargs": {"enable_thinking": false}`.) Four users get twice the total throughput of one: continuous batching (Section 5) works across both Sparks. Each stream is slower, because every token waits for an all-reduce over the link.

> 💡 vLLM listens on every interface, so laptops on your tailnet reach it at `http://<spark-a tailnet name>:8000/v1` with model `nemotron-3-super`, and `/docs` opens the API explorer in a browser. There is no password: Module 08 puts a LiteLLM gateway with keys in front of it.

✓ Checkpoint: `docker exec $VLLM_CONTAINER ray status` shows 2 nodes, and the haiku request returns text from the 70B model served across both Sparks (optional: also from `nemotron-3-super`, with `Model loading took 57.57 GiB` in both nodes' logs).

## 7 · Serve a LoRA fine-tune next to its base model

Modules 09–11 fine-tune models with **LoRA**: a small adapter (tens of MB) trained on top of a frozen base model. vLLM can serve the base model **and** one or more adapters from one server, and each adapter shows up as its own model id in `/v1/models`. You can serve a fine-tune without merging it.

> ⚠ **Course addition.** The NVIDIA vLLM playbook does not cover LoRA. The flags below are vLLM's own (`--enable-lora`, `--lora-modules`, `--max-lora-rank`). Confirm them on your Spark before you rely on them, because flags change between vLLM versions:

```bash
# on: spark
docker run --rm --entrypoint vllm vllm/vllm-openai:latest serve --help=all 2>/dev/null | grep -i -- '--.*lora' \
  || docker run --rm --entrypoint vllm vllm/vllm-openai:latest serve --help | grep -i -- '--.*lora'
```

The command below serves `Qwen/Qwen3-8B` (an ungated base model that the SGLang playbook uses as its default) plus an adapter saved in `~/w25/adapters/hotel-ft` on the Spark. Replace the base model with the `base_model_name_or_path` from your adapter's `adapter_config.json`, and set `--max-lora-rank` to at least the adapter's `r`:

```bash
# on: spark
docker run -d \
  --name vllm-server \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/w25/adapters:/adapters" \
  --entrypoint '' \
  vllm/vllm-openai:latest \
  vllm serve Qwen/Qwen3-8B \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.5 \
    --max-num-seqs 9 \
    --enable-lora \
    --lora-modules hotel-ft=/adapters/hotel-ft \
    --max-lora-rank 16
curl -s http://localhost:8000/v1/models | python3 -m json.tool | grep '"id"'
```

**Expected output** (EXAMPLE — illustrative shape, not a measurement: one id for the base, one for the adapter)

```
            "id": "Qwen/Qwen3-8B",
            "id": "hotel-ft",
```

Clients pick the adapter with `"model": "hotel-ft"` and the base with `"model": "Qwen/Qwen3-8B"`. Module 08's LiteLLM gateway routes to either, and Module 13 evaluates the adapter against the base before you decide to merge it. You will build exactly this command in the exercise below.

✓ Checkpoint: you can explain why `--max-lora-rank` must be at least the adapter's rank, and which path `--lora-modules` needs (the path **inside** the container).

## Labs — run them here

**labs/lab01_launch_builder.py** — Size vLLM's memory flags with arithmetic (one Spark and two), then print a ready `docker run`.

**labs/lab02_first_server.py** — Start the playbook's base server (only with `--yes`), wait for `/health`, and send the `12*17` test.

**labs/lab03_continuous_batching.py** — Send 1, 2, 4 and 8 parallel requests and compare total tok/s with per-stream tok/s.

**labs/lab04_tool_calling.py** — One full tool-calling round trip through the OpenAI API, ending in a grounded answer.

Lab 01 is arithmetic and runs anywhere. Lab 02 drives the Spark over SSH (DRY without one). Labs 03 and 04 call vLLM on the Spark, or the labelled laptop stand-in.

## Try it yourself

**Exercise 05 — write a correct vLLM launch command.** Open `week25/05_vllm/exercises/ex05_launch_command.py`. It has four `TODO`s, each returning a list of command-line tokens (or a number):

1. `docker_flags()`: the container flags from the base configuration: GPU, `--ipc host`, port 8000, the HF cache.
2. `max_full_length_seqs(weights_gb, kv_gb_per_seq, util)`: how many full-context sequences fit, which becomes `--max-num-seqs`.
3. `tool_flags(family)`: auto tool choice plus the right `--tool-call-parser` for Qwen3.6, Nemotron and Gemma 4.
4. `lora_flags(name, path, rank)`: serve a LoRA adapter next to its base model.

The offline checker tests each function, then assembles two full commands (the agent-ready server and a base + LoRA server) and parses them the way a shell would.

```bash
# on: laptop
.venv/bin/python week25/05_vllm/exercises/ex05_launch_command.py
```

**Expected output** (once all four TODOs are done, captured on this Mac; the two printed commands are trimmed here)

```
✓ docker_flags: --gpus all · --ipc host · -p 8000:8000 · HF cache → /root/.cache/huggingface
✓ max_full_length_seqs: Qwen3-32B@0.8 → 9 · Llama-70B-NVFP4@0.8 → 5 · 70B-bf16 → 0 · 20 GB@0.4 → 13
✓ tool_flags: auto tool choice on · qwen3.6→qwen3_xml · nemotron→qwen3_coder · gemma4→gemma4
✓ lora_flags: --enable-lora · --lora-modules hotel-ft=/adapters/hotel-ft · --max-lora-rank ≥ 16
✓ both assembled commands parse: docker flags, IMAGE, then `vllm serve <model>` flags

▣ A · agent-ready server (assembled from your functions; the full recipe adds speed flags)
$ docker run -d \
  --name vllm-server \
  --gpus all \
  …
  vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.4 \
    --max-num-seqs 4 \
    --kv-cache-dtype fp8 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml
```

<details><summary>Hint — why does the 70B bf16 case return 0?</summary>

The slice is 0.8 × 128 = 102.4 GB. The weights alone are 141.2 GB, so the KV room is negative. Return 0 rather than a negative number: no full-length sequence fits, and vLLM would refuse to start.

</details>

<details><summary>Hint — what goes after <code>--lora-modules</code>?</summary>

One `NAME=PATH` pair per adapter. `NAME` is the model id clients send. `PATH` is where the adapter sits **inside the container**: `/adapters/hotel-ft`, because the command mounts `$HOME/w25/adapters` at `/adapters`.

</details>

<details><summary>Stretch — two adapters, one server</summary>

Add a second adapter to `lora_flags` (vLLM accepts several `NAME=PATH` pairs after `--lora-modules`). How many adapters can be active in one batch? Look up `--max-loras` in `vllm serve --help=all` on your Spark.

</details>

✓ Checkpoint: all five checker lines are ✓.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` at start-up | Lower `--max-model-len` and `--max-num-seqs`, or lower `--gpu-memory-utilization` (playbook). Use lab 05-1 to see which term is too big |
| Out of memory even though the model should fit | Unified memory: the OS page cache still holds files. The playbook flushes it: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| `nvidia-smi --query-gpu` shows memory as `N/A` | Expected on unified memory (playbook). Use plain `nvidia-smi`, or `free -g` |
| `content: null` with `finish_reason: length` | A reasoning model spent the budget thinking. Raise `max_tokens` (the playbook uses 4096) |
| Tool calls arrive as text in `content`, `tool_calls` is empty | The server was started without `--enable-auto-tool-choice`, or with a `--tool-call-parser` for another model family (Section 4) |
| An error that auto tool choice needs `--enable-auto-tool-choice` and `--tool-call-parser` | Your client sent `tools`, but the server was not started for tool calling. Restart with both flags |
| `Server not responding on port 8000` | Port in use (NIM also uses 8000, Module 06). `lsof -i :8000`, or map `-p 8001:8000` |
| `exec format error` / missing ARM64 image | Use an image from Section 2. The Spark is aarch64 |
| Gated repo / 401 from Hugging Face | Accept the model licence on its Hugging Face page, then `hf auth login` on the Spark |
| `rm: cannot remove …/models--…: Permission denied` | The container downloaded as root: `sudo rm -rf $HOME/.cache/huggingface/hub/<model>` (playbook) |
| Node 2 missing from `ray status` | QSFP link or IP: redo Module 02's checks, and make sure both Sparks use the same `MN_IF_NAME` and image tag |
| Ray cluster vanished after an SSH drop | `run_cluster.sh` has an EXIT trap: always start it inside `tmux` |
| `run_cluster.sh` prints nothing for minutes, `ray status` says Ray is not installed | The patched start-up `pip install ray` stalled (seen once on 2026-10-03; the retry finished in seconds). `docker rm -f node-NNNN`, then start it again |
| `unrecognized arguments: --swap-space 0` or `` `ray` does not support async scheduling yet `` | Model-card flags for other vLLM versions or one machine. Drop them, as in the Nemotron command in Section 6 |
| A model copied from the other Spark is tiny: `du -shL ~/.cache/huggingface/hub/models--<org>--<name>/snapshots` shows KB, not GB | Only the model's folder of links was copied, not the shared `hub/blobs/` store. Copy the whole `~/.cache/huggingface/hub/` folder (Section 6, Step 5) |

## Next

Continue to [Lab 06 — SGLang, TensorRT-LLM, NIM and Nemotron: the engine bake-off](../06_sglang_trtllm_nim/TUTORIAL.md): launch the other three serving engines, serve Nemotron Nano and Super, and compare them all with one fair benchmark.
