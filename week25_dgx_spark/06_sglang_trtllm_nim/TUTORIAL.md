# ▶ Spark Lab 06 — SGLang, TensorRT-LLM, NIM and Nemotron: the engine bake-off

> Part of Week 25 · DGX Spark: fine-tune, serve, and build sandboxed agents. You type the commands, you see the real output. Every lab also runs in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's playbook, or a clearly marked EXAMPLE.

**What you'll actually do**
- Launch three more serving engines from their playbooks: **SGLang** on :30000, **TensorRT-LLM** on :8355, **NIM** on :8000.
- Use SGLang's two strengths: schema-constrained JSON and prefix caching you can see in `cached_tokens`.
- Serve NVIDIA's own **Nemotron 3 Nano** (vLLM) and **Nemotron 3 Super** (vLLM or TensorRT-LLM) on one Spark.
- Run a **fair bake-off**: same model, same prompts, same settings, one engine at a time, medians.
- Turn a workload's needs into an engine choice, with the playbook fact behind every pick.

**Time** ~60 min · **Difficulty** intermediate · **Hardware** 1 Spark (or none: DRY mode + laptop stand-in)

**Official playbooks covered:** [SGLang](https://build.nvidia.com/spark/sglang) · [TensorRT-LLM](https://build.nvidia.com/spark/trt-llm) · [NIM for LLMs](https://build.nvidia.com/spark/nim-llm) · [Nemotron](https://build.nvidia.com/spark/nemotron)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Module 05 done | you started and stopped `vllm-server` once | this module compares against vLLM |
| No engine left running | `docker ps` on the Spark shows no serving container | every engine reserves most of the 128 GB (Section 1) |
| A Hugging Face login on the Spark | `hf auth whoami` | SGLang, TensorRT-LLM and Nemotron download from Hugging Face |
| An NGC API key (for NIM) | from <https://ngc.nvidia.com/setup/api-key> | NIM containers and their models come from `nvcr.io` |
| ~100 GB free disk | `df -h /` | three container images plus models |

```bash
# on: spark
docker ps --format '{{.Names}}  {{.Image}}  {{.Ports}}'
hf auth whoami
df -h / | tail -1
```

> 🔐 Log in to NGC **on the Spark**, once, by typing the key at the prompt: `docker login nvcr.io --username '$oauthtoken'`. The NIM playbook pipes `$NGC_API_KEY` into the same command. Either way, never paste the key into a lab or the Lab Runner.

✓ Checkpoint: `docker ps` on the Spark lists no serving container, and you have an NGC API key ready.

## 1 · Four engines, one API

Every engine in this module speaks the same OpenAI-compatible API, so the labs, LiteLLM (Module 08) and NAT (Module 14) talk to all of them with one client. They differ in how they get speed and in how much setup they need:

| Engine | Image (exact tag from the playbook) | Port | Serve command | Playbook tagline / strength |
|---|---|---|---|---|
| vLLM (Module 05) | `vllm/vllm-openai:latest` | 8000 | `vllm serve` | continuous batching, PagedAttention; the agent-ready recipe |
| **SGLang** | `lmsysorg/sglang:latest-cu130` | 30000 | `sglang serve --model-path` | RadixAttention prefix cache, xGrammar structured output |
| **TensorRT-LLM** | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` (single node) · `1.3.0rc5` (multi-node) | 8355 | `trtllm-serve` | "Lower-latency responses and higher throughput for the largest models" |
| **NIM** | `nvcr.io/nim/meta/llama-3.1-8b-instruct-dgx-spark:latest` | 8000 | (built in) | "Prebuilt, GPU-optimized model containers with a ready-to-use HTTP endpoint" |

**Run one engine at a time.** Two reasons:

1. **Ports.** NIM and vLLM both use 8000. The playbooks' fix for a busy port is to map another host port, for example `-p 8001:8000`.
2. **Memory.** Each engine reserves a fixed share of the unified memory at start-up: vLLM `--gpu-memory-utilization 0.8`, SGLang `--mem-fraction-static 0.85`, TensorRT-LLM `free_gpu_memory_fraction: 0.9`. Two of them at those settings do not fit in 128 GB.

```bash
# on: spark
docker rm -f vllm-server sglang-server trtllm-server nim-llm-demo 2>/dev/null; docker ps
```

✓ Checkpoint: you can name each engine's port, and say why NIM and vLLM cannot both run with their default settings.

## 2 · SGLang: prefix caching and structured JSON

SGLang's **RadixAttention** keeps the KV cache of every prefix it has seen in a tree. When the next request starts with the same tokens (the same system prompt, the same chat history, the same RAG context), SGLang skips that prefill work. Agents and multi-turn chat repeat long prefixes on every turn, so this saves a lot. Its **structured output** (xGrammar) forces the answer to match a JSON schema while it is generated.

The playbook's environment and base configuration. `Qwen/Qwen3-8B` is its default: a dense 8B model with a fast warm-up, ungated:

```bash
# on: spark
export HF_TOKEN=""                        # empty is fine for public models such as Qwen/Qwen3-8B
export MODEL_HANDLE="Qwen/Qwen3-8B"
export MAX_MODEL_LEN=8192
export SGLANG_IMAGE="lmsysorg/sglang:latest-cu130"
docker pull "$SGLANG_IMAGE"
docker run -d \
  --name sglang-server \
  --gpus all \
  --ipc host \
  --cap-add SYS_NICE \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 30000:30000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub" \
  "$SGLANG_IMAGE" \
  sglang serve --model-path "$MODEL_HANDLE" \
    --host 0.0.0.0 \
    --port 30000 \
    --context-length $MAX_MODEL_LEN \
    --mem-fraction-static 0.85 \
    --attention-backend flashinfer \
    --enable-cache-report \
    --trust-remote-code
timeout 900 bash -c 'until curl -sf http://localhost:30000/health > /dev/null 2>&1; do sleep 10; done' \
  || { echo "Server failed to start within 900s"; docker logs sglang-server | tail -50; exit 1; }
```

| Flag | Playbook note |
|---|---|
| `--context-length` | SGLang's name for vLLM's `--max-model-len` |
| `--mem-fraction-static 0.85` | SGLang's `--gpu-memory-utilization`. On a Spark, lower it toward `0.75` under memory pressure |
| `--attention-backend flashinfer` | the validated attention backend for Blackwell |
| `--enable-cache-report` | fills `usage.prompt_tokens_details.cached_tokens`, so you can see the prefix cache work |
| `--quantization modelopt_fp4` | add it for the NVFP4 checkpoints in the support list (for example `nvidia/Qwen3-32B-FP4`) |

> 💡 The first launch downloads the weights and captures CUDA graphs. The playbook plans ~10–15 minutes for `Qwen/Qwen3-8B`, and longer for large MoE models.

Ask it something. Qwen3 thinks before it answers, so give it room:

```spark
{"target": "sglang", "which": "a", "model": "Qwen/Qwen3-8B",
 "messages": [{"role": "user", "content": "What is the difference between speed and velocity? Two sentences."}],
 "max_tokens": 150}
```

Lab 06-3 runs the playbook's Steps 6 and 7 from Python: a `json_schema` request whose answer the lab checks against the schema, and the two-turn physics-tutor conversation whose second turn should report `cached_tokens` > 0:

```bash
# on: laptop
.venv/bin/python week25/06_sglang_trtllm_nim/labs/lab03_sglang_features.py
```

**Expected output** (captured on this Mac: no Spark, so Ollama stood in. It is not SGLang)

```
◆ endpoint: LAPTOP STAND-IN · Ollama on this Mac (not SGLang, not the Spark) · model gemma3:4b

▣ STEP 1 · structured output — response_format json_schema (playbook Step 7; max_tokens 150, so answers are kept short)
· ANSWER  { "languages": [     {         "name": "Python",         "primary_use": "Data science, web development, scripting",         "year_created": 1991     },     …
✓ valid JSON matching the schema · 3 languages · Python (1991), JavaScript (1995), Java (1995)

▣ STEP 2 · prefix caching — the playbook's two-turn conversation (Step 6)
│ turn 1: prompt_tokens   74 · cached_tokens 0
│ turn 2: prompt_tokens  194 · cached_tokens 69
◆ LAPTOP STAND-IN: Ollama keeps its own prompt cache, so it may report cached tokens too. That shows the API field, not SGLang's RadixAttention.
```

Two changes from the playbook: the lab caps `max_tokens` at 150 (the playbook uses 512 for the JSON request) and asks for short answers so the JSON is not cut off. A first run with long answers did get cut off, and `json.loads` failed. That is the failure mode to watch for. The playbook's other signal is in the server log:

```bash
# on: spark
docker logs sglang-server 2>&1 | grep "cached-token" | tail -10
```

The playbook says to look for `#cached-token` values greater than 0 on later turns.

> ⚠ Hybrid Mamba/SSM models such as `Qwen/Qwen3.6-35B-A3B` always report 0 cached tokens: SGLang skips cross-request prefix reuse for them. Test prefix caching with a standard-attention model such as `Qwen/Qwen3-8B` (playbook).

```bash
# on: spark
docker stop sglang-server && docker rm sglang-server
```

✓ Checkpoint: lab 06-3 prints `✓ valid JSON matching the schema`, and on SGLang turn 2 reports `cached_tokens` above 0.

## 3 · TensorRT-LLM: NVIDIA's optimized runtime on :8355

TensorRT-LLM is NVIDIA's library of optimized kernels, memory management, quantization and parallelism for inference. `trtllm-serve` wraps it in an OpenAI-compatible server. Most tuning lives in a small YAML file passed with `--extra_llm_api_options`, not in command-line flags.

Check that the container sees the GPU, then run the playbook's default path, Llama 3.1 8B Instruct at NVFP4. It uses `--network host`, so no `-p` is needed: the server listens on the Spark's port 8355 directly:

```bash
# on: spark
export HF_TOKEN="<YOUR_HUGGINGFACE_TOKEN>"     # or rely on the token from `hf auth login`
export MODEL_HANDLE="nvidia/Llama-3.1-8B-Instruct-FP4"
export TRTLLM_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13"
mkdir -p "$HOME/.cache/huggingface"
docker run --rm --gpus all "$TRTLLM_IMAGE" python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"

docker run --name trtllm-server --rm -it \
  --gpus all \
  --ipc host \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  "$TRTLLM_IMAGE" \
  bash -c '
    hf download "$MODEL_HANDLE" &&
    cat > /tmp/extra-llm-api-config.yml <<EOF
print_iter_log: false
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: 0.9
cuda_graph_config:
  enable_padding: true
disable_overlap_scheduler: true
EOF
    trtllm-serve "$MODEL_HANDLE" \
      --host 0.0.0.0 \
      --port 8355 \
      --max_batch_size 64 \
      --trust_remote_code \
      --extra_llm_api_options /tmp/extra-llm-api-config.yml
  '
```

| Setting | Role |
|---|---|
| `kv_cache_config.free_gpu_memory_fraction: 0.9` | TensorRT-LLM's KV cache share, like vLLM's utilization flag |
| `--max_batch_size 64` | the most sequences in one batch, like vLLM's `--max-num-seqs` |
| `cuda_graph_config.enable_padding` | pads batches to captured CUDA-graph sizes so they replay fast |

This runs in the foreground (`-it`). Leave that terminal open and test from a second one with the playbook's request:

```bash
# on: spark
curl -s http://localhost:8355/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "nvidia/Llama-3.1-8B-Instruct-FP4", "messages": [{"role": "user", "content": "Explain tensor parallelism in two sentences."}], "max_tokens": 64}'
```

```spark
{"target": "trtllm", "which": "a", "model": "nvidia/Llama-3.1-8B-Instruct-FP4",
 "messages": [{"role": "user", "content": "Explain tensor parallelism in two sentences."}], "max_tokens": 64}
```

What else the playbook covers, and what it does not yet:

- **Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16** has its own recipe with `--reasoning_parser nano-v3 --tool_parser qwen3_coder` and a `nano_v3.yaml` (see the playbook's Step 5).
- **gpt-oss-20b / 120b** need the `TIKTOKEN_ENCODINGS_BASE` setup at the start of the `bash -c` script (see the playbook).
- **Multi-node** uses OpenMPI instead of Ray: a hostfile with one QSFP IP per Spark, the playbook's `trtllm-mn-entrypoint.sh`, image `1.3.0rc5`, and `trtllm-llmapi-launch trtllm-serve … --tp_size 2`. The validated model there is `nvidia/Qwen3-235B-A22B-FP4`.
- **Agent-ready Qwen3.6-35B-A3B**: the playbook says its launch settings are "pending validation" and tells you not to borrow parser settings from another model family. For agents today, use vLLM (Module 05).

Stop it with `Ctrl+C` in the serving terminal. `--rm` removes the container.

✓ Checkpoint: the curl on :8355 returns a `choices` array, and you can say where TensorRT-LLM keeps its KV cache setting.

## 4 · NIM: one prebuilt container per model

A NIM is a container with the model, the engine and a tuned configuration already inside. You pick a container instead of a model and a set of flags. The playbook's default for DGX Spark is the Llama 3.1 8B Instruct NIM. More are listed in the [NGC catalog](https://catalog.ngc.nvidia.com), for example Qwen3-32B for DGX Spark.

```bash
# on: spark
docker login nvcr.io --username '$oauthtoken'        # paste the NGC API key at the prompt
export NGC_API_KEY="<YOUR_NGC_API_KEY>"               # the container needs it to download model assets
export CONTAINER_NAME="nim-llm-demo"
export IMG_NAME="nvcr.io/nim/meta/llama-3.1-8b-instruct-dgx-spark:latest"
export LOCAL_NIM_CACHE=~/.cache/nim
export LOCAL_NIM_WORKSPACE=~/.local/share/nim/workspace
mkdir -p "$LOCAL_NIM_WORKSPACE" && chmod -R a+w "$LOCAL_NIM_WORKSPACE"
mkdir -p "$LOCAL_NIM_CACHE" && chmod -R a+w "$LOCAL_NIM_CACHE"
docker run -it --rm --name=$CONTAINER_NAME \
  --gpus all \
  --shm-size=16GB \
  -e NGC_API_KEY=$NGC_API_KEY \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -v "$LOCAL_NIM_WORKSPACE:/opt/nim/workspace" \
  -p 8000:8000 \
  $IMG_NAME
```

> ⚠ `export NGC_API_KEY=…` puts the key in that shell's history. Prefer `read -rs NGC_API_KEY && export NGC_API_KEY` (it reads the key without echoing it) on the Spark.

The served model id is `meta/llama-3.1-8b-instruct`, not the Hugging Face handle. The playbook's test request:

```bash
# on: spark
curl -X 'POST' 'http://0.0.0.0:8000/v1/chat/completions' \
  -H 'accept: application/json' -H 'Content-Type: application/json' \
  -d '{"model": "meta/llama-3.1-8b-instruct", "messages": [{"role": "system", "content": "detailed thinking on"}, {"role": "user", "content": "Can you write me a song?"}], "top_p": 1, "n": 1, "max_tokens": 15, "frequency_penalty": 1.0, "stop": ["hello"]}'
```

NIM uses port 8000 like vLLM, so the runner reaches it with the `nim` target:

```spark
{"target": "nim", "which": "a", "model": "meta/llama-3.1-8b-instruct",
 "messages": [{"role": "user", "content": "Can you write me a song? Four lines."}], "max_tokens": 80}
```

| | NIM | vLLM / SGLang / TensorRT-LLM |
|---|---|---|
| You choose | a container | a model, an image, and flags |
| Tuning | done for you, per model | yours: memory, batch, parsers |
| Models | the NIM catalog | any supported Hugging Face checkpoint, including your fine-tunes |
| Login | NGC API key | Hugging Face token for gated models |

Stop it with `Ctrl+C` (the container uses `--rm`). The model stays in `~/.cache/nim`. The playbook removes it with `rm -rf "$LOCAL_NIM_CACHE"` only when you need the disk space back.

✓ Checkpoint: the NIM answers on :8000 with model `meta/llama-3.1-8b-instruct`, and `/v1/models` shows that id.

## 5 · Nemotron Nano and Super on one Spark

Nemotron 3 is NVIDIA's open model family for reasoning, tool use and long context. The Nemotron playbook serves two sizes on one Spark:

| | Nemotron 3 Nano Omni | Nemotron 3 Super |
|---|---|---|
| Checkpoint | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` (local weights) | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` |
| Engine and image | vLLM `vllm/vllm-openai:v0.20.0` | vLLM `vllm/vllm-openai:cu130-nightly` **or** TensorRT-LLM `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc9` |
| Port · served name | 8000 · `nemotron_3_nano_omni` | vLLM: 8000 · `nemotron-3-super` · TensorRT-LLM: **8123** · read it from the log |
| Memory choices | `--max-num-seqs 8`, `--max-model-len 131072`, `--gpu-memory-utilization 0.8` | `--max-num-seqs 4`, `--max-model-len 1000000`, `--gpu-memory-utilization 0.90`, `--kv-cache-dtype fp8` |
| Parsers | `--reasoning-parser nemotron_v3` · `--tool-call-parser qwen3_coder` | vLLM: `super_v3` plugin · TensorRT-LLM: `nano-v3` · both `qwen3_coder` |

**Nano (vLLM).** Download the Omni weights to a folder on the Spark, then point `WEIGHTS` at it. The base image lacks audio packages, so the command installs `vllm[audio]` first:

```bash
# on: spark
WEIGHTS=/path/to/nemotron-3-nano-omni-weights
docker pull vllm/vllm-openai:v0.20.0
docker run --rm -it \
  --gpus all \
  --ipc=host -p 8000:8000 \
  --shm-size=16g \
  --name vllm-nemotron-omni \
  -v "${WEIGHTS}:/model:ro" \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.20.0 -c  \
  "pip install vllm[audio] && vllm serve /model \
  --served-model-name=nemotron_3_nano_omni \
  --max-num-seqs 8 \
  --max-model-len 131072 \
  --port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.8 \
  --limit-mm-per-prompt '{\"video\": 1, \"image\": 1, \"audio\": 1}' \
  --media-io-kwargs '{\"video\": {\"fps\": 2,  \"num_frames\": 256}}' \
  --allowed-local-media-path=/ \
  --enable-prefix-caching \
  --max-num-batched-tokens 32768 \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder"
```

If it runs out of memory, the playbook lowers `--gpu-memory-utilization` to `0.70` first, then `--max-model-len` to `32768`. Because it is served by vLLM on :8000, the `vllm` target reaches it:

```spark
{"target": "vllm", "which": "a", "model": "nemotron_3_nano_omni",
 "messages": [{"role": "user", "content": "New York is a great city because..."}], "max_tokens": 150}
```

**Super (vLLM path).** Download the reasoning-parser plugin first. The four environment variables are the playbook's single-GPU fixes (Marlin NVFP4 kernels, long context, all-reduce):

```bash
# on: spark
wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/raw/main/super_v3_reasoning_parser.py
docker pull vllm/vllm-openai:cu130-nightly
docker run --rm -it --gpus all \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e HF_TOKEN=$HF_TOKEN \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/super_v3_reasoning_parser.py:/app/super_v3_reasoning_parser.py \
  -p 8000:8000 \
  vllm/vllm-openai:cu130-nightly \
    --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --served-model-name nemotron-3-super \
    --host 0.0.0.0 \
    --port 8000 \
    --async-scheduling \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --data-parallel-size 1 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --max-num-seqs 4 \
    --max-model-len 1000000 \
    --moe-backend marlin \
    --mamba_ssm_cache_dtype float32 \
    --quantization fp4 \
    --speculative_config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
    --reasoning-parser-plugin /app/super_v3_reasoning_parser.py \
    --reasoning-parser super_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
```

```spark
{"target": "vllm", "which": "a", "model": "nemotron-3-super",
 "messages": [{"role": "user", "content": "Summarize what Latent MoE changes for routing traffic."}], "max_tokens": 150}
```

**Super (TensorRT-LLM path).** Download the checkpoint into a local folder, write the playbook's `extra-llm-api-config.yml` next to it (Step 9 of the Nemotron Super tab; copy it exactly), then:

```bash
# on: spark
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc9
hf download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 --local-dir ./nemotron-super-nvfp4
docker run --rm -it --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -e TLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -v "$(pwd)":/workspace \
  -w /workspace \
  -p 8123:8123 \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc9 \
  trtllm-serve nemotron-super-nvfp4 \
    --host 0.0.0.0 \
    --port 8123 \
    --max_batch_size 8 \
    --tp_size 1 --ep_size 1 \
    --max_num_tokens 8192 \
    --trust_remote_code \
    --reasoning_parser nano-v3 \
    --tool_parser qwen3_coder \
    --extra_llm_api_options extra-llm-api-config.yml \
    --max_seq_len 1048576
```

> ⚠ This recipe uses port **8123**, not TensorRT-LLM's usual 8355. Lab 06-1 probes both. To point the ⚡ blocks and other labs at it, set `SPARK_URL_TRTLLM=http://spark-a:8123/v1` in 🖥 Spark setup, and read the served model name from the `trtllm-serve` log (the playbook uses a placeholder).

Why Super fits one Spark (from the playbook's architecture notes): **LatentMoE** runs experts in a compressed dimension and activates about 12B of 120B parameters per token. **MTP** (one multi-token-prediction layer baked into the checkpoint) drafts 3 tokens for speculative decoding (Module 07). **Mamba-2 hybrid** layers keep an SSM state instead of a growing KV cache, which is why the TensorRT-LLM config sets `enable_block_reuse: false`: Mamba state is not prefix-cacheable.

> 💡 Two Sparks? The **FP8** checkpoint (128.4 GB, more precision than NVFP4) runs across both with vLLM tensor parallel: [Module 05, Section 6](../05_vllm/TUTORIAL.md).

✓ Checkpoint: one Nemotron model answers through the OpenAI API, and you can say which port and served name each Nemotron recipe uses.

## 6 · A fair bake-off

"Engine X is faster" means nothing unless both engines did the same work. The rules the exercise checks:

| Keep the same | Why |
|---|---|
| model **and** precision | Llama 3.1 8B Instruct is in every engine's list: `nvidia/Llama-3.1-8B-Instruct-NVFP4` (vLLM), `nvidia/Llama-3.1-8B-Instruct-FP4` (SGLang, TensorRT-LLM), the Llama 3.1 8B NIM for DGX Spark |
| prompts, `max_tokens`, concurrency | different lengths are different work |
| `temperature: 0` | greedy decoding does the same work every run |
| a warm-up request first | the first call pays for loading and CUDA-graph capture |
| **one engine running** | engines share the 128 GB and the 273 GB/s |
| report **medians** of several runs | one slow outlier or one lucky run does not decide the result |

Lab 06-1 applies these rules to every engine that answers:

```bash
# on: laptop
.venv/bin/python week25/06_sglang_trtllm_nim/labs/lab01_engine_bakeoff.py
```

**Expected output** (captured on this Mac: no Spark, so every Spark endpoint is down and only the laptop row ran)

```
▣ STEP 1 · probe every OpenAI endpoint on the Spark
│ engine                                port   base URL         status  models
│ ────────────────────────────────────  ─────  ───────────────  ──────  ──────
│ vLLM / NIM                            8000   (no Spark host)  ○ down  —
│ SGLang                                30000  (no Spark host)  ○ down  —
│ TensorRT-LLM                          8355   (no Spark host)  ○ down  —
│ TensorRT-LLM (Nemotron Super recipe)  8123   (no Spark host)  ○ down  —
│ llama.cpp                             30080  (no Spark host)  ○ down  —
│ LM Studio                             1234   (no Spark host)  ○ down  —
│ Ollama                                11434  (no Spark host)  ○ down  —

▣ STEP 2 · run 3 prompts on each engine that is up (warm-up first · temperature 0 · max_tokens 100)
→ no Spark engine is up · running the method on Ollama on THIS laptop (gemma3:4b) as a LAPTOP STAND-IN

▣ STEP 3 · comparison
│ engine                              model      median TTFT  median tok/s  tokens  first answer starts
│ ──────────────────────────────────  ─────────  ───────────  ────────────  ──────  ────────────────────────────────────────────────
│ LAPTOP STAND-IN (Ollama, this Mac)  gemma3:4b  2286 ms      77.6          256     Tensor parallelism distributes a single tensor a
◆ LAPTOP STAND-IN: this row shows the method, not an engine. Its numbers are this Mac's and are never compared with a Spark engine.
```

On your Spark, the loop is: start one engine (Sections 2–5 or Module 05) → run lab 06-1 → stop the engine → start the next → run lab 06-1 again. Each run adds that engine's row. Keep the tables. For many users at once, point lab 05-3 at each engine's URL too: an engine can win at one user and lose at eight.

✓ Checkpoint: you have a lab 06-1 table with at least one Spark engine row (or, in DRY mode, the laptop row), and you can list three things that would make a comparison unfair.

## 7 · Which engine?

Measurements tell you which engine is fastest. Your workload's **hard needs** decide which engines are even candidates. Lab 06-2 scores each engine only on what the Spark playbooks show, rules out engines with no documented path for a need, and prints the playbook fact behind each pick (paraphrased; words in quotes are verbatim):

```bash
# on: laptop
.venv/bin/python week25/06_sglang_trtllm_nim/labs/lab02_which_engine.py --need tools,two-sparks
```

**Expected output** (offline, the same on every machine)

```
▣ STEP 2 · six workloads
│ workload                                              needs           pick                ruled out
│ ────────────────────────────────────────────────────  ──────────────  ──────────────────  ───────────
│ Hotel concierge agent (Module 14): tool calls, long…  tools+prefix    vLLM :8000          NIM
│ RAG answers that must be valid JSON for a dashboard   json+prefix     SGLang :30000       —
│ Llama 3.3 70B at bf16 — bigger than one Spark         two-sparks      vLLM :8000          SGLang, NIM
│ Nemotron 3 Super for reasoning, one Spark             nemotron-super  vLLM :8000          SGLang, NIM
│ A demo tomorrow, supported container, no tuning       least-setup     NIM :8000           —
│ Lowest latency for one validated model                latency         TensorRT-LLM :8355  —

▣ STEP 4 · your workload: tools + two-sparks
│ engine        score
│ ────────────  ─────────
│ vLLM          4
│ SGLang        ruled out
│ TensorRT-LLM  3
│ NIM           ruled out
═ pick: vLLM on :8000
```

The rules of thumb that fall out:

- **Agents on one Spark → vLLM.** It has the only validated agent-ready recipe, and tool calling is proven end to end (lab 05-4).
- **JSON-heavy RAG, shared prompts → SGLang.** Structured output and prefix caching are its documented strengths.
- **Two Sparks → vLLM (Ray) or TensorRT-LLM (MPI).** The SGLang and NIM playbooks have no multi-node path on Spark.
- **Fastest route to a supported demo → NIM**, if the model you need has a DGX Spark NIM.
- **Latency for one model you will run for months → TensorRT-LLM**, then measure it against vLLM with lab 06-1.

The scorecard reflects what the playbooks *show*. A 1 means "try it and measure", not "unsupported".

✓ Checkpoint: you can pick an engine for the Module 20 capstone (a tool-calling agent that serves a fine-tune) and name the playbook fact behind the choice.

## Labs — run them here

**labs/lab01_engine_bakeoff.py** — Probe every OpenAI endpoint on the Spark and run the same prompts on each engine that is up.

**labs/lab02_which_engine.py** — Turn a workload's needs into an engine choice, with the playbook fact behind every pick.

**labs/lab03_sglang_features.py** — SGLang's schema-constrained JSON and its prefix cache, checked in Python.

Labs 01 and 03 call the engines on your Spark, or the labelled laptop stand-in. Lab 02 is offline and runs anywhere.

## Try it yourself

**Exercise 06 — plan a fair bake-off.** Open `week25/06_sglang_trtllm_nim/exercises/ex06_fair_bakeoff.py`. It has four `TODO`s:

1. `PORTS`: each engine's OpenAI port, from the playbooks (including the Nemotron Super TensorRT-LLM recipe).
2. `port_conflicts(ports)`: every pair of engines that share a port.
3. `fairness_problems(a, b)`: every reason a comparison of two runs is unfair.
4. `summarise(runs)`: the median TTFT and tok/s of repeated runs.

```bash
# on: laptop
.venv/bin/python week25/06_sglang_trtllm_nim/exercises/ex06_fair_bakeoff.py
```

**Expected output** (once all four TODOs are done, captured on this Mac)

```
✓ PORTS: vLLM 8000 · SGLang 30000 · TensorRT-LLM 8355 · NIM 8000 · Nemotron Super TRT-LLM 8123
✓ port_conflicts: finds every shared port · on a Spark: NIM and vLLM both want :8000
✓ fairness_problems: identical → fair · precision, temperature, warm-up, a second engine, max_tokens + concurrency, model + prompts all caught
✓ summarise: median of 3 runs (TTFT 140 ms, 38.0 tok/s) — one slow outlier does not decide the result

▣ your plan, applied to four engines serving Llama 3.1 8B Instruct
│ vllm    :8000   start → warm-up → 3 prompts → stop
│ nim     :8000   start → warm-up → 3 prompts → stop
│ trtllm  :8355   start → warm-up → 3 prompts → stop
│ sglang  :30000  start → warm-up → 3 prompts → stop
◆ nim and vllm share :8000 — run them one after the other, or map one to -p 8001:8000
```

<details><summary>Hint — why the median and not the mean or the best?</summary>

In the test data one run took 900 ms to its first token (an outlier: maybe another process woke up). The mean would be 387 ms, which describes no real run. The best run (120 ms) is luck. The median (140 ms) is what a typical request sees. Python has `statistics.median`.

</details>

<details><summary>Hint — how many problems for <code>{"max_tokens": 256, "concurrency": 8}</code>?</summary>

Two: one per broken rule. Build the list with one string per rule you check, and return it, even when it is empty.

</details>

<details><summary>Stretch — make the bake-off concurrent</summary>

Add a `concurrency` column to lab 06-1 by running each engine at 1 and 8 parallel requests (reuse `run()` from lab 05-3). Does the engine that wins at 1 also win at 8?

</details>

✓ Checkpoint: all four checker lines are ✓.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Bind for 0.0.0.0:8000 failed: port is already allocated` | vLLM or NIM is still running. `docker ps`, then stop it, or map `-p 8001:8000` |
| Out of memory when starting a second engine | Each engine reserves most of the memory (Section 1). Stop the first one. On a Spark, also flush the page cache: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| SGLang out of memory | Lower `--mem-fraction-static` (for example `0.7`) and/or `--context-length` (playbook) |
| SGLang slow first request | Kernel JIT and CUDA-graph capture. Wait for the ready message in `docker logs sglang-server` |
| SGLang `cached_tokens` is 0 or `n/a` | Add `--enable-cache-report`. For hybrid Mamba/SSM models (Qwen3.6-35B-A3B), 0 is expected (playbook) |
| `json_schema` response_format returns an error on SGLang | Use `lmsysorg/sglang:latest-cu130` (playbook) |
| JSON from lab 06-3 does not parse | `max_tokens` cut the answer off. Raise it (the playbook uses 512) or ask for shorter fields |
| TensorRT-LLM out of memory while loading weights | `TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=1` before starting the container (playbook) |
| TensorRT-LLM does not answer on 8355 | Still loading, or it exited: check the serving terminal and `lsof -i :8355`. For Nemotron Super the port is 8123 |
| NIM `Invalid credentials` at `docker login` | The username is literally `$oauthtoken` (in single quotes); paste the key with no extra spaces |
| NIM returns 404 for the model | Use `"model": "meta/llama-3.1-8b-instruct"`, the NIM's own id, not a Hugging Face handle |
| Nemotron Super: `Error loading reasoning parser` | Run the `wget` for `super_v3_reasoning_parser.py` and start `docker run` from that directory (playbook) |
| Nemotron: `curl` returns model / 404 errors | Use the served name: `nemotron_3_nano_omni` (Nano) or `nemotron-3-super` (Super on vLLM) |

## Next

Continue to [Lab 07 — NVFP4 quantization and speculative decoding](../07_nvfp4_speculative/TUTORIAL.md): quantize a model to NVFP4 yourself, and speed up decoding with a draft model or MTP.
