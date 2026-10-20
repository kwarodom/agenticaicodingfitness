# ▶ Spark Lab 04 — llama.cpp + LM Studio: GGUF and quantized models

> Part of Week 25 · DGX Spark: fine-tune, serve, and build sandboxed agents. You type the commands, you see the real output. Every lab also runs in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's playbook, or a clearly marked EXAMPLE.

**What you'll actually do**
- Build llama.cpp from source with CUDA for the GB10 (`CMAKE_CUDA_ARCHITECTURES=121a-real`), exactly as the playbook does.
- Open a real GGUF file and see what `Q4_K_M` stores, tensor by tensor, down to the byte.
- Quantize a million weights yourself and measure what each bit you remove costs.
- Serve the playbook's `Qwen3.6-35B-A3B` GGUF with `llama-server` on port **30080** and call its OpenAI API.
- Run LM Studio headless (`llmster` and the `lms` CLI) on port 1234.
- Decide when to use llama.cpp, Ollama, LM Studio or vLLM.

**Time** ~50 min · **Difficulty** intermediate · **Hardware** 1 DGX Spark (or none: DRY mode + arithmetic labs + an optional laptop llama-server)

**Official playbooks covered:** [llama.cpp](https://build.nvidia.com/spark/llama-cpp) · [LM Studio](https://build.nvidia.com/spark/lm-studio)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Module 03 done | you can call Ollama through `/v1` | this module reuses the same OpenAI client ideas |
| Build tools on the Spark | `git --version`, `cmake --version` (3.14+), `nvcc --version` | the playbook builds llama.cpp from source |
| ~40 GB free disk on the Spark | `df -h /` | the playbook: "at least ~40 GB free disk for the example download plus build artifacts" |
| ~30 GB free memory on the Spark | `free -g` | the playbook: "about 30 GB free for the example model" |
| numpy in the repo `.venv` | `.venv/bin/python -c "import numpy"` | lab 03 quantizes a million weights |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
.venv/bin/python -c "import numpy; print('numpy', numpy.__version__)"
```

**Expected output** (captured on this Mac)

```
numpy 2.4.2
```

> ⚠ **Port change.** The llama.cpp playbook serves on **port 30000**. SGLang (Module 06) also defaults to 30000, so this course runs `llama-server` on **30080** (`sparkkit.PORTS["llamacpp"]`) and the two engines can run side by side. Every command below uses 30080. When you read the playbook, replace 30000 with 30080.

✓ Checkpoint: numpy imports, and you know which port this course uses for llama-server and why.

## 1 · llama.cpp, Ollama and LM Studio: one engine, three wrappers

[llama.cpp](https://github.com/ggml-org/llama.cpp) is a C/C++ inference engine. Its tensor library, **GGML**, defines the **GGUF** file format: one file that holds the weights (usually quantized), the tokenizer, the chat template and the model's settings. The playbook's words: "a lightweight C/C++ inference stack … load GGUF weights and expose chat through `llama-server`'s OpenAI-compatible HTTP API."

Ollama (Module 03) and LM Studio both run GGUF models and both put a friendlier layer on top:

| | llama.cpp (`llama-server`) | Ollama | LM Studio (`llmster` / `lms`) |
|---|---|---|---|
| You install | build from source (this module) | one script | one script |
| Models come from | any GGUF on Hugging Face (`-hf org/repo:QUANT`) | the Ollama library (`ollama pull`) | the LM Studio catalog (`lms get`) |
| Model file | the plain `.gguf`, in `~/.cache/huggingface/hub` | GGUF blobs in Ollama's own store | files under `~/.lmstudio/models/` |
| One server holds | one model per process (by default) | many, loaded and unloaded on demand | the models you `lms load` |
| Knobs | every flag: context, speculative decoding, GPU layers, slots | a few env vars and a Modelfile | CLI flags and a GUI |
| Default port | 8080 (playbook: 30000, **course: 30080**) | 11434 | 1234 |
| API | OpenAI `/v1` + `/health` + `timings` | native `/api` + OpenAI `/v1` | OpenAI `/v1` + LM Studio REST `/api/v1` |

A GGUF from one tool does not always load in another. This Mac's Ollama copy of `gemma3:4b` is a valid GGUF (lab 02 reads it), but when we pointed Homebrew's `llama-server` at it, it stopped with:

```
E llama_model_load: error loading model: error loading model hyperparameters: key not found in model: gemma3.attention.layer_norm_rms_epsilon
```

Ollama packages some models its own way. Download GGUFs for llama.cpp from Hugging Face, as the playbook does.

✓ Checkpoint: you can say what a GGUF file contains, and why the same model can have different files in Ollama and on Hugging Face.

## 2 · Build llama.cpp for CUDA on the GB10

The playbook's steps 1–3, on the **Spark**:

```bash
# on: spark
sudo apt update
sudo apt install -y git clang cmake libcurl4-openssl-dev libssl-dev
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp
cmake -B build -DGGML_NATIVE=ON -DGGML_CUDA=ON -DGGML_CURL=ON -DGGML_RPC=ON -DCMAKE_CUDA_ARCHITECTURES=121a-real
cmake --build build --config Release --target llama-server -j
```

What the flags mean:

| Flag | Why |
|---|---|
| `-DGGML_CUDA=ON` | build the CUDA backend, so the GB10 GPU does the work |
| `-DCMAKE_CUDA_ARCHITECTURES=121a-real` | compile for the GB10's compute capability 12.1 (sm_121). The playbook's troubleshooting: "Build errors mentioning wrong GPU arch → use `-DCMAKE_CUDA_ARCHITECTURES=121a-real`" |
| `-DGGML_NATIVE=ON` | tune the CPU code for this machine's Arm cores |
| `-DGGML_CURL=ON` | lets `-hf` download models from Hugging Face |
| `-DGGML_RPC=ON` | builds llama.cpp's RPC backend (for spreading a model across machines; not used in this course) |
| `--target llama-server` | build only the server, not every tool |

The playbook says the build "usually takes on the order of 5–10 minutes". Lab 01 checks the tools, then (with `--build`) runs the clone and build **in the background** with a log, so the Lab Runner's 15-minute limit cannot kill it. The `sudo apt` line is printed for you to type; labs never use sudo.

```bash
# on: laptop
.venv/bin/python week25/04_llama_cpp_lm_studio/labs/lab01_build_llama_cpp.py            # read-only checks
.venv/bin/python week25/04_llama_cpp_lm_studio/labs/lab01_build_llama_cpp.py --build    # start the build
```

**Expected output** (DRY mode, captured on this Mac without `--build`)

```
▣ STEP 1 · build tools: git, cmake ≥ 3.14, the CUDA compiler, an Arm CPU
$ uname -m && git --version && cmake --version | head -1 && (nvcc --version || /usr/local/cuda/bin/nvcc --version) | tail -1   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
aarch64
git version 2.43.0
cmake version 3.28.3
Build cuda_13.0.r13.0/compiler.xxxxxxxx_0
◆ nvcc not found? The playbook's fix: export PATH=/usr/local/cuda/bin:$PATH, then run CMake again from a clean build dir.
…
▣ STEP 3 · clone and build llama-server with CUDA for GB10 (sm_121)
→ add --build to run, in the background:
  git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp && cd ~/llama.cpp
  cmake -B build -DGGML_NATIVE=ON -DGGML_CUDA=ON -DGGML_CURL=ON -DGGML_RPC=ON -DCMAKE_CUDA_ARCHITECTURES=121a-real
  cmake --build build --config Release --target llama-server -j
```

Follow a running build from the laptop:

```bash
# on: laptop
ssh spark-a tail -f ~/w25/logs/llama-build.log
```

When it finishes, `llama-server` is at `~/llama.cpp/build/bin/llama-server`.

✓ Checkpoint: `ssh spark-a ls ~/llama.cpp/build/bin/llama-server` finds the binary, or in DRY mode you can explain what `121a-real` selects.

## 3 · GGUF inside out: what `Q4_K_M` really stores

GGML quantizes weights in **blocks**. Each block stores a few integers per weight plus one or two small scales, so the real cost per weight is a little above the integer width:

| GGML type | weights per block | bytes per block | bits per weight | what is in a block |
|---|---|---|---|---|
| `F16` / `BF16` | 1 | 2 | 16 | no quantization |
| `Q8_0` | 32 | 34 | 8.5 | 32 × int8 + one fp16 scale |
| `Q6_K` | 256 | 210 | 6.5625 | 6-bit values + 8-bit sub-scales + fp16 scale |
| `Q5_K` | 256 | 176 | 5.5 | 5-bit values + 6-bit sub-scales + two fp16 scales |
| `Q4_K` | 256 | 144 | 4.5 | 4-bit values + 6-bit sub-scales + two fp16 scales |
| `Q4_0` | 32 | 18 | 4.5 | 32 × 4-bit + one fp16 scale |
| `Q3_K` | 256 | 110 | 3.4375 | 3-bit values + sub-scales |
| `Q2_K` | 256 | 84 | 2.625 | 2-bit values + 4-bit sub-scales |

A name like **`Q4_K_M` is a recipe, not a type**: most tensors get `Q4_K`, the most sensitive ones (such as half of the `ffn_down` layers and the embeddings) get `Q6_K`, and tiny ones stay `F32`. You do not have to take this on trust. Lab 02 reads a real GGUF header, adds up every tensor's bytes from the table above, and checks the total against the file size:

```bash
# on: laptop
.venv/bin/python week25/04_llama_cpp_lm_studio/labs/lab02_gguf_inside.py
```

With no argument it opens the smallest GGUF that Ollama has downloaded on your machine. It reads only the header (a few MB), never the weights.

**Expected output** (captured on this Mac, reading Ollama's `gemma3:4b` file)

```
▣ STEP 1 · read the header of gemma3:4b
│ GGUF version  3 · 35 metadata keys · 883 tensors · header 15.8 MB (tokenizer included)
│ architecture  gemma3 · layers 34 · context 131072
│ file_type     15 → Q4_K_M   (what the file calls itself)

▣ STEP 2 · what each tensor is actually stored as
│ type  tensors  weights  bytes     bits/weight  share of file         e.g.
│ ────  ───────  ───────  ────────  ───────────  ────────────────────  ─────────────────────────────
│ Q4_K  205      2.721 B  1.531 GB  4.500        46.1% ██████░░░░░░░░  blk.0.attn_k.weight
│ Q6_K  34       1.159 B  0.950 GB  6.562        28.6% ████░░░░░░░░░░  token_embd.weight
│ F16   165      0.419 B  0.839 GB  16.000       25.2% ████░░░░░░░░░░  mm.mm_input_projection.weight
│ F32   479      0.001 B  0.003 GB  32.000        0.1% ░░░░░░░░░░░░░░  mm.mm_soft_emb_norm.weight

▣ STEP 3 · does the arithmetic match the file?
│ sum of tensor bytes from block sizes     3,322,976,704
│ file size − header                       3,322,976,731   (includes ≤ 32-byte alignment padding per tensor)
✓ block sizes × tensor shapes = the file, to within alignment padding
◆ effective bits per weight for the whole file: 6.18 (4.30 B weights, 3.32 GB)

▣ STEP 4 · where the bits go: transformer layers vs embeddings vs vision
│ group                       weights  bytes     bits/weight
│ ──────────────────────────  ───────  ────────  ───────────
│ transformer layers (blk.*)  3.209 B  1.932 GB  4.82
│ vision tower (v.*, mm.*)    0.420 B  0.840 GB  16.02
│ embeddings, output, norms   0.671 B  0.551 GB  6.56
◆ A K-quant recipe keeps most layer weights at its base type (Q4_K = 4.5 bits for Q4_K_M) and gives the most sensitive ones more (Q6_K = 6.56 bits). Tiny tensors (norms) stay F32.
◆ This file also carries a vision encoder, stored at F16: it counts toward the download and memory, but text generation does not read it.
```

Three things this real file teaches:

1. **The table is exact.** 883 tensors, and the computed bytes miss the file size by 27 bytes, which is alignment padding.
2. **"Q4" is 4.82 bits in the layers and 6.18 bits for the whole file.** The layers are close to sparkkit's 4.85-bit average for `Q4_K_M`. The whole file is heavier because Ollama's gemma3 packs a 0.84 GB F16 vision encoder into it. Size downloads by the whole file; judge quality by the layers.
3. **Small models break the recipe.** We also pointed lab 02 at Qwen's own `qwen2.5-0.5b-instruct-q4_k_m.gguf` (`--path FILE.gguf`). There, 133 of 170 quantized tensors are `Q5_0`, not `Q4_K`. K-quants pack 256 weights per super-block, and that model's rows are 896 wide (not a multiple of 256), so the quantizer falls back to a 32-weight type. The lab says so when it happens.

On the Spark, run it on the playbook's download once lab 01 `--serve` has fetched it:

```bash
# on: spark
cd ~/agenticaicodingfitness      # wherever you cloned this repo on the Spark
python3 week25/04_llama_cpp_lm_studio/labs/lab02_gguf_inside.py \
  --path '~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/*/*.gguf'
```

✓ Checkpoint: lab 02's step 3 prints ✓, and you can explain why a `Q4_K_M` file averages more than 4.5 bits per weight.

## 4 · How many bits? Size against error

Fewer bits mean a smaller file, more room for KV cache, and faster decode (fewer bytes read per token, Module 01). The price is rounding error. Lab 03 measures that price with ggml's own `Q8_0`, `Q4_0` and `Q4_1` recipes on a million synthetic weights (Gaussian, with 0.1% outliers, as real checkpoints have):

```bash
# on: laptop
.venv/bin/python week25/04_llama_cpp_lm_studio/labs/lab03_quant_tradeoffs.py
```

**Expected output** (arithmetic with a fixed random seed: the same on every machine)

```
▣ STEP 1 · quantize 1,048,576 weights with each recipe and measure the error
│ recipe  bits/w  size      rel. error  SNR                           how
│ ──────  ──────  ────────  ──────────  ────────  ──────────────────  ─────────────────────────────────────────────
│ F16     16       2.10 MB    0.02%      73.7 dB  ░░░░░░░░░░░░░░░░░░  half precision, no blocks
│ Q8_0    8.5      1.11 MB    0.62%      44.1 dB  ░░░░░░░░░░░░░░░░░░  ggml Q8_0: 32 × int8 + fp16 scale
│ sym-6   6.5      0.85 MB    2.56%      31.9 dB  █░░░░░░░░░░░░░░░░░  same recipe, 6-bit values
│ Q4_1    5        0.66 MB    8.21%      21.7 dB  ██░░░░░░░░░░░░░░░░  ggml Q4_1: 32 × 4-bit + fp16 scale + fp16 min
│ Q4_0    4.5      0.59 MB    9.91%      20.1 dB  ███░░░░░░░░░░░░░░░  ggml Q4_0: 32 × 4-bit + fp16 scale
│ sym-3   3.5      0.46 MB   24.14%      12.3 dB  ███████░░░░░░░░░░░  same recipe, 3-bit values
│ sym-2   2.5      0.33 MB   63.88%       3.9 dB  ██████████████████  same recipe, 2-bit values

▣ STEP 2 · why blocks: one outlier ruins the scale for everyone who shares it
│ weights per scale  bits/w  rel. error  SNR
│ ─────────────────  ──────  ──────────  ────────  ──────────────────
│ 32                 4.500    11.18%      19.0 dB  ███░░░░░░░░░░░░░░░
│ 256                4.062    20.97%      13.6 dB  ██████░░░░░░░░░░░░
│ 4,096              4.004    55.15%       5.2 dB  █████████████████░
│ whole tensor       4.000    92.33%       0.7 dB  ██████████████████

▣ STEP 3 · what it means for the playbook's model: Qwen3.6-35B-A3B on one Spark
│ quant   bits/w  weights   left of 128 GB  decode ceiling  share of memory
│ ──────  ──────  ────────  ──────────────  ──────────────  ────────────────
│ BF16    16       70.0 GB   58.0 GB           46 tok/s     █████████░░░░░░░
│ Q8_0    8.5      37.2 GB   90.8 GB           86 tok/s     █████░░░░░░░░░░░
│ Q6_K    6.6      28.9 GB   99.1 GB          110 tok/s     ████░░░░░░░░░░░░
│ Q5_K_M  5.7      24.9 GB  103.1 GB          128 tok/s     ███░░░░░░░░░░░░░
│ Q4_K_M  4.85     21.2 GB  106.8 GB          150 tok/s     ███░░░░░░░░░░░░░
│ Q3_K_M  3.9      17.1 GB  110.9 GB          187 tok/s     ██░░░░░░░░░░░░░░
│ Q2_K    2.625    11.5 GB  116.5 GB          277 tok/s     █░░░░░░░░░░░░░░░
◆ The playbook's pick is unsloth's UD-Q4_K_XL (an Unsloth 'dynamic' ~4-bit recipe that keeps more tensors at higher precision). The playbook budgets about 30 GB of memory and a ~35 GB-order download for it.
◆ On a Spark, even BF16 of this 35B model fits. You quantize here for SPEED (fewer bytes per token) and for room: KV cache for long contexts and several models loaded side by side.
```

How to read it:

- **About 6 dB per bit.** Every bit you drop roughly doubles the error. From 8 to 4 bits the error goes from 0.6% to about 10%; below 4 bits it climbs fast.
- **Blocks are why 4-bit works at all.** With one scale for the whole tensor, the outliers stretch the scale and 92% of the signal is lost. A 32-weight block costs half a bit and brings the error down to 11%.
- **On a 128 GB Spark, quantization is about speed and room, not fit.** For the playbook's 35B MoE model, BF16 fits, but Q4_K_M reads about 3.3× fewer bytes per token, so its decode ceiling is ~150 tok/s instead of ~46.

> ⚠ This lab measures how well the **numbers** survive, not how well the **model** answers. Quality depends on the model and the task. The [Run local LLMs playbook](https://build.nvidia.com/spark/llms) puts it this way: "Aggressive quantization can reduce response quality. NVFP4 or Q4_K_M are a good balance of throughput, accuracy, and memory." Module 07 covers NVFP4 and Module 13 measures quality on a real model.

✓ Checkpoint: you can say, from the table, why Q8_0 is a safe default when memory allows and why Q2_K is a last resort.

## 5 · Serve a GGUF with llama-server on :30080

The playbook serves the MTP-enabled Qwen3.6-35B-A3B GGUF. `-hf org/repo:QUANT` downloads the file into `~/.cache/huggingface/hub` on first use (and loads a vision `mmproj` file automatically when the model has one). From `~/llama.cpp/build`, with the course port:

```bash
# on: spark
cd ~/llama.cpp/build
./bin/llama-server \
  -hf unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL \
  --host 0.0.0.0 \
  --port 30080
```

The playbook's faster variant adds **MTP speculative decoding** (Multi-Token Prediction: the model drafts several tokens and verifies them in one pass; Module 07 explains why this speeds up decode) and keeps Qwen's thinking blocks in the history for agent sessions:

```bash
# on: spark
cd ~/llama.cpp/build
./bin/llama-server \
  -hf unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL \
  --host 0.0.0.0 \
  --port 30080 \
  --chat-template-kwargs '{"preserve_thinking": true}' \
  --spec-type draft-mtp \
  --spec-draft-n-max 3
```

Keep that terminal open, or let lab 01 start it in the background (`--serve`, log in `~/w25/logs/llama-server.log`). The server is ready when the log says it is listening:

**Expected output** (REFERENCE — quoted from the llama.cpp playbook; the playbook's run uses port 30000, yours says 30080)

```
0.14.322.968 I srv    load_model: speculative decoding context initialized
0.14.322.970 I slot   load_model: id  0 | task -1 | new slot, n_ctx = 262144
0.14.322.972 I slot   load_model: id  1 | task -1 | new slot, n_ctx = 262144
0.14.322.972 I slot   load_model: id  2 | task -1 | new slot, n_ctx = 262144
0.14.322.973 I slot   load_model: id  3 | task -1 | new slot, n_ctx = 262144
0.14.323.063 I srv    load_model: prompt cache is enabled, size limit: 8192 MiB

...
0.14.342.935 I srv  llama_server: model loaded
0.14.342.939 I srv  llama_server: server is listening on http://0.0.0.0:30000
0.14.342.944 I srv  update_slots: all slots are idle
```

Four **slots** means four requests can run at once, each with a 262,144-token context: the playbook notes that llama-server "tries to fit the full model context with the ability to serve 4 concurrent requests". For agents and coding it recommends at least 32,768 tokens (`--ctx-size` / `-c`), preferably 100,000 or more. Lower `-c` if you hit out-of-memory.

Test it the playbook's way (on the Spark, with the course port):

```bash
# on: spark
timeout 900 bash -c 'until curl -sf http://127.0.0.1:30080/health > /dev/null 2>&1; do sleep 5; done' || exit 1
curl -X POST http://127.0.0.1:30080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL",
    "messages": [{"role": "user", "content": "New York is a great city because..."}],
    "max_tokens": 100
  }'
```

`--host 0.0.0.0` makes the server reachable from your LAN and tailnet (`http://spark-a:30080/v1`), with no authentication. From the laptop you can also tunnel it: `ssh -N -L 30080:localhost:30080 spark-a`.

Lab 04 makes the same call from Python. It uses the Spark's llama-server when it answers, otherwise one running on your laptop on port 30080, otherwise the laptop's Ollama. To try the middle option, we ran a Homebrew `llama-server` on this Mac with a small GGUF from Hugging Face (`qwen2.5-0.5b-instruct-q4_k_m.gguf`, started with `--alias qwen2.5-0.5b-instruct-q4_k_m --port 30080 -c 4096`):

```bash
# on: laptop
.venv/bin/python week25/04_llama_cpp_lm_studio/labs/lab04_llama_server_lm_studio.py
```

**Expected output** (LAPTOP STAND-IN — a 0.5B model on this Mac; these speeds say nothing about the Spark)

```
▣ STEP 1 · llama-server — where does :30080 answer?
● http://localhost:30080/v1 · LAPTOP STAND-IN (llama-server on this laptop, not the Spark)
→ GET http://localhost:30080/health → {'status': 'ok'}
→ GET http://localhost:30080/v1/models → ['qwen2.5-0.5b-instruct-q4_k_m']

▣ STEP 2 · one chat completion, and the server's own `timings`
→ POST http://localhost:30080/v1/chat/completions  {"model": "qwen2.5-0.5b-instruct-q4_k_m", "max_tokens": 100}
· ANSWER  New York is a great city for many reasons, but one of the most significant and enduring is its vibrant cultural scene and rich history. The city boasts a diverse array of neighborhoods and cultural institutions, including museums, theaters, and galleries, all of which attract both locals and tourist
◆ usage {'completion_tokens': 100, 'prompt_tokens': 37, 'total_tokens': 137, 'prompt_tokens_details': {'cached_tokens': 0}} · finish_reason=length · 0.37 s wall
│ timings (llama.cpp only)  tokens  time    rate
│ ────────────────────────  ──────  ──────  ───────────
│ prefill                   37      29 ms   1265 tok/s
│ decode                    100     320 ms  309.4 tok/s

▣ STEP 3 · the same request, streamed (sparkkit.chat measures TTFT on the client)
→ POST http://localhost:30080/v1/chat/completions  model=qwen2.5-0.5b-instruct-q4_k_m  stream=True
· ANSWER  New York is a great city because it offers a unique blend of history, culture, and innovation. …
◆ LAPTOP STAND-IN (not Spark numbers) · qwen2.5-0.5b-instruct-q4_k_m · TTFT 11 ms · 100 tok in 0.4s · 241.2 tok/s
```

The response has the OpenAI shape plus a **`timings`** block that only llama.cpp adds: prefill and decode speed as the server measured them (the playbook: "fields vary by llama.cpp version"). `finish_reason=length` means it stopped at `max_tokens`, as in the playbook's example response. Try it inline: this ⚡ block goes to the Spark's llama-server, or to the laptop stand-in when the Spark is not connected.

```spark
{"target": "llamacpp", "which": "a", "model": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL",
 "messages": [{"role": "user", "content": "New York is a great city because..."}],
 "max_tokens": 100}
```

✓ Checkpoint: `curl -sf http://127.0.0.1:30080/health` on the Spark returns OK, and you can point to the `timings` block in a response.

## 6 · LM Studio headless: llmster and `lms` on port 1234

LM Studio is usually a desktop app. The [LM Studio playbook](https://build.nvidia.com/spark/lm-studio) installs **llmster**, its headless daemon, on the Spark and drives it with the `lms` CLI. It serves an OpenAI-compatible API and LM Studio's own REST API on port **1234**.

```bash
# on: spark
curl -fsSL https://lmstudio.ai/install.sh | bash      # then follow the printed hint to add lms to your PATH
lms server start --bind 0.0.0.0 --port 1234
lms get nvidia/nemotron-3-nano-omni                   # the playbook's example model
lms ls
lms load nvidia/nemotron-3-nano-omni
```

The playbook asks for "minimum 65 GB memory for model inference (70 GB or above recommended for the example model)" and the same on disk, so check `free -g` first, or unload other servers. Its validated models:

| Model | `lms` path | Note |
|---|---|---|
| Nemotron 3 Nano Omni | `nvidia/nemotron-3-nano-omni` | the playbook's example |
| Qwen3.6-35B-A3B | `qwen/qwen3.6-35b-a3b` | the playbook's "agent-ready" pick; load with `--context-length 65536` for long agent sessions |
| GPT-OSS-120B | `openai/gpt-oss-120b` | |

Check it from the laptop (the playbook's command; find the Spark's address with `hostname -I` on the Spark, or use your tailnet name):

```bash
# on: laptop
curl http://spark-a:1234/api/v1/models
```

Lab 04, step 4, does the same probe and chats with the first loaded model. With no LM Studio running anywhere, it prints the command to start one:

**Expected output** (captured on this Mac, which runs no LM Studio)

```
▣ STEP 4 · LM Studio (llmster) — where does :1234 answer?
○ no LM Studio server on the Spark or on localhost:1234
$ lms ls   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
(the models you downloaded with `lms get …`, with their sizes)
→ start it on the Spark (playbook step 3): lms server start --bind 0.0.0.0 --port 1234
```

**LM Link** (optional, in the playbook) links the Spark and your laptop over an end-to-end encrypted Tailscale-based network, so models on the Spark appear at `localhost:1234` on the laptop without binding to `0.0.0.0`. It needs an account at lmstudio.ai/link and is free in preview for up to 2 users.

Cleanup, from the playbook: stop the server, remove `~/.lmstudio/llmster` to uninstall, and delete `~/.lmstudio/models/` to free the disk.

✓ Checkpoint: `curl http://spark-a:1234/api/v1/models` lists your loaded model, or you can explain why the playbook needs 65 GB for its example.

## 7 · Which engine when?

You now have three ways to serve a GGUF, and Module 05 adds vLLM. This is the course's rule of thumb (not an NVIDIA ranking):

| You want | Pick | Because |
|---|---|---|
| a model running in two minutes, many models to try | **Ollama** | one pull command, auto load/unload, the laptop and the Spark behave the same |
| a specific GGUF from Hugging Face, with every knob | **llama.cpp** | any quant, `-c`, MTP speculative decoding, slots, `timings`; one binary you built |
| a GUI, a model catalog, or remote use without opening ports | **LM Studio** | `lms` plus LM Link's encrypted link |
| many users at once, FP8 / NVFP4 weights, two Sparks | **vLLM** (Module 05) | continuous batching and paged KV cache; tensor parallel across Sparks |
| the last bit of speed for one model | **TensorRT-LLM / SGLang** (Module 06) | compiled kernels, the engine bake-off |

They all speak `/v1/chat/completions`, so the client code from Module 03 does not change. Only the base URL and the model id do. Module 08 puts LiteLLM in front so even those stop mattering.

✓ Checkpoint: for each of your own use cases (one chat user, an agent with tools, a team of ten), you can name an engine and say why.

## Labs — run them here

**labs/lab01_build_llama_cpp.py** — Check the build tools, then build llama.cpp for CUDA and serve on :30080 (both opt-in).

**labs/lab02_gguf_inside.py** — Read a real GGUF header, count its quant types, and prove the file size from GGML's block sizes.

**labs/lab03_quant_tradeoffs.py** — Quantize a million weights, measure the error per bit, and size the playbook's model per quant.

**labs/lab04_llama_server_lm_studio.py** — Call llama-server and LM Studio through their OpenAI APIs and read llama.cpp's `timings`.

Lab 01 runs LIVE on your Spark or DRY. Labs 02 and 03 run on the laptop (lab 02 also on the Spark with `--path`). Lab 04 calls the Spark's servers, else servers on your laptop (labelled), else the laptop's Ollama.

## Try it yourself

**Exercise 04 — pick a quant that fits.** Open `week25/04_llama_cpp_lm_studio/exercises/ex04_pick_a_quant.py`. Three `TODO`s:

1. `bits_per_weight(block_bytes, block_weights)`: the cost of one GGML block type.
2. `gguf_gb(params_b, bpw)`: the size of a GGUF file.
3. `pick_quant(params_b, budget_gb)`: the highest-precision quant that fits a memory budget.

```bash
# on: laptop
.venv/bin/python week25/04_llama_cpp_lm_studio/exercises/ex04_pick_a_quant.py
```

**Expected output** (once all three TODOs are done)

```
✓ bits_per_weight: Q8_0 8.5 · Q6_K 6.5625 · Q4_K 4.5 · Q2_K 2.625
✓ gguf_gb: Qwen3.6-35B-A3B at Q4_K_M ≈ 21.2 GB · 8B at BF16 = 16 GB
✓ pick_quant: 35B→Q8_0 · 70.6B→Q6_K · 405B→None · 8B→BF16

▣ your picker, applied: 1 Spark keeps 60 GB free for KV cache and a second model; 2 Sparks keep 10 GB
│ Qwen3.6-35B-A3B  on 1 Spark   budget  68 GB → Q8_0           37.2 GB
│ Llama 3.3 70B    on 1 Spark   budget  68 GB → Q6_K           57.9 GB
│ Llama 3.1 405B   on 1 Spark   budget  68 GB → does not fit    —
│ Llama 3.1 405B   on 2 Sparks  budget 246 GB → Q4_K_M        245.5 GB
```

<details><summary>Hint — sorting by quality</summary>

"Highest precision" means the largest bits per weight. Sort `quants.items()` by the value, largest first, and return the first name whose `gguf_gb()` is within the budget.

</details>

<details><summary>Stretch — add the KV cache</summary>

Import `kv_cache_gb` from sparkkit and subtract a 32K-context KV cache for Llama 3.3 70B (80 layers, 8 KV heads, head dim 128) from the budget before you pick. Does the 70B still get Q6_K on one Spark with 4 concurrent users? This is the question llama-server's slots and `-c` answer for you.

</details>

✓ Checkpoint: all three checker lines are ✓.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cmake` fails with "CUDA not found" | `export PATH=/usr/local/cuda/bin:$PATH`, then re-run CMake from a clean build directory (`rm -rf build` first) |
| Build errors mentioning the GPU architecture | use `-DCMAKE_CUDA_ARCHITECTURES=121a-real`, as in Section 2 |
| `curl: (7) Failed to connect` on port 30080 | still loading, crashed, or wrong host. Wait for "listening", run `ss -tln \| grep 30080`, read `~/w25/logs/llama-server.log` for OOM or path errors |
| You followed the playbook and nothing answers on 30080 | the playbook uses `--port 30000`. Use `--port 30080` in this course, or set `SPARK_URL_LLAMACPP=http://spark-a:30000/v1` |
| "CUDA out of memory" when llama-server starts | lower the context (`-c 32768`, or `-c 4096` to test) or pick a smaller quant from the same repo. Stop Ollama or LM Studio models you are not using |
| GGUF download stalls | re-run the same `-hf` command: it resumes partial files |
| `key not found in model` when loading an Ollama blob | Ollama packages some models its own way. Download the GGUF from Hugging Face instead (Section 1) |
| `lms: command not found` | `source ~/.bashrc` (or the file the installer named), as the playbook says |
| LM Studio "model not found" | `lms ls` to check the download, then `lms load <model>` |
| Memory pressure although the model should fit | the playbooks' UMA note: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |

## Next

Continue to [Lab 05 — vLLM: high-throughput serving](../05_vllm/TUTORIAL.md): serve many users at once with continuous batching, call tools through vLLM, and split one model across two Sparks.
