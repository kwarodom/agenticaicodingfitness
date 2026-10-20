# ▶ Spark Lab 03 — Ollama + Open WebUI: your first model server

> Part of Week 25 · DGX Spark: fine-tune, serve, and build sandboxed agents. You type the commands, you see the real output. Every lab also runs in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's playbook, or a clearly marked EXAMPLE.

**What you'll actually do**
- Install Ollama on your Spark, pull the playbooks' first model (`gpt-oss:20b`), and see where models live.
- Decide who can reach port 11434: localhost only, an SSH tunnel, or `OLLAMA_HOST=0.0.0.0`.
- Call the same server two ways: Ollama's native `/api/chat` and the OpenAI-compatible `/v1/chat/completions`.
- Tame thinking models: read the `reasoning` field, and switch it off with `reasoning_effort: "none"`.
- Measure time-to-first-token and tokens/second, and compare them with Module 01's bandwidth ceiling.
- Run a full tool-calling loop, then put a chat UI on top with Open WebUI.

**Time** ~50 min · **Difficulty** beginner · **Hardware** 1 DGX Spark (or none: DRY mode + the laptop's Ollama as a labelled stand-in)

**Official playbooks covered:** [Open WebUI with Ollama](https://build.nvidia.com/spark/open-webui) · [Ollama](https://build.nvidia.com/spark/ollama) · [Run local LLMs](https://build.nvidia.com/spark/llms)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Module 01 done | `ssh -o BatchMode=yes spark-a true` | labs send commands to the Spark over SSH |
| Docker without sudo on the Spark | `docker ps > /dev/null` prints nothing | Open WebUI runs as a container (Section 7) |
| ~40 GB free disk on the Spark | `df -h /` | ~15 GB for `gpt-oss:20b`, ~7 GB for the Open WebUI image, plus an agent model |
| Optional: Ollama on your laptop | `curl -s localhost:11434/api/version` | lets the HTTP labs run for real without a Spark, labelled LAPTOP STAND-IN |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
curl -s localhost:11434/api/version; echo
.venv/bin/python week25/common/sparkkit.py | tail -2
```

**Expected output** (captured on this Mac, which has Ollama but no Spark configured)

```
{"version":"0.34.4"}
│ litellm   (no host)                                  ○ down
│ laptop    http://localhost:11434/v1                  nemotron-3.5-lightning:latest, nemotron-3-nano:latest, gemma3:4b, kimi-k2.7-code:cloud
```

> 💡 **The laptop stand-in, and what it is not.** When the Spark's Ollama is not reachable, labs 02–04 and the ⚡ blocks use Ollama on your laptop. The answers, token counts and tool calls are real. The *speed* is your laptop's, so every such line says `LAPTOP STAND-IN`. Never compare it with a Spark number.

✓ Checkpoint: `ssh spark-a true` works (or you have decided to follow along DRY), and you know whether your laptop runs Ollama.

## 1 · Install Ollama on the Spark and pull a model

Ollama is a model server built on llama.cpp's GGML engine (Module 04). It downloads models from its own library as ready-to-run GGUF files, keeps them in one store, loads them on demand, and serves them on port **11434**. It is the fastest way from "empty Spark" to "I can call a model".

The [Ollama playbook](https://build.nvidia.com/spark/ollama) checks for an existing install first, then uses the official script:

```bash
# on: spark
ollama --version
curl -fsSL https://ollama.com/install.sh | sh
```

The script installs the `ollama` binary and a **systemd service** called `ollama` that starts at boot. (You can see this in the playbook's uninstall steps: `sudo systemctl stop ollama`, `sudo rm /usr/local/bin/ollama`, `sudo userdel ollama`.)

Which model first? The playbooks name three, and this course uses two of them:

| Model | Where the playbook uses it | Size on disk | This course |
|---|---|---|---|
| `gpt-oss:20b` | Open WebUI playbook, first model | about 15 GB | the default model in this module's labs |
| `qwen3.6:35b-a3b` | Open WebUI playbook, "agent-ready" pick for DGX Spark | — | tool calling on the Spark (lab 04) |
| `qwen2.5:32b` | older Ollama playbook | 18 GB | not used |

```bash
# on: spark
ollama pull gpt-oss:20b
ollama list
```

The playbook shows what a pull looks like for its own example model:

**Expected output** (REFERENCE — quoted from the Ollama playbook, for `ollama pull qwen2.5:32b`)

```
pulling manifest
pulling 58574f2e94b9: 100% ████████████████████████████  18 GB
pulling 53e4ea15e8f5: 100% ████████████████████████████ 1.5 KB
pulling d18a5cc71b84: 100% ████████████████████████████  11 KB
pulling cff3f395ef37: 100% ████████████████████████████  120 B
pulling 3cdc64c2b371: 100% ████████████████████████████  494 B
verifying sha256 digest
writing manifest
success
```

One big layer is the GGUF weights. The small ones are the chat template, license and default parameters. Lab 01 runs these checks for you (and starts the pull in the background with `--pull`, so the Lab Runner's 15-minute limit does not kill it):

```bash
# on: laptop
.venv/bin/python week25/03_ollama_open_webui/labs/lab01_ollama_models.py
```

**Expected output** (captured on this Mac: the Spark steps are DRY, step 5 asks the laptop's Ollama)

```
▣ STEP 2 · which models has it pulled?
$ ollama list   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
NAME           ID              SIZE     MODIFIED
gpt-oss:20b    <12-hex-id>     <size>   <when>
→ add --pull to start `ollama pull gpt-oss:20b` in the background (~15 GB download).
…
▣ STEP 5 · ask the native API what each model really is
◆ the Spark's Ollama is not reachable over HTTP (no host); open a tunnel or set OLLAMA_HOST on the Spark (Section 2).

→ GET http://localhost:11434/api/tags  +  POST /api/show for each model   [THIS laptop — LAPTOP STAND-IN, not the Spark]
│ model                          on disk  params  quant   context    kind      capabilities
│ ─────────────────────────────  ───────  ──────  ──────  ─────────  ────────  ──────────────────────────────
│ nemotron-3.5-lightning:latest  25.4 GB  32.9B   Q4_K_M  1,048,576  MoE ×128  tools, thinking
│ nemotron-3-nano:latest         24.3 GB  31.6B   Q4_K_M  1,048,576  MoE ×128  tools, thinking
│ gemma3:4b                      3.3 GB   4.3B    Q4_K_M  131,072    dense     vision
│ gemma4:12b                     7.6 GB   11.9B   Q4_K_M  262,144    dense     vision, audio, tools, thinking
│ gemma4:latest                  9.6 GB   8.0B    Q4_K_M  131,072    dense     vision, audio, tools, thinking
```

Read the last table like a spec sheet. **quant** `Q4_K_M` is the GGUF format Module 04 takes apart. **capabilities** says which models can call tools (`tools`) and which think out loud (`thinking`). **MoE ×128** means 128 experts, of which only a few run per token, which is why a 31.6B model can be quick.

✓ Checkpoint: `ollama list` on the Spark shows `gpt-oss:20b`, or you can explain what `--pull` in lab 01 does and where its log goes (`~/w25/logs/ollama-pull.log`).

## 2 · Who can reach port 11434: localhost, a tunnel, or OLLAMA_HOST

By default Ollama listens on **127.0.0.1:11434**: only programs on the Spark itself can call it. The Ollama binary documents this in its own help text:

```bash
# on: laptop
ollama serve --help | grep -E 'OLLAMA_(HOST|KEEP_ALIVE|NUM_PARALLEL|MAX_LOADED|CONTEXT_LENGTH)'
```

**Expected output** (captured on this Mac, Ollama 0.34.4; the same version on the Spark prints the same list)

```
      OLLAMA_HOST                   IP Address for the ollama server (default 127.0.0.1:11434)
      OLLAMA_CONTEXT_LENGTH         Context length to use unless otherwise specified (default: 4k/32k/256k based on VRAM)
      OLLAMA_KEEP_ALIVE             The duration that models stay loaded in memory (default "5m")
      OLLAMA_MAX_LOADED_MODELS      Maximum number of loaded models per GPU
      OLLAMA_NUM_PARALLEL           Maximum number of parallel requests
```

You have three ways to call the Spark's Ollama from your laptop:

| Way | How | Who else can reach it | Use when |
|---|---|---|---|
| **SSH tunnel** (recommended) | `ssh -N -L 21434:localhost:11434 spark-a` | nobody | always, for labs |
| **NVIDIA Sync custom app** | the Ollama playbook: Name `Ollama Server`, Port `11434`, no launch script | nobody | you use NVIDIA Sync anyway |
| **Bind to all interfaces** | `OLLAMA_HOST=0.0.0.0:11434` on the Spark | every device on your LAN and tailnet, with **no authentication** | a trusted lab network only |

**The tunnel.** Your laptop may already run Ollama on 11434, so forward the Spark's port to **21434** (the same trick as Module 01, lab 03), then point the labs at it:

```bash
# on: laptop
ssh -N -L 21434:localhost:11434 spark-a &
curl -s http://localhost:21434/api/version; echo
export SPARK_URL_OLLAMA=http://localhost:21434/v1     # or set it in 🖥 Spark setup
```

`SPARK_URL_OLLAMA` is only used when `SPARK_HOST` is also set and reachable (LIVE mode). Otherwise the labs fall back to the laptop stand-in, and say so.

**Binding to all interfaces.** This is not in the NVIDIA playbooks. It is the standard systemd override from Ollama's own FAQ. The service reads `OLLAMA_HOST` from its unit file, so exporting it in your shell does nothing:

```bash
# on: spark
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="OLLAMA_HOST=0.0.0.0:11434"\n' | \
  sudo tee /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
ss -tln | grep 11434
```

The last line should now show `*:11434` or `0.0.0.0:11434` instead of `127.0.0.1:11434`. To undo it, delete `override.conf` and run the last two commands again.

> ⚠ Ollama has no API keys. Anyone who can reach 0.0.0.0:11434 can use your models, pull new ones, and delete them. Keep the tunnel, or put LiteLLM with keys in front (Module 08).

The same override file is where the other variables go: `OLLAMA_KEEP_ALIVE=30m` keeps a model loaded between labs, and `OLLAMA_NUM_PARALLEL` lets several requests share one loaded model. Section 5 shows why the first matters.

✓ Checkpoint: `curl -s http://localhost:21434/api/version` answers through your tunnel, and you can say why `export OLLAMA_HOST=…` in a shell does not change the running service.

## 3 · Two APIs, one server

Ollama speaks two HTTP dialects on the same port:

| | Native API | OpenAI-compatible API |
|---|---|---|
| Chat | `POST /api/chat` | `POST /v1/chat/completions` |
| List models | `GET /api/tags` | `GET /v1/models` |
| Model details | `POST /api/show` (template, capabilities, context length) | — |
| Pull / delete | `POST /api/pull`, `DELETE /api/delete` | — |
| Output limit | `"options": {"num_predict": 200}` | `"max_tokens": 200` |
| Thinking switch | `"think": false` | `"reasoning_effort": "none"` |
| Timings | `eval_count`, `eval_duration`, `load_duration` (nanoseconds) | `usage` token counts only |
| Streaming | newline-delimited JSON, on by default | Server-Sent Events (`data: …`), with `"stream": true` |
| Who uses it | the `ollama` CLI, Open WebUI | LiteLLM, NAT, OpenClaw, every OpenAI SDK — the rest of this week |

The playbook's own test calls the native API through the tunnel:

```bash
# on: laptop
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5:32b",
  "messages": [{"role": "user", "content": "Write me a haiku about GPUs and AI."}],
  "stream": false
}'
```

(That is the playbook's command, which assumes NVIDIA Sync forwards the Spark to local port 11434. With the course tunnel, use port 21434 and `gpt-oss:20b`.)

**Expected output** (REFERENCE — quoted from the Ollama playbook; it calls this the "expected response format")

```json
{
  "model": "qwen2.5:32b",
  "created_at": "2024-01-15T12:30:45.123Z",
  "message": {
    "role": "assistant",
    "content": "Silicon power flows\nThrough circuits, dreams become real\nAI awakens"
  },
  "done": true
}
```

A real response carries more than that: timing fields that the OpenAI format does not have. Lab 02 sends one question both ways and prints them:

```bash
# on: laptop
.venv/bin/python week25/03_ollama_open_webui/labs/lab02_native_vs_openai.py
```

**Expected output** (LAPTOP STAND-IN — captured on this Mac with nemotron-3-nano; your numbers will differ)

```
▣ STEP 1 · native API, thinking on (the model's default)
→ POST http://localhost:11434/api/chat  {"model": "nemotron-3-nano:latest", "stream": false, "options": {"num_predict": 200}, "think": true}
· content   '51'
~ thinking  73 chars: 'We need to answer with the number only. 17 * 3 = 51. So output just "51".'…
│ native field          value    meaning
│ ────────────────────  ───────  ─────────────────────────────────────────────────
│ load_duration         36.86 s  loading weights into memory (0 if already loaded)
│ prompt_eval_count     31       input tokens
│ prompt_eval_duration  0.212 s  prefill time
│ eval_count            32       output tokens, thinking included
│ eval_duration         0.607 s  decode time → 52.7 tok/s server-side
│ total_duration        42.22 s  (wall clock seen by this script: 42.23 s)

▣ STEP 2 · OpenAI-compatible API, same question
→ POST http://localhost:11434/v1/chat/completions  {"model": "nemotron-3-nano:latest", "max_tokens": 200}
· content    '51'
~ reasoning  139 chars (Ollama's /v1 puts thinking in a `reasoning` field)
◆ usage: prompt_tokens=31 completion_tokens=52 · 7.89 s wall · no server timings in the OpenAI format
```

Look at `load_duration`: **36.9 of the 42.2 seconds** went to loading the 24 GB model into memory, because other programs on this laptop had loaded different models in the meantime. On the Spark the same thing happens when you switch models. Section 5 comes back to it.

✓ Checkpoint: you can name the native field that gives the server's own tok/s (`eval_count ÷ eval_duration`) and the endpoint the rest of the week's tools use (`/v1/chat/completions`).

## 4 · Thinking models: the reasoning field and `reasoning_effort: "none"`

Models like `gpt-oss`, Qwen3.x, Nemotron 3 and Gemma 4 can **think** before they answer: they write a hidden chain of reasoning, then the answer. Ollama keeps the two apart:

- native API: `message.thinking` next to `message.content`;
- OpenAI API: `message.reasoning` (or `delta.reasoning` when streaming) next to `content`. vLLM and SGLang call the same field `reasoning_content` (Modules 05–06); sparkkit reads both.

Thinking costs tokens, and tokens are time. Step 3 of lab 02 switches it off both ways:

**Expected output** (LAPTOP STAND-IN — same run as above)

```
▣ STEP 3 · thinking off, both ways
→ POST http://localhost:11434/api/chat  {"model": "nemotron-3-nano:latest", "stream": false, "options": {"num_predict": 200}, "think": false}
→ POST http://localhost:11434/v1/chat/completions  {"model": "nemotron-3-nano:latest", "max_tokens": 200, "reasoning_effort": "none"}
│ API         setting                     output tokens  answer
│ ──────────  ──────────────────────────  ─────────────  ──────
│ native      "think": true               32             '51'
│ native      "think": false              3              '51'
│ OpenAI /v1  (default)                   52             '51'
│ OpenAI /v1  "reasoning_effort": "none"  3              '51'
```

Same answer, 3 tokens instead of 32–52. The thinking length also varies from run to run: in two other runs of the same lab on this laptop, first the thinking-on native call and then the default `/v1` call used **all 200 tokens** on reasoning and returned an **empty** `content` (`completion_tokens=200`, answer `''`). That is the classic thinking-model bug: `max_tokens` is shared by thinking and answer.

The rule for this course:

| You want | Native API | OpenAI API (what LiteLLM, NAT and agents send) |
|---|---|---|
| a direct, fast answer | `"think": false` | `"reasoning_effort": "none"` |
| the model to reason (maths, planning) | `"think": true` and a larger `num_predict` | leave `reasoning_effort` out, raise `max_tokens` |

`sparkkit.chat_any()` sends `reasoning_effort: "none"` to Ollama unless a lab passes `think=True`. The ⚡ block below does the same, so try it: it asks the Spark's Ollama, or the laptop stand-in if the Spark is not connected.

```spark
{"target": "ollama", "which": "a", "model": "gpt-oss:20b",
 "messages": [{"role": "system", "content": "Answer in one short sentence."},
              {"role": "user", "content": "Why does memory bandwidth limit how fast one conversation generates tokens?"}],
 "max_tokens": 120}
```

✓ Checkpoint: you can explain why a thinking model can return an empty answer, and which field turns thinking off through `/v1`.

## 5 · Streaming: measure TTFT and tokens per second

A chat UI streams: tokens appear as they are generated. Two numbers describe what a user feels:

- **TTFT, time to first token**: loading the model (if it is not in memory) plus **prefill**, which reads the whole prompt in one parallel pass.
- **Decode speed, tokens/second**: one token at a time. Each token re-reads every active weight from memory, so Module 01's formula caps it: `tok/s ≤ bandwidth ÷ bytes read per token`.

Lab 03 streams one prompt four times through `/v1` with `"stream": true` and `stream_options: {"include_usage": true}` (without that, Ollama sends no token count in a stream), then asks the native API for the server's own count, then compares with the ceiling:

```bash
# on: laptop
LAPTOP_BW_GBS=410 .venv/bin/python week25/03_ollama_open_webui/labs/lab03_speed_benchmark.py
```

`LAPTOP_BW_GBS` is your laptop's memory bandwidth from its spec sheet (410 GB/s for this Mac's Apple M4 Max configuration). On the Spark the lab uses 273 GB/s and ignores it.

**Expected output** (LAPTOP STAND-IN — captured on this Mac with gemma3:4b, a small dense model)

```
▣ STEP 1 · what are we timing?
◆ gemma3:4b: 3.34 GB on disk · dense (every weight read per token) · ≈ 3.34 GB read per token

▣ STEP 2 · stream the same prompt 4 times (run 0 may include loading the model)
→ POST http://localhost:11434/v1/chat/completions  model=gemma3:4b  stream=True
│ run              TTFT   tokens  total   tok/s
│ ───────────────  ─────  ──────  ──────  ─────  ────────────────────
│ run 0 (warm-up)  73 ms  160     1.70 s  98.4   █████████████░░░░░░░
│ run 1            38 ms  160     1.71 s  95.5   █████████████░░░░░░░
│ run 2            48 ms  160     1.71 s  96.1   █████████████░░░░░░░
│ run 3            42 ms  156     1.86 s  85.6   ███████████░░░░░░░░░
◆ median over runs 1–3: TTFT 42 ms · decode 95.5 tok/s (client-side: tokens after the first ÷ time after the first). Source: LAPTOP STAND-IN (Ollama on this laptop, not the Spark).

▣ STEP 3 · the server's own count (native /api/chat: eval_count ÷ eval_duration)
│ phase             tokens  time     rate
│ ────────────────  ──────  ───────  ─────────────────────────────────
│ decode            160     1.82 s   87.9 tok/s
│ prefill (prompt)  31      0.039 s  805 tok/s
│ model load        —       0.00 s   0 s when it was already in memory

▣ STEP 4 · compare with the bandwidth ceiling (Module 01)
│ model      bandwidth                 read/token  ceiling    measured    of ceiling
│ ─────────  ────────────────────────  ──────────  ─────────  ──────────  ──────────
│ gemma3:4b  410 GB/s (LAPTOP_BW_GBS)  3.34 GB     123 tok/s  95.5 tok/s  78%
◆ The file size includes weights that text generation never reads (gemma3:4b carries a 0.84 GB vision encoder — Module 04, lab 02), so the true ceiling is higher. Kernel overheads, sampling and other requests sharing the server keep real speed below it.
◆ on the Spark, gpt-oss:20b (~15 GB, 3.6B of 21B active) reads ≈ 2.6 GB per token → ceiling ≈ 106 tok/s (arithmetic). Run this lab on the Spark to measure it.
◆ Do not compare the laptop's tok/s with the Spark's: different chips, different models.
```

What to take from it:

1. **Client and server roughly agree.** 95.5 tok/s measured by the script, 87.9 tok/s reported by Ollama for a separate request. Measure on the client when you cannot see the server (vLLM, a gateway); read the server's numbers when you can.
2. **A shared server lies to benchmarks.** An earlier run of this same lab, while other programs were using the laptop's Ollama, measured anything from **6.8 to 95 tok/s** and TTFTs up to 6 s. Benchmark on an idle server, run several times, and report the median.
3. **TTFT is tiny, until a model has to load.** 38–73 ms when the model is warm. The exercise replays a stream where TTFT was **18.6 s**. On the Spark, set `OLLAMA_KEEP_ALIVE` (Section 2) so your model stays in memory.
4. **The ceiling is the yardstick.** `gpt-oss:20b` is a Mixture-of-Experts model: of ~15 GB on disk, only ~2.6 GB is read per token, so its ceiling on the Spark is about **106 tok/s**. When you run lab 03 on the Spark, the "of ceiling" column tells you how close Ollama gets.

✓ Checkpoint: you ran lab 03 and can point to the TTFT and tok/s columns, and you can say why the laptop's 95.5 tok/s says nothing about the Spark.

## 6 · Tool calling: the model asks, your code answers

Agents (Modules 14–18) are built on one mechanism. You send a list of **tools** (name, description, JSON-schema parameters). The model does not run anything: it replies with **`tool_calls`**, a function name and JSON arguments. Your program runs the function and sends the result back as a `role: "tool"` message. Then the model writes the answer.

Lab 04 gives the model two real tools, Module 01's formulas from sparkkit, and asks a question it can only answer with them. It checks `/api/show` first, because only models with the `tools` capability can do this. On the Spark it uses the playbook's agent-ready `qwen3.6:35b-a3b`.

```bash
# on: laptop
.venv/bin/python week25/03_ollama_open_webui/labs/lab04_tool_calling.py
```

**Expected output** (LAPTOP STAND-IN — captured on this Mac with nemotron-3-nano)

```
▣ STEP 1 · can this model call tools? (POST /api/show → capabilities)
◆ nemotron-3-nano:latest: completion, tools, thinking

▣ STEP 2 · ask, with two tools on offer
→ POST http://localhost:11434/v1/chat/completions · tools=[weights_gb, decode_ceiling_tok_s]
· ANSWER
→ tool_call weights_gb({"params_b":117,"fmt":"mxfp4"})
→ tool_call decode_ceiling_tok_s({"active_params_b":5.1,"fmt":"mxfp4"})
◆ LAPTOP STAND-IN (not Spark numbers) · nemotron-3-nano:latest · TTFT 11370 ms · 85 tok in 11.4s · 7.5 tok/s

▣ STEP 3 · run 2 tool call(s) here, send the results back as role=tool
│ weights_gb({"params_b":117,"fmt":"mxfp4"}) → 62.2
│ decode_ceiling_tok_s({"active_params_b":5.1,"fmt":"mxfp4"}) → 100.8
· ANSWER  The weights of the **gpt-oss-120b** model (117B total parameters in **mxfp4** format) occupy **62.2 GB** of storage.

          On a **DGX Spark** (with 273 GB/s memory bandwidth), the decode ceiling for this model is **100.8 tokens per second**.
          …
◆ ground truth from sparkkit: weights 62.2 GB · ceiling 100.8 tok/s — does the final answer quote these numbers?
```

Three details worth noticing:

- The first reply has an **empty answer** and two tool calls, **in parallel**: the model asked for both numbers at once.
- Tool-calling requests are sent **without streaming** (sparkkit turns `stream` off when `tools` are present), so "TTFT" there is the whole response time, and here it includes a model load.
- The final numbers are right because *your code* computed them. The model only chose which function to call and with what arguments. Modules 15–16 put a sandbox and a policy exactly in that gap.

✓ Checkpoint: you can list the four messages in the loop (user → assistant with `tool_calls` → `tool` → assistant) and say which side runs the function.

## 7 · Open WebUI: a chat UI in one docker run

[Open WebUI](https://build.nvidia.com/spark/open-webui) is a self-hosted, ChatGPT-style web app. The playbook runs the **`ghcr.io/open-webui/open-webui:ollama`** image, which bundles its **own** Ollama inside the container. It does not use the Ollama you installed in Section 1.

```bash
# on: spark
docker ps > /dev/null                       # blank output = Docker works without sudo
docker pull ghcr.io/open-webui/open-webui:ollama
docker run -d -p 12000:8080 --gpus=all \
  -v open-webui:/app/backend/data \
  -v open-webui-ollama:/root/.ollama \
  --name open-webui ghcr.io/open-webui/open-webui:ollama
```

This is the `docker run` line from the playbook's NVIDIA Sync launch script. The playbook's **desktop** path uses `-p 8080:8080` instead; this course uses **12000** (the port in `sparkkit.PORTS` and in Module 01's tunnel) so it never collides with other services on 8080.

| Flag | Meaning |
|---|---|
| `-p 12000:8080` | the web app listens on 8080 inside the container, 12000 on the Spark |
| `--gpus=all` | the bundled Ollama gets the GB10 GPU (without it: "GPU not detected", slow CPU inference) |
| `-v open-webui:/app/backend/data` | accounts and chat history survive restarts |
| `-v open-webui-ollama:/root/.ollama` | the bundled Ollama's models: a **second** model store, separate from Section 1's |

Open it through a tunnel (or add it as an NVIDIA Sync custom app on port 12000, as the playbook does):

```bash
# on: laptop
ssh -N -L 12000:localhost:12000 spark-a
# open http://localhost:12000
```

Then follow the playbook in the browser:

1. **Get Started** → create the admin account. It is stored on the Spark only.
2. **Select a model** → type `gpt-oss:20b` → **Pull "gpt-oss:20b" from Ollama.com**. The container ships with no model.
3. Select it and send **Write me a haiku about GPUs**. The first reply can take up to 30 seconds while the model loads onto the GPU.
4. For agents, the playbook recommends `qwen3.6:35b-a3b` and a context window of **at least 32K tokens**, 64K or more if memory allows.

Because the bundled Ollama publishes no port, `ollama list` on the Spark does not show the models you pulled in the UI. Look inside the container instead (lab 01, step 3, does this):

```bash
# on: spark
docker exec open-webui ollama list
```

> 💡 Two Ollamas holding the same 15 GB model means 30 GB of disk and, if both are loaded, twice the memory. On a 128 GB unified-memory machine that matters. Stop the container when you are done: `docker stop open-webui`. The playbook's cleanup (`docker rm`, `docker rmi`, `docker volume rm open-webui open-webui-ollama`) deletes chat history and models permanently.

✓ Checkpoint: `http://localhost:12000` shows Open WebUI through your tunnel and answers the haiku prompt, and you can explain why `ollama list` on the host does not list its models.

## Labs — run them here

**labs/lab01_ollama_models.py** — Check the Spark's Ollama install, models and port binding, then describe every model through the native API.

**labs/lab02_native_vs_openai.py** — Send one question to the native `/api/chat` and the OpenAI `/v1` API, then count what thinking costs.

**labs/lab03_speed_benchmark.py** — Stream a prompt, measure TTFT and tok/s, and compare them with the bandwidth ceiling.

**labs/lab04_tool_calling.py** — Run a full tool-calling loop with Module 01's memory and speed formulas as the tools.

Lab 01 runs its shell steps LIVE on your Spark or DRY. Labs 02–04 call the Spark's Ollama when it answers, else the laptop stand-in (labelled), else they stop with a DRY note. Each request asks for at most 200 tokens.

## Try it yourself

**Exercise 03 — the stream meter.** Open `week25/03_ollama_open_webui/exercises/ex03_stream_meter.py`. It replays a **real** stream captured from Ollama's `/v1` endpoint on the course laptop (nemotron-3-nano, thinking on), with the arrival time of every line. Four `TODO`s:

1. `parse_sse(lines)`: turn Server-Sent Events into JSON chunks.
2. `split_text(chunks)`: join the `content` pieces and the `reasoning` pieces separately.
3. `speed(timed)`: TTFT, token count and tok/s, with the same rule `sparkkit.chat()` uses.
4. `request_body(...)`: the streaming request, with `include_usage` and thinking off.

```bash
# on: laptop
.venv/bin/python week25/03_ollama_open_webui/exercises/ex03_stream_meter.py
```

**Expected output** (once all four TODOs are done; the checker is offline, so this is the same on every machine)

```
✓ parse_sse: 27 JSON chunks, the last one carries usage, [DONE] not included
✓ split_text: answer '51' · reasoning 'We need to answer with the number only: 17 * 3 = 51 …'
✓ speed: TTFT 18649 ms · 31 tokens · ≈ 82.7 tok/s after the first token
✓ request_body: stream + include_usage · reasoning_effort 'none' only when think=False

▣ your meter, applied to the recorded stream (LAPTOP STAND-IN)
│ answer '51' · reasoning 68 chars · 31 tokens billed
│ TTFT 18.6 s · then 82.7 tok/s
```

<details><summary>Hint — what does one SSE line look like?</summary>

`data: {"id":"chatcmpl-667","object":"chat.completion.chunk",…,"choices":[{"index":0,"delta":{"content":"","reasoning":" need"},"finish_reason":null}]}`. Strip `data:`, `json.loads` the rest. Blank lines separate events, and the stream ends with `data: [DONE]`, which is not JSON.

</details>

<details><summary>Stretch — point your meter at the Spark</summary>

Send `request_body("gpt-oss:20b", …)` to `http://localhost:21434/v1/chat/completions` through your tunnel, record `(time, line)` pairs while you read the response, and feed them to `speed()`. Compare with lab 03's numbers for the same model, and try `think=True` to see how much TTFT the reasoning adds before the first *answer* token.

</details>

✓ Checkpoint: all four checker lines are ✓.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `curl: (7) Failed to connect to localhost port 11434` from the laptop | the tunnel is not running, or it forwards the wrong port. Start `ssh -N -L 21434:localhost:11434 spark-a` and call port **21434** |
| Labs say `LAPTOP STAND-IN` although the tunnel works | `SPARK_HOST` is not set or not reachable, so the labs are in DRY mode. Set it in 🖥 Spark setup, and set `SPARK_URL_OLLAMA=http://localhost:21434/v1` |
| `model "gpt-oss:20b" not found` | the model was pulled in Open WebUI (the container's Ollama), not on the host. `ollama pull gpt-oss:20b` on the Spark, or check with `docker exec open-webui ollama list` |
| Empty answer, output tokens == `max_tokens` | a thinking model spent the budget on reasoning. Send `reasoning_effort: "none"` (or `think: false`), or raise `max_tokens` |
| First request takes 10–40 s, later ones are fast | the model was loading (`load_duration` in the native API). Set `OLLAMA_KEEP_ALIVE` in the service override, and avoid switching models between requests |
| `does not support tools` error | that model has no `tools` capability (see lab 01's table). Use `qwen3.6:35b-a3b` or `gpt-oss:20b` on the Spark |
| Open WebUI: "GPU not detected" or very slow replies | the container was started without `--gpus=all`. `docker rm -f open-webui` and run the `docker run` line again (the volumes keep your data) |
| `Port 12000 already in use` | another service holds it. `ss -tlnp \| grep 12000`, or map a different host port: `-p 12001:8080` |
| Memory pressure although the model should fit | the playbooks' UMA note: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`. Also check that two Ollamas are not holding models at once |

## Next

Continue to [Lab 04 — llama.cpp + LM Studio](../04_llama_cpp_lm_studio/TUTORIAL.md): build the engine inside Ollama yourself, open a GGUF file to see what `Q4_K_M` really means, and serve it with `llama-server` and LM Studio.
