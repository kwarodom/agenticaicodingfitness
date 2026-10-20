# ▶ Spark Lab 08 — LiteLLM: one gateway for every engine and both Sparks

> Part of Week 25 · DGX Spark: fine-tune, serve, and build sandboxed agents. You type the commands, you see the real output. Every lab also runs in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's playbook, or a clearly marked EXAMPLE.

**What you'll actually do**
- See why four engines on each of two Sparks need one front door: one OpenAI endpoint, model aliases, fallbacks, retries, load balancing and a master key.
- Generate one LiteLLM `config.yaml` from your Spark settings: Ollama :11434, vLLM :8000, SGLang :30000 and TensorRT-LLM :8355 on Spark A and Spark B, with laptop fallbacks.
- Start the LiteLLM proxy **for real on this laptop**, call it with the OpenAI SDK through aliases, and watch the master key turn away a call without it.
- Point aliases at a Spark that is down and watch the fallback, the retry and the cooldown in the response headers.
- Decide where the gateway lives (laptop or Spark A) and move it there.

**Time** ~45 min · **Difficulty** intermediate · **Hardware** none (every lab runs on the laptop with Ollama; with Sparks, the same aliases route to them)

**Official playbooks covered:** *course-original.* NVIDIA has no LiteLLM playbook. This module puts one gateway in front of the servers from [Open WebUI with Ollama](https://build.nvidia.com/spark/open-webui) · [vLLM](https://build.nvidia.com/spark/vllm) · [SGLang](https://build.nvidia.com/spark/sglang) · [TRT-LLM](https://build.nvidia.com/spark/trt-llm). Every LiteLLM flag and config key here was checked against the installed package (LiteLLM 1.89.0: `litellm --help` and the `litellm.Router` signature).

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| LiteLLM proxy 1.89 in `week25/.venv-litellm` | `week25/.venv-litellm/bin/litellm --version` | the repo `.venv` has the LiteLLM library but not the `[proxy]` extras the server needs |
| Ollama on the laptop, two small models | `ollama list` shows `gemma3:4b` and `nemotron-3-nano:latest` | the laptop fallbacks every lab calls |
| Port 4000 free | `lsof -iTCP:4000 -sTCP:LISTEN` prints nothing | the labs use the next free port if 4000 is taken, and say so |
| Modules 03–07 (optional) | `curl http://spark-a:8000/v1/models` | with engines running, the Spark aliases answer for real |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
week25/.venv-litellm/bin/litellm --version
ollama list | grep -E "gemma3:4b|nemotron-3-nano"
```

**Expected output** (captured on this Mac)

```
LiteLLM: Current Version = 1.89.0

nemotron-3-nano:latest           b725f1117407    24 GB     8 weeks ago
gemma3:4b                        a2af6cc3eb7f    3.3 GB    8 weeks ago
```

If `week25/.venv-litellm` is missing, create it (a week25-local venv, nothing global):

```bash
# on: laptop
uv venv week25/.venv-litellm --python 3.13
uv pip install --python week25/.venv-litellm/bin/python 'litellm[proxy]==1.89.0' openai pyyaml
```

> ⚠ `.venv/bin/litellm` (the repo venv) exits with `ImportError: Missing dependency No module named 'backoff'. Run pip install 'litellm[proxy]'`. That is why this module uses its own venv for the proxy. The labs themselves still run with `.venv/bin/python`.

✓ Checkpoint: `week25/.venv-litellm/bin/litellm --version` prints 1.89.0, and both laptop models are listed.

## 1 · Why a gateway?

By Module 07 you can have up to eight OpenAI-compatible servers: four engines on each Spark. Every client (a notebook, an agent, Open WebUI) would need each URL, each model id, and its own failover code. A gateway moves all of that to one place:

| Without a gateway | With LiteLLM on :4000 |
|---|---|
| `http://spark-a:8000/v1`, model `nvidia/Llama-3.1-8B-Instruct-FP8` | `http://gateway:4000/v1`, model **`chat`** |
| Swap vLLM for SGLang → edit every client | edit one line of `config.yaml` |
| Spark A reboots → errors in every app | the router retries the other Spark, then falls back to the laptop |
| Two Sparks → clients pick one | one alias, two deployments, load-balanced |
| Engines have **no authentication** | one **master key** in front (the engines still need to stay private: Section 3) |
| No request log | every reply carries `x-litellm-*` headers: which deployment answered, retries, fallbacks, timings |

Clients only ever see **aliases** (`model_name` in the config). Each alias maps to one or more **deployments** (`litellm_params`: a provider-prefixed model id plus an `api_base`). The provider prefix tells LiteLLM how to talk to the server:

| Engine (port) | LiteLLM `model:` | Why that prefix |
|---|---|---|
| vLLM (:8000) | `hosted_vllm/<served model id>` | LiteLLM's provider for self-hosted vLLM |
| Ollama (:11434), SGLang (:30000), TensorRT-LLM (:8355) | `openai/<served model id>` | all three serve the OpenAI API on `/v1` |

`api_key: none` is a placeholder: the engines do not check it, and the OpenAI client code inside LiteLLM expects some value.

✓ Checkpoint: you can say what an alias is, what a deployment is, and why `chat` can have two deployments.

## 2 · Build the config from your Spark settings: lab 01

Lab 01 reads the same settings as every Week 25 lab (`SPARK_HOST`, `SPARK_API_HOST`, Module 01). It asks each engine that answers which model it serves (`GET /v1/models`) and falls back to the course's default model ids. Then it writes one file, `week25/08_litellm_gateway/.runs/litellm.config.yaml` (gitignored), and validates it with PyYAML **and** with LiteLLM's own `Router`.

```bash
# on: laptop
.venv/bin/python week25/08_litellm_gateway/labs/lab01_build_config.py
```

**Expected output** (captured on this Mac with no Spark configured, so the hosts are the placeholders `spark-a` / `spark-b`)

```
▣ STEP 1 · where each alias points
│ Spark A HTTP host: (not set → placeholder spark-a) · Spark B: (not set → placeholder spark-b)
│ alias            LiteLLM model (provider/id)                   api_base                   model id from
│ ───────────────  ────────────────────────────────────────────  ─────────────────────────  ───────────────────
│ spark-a/ollama   openai/gpt-oss:20b                            http://spark-a:11434/v1    course default
│ spark-a/vllm     hosted_vllm/nvidia/Llama-3.1-8B-Instruct-FP8  http://spark-a:8000/v1     course default
│ spark-a/sglang   openai/Qwen/Qwen3-8B                          http://spark-a:30000/v1    course default
│ spark-a/trtllm   openai/nvidia/Llama-3.1-8B-Instruct-FP4       http://spark-a:8355/v1     course default
│ spark-b/ollama   openai/gpt-oss:20b                            http://spark-b:11434/v1    course default
│ …
│ chat             hosted_vllm/nvidia/Llama-3.1-8B-Instruct-FP8  http://spark-a:8000/v1     load-balanced A + B
│ chat             hosted_vllm/nvidia/Llama-3.1-8B-Instruct-FP8  http://spark-b:8000/v1     load-balanced A + B
│ laptop-fast      openai/gemma3:4b                              http://localhost:11434/v1  fallback
│ laptop-nemotron  openai/nemotron-3-nano:latest                 http://localhost:11434/v1  fallback

▣ STEP 3 · validate — PyYAML, then LiteLLM's own Router
✓ PyYAML: the file parses back to exactly the generated config
✓ every fallback names aliases that exist
✓ master_key is read from the environment (os.environ/LITELLM_MASTER_KEY), not stored in the file
✓ litellm 1.89.0 Router accepted 12 deployments in 11 aliases · strategy simple-shuffle
```

The generated file, trimmed to one deployment per kind. **This is the pattern the capstone (Module 20) reuses:**

```yaml
model_list:
- model_name: spark-a/vllm                  # one alias per engine per Spark
  litellm_params:
    model: hosted_vllm/nvidia/Llama-3.1-8B-Instruct-FP8
    api_base: http://spark-a:8000/v1
    api_key: none
- model_name: chat                          # SAME alias twice → load-balanced across the Sparks
  litellm_params: {model: hosted_vllm/nvidia/Llama-3.1-8B-Instruct-FP8, api_base: http://spark-a:8000/v1, api_key: none}
- model_name: chat
  litellm_params: {model: hosted_vllm/nvidia/Llama-3.1-8B-Instruct-FP8, api_base: http://spark-b:8000/v1, api_key: none}
- model_name: laptop-fast                   # the fallback that is always there
  litellm_params: {model: openai/gemma3:4b, api_base: http://localhost:11434/v1, api_key: none}
- model_name: laptop-nemotron               # a thinking model, told not to think
  litellm_params:
    model: openai/nemotron-3-nano:latest
    api_base: http://localhost:11434/v1
    api_key: none
    extra_body: {reasoning_effort: none}
router_settings:
  routing_strategy: simple-shuffle          # also: least-busy, latency-based-routing, usage-based-routing-v2 …
  num_retries: 1                            # retry once on another deployment of the same alias
  timeout: 120                              # seconds per call
  allowed_fails: 1                          # failures before a deployment is benched …
  cooldown_time: 30                         # … for this many seconds
  fallbacks:
  - chat: [laptop-fast]                     # whole alias down → try this alias instead
  - spark-a/vllm: [laptop-fast]
litellm_settings:
  drop_params: true                         # drop OpenAI params an engine does not support
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY # read the key from the environment; never write it here
```

Two details that cost time if you guess them:

- **`extra_body: {reasoning_effort: none}`** is how the nemotron alias turns thinking off in Ollama. Putting `reasoning_effort: none` straight into `litellm_params` fails on this version with `UnsupportedParamsError: openai does not support parameters: ['reasoning_effort']` (captured on this Mac). `extra_body` passes it to the engine untouched.
- **`router_settings` keys must be `Router()` arguments.** The proxy passes only valid ones and logs `Key '…' is not a valid argument for Router.__init__(). Ignoring this key.` for the rest (read from the installed `proxy_server.py`), so at start-up a typo is only a log line. Lab 01 passes `router_settings` straight to `litellm.Router(...)`, which is stricter. On this Mac, `num_retrys: 1` failed there with `TypeError: … unexpected keyword argument 'num_retrys'. Did you mean 'num_retries'?`, and `routing_strategy: fastest` with `ValueError: Invalid routing_strategy`. Run lab 01 (or the exercise checker) after every hand edit.

✓ Checkpoint: `.runs/litellm.config.yaml` exists and all four lab 01 checks are ✓.

## 3 · Start it for real: aliases and the master key (lab 02)

Lab 02 starts the proxy as a child process, exactly as you would by hand:

```bash
# on: laptop
export LITELLM_MASTER_KEY="sk-$(openssl rand -hex 16)"      # a fresh key; keep it out of files and shell history
week25/.venv-litellm/bin/litellm --config week25/08_litellm_gateway/.runs/litellm.config.yaml \
  --host 127.0.0.1 --port 4000 --telemetry False
```

`--config`, `--host`, `--port` and `--telemetry` are flags from `litellm --help` in 1.89.0. The lab also sets `LITELLM_LOCAL_MODEL_COST_MAP=True`, so the proxy uses the price list bundled in the package instead of downloading one. The lab generates a random master key per run and passes it through the environment, never on the command line and never printed. It stops the proxy when it ends, even on Ctrl-C.

```bash
# on: laptop
.venv/bin/python week25/08_litellm_gateway/labs/lab02_gateway_live.py
```

**Expected output** (captured on this Mac: a real LiteLLM proxy, real calls to the laptop's Ollama, labelled LAPTOP STAND-IN)

```
▣ STEP 1 · start LiteLLM with lab 01's config
$ LITELLM_MASTER_KEY=sk-w25-… week25/.venv-litellm/bin/litellm --config week25/08_litellm_gateway/.runs/litellm.config.yaml --host 127.0.0.1 --port 4000 --telemetry False
◆ gateway up on http://127.0.0.1:4000 after 2.1 s (pid 41369, log 08_litellm_gateway/.runs/litellm-4000.log)

▣ STEP 2 · GET /v1/models — the aliases, not the engines' model ids
│ spark-a/ollama · spark-a/vllm · spark-a/sglang · spark-a/trtllm · spark-b/ollama · spark-b/vllm · spark-b/sglang · spark-b/trtllm · chat · laptop-fast · laptop-nemotron

▣ STEP 3 · call two aliases through the OpenAI SDK (LAPTOP STAND-IN: Ollama on this Mac)
→ POST http://127.0.0.1:4000/v1/chat/completions · model=laptop-fast
· ANSWER  NVIDIA Tesla GPUs.
→ POST http://127.0.0.1:4000/v1/chat/completions · model=laptop-nemotron
· ANSWER  An API gateway acts as a single entry point that routes client requests to the appropriate backend services, handling tasks like request routing, authentication, rate limiting, and response aggregation.
│ alias asked for  model in reply   x-litellm-model-api-base   tokens  time
│ ───────────────  ───────────────  ─────────────────────────  ──────  ─────
│ laptop-fast      laptop-fast      http://localhost:11434/v1  6       0.5 s
│ laptop-nemotron  laptop-nemotron  http://localhost:11434/v1  35      9.1 s

▣ STEP 4 · streaming works through the gateway too
· STREAM  1, 2, 3, 4, 5, 6, 7, 8

▣ STEP 5 · the master key: three requests to GET /v1/models
│ Authorization   HTTP  type              message
│ ──────────────  ────  ────────────────  ───────────────────────────────────────────
│ no key          401   auth_error        Authentication Error, No api key passed in.
│ a wrong key     400   no_db_connection  No connected db.
│ the master key  200   11 aliases
✓ no key → 401: the gateway refuses anonymous callers
✓ wrong key → 400: refused (no database, so LiteLLM cannot look up virtual keys and says so)
✓ master key → 200

▣ STEP 6 · …but the engine behind it has no lock
│ GET http://localhost:11434/v1/models with NO key → HTTP 200, 7 models
◆ gateway stopped (pid 41369); port 4000 is free again: True
```

Three things to notice:

1. **The reply names the alias, not the engine.** The `model` field says `laptop-fast`. The `x-litellm-model-api-base` header says which server actually answered. (gemma3:4b's answer is wrong, by the way. A DGX Spark has a GB10, not a Tesla. A 4B model is a fallback, not an oracle.)
2. **A wrong key gets 400, not 401.** Without a database, LiteLLM cannot look up *virtual keys* (per-team keys with budgets). So any key that is not the master key fails with `no_db_connection`. It is still refused. Adding a Postgres `DATABASE_URL` turns on virtual keys, which is beyond this module.
3. **Step 6 is the catch.** The gateway only protects callers that go through it. Ollama, vLLM, SGLang and TensorRT-LLM have no authentication. On the Spark, keep engine ports private (bound to `127.0.0.1` when the gateway runs on the same Spark, or reachable only over the Spark-to-Spark link), and publish only :4000. The playbooks start servers with `--host 0.0.0.0` or `--network host`, which exposes them on every interface, including your tailnet (Module 01, Section 3).

Any OpenAI client works the same way, with `curl` too:

```bash
# on: laptop
curl -s http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"model": "laptop-fast", "messages": [{"role": "user", "content": "Say hello in three words."}], "max_tokens": 20}'
```

✓ Checkpoint: lab 02 ends with three ✓ lines in Step 5 and `port 4000 is free again: True`.

## 4 · When a Spark is down: fallbacks, retries, cooldown, load balancing (lab 03)

Three router features decide what a client sees when a Spark is off:

| Feature | Config | Acts on |
|---|---|---|
| **retry** | `num_retries: 1` | a failed call is sent again, to another deployment of the **same alias** if there is one |
| **cooldown** | `allowed_fails: 1`, `cooldown_time: 30` | a deployment that keeps failing is benched for 30 s, so callers stop waiting on it |
| **fallback** | `fallbacks: [{chat: [laptop-fast]}]` | when an alias fails as a whole, the router tries **another alias** |

Lab 03 adds two demo aliases to lab 01's config: `one-spark-down` (Spark A's vLLM plus the laptop, two deployments of one alias) and `laptop-pair` (the laptop's Ollama under two names, standing in for two Sparks).

```bash
# on: laptop
.venv/bin/python week25/08_litellm_gateway/labs/lab03_fallback_and_balance.py
```

**Expected output** (captured on this Mac with no Spark configured: the "Spark" deployments are down, the laptop answers, LAPTOP STAND-IN)

```
▣ STEP 1 · spark-a/vllm, twice — one deployment on Spark A, fallback on the laptop
│ alias asked   HTTP  answered by (group)  api_base                   retries  fallbacks  time   reply
│ ────────────  ────  ───────────────────  ─────────────────────────  ───────  ─────────  ─────  ──────
│ spark-a/vllm  200   laptop-fast          http://localhost:11434/v1  0        1          4.1 s  Ready.
│ spark-a/vllm  200   laptop-fast          http://localhost:11434/v1  0        1          1.5 s  Ready.
✓ Spark A's vLLM did not answer → both calls fell back to laptop-fast (Ollama on this Mac, LAPTOP STAND-IN)

▣ STEP 2 · chat — load-balanced across Spark A and Spark B vLLM
│ chat         200   laptop-fast          http://localhost:11434/v1  0        1          3.2 s  Ready.

▣ STEP 3 · one-spark-down × 6 — retries inside the group, then cooldown
│ call  HTTP  api_base that answered     retries  time
│ ────  ────  ─────────────────────────  ───────  ──────
│ 1     200   http://localhost:11434/v1  1        6.5 s
│ 2     200   http://localhost:11434/v1  1        13.6 s
│ 3     200   http://localhost:11434/v1  0        0.1 s
│ 4     200   http://localhost:11434/v1  0        0.2 s
│ 5     200   http://localhost:11434/v1  0        0.2 s
│ 6     200   http://localhost:11434/v1  0        0.2 s
│ calls that first landed on the dead Spark and were retried on the laptop: [1, 2]

▣ STEP 4 · laptop-pair × 8 — the balancer at work (two names for the SAME laptop Ollama)
│ http://127.0.0.1:11434/v1      ████████████ 4
│ http://localhost:11434/v1      ████████████ 4
✓ simple-shuffle spread the calls over both deployments

▣ STEP 5 · the gateway's own log
│ INFO:     127.0.0.1:55921 - "POST /v1/chat/completions HTTP/1.1" 200 OK
│ … 17 requests logged, 17 of them HTTP 200 — the failed Spark attempts are not in this log
```

Read it row by row:

- **Step 1.** `spark-a/vllm` has only one deployment, so there is nothing to retry *within* the alias. Every call tries Spark A, fails, and falls back (`fallbacks 1`). In a `--detailed_debug` run on this Mac, the router tried `spark-a:8000` again on the second call. A single-deployment alias is not skipped. Here the failure is instant, because the placeholder host does not resolve. A Spark that resolves but is switched off costs a connection timeout on every call. Give critical aliases two deployments.
- **Step 3** is the interesting one. Calls 1 and 2 landed on the dead Spark and were **retried** on the laptop (`retries 1`). The client still saw HTTP 200. The Spark had now failed more than `allowed_fails: 1`, so it went into **cooldown** (a `--detailed_debug` run on this Mac logged it in the router's cooldown list after the second failure), and calls 3–6 went straight to the laptop (`retries 0`). Shuffle is random, so your call numbers will differ.
- **Step 4.** `simple-shuffle` spread 8 calls 4 and 4 in this run. With two real Sparks, that halves the load on each. The router also supports `least-busy` and `latency-based-routing` (valid values of `routing_strategy` in 1.89.0).
- **Times** are mostly the laptop's Ollama generating a one-word reply on a machine shared with other work. They are not the router's cost, and certainly not Spark numbers.

The default log is one access line per request, all `200`. The failures happened *inside* the gateway. To see every routing decision, add `--detailed_debug` to the `litellm` command.

✓ Checkpoint: in your run of Step 3, some calls show `retries 1` and later calls `retries 0`, and you can explain why using `allowed_fails` and `cooldown_time`.

## 5 · Where should the gateway run?

| | On the laptop (this module's labs) | On Spark A (recommended to share, and for the capstone) |
|---|---|---|
| Who can use it | apps on this laptop | anyone on your tailnet with the key |
| Engine URLs | `http://spark-a:8000/v1`, so the engines must be reachable from the laptop | `http://localhost:8000/v1` for Spark A. Engines can stay on `127.0.0.1` |
| Spark B | over the tailnet | over the Spark-to-Spark link or tailnet |
| Laptop fallback | `localhost:11434` | only if the laptop's Ollama listens on the tailnet (`OLLAMA_HOST=0.0.0.0`). Otherwise fall back to Spark A's Ollama |
| Lives as long as | your laptop is awake | the Spark is on |

Lab 01 writes the Spark A variant too. Spark A's engines become `localhost`. The laptop fallbacks are dropped, or pointed at your laptop's tailnet name if you pass it. With a Spark configured, the file is copied to `~/w25/litellm/config.yaml` on Spark A:

```bash
# on: laptop
.venv/bin/python week25/08_litellm_gateway/labs/lab01_build_config.py --on-spark
# or keep the laptop fallback, via the tailnet (the laptop must run: OLLAMA_HOST=0.0.0.0 ollama serve)
.venv/bin/python week25/08_litellm_gateway/labs/lab01_build_config.py --on-spark --laptop-url http://<laptop-tailnet-name>:11434/v1
```

**Expected output** (captured on this Mac, DRY: nothing copied)

```
▣ STEP 4 · the variant for a gateway ON Spark A
│ alias           api_base
│ ──────────────  ─────────────────────────
│ spark-a/ollama  http://localhost:11434/v1
│ spark-a/vllm    http://localhost:8000/v1
│ …
│ chat            http://localhost:8000/v1
│ chat            http://spark-b:8000/v1
│ fallbacks: [{'chat': ['spark-a/ollama']}] …
◆ no --laptop-url: the laptop fallbacks are dropped (localhost on a Spark is the Spark), and `chat` falls back to Spark A's Ollama instead.
$ scp litellm.spark.yaml <spark>:~/w25/litellm/config.yaml   [DRY]
```

Then, on Spark A, install the same LiteLLM version in a venv and start it in the background. These are the commands used on this Mac, adapted to the Spark's paths. The course has not run them on a Spark yet. If a wheel fails to build on aarch64, the error names the package:

```bash
# on: spark
sudo apt install -y python3-venv                    # only if the next line says ensurepip is not available
python3 -m venv ~/w25/litellm-venv
~/w25/litellm-venv/bin/pip install 'litellm[proxy]==1.89.0'
mkdir -p ~/w25/litellm ~/w25/logs
( umask 077; echo "sk-$(openssl rand -hex 16)" > ~/w25/litellm/master.key )
LITELLM_MASTER_KEY="$(cat ~/w25/litellm/master.key)" LITELLM_LOCAL_MODEL_COST_MAP=True \
  nohup ~/w25/litellm-venv/bin/litellm --config ~/w25/litellm/config.yaml \
  --host 0.0.0.0 --port 4000 --telemetry False > ~/w25/logs/litellm.log 2>&1 &
curl -s http://localhost:4000/health/liveliness
```

`/health/liveliness` answers `"I'm alive!"` without a key (captured on this Mac). On the laptop, read the key over SSH into your shell only, never into a file in the repo:

```bash
# on: laptop
export LITELLM_MASTER_KEY="$(ssh spark-a cat ~/w25/litellm/master.key)"
curl -s http://spark-a:4000/v1/models -H "Authorization: Bearer $LITELLM_MASTER_KEY"
```

> 💡 LiteLLM also publishes a container image (see the Docker page of the LiteLLM docs). If you prefer Docker on the Spark, check that the tag you pull has an **arm64** build and matches 1.89. The course has not tested it.

**From the Lab Runner.** The ⚡ blocks below go to `litellm` on Spark A (:4000). The runner never stores your master key, so it sends no `Authorization` header. A keyed gateway answers them with 401, and the block shows the reference text. With no gateway running on the Spark, the runner falls back to the laptop's Ollama directly (labelled LAPTOP STAND-IN), not through a gateway. For keyed calls, use the labs, `curl` or the OpenAI SDK with your key.

```spark
{"target": "litellm", "which": "a", "model": "chat", "messages": [{"role": "user", "content": "In one sentence: why put a gateway in front of two DGX Sparks?"}], "max_tokens": 120}
```

```spark
{"target": "litellm", "which": "a", "model": "laptop-fast", "messages": [{"role": "user", "content": "Say which alias you are, in five words."}], "max_tokens": 40}
```

✓ Checkpoint: you have picked a place for the gateway. Either lab 01 `--on-spark` wrote `litellm.spark.yaml`, or you have decided to keep the gateway on the laptop for now.

## Labs — run them here

**labs/lab01_build_config.py** — Generate one LiteLLM config for every engine on both Sparks with laptop fallbacks, and validate it with PyYAML and LiteLLM's Router (`--on-spark` writes the Spark A variant).

**labs/lab02_gateway_live.py** — Start the LiteLLM proxy on this laptop, call it through aliases with the OpenAI SDK (plain and streaming), and watch the master key refuse calls.

**labs/lab03_fallback_and_balance.py** — Point aliases at a Spark that is down and watch fallbacks, retries, cooldown and load balancing in the `x-litellm-*` headers.

All three run on the laptop. Labs 02 and 03 start a real proxy and always stop it on exit. `labs/_gateway.py` is their shared helper (config builder and proxy process), not a lab.

## Try it yourself

**Exercise 08 — complete the gateway.** Open `week25/08_litellm_gateway/exercises/ex08_complete_the_gateway.py`. The YAML at the top has one `chat` deployment on Spark A and four `TODO`s:

1. A second `chat` deployment: the same model on Spark B (load balancing).
2. `fallbacks`: `chat` → `laptop-fast`.
3. `num_retries: 1`, `timeout: 120`, `allowed_fails: 1`, `cooldown_time: 30`.
4. A master key read from the environment, `os.environ/LITELLM_MASTER_KEY`.

```bash
# on: laptop
.venv/bin/python week25/08_litellm_gateway/exercises/ex08_complete_the_gateway.py
```

**Expected output** (once all four TODOs are done. Captured on this Mac from the reference solution)

```
✓ TODO 1: `chat` has two deployments — vLLM on spark-a:8000 and spark-b:8000, same model
✓ TODO 2: chat falls back to laptop-fast
✓ TODO 3: num_retries 1 · timeout 120 · allowed_fails 1 · cooldown_time 30
✓ TODO 4: master_key comes from the environment, not from the file
✓ LiteLLM Router loaded it: 3 deployments, aliases ['chat', 'laptop-fast']

═ saved to week25/08_litellm_gateway/.runs/ex08.config.yaml. Try it live (it will fall back to your laptop):
  export LITELLM_MASTER_KEY=sk-$(openssl rand -hex 12)
  week25/.venv-litellm/bin/litellm --config week25/08_litellm_gateway/.runs/ex08.config.yaml --port 4000
```

Starting the saved file that way and asking for `chat` returned HTTP 200 on this Mac, with `x-litellm-model-group: laptop-fast` and `x-litellm-attempted-fallbacks: 1`.

<details><summary>Hint — where do the TODO 2 and 3 keys go?</summary>

All five are `router_settings` keys, at the same indentation as `routing_strategy`. `fallbacks` is a list of one-key maps: `fallbacks: [{"chat": ["laptop-fast"]}]`.

</details>

<details><summary>Hint — why not paste the key into the file?</summary>

Config files get copied, committed and shared. `os.environ/LITELLM_MASTER_KEY` tells LiteLLM to read the environment variable at start-up, so the file is safe to share and the key stays in your shell.

</details>

<details><summary>Stretch — send Spark A's Ollama to Spark B</summary>

Add a fallback `spark-a/ollama: [spark-b/ollama, laptop-fast]` to lab 01's config. Fallbacks are tried in order. Which alias answers when only Spark B is up?

</details>

✓ Checkpoint: all four TODO lines and the Router line are ✓.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: Missing dependency No module named 'backoff'` | You ran `.venv/bin/litellm`. Use `week25/.venv-litellm/bin/litellm` (Section 0) |
| `port 4000 is taken on this laptop — using 4001 instead` | Something else listens on 4000 (maybe another gateway). The labs pick the next free port. By hand: `lsof -iTCP:4000 -sTCP:LISTEN` |
| `Authentication Error, No api key passed in.` (401) | Send `Authorization: Bearer $LITELLM_MASTER_KEY` |
| `No connected db.` (400) | The key is not the master key. Without a database, only the master key works |
| `UnsupportedParamsError: openai does not support parameters: ['reasoning_effort']` | Move it into `extra_body: {reasoning_effort: none}` (Section 2), or set `litellm_settings: drop_params: true` to drop it |
| A setting in `router_settings` has no effect | The log says `Key '…' is not a valid argument for Router.__init__(). Ignoring this key.` Run lab 01: its Router check names the typo |
| Every call to a Spark alias takes seconds, then falls back | A single-deployment alias retries the dead Spark every time, and a resolvable but powered-off host costs a connect timeout. Give the alias a second deployment, or lower `timeout` |
| The gateway is still running after a lab crashed | `pgrep -fl "litellm --config"`, then `kill <pid>`. The labs stop it themselves on normal exit and Ctrl-C |

## Next

Continue to [Lab 09 — fine-tune with LLaMA Factory](../09_llama_factory/TUTORIAL.md): train LoRA, QLoRA and full fine-tunes on your Spark. Module 13 then serves the result and routes it through this gateway.
