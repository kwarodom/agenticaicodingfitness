# ▶ Hands-on Lab 02 — NIM Microservices

> Part of Week 23 · The Open Superintelligence Stack. Companion explainer app: `.venv/bin/python week23/02_nim_microservices/tutorial_server.py` → http://127.0.0.1:8101. This lab is the DO-IT side: you type the commands, you see the output.

**What you'll actually do**
- Get an `nvapi-` key from build.nvidia.com and call a **hosted NIM** with curl and the OpenAI SDK.
- Deploy a **real NIM container** on a DGX Spark (`docker login nvcr.io` → one `docker run` → `/v1` endpoint on :8000).
- Fingerprint an unknown endpoint like an ops engineer: which of NIM / Ollama / vLLM is behind `/v1`?
- Do the **base_url swap drill** — migrate a client Ollama → NIM → cloud with zero code changes.
- Run the **fit math**: decide which Nemotron NIMs a 128 GB Spark can actually load, before pulling 50 GB.

**Time** ~45 min · **Difficulty** beginner→intermediate · **Cost** $0 on Spark/local; cloud path is usage-billed.

## 0 · Pick your path

| Path | You have | You'll use |
|---|---|---|
| A 🖥️ | a DGX Spark (or any Linux GPU box) | real commands over SSH |
| B ☁️ | an nvapi- key from build.nvidia.com | hosted NIM endpoints, same API |
| C 💻 | just this laptop | local Ollama (localhost:11434) or the app's SIM mode |

(every numbered step below marks which paths it applies to)

## 1 · Get your key at build.nvidia.com (A + B)

Everything NIM starts with one free key — it is both your NGC container-pull credential and your hosted-API bearer token.

```
open https://build.nvidia.com
# sign in (free NVIDIA Developer account, no credit card) → open any model page
# → click "Get API Key" — the key starts with nvapi- and is ~86 chars ending ==
export NGC_API_KEY=nvapi-...        # for docker pulls from nvcr.io (path A)
export NVIDIA_API_KEY=$NGC_API_KEY  # for the hosted API (path B)
```

**Expected output**
```
(a key in your clipboard that looks like)
nvapi-AbC1...xyz==
```

✓ Checkpoint: `echo $NGC_API_KEY | head -c 6` prints `nvapi-`. C-path: skip this step — local Ollama needs no key.

## 2 · Call a hosted NIM — the cloud on-ramp (B; C-equivalent below)

Goal: prove the OpenAI contract against NVIDIA's hosted catalog before touching any hardware. **Never hardcode model ids** — list the live ones first (Super/Ultra id suffixes drift; verify with `/v1/models`).

```bash
# B ☁️ — list what's live (this is the source of truth for model ids)
curl -s https://integrate.api.nvidia.com/v1/models \
  -H "Authorization: Bearer $NVIDIA_API_KEY" | python3 -c \
  "import json,sys; [print(m['id']) for m in json.load(sys.stdin)['data'][:10]]"

# B ☁️ — one chat completion (id verified on build.nvidia.com)
curl -s https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
       "messages":[{"role":"user","content":"What is NVFP4? One sentence."}],
       "max_tokens":200}'
```

```bash
# C 💻 — the identical call against local Ollama (same API, different base_url)
curl -s http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"gemma4","messages":[{"role":"user","content":"What is NVFP4? One sentence."}],"max_tokens":200}'
```

**Expected output**
```
nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
meta/llama-3.1-8b-instruct
... (dozens more)
{"choices":[{"message":{"content":"NVFP4 is NVIDIA's 4-bit floating-point
format that shrinks model weights ~4x with minimal accuracy loss..."}}], ...}
```

✓ Checkpoint: you got a JSON completion back from `integrate.api.nvidia.com/v1` (or `localhost:11434/v1`). Note the honest label: the cloud path is usage-billed and off-box — convenient, **not sovereign**. Free tier is rate-limited (~40 req/min commonly reported); a 403 "missing public API endpoints permission" means your org/account lacks access to that model — pick another id from step 1's list.

## 3 · Deploy a NIM on the Spark — one signed container (A; C-equivalent below)

Goal: one `docker run` → model + auto-selected optimized engine (TensorRT-LLM/vLLM/SGLang) + OpenAI API on :8000. **Spark caveat first**: the GB10 is aarch64 — only NIMs with a `-dgx-spark` (ARM64+Blackwell) build run locally. Verified: `llama-3.1-8b-instruct`, `qwen3-32b`. A Nemotron-3 Nano NIM for Spark is *not* verified — check `nvcr.io/nim/nvidia/` for a `-dgx-spark` tag first; fall back to Ollama's `nemotron-3-nano` pull or the cloud.

```bash
# A 🖥️ — on the Spark (ssh in, or use the app's 🖥️ DGX console)
echo $NGC_API_KEY | docker login nvcr.io --username '$oauthtoken' --password-stdin

export LOCAL_NIM_CACHE=~/.cache/nim && mkdir -p "$LOCAL_NIM_CACHE"
docker run -d --name nim --gpus all --shm-size 16GB -p 8000:8000 \
  -e NGC_API_KEY=$NGC_API_KEY -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  nvcr.io/nim/meta/llama-3.1-8b-instruct-dgx-spark:latest

# first start downloads 10–50 GB of weights — watch it, then poll readiness:
docker logs -f nim            # Ctrl-C when you see "Uvicorn running"
curl -s http://localhost:8000/v1/health/ready && echo " ← NIM is warm"
curl -s http://localhost:8000/v1/models
```

**Expected output**
```
Login Succeeded
6f1e2a...   (container id)
... [NIM] downloading engine profile for GB10 ... Uvicorn running on 0.0.0.0:8000
{"object":"health.response","message":"Service is ready."} ← NIM is warm
{"object":"list","data":[{"id":"meta/llama-3.1-8b-instruct", ...}]}
```

✓ Checkpoint: `/v1/health/ready` returns 200 and `/v1/models` lists exactly one model — the classic NIM fingerprint.

```bash
# C 💻 — no Spark? The concept-equivalent on this laptop is one Ollama pull:
ollama run gemma4        # one command → model + engine + OpenAI API on :11434/v1
# or run the explainer app in SIM: .venv/bin/python week23/02_nim_microservices/tutorial_server.py
```

Also see `demos/step01_deploy_nim.py` (Chapter 2 in the web app) — it walks this same deploy narratively in REAL or SIM mode.

## 4 · Point this repo at your endpoint (A + B + C)

Goal: make every demo and lab in this folder hit *your* endpoint, using the repo's standard `DGX_*` env resolution (`config.py`).

```bash
# A 🖥️ — your Spark NIM (LAN or Tailscale hostname)
export DGX_BASE_URL=http://<spark>:8000/v1

# B ☁️ — build.nvidia.com hosted NIMs
export DGX_CONN=cloud DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1 DGX_API_KEY=$NVIDIA_API_KEY

# C 💻 — local Ollama (also the automatic fallback when nothing else is set)
export DGX_CONN=local
```

**Expected output** (verify with the quick probe)
```
$ .venv/bin/python -c "import sys; sys.path.insert(0,'week23/02_nim_microservices'); \
import config; print(config.MODE, config.CONN, config.safe_base_url(), config.MODEL)"
real local http://localhost:11434/v1 qwen3.6:35b-a3b-q8_0
```
(the last field is whatever model your endpoint actually serves — e.g. `gemma4:12b` on a laptop Ollama, `meta/llama-3.1-8b-instruct` on a Spark NIM)

✓ Checkpoint: `config.MODE` prints `real` (or `sim` if nothing is up — every lab still runs and shows the real commands).

## 5 · Call your NIM — same OpenAI API, three ways (A + B + C)

Goal: the punchline of this layer — the client never knows (or cares) what serves it.

```bash
# curl (swap the base for :11434/v1 on path C, integrate.api.nvidia.com/v1 on B)
curl -s http://<spark>:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"meta/llama-3.1-8b-instruct","messages":[{"role":"user","content":"Hello"}],"max_tokens":60}'
```

```python
# Python — the ONLY thing that ever changes is base_url / api_key / model
from openai import OpenAI
client = OpenAI(base_url="http://<spark>:8000/v1", api_key="not-needed")
r = client.chat.completions.create(model="meta/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": "Hello"}], max_tokens=60, stream=True)
for chunk in r:
    print(chunk.choices[0].delta.content or "", end="")
```

**Expected output**
```
{"choices":[{"message":{"content":"Hello! How can I help you today?"}}], ...}
Hello! How can I help you today?
```

✓ Checkpoint: identical code worked against your endpoint. `demos/step03_call_nim.py` (Chapter 4) runs this live from the web app with streaming and tok/s metrics — the labs below make you *prove* it across endpoints.

## Labs (run these)

**labs/lab01_fingerprint_endpoint.py** — probes `/v1/models`, Ollama's native `/api/tags`, and NIM's `/v1/health/ready` against whatever `config.py` resolved, names the server type from the fingerprint (a NIM lists exactly one model *and* answers health/ready; Ollama exposes `/api/tags`; vLLM is a bare single-model OpenAI contract), then makes one real call.
```bash
cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/02_nim_microservices/labs/lab01_fingerprint_endpoint.py
```
Look for: the three probe lines and the `✓ verdict` naming your server. Modify it: add a fourth probe for vLLM's `GET /version` (root, no `/v1`) and extend the verdict logic.

**labs/lab02_base_url_swap.py** — the promotion drill: sends the same prompt with the same code to endpoint A (your config-resolved endpoint) and endpoint B (build.nvidia.com, if `NVIDIA_API_KEY=nvapi-...` is exported), streams both, and prints a side-by-side of model / first-token latency / tok/s. Zero client code differs between the two calls.
```bash
cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/02_nim_microservices/labs/lab02_base_url_swap.py
```
Look for: with both endpoints answering (a live endpoint A *and* the nvapi- key), the `side-by-side` table and the line "code changed to migrate: 0 lines"; with no key, endpoint B prints the real steps to get one plus a labeled expected-output sample. Modify it: add a third endpoint (e.g. a Tailscale-tunneled Spark) and make the table three columns.

**labs/lab03_spark_fit_check.py** — offline deploy-planning math (no GPU, no endpoint, always runs): `params_B × GB-per-B(precision) × 1.18 overhead` vs ~115 GB usable on a 128 GB Spark, applied to the Nemotron 3 family, plus the aarch64 `-dgx-spark` tag gate.
```bash
cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/02_nim_microservices/labs/lab03_spark_fit_check.py
cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/02_nim_microservices/labs/lab03_spark_fit_check.py 70 q4
```
Look for: Super fits only at Q4/NVFP4; Ultra fits nowhere (cloud-only). Modify it: add a "DGX Station" mode using `config.DGX_SPECS["DGX Station"]` and re-run the table against 784 GB.

## Try it yourself

**1. Which server am I talking to — without lab01?** Using only `curl`, determine whether `http://localhost:11434/v1` is a NIM, Ollama, or vLLM.
<details><summary>Solution</summary>

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:11434/api/tags        # 200 → Ollama
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:11434/v1/health/ready # 200 → NIM
curl -s http://localhost:11434/v1/models                                        # only this → vLLM-style
```
`/api/tags` answering is the Ollama giveaway; `/v1/health/ready` is the NIM giveaway; a NIM's `/v1/models` usually lists exactly one model, Ollama lists many.
</details>

**2. The 405B question.** Your team wants Llama-3.1-405B on Sparks at Q4. How many Sparks, and does it fit?
<details><summary>Solution</summary>

```bash
.venv/bin/python week23/02_nim_microservices/labs/lab03_spark_fit_check.py 405 q4
```
Needed ≈ 405 × 0.60 × 1.18 ≈ **287 GB** → ✗ on one Spark (~115 GB budget) and ✗ even on **two linked Sparks** (~230 GB). At **NVFP4** it needs ≈ 405 × 0.55 × 1.18 ≈ 263 GB — still over 230. The runbook's "~405B on 2 Sparks" ceiling assumes aggressive quantization and minimal KV-cache headroom; the honest answer is "borderline at best — verify on the hardware, or use the cloud NIM." The fit-math lab exists precisely so you catch this before a 200 GB pull.
</details>

**3. Wrap your own model.** Sketch (don't run) the `docker run` that would serve your Week-19 fine-tuned LoRA as a NIM.
<details><summary>Solution</summary>

```bash
docker run --gpus all -p 8000:8000 \
  -e NIM_FT_MODEL=/models/hvac-assistant \
  -v $PWD/hvac-assistant:/models/hvac-assistant \
  nvcr.io/nim/nvidia/llm-nim:latest
```
This is the pattern from `demos/step04_catalog_custom.py` (Chapter 5) — the generic `llm-nim` base image serving your weights with a production engine + OpenAI API. On a Spark, first verify the image ships an aarch64 build.
</details>

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `404 model not found` on a completion | wrong model id for this endpoint | `GET {base}/v1/models` and use an id from THAT list — ids differ per server (Ollama tag vs `nvidia/...` cloud id) |
| `401 Unauthorized` | missing/expired key | cloud needs `Authorization: Bearer nvapi-...`; ngrok tunnels may need basic-auth (`user:pass` in the URL or key) |
| `403 missing public API endpoints permission` | your build.nvidia.com org can't access that hosted model | pick another id from `/v1/models`; known issue on some personal orgs for Nemotron Super |
| `404` on every route / `Connection refused` | base URL missing the port or `/v1` | Ollama is `:11434/v1`, NIM/vLLM are `:8000/v1`; the app's 🔌 Connection auto-appends `/v1` |
| `exec format error` at `docker run` on the Spark | x86-only image on aarch64 | only pull NIMs with a `-dgx-spark` / ARM64 tag; check the container page first |
| NIM container up but calls hang/refuse | engine still warming (10–50 GB first-start download) | `docker logs -f nim`; wait for `GET /v1/health/ready` → 200 |
| `port 8000 already allocated` | vLLM / TRT / Dynamo / another NIM owns :8000 | only one can be REAL at a time — `docker rm -f nim`, or map `-p 8001:8000` and point `DGX_BASE_URL` at :8001/v1 |
| Everything prints `MODE: SIM` | no endpoint reachable | that's a feature — real commands still shown; bring up Ollama or set `DGX_BASE_URL` to go REAL |

## Next

→ ../03_dynamo_serving/TUTORIAL.md (NVIDIA Dynamo — disaggregated prefill/decode serving, KV-cache-aware routing, SLO planner) — one NIM serves one box; Dynamo is what turns many of these OpenAI-compatible workers into a datacenter-scale, SLO-aware serving fleet.
