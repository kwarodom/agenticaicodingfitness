# DGX Spark Runbook — what runs where, and how the app should detect it

**Audience:** the Stack Navigator backend/frontend agents.
**Ground rules:** every stack layer gets a verdict — `SPARK-1` (runs on one DGX Spark),
`SPARK-2` (needs two Sparks over the QSFP 200GbE link), `CLOUD` (build.nvidia.com NIM API),
or `SIM` (simulate; real commands still printed). The app must reuse the exact
`DGX_CONN` resolution pattern from `week23/01_nemotron_models/config.py` (copied per-app,
Week 23 convention) and the SSH console from `dgxsh.py`.

Sources: `week23/01_nemotron_models/config.py` (DGX_SPECS + connection logic),
`week19/sovereign_dgx/demos/step03…step08` (verified-on-Spark commands),
`week23/README.md`, NVIDIA dgx-spark-playbooks, build.nvidia.com. Items marked
**[UNCERTAIN]** were not verifiable from the repo or a primary source — probe at runtime,
never hardcode.

---

## 1. DGX Spark capability matrix

### 1.1 GB10 hardware facts (mirror of `config.DGX_SPECS["DGX Spark"]` — keep in sync)

| Fact | Value |
|---|---|
| Chip | NVIDIA GB10 Grace Blackwell Superchip |
| Memory | **128 GB LPDDR5X unified** (CPU+GPU coherent — no host/device copies) |
| Memory bandwidth | **273 GB/s** (this, not FLOPs, bounds single-stream decode tok/s) |
| Compute | ~**1 PFLOP sparse FP4** (1000 TOPS); NVFP4/FP8 native on Blackwell |
| CPU | 20-core Arm (10x Cortex-X925 + 10x A725) — **aarch64/ARM64, DGX OS (Ubuntu-based)** |
| NIC | ConnectX-7 **200GbE QSFP** (the 2-Spark link; NOT the RJ45 jack) |
| Power | 240 W |
| Practical model ceiling | ~**200B params on 1 Spark** (quantized); ~**405B on 2 Sparks** |

ARM64 is the single biggest ops caveat: every container must be `linux/arm64`
(NGC ships Spark-specific tags; random Docker Hub x86 images will not run).

### 1.2 Memory rule of thumb (weights only; multiply by ~1.18 for KV-cache/runtime overhead)

| Precision | GB per B params | 128 GB Spark fits (x1.18 overhead, ~90% usable) |
|---|---|---|
| FP16/BF16 | ~2.0 | ~48B |
| Q8 / FP8 | ~1.06 | ~92B |
| Q4 (GGUF q4_K_M) | ~0.6 | ~160B |
| NVFP4 | ~0.55 | ~**177B** (repo uses `128*0.9/(0.55*1.18)` = same formula for N Sparks) |

Long context eats the margin: KV-cache grows linearly with context; budget 10–30 GB extra
for agentic (100k+ token) workloads.

### 1.3 Nemotron 3 tier placement (the family taught in App 01)

| Tier | Size | 1 Spark | 2 Sparks (TP=2, 256 GB) | Cloud |
|---|---|---|---|---|
| **Nano 30B-A3B** (incl. Omni/Reasoning) | 30B MoE, ~3B active | **YES** — comfortable at Q8 (~35 GB), fast (MoE, few active params) | n/a | yes |
| **Super 120B-A12B** | 120B MoE, ~12B active | **YES at Q4/NVFP4** (~78 GB weights); tight at Q8 (~150 GB → no) | comfortable | yes |
| **Ultra 550B-A55B** | 550B MoE | no (needs ~357 GB even at NVFP4) | **no** (550B x ~0.65 GB/B ≈ 357 GB > 256 GB; also above the ~405B 2-Spark ceiling only in memory-per-param terms — it simply does not fit) | **CLOUD-ONLY** |
| RAG / Speech / Safety (small aux models) | <10B | YES | n/a | yes |

Practical default on the user's Spark today (empirically verified, per `config._PREFERRED`):
`qwen3.6:35b-a3b-q8_0` is the workhorse (~3x tok/s of `nemotron-3-super:120b` on GB10);
Super is one dropdown click away; `nemotron-3-nano` is a small *reasoning* model — don't
use it for terse classify() calls.

2-Spark recipe (App 01 ch.6 / week19 step08): cable QSFP-to-QSFP, static IPs
(e.g. 192.168.100.10/.11), verify with NCCL `all_reduce_perf` (expect tens of GB/s busbw),
then `mpirun -H ip0,ip1 -np 2 trtllm-serve <model> --tp_size 2 --port 8355`.

---

## 2. Per-stack-layer runbook

Conventions: commands prefixed `[SPARK]` run on the DGX (via `dgxsh.py` SSH console or
directly); `{base}` is the resolved OpenAI-compatible base URL. Every layer's probe should
use the shared 2 s-timeout GET described in section 4.

### 2.1 Ollama on Spark — verdict SPARK-1 (the default REAL path)

```bash
[SPARK] curl -fsSL https://ollama.com/install.sh | sh        # needs Ollama 0.15+ for Blackwell CUDA
[SPARK] ollama pull qwen3.6:35b-a3b-q8_0                     # 35B MoE default workhorse
[SPARK] ollama pull nemotron-3-nano                          # reasoning tier
# already serving — no extra start command needed
```
- **Port:** `11434`. OpenAI API at `/v1`, native API at `/api`.
- **Expose off-box:** `OLLAMA_HOST=0.0.0.0 ollama serve`, or Tailscale (the repo default:
  `DGX_TUNNEL_URL=http://<spark>.<tailnet>.ts.net:11434/v1`).
- **Probe:** `GET {base}/models` (OpenAI shape) **and** `GET {base minus /v1}/api/tags`
  (Ollama native) — `config.endpoint_up()` already tries both, local-first order.
- Bigger context: `/set parameter num_ctx 40960` (or `OLLAMA_CONTEXT_LENGTH`).

### 2.2 vLLM — verdict SPARK-1 (concurrency / production API)

Use NVIDIA's ARM64 container from NGC, not pip (Blackwell + aarch64 wheels are the trap):
```bash
[SPARK] export HF_TOKEN=hf_...   VLLM_TAG=26.05.post1-py3    # check catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm
[SPARK] docker pull nvcr.io/nvidia/vllm:$VLLM_TAG
[SPARK] docker run -d --name vllm --gpus all --ipc host -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  nvcr.io/nvidia/vllm:$VLLM_TAG vllm serve openai/gpt-oss-20b \
    --gpu-memory-utilization 0.9 --max-model-len 8192
```
- **Port:** `8000` → base `http://<spark>:8000/v1`.
- **Probe:** `GET {base}/models` (also `GET /health` returns 200 on vLLM).
- Rule of thumb (playbooks): concurrency → vLLM/SGLang; one user → Ollama/llama.cpp;
  on the Spark's ARM64 container vLLM often *beats* TRT-LLM — don't assume TRT wins.

### 2.3 TensorRT-LLM — verdict SPARK-1 (single model, max effort) / SPARK-2 (big models)

```bash
[SPARK] docker pull nvcr.io/nvidia/tensorrt-llm/release:<latest-arm64-tag>   # check NGC for Spark tag
[SPARK] docker run -d --gpus all --ipc host -p 8000:8000 <image> \
  trtllm-serve nvidia/Nemotron-3-Nano-... --port 8000
# 2-Spark tensor parallel (the Ultra/235B story):
[SPARK] mpirun -H 192.168.100.10,192.168.100.11 -np 2 \
  trtllm-serve nvidia/Qwen3-235B-A22B-NVFP4 --tp_size 2 --port 8355
```
- **Ports:** `8000` single-node (repo convention), `8355` in the 2-Spark demo.
- **Probe:** `GET {base}/models` (trtllm-serve is OpenAI-compatible).
- Caveat: compile/engine-build step on first serve; NVIDIA-specific; ARM64 container required.

### 2.4 NIM containers on Spark — verdict SPARK-1 for the small set of ARM64 NIMs, else CLOUD

Only NIMs with a **`-dgx-spark`** (ARM64+Blackwell) build run locally. Verified examples:
Llama 3.1 8B Instruct, Qwen3-32B. Most catalog NIMs are x86-only → use the cloud API.
```bash
[SPARK] echo $NGC_API_KEY | docker login nvcr.io --username '$oauthtoken' --password-stdin
[SPARK] docker run -d --name nim --gpus all --shm-size 16GB -p 8000:8000 \
  -e NGC_API_KEY=$NGC_API_KEY -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  nvcr.io/nim/meta/llama-3.1-8b-instruct-dgx-spark:latest
```
- **Port:** `8000` → `http://<spark>:8000/v1`.
- **Probe:** `GET {base}/models`; NIMs also expose `GET /v1/health/ready` (200 when warm).
- Caveats: 10–50 GB weight download on first start; `--shm-size 16GB` minimum; NGC key is an
  86-char base64 string ending `==`; production needs NVIDIA AI Enterprise (bundled with DGX);
  free to pull for dev/eval. Nemotron-3 **Nano** NIM for Spark: **[UNCERTAIN]** — check
  `nvcr.io/nim/nvidia/` for a `-dgx-spark` tag at runtime; fall back to Ollama's
  `nemotron-3-nano` pull or cloud.

### 2.5 Dynamo — verdict SPARK-1 (single-node demo) / SPARK-2 (real disaggregation) / SIM

```bash
[SPARK] uv pip install "ai-dynamo[vllm]"          # or the nvcr.io dynamo container
[SPARK] docker compose up -d etcd nats            # Dynamo needs etcd + NATS control plane
[SPARK] python -m dynamo.frontend --http-port 8000 &
[SPARK] python -m dynamo.vllm --model Qwen/Qwen3-0.6B     # worker registers with the frontend
```
- **Port:** frontend `8000` (OpenAI-compatible); etcd `2379`, NATS `4222`.
- **Probe:** `GET {base}/models` on the frontend.
- Real disaggregated prefill/decode + NIXL KV transfer needs 2 nodes → 2 Sparks over the
  200GbE link; on 1 Spark the app should run Dynamo single-pool and **SIM** the scaling
  charts (that is exactly what App 03 does). **[UNCERTAIN]**: exact module invocation names
  drift between Dynamo releases — treat the commands as the documented pattern, verify
  against `docs.dynamo.nvidia.com` when the console is live.

### 2.6 NeMo Agent Toolkit (NAT, formerly AIQ toolkit) — verdict SPARK-1 (pure Python, any arch)

```bash
[SPARK] uv pip install "nvidia-nat[all]"           # package renamed from aiqtoolkit; both names may resolve
[SPARK] nat serve --config_file workflow.yaml --host 0.0.0.0 --port 8001
```
- **Port:** `8000` by default — collides with vLLM/NIM, so the navigator should run it on
  `8001` **[UNCERTAIN: default port; verify `nat serve --help`]**.
- **Probe:** it serves FastAPI — `GET {base_without_v1}/docs` (200) is the reliable liveness
  check; the workflow endpoint is `POST /generate`.
- NAT is CPU-side orchestration calling any OpenAI endpoint — no GPU/ARM constraints; point
  its `llms:` block at the Ollama/vLLM/NIM base URL from section 4.
- Observability hook (week19 step07, verified pattern): add
  `general.telemetry.tracing.phoenix: {_type: phoenix, endpoint: http://localhost:6006/v1/traces, project: ...}`.

### 2.7 AI-Q Research Blueprint — verdict CLOUD-assisted SPARK-1, mostly SIM

The full blueprint (`NVIDIA-AI-Blueprints/aiq-research-assistant`) is a docker-compose of
many services (frontend, backend, NeMo Retriever NIMs, Milvus). The retrieval/rerank NIMs
are x86-heavy; on a Spark run the *agent* locally and point model slots at cloud NIMs:
```bash
[SPARK] git clone https://github.com/NVIDIA-AI-Blueprints/aiq-research-assistant && cd aiq-research-assistant
[SPARK] export NVIDIA_API_KEY=nvapi-...            # cloud NIMs for embed/rerank/ingest
[SPARK] docker compose -f deploy/compose.yaml up -d    # [UNCERTAIN: exact compose path/profiles]
```
- **Ports [UNCERTAIN]:** backend ~`8051`, frontend ~`3000` in the blueprint defaults — probe,
  don't hardcode.
- **Probe:** `GET http://<spark>:8051/health` else fall back to SIM (App 05 ships `sim.py`).
- Verdict for the navigator: mark AI-Q **SIM by default**, REAL when its backend health
  endpoint answers.

### 2.8 NeMo Guardrails — verdict SPARK-1 (pure Python)

```bash
[SPARK] uv pip install nemoguardrails
[SPARK] nemoguardrails server --config ./rails_config --port 8500
```
- **Port:** server default is `8000` → run on `8500` to avoid the vLLM/NIM collision
  **[UNCERTAIN: confirm default with `nemoguardrails server --help`]**.
- **Probe:** `GET http://<spark>:8500/v1/rails/configs` (the server lists configs there)
  **[UNCERTAIN — if 404, fall back to `GET /docs`]**.
- The rails' `models:` block points at the local Ollama/vLLM endpoint (`engine: openai`,
  custom `base_url`) — fully sovereign. OpenShell/NemoClaw runtime pieces: treat as **SIM**
  (App 07 pattern); no public installable verified.

### 2.9 Phoenix / NeMo Relay observability — verdict SPARK-1 (Phoenix), SIM (Relay)

```bash
[SPARK] uv pip install arize-phoenix
[SPARK] phoenix serve            # UI + OTLP-HTTP on :6006, OTLP-gRPC on :4317
```
- **Ports:** `6006` (UI + `/v1/traces` OTLP-HTTP), `4317` (gRPC).
- **Probe:** `GET http://<spark>:6006` (any 200) — matches week19 `phoenix_up()`
  (`PHOENIX_ENDPOINT`, default `http://localhost:6006`, 2 s timeout).
- Run Phoenix on the DGX/VPN only — traces contain prompts/PII.
- **NeMo Relay** (App 08): productized router/gateway; no verified public container yet →
  **SIM**, with the OTel-to-Phoenix path shown as the real substrate. **[UNCERTAIN:
  Relay availability — re-check build.nvidia.com.]**

### 2.10 NeMo RL / NeMo Gym — verdict SPARK-1 for small-policy GRPO demos, SPARK-2 or SIM beyond

```bash
[SPARK] git clone https://github.com/NVIDIA-NeMo/RL nemo-rl && cd nemo-rl
[SPARK] uv sync                                     # ARM64 wheels: torch OK on aarch64+CUDA via NGC pytorch container if pip fails
[SPARK] uv run python examples/run_grpo_math.py policy.model_name=Qwen/Qwen2.5-1.5B cluster.gpus_per_node=1
```
- Sizing: GRPO on a 1–8B policy fits 1 Spark (training needs ~4–6x weight memory for
  optimizer/grads/rollouts); Nemotron Nano 30B LoRA is borderline; anything bigger →
  2 Sparks (`cluster.tp=2`, App 10's framing) or SIM.
- **No serving port** — it is a batch job. The navigator should treat this layer's REAL
  signal as "SSH reachable + `nvidia-smi` shows a training process", i.e. use `dgxsh.status()`
  rather than an HTTP probe; otherwise SIM.
- Safer path if pip wheels fight ARM64: run inside `nvcr.io/nvidia/pytorch:25.11-py3`
  (the same container week19 uses for NCCL tests).

### 2.11 Data Flywheel (Curator → Customizer → Evaluator) — verdict SIM on Spark, CLOUD/DC for real

The blueprint (`github.com/NVIDIA-AI-Blueprints/data-flywheel`) runs on **NeMo microservices,
which want a Kubernetes cluster with x86 + datacenter GPUs** — not a fit for one ARM64 Spark.
- Realistic Spark story: OBSERVE on the Spark (Phoenix traces), CURATE with
  `nemo-curator` (pip, CPU-ok), CUSTOMIZE via week19's `dgx_finetune` LoRA path
  (PEFT on the Spark works for <=12B), EVALUATE with an LLM-judge against the local endpoint.
- **Probe:** none — mark **SIM** unless the user points `FLYWHEEL_BASE_URL` at a real NeMo
  microservices deployment (`GET {url}/v1/models` **[UNCERTAIN: NMP health route]**).

### 2.12 Verdict summary table (what the navigator UI should render)

| Layer | 1 Spark | 2 Sparks | Cloud | Default port | Probe |
|---|---|---|---|---|---|
| Ollama | REAL | — | — | 11434 | `/v1/models` or `/api/tags` |
| vLLM | REAL | — | — | 8000 | `/v1/models` |
| TensorRT-LLM | REAL | REAL (TP=2) | — | 8000 / 8355 | `/v1/models` |
| NIM | REAL (ARM64 tags only) | — | REAL | 8000 | `/v1/models`, `/v1/health/ready` |
| Dynamo | REAL (1 pool) | REAL (disagg) | — | 8000 (+2379/4222) | `/v1/models` |
| NeMo Agent Toolkit | REAL | — | — | 8001 (moved) | `/docs` |
| AI-Q | partial (cloud NIMs) | — | REAL | ~8051 | `/health` [UNCERTAIN] |
| Guardrails | REAL | — | — | 8500 (moved) | `/v1/rails/configs` [UNCERTAIN] |
| Phoenix | REAL | — | — | 6006 (+4317) | `GET /` |
| NeMo Relay | SIM | — | ? | — | — |
| NeMo RL/Gym | REAL (small) | REAL (tp=2) | — | none | SSH `nvidia-smi` |
| Data Flywheel | SIM | — | DC/k8s | — | — |
| Nemotron Ultra 550B | — | — | **CLOUD-ONLY** | — | cloud `/v1/models` |

---

## 3. build.nvidia.com cloud path (the on-ramp)

1. **Get a key:** sign in at `https://build.nvidia.com` (free NVIDIA Developer Program
   account, no credit card), open any model page, click "Get API Key" / "Build with this NIM".
   Key starts with **`nvapi-`**; reported valid ~6 months.
2. **Endpoint:** `https://integrate.api.nvidia.com/v1` — fully OpenAI-compatible
   (`/chat/completions`, `/models`, streaming). This is exactly the repo's
   `DGX_CONN=cloud` + `DGX_CLOUD_URL` + `DGX_API_KEY` preset (`week23 README`, "Cloud on-ramp").
3. **Curl example:**
```bash
curl https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
       "messages":[{"role":"user","content":"What is NVFP4?"}],
       "max_tokens":512, "stream":false}'
```
4. **Model IDs** (list live IDs via `GET /v1/models` — the app should never hardcode):
   - `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` — **verified** on build.nvidia.com.
   - `nvidia/nemotron-3-super-...` and `nvidia/nemotron-3-ultra-...` — **[UNCERTAIN exact
     suffixes]**; a developer-forum post confirms Nemotron 3 Super is served but some
     personal orgs hit 403 "missing public API endpoints permission" — surface that error
     hint in the UI.
   - Older stable fallbacks: `nvidia/nvidia-nemotron-nano-9b-v2`,
     `nvidia/llama-3.1-nemotron-70b-instruct`, `meta/llama-3.1-8b-instruct`.
   - Note `config._PREFERRED` matches by substring, so namespaced cloud IDs containing
     "nemotron" are picked up automatically.
5. **Limits:** free tier is rate-limited (~**40 requests/minute** commonly reported;
   NVIDIA does not publish exact per-model limits — your ceiling shows in your account).
   New accounts get starter inference credits; messaging has shifted toward rate-limited
   trial rather than hard credit counts. **[UNCERTAIN: exact credit numbers — do not print
   specific credit amounts in the UI.]** Cloud mode must display the repo's honest label:
   `via cloud provider · cloud usage billed` (`config.cost_note()`), i.e. NOT sovereign.

---

## 4. Connection probing spec (for the navigator backend)

Reuse `_resolve_connection()` + `endpoint_up()` + `list_local_models()` from
`week23/01_nemotron_models/config.py` verbatim, with these normative rules:

**Resolution order (first match wins):**
1. **`DGX_BASE_URL`** (or legacy `EDGE_BASE_URL`) explicit → use as-is; infer the label
   from the hostname (localhost/RFC1918/`.local` → `local`; ngrok/trycloudflare/loca.lt/
   `ts.net`/"tunnel" → `tunnel`; else `cloud`).
2. **`DGX_CONN`** = `local` → `http://localhost:11434/v1`; `tunnel` → `DGX_TUNNEL_URL`
   (default `http://your-spark.your-tailnet.ts.net:11434/v1`); `cloud` → `DGX_CLOUD_URL`
   (repo default `https://ollama.com/v1`; the navigator should prefer the
   `https://integrate.api.nvidia.com/v1` preset).
3. **Default:** `tunnel` to the Spark; if unreachable AND neither `DGX_CONN` nor
   `DGX_BASE_URL` was set, fall back once to `local` `http://localhost:11434/v1`
   (Ollama default) before surrendering to SIM. `DGX_MODE=sim|real` force-overrides.

**Probe procedure (per endpoint):**
- Timeout **2 s** per request (config.py uses 3–4 s; the navigator wants snappier UI —
  2 s is the spec; probes run in a background thread, never block page render).
- Try, in order (reverse the pair when `CONN == "cloud"` since cloud has no Ollama API):
  1. `GET {base}/models` — OpenAI shape (`{"data":[{"id":...}]}`) — vLLM/NIM/TRT/Dynamo/cloud.
  2. `GET {base minus '/v1'}/api/tags` — Ollama native (`{"models":[{"name":...}]}`).
- **Auth header:** `Authorization: Bearer {DGX_API_KEY}` only when key set and CONN != local;
  if the key contains `:` treat it as basic-auth `user:pass` (ngrok `--basic-auth` tunnels);
  Anthropic hosts get `x-api-key` + `anthropic-version` instead (all already implemented in
  `config._open()`).
- Any 2xx on either probe → **REAL** for that endpoint; both fail → **SIM**.
- Auto-append `/v1` if the user-supplied URL has an empty path (`apply_connection._norm`).

**Per-layer probes on the Spark host** (run after the base connection resolves; `H` = Spark
hostname from `DGX_SSH_HOST` or the tunnel URL): Ollama `H:11434`, vLLM/NIM/TRT/Dynamo
`H:8000/v1/models` (+ NIM `H:8000/v1/health/ready`), NAT `H:8001/docs`, Guardrails
`H:8500/v1/rails/configs`, Phoenix `H:6006/`, AI-Q `H:8051/health`. Anything unprobeable
over HTTP (NeMo RL, flywheel) uses `dgxsh.status()` (SSH, `BatchMode=yes`,
`ConnectTimeout=8`, env `DGX_SSH_HOST`/`DGX_SSH_USER`/`DGX_SSH_PORT`/`DGX_SSH_KEY`).

**Port bookkeeping:** Week 23 apps own 8100–8111 on the laptop; the navigator must not
collide (pick e.g. 8099 or 8112). On the Spark, 8000 is contested (vLLM vs NIM vs TRT vs
Dynamo frontend) — only one can be REAL at a time; the navigator should report *which*
service answered by inspecting the `/v1/models` payload (Ollama lists many models; a NIM
usually lists exactly one; vLLM lists the single `--model` it was started with).
