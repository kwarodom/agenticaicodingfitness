# ▣ Week 19 — Sovereign & Self-Evolving AI on a DGX

A **guided, hands-on walkthrough**. Week 19 is five interactive web apps (in the
Week 18 style) that take the agent stack you've built all course and run it
**sovereignly on an NVIDIA DGX** — the model, the serving gateway, the
fine-tuning, the observability, and a self-evolving agent's memory, all on your
own hardware. Grounded in NVIDIA's
[`dgx-spark-playbooks`](https://github.com/NVIDIA/dgx-spark-playbooks).

This page walks you through them **in order**, chapter by chapter. Follow it top
to bottom; each app builds on the one before.

> **You do NOT need a DGX (or any GPU) to do this course.** Every app auto-detects
> its mode: **REAL** if a live model endpoint is reachable, otherwise **SIM** — a
> faithful simulator so you can learn every concept offline. Cloud cost is always **$0**.

---

## 💻 vs 🖥️ — which commands run where (READ THIS FIRST)

This is the one thing to get straight. There are **two machines**, and they do
**different jobs**:

| | 💻 **Your laptop** | 🖥️ **The DGX Spark** |
|---|---|---|
| **Its job** | runs the **tutorial** (the teaching web apps + your browser) | runs the **real AI workloads** (the actual models, training, serving) |
| **What you run on it** | `uv pip install …` and `.venv/bin/python …/tutorial_server.py` — then open the page in a browser | the GPU commands the chapters show you: `docker run … nvcr.io/nvidia/…`, `trtllm-serve`, `nemo-automodel`, `litellm --config`, big `ollama` models, `mpirun` across Sparks |
| **Needs a GPU?** | No | Yes (that's the whole point of the DGX) |

**The golden rule:**

- 💻 **Every command in THIS file** (the `uv pip install` and `tutorial_server.py`
  lines) runs on **your laptop**. That's how you launch the tutorial.
- 🖥️ **Every command a chapter PRINTS inside the app** (anything with `docker`,
  `nvcr.io/...`, `trtllm-serve`, `nemo`, `litellm`, `mpirun`, or pulling a 35B+
  model) is a **DGX Spark** command. The app *shows* it so you can copy-paste it
  onto a real DGX later — it does **not** run it on your laptop.

So: **you drive the tutorial from your laptop; the DGX commands are the homework
you run on real hardware when you have it.**

### Pick your setup

**Setup A — Laptop only, no DGX (the default, $0).**
Do nothing extra. Launch any app on your laptop; it runs in **SIM** and simulates
the DGX. You see every DGX command printed, but nothing GPU-heavy executes. Best
for learning the concepts.

**Setup B — Laptop with a small local model (REAL, on a Mac/PC).**
Run a *small* model on the laptop itself so generations are real:
```bash
# 💻 on your laptop
curl -fsSL https://ollama.com/install.sh | sh
ollama run gemma4:12b          # small enough for a laptop
```
The apps auto-detect `http://localhost:11434` and flip to **REAL**. The big-iron
commands (vLLM containers, 70B fine-tunes, multi-Spark) are still just *shown* —
your laptop can't run those, and that's fine.

**Setup C — You have a DGX Spark (fully REAL).** Use the **connection switch**
below — it's the same for all five apps.

You can switch setups any time — the apps re-detect on each launch.

---

## 🔌 The connection switch — `DGX_CONN` (local · tunnel · cloud)

Your DGX might be on your desk, in another building, or you might not have one at
all. All five apps share **one switch** for *where the model lives*. Set it on
your laptop (💻) before launching an app:

```bash
export DGX_CONN=local     # ① a DGX on your LAN / this laptop   (default)
export DGX_CONN=tunnel    # ② a DGX on another network, over a tunnel (ngrok/cloudflared/tailscale)
export DGX_CONN=cloud     # ③ a hosted provider (Ollama Cloud, etc.) or Claude (App 4)
```

The active connection is shown in each app's **mode pill** (top-right) and in the
run output (`connection = …`). If the chosen endpoint isn't reachable, the app
simply falls back to **SIM** — you're never stuck.

### ① `local` — DGX on your LAN (or this laptop)
```bash
# 🖥️ on the DGX — serve a model
ollama run qwen3.6:35b-a3b-q8_0
# 💻 on your laptop
export DGX_CONN=local
export DGX_BASE_URL=http://my-dgx-spark.local:11434/v1   # omit for localhost
```

### ② `tunnel` — DGX in another network, reached over the internet
Your DGX is behind a firewall/NAT. Expose its Ollama port with a tunnel, then
point the apps at the public URL. **Run the tunnel command ON the DGX (🖥️):**

```bash
# 🖥️ on the DGX — pick ONE tunnel tool:

# ngrok:
ngrok http 11434                         # → forwards https://<id>.ngrok-free.app → :11434
# Cloudflare Tunnel (no account needed for a quick tunnel):
cloudflared tunnel --url http://localhost:11434
# Tailscale (private mesh VPN — most secure, no public URL):
tailscale up                             # then use the DGX's 100.x.y.z / *.ts.net name

# 💻 on your laptop — point at the tunnel URL it printed:
export DGX_CONN=tunnel
export DGX_TUNNEL_URL=https://<id>.ngrok-free.app/v1     # note the trailing /v1
export DGX_API_KEY=<token>               # only if your tunnel adds auth (optional)
```
> ⚠️ A public tunnel exposes your model to the internet. Prefer **Tailscale**
> (private, no public URL) for anything sensitive, or add auth to ngrok/cloudflared.
> Ollama itself has no auth — never leave a bare `ngrok http 11434` up long-term.

<details>
<summary><b>📖 Full ngrok walkthrough — expose a remote DGX over the cloud, step by step</b></summary>

Two machines: 🖥️ = the DGX, 💻 = your laptop.

```
💻 laptop  ──HTTPS──▶  ngrok cloud  ──secure tunnel──▶  🖥️ DGX :11434 (Ollama)
```

#### Step 1 · 🖥️ On the DGX — make Ollama listen on all interfaces

By default Ollama only listens on `127.0.0.1`, so ngrok can't reach it. Bind it to `0.0.0.0`:

```bash
# 🖥️ DGX — edit the Ollama service
sudo systemctl edit ollama.service
```
Add these lines in the editor, save, exit:
```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```
Then reload and pull a model:
```bash
# 🖥️ DGX
sudo systemctl daemon-reload
sudo systemctl restart ollama
ollama pull qwen3.6:35b-a3b-q8_0          # the model the tutorials default to
curl http://localhost:11434/api/tags       # ✅ should list your models
```

> No systemd? Just run `OLLAMA_HOST=0.0.0.0 ollama serve` in a terminal instead.

#### Step 2 · 🖥️ On the DGX — install ngrok and add your authtoken

```bash
# 🖥️ DGX (Ubuntu/ARM64 — DGX OS)
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
  | sudo tee /etc/apt/sources.list.d/ngrok.list \
  && sudo apt update && sudo apt install ngrok

# grab your token from https://dashboard.ngrok.com/get-started/your-authtoken
ngrok config add-authtoken <YOUR_NGROK_AUTHTOKEN>
```

#### Step 3 · 🖥️ On the DGX — start the tunnel (with the two critical flags)

```bash
# 🖥️ DGX
ngrok http 11434 \
  --host-header="localhost:11434" \
  --basic-auth="dgx:choose-a-strong-password"
```

- `--host-header="localhost:11434"` — **the #1 gotcha.** Ollama rejects requests whose `Host` header isn't localhost (you'd get `403 Forbidden`). This rewrites it so Ollama accepts the tunneled request.
- `--basic-auth="user:pass"` — password-protects the public URL. **Strongly recommended** — without it, anyone with the URL can use your GPU.

ngrok prints a line like:
```
Forwarding   https://a1b2c3d4.ngrok-free.app -> http://localhost:11434
```
Copy that `https://…ngrok-free.app` URL. **Leave this terminal running** (the tunnel lives as long as the command runs).

#### Step 4 · 💻 On your laptop — verify the tunnel works

```bash
# 💻 laptop  (use the user:pass from Step 3)
curl -u dgx:choose-a-strong-password https://a1b2c3d4.ngrok-free.app/api/tags
# ✅ should return the same model list you saw on the DGX
```
If you see your models, the DGX is reachable over the cloud. 🎉

#### Step 5 · 💻 On your laptop — point the tutorials at it

```bash
# 💻 laptop
cd /Users/altodev/Desktop/agenticaicodingfitness

export DGX_CONN=tunnel
# include the basic-auth creds in the URL, and the trailing /v1:
export DGX_TUNNEL_URL=https://dgx:choose-a-strong-password@a1b2c3d4.ngrok-free.app/v1

# launch any tutorial — the pill should read "REAL · tunnel · qwen3.6…"
.venv/bin/python week19/sovereign_dgx/tutorial_server.py     # → http://127.0.0.1:8092
```

That's it — every Week 19 app now runs its real inference on your remote DGX. (If you skipped `--basic-auth`, just use `https://a1b2c3d4.ngrok-free.app/v1` with no `user:pass@`.)

#### Troubleshooting

| Symptom | Fix |
|---|---|
| `403 Forbidden` | You forgot `--host-header="localhost:11434"` on ngrok, **or** `OLLAMA_HOST` isn't `0.0.0.0` (Step 1). |
| `Connection refused` / tunnel can't reach Ollama | Ollama isn't on `0.0.0.0` — recheck Step 1 (`curl localhost:11434/api/tags` on the DGX). |
| `model not found` | `ollama pull qwen3.6:35b-a3b-q8_0` on the DGX, or set `export DGX_MODEL=<a model you have>`. |
| App shows **SIM**, not REAL | The laptop can't reach the URL — re-run the Step 4 curl; check the tunnel terminal is still running and the URL is current. |
| URL changed after restart | Free ngrok gives a **new random URL each launch** — just re-export `DGX_TUNNEL_URL`. A paid **reserved domain** (or **Tailscale**) gives a stable address. |

#### Two things worth knowing

- **Security:** a public ngrok URL exposes your GPU to the internet. Always use `--basic-auth` (above), keep the tunnel up only while you need it, and rotate the password. For sensitive work, **Tailscale** (`DGX_CONN=tunnel` with a `*.ts.net` address) is safer — a private mesh VPN with no public URL.
- **Keep it up unattended:** run the tunnel in `tmux`/`screen`, or as a small systemd unit, so it survives your SSH session closing.

</details>

### ③ `cloud` — a hosted provider, no DGX at all
```bash
# 💻 on your laptop — Ollama Cloud (or any OpenAI-compatible host):
export DGX_CONN=cloud
export DGX_CLOUD_URL=https://ollama.com/v1     # default; any OpenAI-compatible URL works
export DGX_API_KEY=<your-cloud-key>
```
**App 4** (the self-evolving agent) has an extra cloud brain — **Claude**:
```bash
export BRAIN=claude
export ANTHROPIC_API_KEY=sk-ant-...
```

### Cheat sheet — the connection env vars

| Variable | Used by | Meaning |
|---|---|---|
| `DGX_CONN` | all apps | `local` (default) · `tunnel` · `cloud` |
| `DGX_BASE_URL` | all apps | explicit endpoint — **overrides** `DGX_CONN` and auto-labels it |
| `DGX_TUNNEL_URL` | `tunnel` | the public tunnel URL (include trailing `/v1`) |
| `DGX_CLOUD_URL` | `cloud` | hosted endpoint (default `https://ollama.com/v1`) |
| `DGX_API_KEY` | tunnel/cloud | bearer token, if the endpoint needs one |
| `BRAIN` | App 4 only | `local` · `claude` · `sim` — *who* thinks (works with any `DGX_CONN`) |

> `DGX_CONN` and `BRAIN` are independent: `BRAIN` chooses **who** thinks (your
> local model vs Claude); `DGX_CONN` chooses **how** you reach the local model
> (LAN vs tunnel vs cloud Ollama).

### Quick lookup — "where do I run THIS command?"

| If the command starts with / contains… | Run it on… |
|---|---|
| `uv pip install …`, `.venv/bin/python …tutorial_server.py` | 💻 **Laptop** (launches the tutorial) |
| `export DGX_CONN=…`, `export DGX_BASE_URL=…`, `export BRAIN=…`, `export DGX_MODE=…` | 💻 **Laptop** (configures the tutorial before launch) |
| `ngrok http 11434`, `cloudflared tunnel …`, `tailscale up` | 🖥️ **DGX Spark** (exposes it so the laptop can reach it) |
| `ollama run gemma4:12b` (small model) | 💻 **Laptop** is fine (or 🖥️ DGX) |
| `ollama run qwen3.6:35b…` (large model) | 🖥️ **DGX Spark** |
| `docker run … nvcr.io/nvidia/…` (vLLM, TensorRT-LLM, NeMo) | 🖥️ **DGX Spark** |
| `trtllm-serve`, `mpirun`, NVFP4 quantization | 🖥️ **DGX Spark** |
| `nemo-automodel` / Unsloth training | 🖥️ **DGX Spark** |
| `litellm --config …` (the proxy), `python -m phoenix.server.main serve` | 🖥️ **DGX Spark** (or wherever your models run) |
| anything printed **inside a chapter's grey box** | 🖥️ **DGX Spark** (copy-paste it there when ready) |

Rule of thumb: **if it needs a GPU, it's a DGX command.** If it just opens the
tutorial or points it somewhere, it's a laptop command.

---

## Before you begin

```bash
# 💻 on your laptop — that's where the tutorial lives
cd /Users/altodev/Desktop/agenticaicodingfitness
```

**How each app launches** — these run **on your laptop (💻)**; you'll repeat this
pattern five times:

```bash
# 💻 LAPTOP
uv pip install -r week19/<app>/requirements.txt          # deps into the repo .venv (3.13)
.venv/bin/python week19/<app>/tutorial_server.py         # then open the printed http://127.0.0.1:PORT
```

In every app's web page: pick a chapter on the left → read the **concept** → click
**View demo source** → click **Run**, and watch it stream. The pill (top-right)
tells you REAL vs SIM. The 🧹 button resets that app's generated files.

> Inside the chapters you'll see grey **shell blocks** — those are the **🖥️ DGX
> Spark** commands. Read them to learn the real workflow; run them on a DGX when
> you have one. The tutorial never runs them on your laptop.

**The recommended order is just 1 → 2 → 3 → 4 → 5.** Apps 2–4 assume the ideas
from App 1; App 5 is the production serving layer that ties App 1's runtimes
together. Budget ~30–45 min per app.

---

## Stop 1 · `sovereign_dgx` — run, serve & manage models on a DGX  ·  port 8092

**Goal:** get a model running on a DGX and understand the serving runtimes
(Ollama, vLLM, llama.cpp), what fits in 128 GB, quantization, and scale-out.

```bash
# 💻 LAPTOP — launch the tutorial here
uv pip install -r week19/sovereign_dgx/requirements.txt
.venv/bin/python week19/sovereign_dgx/tutorial_server.py     # → http://127.0.0.1:8092
```

> In this app especially, the chapters print lots of **🖥️ DGX Spark** commands
> (`ollama pull`, `docker run … vllm`, `trtllm-serve`, `mpirun`). Those are for the
> DGX — your laptop just shows them. The web app itself runs fine on the laptop.

Work through the chapters in this order:

1. **Ch 1 · Why sovereign AI on a DGX** *(read)* — own the hardware, data, weights, software.
2. **Ch 2 · Your first sovereign inference** — a drop-in OpenAI call where only `base_url` changed; nothing leaves the box.
3. **Ch 3 · DGX hardware — what fits in 128 GB** — memory **bandwidth**, not TOPS, sets tok/s; see which models fit a Spark.
4. **Ch 4 · Ollama — the one-command path** — install → pull → serve; watch the GB10 fill as weights load.
5. **Ch 5 · vLLM — throughput serving** — PagedAttention + batching for many concurrent users.
6. **Ch 6 · llama.cpp — GGUF + native CUDA** — one binary, any GGUF, the Q4_K_M sweet spot.
7. **Ch 7 · Bake-off — pick the right server** — Ollama vs vLLM vs llama.cpp vs TensorRT-LLM, same OpenAI API.
8. **Ch 8 · Model management & NVFP4** — the VRAM math; quantize a 70B to fit on one Spark.
9. **Ch 9 · Scaling out — two DGX Sparks** — 200GbE link → 256 GB + tensor parallelism for a 235B.
10. **Ch 10 · Sovereignty & air-gap audit** — turn "sovereign" into a CI check.

✅ **You can now** serve any open-weight model on a DGX and reason about memory,
runtime choice, and scale.

---

## Stop 2 · `dgx_finetune` — adapt a model to YOUR domain  ·  port 8093

**Goal:** bake your own domain knowledge into the weights with LoRA/QLoRA — your
training data never leaves the building.

```bash
# 💻 LAPTOP — launch the tutorial here
uv pip install -r week19/dgx_finetune/requirements.txt
.venv/bin/python week19/dgx_finetune/tutorial_server.py      # → http://127.0.0.1:8093
```

> Training a real model is a **🖥️ DGX Spark** job — the chapters print the actual
> `nemo-automodel` / Unsloth commands and *simulate* the training run on your laptop
> (you can't train a 70B on a laptop). Dataset prep + the eval run for real anywhere.

1. **Ch 1 · Why fine-tune on a DGX** *(read)* — fine-tune changes the weights; the whole loop is sovereign.
2. **Ch 2 · Build a domain SFT dataset** — turns documents into real JSONL train/val splits in `.sandbox/`.
3. **Ch 3 · LoRA vs QLoRA vs full SFT** — the VRAM math; QLoRA fits a 70B fine-tune on one Spark.
4. **Ch 4 · The NeMo AutoModel recipe** — writes a real QLoRA YAML + the container launch command.
5. **Ch 5 · The Unsloth fast path** — ~2× faster, exports straight to GGUF for Ollama.
6. **Ch 6 · Watch the training run** — a simulated run with a real loss curve, LR schedule, checkpoints.
7. **Ch 7 · Evaluate — before vs after** — the held-out test that proves it worked (the judge is your own model in REAL mode).
8. **Ch 8 · Export, quantize & serve** — merge → GGUF/NVFP4 → Ollama; your app changes only the model name.

✅ **You can now** turn a general model into a domain expert on a DGX, and serve
the result — without a single example leaving the box.

---

## Stop 3 · `dgx_observability` — see, measure & judge the agent  ·  port 8094

**Goal:** trace a sovereign agent with OpenTelemetry → Arize Phoenix, derive the
metrics that matter, evaluate traces, and rebuild the agent with the NeMo Agent
Toolkit.

```bash
# 💻 LAPTOP — launch the tutorial here
uv pip install -r week19/dgx_observability/requirements.txt
.venv/bin/python week19/dgx_observability/tutorial_server.py # → http://127.0.0.1:8094
```

> The tracing runs on your laptop with no extra installs. To send traces to a **real
> Phoenix UI**, the chapter prints the **🖥️ DGX** commands (`pip install arize-phoenix`,
> `python -m phoenix.server.main serve`) — run those where your model runs.

1. **Ch 1 · Why observe a sovereign agent** *(read)* — tracing + metrics + evals; on a DGX the traces stay on-prem too.
2. **Ch 2 · Trace a sovereign agent** — one run = a trace of nested AGENT → LLM → TOOL spans.
3. **Ch 3 · Phoenix + OTel GenAI conventions** — Phoenix is self-hosted; spans follow `gen_ai.*`.
4. **Ch 4 · The metrics that matter on a DGX** — latency waterfall, tokens, LLM-calls, correlated with GPU util.
5. **Ch 5 · Evaluate traces (LLM-as-judge)** — catch a mis-triaged action; the judge is your own model.
6. **Ch 6 · NAT — register a tool** — `FunctionBaseConfig` + `@register_function` + `FunctionInfo`.
7. **Ch 7 · NAT — a YAML workflow on your DGX** — supervisor + HITL; the only sovereignty knob is `base_url`.
8. **Ch 8 · NAT observability → Phoenix** — one telemetry block streams every call to Phoenix on the box.

✅ **You can now** make a sovereign agent fully observable and build it the NVIDIA
way (NeMo Agent Toolkit) — all on-prem.

---

## Stop 4 · `self_evolving_agent_v2` — an agent that learns, on the DGX  ·  port 8095

**Goal:** the Week 18 self-evolving agent, made sovereign — a **switchable brain**
(DGX ↔ Claude ↔ sim) and a tripartite **memory that lives on the DGX** and gets
smarter with use. **Run Ch 3 → 4 → 5 → 6 in order** — they build shared memory.

```bash
# 💻 LAPTOP — launch the tutorial here
uv pip install -r week19/self_evolving_agent_v2/requirements.txt
.venv/bin/python week19/self_evolving_agent_v2/tutorial_server.py  # → http://127.0.0.1:8095
# pick a brain:  BRAIN=local (DGX/Ollama) · BRAIN=claude (needs ANTHROPIC_API_KEY) · BRAIN=sim
```

> The agent and its memory run wherever you launch this (laptop in SIM, or pointed
> at a 🖥️ DGX via `DGX_BASE_URL` for a real local brain). Nothing here is GPU-heavy.

1. **Ch 1 · A sovereign self-evolving agent** *(read)* — switchable brain + sovereign memory.
2. **Ch 2 · The switchable brain** — one env var swaps the model; the memory engine is identical.
3. **Ch 3 · Episodic memory** — every run appends to the on-DGX event log.
4. **Ch 4 · Consolidation (the "sleep" loop)** — distill episodes → durable facts + a reusable skill.
5. **Ch 5 · Procedural memory** — inspect the skill the agent wrote for itself.
6. **Ch 6 · Prove it evolved — cold vs warm** — the same task, before vs after recall; warm wins.
7. **Ch 7 · Sovereign-memory audit** — confirm brain AND memory are both on the DGX.

✅ **You can now** run an agent whose mind and memories never leave your hardware,
and that improves with every run.

---

## Stop 5 · `dgx_litellm` — the serving gateway (LiteLLM)  ·  port 8096

**Goal:** put one OpenAI-compatible URL in front of *all* the backends from Stop 1,
with routing, fallbacks, hot-swap, virtual keys/budgets, and logging — the
production serving layer of the sovereign stack.

```bash
# 💻 LAPTOP — launch the tutorial here
uv pip install -r week19/dgx_litellm/requirements.txt
.venv/bin/python week19/dgx_litellm/tutorial_server.py       # → http://127.0.0.1:8096
```

> The **real LiteLLM proxy** is a **🖥️ DGX** service — the chapters print
> `pip install 'litellm[proxy]'` and `litellm --config …` to run there. On your
> laptop the gateway's routing/keys/fallbacks are simulated so you see how they work.

1. **Ch 1 · Why a gateway on a DGX** *(read)* — a real deployment runs several runtimes; LiteLLM unifies them.
2. **Ch 2 · Install LiteLLM & write the config** — writes a real `config.yaml` mapping aliases → backends.
3. **Ch 3 · One endpoint, many models** — same base_url + key, only the model name changes.
4. **Ch 4 · Routing & load-balancing across Sparks** — shuffle / usage-based / latency strategies.
5. **Ch 5 · Fallbacks & reliability** — retries, cooldowns, fallback chains keep the gateway up.
6. **Ch 6 · Model management & hot-swap** — offer more models than fit in VRAM, loaded on demand.
7. **Ch 7 · Virtual keys, budgets & rate limits** — per-team allow-lists and on-prem quotas.
8. **Ch 8 · Logging & observability callbacks** — stream every call to Phoenix on the DGX (loops back to Stop 3).

✅ **You can now** serve many models in production behind a single governable,
observable endpoint — the control point of a sovereign DGX deployment.

---

## The whole journey, in one picture

```
Stop 1  sovereign_dgx ........ run & serve a model on the DGX          (:8092)
   │
Stop 2  dgx_finetune ......... make it a domain expert (LoRA/QLoRA)    (:8093)
   │
Stop 3  dgx_observability .... trace + evaluate it (Phoenix + NAT)     (:8094)
   │
Stop 4  self_evolving_agent_v2  give it sovereign, evolving memory      (:8095)
   │
Stop 5  dgx_litellm .......... front all backends with one gateway     (:8096)

   ⛔ across all five: not a prompt, weight, trace, key, or token leaves the box.
```

---

## Reference — the DGX hardware (grounded, accurate)

| | DGX Spark | DGX Station |
|---|-----------|-------------|
| Chip | GB10 Grace Blackwell | GB300 Grace Blackwell Ultra |
| Memory | **128 GB** LPDDR5X unified | up to **784 GB** coherent |
| Bandwidth | ~273 GB/s (the decode bottleneck) | HBM3e, much higher |
| Compute | ~1 PFLOP FP4 | ~20 PFLOP-class FP4 |
| Fits | ~200B params (quantized); 2 linked → ~405B | 670B-class in one box |

LLM **decode is memory-bandwidth bound**, so memory — not TOPS — decides which
model fits and how fast it runs. NVFP4 (Blackwell-native 4-bit float) is the lever
that brings 70B-class models on-desk.

---

## Reference — modes & env vars

| App(s) | Switch | Values |
|--------|--------|--------|
| all | `DGX_CONN` | `local` (default) · `tunnel` · `cloud` — *where the model lives* |
| all | `DGX_BASE_URL` | explicit endpoint — overrides `DGX_CONN`, auto-labels it |
| `tunnel` | `DGX_TUNNEL_URL` | public tunnel URL (with `/v1`) |
| `cloud` | `DGX_CLOUD_URL` | hosted endpoint (default `https://ollama.com/v1`) |
| tunnel/cloud | `DGX_API_KEY` | bearer token if the endpoint needs one |
| 1, 2, 3, 5 | `DGX_MODE` | `auto` (default) · `sim` · `real` — force simulator |
| 4 | `BRAIN` | `auto` · `local` (DGX) · `claude` · `sim` — *who* thinks |
| all | `DGX_MODEL` | pin a specific model |
| 5 | `LITELLM_BASE_URL` | a running LiteLLM proxy (default `http://localhost:4000`) |

Each app is standalone — `README.md`, `requirements.txt`, `config.py`, a
"make-it-visible" engine, `tutorial_server.py` (FastAPI), `static/guide.html`, and
a `demos/` folder with one runnable demo per chapter — exactly the Week 18 layout.
Each app's own `README.md` has a deeper dive.

---

## How Week 19 builds on the course

```
W3 tool-use → W4–5 agent loops → W9/15 multi-agent → W10 observability
→ W16 NeMo Agent Toolkit → W17 long-running/distributed → W18 sovereign edge
                                                              │
   WEEK 19: take the WHOLE stack onto a DGX — run/serve (1), fine-tune (2),
   observe with Phoenix + NAT (3), a self-evolving agent whose brain AND
   memory stay on the box (4), one LiteLLM gateway over every backend (5).
```

Earlier weeks called frontier models in the cloud. Week 19 proves the same
capabilities run **entirely on your own DGX**: private by physical design,
offline-capable, and $0 per token.
