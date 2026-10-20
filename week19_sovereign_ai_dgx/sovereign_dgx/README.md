# ▣ Week 19 · App 1 — Sovereign AI on a DGX

An **interactive, explainable web app + runnable demos** for running, serving, and
managing open-weight models **on a DGX** — entirely on-prem, $0 per token.

Where NVIDIA's [`dgx-spark-playbooks`](https://github.com/NVIDIA/dgx-spark-playbooks)
*document* the commands, this app lets you **watch them run** and narrates every
step so the "magic" of on-device AI is never a black box:

```
▣ DGX ✓  →  » PROMPT  →  ~ REASON (on-device)  →  · ANSWER  →  ◆ tok/s · $0
                                 ⛔ never leaves the box
```

## Two modes — auto-detected, no config needed

| Mode | When | What runs |
|------|------|-----------|
| **REAL** | a live OpenAI-compatible endpoint is reachable (Ollama / vLLM / llama.cpp on this laptop, or a DGX you point `DGX_BASE_URL` at) | genuine on-device inference + real GPU telemetry |
| **SIM** | nothing reachable | a faithful **DGX Spark simulator** (GB10, 128 GB): mock `nvidia-smi`, real model registry, token-by-token streaming at plausible tok/s |

> **Either way, cloud cost is $0.** SIM mode means you can learn every DGX concept
> on a plane with no GPU — and the **exact real commands are always printed**, so
> SIM is a true dry-run of the hardware. The MODE pill (top-right) shows which.

---

## Quick start

```bash
# 1) install deps into the repo's uv-managed .venv (Python 3.13)
uv pip install -r week19/sovereign_dgx/requirements.txt

# 2) (optional) go REAL — any OpenAI-compatible local server works
curl -fsSL https://ollama.com/install.sh | sh
ollama run qwen3.6:35b-a3b-q8_0       # DGX Spark / Ubuntu GPU
# ollama run gemma4:12b               # Mac / smaller boxes

# 3) the interactive web app  (recommended — click through all 10 chapters)
.venv/bin/python week19/sovereign_dgx/tutorial_server.py
# → open http://127.0.0.1:8092

# 4) or run any demo straight from the terminal
.venv/bin/python week19/sovereign_dgx/demos/step03_ollama_on_dgx.py
```

Point at a real DGX, or force a mode:

```bash
export DGX_BASE_URL=http://my-dgx-spark.local:11434/v1   # a real DGX endpoint
export DGX_MODEL=qwen3.6:35b-a3b-q8_0                    # pin a model
export DGX_MODE=sim     # force the simulator   |   DGX_MODE=real to require live
```

---

## Layout

```
week19/sovereign_dgx/
├── README.md                     → this file
├── requirements.txt              → openai + FastAPI (openai only used in REAL mode)
├── config.py                     → endpoint detection, REAL/SIM mode, DGX specs
├── dgxsim.py                     → the DGX Spark simulator (registry, GB10 telemetry)
├── dgxview.py                    → the "make sovereign inference visible" engine (real+sim)
├── tutorial_server.py            → FastAPI control plane for the web app
├── static/guide.html             → the clickable, streaming tutorial UI
└── demos/                        → one runnable demo per chapter
    ├── step01_dgx_hello.py          Ch 2 · first sovereign inference (real or sim)
    ├── step02_dgx_hardware.py       Ch 3 · DGX Spark vs Station; what fits in 128 GB
    ├── step03_ollama_on_dgx.py      Ch 4 · Ollama — the one-command path + GPU telemetry
    ├── step04_vllm_on_dgx.py        Ch 5 · vLLM — PagedAttention throughput serving
    ├── step05_llamacpp_on_dgx.py    Ch 6 · llama.cpp — GGUF, native CUDA, full control
    ├── step06_runtime_bakeoff.py    Ch 7 · Ollama vs vLLM vs llama.cpp vs TRT-LLM
    ├── step07_model_management.py   Ch 8 · model fleet + NVFP4 quantization math
    ├── step08_multi_spark.py        Ch 9 · link two Sparks → 256 GB + tensor parallel
    └── step09_sovereignty_audit.py  Ch 10 · air-gap / sovereignty audit (CI-style)
```

---

## The 10 chapters → what you'll watch happen

| Ch | Demo | The one thing to notice |
|----|------|--------------------------|
| 1  | *(concept)* | Sovereign = own hardware + data + weights + software; a DGX makes it desk-side. |
| 2  | `step01` | A drop-in OpenAI call — only `base_url` changed — and **nothing left the box**. |
| 3  | `step02` | **Memory bandwidth**, not TOPS, sets tok/s; a 35B MoE beats a 32B dense. |
| 4  | `step03` | One install → an OpenAI API on `:11434`; watch the **GB10 fill** as weights load. |
| 5  | `step04` | Same API, **PagedAttention** → aggregate throughput scales with concurrency. |
| 6  | `step05` | **GGUF + Q4_K_M** = one binary, any weight; the engine Ollama wraps, raw. |
| 7  | `step06` | Four runtimes, **one OpenAI API** — your app code never changes. |
| 8  | `step07` | **NVFP4** turns a 165 GB BF16 70B into ~45 GB — frontier models go on-desk. |
| 9  | `step08` | Two Sparks over **200GbE** → 256 GB + **tensor parallel** for a 235B model. |
| 10 | `step09` | A live **sovereignty audit** you can run in CI — it even flags a `*-cloud` model. |

---

## Where this sits in Week 19

```
App 1 (THIS) sovereign_dgx ── run + serve + manage models on a DGX
        │
        ├─ App 2  dgx_finetune ........ adapt a model to YOUR domain (LoRA/QLoRA, NeMo)
        ├─ App 3  dgx_observability ... trace agents with Phoenix + NeMo Agent Toolkit
        └─ App 4  self_evolving_agent_v2  the Week 18 agent, now driven by a DGX model
```

Week 18 took the agent stack off the cloud onto edge devices. Week 19 scales that
up to a **DGX**: bigger models, real serving runtimes, a managed model fleet,
multi-node scale-out, and an auditable air-gap — then builds fine-tuning,
observability, and a self-evolving agent on top, all sovereign.
