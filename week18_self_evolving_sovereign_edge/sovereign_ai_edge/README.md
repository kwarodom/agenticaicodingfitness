# ▣ Week 18 — Sovereign AI at the Edge

An **interactive, explainable web app + runnable demos** for the
*Sovereign AI at the Edge* tutorial
(`week18/sovereign_ai_edge_tutorial.pdf`).

Where the PDF *explains* sovereign edge AI — owning your hardware, models, and
compute with **zero cloud dependency** — this folder lets you **watch it run** on
your own machine. Every demo makes a **real** call to a model running locally and
narrates it so the "magic" of on-device AI is never a black box:

```
▣ LOCAL ✓  →  » PROMPT  →  ~ REASON (on-device)  →  · ANSWER  →  ◆ tok/s · $0
                                   ⛔ never leaves the machine
```

> **These are real local inferences, not mocks.** By default the demos call
> **Ollama's OpenAI-compatible endpoint** at `http://localhost:11434/v1` — the
> exact same API as the cloud, served from your hardware. No `ANTHROPIC_API_KEY`,
> no `OPENAI_API_KEY`, **no cloud cost** — tokens are free because it's your GPU.

---

## Quick start

```bash
# 0) a local model server must be running (any OpenAI-compatible one works)
curl -fsSL https://ollama.com/install.sh | sh
ollama run gemma4:12b        # Mac / DGX Spark / Ubuntu GPU
# ollama run gemma4:2b       # Raspberry Pi / low-RAM devices

# 1) install deps into the repo's uv-managed .venv (Python 3.13)
uv pip install -r week18/sovereign_ai_edge/requirements.txt

# 2) the interactive web app  (recommended — click through all 10 chapters)
.venv/bin/python week18/sovereign_ai_edge/tutorial_server.py
# → open http://127.0.0.1:8091

# 3) or run any demo straight from the terminal
.venv/bin/python week18/sovereign_ai_edge/demos/step01_local_inference.py
```

If no local endpoint is running, the web app still loads and the demos print
**how to start one** instead of pretending to call a model.

Point at a different endpoint/model with env vars:

```bash
export EDGE_BASE_URL=http://localhost:11434/v1   # any OpenAI-compatible server
export EDGE_MODEL=gemma4:12b                      # pin a specific model
```

---

## What the web app gives you

For each of the 10 tutorial chapters the guide shows:

1. **The concept** — the *why*, distilled from the PDF.
2. **The exact source** — a "View demo source" toggle (no hidden magic).
3. **A live, on-device run** — click *Run this on-device* and watch the local
   model stream into the browser, colour-coded so you can read it at a glance:
   - <kbd>local ✓</kbd> · <kbd>prompt</kbd> · <kbd>reasoning</kbd> ·
     <kbd>answer</kbd> · <kbd>ACT</kbd> (a tool call) · <kbd>OBSERVE</kbd> ·
     <kbd>tok/s</kbd>.

The shared engine is [`edgeview.py`](edgeview.py) — a thin wrapper over the
OpenAI SDK pointed at your **local** endpoint. It separates the model's private
**reasoning** channel from its **answer**, measures real tokens/sec, runs the
local **function-calling** loop, and prints `cloud cost: $0.0000` on every call.

---

## Layout

```
week18/sovereign_ai_edge/
├── README.md                     → this file
├── requirements.txt              → openai + FastAPI (local-only)
├── config.py                     → local endpoint, model auto-detect, paths
├── edgeview.py                   → the "make sovereign inference visible" engine
├── tutorial_server.py            → FastAPI control plane for the web app
├── static/guide.html             → the clickable, streaming tutorial UI
└── demos/                        → one runnable demo per tutorial chapter
    ├── step01_local_inference.py    Ch 2 · first sovereign inference (proof it's local)
    ├── step02_hardware_advisor.py   Ch 3 · pick platform/model/quant from your RAM
    ├── step03_model_explorer.py     Ch 4 · Gemma 4 family + real local bake-off
    ├── step04_quant_calculator.py   Ch 5 · quant formats, VRAM math, inspect real model
    ├── step05_finetune_recipe.py    Ch 6 · NeMo LoRA recipe + domain-adaptation effect
    ├── step06_serving_runtimes.py   Ch 7 · OpenAI-compatible serving + latency/throughput
    ├── step07_tool_calling_agent.py Ch 8 · sovereign agent — local function calling
    ├── step08_smart_hotel_mas.py    Ch 9 · Smart-Hotel HVAC agent clears an alarm queue
    └── step09_airgap_audit.py       Ch 10 · live security / air-gap sovereignty audit
```

---

## The 10 chapters → what you'll watch happen

| Chapter | Demo | The one thing to notice |
|---------|------|--------------------------|
| 1  | *(concept)* | Sovereign = own the hardware, data, models & compute; edge = compute at the data. |
| 2  | `step01` | A drop-in OpenAI call — only the `base_url` changed — and **nothing left the machine**. |
| 3  | `step02` | RAM, not TOPS, picks your platform: decode is **memory-bandwidth bound**. |
| 4  | `step03` | Same prompt, every local Gemma you have — a **real bake-off** of quality vs tok/s. |
| 5  | `step04` | 31B is 75 GB in BF16 but **~14 GB at Q4_K_M** — quantization makes the edge possible. |
| 6  | `step05` | A system prompt steers per-call; **LoRA bakes the domain into the weights** for good. |
| 7  | `step06` | The sovereign API is the **same** API — measured TTFT + throughput, served from localhost. |
| 8  | `step07` | The local model **READS a sensor and DISPATCHES** a work order — it acts, on-device. |
| 9  | `step08` | One agent **clears a hotel alarm queue**, CRITICAL for the guest room — zero cloud. |
| 10 | `step09` | A live **sovereignty audit**: endpoint local? no cloud creds? it even flags a `*-cloud` model. |

---

## Cost & safety notes

- **Cloud cost is always $0** — every token is generated on your hardware.
- Gemma 4 in Ollama is a *thinking* model: demos surface its **reasoning channel**
  as the `~ REASON` step, which is why a full click-through is genuinely
  explainable (you see the model think, then answer).
- `step05` writes a NeMo LoRA recipe + SFT dataset to a throwaway `.sandbox/`;
  `🧹 Clean sandbox` in the web app (or `POST /api/cleanup`) removes it.
- `step09` is a real audit — on this repo it correctly flags any `*-cloud`
  passthrough model as **not sovereign**.

---

## Where this sits in the course

```
tool-use (W3) → agent loops (W4–5) → MAS (W9/W15) → long-running/distributed (W17)
                                                  │
   THIS WEEK (W18): take the whole agent stack OFF the cloud —
   sovereign hardware, open-weight models, local serving, on-device agents,
   and an auditable air-gap — demonstrated on real inferences you can watch run.
```

Earlier weeks built agents that called frontier models in the cloud. Week 18
shows the same capabilities running **entirely on your own hardware**: private by
physical design, offline-capable, compliant by physics, and $0 per token.
