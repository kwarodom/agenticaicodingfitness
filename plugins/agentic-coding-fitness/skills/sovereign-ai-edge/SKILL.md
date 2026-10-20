---
name: sovereign-ai-edge
description: "Teach Sovereign AI at the Edge (Week 18) — running the WHOLE agent stack on hardware you own, with zero cloud dependency and $0-per-token cost. Covers what 'sovereign' + 'edge' mean (own the hardware, data, models & compute, run it where the data is), drop-in local inference via an OpenAI-compatible endpoint (Ollama at http://localhost:11434/v1 — only base_url changes), hardware sizing (decode is memory-bandwidth bound, so RAM not TOPS picks your platform), the Gemma open-weight model family, quantization math (a 31B model is ~75 GB in BF16 but ~14 GB at Q4_K_M), LoRA/NeMo fine-tuning to bake a domain into the weights, local serving runtimes (TTFT + throughput), sovereign tool-calling/function-calling agents that act on-device, a Smart-Hotel HVAC multi-agent demo, and an air-gap sovereignty audit. Use when someone asks 'how do I run agents locally / offline / on-device / on my own GPU?', mentions sovereign AI, edge AI, Ollama, local LLM, open-weight models, Gemma, quantization (Q4_K_M/GGUF/VRAM), air-gap, data residency/compliance, $0 inference, or is reviewing Week 18."
when_to_use: "Learner wants to run the agent stack on their OWN hardware — local/offline/on-device/air-gapped, $0 per token, for privacy/compliance/data-residency — or asks about sovereign AI, edge AI, Ollama or another OpenAI-compatible local server, open-weight models (Gemma), quantization & VRAM math, LoRA/NeMo fine-tuning, local serving throughput, on-device tool-calling agents, or is catching up on Week 18."
---

# Sovereign AI at the Edge — The Whole Stack, Off the Cloud (Week 18)

> **The one idea:** Every earlier week called a *frontier model in the cloud*. **Sovereign edge AI runs the same capabilities entirely on hardware you own** — private by physical design, offline-capable, compliant by physics, and **$0 per token** because the GPU is yours. *Sovereign* = you own the hardware, data, models & compute. *Edge* = the compute runs where the data is.

```
▣ LOCAL ✓  →  » PROMPT  →  ~ REASON (on-device)  →  · ANSWER  →  ◆ tok/s · cloud cost $0
                                   ⛔ nothing ever leaves the machine
```

The unlock is that **sovereignty is almost free to adopt**: a local server (Ollama, vLLM, llama.cpp…) exposes the *exact same* OpenAI-compatible API as the cloud. Going sovereign is often a **one-line change** — you point `base_url` at `localhost`.

---

## The drop-in move — only `base_url` changes

```python
from openai import OpenAI

# Cloud:  client = OpenAI()                         # api.openai.com, costs $, data leaves
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # sovereign

resp = client.chat.completions.create(
    model="gemma4:12b",
    messages=[{"role": "user", "content": "Summarise this incident report."}],
)
# nothing left the machine · no ANTHROPIC_API_KEY / OPENAI_API_KEY · cloud cost: $0.0000
```

That's the whole trick: the sovereign API **is** the same API. Everything you learned about tool-calling, agent loops, and MAS transfers unchanged — it just runs on your box.

---

## The five things that make it real (not just "a local model")

| Concern | The sovereign reality | The counter-intuitive fact |
|---|---|---|
| **Hardware sizing** | pick platform by **RAM/VRAM**, not headline TOPS | decode is **memory-bandwidth bound** — bandwidth, not raw compute, sets tok/s |
| **Models** | **open-weight** families (Gemma 4, etc.) you can download & own | a "thinking" model exposes a private **reasoning** channel separate from its answer |
| **Quantization** | shrink weights to fit consumer hardware | a 31B model is **~75 GB in BF16** but **~14 GB at Q4_K_M** — quantization is what makes the edge possible |
| **Fine-tuning** | **LoRA / NeMo** bakes a domain into the weights | a system prompt steers *per call*; LoRA changes the weights *for good* |
| **Serving** | OpenAI-compatible runtimes; measure **TTFT + throughput** | same API, but you now own the latency/throughput tradeoff |

---

## Sovereign agents — the loop, on-device

Sovereignty isn't just chat: a local model can run the full **tool-calling loop** (`tool-use`, `agent-loops`) without any network. The model **reads a sensor and dispatches a work order** — it *acts*, on your hardware:

```python
# A sovereign function-calling agent: same tool_use loop as Week 3/5, zero cloud.
tools = [{"type": "function", "function": {
    "name": "read_room_temp",
    "description": "Read current temperature for a hotel room sensor.",
    "parameters": {"type": "object", "properties": {"room": {"type": "string"}}},
}}]
resp = client.chat.completions.create(model="gemma4:12b", tools=tools,
    messages=[{"role": "user", "content": "Room 412 alarm — diagnose and act."}])
# → the LOCAL model calls read_room_temp(412), OBSERVEs, then dispatches an HVAC work order
```

The Week 18 capstone runs a **Smart-Hotel HVAC agent that clears an alarm queue** — resolving routine alarms and escalating the critical guest-room incident — entirely on-device, $0.

---

## Prove it's sovereign — the air-gap audit

"Local" is a claim; **audit it.** Week 18 ships a real sovereignty audit that checks: is the endpoint actually `localhost`? Are there *no* cloud credentials in the environment? It even flags a `*-cloud` passthrough model as **not sovereign** — catching the case where a "local" config is secretly proxying to a cloud API.

> Why this matters: sovereignty's payoffs — privacy, offline operation, and **compliance by physics** (data that never leaves the building can't leak) — only hold if the air-gap is *real*. The audit turns "we think it's local" into "we verified it's local."

---

## When sovereign edge wins (and when it doesn't)

| Reach for sovereign edge when… | Stay on the cloud when… |
|---|---|
| data **cannot** leave (PII, PHI, defense, on-prem industrial) | you need the absolute frontier model and data is non-sensitive |
| you need **offline / air-gapped** operation | traffic is spiky and you don't want to own idle GPUs |
| **per-token cost** at scale dominates (every token is free on your GPU) | you have no local hardware budget / ops capacity |
| **latency** to a far cloud region hurts (compute at the data) | model size needed simply won't fit your hardware even quantized |

> 📁 Class repo: `week18_self_evolving_sovereign_edge/sovereign_ai_edge/` — a clickable, streaming web app (`tutorial_server.py` → `http://127.0.0.1:8091`) plus 9 runnable on-device demos, one per chapter: `step01_local_inference` (proof it's local) → `step02_hardware_advisor` (RAM-based sizing) → `step03_model_explorer` (real Gemma bake-off) → `step04_quant_calculator` (VRAM math) → `step05_finetune_recipe` (NeMo LoRA) → `step06_serving_runtimes` (TTFT/throughput) → `step07_tool_calling_agent` (on-device function calling) → `step08_smart_hotel_mas` (clear an HVAC alarm queue) → `step09_airgap_audit` (live sovereignty audit). The shared engine `edgeview.py` separates the model's reasoning channel from its answer, measures real tok/s, and prints `cloud cost: $0.0000` every call. Needs a local OpenAI-compatible server (`ollama run gemma4:12b`, or `gemma4:2b` on a Pi); point elsewhere with `EDGE_BASE_URL` / `EDGE_MODEL`. If nothing's running, the app loads and prints *how to start one* instead of faking a call. Full write-up: `week18_self_evolving_sovereign_edge/sovereign_ai_edge/README.md` + `week18_self_evolving_sovereign_edge/sovereign_ai_edge_tutorial.pdf`.

---

## 🧪 Guided lab (offer this)

### Warm-up (5 min, pass/fail)

Answer out loud:
1. Define **sovereign** and **edge** in one sentence each.
2. Picking hardware: why does **RAM/bandwidth** matter more than headline TOPS for decode?
3. A 31B model is ~75 GB in BF16 — roughly what does it become at **Q4_K_M**, and why does that matter? (~14 GB; it's what lets it fit consumer hardware)

**Pass/fail:** all three correct without peeking.

### Skill Drill A — First sovereign inference (10 min, $0)

Start a local server and run the proof-it's-local demo. Confirm the output reports `cloud cost: $0.0000`.

```bash
ollama run gemma4:12b           # or gemma4:2b on low-RAM hardware
.venv/bin/python week18_self_evolving_sovereign_edge/sovereign_ai_edge/demos/step01_local_inference.py
```

**Done =** a real local completion **and** evidence nothing left the machine (cost $0, localhost endpoint).

### Skill Drill B — Sovereign tool-calling agent (20 min, $0)

Run `step07_tool_calling_agent.py` (and/or the `step08` Smart-Hotel queue). Watch the **local** model call a tool, OBSERVE, and act.

**Done =** you can point to the on-device `tool_use` → `tool_result` round-trip and confirm zero network calls.

### Skill Drill C — Audit the air-gap (10 min, $0)

Run `step09_airgap_audit.py`. Then deliberately point `EDGE_MODEL` at a `*-cloud` model and re-run.

**Done =** the audit passes for the real local model and **flags the `*-cloud` model as not sovereign** — you can explain each check (localhost? no cloud creds? no passthrough model?).

### Weighted evaluation criteria

| # | Criterion | Weight |
|---|---|---|
| 1 | Defines sovereign + edge; states the `base_url`-only drop-in move | 25% |
| 2 | Drill A: a real local inference with `cloud cost: $0` | 25% |
| 3 | Drill B: on-device tool-calling round-trip identified | 20% |
| 4 | Drill C: audit flags a non-sovereign config; learner names the checks | 20% |
| 5 | Can state one quantization/sizing fact (Q4_K_M ~14 GB **or** bandwidth-bound decode) | 10% |

**Pass = 4 of 5** (criteria 1 and 2 mandatory).

### Stretch

- Run the **bake-off** (`step03`): same prompt across every local Gemma you have — compare quality vs tok/s.
- Read the **LoRA/NeMo** recipe (`step05`) and articulate when baking a domain into the weights beats a system prompt.
- Take a cloud agent you built earlier in the course and make it sovereign by changing only `base_url` — then audit it.

End by framing the throughline: "the loop (`agent-loops`), memory (`self-evolving-agents`), and multi-agent patterns (`multi-agent-systems`) you learned all run unchanged here — Week 18 just proves the whole stack can live on hardware you own."
