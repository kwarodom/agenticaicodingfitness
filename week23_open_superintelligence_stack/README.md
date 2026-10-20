# ▣ Week 23 — The Open Superintelligence Stack (NVIDIA) for long-running, self-evolving, sovereign agents

<img src="media/hero-banner.jpg" alt="The Open Superintelligence Stack — Agent = Model + Harness" width="100%">

Twelve **interactive, explainable web apps** (Week 19 style: dual-mode SIM/REAL, animated
inline-SVG **software-architecture diagrams**, runnable demos, one chapter at a time) that teach
NVIDIA's 2026 **"Open Stack for Self-Improving Agents"** — and then combine all of it into one
**capstone** that runs a real building.

> **The through-line of the whole week:  Agent = Model + Harness.**
> The *model* reasons; the *harness* (context, orchestration, tools, memory, security/governance,
> observability, self-improvement) makes it an agent. Week 23 walks the NVIDIA stack that supplies
> both — open **Nemotron** models + the **NeMo / NIM / Dynamo / AI-Q / OpenShell / NemoClaw** harness —
> for agents that run for a long time, improve themselves, and stay sovereign.

> **Dual-mode, always $0.** Every app auto-detects a live endpoint (Ollama / vLLM / NIM / a DGX you
> point `DGX_CONN` at) and runs **REAL**, else a faithful **SIM** so every concept is learnable with
> no GPU. Real commands are always shown; cloud cost $0.

---

## How to use this folder — the learning path

**The folders are numbered `01…12` in the order you should learn them.** Each is a standalone
interactive app (its own `config.py`, engine, `tutorial_server.py`, `static/guide.html`, `demos/`) —
launch it, read the concept, view the demo source, and click **Run**. Budget ~20–40 min each.
Cross-references inside the apps ("… App 7") use these same numbers.

**New — the hands-on track.** Every app `01…12` now pairs the interactive explainer
(`tutorial_server.py`) with a DIY lab: a `TUTORIAL.md` you follow step-by-step plus 3 runnable
scripts in `labs/`. Recommended rhythm: **read-then-do** — click through the app's chapters first,
then work the TUTORIAL and run the labs; each one finishes in under a minute and degrades to a
clearly-labeled SIM when no endpoint is reachable. The TUTORIALs chain `01 → 12` via their **Next** links.
Prefer the browser? [`00_lab_runner`](00_lab_runner/) (port 8113) serves the whole track as one step-by-step course and runs the labs for you.

Work top-to-bottom, or jump by **phase**:

### Phase 0 · The hub — recommended entry point
| # | Folder | Port | What it is |
|---|--------|------|------------|
| 00 | [`00_stack_navigator`](00_stack_navigator/) | 8112 | **Stack Navigator** — the Week 23 hub: one animated map of the whole five-layer stack (hardware → runtime → model → harness → flywheel, 16 nodes), a 16-stop guided tour, per-layer runnable demos (REAL or SIM), and one-click launch dots for all 12 apps below. Open this first to see where everything fits. |
| 00 | [`00_lab_runner`](00_lab_runner/) | 8113 | **Lab Runner** — the hub's hands-on twin: all 12 `TUTORIAL.md` files as one step-by-step course in the browser, with checkpoints you tick off and every `labs/*.py` script runnable server-side, output streamed live (REAL or SIM). Open this to *do* the track, not just read it. |

<p>
  <a href="00_stack_navigator/"><img src="media/00-stack-navigator.jpg" alt="00 · Stack Navigator — the Week 23 hub" width="240"></a>
</p>

### Phase 1 · The Model — start here
| # | Folder | Port | What you learn |
|---|--------|------|----------------|
| 01 | [`01_nemotron_models`](01_nemotron_models/) | 8100 | The open **Nemotron 3** family (Nano 30B-A3B · Super 120B-A12B · Ultra 550B-A55B + RAG/Speech/Safety): hybrid Mamba-Transformer MoE, 1M context, reasoning (RLM) + tool-calling. The **MODEL** layer — everything else is harness around it. |

<p>
  <a href="01_nemotron_models/"><img src="media/01-nemotron-models.jpg" alt="01 · The Nemotron 3 model family" width="240"></a>
</p>

### Phase 2 · Serve it sovereignly
| # | Folder | Port | What you learn | Prereq |
|---|--------|------|----------------|--------|
| 02 | [`02_nim_microservices`](02_nim_microservices/) | 8101 | **NIM** — one signed container = model + optimized engine (TensorRT-LLM/vLLM/SGLang) + OpenAI API; NIM vs raw vLLM/Ollama; `build.nvidia.com`. | 01 |
| 03 | [`03_dynamo_serving`](03_dynamo_serving/) | 8102 | **NVIDIA Dynamo** — serve at scale: disaggregated prefill/decode, KV-cache-aware routing, SLO Planner, NIXL KV transfer. | 02 |

<p>
  <a href="02_nim_microservices/"><img src="media/02-nim-microservices.jpg" alt="02 · NIM signed containers" width="240"></a>
  <a href="03_dynamo_serving/"><img src="media/03-dynamo-serving.jpg" alt="03 · Dynamo disaggregated serving" width="240"></a>
</p>

### Phase 3 · Build real agents (the harness)
| # | Folder | Port | What you learn | Prereq |
|---|--------|------|----------------|--------|
| 04 | [`04_agent_skills`](04_agent_skills/) | 8103 | **NVIDIA Agent Skills** (`github.com/NVIDIA/skills`) — portable, framework-agnostic capabilities that connect **frontier agents** (Claude/GPT/Nemotron) to your business; Skills + MCP + A2A. | 01 |
| 05 | [`05_aiq_research_lab`](05_aiq_research_lab/) | 8104 | **AI-Q Open Agent Blueprint** — "a research lab for any domain": Intent Router (Nano) → Deep Agent (Super) → planning + researcher sub-agents; tools via the **NeMo Agent Toolkit**. | 04 |
| 06 | [`06_nemoclaw`](06_nemoclaw/) | 8105 | **NemoClaw** (`github.com/NVIDIA/NemoClaw`) — **build specialized agents**: base model + persona + skills + tools + signed policy; orchestrate a fleet. | 04, 05 |

<p>
  <a href="04_agent_skills/"><img src="media/04-agent-skills.jpg" alt="04 · Portable Agent Skills" width="240"></a>
  <a href="05_aiq_research_lab/"><img src="media/05-aiq-research-lab.jpg" alt="05 · AI-Q deep-research lab" width="240"></a>
  <a href="06_nemoclaw/"><img src="media/06-nemoclaw.jpg" alt="06 · NemoClaw agent fleet" width="240"></a>
</p>

### Phase 4 · Run it safely
| # | Folder | Port | What you learn | Prereq |
|---|--------|------|----------------|--------|
| 07 | [`07_guardrails_openshell`](07_guardrails_openshell/) | 8106 | Safe long-running autonomy — **OpenShell Secure Runtime** (policy authoring/signing, gateway, sandboxes, egress allowlist, privacy router) + **NeMo Guardrails**. | 06 |

<p>
  <a href="07_guardrails_openshell/"><img src="media/07-guardrails-openshell.jpg" alt="07 · OpenShell guardrails and sandbox" width="240"></a>
</p>

### Phase 5 · Observe, measure & optimize
| # | Folder | Port | What you learn | Prereq |
|---|--------|------|----------------|--------|
| 08 | [`08_nemo_relay`](08_nemo_relay/) | 8107 | **NeMo Relay** — observe every call, **Agent Insights via Phoenix** (trace/span tree, latency, cost), a Router/Gateway that right-sizes the model per request; OTel export to Phoenix/Datadog/LangSmith. | 05 |
| 09 | [`09_inference_economics`](09_inference_economics/) | 8108 | **AI Performance & Evaluation** — tokens as the unit of work; cost/M-token, throughput per GPU & per **Megawatt**; tokens → **goodput** (cost per successful task); LLM-judge + golden-set evaluation. | 03, 08 |

<p>
  <a href="08_nemo_relay/"><img src="media/08-nemo-relay.jpg" alt="08 · NeMo Relay observability" width="240"></a>
  <a href="09_inference_economics/"><img src="media/09-inference-economics.jpg" alt="09 · Inference economics" width="240"></a>
</p>

### Phase 6 · Make it self-improve
| # | Folder | Port | What you learn | Prereq |
|---|--------|------|----------------|--------|
| 10 | [`10_nemo_gym_rl`](10_nemo_gym_rl/) | 8109 | **NeMo Gym + NeMo RL** — verifiable-reward environments, multi-environment rollouts, GRPO post-training — how Nemotron itself was forged. | 09 |
| 11 | [`11_data_flywheel`](11_data_flywheel/) | 8110 | **NeMo Data Flywheel** — production logs → Curator → Customizer (LoRA/SFT/DPO/GRPO) → Evaluator (LLM-judge) → distill to a cheaper model → promote → repeat. | 10 |

<p>
  <a href="10_nemo_gym_rl/"><img src="media/10-nemo-gym-rl.jpg" alt="10 · NeMo Gym RL training" width="240"></a>
  <a href="11_data_flywheel/"><img src="media/11-data-flywheel.jpg" alt="11 · The data flywheel loop" width="240"></a>
</p>

### Phase 7 · Capstone — combine everything
| # | Folder | Port | What you build | Prereq |
|---|--------|------|----------------|--------|
| 12 | [`12_capstone_smart_hotel`](12_capstone_smart_hotel/) | 8111 | **Sovereign Autonomous Hotel** — AltoTech Grand Bangkok run by a self-improving agent fleet that combines **every app above** into one running system (tools mutate a real hotel state; the room-1203 alarm runs end-to-end). Continues Weeks 11/14/15. | all |

<p>
  <a href="12_capstone_smart_hotel/"><img src="media/12-capstone-smart-hotel.jpg" alt="12 · Sovereign smart hotel, Bangkok" width="240"></a>
</p>

<img src="media/capstone-cinematic.jpg" alt="The Sovereign Autonomous Hotel — AltoTech Grand Bangkok at night, run by a self-improving agent fleet" width="720">

**Fastest useful path (≈2 hrs):** 01 → 02 → 04 → 05 → 07 → 12. Add 03/08/09/10/11 for the full stack.

Each app has the same connection switch (`DGX_CONN=local|tunnel|cloud`), model picker, 🔌/🖥️ guidance,
and — new this week — an **animated architecture diagram** in its intro plus per-chapter charts,
sequence diagrams, and dataflow pipelines (all inline SVG/CSS, no external libraries — sovereign).

---

## The Open Superintelligence Stack (what the deck showed)

```
                 ┌───────────────────────────────────────────────┐
   Agent  =      │  HARNESS   (context · orchestration · tools ·  │   ← AI-Q (05), Skills (04),
   Model +       │            memory · security · observability)  │     NemoClaw (06), OpenShell (07),
   Harness       │                                                │     NeMo Relay (08)
                 ├───────────────────────────────────────────────┤
                 │  MODEL     Nemotron 3 (Nano/Super/Ultra,        │   ← 01: open, RL-post-trained,
                 │            RAG/Speech/Safety)                    │     1M ctx, reasoning + tools
                 ├───────────────────────────────────────────────┤
                 │  RUNTIME   NIM · Dynamo · TensorRT-LLM ·         │   ← 02, 03: sovereign,
                 │            OpenShell sandboxes                   │     disaggregated, policy-guarded
                 ├───────────────────────────────────────────────┤
                 │  FLYWHEEL  Curator/Customizer/Evaluator +        │   ← 10, 11: self-improving loop,
                 │            NeMo Gym/RL + NeMo Relay              │     measured by economics (09)
                 └───────────────────────────────────────────────┘
   Hardware: DGX Spark (GB10) ·· 2× Spark over QSFP 200GbE ·· Blackwell / Vera Rubin
```

**Agent Skills catalog** (`github.com/NVIDIA/skills`, taught in **04**) reused across the apps:
AI-Q (deep research) · NeMo Retriever (doc intelligence) · NeMo RL & Gym · NeMo Evaluator ·
NeMo Curator · NeMo Anonymizer · NeMo Data Designer · cuOpt · cuDF · VSS (video) · Voice Chat · TensorRT-LLM.

---

## Quick start (any app)

```bash
uv pip install -r week23/01_nemotron_models/requirements.txt
.venv/bin/python week23/00_stack_navigator/tutorial_server.py     # → http://127.0.0.1:8112 — the hub; start here
.venv/bin/python week23/00_lab_runner/tutorial_server.py          # → http://127.0.0.1:8113 — drive the hands-on track from the browser
.venv/bin/python week23/01_nemotron_models/tutorial_server.py     # → http://127.0.0.1:8100
.venv/bin/python week23/01_nemotron_models/labs/lab01_reasoning_tax.py   # hands-on lab — self-terminates, SIM-safe
```
Open the URL, use the **🔌 Connection** panel (local / tunnel / cloud), pick a model, and run the
chapters. With nothing reachable it runs in **SIM** — every concept still works, $0. Every app is
launched the same way; just swap the folder name and use its port from the table.

**Cloud on-ramp (try before you buy a DGX).** The **☁️ Cloud** connection has one-click presets for
NVIDIA `build.nvidia.com` / NIM cloud (`https://integrate.api.nvidia.com/v1`, key `nvapi-…`),
Hugging Face (`https://router.huggingface.co/v1`, key `hf_…`), Ollama Cloud, OpenRouter, and Claude —
any OpenAI-compatible endpoint. Run the exact same chapters against a hosted Nemotron NIM, then move
to your own DGX with no code change. In cloud mode the apps show **`cloud usage billed`** — convenient,
but off-box, i.e. *not* the sovereign path the workshop teaches.

**2-Spark chapters** (Nemotron Ultra, Dynamo multi-node): link the two Sparks over the **QSFP 200GbE**
port, verify with NCCL, then serve/route across both — the apps print the exact `mpirun` / Dynamo
commands and simulate the scaling if you have one Spark.

---

## Where this sits in the course

```
W18 sovereign edge · W19 sovereign DGX (serve/tune/observe/gateway/self-evolving)
                                        │
   WEEK 23: the OPEN SUPERINTELLIGENCE STACK — the NVIDIA production layer above.
   Learn it in order 01→12: the open Nemotron MODEL (01), serve it with NIM + Dynamo
   (02–03), build agents with Skills + AI-Q + NemoClaw (04–06), run them safely under
   OpenShell (07), observe/measure with NeMo Relay + economics (08–09), make them
   self-improve with NeMo Gym RL + the Data Flywheel (10–11), then COMBINE all of it
   in the sovereign autonomous-hotel capstone (12). Agent = Model + Harness, end to end.
```

---

## Sources

NVIDIA deck (photographed, in `week23/reference/` — reference only, not distributed): Open Superintelligence Stack
(Prime Intellect × NVIDIA); NVIDIA Agent Toolkit; AI-Q Open Agent Blueprint;
OpenShell Secure Runtime; NemoClaw; NeMo Relay; inference-economics session.

Web references:
- [Nemotron (foundation models)](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/) · [Nemotron 3 debut](https://nvidianews.nvidia.com/news/nvidia-debuts-nemotron-3-family-of-open-models) · [Nemotron 3 Nano tech blog](https://developer.nvidia.com/blog/nvidia-nemotron-3-nano-omni-powers-multimodal-agent-reasoning-in-a-single-efficient-open-model/)
- [NIM microservices](https://www.nvidia.com/en-us/ai-data-science/products/nim-microservices/) · [Sovereign AI factories](https://blogs.nvidia.com/blog/sovereign-ai-agents-factories/)
- [Data Flywheel Blueprint (GitHub)](https://github.com/NVIDIA-AI-Blueprints/data-flywheel) · [Data flywheel tech blog](https://developer.nvidia.com/blog/maximize-ai-agent-performance-with-data-flywheels-using-nvidia-nemo-microservices/) · [NeMo microservices](https://www.nvidia.com/en-us/ai-data-science/products/nemo/)
- [NVIDIA Dynamo](https://developer.nvidia.com/dynamo) · [Dynamo disaggregated serving docs](https://docs.dynamo.nvidia.com/dynamo/design-docs/disaggregated-serving)
- [NeMo Gym / RL (agent RL blog)](https://developer.nvidia.com/blog/mastering-agentic-techniques-ai-agent-reinforcement-learning/)
- [NeMo Guardrails](https://developer.nvidia.com/nemo-guardrails) · [OpenShell (safe self-evolving agents)](https://developer.nvidia.com/blog/run-autonomous-self-evolving-agents-more-safely-with-nvidia-openshell/) · [NemoClaw](https://www.nvidia.com/en-us/ai/nemoclaw/)

**Media:** course images generated with Higgsfield (nano banana pro / kling 3.0 turbo), art-directed for
the course (see [`media/manifest.md`](media/manifest.md)); an 8-second cinematic intro —
"Agent = Model + Harness" assembling itself — is at [`media/intro-video.mp4`](media/intro-video.mp4);
full-resolution masters live in `media/originals/` (gitignored, local only).
