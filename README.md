# 🏋️‍♂️ Agentic Coding Fitness @ Rust Tech Bar

Welcome to the repository for the **Agentic Coding Fitness** event series, hosted weekly at Rust Bar, Ban Tad Thong! 

This repository contains all the code, tools, and examples built during our hands-on "Vibe Coding" sessions. It serves as a living codebase demonstrating how to transition from basic AI API calls to building sophisticated, multi-agent systems and real-world IoT integrations.

**Event Details**: [Luma Event Page](https://lu.ma/jy6d10xq)
- **When**: Every Tuesday, 18:00 – 20:00
- **Where**: Rust Bar, Ban Tad Thong (Bangkok)

## 🤖 What is Agentic Coding Fitness?
Think of this as a "fitness center" for your coding brain—but instead of lifting weights, we are building AI muscle muscle memory. We focus on **Agentic AI**: moving beyond simple prompt-and-response mechanisms to build AI that can think, plan, decide, and collaborate using multi-agent systems.

We emphasize a **practice-first** approach (Vibe Coding). No long lectures, just shipping workable solutions that interact with the real world!

## 📚 Catching up? Install the Bootcamp Plugin

Missed a session or want to review at your own pace? We packaged the **entire course (weeks 2–18 and 27)** into a shareable **Claude Code plugin** — **21 bite-sized skills** (one per concept) that teach the idea, show runnable code (pointing at the real `weekN/` files here), and walk you through a hands-on **$0 lab** (a tiny `MockLLM`, or fully offline checkpoints, so you need no API key to start). Just ask Claude in plain English and the right skill loads automatically.

**Install** (run these in any Claude Code session):

```
/plugin marketplace add kwarodom/agenticaicodingfitness
/plugin install agentic-coding-fitness@agentic-coding-fitness
```

Then try: *"Recap the whole course and tell me which skill to start with."*

### 🔄 Already installed? Pull the latest version mid-session

We ship new skills as the course grows (we're on **v2.3.0 — 21 skills**). To grab the newest version **without restarting**, run these three in your current session:

```
/plugin marketplace update agentic-coding-fitness                 # 1. refresh the catalog from GitHub
/plugin install agentic-coding-fitness@agentic-coding-fitness     # 2. fetch the latest version
/reload-plugins                                                   # 3. activate the new skills now
```

> There's no separate `/plugin update` command — **reinstalling** pulls the latest version from the refreshed marketplace. Step 1's argument is the *marketplace name* (`agentic-coding-fitness`), not the GitHub repo.

**Prefer clicking?** Run `/plugin` for the interactive manager: **Marketplaces** tab → select *agentic-coding-fitness* → **Update**, then **Installed** tab → select the plugin → **Reinstall**, then `/reload-plugins`. (You can also toggle **Enable auto-update** on the marketplace so new versions are fetched at startup.)

Covers: LLM basics · tool use · agent loops (now incl. the Week 18 Claude Agent SDK production loop) · MCP & skills · RAG · multi-agent systems · production & observability · agent evaluation/CI · knowledge-graph memory · production GraphRAG · choosing models & patterns · the NVIDIA NeMo Agent Toolkit · long-running & distributed agents (Google ADK durable sessions, pause/resume, auth.md, A2A fleets) · self-evolving agents (tripartite memory + consolidation) · sovereign AI at the edge (local/$0 inference) · sovereign & self-evolving AI on an NVIDIA DGX (serve/fine-tune/observe/gateway) · the Week 27 software factory (unattended Build/QA/Review lanes, merge gates, guard hooks, plugin evals, telemetry → tickets) · vibe-coding & security · the A2A protocol · skill-authoring. See [`plugins/agentic-coding-fitness/`](plugins/agentic-coding-fitness/) for details.

## 📂 Repository Contents 

The project is structured week-by-week as our complexity scales up — from a single API call to long-running distributed fleets, self-evolving memory, and sovereign agents running entirely on hardware you own. Each week maps to a plugin skill (above) that recaps it with a $0 lab.

### Phase ① Foundation — talk → tools → agents

#### 🔹 Week 2: Claude API Foundations
Talking to modern LLMs programmatically. → skill `llm-fundamentals`
- `week2/claudeapicall.py`: basic single-turn API requests · `week2/claudestreamingapi.py`: streaming tokens · `week2/claudemulti_turn.py`: conversational state & history · `week2/lab/`: $0 practice drills.

#### 🔹 Week 3: Tool Use & Smart Assistants
Teaching agents to call external services (function calling). → skill `tool-use`
- `week3/toolsuse.py`: function calling (weather, calculator, web search) · `week3/buildsmartassistant3tools.py`: a full assistant · **Tapo Smart Plug Integration** (`check_tapo.py`, `scan.py`, `tapo_config.json`): a local HTTP wrapper so Claude controls TP-Link Tapo L530 lights.

#### 🔹 Week 4: Autonomous Pipelines & Hardware
Chaining actions and reaching into physical IoT. → skill `agent-loops`
- `week4/pipeline.py`: an autonomous research pipeline (web search → multi-agent synthesis → self-scoring → Markdown reports, with NotebookLM export) · `week4/dronecontrol.py`: flight patterns on a DJI Tello drone (`djitellopy`) · `week4/openrouterfreemodel.py`: a free-model gateway.

#### 🔹 Week 5: The Agent Loop
The reusable REASON → ACT → OBSERVE loop that turns a tool-user into an agent. → skill `agent-loops`
- `week5/autoagent.py`: the reusable bounded `Agent` class (ReAct + stop conditions).

### Phase ② Strength — single-agent mastery, reusable tools & knowledge

#### 🔹 Week 6: Full-Stack Agent App (deload / integration)
An agent put behind a real API. → skill `vibe-coding-and-security`
- `week6/src/` (Express/TypeScript/Postgres) · `week6/CLAUDE.md` + `AGENTS.md`: context engineering in practice.

#### 🔹 Week 7: MCP & Skills
Reusable tools (MCP) and reusable know-how (Skills). → skill `mcp-and-skills`
- `week7/mcpserver.py`, `week7/mcpfilesystem.py`: MCP servers · `week7/agent.py`, `week7/agenttooldt.py`: an MCP client agent · `week7/skill.md`: a worked Skill.

#### 🔹 Week 8: RAG — Knowledge Agents
Ground answers in your own documents (and prove it with RAGAS). → skill `rag-knowledge-agents`
- `week8/Week8_RAG_Knowledge_Agents_Lab.pdf`: the RAG lab.

### Phase ③ Endurance — systems that run reliably

#### 🔹 Week 9: Multi-Agent Systems
Sequential / router / parallel-swarm orchestration across frameworks. → skill `multi-agent-systems`
- `week9/ex1_crewai_sequential.py` (CrewAI) · `week9/ex2_LangGraphSupportGraph.py` (LangGraph router) · `week9/ex3_ParallelSwarm.py` (asyncio swarm) · plus AG2/Anthropic comparisons and 3 workshop PDFs.

#### 🔹 Week 10: Production & Observability
Make a prototype something you can *see, stop, and afford*. → skills `production-and-observability`, `agent-evaluation`
- `week10/notebooks/01_hello_graph.py` → `05_hybrid_sdk.py`: a support-routing system gaining a supervisor, `SqliteSaver` checkpointing + HITL `interrupt()`, LangSmith tracing, then a Claude Agent SDK hybrid · `week10/GUIDE.md`, `solutions/`.

#### 🔹 Week 11: Mastery — Models & Patterns
Pick the right model, framework, and pattern; the 12-pattern taxonomy. → skills `models-and-patterns`, `agent-drills`
- `week11/index.html`: model wizard + pattern playground + quiz · `week11/exercises/`: **14 graded MAS drills** (ex01–ex14, Beginner → Expert).

### Phase ④ Performance — memory, GraphRAG, production frameworks, fleets

#### 🔹 Week 14: Agent Memory with Knowledge Graphs
Durable memory agents remember across runs (Neo4j + GraphRAG). → skill `agent-memory-graphs`
- `week14/agent_memory.py`, `week14/hotel_kg_builder.py`, `week14/lab1_hotel_mas.py` · `week14/NEO4J_TUTORIAL.md` · `week14/pi-structured-extraction/`: a structured-extraction sub-project.

#### 🔹 Week 15: Production GraphRAG
Cypher + GDS, ingestion, GraphRAG across 7 frameworks, and **evaluating** it. → skills `knowledge-graph-mastery`, `agent-evaluation`
- `week15/kg_mastery/`: the 6-part code companion (fundamentals → building → GraphRAG → evaluation/RAGAS+CI → use cases → reference) · `week15/smart_hotel_mas/`: a 5-agent CrewAI system over a 4-layer memory stack.

#### 🔹 Week 16: Production Frameworks — NVIDIA NeMo Agent Toolkit
Config-driven multi-agent: register tools, compose YAML workflows, observe. → skill `nemo-agent-toolkit`
- `week16/adding_tools_to_agents.ipynb`: tool registration + LlamaIndex RAG tool · `week16/multi_agent_orchestration.ipynb`: supervisor → specialists with HITL.

#### 🔹 Week 17: Long-Running & Distributed Agents (Google ADK + A2A)
Agents that **pause for days and resume without losing context**, and delegate across services. → skills `long-running-and-distributed-agents`, `a2a-protocol`
- `week17/checkpoints/checkpoint1_state_machine.py` → `checkpoint6_fleet.py`: 6 offline steps (durable state → restart-survival → webhook resume → sub-agents → A2A cards → fleet capstone) · `week17/hr_onboarding/`: a live ADK onboarding agent · `week17/authmd_adk/`: **auth.md** × ADK — store the durable grant, re-mint a scoped token at every wake.

### Phase ⑤ Sovereignty & Self-Improvement — the stack you own, that gets better

#### 🔹 Week 18: Production Loops, Self-Evolving Memory & Sovereign Edge AI
Three interactive web apps + runnable demos that take the agent stack to production, make it *learn*, and take it *off the cloud*. → skills `agent-loops` (extended), `self-evolving-agents`, `sovereign-ai-edge`
- **`week18/agent_loop/`** — the loop as a *production* discipline via the **Claude Agent SDK**: built-in & custom tools, PreToolUse/PostToolUse safety hooks, resumable sessions, subagent orchestration, and `max_turns`/`max_budget_usd` caps. A clickable streaming web app (`tutorial_server.py`, port 8090) + 9 demos (`step01_hello_agent` → `step09_production`). Uses your `claude` CLI sign-in — **no API key**.
- **`week18/self_evolving_agent/`** — turn a *stateless* agent into one that **remembers, learns, and gets cheaper** via the **Tripartite Memory Model** (episodic `SessionDB` + semantic `MEMORY.md`/`USER.md` + procedural `SKILL.md` library) and a background **consolidation** loop → compound returns (~64% fewer turns / ~66% lower cost by run 5). Live visualizer (port 8088) + step-by-step guide (port 8090); 7 checkpoints (1–6 offline, $0).
- **`week18/sovereign_ai_edge/`** — run the **whole stack on hardware you own** with zero cloud dependency and **$0 per token**: local OpenAI-compatible inference (Ollama), RAM-based hardware sizing, quantization math, LoRA/NeMo fine-tuning, on-device tool-calling agents, a Smart-Hotel HVAC demo, and a live air-gap **sovereignty audit**. Web app on port 8091 + 9 demos.
- Comprehensive write-ups per folder (`README.md`/`TUTORIAL.md`) plus tutorial PDFs: `agent_loop_comprehensive_tutorial.pdf`, `self_evolving_agent_tutorial.pdf`, `sovereign_ai_edge_tutorial.pdf`.

#### 🔹 Week 19: Sovereign & Self-Evolving AI on a DGX
Five interactive web apps that take the whole stack onto an **NVIDIA DGX** — run/serve, fine-tune, observe, self-evolve, and gateway — grounded in NVIDIA's [`dgx-spark-playbooks`](https://github.com/NVIDIA/dgx-spark-playbooks). Every app runs **REAL** (a live Ollama/vLLM/DGX endpoint) or **SIM** (a faithful simulator — no GPU needed); cloud cost always **$0**.
- 👉 **Start here: the step-by-step walkthrough → [`week19/README.md`](week19/README.md)** — walks you through all five apps in order, chapter by chapter.
- **`week19/sovereign_dgx/`** (port 8092) — run + serve + manage models on a DGX: **Ollama, vLLM, llama.cpp**, TensorRT-LLM, NVFP4 quantization, multi-Spark scale-out, air-gap audit.
- **`week19/dgx_finetune/`** (port 8093) — adapt a model to **your domain**: LoRA/QLoRA with **NeMo AutoModel** + **Unsloth**, dataset prep, training loop, eval, GGUF/NVFP4 export.
- **`week19/dgx_observability/`** (port 8094) — **see, measure, judge** a sovereign agent: OpenTelemetry tracing → **Arize Phoenix**, metrics, LLM-as-judge evals, + a **NeMo Agent Toolkit** workflow.
- **`week19/self_evolving_agent_v2/`** (port 8095) — the Week 18 self-evolving agent, made sovereign: a **switchable brain** (DGX ↔ Claude) + tripartite **memory on the DGX** that learns over time.
- **`week19/dgx_litellm/`** (port 8096) — the **serving gateway**: one OpenAI URL over all backends with **LiteLLM** — routing, fallbacks, hot-swap, virtual keys/budgets, logging → Phoenix.

#### 🔹 Week 21: Physical AI & Digital Twins for Buildings
**Digital twin = scene (OpenUSD) + state (live data) + simulation (physics) + agents.** Thirteen interactive apps (SIM/REAL) on NVIDIA Omniverse + OpenUSD and the Physical AI stack, from the three-computer model to a running building.
- 👉 **Start here → [`week21/README.md`](week21/README.md)** · apps `01_physical_ai_landscape` → `13_capstone_smart_city` on ports **8200–8212** (e.g. `.venv/bin/python week21/01_physical_ai_landscape/tutorial_server.py`).
- OpenUSD foundations · BIM → USD · scene assembly · live BACnet/Modbus → MQTT binding · EnergyPlus simulation and what-if sweeps · PhysicsNeMo surrogates · RL building controls · a self-evolving operator and a staff copilot · two capstones (a hotel twin, a sovereign smart city).

### Phase ⑥ The open stack, typed decisions & your own hardware

#### 🔹 Week 23: The Open Superintelligence Stack (NVIDIA)
**Agent = model + harness.** Twelve interactive apps that walk NVIDIA's open stack for long-running, self-evolving, sovereign agents, then combine it into a capstone that runs a building.
- 👉 **Start here → [`week23/README.md`](week23/README.md)** · hub `00_stack_navigator` (port **8112**) · hands-on **Lab Runner** `00_lab_runner` (port **8113**) · apps `01`–`12` on ports **8100–8111**.
- Nemotron models · NIM microservices · Dynamo serving · agent skills · AI-Q research lab · NemoClaw · guardrails + OpenShell · NeMo Relay · inference economics · NeMo Gym RL · the data flywheel · capstone smart hotel.

#### 🔹 Week 24: Typed AI Decisions with Jev (TypeSafe System One)
**Jev judges, your code decides.** Jev returns typed judgments (`choice`, `noul`, `score`) instead of prose, so code can route, rank and verify, with actions kept behind deterministic policy.
- 👉 **Start here:** `.venv/bin/python week24/00_jev_lab_runner/tutorial_server.py` → **http://127.0.0.1:8124** · [`week24/README.md`](week24/README.md)
- 12 modules, EN + ไทย: hello Jev · the three primitives · question design · intent routing · email triage · HR evidence · AFDD alarm triage · leads/RAG · evaluation and cost · Jev vs Laya · Jev + an LLM of your choice · a hotel copilot capstone. Runs **LIVE** with `TYPESAFE_API_KEY`, or **DRY** from recorded answers at $0.

#### 🔹 Week 25: DGX Spark — Fine-Tune, Serve & Build Sandboxed Agents
NVIDIA's official [DGX Spark playbooks](https://build.nvidia.com/spark), hands-on, on one Spark or two cabled together.
- 👉 **Start here:** `.venv/bin/python week25/00_spark_lab_runner/tutorial_server.py` → **http://127.0.0.1:8125** · [`week25/README.md`](week25/README.md)
- 21 modules, EN + ไทย: connect and budget a Spark · two Sparks over QSFP + NCCL · **serve** with Ollama, llama.cpp, vLLM, SGLang, TensorRT-LLM, NIM, NVFP4 and speculative decoding · a **LiteLLM** gateway · **fine-tune** with LLaMA Factory, Unsloth, PyTorch/NeMo and VLM/FLUX · evaluate → serve → route your fine-tune · **agents** with NeMo Agent Toolkit, OpenShell, NemoClaw, OpenClaw/Hermes and local coding agents · a capstone that puts a fine-tuned router, as a tool, behind a gateway for an agent in a sandbox · an atlas of every other playbook.
- Labs drive your Spark over SSH (**LIVE**), or run **DRY** with every output labelled RECORDED, REFERENCE or EXAMPLE. Agent and gateway labs run for real against Ollama on your laptop.

#### 🔹 Week 26: NemoClaw on DGX Spark — Claws from Beginner to Expert
Build, sandbox, trace, benchmark and harden **claws**: always-on, tool-using agents in NVIDIA **OpenShell** sandboxes, installed with **NemoClaw**, with agent logic written in **NeMo Agent Toolkit (NAT)**. The running example is **Alto Ops Claw**, a hotel chiller-plant assistant.
- 👉 **Start here:** `.venv/bin/python week26/00_reef_lab_runner/tutorial_server.py` → **http://127.0.0.1:8126** · [`week26/README.md`](week26/README.md) (one-time setup: the `week26/.venv-nat` and `week26/.venv-openshell` venvs)
- 8 modules, EN + ไทย: what a claw is · your first claw · OpenShell policy as code · NAT claws (custom tools, REST, MCP both ways, in a sandbox) · tracing · benchmarking · hardening · the Alto Ops Claw capstone, graded on evidence.
- The **🪸 Reef Lab Runner** reuses your Week 25 `SPARK_HOST`. Without a Spark, Spark steps run **DRY** with labelled output, and laptop labs (NAT on Ollama, `nat serve`, MCP, eval, the policy parser) run for real. Nothing changes your Spark unless you turn on **🔓 Allow changes**. The full React app from the spec is in [`week26/alto-reef/`](week26/alto-reef/).

### Phase ⑦ The factory — agents that ship unattended

#### 🔹 Week 27: Software Factory — an org chart made of loops
Build the thing that builds the software: linted tickets on a kanban, drained by unattended **Build → QA → Review** lanes (one Claude Code skill each, one worktree per issue, state only in GitHub), a **merge gate** that merges at the reviewed SHA unless a human-only label is present, deterministic **guard hooks**, behavioural **plugin evals**, and a **telemetry → tickets** loop that makes it self-improving. Reference: Eric Tech's open-source super-board. → skill `software-factory`
- 👉 **Start here:** `cd week27/00_alto_mini && make seed && make test && make dev` → **http://127.0.0.1:8127** · [`week27/README.md`](week27/README.md)
- 7 labs: run someone else's factory · the ticket + Builder lane · the QA lane (test-gap ledger, red-first, forensics) · Review + merge gate (review remembers, adversarial truth-check) · guards + evals · telemetry to tickets · a 48-hour capstone graded on honest metrics.
- **Alto Mini** (`week27/00_alto_mini/`) is the target app with seeded, ticketable bugs; [`week27/factory/`](week27/factory/) is the starter kit (lane skills, hooks + tests, `factory-run.sh`, `merge-gate.sh`, stub Sentry/PostHog collectors, three eval cases with an offline `gh` stub). Offline parts are tested; the live-CLI run is Lab 05's job.

### 🦾 Special track: Agentic Robotics (SO-ARM101)
From a simulated SO-ARM101 arm to a tool-using robot agent, in numbered lessons: MuJoCo manual control → record/replay → kinematic pick-and-place → active perception with the wrist camera → semantic scene → **agentic manipulation** (safe robot tools for an LLM) → a guided full pipeline. A workshop portal (`08_workshop_portal/`, **http://127.0.0.1:8000**) operates the simulation or a safety-gated LeRobot path to the physical arm.
- 👉 **Start here → [`week_agentic_robotic/README.md`](week_agentic_robotic/README.md)** (`00_getting_started/check_setup.py` first).

> Weeks 12, 13, 20 and 22 are not published in this repository.

---

## 🛠️ Getting Started

### 1. Requirements
- **Python 3.13** (the repo's `pyproject.toml` requires it) and [**uv**](https://docs.astral.sh/uv/) (recommended), or plain `venv` + `pip`.
- Optional: [**Ollama**](https://ollama.com) for free local models (weeks 18–26 use it as a stand-in when no GPU box is around). Optional: Docker, for Neo4j (weeks 14–15) and NVIDIA containers (weeks 19–26).

Clone the repository and create the shared virtual environment. The labs call `.venv/bin/python`, so keep it at the repo root:
```bash
git clone https://github.com/kwarodom/agenticaicodingfitness.git
cd agenticaicodingfitness
uv venv -p 3.13 .venv                    # or: python3.13 -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
uv pip install -r requirements.txt       # or: pip install -r requirements.txt
```
Weeks with extra dependencies (frameworks, web apps, GPU tooling) list them in their own `README.md` or `requirements.txt`. Week 25 keeps its NeMo Agent Toolkit and LiteLLM tools in their own venvs; see [`week25/README.md`](week25/README.md). Week 26 does the same for NAT 1.9 and the OpenShell CLI (Python 3.12); see [`week26/README.md`](week26/README.md).

### 3. Environment Variables
Copy the template and fill in **only the keys for the weeks you are doing**:
```bash
cp .env.example .env
```
[`.env.example`](.env.example) lists every variable the code reads, grouped by the week that needs it. Most weeks need just `ANTHROPIC_API_KEY`, and many labs run with no key at all ($0 / DRY / SIM modes, or a local model). Some weeks ship their own template as well: `week6/`, `week10/`, `week15/code/`, `week23/`, `week24/`, `week25/` and `week26/`.

> 🔐 `.env` is gitignored. Never commit a key, paste one into code or a notebook, or show one in a screenshot. If a key is ever exposed, **revoke it at the provider**. Deleting it from the code does not un-publish it.

### 4. Hardware Configuration (Optional)
- **Tapo Lights**: Edit `tapo_config.json` with your TP-Link account credentials and local IP address of your light bulb. 
- **Tello Drone**: Connect your computer directly to the Tello's Wi-Fi network before running `week4/dronecontrol.py`.

---

## 🎯 Who is this for?
- **Developers & Programmers** looking to elevate their workflow with AI.
- **Tech, Startup, and Product Innovators**.
- Anyone with basic coding knowledge ready to embrace the future of **AI-native, Agent-based development**.

## 🌟 Our Goal 
- Build **Real Stuff**
- Solve **Real Problems**
- Generate **Real Impact**

Come join us every Tuesday, stretch those brain muscles, and let's craft the future of Agentic AI together! 💪🤖
