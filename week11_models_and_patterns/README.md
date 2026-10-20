# Week 11 — Agent Forge: Mastery Lab

A self-contained web app to **revise, summarize, learn, play, and test** every concept of agents and multi-agent systems from the AltoTech Agentic Coding Fitness 10-week program.

## How to open

Just double-click `index.html` — it works in any modern browser (Chrome / Safari / Firefox / Edge). No install, no backend.

```bash
open /Users/altodev5/AgenticCoding/week11/index.html      # macOS
```

Or drag the file into your browser.

## Live API mode (real Claude / OpenAI calls)

In the **Playground** tab, choose **⚡ Live API** mode and paste your keys:

- **Anthropic key** (`sk-ant-...`) — get from https://console.anthropic.com/
- **OpenAI key** (`sk-...`) — get from https://platform.openai.com/api-keys

Keys are stored in your browser's `localStorage` only — never sent anywhere except directly to `api.anthropic.com` / `api.openai.com`.

Anthropic browser calls use the `anthropic-dangerous-direct-browser-access: true` header (this is fine for personal experimentation; for production, route through a backend).

The simulator (🧪 Sim) mode still works without keys — useful when you just want to see the topology animation.

## What's inside

| Module | What it does |
|---|---|
| **Overview** | One-glance journey map of all 10 weeks |
| **10-Week Review** | Per-week summary, key concepts, code snippet from your `weekN/` folders, CEO tip |
| **LLM Model Atlas** | Side-by-side compare table of 15 frontier + OSS models (Claude, GPT, Gemini, Llama, Mistral, DeepSeek, Qwen, Grok, etc.) — context, price, capabilities |
| **Pick-the-Right-Model Wizard** | 4-step quiz that recommends the top 3 models for your task, budget, latency, and hosting needs |
| **Capability Radar** | Overlay up to 4 models on a 5-axis radar (Code · Reason · Write · Vision · Speed) |
| **Frameworks** | LangGraph · CrewAI · AutoGen/AG2 · OpenAI Agents SDK · Anthropic Agent SDK · MCP · LlamaIndex · A2A — strengths, weaknesses, when-to-use, code, docs |
| **Playground** | TensorFlow-Playground-style simulator: pick topology (sequential / parallel / hierarchical / swarm / router / reflection), agents, model, tools, reflection rounds, task — watch the graph animate, see latency · cost · quality estimates |
| **Pattern Library** | 17 canonical patterns (ReAct, Reflection, Plan-and-Execute, Router, Swarm, Tool-Use, Self-Consistency, Tree-of-Thought, RAG, HITL, Guardrails, …) |
| **Mastery Quiz** | 25 multiple-choice questions across all 10 weeks with explanations and a running score |
| **Resources** | Curated reading list — papers, docs, cookbooks, benchmarks |

## Mapping to your working folder

The 10-week summaries reference the actual files in your `~/AgenticCoding/` directory:

- Week 2 → `week2/claudeapicall.py`, `claudemulti_turn.py`, `claudestreamingapi.py`
- Week 3 → `week3/toolsuse.py`, `buildsmartassistant3tools.py`, `hwadd*tool.py`
- Week 4 → `week4/dronecontrol.py`, `droneeyes.py`, `openrouterfreemodel.py`, `pipeline.py`
- Week 5 → `week5/autoagent.py`, `sample.py`
- Week 6 → `week6/AGENTS.md`, `CLAUDE.md`, `memory-notes.md`, `src/`
- Week 7 → `week7/mcpserver.py`, `mcpfilesystem.py`, `agent.py`, `agenttooldt.py`, `skill.md`
- Week 8 → `week8/Week8_RAG_Knowledge_Agents_Lab.pdf`
- Week 9 → `week9/ex1_crewai_sequential.py`, `ex2_LangGraphSupportGraph.py`, `ex3_ParallelSwarm.py`, `ag2_test.py`, `crewai_test.py`, `langraph_test.py`, `anthropic_test.py`
- Week 10 → production / eval / A2A (no folder yet — synthesized from the syllabus)

## Tips for using it

1. **Start at Overview**, click the journey nodes to jump into any week.
2. **Open the Playground** and try every topology with the same task — see how cost and quality change.
3. **Take the Quiz cold** to find your weak spots, then re-read those weeks.
4. **Use the Wizard** every time you start a new project — it forces you to think about cost vs. quality vs. latency vs. context.

## Tech

- 100% vanilla HTML / CSS / JavaScript in a single file
- No dependencies, no build step, no internet required to run
- ~80 KB, ~1300 lines

— Built for AltoTech Global · Week 11 · 2026-04-27
