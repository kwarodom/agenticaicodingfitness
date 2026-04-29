# Multi-Agent Systems — Coding Exercises

14 hands-on exercises drawn from `week9/week9_3_mas_coding_exercises.pdf`.
Originally split across CrewAI, LangGraph, AutoGen, and the Anthropic SDK.
This implementation deliberately omits CrewAI — the five exercises that
specified it use a `dataclass Agent` + raw `anthropic` SDK pattern instead,
so the coordination concept is preserved without the framework dependency.

## Setup

All exercises read `ANTHROPIC_API_KEY` from `.env` at the repo root and use
`claude-haiku-4-5-20251001`. Activate the project venv before running:

```bash
cd /Users/jirayutchatphet/Code/AltoTech/agenticaicodingfitness
source .venv/bin/activate
python week11/exercises/ex01_two_agent_dialogue/ex01_two_agent_dialogue.py
```

Required packages (already installed in `.venv`):
`anthropic`, `langgraph`, `langchain-anthropic`, `autogen-agentchat`,
`autogen-ext[anthropic]`, `python-dotenv`.

## Exercises by tier

### Beginner — core concepts, raw API calls, 2-agent dialogues

| # | Folder | Framework | Key concept |
|---|---|---|---|
| 01 | `ex01_two_agent_dialogue/` | Anthropic | Independent message histories, turn-taking |
| 02 | `ex02_research_brief/` | Anthropic (was CrewAI) | Sequential pipeline, Researcher → Writer |
| 03 | `ex03_langgraph_router/` | LangGraph | StateGraph, conditional edges, START/END |

### Easy — 3-agent pipelines, state machines, parallel calls

| # | Folder | Framework | Key concept |
|---|---|---|---|
| 04 | `ex04_3agent_product_pipeline/` | Anthropic (was CrewAI) | 3-stage chain: Scout → Analyst → Strategist |
| 05 | `ex05_langgraph_faq_loop/` | LangGraph | `Annotated[list, operator.add]`, cycles |
| 06 | `ex06_parallel_fact_checker/` | Anthropic + asyncio | `asyncio.gather`, confidence-weighted voting |

### Intermediate — tool use, group chats, escalation logic

| # | Folder | Framework | Key concept |
|---|---|---|---|
| 07 | `ex07_ag2_code_review/` | AutoGen v0.4+ | `RoundRobinGroupChat`, deterministic turn order |
| 08 | `ex08_langgraph_escalation/` | LangGraph | L1 → L2 → Emergency, multi-field state, cycles |
| 09 | `ex09_market_research_swarm/` | Anthropic asyncio (was CrewAI) | Sequential prerequisite → parallel → synthesis |

### Advanced — self-healing, negotiation, hybrid frameworks

| # | Folder | Framework | Key concept |
|---|---|---|---|
| 10 | `ex10_langgraph_self_healing/` | LangGraph | Bounded retries, AI-powered repair, multi-route routing |
| 11 | `ex11_negotiation_agents/` | Anthropic | Stateful agent class, BATNA, terminal-keyword detection |
| 12 | `ex12_hybrid_energy_optimizer/` | Anthropic + LangGraph (was CrewAI + LangGraph) | Two-phase: analysis crew → execution graph with rollback |

### Expert — autonomous systems, long-horizon planning, full MAS

| # | Folder | Framework | Key concept |
|---|---|---|---|
| 13 | `ex13_autonomous_research/` | LangGraph + tool_use | Planner-executor-critic, native Claude tool calls |
| 14 | `ex14_hotel_ops_command_center/` | AutoGen + Anthropic (was AutoGen + CrewAI) | `SelectorGroupChat` over 4 dept agents with pre-computed sub-crew briefs |

## Implementation notes

- **AutoGen API**: PDF code uses AutoGen v0.2 (`import autogen`,
  `autogen.AssistantAgent`). Installed package is v0.4+ (`autogen_agentchat`,
  `autogen_ext`). Ex 07 and Ex 14 use the modern split-package API.
- **No CrewAI by design**: exercises 02, 04, 09, 12, 14 use a `dataclass Agent`
  with `system_prompt()` and `run()` methods — same coordination patterns
  (sequential, context-chaining, sub-crews) without the framework.
- **Output files** appear next to each exercise after the first run
  (e.g., `ex02_brief_output.md`, `ex09_swarm_output.md`,
  `ex12_*_output.md`).
