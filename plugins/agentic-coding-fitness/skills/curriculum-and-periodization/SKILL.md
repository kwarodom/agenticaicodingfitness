---
name: curriculum-and-periodization
description: "The syllabus and learning-path index for the whole program — what to learn, in what order, and why. Maps the blueprint's 16-week progressive-overload plan (Foundation → Strength → Endurance → Performance, with deload weeks) onto THIS repo's real weeks 2–17, names the plugin skill that recaps each week, and hands out suggested paths for total beginners through production engineers. Covers the four overload levers (intensity, volume, complexity, autonomy), the daily Warm-Up → Main Workout → Cool-Down → Recovery session, and the four certification tiers. Use when someone asks 'where do I start?', 'what's the learning path / curriculum / syllabus?', 'what order should I learn this in?', 'how is the course structured?', 'what week am I on / what's next?', or wants a big-picture map of the whole program."
when_to_use: "Learner wants to know where to start, what order to learn the skills in, how the whole program is structured into phases/weeks, or which two skills to do next — i.e. they need the syllabus and a path, not a single concept."
---

# Curriculum & Periodization — Train in the Right Order

> **The one idea:** Don't bench-press before you can do a push-up. Skills build on skills, so we *periodize* the training — phases that progressively overload you (Foundation → Strength → Endurance → Performance), with deliberate deload weeks to consolidate. This skill is the map: where you are, what's next, and why that order.

This is the **index** for every other skill in the plugin. Each concept lives in its own sibling skill; this one orders them. If you only read one skill first, read this, then jump to your gap.

---

## Part A — The four phases (progressive overload)

Athletes don't lift max weight on day one. They **periodize**: cycle through phases that each stress a different capacity, raising the load gradually so the body adapts without breaking. Learning agentic coding works the same way — you build *work capacity* before *strength*, and *strength* before *endurance under production load*.

| Phase | Fitness analog | What you build | Competence stage | Recap skills |
|---|---|---|---|---|
| **① Foundation** | Preparatory — 50–75% intensity, high volume | The base: talk to a model, give it tools, run the loop | Conscious incompetence | `llm-fundamentals`, `tool-use`, `agent-loops` |
| **② Strength** | Basic strength — heavier loads, lower volume | Single-agent mastery under constraint; reusable tools & knowledge | Conscious competence | `mcp-and-skills`, `rag-knowledge-agents` |
| **③ Endurance** | Hypertrophy — sustained moderate load | Systems that *run reliably*: many agents, eval, observability, cost | Consolidated competence | `multi-agent-systems`, `agent-evaluation`, `production-and-observability`, `models-and-patterns` |
| **④ Performance** | Peaking — 85–100%, explosive | Durable memory, GraphRAG, production frameworks, long-running & distributed fleets | Unconscious competence | `agent-memory-graphs`, `knowledge-graph-mastery`, `nemo-agent-toolkit`, `long-running-and-distributed-agents` |

### The four overload levers

You make training harder by turning exactly the knobs that matter — never all at once:

| Lever | Turn it up by… | In coding that means… |
|---|---|---|
| **Intensity** | tighter constraints, stricter requirements | harder problems, stricter eval gates |
| **Volume** | more reps/projects per week | more exercises, bigger codebases |
| **Complexity** | new frameworks & failure modes | LangGraph, GraphRAG, multi-agent orchestration |
| **Autonomy** | less hand-holding | from copy-the-snippet → design-it-yourself → ship-it |

> 🏋️ **Deload weeks** are not breaks — they're consolidation. The blueprint drops all three of intensity/volume/complexity by ~40% at **weeks 4, 8, 12** so the new skill *sticks* (supercompensation: you get stronger *after* the rest, not during the grind). In this repo, the natural deloads are the integration weeks (the full-stack app in week06_fullstack_agent_app, the mastery recap in week11_models_and_patterns) where you compose what you already learned instead of adding a new pattern.

> ⚠️ **Conceptual:** the clean "16 weeks, four phases, deloads at 4/8/12" structure (and the *16-Week Progressive Overload Profile* figure where intensity/volume/complexity rise and dip together) is the **blueprint's ideal**, drawn from fitness periodization + the 4C/ID instructional model. The **runnable** curriculum in this repo is **weeks 2–17** — see Part B for the honest mapping.

---

## Part B — The week-by-week map (blueprint ideal → this repo)

The blueprint describes an idealized 16 weeks. This repo has the **real, runnable weeks 2–17** (one past the blueprint). They line up *closely* but not perfectly — the table below is the honest bridge. Cite the repo column; treat the blueprint column as the "why this order."

| Real week (this repo) | Concept | Recap skill | Blueprint's idealized slot |
|---|---|---|---|
| **week02_claude_api_foundations** — LLM basics | messages, multi-turn, streaming | `llm-fundamentals` | ~W1 LLM architecture |
| **week03_tool_use** — tool use | function calling, multi-tool assistant | `tool-use` | ~W2 ReAct / function calling |
| **week04_autonomous_pipelines** — pipelines + IoT | chained steps, drones & lights, OpenRouter | `agent-loops` | ~W4 prompt chaining + routing |
| **week05_agent_loop** — the agent loop | REASON→ACT→OBSERVE, reusable Agent class | `agent-loops` | ~W2–3 ReAct + reflection |
| **week06_fullstack_agent_app** — full-stack app | agent behind a real API (Express/TS/Postgres) | — (integration; like a deload) | ~W11 production patterns |
| **week07_mcp_and_skills** — MCP & Skills | reusable tools (MCP) + reusable know-how | `mcp-and-skills` | **W1 in the blueprint** — it front-loads MCP |
| **week08_rag_knowledge_agents** — RAG | ground answers in your documents | `rag-knowledge-agents` | ~W8 RAG (matches) |
| **week09_multi_agent_systems** — multi-agent | sequential / parallel swarm / router; CrewAI, LangGraph, AG2 | `multi-agent-systems` | ~W5–9 orchestration |
| **week10_production_observability** — production & eval | LangGraph guide, testing, monitoring, cost | `production-and-observability`, `agent-evaluation` | ~W10–11 eval + observability |
| **week11_models_and_patterns** — mastery | pick models/frameworks/patterns; pattern playground | `models-and-patterns` | ~W12 integration + governance |
| **week14_agent_memory_knowledge_graphs** — graph memory | Neo4j, GraphRAG, durable multi-agent memory | `agent-memory-graphs` | ~W13–14 capstone-grade memory |
| **week15_production_graphrag** — production GraphRAG | Cypher+GDS, ingestion, 7 frameworks, RAGAS, CI gate | `knowledge-graph-mastery` | ~W15–16 polish + certification |
| **week16_nemo_agent_toolkit** — production frameworks | NVIDIA NeMo Agent Toolkit: register tools, YAML workflows, supervisor + HITL, observability | `nemo-agent-toolkit` | ~W12 framework adoption + governance |
| **week17_long_running_distributed_agents** — long-running & distributed | Google ADK durable state machines, pause/resume, webhooks, auth.md credentials, A2A fleet | `long-running-and-distributed-agents`, `a2a-protocol` | ~W13–14 capstone-grade orchestration |

**The biggest divergence to call out:** the blueprint puts **MCP in Week 1** ("the USB port of AI coding" — the gateway skill). This repo reaches MCP in **week07_mcp_and_skills**, *after* you've felt the pain of hand-wiring tools (week03_tool_use) and run a full agent loop (week05_agent_loop). Both are defensible: the blueprint front-loads the ecosystem standard; this repo front-loads the fundamentals so MCP lands as "oh, *that's* what this standardizes." There's no week12/13 here, and the repo runs one week past the blueprint's 16 — **week17_long_running_distributed_agents** is the real fleet/long-running capstone (durable agents + A2A) that the blueprint only gestured at.

> 📁 Class repo: `week02_claude_api_foundations/claudeapicall.py` — your very first API call (Foundation starts here).
> 📁 Class repo: `week03_tool_use/toolsuse.py` — the model calling your functions for the first time.
> 📁 Class repo: `week05_agent_loop/autoagent.py` — the reusable REASON→ACT→OBSERVE Agent class.
> 📁 Class repo: `week06_fullstack_agent_app/src/server.ts` — an agent put behind a real Express/TS/Postgres API (the "deload" integration week).
> 📁 Class repo: `week07_mcp_and_skills/skill.md` + `week07_mcp_and_skills/mcpserver.py` — where MCP and Skills finally arrive (week 7 here, week 1 in the blueprint).
> 📁 Class repo: `week08_rag_knowledge_agents/Week8_RAG_Knowledge_Agents_Lab.pdf` — the RAG lab.
> 📁 Class repo: `week09_multi_agent_systems/ex2_LangGraphSupportGraph.py` — a router/graph multi-agent system.
> 📁 Class repo: `week10_production_observability/README.md` — the production/eval week (testing, monitoring, cost).
> 📁 Class repo: `week11_models_and_patterns/README.md` — the mastery recap: model wizard, pattern playground, 25-question quiz.
> 📁 Class repo: `week14_agent_memory_knowledge_graphs/agent_memory.py` + `week14_agent_memory_knowledge_graphs/lab1_hotel_mas.py` — durable graph memory for a multi-agent hotel.
> 📁 Class repo: `week15_production_graphrag/kg_mastery/README.md` — the full production-GraphRAG deep dive (Parts 1–6).
> 📁 Class repo: `week16_nemo_agent_toolkit/multi_agent_orchestration.ipynb` — NeMo Agent Toolkit: register tools, compose a YAML workflow, supervisor → specialists with HITL.
> 📁 Class repo: `week17_long_running_distributed_agents/checkpoints/checkpoint1_state_machine.py` → `checkpoint6_fleet.py` — long-running durable agents (pause/resume) + A2A fleet, offline; plus the live `week17_long_running_distributed_agents/hr_onboarding/` and `week17_long_running_distributed_agents/authmd_adk/` services.

> The single best "you are here" overview is the one-page course map in `models-and-patterns` — it lines up the taught weeks against their skills in a single screen.

---

## Part C — Daily session structure

Every training day (the blueprint's 4C/ID workout) has the same four beats. Use this shape for *any* skill's guided lab.

| Beat | Time | What you do | Why |
|---|---|---|---|
| **Warm-Up** | 10–15 min | Quick recall of yesterday's concept; skim the new one | Spaced repetition; primes memory |
| **Main Workout** | 45–60 min | Micro-lesson → guided practice → solo build → stretch challenge | The actual overload (one new pattern) |
| **Cool-Down** | 10–15 min | Write down what you learned, note confusions, **commit the code** | Encodes it into long-term memory |
| **Recovery** | (deload days) | Light review, community, *no new concepts* | Consolidation; supercompensation |

> ⚠️ **Conceptual:** the four certification tiers below are the blueprint's stretch goal, not something this repo issues. Use them as a self-assessment ladder — "can I do this unaided?" — one tier per phase:

| Tier | Earned after phase | You can… |
|---|---|---|
| **Agentic AI Fundamentals** | Foundation | call a model, give it tools, run a loop |
| **Agentic AI Developer** | Strength | ship a single agent with RAG/MCP under constraint |
| **Agentic AI Engineer** | Endurance | run a multi-agent system with eval + monitoring + cost control |
| **Agentic AI Architect** | Performance | design durable-memory GraphRAG systems end-to-end |

---

## Part D — Suggested paths (pick one)

Same ladders as the plugin README, condensed — pick the row that matches where you are and walk the skills left to right.

| If you are… | Walk this path |
|---|---|
| **A total beginner** | `llm-fundamentals` → `tool-use` → `agent-loops` → then pick what interests you |
| **"I can call the API, what's next?"** | `tool-use` → `agent-loops` → `mcp-and-skills` |
| **"I want agents that collaborate"** | `agent-loops` → `multi-agent-systems` → `agent-memory-graphs` |
| **"I want it to use my data"** | `rag-knowledge-agents` → `agent-memory-graphs` → `knowledge-graph-mastery` |
| **"I want production I can trust"** | `production-and-observability` → `agent-evaluation` → `knowledge-graph-mastery` (build, monitor, then *prove* it with RAGAS) |
| **"Just give me the big picture"** | `models-and-patterns` (the one-page map), then this skill |

**Cross-cutting skills** (not tied to one week — pull them in anytime): `agent-drills` (daily warm-up reps), `vibe-coding-and-security` (the safety layer for AI-written code), `a2a-protocol` (agents talking across systems), `skill-authoring` (write your own skills). For warm-up reps that fit the daily structure in Part C, `agent-drills` is the gym.

---

## 🧪 Guided lab (offer this): *Place yourself on the map*

No API key, no code — this is a **planning kata** ($0). Goal: locate yourself and commit to your next two skills.

**Warm-up (5–10 min, pass/fail).** Answer out loud / on paper:
1. Which **phase** (Foundation / Strength / Endurance / Performance) are you in *right now*?
2. Name the **last real repo week** you actually ran (e.g. `week05_agent_loop/autoagent.py`).
3. Name the **skill** that recaps it (from Part B's table).
> ✅ Pass the warm-up if all three are concrete and consistent (the week and skill match Part B).

**Skill Drill (15–30 min, runnable, $0).** Build your personal plan as a tiny script — a "MockLLM" stub stands in for a coach so it runs with no key:

```python
# plan_kata.py  — runs at $0, no API key. Picks your next two skills from where you are.
SYLLABUS = [  # (real repo week, recap skill, phase)
    ("week02_claude_api_foundations",  "llm-fundamentals",       "Foundation"),
    ("week03_tool_use",  "tool-use",               "Foundation"),
    ("week04_autonomous_pipelines",  "agent-loops",            "Foundation"),
    ("week05_agent_loop",  "agent-loops",            "Foundation"),
    ("week07_mcp_and_skills",  "mcp-and-skills",         "Strength"),
    ("week08_rag_knowledge_agents",  "rag-knowledge-agents",   "Strength"),
    ("week09_multi_agent_systems",  "multi-agent-systems",    "Endurance"),
    ("week10_production_observability", "production-and-observability", "Endurance"),
    ("week11_models_and_patterns", "models-and-patterns",    "Endurance"),
    ("week14_agent_memory_knowledge_graphs", "agent-memory-graphs",    "Performance"),
    ("week15_production_graphrag", "knowledge-graph-mastery","Performance"),
    ("week16_nemo_agent_toolkit", "nemo-agent-toolkit",     "Performance"),
    ("week17_long_running_distributed_agents", "long-running-and-distributed-agents", "Performance"),
]

class MockLLM:
    """Stand-in 'coach' — deterministic, no API. Real coach would call client.messages.create()."""
    def next_two(self, last_week_done: str):
        weeks = [w for w, _, _ in SYLLABUS]
        i = weeks.index(last_week_done)
        return SYLLABUS[i + 1 : i + 3]  # the next two rungs

# ---- edit ONE line: the last repo week you actually ran ----
LAST_DONE = "week05_agent_loop"

plan = MockLLM().next_two(LAST_DONE)
print(f"You are in: {dict((w, p) for w, _, p in SYLLABUS)[LAST_DONE]} phase\n")
print("Your next two skills:")
for week, skill, phase in plan:
    print(f"  → {skill:<28} ({week}, {phase} phase)")
assert len(plan) >= 1, "You've reached the end — go build a capstone."
print("\nWarm-Up → Main Workout → Cool-Down → commit. One pattern at a time.")
```

Run it: `python plan_kata.py`. It prints your phase and your **next two skills** — your actual plan for this week.

**Weighted evaluation criteria** (score yourself):
| Criterion | Weight |
|---|---|
| Correctly identified your current **phase** (Part A) | 1 |
| Correctly named the **real repo week + recap skill** you last ran (Part B) | 1 |
| Script runs and prints two next skills (or a sensible end-of-syllabus message) | 1 |
| Chosen path matches one of the Part D ladders (you can justify *why* in a sentence) | 1 |
| You can explain **one** way the blueprint's 16-week ideal differs from this repo (e.g. MCP in W1 vs week07_mcp_and_skills) | 1 |

**Pass = 4/5 criteria.** If you pass, your homework is literally the two skills the script printed — go run their guided labs.
