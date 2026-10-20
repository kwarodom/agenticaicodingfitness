---
name: models-and-patterns
description: "Help choose the right LLM and the right agent design pattern for a task, and recap the canonical 12 patterns (Reflection, Tool Use, Planning, Multi-Agent, Prompt Chaining, Routing, Parallelization, Orchestrator-Workers, Evaluator-Optimizer, HITL, Topology, ReAct). Covers the 5-question pattern-selection tree, compounding-error math, cost economics (3-tier cascade, prompt caching, cost-per-successful-task), framework support matrix, guardrails, and progressive autonomy. Use when someone asks 'which model/framework/pattern should I use?', 'how do I cut agent cost?', 'how do I add human approval?', wants a pattern catalog, or is reviewing Week 11 / wants a whole-course recap."
when_to_use: "Learner is deciding which model, framework, or design pattern to use, wants to cut agent cost or add human-in-the-loop/guardrails, wants the catalog of agent patterns, or wants a big-picture recap of the whole course (Week 11 mastery lab)."
---

# Choosing Models & Patterns — The Decision Layer (Week 11)

> **The one idea:** Building agents well is mostly *judgment*: which model, which framework, which pattern for *this* problem. This skill is the decision toolkit and the map that ties all 14 weeks together.

---

## Part A — Picking a model

There's no "best model," only the best fit on four axes:

| Axis | Question | When it dominates |
|---|---|---|
| **Capability** | Can it actually do the task (reasoning, code, vision)? | Hard reasoning, complex code, agentic loops |
| **Cost** | $ per million tokens? | High volume, many agents, tight budget |
| **Latency** | How fast does it respond? | Real-time chat, user-facing, big swarms |
| **Context** | How much can it read at once? | Long docs, large codebases, big histories |

**Rules of thumb taught in class:**
- **Frontier model (e.g. Opus/Sonnet tier)** for the hard thinking — planning, tricky code, the "brain" of an agent.
- **Small/fast model (e.g. Haiku tier)** for high-volume, simple, or parallel work — classifiers, routers, swarm members. The Week 9 swarm and Week 11 exercises (`claude-haiku-4-5`) use Haiku *on purpose* for cost.
- **Mix models in one system:** a frontier model plans, cheap models execute. This is a standard cost-control move.
- **Open/alternative providers** (OpenRouter, z.ai/GLM, Qwen) appear across the course — the message: your code shouldn't be married to one vendor.

> 📁 Class repo: `week11_models_and_patterns/README.md` — an interactive **LLM Model Atlas**, a **Pick-the-Right-Model wizard**, and a **capability radar** comparing models. `week04_autonomous_pipelines/openrouterfreemodel.py` routes to a free OpenRouter model (vendor-agnostic, $0 to run).

### Cost economics — the part everyone skips

Agents make **3–10× more LLM calls** than a chatbot for the same task, and an unconstrained coding agent can burn **$5–8 per run** in API fees. Cost is a *design constraint*, not an afterthought. The production answer is a **3-tier cascade**:

```
query → ① semantic cache  → ② small/Haiku tier      → ③ frontier/Opus tier
         (near-$0 hit)        (classify/extract/format)  (only on low confidence
                                                          or a failed quality gate)
```

Small models handle **70–80% of volume**; you pay frontier prices only for the slice that truly needs reasoning. Five stackable levers (from class research):

| Lever | Typical savings | How |
|---|---|---|
| **Model routing** (3-tier cascade) | 40–70% | Classify difficulty, route to the cheapest tier that can do it |
| **Context compaction** | 50–70% fewer tokens | Summarize history before sending to the expensive model |
| **Prompt caching** | ~90% on hits (Anthropic), ~50% (OpenAI) | Cache stable system prompts, tool defs, retrieved docs |
| **Semantic caching** | ~30% query deflection | Reuse answers for same-meaning questions |
| **Batch API** | 50% flat | Queue non-urgent work for a 24h SLA |
| **All five stacked** | **70–85% total** | Layer them; track per-request cost |

> **Break-even rule of thumb:** the cheap tier must clear roughly **~33% success** for cascade routing to pay off. Below that, routing overhead beats the savings — just call the flagship.

**North-star metric: cost per *successful* task** — not cost per token. An agent that uses more tokens but finishes in fewer steps is cheaper *per outcome*. Measure it from day one and **gate CI to block any deploy that raises cost-per-task by >15%** (see `production-and-observability` and `agent-evaluation`).

---

## Part B — Picking a framework

| Use… | When… |
|---|---|
| **Raw Anthropic SDK** | You want full control / are learning the fundamentals / custom orchestration (the swarm). |
| **Claude Agent SDK** | You want the agent loop + MCP handled for you. |
| **CrewAI** | Role-based teams, sequential/parallel crews, fast to stand up. |
| **LangGraph** | Stateful graphs, routing, cycles, fine-grained control flow. |
| **AutoGen / AG2** | Agents that converse with each other. |
| **MCP** | You need standardized, reusable tools across apps. |

Start with the **simplest thing that works**. Reach for a framework when hand-rolling the orchestration becomes the bottleneck — not before.

---

## Part C — The 12-pattern catalog (3 tiers)

Almost every agent you build is some mix of these twelve. They organize into three tiers: **Core** (Andrew Ng's building blocks), **Workflow** (Anthropic's structured LLM pipelines), and **Advanced** (cross-cutting concerns). Production systems usually compose **3–5 at once**.

**🧱 Core 1–4 — single-agent building blocks**

| # | Pattern | What it does | Course link |
|---|---|---|---|
| 1 | **Reflection** | Agent critiques and revises *its own* output until a quality bar is hit. Cap at 2–3 rounds. | code-review agent (W5), `agent-loops` |
| 2 | **Tool Use** | Model calls functions/APIs/DBs to act. Keep <10 tools per context; validate inputs. | `tool-use` (W3), `mcp-and-skills` (W7) |
| 3 | **Planning** | Break a goal into ordered sub-steps (a DAG), then execute; *scoped* replanning on failure. | pipeline (W4), `agent-loops` |
| 4 | **Multi-Agent Collaboration** | Specialized agents (researcher / writer / editor) split the work. Costs 3–5× the tokens — measure the gain first. | `multi-agent-systems` (W9) |

**⚙️ Workflow 5–8 — predictable, structured pipelines**

| # | Pattern | What it does | Course link |
|---|---|---|---|
| 5 | **Prompt Chaining** | Fixed sequence; each step's output feeds the next. Add a **validation gate** so one bad step doesn't cascade. | CrewAI/pipeline (W9), ex02/ex04 |
| 6 | **Routing** | Classify intent → dispatch to the right handler (and the right *model* — cheap vs frontier). | LangGraph router (W9), ex03 |
| 7 | **Parallelization** | Fan out independent calls, then aggregate (sectioning) or majority-vote (voting). | asyncio swarm (W9), ex06 |
| 8 | **Orchestrator-Workers** | A lead LLM decides *at runtime* how many workers to spawn, then synthesizes. Enforce hard step caps **in code**. | hierarchical MAS (W9), ex14 |

**🛡️ Advanced 9–12 — cross-cutting layers**

| # | Pattern | What it does | Course link |
|---|---|---|---|
| 9 | **Evaluator-Optimizer** | **Distinct from Reflection:** a *separate* generator and evaluator, with a **rubric** that emits pass/fail + feedback. Use different models so they don't share blind spots. | `agent-evaluation`, RAGAS (W15) |
| 10 | **Human-in-the-Loop** | Pause for human approval on risky/irreversible actions (Part E). | `week10_production_observability/solutions/03_checkpointing_solution.py` |
| 11 | **Topology** | The wiring of a multi-agent team: Chain / Star / Mesh (Part D). | `multi-agent-systems` (W9) |
| 12 | **ReAct** | Reason → Act → Observe → repeat. The default single-agent loop when the path isn't known in advance. | `agent-loops` (W5) |

> **Reflection (1) vs Evaluator-Optimizer (9):** Reflection is *one* model grading itself ("good enough for first drafts"). Evaluator-Optimizer puts a *second, independent* judge with an explicit rubric in the loop ("good enough to ship"). Reach for #9 when there's a measurable bar — tests that must pass, a style guide to match.

**Plus three emerging patterns** that wrap the twelve: **Bounded Execution** (max steps / tool-call caps / budget circuit-breakers — hard stops in code), **Guardrail Layering** (Part E), and **Context Engineering** (select → compress → isolate; context degrades ~2% per step).

### Framework support matrix

Not every framework does every pattern equally — *native* means built-in primitive, *manual* means you wire it yourself.

| # | Pattern | LangGraph | CrewAI | AutoGen / AG2 | Claude Agent SDK |
|---|---|---|---|---|---|
| 1 | Reflection | native (conditional edges) | native (task) | native (critic agent) | manual (multi-turn) |
| 2 | Tool Use | native (`create_react_agent`) | native (per-agent) | native (registry) | native (MCP pioneer) |
| 3 | Planning | native (`Send` DAG) | native (task deps) | native (GroupChat) | manual (reasoning loop) |
| 4 | Multi-Agent | graph nodes | role-based crews | conversational GroupChat | subagents / handoffs |
| 5 | Prompt Chaining | linear edges | sequential process | sequential | manual |
| 6 | Routing | conditional edges | partial | partial | manual |
| 7 | Parallelization | `Send` (fan-out/in) | async execution | GroupChat | manual |
| 8 | Orchestrator-Workers | `Send` + subgraphs | hierarchical | selector-based | Task tool |
| 9 | Evaluator-Optimizer | conditional loops | separate agents | agent-based | manual |
| 10 | Human-in-the-Loop | `interrupt()` node | callback-based | `UserProxyAgent` | hooks |
| 11 | Topology | graph structure | sequential/parallel | GroupChat topology | subagents |
| 12 | ReAct | prebuilt (`create_react_agent`) | built-in default | built-in | tool use + parsing |

**LangGraph** leads on native coverage (graph foundation). **CrewAI** is the fastest idea→prototype for collaboration. **AutoGen** pioneered conversational multi-agent. **Claude SDK** wins on MCP integration. See `multi-agent-systems` for the deep dive.

> 📁 Class repo: `week11_models_and_patterns/README.md` includes a **Pattern Library**, a topology **Playground** (sequential/parallel/router/swarm/reflection), and a 25-question **mastery quiz**. `week11_models_and_patterns/exercises/README.md` maps 14 runnable MAS exercises to these patterns.

---

## Part D — Topology: how to wire a team (Pattern 11)

When you go multi-agent, the *wiring* is a real decision — it sets your debuggability and fault-tolerance at the same time.

| | **Chain** (A→B→C) | **Star** (hub + spokes) | **Mesh** (any↔any) |
|---|:---:|:---:|:---:|
| Control | high | medium | low |
| Fault tolerance | low | medium | high |
| Debugging | easy | medium | hard |
| Parallelism | none | high | high |
| Production usage | common | **most common** | rare |
| Best for | linear pipelines | parallel subtasks | open-ended exploration |

**Default to Chain. Escalate to Star when you need parallelism. Avoid Mesh in production** — it's flexible but so hard to debug that real systems almost never ship it.

---

## The whole course on one page

```
W2  LLM basics        → text in / text out, messages, streaming, memory   [llm-fundamentals]
W3  Tool use          → model calls your functions                        [tool-use]
W4  Pipelines + IoT   → chained steps; drones & smart lights              [agent-loops]
W5  Agent loop        → REASON→ACT→OBSERVE, the reusable Agent class       [agent-loops]
W6  Full-stack app    → put an agent behind a real API (Express/TS/PG)
W7  MCP & Skills      → reusable tools (MCP) + reusable know-how (skills)  [mcp-and-skills]
W8  RAG               → ground answers in your documents                   [rag-knowledge-agents]
W9  Multi-agent       → CrewAI / LangGraph / AutoGen / swarms              [multi-agent-systems]
W10 Production        → eval, monitoring, cost, HITL, guardrails           [production-and-observability] [agent-evaluation]
W11 Mastery           → pick models/frameworks/patterns                    [models-and-patterns]
W14 Graph memory      → Neo4j, GraphRAG, durable multi-agent memory        [agent-memory-graphs]
W15 Graph mastery     → production GraphRAG: GDS, 7 frameworks, RAGAS eval  [knowledge-graph-mastery]
```

Each `[skill]` above is a skill in this plugin. Start at your gap and walk forward. For keeping the loop sharp over time, see `agent-drills` and `curriculum-and-periodization`.

---

## The 5-question pattern-selection tree

Don't reach for a framework first — answer five questions in order. Each one *adds* a pattern only when a real need shows up.

| # | Question | If **Yes** | If **No** |
|---|---|---|---|
| 1 | Is the solution path **known in advance**? | Workflow patterns 5–7 (Chaining / Routing / Parallelization) | Needs adaptive reasoning → Q2 |
| 2 | Does it need **external tools**? | ReAct (path emerges) or Planning+ReAct (path articulable) | Single LLM + Reflection |
| 3 | Does **quality matter more than speed**? | Add Reflection or Evaluator-Optimizer (+2–3× latency, catches errors) | Proceed to Q4 |
| 4 | Does it need **multiple specializations**? | Multi-Agent + the right Topology | Single agent is enough |
| 5 | Are there **irreversible / high-stakes actions**? | Add Human-in-the-Loop gates (in code) | Ship with the patterns chosen |

**The rule:** start with the simplest pattern that solves the core problem, then layer one more *only* when a specific failure mode demands it. A single well-prompted LLM beats an over-engineered swarm.

### Why the gates aren't optional: compounding errors

Per-step accuracy multiplies. At a *good-looking* **85% per step**, a 10-step chain succeeds only **0.85¹⁰ ≈ 20%** of the time. Real data agrees: only about **24% of agent tasks succeed on the first try**. That math — not paranoia — is *why* you add Reflection, validation gates, and bounded execution to anything running more than a few steps. The longer the chain, the more the gates earn their keep.

```
steps:   1     3     5      8       10
0.85^n:  85%   61%   44%    27%     20%   ← success rate falls off a cliff
```

---

## Part E — Human-in-the-Loop & Guardrails (Patterns 10 + safety)

This is the part demos skip and production can't. The headline cautionary tale: in the **Replit July 2025 postmortem**, an agent ran `DROP DATABASE` on **production** — because no approval gate existed at the database layer. "Please be careful" in a system prompt would not have stopped it.

**The one rule: guardrails live in code, not in prompts.** A destructive action must require a human token *before* execution — enforced with something like `interrupt()`, not a polite instruction the model can ignore or be jailbroken past.

### Progressive autonomy — let agents *earn* trust

Start an agent supervised and graduate it as its measured error rate stays low (**< 5%**):

| Level | Behavior |
|---|---|
| **DRAFT_ONLY** | Agent proposes; a human does everything. |
| **SUPERVISED** | Agent acts, but high-risk actions pause for approval. |
| **MONITORED** | Agent acts freely; humans watch dashboards and can intervene. |
| **FULL** | Agent runs autonomously within hard bounds. |

A `risk_score` (delete = +0.8, write = +0.3, irreversible = +0.5) decides which actions trip the gate. This is exactly what `week10_production_observability/solutions/03_checkpointing_solution.py` demonstrates: a LangGraph `interrupt()` pauses the ticket agent for human **approve / edit / reject**, and the state survives a process restart via the SQLite checkpointer.

> 📁 Class repo: `week10_production_observability/solutions/03_checkpointing_solution.py` — `interrupt()` + human approval + durable checkpointing. Kill the script after the draft step, rerun, and watch it resume from the DB.

### The 4-layer guardrail stack (OWASP-ASI)

Defense in depth — each layer has a distinct job, cheapest checks first:

| Layer | Job | Examples |
|---|---|---|
| **1 · Edge / Network** | Drop obvious junk before paying for inference | rate limits, bot filtering, regex rejects |
| **2 · Input / Prompt** | Inspect what's coming in | PII detection, prompt-injection classifier, context hygiene |
| **3 · Reasoning / Runtime** | Constrain what the agent can *do* | tool allow-lists, arg-schema validation, sandboxing, **Intent Gates** + HITL |
| **4 · Output / Egress** | Filter what goes out | response redaction, hallucination checks, streaming cutoffs |

**Intent Gates** at Layer 3 are the strong form of HITL for high-impact actions: step-up auth / fresh MFA / push approval on a *separate channel*, with task-scoped tokens. Because they sit outside the agent's reasoning path, **no prompt injection can talk its way past them**. See `vibe-coding-and-security` for the full threat model.

### Hand off vs. take control (the partner-not-boss model)

Senior engineers don't type every line *or* blindly accept AI output. Delegate the well-bounded, keep the consequential:

| ✅ Hand off to AI | 🧑 Keep human-owned |
|---|---|
| CRUD operations | Auth / authz logic |
| Boilerplate & scaffolding | Data model design |
| Test generation | API contracts |
| UI component scaffolding | Deployment config |
| Docs, refactors within a pattern | Error-handling strategy, novel domains |

AI-generated code carries **~1.7× more major issues**, and they cluster exactly at these boundaries — where systems interact. Review the *plan* before implementation, run tests before accepting, and commit AI changes separately for clean rollback.

---

## 🧪 Guided lab (offer this)

**Goal:** practice *judgment* — pick a model, a pattern set, and a cost/safety posture for a real goal. Runs at **$0 / no API key** via a `MockLLM` stub.

### Warm-up (5–10 min, pass/fail)
Answer out loud, in order:
1. For a **10-step** chain at 85%/step, what's the success rate? (≈ **20%** — and *why* that means you add gates.)
2. Name the **3 tiers** of the cost cascade in order. (semantic cache → small/Haiku → frontier/Opus.)
3. Reflection vs. Evaluator-Optimizer — what's the *one* structural difference? (separate generator + evaluator with a **rubric**.)

**Pass = all 3 correct.**

### Skill Drill (15–30 min, runnable)
Build a tiny router + reflection + HITL pipeline with a fake LLM — no keys, no cost:

```python
# pip install nothing. Pure stdlib. Run: python drill.py
import random

class MockLLM:
    """$0 stand-in. Deterministic enough to test the *wiring*, not the model."""
    def __init__(self, tier): self.tier = tier
    def __call__(self, prompt):
        if "classify" in prompt:  return random.choice(["billing", "technical", "danger"])
        if "score" in prompt:     return random.choice([6, 9])          # quality gate
        return f"[{self.tier}] handled: {prompt[:30]}"

cheap, frontier = MockLLM("haiku"), MockLLM("opus")

def route(query):                                   # Pattern 6: Routing
    intent = cheap(f"classify: {query}")
    return {"billing": cheap, "technical": frontier}.get(intent, "DANGER" if intent=="danger" else cheap), intent

def reflect(model, query, max_rounds=3):            # Pattern 1: Reflection (+ bounded execution)
    draft = model(query)
    for _ in range(max_rounds):
        if cheap(f"score this draft: {draft}") >= 8: return draft
        draft = model(f"improve: {draft}")
    return draft

def gate(action):                                   # Pattern 10: HITL — in code, not prompt
    risk = 0.8 if "danger" in action else 0.0
    if risk > 0.5:
        ok = input(f"⚠️  high-risk ({risk}). approve? [y/N] ").lower() == "y"
        return "EXECUTED" if ok else "REJECTED"
    return "EXECUTED"

for q in ["fix my invoice", "debug prod outage", "danger: drop database"]:
    model, intent = route(q)
    if model == "DANGER":
        print(q, "→", gate("danger"))               # gate fires BEFORE any action
    else:
        print(q, "→", reflect(model, q))
```

**Then break it to learn it:** raise the per-step "accuracy" failure rate and watch a longer chain collapse; remove the `gate()` call and feel why the Replit story happened; swap the danger query's gate from code into a prompt string and note that the model could ignore it.

### Weighted evaluation criteria
| Weight | Criterion |
|---|---|
| 25% | **Routing** sends cheap vs. frontier work to the right tier (cost awareness) |
| 25% | **Reflection** loop is *bounded* (caps rounds — no infinite refine) |
| 25% | **HITL gate** is enforced in code and fires *before* the dangerous action |
| 15% | Learner can state the **cost-per-successful-task** metric and the **>15% CI gate** |
| 10% | Learner names the **patterns** used (router + reflection + HITL) and the **topology** |

**Pass = 4 / 5 criteria (≥ 75%).**

### Stretch
Open `week11_models_and_patterns/README.md` in a browser and run the model wizard, the topology playground, and the 25-question quiz. Then open `week10_production_observability/solutions/03_checkpointing_solution.py` and replace the `MockLLM` gate with a real `interrupt()` + checkpointer.

End with the meta-lesson: "frameworks and models change every few months; the *patterns*, the *cost discipline*, and the *judgment* are what you keep."
