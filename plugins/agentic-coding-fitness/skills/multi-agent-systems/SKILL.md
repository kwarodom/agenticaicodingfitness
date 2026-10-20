---
name: multi-agent-systems
description: "Teach multi-agent systems (MAS) — coordinating several specialized agents instead of one do-everything agent. Covers the 6 orchestration patterns (supervisor-worker, hierarchical, sequential pipeline, parallel swarm, router/dispatcher, debate), an If-X→pattern decision table, communication topologies (chain/star/mesh) and state management (shared/message-passing/event-driven/blackboard), plus the production frameworks (LangGraph create_supervisor + checkpointers, CrewAI processes, Claude Agent SDK subagents). Use when someone asks 'how do I make agents work together?' or 'which pattern do I pick?', mentions CrewAI/LangGraph/AutoGen/swarms/supervisors/checkpointing/HITL, or is reviewing Week 9 / Week 10."
when_to_use: "Learner wants multiple agents collaborating, is choosing an orchestration pattern or topology, asks about CrewAI / LangGraph / AutoGen / supervisors / swarms / checkpointing / HITL, or is catching up on Week 9–10."
---

# Multi-Agent Systems — Many Specialists, One Goal (Week 9–10)

> **The one idea:** Instead of one agent juggling everything, use a *team* of focused agents — each with a narrow role — and a coordination pattern that decides who does what, when. A researcher + writer + editor beats one agent told to "research, write, and edit."

**Why bother?** Specialization (sharper prompts per role), parallelism (speed), modularity (swap one agent without breaking others), and separation of concerns.

---

## Six orchestration patterns worth knowing (this is the real lesson)

Every MAS answers one question: **who decides what happens next?** These six are the different answers. You'll code **Sequential, Router, and Parallel** hands-on in Week 9 (`ex1`–`ex3`); **Supervisor-Worker** and **Hierarchical** arrive as the Week 10 production upgrades. Of the six, **Supervisor-Worker** is the one you'll actually ship.

### 1. Supervisor-Worker — the production default 🏆
A central **supervisor** reads the request, splits it into subtasks, delegates each to a specialist **worker** (with isolated context), and aggregates the results. It can *replan* mid-run — perfect when the subtasks aren't known up front.
**Use for:** almost everything. This is the most-deployed MAS architecture in production. Anthropic's own research system used an orchestrator-worker shape with parallel subagents.
**Cost:** every message routes through the supervisor → latency grows past ~7–10 workers.

### 2. Hierarchical — supervisor-worker as a tree
Extend pattern 1: a root orchestrator manages **domain supervisors**, each managing its own workers (nutrition team, workout team, recovery team). Two levels (router → specialists) beats both flat *and* deep 3+ level trees on consistency.
**Use for:** 20+ agents across distinct domains that each need their own guardrails.

### 3. Sequential pipeline — assembly line
Agent A's output feeds Agent B feeds Agent C. Order is fixed.
**Use for:** workflows with clear stages (research → write → edit). Predictable and cheap — but brittle: a failure at any stage blocks everything downstream. Use only when the dependency is *genuine*, not an artifact of lazy decomposition.

### 4. Parallel swarm — fan-out, then aggregate
Many agents work *at the same time* on independent sub-tasks; an aggregator merges results. Members self-organize around shared memory, claiming work by capability.
**Use for:** independent analyses (audit a building from 5 angles at once). ~3–5× faster than sequential. (Most production "swarms" are really supervisors with dynamic task assignment.)

### 5. Router / dispatcher — triage desk
One classifier agent inspects the input and routes it to the right specialist.
**Use for:** mixed inputs (support tickets → billing vs. technical vs. general).

### 6. Debate — adversarial stress-test
Two+ agents argue opposing positions while a **judge** evaluates ("Devil-Angel": one finds risks in a workout plan, one defends it). Surfaces blind spots a single agent misses.
**Use for:** high-stakes decisions where a missed failure mode costs more than the extra tokens. (Reserve it — you're running 3+ agents on one problem.)

> **A seventh you'll hear named:** *Peer-to-peer / mesh* — agents coordinate with **no** central supervisor. In practice it's a *wiring* choice more than a delegation strategy, so it's covered below under **communication topologies**, not as a seventh pattern here.

```
Supervisor:  request → supervisor → [worker, worker] → supervisor → answer
Hierarchical: root → [supervisor → [worker,worker]], [supervisor → [worker]]
Sequential:  A → B → C
Parallel:    A ┐
             B ┼→ aggregate
             C ┘
Router:      classify → (billing | technical | general)
Debate:      pro ⇄ con  → judge → verdict
```

---

## Decision framework — *If your problem has X → pick Y*

| If your problem has… | Best pattern | Why | Avoid when |
|---|---|---|---|
| Unknown subtasks, needs dynamic decomposition | **Supervisor-Worker** | Coordinator replans as context evolves | Pipeline (can't adapt) |
| 20+ agents across multiple domains | **Hierarchical** | Layered isolation prevents supervisor overload | Flat supervisor (bottleneck) |
| Fixed linear workflow, known steps | **Sequential pipeline** | Predictable, lowest overhead | Swarm (needless complexity) |
| Independent analyses you can run at once | **Parallel swarm** | Shared context enables emergent speed-up | Pipeline (serializes for no reason) |
| Mixed inputs needing triage | **Router / dispatcher** | One classifier sends each to the right specialist | Mesh (over-connected) |
| High-stakes decision needing stress-testing | **Debate** | Adversarial critique surfaces blind spots | Supervisor (single viewpoint) |

**The rule that saves projects:** *default to Supervisor-Worker* unless you have a specific reason not to. Teams under ~50 people rarely need more than 5–7 agents, and a flat supervisor beats a hierarchy on latency *and* debuggability at that scale. **Only add a 3rd level once a single supervisor manages more than ~7 agents.** (Gartner: ~40% of multi-agent projects get cancelled — usually from orchestration complexity nobody needed.) See `models-and-patterns` for the model/framework half of the decision.

---

## Pattern 1 in CrewAI — sequential crew

CrewAI models a team as **Agents** (role + goal + backstory) and **Tasks**, run by a **Crew** with a process.

```python
from crewai import Agent, Task, Crew, Process, LLM

llm = LLM(model="openai/glm-5", base_url="https://api.z.ai/api/paas/v4/", api_key=...)

researcher = Agent(role="Senior Research Analyst",
                   goal="Find the latest trends in building energy optimization",
                   backstory="Expert energy analyst tracking IoT + AI HVAC.",
                   llm=llm, tools=[search_tool], allow_delegation=False)
writer = Agent(role="Technical Content Writer",
               goal="Write an engaging blog post from the research",
               backstory="Writes for the engineering blog.", llm=llm)
editor = Agent(role="Chief Editor",
               goal="Polish for clarity, accuracy, brand voice",
               backstory="Senior editor.", llm=llm)

research_task = Task(description="Research AI-driven building energy trends...",
                     expected_output="5 findings with sources", agent=researcher)
writing_task  = Task(description="Write a 600-word post from the brief...",
                     expected_output="A 600-word markdown post", agent=writer)
editing_task  = Task(description="Review and polish...",
                     expected_output="Publication-ready post", agent=editor,
                     output_file="post.md")

crew = Crew(agents=[researcher, writer, editor],
            tasks=[research_task, writing_task, editing_task],
            process=Process.sequential, verbose=True)
print(crew.kickoff())
```

The **role + goal + backstory** is just a well-structured system prompt per agent. Output of each task flows into the next.

> 📁 Class repo: `week09_multi_agent_systems/ex1_crewai_sequential.py`

---

## Pattern 3 in LangGraph — router as a state machine

LangGraph models the system as a **graph**: nodes are steps, a shared **state** flows through, and **conditional edges** pick the path.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

class TicketState(TypedDict):
    ticket_text: str
    category: str
    response: str

def classify(state):          # router node → sets state["category"]
    ...  # ask the LLM: billing / technical / general
    return {"category": category}

def billing_agent(state):   ...   # specialist nodes
def technical_agent(state): ...
def general_agent(state):   ...

g = StateGraph(TicketState)
g.add_node("classify", classify)
g.add_node("billing", billing_agent)
g.add_node("technical", technical_agent)
g.add_node("general", general_agent)
g.set_entry_point("classify")
g.add_conditional_edges("classify", lambda s: s["category"],
    {"billing": "billing", "technical": "technical", "general": "general"})
for n in ["billing", "technical", "general"]:
    g.add_edge(n, END)
app = g.compile()

app.invoke({"ticket_text": "I don't recognize a charge on my invoice."})
```

This is centralized planning / BDI-style deliberation: decide **what** (classify) then **how** (route to specialist).

> 📁 Class repo: `week09_multi_agent_systems/ex2_LangGraphSupportGraph.py`

---

## Pattern 2 — parallel swarm with asyncio

No framework needed — just `asyncio.gather` to fan out and aggregate. Five specialists audit a building concurrently:

```python
import asyncio, anthropic
client = anthropic.AsyncAnthropic()
MODEL = "claude-haiku-4-5-20251001"

SPECIALISTS = {
    "energy":      "Energy Efficiency Auditor",
    "comfort":     "Occupant Comfort Analyst",
    "maintenance": "Predictive Maintenance Engineer",
    "safety":      "Safety Compliance Officer",
    "cost":        "Cost Optimization Analyst",
}

async def run_specialist(name, role):
    resp = await client.messages.create(
        model=MODEL, max_tokens=300,
        system=f"You are a {role}. Give a concise 3-bullet audit finding.",
        messages=[{"role": "user", "content": "Audit a 50-floor Bangkok office tower."}])
    return name, resp.content[0].text

async def run_audit():
    results = await asyncio.gather(*(run_specialist(n, r) for n, r in SPECIALISTS.items()))
    for name, finding in results:
        print(f"--- {name} ---\n{finding}\n")

asyncio.run(run_audit())   # all 5 finish in ~the time of the slowest, not the sum
```

This is **distributed sensing + stigmergy**: agents work independently and write to shared results; an aggregator synthesizes.

> 📁 Class repo: `week09_multi_agent_systems/ex3_ParallelSwarm.py`

---

## Communication topologies — how the wires are arranged

A *pattern* says who decides; a *topology* says who can talk to whom. Same diagram, two trade-offs that matter at 2 AM: fault tolerance and observability.

| Dimension | **Chain** (A→B→C) | **Star** (hub-and-spoke) | **Mesh** (any↔any) |
|---|---|---|---|
| Communication edges | n−1 (linear) | 2n (hub↔spokes) | up to n(n−1) |
| Single point of failure | none, but cascades | **hub is SPOF** | no SPOF |
| Latency at 6 agents | cumulative | ≤2 hops | 1 hop any pair |
| Observability | high (linear trace) | **highest** (one central log) | lowest (distributed) |
| Best agent count | 3–6 | 3–7 per hub | 2–4 |
| Compliance / audit | medium | **strong** (single audit trail) | weak |

**Takeaway:** Star == Supervisor-Worker's wiring, and its single central log is *why* supervisors are so easy to audit. Mesh (peer-to-peer) gives the most fault tolerance but the worst debuggability — reserve it for 2–4 agent brainstorming clusters, almost never for transaction-heavy production.

---

## State management — what the agents *remember*

Topology decides who talks; **state model** decides what they share. Four choices:

| Model | What it is | Watch out for |
|---|---|---|
| **Shared state** | All agents read/write one store (a LangGraph `TypedDict`, CrewAI `memory=True`). | One agent's hallucination propagates to all downstream. **Add validation gates between writes and reads.** |
| **Message passing** | Each agent keeps local state; they exchange schema-validated messages. | More plumbing, but clean failure isolation. (The `a2a-protocol` task/message model is standardized message passing.) |
| **Event-driven** | Pub/sub — `WorkoutCompleted` fires the Tracker, Recovery, and Feed agents at once. | Eventual consistency: a subscriber may process seconds/minutes late. |
| **Blackboard** | A shared workspace where agents post **structured artifacts** (findings, decisions), not chat logs. | Central write bottleneck → scales poorly past a handful of agents. |

### ⚠️ The anti-pattern that blows your token budget: unbounded context accumulation

Agents that append the **full transcript** every step will eventually blow the context window, and token cost climbs *exponentially* with run length. This is one of the top causes of MAS bills exploding. The fix is three habits:

1. **Store artifacts, not transcripts.** Put the *meal plan*, the *audit finding*, the *decision* into shared state — not the 4,000-token conversation that produced it.
2. **Require source attribution.** Every claim added to shared state names where it came from, so a later validation gate can check it.
3. **Summarize per milestone.** Expire or compress state after each major step instead of carrying everything forever.

This connects straight to `production-and-observability` (cost + tracing) — the same discipline that keeps a long-running agent affordable.

---

## Framework cheat-sheet

| Framework | Best at | Mental model |
|---|---|---|
| **CrewAI** | Role-based teams, sequential/parallel crews | Agents + Tasks + Crew |
| **LangGraph** | Stateful graphs, routing, cycles, control flow | Nodes + shared state + edges |
| **AutoGen / AG2** | Conversational agents talking to each other | Chat between agents |
| **NeMo Agent Toolkit** (W16) | Config-driven production multi-agent on the NVIDIA stack; built-in eval/observability | Register tools → compose in YAML → observe (wraps LangGraph/LlamaIndex) — see `nemo-agent-toolkit` |
| **Google ADK** (W17) | Long-running agents that pause for days, durable sessions, in-process `sub_agents` | Durable state machine + `DatabaseSessionService` — see `long-running-and-distributed-agents` |
| **Anthropic SDK + asyncio** | Full control, custom orchestration, swarms | You write the loop |

There's no "best" — pick by the shape of your problem. Stuck choosing? See the `models-and-patterns` skill.

> 📁 Class workshop PDFs: `week09_multi_agent_systems/week9_2_multi_agent_theory_workshop.pdf` (BDI, FIPA, coordination theory), `week09_multi_agent_systems/week9_3_mas_coding_exercises.pdf`.

---

## Going to production — the Week 10 upgrades

Week 9 teaches the shapes. **Week 10 makes them survive a restart.** Three upgrades turn a demo into something you can run at 2 AM.

### Supervisor in LangGraph (the tool-calling pattern)

The current recommendation is to build the supervisor as a **tool-calling LLM** — each specialist becomes a `@tool`, and `create_react_agent` gives you the ReAct loop for free. Simpler than the `langgraph-supervisor` wrapper, and you keep full control of context.

```python
from langgraph.prebuilt import create_react_agent

@tool
def technical_specialist(ticket: str) -> str:
    """Handle TECHNICAL questions (APIs, devices, errors)."""
    return llm.invoke(f"You are a senior support engineer...\n\n{ticket}").content
# ...billing_specialist, general_specialist...

supervisor = create_react_agent(
    model=llm,
    tools=[technical_specialist, billing_specialist, general_specialist],
    prompt="Read the ticket, pick exactly ONE specialist tool, call it, then stop. "
           "Do not answer the ticket yourself.",
)
```

If you use the dedicated `create_supervisor()` helper instead, the key knob is **`output_mode`**: `full_history` streams every agent's messages (the audit trail you need for compliance / debugging) while `last_message` returns only the final answer (much cheaper). Choose by whether you must *explain* the reasoning later. **Nested two-level hierarchy** = compile a research-team supervisor and a writing-team supervisor as named subgraphs, then pass them as "agents" to a top-level supervisor.

> 📁 Class repo: `week10_production_observability/notebooks/02_supervisor.py` — the tool-calling supervisor routing tickets to 3 specialists.

### Checkpointers — time-travel debugging + HITL

`create_react_agent` (and any `StateGraph`) takes a **`checkpointer`**. A `SqliteSaver` writes state after every node, so the graph can be killed and resumed from any step — that's your time-travel debugger. Pair it with `interrupt()` to **pause before a risky action** (send an email, charge a card) and wait for a human:

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, interrupt
import sqlite3

memory = SqliteSaver(sqlite3.connect("checkpoints.db", check_same_thread=False))

def human_approval(state):
    decision = interrupt({"draft": state["draft_response"],
                          "question": "Approve sending? (approve/edit/reject)"})
    return {"approved": decision.get("action") == "approve"}

app = g.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "ticket-001"}}
app.invoke(initial, config=config)              # runs, then PAUSES at interrupt()
app.invoke(Command(resume={"action": "approve"}), config=config)  # resumes from disk
```

Kill the process between those two `invoke`s and it *still* resumes — state lives on disk, keyed by `thread_id`. The **`Send` API** does the same dynamic scatter-gather you hand-rolled with `asyncio.gather`, but inside the graph.

> 📁 Class repo: `week10_production_observability/notebooks/03_checkpointing.py` — `SqliteSaver` + `interrupt()` for HITL, with a kill-and-resume challenge.

### Durable execution — pick by blast radius

| Checkpointer | Survives | Use for |
|---|---|---|
| **MemorySaver** | nothing (RAM only) | demos, tests |
| **SqliteSaver** | process restart (one box) | local dev, single-node prod |
| **PostgresSaver** | multi-replica, concurrent threads | real production, shared state |
| **Temporal** (workflow engine) | infra failure, retries, weeks-long runs | long-horizon / mission-critical orchestration |

### CrewAI — two processes + production guardrails

| Process | Behaviour |
|---|---|
| `Process.sequential` | tasks run one after another, each output chaining forward (Week 9 ex1) |
| `Process.hierarchical` | injects a **manager** agent that delegates dynamically — CrewAI's Supervisor pattern |

> CrewAI's `Process` enum has **exactly these two** modes — there is no `Process.parallel`. To run independent tasks concurrently you don't switch the process, you set `async_execution=True` on the tasks (or call `crew.kickoff_async()`).

Hierarchical mode is convenient but **leaky**: the manager sometimes does the work itself, or delegates to everyone regardless of specialization. Production hardening: set `allow_delegation=False` on workers, cap `max_iter`, and add a **validator task** that checks the output before it ships. YAML config lets non-coders tweak prompts.

### Claude Agent SDK — multi-agent primitives

- **Subagents** — a parent spawns specialized children via the Task tool, each with isolated context and **restricted tools** (a code-analyzer gets `Read/Grep/Glob`; a reporter gets `Read/Write`). Constraint by design: **one level deep**, keeping the tree flat and predictable.
- **Agent Teams** (experimental) — subagents that message each other through a shared task list. Higher token cost, better quality on iteratively-refined problems.
- **Fork Context** (experimental) — a subagent inherits the parent's full history instead of starting fresh (`CLAUDE_CODE_FORK_SUBAGENT=1`). Great for "analyze the last 30 days" tasks that need the conversation so far.

**Mixing frameworks?** That's the real end-state — a LangGraph orchestrator delegating to a CrewAI nutrition team. The standard for that cross-framework hop is the `a2a-protocol` skill (Agent Cards + task delegation); MCP connects each agent to its *tools*, A2A connects *agents to agents*.

> 📁 Class repo: `week15_production_graphrag/smart_hotel_mas/README.md` — a 5-agent CrewAI crew (Sensor · Energy · Memory · Alert · Report) over a 4-layer memory stack; the in-process capstone.
> 📁 **Beyond one process:** `week17_long_running_distributed_agents/` stretches this MAS across *time* (long-running, pause-for-days, durable sessions) and *services* (A2A across teams/frameworks) — the `long-running-and-distributed-agents` and `a2a-protocol` skills. That's the L3 fleet layer above this one.

---

## Single agent vs. multi-agent — don't over-build
✅ **Go MAS** when sub-tasks are genuinely distinct, parallelizable, or need different expertise.
❌ **Stay single-agent** when one focused agent with a few tools already does the job — MAS adds latency, cost, and coordination bugs. Most problems don't need a swarm.

---

## 🧪 Guided lab (offer this)

### Warm-up (5–10 min, pass/fail)
Given three problems, name the right pattern out loud and justify in one line:
1. "Route a support ticket to billing / technical / general." → **Router**
2. "User asks an open-ended fitness question that might need research *or* a 12-week plan *or* both — you don't know up front." → **Supervisor-Worker** (subtasks unknown → coordinator must replan).
3. "Audit a building for energy, comfort, safety, cost — all independent." → **Parallel swarm**.
**Pass** = correct pattern *and* a one-sentence reason for all three.

### Skill Drill (15–30 min, runnable, $0 / no API key)
Build a **supervisor with a checkpointer and a human-in-the-loop interrupt** — the Week 10 core — using a `MockLLM` so it runs free and offline. Then *kill the process mid-run and resume*.

```python
# pip install langgraph   (no API key, no network)
import sqlite3
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, interrupt

class MockLLM:
    """$0 stand-in: deterministic 'routing' + 'drafting', zero tokens."""
    def route(self, text: str) -> str:
        t = text.lower()
        if "charge" in t or "invoice" in t: return "billing"
        if "error" in t or "load" in t:     return "technical"
        return "general"
    def draft(self, route: str, text: str) -> str:
        return f"[{route} specialist] Thanks for reaching out — we're on it."

llm = MockLLM()

class S(TypedDict):
    ticket: str
    route: str | None
    draft: str | None
    approved: bool | None
    sent: str | None

def supervise(s): return {"route": llm.route(s["ticket"])}
def specialist(s): return {"draft": llm.draft(s["route"], s["ticket"])}
def human(s):
    d = interrupt({"draft": s["draft"], "q": "approve / reject?"})
    return {"approved": d.get("action") == "approve"}
def send(s):
    return {"sent": s["draft"] if s.get("approved") else "[REJECTED — not sent]"}

g = StateGraph(S)
for name, fn in [("supervise", supervise), ("specialist", specialist),
                 ("human", human), ("send", send)]:
    g.add_node(name, fn)
g.add_edge(START, "supervise"); g.add_edge("supervise", "specialist")
g.add_edge("specialist", "human"); g.add_edge("human", "send"); g.add_edge("send", END)

memory = SqliteSaver(sqlite3.connect("lab.db", check_same_thread=False))
app = g.compile(checkpointer=memory)
cfg = {"configurable": {"thread_id": "t1"}}

app.invoke({"ticket": "I was charged twice on my invoice.",
            "route": None, "draft": None, "approved": None, "sent": None}, cfg)
print("PAUSED — state is on disk in lab.db")          # <-- kill the process here, restart, then:
out = app.invoke(Command(resume={"action": "approve"}), cfg)
print(out["route"], "->", out["sent"])
```

**Extend it:** (a) add a 4th `escalation_specialist` route for "cancel my account"; (b) switch the interrupt to `{"action": "reject"}` and prove `send` emits `[REJECTED — not sent]`; (c) swap `SqliteSaver` → `MemorySaver` and show state does *not* survive a restart.

### Weighted evaluation criteria
| # | Criterion | Weight |
|---|---|---|
| 1 | Supervisor routes all three sample tickets to the correct specialist | 20% |
| 2 | Graph **pauses** at `interrupt()` and only sends after resume | 25% |
| 3 | **Kill-and-resume works** — restart the process, resume from `lab.db`, output completes | 25% |
| 4 | Reject path emits `[REJECTED — not sent]`; `MemorySaver` loses state on restart | 15% |
| 5 | Learner names the topology (Star) + state model (shared-state) and the over-build check (could 1 agent do it?) | 15% |

**Pass = 4 of 5 criteria** (criteria 2 and 3 are mandatory — durability is the whole point of Week 10).

### Stretch — the 14 drills
For more reps, point them at `week11_models_and_patterns/exercises/README.md`: **14 graded MAS exercises** (Beginner → Expert) spanning 2-agent dialogue, sequential pipelines, LangGraph routers/cycles, parallel fact-checkers, AutoGen group chat, self-healing graphs, negotiation, and a hotel-ops command center. Treat them as a kata ladder — see the `agent-drills` skill for how to structure timed reps.

End on the judgment call: "the pattern is the design decision — frameworks are just how you spell it. And a supervisor that can't survive a restart is a demo, not a system."
