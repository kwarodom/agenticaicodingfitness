---
name: production-and-observability
description: "Turn a working agent prototype into a production system you can SEE, STOP, and AFFORD — tracing every step (LangSmith, OpenTelemetry gen_ai.* conventions), human-in-the-loop approval gates with interrupt() + durable SqliteSaver checkpointing, progressive autonomy, OWASP-ASI guardrails, and cost governance (per-run/per-tenant budgets, circuit breakers, CI cost-regression gates). Use when someone asks 'how do I ship this agent?', mentions observability/tracing/LangSmith/OTel, human approval, checkpointing/resume/time-travel, runaway cost, guardrails, or is reviewing Week 10."
when_to_use: "Learner is moving an agent from prototype to production and needs to observe it (tracing), control it (human-in-the-loop + checkpointing + guardrails), or budget it (cost governance) — or is catching up on Week 10."
---

# Production & Observability — See it, Stop it, Afford it (Week 10)

> **The one idea:** A prototype that works *once* is not a production agent. Production means three things you can do to a running system: **see it** (every step is traced), **stop it** (a human can approve or halt risky actions, and state survives a crash), and **afford it** (cost has a ceiling, not a surprise). Gartner predicts 40%+ of agentic projects get cancelled by 2027 — almost always because one of these three was missing.

```
PROTOTYPE → ✅ SEE IT  (trace every step)
           → ✅ STOP IT (approve / pause / resume)
           → ✅ AFFORD IT (budget + circuit breaker)  → PRODUCTION
```

> 📁 Class repo: `week10_production_observability/README.md` — a 5-notebook Support Ticket Routing System that builds exactly this: supervisor → HITL → checkpointing → tracing → SDK hybrid.

---

## Part A — See it (Observability)

**You cannot fix what you cannot see.** Unlike a normal service that fails with a stack trace, an agent can produce a *plausible wrong answer*, silently loop on a tool, or drift in reasoning — without throwing a single exception. Tracing is how you make those failures visible.

### What to log on every step

| Capture | Why it matters |
|---|---|
| **Inputs / prompt** | Reproduce the exact call that misbehaved |
| **Tool calls + args** | Catch silent loops and wrong-tool picks |
| **Output + finish reason** | See *why* it stopped (length? tool? stop word?) |
| **Tokens (in/out)** | Feeds cost and context-bloat alarms |
| **Latency per step** | Find the slow node in a multi-step graph |
| **Cost per step** | Roll up to cost-per-task (Part C) |

### LangSmith — the zero-friction option (grounded)

For LangChain/LangGraph teams, tracing is a couple of env vars. Set them **before** importing the modules you want traced:

```python
import os
# Only flip tracing ON when a key is present, else the uploader 401s on every run.
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "w10-hands-on"
```

That's it — every node's inputs, outputs, latency, token count, and state diff now show up in the LangSmith UI, no per-call wiring. The Week 10 lab uses this to diff a **weak** prompt against a **strong** one and *see* which tickets flipped category.

> 📁 Class repo: `week10_production_observability/notebooks/04_langsmith.py` — turn tracing on, run two prompts, compare runs. Solution `week10_production_observability/solutions/04_langsmith_solution.py` adds a 20-ticket eval set and an accuracy report (links to `agent-evaluation`).

### OpenTelemetry GenAI conventions — the vendor-neutral standard

> ⚠️ **Conceptual** — not in the Week 10 repo. This is the industry standard you'll meet once you outgrow a single vendor.

If you don't want to be married to one tracing vendor, instrument with **OpenTelemetry's GenAI semantic conventions**: a standard set of `gen_ai.*` span attributes so *any* backend can read your traces.

| Attribute | What it holds |
|---|---|
| `gen_ai.operation.name` | chat / tool / embedding / agent invocation |
| `gen_ai.request.model` | which model the call used |
| `gen_ai.usage.input_tokens` | prompt tokens (→ cost) |
| `gen_ai.usage.output_tokens` | completion tokens (→ cost) |
| `gen_ai.response.finish_reasons` | why it stopped |

Auto-instrumentation now covers 40+ frameworks (LangChain, CrewAI, LlamaIndex, OpenAI Agents SDK), so one layer spans a mixed stack. Teams with real observability report ~2.2x better reliability than log-only teams.

**OSS backend:** **Arize Phoenix** is the popular open-source, OTel-native option — a single Docker container that ingests spans over OTLP. (Also conceptual here — not in the repo, but worth knowing as the free self-hosted alternative to LangSmith's cloud-only model.)

---

## Part B — Stop it (Human-in-the-loop + durable state)

A production agent will eventually try to do something irreversible — send the email, charge the card, delete the row. **Stop it** is two capabilities: a **gate** a human can hold (`interrupt()`), and **durable state** so the agent can wait hours for that human and survive a restart (`SqliteSaver`).

### `interrupt()` — pause for human approval (grounded)

`interrupt()` freezes the graph mid-run and hands a payload to a human. The graph stays paused (and persisted) until you resume it with the human's decision.

```python
from langgraph.types import Command, interrupt

def human_approval(state):
    decision = interrupt({                       # graph PAUSES here
        "action": "approve_reply",
        "draft": state["draft_response"],
        "question": "Approve sending this reply? (approve / edit / reject)",
    })
    if isinstance(decision, dict) and decision.get("action") == "approve":
        return {"approved": True}
    if isinstance(decision, dict) and decision.get("action") == "edit":
        return {"draft_response": decision["text"], "approved": True}
    return {"approved": False}

# ... later, after a human reviews ...
app.invoke(Command(resume={"action": "approve"}), config=config)   # RESUMES from the pause
```

### `SqliteSaver` — checkpoint so it survives restarts (grounded)

A checkpointer writes the full graph state to disk **after every node**. That gives you three superpowers: **resume** after a crash, **wait** for a slow human without holding memory, and **time-travel** (inspect or replay any prior checkpoint).

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)              # state persisted on disk, not just RAM
app = g.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "ticket-demo-001"}}
app.invoke(initial, config=config)      # runs, then pauses at interrupt() — state is on disk

# Inspect the history (time-travel):
for c in memory.list(config):
    print(c.metadata.get("step"))
```

Kill the Python process after the interrupt fires, restart, re-invoke with the same `thread_id` — the graph resumes from the checkpoint. That is the difference between an in-memory toy and something you can run in production.

> 📁 Class repo: `week10_production_observability/notebooks/03_checkpointing.py` — draft → `human_approval` (interrupt) → send, persisted to `checkpoints.db`. Solution `week10_production_observability/solutions/03_checkpointing_solution.py` adds the `edit` path, history inspection, and a restart-survival demo.

> 💡 In-memory checkpointers lose everything on restart; `SqliteSaver` survives a single host. For workflows that span services or wait *days*, a durable-execution engine (Temporal, Postgres-backed savers) is the next rung — same idea, bigger blast radius.

### Progressive autonomy — agents earn trust

Don't hand a new agent the keys. Start it read-only and promote it only as its error rate stays low (e.g. below 5%):

| Level | What the agent may do |
|---|---|
| **DRAFT_ONLY** | Propose only. A human does everything. |
| **SUPERVISED** | Acts, but every *risky* action hits an approval gate. |
| **MONITORED** | Acts freely; humans watch traces and can intervene. |
| **FULL** | Autonomous within hard guardrails. |

The gate fires on a **risk score**, not on everything — cheap, reversible actions sail through; `delete` / irreversible / high-impact actions interrupt for a human:

```python
def risk_score(action: dict) -> float:
    risk = 0.0
    if action.get("type") == "delete":   risk += 0.8
    if action.get("type") == "write":    risk += 0.3
    if action.get("irreversible"):       risk += 0.5
    return min(risk, 1.0)
# SUPERVISED level: if risk > 0.5 → interrupt() for approval, else just execute.
```

### The OWASP-ASI 4-layer guardrail stack — the safety net

> ⚠️ **Conceptual** — not in the Week 10 repo. The repo gives you the *gate* (`interrupt()`); the full stack is the surrounding armor you build for real deployments.

The OWASP Top 10 for Agentic Applications (ASI, Dec 2025) names risks unique to autonomy — Goal Hijack, Tool Misuse, Privilege Abuse, Rogue Agents. The deployable answer is **defense in depth**, four layers each with one job:

| Layer | Job | Example |
|---|---|---|
| **1 — Edge / Network** | Drop obvious junk cheaply | rate limit, bot/IP blocks, regex reject |
| **2 — Input validation** | Inspect the request | PII detection, prompt-injection classifier |
| **3 — Reasoning / runtime** | Constrain *actions* | tool allow-lists, arg-schema checks, sandbox, **HITL approval** ← `interrupt()` lives here |
| **4 — Output / egress** | Filter what leaves | response redaction, hallucination check, streaming cutoff |

Key insight: an **intent gate** at Layer 3 (e.g. step-up MFA on a separate channel) is *architecturally outside* the agent's reasoning path — so **no prompt injection can talk its way past it**.

---

## Part C — Afford it (Cost governance)

Agents make **3–10x more LLM calls** than a chatbot for the same task, and an unconstrained agent on one coding task can burn **$5–8** in API fees. 96% of enterprises report costs *over* projection. Cost is not a post-launch cleanup — it's a design constraint, like latency.

### Three guardrails that stop the bleeding

| Guardrail | Rule |
|---|---|
| **Per-run cap** | Max iterations / max tokens per task. Halt, don't spiral. |
| **Per-tenant budget** | Daily cap per customer — **alert at 80%, auto-pause at 100%**. |
| **Circuit breaker** | A monitor that *cuts off* spend the moment a threshold breaches. |

The per-tenant cap is your defense against the **"denial of wallet"** attack — adversarial or buggy input that triggers unlimited expensive inference. A tiny circuit breaker:

```python
class BudgetExceeded(Exception): ...

class Budget:
    def __init__(self, limit_usd: float):
        self.limit, self.spent = limit_usd, 0.0
    def charge(self, usd: float):
        self.spent += usd
        if self.spent >= self.limit:               # pause-at-100%
            raise BudgetExceeded(f"${self.spent:.2f} ≥ ${self.limit:.2f}")
        if self.spent >= 0.8 * self.limit:          # alert-at-80%
            print(f"⚠️  budget at {self.spent/self.limit:.0%}")
```

### The metric that matters: cost per *successful* task

Don't optimize cost-per-token — optimize **cost per successful task completion**. An agent that uses *more* tokens but finishes in *fewer steps* can be cheaper per outcome. Measure it from day one.

### Put it in CI — a cost-regression gate

Run your golden eval set on every change and **block the deploy** if cost-per-task jumps more than a threshold (e.g. **+15%**):

```python
# Pseudocode for a CI gate (pairs with agent-evaluation's golden dataset)
baseline = 0.012  # $/task on main
current  = run_eval_and_measure_cost_per_task()
if current > baseline * 1.15:
    raise SystemExit(f"COST REGRESSION: ${current:.4f}/task > +15% of ${baseline:.4f}")
```

> 🔗 The *how* of getting cost down — 3-tier model routing, prompt/semantic caching, context compaction, batch API — lives in `models-and-patterns` (the full 70–85% savings cascade). This skill is about putting a **ceiling and an alarm** on whatever it costs; that skill is about lowering the number. Don't duplicate the table — go read it there.

---

## How the three fit together

```
        ┌─ SEE IT  → trace every step (LangSmith / OTel gen_ai.*)
agent ──┼─ STOP IT → interrupt() gate + SqliteSaver durability + guardrail stack
        └─ AFFORD IT → per-tenant budget + circuit breaker + CI cost gate
```

`agent-loops` builds the loop; `multi-agent-systems` builds the team; **this skill makes either one safe to ship**; `agent-evaluation` proves it still works after every change.

---

## 🧪 Guided lab (offer this)

Take an agent loop you already have and give it the three production powers — **at $0, no API key** (a `MockLLM` stub stands in for the model).

### Warm-up (5–10 min, pass/fail)
Wrap a single agent step in a **print-trace** decorator that logs, on every call: the **input**, the **tool/output**, **token count** (estimate `len(text)//4`), and **latency** (`time.perf_counter()` delta).
- **Pass = ** running one step prints all four fields. Fail = any field missing.

### Skill Drill (15–30 min, runnable at $0)
Add **Stop it** and **Afford it** to a `MockLLM` agent loop:

```python
import time

class MockLLM:
    """$0 stand-in. Returns a canned 'action' so the lab runs with no API key."""
    def __init__(self): self.calls = 0
    def step(self, task: str) -> dict:
        self.calls += 1
        # pretend the agent wants to delete something on call 2
        return {"type": "delete", "target": "row-42"} if self.calls == 2 \
               else {"type": "read", "target": "row-42"}

class BudgetExceeded(Exception): ...

def risk_score(a: dict) -> float:
    return 0.8 if a.get("type") == "delete" else 0.0

def approve(action: dict) -> bool:
    # In a real app this is interrupt() + a human. Here: auto-reject deletes.
    return action["type"] != "delete"

def run(task: str, budget_usd=0.10, cost_per_call=0.03, max_iters=5):
    llm, spent, warned = MockLLM(), 0.0, False
    for i in range(max_iters):
        action = llm.step(task)
        spent += cost_per_call
        if not warned and spent >= 0.8 * budget_usd:  # 80% alert — checked BEFORE the breaker
            print(f"⚠️  budget at {spent/budget_usd:.0%}")
            warned = True
        if spent >= budget_usd:                       # circuit breaker (hard stop)
            raise BudgetExceeded(f"${spent:.2f} ≥ ${budget_usd:.2f}")
        if risk_score(action) > 0.5:                  # HITL gate
            if not approve(action):
                print(f"step {i}: BLOCKED risky action {action}")
                continue
        print(f"step {i}: ran {action}  (spent ${spent:.2f})")

run("clean up old rows")
```

**Make it pass:**
1. The risky `delete` on call 2 is **blocked** by the approval gate (printed, not executed).
2. The loop **trips the circuit breaker** (raise `BudgetExceeded`) at step 3 — before `max_iters` — because 5 × `cost_per_call` would overshoot `budget_usd`.
3. The **80% alert** prints once (at step 2, `budget at 90%`) *before* the breaker fires — it's checked ahead of the hard stop so the warning is never skipped.
4. Add a **trace line** per step (from the warm-up) so every step shows input + action + cost.
5. Add a **per-run iteration cap** (`max_iters`) and prove the loop can't run forever.

### Weighted evaluation criteria

| # | Criterion | Weight |
|---|---|---|
| 1 | Approval gate blocks the risky `delete` (not executed) | 25% |
| 2 | Circuit breaker raises `BudgetExceeded` at the cap | 25% |
| 3 | 80%-budget alert fires before the breaker | 15% |
| 4 | Per-step trace logs input + action + cost | 20% |
| 5 | Iteration cap proves the loop terminates | 15% |

**Pass = 4 / 5 criteria.** (Criteria 1 and 2 are non-negotiable — a "production" agent that can neither be *stopped* nor *capped* fails regardless of the rest.)

**Stretch:** swap the auto-reject `approve()` for a real LangGraph `interrupt()` + `SqliteSaver` using `week10_production_observability/notebooks/03_checkpointing.py` as the template, then kill and restart the process to prove the run resumes from disk.
