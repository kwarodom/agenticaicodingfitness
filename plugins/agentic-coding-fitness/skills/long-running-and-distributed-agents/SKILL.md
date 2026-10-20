---
name: long-running-and-distributed-agents
description: "Teach agents that survive real time and service boundaries — the Week 17 layer above an in-process multi-agent system. Two halves: (1) LONG-RUNNING agents that pause for days and resume without losing context, built on Google ADK's durable state machine (current_step in persisted session state, NOT chat history), DatabaseSessionService persistence, before_agent_callback init, and event-driven resume via webhooks + Runner.run_async(state_delta=…); plus the credential problem auth.md solves — store the durable GRANT, not the short-lived token, and re-mint a scoped credential every time the agent wakes. (2) DISTRIBUTED agents that delegate across teams/frameworks via A2A (in-process sub_agents vs. cross-service Agent Cards). Use when someone asks 'how do I build an agent that pauses for days and resumes?', 'how does an agent keep credentials valid while it sleeps?', mentions Google ADK / long-running agents / durable sessions / DatabaseSessionService / state_delta / webhooks-resume / auth.md / pause-and-resume / human-in-the-loop across days, or is reviewing Week 17."
when_to_use: "Learner is outgrowing a single in-process agent run and needs it to survive days of idle time and process restarts (pause/resume, durable state machine, ADK DatabaseSessionService, webhook-driven resume), keep credentials valid while it sleeps (auth.md durable grant + re-mint), or delegate to agents owned by other teams/frameworks (sub_agents vs A2A). Also for catching up on Week 17."
---

# Long-Running & Distributed Agents — Surviving Time and Boundaries (Week 17)

> **The one idea:** A chatbot remembers by *replaying its whole conversation*. A long-running agent **can't** — it may sleep for DAYS between messages. So progress lives in an **explicit, durable state machine** (`current_step`), persisted on every tool call and injected into the prompt — *not* in chat history. That one shift is what lets an agent pause for a long weekend, survive a container restart, and wake up correct.

This skill is the **L3 fleet layer** above `multi-agent-systems`. Week 9/15 ran a team of agents in *one process, in real time*. Week 17 stretches that across **calendar time** (long-running) and **service boundaries** (distributed).

```
        Week 15: 5 agents, ONE process, runs while alive
                          │
            ┌─────────────┴──────────────┐
   LONG-RUNNING (survives days)   DISTRIBUTED (survives org/framework boundaries)
   durable state + ADK sessions          A2A agent cards + delegation
   + auth.md (credentials that              (see also: a2a-protocol)
     outlive a single token)
```

> ✅ **Grounded & runnable.** Built on `week17_long_running_distributed_agents/` — the `checkpoints/` run **offline** (no key, no network; they construct the real ADK agents and mock only the LLM call), and `week17_long_running_distributed_agents/hr_onboarding/` + `week17_long_running_distributed_agents/authmd_adk/` are **live FastAPI services** you can drive end-to-end. Based on Google's *"Build long-running AI agents that pause, resume, and never lose context with ADK"* and WorkOS's `auth.md` protocol.

---

## Part A — Why stateless chatbots break over days

A long-running workflow (onboard a hire over weeks; run a maintenance work-order that waits days for a vendor part) cannot lean on conversation history. Three failure modes (from `week17_long_running_distributed_agents/REFERENCE.md`):

| Failure mode | What goes wrong if state = chat history |
|---|---|
| **Context pollution** | Days of unrelated messages bury the one fact that drives the next step. |
| **Token-cost explosion** | Replaying a 3-week transcript every wake-up costs more each time — unbounded. |
| **Reasoning hallucination** | The model re-derives "where am I?" from noise and guesses wrong. |

**The cure — a durable state machine.** Behaviour is driven by an explicit `current_step` that is *persisted* and *interpolated into the instruction*, never re-derived from history:

```python
class OnboardingStep(str, Enum):          # week17_long_running_distributed_agents/hr_onboarding/onboarding_steps.py
    START = "START"
    WELCOME_SENT = "WELCOME_SENT"          # ⏸ paused — waiting on a signed contract
    DOCUMENTS_SIGNED = "DOCUMENTS_SIGNED"
    IT_PROVISIONED = "IT_PROVISIONED"      # ⏸ paused — waiting on laptop delivery
    HARDWARE_DELIVERED = "HARDWARE_DELIVERED"
    COMPLETED = "COMPLETED"
```

The ⏸ states are **pause gates**: the agent must *refuse to advance* through them until an external signal arrives. That refusal is the whole point — a long-running agent that "helpfully" skips the approval gate is a bug.

---

## Part B — The four ADK building blocks (the long-running engine)

Google ADK gives you exactly four primitives. Memorize the mapping — everything in `week17_long_running_distributed_agents/checkpoints/` is these four:

| Need | ADK primitive | Where you see it |
|---|---|---|
| Atomic checkpoint on **every** tool call | `ToolContext.state` (a dict you mutate) | CP1 tools |
| State exists **before the first turn** | `before_agent_callback` | CP1 `initialize_work_order` |
| Durable persistence (local SQLite ↔ prod Cloud SQL) | `DatabaseSessionService` | CP2 |
| **Event-driven resume** with a pre-inference state patch | `Runner.run_async(state_delta=…)` | CP3 |
| Focused in-process delegation | `sub_agents=[…]` | CP4 |

```python
# CP1 — a tool that checkpoints by mutating state (no DB call in your code)
def advance_step(tool_context: ToolContext) -> dict:
    tool_context.state["current_step"] = "DOCUMENTS_SIGNED"   # persisted atomically
    return {"status": "ok", "step": tool_context.state["current_step"]}

# The agent's instruction is INTERPOLATED from state, not from history:
instruction = f"You are onboarding a hire. Current step: {state['current_step']}. " \
              f"Do ONLY what this step allows; refuse to skip ahead."
```

**Persistence is one line of config.** `DatabaseSessionService(db_url="sqlite+aiosqlite:///sessions.db")` for dev; swap the URL to Cloud SQL for prod and *nothing else changes*. The "3-day pause" is just a row sitting in a table — scale-to-zero safe.

**Resume is a push, not a poll.** A webhook applies a `state_delta` *before* the next inference, so the model wakes already knowing what changed:

```python
# CP3 — an external event resumes the agent
@app.post("/webhooks/document_signed")
async def document_signed():
    async for _ in runner.run_async(
        user_id=uid, session_id=sid,
        state_delta={"current_step": "DOCUMENTS_SIGNED"},   # patch state, THEN think
        new_message=types.Content(parts=[types.Part(text="Contract signed — continue.")]),
    ):
        ...
```

> 📁 `week17_long_running_distributed_agents/checkpoints/checkpoint1_state_machine.py` … `checkpoint3_webhook_resume.py` — durable state → restart-survival → webhook resume, each offline and self-asserting.
> 📁 `week17_long_running_distributed_agents/hr_onboarding/server.py` — the live version: `/onboard`, `/chat`, `/webhooks/document_signed`, `/webhooks/hardware_delivered`, `/status`, plus a web visualizer.

---

## Part C — Credentials that outlive a single token (auth.md)

Durable *state* is only half of "survive for days." The other half is durable *access*. **auth.md tokens are short-lived by design; your agent may sleep for days.** A token minted on Monday is dead by the time the agent wakes on Thursday.

> **The rule:** store the **durable grant, not the token**. Re-mint a fresh, scoped, short-lived credential **every time the agent wakes**.

```python
# week17_long_running_distributed_agents/authmd_adk/authmd_client.py — the seam between the two halves
grant = AuthGrant(...)                 # durable, token-FREE; THIS is what lives in session state
token = await AuthMdClient().acquire(grant, scope="sites.read")   # minted fresh at wake time
```

This composes existing OAuth RFCs (RFC 9728 discovery; no new crypto). The teaching points from `week17_long_running_distributed_agents/authmd_adk/`:

- **Store the grant in session state** (`auth_grants`), alongside `current_step` — both are durable, neither is a secret that expires.
- **Re-mint per wake** via a `before_tool_callback` (`AuthInjector`) that calls `acquire()` and injects a fresh token just-in-time.
- **Least privilege per sub-agent.** The read sub-agent gets a `sites.read` token; it is *refused 403* if it tries a write. The apply sub-agent gets `control.write` only after a human approves.
- **The approval gate IS the user-claimed flow.** The agent parks at `ANALYZED`, a webhook emails an OTP, the OTP completes the claim, and only then is the write-scoped token issued.
- **Revocable.** A `/revoke` + logout token kills access immediately; the next `acquire()` fails closed.

> ⚠️ **Honest framing (from the deck's reality-check):** the **ADK foundation is solid and production-grade**; **auth.md is real but early** (a WorkOS-authored open protocol in early access — adoption is thin, the spec may move). Design around the **generic auth-grant abstraction** (`AuthGrant` / `acquire()`) so you use auth.md where a service publishes one and fall back to a plain OAuth refresh exchange everywhere else.
> 📁 `week17_long_running_distributed_agents/authmd_adk/run_authmd_demo.py` — the entire protocol offline (discovery → verified token → **wake-time re-mint** → OTP gate → least-privilege 403 → revoke → audit log). `run_full_demo.py` — the same gate driven by a live ADK agent.

---

## Part D — Distributed: in-process sub-agents vs. A2A across services

Once one agent needs *another* agent, you have two tools — and picking wrong is the classic over-build.

| You own all the code? | Reach for | Why |
|---|---|---|
| **Yes** — same import graph, same deploy | ADK `sub_agents=[…]` (CP4) | A function call's cost, focused context, no network, no protocol |
| **No** — other team, other framework, other server | **A2A** (CP5–6) | Agent Card discovery + a task lifecycle across a boundary you can't import |

A2A is the *horizontal* counterpart to MCP's *vertical* (MCP = agent→tools; A2A = agent→agent). The mechanics — Agent Cards at `/.well-known/agent.json`, the six-state task lifecycle (`submitted → working → completed`, plus `input-required` for network HITL, `failed`, `canceled`), and polling/SSE/webhook follow patterns — are the **`a2a-protocol`** skill; read it for the deep dive. The capstone wires both together:

```
MaintenanceCoordinator (ADK · durable session, pauses for days)
   ├─ in-process sub_agent ──▶ EnergyAgent ──MCP──▶ occupancy / energy data
   └─ A2A across services  ──▶ Acme HVAC Parts Agent  (another org, other framework)
                                 └─ input-required ──▶ resumed via webhook (Part B)
```

> **Climb to A2A only when the team must span services or organizations.** Below that bar, an in-process `sub_agent` is the right, cheaper tool. Week 17 is about recognizing the boundary and crossing it *cleanly*.
> 📁 `week17_long_running_distributed_agents/checkpoints/checkpoint4_sub_agents.py` (the wall: a vendor you can't import) → `checkpoint5_a2a_cards.py` (cards + lifecycle) → `checkpoint6_fleet.py` (long-running + distributed, end-to-end).

---

## Where this sits in the course

```
agent-loops (W4–5) → tools (W3) → MCP (W7) → MAS in one process (W9, W15)
                                                        │
                                              ┌─────────┴──────────┐
                                   long-running (durable)   distributed (A2A)
                                              └─────────┬──────────┘
                                                   Week 17 — fleet
```

Prerequisites worth having first: `multi-agent-systems` (the in-process baseline you're stretching), `mcp-and-skills` (the vertical half), and `production-and-observability` (checkpointing/HITL — ADK's `DatabaseSessionService` is the same durability instinct as LangGraph's `SqliteSaver`, one layer up).

---

## 🧪 Guided lab (offer this): *a durable agent that pauses, dies, and resumes*

**You'll need:** nothing — no API key, no install. A `MockLLM` + a JSON file stand in for ADK's `DatabaseSessionService`, so the durable state machine runs at **$0** and you can literally kill the process between steps.

### Warm-up (5–10 min, pass/fail)
Answer out loud:
1. Why can't a 3-week onboarding agent rely on conversation history? (Name **one** of the three failure modes.)
2. Which ADK primitive persists state on every tool call, and which one resumes the agent from an external event?
3. Your agent sleeps Mon→Thu. The token it minted Monday is dead Thursday. What do you store in session state instead, and when do you mint the token?
> ✅ **Pass** = all three correct (history→pollution/cost/hallucination; `ToolContext.state` + `Runner.run_async(state_delta=…)`; store the **grant**, mint the token **at wake time**).

### Skill Drill (15–30 min, runnable, $0)
Build a durable state machine that **survives a process restart** and **resumes from a webhook-style event** — the heart of CP1–CP3 with no dependencies.

```python
# durable_agent_drill.py — runs with plain `python`, no key, $0.
# Models ADK's DatabaseSessionService with a JSON file so you can kill & resume.
import json, os, sys

STATE_FILE = "session.json"
STEPS = ["START", "WELCOME_SENT", "DOCUMENTS_SIGNED", "IT_PROVISIONED",
         "HARDWARE_DELIVERED", "COMPLETED"]
PAUSE_GATES = {"WELCOME_SENT", "IT_PROVISIONED"}   # ⏸ need an external signal to pass

def load():                                  # ← DatabaseSessionService.get_session()
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {"current_step": "START", "grant": "durable-grant-abc"}  # grant, NOT a token

def save(state):                             # ← ToolContext.state checkpoint (atomic-ish)
    json.dump(state, open(STATE_FILE, "w"))

class MockLLM:
    """$0 stand-in. A long-running agent acts ONLY on current_step — never history."""
    def act(self, state):
        step = state["current_step"]
        token = f"token-for({state['grant']})@{step}"   # re-MINTED every wake, scoped to step
        if step in PAUSE_GATES:
            return f"⏸ paused at {step}: refusing to advance until an external event arrives. ({token})"
        nxt = STEPS[STEPS.index(step) + 1] if step != "COMPLETED" else "COMPLETED"
        return f"▶ at {step}: minted {token}; advancing → {nxt}"

def resume(event_delta=None):                # ← Runner.run_async(state_delta=…)
    state = load()
    if event_delta:                          # webhook patches state BEFORE the agent thinks
        state["current_step"] = event_delta
    print(MockLLM().act(state))
    # advance only if not gated
    if state["current_step"] not in PAUSE_GATES and state["current_step"] != "COMPLETED":
        state["current_step"] = STEPS[STEPS.index(state["current_step"]) + 1]
    save(state)
    return state

if __name__ == "__main__":
    # `python durable_agent_drill.py`            → advance one step (or report a pause)
    # `python durable_agent_drill.py sign`       → fire the "document_signed" webhook
    # `python durable_agent_drill.py deliver`    → fire the "hardware_delivered" webhook
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    delta = {"sign": "DOCUMENTS_SIGNED", "deliver": "HARDWARE_DELIVERED"}.get(arg)
    s = resume(delta)
    print(f"   state on disk: current_step={s['current_step']}  (grant kept, token discarded)")
```

Run the sequence — **kill the process between every line** (that's the point; state lives on disk):
```
python durable_agent_drill.py            # START → WELCOME_SENT, then ⏸ pauses
python durable_agent_drill.py sign       # webhook opens the gate → DOCUMENTS_SIGNED → IT_PROVISIONED ⏸
python durable_agent_drill.py deliver    # webhook opens the gate → HARDWARE_DELIVERED → COMPLETED
```

**Extend it (pick any two):** (a) make the agent *fail closed* if `grant` is missing; (b) add a `sites.read` vs `control.write` scope and refuse a write before the approval gate; (c) prove durability — delete the `resume` call's `save()` and watch it never progress across restarts.

### Weighted evaluation criteria (pass = **4 / 5**)
| # | Criterion | Weight |
|---|---|---|
| 1 | State **survives a process restart** — you killed it between steps and it resumed from `session.json` | ●●● |
| 2 | The agent **refuses to advance** through a ⏸ pause gate until a webhook event arrives | ●●● |
| 3 | A fresh token is **re-minted from the durable grant at each wake** (never stored across restarts) | ●● |
| 4 | A webhook event patches state **before** the agent acts (push, not poll) | ●● |
| 5 | Learner can state, in one sentence each: in-process `sub_agent` vs A2A, and ADK vs auth.md (state vs credentials) | ● |

**Pass threshold: 4 of 5.** Criteria 1 and 2 are non-negotiable — an agent that loses state on restart, or barrels through an approval gate, is a demo, not a long-running system. Close on the boundary call: *"durable state (ADK) keeps the agent correct across days; durable grants (auth.md) keep it authorized; sub-agents handle teammates you own and A2A handles the ones you don't."*
