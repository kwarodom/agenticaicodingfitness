# 🏨 Week 17 — Long-Running Agents & Fleet Orchestration (Google ADK + A2A)

Runnable Python that takes the **week15 smart-hotel MAS** — 5 agents in *one*
CrewAI process — and stretches it across **real time** and **service boundaries**:

- **Long-running** — a maintenance work-order that **pauses for days** waiting on
  a vendor part and **resumes without losing context**, built with Google ADK's
  durable-state-machine pattern.
- **Distributed** — agents owned by *different teams on different frameworks*
  discover and delegate to each other over the **A2A protocol** (the
  agent-to-agent counterpart to MCP).

Based on Google's [*Build long-running AI agents that pause, resume, and never
lose context with ADK*](https://developers.googleblog.com/build-long-running-ai-agents-that-pause-resume-and-never-lose-context-with-adk/),
adapted to this course's hotel domain, with agent communication via **A2A**.

> **Every checkpoint runs OFFLINE — no API key, no network.** They are
> *construction + simulation* demos: they build the real ADK agents and walk the
> real A2A lifecycle, mocking only the LLM call. Install `google-adk` / `a2a-sdk`
> to light up the live wiring; you don't need them to learn the patterns.

```
The two problems week15 left open, and what solves each:

  Single process, lives only while running   →  Long-running: durable state
  ───────────────────────────────────────       machine + DatabaseSessionService
                                                 + event-driven resume (ADK)

  All 5 agents must be one team's code        →  Distributed: Agent Cards +
  in one import graph                            task lifecycle delegation (A2A)
```

---

## The mental model

```
                MCP  =  VERTICAL  (an agent → its own tools/data)
                A2A  =  HORIZONTAL (an agent → another agent, across services)

  MaintenanceCoordinator   (ADK · durable session state, pauses for days)
     │
     ├─ in-process sub_agent ──▶ EnergyAgent ──MCP──▶ occupancy / energy data
     │
     └─ A2A across services  ──▶ Acme HVAC Parts Agent   (another org, other framework)
                                    └─ input-required ──▶ resumed via webhook
```

- **MCP** (Week 7) wired an agent *down* to tools. **A2A** (this week) wires an
  agent *across* to peer agents you don't own and never imported.
- The work-order is the **durable state machine**: behaviour is driven by
  `current_step` in persisted session state, **never** by replaying chat history.
  That is what lets it pause for a long weekend and wake up correct.

---

## Layout

```
week17/
├── config.py                       → WorkOrderStep state machine, MODEL, DB URL,
│                                      MockLLM, capability probes, print helpers
├── requirements.txt                → FastAPI (CP3); ADK / A2A optional
├── README.md                       → this tutorial
├── REFERENCE.md                    → ADK API map, A2A card spec + lifecycle,
│                                      A2A-vs-MCP, production deployment notes
└── checkpoints/                    → 6 self-contained steps (~15 min each)
    ├── checkpoint1_state_machine.py    Durable state machine + ADK Agent wiring
    ├── checkpoint2_durable_sessions.py Pause/resume across a process restart
    ├── checkpoint3_webhook_resume.py   Event-driven resume (webhook + state_delta)
    ├── checkpoint4_sub_agents.py       In-process delegation — and its wall
    ├── checkpoint5_a2a_cards.py        A2A agent cards + six-state task lifecycle
    └── checkpoint6_fleet.py            Capstone: ADK long-running + A2A end-to-end
```

---

## Setup

No services, no keys. Only **Checkpoint 3** needs a package (FastAPI); the rest
are pure stdlib.

```bash
# from the repo root, using the repo's uv-managed .venv (Python 3.13)
uv pip install -r week17/requirements.txt        # really only needed for CP3
```

Want to see the **real** ADK agent objects instead of the simulation? Add the
optional dep (no Google key required — the checkpoints only *construct* agents):

```bash
uv pip install google-adk        # CP1, CP2, CP4 then print live Agent wiring
```

---

## Run order

Work through `checkpoints/` 1 → 6. Each prints what it did and asserts its own
result, so a clean exit means it worked.

| CP | What you build | Needs | The one idea |
|----|----------------|-------|--------------|
| 1 | Durable **state machine** + ADK `Agent`, `ToolContext` tools, state-interpolated instruction | stdlib | Behaviour comes from `current_step` in state, not chat history |
| 2 | **DatabaseSessionService** durability (shown via a SQLite mirror) | stdlib | State survives a process restart — the "3-day pause" is just a DB row |
| 3 | **Webhook** → `Runner.run_async(state_delta=…)` resume | FastAPI | External events resume the agent (push, not poll) |
| 4 | ADK **`sub_agents`** in-process delegation | stdlib | Focused sub-agents keep reasoning sharp — but they must be *your* code |
| 5 | **A2A** Agent Card + six-state task lifecycle (`input-required` HITL) | stdlib | Delegate to an agent you don't own, discovered via `/.well-known/agent.json` |
| 6 | **Capstone**: long-running + distributed, end-to-end | FastAPI optional | The L3 fleet layer above week15's single-process MAS |

```bash
.venv/bin/python week17/checkpoints/checkpoint1_state_machine.py
.venv/bin/python week17/checkpoints/checkpoint2_durable_sessions.py
.venv/bin/python week17/checkpoints/checkpoint3_webhook_resume.py
.venv/bin/python week17/checkpoints/checkpoint4_sub_agents.py
.venv/bin/python week17/checkpoints/checkpoint5_a2a_cards.py
.venv/bin/python week17/checkpoints/checkpoint6_fleet.py
```

---

## The four ADK building blocks (from the blog)

| Need | ADK primitive | Where you see it |
|------|---------------|------------------|
| Atomic checkpoint on every tool call | `ToolContext.state` | CP1 tools |
| State exists before the first turn | `before_agent_callback` | CP1 `initialize_work_order` |
| Durable persistence (local SQLite ↔ prod Cloud SQL) | `DatabaseSessionService` | CP2 |
| Event-driven resume with a state checkpoint | `Runner.run_async(state_delta=…)` | CP3 |
| Focused delegation | `sub_agents=[…]` | CP4 |

## The A2A protocol (agent communication)

| Half | What it is | Where you see it |
|------|------------|------------------|
| **Discovery** | An **Agent Card** at `/.well-known/agent.json` — `name`, `url`, `authSchemes`, `skills[]` (each with an `input_schema`) | CP5 `VENDOR_CARD` + `validate_card` |
| **Delegation** | A **task** with six states: `submitted → working → completed`, plus `input-required` (network HITL), `failed`, `canceled` | CP5 `MockVendorAgent` + `orchestrator_delegate` |

Three ways to follow a long task: **polling**, **SSE streaming**, **webhooks**
(push). CP3's delivery webhook is exactly the webhook pattern. ADK's
`RemoteA2aAgent` lets you drop a remote A2A agent into `sub_agents=[…]` so
delegating across the network *feels* like calling a local sub-agent.

---

## Notes & honesty

- **A2A is conceptual in this repo.** These checkpoints mock the remote agent so
  the *protocol* (cards, lifecycle, `input-required`) is what you internalize.
  The real shape (`a2a-sdk` / `python_a2a` / ADK `RemoteA2aAgent`) is printed by
  CP5 as guarded, illustrative code. This mirrors the plugin's `a2a-protocol`
  skill, which maps A2A onto week15's hotel MAS as the **L3 fleet** layer.
- **CP2 uses a SQLite mirror, not the real `DatabaseSessionService`.** ADK's
  service persists session state for you against any SQLAlchemy URL; we show that
  exact config and run an equivalent SQLite store so the restart is observable
  with zero dependencies. Swap the URL (`sqlite+aiosqlite://` → Cloud SQL) for
  production — nothing else changes.
- **Model:** `gemini-2.0-flash` (matches the repo's other ADK demo; the blog
  shows `gemini-3.1-flash-lite`). Swap freely — no checkpoint actually calls
  Gemini.
- The checkpoints share `week17_sessions.db` (gitignored) and re-seed it as
  needed, so you can run them in any order.

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

You climb to A2A **only when** the team must span services or organizations.
Below that bar, a function call or an in-process `sub_agent` (CP4) is the right,
cheaper tool. Week 17 is about recognizing the boundary and crossing it cleanly.
