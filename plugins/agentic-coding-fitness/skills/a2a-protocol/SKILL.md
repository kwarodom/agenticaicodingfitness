---
name: a2a-protocol
description: "Teach the Agent-to-Agent (A2A) protocol — the cross-framework counterpart to MCP that lets one agent discover, authenticate to, and delegate work to OTHER agents (built by other teams, on other frameworks). Covers Agent Cards at /.well-known/agent.json, the six-state task lifecycle (submitted → working → input-required → completed/failed/canceled), the three interaction patterns (polling, SSE streaming, webhooks), the A2A-vs-MCP complementary split (MCP = agent→tools, A2A = agent→agent), and the shape of an a2a-sdk agent. Now grounded in runnable Week 17 checkpoints (week17_long_running_distributed_agents/checkpoints/checkpoint5_a2a_cards.py + checkpoint6_fleet.py) that walk the real card + lifecycle offline, mocking only the remote peer. Use when someone asks 'how do agents on different frameworks talk to each other?', 'what's A2A / agent cards / agent.json?', 'A2A vs MCP', 'how do I split my multi-agent system across services?', or is reviewing Week 15/17 multi-agent / fleet-orchestration material."
when_to_use: "Learner is outgrowing a single in-process multi-agent codebase and wants agents (possibly on different frameworks/teams/services) to discover and delegate to each other, asks about A2A, Agent Cards, /.well-known/agent.json, the A2A task lifecycle, or how A2A differs from and complements MCP."
---

# A2A Protocol — Agents Calling Other Agents (cross-cutting)

> **The one idea:** `mcp-and-skills` connected an agent to **tools**. **A2A** connects an agent to **other agents** — possibly written by another team, in another framework, running on another server. MCP is *vertical* (agent → its tools); A2A is *horizontal* (agent ↔ agent).

> ✅ **Now grounded — runnable in `week17_long_running_distributed_agents/checkpoints/`.** A2A used to be pure map; **Week 17 makes the protocol runnable offline.** `checkpoint5_a2a_cards.py` builds and validates a real Agent Card and walks the full six-state task lifecycle (including `input-required` HITL); `checkpoint6_fleet.py` is the capstone — an ADK long-running coordinator delegating across an A2A boundary, end-to-end. **What's still mocked is only the *remote peer agent*** (a `MockVendorAgent`), so you internalize the cards + lifecycle + handoff without standing up a second org's server. The real wire shape (`a2a-sdk` / `python_a2a` / ADK `RemoteA2aAgent`) is printed as guarded, illustrative code. For the long-running half of Week 17 (durable state, pause/resume, auth.md) see the **`long-running-and-distributed-agents`** skill. The original in-process baseline you're stretching is `week15_production_graphrag/smart_hotel_mas` (5 agents, one CrewAI crew); its `INTEGRATION_ARCHITECTURE.md` puts **A2A at "L3 — Fleet Orchestration"**, the layer above the MCP-driven supervisory agents.

```
mcp-and-skills:   agent ──MCP──▶ [ database · files · web · calendar ]      (vertical)
a2a-protocol:     agent ──A2A──▶ agent ──A2A──▶ agent                        (horizontal)
                    │              │
                    └─MCP─▶ tools  └─MCP─▶ tools     ← each agent still uses MCP for ITS tools
```

A2A was launched by Google in April 2025 and donated to the Linux Foundation; the ecosystem (Atlassian, Salesforce, SAP, ServiceNow, and IBM's merged ACP) is the cross-vendor bet, the way MCP is for tools.

---

## Part A — Agent Cards: the public capability manifest

Before one agent can delegate to another, it has to **discover** what the other can do and **how to talk to it** — without reading its source. A2A solves this exactly the way a browser fetches a favicon: every A2A agent publishes a JSON **Agent Card** at a well-known URL.

```
GET https://nutrition.example.com/.well-known/agent.json
```

> **Spec note:** the current A2A spec (v0.3+) serves the card at `/.well-known/agent-card.json`; the older `/.well-known/agent.json` shown throughout this (conceptual) skill is the **legacy fallback**. When you point discovery at a real agent, check `agent-card.json` first.

The card is a contract. A calling agent reads it and learns four things:

| Card field | What the caller learns | Example |
|---|---|---|
| `name` / `description` | *Is this the specialist I need?* | "Nutrition Specialist — analyzes diet, builds meal plans" |
| `url` | *Where do I send the task?* | `https://nutrition.example.com/a2a` |
| `authSchemes` | *Can I even authenticate?* | `oauth2` / `apiKey` / `bearer` / `none` |
| `skills[]` | *What exactly can it do, and what input does each skill need?* | `meal-plan-generation` + an input JSON schema |

```jsonc
// /.well-known/agent.json — the manifest a remote specialist publishes
{
  "schemaVersion": "1.0",
  "humanReadableId": "fitness/nutrition-specialist",
  "agentVersion": "1.2.0",
  "name": "Nutrition Specialist",
  "description": "Analyzes dietary data and generates meal plans",
  "url": "https://nutrition.example.com/a2a",
  "capabilities": { "a2aVersion": "1.0", "supportsPushNotifications": true },
  "authSchemes": [ { "scheme": "oauth2", "tokenUrl": "https://auth.example.com/token" } ],
  "skills": [
    {
      "id": "meal-plan-generation",
      "name": "Meal Plan Generation",
      "description": "Creates meal plans given calories, macros, and dietary restrictions",
      "input_schema": {
        "type": "object",
        "properties": {
          "calories": { "type": "integer" },
          "protein_g": { "type": "integer" },
          "restrictions": { "type": "array", "items": { "type": "string" } }
        },
        "required": ["calories"]
      }
    }
  ]
}
```

**The discovery handshake** a calling agent runs:

1. **Fetch** `…/.well-known/agent.json`.
2. **Verify** the `authSchemes` match the credentials it holds (e.g. it has an OAuth2 token → good).
3. **Match** the task to a `skills[].id` and check its `input_schema`.
4. **Submit** the task to `url` — and only now does the real work start (Part B).

> The Agent Card is to A2A what `tools/list` is to MCP, and what `SKILL.md` frontmatter is to a skill (`mcp-and-skills`): a small, public, machine-readable "here's what I can do and how to call me." Same instinct, one level up the ladder — from a *tool* to a *whole agent*.

---

## Part B — The task lifecycle: six states, three ways to watch

When you call a tool over MCP it returns more or less instantly. When you delegate to a whole *agent*, the work can take seconds, minutes, or pause to ask a human. So A2A models delegation as a **task with a lifecycle**, not a function call. Six states:

| State | Meaning | Who moves it |
|---|---|---|
| `submitted` | Task received, not started | caller → callee |
| `working` | Agent is processing | callee |
| `input-required` | Paused — needs more input (often a human) | callee → caller |
| `completed` | Done, result available ✅ | callee |
| `failed` | Unrecoverable error ❌ | callee |
| `canceled` | Caller killed it | caller |

```
submitted → working → completed          (happy path)
                ├────→ input-required ──▶ (resume) ──▶ working → completed
                ├────→ failed
   (caller) ───────────────────────────────────────▶ canceled
```

The **`input-required`** state is the one to remember — it's how A2A does **human-in-the-loop across a network boundary**. A medical-screening agent that hits a contraindication transitions to `input-required` ("Client reports chest pain — require physician clearance?"); the orchestrator surfaces that to a human and the task *resumes* when the answer comes back. That's the HITL pattern from `models-and-patterns`, now spanning two separately-deployed agents.

Because tasks can be long, A2A gives the caller **three ways to follow progress** — pick by how long the work runs:

| Pattern | How it works | Use when |
|---|---|---|
| **Polling** (request/response) | `POST /a2a/tasks`, then `GET /a2a/tasks/{id}` on a loop | Short tasks; simplest to build |
| **SSE streaming** | Server-Sent Events push state/partial output as it happens | You want live progress / token-by-token output |
| **Webhooks** (push notifications) | Callee POSTs to your callback URL when state changes | Long-running async work — minutes/hours; don't hold a connection open |

> The `supportsPushNotifications` flag back in the Agent Card (Part A) is the callee *telling you up front* whether webhooks are even an option. The card and the lifecycle are two halves of one protocol.

---

## Part C — A2A vs MCP: complementary, not competing

These get confused constantly because both let an AI system "talk to something external." They solve **different** problems and the production answer is to use **both**.

| Dimension | **MCP** (`mcp-and-skills`) | **A2A** (this skill) |
|---|---|---|
| Connects | agent → **tool / data source** | agent → **another agent** |
| Direction | **vertical** (model → system) | **horizontal** (agent ↔ agent) |
| Purpose | standardize tool & data access | delegate tasks across frameworks |
| Discovery | `tools/list` on a server | **Agent Card** at `/.well-known/agent.json` |
| Architecture | client–server (host initiates) | peer-to-peer delegation |
| Scope | one agent's tool access | multi-agent orchestration |
| Governance | Anthropic (open standard) | Linux Foundation |
| Best for | one agent → DBs, APIs, files | many agents, built by different teams |

**The canonical production pattern uses both at once:** each agent reaches its **tools** via MCP, and reaches its **peers** via A2A.

```
Orchestrator: "Design a fitness program for a client with hypertension."
   │
   ├─A2A─▶ Medical Screening Agent ──MCP──▶ health-records DB   (checks contraindications)
   │            └─ flags chest pain → task state: input-required → human clears it
   ├─A2A─▶ Programming Agent       ──MCP──▶ exercise DB + calendar
   └─A2A─▶ Nutrition Agent         ──MCP──▶ food/macro DB
```

A2A is the **horizontal coordination**; MCP is the **vertical governance** (each specialist only gets the tools its role allows). Together they form a security boundary that scales with the system.

**Reach for which?**
- **MCP** — you want one agent to *use a tool or read data* in a reusable, standard way. (You're here ~95% of the time in this course.)
- **A2A** — you have **genuinely separate agents** (different teams, frameworks, deploys) that must delegate to each other. Below that bar, just call a function or stay in one process — see the honest note at the top.

---

## Part D — Building one (the shape of an a2a-sdk agent)

Three responsibilities: **expose a card**, **accept a task**, **report progress**. The official SDKs (`a2a-sdk`, plus Go/JS/Java/.NET) share these concepts, so you can build polyglot systems. The shape below mirrors the research's `python_a2a` example — a Workout Coach that advertises three skills and routes incoming messages by intent.

```python
# ⚠️ ILLUSTRATIVE wire shape (CP5 prints a guarded version of this). pip install a2a-sdk
from python_a2a import A2AServer, Message, TextContent, MessageRole, run_server

AGENT_CARD = {                                   # 1) EXPOSE A CARD (Part A)
    "schemaVersion": "1.0",
    "humanReadableId": "fitness/workout-coach",
    "name": "Workout Coach",
    "description": "AI coach for workout plans, progress tracking, and recovery.",
    "url": "http://localhost:5001/a2a",
    "capabilities": {"a2aVersion": "1.0", "supportsPushNotifications": False},
    "authSchemes": [{"scheme": "none"}],
    "skills": [
        {"id": "workout-planning",  "name": "Workout Planning",  "description": "Create programs"},
        {"id": "progress-tracking", "name": "Progress Tracking", "description": "Analyze performance"},
        {"id": "recovery-advice",   "name": "Recovery Advice",   "description": "Recovery recommendations"},
    ],
}

class WorkoutCoachAgent(A2AServer):
    def handle_message(self, message):           # 2) ACCEPT A TASK
        text = message.content.text.lower()
        if any(k in text for k in ("plan", "program", "routine")):
            return self._reply(message, "# 3-day Strength Program\n- Squat 5x5 ...")
        if any(k in text for k in ("recovery", "rest", "sleep")):
            return self._reply(message, "Sleep 7-9h; protein 1.6-2.2 g/kg; deload week 4.")
        return self._reply(message, "I can plan workouts, track progress, advise recovery.")

    def _reply(self, message, text):             # 3) REPORT BACK (structured response)
        return Message(content=TextContent(text=text), role=MessageRole.AGENT,
                       parent_message_id=message.message_id,
                       conversation_id=message.conversation_id)

if __name__ == "__main__":
    run_server(WorkoutCoachAgent(), host="0.0.0.0", port=5001)
    # Card auto-served at http://localhost:5001/.well-known/agent.json
```

Test it the way a *calling* agent would: `GET http://localhost:5001/.well-known/agent.json` **first** (read the card, verify auth & skills), *then* send a task message. Higher-level wrappers (Google ADK's `RemoteA2aAgent`) let you compose a remote agent so delegating across the network feels like calling a local function — discover endpoint → negotiate capability → submit task → handle the lifecycle, never knowing the other side's framework.

> 📁 **Runnable now:** `week17_long_running_distributed_agents/checkpoints/checkpoint5_a2a_cards.py` — builds + validates a real Agent Card and walks the six-state lifecycle (`MockVendorAgent` + `orchestrator_delegate`), offline. `checkpoint6_fleet.py` — the capstone: ADK long-running coordinator + A2A delegation end-to-end. `week17_long_running_distributed_agents/REFERENCE.md` — the A2A card spec, lifecycle, and A2A-vs-MCP table.
> 📁 Roadmap anchor: `week15_production_graphrag/smart_hotel_mas/INTEGRATION_ARCHITECTURE.md` — its three-layer model puts **A2A at "L3 — Fleet Orchestration (portfolio)"**, above MCP-driven L2 supervisory agents.
> 📁 In-process baseline: `week15_production_graphrag/smart_hotel_mas/README.md` — the 5 agents (Sensor/Energy/Memory/Alert/Report) run as **one CrewAI crew in one process**. That's `multi-agent-systems`. A2A is what you reach for to split them across services.

**Where this fits:** you've gone single-agent (`agent-loops`) → tools (`tool-use`, `mcp-and-skills`) → a team in one process (`multi-agent-systems`) → and A2A is the next rung *only when that team must span services or organizations.* Don't climb it early.

---

## 🧪 Guided lab (offer this)

**You'll need:** no API key, no install — this is design + a $0 mock. The point is to internalize the **card** and the **handoff**, not to stand up a server.

### Warm-up (5–10 min) — hand-write an Agent Card *(binary pass/fail)*
Take one agent you've already built in this course (e.g. a `week15_production_graphrag/smart_hotel_mas` agent, or any `multi-agent-systems` role) and write its `/.well-known/agent.json` by hand.
- **Pass** = the card is valid JSON and contains all four required pieces: `name`, `url`, at least one `authScheme`, and a `skills[]` array where every skill has an `id`, a `description`, and an `input_schema`.
- **Fail** = missing any of those, or `skills` is just free text with no input schema.

### Skill Drill (15–30 min) — sketch a two-agent A2A handoff with a MockLLM *(runnable, $0)*
Stub a tiny **Orchestrator → Specialist** delegation. No network, no key — a `MockLLM` stands in for both agents so it runs offline.

```python
# a2a_handoff_drill.py  —  runs with plain `python`, no install, $0
import json

class MockLLM:                       # deterministic stand-in for a real model
    def __call__(self, prompt):
        return "meal-plan-generation" if "nutrition" in prompt.lower() else "no-match"

# --- the Specialist publishes a card (Part A) ---
SPECIALIST_CARD = {
    "name": "Nutrition Specialist",
    "url": "http://specialist.local/a2a",
    "authSchemes": [{"scheme": "none"}],
    "skills": [{"id": "meal-plan-generation", "description": "Build meal plans",
                "input_schema": {"type": "object", "properties": {"calories": {"type": "integer"}},
                                 "required": ["calories"]}}],
}

REQUIRED_CARD_FIELDS = ["name", "url", "authSchemes", "skills"]

def validate_card(card):             # the discovery-side check
    assert all(f in card for f in REQUIRED_CARD_FIELDS), "card missing a required field"
    for s in card["skills"]:
        assert {"id", "input_schema"} <= s.keys(), "skill missing id/input_schema"
    return True

def orchestrator_delegate(card, user_msg, llm):
    validate_card(card)                                  # 1) discover + verify
    skill_id = llm(f"Which skill handles: {user_msg}")   # 2) match skill
    skill = next((s for s in card["skills"] if s["id"] == skill_id), None)
    if not skill:
        return {"state": "failed", "reason": "no matching skill"}
    task = {"state": "submitted", "skill": skill_id, "input": {"calories": 2200}}
    # 3) walk the lifecycle (mocked specialist)
    task["state"] = "working"
    task["state"], task["result"] = "completed", "Meal plan: 2200 kcal, 5 meals."
    return task

if __name__ == "__main__":
    result = orchestrator_delegate(SPECIALIST_CARD,
                                   "Plan nutrition for a cutting phase", MockLLM())
    print(json.dumps(result, indent=2))
    assert result["state"] == "completed"
    print("handoff OK")
```

**Then extend it (pick any two):** add an `input-required` branch (specialist pauses, asks "any allergies?", caller answers, task resumes); make `validate_card` reject a card with no `authSchemes`; add a second skill and have the MockLLM route to the wrong one and watch it `fail`.

### Weighted evaluation criteria (pass = **4 / 5**)
| # | Criterion | Weight |
|---|---|---|
| 1 | Warm-up card is valid JSON with all four required fields (`name`, `url`, `authSchemes`, `skills`) | ●●● |
| 2 | Every skill in the card has an `id` **and** an `input_schema` (not free text) | ●●● |
| 3 | The drill runs to `state: completed` and the final assert passes ($0, no key) | ●● |
| 4 | The orchestrator **validates the card before delegating** (discovery handshake, not a blind call) | ●● |
| 5 | Learner can state, in one sentence each, **MCP vs A2A** and when `input-required` fires | ● |

**Pass threshold: 4 of 5.** Criteria 1 and 2 (the card) are non-negotiable — a wrong card means no one can discover you. Close by placing A2A on the ladder out loud: "`mcp-and-skills` wired me to tools; `multi-agent-systems` put a team in one process; A2A is for when that team has to live in different services."
