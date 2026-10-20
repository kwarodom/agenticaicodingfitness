# 🧑‍💼 HR Onboarding — a long-running ADK agent with persistent sessions

A **real, runnable** Google ADK agent (not a mock) that runs the blog's
canonical long-running workflow: onboard a new hire across days/weeks, **pausing**
for a contract signature and a laptop delivery, **resuming** from webhooks, and
**never losing context** because state lives in a durable session store — not in
the conversation history.

> This is the live counterpart to `week17_long_running_distributed_agents/checkpoints/` (which are offline
> teaching demos). Here the agent actually calls an LLM, persists every session
> to SQLite, and runs as a long-lived FastAPI service you can restart mid-flow.

```
START ──welcome packet──▶ WELCOME_SENT ⏸  (waiting on signature — days)
                              │  POST /webhooks/document_signed
                              ▼
                         DOCUMENTS_SIGNED ──▶ it_agent provisions accounts ──▶ IT_PROVISIONED ⏸
                              │                                                  (waiting on laptop)
                              │  POST /webhooks/hardware_delivered
                              ▼
                         HARDWARE_DELIVERED ──day-one schedule──▶ COMPLETED
```

---

## Why this is "long-running"

| Property | How | Where |
|----------|-----|-------|
| **Persistent sessions** | `DatabaseSessionService` (SQLite locally, Cloud SQL in prod) — state survives a process restart | `agent.build_session_service` |
| **Durable state machine** | behaviour driven by `current_step` in state, not chat history | `onboarding_steps.py` |
| **Pause** | the agent parks at `WELCOME_SENT` / `IT_PROVISIONED` and refuses to advance | `COORDINATOR_INSTRUCTION` |
| **Event-driven resume** | webhooks apply a `state_delta` before the next inference | `resume_handler.py` |
| **Background process** | one FastAPI/uvicorn service holds no state in memory; scale-to-zero safe | `server.py` |
| **Focused delegation** | coordinator transfers IT provisioning to a sub-agent | `agent.build_root_agent(sub_agents=[it_agent])` |

---

## Files

```
hr_onboarding/
├── onboarding_steps.py   → the OnboardingStep state machine + initial_state()
├── config.py             → model selection (alto/anthropic/gemini), DB URL, .env loader
├── agent.py              → tools (ToolContext), instructions, coordinator + it_agent, Runner
├── resume_handler.py     → OnboardingResumeHandler: run_async(state_delta=…) on a webhook
├── server.py             → the long-running FastAPI service (also serves the visualizer at /)
├── client.py             → a driver that walks one onboarding through the full lifecycle
├── tutorial_server.py    → interactive web guide (Steps 0–13 + cleanup); manages the service
├── static/guide.html     → the clickable step-by-step guide (live terminal output)
├── static/index.html     → the live visualizer (state machine · sub-agents · timeline)
├── TUTORIAL.md           → the same walkthrough as prose, runnable by hand
└── requirements.txt
```

---

## Try it in the browser (recommended)

Two web views, both served with one command each:

```bash
uv pip install -r week17_long_running_distributed_agents/hr_onboarding/requirements.txt
.venv/bin/python week17_long_running_distributed_agents/hr_onboarding/tutorial_server.py    # → http://127.0.0.1:8070
```

- **Interactive guide** (`tutorial_server.py`, port 8070) — click through every
  step of `TUTORIAL.md`, watch each command stream live in an in-page terminal,
  start/stop the service, and demolish/start-over. Auto-picks a free port if
  8070 is taken (`ONBOARDING_GUIDE_PORT` to override).
- **Live visualizer** (the service's own `/`, port 8077) — a real-time view of
  the state machine (with ⏸ pause badges), the coordinator → `it_agent`
  delegation, the new-hire record filling in, and an event timeline. Drive the
  whole onboarding with buttons. Open it once the service is running, or via the
  guide's "Open the live visualizer" step.

Prefer the command line? The `Setup` / `Run it` sections below still apply.

---

## Setup

```bash
uv pip install -r week17_long_running_distributed_agents/hr_onboarding/requirements.txt
```

### Pick a model (it's model-agnostic via LiteLlm)

ADK defaults to Gemini, but this agent runs on whatever endpoint you have. Choose
with `ONBOARDING_PROVIDER`:

| `ONBOARDING_PROVIDER` | Endpoint | Needs | Notes |
|-----------------------|----------|-------|-------|
| `alto` *(default)* | AltoTech LiteLLM gateway (OpenAI-compatible) | `ALTO_LLM_API_KEY`, gateway reachable | Open models: `qwen3`, `llama3.3`, `nemotron`. Set `ONBOARDING_MODEL=qwen3`. |
| `anthropic` | Claude via LiteLlm | a **valid** `ANTHROPIC_API_KEY` | Best tool-use quality. `ONBOARDING_MODEL=anthropic/claude-sonnet-4-6`. |
| `gemini` | Google Gemini (the blog's model) | `GOOGLE_API_KEY` or Vertex ADC | `ONBOARDING_MODEL=gemini-2.0-flash`. |

```bash
# examples
export ONBOARDING_PROVIDER=alto       ONBOARDING_MODEL=qwen3
export ONBOARDING_PROVIDER=anthropic  ONBOARDING_MODEL=anthropic/claude-sonnet-4-6
export ONBOARDING_PROVIDER=gemini     ONBOARDING_MODEL=gemini-2.0-flash
```

The repo-root `.env` is auto-loaded (no python-dotenv needed).

---

## Run it

**Terminal 1 — the long-running service:**

```bash
.venv/bin/python week17_long_running_distributed_agents/hr_onboarding/server.py
# or as a background process:
nohup .venv/bin/python week17_long_running_distributed_agents/hr_onboarding/server.py > /tmp/onboarding.log 2>&1 &
```

**Terminal 2 — drive a full onboarding:**

```bash
.venv/bin/python week17_long_running_distributed_agents/hr_onboarding/client.py
```

The client creates a session, kicks off onboarding (→ pauses at `WELCOME_SENT`),
**tries to skip the wait and is refused**, fires `document_signed` (→ IT
provisioning), supplies a hardware tracking id, fires `hardware_delivered`, and
ends at `COMPLETED` — printing the durable checkpoint after each step.

### Prove persistence survives a restart

```bash
# start an onboarding, let it pause at WELCOME_SENT, then Ctrl-C the server.
# restart server.py — the session is still there:
curl http://127.0.0.1:8077/status/hr_coordinator/<session_id>
# → {"current_step": "WELCOME_SENT", ...}  (nothing was lost)
```

### API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/onboard` | create a durable session at `START` |
| POST | `/chat` | send a message to the coordinator (one agent turn) |
| POST | `/webhooks/document_signed` | resume: contract signed |
| POST | `/webhooks/hardware_delivered` | resume: laptop delivered |
| GET | `/status/{user_id}/{session_id}` | read the checkpoint (no LLM call) |
| GET | `/healthz` | liveness |

---

## Notes

- **Tool-use quality varies by model.** The coordinator relies on function
  calling and sub-agent transfer. Claude/Gemini handle this most reliably; the
  open models on the gateway (`qwen3`, `llama3.3`) generally support tool calls
  via the OpenAI-compatible path but may need a couple of nudges.
- **Production swap:** change `DB_URL` from `sqlite+aiosqlite://` to a Cloud SQL
  URL — nothing else changes. Host on Vertex AI Agent Engine (`AdkApp`) for
  managed scale-to-zero. The same checkpoint-and-resume code runs in both.
- **ADK dev UI:** because `agent.py` follows ADK conventions, you can also poke
  the agent with `adk web` / `adk run` once a model endpoint is configured.
