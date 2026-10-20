# 🔐 auth.md × ADK — authenticating long-running agents, end to end

A **real, runnable** companion to the slide deck
`AltoTech_authmd_x_ADK_Tutorial.pptx`. It joins the two halves of one problem:

- **ADK** keeps a long-running agent's *state* alive across days of idle time —
  a durable state machine, persistent sessions, event-driven wake-up. (This is
  the `week17/hr_onboarding` lesson.)
- **auth.md** keeps a long-running agent's *credentials* valid — scoped,
  short-lived, revocable tokens an agent re-mints every time it wakes.

> **THE GAP the deck names:** auth.md tokens are short-lived *by design*; your
> agent may sleep for days. So the agent stores the **durable grant, not the
> token**, and re-mints a credential each time it wakes.

```
        MCP  =  VERTICAL   (an agent → its own tools/data)         ← Week 7
        A2A  =  HORIZONTAL (an agent → another agent)              ← week17/checkpoints
     auth.md =  the CREDENTIAL an agent presents to a service it doesn't own  ← here

  ┌─ PART A · the AltoTech Energy API (app_server.py) ─────────────────────────┐
  │  publishes auth.md + PRM + AS metadata · POST /agent/auth (verified+claimed)│
  │  /claim · /claim/complete · /revoke · issues SCOPED SHORT-LIVED tokens      │
  └────────────────────────────────────────────────────────────────────────────┘
                       ▲  discover · register · re-mint · revoke
  ┌─ PART B · the long-running ADK work-order agent (agent_server.py) ──────────┐
  │  state machine + DatabaseSessionService · auth_grants in state (not tokens) │
  │  before_tool_callback re-mints a fresh token · scoped per sub-agent          │
  └────────────────────────────────────────────────────────────────────────────┘
```

The work order: optimise one site's HVAC. **Read** the telemetry, **pause** for a
human to approve the change, then **apply** a new setpoint — across "days".

```
START ──analyze (read_agent · sites.read)──▶ ANALYZED ⏸  (waiting on human approval)
                                                │  /webhooks/request_approval → app emails an OTP
                                                │  /webhooks/approved (OTP)    → claim completes
                                                ▼
                                           APPROVED ──apply (apply_agent · control.write)──▶
                                           APPLIED ──verify──▶ COMPLETED
```

---

## Three ways to run it

### 0. The interactive **web guide** (recommended — clickable, no terminal)

A browser walkthrough of the *entire* tutorial (Steps 0–15 + cleanup): click a
step, watch its command stream live in an in-page terminal, start/stop the
servers, and demolish/start-over — all with one process.

```bash
uv pip install -r week17/authmd_adk/requirements.txt   # or the offline subset for Part 1
.venv/bin/python week17/authmd_adk/tutorial_server.py
# → open http://127.0.0.1:8080
```

It manages the Part A / Part B servers for you, shows the exact command behind
every step (copy it to run by hand), and links to the **live visualizer**
(`static/index.html`, served at the agent server's `/`) — a real-time view of
the state machine, auth grants, OTP inbox, and event timeline you can drive with
buttons. Prefer the command line? Use options 1 and 2 below.

### 1. The auth.md protocol, **fully offline** (no ADK, no model, no API key)

One command boots Part A in a thread and walks the *entire* protocol with the
Part B client — real RS256/JWKS crypto, zero external network:

```bash
uv pip install "fastapi" "uvicorn[standard]" httpx pydantic "pyjwt[crypto]"
.venv/bin/python week17/authmd_adk/run_authmd_demo.py
```

You'll watch: discovery → agent-verified token → **wake-time re-mint** (a brand
new token from the same durable grant) → the user-claimed OTP gate → **least
privilege** (the `sites.read` token is refused `403` on a write) → apply →
revocation (`401`) → the audit log.

### 2. The long-running ADK agent, **end to end** (needs `google-adk` + a model)

```bash
uv pip install -r week17/authmd_adk/requirements.txt
.venv/bin/python week17/authmd_adk/app_server.py     # terminal 1 — Part A
.venv/bin/python week17/authmd_adk/agent_server.py    # terminal 2 — Part B
.venv/bin/python week17/authmd_adk/run_full_demo.py   # terminal 3 — the driver
```

The driver creates a work order, lets the agent analyze + park at `ANALYZED`,
**tries to skip approval (the agent refuses)**, fires the approval webhook
(emails an OTP), then the approved webhook with that OTP — the claim completes,
a fresh `control.write` token is injected via `state_delta`, and the setpoint is
applied. The model runs on the repo's Alto gateway by default; override with
`AUTHMD_PROVIDER=anthropic` or `=gemini` (see `config.py`).

---

## Part A — make your app agent-ready (`app_server.py`)

Every endpoint the deck's slides A1–A6 require, in one FastAPI service:

| Slide | What | Where |
|------|------|-------|
| A1 | `GET /auth.md` — the human/agent-readable discovery doc | `auth_md()` |
| A2 | `GET /.well-known/oauth-protected-resource` (PRM, RFC 9728) | `prm()` |
| A2 | `GET /.well-known/oauth-authorization-server` (the `agent_auth` block) | `as_metadata()` |
| A2 | `401` + `WWW-Authenticate: Bearer resource_metadata="…"` hint | `_require_scope()` |
| A3 | `POST /agent/auth` — one endpoint, dispatch on `type` | `agent_auth()` |
| A4 | **agent-verified** — verify the ID-JAG against the provider JWKS, issue a token (no refresh token) | `_handle_verified()` |
| A5 | **user-claimed** — the OTP ceremony (`/claim`, `/claim/complete`) | `_handle_anonymous` / `_handle_email_required` / `agent_auth_claim*` |
| A6 | matching + JIT, revocation (`/revoke`, logout tokens), replay cache, hashed secrets | `_match_or_provision`, `agent_auth_revoke`, `_SEEN_JTI` |

`idjag_provider.py` is a self-contained **trusted provider** (stands in for
OpenAI / Anthropic / Cursor): it mints real RS256 ID-JAGs and serves a JWKS. Its
signing key is **persisted to `idjag_signing_key.pem`** so Part A and Part B —
separate processes — share it (a real provider's key is stable, not regenerated
each boot).

## Part B — consume auth.md from the ADK agent

| Slide | What | Where |
|------|------|-------|
| B1 | durable state machine + persistent sessions + event-driven resume | `work_order.py`, `agent.py`, `resume_handler.py` |
| B2 | **store the grant, not the token** — `auth_grants` in session state | `work_order.initial_grants()`, `authmd_client.AuthGrant` |
| B3 | **acquire credentials at wake time** — `before_tool_callback` re-mints | `agent.AuthInjector`, `authmd_client.acquire()` |
| B4 | the approval gate **is** the user-claimed flow; **scope per sub-agent** | `resume_handler.approved()`, `read_agent`/`apply_agent` |

`authmd_client.py` is the seam that bridges both halves. `AuthGrant` is the
durable, token-free thing stored in state; `AuthMdClient.acquire()` turns it into
a fresh credential on demand by running real auth.md discovery + the right flow.
It's deliberately generic — the same call is where you'd fall back to a plain
OAuth refresh exchange for a service that doesn't publish an auth.md.

---

## Files

```
authmd_adk/
├── config.py            → app/agent base URLs, DB URL, model selection, .env loader
│  PART A (offline-capable: pure HTTP + crypto)
├── app_server.py        → the agent-ready AltoTech Energy API (all of A1–A6)
├── idjag_provider.py    → mock trusted provider: RS256 ID-JAG signing + JWKS (real crypto)
│  PART B (the auth bridge is offline-capable; the agent needs google-adk + a model)
├── authmd_client.py     → AuthGrant + AuthMdClient.acquire()/start_claim()/complete_claim()/revoke()
├── work_order.py        → WorkOrderStep state machine + auth_grants block
├── agent.py             → coordinator + read_agent + apply_agent; AuthInjector before_tool_callback
├── resume_handler.py    → request_approval (start claim) + approved (complete claim, re-mint, resume)
├── agent_server.py      → the long-running FastAPI/ADK service (also serves the live visualizer at /)
│  WEB UI
├── tutorial_server.py   → ⓪ interactive guide control-plane (Steps 0–15 + cleanup); manages both servers
├── static/guide.html    → the clickable step-by-step guide (terminal output, start/demolish)
├── static/index.html    → the live visualizer (state machine · grants · OTP · timeline)
│  DRIVERS (command line)
├── run_authmd_demo.py   → ① offline protocol walk-through (no ADK/model)
├── run_full_demo.py     → ② end-to-end ADK pause/resume across the approval gate
└── requirements.txt
```

---

## Reality check (from the deck's final slide)

- **The ADK foundation is solid** — official Google guidance, mainstream
  production patterns. Build on it now. (See `week17/hr_onboarding` for the bare
  ADK version without auth.)
- **auth.md is real but early** — a WorkOS-authored open protocol in early
  access. It composes existing OAuth RFCs (no new crypto); adoption is still
  thin and the spec may change. Design around the generic **auth-grant**
  abstraction (`AuthGrant` / `acquire()`) so you can use auth.md where a service
  publishes one and fall back to standard OAuth refresh everywhere else.
- **What's faithful here vs. stubbed for teaching.** Faithful: RFC 9728
  discovery shapes, real RS256/JWKS ID-JAG verification, `jti` replay
  protection, SHA-256-only storage of OTP/claim secrets, the no-refresh-token
  rule. Stubbed: the credential store is an in-process dict (use a DB), the
  `_demo/inbox` endpoint stands in for a real email (an OTP must never appear in
  an API response in production — it's there only so the offline demo can "read
  the user's email"), and rate-limiting is a note, not middleware.

> Sources: `workos.com/auth-md/docs/apps` · `github.com/workos/auth.md` ·
> `developers.googleblog.com` (ADK long-running agents).
