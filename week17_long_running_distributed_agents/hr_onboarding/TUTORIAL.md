# 🧭 Tutorial — a long-running ADK agent, step by step

A hands-on walkthrough of `week17/hr_onboarding/`: a **real** Google ADK agent
that onboards a new hire across *days or weeks* — sending a welcome packet,
**pausing** for a contract signature, delegating IT provisioning to a sub-agent,
**pausing** again for a laptop delivery, and **resuming** from webhooks without
ever losing its place.

> This whole tutorial uses a **live LLM** (unlike the offline
> `week17/checkpoints/` demos). You need `google-adk` + one model credential.

Time: ~25 min.

> 💡 **Prefer clicking to typing?** Every step below is also a button in the
> interactive web guide — it streams each command's output in the page, manages
> the service for you, and links to a live visualizer:
> ```bash
> .venv/bin/python week17/hr_onboarding/tutorial_server.py   # → http://127.0.0.1:8070
> ```
> The text walkthrough below is the same content, runnable by hand.

---

## The one idea to hold onto

```
A chatbot remembers by REPLAYING its whole conversation history.
A long-running agent can't — it may sleep for DAYS between messages.

   So progress lives in an EXPLICIT, DURABLE state machine (current_step),
   persisted to disk on every tool call, and the agent RESUMES from a webhook
   that advances that state BEFORE the next time the model thinks.
```

The six steps of the machine:

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

# Part 1 — understand the agent

## Step 0 · Setup

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
uv pip install -r week17/hr_onboarding/requirements.txt
.venv/bin/python -c "import google.adk, litellm; print('ready')"
```

**Checkpoint:** prints `ready`. The agent is model-agnostic; by default it runs
on the repo's Alto gateway. To switch: `export ONBOARDING_PROVIDER=anthropic`
(uses `ANTHROPIC_API_KEY`) or `=gemini` (needs `GOOGLE_API_KEY`). See `config.py`.

## Step 1 · The durable state machine

Open `onboarding_steps.py`. The key ideas:

```python
class OnboardingStep:
    START = "START"
    WELCOME_SENT = "WELCOME_SENT"          # paused: waiting for signed contract
    DOCUMENTS_SIGNED = "DOCUMENTS_SIGNED"   # set by the document_signed webhook
    IT_PROVISIONED = "IT_PROVISIONED"       # paused: waiting for hardware
    HARDWARE_DELIVERED = "HARDWARE_DELIVERED"
    COMPLETED = "COMPLETED"

    # the two steps where the agent must REFUSE to advance on its own:
    PAUSED_STEPS = {WELCOME_SENT: "document_signed",
                    IT_PROVISIONED: "hardware_delivered"}
```

`initial_state()` seeds every key the agent's instruction interpolates
(`current_step`, `new_hire_details`, `pending_signals`) — they must all exist up
front or ADK's `{var}` templating raises on turn one.

## Step 2 · The agent, its tools, and the sub-agent

Open `agent.py`. Three things make this long-running:

- **Tools advance the state machine atomically.** `send_welcome_packet` sets
  `current_step = WELCOME_SENT`; the Runner persists that `state_delta`. If the
  process crashes right after, the state is already on disk.
- **A focused sub-agent.** The coordinator `transfer`s IT provisioning to a
  narrow `it_agent` (`provision_software_accounts` only) — keeping each prompt
  sharp even after weeks of accumulated state.
- **Durable sessions.** `build_session_service()` returns a
  `DatabaseSessionService` (SQLite locally, Cloud SQL in prod) — the same object
  the server and the webhooks share.

## Step 3 · The resume handler

Open `resume_handler.py`. A paused onboarding is **not polled**. When the real
world produces a signal, a webhook calls `Runner.run_async(state_delta=…)`, which
advances `current_step` **before** the next inference — so the model wakes up
already seeing the new step and can't pretend it got there on its own.

---

# Part 2 — run it and drive one onboarding

## Step 4 · Start the long-running service

```bash
.venv/bin/python week17/hr_onboarding/server.py
```

**Checkpoint:** prints `✅ onboarding service up — model=…`. It listens on
`http://127.0.0.1:8077` and holds **no** onboarding state in memory — everything
is in the SQLite session store. Leave it running.

> The endpoints: `POST /onboard`, `POST /chat`, `POST /webhooks/document_signed`,
> `POST /webhooks/hardware_delivered`, `GET /status/{user}/{session}`.

## Step 5 · Create an onboarding session

```bash
.venv/bin/python - <<'PY'
import httpx, uuid
sid = "onb-" + uuid.uuid4().hex[:8]; open("/tmp/hr_sid.txt","w").write(sid)
r = httpx.post("http://127.0.0.1:8077/onboard", json={"session_id": sid}, timeout=60).json()
print("session:", sid, "| step:", r["current_step"])
PY
```

**Checkpoint:** `step: START` — a durable session created with no LLM call.

## Step 6 · Send the welcome packet (first pause)

```bash
.venv/bin/python - <<'PY'
import httpx
sid = open("/tmp/hr_sid.txt").read().strip()
r = httpx.post("http://127.0.0.1:8077/chat", json={"session_id": sid,
    "message": "Start onboarding for Jane Doe, email jane@example.com, start date 2026-07-01."},
    timeout=300).json()
print("agent:", (r.get("reply") or "").strip()[:240])
print("step:", r["current_step"], "| waiting_for:", r.get("paused_waiting_for"))
print("details:", r.get("new_hire_details"))
PY
```

**Checkpoint:** `step: WELCOME_SENT`, `waiting_for: document_signed`, and the
new-hire details are now in durable state. The agent has **paused**.

## Step 7 · Try to skip the wait (the safety gate)

```bash
.venv/bin/python - <<'PY'
import httpx
sid = open("/tmp/hr_sid.txt").read().strip()
r = httpx.post("http://127.0.0.1:8077/chat", json={"session_id": sid,
    "message": "Can we skip the signature and provision IT accounts now?"}, timeout=300).json()
print("agent:", (r.get("reply") or "").strip()[:240])
print("step:", r["current_step"], "(should STILL be WELCOME_SENT)")
PY
```

**Checkpoint:** the agent **refuses** and the step stays `WELCOME_SENT`. It will
not advance until the real signal arrives.

## Step 8 · Inspect the durable state (no LLM)

```bash
.venv/bin/python - <<'PY'
import httpx, json
sid = open("/tmp/hr_sid.txt").read().strip()
print(json.dumps(httpx.get(f"http://127.0.0.1:8077/status/hr_coordinator/{sid}").json(), indent=2))
PY
```

**Checkpoint:** you read the checkpoint (step, pending signal, details) with **no
model call** — proof that progress lives in the session store, not the chat.

## Step 9 · Webhook: document signed → delegate to it_agent

```bash
.venv/bin/python - <<'PY'
import httpx
sid = open("/tmp/hr_sid.txt").read().strip()
r = httpx.post("http://127.0.0.1:8077/webhooks/document_signed",
               json={"session_id": sid}, timeout=300).json()
print("agent:", (r.get("reply") or "").strip()[:240])
print("step:", r["current_step"], "| details:", r.get("new_hire_details"))
PY
```

**Checkpoint:** `step: IT_PROVISIONED` and the details now include a
`corp_username` + `accounts` — the coordinator **delegated to `it_agent`**, which
provisioned them, then handed control back.

## Step 10 · Provide the hardware tracking id (second pause)

```bash
.venv/bin/python - <<'PY'
import httpx
sid = open("/tmp/hr_sid.txt").read().strip()
r = httpx.post("http://127.0.0.1:8077/chat", json={"session_id": sid,
    "message": "The laptop shipped, tracking id 1Z999AA10123456784."}, timeout=300).json()
print("agent:", (r.get("reply") or "").strip()[:240])
print("step:", r["current_step"], "| pending:", r.get("pending_signals"))
PY
```

**Checkpoint:** the tracking id is recorded and the agent **pauses again**,
`pending: ['hardware_delivered']`.

## Step 11 · Webhook: hardware delivered → complete

```bash
.venv/bin/python - <<'PY'
import httpx
sid = open("/tmp/hr_sid.txt").read().strip()
r = httpx.post("http://127.0.0.1:8077/webhooks/hardware_delivered",
               json={"session_id": sid}, timeout=300).json()
print("agent:", (r.get("reply") or "").strip()[:240])
print("step:", r["current_step"])
PY
```

**Checkpoint:** `step: COMPLETED` — the agent sent the day-one schedule and
finished. 🎉

---

# Part 3 — prove it's really durable

The headline claim: a paused onboarding survives a full process restart.

1. **Pause a fresh onboarding:**
   ```bash
   .venv/bin/python - <<'PY'
   import httpx, uuid
   sid = "onb-" + uuid.uuid4().hex[:8]; open("/tmp/hr_dur.txt","w").write(sid)
   B = "http://127.0.0.1:8077"
   httpx.post(f"{B}/onboard", json={"session_id": sid}, timeout=60)
   httpx.post(f"{B}/chat", json={"session_id": sid,
       "message": "Start onboarding for Sam Lee, email sam@example.com, start date 2026-08-01."}, timeout=300)
   print("session:", sid, "step:", httpx.get(f"{B}/status/hr_coordinator/{sid}").json()["current_step"])
   PY
   ```
   **Checkpoint:** `step: WELCOME_SENT`.

2. **Restart the service** — `Ctrl-C` the `server.py` terminal and run it again.
   In-memory state is gone; the SQLite session is not.

3. **Resume the same session:**
   ```bash
   .venv/bin/python - <<'PY'
   import httpx
   sid = open("/tmp/hr_dur.txt").read().strip(); B = "http://127.0.0.1:8077"
   print("after restart, step =", httpx.get(f"{B}/status/hr_coordinator/{sid}").json()["current_step"])
   r = httpx.post(f"{B}/webhooks/document_signed", json={"session_id": sid}, timeout=300).json()
   print("resumed → step =", r["current_step"])
   PY
   ```
   **Checkpoint:** `after restart, step = WELCOME_SENT` (survived!), then
   `resumed → step = IT_PROVISIONED`. The agent woke from a cold restart and
   continued exactly where it paused. **That is the long-running story.**

---

## The one-shot driver

Everything in Parts 2–3 is also scripted end to end in `client.py`:

```bash
.venv/bin/python week17/hr_onboarding/server.py     # terminal 1
.venv/bin/python week17/hr_onboarding/client.py     # terminal 2
```

## Cleanup

```bash
# stop the server (Ctrl-C), then:
rm -f week17/hr_onboarding/onboarding_sessions.db    # wipe all sessions
```

(Or use the **Cleanup & teardown** steps in the web guide.)

---

## Troubleshooting

| Symptom | Fix |
|--------|-----|
| `server.py` won't bind / port busy | Another process holds 8077 — `ONBOARDING_PORT=8078 .venv/bin/python …/server.py` (and point the client at it). |
| Agent turn errors on the model | Check `ONBOARDING_PROVIDER` and that the matching key is set; read the `server.py` terminal traceback. |
| Agent advances without the webhook | It shouldn't — if it does, your model is ignoring the `WELCOME_SENT`/`IT_PROVISIONED` gate; try a stronger model via `ONBOARDING_PROVIDER`. |
| Want a clean slate | Stop the server and delete `onboarding_sessions.db`. |
