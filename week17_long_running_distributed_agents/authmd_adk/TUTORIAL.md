# 🧭 Tutorial — auth.md × ADK, step by step

A hands-on walkthrough of `week17/authmd_adk/`. You'll run real code at every
step and watch the two halves of the deck come together:

- **ADK** keeps a long-running agent's **state** alive across days of idle time.
- **auth.md** keeps that agent's **credentials** valid — scoped, short-lived,
  revocable tokens it re-mints every time it wakes.

> Work through it top to bottom. **Part 1** (Steps 0–9) needs *no model and no
> API key* — pure HTTP + crypto. **Part 2** (Steps 10–15) runs the live ADK
> agent and needs `google-adk` + a model.

Time: ~30 min for Part 1, ~20 min for Part 2.

> 💡 **Prefer clicking to typing?** Every step below is also a button in the
> interactive web guide — it streams each command's output in the page and
> manages the servers for you:
> ```bash
> .venv/bin/python week17/authmd_adk/tutorial_server.py   # → http://127.0.0.1:8080
> ```
> The text walkthrough below is the same content, runnable by hand.

---

## The one idea to hold onto

```
auth.md tokens are SHORT-LIVED by design.   Your agent may SLEEP for days.
        └────────────────────┬───────────────────────────┘
                       so the agent stores the
              DURABLE GRANT  (what it's allowed to do)
                    NOT the TOKEN  (which will have expired),
                 and RE-MINTS a fresh credential each time it wakes.
```

Everything below is in service of that sentence.

---

# Part 1 — the auth.md protocol (offline, no model)

## Step 0 · Setup

You only need the offline subset for Part 1:

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
uv pip install "fastapi" "uvicorn[standard]" httpx pydantic "pyjwt[crypto]"
```

Sanity check:

```bash
.venv/bin/python -c "import fastapi, httpx, jwt, cryptography; print('ready')"
```

**Checkpoint:** prints `ready`.

---

## Step 1 · See the whole thing run once

Before dissecting it, watch the entire protocol fly past. This one command boots
the app (Part A) in a background thread and drives it with the client (Part B):

```bash
.venv/bin/python week17/authmd_adk/run_authmd_demo.py
```

You'll see 8 labelled sections scroll by: discovery → agent-verified token →
wake-time re-mint → user-claimed OTP → least privilege (a `403`) → apply →
revocation (a `401`) → audit log, ending with:

```
✅ auth.md protocol walk-through complete (offline, no model).
```

**Checkpoint:** it ends with the ✅ line. Now let's understand each section. Open
`run_authmd_demo.py` beside this tutorial — its 8 numbered blocks map 1:1 to the
steps below.

---

## Step 2 · Discovery — how an agent *finds* how to authenticate (slides A1–A2)

An agent that has never met your service needs to learn how to register. The
discovery chain is **auth.md → PRM → AS metadata**. Start the app on its own so
you can poke it by hand:

```bash
.venv/bin/python week17/authmd_adk/app_server.py
```

In a second terminal:

```bash
# A1 — the human/agent-readable entry point
curl -s http://127.0.0.1:8088/auth.md

# A2 — Protected Resource Metadata (RFC 9728): machine source of truth
curl -s http://127.0.0.1:8088/.well-known/oauth-protected-resource | python3 -m json.tool

# A2 — the agent_auth block: where to register, claim, revoke
curl -s http://127.0.0.1:8088/.well-known/oauth-authorization-server | python3 -m json.tool
```

Now ask the protected API for data **with no token** and read the response
*headers*:

```bash
curl -s -D - http://127.0.0.1:8088/sites/site-bkk-01/energy -o /dev/null
```

**Checkpoint:** you get `401 Unauthorized` and a header

```
WWW-Authenticate: Bearer resource_metadata="http://127.0.0.1:8088/.well-known/oauth-protected-resource"
```

That header is the **401 discovery hint** — even an agent that stumbled onto the
API blind now knows exactly where to look. See it in code at
`app_server.py:_require_scope()`.

> Leave this app server running for Steps 3–9.

---

## Step 3 · `POST /agent/auth` — one endpoint, two flows (slide A3)

Registration is a single endpoint that **dispatches on a `type` field**. Read
`agent_auth()` in `app_server.py`:

```python
@app.post("/agent/auth")
async def agent_auth(req: AuthRequest) -> dict:
    if req.type == "identity_assertion":   # agent-verified  → Step 4
        return _handle_verified(req)
    if req.type == "anonymous":            # user-claimed (pre-claim creds)
        return _handle_anonymous(req)
    if req.type == "verified_email":       # user-claimed (email-required) → Step 5
        return _handle_email_required(req)
```

Two real flows live behind it. Pick by asking: *is a human in the loop?*

| Flow | `type` | Human? | Used in our agent by |
|------|--------|--------|----------------------|
| **agent-verified** | `identity_assertion` | No — a provider vouches | the `read_agent` (`sites.read`) |
| **user-claimed** | `verified_email` | Yes — confirms an OTP | the approval gate (`control.write`) |

---

## Step 4 · Agent-verified — a provider vouches, no human (slide A4)

Here a **trusted provider** (think OpenAI / Anthropic / Cursor) signs an
**ID-JAG** — a JWT asserting *who the human behind the agent is*. The app
verifies that signature against the provider's JWKS and issues a token. No human
interaction.

Mint a real ID-JAG and exchange it by hand:

```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "week17/authmd_adk")
import idjag_provider, httpx

# 1) the trusted provider signs an ID-JAG for our user
id_jag = idjag_provider.mint_id_jag(
    subject="energy-ops-bot",
    email="ops@altotech.ai",
    audience="http://127.0.0.1:8088/",
)
print("ID-JAG (first 40 chars):", id_jag[:40], "...")

# 2) exchange it for a scoped token
r = httpx.post("http://127.0.0.1:8088/agent/auth", json={
    "type": "identity_assertion", "id_jag": id_jag, "scopes": ["sites.read"]})
print("token response:", r.json())

# 3) use the token on the protected API
tok = r.json()["access_token"]
data = httpx.get("http://127.0.0.1:8088/sites/site-bkk-01/energy",
                 headers={"Authorization": f"Bearer {tok}"}).json()
print("energy:", data)
PY
```

**Checkpoint:** the token response has `"flow": "agent_verified"`, a short
`"expires_in": 300`, and crucially **`"refresh_token": null`**.

> **Why no refresh token?** (Slide A4.) The agent doesn't keep a long-lived
> refresh credential — it just presents a *fresh ID-JAG* whenever it needs a new
> token. That is exactly what makes wake-time re-minting clean (Step 8). The
> verification steps — decode header → check issuer trust list → fetch JWKS →
> verify signature → validate `aud`/`exp`/`jti` → match user — are in
> `_handle_verified()`.

The signer is `idjag_provider.py`, a self-contained provider with a real RSA key
(persisted to `idjag_signing_key.pem` so the app and agent processes share it).

---

## Step 5 · User-claimed — a human confirms an OTP (slide A5)

When you *want* a human in the loop (e.g. approving a control action), use the
claimed flow. The app emails a one-time code; the credential is withheld until
the user confirms it. Walk the ceremony:

```bash
.venv/bin/python - <<'PY'
import httpx
B = "http://127.0.0.1:8088"

# 1) start the email-required claim → app "emails" an OTP, returns a claim_token
start = httpx.post(f"{B}/agent/auth", json={
    "type": "verified_email", "email": "facility.manager@altotech.ai",
    "scopes": ["control.write"]}).json()
print("started:", start)                       # note: no credential yet

# 2) the user opens their email and reads the code (demo stand-in endpoint)
otp = httpx.get(f"{B}/_demo/inbox/facility.manager@altotech.ai").json()["otp"]
print("OTP the human reads:", otp)

# 3) complete the claim with the OTP → NOW a scoped credential is issued
done = httpx.post(f"{B}/agent/auth/claim/complete", json={
    "claim_token": start["claim_token"], "otp": otp}).json()
print("claimed:", done)
PY
```

**Checkpoint:** step 1 returns `"credential": null` (withheld), and step 3
returns an `access_token` with `["control.write"]`.

> The `_demo/inbox` endpoint is a **teaching stand-in** for a real email — in
> production an OTP must never appear in an API response. The app stores only
> SHA-256 *hashes* of the OTP and claim tokens (`app_server.py:_send_otp`).

---

## Step 6 · Least privilege — scopes are enforced (slide A6 / B4)

A token only works for what it was scoped to. Prove it: try the **`sites.read`**
token from Step 4 on the **write** endpoint.

```bash
# reuse a sites.read token, then attempt a control.write call
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "week17/authmd_adk")
import idjag_provider, httpx
B = "http://127.0.0.1:8088"
jag = idjag_provider.mint_id_jag(subject="x", email="ops@altotech.ai",
                                 audience=f"{B}/")
tok = httpx.post(f"{B}/agent/auth", json={
    "type": "identity_assertion", "id_jag": jag, "scopes": ["sites.read"]}).json()["access_token"]
r = httpx.post(f"{B}/sites/site-bkk-01/setpoint", json={"setpoint_c": 25.5},
               headers={"Authorization": f"Bearer {tok}"})
print(r.status_code, r.json())
PY
```

**Checkpoint:** `403 {'error': "token lacks required scope 'control.write'"}`.
A read credential can never write. This is what makes per-sub-agent scoping
(Step 14) meaningful.

---

## Step 7 · Revocation (slide A6)

Tokens are revocable. Mint one, use it, revoke it, watch the next call fail:

```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "week17/authmd_adk")
from authmd_client import AuthMdClient, AuthGrant
B = "http://127.0.0.1:8088"
c = AuthMdClient(B)
cred = c.acquire(AuthGrant("altotech_read", "agent_verified", ["sites.read"],
                           subject="x", email="ops@altotech.ai"))
print("revoke:", c.revoke(token=cred.token))
import httpx
print("after revoke:",
      httpx.get(f"{B}/sites/site-bkk-01/energy",
                headers={"Authorization": cred.bearer()}).status_code)
PY
```

**Checkpoint:** `revoke: {'revoked': True}` then `after revoke: 401`.

---

## Step 8 · Wake-time re-mint — the punchline (slides B2–B3)

This is *the* idea. The agent stores a **grant** (durable, token-free) and calls
`acquire()` to get a **fresh token on demand**. Two `acquire()` calls from the
same grant return two *different* tokens — so an expired token is never a
problem.

```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "week17/authmd_adk")
from authmd_client import AuthMdClient, AuthGrant

# the DURABLE grant — this is what lives in ADK session state. No token in it.
grant = AuthGrant(service="altotech_read", flow="agent_verified",
                  scopes=["sites.read"], subject="energy-ops-bot",
                  email="ops@altotech.ai")
print("grant persisted as:", grant.as_state())

c = AuthMdClient("http://127.0.0.1:8088")
t1 = c.acquire(grant).token      # "wake" #1
t2 = c.acquire(grant).token      # "wake" #2, days later
print("token 1:", t1[:24], "…")
print("token 2:", t2[:24], "…")
print("different? ->", t1 != t2)
PY
```

**Checkpoint:** `different? -> True`. Read `AuthMdClient.acquire()` in
`authmd_client.py` — each call signs a *new* ID-JAG and exchanges it. Notice
`AuthGrant.as_state()` contains scopes and references but **no token**: that's
slide B2, "store the grant, not the token."

---

## Step 9 · The audit trail

Every state change the app made during your poking is recorded:

```bash
curl -s http://127.0.0.1:8088/_demo/audit | python3 -m json.tool
```

**Checkpoint:** you see `verified_issued`, `otp_sent`, `claim_issued`,
`setpoint_applied`, `revoked_token` events. Then stop the app server (`Ctrl-C`).

🎉 **You now understand the whole auth.md protocol.** Part 2 plugs it into a
long-running agent that actually sleeps and wakes.

---

# Part 2 — the long-running ADK agent

## Step 10 · Setup for the agent

Install the rest (ADK + a model driver):

```bash
uv pip install -r week17/authmd_adk/requirements.txt
```

The agent is model-agnostic. By default it uses the repo's Alto gateway; no
extra setup needed. To switch: `export AUTHMD_PROVIDER=anthropic` (uses the
repo's `ANTHROPIC_API_KEY`) or `=gemini` (needs `GOOGLE_API_KEY`). See
`config.py`.

---

## Step 11 · The work order — a durable state machine (slide B1)

Open `work_order.py`. The agent's behaviour is driven by an explicit
`current_step`, **not** by chat history:

```
START ──analyze (sites.read)──▶ ANALYZED ⏸  (waiting on human approval)
                                   │  /webhooks/request_approval → app emails an OTP
                                   │  /webhooks/approved (OTP)    → claim completes
                                   ▼
                              APPROVED ──apply (control.write)──▶ APPLIED ──▶ COMPLETED
```

And look at `initial_grants()` — the `auth_grants` block that lives in session
state. **Two grants, least privilege**, one per path:

```python
"altotech_read":  {"flow": "agent_verified", "scopes": ["sites.read"],  ...}
"altotech_write": {"flow": "user_claimed",   "scopes": ["control.write"], ...}
```

No tokens — just what each path is allowed to do (slide B2).

---

## Step 12 · Start both services

The agent (Part B) calls the app (Part A) over HTTP, so run both. **Three
terminals**, all from the repo root:

```bash
# terminal 1 — Part A: the agent-ready Energy API
.venv/bin/python week17/authmd_adk/app_server.py

# terminal 2 — Part B: the long-running ADK agent service
.venv/bin/python week17/authmd_adk/agent_server.py
```

**Checkpoint:** terminal 2 prints `✅ work-order service up — model=…`. Both
respond to `curl -s http://127.0.0.1:8088/healthz` and `:8089/healthz`.

---

## Step 13 · Drive one work order end to end

In terminal 3:

```bash
.venv/bin/python week17/authmd_adk/run_full_demo.py
```

Watch the six steps print. The important moments:

- **Step 2 (kick off):** the agent transfers to `read_agent`, which mints a
  `sites.read` token *autonomously* (agent-verified, no human) and parks at
  `ANALYZED` recommending a setpoint.
- **Step 3 (try to skip):** you ask it to apply now; it **refuses** — the
  idle-time safety gate. This is the agent honouring `current_step`.
- **Step 4 (request_approval):** the app emails the facility manager an OTP; the
  grant's `claim_ref` is persisted (`claimed: True` appears).
- **Step 5 (approved):** the OTP is confirmed → the claim completes → a fresh
  `control.write` token is injected via `state_delta` → the `apply_agent` writes
  the setpoint → `COMPLETED`.

**Checkpoint:** the run ends with `✅ work order COMPLETED` and
`recommended=25.5°C  applied=25.5`.

---

## Step 14 · Where the auth plugs in (slides B3–B4)

Now connect what you saw to three spots in the code:

1. **`agent.py:AuthInjector`** — the `before_tool_callback`. Right before a tool
   hits the API, it picks the tool's grant, **re-mints a fresh token**, and drops
   it into state. For the read path it acquires autonomously; for the write path
   it expects the token the approval webhook injected (slide B3).

2. **`resume_handler.py:approved()`** — the approval gate **is** the user-claimed
   flow (slide B4). It completes the claim, mints the `control.write` token, and
   advances the step **in one `state_delta`** — authorise + resume together.

3. **`agent.py:build_read_agent` / `build_apply_agent`** — least privilege per
   sub-agent (slide B4): `read_agent` carries only `sites.read`, `apply_agent`
   only `control.write`. Neither can do the other's job (you proved the `403` in
   Step 6).

Confirm the auth flows actually fired over the wire:

```bash
curl -s http://127.0.0.1:8088/_demo/audit | python3 -m json.tool
```

**Checkpoint:** `verified_issued` (the read_agent), then `otp_sent` /
`claim_issued` (the approval), then `setpoint_applied`.

---

## Step 15 · Prove durability (the long-running claim)

The whole point is surviving restarts. The session is in SQLite
(`work_order_sessions.db`), so a work order paused at `ANALYZED` outlives the
process. Try it:

1. Create + kick off a work order, but **stop before approving**:
   ```bash
   .venv/bin/python - <<'PY'
   import httpx, uuid
   sid = "wo-" + uuid.uuid4().hex[:8]; print("session:", sid)
   A = "http://127.0.0.1:8089"
   httpx.post(f"{A}/work_order", json={"session_id": sid}, timeout=180)
   httpx.post(f"{A}/chat", json={"session_id": sid,
              "message": "Begin the energy work order."}, timeout=180)
   print(httpx.get(f"{A}/status/energy_ops/{sid}").json()["current_step"])
   open("/tmp/authmd_sid.txt","w").write(sid)
   PY
   ```
   **Checkpoint:** prints `ANALYZED`.

2. **Restart `agent_server.py`** (Ctrl-C terminal 2, run it again). The in-memory
   state is gone; the SQLite session is not.

3. Resume the *same* session — it picks up exactly where it paused:
   ```bash
   .venv/bin/python - <<'PY'
   import httpx
   sid = open("/tmp/authmd_sid.txt").read().strip()
   A, B = "http://127.0.0.1:8089", "http://127.0.0.1:8088"
   print("after restart, step =",
         httpx.get(f"{A}/status/energy_ops/{sid}").json()["current_step"])
   httpx.post(f"{A}/webhooks/request_approval", json={"session_id": sid}, timeout=180)
   otp = httpx.get(f"{B}/_demo/inbox/facility.manager@altotech.ai").json()["otp"]
   r = httpx.post(f"{A}/webhooks/approved", json={"session_id": sid, "otp": otp}, timeout=180)
   print("final step =", r.json()["current_step"])
   PY
   ```
   **Checkpoint:** `after restart, step = ANALYZED` then `final step = COMPLETED`
   — the agent woke from a cold restart with a freshly re-minted token. **That is
   the long-running + auth.md story in one run.**

---

## Where to go next

- **Swap the model:** re-run Step 13 with `AUTHMD_PROVIDER=anthropic`.
- **Add a scope:** add `reports.read` to `SCOPES_SUPPORTED` in `app_server.py`,
  give a sub-agent a grant for it, and watch least privilege hold.
- **Fall back to plain OAuth:** `acquire()` (`authmd_client.py`) is the seam —
  for a service with no `auth.md`, it's where you'd do a refresh-token exchange
  instead. The agent never notices.
- **Read the reality check** in `README.md` — what's production-faithful here
  vs. stubbed for teaching (in-process token store, the `_demo/inbox` email
  stand-in, rate-limiting as a note).

---

## Troubleshooting

| Symptom | Fix |
|--------|-----|
| `app_server did not come up` in Step 1 | Port 8088 busy — set `AUTHMD_APP_PORT=8093` and retry. |
| `401` from `/agent/auth` in Part 2 | The app and agent processes disagree on the signing key. Stop both, delete `week17/authmd_adk/idjag_signing_key.pem`, restart both (they'll regenerate + share it). |
| `run_full_demo.py` says a server isn't reachable | Start `app_server.py` *and* `agent_server.py` first (Step 12). |
| Agent turn hangs / errors on the model | Check `AUTHMD_PROVIDER` and that the corresponding key is set; tail `agent_server.py`'s terminal for the traceback. |
| Want a clean slate | Stop both servers; delete `work_order_sessions.db` and `idjag_signing_key.pem`. |
