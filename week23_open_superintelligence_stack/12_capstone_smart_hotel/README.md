# ▣ Capstone — Sovereign Autonomous Hotel Operations

**AltoTech Grand Bangkok**, now run by a self-improving, sovereign agent fleet on your DGX —
the entire Week 23 stack combined into **one running system**. It continues the smart-hotel
demo from earlier weeks (Week 11 ops command center, Week 14 HVAC multi-agent + memory,
Week 15 hotel GraphRAG, and the Week 23 room-1203 tool-calling demo) and makes it real:
**the tools genuinely mutate a hotel state**, so swapping the SIM brain for a Nemotron
endpoint gives you a production-shaped autonomous building.

> **Agent = Model + Harness.** The model reasons; the harness (orchestration, skills, tools,
> policy, observability, self-improvement) makes it a trustworthy operator. This capstone is
> the whole harness, wired around open Nemotron models.

---

## Run it

```bash
uv pip install -r week23/12_capstone_smart_hotel/requirements.txt
.venv/bin/python week23/12_capstone_smart_hotel/tutorial_server.py     # → http://127.0.0.1:8111
```

Open the URL and click the six chapters. With no GPU it runs in **SIM** (a deterministic
brain drives the *real* tools/policy/relay/flywheel — $0). Point the **🔌 Connection** panel
at a Nemotron **NIM / DGX** endpoint (local / tunnel / cloud) to run the specialists on a
real model with native tool-calling. Every demo is also runnable standalone:

```bash
cd week23/capstone_smart_hotel
DGX_MODE=sim ../../.venv/bin/python demos/step02_critical_alarm.py
```

---

## What each chapter shows (and which Week 23 app it uses)

| Ch | Scenario | Week 23 apps exercised |
|----|----------|------------------------|
| 1 | The sovereign autonomous hotel — architecture | all |
| 2 | **Morning ops brief** — AI-Q Deep Agent plans a sweep, fans out to specialists | 7 · 10 · 6 · 8 |
| 3 | **CRITICAL alarm, room 1203** — Maintenance agent triages via SOP RAG, dispatches under policy | 10 · 9 · 6 · 8 |
| 4 | **VIP request** — the signed policy blocks an unsafe autonomous setpoint change → human | 6 |
| 5 | **Observe & optimize** — Phoenix-style trace, router right-sizing, inference economics | 8 · 11 |
| 6 | **Self-improve** — verifiable rewards + Data Flywheel curate → distill a cheaper Nano | 3 · 5 |

---

## Architecture

```
   🏨 event / alarm / request
        │
        ▼
   Intent Router  (Nemotron Nano)                         ── AI-Q, App 5
        │  simple → specialist   ·   hard → escalate
        ▼
   Deep Agent  (Nemotron Super) — plan & delegate         ── AI-Q, App 5
        │
        ├──► Energy·HVAC agent  ┐
        ├──► Maintenance agent  ├─ NemoClaw specialists    ── App 6
        └──► Guest agent        ┘   persona+model+skills+tools
                 │
                 ▼   each tool call…
        🛡 OpenShell Gateway  (signed policy)              ── App 7
           allow / deny (setpoint bounds · VIP protection · egress allowlist)
                 │ allow
                 ▼
        Tools:  get_room_telemetry · set_setpoint · dispatch_work_order
                guest_profile · search_sop (NeMo Retriever RAG over SOPs)  ── App 4
                 │  (mutates real HotelState)
                 ▼
        NeMo Relay — observes every span, right-sizes the model            ── App 8
                 │→ Phoenix / Datadog / LangSmith (OTel)  +  economics      ── App 9
                 ▼
        Data Flywheel — verifiable reward per decision → curate → distill   ── Apps 11 · 10

   Model served by NIM, scaled by Dynamo, on a DGX Spark                    ── Apps 1 · 2 · 3
```

### Files
```
config.py · view.py · sim.py      shared Week 23 infra (connection, render, SIM models)
tutorial_server.py                 the web control plane (serves the guide, runs the demos)
static/guide.html                  the interactive guide + animated architecture visuals
hotel/
  world.py       HotelState, the tools, the SOP corpus (the real, mutating world)
  policy.py      OpenShell signed policy + Gateway (the safety choke-point)
  relay.py       NeMo Relay: spans, Phoenix trace, model router, economics
  brain.py       NemotronBrain (REAL tool-calling) · SimBrain (deterministic, no GPU)
  agents.py      NemoClaw Specialist loop + AI-Q Orchestrator (router + deep agent)
  flywheel.py    verifiable rewards + curation + distillation report
  runtime.py     wires it all together
demos/           step01…step05 — the runnable scenarios the web app executes
```

**The key design choice:** the *harness* (tools, policy, relay, flywheel) is real and identical
in SIM and REAL — only the **brain** swaps. That is what makes it "usable in real life": replace
`hotel/world.py`'s tools with your BMS/PMS calls, sign `hotel/policy.py` with your org key, point
the connection at your Nemotron NIM, and the same code runs a real building — autonomous,
self-improving, and sovereign: nothing leaves the box.
