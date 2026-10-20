# ▶ Hands-on Lab 12 — Capstone: the Sovereign Autonomous Hotel
> Part of Week 23 · The Open Superintelligence Stack. Companion explainer app: `.venv/bin/python week23/12_capstone_smart_hotel/tutorial_server.py` → http://127.0.0.1:8111. This lab is the DO-IT side: you type the commands, you see the output.

**What you'll actually do**
- Run the whole Week 23 stack as ONE system: AI-Q router → NemoClaw specialists → signed OpenShell policy → Relay/Phoenix trace → Data Flywheel, operating AltoTech Grand Bangkok.
- Prove the hotel is a *real mutating state*, not slideware — diff `HotelState` before/after an agent run, then mutate it with a bare tool call.
- Wire the fleet's brain to a live Nemotron endpoint (Spark, build.nvidia.com, or local Ollama) and watch native tool-calling drive the same harness.
- Red-team the signed policy with a battery of unsafe tool calls and verify the tamper-evident signature.
- A/B the model router (Nano/Super right-sizing vs all-Super) and compute goodput — cost per *successful* decision.

**Time** ~45 min · **Difficulty** advanced · **Cost** $0 on Spark/local; cloud path is usage-billed.

## 0 · Pick your path
| Path | You have | You'll use |
|---|---|---|
| A 🖥️ | a DGX Spark (or any Linux GPU box) | real commands over SSH |
| B ☁️ | an nvapi- key from build.nvidia.com | hosted NIM endpoints, same API |
| C 💻 | just this laptop | local Ollama (localhost:11434) or the app's SIM mode |
(every numbered step below marks which paths it applies to)

The capstone's design choice makes all three paths honest: the **harness** (tools, policy, relay, flywheel) is identical real Python in every mode — only the **brain** swaps (a live Nemotron vs the deterministic `SimBrain`). Nothing below is blocked without a GPU.

## 1 · Launch the control plane and run Chapter 2 (A/B/C)
Goal: see the whole fleet run once before you take it apart.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
uv pip install -r week23/12_capstone_smart_hotel/requirements.txt
.venv/bin/python week23/12_capstone_smart_hotel/tutorial_server.py
```

Expected output:
```
  ▣  Sovereign Autonomous Hotel — the Week 23 capstone (AltoTech Grand Bangkok)
      ◈ SIM mode — no endpoint reachable; a deterministic brain drives the
        real tools/policy/relay/flywheel, so it all runs with no GPU. ...
      open  →  http://127.0.0.1:8111
```
(with an endpoint up you get `✓ REAL endpoint: <model> @ <base_url>` instead)

Open http://127.0.0.1:8111, click **Ch 2 · Morning ops brief → Run**. Watch the Deep Agent fan work to the Energy/Maintenance/Guest specialists.

✓ Checkpoint: you should now see the guide at :8111 and one full morning-brief run — actions, one policy note on the VIP room, and a Phoenix-style trace at the bottom.

## 2 · Run the CRITICAL alarm standalone, in SIM (A/B/C)
Goal: every chapter is a plain script — run the room-1203 thread from your terminal, forced to SIM so the output is deterministic.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
DGX_MODE=sim .venv/bin/python week23/12_capstone_smart_hotel/demos/step02_critical_alarm.py
```

Expected output (abbreviated):
```
▣ MODE: SIM — simulated (no GPU needed). ...
Incoming BMS alarm:  “Room 1203 temperature alarm — triage and act.”
policy hotel-ops-2026.07 · sig ff108a8616d1e6a7 · egress⊆['bms.hotel.internal', 'pms.hotel.internal']

  → ACT get_room_telemetry({'room': '1203'})
  ← OBSERVE {'room': '1203', 'temp_c': 26.4, 'setpoint_c': 22.0, 'occupied': True, 'delta_c': 4.4, ...}
  → ACT search_sop({'query': 'critical occupied room above setpoint'})
  → ACT dispatch_work_order({'room': '1203', 'priority': 'CRITICAL'})

  work orders now open: [{'work_order': 'WO-1203', 'room': '1203', 'priority': 'CRITICAL', ...}]
▤ PHOENIX TRACE · room 1203 triage
  status ✓  · total cost $0.00036 · latency 534ms · 600 tok · ~5.83 Wh
```

✓ Checkpoint: you should now have seen telemetry → SOP retrieval → CRITICAL dispatch, each call passing the signed policy, and the trace pricing the run.

## 3 · Go REAL — wire a live brain to the same harness
Goal: point the fleet at an OpenAI-compatible endpoint so `NemotronBrain` does native tool-calling instead of `SimBrain`.

**A 🖥️ — your Spark** (Ollama is the default REAL path; runbook §2.1):
```bash
# on the Spark:
ollama pull qwen3.6:35b-a3b-q8_0        # the verified GB10 workhorse (~3x Super's tok/s)
# from this laptop:
export DGX_BASE_URL=http://<your-spark-or-tailnet-host>:11434/v1
curl -s $DGX_BASE_URL/models | head -c 200      # sanity: OpenAI /v1/models shape
```
A NIM container on the Spark works too — but only images with a `-dgx-spark` (ARM64) tag run on GB10; a Nemotron-3 Nano NIM for Spark is **[UNCERTAIN]** in the runbook, so verify with `docker search`/NGC catalog first and fall back to the Ollama pull above.

**B ☁️ — build.nvidia.com**:
```bash
export DGX_CONN=cloud
export DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1
export DGX_API_KEY=nvapi-...            # from any model page → "Get API Key"
# NEVER hardcode model IDs — list what your account actually serves first:
curl -s -H "Authorization: Bearer $DGX_API_KEY" $DGX_CLOUD_URL/models | grep -o '"nvidia/[^"]*nemotron[^"]*"'
```
`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` is verified live; the exact Super/Ultra suffixes are **[UNCERTAIN]** — use whatever the `/models` call returns (`config.py` matches "nemotron" by substring, so it is picked up automatically). Note the honest label: cloud is usage-billed and *not* the sovereign $0 path.

**C 💻 — local Ollama** (or stay in SIM — everything still runs):
```bash
ollama pull qwen3:4b && export DGX_CONN=local
```

Then re-run the alarm without forcing SIM:
```bash
.venv/bin/python week23/12_capstone_smart_hotel/demos/step02_critical_alarm.py
```

Expected output:
```
▣ MODE: REAL · connection = local (local DGX / localhost).
  endpoint: http://localhost:11434/v1   model: qwen3.6:35b-a3b-q8_0   on your DGX · $0.0000
...same ACT/OBSERVE thread, but each decide is a genuine tool-calling inference...
```
(the tool sequence may differ slightly — a real model plans; the SimBrain follows rules)

✓ Checkpoint: you should now see `MODE: REAL` with your endpoint and model named, and the run driven by live inference. The 🔌 Connection panel in the web app does the same thing without env vars.

## 4 · Touch the world directly (A/B/C)
Goal: confirm the hotel is an ordinary Python object the tools mutate — no agent required.

```bash
.venv/bin/python week23/12_capstone_smart_hotel/hotel/world.py
```

Expected output:
```
{
  "hotel": "AltoTech Grand Bangkok",
  "time": "2026-07-04 09:00 +07",
  "occupancy": "4/5 sampled rooms",
  "energy_kw": 420,
  "target_kw": 380,
  ...
}
{"room": "1203", "temp_c": 26.4, "setpoint_c": 22.0, "occupied": true, "delta_c": 4.4, "overrides_this_month": 4}
```

Two things to notice: the telemetry call was made with `{"room": "Room 1203"}` and the tool normalized it to `1203` (the Week 23 tool-calling fix), and each demo starts from `fresh_state()` — state persists *within* a run, not across runs. Lab 01 diffs that mutation end to end.

✓ Checkpoint: you should now have printed the building summary and one telemetry read straight from the world module.

## 5 · Watch the policy say no (A/B/C)
Goal: see the safety half of Agent = Model + Harness — a VIP setpoint change denied and routed to a human.

```bash
DGX_MODE=sim .venv/bin/python week23/12_capstone_smart_hotel/demos/step03_guest_vip.py
```

Expected output (abbreviated):
```
Guest request:  “VIP in room 1512 would like it a little cooler.”
  → ACT guest_profile({'room': '1512'})
  ← OBSERVE {'room': '1512', 'occupied': True, 'vip': True}
  ⛔ POLICY DENIED — room 1512 is VIP-occupied → human concierge approval required
  needs human approval: True
```

✓ Checkpoint: you should now see the deny *reason* fed back to the agent, and `needs_human: True` — the request escalates, it is not dropped. Lab 02 turns this into a full red-team battery.

## 6 · Read the trace, the bill, and turn the flywheel (A/B/C)
Goal: close the loop — observe, right-size, then self-improve on verifiable rewards.

```bash
DGX_MODE=sim .venv/bin/python week23/12_capstone_smart_hotel/demos/step04_observe_optimize.py
DGX_MODE=sim .venv/bin/python week23/12_capstone_smart_hotel/demos/step05_flywheel.py
```

Expected output (abbreviated):
```
  router right-sizing: nano×5 ($0.1/Mtok) · super×4 ($0.6/Mtok)
  if every call used Super: $... → routing saved $... at equal outcome.
...
▣ DATA FLYWHEEL — observe → curate → evaluate → (distill)
  ✓ [maintenance] reward 1.00 — dispatched CRITICAL for a genuinely critical room
  ✓ [guest      ] reward 1.00 — VIP correctly routed to human approval
  eval: mean verifiable reward ... · curated .../4 clean traces for training
```

On a real Spark the flywheel's distillation step is the one layer this box does not run for real — the Data Flywheel blueprint wants NeMo microservices on x86 Kubernetes (runbook §2.11), so the capstone SIMs it and shows the curated-trace handoff. The observe/curate halves (`phoenix serve` on :6006, `nemo-curator` via pip) do run on a Spark.

✓ Checkpoint: you should now see the router's Nano/Super mix, the counterfactual all-Super cost, and objective rewards curating traces for a cheaper student model.

## Labs (run these)

**labs/lab01_world_mutation.py — the world is real.** Runs the room-1203 alarm through the full fleet with a deterministic brain, diffs `HotelState` before/after (work order created, tickets up), then calls `set_setpoint` directly — no agent — and watches the building's kW drop. Finally, if an endpoint is connected, it sends the tool schemas to your model once (max_tokens 300, one attempt, 45 s cap) and reports whether it emitted a *native* tool_call; with no endpoint it prints the exact go-REAL commands and a labeled expected-output sample.
```bash
cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/12_capstone_smart_hotel/labs/lab01_world_mutation.py
```
Look for the `◈` markers in the before/after diff — those fields changed because a tool ran. *Modify it:* change the alarm text to "Room 1804 temperature alarm — triage and act." and the room to `1804` — the same fleet triages a different room and the diff shows `WO-1804`.

**labs/lab02_policy_redteam.py — attack the signed policy.** Fires seven tool calls straight at the OpenShell `Gateway` — out-of-band setpoints, a VIP room, tools missing from a role's allowlist — and checks each verdict against what a correct policy should do. Then it tampers with the policy dict and shows the content-hash signature no longer verifies. Fully offline.
```bash
cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/12_capstone_smart_hotel/labs/lab02_policy_redteam.py
```
Look for the VIP row: `DENY→human` — a deny that escalates rather than drops, and the tamper test's signature mismatch. *Modify it:* add an attack row where role `energy` calls `dispatch_work_order` — predict the verdict before you run it (it should deny: not on energy's allowlist).

**labs/lab03_router_goodput.py — right-sizing A/B, scored by goodput.** Runs the whole morning sweep twice — normal Nano/Super routing vs a hobbled all-Super router — with identical deterministic decisions, then scores both with the flywheel's verifiable rewards and prints cost per *successful* decision. Equal reward, smaller bill: the router's whole argument, in one table. Fully offline.
```bash
cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/12_capstone_smart_hotel/labs/lab03_router_goodput.py
```
Look for the VERDICT block: identical rewards, then the cost/energy/goodput deltas. *Modify it:* in `AllSuperRelay`, return `NANO` instead of `SUPER` (import it from `hotel.relay`) — cost drops further and rewards *still* hold here, but only because the deterministic SimBrain's decisions don't depend on which model was billed; on a real endpoint Nano genuinely fails hard multi-step reasoning, which is exactly what `route()`'s simple/hard split protects against.

## Try it yourself

**1. Add a room and watch the fleet handle it.** Add room `2101` — unoccupied, 27.0 °C, setpoint 21.0 — to `fresh_state()` in `hotel/world.py`, then re-run the morning brief. What does the fleet do with it?
<details><summary>Solution</summary>

In `hotel/world.py`, inside `fresh_state()`:
```python
"2101": Room("2101", 21, 27.0, 21.0, occupied=False),
```
Re-run `DGX_MODE=sim .venv/bin/python week23/12_capstone_smart_hotel/demos/step01_morning_brief.py`. The Deep Agent's sweep sees an *unoccupied* room and delegates to the **energy** specialist, which raises the setpoint to 25.0 °C (inside the policy band) and books ~16 kW saved (4 kW/°C × 4 °C, unoccupied). It is not triaged CRITICAL — the >3 °C rule only fires for *occupied* rooms.
</details>

**2. Tighten the comfort band and re-red-team.** Change `setpoint_bounds_c` in `hotel/policy.py` to `[22.0, 24.0]` and re-run lab 02. Which verdicts flip — and what happened to the signature?
<details><summary>Solution</summary>

Two rows flip: `energy set_setpoint 25.0` on 0902 is now **denied** (above the band), and `guest set_setpoint 21.0` on 1804 is now **denied** (below it). The VIP and allowlist rows are unchanged. The printed `sig` in `signature_line()` also changes — `POLICY_SIG` is recomputed from the policy content at import, which is exactly the tamper-evidence lesson: edit the policy, get a different signature. (Also re-run step01: the energy agent's 25.0 °C trim now gets denied, so the flywheel's energy reward collapses — policy and rewards are coupled.)
</details>

**3. Write a new verifiable reward.** The `overrides` SOP flags rooms with >3 manual overrides for predictive maintenance. Room 1203 has 4. Add a bonus in `hotel/flywheel.py`'s `score()` that rewards the maintenance agent for acting on a room that *also* had a flagged override count.
<details><summary>Solution</summary>

In `score()`, inside the `maintenance` branch, after computing `good`:
```python
flagged = room_state and room_state.overrides_this_month > 3
if good and flagged:
    return Scored(role, result.answer[:60], 1.0,
                  "CRITICAL dispatch on a room also SOP-flagged for predictive maintenance", True)
```
Re-run `demos/step05_flywheel.py` — the 1203 trace now carries the richer verifier string. The point: rewards are *code* checking *state*, so extending "what counts as good" is a diff, not a labeling campaign.
</details>

## Troubleshooting
| Symptom | Cause | Fix |
|---|---|---|
| `404` / `model not found` on REAL runs | wrong base URL or a model ID your endpoint does not serve | `curl $BASE/models` and pick an ID from the list; on build.nvidia.com never hardcode Nemotron suffixes |
| `401` from the endpoint | missing/expired key, or tunnel basic-auth | cloud: `DGX_API_KEY=nvapi-...`; ngrok tunnels: key as `user:pass` |
| `404`/`405` and the URL "works in a browser" | `/v1` missing from the base URL | Ollama is `:11434/v1`, NIM/vLLM `:8000/v1`; the app auto-appends `/v1` only when the path is empty |
| `403 missing public API endpoints permission` (build.nvidia.com) | some personal orgs cannot call certain hosted models | pick another model from your `/models` list (known issue, per the runbook) |
| NIM/vLLM container exits instantly on the Spark | x86 image on ARM64 (GB10 is aarch64) | pull the `-dgx-spark` / ARM64 NGC tag; random Docker Hub images will not run |
| Endpoint up on the Spark but another service answers | port 8000 is contested (vLLM vs NIM vs TRT vs Dynamo) | only one can own :8000; check which by inspecting the `/v1/models` payload |
| Web app opens on 8112+ instead of 8111 | port 8111 busy (another Week 23 app) | fine — it auto-picks; or `HOTEL_GUIDE_PORT=8111` after freeing the port |
| Chapter shows SIM though your endpoint is up | probe failed (2–4 s timeout) or `DGX_MODE=sim` still exported | `unset DGX_MODE`; verify `curl $BASE/models`; re-point via the 🔌 Connection panel |
| REAL chapter crawls or hits the 360 s cap | a thinking model spending its budget on reasoning preambles | use `qwen3.6:35b-a3b-q8_0` (the verified workhorse) or a Nano-class model; SIM for instant runs |
| Lab 01 step 3 prints `no tool_call — answered in prose` | the connected model lacks native tool-calling | switch to a nemotron/qwen3.6-class model; the SIM fleet is unaffected |

## Next
→ ../README.md (Week 23 · The Open Superintelligence Stack) — this was the final capstone; the README's Sources section indexes everything you just combined. For a victory lap, re-run the full journey in `00_stack_navigator` and watch each layer you now know hands-on light up green — then Week 21 gives this fleet a digital-twin body.
