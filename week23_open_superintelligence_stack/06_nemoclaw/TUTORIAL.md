# ▶ Hands-on Lab 06 — NemoClaw: build specialized agents

> Part of Week 23 · The Open Superintelligence Stack. Companion explainer app: `.venv/bin/python week23/06_nemoclaw/tutorial_server.py` → http://127.0.0.1:8105. This lab is the DO-IT side: you type the commands, you see the output.

**What you'll actually do**
- Author a NemoClaw agent SPEC by hand — base model + persona + skills + tools + signed policy — and run it through a 5-point lint (including Spark fit-math).
- A/B the persona: the same base model answers the same question bare vs wearing the spec, so you *see* composition-before-fine-tuning.
- Watch a real tool-call loop where attached skills give the specialist hands (demos/step02).
- Put an OpenShell-style policy gate in front of a honeypot tool and count the denials — naive runtime vs gated runtime, same model, same task.
- Route a batch of tasks across a fleet of specialists, score the model-as-supervisor against a keyword baseline, and do the Nano-vs-Super right-sizing math.

**Time** ~45 min · **Difficulty** intermediate · **Cost** $0 on Spark/local; cloud path is usage-billed.

## 0 · Pick your path

| Path | You have | You'll use |
|---|---|---|
| A 🖥️ | a DGX Spark (or any Linux GPU box) | real commands over SSH |
| B ☁️ | an nvapi- key from build.nvidia.com | hosted NIM endpoints, same API |
| C 💻 | just this laptop | local Ollama (localhost:11434) or the app's SIM mode |

(every numbered step below marks which paths it applies to)

One honesty note up front: NemoClaw and OpenShell are *young* NVIDIA projects. The DGX Spark runbook's verdict is **SIM** for their runtime pieces — "no public installable verified". So this tutorial teaches the spec/equip/guard/fleet workflow with real model calls and real gate code, and treats the NemoClaw CLI itself as something you verify first:

```bash
git clone https://github.com/NVIDIA/NemoClaw && cat NemoClaw/README.md
```

If the repo's README documents an installer, follow it; until you've verified that yourself, everything here runs against plain OpenAI-compatible endpoints — which is what the framework wraps anyway.

## 1 · Launch the companion app and confirm your mode  (A · B · C)

Goal: get the explainer running and see whether you're REAL or SIM before touching anything else.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
.venv/bin/python week23/06_nemoclaw/tutorial_server.py
```

Expected output:

```
  ▣  NVIDIA NemoClaw — build specialized agents on your DGX
      ◈ SIM mode — no endpoint reachable, simulating the base model.
        author + run specialized agents with no GPU. Go REAL anytime:
        ollama run qwen3.6:35b-a3b-q8_0   (or set DGX_BASE_URL)
      open  →  http://127.0.0.1:8105
```

(If you see `✓ REAL base model: … @ …` instead, an endpoint is already up — skip step 2.)

✓ Checkpoint: you should now see the guide at http://127.0.0.1:8105 with Ch 1–5 listed, and know your mode.

## 2 · Wire up a base model  (A / B / C — pick one)

Goal: give the specialists a real brain. All three paths end at the same OpenAI-compatible `/v1` API — that's the point.

**A 🖥️ — Spark on your tailnet.** On the Spark (via the app's 🖥️ DGX console or plain SSH):

```bash
ollama pull qwen3.6:35b-a3b-q8_0     # the verified GB10 workhorse (~3× Super's tok/s)
ollama pull nemotron-3-nano          # the reasoning tier
```

Then on the laptop, before launching the app or labs:

```bash
export DGX_BASE_URL=http://your-spark.your-tailnet.ts.net:11434/v1
```

**B ☁️ — build.nvidia.com.** Get a key (free developer account, key starts `nvapi-`), then:

```bash
export DGX_CONN=cloud
export DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1
export DGX_API_KEY=nvapi-...
```

Nemotron 3 Super/Ultra cloud IDs are **[UNCERTAIN]** in the runbook — verify what your account can see before pinning a model:

```bash
curl -s https://integrate.api.nvidia.com/v1/models \
  -H "Authorization: Bearer $DGX_API_KEY" | grep -o '"id":"[^"]*nemotron[^"]*"'
```

`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` is verified; `nvidia/nvidia-nemotron-nano-9b-v2` and `meta/llama-3.1-8b-instruct` are stable fallbacks. `config.py` matches "nemotron" by substring, so a live ID is picked up automatically.

**C 💻 — local Ollama (or nothing).**

```bash
ollama pull qwen3.6:35b-a3b-q8_0     # or any model you already have: gemma4, llama3.1:8b …
```

No Ollama at all? Do nothing — SIM mode runs every chapter and lab with honest `[simulated]` labels.

Verify whichever path you chose:

```bash
curl -s ${DGX_BASE_URL:-http://localhost:11434/v1}/models | head -c 300
```

Expected output:

```
{"object":"list","data":[{"id":"qwen3.6:35b-a3b-q8_0","object":"model", …
```

✓ Checkpoint: you should now have `/v1/models` answering (paths A/B/C-with-Ollama), or a deliberate decision to run SIM.

## 3 · Meet the spec — an agent is authored, not coded  (A · B · C)

Goal: read one complete NemoClaw agent spec and hear its persona speak.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
.venv/bin/python week23/06_nemoclaw/demos/step01_define_agent.py
```

Expected output (abbreviated, REAL mode):

```
  PART 1 — Define a specialized agent   [BEGINNER]
▣ MODE: REAL · connection = local (local DGX / localhost).
A NemoClaw agent = base model + persona + skills + tools + signed policy.
{
  "name": "hvac-reliability-expert",
  "base_model": "nemotron-3-nano:30b-a3b",
  …
  "policy": {"sandbox": "openshell", "egress_allowlist": ["telemetry.internal"], …}
}
  · ANSWER:
I am the HVAC reliability engineer for the hotel portfolio; I diagnose comfort and
energy issues from telemetry … and I may not act outside my signed policy.
  ◆ ~62 tok in 3.1s = ~20.0 tok/s · on your DGX · $0.0000
```

In SIM mode the JSON is identical and the answer is labeled `ANSWER (simulated)`.

✓ Checkpoint: you should now be able to name all five parts of a spec without looking: base model, persona, skills, tools, signed policy.

## 4 · Equip it — skills give the persona hands  (A · B · C)

Goal: watch the discover → attach → invoke loop where the model actually calls the tools its skills expose.

```bash
.venv/bin/python week23/06_nemoclaw/demos/step02_equip_skills.py
```

Expected output (abbreviated, REAL mode):

```
Agent Skills catalog (App 4) — capabilities you can attach to the specialist:
  • nemo-retriever   RAG over YOUR runbooks & manuals       → tool: get_runbook
  • cuopt            GPU-accelerated setpoint optimization  → tool: optimize_setpoints
  → ACT get_runbook({'symptom': 'overheating'})
  ← OBSERVE {"step": "RB-07: check damper actuator…", "source": "runbook-RAG"}
  → ACT optimize_setpoints({'room': '1203'})
  ← OBSERVE {"room": "1203", "setpoint_c": 22.0, "kwh_saved": 3.1}
  · answer: Per RB-07, check the damper actuator first; set 22.0°C to save ~3.1 kWh.
```

C-path note: small local models sometimes skip a tool and answer directly — re-run once, or switch the model in the app's dropdown. SIM prints the same loop as a labeled trace.

✓ Checkpoint: you should now see at least one `→ ACT … / ← OBSERVE …` pair — the specialist doing, not just saying.

## 5 · The gate — run it under a signed policy  (A · B · C)

Goal: see every tool call checked by a policy gateway *before* it executes.

```bash
.venv/bin/python week23/06_nemoclaw/demos/step03_run_openshell.py
```

Expected output (abbreviated):

```
The specialist runs inside an OpenShell sandbox under this SIGNED policy:
{ "signed_by": "ops-team", "allow_tools": ["get_room_telemetry", "dispatch_work_order"], … }
  ✓ POLICY get_room_telemetry allowed → execute in sandbox
  ← OBSERVE {"temp_c": 26.4, "setpoint_c": 22.0, "occupied": true}
  ✓ POLICY dispatch_work_order allowed → execute in sandbox
  ← OBSERVE {"work_order": "WO-1203", "priority": "CRITICAL", "status": "dispatched"}
```

Notice the demo never actually *denies* anything — the model was only offered allowed tools. That gap is exactly what `labs/lab02_policy_gate.py` closes (a honeypot exfil tool + a red-team replay). The gate pattern here is faithful to OpenShell's design; the installable runtime itself is **[UNCERTAIN]** per the runbook — verify with the `git clone` from section 0 before assuming a CLI exists.

✓ Checkpoint: you should now have a run where every `→ ACT` line was preceded by a `✓ POLICY … allowed` line.

## 6 · The fleet — a supervisor routes to authored experts  (A · B · C)

Goal: one task, three specialists, and a supervisor that picks the right one.

```bash
.venv/bin/python week23/06_nemoclaw/demos/step04_fleet.py
```

Expected output (abbreviated, REAL mode):

```
  • hvac     [nemotron-3-nano ] comfort/energy from telemetry    tools: get_runbook, optimize_setpoints
  • finance  [nemotron-3-super] budgets, invoices, forecasts     tools: query_ledger, run_forecast
  · decision → HVAC   (model: gemma3 · on your DGX · $0.0000)
  → supervisor routed to: hvac specialist
  Selected specialist: hvac  [nemotron-3-nano] — comfort/energy from telemetry
```

(The `auto-picked gemma3 for this terse call` line you may see is deliberate — thinking models burn a terse token budget on reasoning, so `view.classify()` swaps in a direct-answer model when one exists.)

✓ Checkpoint: you should now have seen a routing decision made by a model (REAL) or a labeled trace (SIM). One specialist is a spec; a fleet is a system.

## Labs (run these)

**labs/lab01_author_spec.py — author + lint a spec, then A/B the persona.** You write a *new* specialist (an energy analyst) and lint it: all 7 fields, Spark fit-math (Q8 GB/B × 1.18 overhead vs 115 GB usable), signed policy, non-empty egress allowlist, tools ⊆ skills. Then the same base model answers the same question bare vs wearing the persona (if your workhorse is a thinking model, the lab auto-picks a direct-answer one — you'll see a `◆ auto-picked …` line).

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/06_nemoclaw/labs/lab01_author_spec.py
```

Look for: five `✓` lint lines, then two answers — the persona'd one should be terser, cite the meter reading, and refuse to actuate. Modify it: change `base_model` to `nemotron-3-ultra` and re-run — the fit-math check should fail (~688 GB needed vs 115 GB usable): Ultra is cloud-only.

**labs/lab02_policy_gate.py — naive runtime vs the policy gate.** The tool list includes a honeypot (`post_report(url=…)` reaching any host) and the task nudges the agent to exfiltrate. Every requested call is judged twice — naive (executes everything) vs gated (signed tool allowlist + egress allowlist). A canned red-team replay guarantees denials even offline.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/06_nemoclaw/labs/lab02_policy_gate.py
```

Look for: `naive runtime : EXECUTES it   ☠ data would leave the building` directly above `gated runtime : ✗ DENY — egress to 'analytics.example.com' not allowlisted`, and `2/4 calls denied` in the replay. Modify it: add `"analytics.example.com"` to `POLICY["egress_allowlist"]` and re-run — the replay drops to 1/4 denied, which is exactly why the policy is *signed* by someone who isn't the agent.

**labs/lab03_fleet_router.py — route a batch, score the router, right-size the fleet.** Three tasks routed twice — keyword baseline (local code) and model-as-supervisor (real calls) — scored against the human-expected specialist, followed by illustrative Nano-vs-Super GPU-hours math.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/06_nemoclaw/labs/lab03_fleet_router.py
```

Look for: the two score lines (`keyword score: 3/3`, `model-router score: n/3`) and the right-sizing result (~44% GPU-hours saved by routing 2/3 of traffic to Nano-class). Modify it: add a fourth task that contains *no* routing keywords ("Guests on floor 12 are unhappy this week — find out why") — watch the keyword baseline guess `code` while the model router does better.

## Try it yourself

1. **Add a fourth specialist.** Give lab03's fleet a `guest` desk (Nano-based, "guest experience from reviews and requests") plus one task for it. Which router finds it first?

<details><summary>Solution</summary>

In `labs/lab03_fleet_router.py`, add to `FLEET`: `"guest": ("nemotron-3-nano", "guest experience from reviews & requests"),` and to `TASKS`: `("A VIP in suite 1801 complained twice about slow room service.", "guest")`. The keyword router has no `guest` branch — worse, "slow **room** service" trips its `"room"` keyword, so it confidently misroutes to `hvac` (✗); the model router usually names `guest` (✓). That asymmetry — heuristics misfire silently and need maintenance, models generalize — is the argument for a model supervisor.
</details>

2. **Make the persona enforce the policy in words.** Extend lab01's `system_prompt` so the specialist itself refuses to *propose* anything touching a host outside `telemetry.internal`, then ask it to "email this analysis to the vendor". Does the prompt hold? Why do you still need lab02's gate?

<details><summary>Solution</summary>

Append to `SPEC["system_prompt"]`: `" You may only reference hosts on your egress allowlist: telemetry.internal. Refuse anything else."` The model will usually refuse — but re-phrase the request two or three ways and it may comply, because a prompt is advice, not enforcement. The gate in lab02 denies the call *outside* the model's process regardless of what the model was talked into. Prompt = first layer; signed policy at a gateway = the boundary. That is App 07's whole thesis.
</details>

3. **Right-size with your real numbers.** Time a real ~200-token generation on your endpoint (lab01 prints one), replace lab03's `super_tps = 13.0` with your measured tok/s, and recompute the routed-vs-all-Super savings.

<details><summary>Solution</summary>

Run lab01 in REAL mode and note the elapsed time for the persona answer; tok/s ≈ tokens/seconds (or read the `◆ ~N tok in Ns = ~X tok/s` line from any demo). Edit `super_tps` in lab03's Round 3 to your value (keep the ×3 Nano multiplier — the runbook's GB10 MoE observation). The *percentage* saved barely moves (~44%) because it depends on the ratio, not the absolute speed — which is why right-sizing pays on every deployment tier.
</details>

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `404 model not found` on every call | model ID not on this endpoint | `curl $BASE/models` and pick a listed ID; cloud IDs are namespaced (`nvidia/…`) |
| `404/405` immediately, no model list | base URL missing the port or `/v1` | Ollama is `:11434/v1`, NIM/vLLM `:8000/v1`; the app's 🔌 Connection panel auto-appends `/v1` |
| `401 Unauthorized` | missing/wrong key, or tunnel basic-auth | cloud: `export DGX_API_KEY=nvapi-…`; ngrok basic-auth: key as `user:pass` |
| `403 missing public API endpoints permission` (cloud Super) | some personal build.nvidia.com orgs can't call Super/Ultra | use the verified Nano ID or a fallback (`meta/llama-3.1-8b-instruct`) |
| `exec format error` pulling a container on the Spark | x86 image on aarch64 | use `linux/arm64` NGC tags (e.g. `…-dgx-spark`) — never random Docker Hub images |
| port 8000 refused/wrong service on the Spark | vLLM/NIM/TRT/Dynamo all default to 8000 — only one wins | `docker ps` on the Spark; inspect `/v1/models` payload to see *which* service answered |
| lab answer is empty / "(model spent the whole budget thinking)" | thinking model (qwen3.6, gemma4, nemotron-3) used its `max_tokens` on reasoning | pull a direct model (`ollama pull gemma3`) — labs 01/03 and `view.classify()` auto-swap to it — or raise `max_tokens` |
| `⚠ another demo is already running` in the app | the server serializes runs with a lock | wait for the running chapter to finish, then re-click |
| step02/step03 answers without calling tools (C path) | small local model skipped tool use | re-run, lower the temperature, or pick a stronger model in the dropdown |

## Next

→ ../07_guardrails_openshell/TUTORIAL.md (Safe autonomy — NeMo Guardrails + OpenShell secure runtime) — you just built powerful specialists and met the gate informally; next you author, sign, and test the rails and policies that make their autonomy safe for real.
