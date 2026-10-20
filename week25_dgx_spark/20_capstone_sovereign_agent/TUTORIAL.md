# ▶ Spark Lab 20 — Capstone: a sovereign hotel agent — fine-tune → serve → gateway → NAT agent in a sandbox

> Part of Week 25 · DGX Spark: fine-tune, serve, and build sandboxed agents. You type the commands, you see the real output. Every lab also runs in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's playbook, or a clearly marked EXAMPLE.

**What you'll actually do**
- Put this week's pieces into one system across **two Sparks**: the fine-tuned router on Spark B, and the agent's brain, the gateway and the sandbox on Spark A.
- Turn the **fine-tuned router into a tool** that a NeMo Agent Toolkit agent calls before it acts.
- Run four **acceptance scenarios** end to end (smoke, a hot room, a restaurant question, a Thai message) and let code judge each one.
- Fence the agent with an **OpenShell policy** that allows exactly two ways out, and check it against ten attempted escapes.
- Read one **scorecard** that says, honestly, what is proven and what still needs a Spark.

**Time** ~75 min · **Difficulty** advanced · **Hardware** 2 Sparks (1 works with two vLLMs on one; none: the whole wiring runs on the laptop stand-in)

**Official playbooks covered:** [vLLM](https://build.nvidia.com/spark/vllm) (the agent-ready recipe) · [OpenShell](https://build.nvidia.com/spark/openshell) · [LLaMA Factory](https://build.nvidia.com/spark/llama-factory) · [Connect two Sparks](https://build.nvidia.com/spark/connect-two-sparks), all through the modules that taught them. The glue (router-as-a-tool, gateway aliases, acceptance judge, capstone policy) is course-original.

## 0 · Before you start

The capstone reuses, rather than repeats, the earlier modules. Know where each piece comes from:

| Piece | Module | What the capstone takes |
|---|---|---|
| Hotel router fine-tune | 09 | the LoRA adapter `~/w25/m09/saves/qwen3-4b-hotel/lora/sft` and its system prompt |
| Scoring and serving the fine-tune | 13 | `_hotel_eval.py` (the ship gate), vLLM with `--lora-modules hotel-ft=…` |
| Agent-ready vLLM | 05 | the Qwen3.6-35B-A3B-NVFP4 command and its three agent flags |
| Gateway | 08 | `_gateway.py` (config writer + proxy runner), the on-Spark install |
| Agent | 14 | NAT 1.9, the `hotel_ops_nat` plugin (`room_temperature`, `create_maintenance_ticket`) |
| Sandbox | 15 | `policykit`, the OpenShell sequence from lab 15-4 |
| Two Sparks | 01, 02 | `SPARK_HOST` and `SPARK_HOST2`, both answering `ssh -o BatchMode=yes` |

On the laptop you need the NAT venv with both plugins, and the LiteLLM venv:

```bash
# on: laptop
uv pip install --python week25/.venv-nat/bin/python --no-deps \
  -e week25/14_nat_agents/hotel_ops_nat -e week25/20_capstone_sovereign_agent/hotel_capstone_nat
week25/.venv-nat/bin/nat info components -t function | grep -E "route_guest|room_temp|maintenance"
ls week25/.venv-litellm/bin/litellm
```

✓ Checkpoint: `nat info components` lists `route_guest_message`, `room_temperature` and `create_maintenance_ticket`, and the LiteLLM binary exists.

## 1 · What you are building

A night-shift operations agent for a small hotel. A guest writes, in English or Thai; the agent classifies the message with **the model you fine-tuned for exactly that**, reads the room if it is about temperature, opens a ticket when engineering or security must act, and answers. Nothing leaves your two Sparks.

```text
                    Spark A — agent side                                   Spark B — router side
 ┌──────────────────────────────────────────────────────────┐   ┌───────────────────────────────────┐
 │  OpenShell sandbox (policy: 2 ways out)                   │   │  vLLM :8000                        │
 │   NAT tool_calling_agent                                  │   │   Qwen3-4B-Instruct-2507           │
 │     tools: route_guest_message · room_temperature ·       │   │   + LoRA hotel-ft (Module 09)      │
 │            create_maintenance_ticket                      │   │                                   │
 │        │ brain: inference.local      │ router tool        │   │  LLaMA Factory: retrain the next   │
 │        ▼                             ▼                    │   │  adapter here while this one serves│
 │  LiteLLM gateway :4000 (master key)                        │   └───────────────▲───────────────────┘
 │     agent-brain  → vLLM 127.0.0.1:8000 (Qwen3.6 NVFP4) ───┼───── hotel-router ─┘ (LAN / ConnectX-7)
 └──────────────────────────────────────────────────────────┘
```

Three design decisions, and why:

1. **The router is a tool, not the agent's job.** The brain (a 35B MoE) could guess departments. But you measured, in Module 13, that guessing misses emergencies and mislabels Thai requests. A 4B model trained for the task, called as a tool, is cheaper and checkable.
2. **Every model call goes through one gateway.** The gateway is one place to put a key, log fallbacks and swap a model. Clients (the agent included) never know which Spark answers.
3. **The sandbox is fenced to exactly two destinations.** The agent can reach its brain and the router, and that is all: not the internet, not Spark B directly, and not your home directory.

✓ Checkpoint: you can say which Spark serves which model, and why the router talks to Spark B only through the gateway.

## 2 · The plan and the preflight: lab 20-1

```bash
# on: laptop
.venv/bin/python week25/20_capstone_sovereign_agent/labs/lab01_plan_and_preflight.py
```

**Expected output** (captured on this Mac in DRY mode: the plan is arithmetic, the Spark rows are EXAMPLE values)

```
▣ STEP 1 · the plan: what runs where (arithmetic, not a measurement)
│ where    service                        port  model / role                                  weights  module
│ ───────  ─────────────────────────────  ────  ────────────────────────────────────────────  ───────  ───────
│ Spark A  vLLM · agent brain             8000  nvidia/Qwen3.6-35B-A3B-NVFP4                  20.0 GB  05
│ Spark A  LiteLLM gateway                4000  agent-brain · hotel-router aliases            0.5 GB   08
│ Spark A  OpenShell sandbox + NAT agent  —     route_guest_message · room_temperature · tic  1.0 GB   14 · 15
│ Spark B  vLLM · base + LoRA hotel-ft    8000  Qwen/Qwen3-4B-Instruct-2507 + hotel-ft        8.1 GB   09 · 13
│ Spark A: vLLM reserves 40% of 128 GB =  51.2 GB  ███████████░░░░░░░░░░░░░░░░░
│ Spark B: vLLM reserves 30% of 128 GB =  38.4 GB  ████████░░░░░░░░░░░░░░░░░░░░
◆ Spark A keeps ~77 GB free for the gateway, the sandbox and the OS. Spark B keeps ~90 GB free: enough to retrain the next hotel adapter (Module 09, LoRA on 4B) while the current one serves.
◆ Why two Sparks: the brain and the router scale and fail separately, and retraining never touches the agent's Spark. One Spark also works: run both vLLMs on A with --gpu-memory-utilization 0.4 + 0.3 on ports 8000/8001.
…
│ Spark    GPU   free mem  free disk  needs ports  busy  ready
│ ───────  ────  ────────  ─────────  ───────────  ────  ─────────
│ Spark A  GB10  112 GB    3100 GB    8000 + 4000  none  ◈ example
│ Spark B  GB10  112 GB    3100 GB    8000         none  ◈ example
═ Plan fits: 20 GB brain + 8 GB router on two 128 GB Sparks, each well under its reservation. Next: lab 20-2.
```

The 20 GB is NVFP4 arithmetic from Module 01 (35B × ~4.5 bits). The 8.1 GB is Qwen3-4B at bf16. Both are well inside their reservations, and the rest of each reservation becomes KV cache.

✓ Checkpoint: in LIVE mode both Sparks show GB10, free ports and ✓; in DRY mode you can explain each column.

## 3 · Bring it up: lab 20-2

Lab 20-2 prints the three launch sequences with your hosts filled in, and writes the gateway config for Spark A. With `--launch` it starts them in order and waits for each `/v1/models`:

```bash
# on: laptop
.venv/bin/python week25/20_capstone_sovereign_agent/labs/lab02_bring_up.py            # plan + config
.venv/bin/python week25/20_capstone_sovereign_agent/labs/lab02_bring_up.py --launch   # start all three
```

**Expected output** (captured on this Mac in DRY mode, first lines)

```
▣ STEP 1 · the gateway config the Spark will run (no laptop fallbacks: the sandbox must stay on the Sparks)
│ alias         backend                                   api_base (as seen from Spark A)
│ ────────────  ────────────────────────────────────────  ───────────────────────────────
│ agent-brain   hosted_vllm/nvidia/Qwen3.6-35B-A3B-NVFP4  http://localhost:8000/v1
│ hotel-router  hosted_vllm/hotel-ft                      http://spark-b:8000/v1
```

What each sequence changes from its source module, printed next to it:

| Service | Source | Course changes (and why) |
|---|---|---|
| Router on B | Module 13, path A | none; the LoRA flags are vLLM's own CLI (a course addition, see Module 05 §7) |
| Brain on A | Module 05 §4 (older vLLM playbook), quoted | `-d --name` instead of `-it`; `-p 127.0.0.1:8000:8000` so only the gateway on A can reach it; no `-e HF_TOKEN` (the model is not gated) |
| Gateway on A | Module 08 §7 | the config has **no laptop fallbacks**: on the Spark, a silent fallback to someone's laptop would leak guest messages off the Sparks |

The gateway binds `0.0.0.0:4000` on purpose. The OpenShell playbook's troubleshooting says the provider URL must use the Spark's IP, not `localhost`, because OpenShell's gateway runs in Docker. The master key is what protects it.

> 💡 **Optional hardening:** vLLM accepts `--api-key <key>`, which makes Spark B's router refuse any caller that is not your gateway. Put the same key in the gateway config as the deployment's `api_key`. This is vLLM's own flag, not in a playbook.

✓ Checkpoint: with two Sparks, `curl http://spark-b:8000/v1/models` lists `hotel-ft`, and on Spark A `curl -H "Authorization: Bearer $(cat ~/w25/litellm/master.key)" localhost:4000/v1/models` lists both aliases.

## 4 · The router becomes a tool

`hotel_capstone_nat/` is a NAT plugin package with one tool, built the same way as Module 14's:

```text
route_guest_message(message)                       → JSON the agent reads
  POST {gateway}/chat/completions
    model: hotel-router                              (the gateway alias, never a hostname)
    system: Module 09's exact training prompt        (a different prompt is a different task)
    temperature: 0
  parse strictly → {department, priority, reply}     (invalid JSON → an error, never a guess)
  + served_by, x-litellm-attempted-fallbacks        (so a dead fine-tune is visible)
```

Two details matter more than the code:

- **The system prompt is Module 09's, byte for byte.** A LoRA adapter learned the task under that prompt. Send a different one and you are measuring a different model.
- **No repair.** If the router returns `"priority": urgent` without quotes (the Module 13 failure), the tool returns an error, and the agent's system prompt tells it to hand the guest to a human rather than guess.

`configs/capstone_agent.yml` wires it up. The key parts:

```text
llms.brain            base_url ${CAPSTONE_BRAIN_URL}, model agent-brain, api_key ${HOTEL_GATEWAY_KEY}
functions             route_guest_message (base_url ${CAPSTONE_GATEWAY}, model hotel-router)
                      room_temperature · create_maintenance_ticket   (Module 14's plugin)
workflow              tool_calling_agent, max_iterations 6, and the rules:
                        1. route first  2. read the room for temperature  3. ticket only for engineering/security
                        4. two sentences: what you did (quote the ticket id) + the router's reply
```

The key only ever lives in the environment. The YAML names the variable, never the value.

✓ Checkpoint: you can explain why the tool refuses to "fix" invalid JSON, and why it must send Module 09's exact system prompt.

## 5 · End to end: lab 20-3

Lab 20-3 starts the gateway (both aliases with laptop fallbacks) and runs the NAT agent once per scenario. It then judges each run from **NAT's own trace** (which tools ran, in what order, with what output) and the ticket log. It judges what the agent did, not what it says it did:

```bash
# on: laptop
.venv/bin/python week25/20_capstone_sovereign_agent/labs/lab03_agent_end_to_end.py
```

**Expected output** (captured on this Mac: real NAT 1.9 and LiteLLM 1.89; no Spark, so both aliases fell back to gemma4:12b as a LAPTOP STAND-IN. Two full runs gave the same verdicts)

```
▣ STEP 3 · scenario 'smoke': Hello, there's a strong smell of smoke in the corridor near room 522. Please hurry.
→ route_guest_message({'message': "Hello, there's a strong smell of smoke in the corridor ne)
  ← {"ok": true, "department": "security", "priority": "urgent", "reply": "We have alerted our security team to investigate the smoke smell immediately.",
→ create_maintenance_ticket({'room': '522', 'issue': 'Strong smell of smoke in the corridor near r)
  ← Created ticket MT-9C47E8 for room 522 (priority urgent): Strong smell of smoke in the corridor near room 522.
· ANSWER  I have opened an urgent maintenance ticket (MT-9C47E8). We have alerted our security team to investigate the smoke smell immediately.
✓ routed first
✓ router said security
✓ ticket for room 522
✓ ticket priority urgent

▣ STEP 3 · scenario 'hot-room': Guest in room 808: it's really hot in here and the air conditioning just blows air.
→ route_guest_message({'message': "it's really hot in here and the air conditioning just blo)
  ← {"ok": true, "department": "engineering", "priority": "urgent", "reply": "I'm sorry for the discomfort; I have notified our engineering team to inspec
→ room_temperature({'room': '808'})
  ← Room 808: 27.9 °C, setpoint 23.0 °C (+4.9 °C, too warm). HVAC status: fault: fan coil unit not responding.
→ create_maintenance_ticket({'room': '808', 'issue': 'Air conditioning blowing air but not cooling)
  ← Created ticket MT-8E1E86 for room 808 (priority urgent): Air conditioning blowing air but not cooling; fan coil unit not responding.
✓ routed first
✓ router said engineering
✓ called room_temperature
✓ ticket for room 808

▣ STEP 3 · scenario 'restaurant': Could you recommend a good seafood restaurant within walking distance? Thank you.
→ route_guest_message({'message': 'Could you recommend a good seafood restaurant within walk)
  ← {"ok": true, "department": "concierge", "priority": "normal", "reply": "I would be happy to provide some excellent local seafood restaurant recommenda
✓ routed first
✓ router said concierge
✓ no ticket opened

▣ STEP 3 · scenario 'thai-tv': สวัสดีครับ ทีวีในห้อง 1203 เปิดไม่ติด
→ route_guest_message({'message': 'สวัสดีครับ ทีวีในห้อง 1203 เปิดไม่ติด'})
  ← {"ok": true, "department": "engineering", "priority": "urgent", "reply": "รับทราบครับ ช่างจะรีบดำเนินการตรวจสอบทีวีในห้อง 1203 ให้โดยเร็วที่สุดครับ",
→ create_maintenance_ticket({'room': '1203', 'issue': 'ทีวีในห้อง 1203 เปิดไม่ติด', 'priority': 'u)
  ← Created ticket MT-19010C for room 1203 (priority urgent): ทีวีในห้อง 1203 เปิดไม่ติด
✓ routed first
✓ router said engineering
✓ ticket for room 1203
✕ ticket priority normal
◆ gateway stopped (pid 57879); port 4000 is free again: True

▣ STEP 4 · acceptance summary
│ scenario    checks passed  time
│ ──────────  ─────────────  ────  ─
│ smoke       4/4            9s    ✓
│ hot-room    4/4            10s   ✓
│ restaurant  3/3            7s    ✓
│ thai-tv     3/4            10s   ✕
═ NOT accepted yet — read the ✕ lines: a routing miss is the router's job (retrain, Module 09/13), a tool-order or ticket miss is the agent's (prompt, Module 14).
```

How to read it:

- **The wiring is proven.** Gateway, NAT, a custom tool, two plugin packages and the tool order all worked, for real, on a laptop. The agent routed first every time, read room 808 before ticketing it, and opened no ticket for the restaurant.
- **The one failure is the finding you already had.** Module 13 measured that a prompt-only model marks normal Thai requests urgent, and here it does it again, inside a working agent: a TV that won't turn on became an urgent ticket at 3 a.m. The acceptance judge caught it, and the ✕ tells you which layer to fix: the **router**, by retraining in Module 09 with more Thai normal-priority examples. The agent's prompt is not at fault.
- **Look at the hot-room input.** The agent passed `"it's really hot in here…"` to the router, dropping `"Guest in room 808:"`, even though the prompt says "the guest's exact words". It worked, but agents paraphrase tool inputs. For a stricter system, pass the raw message into the tool from code instead of trusting the model to copy it.

With both Sparks serving, the same lab runs the fine-tuned router, and the Thai row is the one to watch.

✓ Checkpoint: you ran lab 20-3, and you can say for each ✕ whether the fix belongs to the router, the agent's prompt, or a tool.

## 6 · The fence: lab 20-4, steps 1–3

The sandbox policy is built with Module 15's `policykit`, and written to `policies/capstone_policy.yaml`:

| Network group | Endpoint | Binaries | When |
|---|---|---|---|
| `inference` | `inference.local:443` | the venv's Python, `/usr/bin/python3*`, `curl` | always: the brain, via `openshell inference set` → the gateway's `agent-brain` |
| `gateway` | `host.openshell.internal:4000` (`access: full`) | the venv's Python only | always: the router tool |
| `pypi` | `pypi.org`, `files.pythonhosted.org` | Python, pip | setup only, removed with `openshell policy update … --remove-rule pypi` |

```bash
# on: laptop
.venv/bin/python week25/20_capstone_sovereign_agent/labs/lab04_sandbox_and_scorecard.py
```

**Expected output** (captured on this Mac; step 2 is policykit's teaching model, not OpenShell)

```
▣ STEP 1 · build and validate the capstone policy (Module 15's policykit)
✓ policy valid · 3 network groups (inference, gateway, pypi) · 0 warning(s)

▣ STEP 2 · what would it allow? (policykit's teaching model — not OpenShell)
│ attempted action (after setup)  decision                  why
│ ──────────────────────────────  ─────────────────────  ─  ────────────────────────────────────────────────────
│ router tool → gateway           allow                  ✓  network_policies.gateway: host.openshell.internal:4…
│ brain → inference.local         inspect_for_inference  ✓  inference.local is handled by the proxy's inference…
│ router tool → Spark B directly  deny                   ✓  spark-b:8000 is not in network_policies (default de…
│ curl → gateway                  deny                   ✓  host.openshell.internal:4000 is listed, but not for…
│ agent → api.openai.com          deny                   ✓  api.openai.com:443 is not in network_policies (defa…
│ write tickets.jsonl             allow                  ✓  under read_write /sandbox
│ write /etc/hosts                deny                   ✓  /etc is read_only (Landlock: Permission denied)
│ read ~/.ssh on the host         deny                   ✓  not in read_only or read_write (Landlock: Permissio…
│ run as root                     deny                   ✓  the sandbox runs as 'sandbox'; it cannot become 'ro…
│ pip install (after setup)       deny                   ✓  pypi.org:443 is not in network_policies (default de…
✓ every attempt lands where the plan says: two ways out, nothing else
```

A real bug this step caught while the capstone was being written: the first version of the `gateway` rule had no `access` mode. policykit warned *"no access mode … the proxy denies it with CONNECT 403"*, a rule it takes from the NemoClaw applications playbook. The router tool would have been blocked inside the sandbox while every laptop test passed. Validate before you push.

Step 3 prints the OpenShell sequence on Spark A. It is Module 15 lab 4's sequence, re-pointed at the gateway: create a provider for the gateway, `inference set --model agent-brain`, create the sandbox with the policy, upload both folders, install NAT while `pypi` is open, remove `pypi`, run the agent, and prove the fence with `curl https://example.com` (it must fail). Every command that changes the Spark runs only with `--yes`.

> ⚠ **Course assumptions, check them the first time on a Spark.** (1) The NemoClaw applications playbook shows `host.openshell.internal` reaching the host's vLLM on `:8000`; the capstone uses the same name for the gateway's `:4000`. (2) The gateway key is passed with `--env` on the command line, where other users of the Spark could see it in the process list. On a shared Spark, use an OpenShell provider credential instead. Watch `openshell logs hotel-capstone --source sandbox` for `decision=deny` lines on your first run.

✓ Checkpoint: step 2 shows ✓ on all ten attempts, and you can say why "router tool → Spark B directly" must be denied even though Spark B is yours.

## 7 · The scorecard: what "done" means

Step 4 of lab 20-4 collects every gate into one table:

**Expected output** (captured on this Mac after the full lab 20-3 run above)

```
▣ STEP 4 · the capstone scorecard
│ gate                                             evidence
│ ────────────────────────────────────────────  ─  ────────────────────────────────────────────────────
│ router ship gate (Module 13)                  —  no fine-tune predictions yet (needs Module 09 on a …
│ agent scenarios (lab 20-3)                    ✕  3/4 accepted · LAPTOP STAND-IN · 2026-09-29 13:32  …
│ sandbox policy valid + fence (this lab)       ✓  2 runtime groups after setup · 10 attempts checked …
│ sandbox applied on Spark A (this lab, --yes)  —  DRY: not applied
═ Not complete yet: every — needs a Spark, every ✕ names the module to revisit. That is the honest state of a stack built on a laptop: the wiring is proven, the fine-tune and the sandbox are waiting for their Sparks.
```

The capstone is complete when every row is ✓ **on real Sparks**. The first live pass, in order:

1. Module 09 on Spark B: train, then predict. → Module 13 lab 02 scores it (row 1).
2. Lab 20-2 `--launch`: router on B, brain and gateway on A.
3. Lab 20-3 against the Spark gateway (row 2). The Thai row is the one to watch.
4. Lab 20-4 `--yes`: the sandbox on Spark A (row 4), then `openshell logs` to confirm both routes and the denials.
5. Re-run any lab with `SPARK_RECORD=1`, so DRY mode replays your real transcripts from then on.

✓ Checkpoint: you can read your own scorecard and name, for every row that is not ✓, the next command to run.

## Labs — run them here

**labs/lab01_plan_and_preflight.py** — What runs on which Spark, the memory each reserves, and read-only readiness checks on both Sparks.

**labs/lab02_bring_up.py** — The three launch sequences with your hosts, the Spark-side gateway config, and an opt-in launch that waits for each server.

**labs/lab03_agent_end_to_end.py** — Four guest scenarios through gateway, NAT agent, router tool and hotel tools, judged from NAT's own trace.

**labs/lab04_sandbox_and_scorecard.py** — The capstone OpenShell policy, ten attempted escapes, the Spark-side sandbox sequence, and the final scorecard.

`labs/_capstone.py` (the plan, gateway config, scenarios and judge) and `hotel_capstone_nat/` (the router tool) are shared, not labs.

## Try it yourself

**Exercise 20 — the acceptance judge.** Open `week25/20_capstone_sovereign_agent/exercises/ex20_acceptance.py`. Three `TODO`s:

1. `judge(scenario, calls, tickets)`: routed first, right department, the ticket rule (tickets only for engineering and security), and the priority when a ticket is expected.
2. `THAI_SCENARIO`: a Thai message about a normal engineering problem, the scenario that catches the bug lab 20-3 found.
3. `verdict(results)`: one failed check anywhere blocks acceptance, and names itself.

```bash
# on: laptop
.venv/bin/python week25/20_capstone_sovereign_agent/exercises/ex20_acceptance.py
```

**Expected output** (once all three TODOs are done)

```
✓ judge: 6 recorded runs · smoke ✓ · restaurant ✓ · stray ticket ✕ · Thai urgency ✕ · acted first ✕ · misroute ✕
✓ Thai scenario: "สวัสดีค่ะ ไฟในห้องน้ำห้อง 402 กะพริบ" → engineering · normal
✓ verdict: one failed check anywhere blocks acceptance, and names it (thai: priority)

═ Add THAI_SCENARIO to _capstone.SCENARIOS and re-run lab 20-3: the laptop stand-in may well fail it;
  the fine-tuned router (trained on Thai examples) is what has to pass it.
```

<details><summary>Hint — why judge from the trace instead of the answer text?</summary>

The agent's answer is prose, and it can claim "I opened a ticket" without having called the tool. NAT's trace records every tool call with its input and output, and the ticket log records every ticket. Judge what happened, not what was said.

</details>

<details><summary>Stretch — close the loop for real</summary>

Add ten Thai normal-priority messages to Module 09's training data (`week25/09_llama_factory/data/hotel_ops.json`), retrain on Spark B, score with Module 13 lab 02, and if the gate passes, swap the adapter folder and restart the router container. Re-run lab 20-3. This is the whole week in one loop: data → fine-tune → gate → serve → agent → acceptance.

</details>

✓ Checkpoint: all three checker lines are ✓.

## Troubleshooting

| Symptom | Fix |
|---|---|
| lab 20-3: `tool not registered: route_guest_message` | Install both plugins into the NAT venv (section 0); `nat info components -t function` must list all three tools |
| Every scenario: `(no answer) … Connection error` | The gateway did not start, or `HOTEL_GATEWAY_KEY` is missing. The lab sets both; for a manual `nat run`, export them first |
| The router tool returns `router returned invalid JSON` | That is the tool working: the router wrote something code cannot parse (Module 13's `urgent` without quotes). Fix the router, not the tool |
| All tool calls show `fallbacks: 1` on the Spark | `hotel-router` or `agent-brain` is not being served. Check `curl http://spark-b:8000/v1/models` and `docker logs w25-router` / `w25-brain` |
| Inside the sandbox: `CONNECT tunnel failed, response 403` on port 4000 | The `gateway` rule needs an access mode (`access: full`), and the calling binary must be listed; compare with `policies/capstone_policy.yaml` |
| Inside the sandbox: `inference.local` fails | `openshell inference get` must show the `capstone-gateway` provider and `agent-brain`; the provider URL must use the Spark's IP, not localhost |
| The agent answers without calling `route_guest_message` | Thinking or small models sometimes skip tools. Check `agent-brain` is the Qwen3.6 recipe with `--enable-auto-tool-choice`, or use a stronger laptop stand-in |

## Next

Continue to [Lab 21 — the playbook atlas](../21_playbook_atlas/TUTORIAL.md): every other official DGX Spark playbook, grouped by what you want to build next.
