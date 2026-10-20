# ▶ Hands-on Lab 04 — NVIDIA Agent Skills

> Part of Week 23 · The Open Superintelligence Stack. Companion explainer app: `.venv/bin/python week23/04_agent_skills/tutorial_server.py` → http://127.0.0.1:8103. This lab is the DO-IT side: you type the commands, you see the output.

**What you'll actually do**
- Clone the real NVIDIA skills catalog and inspect what a Skill physically is (a dir + SKILL.md).
- Author your own `hotel-telemetry` skill package and run an agent's discovery pass over it.
- Measure progressive disclosure yourself: metadata tokens vs full-body tokens, on your own files.
- Load your skill into a LIVE model on your endpoint and watch the discover → invoke → answer loop.
- Drive the same capability over both open transports by hand: MCP JSON-RPC frames and an A2A Agent Card + task lifecycle.
- Install your skill into a real agent (`~/.claude/skills/`) so it survives beyond this lab.

**Time** ~45 min · **Difficulty** intermediate · **Cost** $0 on Spark/local; cloud path is usage-billed.

## 0 · Pick your path

| Path | You have | You'll use |
|---|---|---|
| A 🖥️ | a DGX Spark (or any Linux GPU box) | real commands over SSH |
| B ☁️ | an nvapi- key from build.nvidia.com | hosted NIM endpoints, same API |
| C 💻 | just this laptop | local Ollama (localhost:11434) or the app's SIM mode |

(every numbered step below marks which paths it applies to)

## 1 · Get the real catalog (A + B + C)

Skills are files, not a service — cloning works on any machine, GPU or not.

```bash
git clone https://github.com/NVIDIA/skills ~/nvidia-skills
ls ~/nvidia-skills
```

**Expected output**

```
Cloning into '/Users/you/nvidia-skills'...
LICENSE  README.md  ai-q/  cudf/  cuopt/  nemo-retriever/  ...
```

The catalog is young — dir names shift between pulls; the durable thing is the *shape*: one directory per skill, a `SKILL.md` with frontmatter metadata inside. Look at one:

```bash
find ~/nvidia-skills -name "SKILL.md" | head -3
head -20 "$(find ~/nvidia-skills -name 'SKILL.md' | head -1)"
```

**Expected output**

```
/Users/you/nvidia-skills/<skill>/SKILL.md
---
name: <skill>
description: <what it does and WHEN an agent should use it>
---
# <skill>
...instructions the agent loads on demand...
```

✓ Checkpoint: you should now have the catalog on disk and have seen that a Skill = frontmatter (cheap, always read) + body (loaded only on demand).

## 2 · Author your own skill and measure progressive disclosure (A + B + C)

Lab 01 writes a two-skill catalog to `.sandbox/skills/`, runs the discovery scan an agent runs, and computes the metadata-vs-body token math on your actual files. No endpoint needed.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
.venv/bin/python week23/04_agent_skills/labs/lab01_author_a_skill.py
```

**Expected output**

```
▣ STEP 1 · AUTHOR — writing 2 skill packages to .../04_agent_skills/.sandbox/skills
  ✓ hotel-telemetry/SKILL.md  (2029 chars)
  ✓ ipmvp-savings/SKILL.md  (423 chars)
▣ STEP 2 · DISCOVER — scan the catalog, read frontmatter only:
  ◈ hotel-telemetry — Query room sensors and file work orders for the hotel BMS…
  ◈ ipmvp-savings   — Compute IPMVP-verified energy savings (baseline - reporting…
▣ STEP 3 · MEASURE — metadata-only vs full-body context cost (~4 chars/tok):
  hotel-telemetry  metadata ≈   55 tok   full body ≈  507 tok
  ipmvp-savings    metadata ≈   52 tok   full body ≈  106 tok
  CATALOG TOTAL    metadata ≈  107 tok   full body ≈  613 tok
▣ STEP 4 · LOAD ON DEMAND — task: 'Room 1203 is reading 29.4C…'
  ✓ loaded 507 tok for ONE skill — the other stayed cold.
```

✓ Checkpoint: you should now see the concrete number progressive disclosure saves — discovery costs ~tens of tokens per skill; bodies load only when a task matches.

## 3 · Point an endpoint at the labs (A / B / C)

The labs inherit the endpoint from `config.py` (same `DGX_CONN` resolution as every Week 23 app). Pick ONE:

**A 🖥️ your Spark over Tailscale** (sovereign, $0 — the runbook's default REAL path, Ollama on port 11434):

```bash
export DGX_CONN=tunnel
export DGX_TUNNEL_URL=http://<your-spark>.<your-tailnet>.ts.net:11434/v1
curl -s $DGX_TUNNEL_URL/models | head -c 300   # sanity: OpenAI-shaped model list
```

If the Spark has no models yet, over SSH (or the app's 🖥️ DGX console):

```bash
ssh <user>@<your-spark> "ollama pull qwen3.6:35b-a3b-q8_0"
```

**B ☁️ build.nvidia.com** (usage-billed, off-box — NOT the sovereign path, honest label and all):

```bash
export DGX_CONN=cloud
export DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1
export DGX_API_KEY=nvapi-...        # from any model page on build.nvidia.com
curl -s $DGX_CLOUD_URL/models -H "Authorization: Bearer $DGX_API_KEY" | head -c 300
```

`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` is a verified live ID; Super/Ultra ID suffixes are not — list live IDs with the `curl` above and pin with `export DGX_MODEL=<id>` rather than trusting any doc (some personal orgs also hit 403 "missing public API endpoints permission" on newer models).

**C 💻 local Ollama on this laptop**:

```bash
export DGX_CONN=local
ollama list          # any tool-capable model works; pull one if empty
```

**Expected output** (any path)

```
{"object":"list","data":[{"id":"qwen3.6:35b-a3b-q8_0", ...
```

✓ Checkpoint: `curl` on `/models` returns a JSON model list. No endpoint at all? Fine — lab 02 degrades gracefully and labs 01/03 never needed one.

## 4 · Load your skill into a live agent (A + B + C — SIM fallback built in)

Lab 02 reads the `hotel-telemetry` skill you authored in step 2, hands it to the model on your endpoint as a system-prompt + tool surface, and runs the genuine tool-call loop: the model decides the skill applies, calls `query_room_telemetry`, sees the reading is >3°C over band with an alarm, files the work order via `file_work_order` (exactly as the SKILL.md instructs), and answers from the results.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
.venv/bin/python week23/04_agent_skills/labs/lab02_load_skill_live.py
```

**Expected output** (REAL; with no endpoint you get the same trace under a clear `[no endpoint — showing expected output]` label)

```
▣ connection: tunnel (DGX over a tunnel) · endpoint: http://…:11434/v1
▣ STEP 1 · DISCOVER — the agent reads the skill file first:
  ✓ found .sandbox/skills/hotel-telemetry/SKILL.md (authored in lab01)
▣ STEP 2 · INVOKE — live tool-call loop (max 4 hops, 60 s/request, 90 s total):
  → ACT query_room_telemetry({'room': '1203'})
  ← OBSERVE {"temp_c": 29.4, "setpoint_c": 23.0, "alarm": "HIGH_TEMP"}
  → ACT file_work_order({'room': '1203', 'summary': 'Room temperature is 29.4C…'})
  ← OBSERVE {"work_order": "WO-4211", "status": "filed", "sop": "SOP-HVAC-07"}
  · ANSWER: Room 1203 is at 29.4°C against a 23°C setpoint … work order WO-4211 filed per SOP-HVAC-07.
  ◆ 2 tool hop(s) in 38.2s · on your DGX · $0.0000
```

✓ Checkpoint: you should now have watched a model you never modified gain your capability from a file — and seen the honest cost line (`$0.0000` sovereign vs `cloud usage billed`).

## 5 · The same skill over MCP and A2A (A + B + C — fully offline)

Lab 03 exposes the identical capability through both open standards: you drive real MCP JSON-RPC frames (`initialize` → `tools/list` → `tools/call`) against an in-process server, then read the A2A Agent Card and step the six-state task lifecycle. This is Ch 5 of the explainer, but you type it.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
.venv/bin/python week23/04_agent_skills/labs/lab03_mcp_a2a_bridge.py
```

**Expected output**

```
▣ PART A · MCP — agent→TOOLS. Drive the handshake by hand (JSON-RPC 2.0):
  → {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
  ← {"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "query_room_telemetry"…
▣ PART B · A2A — agent→AGENT. …Agent Card a peer fetches from /.well-known/agent.json
  ◈ state: submitted    — peer POSTs the task to the agent
  ◈ state: working      — agent loads its hotel-telemetry skill…
  ◈ state: completed    — artifact returned to the delegating agent
```

✓ Checkpoint: you can now say — with frames you've seen — why MCP and A2A are complementary: MCP moved the *tool call*, A2A moved the *task*, and the SKILL.md never changed.

## 6 · Install the skill into a real agent (A + C)

Make it permanent — Claude Code loads skills from `~/.claude/skills/` natively; the NVIDIA catalog installs the same way:

```bash
mkdir -p ~/.claude/skills
cp -r /Users/altodev/Desktop/agenticaicodingfitness/week23/04_agent_skills/.sandbox/skills/hotel-telemetry ~/.claude/skills/
ls ~/.claude/skills/hotel-telemetry
```

**Expected output**

```
SKILL.md
```

On a Spark (path A) the sovereign version is the same two commands over SSH, pairing the skill with the on-box model from step 3.

✓ Checkpoint: a fresh agent session can now discover `hotel-telemetry` by its frontmatter — you've shipped institutional knowledge as a versioned file.

## Labs (run these)

**labs/lab01_author_a_skill.py** — authors a two-skill catalog to `.sandbox/skills/`, runs the discovery scan (frontmatter only), and measures the progressive-disclosure token math on your files. Run: `cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/04_agent_skills/labs/lab01_author_a_skill.py`. Look for: the CATALOG TOTAL line — metadata is ~10× cheaper than bodies. Modify it: add a third skill of your own (a runbook you actually use) and rerun — watch the totals move.

**labs/lab02_load_skill_live.py** — loads the lab01 skill into a live model via the OpenAI tool-call loop and times the hops; prints real setup commands + a labeled expected-output sample when no endpoint answers. Run: `cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/04_agent_skills/labs/lab02_load_skill_live.py`. Look for: the model choosing to call `query_room_telemetry` unprompted — that decision came from the skill's when-to-use metadata. Modify it: change the user question to something the skill does NOT cover ("what's the weather in Paris?") and confirm the model answers without invoking the tool.

**labs/lab03_mcp_a2a_bridge.py** — the same tool served as MCP JSON-RPC frames (genuinely dispatched in-process) and as an A2A Agent Card + six-state task lifecycle; offline, stdlib only. Run: `cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/04_agent_skills/labs/lab03_mcp_a2a_bridge.py`. Look for: the `tools/call` response carrying 29.4 — it executed, it isn't a canned string. Modify it: add a second tool `file_work_order` to `tools/list` and dispatch a `tools/call` for it.

## Try it yourself

1. **Skill triage.** Give lab01's catalog five skills and a batch of six tasks; write a 10-line matcher that picks a skill per task from *metadata only*, then check each pick against the full body. How often does metadata alone suffice?
<details><summary>Solution</summary>
Score each task against each skill's `description` with simple keyword overlap (`set(task.lower().split()) & set(desc.lower().split())`, take the argmax). On operationally-worded tasks metadata picks correctly ~5/6 times — which is exactly why the frontmatter's "Use when…" sentence is the most load-bearing line in a SKILL.md. The miss is usually a task phrased in synonyms the description lacks; fix the description, not the matcher.
</details>

2. **Prove framework-agnosticism.** Run lab02 twice against two different endpoints (e.g. local Ollama and build.nvidia.com with `DGX_CONN=cloud DGX_MODEL=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`) without touching the skill file. Diff the two ANSWER lines.
<details><summary>Solution</summary>
`DGX_CONN=local .venv/bin/python week23/04_agent_skills/labs/lab02_load_skill_live.py > /tmp/a.txt`, then export the cloud vars from step 3B and redirect to `/tmp/b.txt`; `diff /tmp/a.txt /tmp/b.txt`. Wording differs, but both call the same tool with `{'room': '1203'}` and both cite 29.4°C/SOP-HVAC-07 — the capability was in the file, not the model. Also note the cost lines differ: `$0.0000` vs `cloud usage billed`.
</details>

3. **A2A input-required.** Lab 03 walks submitted → working → completed. Extend PART B2 so the agent hits `input-required` (e.g. the room id is missing from the task message) and resumes to `completed` after the "peer" supplies it.
<details><summary>Solution</summary>
Start the task with `"message": "Investigate the high-temp alarm."` (no room). In the working step, if no room id parses out, set `task["state"] = "input-required"` and print the question ("which room?"); then simulate the peer's reply `{"room": "1203"}`, set state back to `working`, call the tool, and complete. That pause-for-the-caller hop is the state MCP has no equivalent for — it's the heart of agent-to-agent delegation (Week 17).
</details>

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `404` / model-not-found on lab02 | wrong model ID for that endpoint | `curl $BASE/models` and `export DGX_MODEL=<an id from the list>` — never trust a hardcoded ID, esp. cloud Nemotron Super/Ultra suffixes |
| `401 Unauthorized` | missing/stale key | cloud needs `DGX_API_KEY=nvapi-…`; ngrok basic-auth tunnels take `user:pass` as the key |
| `404` on every route / connection works in browser but not labs | `/v1` missing from the base URL | Ollama is `:11434/v1`, NIM/vLLM `:8000/v1`; the app auto-appends `/v1` only for empty paths |
| `403 missing public API endpoints permission` (cloud) | some personal build.nvidia.com orgs lack access to newer Nemotron endpoints | pick a fallback ID (`nvidia/nvidia-nemotron-nano-9b-v2`, `meta/llama-3.1-8b-instruct`) from your live `/models` list |
| model answers but never calls the tool | model lacks tool-calling support | pick a tool-capable model (qwen3.6, llama3.3, nemotron-3); tiny quantized chat models often skip tools |
| `exec format error` pulling containers on the Spark | x86 image on ARM64 | the Spark is aarch64 — use NGC's `-dgx-spark`/arm64 tags only (runbook §1.1) |
| port 8000 refused/odd payload on the Spark | vLLM/NIM/TRT/Dynamo all contest :8000 — only one runs at a time | check what answered: Ollama lists many models, a NIM exactly one; skills labs prefer Ollama on :11434 anyway |
| explainer app on the wrong port | 8103 busy | it auto-picks the next free port and prints it; pin with `SKILLS_GUIDE_PORT` |
| lab02 says lab01 not run yet | `.sandbox/skills/` missing | run lab01 first (or let lab02 use its inline copy — same skill) |

## Next

→ ../05_aiq_research_lab/TUTORIAL.md (AI-Q Open Agent Blueprint — intent router + deep agent + researcher fan-out on NeMo Agent Toolkit) — you just built the portable capability; AI-Q is the first harness that wires a whole catalog of them into a working research agent.
