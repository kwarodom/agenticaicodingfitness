# ▶ Hands-on Lab 05 — AI-Q Open Agent Blueprint: router → deep agent → researcher fan-out

> Part of Week 23 · The Open Superintelligence Stack. Companion explainer app: `.venv/bin/python week23/05_aiq_research_lab/tutorial_server.py` → http://127.0.0.1:8104. This lab is the DO-IT side: you type the commands, you see the output.

**What you'll actually do**
- Wire this folder's demos and labs to a real Nemotron endpoint — your Spark, build.nvidia.com, or local Ollama.
- Route a query by hand with `curl` and watch the shallow-vs-deep triage decision come back.
- Measure router accuracy AND the blended-cost saving on your own query mix (lab01) — the honest version of NVIDIA's "~50% cheaper" claim.
- Generate a machine-readable research plan, write it to a filesystem To-Do, and fan out two researcher sub-agents concurrently with a measured speedup (lab02).
- Close a real tool loop — model picks a tool as JSON, you execute it, the observation grounds the final answer — then map it to a NeMo Agent Toolkit `workflow.yml` (lab03).
- Install the NeMo Agent Toolkit for real, and (A-path) clone the actual AI-Q blueprint repo.

**Time** ~45 min · **Difficulty** intermediate · **Cost** $0 on Spark/local; cloud path is usage-billed.

## 0 · Pick your path
| Path | You have | You'll use |
|---|---|---|
| A 🖥️ | a DGX Spark (or any Linux GPU box) | real commands over SSH |
| B ☁️ | an nvapi- key from build.nvidia.com | hosted NIM endpoints, same API |
| C 💻 | just this laptop | local Ollama (localhost:11434) or the app's SIM mode |

(every numbered step below marks which paths it applies to)

## 1 · Launch the explainer and read your mode line — A/B/C

Goal: get the companion app up and learn how this folder auto-detects REAL vs SIM.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
.venv/bin/python week23/05_aiq_research_lab/tutorial_server.py
```

Expected output (SIM shown; if a local Ollama is already up you'll instead see `✓ REAL endpoint: <model> @ http://localhost:11434/v1`):
```
  ▣  NVIDIA AI-Q — an open deep-research lab for any domain
      ◈ SIM mode — no endpoint reachable, simulating the AI-Q agents.
        every chapter is learnable with no GPU. Go REAL anytime:
        ollama run nemotron-3-nano   (or set DGX_BASE_URL)
      open  →  http://127.0.0.1:8104
```

✓ Checkpoint: you should now see the guide at http://127.0.0.1:8104 and know whether you're in REAL or SIM. SIM is a full dry-run — every chapter still works. Leave the server running in its own terminal.

## 2 · Point at a live Nemotron endpoint — A/B/C (pick one)

Goal: give the router and researchers a real model to call. `config.py` resolves the connection from env vars, so demos and labs inherit it for free.

**A — your DGX Spark** (Ollama on the Spark, reached over LAN or Tailscale):
```bash
ssh <you>@<spark> "ollama pull nemotron-3-nano"      # once, on the Spark
export DGX_BASE_URL=http://<spark-host>:11434/v1     # laptop side; .ts.net names work too
curl -s $DGX_BASE_URL/models | head -c 300
```

**B — build.nvidia.com hosted NIMs** (usage-billed, data leaves the box):
```bash
export DGX_CONN=cloud
export DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1
export DGX_API_KEY=nvapi-...                          # from build.nvidia.com
export DGX_MODEL=nvidia/nemotron-3-nano-30b-a3b       # router tier; super-120b-a12b = deep tier
curl -s -H "Authorization: Bearer $DGX_API_KEY" $DGX_CLOUD_URL/models | head -c 300
```

**C — this laptop, local Ollama** ($0, sovereign):
```bash
ollama pull nemotron-3-nano       # or any model you already have (qwen3.6, gemma4, …)
curl -s http://localhost:11434/v1/models | head -c 300
```

Expected output (any path):
```
{"object":"list","data":[{"id":"nemotron-3-nano:30b-a3b","object":"model", ...
```

✓ Checkpoint: `curl .../models` returns a JSON model list. Restart the tutorial server (or use its 🔌 Connection panel) and the banner should flip to `✓ REAL endpoint`. No endpoint at all? Stay in SIM — steps 3–6 all have a SIM story.

## 3 · Route one query by hand — A/B/C

Goal: be the Intent Router yourself, once, so the labs' automation isn't magic. This is Ch 2 of the explainer done with bare `curl`.

```bash
# Path C shown; A: swap in $DGX_BASE_URL; B: add -H "Authorization: Bearer $DGX_API_KEY"
curl -s http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "nemotron-3-nano:30b-a3b",
  "max_tokens": 300, "temperature": 0,
  "messages": [{"role": "user", "content":
    "You are the AI-Q Intent Router. Answer with one word — shallow for a single-source lookup, deep for multi-step research. Query: Compare the 3-year TCO of a DGX Spark vs a frontier API for a 10-analyst team."}]
}' | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'][-400:])"
```

Expected output (abbreviated — Nemotron Nano is a reasoning model, so a thinking preamble may precede the answer):
```
...multi-source, needs cost modeling over time... deep
```

C-path without any model: run the canned version instead — `.venv/bin/python week23/05_aiq_research_lab/demos/step01_intent_router.py` prints the same routing decision in SIM.

✓ Checkpoint: you got a `deep` verdict for a hard query. Try a trivial query ("Who founded NVIDIA?") and confirm it comes back `shallow`. That one-word decision is the ~50%-cost lever.

## 4 · Run the four demos from the CLI — A/B/C

Goal: see the whole blueprint shape — route → plan → fan-out → tool bus — end to end, in your terminal (the web app runs these same files).

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
.venv/bin/python week23/05_aiq_research_lab/demos/step01_intent_router.py
.venv/bin/python week23/05_aiq_research_lab/demos/step02_deep_agent_plan.py
.venv/bin/python week23/05_aiq_research_lab/demos/step03_researchers_fanout.py
.venv/bin/python week23/05_aiq_research_lab/demos/step04_tools_toolkit.py
```

Expected output (step03, SIM shown — REAL runs one researcher live):
```
  → DISPATCH Researcher A (Super): 'DGX 3-yr hardware + power TCO'
  → DISPATCH Researcher B (Super): 'frontier API $/1M tok + team volume'
    · A → ACT web_search('DGX Spark price power draw depreciation')
  ← Researcher A findings → Memory   ← Researcher B findings → Memory
```

✓ Checkpoint: all four demos exit 0 and you can name the four moving parts: Intent Router (Nano), Deep Agent + Planning Sub-Agent, parallel Researchers (Super), NeMo Agent Toolkit tool bus.

## 5 · Install the real NeMo Agent Toolkit — A/B/C

Goal: the tool bus in Ch 5 is a real, pure-Python, any-architecture package. Install it and prove it runs — this works identically on the Spark (aarch64) and your laptop.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
.venv/bin/pip install "nvidia-nat[all]"     # package renamed from aiqtoolkit; both names may resolve
.venv/bin/nat --help
```

Expected output:
```
Usage: nat [OPTIONS] COMMAND [ARGS]...
Commands:
  run       Run a workflow from a config file
  serve     Serve a workflow as an API
  eval      ...
```

Its `workflow.yml` points an `llms:` block at ANY OpenAI-compatible base URL — the exact endpoint you wired in step 2. To serve a workflow: the runbook marks NAT's default port **[UNCERTAIN]** — verify with `nat serve --help` first, then run it on 8001 to dodge the vLLM/NIM collision on 8000:

```bash
.venv/bin/nat serve --config_file workflow.yaml --host 0.0.0.0 --port 8001
# liveness probe:  curl -s http://localhost:8001/docs   (FastAPI docs page = alive)
```

✓ Checkpoint: `nat --help` prints the command list. lab03 prints the minimal `workflow.yml` that matches its hand-rolled loop — that's your starting config.

## 6 · Clone the real AI-Q blueprint — A (B for the model slots; C uses SIM)

Goal: know where the production version lives. The full blueprint is a docker-compose of many services (frontend, backend, NeMo Retriever NIMs, Milvus); the retrieval/rerank NIMs are x86-heavy, so on a Spark you run the *agent* locally and point model slots at cloud NIMs.

```bash
# A-path, on the Spark:
git clone https://github.com/NVIDIA-AI-Blueprints/aiq-research-assistant && cd aiq-research-assistant
export NVIDIA_API_KEY=nvapi-...              # B: cloud NIMs fill the embed/rerank/ingest slots
# the runbook marks the exact compose path/profiles [UNCERTAIN] — verify with `ls deploy/` first:
docker compose -f deploy/compose.yaml up -d
```

Expected output (abbreviated):
```
Cloning into 'aiq-research-assistant'...
[+] Running ... aira-backend  Started
```

Ports are **[UNCERTAIN]** in the runbook (backend ~8051, frontend ~3000) — probe, don't hardcode: `curl -s http://localhost:8051/health`. If the health endpoint answers you're REAL; anything else, this folder's SIM already taught you the same architecture. **C-path equivalent:** the explainer app at :8104 in SIM mode *is* this blueprint's teaching twin — nothing is blocked.

✓ Checkpoint: repo cloned and you know the probe URL; or (C) you can say precisely which services the compose file would start and why the NIMs stay in the cloud on aarch64.

## Labs (run these)

**labs/lab01_router_economics.py — measure the routing claim.** Classifies six queries shallow/deep against your live endpoint, scores accuracy, then computes the blended-cost saving assuming a deep run ≈ 9× a shallow answer.
```bash
cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/05_aiq_research_lab/labs/lab01_router_economics.py
```
Look for: the per-query ✓/✗ marks with latency, then `routing economics on this mix` — a saving % that depends entirely on how many queries stay shallow. *Modify it:* replace `QUERIES` with six questions from your own domain (hotel ops, energy audits) and see how the saving moves with your real mix.

**labs/lab02_plan_fanout.py — plan to file, fan out for real.** Asks the model for a JSON plan with `parallel_ok` flags, writes it to `.sandbox/todo.md` (the blueprint's filesystem To-Do), then runs two researcher sub-agents concurrently and reports sum-of-latencies vs wall clock.
```bash
cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/05_aiq_research_lab/labs/lab02_plan_fanout.py
```
Look for: the `∥` / `⇢` marks on each TODO (parallel-legal vs sequential) and the final `fan-out speedup: ~1.9x` line — concurrency is the wall-clock half of the fan-out argument. *Modify it:* raise `max_workers` and the `[:2]` pick to 3 researchers; does the speedup scale, or does your endpoint serialize requests?

**labs/lab03_tool_bus.py — close the tool loop.** A 3-tool registry over real data (`config.DGX_SPECS`, the endpoint's model list, a safe calculator). The model picks a tool as JSON, the lab executes it, and the observation grounds a one-sentence answer. Ends by printing the equivalent NAT `workflow.yml`.
```bash
cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/05_aiq_research_lab/labs/lab03_tool_bus.py
```
Look for: the four arrows — `picks` → `EXECUTE` → `OBSERVE` → `final` — each one an observable event; and the failure branch if the model returns an invalid pick (that's the mismatch NAT's typed schemas exist to prevent). *Modify it:* add a fourth tool `todo_read()` that returns lab02's `.sandbox/todo.md`, and change `TASK` to "what's next on the research plan?".

## Try it yourself

**1. Break the router, then fix it with the prompt.** Add an ambiguous query to lab01 — "What is a DGX Spark?" — that could go either way. Get the router to route it *shallow* consistently.
<details><summary>Solution</summary>

Add `("What is a DGX Spark?", "shallow")` to `QUERIES`. If the model says `deep`, sharpen `ROUTER_PROMPT` with a tie-break rule, e.g. append: *"If a query can be answered from one spec sheet or one encyclopedia entry, it is shallow — definitions are shallow."* Rerun; accuracy should return to 7/7. This hand-tuning is exactly what the NAT Optimizer automates against a golden set (Week 10/15).
</details>

**2. Two-tier routing for real.** In lab01, route with a small model but make lab02's researchers use a bigger one — the actual Nano/Super split.
<details><summary>Solution</summary>

Run lab01 with `DGX_MODEL=nemotron-3-nano:30b-a3b` (or any small model you have), then run lab02 with `DGX_MODEL=qwen3.6:35b-a3b-q8_0` (or `nvidia/nemotron-3-super-120b-a12b` on the B path). Since labs read `config.MODEL` once at import, the env var is the whole switch: `DGX_MODEL=<small> .venv/bin/python .../lab01_router_economics.py`. Compare the latencies printed by each — the ratio is your live Nano-vs-Super cost intuition.
</details>

**3. Make the fan-out honest about dependencies.** lab02 currently fans out any `parallel_ok` steps. Add a guard that refuses to dispatch a step whose text references an earlier step's output ("using the above", "from step 2").
<details><summary>Solution</summary>

Before `picks = ...`, filter: `plan = [(s, p and not re.search(r"(above|previous|step \d)", s, re.I)) for s, p in plan]`. Rerun — you may see fewer `∥` marks. The deeper lesson: `parallel_ok` is a *claim by the planner*; production orchestrators (and the real Deep Agent) validate the dependency graph before dispatch rather than trusting it.
</details>

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `404` / model not found | wrong model id for this endpoint | `curl $BASE/models` and copy an id exactly; cloud ids are namespaced (`nvidia/nemotron-3-nano-30b-a3b`), Ollama ids are tagged (`nemotron-3-nano:30b-a3b`) |
| `401 Unauthorized` | missing/wrong key | B-path needs `Authorization: Bearer nvapi-…`; tunnels may need basic-auth creds in the URL |
| `404` on every route, even `/models` | base URL missing `/v1` | Ollama is `:11434/v1`, NIM/vLLM `:8000/v1`; the app's 🔌 Connection panel auto-appends it, raw env vars don't |
| labs print `[no endpoint — showing expected output]` | no reachable endpoint → SIM | that's by design; wire step 2, restart, rerun |
| router answer truncated / no label found | thinking model spent the budget on reasoning preamble | keep `max_tokens` ≥ 300 for classification, or pin a direct-answer model; the labs already send `reasoning_effort: none` (endpoints that reject it get a plain retry) and extract the *last* label mention as the fallback |
| `docker pull` fails on the Spark with exec/arch errors | x86-only image on aarch64 | expected for the blueprint's retrieval NIMs — point those slots at cloud NIMs (`NVIDIA_API_KEY`), run the agent locally |
| NAT/blueprint service won't bind | port 8000 contention (vLLM/NIM/Dynamo all default there) | run NAT on `--port 8001` (verify default with `nat serve --help` — runbook marks it [UNCERTAIN]) |
| lab02 speedup ≈ 1.0x | endpoint serializes concurrent requests | normal for single-instance Ollama under load; the pattern still holds where servers batch (NIM/vLLM/Dynamo — App 03) |
| `✗ call failed (APITimeoutError)` | your local model generates slower than the lab's per-call cap (a 12B model on a laptop ≈ 8 tok/s) | rerun (a warm model is faster), or pin a smaller model from `curl $BASE/models`, e.g. `DGX_MODEL=gemma3:4b`; the labs score what they got and exit cleanly |

## Next
→ ../06_nemoclaw/TUTORIAL.md (NemoClaw — build specialized agents: base model + persona + skills + tools + signed policy) — you've orchestrated generic researchers; next you forge the *specialists* worth dispatching, each with its own persona, tool grants, and a signed policy.
