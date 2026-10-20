# ▶ Hands-on Lab 11 — NeMo Data Flywheel

> Part of Week 23 · The Open Superintelligence Stack. Companion explainer app: `.venv/bin/python week23/11_data_flywheel/tutorial_server.py` → http://127.0.0.1:8110. This lab is the DO-IT side: you type the commands, you see the output.

**What you'll actually do**
- Generate real "production traces" by hitting your own endpoint, so you have logs to curate.
- Run a working mini-Curator pipeline — dedup → quality filter → PII scrub → LLM-judge label — on 14 messy traces and watch the funnel shrink.
- Harvest live teacher completions into a `train.jsonl` in the exact messages format NeMo Customizer / PEFT consume — distillation data, built by you.
- Run a real A/B promotion gate: two models on your endpoint, a blind LLM-judge, and a hard promote/keep decision.
- See the honest Spark story for the real blueprint: what runs on one ARM64 box, and what needs a datacenter.

**Time** ~45 min · **Difficulty** advanced · **Cost** $0 on Spark/local; cloud path is usage-billed.

## 0 · Pick your path

| Path | You have | You'll use |
|---|---|---|
| A 🖥️ | a DGX Spark (or any Linux GPU box) | real commands over SSH |
| B ☁️ | an nvapi- key from build.nvidia.com | hosted NIM endpoints, same API |
| C 💻 | just this laptop | local Ollama (localhost:11434) or the app's SIM mode |

(every numbered step below marks which paths it applies to)

## 1 · Point the labs at a model (A/B/C)

Everything in this folder resolves its endpoint through `config.py` — set it once, and the app, demos, and labs all follow.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness

# A — your Spark over the network (Ollama on the Spark):
export DGX_BASE_URL=http://<your-spark-host>:11434/v1

# B — build.nvidia.com hosted NIMs (usage-billed, not the sovereign path):
export DGX_CONN=cloud
export DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1
export DGX_API_KEY=nvapi-...        # from any model page → "Get API Key"

# C — local Ollama on this laptop (also the fallback if nothing is set):
export DGX_CONN=local               # expects ollama serve on :11434

# verify — you should see a JSON model list, not an error:
curl -s "${DGX_BASE_URL:-http://localhost:11434/v1}/models" | head -c 300
```

**Expected output**
```
{"object":"list","data":[{"id":"qwen3.6:35b-a3b-q8_0","object":"model",...},{"id":"gemma3:4b",...}]}
```

✓ Checkpoint: you should now see at least one model id from `/v1/models`. No endpoint at all? Fine — every lab below degrades to a clearly-labeled `[no endpoint — showing expected output]` mode, and the explainer app runs in SIM.

## 2 · Walk the explainer's four chapters (A/B/C)

One goal: see the whole loop once before you build its pieces by hand.

```bash
.venv/bin/python week23/11_data_flywheel/tutorial_server.py
# → open http://127.0.0.1:8110 and click Run on Ch 2–5
# or run the same chapters headless:
.venv/bin/python week23/11_data_flywheel/demos/step01_the_loop.py
.venv/bin/python week23/11_data_flywheel/demos/step04_evaluate_promote.py
```

**Expected output** (step04, abbreviated)
```
  round  student acc  teacher acc  student $/tok    verdict
  ──────────────────────────────────────────────────────────
  1            75%          91%          0.15x   …keep teacher
  ...
  4            88%          91%          0.18x   …keep teacher

What just happened: over 4 turns the small student closed the quality gap to a
120B teacher while serving at ~1/7th the cost — so it gets promoted to production.

┌─ the promoted student
```

✓ Checkpoint: you should now have seen the four-stage loop (observe → curate → customize → evaluate) and the promotion arc. Everything after this step is you doing those stages for real.

## 3 · OBSERVE — make some production traces (A/B/C)

You can't curate what you didn't log. Fire three real requests and append them to a raw log file — this is the "① OBSERVE" stage at toy scale (App 08's Relay/Phoenix does it at fleet scale).

```bash
BASE="${DGX_BASE_URL:-http://localhost:11434/v1}"
MODEL=$(curl -s "$BASE/models" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
mkdir -p week23/11_data_flywheel/.sandbox
for Q in "Is breakfast included?" "AC in 305 is rattling" "Late checkout for room 210?"; do
  curl -s "$BASE/chat/completions" -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${DGX_API_KEY:-dgx}" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":80}" \
    | python3 -c "import sys,json;r=json.load(sys.stdin);print(json.dumps({'prompt':'$Q','answer':r['choices'][0]['message']['content'][:200],'ok':True}))" \
    >> week23/11_data_flywheel/.sandbox/raw_logs.jsonl
done
wc -l week23/11_data_flywheel/.sandbox/raw_logs.jsonl
```

**Expected output**
```
       3 week23/11_data_flywheel/.sandbox/raw_logs.jsonl
```

C-path with no endpoint: skip this step — lab01 ships 14 built-in raw traces, so nothing downstream is blocked.

✓ Checkpoint: you should now have a `raw_logs.jsonl` of genuine request/response pairs (or know lab01 will supply them).

## 4 · CURATE — run the funnel yourself (A/B/C)

The demo prints a static 1M→62k funnel; this lab runs a real one on real rows.

```bash
.venv/bin/python week23/11_data_flywheel/labs/lab01_curate_logs.py
```

**Expected output** (abbreviated)
```
  raw logs            14 rows   everything the agent saw
  dedup               11 rows   exact + normalized near-dups dropped
  quality filter       9 rows   keep ok=True, ≥3 words, non-empty answer
  PII scrub            9 rows   redacted email/phone/secret in 3 traces
◈ Stage 4 — LLM-judge labels (teacher grades the keepers):
  KEEP  ← 'How do I reset the thermostat in room 412?'
  ...
✓ wrote 9 curated traces → …/.sandbox/curated.jsonl
```

✓ Checkpoint: you should now have `.sandbox/curated.jsonl` and have watched every Curator stage change the row count for a reason you can read in the code.

## 5 · CUSTOMIZE — build the distillation dataset (A/B/C)

Distillation = fine-tune a small student on (prompt → big-teacher answer) pairs. This lab harvests the teacher side live and writes the training file.

```bash
.venv/bin/python week23/11_data_flywheel/labs/lab02_distill_dataset.py
cat week23/11_data_flywheel/.sandbox/train.jsonl | python3 -m json.tool --json-lines | head -12
```

**Expected output** (abbreviated)
```
◈ Spark fit-math (runbook §1.2 — weights ≈ GB/B-param · ×1.18 overhead):
  serve  Nano 30B @ Q8     ≈ 30 × 1.06 × 1.18 ≈  38 GB  → fits a 128 GB Spark
  ...
◈ Harvesting 3 teacher completions from qwen3.6:35b-a3b-q8_0:
  1. [ 3.2s] » A hotel guest reports the AC in room 305 is rattl…
     teacher · I file a work order for room 305 and dispatch an engineer…
✓ wrote 3 training rows → …/.sandbox/train.jsonl
```

A-path, the real fine-tune: the runbook's verdict is that PEFT/LoRA on the Spark works for **≤12B students** (week19's `dgx_finetune` path); Nano-30B LoRA is borderline on one Spark, 2 Sparks (TP=2) beyond that. Full NeMo Customizer wants NeMo microservices on datacenter GPUs.

✓ Checkpoint: you should now have a `train.jsonl` where every row is `{"messages": [system, user, assistant]}` with the teacher's answer as the label — the exact input shape a LoRA/SFT job consumes.

## 6 · EVALUATE + PROMOTE — run the gate (A/B/C)

One goal: make an actual promotion decision between two live models, the way NeMo Evaluator gates a candidate.

```bash
# best with two models on the endpoint — e.g. on local Ollama:
ollama pull gemma3:4b   # a small "student" alongside your workhorse   (A/C)
.venv/bin/python week23/11_data_flywheel/labs/lab03_eval_gate.py
```

**Expected output** (abbreviated)
```
◈ teacher = qwen3.6:35b-a3b-q8_0
◈ student = gemma3:4b
  Q1  judge → TIE         teacher  6.1s · student  1.4s
  Q2  judge → A(teacher)  teacher  5.8s · student  1.5s
  quality: student ties-or-wins 1/2
  cost proxy: student 4.2× faster wall-clock (2.9s vs 11.9s total)
  ── GATE: …keep teacher — the student must tie-or-win on ALL items
```

✓ Checkpoint: you should now have watched a gate refuse (or grant) a promotion for stated, reproducible reasons — quality parity on every golden item AND lower cost. "…keep teacher" is a *correct* outcome, not a failure.

## 7 · The real blueprint on real hardware (A)

One goal: know exactly where the toy ends and the product begins. The Data Flywheel Blueprint runs on NeMo microservices, which want a Kubernetes cluster with x86 + datacenter GPUs — the runbook's verdict for one ARM64 Spark is **SIM**, with the honest on-Spark scope being: Curator runs (CPU-ok), small LoRA fine-tunes, and LLM-judge evaluation against the local endpoint.

```bash
# on the Spark:
git clone https://github.com/NVIDIA-AI-Blueprints/data-flywheel
uv pip install nemo-curator          # pip package, CPU-ok — the CURATE stage for real
```

**Expected output**
```
Cloning into 'data-flywheel'...
Successfully installed nemo-curator-…
```

The runbook marks the NeMo-microservices health probe `GET {url}/v1/models` as **[UNCERTAIN: NMP health route]** — if you do stand up a real NeMo microservices deployment, verify the route with `curl -s $FLYWHEEL_BASE_URL/v1/models` first rather than assuming it. C-path equivalent: the labs above plus the app's SIM mode already exercise every stage's logic.

✓ Checkpoint: you should now be able to say which flywheel stages run on one Spark (curate, small LoRA, evaluate) and which don't (full Customizer/microservices) — and why.

## Labs (run these)

**labs/lab01_curate_logs.py** — runs a genuine four-stage curation pipeline (dedup → quality filter → PII scrub → LLM-judge label) on 14 messy bundled traces and writes `.sandbox/curated.jsonl`. Run: `cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/11_data_flywheel/labs/lab01_curate_logs.py`. Look for: the row count dropping at each stage and `<EMAIL>`/`<PHONE>`/`<SECRET>` tags replacing real PII. Modify it: add a fourth PII pattern for room numbers (`r"room \d{3}"` → `<ROOM>`) and decide whether that helps or destroys the training signal — it's the classic over-scrubbing tradeoff.

**labs/lab02_distill_dataset.py** — calls your live model as the teacher on 3 domain prompts and writes `.sandbox/train.jsonl` in Customizer/PEFT messages format, plus prints Spark fit-math for the fine-tune. Run: `cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/11_data_flywheel/labs/lab02_distill_dataset.py`. Look for: the assistant turn in each row being the teacher's actual words — that IS the distillation label. Modify it: replace the hardcoded `PROMPTS` with prompts read from lab01's `curated.jsonl`, closing the curate→customize pipe for real.

**labs/lab03_eval_gate.py** — picks a teacher and a student from your endpoint, answers 2 golden questions with both, has a blind LLM-judge score each pair, then applies the hard promotion rule (tie-or-win on ALL items and cheaper). Run: `cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/11_data_flywheel/labs/lab03_eval_gate.py`. Look for: the judge verdicts, the latency-based cost proxy, and the gate's stated reason. Modify it: the judge always sees the student as "B" — randomize A/B order per question and see whether verdicts change (position bias is a real LLM-judge failure mode).

## Try it yourself

**1. Poison the flywheel, then catch it.** Add a trace to lab01's `RAW` with `ok: True` but a confidently wrong answer (e.g. "The wifi password is 1234" when it isn't). Which stage should catch it, and does it?

<details><summary>Solution</summary>

Nothing in dedup/quality/PII catches it — the trace is unique, "successful", and clean. Only the LLM-judge label stage *can* catch it, and only if the judge has grounds to know the answer is wrong (it usually doesn't, from the trace alone). This is the runbook's warning made concrete: flywheels amplify what you feed them, and "ok=True" from a happy user is not ground truth. Real deployments add a held-out golden set and human spot-checks precisely because the judge can't verify facts it never saw.
</details>

**2. Make the gate honest about tokens, not seconds.** lab03 uses wall-clock latency as the cost proxy. Change it to compare `usage.completion_tokens / elapsed` (tok/s) and tokens generated, then recompute the speedup.

<details><summary>Solution</summary>

In `ask()`, return `r.usage.completion_tokens` alongside the text and elapsed time (guard with `getattr(r, "usage", None)` — some servers omit it). Accumulate tokens per model, then report `student_tok_s = s_tokens/s_lat` vs teacher and cost-per-answer as tokens generated. You'll often find the "faster" student also generated fewer tokens — latency conflates speed with verbosity, which is exactly why App 09 measures cost per *successful task*, not per second.
</details>

**3. Spin one full turn.** Chain the labs: step 3's `raw_logs.jsonl` → lab01 (modified to read it) → lab02 (modified to read `curated.jsonl`) → lab03. One command per stage, artifacts flowing through `.sandbox/`.

<details><summary>Solution</summary>

In lab01, replace `RAW` with `RAW = [json.loads(l) for l in open(config.SANDBOX/"raw_logs.jsonl")]` (keep the bundled list as fallback if the file is missing). In lab02, `PROMPTS = [json.loads(l)["prompt"] for l in open(config.SANDBOX/"curated.jsonl")][:3]`. lab03 needs no change — it gates whatever model you'd have tuned. You now have the full observe→curate→customize-data→evaluate loop as four auditable files, which is the whole flywheel minus the GPU-hours.
</details>

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `404 model not found` | model id doesn't exist on THIS endpoint | `curl $BASE/models` and use an id from the list; never hardcode cloud ids — runbook marks Super/Ultra cloud suffixes [UNCERTAIN] |
| `401 Unauthorized` | missing/wrong key on tunnel or cloud | export `DGX_API_KEY=nvapi-…` (cloud) or put `user:pass` tunnel creds in the URL |
| `404` on every route | `/v1` missing from the base URL | Ollama is `:11434/v1`, NIM/vLLM `:8000/v1` — `config.apply_connection` auto-appends it in the app, but your shell exports must include it |
| Labs print `[no endpoint — showing expected output]` | nothing answering at the resolved BASE_URL | start `ollama serve` (C), or export `DGX_BASE_URL` to your Spark (A) / cloud preset (B) |
| lab03 says "only one model on the endpoint" | one model can't A/B itself meaningfully | `ollama pull gemma3:4b` (or any second model) and rerun |
| `APITimeoutError` on the first call to a model | cold weight load / teacher↔student swap exceeds the 25s per-call cap | the labs retry once (the load finishes server-side); if it persists, pre-warm with `ollama run <model> ""` or just rerun the lab |
| Judge answers eaten / verdicts always TIE | thinking model spent the token budget on reasoning | the labs send `reasoning_effort:"none"` (skipping the preamble on Ollama-compatible servers; servers that reject it get a plain retry) and extract the LAST label from reasoning+content; raise `max_tokens` if your model still reasons at length |
| `exec format error` pulling blueprint containers on the Spark | x86 image on ARM64 | Spark needs `linux/arm64` images (NGC Spark tags); the blueprint's microservices are x86/DC-sized — expected, per runbook §2.11 |
| Port 8000 refused on the Spark | NIM/vLLM/Dynamo all contend for :8000 | `ss -tlnp \| grep 8000` on the Spark; stop the other server or move one (this app itself uses laptop port 8110) |
| Cloud calls suddenly rate-limited | build.nvidia.com free tier (~40 req/min commonly reported) | space out lab runs, or switch to path A/C where tokens are $0 |

## Next

→ ../12_capstone_smart_hotel/TUTORIAL.md (Capstone — sovereign autonomous hotel) — the capstone wires every layer together, and the flywheel you just spun is the one that makes its agents get cheaper and better from their own traffic.
