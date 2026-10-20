# ▶ Hands-on Lab 09 — Inference Economics: cost/M-token, throughput per GPU & per MW, goodput

> Part of Week 23 · The Open Superintelligence Stack. Companion explainer app: `.venv/bin/python week23/09_inference_economics/tutorial_server.py` → http://127.0.0.1:8108. This lab is the DO-IT side: you type the commands, you see the output.

**What you'll actually do**
- Measure your own endpoint's REAL tokens/sec — first with `ollama run --verbose`, then from a streamed API call with TTFT split out
- Turn watts × electricity price ÷ throughput into $/M-token for YOUR box and compare it to a cloud list price
- Compute throughput with both denominators — tokens/s per GPU and tokens/s per Megawatt
- Benchmark **goodput** (cost per SUCCESSFUL task) with self-verifying micro-tasks your model actually answers
- Run an LLM-as-judge over a full golden set — and score the judge itself against ground truth

**Time** ~45 min · **Difficulty** intermediate · **Cost** $0 on Spark/local; cloud path is usage-billed.

## 0 · Pick your path

| Path | You have | You'll use |
|---|---|---|
| A 🖥️ | a DGX Spark (or any Linux GPU box) | real commands over SSH |
| B ☁️ | an nvapi- key from build.nvidia.com | hosted NIM endpoints, same API |
| C 💻 | just this laptop | local Ollama (localhost:11434) or the app's SIM mode |

(every numbered step below marks which paths it applies to)

## 1 · Launch the companion app and check your mode (A/B/C)

Goal: see which mode config.py auto-detected — REAL (a live endpoint answered) or SIM (illustrative constants; formulas still learnable, $0, no GPU).

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
.venv/bin/python week23/09_inference_economics/tutorial_server.py
```

Expected output:

```
  ▣  AI Performance & Evaluation — the economics of intelligence
      ◈ SIM mode — no endpoint reachable, using illustrative constants.
        every formula is learnable with no GPU. Go REAL anytime:
        ollama run qwen3.6:35b-a3b-q8_0   (or set DGX_BASE_URL)
      open  →  http://127.0.0.1:8108
```

(If an endpoint is up you'll see `✓ REAL endpoint: <model> @ <base_url>` instead.)

✓ Checkpoint: browser open on :8108, Ch 1–5 listed on the left, and you know whether you're REAL or SIM. Leave it running — steps below use a second terminal.

## 2 · Point at a real endpoint (A/B/C)

Goal: give the demos and labs a live OpenAI-compatible API to MEASURE against. All three paths converge on the same env vars (config.py resolves them).

**A — DGX Spark** (Ollama serves on :11434 out of the box):

```bash
# on the Spark (once):
ollama pull qwen3.6:35b-a3b-q8_0
# on the laptop — over Tailscale or LAN:
export DGX_BASE_URL=http://<your-spark>.<your-tailnet>.ts.net:11434/v1
curl -s $DGX_BASE_URL/models | head -c 300
```

**B — build.nvidia.com cloud on-ramp** (usage-billed — NOT the sovereign $0 path):

```bash
export DGX_CONN=cloud
export DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1
export DGX_API_KEY=nvapi-...        # from any model page → "Get API Key"
# ALWAYS list live model IDs first — the runbook marks exact Super/Ultra
# suffixes as uncertain, so verify with /v1/models before hardcoding one:
curl -s -H "Authorization: Bearer $DGX_API_KEY" \
  https://integrate.api.nvidia.com/v1/models | python3 -m json.tool | grep '"id"' | head -5
export DGX_MODEL=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning   # verified ID
```

**C — local Ollama on this laptop:**

```bash
ollama serve &            # if not already running
ollama pull qwen3:4b      # any small model works for the math
export DGX_CONN=local
curl -s http://localhost:11434/v1/models | head -c 300
```

Expected output (any path):

```
{"object":"list","data":[{"id":"qwen3:4b","object":"model", ...
```

✓ Checkpoint: `curl .../v1/models` returns a JSON model list. No model list → no REAL mode; C-path learners can still do every step in SIM.

## 3 · Measure raw tokens/sec by hand (A/C; B → use lab01)

Goal: get the one number every formula in this app divides by — YOUR tok/s.

**A/C — Ollama prints its own timing with `--verbose`:**

```bash
ollama run qwen3:4b --verbose "Write one sentence about chillers."
```

Expected output (abbreviated — the two `rate` lines are the payload):

```
Chillers remove heat from a liquid via vapor-compression ...
total duration:       2.1s
prompt eval rate:     312.44 tokens/s     ← PREFILL (reading the prompt)
eval rate:            42.17 tokens/s      ← DECODE (generating) — use this one
```

**B —** the hosted API has no `--verbose`; lab01 below measures tok/s from a streamed call the same way for any endpoint (mind the free tier's rate limiting).

✓ Checkpoint: you have a decode tok/s number written down. Prefill ≫ decode is normal — decode is bound by memory bandwidth (273 GB/s on a Spark), not FLOPs.

## 4 · Do the $/M-token math on paper (A/B/C — pure arithmetic, works in SIM too)

Goal: Ch 2's formula, `cost/Mtok = infra cost ÷ tokens`, with your numbers. A Spark draws ~240 W under load; substitute your box's watts and tariff.

```bash
python3 - <<'PY'
watts, kwh_price, tok_s = 240, 0.15, 42.0     # ← EDIT: your watts, tariff, step-3 tok/s
box_usd, years = 3999, 3
elec_hr  = watts/1000 * kwh_price             # $ of electricity per hour
amort_hr = box_usd / (years*8760)             # $ of hardware per hour
tok_hr   = tok_s * 3600
print(f"electricity only : ${elec_hr/tok_hr*1e6:.4f} / 1M tokens")
print(f"+ amortized box  : ${(elec_hr+amort_hr)/tok_hr*1e6:.4f} / 1M tokens")
print(f"cloud list price : $1.80 / 1M tokens (illustrative)")
PY
```

Expected output:

```
electricity only : $0.2381 / 1M tokens
+ amortized box  : $1.2445 / 1M tokens
cloud list price : $1.80 / 1M tokens (illustrative)
```

✓ Checkpoint: you can explain why the amortization line dominates the electricity line at single-stream tok/s — and why batching (more tok/s from the same watts) is the whole serving-optimization game (App 03 · Dynamo). The dollar figures are illustrative teaching constants, not price quotes.

## 5 · Throughput's two denominators — per GPU and per MW (A/B/C)

Goal: Ch 3's lens. Same tokens/s, divided by silicon and by POWER — at datacenter scale you run out of megawatts before you run out of GPU budget.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
.venv/bin/python week23/09_inference_economics/demos/step02_throughput.py
```

Expected output (abbreviated; in REAL the `1×` row is YOUR measured tok/s):

```
  serving config                tok/s  tok/s / GPU   tok/s / MW
  ──────────────────────────────────────────────────────────────
  1× DGX Spark (Nano)              54           54      225,000
  8× GPU node (Super)            3200          400      571,429
  Dynamo cluster (disagg)       42000          656      954,545
```

✓ Checkpoint: you can read the table both ways — per-GPU says "did we tune the engine well", per-MW says "how much product per grid connection". Note the Spark's per-MW number is respectable precisely because it sips 240 W.

## Labs (run these)

**labs/lab01_measure_cost_per_mtok.py** — streams one real generation, splits TTFT from decode rate, counts tokens (exact `usage` when the server reports it, honest ~4-chars/tok estimate otherwise), then derives $/M-token from watts + tariff + amortization and compares to a cloud list price. This is step 3+4 done properly, for ANY endpoint including path B.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/09_inference_economics/labs/lab01_measure_cost_per_mtok.py
```

Look for: the `TTFT … tok/s decode` line, and whether tokens were `exact (usage)` or `estimated`. On path B it flags that the wall-power math is hypothetical (the cloud call itself is usage-billed). Modify it: set `LAB_WATTS`/`LAB_KWH_PRICE` to your real wall numbers, then change `PROMPT` to something 3× longer and watch $/Mtok stay put while TTFT grows — cost is per token, latency is per request.

**labs/lab02_goodput_bench.py** — Ch 4 hands-on: four self-verifying micro-tasks (17×23, string reversal, primality, a capital) run live, get graded by a word-boundary match on the exact expected answer (terse or chatty replies both count; `<think>` traces stripped), and produce a MEASURED success rate → `$/success = $/attempt ÷ success_rate`, plus a sensitivity table for 35%/70%/95% success.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/09_inference_economics/labs/lab02_goodput_bench.py
```

Look for: any `✗ FAIL` row (small models often miss the reversal), and how one failure moves $/SUCCESS while $/attempt is unchanged. Modify it: add a fifth task your model will plausibly fail (e.g. "How many r's in 'strawberry'? Reply with only the number." → `3`) and re-run — watch goodput degrade with the success rate.

**labs/lab03_llm_judge.py** — Ch 5 hands-on, one level deeper than demos/step04_evaluate.py (which judges a single case live): the live endpoint judges ALL five golden answers, and each verdict is compared to ground truth. You get two scores — the golden-set score (how good the answers were) and JUDGE AGREEMENT (how good the judge is), gated at 80%.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/09_inference_economics/labs/lab03_llm_judge.py
```

Look for: whether the judge FAILs `2 + 2 = 5` and `17 = 3 × 6` — a weak judge PASSes them, agreement drops below the gate, and the lab tells you not to trust any goodput number built on it. Modify it: swap `config.MODEL` for a smaller judge (`DGX_MODEL=qwen3:4b` env) and compare agreement — you just measured why judge choice is an economics decision.

All three labs are graceful without an endpoint: they print the exact commands to start one, plus a clearly-labeled `[no endpoint — showing expected output]` sample — never fabricated as real.

## Try it yourself

**1. Your break-even utilization.** Using step 4's math: at your measured tok/s, how many hours/day must the box serve tokens before owning beats a $1.80/Mtok cloud API for 10M tokens/day?

<details><summary>Solution</summary>

Cloud: 10M × $1.80/1M = $18.00/day. Your box costs (elec_hr + amort_hr) × 24 ≈ ($0.036 + $0.152) × 24 ≈ $4.51/day whether it serves or idles — so if it can PRODUCE 10M tokens/day it wins immediately (~4× cheaper). The real constraint is capacity: at 42 tok/s single-stream you make only 42×86400 ≈ 3.6M tokens/day, so you need ~3× concurrency (batching — Dynamo's job) to hit 10M. Ownership economics are a throughput problem, not a price problem.
</details>

**2. Cheap-model trap, with your numbers.** Take lab02's measured success rate and tokens/attempt. At what per-token price would a hypothetical 95%-success model be worth switching to?

<details><summary>Solution</summary>

Set the two $/success equal: `p95 × toks/1e6 ÷ 0.95 = p_yours × toks/1e6 ÷ rate_yours`, so `p95 = p_yours × 0.95 ÷ rate_yours`. If your model measured 75% at $0.45/Mtok, anything cheaper than $0.45 × 0.95/0.75 = $0.57/Mtok is a win — a 27% HIGHER sticker price still beats you per successful task. Run lab02 with `LAB_MTOK=0.57` to confirm the crossover.
</details>

**3. Break the judge, then fix it with the rubric.** Change lab03's judge prompt to the vague `"Is this a good answer? PASS or FAIL."` and re-run; then restore the strict version. What happens to agreement?

<details><summary>Solution</summary>

Vague rubrics make judges lenient — `2 + 2 = 5` often gets a PASS because it's "a confident, well-formatted answer", and agreement drops (commonly to 60–80%). The strict "Is the answer correct?" framing anchors the judge on correctness, not style. That one prompt line is the difference between an eval gate and a rubber stamp — the same reason Week 10/15's golden sets ship rubrics, not vibes.
</details>

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `404 model not found` on API calls | wrong model ID for THIS endpoint | `curl $BASE/models` and use an id from the list; set `DGX_MODEL=<id>` — never trust a hardcoded cloud ID |
| `401 Unauthorized` | missing/expired key, or tunnel basic-auth | cloud: `DGX_API_KEY=nvapi-...`; ngrok tunnel: key as `user:pass` (config.py sends Basic) |
| `404`/`405` on every route | base URL missing the port or `/v1` | Ollama is `:11434/v1`, NIM/vLLM `:8000/v1`; config.py auto-appends `/v1` only when the path is empty |
| App says SIM though Ollama runs | endpoint resolves to the (unreachable) Spark tunnel default | `export DGX_CONN=local` (or `DGX_BASE_URL=http://localhost:11434/v1`) and restart |
| `exec format error` pulling images on the Spark | x86 image on aarch64 | Spark is ARM64 — use NGC `linux/arm64` tags (e.g. `-dgx-spark` NIMs), never random Docker Hub x86 images |
| Port 8000 already in use on the Spark | vLLM/NIM/TRT/Dynamo all default to 8000 | only one can own it — `docker ps` to see which; move the newcomer (`-p 8001:8000`) |
| Cloud calls suddenly erroring/slow | free-tier rate limiting (~40 req/min commonly reported; exact limits unpublished) | pace lab02/lab03 or point `DGX_CONN=local` at Ollama |
| lab02/lab03 verdicts look wrong on a thinking model | reasoning trace confused the extraction | the labs request `reasoning_effort: "none"` (dropped automatically if the server rejects it), strip `<think>…</think>`, and match on word boundaries / last mention; if it persists, pin a direct-answer model: `DGX_MODEL=gemma3:4b` |
| lab rows show `⏱ timeout` / "wall-clock budget hit" | endpoint decodes too slowly for the 25 s/call · 50 s/lab budgets | the labs score what completed — a timeout counts as a FAILED task on purpose (too-slow IS a goodput miss); warm the model first (`ollama run <model> "hi"`) or pick a smaller one |
| tok/s absurdly high or low in lab01 | server didn't report `usage`, so tokens were estimated | fine for teaching; for exact counts use an endpoint that honors `stream_options.include_usage` or read `ollama run --verbose` |

## Next

→ ../10_nemo_gym_rl/TUTORIAL.md (NeMo Gym + NeMo RL — verifiable-reward environments, rollouts, GRPO post-training) — you can now MEASURE success and price it; Gym turns that same pass/fail signal into a reward the model is trained against, closing the loop from evaluation to improvement.
