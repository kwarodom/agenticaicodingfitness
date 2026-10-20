# ▶ Hands-on Lab 03 — NVIDIA Dynamo: disaggregated serving, KV-aware routing, SLO planning
> Part of Week 23 · The Open Superintelligence Stack. Companion explainer app: `.venv/bin/python week23/03_dynamo_serving/tutorial_server.py` → http://127.0.0.1:8102. This lab is the DO-IT side: you type the commands, you see the output.

**What you'll actually do**
- Measure TTFT and decode tok/s on real streamed calls and prove *prefill is compute-bound, decode is memory-bound* — the fact Dynamo's whole architecture rests on.
- Reuse a ~500-token agent prefix and watch warm TTFT collapse, then build a 20-line KV-cache-aware router in pure stdlib.
- Declare TTFT/ITL SLOs, ramp concurrency 1→2→4 against one worker, and generate by hand the exact breach signal the SLO Planner autoscales on.
- Compute your own $/1M-tokens from measured throughput and the Spark's 240 W wall power.
- (Path A) Stand up a single-node Dynamo frontend + worker on the Spark and probe its OpenAI-compatible endpoint.

**Time** ~45 min · **Difficulty** intermediate · **Cost** $0 on Spark/local; cloud path is usage-billed.

## 0 · Pick your path
| Path | You have | You'll use |
|---|---|---|
| A 🖥️ | a DGX Spark (or any Linux GPU box) | real commands over SSH |
| B ☁️ | an nvapi- key from build.nvidia.com | hosted NIM endpoints, same API |
| C 💻 | just this laptop | local Ollama (localhost:11434) or the app's SIM mode |
(every numbered step below marks which paths it applies to)

One honest note up front: Dynamo itself is **not a callable cloud model** — build.nvidia.com serves models *behind* fabric like this, it doesn't sell you the fabric. So path B/C do every measurement lab against any OpenAI-compatible endpoint (the physics of prefill/decode are the same), and study the multi-worker scaling through the app's SIM. Path A can additionally run a real single-node Dynamo, and 2 QSFP-linked Sparks make real disaggregation.

## 1 · Ground yourself in the explainer (A/B/C)
Goal: know the four pieces (disaggregated P/D, KV-aware routing, SLO Planner, NIXL) before you measure them.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
.venv/bin/python week23/03_dynamo_serving/tutorial_server.py
# open http://127.0.0.1:8102 — run Ch 2 (four pieces) and Ch 3 (disaggregation), or standalone:
.venv/bin/python week23/03_dynamo_serving/demos/step01_what_is_dynamo.py
```

Expected output:
```
▣  NVIDIA Dynamo — serving long-running agents at scale
    ◈ SIM mode — no endpoint reachable, simulating a DGX Spark.   (or ✓ REAL …)
    open  →  http://127.0.0.1:8102
```
✓ Checkpoint: you can name the four pieces and say which inference phase is compute-bound and which is memory-bound.

## 2 · Point the labs at a live endpoint (A/B/C)
Goal: get `config.py` resolving a real OpenAI-compatible endpoint so the labs run REAL, not SIM.

**Path C — local Ollama (simplest):**
```bash
ollama serve &                        # if not already running on :11434
ollama pull qwen3.6:35b-a3b-q8_0      # or any model you have; labs auto-pick
curl -s http://localhost:11434/v1/models | head -c 200
```

**Path A — your Spark over Tailscale/LAN:**
```bash
export DGX_BASE_URL=http://<your-spark>.<your-tailnet>.ts.net:11434/v1   # note the /v1
```

**Path B — build.nvidia.com (usage-billed, not sovereign):**
```bash
export DGX_CONN=cloud DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1 DGX_API_KEY=nvapi-...
curl -s https://integrate.api.nvidia.com/v1/models -H "Authorization: Bearer $DGX_API_KEY" | head -c 300
# never hardcode model IDs — list them live. nvidia/nemotron-3-nano-omni-30b-a3b-reasoning is verified.
```

Expected output (any path):
```
{"object":"list","data":[{"id":"qwen3.6:35b-a3b-q8_0", ...
```
✓ Checkpoint: `/v1/models` answers with a JSON model list. No endpoint at all? Fine — every lab prints the real commands plus a clearly-labeled expected-output sample instead of crashing.

## 3 · (A only) Stand up single-node Dynamo on the Spark
Goal: run the actual Dynamo frontend + one vLLM worker, so "frontend / worker / control plane" stops being a diagram.

The runbook marks the exact module invocations **[UNCERTAIN]** — they drift between Dynamo releases — so verify with `python -m dynamo.frontend --help` and docs.dynamo.nvidia.com first, then:

```bash
# on the Spark (SSH, or the app's 🖥️ DGX console)
uv pip install "ai-dynamo[vllm]"            # or the nvcr.io dynamo container (linux/arm64!)
docker compose up -d etcd nats              # Dynamo's control plane: etcd :2379, NATS :4222
python -m dynamo.frontend --http-port 8000 &
python -m dynamo.vllm --model Qwen/Qwen3-0.6B   # worker registers with the frontend
curl -s http://localhost:8000/v1/models
```

Expected output:
```
{"object":"list","data":[{"id":"Qwen/Qwen3-0.6B","object":"model", ...
```
✓ Checkpoint: the frontend answers `/v1/models` on :8000 — the same OpenAI surface as NIM/vLLM/Ollama, which is the point: Dynamo is a drop-in scale-out layer, not a new API. Port 8000 is contested on the Spark (vLLM vs NIM vs TRT vs Dynamo) — stop the others first. **C-path equivalent:** the app's Ch 2/3 in SIM mode shows the same topology with the real commands printed.

## 4 · Measure prefill vs decode yourself (A/B/C)
Goal: prove with a stopwatch that TTFT scales with prompt length while decode tok/s stays flat.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
.venv/bin/python week23/03_dynamo_serving/labs/lab01_prefill_vs_decode.py
```

Expected output (abbreviated; your numbers will differ):
```
▣ LAB 01 · Prefill vs decode — the two phases, measured
  prompt    in tok  TTFT(ms)  decode tok/s  ITL(ms)
  SHORT         11       142          51.8     19.3
  LONG         849       891          50.9     19.6
✓ TTFT: 6.3x longer on the long prompt — prefill work scales with input length …
✓ decode: 51.8 vs 50.9 tok/s — near-constant (memory-bandwidth-bound).
```
✓ Checkpoint: you have two rows where TTFT diverges sharply and decode tok/s doesn't. That asymmetry — visible on your own hardware — is the entire case for disaggregating the pools. (On the Spark, decode is bounded by the GB10's 273 GB/s memory bandwidth, not its 1 PFLOP FP4 compute.)

## 5 · KV-cache reuse, then route on it (A/B/C)
Goal: see a warm prefix skip prefill, then implement the routing decision that exploits it fleet-wide.

```bash
.venv/bin/python week23/03_dynamo_serving/labs/lab02_kv_cache_router.py
```

Expected output (abbreviated):
```
── Part 1 · cold vs warm prefix (measured) ─────
  call 1 (cold-ish prefix)  TTFT    780 ms
  call 2 (warm prefix)      TTFT    160 ms   → 4.9x
── Part 2 · a cache-aware router in 20 lines ───
  #1   A (dc27244c)   ◈ MISS    spark-0
  #3   A (dc27244c)   ✓ HIT     spark-0
  3/6 requests skipped prefill entirely (50% warm) …
```
✓ Checkpoint: call 2's TTFT dropped (Ollama caches prefixes; a cloud gateway may not — the lab says so honestly), and the router prints HIT/MISS decisions. An agent fleet re-sends the same system prompt millions of times; reuse × routing is where the sim's ~1.8x comes from.

## 6 · Hold an SLO, then price your tokens (A/B/C)
Goal: generate the SLO Planner's input signal by hand, and end with a real $/1M-tokens number.

```bash
.venv/bin/python week23/03_dynamo_serving/labs/lab03_slo_economics.py
# stricter SLO or your local electricity price:
SLO_TTFT_MS=500 ELEC_USD_KWH=0.12 .venv/bin/python week23/03_dynamo_serving/labs/lab03_slo_economics.py
```

Expected output (abbreviated):
```
  SLO declared: TTFT ≤ 2000 ms · ITL ≤ 100 ms
   conc  worst TTFT  mean ITL  agg tok/s   SLO
      1       410ms    19.8ms       50.4   ✓ ✓
      4      2350ms    26.7ms      151.9   ⚠ TTFT breach ✓
◈ economics at your best aggregate rate (151.9 tok/s):
  240 W × $0.15/kWh ÷ throughput → $0.0658 / 1M tokens (electricity only …)
```
✓ Checkpoint: you can point at the row where YOUR worker breaks the SLO and say which pool the Planner would grow (TTFT breach → prefill queueing; ITL creep → decode saturating). Note the throughput row: aggregate tok/s *rose* with concurrency even as TTFT broke — throughput and latency SLOs pull in opposite directions, which is why you declare objectives instead of maximizing one number. Path B: the wall-power math applies to a box you own; the lab labels cloud calls usage-billed.

## 7 · (A ×2 Sparks) Real disaggregation — where SIM ends
Goal: know the real 2-node recipe, even if you run it later.

Real disaggregated prefill/decode with NIXL moving KV cache between nodes needs **two Sparks over the QSFP 200GbE link** (the QSFP port, not the RJ45 jack): cable them, set static IPs, verify the fabric, then start prefill and decode workers on separate nodes registering with one frontend. The exact worker flags drift between releases — verify against docs.dynamo.nvidia.com before trusting any blog post.

```bash
# on each Spark: cable QSFP-to-QSFP, then static IPs on the 200GbE interface
sudo ip addr add 192.168.100.10/24 dev <qsfp-if>     # .11 on the second Spark
ping -c 2 192.168.100.11                              # fabric reachable?
# verify real bandwidth with NCCL before blaming Dynamo for anything:
mpirun -H 192.168.100.10,192.168.100.11 -np 2 ./build/all_reduce_perf -b 1G -e 1G
```

Expected output:
```
# all_reduce_perf: busbw column in the tens of GB/s over the 200GbE link
  1073741824  ...  busbw 21.4 GB/s
```
✓ Checkpoint: you can say what NIXL does (moves KV cache prefill→decode across nodes fast enough to make the split worth it) and why one Spark can't demo it for real. **B/C-path equivalent:** `demos/step02_disaggregated.py` and `demos/step03_slo_planner.py` simulate exactly this topology, with the stacked wins labeled as sim-derived (~4.4x throughput, ~0.24x cost/token — real value depends on your prefix-sharing rate).

## Labs (run these)
All three run from the repo root, inherit `config.py`'s endpoint resolution (local Ollama / Spark tunnel / nvapi- cloud), finish in under a minute, and degrade gracefully to a labeled expected-output sample when no endpoint is up.

- **labs/lab01_prefill_vs_decode.py** — streams a short and a ~850-token prompt, measures TTFT / decode tok/s / ITL for each. Run: `cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/03_dynamo_serving/labs/lab01_prefill_vs_decode.py`. Look for: TTFT growing several-x with prompt length while decode tok/s barely moves. **Modify it:** add a third MEDIUM row (`PASSAGE * 4`) and check TTFT growth is roughly linear in input tokens.
- **labs/lab02_kv_cache_router.py** — measures cold-vs-warm TTFT on a shared ~500-token system prompt, then runs a stdlib prefix-hash router printing HIT/MISS per request. Run: `cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/03_dynamo_serving/labs/lab02_kv_cache_router.py`. Look for: the warm-call TTFT drop, and the hit-rate line. **Modify it:** change the request mix to `["A"]*5 + ["B"]` and watch the hit rate jump — long-lived agents with one shared prefix are cache-routing's best case.
- **labs/lab03_slo_economics.py** — declares TTFT/ITL SLOs, ramps 1→2→4 concurrent streams with a thread pool, flags breaches, then computes $/1M tokens from measured tok/s and 240 W wall power. Run: `cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/03_dynamo_serving/labs/lab03_slo_economics.py`. Look for: which SLO breaks first as concurrency climbs. **Modify it:** set `SLO_TTFT_MS=300` and find the largest concurrency your endpoint can hold — that number is your one-worker capacity, the thing the Planner multiplies.

## Try it yourself

1. **Find your prefix break-even.** Using lab02, shrink `SYSTEM` (change `* 6` to `* 1`) and re-run. At what prefix length does warm-vs-cold stop mattering on your endpoint?
<details><summary>Solution</summary>
With `* 1` the prefix is ~80 tokens; prefill of 80 tokens takes tens of ms, so cold≈warm and the speedup ratio falls toward 1.0x. Somewhere between `* 2` and `* 6` (a few hundred tokens) the gap becomes obvious. Lesson: cache-aware routing pays in proportion to prefix length × reuse frequency — a 4k-token agent system prompt re-sent every turn is the best case; a chat app with unique short prompts gains almost nothing. That's also the honest scope note from the explainer: low-QPS, no-shared-prefix workloads don't need Dynamo.
</details>

2. **Compare two endpoints like a router would.** Run lab01 twice — once against local Ollama, once against build.nvidia.com (`DGX_CONN=cloud DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1 DGX_API_KEY=nvapi-...`) — and explain the TTFT difference.
<details><summary>Solution</summary>
Cloud TTFT includes network RTT + gateway queueing on top of prefill, so its SHORT-prompt TTFT is usually worse than local even though the datacenter GPU prefills faster; the LONG/SHORT TTFT *ratio* is what isolates prefill compute. Decode tok/s on the hosted endpoint reflects a datacenter GPU's memory bandwidth (and per-user rate shaping), not your Spark's 273 GB/s. This is exactly the comparison App 08 (NeMo Relay) automates when routing between endpoints.
</details>

3. **Amortize the box.** Extend lab03's cost line: add the Spark's purchase price (~$4,000 over 3 years of 24/7 operation) to the electricity-only $/1M tokens.
<details><summary>Solution</summary>
Amortization = 4000 / (3 × 365 × 24) ≈ $0.152/hour. Convert to per-M-tokens at your measured rate: `0.152 / (tps * 3600 / 1e6)` — at 150 tok/s that's ≈ $0.28/1M tokens, dwarfing the ~$0.07 of electricity. In lab03, add `AMORT_USD_H = 4000/(3*365*24)` and fold `AMORT_USD_H / (best_tps*3.6e-3)` into `cost`. Lesson: at Spark scale the *hardware*, not the power, dominates $/token — which is why utilization (what Dynamo maximizes) is the real economic lever.
</details>

## Troubleshooting
| Symptom | Cause | Fix |
|---|---|---|
| `404 model not found` | wrong model ID for this endpoint, or wrong endpoint entirely | `curl {base}/models` and use an ID from the live list — never hardcode; cloud IDs are namespaced (`nvidia/…`) |
| `401 Unauthorized` | missing/stale key, or tunnel basic-auth | cloud: `DGX_API_KEY=nvapi-…`; ngrok basic-auth: key as `user:pass`; keys reported valid ~6 months |
| `Connection refused` / labs print SIM sample | no endpoint at `BASE_URL` | start Ollama (`ollama serve`), or fix `DGX_BASE_URL` — and check it ends in **`/v1`** (Ollama `:11434/v1`, Dynamo/vLLM/NIM `:8000/v1`) |
| `curl :8000/v1/models` answers but it's not Dynamo | port 8000 contention — vLLM/NIM/TRT/Dynamo all default to 8000 on the Spark | inspect the payload (a NIM lists one model; Ollama lists many); stop the other service or move ports |
| `exec format error` pulling the Dynamo/vLLM container on the Spark | x86 image on aarch64 | use NGC `linux/arm64` Spark tags (`nvcr.io/…`), never random Docker Hub images |
| `python -m dynamo.frontend` → `No module named …` | Dynamo module names drift between releases (runbook flags this [UNCERTAIN]) | `pip show ai-dynamo`, `python -m dynamo.frontend --help`, and verify against docs.dynamo.nvidia.com |
| Dynamo worker starts but frontend lists no models | etcd/NATS control plane not up | `docker compose up -d etcd nats` (etcd :2379, NATS :4222), then restart the worker |
| lab02 shows no warm speedup | endpoint doesn't cache prefixes (some cloud gateways), or call 1 was already warm | expected — the lab prints this caveat; try local Ollama, or restart the model to force a cold call |
| TTFT huge even at concurrency 1 | thinking model reasoning silently before token one, or slow tunnel | the labs request `reasoning_effort="none"`; if your endpoint ignores it (some do for `qwen3.6`/`gemma4`/`nemotron-3` reasoning models), the ramp *trend* is still the lesson, not the absolute number |
| cloud calls suddenly failing mid-lab | build.nvidia.com free-tier rate limit (~40 req/min commonly reported) | wait a minute; the labs send <10 requests total, but shared keys hit it |

## Next
→ ../04_agent_skills/TUTORIAL.md (NVIDIA Agent Skills — portable framework-agnostic capabilities; Skills + MCP + A2A) — you can now serve a model economically at fleet scale; next you give the agents running on it portable capabilities.
