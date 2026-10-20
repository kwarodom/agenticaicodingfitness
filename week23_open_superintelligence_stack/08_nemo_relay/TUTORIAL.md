# ▶ Hands-on Lab 08 — NeMo Relay: observe, trace, right-size, export
> Part of Week 23 · The Open Superintelligence Stack. Companion explainer app: `.venv/bin/python week23/08_nemo_relay/tutorial_server.py` → http://127.0.0.1:8107. This lab is the DO-IT side: you type the commands, you see the output.

**What you'll actually do**
- Build a ~40-line relay that captures a REAL agent turn (tool + LLM calls) as telemetry spans with measured latency and token counts.
- Stand up a real Arize Phoenix on :6006 and auto-instrument an OpenAI-compatible call so a genuine span tree appears in the UI.
- Write and measure a model-right-sizing router: easy prompts → a small model, hard prompts → the big one, priced both ways.
- POST an OTel-shaped trace to Phoenix over OTLP-HTTP, then curate slow/expensive spans into flywheel-ready JSONL.
- Learn the one honest caveat the demos insist on: "equal outcome" from routing is a claim until an eval proves it.

**Time** ~45 min · **Difficulty** intermediate · **Cost** $0 on Spark/local; cloud path is usage-billed.

## 0 · Pick your path
| Path | You have | You'll use |
|---|---|---|
| A 🖥️ | a DGX Spark (or any Linux GPU box) | real commands over SSH |
| B ☁️ | an nvapi- key from build.nvidia.com | hosted NIM endpoints, same API |
| C 💻 | just this laptop | local Ollama (localhost:11434) or the app's SIM mode |
(every numbered step below marks which paths it applies to)

One thing to know before you start: **NeMo Relay itself is early/limited availability — no public container was verifiable when this was written** (the runbook marks it `[UNCERTAIN: Relay availability — re-check build.nvidia.com]`). So verify with a search on build.nvidia.com before expecting a `docker pull`. Everything Relay *does* — spans, Phoenix trace trees, routing, OTel export — is real and buildable today, and that is exactly what you build below.

## 1 · Point the labs at a model (A · B · C)
Goal: the labs inherit their endpoint from `config.py`'s `DGX_CONN` resolution — set it once, everything works.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness

# A — your Spark (Ollama serving on the box; repo default is the Tailscale tunnel)
export DGX_BASE_URL=http://<spark-hostname>:11434/v1

# B — build.nvidia.com hosted NIMs (usage-billed; list live model IDs, never hardcode)
export DGX_CONN=cloud DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1 DGX_API_KEY=nvapi-...

# C — this laptop (Ollama)
ollama pull qwen3.6:35b-a3b-q8_0    # or any model you already have

# verify whichever you picked answers the OpenAI shape:
curl -s ${DGX_BASE_URL:-http://localhost:11434/v1}/models | head -c 300
```

Expected output:
```
{"object":"list","data":[{"id":"qwen3.6:35b-a3b-q8_0","object":"model", ...
```
✓ Checkpoint: you should now see a JSON model list. No endpoint at all? Every lab below still runs — it prints its real commands and a clearly-labeled expected-output sample instead of crashing.

## 2 · Observe — capture one real turn as spans (A · B · C)
Goal: the demos *show* spans; here you *make* them — a mini agent turn (tool → LLM → tool) recorded by a relay you can read in one screen.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
.venv/bin/python week23/08_nemo_relay/labs/lab01_build_a_relay.py
```

Expected output (abbreviated; your numbers WILL differ — they're measured):
```
  ▣ Lab 01 — build a minimal relay (spans from a REAL agent turn)
▣ step 1 — tool span: read this folder's demos (real file I/O, real timing)
  ✓ span [tool ] tool · read_demos               0.8 ms
▣ step 2 — llm span: one REAL call, usage tokens captured from the response
  ✓ span [llm  ] llm · summarize              2223.4 ms
...
  │   llm · summarize              │ llm   │  2223 ms │ $0.000017 │
  │ hermes-turn                    │ agent │  2225 ms │ $0.000000 │
✓ OTel-shaped trace written → .../week23/08_nemo_relay/.sandbox/trace_lab01.json
```
✓ Checkpoint: you should now have `.sandbox/trace_lab01.json` — a real trace in OTel `resourceSpans` shape, with `gen_ai.usage.*` token counts pulled from the actual API response. Lab03 exports this file.

## 3 · Stand up Phoenix for real and watch a span arrive (A · C; B for the model half)
Goal: run the actual trace UI this app teaches. Phoenix is pip-installable and verified `SPARK-1` in the runbook — UI + OTLP-HTTP on **:6006**, OTLP-gRPC on **:4317**.

```bash
# A — on the Spark:  uv pip install arize-phoenix && python -m phoenix.server.main serve
#     then from your laptop:  ssh -L 6006:localhost:6006 <spark>
# C — locally:
cd /Users/altodev/Desktop/agenticaicodingfitness
uv pip install arize-phoenix openinference-instrumentation-openai
.venv/bin/python -m phoenix.server.main serve   # leave running; UI → http://localhost:6006
```

In a second terminal, auto-instrument a real call (B users: swap base_url/key/model for `https://integrate.api.nvidia.com/v1` + `nvapi-…` + a live model ID from `/v1/models`):

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
.venv/bin/python - <<'PY'
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
register(project_name="relay-lab", endpoint="http://localhost:6006/v1/traces")
OpenAIInstrumentor().instrument()
from openai import OpenAI
c = OpenAI(base_url="http://localhost:11434/v1", api_key="dgx")
r = c.chat.completions.create(model="qwen3.6:35b-a3b-q8_0",  # ← any ID from step 1's /models list
    max_tokens=120,
    extra_body={"reasoning_effort": "none"},  # thinking models: answer, don't ruminate
    messages=[{"role": "user", "content": "One sentence: why trace every agent call?"}])
print(r.choices[0].message.content)
PY
```

Expected output (banner abbreviated):
```
🔭 OpenTelemetry Tracing Details 🔭
|  Phoenix Project: relay-lab
|  Collector Endpoint: http://localhost:6006/v1/traces
|  Transport: HTTP + protobuf
Tracing every agent call turns opaque behavior into inspectable, debuggable evidence.
```
✓ Checkpoint: open http://localhost:6006 → project **relay-lab** → Traces: one `ChatCompletion` span with model name, token counts, latency, and the full prompt/response. That last part is the caveat — **traces contain prompts/PII, so keep Phoenix on the box or VPN only** (runbook §2.9).

## 4 · Optimize — route requests to the right-sized model (A · B · C)
Goal: write the router instead of reading about it — a heuristic classifier picks small vs big *from the models actually on your endpoint*, then real calls prove the shape of the saving.

```bash
# optional but worthwhile on A/C: give the router a genuinely small backend
ollama pull gemma3:12b     # any small direct-answer model works

cd /Users/altodev/Desktop/agenticaicodingfitness
.venv/bin/python week23/08_nemo_relay/labs/lab02_rightsize_router.py
```

Expected output (abbreviated, measured):
```
▣ backends picked from your endpoint — small: gemma3:12b   big: qwen3.6:35b-a3b-q8_0
▣ route(  easy) → gemma3:12b
    · 1.1s · 8 out-tok · « SOFTWARE
▣ route(  hard) → qwen3.6:35b-a3b-q8_0
    · 11.4s · 227 out-tok · « Likely causes: impeller wear vs a partially closed...
◈ pricing the SAME measured tokens two ways (illustrative $/M-token tiers):
    relay-routed   $0.000572
    always-big     $0.000630
    → 9% cheaper at (claimed) equal outcome.
```
✓ Checkpoint: you should now see per-request routing decisions with real latencies, and a cost delta. Note the honesty line the lab prints: the saving % is dominated by how much *easy* traffic you have, and "equal outcome" needs App 09's LLM-judge before you trust it in production.

## 5 · Learn — export the trace over OTLP and curate for the flywheel (A · C; B works too)
Goal: do one real leg of the exporter fan-out — POST lab01's trace to Phoenix — then filter it for slow/expensive spans, which is precisely what App 11's Data Flywheel trains on.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
.venv/bin/python week23/08_nemo_relay/labs/lab03_otel_export.py
# Phoenix on another host?  PHOENIX_ENDPOINT=http://<spark>:6006 .venv/bin/python ...
```

Expected output (abbreviated):
```
▣ using the REAL trace lab01 measured → .../.sandbox/trace_lab01.json
▣ probing Phoenix at http://localhost:6006 …
  ✓ HTTP 200 from Phoenix — trace accepted over OTLP-HTTP (protobuf).
◈ 'Learn' — curate slow/expensive spans (the Data Flywheel's feedstock):
  {"span": "llm · summarize", "latency_ms": 2142, "out_tokens": 46, "why": ["slow"], ...}
  → 2 candidate span(s). Scrub PII before any export — spans carry prompts.
```
✓ Checkpoint: you should now see your lab01 span tree inside the Phoenix UI (project *default*), plus curated JSONL lines on stdout. Swap the endpoint and the same OTLP doc would feed Datadog or LangSmith — one instrumentation, many backends, which is the entire Relay pitch.

## Labs (run these)
- **labs/lab01_build_a_relay.py** — a minimal relay records a real tool→LLM→tool turn as spans, prints a Phoenix-style tree from measured numbers, and writes the trace as OTel JSON. Run: `cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/08_nemo_relay/labs/lab01_build_a_relay.py`. Look for: the llm span dwarfing the tool spans, and `gen_ai.usage.*` tokens taken from the response, not guessed. **Modify it:** add a second `llm_step` with a different prompt and confirm two llm spans appear in the tree and in the JSON.
- **labs/lab02_rightsize_router.py** — a heuristic router classifies easy/medium/hard, picks small/big backends from your live endpoint, makes the real calls, and prices routed vs always-big. Run: `cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/08_nemo_relay/labs/lab02_rightsize_router.py`. Look for: which model each request was routed to and the cost delta. **Modify it:** add a fourth request that *should* be easy but contains the word "why" — watch the heuristic misroute it, then fix `route()`.
- **labs/lab03_otel_export.py** — POSTs lab01's trace to Phoenix over OTLP-HTTP and curates slow/expensive spans into JSONL. Run: `cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/08_nemo_relay/labs/lab03_otel_export.py`. Look for: the HTTP 200 from Phoenix and which spans got flagged and why. **Modify it:** lower `SLOW_MS` to 100 and re-run — nearly every span becomes a "candidate", which is the curation-threshold lesson in one line.

## Try it yourself
1. **Find the money span.** Run lab01 three times, then read `.sandbox/trace_lab01.json` and compute which span kind (tool vs llm) accounts for >90% of total latency.
   <details><summary>Solution</summary>

   ```bash
   cd /Users/altodev/Desktop/agenticaicodingfitness
   .venv/bin/python - <<'PY'
   import json
   doc = json.load(open("week23/08_nemo_relay/.sandbox/trace_lab01.json"))
   for rs in doc["resourceSpans"]:
       for ss in rs["scopeSpans"]:
           for s in ss["spans"]:
               ms = (int(s["endTimeUnixNano"]) - int(s["startTimeUnixNano"])) / 1e6
               print(f'{s["name"]:<28} {ms:8.1f} ms')
   PY
   ```
   The llm span dominates — which is why the router (step 4) targets model calls, not tool calls.
   </details>
2. **Break the router, then trust it.** In lab02, route the *hard* pump question to the small model on purpose. Compare its answer to the big model's. Is the outcome actually equal?
   <details><summary>Solution</summary>
   Change `model = big if tier == "hard" else small` to always use `small`, re-run, and diff the two "hard" answers by eye. Usually the small model's diagnosis is shallower or misses the disambiguating test — that gap is exactly what an LLM-judge (App 09) scores systematically. The router only earns the "equal outcome" claim on requests where the judge can't tell the difference.
   </details>
3. **Second backend, same spans.** Send the step-3 instrumented call through a *different* endpoint (e.g. path B's `https://integrate.api.nvidia.com/v1`) and confirm Phoenix shows both traces in the same project with different `llm.model_name`s.
   <details><summary>Solution</summary>
   Re-run the step-3 heredoc with `base_url="https://integrate.api.nvidia.com/v1"`, `api_key="nvapi-..."`, and a model ID taken from `curl -s -H "Authorization: Bearer $DGX_API_KEY" https://integrate.api.nvidia.com/v1/models` (the runbook verifies `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`; Super/Ultra suffixes are [UNCERTAIN] — list, don't guess). In Phoenix, the Traces table now has two rows; the attributes panel shows each call's model. One instrumentation, two providers — that's the gateway idea.
   </details>

## Troubleshooting
| Symptom | Cause | Fix |
|---|---|---|
| `404` model not found | wrong model ID for that endpoint | `curl $BASE/models` and use an ID from the list; cloud IDs are namespaced like `nvidia/...` |
| `401 Unauthorized` | missing/stale key, or tunnel basic-auth | cloud: `DGX_API_KEY=nvapi-...`; ngrok tunnels: creds in the URL (`user:pass@host`) |
| `404`/`405` on every call | `/v1` missing from base URL | Ollama is `:11434/v1`, NIM/vLLM `:8000/v1` — the app's 🔌 Connection auto-appends it |
| `exec format error` pulling containers on Spark | x86 image on aarch64 | Spark is ARM64 — use `linux/arm64` / NGC Spark tags only |
| vLLM/NIM won't start on :8000 | port 8000 contention (both default to it) | run the second server on another port (`--port 8355` is the repo convention) |
| Phoenix UI dead at :6006 | not started, or started on the other box | `.venv/bin/python -m phoenix.server.main serve`; on Spark, `ssh -L 6006:localhost:6006 <spark>` |
| `ModuleNotFoundError: phoenix.otel` | otel extras not installed | `uv pip install arize-phoenix-otel openinference-instrumentation-openai` |
| lab03 gets HTTP 415/400 from Phoenix | endpoint rejected the JSON OTLP encoding | use the step-3 SDK path (`register()` + instrumentor) — protobuf is OTLP-HTTP's canonical encoding |
| lab02's "easy" answers are rambling | routed to a thinking model that spends tokens reasoning | pull a direct-answer small model (`ollama pull gemma3:12b`) — same lesson as `view.classify()` |
| you expected to `docker pull` NeMo Relay | Relay is early/limited availability, [UNCERTAIN] | check build.nvidia.com for current status; meanwhile the Phoenix+OTel+router pattern above IS the substrate |

## Next
→ ../09_inference_economics/TUTORIAL.md (Inference economics — cost/M-token, throughput per GPU and per MW, goodput, LLM-judge eval) — Relay's spans just gave you latency and token counts per call; App 09 turns those exact numbers into $/M-token, goodput, and the eval that proves (or kills) the router's "equal outcome" claim.
