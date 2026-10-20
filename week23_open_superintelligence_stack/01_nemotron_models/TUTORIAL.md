# ▶ Hands-on Lab 01 — Nemotron 3 Open Model Family
> Part of Week 23 · The Open Superintelligence Stack. Companion explainer app: `.venv/bin/python week23/01_nemotron_models/tutorial_server.py` → http://127.0.0.1:8100. This lab is the DO-IT side: you type the commands, you see the output.

**What you'll actually do**
- Point this folder at a real Nemotron endpoint (your Spark, build.nvidia.com, or local Ollama) and verify it answers.
- Watch a reasoning model think — split the private REASON channel from the ANSWER on a live stream, and measure the "reasoning tax".
- Run one raw tool-calling round trip and read all four JSON shapes on the wire.
- Do the fit math yourself: which Nemotron tier fits 1 Spark, 2 Sparks, or only the cloud.
- (Path A) Pull and serve Nemotron Nano on your own DGX Spark.

**Time** ~45 min · **Difficulty** beginner→intermediate · **Cost** $0 on Spark/local; cloud path is usage-billed.

## 0 · Pick your path
| Path | You have | You'll use |
|---|---|---|
| A 🖥️ | a DGX Spark (or any Linux GPU box) | real commands over SSH |
| B ☁️ | an nvapi- key from build.nvidia.com | hosted NIM endpoints, same API |
| C 💻 | just this laptop | local Ollama (localhost:11434) or the app's SIM mode |

(every numbered step below marks which paths it applies to)

## 1 · Point this folder at a Nemotron (A/B/C)
Everything here (app, demos, labs) resolves its endpoint from `config.py` — set it once, all paths use the same OpenAI API.

**A — your Spark.** SSH in (or use the app's 🖥️ DGX console), pull the model, then aim your laptop at it:
```bash
ssh <you>@<spark-host>
ollama pull nemotron-3-nano          # the reasoning tier; ~20 GB download
ollama list
exit
# back on the laptop — Tailscale is the repo default:
export DGX_CONN=tunnel DGX_TUNNEL_URL=http://<spark>.<tailnet>.ts.net:11434/v1
```

**B — build.nvidia.com.** Sign in (free Developer Program account, no credit card), open any model page, click "Get API Key" — the key starts with `nvapi-`:
```bash
export DGX_CONN=cloud
export DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1
export DGX_API_KEY=nvapi-...
```

**C — this laptop.** If you run Ollama locally, `ollama pull nemotron-3-nano` (or keep whatever model you already have — the labs auto-pick). No Ollama at all? Skip the export; the app and labs fall back to SIM and still print every real command.

Verify — one command for all three paths (B uses `$DGX_CLOUD_URL`, A falls back to `$DGX_TUNNEL_URL`, C to local Ollama; local ignores the auth header):
```bash
curl -s "${DGX_CLOUD_URL:-${DGX_TUNNEL_URL:-http://localhost:11434/v1}}/models" \
  -H "Authorization: Bearer ${DGX_API_KEY:-dgx}" | head -c 400
```
Expected output (path B shown; A/C list whatever your box serves — e.g. `nemotron-3-nano` or `gemma4:12b`):
```
{"object":"list","data":[{"id":"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", ...
```
✓ Checkpoint: you should now see a JSON model list containing at least one usable model id. Copy the exact id you'll use — never guess it.

## 2 · Meet the family — then do the fit math (A/B/C)
Goal: know the three tiers cold, and be able to compute what fits where. First run the explainer's own table:
```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
.venv/bin/python week23/01_nemotron_models/demos/step01_family.py
```
Expected output:
```
  Model               Total  Active    Ctx  VRAM(NVFP4)  tok/s  Sparks
  ──────────────────────────────────────────────────────────────────────
  Nemotron 3 Nano       30B    3.0B  1000K         22GB     54       1
  Nemotron 3 Super     120B   12.0B  1000K         78GB     20       1
  Nemotron 3 Ultra     550B   55.0B  1000K        360GB      9       2
  ... (plus the RAG / Speech / Safety 8B specialists)
```
Then do the arithmetic yourself (this is lab03 Part A, see Labs below). The formula: `weights_GB = params_B × GB_per_B × 1.18` vs `usable = 128 GB × 0.9` per Spark. One honest wrinkle: the table above prints `Sparks 2` for Ultra, but run the row yourself and you'll get `550 × 0.55 × 1.18 ≈ 357 GB` — more than TWO Sparks (230 GB usable), so treat Ultra as the CLOUD path today; the 2-Spark TP=2 recipe (step 5) is how you serve models in the ~200–400B class.

✓ Checkpoint: you can say, with numbers, why Nano fits one Spark at Q8, Super needs Q4/NVFP4, and Ultra is cloud-only.

## 3 · Watch it think — REASON → ANSWER on the raw API (A/B/C)
Goal: see the reasoning channel with your own curl, no wrapper. Path B shown (A/C: swap the URL for `http://localhost:11434/v1` or your tunnel, and the model id for `nemotron-3-nano`, drop the auth header for local):
```bash
curl -s https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $DGX_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
       "messages":[{"role":"user","content":"What is NVFP4? Two sentences."}],
       "max_tokens":300}' | python3 -m json.tool | head -30
```
Expected output (abbreviated — the reasoning may appear in a `reasoning`/`reasoning_content` field or inline as `<think>…</think>`, depending on the server):
```
"message": {
    "role": "assistant",
    "reasoning_content": "The user asks about NVFP4. It is NVIDIA's 4-bit floating point...",
    "content": "NVFP4 is NVIDIA's 4-bit floating-point format for Blackwell GPUs. It roughly halves..."
```
Cloud model-id note: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` is verified on build.nvidia.com; the exact Super/Ultra id suffixes are [UNCERTAIN] — verify with `GET /v1/models` (step 1) first rather than trusting any tutorial, this one included. Some personal orgs get 403 "missing public API endpoints permission" on Super — that's an account-scope issue, not a typo.

✓ Checkpoint: you should now see thinking text arrive separately from (or before) the answer text — the RLM pattern from app Ch 4, on the raw wire.

## 4 · Make it call YOUR tools (A/B/C)
Goal: run one tool-call round trip and understand that the model only *emits JSON* — your code executes. The polished loop is `demos/step04_tool_calling.py`; the dissected version is lab02:
```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
.venv/bin/python week23/01_nemotron_models/demos/step04_tool_calling.py   # the loop
.venv/bin/python week23/01_nemotron_models/labs/lab02_toolcall_wire.py   # the wire
```
Expected output (lab02, abbreviated):
```
◈ WIRE ← assistant tool_calls, turn 1 (shape 2 of 4) · finish_reason='tool_calls'
  [{"id": "call_...", "function": {"name": "get_chiller_status", "arguments": "{\"chiller_id\": \"3\"}"}}]
◈ WIRE → role:"tool" result you append (shape 3 of 4)
  {"role": "tool", "tool_call_id": "call_...", "content": "{\"cop\": 3.8, \"state\": \"fault\"...}"}
· FINAL: Chiller 3 is faulted — low refrigerant pressure...
```
✓ Checkpoint: you can name the four wire shapes: tools schema → assistant `tool_calls` (finish_reason `tool_calls`) → your `role:"tool"` result → final answer.

## 5 · Stand it up on your Spark (A — C-path: run the SIM equivalent)
Goal: serve Nemotron on hardware you own. **Real Spark hardware required for the commands below**; C-path learners run `demos/step05_run_on_dgx.py` (or the app's Ch 6), which prints these same commands and simulates the run — nobody is blocked.

One Spark — Ollama (the default REAL path, port 11434):
```bash
[SPARK] curl -fsSL https://ollama.com/install.sh | sh   # needs Ollama 0.15+ for Blackwell
[SPARK] ollama pull nemotron-3-nano
[SPARK] OLLAMA_HOST=0.0.0.0 ollama serve                # expose off-box (or use Tailscale)
```
One Spark — as a NIM container (port 8000). The Nemotron-3 Nano NIM for Spark is [UNCERTAIN] in the runbook — verify a `-dgx-spark` (ARM64) tag exists at catalog.ngc.nvidia.com under `nvcr.io/nim/nvidia/` first; if there is none, stay on Ollama or the cloud NIM:
```bash
[SPARK] echo $NGC_API_KEY | docker login nvcr.io --username '$oauthtoken' --password-stdin
[SPARK] docker run -d --name nim --gpus all --shm-size 16GB -p 8000:8000 \
  -e NGC_API_KEY=$NGC_API_KEY -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  nvcr.io/nim/meta/llama-3.1-8b-instruct-dgx-spark:latest    # a verified ARM64 NIM
```
Two Sparks — the TP=2 recipe (QSFP 200GbE, for ~200–400B-class models):
```bash
[SPARK] docker run --gpus all --network host nvcr.io/nvidia/pytorch:latest \
  all_reduce_perf -b 8 -e 256M -f 2 -g 1                # verify NCCL: expect tens of GB/s busbw
[SPARK] mpirun -H 192.168.100.10,192.168.100.11 -np 2 \
  trtllm-serve <model> --tp_size 2 --port 8355
```
Expected output (Ollama path):
```
pulling manifest... success
>>> the model answers on http://<spark>:11434/v1 — same OpenAI API as steps 1-4
```
✓ Checkpoint: `curl http://<spark>:11434/v1/models` from your laptop lists `nemotron-3-nano`, and re-running step 3 against it costs $0.

## Labs (run these)
All labs inherit the step-1 connection from `config.py`, terminate in under a minute, and degrade gracefully to a labeled expected-output sample if no endpoint is up.

**labs/lab01_reasoning_tax.py** — sends an EASY and a HARD prompt to the same model, splits REASON from ANSWER on the stream, and measures tokens/latency of each: the reasoning tax, quantified.
```bash
cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/01_nemotron_models/labs/lab01_reasoning_tax.py
```
Look for: the HARD prompt spending several times the REASON tokens of the EASY one, and what that does to latency. *Modify it:* add a third MEDIUM prompt and see whether the tax scales smoothly or jumps.

**labs/lab02_toolcall_wire.py** — one tool-calling round trip with every raw JSON message printed: the tools schema, the assistant `tool_calls`, the `role:"tool"` result, the final answer.
```bash
cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/01_nemotron_models/labs/lab02_toolcall_wire.py
```
Look for: `finish_reason='tool_calls'` — the model is paused, waiting for YOUR code. *Modify it:* add a second tool `dispatch_service(chiller_id)` and a system rule for when to call it; watch the model chain two calls.

**labs/lab03_fit_math.py** — Part A computes the full fit table (FP16/Q8/Q4/NVFP4 × Nano/Super/Ultra × 1–2 Sparks) from the runbook formula, no endpoint needed; Part B buries a needle in a 4k-char log haystack and asks the model to find it.
```bash
cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/01_nemotron_models/labs/lab03_fit_math.py
```
Look for: the Ultra row failing even the 2-Spark budget at NVFP4 — sizing is arithmetic, not folklore. *Modify it:* change `SPARK_GB` to 784 (a DGX Station) and see which verdicts flip.

## Try it yourself
1. **Route by difficulty.** Using lab01's measurements, write a 10-line router: if the prompt is under 15 words and contains a digit, answer with `max_tokens=30` and no reasoning expected; otherwise allow 280. Measure total latency across 4 mixed prompts vs sending everything the slow way.
<details><summary>Solution</summary>

```python
def route(prompt):
    terse = len(prompt.split()) < 15 and any(c.isdigit() for c in prompt)
    return {"max_tokens": 30 if terse else 280,
            "messages": [{"role": "user", "content":
                          ("Answer in <=1 sentence: " if terse else "") + prompt}]}
```
Call `client.chat.completions.create(model=config.MODEL, **route(p))` per prompt and sum `time.time()` deltas. Typical result: the terse half returns 3–10× faster — this is the Nano-routes/Super-orchestrates economics from Ch 2, in miniature.
</details>

2. **Break the tool loop on purpose.** In lab02, change the system prompt to omit "Pass chiller_id as a bare digit" and ask about "Chiller #3". What arrives in `arguments`, and where would a naive `STATUS[cid]` lookup fail?
<details><summary>Solution</summary>
The model often emits `{"chiller_id": "#3"}` or `{"chiller_id": "Chiller 3"}` — valid JSON, wrong key. `STATUS["#3"]` returns the error dict. This is why `demos/step04_tool_calling.py` ships a `_room_key()` normalizer: tool implementations must defensively parse model-authored arguments. The fix is either normalization in your tool (lab02's `_impl` already strips a leading `#`) or a stricter schema (`"pattern": "^[1-3]$"`).
</details>

3. **Find the KV-cache cliff.** Extend lab03 Part A: assume KV-cache costs ~0.15 GB per 1k tokens of context for Super at NVFP4. At what context length does Super stop fitting one Spark?
<details><summary>Solution</summary>
Super weights: `120 × 0.55 × 1.18 ≈ 78 GB`. Usable: `128 × 0.9 = 115 GB`. Headroom ≈ 37 GB → `37 / 0.15 ≈ 247k tokens`. So "1M context" on the model card does not mean 1M on one 128 GB Spark with Super — context is a memory budget you spend. (The 0.15 GB/1k figure is illustrative; measure your real deployment — the runbook says budget 10–30 GB for 100k+ agentic runs.)
</details>

## Troubleshooting
| Symptom | Cause | Fix |
|---|---|---|
| 404 model not found | wrong model id for THIS endpoint | `GET {base}/v1/models` and copy an id exactly; cloud Super/Ultra suffixes are [UNCERTAIN] — never hardcode |
| 401 Unauthorized | missing/expired key, or tunnel basic-auth | cloud: `Bearer $DGX_API_KEY` (starts `nvapi-`); ngrok tunnels: key as `user:pass` |
| 403 "missing public API endpoints permission" | build.nvidia.com account scope (seen on Super) | try the verified Nano id or a fallback like `nvidia/nvidia-nemotron-nano-9b-v2`; check account org |
| Connection refused / HTML error page | `/v1` missing from the base URL, or wrong port | Ollama is `:11434/v1`, NIM/vLLM/TRT are `:8000/v1`; `config.py` auto-appends `/v1` only if the path is empty |
| `exec format error` pulling containers on Spark | x86 image on aarch64 (the #1 Spark ops trap) | use `linux/arm64` NGC tags (`-dgx-spark` NIMs, `nvcr.io/nvidia/*` Spark tags), never random Docker Hub images |
| Two servers fight over port 8000 on the Spark | vLLM vs NIM vs TRT vs Dynamo all default to 8000 | run one at a time, or remap (`-p 8001:8000`); identify who answered by the `/v1/models` payload |
| Labs print "[no endpoint — showing expected output]" | nothing reachable at the resolved base URL | re-do step 1; force with `DGX_MODE=real` to see the actual error instead of SIM |
| First call after `ollama run` takes ~a minute | model loading into unified memory | normal on first load; subsequent calls are fast — don't lower the timeout below 30 s |

## Next
→ ../02_nim_microservices/TUTORIAL.md (NIM microservices — one signed container = model + optimized engine + OpenAI API; build.nvidia.com) — you now have the model; NIM is how you package and serve it as a production endpoint instead of a dev-box Ollama.
