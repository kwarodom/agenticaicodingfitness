# ▶ Spark Lab 13 — Close the loop: evaluate, merge, serve and route your fine-tune

> Part of Week 25 · DGX Spark: fine-tune, serve, and build sandboxed agents. You type the commands, you see the real output. Every lab also runs in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's playbook, or a clearly marked EXAMPLE.

**What you'll actually do**
- Measure the **baseline** a fine-tune has to beat: the base model with a good prompt and no training.
- Score Module 09's LoRA fine-tune with the **same rules**: valid JSON, right department, no missed emergency, no invented times.
- Serve the fine-tune three ways: the **adapter on vLLM** without merging, a **merged model** on vLLM, or **GGUF on Ollama**.
- Put it behind one **LiteLLM alias** with fallbacks, and see why a silent fallback can hide a dead fine-tune.
- Write the **ship gate**: the code that decides whether a fine-tune goes live.

**Time** ~45 min · **Difficulty** intermediate · **Hardware** 1 Spark (or none: the baseline, gateway and gate run on the laptop)

**Official playbooks covered:** none directly. This module is **course-original** glue between [LLaMA Factory](https://build.nvidia.com/spark/llama-factory) (Module 09), [vLLM](https://build.nvidia.com/spark/vllm) (Module 05) and the LiteLLM gateway (Module 08). vLLM's LoRA flags are vLLM's own CLI, not an NVIDIA playbook, and the text says so where they appear.

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Module 09's dataset in the repo | `ls week25/09_llama_factory/data/` shows `hotel_ops_eval.json` | 60 held-out guest messages the model never trained on |
| Module 09's training finished on the Spark (optional) | `ls ~/w25/m09/saves/qwen3-4b-hotel/lora/sft` on the Spark | the adapter this module serves and scores |
| Module 09's predict step ran (optional) | `ls ~/w25/m09/saves/qwen3-4b-hotel/lora/predict/` on the Spark | the fine-tune's answers to the 60 messages |
| The LiteLLM venv from Module 08 | `ls week25/.venv-litellm/bin/litellm` | lab 13-4 starts the gateway |
| Ollama on the laptop (optional) | `curl -s localhost:11434/api/tags` | the stand-in for the baseline when no Spark answers |

```bash
# on: laptop
ls week25/09_llama_factory/data/ week25/.venv-litellm/bin/litellm
```

**Expected output**

```
dataset_info.json
hotel_ops.json
hotel_ops_eval.json
week25/.venv-litellm/bin/litellm
```

✓ Checkpoint: the eval file and the LiteLLM binary exist. If you have a Spark, Module 09's adapter folder exists too.

## 1 · The loop, in one picture

Training a model is the middle of the job, not the end. A fine-tune earns a place in production only if it beats the cheaper option, on a test it never saw, by rules your code can check.

```text
 Module 09                     this module
┌──────────┐  adapter  ┌──────────────────────┐   ┌─────────────────────┐   ┌───────────────────┐
│  train   │ ────────► │ predict + SCORE      │──►│ SERVE               │──►│ ROUTE + GATE      │
│  LoRA    │           │ same 60 messages,    │   │ A adapter on vLLM   │   │ one alias for     │
│  on the  │           │ same rules as the    │   │ B merged on vLLM    │   │ clients, fallback,│
│  Spark   │           │ baseline (lab 13-1/2)│   │ C GGUF on Ollama    │   │ ship / don't ship │
└──────────┘           └──────────────────────┘   └─────────────────────┘   └───────────────────┘
       ▲                                                                              │
       └──────────────────────── retrain when the gate says ✕ ────────────────────────┘
```

The task is Module 09's hotel router. A guest message goes in, one line of JSON comes out:

```text
{"department": "security", "priority": "urgent", "reply": "Security is on the way …"}
```

`labs/_hotel_eval.py` holds the scoring rules, and every lab in this module imports it. The fine-tune and the baseline are therefore judged by exactly the same code:

| Gate metric | Needed to ship | Why this bar |
|---|---|---|
| `json_valid` | ≥ 98% | code parses the answer; invalid JSON is an outage, not a typo |
| `department_acc` | ≥ 90% | the router's main job |
| `urgent_recall` | **100%** | a missed "smoke in the corridor" is not a rounding error |
| `reply_policy` | ≥ 95% | the system prompt forbids promising a time the guest did not ask for |

✓ Checkpoint: you can say why `urgent_recall` has a stricter bar than `department_acc`.

## 2 · The baseline: lab 13-1

Before trusting a fine-tune, measure the cheaper alternative: the **base model with a good prompt**. Lab 13-1 sends the 60 held-out messages, with the same system prompt the training data uses, to an un-fine-tuned model. It tries the Spark's vLLM serving `Qwen/Qwen3-4B-Instruct-2507` (the model Module 09 fine-tunes) first, then Ollama on your laptop, then DRY.

```bash
# on: laptop
.venv/bin/python week25/13_finetune_to_serve/labs/lab01_baseline.py
```

**Expected output** (captured on this Mac; no Spark was reachable, so the baseline is gemma4:12b as a LAPTOP STAND-IN, a different and bigger model than the Spark's base)

```
▣ STEP 1 · send 60 held-out guest messages with Module 09's system prompt
→ POST http://localhost:11434/v1/chat/completions · model=gemma4:12b · Ollama on THIS laptop (stand-in, not the Spark)
│ 10/60 answered · 17s
…
│ 60/60 answered · 168s
◆ 60 predictions from gemma4:12b (LAPTOP STAND-IN) → week25/13_finetune_to_serve/.runs/predictions_baseline_laptop.jsonl

▣ STEP 2 · score them with the same rules the fine-tune will face
│ gate metric     baseline  needed to ship
│ ──────────────  ────────  ──────────────  ─
│ json_valid      97%       ≥ 98%           ✕
│ department_acc  97%       ≥ 90%           ✓
│ urgent_recall   80%       ≥ 100%          ✕
│ reply_policy    97%       ≥ 95%           ✓
│ priority accuracy 82% · urgent cases in this set: 10

▣ STEP 3 · where does it go wrong?
│ expected security       got None           × 2
│ not valid router JSON: '{"department": "security", "priority": urgent, "reply": "Thank you for alerting us; our se'
│ not valid router JSON: '{"department": "security", "priority": urgent, "reply": "Thank you for alerting us; our se'
═ baseline does not pass the ship gate (LAPTOP STAND-IN). Lab 13-2 scores the fine-tune's predictions with the same rules; lab 13-4 puts them side by side.
```

Read that result carefully, because it is typical:

- **97% department accuracy looks excellent**, and a demo would pass.
- **The failures cluster on the cases that matter.** Both invalid answers are security emergencies. The model wrote `"priority": urgent` without quotes, so code cannot parse it, and the smoke report is lost. A bigger model with a good prompt still drops 2 of 10 emergencies.
- **Priority accuracy is 82%, and the errors have a language.** The model flagged 9 normal requests as urgent, and **all 9 were in Thai**: 9 of the 10 normal Thai messages ("the TV in room 527 won't turn on") became urgent, against 0 of 40 English ones. An average score hides a bias that a Thai-speaking night team would feel on the first night. Slice every eval by language (and by department) before you trust it.

This is the honest case for fine-tuning a small model on a narrow task: consistency on your format, your languages and your edge cases, rather than general smarts. Module 09's training set mixes English and Thai for exactly this reason. Another fix is to make invalid JSON impossible. vLLM's structured outputs (a JSON schema on the request) constrain decoding so the model cannot write `urgent` without quotes. Try both, and let the gate decide.

✓ Checkpoint: you ran lab 13-1 (Spark or laptop), and `.runs/predictions_baseline_*.jsonl` exists. You can name the metric the baseline fails on.

## 3 · Score the fine-tune: lab 13-2

Module 09's lab 04 already ran LLaMA Factory's predict step on the Spark:

```bash
# on: spark
cd ~/w25/m09 && llamafactory-cli train configs/hotel_predict.yaml
```

It writes `generated_predictions.jsonl`, with one `{"prompt", "predict", "label"}` object per held-out message. (LLaMA Factory's predict output format; the config uses greedy decoding so the scores do not wobble between runs.) Lab 13-2 reads that file over SSH (read-only), scores it with the same gate, and keeps a copy in `.runs/` for lab 13-4:

```bash
# on: laptop
.venv/bin/python week25/13_finetune_to_serve/labs/lab02_score_finetune.py
```

**Expected output** (captured on this Mac in DRY mode: there is no fine-tune to read, so the lab says so and scores the baseline file to show the report format)

```
▣ STEP 1 · get the fine-tune's predictions
$ test -f ~/w25/m09/saves/qwen3-4b-hotel/lora/predict/generated_predictions.jsonl && wc -l < … || echo MISSING   [DRY]
◈ DRY: no fine-tune predictions reachable; scoring predictions_baseline_laptop.jsonl instead

▣ STEP 2 · score 60 predictions · BASELINE predictions_baseline_laptop.jsonl (no fine-tune available — shown so you can read the report)
│ gate metric     score  needed to ship
│ ──────────────  ─────  ──────────────  ─
│ json_valid      97%    ≥ 98%           ✕
│ department_acc  97%    ≥ 90%           ✓
│ urgent_recall   80%    ≥ 100%          ✕
│ reply_policy    97%    ≥ 95%           ✓
…
▣ STEP 4 · slice by guest language — an average can hide a bias
│ language  n   department  priority  normal → urgent  urgent recall
│ ────────  ──  ──────────  ────────  ───────────────  ─────────────
│ en        50  96%         96%       0%               80%
│ th        10  100%        10%       90%              100%
═ BASELINE predictions_baseline_laptop.jsonl (no fine-tune available — shown so you can read the report): does not pass the ship gate.
◆ LLaMA Factory's own metrics (BLEU/ROUGE in predict_results.json) measure word overlap with the label. For a router, the gate above (valid JSON, right department, no missed emergency) is what matters.
```

Step 4 is the slice from section 2, computed: the baseline routes Thai messages to the right department every time, but marks 9 of 10 normal Thai requests urgent. Ten messages is a small sample, and one run of one model is not a verdict on Thai in general. But it is exactly the kind of gap a fine-tune on bilingual data should close, and exactly what an average hides. Compare this row for the fine-tune.

With a Spark, step 2 shows the fine-tune's own numbers. Scored files from any source can be compared with `--file path.jsonl`.

> 💡 **Why not use LLaMA Factory's BLEU and ROUGE?** They count shared words between the answer and the label. A reply that routes a gas smell to *housekeeping* in beautiful English can score well on ROUGE. Score what your code consumes.

✓ Checkpoint: with a Spark, you have `.runs/predictions_finetune_spark.jsonl` and its gate table. Without one, you can explain what lab 13-2 would read and from where.

## 4 · Serve it: adapter, merged, or GGUF (lab 13-3)

A LoRA fine-tune is a ~60 MB folder of adapter weights on top of an unchanged base model. There are three ways to serve it, and lab 13-3 prints all three with your adapter's real base model and rank:

```bash
# on: laptop
.venv/bin/python week25/13_finetune_to_serve/labs/lab03_serve_it.py            # plan only
.venv/bin/python week25/13_finetune_to_serve/labs/lab03_serve_it.py --launch   # copy the adapter + start path A
```

**Expected output** (captured on this Mac in DRY mode; the adapter config is an EXAMPLE shape)

```
▣ STEP 1 · read the adapter's own config: which base, which rank?
$ cat ~/w25/m09/saves/qwen3-4b-hotel/lora/sft/adapter_config.json && ls -la … | grep -E 'safetensors|tokenizer'   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
{"base_model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507", "r": 16, "lora_alpha": 32, …}
✓ base = Qwen/Qwen3-4B-Instruct-2507 · rank r = 16 → --max-lora-rank must be ≥ 16

▣ STEP 2 · three ways to serve it

→ A · adapter on vLLM, no merge (one base can carry many adapters)
  docker run -d --name w25-hotel-vllm --gpus all --ipc host \
    --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$HOME/w25/adapters:/adapters" \
    --entrypoint '' vllm/vllm-openai:latest \
    vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 8192 --gpu-memory-utilization 0.3 \
      --enable-lora --lora-modules hotel-ft=/adapters/hotel-ft --max-lora-rank 16

→ B · merged model on vLLM (one plain model, no LoRA flags)
  cd ~/w25/m09 && llamafactory-cli export configs/hotel_merge.yaml     # writes ~/w25/m09/saves/qwen3-4b-hotel/merged
  …

→ C · GGUF on Ollama (smallest, simplest, slowest to update)
  python ~/llama.cpp/convert_hf_to_gguf.py ~/w25/m09/saves/qwen3-4b-hotel/merged --outtype q8_0 --outfile ~/w25/hotel-router-q8_0.gguf
  …

▣ STEP 3 · start path A on the Spark (opt-in)
◆ Not launched. Re-run with --launch to copy the adapter to ~/w25/adapters/hotel-ft and start vLLM.
```

> ⚠ **Course addition.** Path A's `--enable-lora`, `--lora-modules` and `--max-lora-rank` are vLLM's own flags; no NVIDIA playbook covers LoRA serving. Module 05 section 7 shows how to confirm them for your vLLM version. Path B uses LLaMA Factory's merge config from Module 09. Path C uses llama.cpp's converter from Module 04's build in `~/llama.cpp`. The converter stores the model's chat template in the GGUF. If `ollama run hotel-router` answers in a strange format, add a `TEMPLATE` line to the Modelfile (Ollama's docs show the format).

| | A · adapter on vLLM | B · merged on vLLM | C · GGUF on Ollama |
|---|---|---|---|
| What you ship | a ~60 MB adapter folder | a full model (~8 GB at bf16 for 4B) | one quantized file |
| Retrain → live | swap the folder, restart | merge again, restart | merge, convert, `ollama create` |
| Many fine-tunes on one base | ✓ many `--lora-modules`, one base in memory | ✕ one model each | ✕ one model each |
| Speed | a small LoRA overhead per token | full base speed | llama.cpp speed, smaller memory |
| Best for | iterating, A/B tests, per-customer adapters | the version you ship | edge, laptops, Open WebUI users |

`--gpu-memory-utilization 0.3` leaves room for a second engine or the gateway on the same Spark. A 4B model at bf16 is ~8 GB of weights, and Module 05's lab 01 computes the rest.

✓ Checkpoint: you can say which path you would use while you are still retraining, and which you would use for the version that ships.

## 5 · Route it, and gate it (lab 13-4)

Clients should never hard-code `hotel-ft` or a Spark's hostname. Lab 13-4 writes a LiteLLM config (Module 08's generator) with **one alias**, `hotel-router`, and a fallback chain:

```text
hotel-router  →  hotel-ft on Spark A's vLLM       (the fine-tune)
      └─ fails → hotel-base on Spark A's vLLM     (same base model, prompt-only)
            └─ fails → gemma4:12b on the laptop   (last resort)
```

It starts the gateway on this laptop, sends five held-out messages to `hotel-router`, and reads LiteLLM's response headers to show who really answered. Then it puts the saved prediction files side by side:

```bash
# on: laptop
.venv/bin/python week25/13_finetune_to_serve/labs/lab04_route_and_gate.py
```

**Expected output** (captured on this Mac: a real LiteLLM 1.89 proxy; no Spark, so every call fell back to the laptop)

```
▣ STEP 1 · write the gateway config: hotel-router → hotel-ft → hotel-base → laptop
│ alias         backend model                            api_base
│ ────────────  ───────────────────────────────────────  ─────────────────────────
│ hotel-router  hosted_vllm/hotel-ft                     http://spark-a:8000/v1
│ hotel-base    hosted_vllm/Qwen/Qwen3-4B-Instruct-2507  http://spark-a:8000/v1
│ hotel-laptop  openai/gemma4:12b                        http://localhost:11434/v1
│ fallbacks: hotel-router → hotel-base → hotel-laptop

▣ STEP 2 · start the gateway and send 5 held-out messages to `hotel-router`
$ LITELLM_MASTER_KEY=sk-w25-… week25/.venv-litellm/bin/litellm --config week25/13_finetune_to_serve/.runs/hotel-gateway.yaml --host 127.0.0.1 --port 4000 --telemetry False
◆ gateway up on http://127.0.0.1:4000 after 2.6 s (pid 52797, log 08_litellm_gateway/.runs/litellm-4000.log)
◆ gateway stopped (pid 52797); port 4000 is free again: True
│ guest message                        HTTP  answered by  fallbacks  right dept + JSON
│ ───────────────────────────────────  ────  ───────────  ─────────  ─────────────────
│ Good evening. Could you recommend …  200   gemma4:12b   1          ✓
│ Hello, the bathroom in 1418 wasn't…  200   gemma4:12b   1          ✓
│ Could you recommend a good seafood…  200   gemma4:12b   1          ✓
│ What is the Wi-Fi password? I'm in…  200   gemma4:12b   1          ✓
│ The bathroom in 606 wasn't cleaned…  200   gemma4:12b   1          ✓
◆ Every answer needed a fallback: the fine-tune is not being served right now. Clients noticed nothing — which is exactly why you must LOG `x-litellm-attempted-fallbacks`, or a dead fine-tune goes unnoticed.

▣ STEP 3 · side by side: baseline vs fine-tune, same 60 messages, same gate
│ gate metric     baseline (predictions_baseline_laptop.jsonl)  needed
│ ──────────────  ────────────────────────────────────────────  ──────
│ json_valid      97%                                           ≥ 98%
│ department_acc  97%                                           ≥ 90%
│ urgent_recall   80%                                           ≥ 100%
│ reply_policy    97%                                           ≥ 95%
⚠ no fine-tune predictions: the ship decision needs Module 09's adapter scored on a Spark (lab 13-2).
```

Two lessons from a run where nothing on the Spark was up:

1. **Fallbacks hide failures.** Every request got HTTP 200 and a good answer, so a dashboard of status codes would look perfect while the fine-tune you paid to train served nothing. The `x-litellm-attempted-fallbacks` header is the only sign, so log it and alert on it.
2. **The decision is a table, not a feeling.** With a Spark, step 3 shows a second column for the fine-tune and prints **SHIP** or **DO NOT SHIP**. Keep the baseline file: every future retrain is scored against it.

> 💡 A fallback from `hotel-ft` to `hotel-base` is a **quality** downgrade, not only an availability one. For a safety-critical router you might prefer failing loudly (HTTP 503 and a human dispatcher) to answering with the model the gate rejected. The gateway config is where you make that choice explicit.

✓ Checkpoint: you ran lab 13-4, saw `x-litellm-attempted-fallbacks` in the table, and can explain why an all-200 dashboard is not proof the fine-tune is live.

## Labs — run them here

**labs/lab01_baseline.py** — The prompt-only base model on the 60 held-out hotel messages, scored by the ship gate: the number a fine-tune must beat.

**labs/lab02_score_finetune.py** — Read LLaMA Factory's predictions for the fine-tune from the Spark and score them with exactly the same rules.

**labs/lab03_serve_it.py** — Read the adapter's base and rank, print the three ways to serve it, and (opt-in) start the adapter on vLLM.

**labs/lab04_route_and_gate.py** — One LiteLLM alias with a fallback chain, who really answered each request, and the side-by-side ship decision.

`labs/_hotel_eval.py` is the shared scorer, not a lab. The capstone (Module 20) imports it too.

## Try it yourself

**Exercise 13 — the ship gate.** Open `week25/13_finetune_to_serve/exercises/ex13_ship_gate.py`. Three `TODO`s:

1. `parse_answer(text)`: extract the router's JSON from real model output (clean, inside ```json fences, or after a chatty sentence), and return `None` for invalid JSON.
2. `promises_time(reply, guest_message)`: flag "in 10 minutes" and "within the hour", but not the guest's own booking time echoed back.
3. `ship(metrics, baseline)`: the four gate bars plus "no worse than the baseline", returning the list of rules that failed.

```bash
# on: laptop
.venv/bin/python week25/13_finetune_to_serve/exercises/ex13_ship_gate.py
```

**Expected output** (once all three TODOs are done)

```
✓ parse_answer: clean · fenced · with chatter → dict; unquoted value and prose → None
✓ promises_time: 'in 10 minutes' ✓ · 'within the hour' ✓ · echoed 19:00 ✗ · no time ✗
✓ ship: the good run ships · the bad run is blocked by urgent_recall, reply_policy and the baseline

▣ why the bad run is blocked
│ urgent_recall failed (90%)
│ reply_policy failed (90%)
│ beats baseline department_acc failed (95%)

═ A 95% department score still fails: one missed emergency is worse than ten misrouted towel requests.
```

<details><summary>Hint — why does `{"priority": urgent}` have to be rejected?</summary>

It is exactly what the laptop baseline wrote for two smoke reports in lab 13-1. `json.loads` raises on it, so the router's caller would crash or drop the message. A lenient parser that "fixes" it hides a model bug you need to see in the gate.

</details>

<details><summary>Stretch — structured outputs instead of fine-tuning</summary>

vLLM accepts a JSON schema on the request (`"response_format": {"type": "json_schema", …}`), which constrains decoding to valid JSON. Add it to lab 13-1's request, re-run on the Spark's base model, and see which gate metrics move. Which failures does it fix, and which (a wrong department, a missed urgent) can only training or a better prompt fix?

</details>

✓ Checkpoint: all three checker lines are ✓.

## Troubleshooting

| Symptom | Fix |
|---|---|
| lab 13-2: `generated_predictions.jsonl does not exist` | Run Module 09 lab 04's predict step first (`llamafactory-cli train configs/hotel_predict.yaml` from `~/w25/m09`) |
| vLLM: `LoRA rank 16 is greater than max_lora_rank 8` | Set `--max-lora-rank` to at least the adapter's `r` (lab 13-3 reads it from `adapter_config.json`) |
| vLLM lists the base model but not `hotel-ft` | The `--lora-modules` path is **inside the container** (`/adapters/hotel-ft`); check the `-v "$HOME/w25/adapters:/adapters"` mount and `ls ~/w25/adapters/hotel-ft` |
| `/v1/models` shows `hotel-ft`, but answers look like the base model | The adapter was trained with `template: qwen3_nothink`. Send the same system prompt the training data used, and compare against lab 13-1's baseline to be sure |
| lab 13-4: every row shows a fallback | Nothing serves `hotel-ft` on Spark A's :8000 yet. Run lab 13-3 with `--launch`, then check `curl http://spark-a:8000/v1/models` |
| lab 13-4: `litellm … not found` | Create the venv from Module 08: `uv venv week25/.venv-litellm && uv pip install -p week25/.venv-litellm/bin/python "litellm[proxy]==1.89.0"` |
| Baseline scores swing between runs | Use `temperature: 0` (the labs do) and a fixed eval file; LLaMA Factory's predict config sets `do_sample: false` for the same reason |

## Next

Continue to [Lab 14 — NeMo Agent Toolkit: agents on your Spark's models](../14_nat_agents/TUTORIAL.md): give the router tools and a reasoning loop, served from your Spark.
