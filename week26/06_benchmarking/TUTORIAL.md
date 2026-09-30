# ▶ Reef Lab 06 — Performance benchmarking on DGX Spark: engine, workflow, quality, sandbox tax

> Part of Week 26 · NemoClaw on DGX Spark, beginner to expert. You type the commands, you see the real output. Laptop labs (the NAT and OpenShell CLIs, the policy model) run for real everywhere. Spark labs also run in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's docs or playbooks, or a clearly marked EXAMPLE.

**What you'll actually do**
- Split "is my claw fast?" into four separate questions: engine, workflow, quality and sandbox overhead. Then measure each one with its own tool.
- Work out the single-stream decode ceiling from the Spark's 273 GB/s, and learn why real engines land below it.
- Run a real `nat eval` with the NAT profiler on Alto Ops Claw, and read p50/p95, LLM calls per turn, tokens and a quality score from the files it writes.
- Run a real (tiny) `nat sizing calc`, redo its GPU estimate by hand, and see why two points prove nothing.
- Measure a "tax" the right way: the same eval on two paths, with a verdict on whether the difference beats the noise.
- Time a claw through its own API, as you would Hermes on :8642, and put all four layers in one report.

**Time** ~90 min · **Difficulty** advanced · **Hardware** laptop (NAT 1.9.0 + Ollama) · 1 DGX Spark optional

**Sources:** the course's research tutorial *NemoClaw on DGX Spark — Beginner to Expert* (Part 5, labs L5.1–L5.5), which cites [NAT profiler](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/profiler.html) · [NAT evaluate](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/evaluate.html) · [NAT sizing calculator](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/sizing-calc.html) · [ai-muninn — Nemotron 3 Nano on DGX Spark](https://ai-muninn.com/en/blog/dgx-spark-nemotron-3-nano-w4a16-74-toks) · [Exxact — local agents on DGX Spark](https://www.exxactcorp.com/blog/benchmarks/benchmarking-local-ai-agents-on-nvidia-dgx-spark) · [Exxact — inference engines on DGX Spark](https://www.exxactcorp.com/blog/deep-learning/comparing-inference-engines-on-dgx-spark) · [Classmethod — NAT on DGX Spark](https://dev.classmethod.jp/en/articles/dgx-spark-nemo-agent-toolkit-local-intro/) · [NVIDIA developer forum — Nemotron 3 NVFP4](https://forums.developer.nvidia.com/t/dgx-spark-nemotron3-and-nvfp4-getting-to-65-tps/355261) · [Exxact local-agent-benchmark](https://github.com/Exxact-Software/local-agent-benchmark)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| NAT 1.9.0 with the profiler and eval extras | `week26/.venv-nat/bin/nat --version` | `nat eval`, the profiler and `nat sizing calc` run for real on this laptop |
| Laptop Ollama with `nemotron-3-nano` and `gemma3:4b` | `curl -s http://localhost:11434/api/tags` | the stand-in for vLLM on the Spark: nemotron-3-nano for the agent, gemma3:4b for the fast engine sweep |
| The Alto Ops package (Module 04) | installed editable in `week26/.venv-nat` | the `chiller_kpi` tool the eval questions exercise |
| A DGX Spark (optional) | `vllm-nat` container (research tutorial Lab 3.2), the `alto-ops` sandbox (Lab 3.8), a Hermes claw | the real engine, sandbox and harness numbers |

```bash
# on: laptop
week26/.venv-nat/bin/nat --version
curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | grep -E "nemotron-3-nano|gemma3:4b"
```

**Expected output** (captured on this Mac)

```
nat, version 1.9.0
"name":"nemotron-3-nano:latest"
"name":"gemma3:4b"
```

> 📌 **Two rules for every number in this module.**
> 1. **Laptop numbers are a LAPTOP STAND-IN.** They come from this Mac's Ollama, which other labs may be using at the same time, so they are noisy. They show you the *method*. They are never compared with a Spark figure.
> 2. **Spark figures are quoted, never invented.** No module has run on a real Spark yet. Every Spark number below is a third-party measurement, quoted "per <source>, cited in the research tutorial". Spark output blocks without a source are EXAMPLE shapes with `…` where your numbers go.
>
> The research tutorial was written against the NAT 1.8 docs. This course runs NAT **1.9.0**. Every place 1.9.0 differs is called out below.

✓ Checkpoint: `nat --version` prints 1.9.0 and both models are pulled, or you know that only the DRY and arithmetic parts will run.

## 1 · Four layers, measured separately

"The claw feels slow" can mean four different things. Each one has its own tool:

| Layer | Question | Tool | Key numbers | Lab |
|---|---|---|---|---|
| 1 · Engine / model | How fast does the model decode, alone and under load? | `vllm bench serve`, `ollama run --verbose` | TTFT, tok/s per request, aggregate tok/s | 06-1 |
| 2 · Workflow | How long does one agent turn take, and why? | `nat eval` + the NAT profiler | p50/p95 runtime, LLM calls and tokens per turn | 06-2, 06-3 |
| 3 · Quality | Does the claw get the task right? | NAT evaluators (ragas, trajectory), or an exact check | score per question | 06-2 |
| 4 · Sandbox overhead | What does OpenShell's proxy, TLS interception and Landlock add? | the same `nat eval` on the host and in the sandbox | Δ p95 runtime | 06-4 |

Speed alone is not the answer. The research tutorial cites Exxact's agent benchmark: a fast engine with a model that fails tool calls is slower in practice than a slow, reliable one. Reliability and structured-output discipline mattered more than raw tok/s.

These are the reference numbers the research tutorial collects for the Spark. Each one is one author's measurement on one machine and one software version. Reproduce them. Do not quote them as specs.

| Model / engine (per the research tutorial) | Metric | Value | Source |
|---|---|---|---|
| Nemotron 3 Nano 30B-A3B, vLLM, W4A16 NVFP4 | single-stream decode | 74.75 tok/s | ai-muninn |
| same, W4A4 NVFP4 | single / aggregate c=16 | 58.27 / 786 tok/s | ai-muninn |
| same, W4A16 | aggregate | ~400 tok/s | ai-muninn |
| Nemotron 3 Nano NVFP4, vLLM | single-stream | 65+ tok/s | NVIDIA developer forum |
| nemotron-3-nano:30b, Ollama | avg tok/s over agent tasks | 64.7 | Exxact benchmark |
| nemotron-3-super:120b-a12b, Ollama | avg tok/s; task pass | 16.4; 17/17 | Exxact benchmark |
| qwen3.5:35b-a3b / qwen3.5:122b-a10b, Ollama | avg tok/s | 48.2 / 20.1 | Exxact benchmark |
| gemma4:26b, Ollama vs vLLM | single-stream | ~64 vs ~30 tok/s | Exxact engines |
| gemma4:26b, vLLM | aggregate at 10+ concurrent | >300 tok/s | Exxact engines |
| gemma4:26b, Ollama `OLLAMA_NUM_PARALLEL=4` | aggregate | ~122 tok/s | Exxact engines |
| NAT ReAct + 1 tool on Nemotron 3 Nano FP8 | end-to-end per query | ~13 s | Classmethod |

How to read them, from the same sources:

- **Decode is bandwidth-bound.** Per ai-muninn, the single-stream ceiling for a 3B-active MoE on the Spark sits in the low 80s tok/s, whatever the engine does.
- **What feels usable.** Per Exxact, an agent above ~20 tok/s feels usable and 40+ feels responsive.
- **Which engine for whom.** Per Exxact's engine comparison, Ollama wins single-user latency and vLLM wins multi-user throughput. The Mamba-hybrid Nemotron models did not gain from Ollama parallelism.
- **Watch memory.** Exxact recommends a memory watchdog: unified-memory pressure can hard-reset the Spark.

✓ Checkpoint: for each of the four layers, you can name the tool and the one number you would put in a report.

## 2 · L5.1 — Engine: the ceiling, single stream, concurrency sweep

**Start with arithmetic.** When a model decodes one stream, each new token reads every *active* weight once from memory. So the fastest possible single-stream speed is the memory bandwidth divided by the bytes of active weights per token. Nemotron 3 Nano 30B-A3B activates about 3B parameters per token. The Spark has 273 GB/s:

| Format | Active weights per token | Naive ceiling |
|---|---|---|
| BF16 | 6.0 GB | 46 tok/s |
| FP8 | 3.0 GB | 91 tok/s |
| NVFP4 (~4.5 bits with scales) | ~1.7 GB | 162 tok/s |

This is an upper bound, and real engines land below it. Decode also reads the KV cache and the non-expert layers, and no chip runs at peak bandwidth. That is why ai-muninn's practical ceiling (low 80s) is well under the NVFP4 line. The arithmetic tells you two things: which format can possibly hit your target, and how close to the wall a measured number already is.

**Then measure on the Spark.** Use vLLM's built-in benchmark with random 512-token prompts and 256-token answers. Run a single stream first, then a concurrency sweep:

```bash
# on: spark
docker exec vllm-nat vllm bench serve --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
  --backend openai-chat --endpoint /v1/chat/completions --host 127.0.0.1 --port 8000 \
  --dataset-name random --random-input-len 512 --random-output-len 256 \
  --num-prompts 32 --max-concurrency 1
for c in 2 4 8 16; do docker exec vllm-nat vllm bench serve --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
  --backend openai-chat --endpoint /v1/chat/completions --host 127.0.0.1 --port 8000 \
  --dataset-name random --random-input-len 512 --random-output-len 256 \
  --num-prompts $((c*8)) --max-concurrency $c; done
```

**Expected output** (EXAMPLE — illustrative shape, not a measurement)

```
============ Serving Benchmark Result ============
Successful requests:                     32
Maximum request concurrency:             1
Benchmark duration (s):                  …
Output token throughput (tok/s):         …
Mean TTFT (ms):                          …
Mean TPOT (ms):                          …
==================================================
```

Record three numbers per concurrency level: **TTFT**, **output tok/s per request**, and **aggregate** output tok/s. Then run the same prompt set on Ollama, so you have both engines side by side. `ollama run nemotron-3-nano:30b --verbose` prints its own timing after each answer. If you prefer a ready-made agent task suite, Exxact's harness is open source (linked in Sources).

A quick single-request check from the runner (it reports TTFT and tok/s, and says where it ran):

```spark
{"target": "vllm", "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8", "messages": [{"role": "user", "content": "In four short bullet points, explain why a hotel chiller plant's kW/RT gets worse at low cooling load."}], "max_tokens": 256}
```

**On the laptop, the same method.** Lab 06-1 prints the ceiling table and the reference table, shows the Spark commands, then runs a small sweep against Ollama for real: gemma3:4b, one warm-up, then concurrency 1, 2 and 4 with the same prompt (9 calls in all).

```bash
# on: laptop
.venv/bin/python week26/06_benchmarking/labs/lab06_1_engine_sweep.py
```

**Expected output** (captured on this Mac, the stand-in sweep)

```
◆ c=1: 2 requests · 192 tokens in 3.0s · TTFT p50 57 ms · 75.8 tok/s per request · 63.9 tok/s aggregate
◆ c=2: 2 requests · 192 tokens in 4.1s · TTFT p50 1120 ms · 48.9 tok/s per request · 47.1 tok/s aggregate
◆ c=4: 4 requests · 384 tokens in 8.4s · TTFT p50 3188 ms · 47.5 tok/s per request · 45.9 tok/s aggregate
│ concurrency  requests  TTFT p50  tok/s per request  tok/s aggregate                      
│ ───────────  ────────  ────────  ─────────────────  ───────────────  ────────────────────
│ 1            2         57 ms     75.8               63.9             ████████████████████
│ 2            2         1120 ms   48.9               47.1             ███████████████░░░░░
│ 4            4         3188 ms   47.5               45.9             ██████████████░░░░░░
◆ Aggregate changed only ×0.7: this Ollama is queueing requests (OLLAMA_NUM_PARALLEL) or other labs are using it. The research tutorial cites Exxact: Ollama wins single-user latency, vLLM wins multi-user throughput.
```

Read the shape, not the size. On vLLM you expect aggregate tok/s to climb with concurrency while each request slows down: batching shares the bandwidth. If the aggregate stays flat, as it can on a laptop Ollama, the server is queueing requests one after another (`OLLAMA_NUM_PARALLEL`), or other labs are using it.

✓ Checkpoint: you can explain why the NVFP4 ceiling (162 tok/s) is far above the measured 74.75 tok/s ai-muninn reports, and what a flat aggregate line means.

## 3 · L5.2 — Workflow and quality: `nat eval` with the profiler

An engine benchmark says nothing about an *agent turn*. One Alto Ops question costs two LLM calls (pick the tool, then write the answer), a tool run, and every token of both prompts. `nat eval` runs a dataset through the workflow. With `profiler:` set, it records every LLM and tool event.

**The dataset.** Each JSONL line has a `question` (the input) and an `answer` (the reference). A `structure` block in the dataset config maps other column names. Lab 06-2 computes the reference answers from `chiller_plant.csv` with the same arithmetic as the `chiller_kpi` tool, so nobody types a KPI by hand:

| id | question | answer (from the CSV) |
|---|---|---|
| 1 | What is the average plant kW/RT over the last 6 hours? | 0.901 |
| 2 | Is the chiller plant efficiency in ALARM or OK over the last 6 hours? | ALARM |
| 3 | What is the average plant kW/RT over the last 24 hours? | 0.703 |
| 4 | What was the average cooling load in RT over the last 24 hours? | 629.2 |

The research tutorial asks for 20 questions. A shared laptop gets 4.

**The config.** On the Spark you use the research tutorial's `eval_config.yml`: the Alto Ops workflow from Part 3 plus this `eval:` block. The course copy is `configs/eval_config.spark.yml`.

```yaml
eval:
  general:
    output_dir: ./.tmp/eval/alto_ops/
    max_concurrency: 4
    dataset:
      _type: jsonl
      file_path: ./data/alto_ops_eval.jsonl
    profiler:
      token_uniqueness_forecast: true
      workflow_runtime_forecast: true
      compute_llm_metrics: true
      csv_exclude_io_text: true
      prompt_caching_prefixes:
        enable: true
        min_frequency: 0.5
      bottleneck_analysis:
        enable_nested_stack: true
      concurrency_spike_analysis:
        enable: true
        spike_threshold: 7
  evaluators:
    accuracy:
      _type: ragas
      metric: AnswerAccuracy
      llm_name: local_vllm
    trajectory:
      _type: trajectory
      llm_name: local_vllm
```

The laptop copy, `configs/eval_config.yml`, keeps the same profiler block and changes three things. It uses 4 questions, `max_concurrency: 2`, and trace-based evaluators (`avg_llm_latency`, `avg_num_llm_calls`, `avg_workflow_runtime`, `avg_tokens_per_llm_end`) that need no judge LLM.

> ⚠ **NAT 1.9.0: `nat validate` does not catch a misspelt profiler key.** The profiler model ignores unknown fields, so `token_uniqueness_forecst: true` validates with a ✓ and then does nothing. Lab 06-2 checks each key against `ProfilerConfig.model_fields` instead. All nine keys the research tutorial uses exist in 1.9.0.

**Run it:**

```bash
# on: laptop
week26/.venv-nat/bin/nat eval --config_file week26/06_benchmarking/configs/eval_config.yml
```

```bash
# on: spark
cd ~/alto_ops && nat eval --config_file eval_config.yml
nat eval --config_file eval_config.yml --override eval.general.max_concurrency 1   # serial baseline
```

**Expected output** (captured on this Mac, lab 06-2 step 3)

```
=== EVALUATION SUMMARY ===
Workflow Status: COMPLETED (workflow_output.json)
Total Runtime: 74.75s
Workflow Runtime (p95): 39.76s
LLM Latency (p95): 27.35s

Per evaluator results:
| Evaluator        |   Avg Score | Output File                  |
|------------------|-------------|------------------------------|
| llm_latency      |       17.61 | llm_latency_output.json      |
| llm_calls        |        2    | llm_calls_output.json        |
| workflow_runtime |       35.41 | workflow_runtime_output.json |
| tokens_per_call  |      615.75 | tokens_per_call_output.json  |
```

**What lands in the output directory.** The research tutorial lists these files:

**Expected output** (REFERENCE — quoted from the research tutorial, Lab 5.2)

```
#  workflow_output.json  accuracy_output.json  trajectory_accuracy_output.json  config_effective.yml
#  all_requests_profiler_traces.json  inference_optimization.json  standardized_data_all.csv  workflow_profiling_report.txt
```

NAT 1.9.0 on this laptop wrote all six profiler and config files. The judge files follow a different rule: each evaluator writes `<evaluator key>_output.json`. So the tutorial's `trajectory:` key writes `trajectory_output.json`, not `trajectory_accuracy_output.json`. 1.9.0 also writes `config_original.yml`, `config_metadata.json`, `workflow_profiling_metrics.json`, a `gantt_chart.png`, and one file per trace evaluator.

**Reading the numbers.** `inference_optimization.json` holds p90/p95/p99 and confidence intervals for workflow runtime and LLM latency. `standardized_data_all.csv` has one row per event, which gives you p50 and per-call tokens. `workflow_profiling_report.txt` is the bottleneck tree.

**Expected output** (captured on this Mac, lab 06-2 step 5)

```
│ metric                p50   p90   p95   note           
│ ────────────────────  ────  ────  ────  ───────────────
│ workflow runtime (s)  36.7  39.6  39.8  n=4 · mean 35.4
│ LLM latency (s)       —     26.7  27.4  n=8 · mean 17.6
◆ 95% confidence interval of the MEAN runtime: 30.9–39.9 s. With 4 rows it is wide: that is the noise any later comparison (lab 06-4) has to beat.
│ standardized_data_all.csv · LLM_END rows  count  mean  p50  max
│ ────────────────────────────────────────  ─────  ────  ───  ───
│ prompt_tokens                             8      409   409  450
│ completion_tokens                         8      207   217  305
◆ 8 LLM calls for 4 questions = 2.0 per agent turn (tool call → tool → final answer). A ReAct loop or retries would show up here first.
✓ id 1: expected '0.901' · 'The average plant kW/RT over the last 6 hours is **0.901 kW per RT**. '
✓ id 2: expected 'ALARM' · 'The chiller plant efficiency is currently in **ALARM** status over the last 6 hours. This '
✓ id 3: expected '0.703' · 'The average plant kW/RT over the last 24 hours is **0.703** (status: OK). This indicates t'
✓ id 4: expected '629.2' · 'The average cooling load over the last 24 hours was **629.2 RT** (refrigeration tons). Thi'
◆ no-judge quality check (the reference string appears in the answer): 4/4. It is cheap and exact for numeric KPIs. For free-text answers use the LLM judges (--judge).
```

The research tutorial's pandas one-liner works on 1.9.0. Filter to `LLM_END` rows first, though. The CSV also has `LLM_START` rows with zero tokens, and they halve the mean:

```python
import pandas as pd
df = pd.read_csv("week26/06_benchmarking/.runs/eval/alto_ops/standardized_data_all.csv")
ends = df[df.event_type == "LLM_END"]
print(ends.groupby("llm_name")[["prompt_tokens", "completion_tokens"]].describe())
```

**Quality.** For numeric KPIs, an exact check is cheap and honest: does the reference string appear in the answer? That gave 4/4 above. For free-text answers you need a judge. `lab06_2 --judge` scores one row with the research tutorial's `ragas` AnswerAccuracy and `trajectory` evaluators, using the laptop model as its own judge:

**Expected output** (captured on this Mac, `--judge`)

```
=== EVALUATION SUMMARY ===
Workflow Status: COMPLETED (workflow_output.json)
Total Runtime: 15.99s
Workflow Runtime (p95): 15.99s
LLM Latency (p95): 8.34s

Per evaluator results:
| Evaluator   |   Avg Score | Output File            |
|-------------|-------------|------------------------|
| trajectory  |         1   | trajectory_output.json |
| accuracy    |         0.5 | accuracy_output.json   |
✓ accuracy_output.json: average_score 0.5
✓ trajectory_output.json: average_score 1.0
```

The agent answered row 1 exactly (0.901), and the judge still gave it 0.5. That is the research tutorial's warning in practice: a model judging itself works as a smoke test, and it is biased. For a real report, use a stronger judge (Super or Ultra via NVIDIA endpoints, or a second Spark).

✓ Checkpoint: from one eval you can quote p50 and p95 runtime, LLM calls per turn, mean prompt tokens per call, and a quality score, and you can say which file each came from.

## 4 · L5.3 — Sizing: how many Sparks for a hotel portfolio?

`nat sizing calc` answers the quote question: *how many GPUs for N users at a target latency?* It runs your eval once per concurrency value and records p95 LLM latency and p95 workflow runtime. It fits a straight line, `p95 = slope × concurrency + intercept`, and solves it for your target:

- concurrency one test GPU sustains: **c\* = (target − intercept) / slope**
- GPUs needed: **target users / c\* × test GPU count**

Treat one Spark as one GPU. On the Spark, with the research tutorial's values (40 GM and engineer users, 15 s p95):

```bash
# on: spark
export CONFIG_FILE=eval_config.yml CALC_OUTPUT_DIR=./.tmp/sizing/alto_ops
nat sizing calc --config_file $CONFIG_FILE --calc_output_dir $CALC_OUTPUT_DIR \
  --concurrencies 1,2,3,4,6,8,12,16,24,32 --num_passes 2 \
  --test_gpu_count 1 --target_workflow_runtime 15 --target_users 40
# later, re-fit without re-running:
nat sizing calc --offline_mode --calc_output_dir $CALC_OUTPUT_DIR --test_gpu_count 1 --target_workflow_runtime 10 --target_users 100
```

Three rules from the sizing-calculator docs:

| Rule | Why |
|---|---|
| Use **ten or more** concurrency values | the linear fit needs enough points to be robust |
| Keep the calculator's output dir **separate** from the eval output dir | it writes per-concurrency jobs, and `--offline_mode` re-reads that dir |
| The GPU estimate is **rough — not for production** | it assumes linear scaling; use it for a first quote |

On the laptop, lab 06-3 runs the calculator for real and tiny: concurrency 1 and 2, one pass (3 agent runs), with its own `configs/sizing_config.yml` so the eval output stays separate. It then redoes the estimate by hand and re-fits offline.

```bash
# on: laptop
.venv/bin/python week26/06_benchmarking/labs/lab06_3_sizing.py
```

**Expected output** (captured on this Mac)

```
Targets: LLM Latency ≤ 0.0s, Workflow Runtime ≤ 60.0s, Users = 40
Test parameters: GPUs = 1
Per concurrency results:
|   Concurrency |   p95 LLM Latency |   p95 WF Runtime |   Total Runtime |   GPUs (WF Runtime, Rough) |
|---------------|-------------------|------------------|-----------------|----------------------------|
|             1 |           6.33829 |          10.8256 |         10.8256 |                    7.21707 |
|             2 |          16.1058  |          26.8455 |         27.1506 |                    8.94851 |

=== GPU ESTIMATES ===
Estimated GPU count (Workflow Runtime): 9.8
✓ wrote week26/06_benchmarking/.runs/sizing/alto_ops/online/job_1790740665/ (calc_runner_output.json + two PNG plots)
│ concurrency  measured p95 runtime  line   
│ ───────────  ────────────────────  ───────
│ 1            10.83 s               10.83 s
│ 2            26.85 s               26.85 s
│                                      by hand  nat sizing calc
│ ───────────────────────────────────  ───────  ───────────────
│ slope (s per extra concurrent user)  16.020   16.020         
│ intercept (s)                        -5.194   -5.194         
│ R²                                   —        1.000          
◆ concurrency one GPU sustains at ≤ 60 s: (60 − -5.19) / 16.020 = 4.07
◆ GPUs for 40 users: 40 / 4.07 × 1 test GPU = 9.83  ·  nat sizing calc: 9.83
✓ the calculator's number is this line, nothing more
⚠ R² = 1.000 from 2 points means nothing: two points always make a perfect line. The sizing docs recommend ten or more concurrency values for a robust fit.
```

The by-hand line matches the calculator exactly, so there is no magic in it. Look at R² = 1.000, though. Two points always make a perfect line, so that "perfect fit" is the lesson: it proves nothing. With `--num_passes 1`, NAT 1.9.0 trims the dataset to concurrency × passes rows (1 row at c=1, 2 at c=2), which is how the laptop run stays small.

✓ Checkpoint: given a slope, an intercept, a target and a user count, you can compute the GPU estimate by hand, and you can say why the laptop's R² = 1 means nothing.

## 5 · L5.4 — The sandbox tax

OpenShell is not free. Every model call from inside the sandbox goes to `https://inference.local`. The supervisor intercepts it, the policy engine checks it, and it crosses an extra hop over the veth pair before the gateway forwards it. To put a number on that:

1. Run the **identical** `nat eval` (a) on the Spark host against `http://localhost:8000/v1` and (b) inside the OpenShell sandbox from Lab 3.8 against `https://inference.local/v1`.
2. Run both legs at `max_concurrency` 1 and 4.
3. Compare p95 workflow runtime from `inference_optimization.json`.
4. Publish the delta with `openshell --version`.

No source gives an official overhead figure. Your measurement **is** the reference, so it needs its noise and its versions next to it.

```bash
# on: spark
openshell --version
cd ~/alto_ops && for c in 1 4; do nat eval --config_file eval_config.yml \
  --override eval.general.max_concurrency $c --override eval.general.output_dir ./.tmp/tax/host_c$c/; done
openshell sandbox upload alto-ops ./eval_config.sandbox.yml /sandbox/eval_config.yml
openshell sandbox exec -n alto-ops --workdir /sandbox -- nat eval --config_file /sandbox/eval_config.yml \
  --override eval.general.max_concurrency 1 --override eval.general.output_dir /sandbox/.tmp/tax/sandbox_c1/
openshell sandbox download alto-ops /sandbox/.tmp/tax/sandbox_c1 ./.tmp/tax/sandbox_c1
```

`configs/eval_config.sandbox.yml` differs from the host config only in the path to the model (`base_url: https://inference.local/v1`) and the `/sandbox` file paths. Uploading into a sandbox changes it, so the lab runs that line through `change()`: it runs only with 🔓 Allow changes. Lab 06-4 also parse-checks all three `openshell sandbox` commands with the laptop CLI, which reaches no gateway.

**On the laptop, a stand-in for the second path.** There is no sandbox on a Mac. Lab 06-4 runs the same 2-question eval twice: (A) straight to Ollama, and (B) through a small Python reverse proxy with an allow-list (`benchkit.HopProxy`). The proxy denies anything except `POST /v1/chat/completions` and `GET /v1/models`, and it times itself. It is **not** OpenShell: there is no TLS interception, no Landlock and no network namespace. It gives the method a second path to measure.

```bash
# on: laptop
.venv/bin/python week26/06_benchmarking/labs/lab06_4_sandbox_tax.py
```

**Expected output** (captured on this Mac, steps 2–3)

```
◆ Workflow Runtime (p95): 13.03s
◆ hop proxy on 127.0.0.1:8090 → http://localhost:11434/v1 (allow: POST /v1/chat/completions, GET /v1/models; deny the rest)
$ nat eval --config_file week26/06_benchmarking/configs/eval_config.yml --override eval.general.max_concurrency 1 --override eval.general.output_dir week26/06_benchmarking/.runs/tax/hop_c1/ --override eval.general.dataset.file_path week26/06_benchmarking/.runs/tax_rows.jsonl --override llms.local_llm.base_url http://127.0.0.1:8090/v1   [this laptop]
◆ Workflow Runtime (p95): 23.77s
✓ GET /api/pull through the hop → 403 (denied, like an unlisted endpoint)
■ stopped hop proxy on :8090
◆ the hop itself: 4 allowed · 1 denied · its own time per LLM call ≈ 4.12 ms (max 12.23 ms), next to ≈ 9.3 s upstream

▣ STEP 3 · the delta calculator over the two result directories
│ path                                  p50 s  p95 s  mean s  95% CI of mean  n
│ ────────────────────────────────────  ─────  ─────  ──────  ──────────────  ─
│ A · direct (host stand-in)            11.69  13.03  11.69   9.62–13.76      2
│ B · via hop proxy (sandbox stand-in)  19.41  23.77  19.41   12.70–26.13     2
◆ Δ p95 = +10.74 s (+82.5 %)
⚠ n = 2 per leg: too few rows for a confidence interval to mean much. Treat the verdict below as a demonstration of the check, not as evidence.
⚠ the two confidence intervals overlap: this delta is WITHIN NOISE. Do not publish it as a tax. Add rows and repetitions (`nat eval --reps`) until the intervals separate, or report 'not measurable at n = …'.
◆ The one trustworthy laptop number is the hop's own time: ≈ 4.1 ms per LLM call, measured inside the proxy. Run-to-run noise on this shared Ollama is seconds. How big the tax is on a Spark, no source says, so measure it with enough rows and repetitions to see it above the noise.
```

Here is how the delta calculator decides. If path B came out *faster*, that is drift (other load, a warm cache), not a tax. If the 95% confidence intervals of the two means overlap, the delta is within noise: say "not measurable at n = …" and do not publish a number. Only when the intervals separate is the delta real. The only laptop number you can trust here is the proxy's own time, a few milliseconds per call, next to seconds of run-to-run noise. That is why a real sandbox-tax measurement needs many rows and repetitions (`nat eval --reps`).

✓ Checkpoint: you can list the five things to publish with a sandbox-tax number (OpenShell version, NAT version and model, rows/reps/concurrency, both p95s and Δ, both confidence intervals).

## 6 · L5.5 — Harness-level benchmark, and the four-layer report

OpenClaw and Hermes claws have no NAT profiler, so you time them through the API they expose. For Hermes that is the OpenAI-compatible API on port 8642. Send 20 identical tasks and take p50/p95:

```bash
# on: spark
for i in $(seq 1 20); do
  /usr/bin/time -f "%e" curl -s -X POST http://localhost:8642/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"hermes","messages":[{"role":"user","content":"Summarise /sandbox/data/chiller_plant.csv in 3 bullets"}]}' >/dev/null
done 2>&1 | sort -n | awk '{a[NR]=$1} END {print "p50",a[int(NR*0.5)],"p95",a[int(NR*0.95)]}'
```

Pair each run with two more counts. The `inspect_for_inference` count in `openshell term` gives LLM calls per task. The Phoenix or Langfuse spans (Module 05) give tokens per task.

On the laptop, lab 06-5 runs the same loop for real against a `nat serve` of Alto Ops Claw (`/v1/chat/completions`, port `free_port(8001)`): 5 identical tasks, one at a time. It computes p50/p95 two ways, then builds the four-layer report from the summaries labs 06-1 to 06-4 saved in `.runs/`.

> ⚠ **NAT 1.9.0: `nat serve` needs `greenlet`, and the NAT extras do not pull it in.** The FastAPI front end imports `sqlalchemy.ext.asyncio` at start-up, and that import refuses to load without greenlet. When this course was built, `week26/.venv-nat` did not have it and `nat serve` exited at once. The venv now includes greenlet. If yours does not, lab 06-5 notices, puts a clearly labelled stub on `PYTHONPATH` (`.runs/pyshim/greenlet.py`) and says so. That is safe because the async job store that would use greenlet only starts when Dask is installed. On the Spark, install greenlet into the NAT environment.

```bash
# on: laptop
.venv/bin/python week26/06_benchmarking/labs/lab06_5_harness_bench.py
```

**Expected output** (captured on this Mac)

```
✓ ready in 2.8s → http://127.0.0.1:8001/docs
→ POST http://127.0.0.1:8001/v1/chat/completions × 5 · one at a time · the same task every time
◆ task 1: 17.80 s · '- Over the last 6 hours, the chiller plant averaged **473.6 kW** of power consum'
◆ task 2: 34.58 s · '- Over the last 6 hours, the chiller plant averaged **473.6 kW** of power consum'
◆ task 3: 34.20 s · '- Over the last 6 hours, the chiller plant averaged **473.6 kW** of power consum'
◆ task 4: 19.71 s · '- Over the last 6 hours, the chiller plant averaged **473.6 kW** of power consum'
◆ task 5: 20.41 s · '- Over the last 6 hours, the chiller plant averaged **473.6 kW** of power consum'
■ stopped nat (pid 13682)

▣ STEP 3 · p50 / p95 — interpolated vs the tutorial's nearest-rank awk line
│ method                              p50 s  p95 s
│ ──────────────────────────────────  ─────  ─────
│ interpolated (benchkit.percentile)  20.41  34.50
│ sort | awk a[int(NR*p)]             19.71  34.20
◆ With n = 5, awk's p95 is just the 4th-fastest run: one slow outlier moves it a lot. The tutorial sends 20 tasks for a reason; on the Spark, send 20 or more.
◆ 1 distinct answer text(s) for 5 identical tasks at temperature 0. Check correctness as well as speed.

▣ STEP 4 · the four layers side by side (all LAPTOP STAND-IN, from .runs/summary_*.json)
│ layer          LAPTOP STAND-IN result                                from      measured        
│ ─────────────  ────────────────────────────────────────────────────  ────────  ────────────────
│ 1 engine       gemma3:4b: 66 tok/s single · 14 aggregate at c=4      lab 06-1  2026-09-30 10:55
│ 2 workflow     p50 36.7 · p95 39.8 s · 2.0 LLM calls/turn            lab 06-2  2026-09-30 10:57
│ 3 quality      4/4 correct (reference string in answer)              lab 06-2  2026-09-30 10:57
│ 4 sandbox tax  Δ p95 +10.7 s (within noise) · hop itself 4.1 ms/ca…  lab 06-4  2026-09-30 10:59
│ harness (API)  p50 20.4 s · p95 34.5 s over 5 tasks                  lab 06-5  2026-09-30 11:01
✓ the report has one line per layer, each with where it came from and when
⚠ LAPTOP STAND-IN: every number above is this Mac, with nemotron-3-nano / gemma3 on a shared Ollama. None of them is comparable with a Spark figure. On the Spark, rerun labs 06-1 to 06-5 live and this table fills with yours.
```

Look at the two percentile rows. The research tutorial's `awk a[int(NR*p)]` is a nearest-rank percentile. With 5 runs, its "p95" is just the 4th-fastest run. With 20 runs it is the 19th. Send 20 or more tasks on the Spark, and say which percentile method you used.

✓ Checkpoint: your report has one line per layer, each with its source lab and the time it was measured, and none of the laptop lines sits next to a Spark figure.

## Labs — run them here

**labs/lab06_1_engine_sweep.py** — The bandwidth ceiling, vLLM's benchmark on the Spark, and a real laptop stand-in sweep at concurrency 1, 2 and 4.

**labs/lab06_2_nat_eval_profiler.py** — A real `nat eval` with the profiler on Alto Ops Claw: the dataset from the CSV, the key check, the files 1.9.0 writes, percentiles, tokens and a quality score (`--judge` adds the LLM judges).

**labs/lab06_3_sizing.py** — A real, tiny `nat sizing calc`, the GPU estimate redone by hand, and an offline re-fit.

**labs/lab06_4_sandbox_tax.py** — The sandbox-tax method on the Spark, and a laptop stand-in with two paths to the same model and a within-noise verdict.

**labs/lab06_5_harness_bench.py** — Identical tasks through a claw's API (Hermes on the Spark, `nat serve` on the laptop), p50/p95 two ways, and the four-layer report.

## Try it yourself

`exercises/ex06_bench_math.py` has four TODOs. The checker is offline:

1. Write `percentile(xs, p)` with linear interpolation.
2. Turn ai-muninn's 786 tok/s aggregate at c=16 into a per-session speed (research tutorial Part 5, exercise 4).
3. Write the least-squares `linear_fit(xs, ys)` the sizing calculator uses.
4. Write `gpus_needed(target, users, slope, intercept)`, and make it refuse a target below the intercept.

```bash
# on: laptop
.venv/bin/python week26/06_benchmarking/exercises/ex06_bench_math.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ percentile: p50 of 1..10 = 5.5 · p95 = 9.55 · one value is its own p95
✓ per session: 786 tok/s ÷ 16 sessions ≈ 49 tok/s each (bandwidth is shared)
✓ linear fit: p95 ≈ 1.00 s × concurrency + 12.00 s on the practice points
✓ sizing: ≤ 15 s → c* = 3 per Spark → 40 users need ≈ 13.3 Sparks · a target below the intercept raises
```

Then think through two of the research tutorial's Part 5 exercises:

- Your Spark shows 64 tok/s single-stream on Nemotron 3 Nano with Ollama, but the claw feels slow. Name three non-engine causes, and the tool that exposes each.
- The profiler reports a `concurrency_spike_analysis` spike of 9 at t=42 s with `spike_threshold: 7`. What does it mean, and what would you change?

<details><summary>Hint — why per-session speed drops</summary>

Aggregate throughput is capped by memory bandwidth. Batching lets the engine serve more streams per weight read, so the **total** rises, but each stream gets a smaller share: 786 ÷ 16 ≈ 49 tok/s per session. That is still above Exxact's "40+ feels responsive".

</details>

<details><summary>Hint — three non-engine causes</summary>

(a) Too many LLM round-trips per turn (ReAct loops, retries): look at LLM calls per turn and `workflow_profiling_report.txt`. (b) Long prompts: look at `prompt_tokens` in `standardized_data_all.csv`, and turn on `prompt_caching_prefixes`. (c) Slow tools, or policy denials followed by retries: look at tool spans in Phoenix and `deny` lines in `openshell logs`.

</details>

<details><summary>Hint — the concurrency spike</summary>

Nine NAT functions were running at the same moment, above the threshold of 7, and the profiler lists which ones. Either the agent fanned out too many tool calls, or the eval's `max_concurrency` is too high for one Spark. Lower the concurrency, or gate parallel tool execution.

</details>

✓ Checkpoint: all four checker lines are ✓, and you can answer both discussion questions in a sentence each.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `nat serve` exits: "The SQLAlchemy asyncio module requires that the Python 'greenlet' library is installed" | NAT 1.9.0's FastAPI front end imports `sqlalchemy.ext.asyncio`, and the NAT environment has no greenlet | lab 06-5 adds a labelled stub on `PYTHONPATH` by itself. On the Spark, install greenlet into the NAT environment |
| `nat eval --dataset x.jsonl` fails with "If using all scalar values, you must pass an index" | `--dataset` reads a JSON file, not JSONL | point `eval.general.dataset.file_path` at the JSONL, or `--override eval.general.dataset.file_path x.jsonl` (lab 06-4 does this) |
| a profiler option seems to do nothing | a misspelt key: `nat validate` ignores unknown profiler fields | check the key against `ProfilerConfig.model_fields` (lab 06-2 step 2) |
| the aggregate tok/s stays flat as concurrency rises | Ollama is queueing (`OLLAMA_NUM_PARALLEL`), or other labs share it | expected on a laptop. On the Spark, compare vLLM, which batches |
| `nat sizing calc` stops with "must be greater than the intercept" | the target runtime is below the line's intercept, or noise gave a negative slope | raise the target, add concurrencies and passes, and re-fit with `--offline_mode` |
| the ragas judge scores a correct answer 0.5 | the same small model judges itself | use a stronger judge for real reports; use an exact check for numeric KPIs |
| laptop numbers change a lot between runs | a shared Ollama: other labs run at the same time | that is why every laptop number is labelled LAPTOP STAND-IN. Compare only runs from the same minute, or use the Spark |
| port 8001 or 8090 is busy | another lab's server is running | the labs use `free_port()` and print the port they got |

## Next

[Lab 07 — Expert: threat model, hardening, custom blueprints, remote gateways](../07_hardening/TUTORIAL.md): the threat model for a claw, a production policy, custom blueprints, and remote gateways. The five-line benchmark report you built here becomes the baseline you re-measure after every hardening change.
