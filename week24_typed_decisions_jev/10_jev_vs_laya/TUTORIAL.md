# ▶ Jev Lab 10 — Jev vs Laya: hosted API or open weights?

> Part of Week 24 · Typed AI decisions with Jev. Laya is an open-weights model with the same three primitives (`choice`, `noul`, `score`). This lab is about *choosing between deployment models* and *comparing two models fairly*. We run the Jev side live; the Laya side is optional.

**What you'll actually do**
- Compare managed inference (Jev) with self-operated inference (Laya), dimension by dimension.
- Learn why two published comparisons seem to disagree, and why they are not the same experiment.
- Verify the shared benchmark harness offline (46 + 32 checks), then plan a run without spending anything.
- Run the Jev half of the shared 28-row benchmark live and read the report like a reviewer.
- Probe negation ("do NOT turn off the chiller") in English and Thai.
- Write a referee that decides whether two benchmark runs are comparable.

**Time** ~35 min · **Difficulty** intermediate · **Cost** ≈ $0.001 live (28 + 8 calls) · $0 dry · Laya optional

## 0 · The one-sentence difference

**Jev** is a *managed API*: you send `state` + `questions` to TypeSafe's servers and pay per input token. **Laya** publishes *model weights* (Apache-2.0) and a Python runtime that you run on your own CPU/GPU. Same question types, different responsibilities.

| Dimension | Jev | Laya |
|---|---|---|
| Delivery | Hosted TypeSafe API (also via OpenRouter) | Downloadable checkpoints you run yourself |
| Primitives | `choice`, `noul`, `score` | the same three, same state/questions pattern |
| Version pinning | pin an API id like `jev-1.13.0` | pin the exact weights + runtime (`laya==0.3.20`) yourself |
| Context | 64k per request; 32k for state + longest question | 512 or 1,024 tokens by default (multilingual up to 8,192 if configured) |
| Custom training | none — you customize through state, instructions, criteria | fine-tuning and temperature fitting documented |
| Language | English-primary; test Thai yourself | multilingual checkpoint claims 100+ languages; test Thai yourself |
| Privacy boundary | selected state goes to the provider | inference can stay on your hardware; logs and later cloud calls still need controls |
| Cost | $0.042 per million input tokens, output free | no per-call bill; you pay for hardware, idle capacity, engineering and maintenance |
| You own | input quality, evaluation, failure handling, action policy | all of that **plus** hardware, dependencies, checkpoint lifecycle, capacity, service security |

Start from the **data boundary and the quality target**, then pick the infrastructure. Don't start from the lowest advertised milliseconds.

✓ Checkpoint: you can name one AltoTech workload where the privacy boundary argues for local inference, and one where Jev's larger context argues for the hosted API.

## 1 · Laya is three checkpoints, not one model

These specifications are **reported by the Laya project**, not measured by us:

| Checkpoint | Backbone | Default length | Worth testing for |
|---|---|---|---|
| `english` | ModernBERT-large, 421M | 512 tokens | short English labels, inbox triage |
| `multilingual` | mmBERT-base, 322M | 1,024 (up to 8,192 if set) | Thai/English requests |
| `typed-decisions` | ModernBERT-large, 421M | 1,024 tokens | invoice, support, security, agent-trace tasks like its training data |

**The silent-truncation trap.** Laya reserves ~192–256 tokens for the question and options, and whatever is left (~320–768 tokens) is for state. Its ordinary predictor *quietly cuts off* the rest. A confident HR or incident label could come from a model that never saw the relevant paragraph. The shared benchmark runner therefore **refuses** to run a row that would be truncated (`laya_head_would_truncate`) rather than scoring incomplete input.

✓ Checkpoint: you can explain why a truncated-but-confident answer is worse than an error.

## 2 · Why the two published comparisons "disagree"

**In plain words:** two articles seem to crown different winners, but they ran different races. One compared an out-of-the-box Laya against Jev on one benchmark. The other compared a Laya that was specially trained for the task against a Jev number copied from somewhere else, with different prompts and test sizes. It is like comparing a runner's time on a flat track with another runner's time uphill. Neither result tells you which model is better *for AltoTech*. Only a fair test on your own data (the fair-pilot referee you write in this module's exercise) can.


| Source | Reports | What was actually compared |
|---|---|---|
| Hugging Face community blog | Jev 74.4 vs untuned Laya 54.4 (JevBench composite, 534 decisions) | composite scores, not % correct; secondary-source, not reproduced here |
| ZimaSpace article | fine-tuned Laya 0.766 vs Jev 0.727 accuracy | Laya's model card says Jev was **not run** by that project; prompts and sample sizes differ |

More from the Laya typed-decisions card (400 test cases, 2,000 decisions, fine-tuned on a separate 1,200-case split):

| Metric | Fine-tuned Laya | Jev (published reference) | Note |
|---|---:|---:|---|
| Accuracy ↑ | 0.766 | 0.727 | different prompts/sample sizes |
| Soft accuracy ↑ | 0.471 | 0.580 | a different metric from hard accuracy |
| Brier ↓ | 0.062 | 0.148 | probability error |
| ECE ↓ | 0.213 | 0.144 | more accurate ≠ better calibrated |
| Score MAE ↓ | 0.242 | 0.391 | ordinal error |

The *untuned* English Laya checkpoint scored **0.362** on that same test. A specialized result must not be attributed to every checkpoint. These are **different experiments**, not two controlled studies with opposite winners. In the exercise you will write the referee that says so.

✓ Checkpoint: you can list three reasons the 0.766 vs 0.727 figure is not a head-to-head win.

## 3 · Test the harness before trusting its numbers (lab 01)

A benchmark is software, and software has bugs. Before comparing models, prove offline that the harness sends both models **byte-identical input**, counts failures, and refuses truncation.

```bash
.venv/bin/python week24/10_jev_vs_laya/labs/lab01_benchmark_offline_tests.py
```

**Expected output**

```
▣ STEP 1 · jev_lab.py selftest — request/response contracts for all 7 labs
✓ 46 contract checks passed · live inference tested: False
▣ STEP 2 · python -m unittest test_jev_laya_benchmark -v
✓ gold never in state
✓ mocked end to end equal inputs and outputs
✓ preflight state overflow
✓ setup failure prevents jev billing
✓ record failure no secret
…
◆ Ran 32 tests in 0.186s · 32 passed · 0 failed
⚠ NOT tested: real model quality, Thai accuracy, latency. Tests use mocked answers.
```

✓ Checkpoint: 46 contract checks and 32 unit tests pass, with no key and no network.

## 4 · Plan before you spend (lab 02)

`plan` builds the corpus, hashes it, counts the calls and shows one example request. It sends nothing.

```bash
.venv/bin/python week24/10_jev_vs_laya/labs/lab02_benchmark_plan.py
```

**Expected output**

```
◆ 28 rows · by lab {'intent': 11, 'email': 6, 'hr': 2, 'afdd': 2, 'lead': 2, 'rag': 2, 'point': 3}
◆ languages {'en': 23, 'th': 4, 'mixed': 1} · 43 gold labels (partial on purpose — unlabeled answers are never scored)
◆ mode dry_run_no_inference · providers ['jev', 'laya'] · split smoke · rows 28
◆ max logical calls per provider: 28 (each Jev call may retry up to 4 HTTP attempts)
◆ corpus hash 1ddbb974afc82c82… · 7 question schemas hashed
» state     {"message":"Why is AHU-3 not cooling the hotel lobby? Check yesterday's trends."}
◆ providers ['jev'] · 28 logical Jev calls · est. ≈ $0.0011 at ~900 tok/call
```

Notice that `state` is a **JSON string**, not an object: the runner serializes the state once and sends the same string to both models, so neither gets a friendlier format. The corpus and schema hashes let anyone check later that two runs used the same data and questions.

✓ Checkpoint: you can say how many billed calls a 28-row Jev-only run makes, and why the real number of HTTP attempts could be higher.

## 5 · Run the Jev side live (lab 03)

This is the real harness on the real API: 28 fictional rows covering all seven lab schemas, in EN, TH and mixed. It takes about 15 seconds and costs about a tenth of a cent. Results go to a **new** folder every time; the runner refuses to overwrite evidence.

```bash
.venv/bin/python week24/10_jev_vs_laya/labs/lab03_jev_only_benchmark.py
```

The same run by hand, for your own terminal:

```bash
set -a; source <(grep '^TYPESAFE_API_KEY=' .env); set +a
cd week24/jev_lab
../../.venv/bin/python jev_laya_benchmark.py init --dir /tmp/bench_data
../../.venv/bin/python jev_laya_benchmark.py run --input /tmp/bench_data/smoke.jsonl \
  --providers jev --live-jev --limit 28 --out /tmp/bench-run-01
```

**Expected output**

```
◆ 28 calls · 28 ok · 0 failed · p50 442 ms · p95 1304 ms
◆ 20,142 input tokens · ≈ $0.00085
│ task/question            type    labels  accuracy  coverage  acc. accepted  wrong acc.
│ email/category           choice  6       1.00      1.00      1.00           0
│ hr/python_evidence       choice  2       0.50      1.00      0.50           1
│ intent/complexity        score   1       0.00      0.00      n/a            0
│ intent/intent            choice  11      0.91      0.91      0.90           1
│ point/ambiguous          noul    3       1.00      0.33      1.00           0
│ rag/injection            noul    2       1.00      1.00      1.00           0
│ …
⚠ candidate-demo-2 [hr/python_evidence, en] gold 'unclear' → Jev 'not_stated' (top p 0.99)
⚠ i3 [intent/intent, en] gold 'hvac' → Jev 'energy_mv' (top p 0.87)
⚠ b-mixed-code [intent/complexity, mixed] gold 1 → Jev 0 (top p 0.66)
```

Read the three disagreements, not the averages:

- **candidate-demo-2** — the text says "Lists Python as an interest." Gold says `unclear`, Jev says `not_stated` with 0.99. Both readings are defensible, and a 0.99 label on a debatable case is exactly why HR outputs go to a recruiter and are never used for ranking.
- **i3** — the same half-HVAC, half-M&V request that Lab 09 flagged. It is consistent across runs, which is useful: the label needs adjudication.
- **b-mixed-code** — a mixed Thai/English coding request rated "simple" instead of "one specialist". With one label, accuracy for this question is 0%. One example proves nothing either way.

Also note **p95 = 1,304 ms** while p50 = 442 ms. The tail matters for user experience, and it is where network and retries show up. That is one more reason to measure end-to-end latency, not model-only speed.

✓ Checkpoint: you ran the live benchmark and can explain why "hr/python_evidence accuracy 0.50" says almost nothing about Jev's HR quality.

## 6 · Probe negation (lab 04)

Laya's repository documents a negated cancellation request that still selected "cancel", with high confidence. TypeSafe documents that Jev reads instructions *literally*. Either way, you test it yourself before a classifier goes anywhere near equipment.

```bash
.venv/bin/python week24/10_jev_vs_laya/labs/lab04_negation_probe.py
```

**Expected output**

```
│ message                                       shutdown?  action        conf  outage?  human agrees
│ Turn off chiller 2 now.                       0.98       turn_off      0.97  0.04     ✓
│ Do NOT turn off chiller 2.                    0.02       keep_running  0.92  0.03     ✓
│ Don't turn off chiller 2 unless the high-pr…  0.03       conditional   0.83  0.07     ✓
│ There is no outage. Please send pricing for…  0.02       no_action     0.96  0.03     ✓
│ ปิดชิลเลอร์ 2 ตอนนี้เลย                       0.97       turn_off      0.96  0.07     ✓
│ ห้ามปิดชิลเลอร์ 2 เด็ดขาด                     0.02       keep_running  0.98  0.04     ✓
◆ 8/8 messages match the human reading on all three questions
```

jev-1.13.0 passed all eight on 2026-09-27. That is good news, but it's not permission. The pass depends on the wording: the Noul spells out that a *conditional* request is a "no", and the Choice has a separate `conditional` option. Try the weaker wording below and see whether the conditional case still lands where you want.

```jev
{
  "state": {"message": "Don't turn off chiller 2 unless the high-pressure alarm clears."},
  "questions": {
    "requests_shutdown": {
      "type": "noul",
      "instructions": "Does `message` mention turning off the chiller?"
    },
    "action": {
      "type": "choice",
      "instructions": "Which equipment action does `message` request?",
      "criteria": {
        "turn_off": "Turn the equipment off",
        "keep_running": "Keep the equipment running"
      }
    }
  }
}
```

When we ran it, `requests_shutdown` came back **0.99**. Jev was not wrong: the message literally *mentions* turning off the chiller, and that is what the question asked. A policy reading that noul as "the operator wants it off" would be dangerously wrong. The weak Choice still picked `keep_running`, but only because it had no `conditional` option. Fix the instruction back to "ask for the chiller to be turned off now or unconditionally", add `conditional` and `no_action`, and ask again.

✓ Checkpoint: you found a wording that makes the conditional message look like a shutdown request — and fixed it with better instructions or options, not by trusting the score.

## 7 · Where each model fits at AltoTech — and the hybrid rule

These are proposed experiments, not deployed capabilities:

| Workflow | First experiment | Boundary |
|---|---|---|
| Alto Copilot EN/TH router | hosted Jev vs Laya `multilingual` on the same authorized short inputs | uncertain → clarification; never copy thresholds between models |
| HR and internal inbox | if policy forbids external processing, test Laya locally on minimized evidence | a human reviews every proposal |
| Site-gateway incident triage | Laya where offline operation is required | existing deterministic alarms keep running independently |
| Long documents | Jev's larger context where hosted processing is allowed | enforce the real provider limit |
| Sales / shared inbox | Jev when fast delivery and low model-ops burden dominate | price the whole workflow before buying hardware |

A privacy-aware hybrid:

```text
identity + tenant policy resolved in code
  → data minimization + deterministic calculations
  → local Laya proposal
     → accepted only under separately tested policy
     → uncertain:
        → cloud ALLOWED for this data?  → approved minimal context to Jev
        → cloud PROHIBITED?             → local specialist or human review
  → authorization and human approval stay independent
```

**Never send data to Jev just because Laya was unsure.** Whether data may go to the cloud is a policy decision made *before* escalation, not something a model's uncertainty decides.

✓ Checkpoint: you can explain why "local was uncertain, so ask the cloud" is a data-governance bug.

## 8 · Optional — run Laya locally

Laya needs a separate environment, a PyTorch runtime and a model download on first use. That is why it is optional. The demo script is saved at `week24/10_jev_vs_laya/optional/laya_demo.py`, and its default mode is a dry run that imports nothing:

```bash
.venv/bin/python week24/10_jev_vs_laya/optional/laya_demo.py
```

To actually run it, use an isolated folder so the course `.venv` stays clean:

```bash
mkdir -p ~/laya-tutorial && cd ~/laya-tutorial
python3 -m venv .venv && source .venv/bin/activate
python -m pip install "laya==0.3.20" && python -m pip check
cp /Users/altodev/Desktop/agenticaicodingfitness/week24/10_jev_vs_laya/optional/laya_demo.py .
python laya_demo.py --run            # first run downloads the multilingual checkpoint
python laya_demo.py --run --offline  # proves it works from the local cache
```

Suggested human labels for its three samples: duplicate invoice → `finance`, no outage; Thai "hotel AC stopped" → `support`, outage reported; "There is no outage…" → `sales`, no outage. The first call includes the model load, so never compare its `call_ms_including_any_lazy_load` with a warm Jev call. With Laya installed, the shared runner can then compare both: `jev_laya_benchmark.py run --limit 1 --live-jev --live-laya --out run-smoke-01`.

✓ Checkpoint (optional): `laya_demo.py` prints `dry_run_no_inference` in the course venv — or, if you installed Laya, three records marked `human_review: true, execute: false`.

## Labs — run them here

**labs/lab01_benchmark_offline_tests.py** — Run the 46 contract checks and 32 benchmark unit tests offline: no key, no model.

**labs/lab02_benchmark_plan.py** — Init the 28-row corpus in a temp folder and print the dry-run plan: hashes, call caps, the shared request.

**labs/lab03_jev_only_benchmark.py** — The real shared benchmark, Jev side: 28 live calls, per-question metrics, every disagreement.

**labs/lab04_negation_probe.py** — Eight negation and condition probes in English and Thai; nothing is actuated.

Labs 01 and 02 never call an API. Lab 03 replays a recorded real run in DRY mode.

## Try it yourself

**Exercise 10 — be the referee.** Open `week24/10_jev_vs_laya/exercises/ex10_fair_pilot.py`. It is fully offline:

1. **TODO 1** — fill `PILOT_PLAN`: splits, languages, metrics, where thresholds are tuned, whether to start in shadow mode, and the hybrid-fallback rule.
2. **TODO 2** — implement `is_fair_comparison(run_a, run_b)`. It returns one problem string per broken rule (R1–R7 in the file's docstring), and an empty list when the runs are comparable.

```bash
.venv/bin/python week24/10_jev_vs_laya/exercises/ex10_fair_pilot.py
```

**Expected output**

```
✓ no automatic cloud fallback just because the local model was unsure
│ ✓  identical fair setup                              0                  0
│ ✓  different corpora (the two articles!)             1                  1      R1 different corpus
│ ✓  fine-tuned Laya vs zero-shot Jev                  1                  1      R5 fine-tuned vs zero-shot
│ ✓  smoke split + different prompts + failures hidd…  3                  3      R2 not both on the held-out…
```

<details><summary>Hint — structure of the referee</summary>

Seven independent `if` statements, one per rule. Each appends its own message. Don't `return` early: a reviewer wants **every** problem at once, not just the first.

</details>

<details><summary>Stretch — add R8</summary>

Add a rule for latency fairness: when comparing latencies, both runs must record `hardware` and `warmup_calls`, and a cold first Laya call must not be compared with a warm Jev call. Add a fixture that breaks it.

</details>

✓ Checkpoint: all six fixtures are ✓ and your plan says `cloud_fallback_when_local_uncertain: False`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Missing selected Jev provider key` | The runner reads `TYPESAFE_API_KEY` from the environment only. Lab 03 passes it for you; by hand, `source` it first |
| `Output directory exists` | Deliberate — results are evidence. Use a new `--out` folder |
| `Explicit --live-jev / --live-laya required` | The runner never spends money without a flag per provider |
| `ModuleNotFoundError: laya` on `--run` | Expected in the course venv — Laya is optional; use the isolated `~/laya-tutorial` venv |
| `laya_head_would_truncate` | A fairness guard, not a model answer: shorten the rubric for **both** models in a new, versioned experiment |
| Lab 03 numbers differ from the expected output | Normal run-to-run and version drift on 1–11 labels per question — read the disagreement list |

## Next

Continue to [Lab 11 — Jev + your choice of LLM](../11_jev_plus_llm/TUTORIAL.md): Jev decides, Claude / ChatGPT / Gemini / DeepSeek / Kimi / GLM writes. Then the capstone: [Lab 12](../12_capstone_hotel_copilot/TUTORIAL.md). You'll put all of it together: batched questions, code-side extraction, an uncertainty gate, a safety override and shadow-mode comparison.
