# ▶ Jev Lab 09 — Evaluate before you automate (and know what it costs)

> Part of Week 24 · Typed AI decisions with Jev. A well-formed answer is not a correct answer. This lab turns "it looks right" into numbers you can defend: accuracy, coverage, wrong-accepted, latency and cost.

**What you'll actually do**
- Run a 12-row labeled smoke evaluation (English + Thai) against the intent router from Lab 04.
- Read the four numbers that decide whether a router may act alone — and the one that matters most.
- Sweep thresholds over the **same** answers, with zero new API calls.
- Price 1 → 1,000,000 requests from tokens you actually measured, and check the rate limits.
- Build your own labeled test set and implement macro-F1.

**Time** ~40 min · **Difficulty** intermediate · **Cost** ≈ $0.001 live · $0 dry

## 0 · Why evaluate at all?

Jev always returns valid JSON with probabilities that sum to 1. That guarantees the **interface**, not the **truth**. The only way to know whether a router is good enough to act on its own is to run it on examples where *you already know the right answer* and count.

Four words you will use all lab:

| Word | Meaning | Why you care |
|---|---|---|
| **gold label** | the answer a human decided is correct | what you grade against |
| **coverage** | share of ALL rows the router handled automatically | how much work you save |
| **accepted accuracy** | share of the auto-handled rows that were right | how safe the automation is |
| **wrong accepted** | rows handled automatically **and** wrong | the damage — drive this to 0 |

A router that sends everything to a human has 0% coverage and 0 wrong-accepted: safe and useless. A router that auto-handles everything has 100% coverage and every mistake goes through. The job is choosing the point in between.

✓ Checkpoint: you can explain why "coverage" and "accepted accuracy" pull in opposite directions.

## 1 · Run the smoke evaluation (lab 01)

The data is `week24/09_evaluate_and_cost/data/intent_eval.jsonl` — the 9 teaching rows from `jev_lab.py` plus 3 more Thai rows. Each row is one line of JSON:

```json
{"id": "t3", "language": "th", "state": {"message": "นัดช่างมาบำรุงรักษาเครื่องทำน้ำเย็นสัปดาห์หน้า"}, "expected": {"intent": "facility_ops"}}
```

The `expected` field never reaches Jev: `jev_lab.prepare()` copies only `message` into the state. Leaking the answer key into the input is the most common evaluation bug there is.

```bash
.venv/bin/python week24/09_evaluate_and_cost/labs/lab01_smoke_eval.py
```

**Expected output**

```
▣ STEP 2 · ask Jev — 12 calls, one per row
│ id  lang  gold            predicted       top p  match  gate
│ i1  en    hvac            hvac            1.00   ✓      auto
│ i3  en    hvac            energy_mv       0.86   ✕      auto
│ i9  en    unknown         unknown         0.97   ✓      review
│ t3  th    facility_ops    hvac            0.78   ✕      review
│ …
▣ STEP 3 · score it — reference gate: confidence ≥ .75 AND top ≥ .80 AND margin ≥ .20
◆ raw accuracy (successes only) : 83%
◆ coverage (auto-routed / all)  : 83%  (10 of 12)
◆ accepted accuracy             : 90%
◆ wrong accepted                : 1   ← routed automatically AND wrong
◆ accuracy [en]                 : 7/8
◆ accuracy [th]                 : 3/4
▣ STEP 5 · latency and cost (successful calls)
◆ p50 460 ms · p95 559 ms   (nearest-rank, n=12)
◆ 10354 input tokens total · avg 863/call · $0.000435 for the whole eval
```

Two rows are wrong, and they teach different lessons:

- **t3** (Thai: "book a technician to service the chiller next week") → Jev chose `hvac` with p ≈ 0.78–0.80 — we have seen both values on different live runs of the *same* request. At 0.78 the gate **catches** it (below the 0.80 bar → review); at 0.80 it **slips through** and becomes a second "wrong accepted". That is the real lesson: a threshold sitting right on the edge of your data makes the outcome depend on tiny run-to-run jitter. Pick thresholds with margin (Lab 09-2 shows 0.90 removes both wrong-accepted rows), and re-check them on every new model version.
- **i3** ("compare an HVAC diagnosis with an M&V assessment and resolve the disagreement") → Jev chose `energy_mv` at 0.86 and it was **accepted**. But is Jev wrong? The request is half HVAC, half M&V, and in the same call Jev said `needs_mas = 0.98` ("this needs several experts"). The *label* is debatable. Deciding which is right is called **adjudication**, and you must do it before you trust any accuracy number.

> Your numbers may differ slightly from the sample above: live answers jitter by a few hundredths between runs. `t3` is the row most likely to flip between `review` and `auto`.

✓ Checkpoint: you ran lab 01 and can say which row is the "wrong accepted" one — and argue whether the model or the label is at fault.

## 2 · Feel the ambiguity yourself

Here is row i3 with only the topic question. Press **⚡ Ask Jev**, then try the edits below to watch the probability move between `hvac` and `energy_mv`.

```jev
{
  "state": {"message": "Compare an HVAC engineer's diagnosis with an M&V analyst's savings assessment and resolve their disagreement."},
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Classify the primary topic of `message`.",
      "criteria": {
        "hvac": "HVAC, chiller, AHU, VRF, cooling performance or diagnosis.",
        "energy_mv": "Energy baselines, savings measurement or verification.",
        "facility_ops": "Maintenance workflow, site operations, work orders.",
        "unknown": "Too vague, unsupported topic, or no identifiable request."
      }
    },
    "needs_several_experts": {
      "type": "noul",
      "instructions": "Does `message` explicitly require combining distinct expert analyses or resolving their disagreement?"
    }
  }
}
```

1. Delete `"and resolve their disagreement"`. Does the topic get clearer?
2. Change the message to `"Why is the HVAC diagnosis disagreeing with the M&V baseline?"`.
3. Add a `"mixed_specialists"` option to `criteria`. Now the label set itself admits the ambiguity — often the honest fix.

✓ Checkpoint: you found at least one edit that moves `intent` by more than 0.2.

## 3 · Sweep the threshold — for free (lab 02)

A threshold lives in **your** code, not in the model. So once you have the answers you can try every threshold without calling Jev again. Lab 02 reads the answers Lab 01 saved in `.runs/intent_answers.json`.

```bash
.venv/bin/python week24/09_evaluate_and_cost/labs/lab02_threshold_sweep.py
```

**Expected output**

```
✓ 12 saved answers · 0 new API calls · $0
│ threshold  coverage                accepted acc  wrong accepted
│ 0.50       92%       ███████████░  82%           2
│ 0.70       92%       ███████████░  82%           2
│ 0.80       83%       ██████████░░  90%           1
│ 0.90       75%       █████████░░░  100%          0
│ 0.95       75%       █████████░░░  100%          0
│ 0.99       75%       █████████░░░  100%          0
│ ref gate   83%       ██████████░░  90%           1
│ t3  top p 0.78  pred hvac       gold facility_ops  ✕  → auto only while threshold ≤ 0.78
│ i3  top p 0.86  pred energy_mv  gold hvac          ✕  → auto only while threshold ≤ 0.86
```

At 0.90 both mistakes go to review and wrong-accepted drops to 0, at the cost of one extra row for a human. **But:** we chose 0.90 *by looking at these 12 rows*. Report it on these same rows and you are grading your own homework. The honest workflow is three splits:

| Split | Used for |
|---|---|
| development | writing and fixing questions |
| calibration | choosing thresholds — then **freeze** them |
| test (held out) | the number you report; touched once |

A mistake made at top p = 1.00 can't be caught by any threshold. Only better questions, better labels, or a second check can catch it.

✓ Checkpoint: you can name the threshold that gives 0 wrong-accepted on this set, and explain why you still should not ship it based on this set.

## 4 · The reference evaluator

`week24/jev_lab/jev_lab.py` (the original tutorial script) has the same evaluator built in. Run it on the same file to see the numbers agree. It needs the key in your shell; the built-in ⌨ Terminal already has it.

```bash
set -a; source <(grep '^TYPESAFE_API_KEY=' .env); set +a
.venv/bin/python week24/jev_lab/jev_lab.py evaluate intent \
  --input week24/09_evaluate_and_cost/data/intent_eval.jsonl --limit 12 --live
```

**Expected output**

```
{
  "rows": 12,
  "api_or_validation_failures": 0,
  "raw_accuracy_successes_only": 0.8333333333333334,
  "recommendation_coverage_all_rows": 0.8333333333333334,
  "accepted_route_accuracy": 0.9,
  "wrong_accepted_routes": 1,
  "p50_ms_successes_only": 493.27,
  "p95_ms_successes_only": 744.47,
  "reported_cost_usd_successes_only": null,
  "note": "Toy dataset, not a production benchmark. Failures count against coverage. …"
}
```

`reported_cost_usd` is `null` because the TypeSafe direct API returns token counts but not a dollar figure (OpenRouter returns `usage.cost`). Missing cost is **unknown**, not zero — compute it from tokens, as Lab 03 does.

✓ Checkpoint: your reference run matches lab 01's accuracy, coverage and wrong-accepted.

## 5 · What does it cost? (lab 03)

Jev bills **input tokens only**: **$0.042 per million**. Output is free. Lab 03 is pure arithmetic, so it needs no key.

```bash
.venv/bin/python week24/09_evaluate_and_cost/labs/lab03_cost_calculator.py
```

**Expected output**

```
│ requests   input tokens @1,000 each  classifier cost
│ 1          1,000                     $0.000042
│ 10,000     10,000,000                $0.42
│ 100,000    100,000,000               $4.20
│ 1,000,000  1,000,000,000             $42.00
◆ measured: 12 intent calls, avg 863 input tokens (7 questions + guard text + state)
│ 1,000,000  862,833,333   $36.24
│ one batched call  860               $36.1 per million msgs
│ 7 separate calls  2,240             $94.1 per million msgs
◆ batching saves 62% here — and one round-trip instead of seven.
✓    10,000 req/day → peak ≈ 35 req/min, 499 tok/s fits
⚠   500,000 req/day → peak ≈ 1,736 req/min, 24,966 tok/s → queue, batch, or ask for a higher limit
```

Three takeaways:

1. **Your questions cost more than your messages.** Here a short message is ~30 tokens, and the 7 questions with their criteria are most of the other ~830. Long rubrics are paid for on every call.
2. **Batch questions that share a state.** The state and framing are paid once per call.
3. **The request-rate limit (1,200/min) bites before the token limit** for short messages. A bursty 500k/day workload needs a queue.

A million classifications for ~$36 is cheap. That's exactly why the classifier is rarely what drives cost: the generative model, retries, engineering and human review cost far more. Price the whole workflow, not one API.

✓ Checkpoint: you can estimate the monthly Jev bill for 20,000 inbox emails/day from your own measured tokens per call.

## Labs — run them here

**labs/lab01_smoke_eval.py** — 12 labeled EN/TH rows → accuracy, coverage, wrong-accepted, confusion matrix, latency, cost.

**labs/lab02_threshold_sweep.py** — Re-threshold the saved answers from 0.5 to 0.99 with zero new calls.

**labs/lab03_cost_calculator.py** — Pure arithmetic: price tables, batching savings, rate-limit headroom.

Run lab 01 first; labs 02 and 03 reuse its saved answers (`.runs/intent_answers.json`).

## Try it yourself

**Exercise 09 — your own test set, your own metric.** Open `week24/09_evaluate_and_cost/exercises/ex09_build_a_testset.py`:

1. **TODO 1** — write at least 6 labeled rows for the Alto Copilot router: at least 2 in Thai and at least 1 whose correct answer is `unknown`. Write requests *you* would actually receive.
2. **TODO 2** — implement `macro_f1(pairs)`: the average F1 across classes. The comments spell out TP, FP, FN.

The checker validates your rows and tests `macro_f1()` against four fixtures **before** any API call. When everything is ✓, it evaluates your rows live.

```bash
.venv/bin/python week24/09_evaluate_and_cost/exercises/ex09_build_a_testset.py
```

**Expected output**

```
✓ 7 rows
✓ 2 Thai rows
✓ includes an 'unknown' row
✓ macro_f1 fixture → 0.667
▣ STEP 2 · ask Jev about your 7 rows
│ my-5  en     facility_ops    facility_ops    0.63   ✓      review
│ my-7  en     unknown         unknown         0.90   ✓      review
◆ raw accuracy 100% · macro-F1 1.00 · coverage 71% · wrong accepted 0
```

<details><summary>Hint — macro_f1 in five lines</summary>

Collect `classes = {g for g, _ in pairs} | {p for _, p in pairs}`. For each class count TP, FP and FN with `sum(...)` over the pairs, compute `2*tp / (2*tp + fp + fn)`, and average. Guard the division when a class has no TP, FP or FN.

</details>

<details><summary>Stretch — make the router fail</summary>

Write 3 rows *designed* to break it: a negation ("don't schedule maintenance, just tell me the chiller COP"), a mixed EN/TH abbreviation ("AHU-2 ช่วยเช็ค SAT หน่อย"), and a follow-up with no context ("and the other building?"). A test set with no hard cases measures nothing.

</details>

✓ Checkpoint: all fixture lines are ✓, and you labeled at least one row you are *not* sure about — and wrote down why.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Lab 02 says `.runs/intent_answers.json not found` | Run lab 01 first. Lab 02 then makes the 12 calls itself once |
| Latency shows `no latency in DRY mode` | Replayed answers are not timed — switch to ⚡ Live to measure |
| Accuracy differs a little from the expected output | Normal: another model version (check `model`) or a borderline row. Look at the rows, not the percentage |
| `expected.intent must be one of …` | A label typo in your data file — labels must match the `criteria` keys exactly |
| `reported_cost_usd: null` | The TypeSafe API returns tokens, not dollars. Multiply `input_tokens` by $0.042 / 1,000,000 |

## Next

Continue to [Lab 10 — Jev vs Laya: hosted API or open weights](../10_jev_vs_laya/TUTORIAL.md). It asks when a self-hosted classifier makes sense, and how to compare two models fairly.
