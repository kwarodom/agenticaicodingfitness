---
name: agent-evaluation
description: "Teach how to measure and GATE agent quality so 'it felt right' stops being your test — golden datasets, eval frameworks (RAGAS/DeepEval/Braintrust/Inspect AI), LLM-as-judge with bias mitigation, and a 5-gate CI/CD pipeline for non-deterministic systems. Use when someone asks 'how do I know if my agent is good / got worse?', mentions evals/golden sets/LLM-as-judge/regression testing/CI for prompts, ships an agent they can't score, or is reviewing Weeks 10 & 15."
when_to_use: "Learner wants to score an agent's quality, build a golden test set, run LLM-as-judge, catch regressions on prompt changes, or gate merges in CI — or is catching up on Weeks 10 & 15."
---

# Agent Evaluation — You Can't Ship What You Can't Score (Weeks 10 & 15)

> **The one idea:** "It felt right" is not a test. An agent's output changes from run to run, so you need *numbers* — golden datasets, automatic scorers, and a CI gate that blocks the merge when quality drops. Eyeballing three answers is how 40% of agent projects get cancelled.

```
real questions + expected answers → run the agent → score every answer → gate the merge on the mean
```

This is the general craft of measuring agents. The RAG-specific version lives in `knowledge-graph-mastery` and `rag-knowledge-agents`; this skill is the part that applies to *any* agent.

---

## Part A — Golden datasets (your ground truth)

A **golden dataset** is a versioned list of real `(input, expected_output)` pairs. It's the ruler you measure every change against. Without it, "did my prompt tweak help or hurt?" has no answer.

| Rule | Why |
|---|---|
| **25–100 real pairs** to start | Enough signal to catch regressions; small enough to maintain. Add the questions users *actually* ask, plus edge cases. |
| **Label the expected answer** | Each row needs a `ground_truth` — the known-correct answer, written/checked by a human. |
| **Version it in Git** | Commit the CSV next to the prompts. Then every score is reproducible against a specific dataset version. |
| **Grow from incidents** | The most valuable rows are the ones that caused a production bug. Add them so they can never regress again. |
| **Size up later** | For ~80% expected pass rate at ±5% margin, you need ~250 samples *per scenario*. Start small, grow with the system. |

You can **bootstrap** the set instead of hand-writing every row: RAGAS reads your source docs and synthesizes a diverse mix of question types (simple lookup / reasoning / multi-context), each with a reference answer and the contexts it came from. Then you spot-check and delete the junk.

> 📁 Class repo: `week15_production_graphrag/kg_mastery/part4_evaluation/02_testset_generation.py` — auto-generates a test set from `hotel_docs/*.txt` with a 40/40/20 simple/reasoning/multi-context mix.
> 📁 Class repo: `week15_production_graphrag/kg_mastery/part4_evaluation/hotel_kg_testset.csv` — the resulting golden set: columns `user_input, reference_contexts, reference, persona_name, query_style, ...`. Open it to see what a real row looks like.

The hand-built, graph-grounded version is even stricter — every `ground_truth` is *verified against the live data* before it's committed:

```python
# from 04_cicd_gate_real.py — each answer was checked with Cypher first
GOLD = [
    {"question": "How many alerts have HIGH severity?",
     "ground_truth": "There are 2 alerts with HIGH severity."},
    {"question": "Who performed the maintenance job that resolved alert AL3?",
     "ground_truth": "Somchai performed maintenance job J1, which resolved alert AL3."},
]
```

> 📁 Class repo: `week15_production_graphrag/kg_mastery/part4_evaluation/04_cicd_gate_real.py` — a gold set where each row was verified to exist in the graph, so a low score means *quality*, not data mismatch.

---

## Part B — The eval-framework landscape

Don't hand-roll metrics if a framework already ships them. Four dominate in 2026:

| Framework | What it's for | In this repo? |
|---|---|---|
| **RAGAS** | RAG/graph metrics (faithfulness, answer relevancy, context precision/recall), LLM-as-judge built in, synthetic test-set generation. | ✅ Yes — Week 15 part 4 uses it end-to-end. |
| **DeepEval** | pytest-native — `assert_test()`, `@pytest.mark.parametrize`, 50+ metrics (tool correctness, task completion, hallucination). Best for adding eval to an existing CI with zero new mental model. | ➕ Add it when you want eval to *be* your pytest suite. |
| **Braintrust** | Hosted observability + **sandboxed** Python scorers; three-layer checks (behavior / context / ground-truth) so you can start before you have a gold set. | ➕ Add for production-grade, team-scale eval. |
| **Inspect AI** | UK AISI; 200+ pre-built evals, audit-grade logging, model-graded scoring. Slower but built for regulated / "explain it to an auditor" work. | ➕ Add for compliance-heavy domains. |

RAGAS scores four independent dimensions, each 0–1, using an LLM as the judge — so you learn *which* part broke, not just "it's bad":

```python
# from 01_ragas_eval.py — one LLM-as-judge run, four orthogonal scores
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
results = evaluate(dataset, metrics=metrics, llm=eval_llm, embeddings=eval_embeddings)
#  faithfulness       — is every claim supported by the retrieved context? (hallucination)
#  answer_relevancy   — does the answer address the question (not waffle)?
#  context_precision  — was the retrieved context mostly relevant? (over-retrieval)
#  context_recall     — did retrieval pull ALL the needed facts? (misses)
```

The point of multiple metrics: **low faithfulness + high relevancy = the model is confidently making things up.** A single number can't tell you that.

> 📁 Class repo: `week15_production_graphrag/kg_mastery/part4_evaluation/01_ragas_eval.py` — the four RAGAS metrics with target thresholds and how to read them.
> 📁 Class repo: `week15_production_graphrag/kg_mastery/part4_evaluation/FAILURE_MODES.md` — a "lowest metric → root cause → fix" table (low recall → missing context; low precision → over-retrieval).
> 📁 Class repo: `week10_production_observability/notebooks/04_langsmith.py` — turn on tracing, then *diff* a weak prompt vs a strong one. "You cannot fix what you cannot see."

---

## Part C — LLM-as-Judge (and its blind spots)

When the right answer isn't an exact string ("is this summary good?"), you grade with another LLM against a **rubric**. It scales where humans can't — but it has documented biases that silently corrupt scores if you ignore them.

| Bias | What goes wrong | Mitigation |
|---|---|---|
| **Position** | Favors whichever answer is shown first/last (GPT-4 flipped ~⅓ of preferences when order was swapped). | Randomize order; score both orders, keep only consistent verdicts. |
| **Verbosity** | Longer answers score higher regardless of quality. | Penalize unnecessary length; normalize by length. |
| **Self-preference** | A judge prefers answers in its own model family's style. | Cross-model judging (judge ≠ generator family); ensemble judges. |
| **Instruction-following** | Rewards polite, superficially-compliant answers that miss the actual goal. | Write a *specific* rubric; explicitly penalize shallow compliance. |

**Calibrate against humans before you trust the judge.** Label 30–200 examples by hand, run the judge, and check agreement (correlation / Cohen's Kappa). Iterate the judge *prompt* against its worst examples, validate on a held-out slice, then **freeze it** — "set and forget" for the experiment so the ruler doesn't move under you.

### The 60/30/10 scorer mix (the community standard)

Don't use the LLM judge for everything — it's the expensive, fuzzy option. Layer your scorers:

| Share | Type | Examples |
|---|---|---|
| **60%** | Code / deterministic | exact match, regex, JSON-schema validation, "contains entity X", tool-was-called |
| **30%** | Model-graded | LLM-as-judge with a rubric, RAGAS faithfulness, G-Eval |
| **10%** | Human-in-the-loop | the genuinely ambiguous cases a human must rule on |

Reach for the cheap deterministic check first; escalate to the judge only when the question is genuinely subjective.

---

## Part D — CI/CD for non-deterministic systems

Traditional CI asks "does it compile, do tests pass?" Agent CI must also ask **"did quality regress, did cost blow up?"** — across outputs that change every run. The answer is a **5-gate pipeline** layered on top of normal build/test:

| Gate | Stage | Checks | On fail |
|---|---|---|---|
| **1** | Lint & static | prompt syntax, schema, no dangerous patterns (prompts are code — review them in PRs) | block merge |
| **2** | Offline eval | run golden set, score with the 60/30/10 mix | block if pass-rate < threshold |
| **3** | Cost budget | tokens/request; flag if cost +15% | block if over budget |
| **4** | Canary | route ~5% of live traffic to the new version for ≥1h | auto-rollback on regression |
| **5** | Shadow | run new version on real traffic *without serving it* | feed traces back into the golden set |

**The non-determinism rule: assert on score thresholds and properties, not exact strings.** `expected == actual` breaks the moment the same input gives a different wording. Instead, gate on the *mean* score and on invariants ("answer contains the room number", "no hallucinated entity"). Average over 3+ runs to absorb variance; set an acceptable regression margin (~2%) *before* you test.

Gate 2 is the one you can build today — run the gold set, take the mean per metric, exit non-zero if any metric is below its floor:

```python
# from 03_cicd_gate.py — the gate that BLOCKS a bad merge
THRESHOLDS = {"faithfulness": 0.85, "answer_relevancy": 0.80,
              "context_precision": 0.75, "context_recall": 0.70}

scores = {m: float(rdf[m].mean()) for m in THRESHOLDS}
scores["passed"] = all(scores[m] >= THRESHOLDS[m] for m in THRESHOLDS)
...
sys.exit(0 if scores["passed"] else 1)   # non-zero exit fails the CI job
```

Wire that `sys.exit` into a GitHub Actions job and a prompt tweak that tanks faithfulness **fails the PR check instead of reaching production**:

```yaml
# .github/workflows/eval.yml (abridged from 03_cicd_gate.py)
- name: Run quality gate
  env: { ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }} }
  run: python eval_gate.py   # exits non-zero if any metric regresses
```

> 📁 Class repo: `week15_production_graphrag/kg_mastery/part4_evaluation/03_cicd_gate.py` — the gate mechanics: `THRESHOLDS`, mean-per-metric, `sys.exit`, plus the full GitHub Actions YAML.
> 📁 Class repo: `week15_production_graphrag/kg_mastery/part4_evaluation/04_cicd_gate_real.py` — the same gate against the *real* pipeline + graph-grounded gold set.

**Layer three regression types** (running only "targeted" tests is a documented anti-pattern — a reasoning-prompt change can break tool selection):
- **Direct** — re-run the tests for the capability you changed.
- **Adjacent** — re-run capabilities that *interact with* it (the misses live here).
- **Random spot-check** — 5–10 unrelated cases as a safety net.

> 📁 Class repo: `week15_production_graphrag/kg_mastery/part4_evaluation/05_improvement_loop.py` + `improvement_log.json` — log each iteration's four scores plus the action taken, so you keep changes only when scores rise.

---

## When to invest in eval (and how much)

✅ The output is subjective / non-deterministic and you keep tweaking prompts.
✅ Regressions are expensive (it's user-facing, billed, or safety-relevant).
✅ More than one person edits the prompts.
❌ A one-off script you'll run twice — a manual eyeball is fine; don't build a gate.
❌ The output is a deterministic value with one right answer — a plain unit test already covers it.

Start non-blocking: **measure first, add blocking thresholds once you know the baseline.**

---

## 🧪 Guided lab (offer this)

Build the smallest real eval gate, from a deterministic check up to an LLM-as-judge — runs at **$0, no API key** (the judge is a `MockLLM` stub).

### Warm-up (5–10 min) — a deterministic scorer (binary pass/fail)
Score one answer with code, no model:

```python
def exact_score(answer: str, expected: str) -> int:
    return int(expected.lower() in answer.lower())   # 1 = pass, 0 = fail

assert exact_score("Rooms 204 and 305 had HVAC issues.", "305") == 1
assert exact_score("No issues found.", "305") == 0
print("warm-up PASS")
```
**Pass = both asserts hold.** You just wrote a Gate-2 deterministic scorer (the 60% bucket).

### Skill Drill (15–30 min) — LLM-as-judge over a 5-row golden set ($0)
Write a rubric-based judge, run it over a golden set, and **fail if the mean is below threshold** — exactly like `03_cicd_gate.py`, but with a fake judge so it's free.

```python
import sys

# 1) A tiny golden set (your ground truth, normally a versioned CSV).
GOLD = [
    {"q": "Which rooms had HVAC issues?", "expected": "204 and 305"},
    {"q": "Who resolved the room 305 alert?", "expected": "Somchai"},
    {"q": "What country is supplier BrightLite in?", "expected": "China"},
    {"q": "How many HIGH alerts?", "expected": "2"},
    {"q": "Which room did guest Ben stay in?", "expected": "R201"},
]

# 2) The agent under test (swap in your real one). Row 4 is deliberately wrong.
def agent(q: str) -> str:
    canned = {
        "Which rooms had HVAC issues?": "Rooms 204 and 305 reported HVAC faults.",
        "Who resolved the room 305 alert?": "Technician Somchai resolved it.",
        "What country is supplier BrightLite in?": "BrightLite is based in China.",
        "How many HIGH alerts?": "There are 5 high alerts.",          # WRONG
        "Which room did guest Ben stay in?": "Guest Ben stayed in room R201.",
    }
    return canned.get(q, "I don't know.")

# 3) MockLLM judge — $0, deterministic. Substitutes any LLM-as-judge.
#    Rubric: 1.0 if the expected fact is present, else 0.0.
class MockLLM:
    def judge(self, answer: str, expected: str) -> float:
        return 1.0 if expected.lower() in answer.lower() else 0.0

# 4) Run the gate.
THRESHOLD = 0.80
judge = MockLLM()
scores = [judge.judge(agent(r["q"]), r["expected"]) for r in GOLD]
mean = sum(scores) / len(scores)

print(f"per-row scores: {scores}")
print(f"mean score: {mean:.2f}  (threshold {THRESHOLD})")
sys.exit(0 if mean >= THRESHOLD else 1)   # non-zero = CI fails the merge
```

Run it: it scores **0.80** (4/5 rows pass; the wrong "5 alerts" row fails) — right at the line. Now **break it to learn it:** flip a second canned answer to something wrong → mean drops to 0.60 → `sys.exit(1)` → the gate blocks. Then fix the agent → it passes. That swing is the whole point: the gate caught a regression you didn't eyeball.

**Stretch:** (1) add the position-bias guard — judge each answer twice with the rubric phrased two ways and only count consistent verdicts; (2) replace `MockLLM` with a real Claude/RAGAS judge and re-run; (3) wrap it in a GitHub Actions job (copy the YAML from `03_cicd_gate.py`).

### Evaluation criteria (weighted)
| # | Criterion | Weight |
|---|---|---|
| 1 | Warm-up deterministic scorer returns clean 1/0 and both asserts pass | 15% |
| 2 | Golden set + agent + judge are cleanly separated (you can swap any one) | 25% |
| 3 | Gate computes the **mean** and exits non-zero below threshold (no exact-string asserts) | 30% |
| 4 | "Break it" step demonstrably flips PASS→FAIL, then FAIL→PASS when fixed | 20% |
| 5 | Can name where this maps to the 5-gate pipeline (this is Gate 2) and the 60/30/10 mix | 10% |

**Pass = 4/5 criteria (≥ 70% weighted), and criterion 3 must be one of them.**

End on the mantra: *frameworks change, but "build a golden set, score every change, gate the merge" is the discipline that keeps agents shippable.* Next stops: `production-and-observability` (tracing & monitoring the live system) and `knowledge-graph-mastery` (the RAG-specific deep dive).
