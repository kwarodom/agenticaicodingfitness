# ▶ Harness Lab 06 — Reading a results table: streams, noise, ablations, transfer, pass^k

> Part of Week 31 · Harness²: agents that improve their own harness. You improve the harness of one simulated agent, the **HVAC Expert agent of Ops Assistant** (a fictional always-on operations assistant), from execution evidence, and you see the real output. Everything is synthetic (two fictional hotels, invented alarms and work orders), so every lab is free, offline and safe. Labs run on **🧠 SIM** by default: a deterministic teaching model of an agent and its editors. Flip the runner to **🦙 LIVE** where a lab offers it, and a real local model plays one role.

**What you'll actually do**
- Read the paper's headline numbers the way the paper reports them: the mean of three shuffled streams, with a standard deviation.
- Learn one reading rule (do all three streams agree, and is the gap larger than one SD?) and apply it to every row.
- Run base versus Harness² on both simulated executors, and check whether the weaker one gains more.
- Build the SIM's own ablation table (no library, generic editor, no composition, k = 3, k = 5, sequential) and say honestly which rows the table can and cannot support.
- Promote a harness version on one executor and measure it on the other, then see why pass^3, not pass@3, promotes a version.

**Time** ~45 min · **Difficulty** intermediate · **Hardware** this laptop only (no model needed)

**Sources:** [Harness² paper, arXiv 2610.14224](https://arxiv.org/html/2610.14224) · [google-research/harness2](https://github.com/google-research/harness2) (Apache-2.0; snapshots of `README.md`, `EXPERIMENTS.md` in `common/sources/`) · [Anthropic, Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) · the evals and observability of Week 10

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| This repo's Python | `.venv/bin/python --version` → 3.13 | runs every lab and the Harness Lab Runner |
| Modules 02–05 | you know a step, KEEP_BASE, edit-row verdicts and the library | this module measures what those pieces add |
| No model | every lab here is SIM only | a results table is about the protocol, not about one model's output |

Every number in this module carries one of two labels. **Reported in the paper** means a number or finding from the Harness² paper (arXiv 2610.14224) or its repository, VERIFIED against the source. Only the repository snapshots in `common/sources/` are quoted word for word, so the paper's findings appear here as plain statements, not quotations. **SIM** means you produced it on this laptop with the course's teaching model. The SIM encodes the paper's findings as assumptions, so a SIM table shows how to *read* a result. It never confirms the paper's result.

```bash
# on: laptop
.venv/bin/python week31_harness2_recursive_harnessing/06_reading_the_results/labs/lab06_1_three_streams.py
```

✓ Checkpoint: Lab 06-1 runs in about a second and prints six steps.

## 1 · What the paper reports, and how it reports it

The paper's headline is that one Harness² step helps every configuration it tried. Its averages (as reported in the paper):

| Benchmark | Executor | Base | One step | Gain |
|---|---|---|---|---|
| WorkBuddy-Bench average | Gemini 3.5 Flash / OpenCode | 72.4 | 77.5 | +5.1 |
| WorkBuddy-Bench average | GLM-5 / OpenCode | 64.2 | 75.7 | +11.5 |
| JobBench | Gemini 3.5 Flash | 31.5 | 36.0 | +4.5 |
| LAB | GLM-5 | 59.6 | 71.4 | +11.8 |

Before you trust a gain, ask how the number was made. The paper reports each score as the mean over three independent streams, with the standard deviation taken over three runs in shuffled task order (as reported in the paper).

A stream is the whole task list in one order, with the library read before each task and appended after it (Module 05). Three streams in three orders give three means. Their spread is the noise of the protocol. This course turns the paper's own spread into a rule of thumb (EXAMPLE): the paper's base-harness standard deviations on JobBench are 1.1 and 2.6 points (as reported in the paper), so a difference smaller than about 3 points on a 30-task family is noise unless all three streams agree in direction.

So the course reads every row with two questions:

1. **Do all three streams agree?** Pair stream *s* of one arm with stream *s* of the other (same task order). Are all three differences the same sign?
2. **Is the gap larger than one SD?** Is the difference of the means larger than the larger of the two standard deviations?

Both yes: **beyond noise**. Streams agree but the gap is inside one SD: **direction only** (more streams or more tasks needed). Streams disagree: **within noise**. The course's pre-registered criteria (EXAMPLE design, Module 09) ask the same two questions, which is why this module uses them. `readkit.py` in this folder holds the rule, and Exercise 06 asks you to write it yourself.

Two more results set the scene (as reported in the paper). The weaker executor gains more, so a weaker model with an improved harness can beat a stronger model on its original harness. And the editor need not be the stronger model: with the same model as executor and editor, all five models tested, from GLM-5 to Claude Opus 5, improve on every benchmark.

✓ Checkpoint: you can say what "77.5" stands for (the mean of three stream means), and state the reading rule in one sentence.

## 2 · Three streams in the SIM: base versus Harness², both executors

Lab 06-1 runs the paper's protocol on the course's 17 refine tasks. The editor is fixed (`sim-cloud-editor`) for both executors, as the paper keeps one default editor. The scores come from `bench.judge` after each step; the loop never sees them.

**Measured on this Mac — protocol: open-world streams** (each task's final deliverable after its own step, refine half, library read before and appended after within a stream, three shuffled streams, mean ± SD).

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 2 · The SIM, same protocol: 17 refine tasks × 3 shuffled streams, base vs one Harness² step per task
│ executor   arm              stream 1 / 2 / 3    mean ± SD*
│ ─────────  ───────────────  ──────────────────  ──────────
│ sim-cloud  base harness v1  81.2 / 83.8 / 75.8  80.3 ± 4.1
│ sim-cloud  Harness² k=1     96.5 / 95.0 / 91.5  94.3 ± 2.6
│ sim-local  base harness v1  76.3 / 69.2 / 71.0  72.2 ± 3.7
│ sim-local  Harness² k=1     86.5 / 80.1 / 88.3  85.0 ± 4.3
◆ * measured after each step by bench.judge (hidden truth); the loop never saw a score

▣ STEP 3 · Read each difference stream by stream: do all three agree, and is the gap larger than one SD?
│ executor   per-stream Δ           Δ mean  larger SD  all agree  |Δ| > SD  reading     
│ ─────────  ─────────────────────  ──────  ─────────  ─────────  ────────  ────────────
│ sim-cloud  +15.3 / +11.2 / +15.7  +14.1   4.1        yes        yes       beyond noise
│ sim-local  +10.2 / +10.9 / +17.3  +12.8   4.3        yes        yes       beyond noise
✓ one step helps both SIM executors beyond stream noise — the same shape as the paper's 'every configuration'
```

The SIM's standard deviations (2.6 to 4.3) are larger than the paper's JobBench ones. Seventeen tasks is a small family, and one slip on a short stream moves its mean by several points. That is the first honest lesson: **a smaller task set needs a bigger gap before you believe it.**

Now the second question: does the weaker executor gain more? Compare the gains, not the scores.

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 4 · Does the weaker executor gain more? Compare the GAINS, stream by stream
│ gain per stream  sim-cloud +15.3 / +11.2 / +15.7   sim-local +10.2 / +10.9 / +17.3
│ local gain − cloud gain: -1.3 · per stream -5.1 / -0.3 / +1.6 · larger SD of the gains 3.9 → within noise
◆ sim-local with Harness² (85.0) vs sim-cloud on its base (80.3): +4.7 · per stream +5.3 / -3.7 / +12.5 → within noise (the paper reports the weaker executor with an improved harness ahead, for GLM-5)
⚠ the SIM ENCODES 'weaker executors follow prose less and hand-roll more' as an assumption, so this shows the mechanism; it is not evidence for the paper's finding
```

Here the table does not show it. On the mean the local gain is 1.3 points *smaller*, and the streams disagree (−5.1 / −0.3 / +1.6): within noise. The SIM encodes the mechanism behind the paper's finding, yet 17 tasks and three streams do not show it. That is not evidence against the paper either. The honest sentence is "this table cannot tell which executor gains more".

The second line asks the paper's other question: does the weaker executor with a refined harness beat the stronger one on its base? On the mean it is ahead (85.0 against 80.3), but stream 2 goes the other way (80.1 against 83.8). By the same rule that is within noise too.

✓ Checkpoint: you can explain why "−1.3, per stream −5.1 / −0.3 / +1.6" earns *within noise*, and why a 4.7-point lead on the mean with one stream against it earns the same label.

## 3 · Same-model editing, tool use and what the gain cost

Step 5 of Lab 06-1 lets `sim-local` edit its own harness. The SIM local editor is set to miss a failure 40% of the time, against 25% for the cloud editor.

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 5 · Same-model editing: sim-local as its own editor versus the fixed cloud editor
│ executor   editor            stream 1 / 2 / 3    mean ± SD 
│ ─────────  ────────────────  ──────────────────  ──────────
│ sim-local  sim-cloud-editor  86.5 / 80.1 / 88.3  85.0 ± 4.3
│ sim-local  sim-local-editor  84.4 / 83.3 / 83.6  83.8 ± 0.6
◆ self-editing − cloud editor: -1.2 · per stream -2.1 / +3.2 / -4.7 → within noise
◆ the SIM local editor is set to miss a failure 40% of the time (cloud editor 25%), yet over 17 tasks and three streams that difference does not show: the table cannot tell the two editors apart
```

"Within noise" is not "equal". It means this table cannot tell the two editors apart. For a production agent with a local tier the question matters: if a smaller open-weight model on local hardware can edit its own harness, its traces never leave that machine. The course's EXAMPLE design therefore pre-registers it as its own arm (arm F, Module 09) with its own threshold, instead of reading it off a table like this one.

Step 6 counts what the final runs did. The paper has a tool-use table too (as reported in the paper): tool calls per task rose from 27.4 to 32.8 while the tool-error rate fell from 14.4% to 9.1%. These two numbers are not in the repository snapshots in `common/sources/`, so they are not re-checked here. The lesson plan compresses the point to one line:

> “fewer errors, more useful calls, at a compute cost” — facilitator guide, Part 3

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 6 · Tool use and cost: what the final runs did, and what the loop spent to get them
│ executor   tool calls / task  tool-error rate  executions / task  editor sessions / task
│ ─────────  ─────────────────  ───────────────  ─────────────────  ──────────────────────
│ sim-cloud  2.67 → 3.24        6.6% → 1.8%      1 → 2.4            0 → 4.0               
│ sim-local  2.59 → 3.20        6.8% → 5.5%      1 → 2.5            0 → 4.0               
◆ more calls, fewer errors: scripts, validators and the maintenance check ADD useful work; refinement need not shorten a run
⚠ the base row costs one execution per task; Harness² adds references, probes and editor sessions — the paper publishes no token totals, only that it 'requires additional executions and editor sessions'
```

The SIM shows the same shape: a refined run makes more calls (the `cmms.work_orders` check, the script, the validator) and fewer of them fail. More calls is not worse here; each added call does work the base skipped. The right column is the cost caveat. The paper says it plainly (as reported in the paper): Harness² needs extra executions and editor sessions to propose, probe and compose its harness changes. When it compares Harness² with fixed production harnesses, it adds that refinement spends extra test-time compute.

✓ Checkpoint: you can say why "more tool calls" is not a regression in this table, and name the cost a results table must report next to the gain.

## 4 · The ablation table: which parts show up beyond noise

An ablation removes one part of the method and measures the loss. The paper's ablations at k = 1 (as reported in the paper):

| Ablation | WorkBuddy-Bench | JobBench |
|---|---|---|
| without the library | −3.1 | −1.8 |
| without scoping (generic editors) | −2.9 | −3.7 |
| without composition | −1.6 | −1.9 |

And the component removals from the improved harness: skills −5.3 on average (6.9 on Web), scripts −3.7 (7.4 on Office), prompt text −1.5. The paper adds that skills and scripts make up roughly half the edits on each benchmark.

Read those with Section 1's rule. Against JobBench SDs of 1.1 and 2.6, the composition rows (1.6 and 1.9) are the ones to read most carefully.

Lab 06-2 builds the SIM's own table, every arm as three streams, read against Harness² k = 1. **Measured on this Mac — protocol: open-world streams** (same as Section 2).

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 2 · The SIM's ablation table on sim-cloud (3 shuffled streams each, read against k=1)
│ arm             stream 1 / 2 / 3     mean ± SD*  Δ vs k=1  reading         execs/task  sessions/task  always-on
│ ──────────────  ───────────────────  ──────────  ────────  ──────────────  ──────────  ─────────────  ─────────
│ Harness² k=1    96.5 / 95.0 / 91.5   94.3 ± 2.6  —         —               2.4         4.0            271      
│ no library      97.6 / 93.5 / 90.3   93.8 ± 3.7  -0.5      within noise    2.3         4.0            265      
│ generic editor  94.1 / 90.9 / 88.3   91.1 ± 2.9  -3.2      beyond noise    1.8         3.0            279      
│ no composition  97.6 / 92.1 / 92.7   94.1 ± 3.0  -0.2      within noise    2.4         3.0            265      
│ parallel k=3    98.8 / 100.0 / 95.4  98.1 ± 2.4  +3.7      beyond noise    7.3         11.0           281      
│ parallel k=5    97.6 / 95.7 / 92.3   95.2 ± 2.7  +0.9      direction only  12.5        17.0           284      
│ sequential k=3  98.5 / 92.7 / 97.4   96.2 ± 3.1  +1.9      within noise    4.8         10.0           279      
│ sequential k=5  92.5 / 98.8 / 98.0   96.4 ± 3.4  +2.1      within noise    7.0         16.0           282      
◆ * measured after each step by bench.judge · always-on = mean always-on tokens of the adopted harnesses (base 245)

▣ STEP 3 · The SIM's ablation table on sim-local (3 shuffled streams each, read against k=1)
│ arm             stream 1 / 2 / 3     mean ± SD*  Δ vs k=1  reading         execs/task  sessions/task  always-on
│ ──────────────  ───────────────────  ──────────  ────────  ──────────────  ──────────  ─────────────  ─────────
│ Harness² k=1    86.5 / 80.1 / 88.3   85.0 ± 4.3  —         —               2.5         4.0            270      
│ no library      84.1 / 76.6 / 84.8   81.8 ± 4.5  -3.1      direction only  2.3         4.0            263      
│ generic editor  81.9 / 73.3 / 78.2   77.8 ± 4.3  -7.2      beyond noise    1.8         3.0            275      
│ no composition  86.8 / 78.6 / 81.9   82.4 ± 4.1  -2.5      within noise    2.5         3.0            264      
│ parallel k=3    94.2 / 95.4 / 95.7   95.1 ± 0.8  +10.1     beyond noise    8.0         11.0           286      
│ parallel k=5    97.6 / 92.4 / 95.7   95.2 ± 2.6  +10.3     beyond noise    13.4        17.0           288      
│ sequential k=3  100.0 / 97.1 / 95.9  97.7 ± 2.1  +12.7     beyond noise    5.2         10.0           282      
│ sequential k=5  91.2 / 90.0 / 88.3   89.8 ± 1.5  +4.9      within noise    7.4         16.0           284      
```

Read the rows one at a time.

- **Generic editor.** On sim-cloud it costs 3.2 points in all three streams: beyond noise. On sim-local it costs 7.2 (−4.6 / −6.8 / −10.1): beyond noise too. On both executors its adopted harnesses carry more always-on text (279 against 271 tokens on sim-cloud, 275 against 270 on sim-local), for a lower score. It is the one ablation the SIM shows on both executors.
- **No library, no composition.** On sim-cloud both sit inside one SD and the streams disagree: within noise. On sim-local, no library costs 3.1 points in all three streams (−2.4 / −3.5 / −3.5), but the SD is 4.5: direction only. No composition on sim-local is −2.5 with one stream at +0.3: within noise. The paper's 1.6 to 3.1 points are not shown beyond noise on 17 tasks. The honest sentence is "**not shown here**" (for the sim-local library row, "a consistent direction, not yet sized"), which is neither "no effect" nor "shown". The course's pre-registered criteria for Ops Assistant (EXAMPLE design, Module 09) draw the consequence in advance: adopt the library only if arm C (with it) beats arm B (without it) by more than the stream standard deviation; otherwise ship task-level harnessing only and defer the library.
- **k = 3 and k = 5.** Look at the cost columns first: k = 3 triples the executions, k = 5 roughly quintuples them. On sim-cloud, k = 3 is beyond noise, but k = 5 is not above k = 1 by more than one SD. On sim-local, starting from a lower k = 1 row, parallel k = 3, parallel k = 5 and sequential k = 3 are all beyond noise (+10.1 to +12.7).
- **Sequential k = 5 on sim-local.** The mean is up 4.9 points, more than the SD of 4.3, yet the third stream is exactly +0.0 (88.3 against 88.3), so the streams do not agree: *within noise*. A big mean with a stream that does not go along is exactly the case the rule exists for.

Step 4 makes the k comparison explicit:

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 4 · Two comparisons the table invites: k=5 against k=3, and sequential against parallel at k=5
│ executor   comparison                     per-stream Δ        Δ mean  larger SD  reading     
│ ─────────  ─────────────────────────────  ──────────────────  ──────  ─────────  ────────────
│ sim-cloud  parallel k=5 − parallel k=3    -1.2 / -4.3 / -3.1  -2.9    2.7        beyond noise
│ sim-cloud  sequential k=5 − parallel k=5  -5.1 / +3.1 / +5.7  +1.2    3.4        within noise
│ sim-local  parallel k=5 − parallel k=3    +3.4 / -3.0 / +0.0  +0.1    2.6        within noise
│ sim-local  sequential k=5 − parallel k=5  -6.4 / -2.4 / -7.4  -5.4    2.6        beyond noise
```

The paper reports that both recursion modes improve monotonically as k goes from 1 to 5.

The SIM does not. On sim-cloud, k = 5 is *lower* than k = 3 in all three streams, by just more than one SD: beyond noise. On sim-local the two are level (+0.1, per stream +3.4 / −3.0 / +0.0): within noise. Neither executor shows k = 5 above k = 3. Say that, and say why you might expect it: with five finals the blind selector has more chances to deliver a weaker one. Module 07 counts it. The course's EXAMPLE advice for Ops Assistant does not depend on which curve is right: expect no immediate benefit from k above 1. The paper's recursion gains at k = 5 (2.9 to 3.9 points on WorkBuddy averages, reported in the paper) cost about five times the probes, so an always-on agent should earn them family by family.

The sequential rows are a puzzle on purpose. On sim-local, sequential k = 3 is the best row of the whole table (97.7 ± 2.1), yet the same mode at k = 5 drops to 89.8: below sequential k = 3 in every stream, and below parallel k = 5 beyond noise (−5.4, Step 4). On sim-cloud, sequential against parallel at k = 5 is within noise. One row beyond noise on one executor is a result to re-run with more tasks, not a policy.

✓ Checkpoint: for each ablation row you can say *beyond noise*, *direction only* or *within noise*, and you can explain why "within noise" does not mean "the library does nothing".

## 5 · Transfer: a version refined on one executor, measured on the other

Ops Assistant's EXAMPLE design runs two tiers: a cloud model and a smaller open-weight model on local hardware. One harness tree for both would be cheaper to maintain. The paper says transfer happens, unevenly (as reported in the paper): a harness refined for Gemini 3.5 Flash gives GLM-5 +17.3 points on Code (58.5 → 75.8), Gemini 3.7 Flash +0.3 on Code and +6.4 on Office, Opus 4.8 +2.6 and +1.1. The benefit varies a lot with the executor and the domain.

Lab 06-3 measures it. It promotes one **version** per executor from that executor's stream records with the reviewer's merge (`bench.promote`, `min_support=2`, EXAMPLE), freezes it, and evaluates it on the **test half**, which never entered the loop. **Measured on this Mac — protocol: version** (a promoted, fixed version on the frozen test half, three trials per task, pass^3), repeated under three evaluation seeds for a mean ± SD. This is a different protocol from Sections 2–4, so do not compare these numbers with those.

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 3 · Evaluate every version on the frozen TEST half (17 tasks × 3 trials) on both executors, 3 seeds
│ executor   version               seed 0 / 1 / 2      mean ± SD*  Δ vs base             pass^3*  unsafe
│ ─────────  ────────────────────  ──────────────────  ──────────  ────────────────────  ───────  ──────
│ sim-cloud  base v1               81.8 / 86.0 / 83.8  83.9 ± 2.1  —                     0.49     0     
│ sim-cloud  refined on sim-cloud  97.4 / 96.7 / 98.4  97.5 ± 0.9  +13.6 · beyond noise  0.92     0     
│ sim-cloud  refined on sim-local  93.5 / 95.7 / 96.5  95.2 ± 1.6  +11.4 · beyond noise  0.88     0     
│ sim-local  base v1               70.9 / 71.9 / 73.0  71.9 ± 1.1  —                     0.22     0     
│ sim-local  refined on sim-cloud  94.4 / 91.6 / 92.9  93.0 ± 1.4  +21.0 · beyond noise  0.71     0     
│ sim-local  refined on sim-local  88.5 / 89.6 / 89.9  89.3 ± 0.7  +17.4 · beyond noise  0.69     0     
◆ * measured on the test half by bench.evaluate; nothing in the loop ever read these tasks

▣ STEP 4 · Read transfer per direction
│ sim-cloud version on sim-local: 93.0 vs sim-local's own version 89.3 → +3.6 · beyond noise
│ sim-local version on sim-cloud: 95.2 vs sim-cloud's own version 97.5 → -2.3 · beyond noise
⚠ in the SIM both executors share the same weaknesses and the same editors, so a version moves almost intact. The SIM cannot reproduce the paper's uneven transfer (+17.3 vs +0.3); it can only show how to measure it
```

Three things to read here.

1. **Both versions transfer in the SIM**, in both directions, well beyond noise. That is because the two SIM executors fail in the same ways. Real models differ more, which is the paper's point.
2. **The version refined on the weaker executor is the weaker version, on both executors.** On sim-cloud it trails the cloud-refined version by 2.3 points, beyond noise. On sim-local, its own executor, it trails too: the cloud-refined version scores 3.6 points more there, beyond noise. Step 2 shows one reason: the local streams supported 11 adopted edits against 12, so less passed the reviewer's merge.
3. **Transfer is a measurement per direction.** The course's pre-registered arm E (Module 09) does exactly this: the cloud-refined version on the local executor *and* the local-refined version on the cloud executor. Keep one library per executor, and measure before you share a version.

✓ Checkpoint: you can say why this table cannot be put next to the stream tables of Sections 2–4, and what arm E measures.

## 6 · pass@k versus pass^k

A chief engineer reads the digest every morning. A digest that is right two mornings out of three is a digest nobody trusts. That is the difference between two numbers that look alike. Anthropic's evals guide draws the line: pass@k fits a tool where one success is enough, and pass^k fits an agent that must be consistent.

pass@k is the chance that at least one of k trials succeeds. pass^k is the chance that all k succeed. With independent trials and a per-trial success p, pass@k = 1 − (1 − p)^k and pass^k = p^k.

**Expected output** (captured on this Mac, SIM)

```
│ per-trial success p  k  pass@k = 1 − (1 − p)^k  pass^k = p^k
│ ───────────────────  ─  ──────────────────────  ────────────
│ 0.75                 3  98.4%                   42.2%       
◆ worked example: 75% per trial is about 42% across three trials
│ version               executor   per-trial p  pass@3 if independent  p^3 if independent  pass^3 observed
│ ────────────────────  ─────────  ───────────  ─────────────────────  ──────────────────  ───────────────
│ base v1               sim-cloud  0.69         0.97                   0.32                0.41           
│ base v1               sim-local  0.51         0.88                   0.13                0.23           
│ refined on sim-cloud  sim-cloud  0.98         1.00                   0.94                0.94           
│ refined on sim-cloud  sim-local  0.92         1.00                   0.78                0.77           
│ refined on sim-local  sim-cloud  0.94         1.00                   0.83                0.88           
│ refined on sim-local  sim-local  0.86         1.00                   0.64                0.71           
◆ observed pass^3 is not p^3 (at or above it on 5 of 6 rows): trials of one task are not independent — some tasks pass every time, some never. Measure pass^3 per task; do not derive it from the pass rate
```

Seventy-five percent per trial sounds good. It is 98% as pass@3 and 42% as pass^3. The base harness on sim-cloud passes a trial 69% of the time and passes all three trials on only 41% of the test tasks (seed 0). Observed pass^3 is not p^3. On five of the six rows it is at or above p^3, because the trials of one task are correlated: some tasks always pass, some never do. On one row (the cloud-refined version on sim-local) it is a little below (0.77 against 0.78). Either way the pass rate does not give you pass^3, so compute it per task from real trials (`bench.evaluate` does). Customer-facing families (digests, notifications, Thai drafts) are promoted on pass^3 in the course's EXAMPLE design; pass@k is only a diagnostic.

✓ Checkpoint: you can compute pass^3 at p = 0.9 in your head (about 73%) and say which of pass@3 and pass^3 promotes a digest version.

## Labs — run them here

**labs/lab06_1_three_streams.py** — Three shuffled streams: base versus Harness² for both executors, same-model editing, tool use and cost.

**labs/lab06_2_ablation_table.py** — The SIM's ablation table on both executors, every row read against k = 1 with its cost.

**labs/lab06_3_transfer_and_pass_k.py** — A version promoted on each executor, measured on both on the frozen test half, then pass@k versus pass^k.

## Try it yourself

`exercises/ex06_read_the_table.py` has three TODOs:

1. Write the reading rule `reading_of(a, b)`: beyond noise, direction only, or within noise.
2. Decide which of eight claims the evidence supports (five about a SIM table, two about the paper, one about pass^3).
3. Write `pass_hat(p, k)`.

```bash
# on: laptop
.venv/bin/python week31_harness2_recursive_harnessing/06_reading_the_results/exercises/ex06_read_the_table.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ reading rule matches on all 6 comparisons (e.g. no library vs k=1: within noise)
✓ pass^3 at p = 0.90 is 0.729; at p = 0.75 it is 0.422 (the worked example's ≈ 42%)
✓ all eight claims judged: 1, 3, 6 supported; 2, 4, 5, 7, 8 not supported by this evidence

═ Done. Module 07 asks why a table can mislead even when you read it carefully: three failure modes.
```

<details><summary>Hint — TODO 1</summary>

Pair the lists by position: `diffs = [y - x for x, y in zip(a, b)]`. "Agree" means every difference is positive or every difference is negative (a zero difference breaks agreement). Compare `abs(mean(b) - mean(a))` with `max(stdev(a), stdev(b))` from the `statistics` module.

</details>

<details><summary>Hint — TODO 2</summary>

"Not supported" covers two cases: the table shows nothing beyond noise (claims 2 and 5), or it shows the opposite (claim 4: k = 5 is below k = 3). For claim 7, compare GLM-5's +17.3 with Gemini 3.7 Flash's +0.3 from Section 5. For claim 8, use TODO 3.

</details>

✓ Checkpoint: all three checker lines are ✓.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Your stream means differ from the tutorial | an edited kit, or a different editor | `git diff week31_harness2_recursive_harnessing/common`; the labs fix `editor="sim-cloud-editor"` for both executors |
| `streams(..., mode="sequential")` gives the parallel numbers | `stream(mode=)` is base or harness2; the recursion mode is `recursion=` | pass `recursion="sequential"`, as Lab 06-2 does |
| A row reads "within noise" although its mean moved a lot | one stream went the other way, or did not move | that is the rule working; run more streams or more tasks before you write a sentence about it |
| Lab 06-3 numbers do not match Lab 06-2 | different protocols: fixed versions on the test half vs open-world streams on the refine half | never put the two in one table |

## Next

[Lab 07 — Three ways harnessing fails, and measurement hygiene](../07_failure_modes/TUTORIAL.md): blind selection, a premise every candidate shares, attribution noise, and the rubric an executor should never be able to fetch.
