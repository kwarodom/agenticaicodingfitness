# ▶ Harness Lab 07 — Three ways harnessing fails, and measurement hygiene

> Part of Week 31 · Harness²: agents that improve their own harness. You improve the harness of one simulated agent, the **HVAC Expert agent of Ops Assistant** (a fictional always-on operations assistant), from execution evidence, and you see the real output. Everything is synthetic (two fictional hotels, invented alarms and work orders), so every lab is free, offline and safe. Labs run on **🧠 SIM** by default: a deterministic teaching model of an agent and its editors. Flip the runner to **🦙 LIVE** where a lab offers it, and a real local model plays one role.

**What you'll actually do**
- Run steps at k = 3 and k = 5, judge every final after the fact, and count how often the blind selector delivered a weaker one.
- Carry one wrong workspace fact from one hotel's library into another and watch every candidate inherit it, so no selector can recover.
- Challenge that shared premise with a probe that needs no score, and lift the cap by re-verifying the fact.
- See an edit row change verdict when two more probes run, in both directions.
- Let a simulated executor fetch its own rubric, then close the leak three ways: an egress block, the measurement-word ban and the grader-verdict check.

**Time** ~45 min · **Difficulty** intermediate · **Hardware** this laptop only (no model needed)

**Sources:** [Harness² paper, arXiv 2610.14224](https://arxiv.org/html/2610.14224) · [google-research/harness2](https://github.com/google-research/harness2) (Apache-2.0; snapshots of `prompts.py`, `spec.py`, `library.py`, `EXPERIMENTS.md` in `common/sources/`) · the evals of Week 10, the security controls of Week 28

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| This repo's Python | `.venv/bin/python --version` → 3.13 | runs every lab and the Harness Lab Runner |
| Module 06 | you can read a row as beyond noise, direction only or within noise | this module explains why even a well-read table can mislead |
| Modules 04–05 | edit-row verdicts, repeats, the library and `reverify` | the failure modes live in exactly those pieces |

The paper names three failure modes: selection can prefer a weaker candidate (55 over 95); an incorrect premise can propagate across candidates (losses of 2–11 points); and execution variability confounds attribution.

A fourth risk is not a failure of the loop but of the measurement around it: the executor reaching its own grading criteria. Section 4 covers it. The lesson plan's question for this part is the one to keep in mind:

> “Then: which failure would a human reviewer catch? Blind selection yes; a propagated premise no, since every candidate agrees.” — facilitator guide, Part 4

```bash
# on: laptop
.venv/bin/python week31_harness2_recursive_harnessing/07_failure_modes/labs/lab07_1_blind_selection.py
```

✓ Checkpoint: Lab 07-1 runs in about a second, and you can name the three failure modes before you read their sections.

## 1 · Blind selection: the better final was in the pool

At k > 1 a step produces k finals and a selector delivers one. The selector is blind on purpose. It sees the request, the sources and the anonymised deliverables, never the harness and never a score:

> “The ids are arbitrary and shuffled, and tell you nothing about how an output was … produced. No candidate is privileged and there is no credit for a more elaborate one” — `harness2/core/prompts.py`

Blindness keeps the selector from rewarding a harness for being elaborate. The price is that it can be wrong. The paper does not report a selector accuracy, but its failure analysis has a clear case: selection choosing a 55-scoring output over a 95-scoring one.

Lab 07-1 counts the same thing in the SIM. It runs every refine task at k = 3 and k = 5, three seeds, both executors. After each step, `bench.judge` scores every final, so the lab can see what the selector could not. The SIM selector is a careful reader that never recomputes and notices each visible problem 75% of the time (an assumption in `common/h2kit.py`).

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 2 · k = 3 and k = 5, 17 tasks × 3 seeds, both executors — then judge every final after the fact
│ k  executor   steps  weaker final picked  share  mean delivered*  mean of the k finals*  mean best final*
│ ─  ─────────  ─────  ───────────────────  ─────  ───────────────  ─────────────────────  ────────────────
│ 3  sim-cloud  51     3                    5.9%   98.1             95.6                   99.6            
│ 3  sim-local  51     10                   19.6%  94.3             90.9                   99.0            
│ 5  sim-cloud  51     10                   19.6%  95.2             95.9                   100.0           
│ 5  sim-local  51     9                    17.6%  96.1             92.7                   100.0           
◆ * measured after each step by bench.judge · 'mean best final' is an oracle no real loop has: it needs the hidden truth
◆ the pick beats the average final for k=3 sim-cloud, k=3 sim-local, k=5 sim-local, but NOT for k=5 sim-cloud; every row leaves points behind the best final
```

Three readings.

1. The selector is usually useful: in three rows of four it beats the average final.
2. It always leaves points behind. The best final scores 99.0 to 100 on average; the delivered one 94.3 to 98.1.
3. More finals can mean more chances to deliver a weaker one: on sim-cloud the share rises from 5.9% at k = 3 to 19.6% at k = 5. On sim-local it does not (19.6% at k = 3, 17.6% at k = 5), so read it as a tendency on one executor, not a law: three seeds of 17 tasks are a small sample. The sim-cloud rise is one visible reason why k = 5 did not beat k = 3 there in Module 06.

The worst pick in the run is a 66.7 delivered over two 100s, on a quiet trend check:

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 3 · The worst pick: F1-11 · k=3 · sim-cloud · seed 2 — the selector delivered 66.7 with 100.0 in the pool
│ request: Morning trend check for CH-3. Anything to act on?
│ final           harness    deliverable (what the selector reads)                score*  failed checks*
│ ──────────────  ─────────  ───────────────────────────────────────────────────  ──────  ──────────────
│ #1 ← delivered  KEEP_BASE  diagnosis: low load cycling · proposal: none needed  66.7    diagnosis     
│ #2              KEEP_BASE  diagnosis: within design · proposal: none needed     100.0   —             
│ #3              KEEP_BASE  diagnosis: within design · proposal: none needed     100.0   —             
⚠ the delivered final disagrees with the other 2 finals, which all agree — and nothing on its face says it is wrong: the SIM selector reads each artifact against the request and sources but never recomputes a figure or re-derives a diagnosis
◆ a majority vote would have caught this one, and would have locked in a mistake every final shares (Lab 07-2)
```

All three finals ran the base harness (the step kept it), so the pool holds three executions of one harness, and one of them misread the trend. A person who saw all three would ask why one diagnosis differs. That is why this failure is **catchable by review**. The course's EXAMPLE design has three controls: human review before Serve, a monthly selector audit that passes when fewer than one in five selections prefer the weaker deliverable, and k = 1 by default, so no selection happens until a family earns k > 1. None of them lets the selector see a score.

✓ Checkpoint: you can say why the selector is blind, what that costs, and why a reviewer can catch this failure.

## 2 · A premise that caps the pool

The second failure is worse because nobody sees it. If every candidate inherits the same wrong assumption, every final carries it. The selector, the composer and a reviewer comparing candidates all see agreement.

Module 05 built the setup. The library learns at Harbour View, whose O&M manual sets the condenser-approach limit at 3.0 K. The fault-review skill it adopts carries that limit as a workspace fact. The second hotel, Chelsea Riverside Hotel, has a 4.5 K limit. The repository's editor prompt says what to do with a copied fact:

> “an earlier task's mapping or threshold is a hypothesis here, not a fact.” — `harness2/core/prompts.py`

Lab 07-2 runs the six second-hotel fault reviews with the fact copied unchecked (`reverify=False`) and re-verified. Each task reads a fresh copy of the same Harbour View library, so the comparison isolates the premise.

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 3 · Six fault reviews at Chelsea Riverside Hotel (fictional) — manual limit 4.5 K · k=1 · seed 2
│ library facts     score per task*                    mean*  limit in the adopted harnesses
│ ────────────────  ─────────────────────────────────  ─────  ──────────────────────────────
│ copied unchecked  80 / 100 / 75 / 100 / 75 / 100     88.3   3.0 K                         
│ re-verified       100 / 100 / 100 / 100 / 100 / 100  100.0  4.5 K                         
◆ * measured after each step by bench.judge — the loop never saw a score
```

Now the same at k = 5. Does selection rescue the unchecked library? Lab 07-2 lists every final in every pool:

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 4 · k=5 with the facts copied unchecked: list EVERY final in each pool
│ task     approach K  limits in the pool  all 5 carry 3.0  score per final*             pool best*  delivered*
│ ───────  ──────────  ──────────────────  ───────────────  ───────────────────────────  ──────────  ──────────
│ F1-CH40  3.4         3.0                 yes              80 / 100 / 100 / 60 / 100    100         80        
│ F1-CH41  3.9         —                   no               100 / 100 / 67 / 100 / 100   100         100       
│ F1-CH42  4.2         3.0                 yes              75 / 75 / 75 / 75 / 75       75          75        
│ F1-CH43  5.9         3.0                 yes              100 / 100 / 80 / 80 / 100    100         100       
│ F1-CH44  3.6         3.0                 yes              75 / 75 / 100 / 75 / 75      100         75        
│ F1-CH45  3.8         —                   no               100 / 100 / 100 / 100 / 100  100         100       
⚠ F1-CH42: all five finals carry 3.0 K and the best of them scores 75. The selector cannot deliver a final the pool does not contain
```

F1-CH42 is the clean case. Its approach (4.2 K) is above the copied 3.0 K limit and below the real 4.5 K, so every candidate diagnoses fouling, and all five finals score 75. The pool's best *is* 75. A wider pool (larger k) only adds more copies of the same mistake. On F1-CH40 and F1-CH44 the approach is close to the limit, so some finals land right by luck; the premise does not always cap the pool, it caps it where it decides the answer.

The remedy is a probe that challenges the **shared** assumption instead of comparing candidates with each other. `failkit.premise_probe` runs each candidate with and without its workspace fact and compares the fact with the task's own sources. It uses behaviour and visible sources only, no score:

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 5 · The remedy: a probe that challenges the SHARED assumption (no score, no hidden truth)
│ F1-CH42 · request: Alarm overnight: CH-1 high kW/ton. Review it and propose the next step.
│ the task's own source: "Chelsea Riverside Hotel (fictional) O&M: condenser approach above 4.5 K indicates fouling; design efficiency 0.68 kW/ton."
│ final  fact asserted  source says  diagnosis with fact  without fact       carries the decision  vs source
│ ─────  ─────────────  ───────────  ───────────────────  ─────────────────  ────────────────────  ─────────
│ #1     3.0 K          4.5 K        condenser fouling    condenser fouling  no                    FLAG     
│ #2     3.0 K          4.5 K        condenser fouling    low load cycling   yes                   FLAG     
│ #3     3.0 K          4.5 K        condenser fouling    low load cycling   yes                   FLAG     
│ #4     3.0 K          4.5 K        condenser fouling    low load cycling   yes                   FLAG     
│ #5     3.0 K          4.5 K        condenser fouling    condenser fouling  no                    FLAG     
◆ the fact contradicts the task's own manual in every candidate — a flag that no comparison BETWEEN candidates could raise, because they all agree
│ re-verified pool: limits 4.5 K · score per final* 100 / 100 / 75 / 100 / 100
✓ re-verifying the shared fact against the evidence in front of the editor lifts the cap
```

Two signals, both visible without the hidden truth: removing the fact flips the decision in three of five candidates, and the fact contradicts the manual excerpt the task itself carries. Re-verified, the pool's best is 100 again (one final still misses: that approach is within 1 K of the limit, a judgement call no harness removes). The course's EXAMPLE design for Ops Assistant goes one step further: workspace facts live in the knowledge graph per site, procedures in skills, and harness memory never authorises.

✓ Checkpoint: you can explain why a larger k does not fix a shared premise, and name two signals a premise probe uses that need no score.

## 3 · Attribution noise: one probe is a sample of one

The third failure hides in the edit rows. A probe is one execution of a stochastic agent. When one rollout is all that stands behind a verdict, the verdict can be luck. The reflector's prompt says what to do about it:

> “`confounded` -- and prefer it -- whenever a candidate changed several components at … once and no execution isolates one of them, or when a single rollout is all that … stands behind the difference.” — `harness2/core/prompts.py`

and why it matters:

> “A confounded row is a useful record; an invented … attribution poisons the library for every later task.” — `harness2/core/prompts.py`

Lab 07-3 runs the same step with one probe per candidate and with three (`repeats=1` and `repeats=3`, Module 04).

**Expected output** (captured on this Mac, SIM)

```
◆ F2-00 repeats=1 · probe dom1#0: digest_contract mitigated
⚖ F2-00 repeats=1 · skills/engineer_digest/SKILL.md, systemprompt. → helped
│ F2-00 repeats=1 → COMPOSED · 3 executions
◆ F2-00 repeats=3 · probe dom1#0: digest_contract mitigated
◆ F2-00 repeats=3 · probe dom1#1: digest_contract mitigated
◆ F2-00 repeats=3 · probe dom1#2: digest_contract mitigated, no_reopen regressed
⚖ F2-00 repeats=3 · skills/engineer_digest/SKILL.md, systemprompt. → hurt
│ F2-00 repeats=3 → KEEP_BASE · 4 executions
```

Read it carefully, because it cuts the other way from what you might expect. The digest skill fixed the missing section in all three probes. But the third probe also skipped reopening the file, a behaviour the skill never touched. The kit's composer counts that as a regression (its list of stochastic failures does not include `no_reopen`), so a working edit is dropped as "hurt". More probes surface more variance, and a composer can still blame it on the wrong edit. The other case in the lab (F1-01) goes the expected way: two edits that each looked "helped" on one probe become "confounded" on three, because the middle probe shows neither working.

Across the whole refine half the pattern is clear:

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 3 · Across all 17 refine tasks × 3 seeds × both executors: how verdicts move with the number of probes
│ repeats  steps  edit rows  helped  confounded  no obs. effect  hurt  KEEP_BASE  execs/step
│ ───────  ─────  ─────────  ──────  ──────────  ──────────────  ────  ─────────  ──────────
│ 1        102    101        66%     21%         12%             1%    50         2.3       
│ 3        102    101        73%     24%         1%              2%    49         4.0       
│ verdict with 1 probe  with 3 probes  edit rows
│ ────────────────────  ─────────────  ─────────
│ confounded            helped         17       
│ no observed effect    confounded     11       
│ helped                confounded     9        
│ helped                hurt           1        
◆ 38 of 101 paired edit rows change verdict when two more probes run — in both directions, so the totals above hide most of the movement. A verdict from one probe is a sample of one
```

More than a third of the rows (38 of 101) change verdict, and the totals barely show it. That is why the course's EXAMPLE design asks for repeated probes on high-stakes harness changes, and why a lesson that rests only on confounded rows is a guess. Repeats cost executions (2.3 → 4.0 per step here), and they are still the cheapest way to separate an edit from noise: no new candidate, no new editor session, just the same probe again.

The kit also flags noise inside a single step. `Candidate.noise()` lists a failure that appeared in a probe although no edit touched it, and the composer names it instead of blaming the edit:

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 4 · Candidate.noise(): a failure that appeared in a probe although no edit touched it
│ inv1: edits target hand_arithmetic · noise(): no_cmms
→ compose: inv1: no_cmms appeared in the probe but no edit touched it — execution variance? (repeated executions would tell)
✓ the composer names it as possible execution variance instead of blaming the edit; repeated executions would tell
```

✓ Checkpoint: you can say why a "helped" row from one probe is weak evidence, and give one example where more probes changed a verdict for the worse.

## 4 · A rubric leak, and the three controls around measurement

Harness² keeps measurement out of the loop: the judge runs only after selection commits, and no editor sees a score (Module 04). That only holds if the executor cannot reach the grading criteria either. The repository found out the hard way:

> “In a validation run of this release, a Codex rollout on JobBench downloaded its task's rubric without being instructed to.” — `docs/EXPERIMENTS.md`

Lab 07-3 Part 2 re-enacts that on Ops Assistant's digest task with a **module-local simulation** (`failkit.py`). The "web" is a Python dict with fictional URLs on the hosts the repository names; nothing touches the network. The simulated executor has a web tool, searches for its task unprompted, and if the bench's `RUBRICS.json` is reachable it writes to it. A stand-in model judge grades the text against the rubric's two dimensions.

**Expected output** (captured on this Mac, SIM)

```
│ seed  fetched (open)  claims 'verified'  reopened the file  rubric judge (open)  fetched (blocked)  rubric judge (blocked)  outcome checks* open / blocked
│ ────  ──────────────  ─────────────────  ─────────────────  ───────────────────  ─────────────────  ──────────────────────  ──────────────────────────────
│ 0     2               yes                yes                100.0                0                  50.0                    100 / 100                     
│ 1     2               yes                no                 50.0                 0                  0.0                     75 / 75                       
│ 2     2               yes                yes                100.0                0                  50.0                    100 / 100                     
│ 3     2               yes                no                 50.0                 0                  0.0                     75 / 75                       
◆ * bench.judge checks the product (sections, IDs) — it is the same with and without the leak
⚠ with the rubric in reach, every run adds 'verified against the source systems' — also on seeds 1 and 3, where it never reopened the file and a section is missing. A judge that grades the text rewards a claim, not an act
```

The leak raises the rubric judge by 50 points on every seed and changes nothing the outcome checks measure. That is the shape of the danger: scores go up, the work does not. Three controls close it.

**Control 1: block the hosts at the network level.** The repository's advice is specific about where:

> “apply restrictions at the network level, such as an egress firewall or DNS policy on the executor host, rather than relying on a tool setting.” — `docs/EXPERIMENTS.md`

It lists the hosts: `huggingface.co`, `hf.co` and `*.hf.co`, plus `github.com` and `raw.githubusercontent.com` for benchmark repositories, and asks you to report the restriction alongside results. In the course's EXAMPLE design for Ops Assistant this becomes a grader-isolation test: executor, editor and selector sandboxes try to read the graders, and the OpenShell policy (Week 26) denies it.

**Control 2: the measurement-word ban.** A visible prompt field must describe the work, not how it is measured. The repository's `spec.py` refuses a `BenchPrompts` field that contains benchmark, evaluation, grader, rubric or score, with the message that it

> “must describe the WORK, not its” — `harness2/core/spec.py`

(the message continues "measurement"). **Control 3: the grader-verdict check.** The library refuses a tree that carries grader verdicts before anything is stored; `library.py` checks the tree before it is written. The course's version, `harness.grader_verdicts()`, applies both checks to every tree file, which is stricter than the repository.

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 7 · Control 2 — the measurement-word ban on visible prompt text (the course's version of spec.py)
◆ harness2/core/spec.py refuses a visible field that does not describe the work: "must describe the WORK, not its measurement"
✓ accepted: "Morning digest for the chief engineer: open jobs, overnight alarms, today's PMs, with IDs."
✕ refused: "Morning digest for the chief engineer. It is scored by a rubric: include the word 'verified'." — found rubric, score

▣ STEP 8 · Control 3 — the grader-verdict check: no verdict or measurement language enters the library
✕ leaky tree refused: grader verdicts or measurement words in skills/engineer_digest/SKILL.md
✓ clean tree stored: F2-00-clean
```

The ✕ lines are the controls working: the refused field and the refused tree are what you want stopped. Week 10 taught you to grade the outcome, not the transcript's claims; this section is the other half of that rule: keep the grading criteria where neither the executor nor the editor can read them.

✓ Checkpoint: you can name the three controls (egress block, measurement-word ban, grader-verdict check) and say which one stops the fetch and which two stop grader language entering the harness.

## Labs — run them here

**labs/lab07_1_blind_selection.py** — Steps at k = 3 and k = 5 with every final judged after the fact: how often the blind selector delivers a weaker one, and the worst pick.

**labs/lab07_2_premise_caps_the_pool.py** — One wrong workspace fact copied into every candidate at the second hotel, a premise probe that flags it, and re-verification that lifts the cap.

**labs/lab07_3_noise_and_rubric_leak.py** — Edit-row verdicts with one probe versus three, Candidate.noise(), and a simulated rubric leak closed by an egress block, the word ban and the verdict check.

## Try it yourself

`exercises/ex07_diagnose_the_incident.py` has three TODOs:

1. Diagnose six short incidents: blind selection, premise caps the pool, attribution noise or rubric leak.
2. Name the control for each.
3. Write the egress rule `blocked(host)` for exactly the hosts `EXPERIMENTS.md` lists.

```bash
# on: laptop
.venv/bin/python week31_harness2_recursive_harnessing/07_failure_modes/exercises/ex07_diagnose_the_incident.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ six incidents diagnosed: 1 selection, 2 premise, 3 and 6 noise, 4 and 5 leak
✓ six controls named — none of them shows the loop a score
✓ egress rule right on all 9 hosts (cdn-lfs.hf.co blocked, nothf.co allowed)

═ Done. Module 08 puts these controls into Ops Assistant's nightly harness review, behind a human gate.
```

<details><summary>Hint — TODO 1</summary>

Ask four questions in order. Was a better answer in the pool? (selection) Did every candidate share the mistake? (premise) Does one rollout, or one rollout with several changed components, stand behind the verdict? (noise) Did grading criteria or grader language reach the executor or the harness? (leak)

</details>

<details><summary>Hint — TODO 3</summary>

`fnmatch.fnmatch("cdn-lfs.hf.co", "*.hf.co")` is True and `fnmatch.fnmatch("nothf.co", "*.hf.co")` is False. Match the host against each of the five patterns; do not use `endswith("hf.co")`.

</details>

✓ Checkpoint: all three checker lines are ✓.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Lab 07-2 shows 4.5 K in the unchecked rows | the second-hotel tasks appended to the same library, so a re-verified entry leaked in | the lab gives each task a fresh copy of the Harbour View library; keep that if you change it |
| Your selection counts differ from the tutorial | an edited kit, or a different seed range | `git diff week31_harness2_recursive_harnessing/common`; the lab uses seeds 0–2 |
| Lab 07-3 prints ✕ lines | steps 7 and 8 show the controls refusing bad input | those ✕ lines are the expected result; the lab itself exits 0 |
| You want to see the rubric leak hit a real host | you should not: the lab is a simulation by design | block the hosts on your own executors and report the restriction with your results |

## Next

[Lab 08 — Ops Assistant: the nightly harness review routine with a human gate](../08_nightly_harness_review/TUTORIAL.md): put the controls of this module into Ops Assistant's always-on runtime, with weak-signal task selection, a review packet and a named human who approves every version.
