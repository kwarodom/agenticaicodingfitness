# ▶ Harness Lab 02 — One Propose–Probe–Compose step, by hand and by loop

> Part of Week 31 · Harness²: agents that improve their own harness. You improve the harness of one simulated agent, the **HVAC Expert agent of Ops Assistant** (a fictional always-on operations assistant), from execution evidence, and you see the real output. Everything is synthetic (two fictional hotels, invented alarms and work orders), so every lab is free, offline and safe. Labs run on **🧠 SIM** by default: a deterministic teaching model of an agent and its editors. Flip the runner to **🦙 LIVE** where a lab offers it, and a real local model plays one role.

**What you'll actually do**
- Learn the six moves of one Harness² step, and which of them may read what.
- Re-enact the repository's offline demo in a short script of your own: a ledger, a script with a positive-only filter bug, an editor that removes the filter, a probe, a compose decision and an `EXPERIENCE.md` entry.
- Run one full step of the course's loop on task F1-01, the rejected chiller proposal from the course's worked example, and read every line it prints.
- Run a step on every quiet task and every trap task, count how often the honest answer is KEEP_BASE, and see why a trap task can still get one.
- Play the composer yourself: keep, drop or KEEP_BASE for five contrast tables.

**Time** ~45 min · **Difficulty** intermediate · **Hardware** this laptop only (no model needed)

**Sources:** [Harness² paper, arXiv 2610.14224](https://arxiv.org/html/2610.14224) · [google-research/harness2](https://github.com/google-research/harness2) (Apache-2.0; snapshots of `prompts.py` and `demo.py` in `common/sources/`) · the self-evolving agents of Week 18, the evals of Week 10, the prompt caching of Week 2

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Module 01 done | you can name C1–C8 and say which four are always on | every edit in this module lands in one of them |
| This repo's Python | `.venv/bin/python --version` → 3.13 | runs every lab |
| No model | — | all three labs are SIM; the editors here are rule tables |

```bash
# on: laptop
.venv/bin/python week31_harness2_recursive_harnessing/common/h2kit.py | tail -3
```

**Expected output** (captured on this Mac, SIM)

```
⚖ tool_guidance.md → helped
⚖ skills/chiller_fault_review/SKILL.md, systemprompt → helped
═ COMPOSED · selected composed · EXPERIENCE entry written · always-on 245 → 288 tokens · 4 executions, 4 editor sessions
```

That one line is a whole Harness² step. By the end of this module you can explain every word in it.

✓ Checkpoint: the kit's self-run prints a `COMPOSED` line with always-on 245 → 288 tokens.

## 1 · One step, six moves

Week 18 showed you agents that rewrite their own prompts. Harness² keeps that ambition but adds discipline: **every edit must come from something an execution showed.** One step of the paper has six moves: reference, propose, probe, compose, select and reflect, and each decision rests on a concrete observation, not on a hoped-for score gain.

| Move | What happens | What it may read |
|---|---|---|
| Reference | run the current harness once on the task | the task, the harness |
| Propose | an **INV** editor and a **DOM** editor each write one candidate harness | the trajectory, the deliverable, the request and sources, the library |
| Probe | run each candidate on the same task under the same model | — |
| Compose | merge the edits each probe supports, or answer KEEP_BASE | references, candidates, their probes, the diffs |
| Select | only when k > 1: pick one of k finals, blind | anonymised deliverables, the request, the sources |
| Reflect | write edit rows with verdicts and at most one `EXPERIENCE.md` entry | the whole record |

Two rules from the paper make this work without ground truth. The reference shows behaviour, not correctness, and the composer works from behavioural contrasts alone:

> “compose a refined harness from behavioral contrasts between executions, without ground-truth feedback” — `harness2/README.md`

And no move ever sees a score. In this course only `bench.judge` reads a task's hidden truth, and it runs after the step has committed. Module 04 explains why that boundary matters so much.

At k = 1 there is one reference, one candidate per scope, and no selection. That is the setting the course's EXAMPLE design uses for Ops Assistant's nightly routine, and the one every lab here uses.

✓ Checkpoint: you can name the six moves and say which one runs only when k > 1 (select).

## 2 · The repository's demo, re-enacted by hand

The Harness² repository ships a tiny offline demo. Its docstring says what it is for:

> “A deterministic local adapter for exercising the real adaptation workflow.” — `harness2/demo.py`

> “Scripted edits and selection; no model service or benchmark judge is used.” — `harness2/demo.py`, `DemoProposer`

Its whole world is a ledger CSV and a base harness with one script. The script has a bug that is easy to miss: it keeps only the positive amounts.

> “total = sum((amount for amount in amounts if amount > 0), Decimal(0))” — `harness2/demo.py`, `BASE_SCRIPT`

The scripted editor fixes it with a one-line replace, and the scripted reflection writes the edit row:

> “script.write_text(script.read_text().replace(" if amount > 0", ""), encoding="utf-8")” — `harness2/demo.py`

> “change: dropped the positive-only filter from the amount sum” — `harness2/demo.py`

> 📌 **This course does not run the demo.** Running the cloned repository was refused on this machine, so there is no captured demo output here. The facilitator guide asks learners to run `harness2-demo --output runs/demo --k 2` themselves, in their own clone, as an optional before-class step. Lab 02-1 re-enacts the same idea in its own code.

Lab 02-1 builds the same ledger, the same buggy script and the same request, then walks the loop. The editor's view is the request ("Sum every signed amount"), the source rows and the trajectory. It never sees an answer key.

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 2 · Reference run on ledger-1: evidence of BEHAVIOUR, not of correctness
│ [ACTION] Run scripts/summarize.py on ledger.csv
│ [OBSERVATION] Total: 105.25
◆ observed: credits_omitted: the source has 1 negative row(s); the delivered total equals the positive rows only

▣ STEP 3 · Propose: an INV editor edits the script (the demo's scripted editor does the same one-line replace)
✎ inv1 scripts/summarize.py (C7) modified — removed the positive-only filter
│ - total = sum((amount for amount in amounts if amount > 0), Decimal(0))
│ + total = sum((amount for amount in amounts), Decimal(0))

▣ STEP 4 · Probe the candidate on the SAME ledger and contrast it with the reference
│ reference: Total: 105.25   probe: Total: 85.15
◆ contrast: credits_omitted mitigated

▣ STEP 5 · Repeat the contrast on a second ledger (a repeated execution separates an edit from luck)
│ ledger-2 reference: Total: 35.50   probe: Total: 20.25
◆ contrast: credits_omitted mitigated

▣ STEP 6 · Compose: keep an edit only if its probe mitigated the failure, delivered, and regressed nothing
→ compose: keep scripts/summarize.py — credits_omitted mitigated on 2/2 ledgers, nothing regressed
```

Three things to notice.

1. **The fix is a script edit (C7), not prose.** The always-on text stays 14 tokens and the prefix hash does not move, so the prompt cache from Week 2 stays warm.
2. **The contrast is behavioural.** "The delivered total equals the positive rows only" is something you can see by reading the source next to the trajectory. The demo's own scripted selector checks the same thing against the source ledger.
3. **A third ledger with no credits gives no evidence.** The bug is still in the script, but the trajectory cannot show it, so the honest answer is KEEP_BASE and `NO_ENTRY`.

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 8 · The same loop on ledger-3, which has no credits: nothing to see, so KEEP_BASE
│ ledger-3 reference: Total: 20.50
◆ observed: nothing — the filter bug is real but this trajectory gives no evidence of it
→ compose: KEEP_BASE — the evidence does not argue for any change
✓ NO_ENTRY written: no edit rows, no lesson — the loop only fixes what an execution shows
```

✓ Checkpoint: you can say why ledger-3 gets KEEP_BASE even though its script still has the bug.

## 3 · One step on F1-01, the course's worked example

The course's EXAMPLE design follows one rejected proposal through the loop. A chief engineer rejected a chiller work order with the reason "CH-2 was already on a PM; the agent did not check open work orders." Task F1-01 is that case in the simulator. Lab 02-2 runs `step(task("F1-01"), base_harness())` and prints everything with `rec.show(trajectories=True)`.

**Expected output** (captured on this Mac, SIM)

```
▣ task F1-01 · fault_review · signal rejected_proposal · reviewer: "CH-2 was already on a PM; the agent did not check open work orders."
◆ reference 0: delivered=True · observed: hand_arithmetic, no_cmms, no_analytic_move
┌─ run F1-01 · sim-cloud · seed 0 · harness 773bf3d26eee · prefix ee3bb0d97ccb
│ [ACTION] alarms.list(window='overnight')
│ [OBSERVATION] AL-901 CH-2.kw_per_ton: high kW/ton
│ [ACTION] telemetry.query('CH-2.kw', days=1)
│ [OBSERVATION] 6 hourly rows (kw, tons)
│ [ACTION] docs.search('CH-2 high kW/ton')
│ [OBSERVATION] Harbour View Hotel (fictional) O&M: condenser approach above 3.0 K indicates fouling; design efficiency 0.62 kW/ton.
│ [REASONING] summed kW and tons by hand across six rows
│ [DELIVER] send_to_user(outcome=done)
│ ── deliverable ──
│ diagnosis: condenser fouling
│ kw_per_ton: 0.201
│ proposal: {"type": "ActionProposal", "action": "create work order: CH-2 condenser fouling", "approver": "chief engineer", "evidence": ["AL-901", "CH-2.kw telemetry"]}
└─
✎ inv1 [inv] proposed equipment work without calling cmms.work_orders first → tool_guidance.md
◆ probe inv1#0: hand_arithmetic unchanged, no_analytic_move mitigated, no_cmms mitigated
✎ dom1 [dom] diagnosed without comparing the condenser approach with the manual's limit → skills/chiller_fault_review/SKILL.md, systemprompt.md
◆ probe dom1#0: hand_arithmetic unchanged, no_analytic_move mitigated, no_cmms mitigated
→ compose: inv1: keep tool_guidance.md — no_cmms mitigated in its probe
→ compose: dom1: keep skills/chiller_fault_review/SKILL.md, systemprompt.md — no_analytic_move mitigated in its probe
```

Read it top to bottom, the way the composer does.

- **Reference.** The trajectory has no `cmms.work_orders` call, so the agent never saw the open PM. It also summed kW and tons by hand. Its kW/ton of 0.201 is wrong (it divided by 3.517 once too often), but no editor can know that: there is no answer key. What the editor *can* see is the line `summed kW and tons by hand`.
- **Propose.** The INV editor owns how the agent works, so it adds one tool-guidance line: check work orders before any equipment proposal. The DOM editor owns what a practitioner knows, so it adds the `chiller_fault_review` skill plus one trigger line in `systemprompt.md`.
- **Probe.** Both probes now call `cmms.work_orders`, as in the EXAMPLE walk-through, where both candidates add the recent-maintenance check. The skill names the maintenance check too.
- **Compose.** Each edit's own failure was mitigated in its own probe and nothing regressed, so both are kept.

Two lines deserve a second look. `hand_arithmetic unchanged` appears because **no editor targeted it**: the SIM editor misses a failure 25% of the time, and this was one of those times. "Unchanged" is not "regressed", so it does not block the merge. The failure simply stays unfixed. And `inv1` "mitigated" `no_analytic_move` without touching it: once the agent finds the open work order it proposes nothing new, and that observation only fires on a proposal. A contrast records behaviour, not cause. Module 04 turns that into the `confounded` verdict.

Then the lab puts the EXAMPLE walk-through and the SIM step side by side.

**Expected output** (captured on this Mac, SIM)

```
│ stage          EXAMPLE walk-through                         this lab (SIM, seed 0)                          
│ ─────────────  ───────────────────────────────────────────  ────────────────────────────────────────────────
│ reference      no CMMS call                                 no cmms.work_orders call; kW/ton summed by hand 
│ INV candidate  guidance line + scripts/open_work_orders.py  guidance line only (missed hand arithmetic)     
│ DOM candidate  skills/chiller_fault_review                  skills/chiller_fault_review + trigger line      
│ probes         both add the maintenance check               both call cmms.work_orders                      
│ composer       merges both                                  keeps both                                      
│ cue            maintenance state can contradict the alarm   a fault review where the practitioner's compari…
│ always-on Δ    +0 (a sentence was replaced)                 +43 (the kit only appends)                      
```

The stages match. The details do not, and that is fine: the walk-through is EXAMPLE design, the lab is a SIM run. The kit's editors only append lines, so the always-on text grows by 43 tokens here, while the EXAMPLE editor paid for its new line by replacing an old one. The prompt's rule "if you add always-on text, cut something else" is the one the kit does not model.

✓ Checkpoint: you can point at the trajectory line each kept edit answers, and say why `hand_arithmetic unchanged` did not stop the merge.

## 4 · Edit rows and the EXPERIENCE.md entry

After the step, reflection writes two records. First, one **edit row** per edit, with a verdict: `helped`, `hurt`, `no observed effect` or `confounded`. Then **at most one** lesson:

> “at most one durable entry, and only if the edit rows support it” — `harness2/core/prompts.py`

The entry has four fields. The `cue` names "the SHAPE OF SITUATION the lesson applies to, so it can fire on a different matter" (`prompts.py`). The `lesson` names the operation and the component. The `evidence` cites the edit rows. The `confidence` is high, medium or low.

**Expected output** (captured on this Mac, SIM)

```
┌─ EXPERIENCE.md
│ <edits>
│ component: tool_guidance.md
│ candidates: inv1
│ change: proposed equipment work without calling cmms.work_orders first → tool_guidance.md
│ observed: no_cmms: mitigated
│ verdict: helped
│ 
│ component: skills/chiller_fault_review/SKILL.md, systemprompt.md
│ candidates: dom1
│ change: diagnosed without comparing the condenser approach with the manual's limit → skills/chiller_fault_review/SKILL.md, systemprompt.md
│ observed: no_analytic_move: mitigated
│ verdict: helped
│ 
│ </edits>
│ 
│ <experience>
│ cue: a fault review where the practitioner's comparison is skipped
│ lesson: add a skill naming the analytic move, the artifact contract and the maintenance check
│ evidence: edit row 'skills/chiller_fault_review/SKILL.md, systemprompt.md' (dom1): no_analytic_move: mitigated; reviewer's reason (content): "CH-2 was already on a PM; the agent did not check open work orders."
│ confidence: medium (one task, several rows agree)
│ </experience>
└─
✓ wrote 02_propose_probe_compose/.runs/lab02_2/EXPERIENCE.md — no grader verdict or measurement word in it
```

- The kit chose the skill row for the lesson because it prefers a named component (C6, C7, C8) over always-on text. The EXAMPLE walk-through chose the maintenance-check cue instead. Both rest on a `helped` row.
- The reviewer's reason appears as **content**, quoted, never as an instruction. Module 08 builds on that firewall.
- Confidence is "medium" because it is one task. The kit always writes medium. Lab 02-1 wrote "high" by hand, because a script is exact and the contrast repeated on two ledgers. Module 05 reads the whole contract.
- The paper's caution applies to every entry: "Adoption alone never establishes that an edit helped or will transfer." (`prompts.py`)

Only at the very end does the lab ask `bench.judge`: the reference scored 40 and the composed run 100. One task and one seed is a demonstration, not a measurement.

✓ Checkpoint: you can name the four `EXPERIENCE.md` fields and say what decides between an entry and `NO_ENTRY`.

## 5 · KEEP_BASE is a real answer

The repository tells its composer, in so many words, that doing nothing can be right:

> “KEEP_BASE is a real answer, not a failure. Adaptation that the evidence does not argue for is a regression: it costs context and buys nothing.” — `harness2/core/prompts.py`

Lab 02-3 tests whether the course's loop obeys that. It runs one step on every **quiet** task of the refine half (no trap: the base harness already handles it) and every **trap** task, for three seeds and both executors. The loop never reads which is which; `quiet()` is kit metadata for this count.

**Expected output** (captured on this Mac, SIM)

```
│ executor (+ its editor)  seed  KEEP_BASE on quiet  KEEP_BASE on trap
│ ───────────────────────  ────  ──────────────────  ─────────────────
│ sim-cloud                0     5/5                 2/12             
│ sim-cloud                1     5/5                 3/12             
│ sim-cloud                2     5/5                 4/12             
│ sim-local                0     5/5                 7/12             
│ sim-local                1     5/5                 4/12             
│ sim-local                2     5/5                 2/12             
◆ KEEP_BASE rate: quiet 30/30 = 100% · trap 22/72 = 31%
✓ on quiet tasks the loop never edits: no observation, no proposal, no new version
```

Quiet tasks get KEEP_BASE every time: nothing in the trajectory is wrong, so no editor proposes anything. But 31% of the **trap** steps got KEEP_BASE too. The lab asks why.

**Expected output** (captured on this Mac, SIM)

```
│ why a trap task got KEEP_BASE (3 seeds)  sim-cloud  sim-local
│ ───────────────────────────────────────  ─────────  ─────────
│ reference showed nothing                 7          5        
│ editors proposed nothing                 1          6        
│ edits proposed, none supported           1          2        
```

- **Reference showed nothing.** On F1-00 and F1-03 at seed 0, the agent happened to call `cmms.work_orders` (sim-cloud does that 45% of the time on v1). One reference is one draw. The weakness is still there; it just did not show.
- **Editors proposed nothing.** The editor missed what the trajectory showed. The weaker sim-local editor misses more often (40% against 25%).
- **None supported.** A probe changed nothing or regressed something, so the composer dropped the edit.

So KEEP_BASE answers "did *this* evidence argue for a change?", not "is the harness perfect?". More references (k > 1) or repeated executions would surface the hidden cases. Modules 04 and 06 measure that.

The lab ends by forcing an unsupported skill onto quiet task F1-10. The composer drops it, and the table shows what shipping it would have cost:

**Expected output** (captured on this Mac, SIM)

```
◆ reference observed: nothing wrong · probe contrast: no difference
→ compose: dom1: drop skills/chiller_fault_review/SKILL.md, systemprompt.md — no_analytic_move: no observed change
→ compose: KEEP_BASE — the evidence does not argue for any change
│ outcome                         always-on ≈tokens  prefix hash   files
│ ──────────────────────────────  ─────────────────  ────────────  ─────
│ KEEP_BASE (composer)            245                ee3bb0d97ccb  4    
│ if the forced edit had shipped  264                efa3d6dfa285  5    
✓ the composer dropped it: an edit whose probe changed nothing buys nothing, and would still cost +19 always-on tokens and one cold prompt cache
```

For an always-on agent like Ops Assistant this is the difference between a quiet night and a noisy one. In the course's EXAMPLE design a KEEP_BASE result produces no draft version and no notification. A loop that always edits would page a reviewer every night and break the prompt cache for nothing.

✓ Checkpoint: you can give two different reasons a trap task can end in KEEP_BASE, and say why neither is a bug in KEEP_BASE.

## Labs — run them here

**labs/lab02_1_the_demo_by_hand.py** — The repository demo's idea re-enacted in your own code: ledger, filter bug, reference, edit, probe, compose, EXPERIENCE.md, and a KEEP_BASE ledger.

**labs/lab02_2_one_step_on_f1_01.py** — One full k = 1 step on F1-01 with every trajectory shown, the composer's evidence, the edit rows, and EXPERIENCE.md written under .runs/.

**labs/lab02_3_keep_base_honesty.py** — A step on every quiet and every trap task (three seeds, two executors): the KEEP_BASE rate, why trap tasks get it, and what a forced edit would cost.

## Try it yourself

`exercises/ex02_compose_by_hand.py` gives you five contrast tables and makes you the composer. It has three TODOs:

1. For every candidate, write `("keep" | "drop", reason)`.
2. For every case, write the step's outcome: `COMPOSED` or `KEEP_BASE`.
3. In case D, name the regressed failure the composer only *notes* as possible execution variance.

```bash
# on: laptop
.venv/bin/python week31_harness2_recursive_harnessing/02_propose_probe_compose/exercises/ex02_compose_by_hand.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ every keep/drop decision and its reason match the composer's rule
✓ outcomes right: B and E are KEEP_BASE — no edit earned its place
✓ no_evidence in D is noted as possible variance: stochastic, and no edit touched it

═ Done. Module 03 asks WHERE each kept edit should live: the cheapest effective component.
```

<details><summary>Hint — TODO 1</summary>

Check three things, in order. Did the probe deliver? If not: drop, "not delivered" (the repository's composer prompt: "Never merge an edit from a non-delivering rollout without stating why"). Is the edit's *own* target "mitigated"? If not: drop, "no observed change". Did the candidate regress a targeted or a deterministic failure (hand_arithmetic, no_reopen, blind_retry, no_deliverable)? If so: drop, "regressed". Otherwise: keep, "mitigated".

</details>

<details><summary>Hint — TODO 3</summary>

Case D has two "regressed" entries. One is `hand_arithmetic`, a deterministic failure: if it appears, something changed. The other is a stochastic failure that no edit in that candidate targeted. One probe cannot tell an edit's effect from luck, so the composer writes a note and keeps the edit. Module 04 shows how repeated executions settle it.

</details>

Want more? Re-run Lab 02-2 with `seed=2` in the `h2step(...)` call. The INV editor then catches the hand arithmetic too and adds `scripts/energy_units.py`, and one edit row comes back `confounded`. Read why before Module 04 explains it.

✓ Checkpoint: all three checker lines are ✓.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `harness2-demo: command not found` | the course never installs the repository | run the demo in your own clone, as the facilitator guide describes; nothing in this module needs it |
| Lab 02-1 fails with `FileNotFoundError` on `.runs/` | the module folder is read-only | run from a writable checkout; the lab writes only under `02_propose_probe_compose/.runs/` |
| A KEEP_BASE count differs from this page | an edited kit or a different seed | `git diff week31_harness2_recursive_harnessing/common`; same tree + task + seed gives the same step |

## Next

[Lab 03 — Two scopes and the cheapest effective component](../03_two_scopes_cheapest_component/TUTORIAL.md): see why the loop runs two editors instead of one, and why each reaches for a skill, a script or a plugin before it writes a line of always-on prose.
