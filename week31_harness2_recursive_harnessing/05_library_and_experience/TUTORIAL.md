# ▶ Harness Lab 05 — The improvement library: EXPERIENCE.md, NO_ENTRY and re-verifying facts

> Part of Week 31 · Harness²: agents that improve their own harness. You improve the harness of one simulated agent, the **HVAC Expert agent of Ops Assistant** (a fictional always-on operations assistant), from execution evidence, and you see the real output. Everything is synthetic (two fictional hotels, invented alarms and work orders), so every lab is free, offline and safe. Labs run on **🧠 SIM** by default: a deterministic teaching model of an agent and its editors. Flip the runner to **🦙 LIVE** where a lab offers it, and a real local model plays one role.

**What you'll actually do**
- Read the reflection contract: edit rows first, then at most one lesson with a cue, a lesson, evidence and a confidence, or the single line NO_ENTRY.
- Build a library over one domain's tasks in order, then open it the way the next task does: INDEX.md, one EXPERIENCE.md, the whole stored tree. In LIVE, a local model writes two entries and simple rules check them.
- Run the library ablation on both executors, and read the difference against the standard deviation instead of against your hopes.
- Take a library from the first hotel to a second one, with and without re-verifying the facts it carries, and watch a 3.0 K threshold mislead a 4.5 K hotel.
- Decide where a workspace fact belongs in Ops Assistant: the knowledge graph, not `LongTermMEMORY.md`.

**Time** ~40 min · **Difficulty** intermediate · **Hardware** this laptop only (Ollama with `gemma4:12b` optional, for the LIVE step of Lab 05-1)

**Sources:** [Harness² paper, arXiv 2610.14224](https://arxiv.org/html/2610.14224) · [google-research/harness2](https://github.com/google-research/harness2) (Apache-2.0; snapshots of `prompts.py`, `library.py`, `docs/EXPERIMENTS.md` in `common/sources/`) · the agent memory and knowledge graphs of Weeks 14–15, the self-evolving agents of Week 18, the evals of Week 10

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Module 04 | you can say what `helped` and `confounded` claim | the library stores those rows and builds lessons on them |
| This repo's Python | `.venv/bin/python --version` → 3.13 | runs every lab |
| Disk space | a few hundred KB under `05_library_and_experience/.runs/` | the labs write real library folders you can open (gitignored) |
| Ollama (optional) | `ollama list` shows `gemma4:12b` | only for the LIVE step of Lab 05-1; every lab finishes without it |

Module 02 ran one step and threw the evidence away. Harness² keeps it. After each task, reflection writes what each edit did and, at most, one lesson. The next task in the same domain reads that library before it edits anything. Week 18's self-evolving agents kept a memory of past runs too; the difference here is that every entry is a whole harness tree with its evidence attached, and the rules for writing an entry are strict.

✓ Checkpoint: you can say what the library stores per task (the whole adopted tree plus EXPERIENCE.md) and when the next task reads it (before its own editors start).

## 1 · The reflection contract: rows first, then at most one lesson

Reflection writes two records. The **edit rows** (Module 04) say what each change did. Then comes the part a later task actually learns from:

**Expected output** (REFERENCE — quoted from `harness2/core/prompts.py`, the reflection prompt)

```
"Now write at most one durable entry, and only if the edit rows support it.\n\n"
"The `cue` describes the SHAPE OF SITUATION the lesson applies to, so it can fire on a "
"different matter. Not this task's facts -- the pattern. The `lesson` is the operation "
"to perform on a harness, stated so it can be applied and checked, and it must name "
"the component it concerns.\n\n"
"Too generic is useless ('be thorough'); too specific never fires again ('the fee rose "
"to 250 on Jan 1'). Aim between: 'when an instrument performs several sequential acts "
"and one of them terminates the actor's own authority, order them so the terminating "
"act comes last'.\n\n"
"The `evidence` field cites the edit rows the lesson rests on. A lesson resting only "
"on `confounded` rows is a guess: narrow it to what the record does show, or decline. "
"Adoption alone never establishes that an edit helped or will transfer. If the record "
"shows no concrete, reusable execution contrast, write the single line NO_ENTRY in "
"place of the <experience> block -- the edit rows are still worth writing.",
```

So an entry has four fields:

| field | what it holds | wrong when |
|---|---|---|
| `cue` | the shape of situation the lesson applies to | it names this task's IDs or numbers, or says "be careful" |
| `lesson` | an operation on the harness that names the component | it gives advice to the agent instead of a change to a file |
| `evidence` | the edit rows and the execution contrast | it cites "the version was adopted" |
| `confidence` | high, medium or low, and why | one task stands behind a "high" |

Two lines deserve a second read. "Adoption alone never establishes that an edit helped or will transfer": the selector picked a deliverable, not an edit. And NO_ENTRY is a real answer, exactly like KEEP_BASE in Module 02.

A note on the kit: `h2kit.experience()` always writes confidence **medium**, because every entry rests on one task. The contract allows high, medium and low. The kit never exercises the other two, so do not read anything into the fact that every entry you see says "medium".

✓ Checkpoint: you can name the four fields, and say when the honest entry is the single line NO_ENTRY.

## 2 · A library over one stream: tasks in order

Lab 05-1 runs one step per fault-review task of the refine half, in order, with one library for the domain `hvac_expert/fault_review/hotel_th/sim-cloud` (agent × task family × tenant class × executor). The repository's operating notes set the order rule:

> “Tasks within a domain run in order and read only earlier entries; different domains can run concurrently.” — `harness2/docs/EXPERIMENTS.md`

**Expected output** (captured on this Mac, SIM)

```
│ task   entries read  adapted from the library           step       EXPERIENCE  confidence                           
│ ─────  ────────────  ─────────────────────────────────  ─────────  ──────────  ─────────────────────────────────────
│ F1-00  0             —                                  KEEP_BASE  NO_ENTRY    —                                    
│ F1-01  1             —                                  COMPOSED   entry       medium (one task, several rows agree)
│ F1-02  2             no_analytic_move                   COMPOSED   entry       medium (one task, several rows agree)
│ F1-03  3             —                                  KEEP_BASE  NO_ENTRY    —                                    
│ F1-04  4             hand_arithmetic, no_analytic_move  COMPOSED   entry       medium (one task, several rows agree)
│ F1-05  5             hand_arithmetic, no_analytic_move  COMPOSED   entry       medium (one task, several rows agree)
│ F1-10  6             —                                  KEEP_BASE  NO_ENTRY    —                                    
│ F1-11  7             —                                  KEEP_BASE  NO_ENTRY    —                                    
◆ a quiet trend check (F1-10, F1-11) has nothing to fix: KEEP_BASE and NO_ENTRY — the edit rows are still stored
⚠ the kit always writes confidence 'medium' (one task). The contract allows high | medium | low; the kit never exercises the other two
```

Watch the third column grow. F1-01 learns the fault-review skill. F1-02 adapts it from the library instead of writing one from scratch. By F1-04 the editor adapts two proven components. Half the entries are NO_ENTRY, and that is fine: the quiet trend checks had nothing to teach.

The library is then browsed through its index, which is one line per task:

**Expected output** (captured on this Mac, SIM)

```
┌─ lab05_1_library/hvac_expert__fault_review__hotel_th__sim-cloud/INDEX.md
│ # Library hvac_expert/fault_review/hotel_th/sim-cloud
│ 
│ - F1-00: NO_ENTRY (edit rows only)
│ - F1-01: a fault review where the practitioner's comparison is skipped
│ - F1-02: a deliverable that needs repeated unit arithmetic over telemetry rows
│ - F1-03: NO_ENTRY (edit rows only)
│ - F1-04: a deliverable that needs repeated unit arithmetic over telemetry rows
│ - F1-05: a deliverable that needs repeated unit arithmetic over telemetry rows
│ - F1-10: NO_ENTRY (edit rows only)
│ - F1-11: NO_ENTRY (edit rows only)
└─
```

Each line shows the entry's **cue**, not its task. That is the point of a cue: a later task with a different chiller, a different hotel, or a different alarm can still recognise the situation.

✓ Checkpoint: you can explain why F1-02 shows "adapted from the library" and F1-01 does not.

## 3 · Inside one entry: EXPERIENCE.md and the whole tree

Lab 05-1 opens the F1-01 entry, then a NO_ENTRY one:

**Expected output** (captured on this Mac, SIM)

```
┌─ harnesses/F1-01/EXPERIENCE.md
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
┌─ harnesses/F1-00/EXPERIENCE.md (no reusable contrast)
│ <edits>
│ </edits>
│ 
│ NO_ENTRY
└─
```

Check the entry against the contract. The cue is a shape ("the practitioner's comparison is skipped"), with no chiller tag and no number. The lesson names a component (a skill) and says what goes in it. The evidence cites a row. The reviewer's words are kept, and marked as content, so they inform and never instruct (Module 04).

The entry's folder holds more than the lesson. The repository's library stores the full tree:

> “The whole tree is carried, never a summary.” — `harness2/core/library.py`

**Expected output** (captured on this Mac, SIM)

```
│ file in harnesses/F1-01/harness/      comp.  vs base       ≈tokens
│ ────────────────────────────────────  ─────  ────────────  ───────
│ LongTermMEMORY.md                     C3     same as base  36     
│ guardrails.md                         C2     same as base  80     
│ skills/chiller_fault_review/SKILL.md  C6     added         209    
│ systemprompt.md                       C1     changed       86     
│ tool_guidance.md                      C4     changed       86     
```

Why the whole tree? A summary loses what made the edit work: the trigger line that names the skill, the script a skill invokes, the guidance line next to it. A later editor can copy a component together with everything it needs, and Module 02's validator throws away a skill whose script was left behind.

**LIVE: can a local model write the entry?** In 🦙 LIVE, step 6 of Lab 05-1 gives a local model the contract and the edit rows of two real records: F1-01 (two `helped` rows) and F3-00 (one `confounded` row, from a step with three probes). Simple rules like Exercise 05's check what comes back (2 calls).

**Expected output** (captured on this Mac, LIVE gemma4:12b — a rerun can differ)

```
▣ STEP 6 · LIVE option: does a local model write a lesson the contract accepts — and decline when it should?
── F1-01 (two helped rows) · gemma4:12b · 6.0s → entry
│ <experience>
│ cue: performing equipment work without checking the CMMS system first
│ lesson: tool_guidance.md
│ evidence: proposed equipment work without calling cmms.work_orders first; observed: no_cmms: mitigated
│ confidence: high
│ </experience>
✕ F1-01 (two helped rows): expected entry; rule problems: lesson states no operation (add/move/split…), 'high' on one task (one run — a rerun can differ)
── F3-00 (one confounded row) · gemma4:12b · 2.6s → NO_ENTRY
│ NO_ENTRY
✓ F3-00 (one confounded row): NO_ENTRY, as the record supports
```

In this run `gemma4:12b` declined correctly on the confounded record. On F1-01 its cue is a fair shape, but its "lesson" is only a file name, and it claims high confidence from one task. A library entry like that would tell the next editor where to look but not what to do, and with more certainty than the record holds. That is one reason the course's EXAMPLE design for Ops Assistant puts a named human reviewer in front of every harness version (Module 08).

✓ Checkpoint: you can say why a stored summary ("added a fault-review skill") would be less useful to the next task than the stored tree, and name the two contract rules the LIVE F1-01 entry broke.

## 4 · Reading the library: INDEX → entry → adapt

A later editor does not read every file. The repository's editor prompt tells it how to browse:

> “Browse `INDEX.md` first, then open the ones that look relevant: reusing or adapting a proven component is encouraged when it genuinely fits, but never presume this task's question.” — `harness2/core/prompts.py`

`Library.relevant(task, tree)` is the kit's version. It reads INDEX.md, skips NO_ENTRY lines, opens the entries whose cue matches a failure kind, and hands the editor the components those entries added:

**Expected output** (captured on this Mac, SIM)

```
│ hand_arithmetic    → scripts/energy_units.py, skills/chiller_fault_review/SKILL.md, systemprompt.md, tool_guidance.md
│ no_analytic_move   → skills/chiller_fault_review/SKILL.md, systemprompt.md, tool_guidance.md
│ workspace fact inside a copied skill: - In this workspace, the condenser approach limit is 3.0 K (from the O&M manual excerpt).
✓ 2 failure kinds have a proven component waiting — the editor adapts it instead of writing from scratch
```

Look at the last line of components. The copied skill carries a **workspace fact**: Harbour View's condenser approach limit, 3.0 K. The repository asks editors to write facts like this into the harness, anchored with "in this workspace":

> “VERIFIED FACTS ABOUT THIS WORKSPACE ARE THE MOST VALUABLE THING YOU CAN ENCODE” — `harness2/core/prompts.py`

That is right for the hotel where it was verified. Section 6 shows what it does at the next hotel.

✓ Checkpoint: you can say which file of the F1-02 entry carries the 3.0 K fact, even though F1-02's own cue is about unit arithmetic.

## 5 · One library per domain and per executor, and the ablation

Lab 05-2 runs one stream with the library on for each executor, then lists the libraries that exist afterwards:

**Expected output** (captured on this Mac, SIM)

```
│ library (agent/family/tenant class/executor)  tasks stored  with a lesson
│ ────────────────────────────────────────────  ────────────  ─────────────
│ hvac_expert/digest/hotel_th/sim-cloud         3             3            
│ hvac_expert/digest/hotel_th/sim-local         3             3            
│ hvac_expert/fault_review/hotel_th/sim-cloud   8             4            
│ hvac_expert/fault_review/hotel_th/sim-local   8             4            
│ hvac_expert/helper/hotel_th/sim-cloud         2             1            
│ hvac_expert/helper/hotel_th/sim-local         2             0            
│ hvac_expert/triage/hotel_th/sim-cloud         4             2            
│ hvac_expert/triage/hotel_th/sim-local         4             1            
│ EXPERIMENTS.md: “Libraries are separated by benchmark, executor, mode, budget, and seed.”
```

Eight libraries for two executors and four task families. A `sim-local` task never reads a lesson that `sim-cloud` learned. A lesson is a claim about what worked for one model, and the paper found transfer between models real but uneven (Module 06 measures it).

Does the library help? The ablation runs three streams, each with its own shuffled task order, with the library on and off:

**Expected output** (captured on this Mac, SIM)

```
│ executor   library  stream means (s0 / s1 / s2)  mean ± SD                       
│ ─────────  ───────  ───────────────────────────  ──────────  ────────────────────
│ sim-cloud  on       96.5 / 95.0 / 91.5           94.3 ± 2.6  ███████████████████░
│ sim-cloud  off      97.6 / 93.5 / 90.3           93.8 ± 3.7  ███████████████████░
│ sim-local  on       84.4 / 83.3 / 83.6           83.8 ± 0.6  █████████████████░░░
│ sim-local  off      84.4 / 82.2 / 85.1           83.9 ± 1.5  █████████████████░░░

▣ STEP 3 · Read it honestly: the difference, the SD, and whether all three streams agree
│ facilitator guide: “SDs are 1–3 points: believe differences under ~3 only if all three streams agree.”
│ executor   on − off  SD on / off  per stream (same order)  all agree  size         
│ ─────────  ────────  ───────────  ───────────────────────  ─────────  ─────────────
│ sim-cloud  +0.5      2.6 / 3.7    -1.1 / +1.5 / +1.2       no         within one SD
│ sim-local  -0.1      0.6 / 1.5    +0.0 / +1.1 / -1.5       no         within one SD
⚠ sim-cloud: the streams disagree in sign (-1.1 / +1.5 / +1.2) — no library effect is shown on 17 tasks
⚠ sim-local: the streams disagree in sign (+0.0 / +1.1 / -1.5) — no library effect is shown on 17 tasks
◆ neither difference is a result you could act on. Module 06 reads tables like this; Module 09 pre-registers the test.
```

The honest reading: neither executor shows a library effect. On `sim-cloud` the gap (+0.5) is inside one SD and the streams disagree in sign. On `sim-local` the two arms are level (−0.1): one stream ties, one favours the library and one favours running without it. The same stream seed means the same task order, so the per-stream column is a paired comparison, and here it points nowhere. Compare the paper: without the library the method loses 3.1 points on WorkBuddy-Bench and 1.8 on JobBench, and library provenance appears in 22% to 47% of modified files. The SIM shows how a library is used. It does not reproduce those numbers, and it should not be read as if it did.

✓ Checkpoint: you can say, for each executor, whether this table shows a library effect, using the SD and the per-stream column.

## 6 · The second hotel: re-verify what you copy

The tenant class `hotel_th` has a second fictional hotel, Chelsea Riverside Hotel. Its O&M manual puts the condenser approach limit at 4.5 K, not 3.0 K. Lab 05-3 builds the fault-review library at the first hotel, then runs the second hotel's six fault reviews twice from copies of it: once re-verifying copied facts against the task's own manual (`reverify=True`), once copying them unchecked (`reverify=False`).

The repository's rule for copying from the library:

> “If you want a component from one, COPY IT into `context/harness/` -- and when you do, RE-VERIFY its workspace facts against the evidence in front of you: an earlier task's mapping or threshold is a hypothesis here, not a fact.” — `harness2/core/prompts.py`

**Expected output** (captured on this Mac, SIM)

```
│ task     approach K  truth*             re-verified: fact  diagnosis          score*  unchecked: fact  diagnosis          score*
│ ───────  ──────────  ─────────────────  ─────────────────  ─────────────────  ──────  ───────────────  ─────────────────  ──────
│ F1-CH40  3.4         low load cycling   4.5 K              low load cycling   100.0   3.0 K            condenser fouling  80.0  
│ F1-CH41  3.9         within design      —                  within design      100.0   —                within design      100.0 
│ F1-CH42  4.2         low load cycling   4.5 K              low load cycling   100.0   3.0 K            condenser fouling  75.0  
│ F1-CH43  5.9         condenser fouling  4.5 K              condenser fouling  100.0   3.0 K            condenser fouling  100.0 
│ F1-CH44  3.6         low load cycling   —                  low load cycling   100.0   —                low load cycling   100.0 
│ F1-CH45  3.8         within design      —                  within design      100.0   —                within design      100.0 
◆ * truth and scores measured AFTER each step by bench.judge — no editor saw them
═ seed 0 · re-verified mean 100.0 · unchecked mean 92.5

▣ STEP 3 · What the editor copied, and what re-verification changed
│ copied from the Harbour View entry : - In this workspace, the condenser approach limit is 3.0 K (from the O&M manual excerpt).
│ after reverify_facts on F1-CH40    : - In this workspace, the condenser approach limit is 4.5 K (from the O&M manual excerpt).
│ F1-CH40's own manual excerpt       : Chelsea Riverside Hotel (fictional) O&M: condenser approach above 4.5 K indicates fouling; design efficiency 0.68 kW/ton.
```

F1-CH40 and F1-CH42 sit between the two limits. With the first hotel's 3.0 K they look fouled; with this hotel's 4.5 K they are fine. F1-CH43, at 5.9 K, is fouled under either limit, so the wrong fact did no visible harm there. This is the paper's premise failure in miniature: its failure analysis reports that an incorrect premise can propagate across candidates.

Notice why no contrast caught it. The skill fired and the agent made the right comparison, so the trajectory looked healthy. Only the number was wrong, and only the judge, which reads hidden truth after the step, could see that.

One seed is one story. Lab 05-3 repeats the comparison on seeds 0–4:

**Expected output** (captured on this Mac, SIM)

```
│ seed  re-verified  unchecked  difference  unchecked runs that used 3.0 K  tasks worse unchecked
│ ────  ───────────  ─────────  ──────────  ──────────────────────────────  ─────────────────────
│ 0     100.0        92.5       -7.5        F1-CH40, F1-CH42, F1-CH43       F1-CH40, F1-CH42     
│ 1     74.5         74.5       +0.0        —                               —                    
│ 2     95.8         95.8       +0.0        —                               —                    
│ 3     95.8         95.8       +0.0        F1-CH42                         —                    
│ 4     91.7         88.3       -3.4        F1-CH40                         F1-CH40              
◆ every seed's first-hotel library carries the 3.0 K fact inside a copied skill. It reaches a run only when the copied skill survives composition (a domain editor that writes its own skill from the task's manual overwrites it) and then fires. Seed 3: F1-CH42 used 3.0 K yet scored the same in both arms: the re-verified run used 4.5 K, but 4.2 K sits within 1 K of that limit, where the SIM makes the call uncertain, and it missed too
✓ every task the unchecked arm lost had used the 3.0 K fact; re-verification never scored below copying here
```

The effect shows on two seeds of five, and only where the stale fact reached a run. That is the shape of a premise failure: silent inside the step, and visible only where the stale fact lands on a task it gets wrong.

✓ Checkpoint: you can explain why F1-CH40 changes diagnosis between the two arms and F1-CH43 does not.

## 7 · Where a workspace fact belongs in Ops Assistant

The kit's skill keeps the threshold inside the skill, as the repository's prompt suggests for one workspace. The same prompt warns against the always-on home:

> “Never store this matter's facts as durable memory: they are wrong anchors for the next task.” — `harness2/core/prompts.py`

The course's EXAMPLE design gives Ops Assistant a better home than either: a knowledge graph with Brick identities, provenance and supersession — the same kind of graph memory you built in Weeks 14–15. The design's rule is that tenant and building facts discovered during a probe (a point mapping, a nameplate value, a schedule) go to the graph through the reviewed supersession path, never into `LongTermMEMORY.md`.

| what | home in the EXAMPLE design | why |
|---|---|---|
| the procedure: compare the approach with the manual's limit | a skill (C6) | travels to every hotel in the tenant class |
| the fact: "Chelsea approach limit 4.5 K" | the knowledge graph | provenance, supersession, read at run time per building |
| a short fact true for every task | `LongTermMEMORY.md` (C3), capped | paid on every call, so only if something else is cut |

With the threshold in the graph, the copied skill carries procedure only, and the second hotel reads its own limit. The design adds the invariant that keeps this safe: a harness file informs, it never authorises.

✓ Checkpoint: you can say where each of the three rows above lives, and what goes wrong at the second hotel if the fact lives in a shared skill or in memory.

## Labs — run them here

**labs/lab05_1_build_a_library.py** — One fault-review library built in task order: INDEX.md, an EXPERIENCE.md, the whole stored tree, what Library.relevant hands the next editor, and a LIVE model writing two entries.

**labs/lab05_2_library_ablation.py** — One library per domain and executor, then the library-on versus library-off ablation over three shuffled streams, read against the SD.

**labs/lab05_3_second_hotel.py** — A first-hotel library at the second hotel, with and without re-verifying copied facts, a seed sweep, and where the fact belongs.

## Try it yourself

`exercises/ex05_write_the_lesson.py` has four TODOs:

1. Label three cues: too generic, too specific, or just right.
2. Write the EXPERIENCE entry for a real edit row (F7-00): cue, lesson, evidence and confidence.
3. Decide whether a record whose only row is `confounded` deserves an entry.
4. Choose the home of the second hotel's 4.5 K limit.

```bash
# on: laptop
.venv/bin/python week31_harness2_recursive_harnessing/05_library_and_experience/exercises/ex05_write_the_lesson.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ cues: a too generic (changes no behaviour) · b too specific (never fires again) · c just right
✓ entry A: cue 'a read over a long time window that times out and is retried unchanged' · lesson names a component · evidence cites the row · confidence 'medium — one task, three probes agree, no second task yet'
✓ record B: NO_ENTRY — its only row is confounded (2 of 3 repeats); a lesson on it would be a guess
✓ the 4.5 K limit is a workspace fact: the knowledge graph, with provenance and supersession (EXAMPLE design for this course)

═ Done. Module 06 reads the results table these libraries feed: streams, noise, ablations, transfer, pass^k.
```

The checker uses simple rules (no IDs or digits in the cue, a component and a verb in the lesson, the row cited in the evidence, no "high" on one task, no grader language). Your wording will differ from the solution's; a person still reviews whether it reads well.

<details><summary>Hint — TODO 2</summary>

The cue describes the situation, not the task: "a long-window query that times out" fires on any helper, any point, any hotel. "The 30-day AHU-B1 pull" fires once. The lesson changes a file: say which file (`tool_guidance.md`) and what goes in it (one bounded fallback with a stop condition). The evidence names the row and its contrast (`blind_retry: mitigated` in three probes). One task stands behind it, so confidence is medium at most.

</details>

<details><summary>Hint — TODO 3</summary>

Read the verdict, not the observation. Two of three probes mitigated the storm, so the row is `confounded`, and a lesson resting only on confounded rows is a guess. The prompt allows narrowing to what the record shows, but this record shows no reusable contrast at all.

</details>

✓ Checkpoint: all four checker lines are ✓.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| INDEX.md lists more entries than the tutorial | a library folder from an older run was reused | the labs delete their `.runs/lab05_*` folder at start; if you call `bench.stream(root=…)` yourself, start from an empty root |
| Lab 05-3 shows no difference on seed 0 | an edited kit, or a different library | `git diff week31_harness2_recursive_harnessing/common`; the effect depends on the copied skill carrying the 3.0 K fact |
| Step 6 of Lab 05-1 says "no model call" | the lab ran in 🧠 SIM | switch the runner to 🦙 LIVE, or run with `H2_BRAIN=ollama H2_MODEL=gemma4:12b` |
| Every entry says confidence "medium" | the kit only writes medium (one task per entry) | expected; the contract's high and low are for you to apply by hand |
| Ablation numbers differ slightly from a table elsewhere in the course | a different task set, seed or editor | compare like with like: same tasks, executor, editor and three stream seeds |

## Next

[Lab 06 — Reading a results table: streams, noise, ablations, transfer, pass^k](../06_reading_the_results/TUTORIAL.md): read the base-versus-Harness² table over three shuffled streams, the full ablation set, transfer between executors, and why pass^k is the number a customer feels.
