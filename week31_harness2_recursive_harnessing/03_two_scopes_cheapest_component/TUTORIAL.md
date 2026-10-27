# ▶ Harness Lab 03 — Two scopes and the cheapest effective component

> Part of Week 31 · Harness²: agents that improve their own harness. You improve the harness of one simulated agent, the **HVAC Expert agent of Ops Assistant** (a fictional always-on operations assistant), from execution evidence, and you see the real output. Everything is synthetic (two fictional hotels, invented alarms and work orders), so every lab is free, offline and safe. Labs run on **🧠 SIM** by default: a deterministic teaching model of an agent and its editors. Flip the runner to **🦙 LIVE** where a lab offers it, and a real local model plays one role.

**What you'll actually do**
- Read the two editor templates of the repository side by side and learn the one test that decides which editor owns a failure.
- Learn the repository's "prefer, in this order" ladder: skill, script, subagent, plugin, and always-on text last.
- Map every failure the simulated agent shows on the refine half to its scope, its template line and the component the kit's editors write, with the always-on cost of each.
- Run the generic-editor ablation over three streams and see two results: the score, and the always-on text each arm adds.
- Watch the editor climb from a prose rule to a plugin only after the prose was skipped again.

**Time** ~45 min · **Difficulty** intermediate · **Hardware** this laptop only (no model needed)

**Sources:** [Harness² paper, arXiv 2610.14224](https://arxiv.org/html/2610.14224) · [google-research/harness2](https://github.com/google-research/harness2) (Apache-2.0; snapshot of `prompts.py` in `common/sources/`) · the skills of Week 7, the hooks and mods of Week 29, the prompt caching of Week 2

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Module 02 done | you can read a `rec.show()` and name the six moves | this module asks where each kept edit should live |
| This repo's Python | `.venv/bin/python --version` → 3.13 | runs every lab |
| No model | — | all three labs are SIM; the generic ablation runs 204 loop steps in under a second |

```bash
# on: laptop
.venv/bin/python -c "import sys; sys.path.insert(0, 'week31_harness2_recursive_harnessing/common'); from h2kit import observe; import inspect; print(len(inspect.getsource(observe).splitlines()), 'lines decide what an editor can see')"
```

**Expected output** (captured on this Mac, SIM)

```
28 lines decide what an editor can see
```

Open `common/h2kit.py` and read `observe()`. Every failure kind in this module comes from those lines, and each one is tagged `inv` or `dom`.

✓ Checkpoint: you have read `observe()` and can name one `inv` kind and one `dom` kind.

## 1 · Two editors, two dimensions

The paper's step proposes **two** candidates, not one. Each comes from an editor with its own job description. Lab 03-1 prints the key lines of both templates.

> “You own procedure, tool use and completion -- not subject matter.” — `harness2/core/prompts.py`, the INV template

> “You own the practitioner model the agent lacks:” — `harness2/core/prompts.py`, the DOM template

The INV editor lists five failure shapes and a component for each. The DOM editor turns practitioner knowledge into four kinds of checkable rule.

| INV — how the agent works | DOM — what a practitioner knows |
|---|---|
| TOOL ERROR → one bounded fallback and a stop condition | ANALYTIC MOVE → name the operation, its inputs and result |
| MECHANICAL REPETITION → a script | ARTIFACT CONTRACT → required parts and their checks |
| TRIGGERED WORKFLOW → an on-demand skill | DECISION RULE → trigger, evidence, conflicts, uncertainty |
| IGNORED INVARIANT → a defensive plugin, only after repeated evidence | ENCODING → choose the form; a skill is the default |
| ORDINARY SEQUENCING → concise always-on guidance | |

Each template ends by fencing the other one out:

> “Do not add professional rules, required domain sections or matter facts. A parallel candidate owns them; stay focused on how the agent works.” — the INV template

> “Do not diagnose commands, retries, reading order, output paths, budgeting, scripts or hooks. A parallel candidate owns procedure; stay on practitioner knowledge.” — the DOM template

The repository's second prompt pack states the test in one line each way:

> “The test: if your edit stops making sense when the subject matter changes, it belongs to them, not you.” — `harness2/core/prompts.py`, the INVARIANT editor

> “The test: if your edit would help just as much on a task from a completely different profession, it belongs to them.” — `harness2/core/prompts.py`, the DOMAIN editor

The same fact can land in both scopes, in two different forms. "Call `cmms.work_orders` before any equipment proposal" is a sequencing rule that holds for any equipment: INV. "Reconcile the alarm with the open work orders and say what the PM already covers" is the practitioner's analytic move: DOM. In Lab 02-2 both candidates produced the maintenance check, each in its own form.

Why split at all? One editor that sees everything tends to write one thing: a paragraph of prose. Two editors with fenced jobs write two different kinds of fix, and the composer can judge each one on its own probe. Section 4 measures what happens when you take the split away.

✓ Checkpoint: you can apply the one-line test to "call cmms.work_orders before any equipment proposal" (INV) and to "reconcile the alarm with the open work orders" (DOM), and say why they differ.

## 2 · The ladder: prefer, in this order

Both editors are given the same picture of the harness, and the same warning about cost. Module 01 measured that cost; the prompt says it plainly:

> “ALWAYS-ON PROSE IS THE EXPENSIVE OPTION. Every word of it is in context on every run, competing with the agent's own output budget” — `harness2/core/prompts.py`

Then it gives the order to try components in:

> “So prefer, in this order:” — `harness2/core/prompts.py`

| Rung | Component | When (from the prompt) |
|---|---|---|
| 1 | a **skill** (C6) | "anything with a recognisable trigger"; the default form for new instruction |
| 2 | a **script** (C7) | "a deterministic operation the runs hand-rolled and got wrong, or did slowly" |
| 3 | a **subagent** (C5) | "a bounded investigation whose input is large and whose answer is small" |
| 4 | a **plugin** (C8) | a recurring, mechanical problem in the tool loop |
| 5 | **always-on text** | "last, and only for a short rule that genuinely applies to every run" |

Read rung 5 carefully. It is not "never". The INV template sends ORDINARY SEQUENCING to "concise always-on guidance" on purpose: "check work orders before proposing" is one short line that applies to every fault review. The rule is the one Module 01 taught for Ops Assistant. Pay for always-on text only when it earns its place, and if you add some, cut some.

This ladder also matches the paper's strongest ablation result. As reported in the paper, removing skills from the improved harness loses 5.3 points on average, removing scripts 3.7, and removing prompt text only 1.5.

✓ Checkpoint: you can recite the five rungs in order and say which one the prompt calls "the default form for new instruction" (a skill).

## 3 · Every failure, its scope and its component

Lab 03-1 runs the base harness on all 17 refine tasks (three seeds, both executors), collects every failure kind an editor could see, and asks the kit's editors what they would write for each one.

**Expected output** (captured on this Mac, SIM)

```
│ failure kind      scope  template line (our reading)                  kit writes  ladder rung  Δ always-on  tasks
│ ────────────────  ─────  ───────────────────────────────────────────  ──────────  ───────────  ───────────  ─────
│ digest_contract   DOM    ARTIFACT CONTRACT                            C1 + C6     1 skill      +19          3    
│ no_analytic_move  DOM    ANALYTIC MOVE                                C1 + C6     1 skill      +19          6    
│ nuisance_notify   DOM    DECISION RULE                                C1 + C6     1 skill      +13          1    
│ storm_spam        DOM    DECISION RULE                                C1 + C6     1 skill      +13          1    
│ thai_register     DOM    ENCODING                                     C1 + C6     1 skill      +18          1    
│ blind_retry       INV    TOOL ERROR                                   C4          5 always-on  +38          1    
│ hand_arithmetic   INV    MECHANICAL REPETITION                        C4 + C7     2 script     +23          4    
│ no_cmms           INV    ORDINARY SEQUENCING                          C4          5 always-on  +24          6    
│ no_deliverable    INV    completion: reserve room for every artifact  —           — (no edit)  +0           1    
│ no_evidence       INV    ORDINARY SEQUENCING → IGNORED INVARIANT      C4          5 always-on  +25          6    
│ no_reopen         INV    completion: reopen or check the deliverable  C4 + C7     2 script     +29          3    
```

The "template line" column is the course's reading, not the repository's. Here is how to read the rest.

- **Every DOM row is a skill plus one trigger line.** The `C1` in "C1 + C6" is the line in `systemprompt.md` that names the skill. Without it the skill never loads; the repository's surface prompt calls a missing trigger line "the most violated rule here by a wide margin". The 13–19 always-on tokens are that line only. The skill body loads only when its trigger fires.
- **INV rows split three ways.** A script where the runs hand-rolled something (kW/ton, the digest check). One short always-on line for ordinary sequencing and for a tool error's bounded fallback. And nothing at all for `no_deliverable`: the kit never writes "try harder" prose. It fixes a visible cause instead, such as the blind retry.
- **`no_evidence` starts on rung 5.** The first fix for a skipped rule is to state the rule clearly. Section 5 shows when it may climb to rung 4.

Every row is something a trajectory showed. The "tasks" column counts the refine tasks where it appeared at least once. This is the INV template's instruction made concrete: "match each failure to the cheapest effective component".

✓ Checkpoint: you can say why `hand_arithmetic` gets a script but `no_cmms` gets one always-on line.

## 4 · The generic-editor ablation

What if one editor did everything? The kit's `scoped=False` replaces the INV and DOM editors with **one generic editor** that writes what "one editor rewrites the prompt" tends to write: long, unscoped prose in `systemprompt.md`. Lab 03-2 first shows both editors' fix for the same failure.

**Expected output** (captured on this Mac, SIM)

```
┌─ scoped INV editor → tool_guidance.md  (+24 tokens)
│ - Before any proposal about equipment, check open and recent work orders with cmms.work_orders.
└─
┌─ generic editor → systemprompt.md  (+68 tokens)
│ When working, always be careful and thorough about no cmms. Before any proposal about equipment, check open and recent work orders, because maintenance may already be planned. Think step by step, double-check your work, and explain your reasoning clearly to the reader.
└─
```

Then it runs three open-world streams (shuffled order) over the 17 refine tasks, three ways, with the library off in both loop arms so that only the scoping differs (editor: `sim-cloud-editor` in every arm, as in the `AUTHORING.md` table; Lab 02-3 paired each executor with its own editor instead).

**Expected output** (captured on this Mac, SIM)

```
│ executor   arm                                score* mean ± SD  per stream                              
│ ─────────  ─────────────────────────────────  ────────────────  ──────────────────  ────────────────────
│ sim-cloud  base harness                       80.3 ± 4.1        81.2 / 83.8 / 75.8  ████████████████░░░░
│ sim-cloud  Harness² k=1 · scoped INV + DOM    93.8 ± 3.7        97.6 / 93.5 / 90.3  ███████████████████░
│ sim-cloud  Harness² k=1 · one generic editor  91.1 ± 2.9        94.1 / 90.9 / 88.3  ██████████████████░░
│ sim-local  base harness                       72.2 ± 3.7        76.3 / 69.2 / 71.0  ██████████████░░░░░░
│ sim-local  Harness² k=1 · scoped INV + DOM    81.8 ± 4.5        84.1 / 76.6 / 84.8  ████████████████░░░░
│ sim-local  Harness² k=1 · one generic editor  77.3 ± 3.6        80.4 / 73.3 / 78.2  ███████████████░░░░░
◆ * measured after the run by bench.judge, which reads the hidden truth — nothing in the loop ever sees it
◆ sim-cloud: scoped − generic = +2.7 points · all three streams agree in direction: yes · within one SD — the size is not shown on 17 tasks
◆ sim-local: scoped − generic = +4.5 points · all three streams agree in direction: yes · larger than one SD
```

Read the score honestly. Scoped beats generic on both executors, the same direction as the paper, which reported minus 2.9 and minus 3.7 for the ablation without scoping, on WorkBuddy-Bench and JobBench. On both executors all three streams agree in direction, so the facilitator guide's rule (believe a small difference only if all three streams agree) lets you believe the direction. The size is another matter. On sim-cloud the gap (+2.7) is within one standard deviation, so on 17 tasks its size is **not shown**. On sim-local the gap (+4.5) is larger than one SD, but that size comes from the SIM's assumptions (see the note below), so it shows the mechanism and measures nothing. Module 06 reads results tables like this one in depth.

The second table is the one that does not hide in noise.

**Expected output** (captured on this Mac, SIM)

```
│ executor   arm                                composed  Δ always-on mean  max   components changed          nothing delivered
│ ─────────  ─────────────────────────────────  ────────  ────────────────  ────  ──────────────────────────  ─────────────────
│ sim-cloud  Harness² k=1 · scoped INV + DOM    27/51     +38.3             +68   C1×20, C4×19, C6×22, C7×12  0                
│ sim-cloud  Harness² k=1 · one generic editor  17/51     +100.9            +140  C1×17                       0                
│ sim-local  Harness² k=1 · scoped INV + DOM    25/51     +36.3             +72   C1×16, C4×19, C6×17, C7×13  2                
│ sim-local  Harness² k=1 · one generic editor  14/51     +105.7            +201  C1×14                       2                
✓ sim-cloud: every generic edit landed in C1 (systemprompt.md) — always-on prose, paid on every call
✓ sim-local: every generic edit landed in C1 (systemprompt.md) — always-on prose, paid on every call
```

- The generic editor adds about **100 always-on tokens** per composed step, against about 40 for the scoped pair. Its worst step adds 201.
- Every generic edit lands in **C1**. The scoped pair spreads its edits over skills, scripts, tool guidance and short trigger lines.
- The generic arm also composes less often (17 against 27 steps on sim-cloud, 14 against 25 on sim-local). Its prose names no trigger, no script and no tool, so its probes often show no change and the composer drops them. KEEP_BASE protects the generic arm from itself, a little.

For an always-on agent the token column is the expensive one. Every always-on token is paid on every call, and every change to it is one cold prompt cache. That is why the course's EXAMPLE design for Ops Assistant biases the invariant editor toward on-demand components and makes every review packet show the always-on delta.

> 📌 **SIM, not proof.** The SIM agent follows prose with a fixed probability and follows a fired skill more reliably. Those are assumptions written into `ops_sim.py` to show the mechanism the paper reports. They make the shape of this table likely; they do not measure it.

✓ Checkpoint: you can say which result of the ablation is within noise here (the score gap on sim-cloud), which clears one SD only inside the SIM's assumptions (the score gap on sim-local), and which is not noise on either executor (the always-on growth, all of it in C1).

## 5 · A plugin only after repeated evidence

A plugin is the only component that enforces. Module 01 quoted the reason: "Prose can be stopped reading; a hook cannot." So why not reach for one every time a rule is skipped? Because the INV template sets a bar:

> “- IGNORED INVARIANT: only repeated evidence that clear prose is skipped can justify a defensive” — `harness2/core/prompts.py` (the template names the plugin hook here)

> “It must not block valid completion.” — `harness2/core/prompts.py`

The kit encodes the bar in `inv_edits("no_evidence", tree)`. Lab 03-3 calls it twice.

**Expected output** (captured on this Mac, SIM)

```
✎ tree = v1 base       → files — · always-on line: '- Every ActionProposal carries evidence links: the alarm id, the telemetry window, the document.'
✎ tree = v1 + that line → files ['plugin/require_evidence.ts'] · always-on line: ''
```

On the base tree, the editor writes the rule as one clear line. On a tree that already has that line, and still produced a proposal with no evidence, it writes `plugin/require_evidence.ts`. The kit's stand-in for "repeated evidence" is exactly that: the line is already there (an earlier edit put it in) and this run still skipped it. Two observations, not one.

What does each rung buy? The lab counts empty evidence lists over 40 proposals per row:

**Expected output** (captured on this Mac, SIM)

```
│ executor   harness           proposals  evidence empty  plugin blocked + agent fixed
│ ─────────  ────────────────  ─────────  ──────────────  ────────────────────────────
│ sim-cloud  v1 base           40         16              0                           
│ sim-cloud  + prose line      40         2               0                           
│ sim-cloud  + prose + plugin  40         0               2                           
│ sim-local  v1 base           40         27              0                           
│ sim-local  + prose line      40         11              0                           
│ sim-local  + prose + plugin  40         0               11                          
```

The prose line cuts most of the failures. It cannot cut all of them, and the weaker executor skips it more often (11 of 40 still empty, against 2 on sim-cloud). The hook catches every remaining case, and it adds zero always-on tokens because it lives in the tool loop. These rates are the SIM's assumptions, not a measurement of any real model.

The lab then shows the same climb inside one step. A **sequential** step with k = 2 edits its own checkpoint in round 2 (Module 06 compares the modes). The lab uses seed 19 of F1-02 on sim-local, chosen because the failure comes back on a checkpoint that already has the line. On most seeds of this task it does not come back, and nothing climbs: of seeds 0–29, only seed 19 reaches the plugin. That is the bar working as intended:

**Expected output** (captured on this Mac, SIM)

```
✎ inv1.r1 [inv] ActionProposal sent with an empty evidence list → tool_guidance.md
✎ inv1.r2 [inv] ActionProposal sent with an empty evidence list → plugin/require_evidence.ts
◆ round 1 (editing v1): no_evidence → ['tool_guidance.md'] · round 2 (editing the round-1 checkpoint, which has the line): no_evidence → ['plugin/require_evidence.ts']
✓ the plugin appears only in round 2, after the prose line was in place and the failure came back
```

The excerpt keeps the two `no_evidence` edit lines of the full `rec.show()` and the lab's summary. Finally, the lab checks the template's second sentence on four deliverables that are not proposals. The plugin fired zero times, and all four delivered. In Ops Assistant's EXAMPLE design this hook is the `send_to_user` auto-review. It still never replaces the named-human approval, which stays in platform code (Module 01).

✓ Checkpoint: you can say what has to happen before the editor may write a plugin, and what the plugin must never do.

## Labs — run them here

**labs/lab03_1_two_scopes_and_the_ladder.py** — The INV and DOM templates and the ladder, verbatim, then every failure in the refine half mapped to its scope, template line, component and always-on cost.

**labs/lab03_2_generic_editor_ablation.py** — Base vs scoped vs one generic editor over three streams per executor: the score, and the always-on text each arm adds.

**labs/lab03_3_plugin_after_repeated_evidence.py** — The no_evidence rule with and without the prose line, what each rung buys, a sequential step that climbs to a plugin, and the plugin's no-block check.

## Try it yourself

`exercises/ex03_scope_and_component.py` lists nine EXAMPLE failure classes of Ops Assistant, the course's own design. It has three TODOs:

1. The scope of each failure: INV or DOM.
2. The template line each one falls under.
3. The cheapest effective component for each.

```bash
# on: laptop
.venv/bin/python week31_harness2_recursive_harnessing/03_two_scopes_cheapest_component/exercises/ex03_scope_and_component.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ scopes right: five INV (how the agent works), four DOM (what a practitioner knows)
✓ every failure sits on the right line of its editor's template
✓ cheapest components right: one bounded fallback, two scripts, five skills, one plugin
✓ only failure 4 earns a plugin, and only because clear prose was already skipped repeatedly

═ Done. Module 04 asks how the composer knows an edit helped: contrasts, verdicts and the evidence boundary.
```

Where a failure could take two components (a guidance line *and* a script), the exercise asks for the cheapest one.

<details><summary>Hint — TODO 1</summary>

Use the one-line test from section 1. "Retry a timed-out query once, in chunks" makes sense whatever the building does: INV. "Escalate a high condenser pressure alarm only when…" stops making sense outside HVAC: DOM. Two are tricky. A chiller trip procedure is a fixed multi-step workflow, so the course files it under INV's TRIGGERED WORKFLOW. Reconciling the alarm list with CMMS work orders is a practitioner's comparison, so the course files it under DOM's ANALYTIC MOVE, even though a tool call is involved.

</details>

<details><summary>Hint — TODO 3</summary>

Walk the ladder from the top. A recognisable trigger means a skill. Hand-rolled arithmetic or a mechanical check means a script. A tool error means one bounded fallback in tool guidance. Only the failure that says "despite instructions" has the repeated evidence a plugin needs.

</details>

✓ Checkpoint: all four checker lines are ✓.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Lab 03-2's numbers differ slightly from the table in `AUTHORING.md` | that table runs with the library on; this lab turns it off in both loop arms | compare like with like: the "scoped" arm here is the "no library" column there |
| Lab 03-3 step 4 prints `⚠ the escalation did not happen` | an edited kit (the sequential example depends on `sim-local`, seed 19) | `git diff week31_harness2_recursive_harnessing/common` |

## Next

[Lab 04 — Evidence, not scores: contrasts, verdicts and the evidence boundary](../04_evidence_not_scores/TUTORIAL.md): learn how the composer tells an edit that helped from one that only looked like it, and why the evidence it reads is data, never instructions.
