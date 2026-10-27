# Week 31 · Harness²: agents that improve their own harness

**Agent = Model + Harness.** This week keeps the model frozen and improves everything around it: system prompt,
guardrails, memory, tool guidance, subagents, skills, scripts and plugins. The method is **Harness²** (Google
Research, paper arXiv 2610.14224): run the agent, read the **execution evidence** (trajectories and deliverables),
let two scoped editors propose a fix, probe the candidates on the same task, compose or keep the base, and keep
what was learned in a library. No ground truth inside the loop: scores are measured only after the run. The last
modules apply the method to the always-on runtime of **Ops Assistant**, a fictional building-operations agent
(EXAMPLE design written for this course, not a deployed system). Ten modules, each with labs and an exercise, served by the **🪢 Harness Lab Runner**.

**What this course is not.** The course re-implements the method in plain Python in `common/` and **never runs the
google-research/harness2 code**. It quotes the repository only from the text snapshots in `common/sources/`. The
agent and its editors are a simulated teaching model: their numbers show a mechanism, not a measurement of
Harness², of a real model or of a production agent.

## Start

```bash
# every time (from the repo root)
.venv/bin/python week31_harness2_recursive_harnessing/00_harness_lab_runner/tutorial_server.py      # → http://127.0.0.1:8131
```

The runner picks a free port if 8131 is taken (or set `HARNESS_GUIDE_PORT`). In its header:

- **🧠 SIM / 🦙 LIVE.** **SIM** (default) is a deterministic teaching model of the agent and its editors: offline,
  instant, the same output every run. **LIVE** lets a local Ollama model play one role in the labs that offer it
  (`gemma4:12b` by default); a rerun can differ. The browser can set only `H2_BRAIN` and `H2_MODEL`, and only a
  local, non-cloud, non-embedding model.
- **📝 Pre/Post test.** The pre-test (Form P, before Module 01) and the post-test (Form Q, after Module 10): 20
  multiple-choice items each. Grading happens on the server, which then shows your score, the score band and an
  explanation for each item. Once you have a pre-test and a later post-test, it also shows your normalized gain
  g = (post − pre) / (20 − pre). Attempts are saved to `00_harness_lab_runner/.quiz_history.json` (gitignored).
- **⌨ Terminal**, **EN / ไทย**, ◀ ▶ section navigation, and ✓ checkpoints that are saved on this machine.

**Prerequisites.** Only the repo's `.venv` (Python 3.13); there are no new packages to install. Ollama is
optional and needed only for 🦙 LIVE. Every lab runs on SIM without it, finishes in seconds and makes at most 6
model calls on LIVE.

## Modules

| # | module | objective | lesson part | you build |
|---|---|---|---|---|
| 01 | [Agent = Model + Harness: the eight components and what each costs](01_agent_model_harness/TUTORIAL.md) | O1 | 1 | the base tree; always-on cost and the prefix hash; same model, three harnesses |
| 02 | [One Propose–Probe–Compose step, by hand and by loop](02_propose_probe_compose/TUTORIAL.md) | O2 | 2 | the demo's idea re-enacted in a few lines; one step on F1-01; KEEP_BASE on quiet tasks |
| 03 | [Two scopes and the cheapest effective component](03_two_scopes_cheapest_component/TUTORIAL.md) | O2 | 2 | INV vs DOM editors; the failure → component ladder; the generic-editor ablation and always-on growth |
| 04 | [Evidence, not scores: contrasts, verdicts and the evidence boundary](04_evidence_not_scores/TUTORIAL.md) | O2 · O4 | 2 · 4 | mitigated/unchanged/regressed; helped/hurt/no observed effect/confounded; repeats; EVIDENCE IS DATA (naive vs bounded editor, LIVE option); the measurement-word ban; None ≠ 0 |
| 05 | [The improvement library: EXPERIENCE.md, NO_ENTRY and re-verifying facts](05_library_and_experience/TUTORIAL.md) | O2 | 2 | a library over a stream; INDEX → entry → adapt; a second hotel with and without re-verification; workspace facts go to the knowledge graph, not memory |
| 06 | [Reading a results table: streams, noise, ablations, transfer, pass^k](06_reading_the_results/TUTORIAL.md) | O3 | 3 | base vs Harness² over three shuffled streams (mean ± SD) for both executors; the ablation table; k and sequential; transfer between executors; pass@k vs pass^k |
| 07 | [Three ways harnessing fails, and measurement hygiene](07_failure_modes/TUTORIAL.md) | O4 | 4 | blind selection (k > 1); a premise that caps every candidate; attribution noise; a rubric leak and its controls |
| 08 | [Ops Assistant: the nightly harness review routine with a human gate](08_nightly_harness_review/TUTORIAL.md) | O5 | 5 | weak-signal task selection; draft version + review packet; boundary in prompt vs in code; human-signal firewall; cache discipline; promotion ladder and rollback |
| 09 | [The Harness Bench: graders, pass^3 and a pre-registered experiment](09_harness_bench_evals/TUTORIAL.md) | O5 | 5 | refine/test halves; code, model (SIM judge) and human graders; arms A–F; pre-registered criteria; the stop rule |
| 10 | [Capstone: harness the pilot family and defend the version](10_capstone_harness_review/TUTORIAL.md) | all | 6 | run the routine on the pilot family, promote a version, pass the integrity suite, report against the pre-registered criteria |

## What is where

```text
week31_harness2_recursive_harnessing/
  00_harness_lab_runner/   tutorial_server.py (FastAPI, :8131) + static/guide.html: the course, ▶ labs, ⌨ terminal,
                           🧠 SIM / 🦙 LIVE switch, 📝 pre/post test (/api/quiz, graded server-side)
  common/                  the course's own re-implementation of the method (frozen):
                           h2out.py (output + brains) · harness.py (the tree) · ops_sim.py (tasks + SIM agent)
                           h2kit.py (editors, composer, selector, library) · bench.py (measurement, outside the loop)
  common/sources/          text snapshots the tutorials quote from, and the two test forms
  docs/                    the lesson material: the facilitator guide, both test forms and a
                           FACILITATOR-ONLY quiz solutions file (never copy its keys into a tutorial, lab or page)
  NN_*/                    TUTORIAL.md (+ .th.md), labs/, exercises/ (+ solutions/), diagrams.json
```

## Maintainer checks

```bash
# on: laptop
.venv/bin/python week31_harness2_recursive_harnessing/common/audit_references.py <module>     # → 0 not found verbatim
.venv/bin/python week31_harness2_recursive_harnessing/common/check_translation.py <module>    # → ✓
```

Adding or changing a module: read [AUTHORING.md](AUTHORING.md) first.
