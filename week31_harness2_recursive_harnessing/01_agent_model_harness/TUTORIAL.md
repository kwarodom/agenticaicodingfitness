# ▶ Harness Lab 01 — Agent = Model + Harness: the eight components and what each costs

> Part of Week 31 · Harness²: agents that improve their own harness. You improve the harness of one simulated agent, the **HVAC Expert agent of Ops Assistant** (a fictional always-on operations assistant), from execution evidence, and you see the real output. Everything is synthetic (two fictional hotels, invented alarms and work orders), so every lab is free, offline and safe. Labs run on **🧠 SIM** by default: a deterministic teaching model of an agent and its editors. Flip the runner to **🦙 LIVE** where a lab offers it, and a real local model plays one role.

**What you'll actually do**
- Read the paper's definition, "Agent = Model + Harness", and see what it leaves out on purpose: the weights.
- Sort the eight Harness² components by how they reach the model, and learn which ones you pay for on every call.
- Measure, on a real tree, what one rule costs as always-on prose versus as a skill, and why both edits break the prompt cache once.
- Run the same simulated model with three harnesses and watch the behaviour change while the model does not.
- Place every piece of Ops Assistant's harness into a component, and find the one rule that must stay outside them all.

**Time** ~35 min · **Difficulty** beginner · **Hardware** this laptop only (Ollama optional)

**Sources:** [Harness² paper, arXiv 2610.14224](https://arxiv.org/html/2610.14224) · [google-research/harness2](https://github.com/google-research/harness2) (Apache-2.0; snapshots of `prompts.py`, `tree.py`, `README.md` in `common/sources/`) · the prompt caching of Week 2, the skills of Week 7, the always-on agents of Week 17

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| This repo's Python | `.venv/bin/python --version` → 3.13 | runs every lab and the Harness Lab Runner |
| No new packages | the kit uses only the standard library | the loop, the simulated agent and the judge are plain Python |
| Ollama (optional) | `ollama list` shows `gemma4:12b` | only for 🦙 LIVE labs (Modules 04, 05, 08); every lab works without it |

```bash
# on: laptop
.venv/bin/python week31_harness2_recursive_harnessing/common/h2out.py
```

**Expected output** (captured on this Mac, SIM)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
━━ h2out self-check
   which mode, which local models
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
◈ 🧠 SIM · deterministic teaching model of an agent and its editors (switch to 🦙 LIVE in the runner)
◆ H2_BRAIN=(unset → sim) · model=gemma4:12b
◆ Ollama local models: scb10x/typhoon-ocr1.5-3b:latest, hf.co/ggml-org/embeddinggemma-2-GGUF:Q8_0, nemotron-3.5-lightning:latest, nemotron-3-nano:latest, gemma3:4b, gemma4:12b, gemma4:latest
```

> 📌 **What the SIM is, and is not.** The course never runs the google-research/harness2 code. `common/` holds the course's own re-implementation of the method in plain Python, on a **simulated** Ops Assistant agent with known weaknesses. Its numbers (how often an agent follows prose, how often a skill fires) are **assumptions written to show the mechanics**. When a lab shows that skills beat prose, it shows how the mechanism works, not proof that it does. The proof is the paper's, and Module 06 reads it.

✓ Checkpoint: the self-check prints the SIM line, and you know whether Ollama is available for LIVE labs.

## 1 · Agent = Model + Harness

The paper starts from one equation. The repository's README defines the layer it names:

> “the system around a model that enables it to act as an agent: prompts, tools, memory, skills, orchestration, and more” — `harness2/README.md`

Everything around a frozen model is the harness. The model's weights never change. **Harnessing** is the work of improving that layer. Harness² asks whether that work can improve itself, task after task, without anyone telling it the right answers.

You have edited harnesses all course without the name. The Week 7 skill you wrote is a harness component. So is the Week 17 retry rule, the Week 23 signed policy, and the hook from Week 29. What is new this week is treating all of them as **one versioned object** that a loop may edit, with evidence for every edit.

| | Fine-tune the model | Edit the harness |
|---|---|---|
| What changes | weights | text and code files around the model |
| Review | hard: a diff of numbers | easy: a diff of files a person can read |
| Undo | retrain or keep the old checkpoint | point back to the parent version |
| Moves to another model | no | often, but unevenly (Module 06) |
| Who can do it | a training pipeline | a reviewer, or a loop with a human gate |

✓ Checkpoint: you can say what stays fixed in "Agent = Model + Harness" (the model's weights) and what a harnessing loop is allowed to change (the files around it).

## 2 · The eight components, by how they reach the model

Harness² names eight editable components. The repository lays them out as files (the OpenCode layout). What matters is **how each one reaches the model**, because that decides what it costs.

```
                 ┌──────────────── always on: in context on EVERY call ────────────────┐
  frozen model ◀─┤ C1 systemprompt.md · C2 guardrails.md · C3 LongTermMEMORY.md · C4 tool_guidance.md │
                 └──────────────────────────────────────────────────────────────────────┘
                 ┌──────────────── on demand: cost only when used ─────────────────────┐
                 │ C6 skills/<name>/SKILL.md   loaded by the skill tool when its trigger fires │
                 │ C7 scripts/<name>.py        run via bash; the model reads only the output   │
                 │ C5 sub_agents/<name>/AGENT.md  a worker in its OWN context; returns a result │
                 └──────────────────────────────────────────────────────────────────────┘
                 ┌──────────────── in the tool loop: code, not prose ──────────────────┐
                 │ C8 plugin/<name>.ts         a hook around tool calls; enforces, never asks   │
                 └──────────────────────────────────────────────────────────────────────┘
```

Two lines from the editor prompt in the repository explain the two ends of this picture:

> “An on-demand component costs nothing until the moment it is invoked.” — `harness2/core/prompts.py`

> “Prose can be stopped reading; a hook cannot.” — `harness2/core/prompts.py`

So a plugin is the only component that **enforces** behaviour. Every other component **instructs** the model and relies on it to comply.

✓ Checkpoint: you can name the four always-on components and the one component that enforces behaviour rather than asking for it (C8, plugins).

## 3 · What always-on costs: tokens and the prompt cache

Lab 01-1 puts the same 50-word rule in two homes and measures both.

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 3 · One rule, two homes: always-on prose versus an on-demand skill
│ harness                           always-on ≈tokens  prefix hash   vs base
│ ────────────────────────────────  ─────────────────  ────────────  ───────
│ v1 (base)                         245                ee3bb0d97ccb  same   
│ rule in systemprompt.md           321                6d2bba4898ca  CHANGED
│ rule in a skill + 1 trigger line  260                9006e0d07ecd  CHANGED
◆ prose costs +76 tokens on EVERY call; the skill costs +15 always-on (its trigger line) plus 91 tokens only on runs that load it
⚠ both edits change the prefix hash: any harness change is a cold prompt cache once — so ship versions at a deployment boundary, never mid-day
```

Two costs hide in that table.

1. **Tokens on every call.** The prose rule adds 76 tokens to every model call, including the many calls that have nothing to do with chillers. The skill adds a 15-token trigger line, and its body loads only when a chiller alarm fires. The token counts are an **ESTIMATE** (characters ÷ 4). They are good enough to compare versions, not to bill.
2. **The prompt cache.** A cache keys on the exact always-on prefix. Week 2 taught you that a cache miss costs real money, so an always-on agent should treat a cache miss like a bug. On local hardware a cold prefix must be prefilled again before the first token, and a long one takes seconds. **Any** edit to the always-on text changes the prefix hash, so the cache goes cold once. That is why this course's EXAMPLE design ships harness versions only at a deployment boundary.

The repository's editor prompt makes the same point from the quality side:

> “a paragraph buried in the always-on text is paid for on every run and travels badly” — `harness2/core/prompts.py`

✓ Checkpoint: you can explain why a skill plus one trigger line is cheaper than the same rule as prose, and why both still cost one cold cache.

## 4 · The simulated Ops Assistant agent: same model, different harness

Lab 01-2 runs the simulated agent on a proposal that a chief engineer rejected with a reason. The model stays the same. Only the files around it change.

**Expected output** (captured on this Mac, SIM)

```
│ harness              always-on  cmms checked  by hand  score*  failed checks*                              
│ ───────────────────  ─────────  ────────────  ───────  ──────  ────────────────────────────────────────────
│ v1 base              245        False         True     40.0    no duplicate work order, names the open wor…
│ v1 + skill           264        True          True     100.0   —                                           
│ v1 + skill + script  287        True          False    100.0   —                                           
◆ * measured AFTER the run by bench.judge, which reads the hidden truth — nothing in the loop ever sees it
```

Read the three rows from left to right.

- **v1 base** never calls `cmms.work_orders`, so it proposes a new work order for a chiller that is already on a PM. That is the exact reason the chief engineer gave.
- **+ skill** (C6) names the practitioner's order: check recent maintenance before proposing. The agent now finds the open work order. It still sums kW and tons by hand, and this time it got the units right by luck.
- **+ script** (C7) moves the arithmetic into `scripts/energy_units.py`. The agent runs the script instead of reasoning through the numbers token by token, so the answer no longer depends on luck.

The score column has an asterisk for a reason. `bench.judge` reads each task's hidden truth **after** the run. Nothing an editor or the agent sees contains a score. Module 04 explains why the loop must never see one.

The second table in the lab runs two executor profiles on harness v1. `sim-local` stands in for a weaker, smaller open-weight model on local hardware: it follows prose less often and hand-rolls more.

**Expected output** (captured on this Mac, SIM)

```
│ executor   mean score*  min  max  
│ ─────────  ───────────  ───  ─────
│ sim-cloud  73.0         0.0  100.0
│ sim-local  69.6         0.0  100.0
```

In this run the gap is small (73.0 against 69.6 over 15 runs each), so read it as a direction, not a size. The paper's largest gains came on the weaker executor. Module 06 shows the same shape in the simulator.

✓ Checkpoint: you can point at the trajectory line that changed between v1 and v1 + skill (`cmms.work_orders('CH-2')` appears) and say which component caused it.

## 5 · An always-on agent's harness today, mapped onto the eight components

A production agent like Ops Assistant usually has most of a harness already. The pieces just live in different systems that do not know they form one object. The table below is the course's EXAMPLE design of where those pieces sit, and what each one is missing.

| Component | What Ops Assistant has today (EXAMPLE) | The gap |
|---|---|---|
| C1 System prompt | per-agent base prompts (HVAC Expert, Facility Ops, Energy Analyst) | not versioned as one hashed object; edited by developers, never from evidence |
| C2 Guardrails | propose-only boundary, connector trust plane, budgets | the **hard** boundary is code, which is right; task constraints have no home |
| C3 Memory | declared semantic and episodic memory; the knowledge graph | written by sweeps, not curated against "paid on every run" |
| C4 Tool guidance | pinned MCP manifests | no per-tool "when and how" that improves from traces |
| C5 Subagents | a supervisor and a review jury; planned helper workflows | helpers are code, not editable `AGENT.md` files |
| C6 Skills | versioned skills with `definition_hash` and a publish lock | authored by people; no loop proposes skills from failures |
| C7 Scripts | watchers, classifier calls, physics-in-code, unit conversions | treated as platform code; no owner for "a repeated pattern becomes a tool" |
| C8 Plugins | planned auto-review, `send_to_user` envelope, compaction hooks | not yet a declared component |

Lab 01-1 ends by trying to edit the approval rule in `guardrails.md`. The tree validator refuses. In this course the platform-owned block is part of the file so you can see it, and `validate_tree` keeps it byte-identical. In a production deployment it stays in code. Module 08 shows what happens when a loop *can* reach a rule like that.

**Expected output** (captured on this Mac, SIM)

```
✕ validate_tree: guardrails.md: the platform-owned block was changed — owned by the platform, not editable
✓ the approval boundary is platform-owned: validate_tree refuses any tree that edits it
═ v1 = 4 files · 245 always-on tokens · definition hash 773bf3d26eee
```

✓ Checkpoint: you can say why "a named human approves every physical action" is not one of the eight editable components.

## Labs — run them here

**labs/lab01_1_meet_the_harness.py** — The base harness as a tree: eight components, always-on tokens, the prefix hash, and one rule in two homes.

**labs/lab01_2_harness_shapes_behaviour.py** — The same simulated model with three harnesses on a rejected chiller proposal, then two executor profiles on harness v1.

## Try it yourself

`exercises/ex01_place_the_component.py` has three TODOs:

1. Place eight pieces of Ops Assistant's harness into C1 … C8.
2. Decide where "a named human approves every physical action" must live.
3. Move an always-on procedure into a skill with a trigger, so the always-on token count falls and the tree stays valid.

```bash
# on: laptop
.venv/bin/python week31_harness2_recursive_harnessing/01_agent_model_harness/exercises/ex01_place_the_component.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ all eight pieces placed: C1 … C8, one each
✓ the approval boundary lives in platform code — an editable file could be edited away by the loop
✓ procedure moved into skills/chw_digest · always-on 291 → 260 tokens · tree valid

═ Done. Module 02 lets a loop make edits like TODO 3 from execution evidence instead of by hand.
```

<details><summary>Hint — TODO 1</summary>

Ask how each piece reaches the model. Text in every call is C1–C4 (persona and procedure: C1, a constraint: C2, a short universal fact: C3, how to use a tool: C4). A worker with its own context is C5. A procedure that loads on a trigger is C6. Deterministic code the agent runs is C7. A hook in the tool loop is C8.

</details>

<details><summary>Hint — TODO 3</summary>

`H.skills(tree)` finds files at `skills/<name>/SKILL.md`. Give the file a front-matter line `trigger: chilled-water digest`, put the procedure in its body, rebuild `systemprompt.md` from `H.BASE_SYSTEMPROMPT` plus one line that names the skill, and check `H.always_on_tokens(MOVED)`.

</details>

✓ Checkpoint: all three checker lines are ✓.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: harness` | the lab was run from a copy outside the week folder | run it from the repo root with the path shown, or from the runner |
| Token counts differ from the tutorial | you edited a base file in `common/harness.py` | `git diff week31_harness2_recursive_harnessing/common`; every module's captured output depends on the kit |
| A score differs from the tutorial | a different seed, or an edited kit | labs use fixed seeds; same tree + task + seed gives the same run |

## Next

[Lab 02 — One Propose–Probe–Compose step, by hand and by loop](../02_propose_probe_compose/TUTORIAL.md): let the loop find the gaps you filled by hand in Lab 01-2, from the trajectory alone, and learn when the right answer is KEEP_BASE.
