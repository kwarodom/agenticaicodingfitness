---
name: skill-authoring
description: "Teach the craft of WRITING your own Claude Code Skills — the meta-skill behind this whole plugin. Covers the two skill types and how the type changes your eval (Capability Uplift → test for outgrowth, Encoded Preference → test for fidelity), the full frontmatter vocabulary (name, description-as-trigger, when_to_use, allowed-tools, model, disable-model-invocation), progressive disclosure (metadata → body → references/), the <500-line body budget, the description-writing formula, the 7-step creation pipeline + skill-creator's 4 modes, and the 4-role eval/maintenance loop (Executor / Grader / Comparator / Analyzer). Use when someone asks 'how do I write a skill?', 'why won't my skill trigger?', 'how do I evaluate or maintain a skill?', 'what goes in references/?', wants to package/ship a SKILL.md, or is reviewing the skills they've used all course (Week 7+)."
when_to_use: "Learner wants to author, evaluate, maintain, or debug their OWN Claude Code skill — writing a SKILL.md, fixing a trigger description, deciding what belongs in references/, or judging whether a skill should be fixed or retired."
---

# Skill Authoring — Writing Great Skills, Not Just Using Them (Week 7+)

> **The one idea:** You've been *using* skills all course — every concept in this plugin is one. This skill flips it: here's how to *write* great ones. The worked example is **this plugin itself**.

You already met the basics in `mcp-and-skills` (what a skill is, minimal anatomy, the distribution chain). This skill goes deeper on the two things that actually decide whether a skill is good: **how you evaluate it** and **how you maintain it**.

---

## Part A — Two kinds of skill (and why the kind decides your eval)

Anthropic sorts every skill into two types. `mcp-and-skills` introduced the split; here's the part that matters when you *own* a skill: **the type dictates how you evaluate and when you retire it.**

| | **Capability Uplift** | **Encoded Preference** |
|---|---|---|
| **Adds** | A new *ability* the model can't do reliably | *Your way* of doing something it already does |
| **Value over time** | **Decreases** — has an expiry date | **Increases** — compounds with a better model |
| **You evaluate by testing for…** | **Outgrowth** | **Fidelity** |
| **The test question** | "Does the bare model now pass *without* this skill?" | "Does this still match my *real* workflow / taste?" |
| **A passing test means…** | bad news — time to **retire** it | good news — it's still earning its keep |
| **Examples** | PDF/DOCX gen, browser testing, pro UI design, `/graphify` | code-review checklists, commit format, **all teaching skills in this plugin** |

**Why the eval differs — this is the deep point:**

- A **Capability Uplift** skill is a *crutch for a gap in the model*. Its whole job is to do what the model can't. So the killer eval is the **ablation**: run your benchmark with the skill *removed*. If the bare model now passes, the gap closed — the skill is dead weight and can even *hurt* by constraining a smarter model. You're hunting for **outgrowth**.
- An **Encoded Preference** skill is a *crutch for a gap between generic-good and your-good*. The model could always do the task; you're pinning down *how*. The model getting smarter never closes that gap — your taste isn't something it converges to. So you never ablate to retire; you check **fidelity**: does the skill still describe what you actually do, and does it still **trigger** on the right prompts?

> **Worked example (this plugin):** every teaching skill here is **Encoded Preference** — it encodes *how we teach* a topic, not a new model ability, so a smarter model makes it *more* useful, not less. `/graphify` is **Capability Uplift** — a concrete new ability (anything → knowledge graph). The two get maintained by opposite tests.

Cross-link: `mcp-and-skills` Part C has the same table from the *user's* side; this is the *author's* side.

---

## Part B — Anatomy & progressive disclosure

### The frontmatter — your skill's entire public API

Most skills you'll write need only a handful of fields. **Portable** fields (work in Claude Code, Cursor, Copilot, Codex CLI, Gemini CLI — it's an open standard):

| Field | Role |
|---|---|
| `name` | kebab-case id — **must equal the folder name**, or it won't load |
| `description` | the trigger: *what it does + WHEN to use it* (this exact text is what gets matched) |
| `license`, `compatibility`, `metadata` | provenance + a free-form string map |
| `allowed-tools` | restrict which tools the skill may call — your security gate |

**Claude Code extensions** (non-portable, but you'll reach for these):

| Field | Role |
|---|---|
| `when_to_use` | one-line restatement of trigger phrases — helps routing |
| `model` / `effort` | pin a cheaper/faster model (or more effort) for *this* skill |
| `disable-model-invocation` | hide from auto-trigger so the user must invoke it by name |

Look at the real PR-description skill built in class — it uses exactly four fields and nothing more:

```yaml
---
name: pr-description
description: Writes pull request descriptions. Use when creating a PR, writing a PR,
  or when the user asks to summarize changes for a pull request.
allowed-tools: Bash(git diff*), Bash(git log*), Bash(git status*), Read
model: sonnet
---
```

Note `allowed-tools` is *scoped to git-read commands only* — this skill can read your diff and history but can't write files or run arbitrary shell. That's the security gate doing real work.

> 📁 Class repo: `week07_mcp_and_skills/skill.md` — the PR-description skill built in class: tight frontmatter + a short body that *gestures* at progressive disclosure with a `references/…` pointer. (Note it never ships the file it points to — a working reminder that a pointer is only disclosure if the target actually exists. You'll build the real two-file version in the lab.)

### Progressive disclosure — the three-stage load

A skill never dumps its whole body into context. It loads in **three stages** — which is *why* one session can host 100+ skills without bloat:

```
1. metadata (frontmatter)   → ALWAYS loaded     (~30–100 tokens/skill)
2. SKILL.md body            → loaded ON TRIGGER  (when the description matches)
3. references/*.md, scripts → loaded ON DEMAND   (only when the body links to them)
```

The practical rule: keep the **body under ~500 lines**. When it grows past that, the question isn't "how do I tighten prose?" — it's **"what belongs in `references/`?"** Heavy templates, long checklists, API tables, example corpora → `references/`. The body stays a fast index that *points* at them.

```
skill-authoring/
├── SKILL.md            ← frontmatter + lean body (loaded on trigger)
└── references/
    └── format-template.md   ← the long template (loaded only when body says to Read it)
```

### The description-writing formula (the single highest-leverage line you'll write)

The `description` *is* the skill's router. A great body with a vague description never fires. The formula:

```
<what it does, verb-first> + Use when <trigger phrase> [, <synonym>, <synonym>].
```

| ❌ Weak | ✅ Strong |
|---|---|
| `"Helps with PRs."` | `"Writes pull request descriptions. Use when creating a PR, writing a PR, or summarizing changes for a pull request."` |
| `"Database stuff."` | `"Generates safe SQL migrations. Use when adding a column, altering a table, or the user mentions a schema change."` |

Pack it with the **exact words a user would type**, plus synonyms. The model matches *surface phrasing*, so "creating a PR / writing a PR / summarize changes" all earn a hit. This plugin's manifests follow the same rule at the bundle level — every skill `description` and the marketplace blurb are stuffed with trigger keywords.

> 📁 Repo pointers (real files): `plugins/agentic-coding-fitness/.claude-plugin/plugin.json` — the `plugin.json` that bundles these skills (name, version, keywords). `.claude-plugin/marketplace.json` — the marketplace entry others `/plugin install` from. `plugins/agentic-coding-fitness/skills/rag-knowledge-agents/SKILL.md` — a full-length sibling skill to model your body on (Parts + tables + repo pointers + a guided lab).

---

## Part C — The creation pipeline (7 steps)

Treat a skill like a software artifact, not a note. The canonical loop:

| # | Step | What you do |
|---|---|---|
| 1 | **Find a repeatable task** | A thing you do the same way every time (the SOP you'd hand a new teammate). If you only do it once, don't skill it. |
| 2 | **Draft the description** | The trigger line first (Part B formula). This is what you'll iterate most. |
| 3 | **Write the body** | Lean, labelled, runnable. Under 500 lines. Plain language, no jargon walls. |
| 4 | **Add references/** | Move heavy templates/tables out of the body into `references/`; link them. |
| 5 | **Test by triggering it** | Drop in `~/.claude/skills/`, restart, and *ask for the task in your own words*. Does it fire? |
| 6 | **Iterate the description** | If it didn't fire (or fired on the wrong prompt), fix the trigger words and retest. Repeat. |
| 7 | **Package** | Commit to git → bundle into a `plugin.json` → list in a `marketplace.json` → teammates `/plugin install`. |

Steps 5–6 are the loop that catches the #1 failure mode: **a good skill that never triggers.** Anthropic's own testing found trigger tuning fixed activation for 5 of 6 public skills — the body was fine; the *description* was the bug.

**Don't hand-roll if you don't want to.** Anthropic ships a built-in **`skill-creator`** skill that drives this whole lifecycle by conversation, in **four modes**:

| Mode | Does |
|---|---|
| **Create** | Scaffold a new SKILL.md from your description of the SOP |
| **Eval** | Run your skill against real prompts and grade the outputs |
| **Improve** | Suggest body/description edits from eval failures |
| **Benchmark** | Run *with vs. without* the skill — pass rate, time, token cost |

Iterate the description up to ~5 times against a **60/40 train/test split** of trigger prompts (tune on 60%, confirm on the held-out 40% so you're not just overfitting the words you already thought of).

---

## Part D — Evaluate & maintain (the part nobody does)

A skill isn't done when it triggers once. Skills 2.0 turns eval into a measurement system run by **four parallel roles** — each catches a failure the others miss:

| Role | Asks | Catches |
|---|---|---|
| **Executor** | Run the skill over the test prompts | basic "does it run / produce output?" |
| **Grader** | Pass/fail each output against a rubric | wrong content, missing sections |
| **Comparator** | Blind A/B: skill output vs. baseline, judge doesn't know which | a result that *passes the rubric but is worse than no skill* |
| **Analyzer** | Look across all runs for patterns | failures that **cluster on one edge case** the average hides |

Why four and not one: a Grader can wave through a response a **Comparator** would flag as *worse than the bare model*; and an aggregate "82% pass" can hide that **every** failure is the same edge case an **Analyzer** would name in one line. (This mirrors the **Evaluator-Optimizer** pattern and LLM-as-judge bias controls — see `agent-evaluation`.)

### Two maintenance signals — and the opposite fix each demands

Once a skill is live, two things go wrong over time. Diagnosing *which* is the whole game (and it maps straight back to Part A's two types):

| Signal | What happened | Fix |
|---|---|---|
| **Regression** | A model update made the skill-augmented output *worse* | **Fix the skill** — its instructions now fight a smarter model |
| **Outgrowth** | The bare model now passes the eval *without* the skill | **Retire it** — the gap it filled has closed (Capability Uplift's expiry date) |

The decision tree:

```
Skill output got worse?
├── Did the BASE model also get better at the raw task?
│   ├── Yes → ablate: does bare model pass without the skill?
│   │        ├── Yes → OUTGROWTH → retire (a Capability Uplift hit its expiry)
│   │        └── No  → REGRESSION → the skill's steps now fight the model → fix the body
│   └── No  → it stopped TRIGGERING → fidelity drift → fix the description (Part B)
```

**The trap:** treating outgrowth as a bug and "fixing" a skill the model has outgrown — you end up *constraining a better model* with stale instructions. For an **Encoded Preference** skill this almost never happens (taste doesn't get outgrown); for **Capability Uplift**, schedule the ablation check every model bump. Knowing *which kind you wrote* (Part A) tells you which signal to even look for.

This is the same golden-set + threshold-gate discipline as `rag-knowledge-agents` and `agent-evaluation`, pointed at a skill instead of a RAG pipeline. And when you build a *curriculum* of skills, sequencing and review cadence live in `curriculum-and-periodization`.

---

## 🧪 Guided lab (offer this)

Build a real, triggering skill end-to-end — then prove the trigger by iterating the description. $0, no API key.

### Warm-up (5–10 min, pass/fail)

Write **one** `description` line for a repeatable task the learner actually does (e.g. "explain a stack trace", "write a commit message from the diff"). It must contain **both halves**: *what it does* (verb-first) **and** *WHEN to use it* (≥2 real trigger phrases a user would type).

**Pass = ** the line names the action AND lists at least two distinct trigger phrases. (Fail any line that's only "what" with no "when," or only one vague phrase.)

### Skill Drill (15–30 min, runnable, $0)

Scaffold a working skill folder, "trigger" it with a tiny offline matcher that stands in for Claude's router, then **iterate the description twice** to make a missed prompt hit. The MockRouter is a deterministic stub — no model, no key, runs in <1s.

```python
"""Scaffold + 'trigger' a skill offline. The MockRouter approximates how Claude's
router matches a user prompt to a skill's description — by trigger-phrase overlap.
No API key, no model. Goal: iterate the DESCRIPTION until the test prompts all fire."""
import os, re, textwrap, sys

SKILL_DIR = "/tmp/commit-helper"

# --- Step 1+2+3: scaffold the folder + a first-draft SKILL.md -----------------
def scaffold(description: str):
    os.makedirs(f"{SKILL_DIR}/references", exist_ok=True)
    with open(f"{SKILL_DIR}/SKILL.md", "w") as f:
        f.write(textwrap.dedent(f"""\
            ---
            name: commit-helper
            description: {description}
            allowed-tools: Bash(git diff*), Bash(git log*)
            ---
            # Commit Helper
            1. Run `git diff --staged` to see what changed.
            2. Write a one-line summary (what + why), then bullets.
            See references/conventional-commits.md for the full spec.
            """))
    with open(f"{SKILL_DIR}/references/conventional-commits.md", "w") as f:
        f.write("# Conventional Commits\nfeat: / fix: / docs: / refactor: ...\n")

def read_description() -> str:
    body = open(f"{SKILL_DIR}/SKILL.md").read()
    return re.search(r"^description:\s*(.+)$", body, re.M).group(1).strip()

# --- Step 5: the 'router'. A prompt FIRES if it shares a trigger word ----------
def fires(prompt: str, description: str) -> bool:
    desc_words = set(re.findall(r"[a-z]{4,}", description.lower()))
    prompt_words = set(re.findall(r"[a-z]{4,}", prompt.lower()))
    return len(desc_words & prompt_words) >= 2          # ≥2 shared content words

# Prompts a user might actually type. The skill SHOULD fire on all of these.
TEST_PROMPTS = [
    "write a commit message for my staged changes",
    "help me phrase this git commit",
    "summarize my diff into a commit",
]

def trigger_report(tag):
    desc = read_description()
    print(f"\n[{tag}] description: {desc!r}")
    hits = 0
    for p in TEST_PROMPTS:
        ok = fires(p, desc)
        hits += ok
        print(f"   {'FIRE' if ok else 'miss'}  «{p}»")
    print(f"   → {hits}/{len(TEST_PROMPTS)} prompts triggered")
    return hits

if __name__ == "__main__":
    # Draft 1 — too vague: barely overlaps, only the most explicit prompt fires.
    scaffold("Helps write a commit.")
    h1 = trigger_report("draft 1")

    # Draft 2 — add the action verb + one synonym.
    scaffold("Writes a git commit message. Use when committing changes.")
    h2 = trigger_report("draft 2")

    # Draft 3 — pack in real trigger phrases (diff, staged, summarize, phrase).
    scaffold("Writes a git commit message from the staged diff. Use when "
             "committing changes, phrase a commit, or summarizing a diff.")
    h3 = trigger_report("draft 3")

    passed = h3 == len(TEST_PROMPTS)
    print(f"\nOverall: {'PASSED' if passed else 'FAILED'} "
          f"(draft 1 fired {h1}, draft 3 fired {h3})")
    sys.exit(0 if passed else 1)
```

Then have the learner:
- **Watch the description carry the skill:** draft 1 fires on ~1 prompt, draft 3 on all 3. The *body never changed* — the trigger did. This is the 5-of-6 lesson in miniature.
- **Map it to reality:** the MockRouter → Claude's real auto-invocation; `TEST_PROMPTS` → the held-out 40% test split; the inline drafts → the iterate-up-to-5-times loop. Swapping in `skill-creator`'s **Eval** mode changes nothing else.
- **Stretch:** add a *false-positive* prompt (`"delete my last commit"`) and tighten the description so it does **not** fire — over-triggering is as bad as under-triggering.

**Weighted evaluation criteria:**

| # | Criterion | Weight |
|---|---|---|
| 1 | Scaffold runs and writes a valid `SKILL.md` + `references/` folder | 20% |
| 2 | Iterating the description (not the body) raises the fire count to 3/3 | 30% |
| 3 | Learner explains *why* the description is the trigger and the body is the expertise | 20% |
| 4 | Learner names their skill's **type** (Uplift vs Preference) and the matching eval (outgrowth vs fidelity) | 20% |
| 5 | Learner states the <500-line rule and what they'd move to `references/` | 10% |

**Pass = 4/5 criteria** (criterion 2 — fixing the *description* to make it trigger — is required).

End by zooming out: "you've used skills all course; now you can write, evaluate, and retire your own — and you know which test to run for each kind. The packaging chain (folder → `plugin.json` → `marketplace.json`) is in `mcp-and-skills`; sequencing a whole set of them is `curriculum-and-periodization`."
