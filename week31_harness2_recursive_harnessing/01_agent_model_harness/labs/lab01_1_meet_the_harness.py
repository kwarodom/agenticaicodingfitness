#!/usr/bin/env python3
"""Lab 01-1 · Meet the harness: eight components, what each costs, and what the prompt cache keys on.

Prints the HVAC Expert agent's base harness (v1) as a tree, sorts every file into C1–C8, counts the always-on
tokens, then shows what happens to the always-on budget and the prefix hash when the same 50-word rule is put
in the system prompt versus in a skill. No model is called.

    .venv/bin/python week31_harness2_recursive_harnessing/01_agent_model_harness/labs/lab01_1_meet_the_harness.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import harness as H                                                      # noqa: E402
from h2out import banner, note, ok, result, show_text, step, table, warn  # noqa: E402

banner("Lab 01-1 · Meet the harness", "Agent = Model + Harness · the eight components · always-on cost", brain=False)

step(1, "The eight Harness² components, by how they reach the model")
table([[c, name, path, reach] for c, name, path, reach in H.COMPONENTS], ["code", "component", "file surface", "reaches the model"])

step(2, "Harness v1 of the HVAC Expert agent (Ops Assistant's starting harness, EXAMPLE)")
base = H.base_harness()
for p in H.ALWAYS_ON:
    show_text(p, base[p])
table(H.summary_rows(base), ["path", "comp.", "reach", "≈tokens"])
note(f"always-on tokens ≈ {H.always_on_tokens(base)} (ESTIMATE: characters / 4) · prefix hash {H.prefix_hash(base)}")
note("C5–C8 are empty: no subagents, skills, scripts or plugins yet — the gaps the loop will find in Module 02")

step(3, "One rule, two homes: always-on prose versus an on-demand skill")
RULE = ("When a chiller alarm arrives, compare the condenser approach with the O&M manual's limit, check open and "
        "recent work orders before proposing anything, and write four sections: diagnosis, evidence, recent "
        "maintenance and proposal. Every ActionProposal links the alarm, the telemetry window and the manual.")
as_prose = dict(base)
as_prose["systemprompt.md"] = base["systemprompt.md"] + RULE + "\n"
as_skill = dict(base)
as_skill["skills/chiller_fault_review/SKILL.md"] = f"---\nname: chiller_fault_review\ntrigger: chiller alarm\n---\n{RULE}\n"
as_skill["systemprompt.md"] = base["systemprompt.md"] + "- For a chiller alarm, load the skill chiller_fault_review.\n"
rows = []
for name, tree in (("v1 (base)", base), ("rule in systemprompt.md", as_prose), ("rule in a skill + 1 trigger line", as_skill)):
    rows.append([name, H.always_on_tokens(tree), H.prefix_hash(tree), "same" if H.prefix_hash(tree) == H.prefix_hash(base) else "CHANGED"])
table(rows, ["harness", "always-on ≈tokens", "prefix hash", "vs base"])
d_prose = H.always_on_tokens(as_prose) - H.always_on_tokens(base)
d_skill = H.always_on_tokens(as_skill) - H.always_on_tokens(base)
note(f"prose costs +{d_prose} tokens on EVERY call; the skill costs +{d_skill} always-on (its trigger line) "
     f"plus {H.tokens(as_skill['skills/chiller_fault_review/SKILL.md'])} tokens only on runs that load it")
warn("both edits change the prefix hash: any harness change is a cold prompt cache once — so ship versions at a "
     "deployment boundary, never mid-day")

step(4, "1,000 routine wakes a day: what the always-on text alone costs (EXAMPLE arithmetic)")
wakes, calls = 1000, 3
for name, tree in (("v1", base), ("prose", as_prose), ("skill", as_skill)):
    print(f"│ {name:<6} {H.always_on_tokens(tree) * calls * wakes:>10,} always-on tokens/day")
ok("same rule, smaller daily bill when it lives in a skill — and the skill is a named, reusable file")

step(5, "What the editor may never touch")
print(H.platform_block(base["guardrails.md"]).rstrip())
broken = dict(base)
broken["guardrails.md"] = base["guardrails.md"].replace("approved by a named human", "approved automatically")
errs = H.validate_tree(broken)
for e in errs:
    print(f"✕ validate_tree: {e}")
ok("the approval boundary is platform-owned: validate_tree refuses any tree that edits it") if errs else warn("not caught?!")
result(f"v1 = {len(base)} files · {H.always_on_tokens(base)} always-on tokens · definition hash {H.definition_hash(base)}")
