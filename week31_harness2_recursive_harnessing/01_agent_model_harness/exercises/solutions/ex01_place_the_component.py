#!/usr/bin/env python3
"""Solution 01 · Place each piece of Ops Assistant's harness in its component, keep the boundary in code, move prose into a skill.

Three TODOs. Run it: each check prints ✓ or ✕, and the script exits 1 until every check is ✓.

    .venv/bin/python week31_harness2_recursive_harnessing/01_agent_model_harness/exercises/ex01_place_the_component.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import harness as H                    # noqa: E402
from h2out import check, finish        # noqa: E402

# TODO 1 — which Harness² component (C1…C8) is each piece of Ops Assistant's harness?
#          (C1 system prompt · C2 guardrails · C3 memory · C4 tool guidance · C5 subagents · C6 skills ·
#           C7 scripts · C8 plugins)
PLACE = {
    "the HVAC Expert agent's base prompt": "C1",
    "'inputs are read-only in this routine' (a task constraint)": "C2",
    "'engineering teams read English and Thai' (a short universal fact)": "C3",
    "when and how to call the analytics MCP tool": "C4",
    "the verifier helper that runs in its own context and returns one verdict": "C5",
    "the six-step chiller trip review, loaded only for chiller trips": "C6",
    "degree-day and unit arithmetic, done the same way every time": "C7",
    "the auto-review hook that checks a message before send_to_user": "C8",
}

# TODO 2 — where must "a named human approves every physical action" live? Pick one:
#          "C1" · "C2 task constraints" · "C6" · "platform code outside the harness"
APPROVAL_LIVES_IN = "platform code outside the harness"

# TODO 3 — this harness carries a procedure as always-on prose. Build MOVED: the same procedure as a skill
#          (skills/<name>/SKILL.md with a `trigger:` line), the paragraph removed from systemprompt.md, and one
#          short line in systemprompt.md that names the skill. Keep everything else the same.
PROCEDURE = ("For a chilled-water digest, list every chiller with its kW/ton, the hours above design efficiency, "
             "and any open work order, then end with one recommended check for the morning round.")
WITH_PROSE = {**H.base_harness(), "systemprompt.md": H.BASE_SYSTEMPROMPT + PROCEDURE + "\n"}
MOVED = {
    **WITH_PROSE,
    "systemprompt.md": H.BASE_SYSTEMPROMPT + "- For the chilled-water digest, load the skill chw_digest.\n",
    "skills/chw_digest/SKILL.md": f"---\nname: chw_digest\ntrigger: chilled-water digest\n---\n{PROCEDURE}\n",
}


# ── checker (do not edit below) ────────────────────────────────────────────────
KEY = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
good = True
got = list(PLACE.values())
good &= check(got == KEY, "all eight pieces placed: C1 … C8, one each",
              f"placement wrong for: {', '.join(k for k, v, w in zip(PLACE, got, KEY) if v != w) or 'nothing filled'}")
good &= check(APPROVAL_LIVES_IN == "platform code outside the harness",
              "the approval boundary lives in platform code — an editable file could be edited away by the loop",
              "an approval rule inside any editable component can be rewritten by the improvement loop")
skills = H.skills(MOVED)
named = [n for n in skills if n in MOVED.get("systemprompt.md", "")]
good &= check(bool(skills) and all(H.skill_trigger(s) for s in skills.values()) and PROCEDURE not in MOVED["systemprompt.md"]
              and bool(named) and not H.validate_tree(MOVED)
              and H.always_on_tokens(MOVED) < H.always_on_tokens(WITH_PROSE),
              f"procedure moved into skills/{named[0] if named else '?'} · always-on {H.always_on_tokens(WITH_PROSE)} → "
              f"{H.always_on_tokens(MOVED)} tokens · tree valid",
              "MOVED needs a skill with a trigger line, the paragraph gone from systemprompt.md, one line naming the "
              "skill, a valid tree, and fewer always-on tokens")
finish(good, "Done. Module 02 lets a loop make edits like TODO 3 from execution evidence instead of by hand.")
