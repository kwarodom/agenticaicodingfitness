#!/usr/bin/env python3
"""Lab 01-2 · Same model, different harness: watch the harness change what the agent does.

Runs the SIMULATED Ops Assistant agent (a teaching model, not an LLM) on one rejected chiller proposal three times:
with harness v1, with v1 plus a hand-written skill, and with v1 plus a skill and a script. Then both executor
profiles (sim-cloud, sim-local) run five tasks on v1. Scores come from bench.judge AFTER the runs — the agent
never sees them.

    .venv/bin/python week31_harness2_recursive_harnessing/01_agent_model_harness/labs/lab01_2_harness_shapes_behaviour.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import harness as H                                       # noqa: E402
from bench import checks, judge                           # noqa: E402
from ops_sim import execute, task, tasks              # noqa: E402
from h2kit import SCRIPT_ENERGY, skill_fault_review       # noqa: E402
from h2out import banner, dim, note, result, step, table  # noqa: E402

banner("Lab 01-2 · Same model, different harness", "the SIM executor on a rejected chiller proposal", brain=False)
dim("SIM executor = a deterministic teaching model of an agent with known weaknesses (common/ops_sim.py)")

t = task("F1-01")
step(1, f"The task the nightly routine picked: {t.id}")
print(f"│ request : {t.request}")
print(f"│ signal  : {t.signal} · reviewer's reason: \"{t.human_reason}\"")

base = H.base_harness()
with_skill = {**base, "skills/chiller_fault_review/SKILL.md": skill_fault_review(t),
              "systemprompt.md": base["systemprompt.md"] + "- For a chiller alarm or fault review, load the skill chiller_fault_review.\n"}
with_script = {**with_skill, "scripts/energy_units.py": SCRIPT_ENERGY,
               "tool_guidance.md": base["tool_guidance.md"] + "- kW/ton: run python scripts/energy_units.py on the telemetry rows; never sum them by hand.\n"}

step(2, "Three harnesses, same model (sim-cloud), same seed")
rows = []
for name, tree in (("v1 base", base), ("v1 + skill", with_skill), ("v1 + skill + script", with_script)):
    r = execute(tree, t, seed=0)
    r.show(max_lines=14)
    failed = [c for c, okv in checks(t, r) if not okv]
    rows.append([name, H.always_on_tokens(tree), r.signals.get("cmms_checked"), r.signals.get("hand_arithmetic"),
                 judge(t, r), ", ".join(failed) or "—"])
table(rows, ["harness", "always-on", "cmms checked", "by hand", "score*", "failed checks*"], maxw=44)
note("* measured AFTER the run by bench.judge, which reads the hidden truth — nothing in the loop ever sees it")

step(3, "Two executors on v1: the weaker one leaves more on the table (five tasks × three seeds)")
pick = [x for x in tasks("refine") if x.id in ("F1-01", "F1-02", "F2-01", "F3-00", "F7-00")]
rows = []
for ex in ("sim-cloud", "sim-local"):
    scores = [judge(x, execute(base, x, seed=s, executor=ex)) for x in pick for s in range(3)]
    rows.append([ex, f"{sum(scores) / len(scores):.1f}", min(scores), max(scores)])
table(rows, ["executor", "mean score*", "min", "max"])
note("sim-local stands in for a smaller open-weight model on local hardware: it follows prose less often and hand-rolls more")
result("the model never changed; only the files around it did — that is the harness")
