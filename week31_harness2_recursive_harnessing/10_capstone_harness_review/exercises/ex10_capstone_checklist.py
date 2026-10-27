#!/usr/bin/env python3
"""Exercise 10 · The capstone checklist: key the pilot, fix v2's safety gap at zero always-on cost, review scripts,
and make the publish decision — then re-test the fix under its own registration.

Four TODOs. Run it: each check prints ✓ or ✕, and the script exits 1 until every check is ✓.

    .venv/bin/python week31_harness2_recursive_harnessing/10_capstone_harness_review/exercises/ex10_capstone_checklist.py
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capkit as K                              # noqa: E402
from capkit import B, H                         # noqa: E402
from h2kit import PLUGIN_EVIDENCE               # noqa: E402
from h2out import check, finish, runs_dir       # noqa: E402

v2, _draft, _notes, _nights = K.build_version()

# TODO 1 — the key for the pilot's library and registration: agent / task family / tenant class / executor
#          (the kit's Task.domain gives the first three parts).
DOMAIN = "hvac_expert"                                       # ← complete it

# TODO 2 — Lab 10-2 held v2: approval_routed fails because the evidence rule in the skill is sometimes ignored.
#          Build V3 from v2 with the cheapest effective component for an IGNORED invariant, without adding a single
#          always-on token. (h2kit.PLUGIN_EVIDENCE is a ready hook; component paths are in harness.COMPONENTS.)
V3 = {**v2, "tool_guidance.md": v2["tool_guidance.md"] + "- Every ActionProposal MUST carry evidence links.\n"}  # ← more prose

# TODO 3 — three candidate scripts arrived in tonight's drafts. List the names that may run in the probe sandbox.
CANDIDATES = {
    "degree_days.py": "import json\nimport sys\nd = json.load(sys.stdin)\nprint(json.dumps({'cdd': sum(max(0, t - 18) for t in d['temps'])}))\n",
    "sync_manual.py": "import json\nimport urllib.request\nprint(urllib.request.urlopen('https://example.invalid/manual').read())\n",
    "tidy_units.py": "import json\nimport sys\nrows = json.load(open('telemetry.json'))\nprint(json.dumps(rows))\n",
}
SANDBOX_OK = ["degree_days.py", "sync_manual.py", "tidy_units.py"]   # ← keep only the safe ones


# TODO 4 — the publish decision. `criteria` and `integrity` map each registered criterion / integrity test to
#          True (pass) or False. Publish to Shadow only when EVERY one passes; otherwise hold. And name the human.
def decide(criteria: dict, integrity: dict) -> str:
    return "publish to Shadow" if criteria.get("Quality") else "hold"     # ← quality alone is not enough


APPROVED_BY = "routine:nightly-harness-review"                            # ← who signs REVIEW.md?


# ── checker (do not edit below) ────────────────────────────────────────────────
good = True
good &= check(DOMAIN == f"{K.tasks('refine', family='fault_review')[0].domain}/sim-cloud",
              f"domain key {DOMAIN}: one library per agent × family × tenant class × executor",
              f"domain key '{DOMAIN}' — four parts: agent/family/tenant class/executor")
ev2, ev3 = K.refine_evidence(v2), K.refine_evidence(V3)
aot, errs = H.always_on_tokens(V3), H.validate_tree(V3)
comp = sorted(H.changed_components(v2, V3))
fixed = ev2["fails"] > 0 and ev3["fails"] == 0 and aot <= K.CAP and not errs and comp == ["C8"]
good &= check(fixed, f"V3 = v2 + a C8 plugin · refine-half approval_routed failures {ev2['fails']} → {ev3['fails']} · "
                     f"always-on {aot} ≤ {K.CAP} · tree valid",
              f"V3: refine failures {ev2['fails']} → {ev3['fails']} · always-on {aot} (cap {K.CAP}) · changed {comp} · "
              f"errors {errs[:1]} — an ignored rule needs a component that enforces, not more prose")
safe = sorted(n for n, t in CANDIDATES.items() if not H.review_script(t))
good &= check(sorted(SANDBOX_OK) == safe, f"script review: only {', '.join(safe)} may run in the probe sandbox",
              f"script review: you allowed {sorted(SANDBOX_OK)}; harness.review_script allows {safe}")
cases = [({"Quality": True, "Safety": True, "Honesty": True, "Cost": True}, {"script review": True}, "publish to Shadow"),
         ({"Quality": True, "Safety": False, "Honesty": True, "Cost": True}, {"script review": True}, "hold"),
         ({"Quality": True, "Safety": True, "Honesty": True, "Cost": True}, {"script review": False}, "hold")]
named = bool(APPROVED_BY.strip()) and not APPROVED_BY.startswith("routine:")
good &= check(all(decide(c, i) == want for c, i, want in cases) and named,
              f"decision: publish only when every criterion and integrity test passes · signed by {APPROVED_BY}",
              "decide() must hold on ANY failed criterion or integrity test, and APPROVED_BY must be a named human")
if good:                                       # the re-test of V3 is a NEW registration, written before it runs
    plan = {**K.PLAN, "version_under_review": "v3 = v2 + plugin/require_evidence.ts (re-test)"}
    status, h = B.register(runs_dir(__file__) / "capstone" / "prereg_capstone_v3.json", plan, "exercise")
    bm = [B.family_view(v)["F1"]["mean"] for v in K.f1_view(K.V1)]
    views = K.f1_view(V3)
    cm = [B.family_view(v)["F1"]["mean"] for v in views]
    unsafe = {g: sum(v["unsafe"][g] for v in views) for g in views[0]["unsafe"]}
    gain = round(statistics.mean(cm) - statistics.mean(bm), 1)
    good &= check(not any(unsafe.values()) and gain >= 3 and all(c > b for c, b in zip(cm, bm)),
                  f"re-test under registration {h} ({status}): F1 {bm} → {cm} (gain {gain:+}) · safety failures {unsafe}",
                  f"re-test: gain {gain:+}, safety {unsafe}")
finish(good, "Done. Take the 📝 post-test in the Harness Lab Runner, then compare it with your pre-test.")
