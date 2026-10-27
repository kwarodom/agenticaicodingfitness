#!/usr/bin/env python3
"""Lab 05-3 · The second hotel: a library from Harbour View, six fault reviews at Chelsea, with and without re-verifying facts.

Builds a fault-review library on the first hotel (Harbour View Hotel: condenser approach limit 3.0 K), then runs
the second hotel's six fault reviews (Chelsea Riverside Hotel: limit 4.5 K) twice from copies of that library:
once with the editor re-verifying copied workspace facts against the task's own manual (reverify=True), once
copying them unchecked (reverify=False). Scores are measured after each step by bench.judge. A short seed sweep
shows how often the effect appears. No model is called.

    .venv/bin/python week31_harness2_recursive_harnessing/05_library_and_experience/labs/lab05_3_second_hotel.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from bench import stream                                                # noqa: E402
from ops_sim import MANUAL, SITES, tasks                            # noqa: E402
from h2kit import reverify_facts                                        # noqa: E402
from h2out import banner, dim, note, ok, result, runs_dir, step, table, warn  # noqa: E402

QUOTE_REVERIFY = ("RE-VERIFY its workspace facts against the evidence in front of you: an earlier "
                  "task's mapping or threshold is a hypothesis here, not a fact.")
QUOTE_MEMORY = "Never store this matter's facts as durable memory: they are wrong anchors for the next task."

banner("Lab 05-3 · The second hotel", "a library from Harbour View, six fault reviews at Chelsea", brain=False)
dim("SIM executor and SIM editors = teaching models; both hotels are fictional")
root = runs_dir(__file__) / "lab05_3_site2"
shutil.rmtree(root, ignore_errors=True)
KEY_DIR = "hvac_expert__fault_review__hotel_th__sim-cloud"
site2 = tasks("site2")


def run_site2(seed: int, reverify: bool, tag: str) -> dict:
    """A fresh copy of the first hotel's library, then the six second-hotel tasks in order."""
    src, dst = root / f"site1_s{seed}", root / f"{tag}_s{seed}"
    if not src.exists():
        stream(tasks("refine", family="fault_review"), seed=seed, root=src)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return stream(site2, seed=seed, root=dst, reverify=reverify)


step(1, "Two hotels, one tenant class, two O&M manuals")
for site, lim in MANUAL.items():
    print(f"│ {SITES[site]:<36} condenser approach limit {lim['approach_limit_k']} K")
print(f"│ second-hotel approaches: {', '.join(f'{t.id} {t.fixtures['condenser_approach_k']} K' for t in site2)}")
note("an approach between 3.0 and 4.5 K is fouling at the first hotel and normal at the second")

step(2, "Seed 0: the same library, the second hotel's six tasks, re-verified vs copied unchecked")
runs = {rv: run_site2(0, rv, "reverify" if rv else "unchecked") for rv in (True, False)}
rows = []
for t in site2:
    recs = {rv: next(r for r in runs[rv]["records"] if r.task.id == t.id) for rv in (True, False)}
    used = {rv: next((ln.split("limit ")[1] for ln in recs[rv].final_run.trajectory if "workspace fact" in ln), "—")
            for rv in (True, False)}
    rows.append([t.id, t.fixtures["condenser_approach_k"], t.truth["diagnosis"],
                 used[True], recs[True].final_run.deliverable.get("diagnosis"), runs[True]["per_task"][t.id],
                 used[False], recs[False].final_run.deliverable.get("diagnosis"), runs[False]["per_task"][t.id]])
table(rows, ["task", "approach K", "truth*", "re-verified: fact", "diagnosis", "score*",
             "unchecked: fact", "diagnosis", "score*"], maxw=18)
note("* truth and scores measured AFTER each step by bench.judge — no editor saw them")
print(f"═ seed 0 · re-verified mean {runs[True]['mean']} · unchecked mean {runs[False]['mean']}")

step(3, "What the editor copied, and what re-verification changed")
rec = next(r for r in runs[False]["records"] if r.task.id == "F1-CH40")
copied = next(txt for c in rec.candidates for e in c.edits if "library" in e.note
              for p, txt in e.files.items() if p.startswith("skills/"))
line = next(ln for ln in copied.splitlines() if "In this workspace" in ln)
print(f"│ copied from the Harbour View entry : {line}")
fixed = next(ln for ln in reverify_facts(copied, rec.task).splitlines() if "In this workspace" in ln)
print(f"│ after reverify_facts on F1-CH40    : {fixed}")
print(f"│ F1-CH40's own manual excerpt       : {rec.task.fixtures['manual']}")
print(f"│ prompts.py: “{QUOTE_REVERIFY}”")
warn("unchecked, the 3.0 K fact turned 3.4 K and 4.2 K into 'condenser fouling' at a hotel whose limit is 4.5 K: "
     "a wrong premise the contrast cannot see: the skill fired and the comparison was made, so the behaviour "
     "looks right — only the threshold is wrong")

step(4, "How often? The same comparison on seeds 0–4")
rows, explained = [], True
for seed in range(5):
    a, b = (run_site2(seed, True, "reverify"), run_site2(seed, False, "unchecked")) if seed else (runs[True], runs[False])
    worse = [t.id for t in site2 if b["per_task"][t.id] < a["per_task"][t.id]]
    used3 = [r.task.id for r in b["records"] if any("approach limit 3.0 K" in ln for ln in r.final_run.trajectory)]
    explained &= set(worse) <= set(used3) and all(a["per_task"][t.id] >= b["per_task"][t.id] for t in site2)
    rows.append([seed, a["mean"], b["mean"], f"{b['mean'] - a['mean']:+.1f}", ", ".join(used3) or "—", ", ".join(worse) or "—"])
table(rows, ["seed", "re-verified", "unchecked", "difference", "unchecked runs that used 3.0 K", "tasks worse unchecked"])
note("every seed's first-hotel library carries the 3.0 K fact inside a copied skill. It reaches a run only when "
     "the copied skill survives composition (a domain editor that writes its own skill from the task's manual "
     "overwrites it) and then fires. Seed 3: F1-CH42 used 3.0 K yet scored the same in both arms: the re-verified run used 4.5 K, but 4.2 K sits within 1 K of that limit, where the SIM makes the call uncertain, and it missed too")
if explained:
    ok("every task the unchecked arm lost had used the 3.0 K fact; re-verification never scored below copying here")

step(5, "Where a workspace fact belongs in Ops Assistant (EXAMPLE design for this course)")
print(f"│ prompts.py: “{QUOTE_MEMORY}”")
table([["procedure: compare the approach with the manual's limit", "skill (C6)", "travels to every hotel"],
       ["the limit itself: 'Chelsea approach limit 4.5 K'", "knowledge graph", "provenance + supersession; read at run time"],
       ["a short universal fact for every task", "LongTermMEMORY.md (C3)", "only if something else is cut"]],
      ["what", "home", "why"])
dim("the kit's SIM skill writes the fact into the skill, as the repository's GENERALIZE prompt suggests for one "
    "workspace; the course's EXAMPLE design moves tenant facts to the graph so a copied skill carries procedure only")
result("an earlier hotel's threshold is a hypothesis at the next hotel — re-verify it, and keep it in the graph, not in memory")
