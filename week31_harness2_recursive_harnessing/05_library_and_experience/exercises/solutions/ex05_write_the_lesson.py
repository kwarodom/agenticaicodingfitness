#!/usr/bin/env python3
"""Exercise 05 (solution) · Write the one lesson that transfers: a cue, a lesson, its evidence and an honest confidence — or NO_ENTRY.

Four TODOs, checked offline with simple rules (a person reviews the wording; the rules catch the common
mistakes). Run it: each check prints ✓ or ✕, and the script exits 1 until every check is ✓.

    .venv/bin/python week31_harness2_recursive_harnessing/05_library_and_experience/exercises/solutions/ex05_write_the_lesson.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import harness as H                    # noqa: E402
from h2out import check, finish        # noqa: E402

# TODO 1 — label each candidate cue: "too generic" · "too specific" · "just right".
CUES = {
    "a": "be careful with chillers and double-check everything",
    "b": "CH-2 at Harbour View had WO-4472 open on the night of 14 October",
    "c": "a proposal about equipment whose open maintenance can contradict the alarm",
}
LABELS = {"a": "too generic", "b": "too specific", "c": "just right"}

# TODO 2 — record A comes from a real step (F7-00, sim-cloud, seed 0, repeats=3). Write its EXPERIENCE entry.
RECORD_A = """\
component: tool_guidance.md
candidates: inv1
change: retried the same failing call 3× after a timeout → tool_guidance.md
observed: blind_retry: mitigated, mitigated, mitigated
verdict: helped
"""
ENTRY_A = {
    "cue": "a read over a long time window that times out and is retried unchanged",
    "lesson": "add one bounded fallback with a stop condition to tool_guidance.md: split the window into "
              "chunks once, and if a chunk also fails, stop and report which window failed",
    "evidence": "edit row 'tool_guidance.md' (inv1): blind_retry mitigated in all three probes of one task",
    "confidence": "medium — one task, three probes agree, no second task yet",
}

# TODO 3 — record B (F3-00, sim-cloud, seed 2, repeats=3). Write its entry as a dict like ENTRY_A, or the string
#          "NO_ENTRY" if the record shows no concrete, reusable execution contrast.
RECORD_B = """\
component: skills/alarm_triage/SKILL.md, systemprompt.md
candidates: dom1
change: 6 messages for alarms that share one root cause → skills/alarm_triage/SKILL.md, systemprompt.md
observed: storm_spam: mitigated, unchanged, mitigated
verdict: confounded
"""
ENTRY_B = "NO_ENTRY"      # its only row is confounded: no concrete, reusable contrast

# TODO 4 — a probe at the second hotel established "Chelsea Riverside Hotel: condenser approach limit 4.5 K".
#          Where does that fact go in Ops Assistant (the course's EXAMPLE design)? "LongTermMEMORY.md" · "skills/chiller_fault_review/SKILL.md"
#          · "knowledge graph"
FACT_HOME = "knowledge graph"


# ── checker (do not edit below) ────────────────────────────────────────────────
ID = re.compile(r"\b[A-Z]{1,3}-\d+\b|\bF\d-\d\d\b|\bWO-|\bAL-")
GENERIC = re.compile(r"\b(careful|thorough|double-check|better|good job|be smart|pay attention)\b", re.I)
COMPONENT = re.compile(r"tool_guidance\.md|systemprompt\.md|guardrails\.md|LongTermMEMORY\.md|skills?/|scripts?/|"
                       r"plugin/|sub_agents/|\bC[1-8]\b|\b(skill|script|plugin|subagent|tool guidance)\b", re.I)
VERB = re.compile(r"\b(add|move|replace|remove|split|name|chunk|cap|run)\b", re.I)

good = True
good &= check(LABELS == {"a": "too generic", "b": "too specific", "c": "just right"},
              "cues: a too generic (changes no behaviour) · b too specific (never fires again) · c just right",
              "cue labels wrong — would the cue change behaviour, and would it ever fire on a different task?")

e = ENTRY_A if isinstance(ENTRY_A, dict) else {}
cue, lesson = e.get("cue", ""), e.get("lesson", "")
ev, conf = e.get("evidence", ""), e.get("confidence", "")
probs = []
if "??" in str(e) or not cue:
    probs.append("fill all four fields")
else:
    if len(cue.split()) < 5:
        probs.append("cue: at least five words describing the situation")
    if ID.search(cue) or re.search(r"\d", cue):
        probs.append("cue: no IDs or numbers — they pin it to this task")
    if GENERIC.search(cue) or GENERIC.search(lesson):
        probs.append("cue/lesson: too generic ('be careful' changes no behaviour)")
    if not COMPONENT.search(lesson) or not VERB.search(lesson):
        probs.append("lesson: an operation (add/move/split…) that names the component")
    if ID.search(lesson):
        probs.append("lesson: no task IDs")
    if "tool_guidance.md" not in ev or "mitigated" not in ev:
        probs.append("evidence: cite the edit row (tool_guidance.md) and its contrast (mitigated …)")
    m = re.match(r"\s*(high|medium|low)\b(.*)", conf, re.I)
    if not m or len(m.group(2).strip(" :-—()")) < 4:
        probs.append("confidence: high|medium|low followed by why")
    elif m.group(1).lower() == "high":
        probs.append("confidence: one task stands behind it — 'high' over-claims")
    if H.grader_verdicts({"EXPERIENCE.md": "\n".join(f"{k}: {v}" for k, v in e.items())}):
        probs.append("measurement words or a grader verdict — Library.add would refuse it")
good &= check(not probs, f"entry A: cue '{cue}' · lesson names a component · evidence cites the row · confidence '{conf}'",
              "entry A: " + "; ".join(probs))

good &= check(ENTRY_B == "NO_ENTRY",
              "record B: NO_ENTRY — its only row is confounded (2 of 3 repeats); a lesson on it would be a guess",
              "record B: what verdict stands behind it? A lesson resting only on confounded rows is a guess — "
              "narrow it or decline")
good &= check(FACT_HOME == "knowledge graph",
              "the 4.5 K limit is a workspace fact: the knowledge graph, with provenance and supersession (EXAMPLE design for this course)",
              "a tenant's threshold copied into memory or a shared skill travels to the next hotel as a wrong premise (Lab 05-3)")
finish(good, "Done. Module 06 reads the results table these libraries feed: streams, noise, ablations, transfer, pass^k.")
