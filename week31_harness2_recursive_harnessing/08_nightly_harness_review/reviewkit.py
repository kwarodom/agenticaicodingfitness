#!/usr/bin/env python3
"""reviewkit — Module 08's nightly harness review routine for the SIMULATED Ops Assistant (EXAMPLE design).

The course's EXAMPLE design treats harnessing as one more always-on routine: schedule-woken, a plain
check first, silent when nothing qualifies, and a human gate before any version ships. This file builds that
routine on the course kit (common/): it never edits the kit, and it never reads Task.truth (only bench.py does).

    wakes (a simulated day)  →  plain_check (no model call)  →  select_tasks (≤ N, by priority)
        →  h2kit.step per task in Shadow  →  bench.promote → DRAFT version  →  review packet
        →  publish(reviewer, read_hash) gate  →  send_to_user envelope (notify only if a draft exists)

Plus the pieces Module 08 tests: the human-signal firewall (`selector_input`), cache discipline
(`cold_prefixes`), the promotion ladder and rollback as a pointer change (`Registry`), and a LIVE editor
wrapper (`live_editor`). Everything here is a teaching model: none of it is deployed, none of it is measured.

    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # the module folder
    import reviewkit as R
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import harness as H                                                  # noqa: E402
import h2kit                                                         # noqa: E402
from bench import promote                                            # noqa: E402
from ops_sim import SITES, Task, task                            # noqa: E402
from h2kit import experience_md, observe, step                       # noqa: E402

N_DEFAULT = 3                                                        # EXAMPLE design: N = 3
PRIORITY = ("rejected_proposal", "low_rating", "helper_failed", "review_block", "latency")   # EXAMPLE priority order
ROUTINE = "routine:nightly-harness-review"


# ── a simulated day of wakes, with the human signals Ops Assistant records (EXAMPLE shapes) ────────────
@dataclass
class Wake:
    wake_id: str
    task: Task
    signal: str                         # rejected_proposal | low_rating | helper_failed | sample
    approval: dict | None = None        # {"by", "outcome", "reason"} — a named human's decision on a proposal
    rating: str | None = None           # "not worth it" on a notification or digest
    helper: str | None = None           # "stopped" | "failed"

    @property
    def human_reason(self) -> str:
        return (self.approval or {}).get("reason") or self.task.human_reason


def wake_for(t: Task, n: int) -> Wake:
    """The wake the always-on runtime would have logged for this task, with its human signal."""
    w = Wake(f"wake-{n:03d}", t, t.signal)
    if t.signal == "rejected_proposal":
        w.approval = {"by": f"chief engineer, {SITES[t.site]}", "outcome": "rejected", "reason": t.human_reason}
    elif t.signal == "low_rating":
        w.rating = "not worth it"
    elif t.signal == "helper_failed":
        w.helper = "stopped"
    return w


def day(ids: list[str]) -> list[Wake]:
    return [wake_for(task(i), n) for n, i in enumerate(ids, 1)]


QUIET_DAY = ["F1-10", "F1-11", "F3-02", "F3-03", "F7-01"]                       # every wake is a plain 'sample'
BUSY_DAY = ["F1-10", "F3-02", "F1-01", "F2-00", "F3-03", "F7-00", "F1-11", "F3-00", "F7-01", "F1-02"]


# ── 1 · plain check and selection (no model call) ──────────────────────────────────────────────
def plain_check(wakes: list[Wake]) -> list[Wake]:
    """Wakes with a weak signal. Pure code over the event log: no model is called, nothing is spent."""
    return [w for w in wakes if w.signal in PRIORITY]


def select_tasks(wakes: list[Wake], n: int = N_DEFAULT) -> list[Wake]:
    """Up to n weak-signal wakes, in the EXAMPLE priority order; a recorded human reason breaks ties."""
    weak = plain_check(wakes)
    weak.sort(key=lambda w: (PRIORITY.index(w.signal), 0 if w.human_reason else 1, w.wake_id))
    return weak[:n]


# ── 2 · one night ────────────────────────────────────────────────────────────────────────────────
@dataclass
class Night:
    wakes: list
    weak: list
    selected: list
    records: list
    draft: dict | None
    promote_notes: list
    model_calls: int
    envelope: dict = field(default_factory=dict)


def nightly_review(wakes: list[Wake], current: dict, *, n: int = N_DEFAULT, executor: str = "sim-cloud",
                   editor: str = "sim-cloud-editor", seed: int = 0, proposer_for: dict | None = None,
                   always_on_cap: int | None = None) -> Night:
    """The routine (EXAMPLE design). Shadow: read-only tools, probes in the sandbox, nobody notified.

    proposer_for = {task_id: proposer} lets one task use a LIVE editor; every other task uses the SIM editors.
    """
    weak = plain_check(wakes)
    if not weak:                                    # plain check: nothing qualifies → no model call, silence
        night = Night(wakes, [], [], [], None, [], 0)
        night.envelope = envelope(night)
        return night
    chosen = select_tasks(wakes, n)
    records = []
    for w in chosen:                                # each would be a child workflow in Temporal
        prop = (proposer_for or {}).get(w.task.id)
        records.append(step(w.task, current, k=1, executor=executor, editor=editor, seed=seed, proposer=prop))
    composed = [r for r in records if not r.keep_base]
    draft, notes = (None, ["every task answered KEEP_BASE — no draft, no notification"]) if not composed else \
        promote(current, records, always_on_cap=always_on_cap)
    if draft is not None and draft == current:
        draft, notes = None, notes + ["the reviewer's merge adopted nothing — no draft"]
    calls = sum(r.cost["executions"] + r.cost["editor_sessions"] for r in records)
    night = Night(wakes, weak, chosen, records, draft, notes, calls)
    night.envelope = envelope(night)
    return night


def envelope(night: Night) -> dict:
    """The send_to_user envelope (EXAMPLE design): outcome=done; notify only when a draft version exists."""
    return {"channel": "operator", "outcome": "done", "notify": night.draft is not None,
            "draft": H.definition_hash(night.draft) if night.draft else None,
            "evidence": [f"probe_run:{r.task.id}/{c.label}#{i}" for r in night.records for c in r.candidates
                         for i in range(len(c.probes))]}


# ── 3 · the review packet ───────────────────────────────────────────────────────────────────────────
def packet(current: dict, night: Night, *, version: str, parent: str) -> dict:
    """What the named reviewer reads. Raises if any part carries measurement words (the library rule)."""
    d = night.draft
    entries = []
    for rec in night.records:
        md = experience_md(rec.rows, rec.exp)
        delta = H.always_on_tokens(rec.final_tree) - H.always_on_tokens(rec.base)
        md += f"human_reason: {rec.task.human_reason or '(none recorded)'}\nalways_on_delta_tokens: {delta:+d}\n"
        entries.append({"task": rec.task.id, "signal": rec.task.signal, "keep_base": rec.keep_base, "md": md})
    p = {"HARNESS.json": H.harness_json(d, version=version, parent=parent, created_by=ROUTINE),
         "changes": H.diff(current, d), "components": H.changed_components(current, d),
         "always_on": (H.always_on_tokens(current), H.always_on_tokens(d)),
         "prefix_hash": (H.prefix_hash(current), H.prefix_hash(d)),
         "definition_hash": (H.definition_hash(current), H.definition_hash(d)),
         "promote_notes": night.promote_notes, "experience": entries, "validate": H.validate_tree(d)}
    bad = [e["task"] for e in entries if H.grader_verdicts({"EXPERIENCE.md": e["md"]})]
    if bad:
        raise ValueError(f"packet refused: measurement words in EXPERIENCE for {', '.join(bad)}")
    return p


def write_packet(p: dict, tree: dict, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    H.write_tree(tree, root / "harness")
    (root / "harness" / "HARNESS.json").write_text(p["HARNESS.json"] + "\n", encoding="utf-8")
    for e in p["experience"]:
        (root / f"EXPERIENCE_{e['task']}.md").write_text(e["md"], encoding="utf-8")
    lines = [f"# Review packet — {json.loads(p['HARNESS.json'])['version']} (EXAMPLE, SIM)", "",
             f"- definition hash: {p['definition_hash'][0]} → {p['definition_hash'][1]}",
             f"- prefix hash: {p['prefix_hash'][0]} → {p['prefix_hash'][1]}"
             + (" (CHANGED: ships only at a deployment boundary)" if p['prefix_hash'][0] != p['prefix_hash'][1] else " (same)"),
             f"- always-on tokens: {p['always_on'][0]} → {p['always_on'][1]} ({p['always_on'][1] - p['always_on'][0]:+d})",
             f"- components changed: {', '.join(p['components'])}", "", "## Changes", ""]
    lines += [f"- {c['change']} {c['path']} ({c['component']}, {c['delta_tokens']:+d} tokens)" for c in p["changes"]]
    lines += ["", "## Reviewer's merge", ""] + [f"- {n}" for n in p["promote_notes"]]
    (root / "PACKET.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


# ── 4 · the named-reviewer gate (a skill-publish flow, EXAMPLE design) ──────────────────────────────
class Registry:
    """harness_version rows and one pointer per ladder stage. Promotion and rollback move pointers only."""

    STAGES = ("shadow", "assist", "serve", "write_back")

    def __init__(self, v1: dict):
        self.versions = {"v1": {"tree": v1, "parent": None, "state": "serve", "approved_by": "initial export"}}
        self.pointer = {"serve": "v1"}
        self.audit: list[str] = []

    def add_draft(self, vid: str, tree: dict, parent: str) -> None:
        self.versions[vid] = {"tree": tree, "parent": parent, "state": "draft", "approved_by": None}
        self.audit.append(f"draft {vid} (parent {parent}) by {ROUTINE}")

    def publish(self, vid: str, reviewer: str, read_hash: str) -> tuple[int, str]:
        """Draft → Shadow. 403 if no named human; 409 if the draft changed since the reviewer read it."""
        v = self.versions[vid]
        if not reviewer or reviewer.startswith("routine:"):
            self.audit.append(f"403 {vid}: '{reviewer or '(nobody)'}' is not a named human reviewer")
            return 403, "a harness version needs a named human reviewer; the routine cannot approve its own draft"
        if read_hash != H.definition_hash(v["tree"]):
            self.audit.append(f"409 {vid}: reviewer read {read_hash}, draft is {H.definition_hash(v['tree'])}")
            return 409, "the draft changed after you read it: re-read the packet"
        errs = H.validate_tree(v["tree"])
        if errs:
            self.audit.append(f"422 {vid}: {errs[0]}")
            return 422, errs[0]
        v.update(state="shadow", approved_by=reviewer)
        self.pointer["shadow"] = vid
        self.audit.append(f"200 {vid} → shadow, approved_by {reviewer}")
        return 200, "published to Shadow (mirrored, nobody notified)"

    def promote(self, vid: str, stage: str, gates: dict) -> tuple[bool, str]:
        """One rung up the ladder, only from the rung below, only if every gate holds."""
        state = self.versions[vid]["state"]
        order = ("draft",) + self.STAGES
        if state not in order or order.index(stage) != order.index(state) + 1:
            self.audit.append(f"refuse {vid} → {stage}: it is {state}, not {order[order.index(stage) - 1]}")
            return False, f"refused: {vid} is {state}; the ladder has no skipped rungs"
        failed = [g for g, okv in gates.items() if not okv]
        if failed:
            self.audit.append(f"hold {vid} at {state}: {', '.join(failed)}")
            return False, f"held at {state}: {', '.join(failed)}"
        self.pointer[stage] = vid
        self.versions[vid]["state"] = stage
        self.audit.append(f"{vid} → {stage}")
        return True, f"{vid} is now {stage}"

    def rollback(self, stage: str) -> str:
        """A pointer change to the parent version (plus one cache re-warm). Nothing is deleted."""
        cur = self.pointer[stage]
        parent = self.versions[cur]["parent"]
        self.pointer[stage] = parent
        self.versions[cur]["state"] = "rolled_back"
        self.audit.append(f"rollback {stage}: {cur} → {parent} (pointer change; {cur} kept for the record)")
        return parent

    def tree(self, stage: str) -> dict:
        return self.versions[self.pointer[stage]]["tree"]


# ── 5 · the human-signal firewall ───────────────────────────────────────────────────────────────────
FORBIDDEN = ("human_reason", "signal", "approval", "approved_by", "rating", "reason", "truth", "traps", "helper")


def selector_input(t: Task, runs: list) -> dict:
    """What the anonymised selector may read: the request, the sources, and the deliverables. Nothing else."""
    v = t.visible()
    return {"request": v["request"], "sources": v["sources"],
            "candidates": [{"id": f"c{i}", "deliverable": r.text()} for i, r in enumerate(runs)]}


def input_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def _keys(o) -> set:
    if isinstance(o, dict):
        return set(o) | {k for v in o.values() for k in _keys(v)}
    if isinstance(o, list):
        return {k for v in o for k in _keys(v)}
    return set()


class GuardedTask:
    """Wraps a Task and records which attributes a function reads (to test the kit's selector)."""

    def __init__(self, t: Task):
        object.__setattr__(self, "_t", t)
        object.__setattr__(self, "touched", set())

    def __getattr__(self, name):
        self.touched.add(name)
        return getattr(self._t, name)


def firewall_checks(wake: Wake, runs: list, seed: int = 0) -> list[tuple[str, bool]]:
    """Unit-style checks: the selector's input carries no approval, rating or reason field."""
    import copy
    si = selector_input(wake.task, runs)
    keys = _keys(si)
    out = [(f"selector input has no '{f}' field", f not in keys) for f in FORBIDDEN]
    blob = json.dumps(si, ensure_ascii=False)
    out.append(("the reviewer's words do not appear in the selector input",
                not wake.human_reason or wake.human_reason not in blob))
    t2 = copy.deepcopy(wake.task)
    t2.human_reason, t2.signal = "a completely different reason", "low_rating"
    out.append(("input hash unchanged when reason and signal change",
                input_hash(selector_input(t2, runs)) == input_hash(si)))
    g = GuardedTask(wake.task)
    h2kit.select(g, runs, seed=seed)
    out.append((f"h2kit.select read only {sorted(g.touched)} of the task",
                not (g.touched & {"human_reason", "signal", "truth", "traps"})))
    src = inspect.getsource(h2kit.select)
    out.append(("h2kit.select's source names no human-signal field",
                not any(f in src for f in ("human_reason", "approved_by", "rating", ".truth"))))
    return out


# ── 6 · cache discipline ────────────────────────────────────────────────────────────────────────────
def cold_prefixes(hourly_hash: list[str], start_hash: str) -> list[int]:
    """Hours (0–23) at which the always-on prefix hash differs from the hour before: each is a cold prefix."""
    out, prev = [], start_hash
    for h, x in enumerate(hourly_hash):
        if x != prev:
            out.append(h)
        prev = x
    return out


def schedule(versions: list[tuple[int, dict]], start: dict) -> list[str]:
    """Prefix hash in force each hour, given (ship_hour, tree) events."""
    cur, out = H.prefix_hash(start), []
    ev = {h: tr for h, tr in versions}
    for h in range(24):
        if h in ev:
            cur = H.prefix_hash(ev[h])
        out.append(cur)
    return out


# ── 7 · LIVE: a local model as the editor (one call per scope) ───────────────────────────────────────
TOPIC = {"no_cmms": r"work orders?|cmms", "hand_arithmetic": r"kW/ton|arithmetic|energy_units|units",
         "no_evidence": r"evidence", "blind_retry": r"timeout|chunk|retry", "no_reopen": r"reopen|validate",
         "no_analytic_move": r"approach|manual|diagnos", "storm_spam": r"batch|storm|root cause",
         "nuisance_notify": r"nuisance", "digest_contract": r"section|digest", "thai_register": r"Thai|ไทย"}
LIVE_LOG: list = []


def live_editor(scope: str, ref, t: Task, tree: dict) -> list:
    """h2kit.live_propose, with the edit credited to a failure the editor was shown in its scope.

    The kit's live edits carry obs='live', and compose keeps an edit only when ITS failure kind was mitigated in
    the probe, so an uncredited LIVE edit can never be merged. Here the edit is credited to the observed failure
    (in its scope) whose topic its text names, else to the first one; the probe still decides whether that
    failure was mitigated. Every LIVE edit is kept in LIVE_LOG so the lab can show what the model wrote.
    """
    import re
    edits = h2kit.live_propose(scope, ref, t, tree)
    seen = [o.kind for o in observe(ref, t, human_reason=t.human_reason) if o.scope == scope]
    for e in edits:
        text = e.note + " " + " ".join(e.files.values())
        hit = next((k for k in seen if re.search(TOPIC.get(k, "$^"), text, re.I)), None)
        e.obs = hit or (seen[0] if seen else "live")
        LIVE_LOG.append({"scope": scope, "task": t.id, "credited": e.obs, "note": e.note,
                         "lines": {p: [ln for ln in x.splitlines() if ln and ln not in tree.get(p, "")]
                                   for p, x in e.files.items()}})
    if not edits:
        LIVE_LOG.append({"scope": scope, "task": t.id, "credited": None, "note": "no <edit> block (KEEP_BASE)", "lines": {}})
    return edits
