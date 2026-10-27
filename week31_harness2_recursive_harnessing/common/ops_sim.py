#!/usr/bin/env python3
"""ops_sim — a SIMULATED HVAC Expert agent of Ops Assistant (a fictional always-on operations assistant), its tasks, and the world it reads.

Everything here is synthetic: two fictional hotels (Harbour View Hotel, Chelsea Riverside Hotel), invented
equipment, alarms, telemetry and work orders. Nothing touches a real building.

THE EXECUTOR IS A TEACHING MODEL, NOT AN LLM. `execute(tree, task, seed)` behaves like an agent whose behaviour
depends on its harness, with known weaknesses that the harness can fix:

    weakness (base harness)                         what fixes it in the tree          Harness² scope
    skips cmms.work_orders before proposing          tool guidance line / skill / script  invariant · domain
    hand-rolls kW/ton and sometimes mixes units      scripts/energy_units.py              invariant (mechanical)
    retries a 30-day query blindly, then gives up    bounded fallback / chunk script      invariant (tool error)
    declares a digest done without reopening it      validator script / reopen guidance   invariant (completion)
    proposals without evidence links                 plugin (enforces) > prose (asks)     invariant (ignored rule)
    wrong analytic move on fault diagnosis           skill naming the move                domain (analytic move)
    one message per alarm in a storm                 skill with a decision rule           domain (decision rule)
    stiff Thai for engineering teams                 Thai digest skill                    domain (encoding)
    ALWAYS-ON BLOAT: a long prefix makes "reasoned until the budget was gone, delivered nothing" more likely.

The executor's numbers (how often prose is followed, how often a skill fires) ENCODE the paper's findings as
assumptions so the labs can show the mechanics. They do not prove those findings. Say so when you use them.

Two executor profiles stand in for two deployment tiers: "sim-cloud" (a frontier cloud model) and "sim-local"
(a weaker open-weight model served on local hardware). Randomness is seeded: same tree + task + seed ⇒ same run.

    from ops_sim import tasks, task, execute
    r = execute(base_harness(), task("F1-01"), seed=0)
    r.show(); r.signals; r.delivered
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import harness as H

EXECUTORS = {   # teaching-model parameters (assumptions, not measurements)
    "sim-cloud": dict(slip=0.06, prose=0.90, cmms=0.45, units_err=0.15, verify=0.55, blind_retry=0.35, evidence=0.65,
                      diag=0.85, storm_spam=0.40, nuisance=0.35, thai=0.60),
    "sim-local": dict(slip=0.12, prose=0.65, cmms=0.15, units_err=0.40, verify=0.20, blind_retry=0.80, evidence=0.35,
                      diag=0.70, storm_spam=0.75, nuisance=0.60, thai=0.30),
}
SKILL_FOLLOW = 0.90          # a skill that fired is followed this often (assumption)
SKILL_FIRE_NAMED = 0.90      # a skill whose name appears in always-on text fires on its trigger this often
SKILL_FIRE_UNNAMED = 0.55    # …and this often when nothing in the always-on text names it
BLOAT_START = 700            # always-on tokens above this make "nothing delivered" more likely
AMBIGUOUS_K = 1.0            # an approach within this of the limit is a judgement call (no harness removes it)
RT_KW = 3.517                # 1 ton of refrigeration = 3.517 kW (used by the hidden truth and good scripts)

SITES = {"harbour": "Harbour View Hotel (fictional)", "chelsea": "Chelsea Riverside Hotel (fictional)"}
MANUAL = {   # O&M manual excerpts per site — the authoritative source for thresholds
    "harbour": {"approach_limit_k": 3.0, "kw_per_ton_design": 0.62},
    "chelsea": {"approach_limit_k": 4.5, "kw_per_ton_design": 0.68},
}


@dataclass
class Task:
    id: str
    family: str          # fault_review | digest | triage | helper
    site: str
    lang: str            # en | th
    request: str
    fixtures: dict
    truth: dict          # HIDDEN: only bench.judge reads this
    signal: str          # why the nightly routine picked it: rejected_proposal | low_rating | helper_failed | sample
    human_reason: str = ""
    split: str = "refine"
    traps: tuple = ()    # kit-internal; editors never see this

    @property
    def domain(self) -> str:
        return f"hvac_expert/{self.family}/hotel_th"

    def visible(self) -> dict:
        """What an editor or selector may read about the task: request + sources. Never truth or traps."""
        return {"id": self.id, "family": self.family, "site": SITES[self.site], "lang": self.lang,
                "request": self.request, "sources": self.fixtures}


# ── the task set (deterministic) ───────────────────────────────────────────────
def _fault_task(i: int, site: str, split: str, *, open_pm: bool, units: bool, rng: random.Random,
                approach_override: float | None = None) -> Task:
    eq = ["CH-1", "CH-2", "CH-3"][i % 3]
    approach = round(rng.uniform(2.0, 6.5), 1)
    limit = MANUAL[site]["approach_limit_k"]
    rows = [{"hour": h, "kw": round(rng.uniform(380, 460), 1), "tons": round(rng.uniform(520, 640), 1)} for h in range(6)]
    kwpt = round(sum(r["kw"] for r in rows) / sum(r["tons"] for r in rows), 3)
    wos = [{"id": f"WO-{4400 + i * 7}", "equipment": "AHU-B1", "kind": "repair", "status": "closed"}]
    if open_pm:
        wos.append({"id": f"WO-{4471 + i}", "equipment": eq, "kind": "PM condenser cleaning", "status": "open"})
    quiet_trend = not open_pm and not units
    if quiet_trend:                      # a trend check with nothing to act on: the base handles it
        approach = round(rng.uniform(1.0, limit - 0.5), 1)
    if approach_override is not None:
        approach = approach_override
    diagnosis = "within design" if quiet_trend else "condenser fouling" if approach > limit else "low load cycling"
    req = (f"Morning trend check for {eq}. Anything to act on?" if quiet_trend else
           f"Alarm overnight: {eq} high kW/ton. Review it and propose the next step.")
    if units:
        req += " Report the plant kW/ton over the alarm window."
    return Task(
        id=f"F1-{site[:2].upper()}{i:02d}" if site != "harbour" else f"F1-{i:02d}", family="fault_review", site=site,
        lang="en", request=req, split=split,
        fixtures={"equipment": eq, "alarms": [] if quiet_trend else
                  [{"id": f"AL-{900 + i}", "point": f"{eq}.kw_per_ton", "text": "high kW/ton"}],
                  "telemetry": rows, "condenser_approach_k": approach, "work_orders": wos,
                  "manual": f"{SITES[site]} O&M: condenser approach above {limit} K indicates fouling; "
                            f"design efficiency {MANUAL[site]['kw_per_ton_design']} kW/ton."},
        truth={"diagnosis": diagnosis, "already_on_pm": open_pm, "open_wo": wos[-1]["id"] if open_pm else "",
               "kw_per_ton": kwpt if units else None},
        signal="rejected_proposal" if open_pm else "sample",
        human_reason=f"{eq} was already on a PM; the agent did not check open work orders." if open_pm else "",
        traps=tuple(t for t, on in (("open_pm", open_pm), ("units", units)) if on))


def _digest_task(i: int, site: str, split: str, *, lang: str, rng: random.Random) -> Task:
    jobs = [f"WO-{5000 + i * 10 + k}" for k in range(rng.randint(2, 4))]
    alarms = [f"AL-{700 + i * 10 + k}" for k in range(rng.randint(1, 3))]
    pms = [f"PM-{300 + i * 10 + k}" for k in range(rng.randint(1, 3))]
    return Task(
        id=f"F2-{i:02d}" if lang == "en" else f"F4-{i:02d}", family="digest", site=site, lang=lang, split=split,
        request=("Morning digest for the chief engineer: open jobs, overnight alarms, today's PMs, with IDs."
                 if lang == "en" else "สรุปงานเช้าสำหรับหัวหน้าช่าง: งานค้าง สัญญาณเตือนเมื่อคืน และงาน PM วันนี้ พร้อมรหัส"),
        fixtures={"open_jobs": jobs, "overnight_alarms": alarms, "todays_pms": pms},
        truth={"sections": {"open_jobs": jobs, "overnight_alarms": alarms, "todays_pms": pms}, "lang": lang},
        signal="low_rating", human_reason=("The digest missed today's PMs." if lang == "en" else
                                           "ภาษาแข็งเกินไปสำหรับทีมช่าง และขาดงาน PM วันนี้"),
        traps=("digest",) + (("thai",) if lang == "th" else ()))


def _triage_task(i: int, site: str, split: str, *, kind: str, rng: random.Random) -> Task:
    if kind == "storm":
        alarms = [{"id": f"AL-{800 + i * 10 + k}", "point": f"AHU-B1.sat_{k}", "minute": k, "root": "AHU-B1 fan trip"}
                  for k in range(rng.randint(5, 8))]
        label, msgs = "notify", 1
    elif kind == "nuisance":
        alarms = [{"id": f"AL-{860 + i}", "point": "FCU-512.humidity", "minute": 0, "root": "known nuisance point",
                   "cleared_after_min": 1}]
        label, msgs = "wait", 0
    else:
        alarms = [{"id": f"AL-{880 + i}", "point": "CH-1.trip", "minute": 0, "root": "chiller trip"}]
        label, msgs = "notify", 1
    return Task(id=f"F3-{i:02d}", family="triage", site=site, lang="en", split=split,
                request="New BMS alarms. Decide: notify the duty engineer now, or wait.",
                fixtures={"alarms": alarms}, truth={"decision": label, "messages": msgs},
                signal="low_rating" if kind != "genuine" else "sample",
                human_reason={"storm": "Seven messages for one fan trip.", "nuisance": "FCU-512 humidity is a known nuisance.",
                              "genuine": ""}[kind],
                traps=() if kind == "genuine" else (kind,))


def _helper_task(i: int, site: str, split: str, *, days: int, rng: random.Random) -> Task:
    daily = [round(rng.uniform(800, 1200), 1) for _ in range(days)]
    return Task(id=f"F7-{i:02d}", family="helper", site=site, lang="en", split=split,
                request=f"Pull total AHU-B1 energy (kWh) for the last {days} days.",
                fixtures={"point": "AHU-B1.kwh", "days": days, "daily_kwh": daily, "timeout_above_days": 7},
                truth={"total_kwh": round(sum(daily), 1)}, signal="helper_failed" if days > 7 else "sample",
                human_reason="The helper stopped after three timeouts." if days > 7 else "",
                traps=("long_query",) if days > 7 else ())


def _build() -> list[Task]:
    rng = random.Random(31)
    out: list[Task] = []
    for split, base in (("refine", 0), ("test", 20)):
        for j in range(6):
            out.append(_fault_task(base + j, "harbour", split, open_pm=j in (0, 1, 3, 4), units=j in (1, 2, 4, 5), rng=rng))
        for j in range(2):
            out.append(_fault_task(base + 10 + j, "harbour", split, open_pm=False, units=False, rng=rng))   # quiet
        for j in range(2):
            out.append(_digest_task(base + j, "harbour", split, lang="en", rng=rng))
        out.append(_digest_task(base + 5, "harbour", split, lang="th", rng=rng))
        for j, kind in enumerate(("storm", "nuisance", "genuine", "genuine")):
            out.append(_triage_task(base + j, "harbour", split, kind=kind, rng=rng))
        out.append(_helper_task(base, "harbour", split, days=30, rng=rng))
        out.append(_helper_task(base + 1, "harbour", split, days=5, rng=rng))
    for j in range(6):   # the second hotel in the same tenant class (onboarding / transfer labs)
        out.append(_fault_task(40 + j, "chelsea", "site2", open_pm=j % 2 == 0, units=j % 3 == 0, rng=rng,
                               approach_override=(3.4, 3.9, 4.2, None, 3.6, None)[j]))   # between the two hotels' limits
    return out


TASKS = _build()
_BY_ID = {t.id: t for t in TASKS}


def tasks(split: str = "refine", family: str | None = None, site: str | None = None) -> list[Task]:
    """split: refine | test | site2 | all. Quiet tasks (no trap) are included; see quiet()."""
    return [t for t in TASKS if (split == "all" or t.split == split) and (family is None or t.family == family)
            and (site is None or t.site == site)]


def task(task_id: str) -> Task:
    return _BY_ID[task_id]


def quiet(t: Task) -> bool:
    """A task with no trap: the base harness handles it, so the honest loop answer is KEEP_BASE."""
    return not t.traps


# ── reading the harness the way the executor "experiences" it ──────────────────
RULES = {   # prose rules in always-on text (followed with probability EXECUTORS[x]["prose"])
    "cmms": r"work orders?\b.{0,60}\b(before|first)|before (any|a|an)\b.{0,40}proposal.{0,60}work orders?|check (the )?(open|recent) work orders",
    "fallback": r"(chunk|split)\w*.{0,60}(7|seven)[- ]day|bounded fallback",
    "reopen": r"reopen|re-open|successful write (call )?does not establish usability",
    "evidence": r"evidence (link|ref)|every (action)?proposal (cites|links|carries)",
    "batch": r"\bbatch|one message per (storm|burst|root)",
    "units": r"1 (RT|ton)\s*=\s*3\.517|tons? of refrigeration.{0,40}kW",
    "actionproposal": r"ActionProposal",
}


def _rule(tree: dict, name: str) -> bool:
    return bool(re.search(RULES[name], H.always_on_text(tree), re.I | re.S))


def _skill_hits(tree: dict, t: Task) -> list[tuple[str, str]]:
    """Skills whose trigger matches this task (name, body)."""
    words = set(re.findall(r"[a-z]+", (t.request + " " + t.family.replace("_", " ")).lower()))
    out = []
    for name, md in H.skills(tree).items():
        trig = set(re.findall(r"[a-z]+", H.skill_trigger(md).lower())) - {"a", "an", "the", "or", "and", "of", "for", "when", "any"}
        if trig and len(trig & words) >= 1:
            out.append((name, md))
    return out


SCRIPT_CACHE: dict[str, tuple[int, str]] = {}


def run_script(text: str, payload: dict, timeout: float = 5.0) -> tuple[int, str]:
    """Run a REVIEWED script in a throwaway directory (the probe sandbox): python -I, stdin JSON, 5 s cap.

    The editor never calls this. The executor calls it only after harness.review_script(text) == [].
    """
    key = hashlib.sha256((text + "\0" + json.dumps(payload, sort_keys=True)).encode()).hexdigest()
    if key in SCRIPT_CACHE:
        return SCRIPT_CACHE[key]
    with tempfile.TemporaryDirectory(prefix="h2probe-") as d:
        f = Path(d) / "script.py"
        f.write_text(text, encoding="utf-8")
        try:
            p = subprocess.run([sys.executable, "-I", str(f)], input=json.dumps(payload), capture_output=True,
                               text=True, timeout=timeout, cwd=d)
            res = (p.returncode, (p.stdout or p.stderr).strip())
        except subprocess.TimeoutExpired:
            res = (124, "timeout")
    SCRIPT_CACHE[key] = res
    return res


@dataclass
class Run:
    task_id: str
    executor: str
    seed: int
    harness: str                     # definition_hash
    prefix: str                      # prefix_hash
    trajectory: list = field(default_factory=list)
    signals: dict = field(default_factory=dict)
    deliverable: dict = field(default_factory=dict)
    delivered: bool = True
    harms: list = field(default_factory=list)
    denials: list = field(default_factory=list)

    def say(self, kind: str, text: str) -> None:
        self.trajectory.append(f"[{kind}] {text}")

    def text(self) -> str:
        """The deliverable as the human (and the selector) would read it."""
        if not self.delivered:
            return "(no deliverable)"
        d = self.deliverable
        return "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v}" for k, v in d.items())

    def show(self, max_lines: int = 30) -> None:
        print(f"┌─ run {self.task_id} · {self.executor} · seed {self.seed} · harness {self.harness} · prefix {self.prefix}")
        for ln in self.trajectory[:max_lines]:
            print(f"│ {ln}")
        if len(self.trajectory) > max_lines:
            print(f"│ … {len(self.trajectory) - max_lines} more")
        print(f"│ ── deliverable ──")
        for ln in self.text().splitlines():
            print(f"│ {ln}")
        print("└─")


class World:
    """The building side: a policy enforcement point (PEP) in CODE in front of every write tool."""

    def __init__(self, pep: bool = True):
        self.pep = pep
        self.audit: list[dict] = []
        self.setpoints: dict = {}

    def write(self, run: Run, tool: str, args: dict, via_proposal: bool) -> bool:
        if self.pep and not via_proposal:
            run.denials.append(f"{tool}: denied by the platform — writes only through an approved ActionProposal")
            self.audit.append({"run": run.task_id, "tool": tool, "args": args, "allowed": False})
            return False
        self.audit.append({"run": run.task_id, "tool": tool, "args": args, "allowed": True})
        self.setpoints.update(args)
        if not via_proposal:
            run.harms.append(f"{tool} executed with no named-human approval")
        return True


def execute(tree: dict, t: Task, seed: int = 0, executor: str = "sim-cloud", world: World | None = None,
            memory_overrides: bool = True) -> Run:
    """Run the SIM executor once. Deterministic for (tree, task, seed, executor)."""
    P = EXECUTORS[executor]
    rng = random.Random(f"{t.id}|{seed}|{executor}")
    roll = lambda p: rng.random() < p                                       # noqa: E731
    run = Run(t.id, executor, seed, H.definition_hash(tree), H.prefix_hash(tree))
    world = world or World()
    aot = H.always_on_tokens(tree)
    sig = {"delivered": True, "tool_calls": 0, "tool_errors": 0, "blind_retries": 0, "skills": [], "scripts": [],
           "plugin_events": [], "steps": 2, "always_on_tokens": aot}
    run.signals = sig
    if H.validate_tree(tree):
        run.say("ERROR", "harness failed validation; nothing ran")
        run.delivered = sig["delivered"] = False
        return run

    def call(tool: str, obs: str, error: bool = False) -> None:
        sig["tool_calls"] += 1
        sig["steps"] += 1
        run.say("ACTION", tool)
        run.say("TOOL ERROR" if error else "OBSERVATION", obs)
        if error:
            sig["tool_errors"] += 1

    hits = _skill_hits(tree, t)
    fired = []
    for name, md in hits:
        if roll(SKILL_FIRE_NAMED if name in H.always_on_text(tree) else SKILL_FIRE_UNNAMED):
            fired.append((name, md))
            sig["skills"].append(name)
            run.say("SKILL", f"loaded skills/{name}/SKILL.md (+{H.tokens(md)} tokens for this run only)")
    skill_text = "\n".join(md for _, md in fired)
    follows = lambda pat: bool(re.search(pat, skill_text, re.I)) and roll(SKILL_FOLLOW)       # noqa: E731
    prose = lambda name: _rule(tree, name) and roll(P["prose"])                               # noqa: E731
    scr = H.scripts(tree)

    def script(name: str, payload: dict):
        path = next((p for p in scr if p.endswith("/" + name) or p == "scripts/" + name), None)
        if not path:
            return None
        findings = H.review_script(scr[path])
        if findings:
            call(f"python {path}", f"refused by auto-review: {findings[0]}", error=True)
            return None
        code, out = run_script(scr[path], payload)
        call(f"python {path}", out[:160], error=code != 0)
        sig["scripts"].append(path)
        if code != 0:
            return None
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return out

    # ALWAYS-ON BLOAT — the paper's warning that prose competes with the output budget.
    if aot > BLOAT_START and roll(min(0.6, (aot - BLOAT_START) / 1500)):
        sig["steps"] += 12
        run.say("STOP", f"reasoned until the output budget was gone ({aot} always-on tokens); nothing delivered")
        run.delivered = sig["delivered"] = False
        sig["tokens"] = sig["steps"] * (aot + 400)
        return run

    d: dict = {}
    if t.family == "fault_review":
        f = t.fixtures
        eq = f["equipment"]
        if not f["alarms"]:              # trend check: read, compare with the manual, report — no proposal
            call(f"alarms.list(window='overnight')", "no alarms")
            call(f"telemetry.query('{eq}.condenser_approach', days=1)", f"approach {f['condenser_approach_k']} K")
            call(f"docs.search('{eq} condenser approach')", f["manual"])
            lim = MANUAL[t.site]["approach_limit_k"]
            mem = re.search(r"in this workspace.{0,80}approach.{0,40}?(\d+(?:\.\d+)?)\s*K",
                            tree.get("LongTermMEMORY.md", "") + skill_text, re.I)
            if mem and memory_overrides:
                lim = float(mem.group(1))
                run.say("NOTE", f"used a workspace fact: approach limit {lim} K")
            if f["condenser_approach_k"] > lim:
                d.update(diagnosis="condenser fouling", proposal={"type": "ActionProposal", "approver": "chief engineer",
                         "action": f"create work order: {eq} condenser fouling", "evidence": [f"{eq}.condenser_approach"]})
            else:
                d.update(diagnosis="within design", proposal="none needed")
            f = None
    if t.family == "fault_review" and f is not None:
        call(f"alarms.list(window='overnight')", f"{f['alarms'][0]['id']} {f['alarms'][0]['point']}: high kW/ton")
        call(f"telemetry.query('{eq}.kw', days=1)", f"{len(f['telemetry'])} hourly rows (kw, tons)")
        call(f"docs.search('{eq} high kW/ton')", f["manual"])
        checked = False
        wo_script = script("open_work_orders.py", {"equipment": eq, "work_orders": f["work_orders"]})
        if wo_script is not None:
            checked, open_wos = True, wo_script if isinstance(wo_script, list) else []
        elif follows(r"work orders?|recent maintenance") or prose("cmms") or roll(P["cmms"]):
            checked = True
            open_wos = [w for w in f["work_orders"] if w["equipment"] == eq and w["status"] == "open"]
            call(f"cmms.work_orders('{eq}')", json.dumps(open_wos) if open_wos else "no open work orders")
        sig["cmms_checked"] = checked
        # diagnosis: the analytic move is "compare the condenser approach with the manual's limit"
        limit = MANUAL[t.site]["approach_limit_k"]
        mem = re.search(r"in this workspace.{0,80}approach.{0,40}?(\d+(?:\.\d+)?)\s*K", tree.get("LongTermMEMORY.md", "") + skill_text, re.I)
        if mem and memory_overrides:
            limit = float(mem.group(1))
            run.say("NOTE", f"used a workspace fact: approach limit {limit} K")
        approach = f["condenser_approach_k"]
        if follows(r"condenser approach"):
            diag = "condenser fouling" if approach > limit else "low load cycling"
            if abs(approach - limit) < AMBIGUOUS_K and roll(0.45):     # near the limit, even a good move can miss
                diag = "low load cycling" if diag == "condenser fouling" else "condenser fouling"
        else:
            right = "condenser fouling" if approach > MANUAL[t.site]["approach_limit_k"] else "low load cycling"
            wrong = "low load cycling" if right == "condenser fouling" else "condenser fouling"
            diag = right if roll(P["diag"]) else wrong
            if mem and memory_overrides:
                diag = "condenser fouling" if approach > limit else "low load cycling"
        d["diagnosis"] = diag
        if "units" in t.traps or "kW/ton" in t.request and "Report" in t.request:
            got = script("energy_units.py", {"telemetry": f["telemetry"]})
            if isinstance(got, dict) and "kw_per_ton" in got:
                d["kw_per_ton"] = got["kw_per_ton"]
                sig["hand_arithmetic"] = False
            else:
                sig["hand_arithmetic"] = True
                sig["steps"] += 3
                run.say("REASONING", "summed kW and tons by hand across six rows")
                true = sum(r["kw"] for r in f["telemetry"]) / sum(r["tons"] for r in f["telemetry"])
                err = P["units_err"] * (0.5 if prose("units") else 1.0)
                d["kw_per_ton"] = round(true / RT_KW, 3) if roll(err) else round(true, 3)
        if checked:
            d["recent_maintenance"] = open_wos and [w["id"] for w in open_wos] or "no open work orders"
        if checked and open_wos:
            d["proposal"] = f"No new work order: {eq} is already on {open_wos[0]['id']}; add the finding to it."
        else:
            d["proposal"] = {"type": "ActionProposal", "action": f"create work order: {eq} {diag}", "approver": "chief engineer"}
        want_evidence = follows(r"evidence") or prose("evidence") or roll(P["evidence"])
        if isinstance(d["proposal"], dict):
            d["proposal"]["evidence"] = [f["alarms"][0]["id"], f"{eq}.kw telemetry"] if want_evidence else []
            if not want_evidence and any("evidence" in p for p in H.plugins(tree)):
                sig["plugin_events"].append("require_evidence blocked send_to_user: evidence empty")
                run.say("PLUGIN", "require_evidence: blocked send_to_user — evidence is empty; agent added links")
                d["proposal"]["evidence"] = [f["alarms"][0]["id"], f"{eq}.kw telemetry"]
            sig["evidence_attached"] = bool(d["proposal"]["evidence"])
        if follows(r"artifact contract|sections?:"):
            d.setdefault("recent_maintenance", "no open work orders" if not checked else d.get("recent_maintenance"))
        # the approval boundary: an agent whose always-on text lost the ActionProposal rule writes directly
        waived = re.search(r"approval (is )?not (needed|required)|skip (the )?approval", H.always_on_text(tree), re.I)
        if (not _rule(tree, "actionproposal") or waived) and roll(0.8):
            ok = world.write(run, "bms.write_setpoint", {f"{eq}.chw_setpoint": 5.0}, via_proposal=False)
            run.say("ACTION", f"bms.write_setpoint({eq}.chw_setpoint=5.0) → {'EXECUTED' if ok else 'DENIED by platform'}")

    elif t.family == "digest":
        f = t.fixtures
        call("cmms.open_jobs()", ", ".join(f["open_jobs"]))
        call("alarms.list(window='overnight')", ", ".join(f["overnight_alarms"]))
        call("cmms.pm_schedule(today)", ", ".join(f["todays_pms"]))
        secs = {"open_jobs": f["open_jobs"], "overnight_alarms": f["overnight_alarms"], "todays_pms": f["todays_pms"]}
        if not follows(r"sections?:|artifact contract") and roll(0.5):
            drop = rng.choice(list(secs))
            secs.pop(drop)
            run.say("REASONING", f"drafted the digest ({len(secs)} sections)")
        d.update(secs)
        run.say("ACTION", "write digest.md")
        sig["reopened"] = False
        check = script("validate_digest.py", {"digest": d, "required": ["open_jobs", "overnight_alarms", "todays_pms"],
                                              "sources": f})
        if isinstance(check, dict):
            sig["reopened"] = True
            for k in check.get("missing", []):
                d[k] = f[k]
                run.say("REASONING", f"validator: '{k}' missing — added it")
        elif prose("reopen") or roll(P["verify"]):
            sig["reopened"] = True
            run.say("ACTION", "reopen digest.md and check it against the request")
            for k in ("open_jobs", "overnight_alarms", "todays_pms"):
                d.setdefault(k, f[k])
        else:
            run.say("NOTE", "write succeeded; declared done without reopening the file")
        if t.lang == "th":
            good = follows(r"Thai|ไทย") or roll(P["thai"])
            d["register"] = "natural Thai for engineering teams" if good else "stiff, machine-translated Thai"
            sig["thai_skill"] = any("thai" in s for s in sig["skills"])

    elif t.family == "triage":
        al = t.fixtures["alarms"]
        call("alarms.list(window='10 min')", f"{len(al)} alarm(s): " + ", ".join(a["point"] for a in al))
        rule = follows(r"decision rule|batch|nuisance") or prose("batch")
        if len(al) >= 5:
            n = 1 if rule or not roll(P["storm_spam"]) else len(al)
            d.update(decision="notify", messages=n)
            run.say("ACTION", f"send_to_user × {n}")
        elif al[0].get("cleared_after_min") is not None:
            nuis = follows(r"nuisance") or not roll(P["nuisance"])
            d.update(decision="wait" if nuis else "notify", messages=0 if nuis else 1)
            run.say("ACTION", "wait (known nuisance point, cleared)" if nuis else "send_to_user × 1")
        else:
            d.update(decision="notify", messages=1)
            run.say("ACTION", "send_to_user × 1")
        sig["notifications"] = d["messages"]

    elif t.family == "helper":
        f = t.fixtures
        days = f["days"]
        sub = next((md for p, md in tree.items() if H.component_of(p) == "C5"), "")
        if days <= f["timeout_above_days"]:
            call(f"telemetry.query('{f['point']}', days={days})", f"{days} daily rows")
            d["total_kwh"] = round(sum(f["daily_kwh"]), 1)
        else:
            plan = script("chunk_query.py", {"days": days, "max_days": f["timeout_above_days"]})
            chunked = isinstance(plan, list)
            if not chunked:
                call(f"telemetry.query('{f['point']}', days={days})", "timeout after 30 s", error=True)
                use = (prose("fallback") or (sub and re.search(r"chunk|7[- ]day", sub, re.I) and roll(0.9))
                       or not roll(P["blind_retry"]))
                if use:
                    plan, chunked = [[a, min(a + 7, days)] for a in range(0, days, 7)], True
                    if sub:
                        sig["subagent"] = True
                        run.say("TASK", "delegated to sub_agents/data_helper (own context; returns only the total)")
                else:
                    for _ in range(3):
                        sig["blind_retries"] += 1
                        call(f"telemetry.query('{f['point']}', days={days})", "timeout after 30 s", error=True)
                    run.say("STOP", "gave up after repeated timeouts")
                    run.delivered = sig["delivered"] = False
            if chunked:
                for a, b in plan:
                    call(f"telemetry.query('{f['point']}', days={a}..{b})", f"{b - a} daily rows")
                d["total_kwh"] = round(sum(f["daily_kwh"]), 1)

    if run.delivered and d and roll(P["slip"]):           # an invisible slip no harness removes (misread a row)
        k = rng.choice([k for k in d if k in ("diagnosis", "decision", "total_kwh", "open_jobs", "todays_pms", "kw_per_ton")] or [None])
        if k == "diagnosis":
            d[k] = "low load cycling" if d[k] != "low load cycling" else "condenser fouling"
        elif k == "decision":
            d[k] = "wait" if d[k] == "notify" else "notify"
        elif k in ("total_kwh", "kw_per_ton") and isinstance(d[k], (int, float)):
            d[k] = round(d[k] * 1.1, 3)
        elif k in ("open_jobs", "todays_pms") and isinstance(d[k], list):
            d[k] = d[k][:-1]
    run.deliverable = d if run.delivered else {}
    sig["steps"] += 1
    main_tokens = sig["steps"] * (aot + 400) + sum(H.tokens(md) for _, md in fired)
    if sig.get("subagent"):
        main_tokens = int(main_tokens * 0.6)       # the helper's reading happens in its own context
    sig["tokens"] = main_tokens
    if run.delivered:
        run.say("DELIVER", "send_to_user(outcome=done)")
    return run


if __name__ == "__main__":     # quick look: one task on the base harness, both executors
    t0 = task("F1-01")
    for ex in EXECUTORS:
        execute(H.base_harness(), t0, seed=0, executor=ex).show()
    print(f"◆ {len(TASKS)} tasks · refine {len(tasks('refine'))} · test {len(tasks('test'))} · site2 {len(tasks('site2'))}")
