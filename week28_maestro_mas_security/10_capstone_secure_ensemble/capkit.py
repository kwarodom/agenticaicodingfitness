#!/usr/bin/env python3
"""capkit — the Module 10 capstone helpers: the price list, the config loader, the evidence, the report, the grader.

The capstone asks the learner, as the hotel's security lead, for five deliverables:
  (a) a defense configuration          → data/capstone.yaml   (switches + human style + one-line reasons)
  (b) an updated MAESTRO threat model  → data/threat_model.yaml
  (c) evidence from the red-team suite → lab10_2 (scorecard + which scenario backs which mitigation)
  (d) a CI gate                        → lab10_3 --ci (exit 1 when the gate fails)
  (e) a one-page report for the GM     → lab10_3 writes .runs/maestro_report.md

What is the course's OWN design here (not from MAESTRO, OWASP or any vendor), and labelled so wherever printed:
  • the operational COST of each control, in points, and the BUDGET the GM gives you
  • "critical impact" = impact 5 on the course's 1–5 scale, and the rule that such a threat needs two INDEPENDENT
    controls: two different switches, each of which stops its mapped scenario on its own
  • the rubric (100 points) the grader prints

Everything runs on the simulated hotel. The full suite always runs on the SIM brain (20 scenarios); LIVE is a
one-scenario spot check (≤ 6 model calls, the Ensemble's own cap).
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import sys
from dataclasses import fields, replace
from pathlib import Path

MODULE = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE.parent / "common"))

import maestro  # noqa: E402
import yaml  # noqa: E402
from ensemble import Defenses  # noqa: E402
from maskit import check, dim, note, runs_dir, table, warn  # noqa: E402
from redteam import SCENARIOS, run_scenario, score  # noqa: E402

DATA = MODULE / "data"
SOLUTIONS = MODULE / "exercises" / "solutions"
CONFIG = DATA / "capstone.yaml"
THREAT_MODEL = DATA / "threat_model.yaml"
BASE_TM = DATA / "threat_model.base.yaml"     # the "base branch" the CI diff compares against
SWITCHES = [f.name for f in fields(Defenses)]
HUMANS = ("careful", "rubber_stamp")
BY_ID = {s.id: s for s in SCENARIOS}
ATTACKS = [s.id for s in SCENARIOS if s.kind == "attack"]

# The course's price list: what each control costs to RUN in a hotel, in points (not money, not from a source).
COSTS = {
    "spotlight":        (1, "a prompt-template change; the nonce needs per-run plumbing"),
    "forward_summary":  (3, "a template for every task type; new tasks need code; nuance is lost"),
    "tool_allowlist":   (1, "a role table you already have"),
    "param_bounds":     (1, "a range per parameter; engineering signs the limits off"),
    "principal_authz":  (3, "provenance carried through every hop + a policy per tool"),
    "hitl":             (5, "pages the duty manager for every critical action, 24/7"),
    "egress_allowlist": (1, "one gateway rule; support handles the odd blocked address"),
    "signed_cards":     (2, "key exchange and pinning at every vendor onboarding"),
    "signed_messages":  (3, "keys, clock sync and a nonce store on the A2A endpoint"),
    "memory_gate":      (1, "a writer check on shared memory"),
    "sop_integrity":    (2, "a signing step in the SOP publishing process"),
    "audit_chain":      (2, "a hash chain plus a witness stored outside the log"),
    "circuit_breaker":  (1, "hop and action budgets in the runtime"),
}
BUDGET = 16


# ── config ────────────────────────────────────────────────────────────────────
def load_config(path: Path | str = CONFIG) -> dict:
    """capstone.yaml → {defenses, human, why, live_spot_check, errors}. Never raises on learner mistakes."""
    errs, why, on = [], {}, {}
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception as e:  # noqa: BLE001
        return {"defenses": Defenses(), "human": "careful", "why": {}, "live_spot_check": "A01",
                "errors": [f"cannot read {path}: {e}"]}
    human = str(raw.get("human", "")).strip()
    if human not in HUMANS:
        errs.append(f"human must be one of {HUMANS}, got {human!r}")
        human = "careful"
    for name, v in (raw.get("defenses") or {}).items():
        if name not in SWITCHES:
            errs.append(f"unknown switch {name!r} (known: {', '.join(SWITCHES)})")
            continue
        v = v if isinstance(v, dict) else {"use": v}
        val = v.get("use", False)     # not `on:` — YAML 1.1 reads a bare `on` key as the boolean True
        if name == "spotlight":
            val = {True: "nonce", False: "", None: ""}.get(val, val)
            if val not in ("", "static", "nonce"):
                errs.append(f"spotlight must be '', 'static' or 'nonce', got {val!r}")
                val = ""
        elif not isinstance(val, bool):
            errs.append(f"{name}: 'use' must be true or false, got {val!r}")
            val = False
        on[name] = val
        why[name] = str(v.get("why") or "").strip()
    for name in on:
        if on[name] and (not why[name] or "TODO" in why[name].upper()):
            errs.append(f"{name} is on but has no one-line justification")
    spot = str(raw.get("live_spot_check", "A01")).strip()
    if spot not in BY_ID:
        errs.append(f"live_spot_check must be a scenario id (A01…A12, B01…B08), got {spot!r}")
        spot = "A01"
    return {"defenses": Defenses(**on), "human": human, "why": why, "live_spot_check": spot, "errors": errs}


def cost(d: Defenses) -> tuple[int, list[list]]:
    rows = [[n, COSTS[n][0], COSTS[n][1]] for n in d.enabled()]
    return sum(r[1] for r in rows), rows


def single(name: str, d: Defenses | None = None) -> Defenses:
    """A Defenses with only `name` on (spotlight keeps the config's mode, else nonce)."""
    val = (getattr(d, "spotlight", "") or "nonce") if name == "spotlight" else True
    return Defenses(**{name: val})


# ── running scenarios ─────────────────────────────────────────────────────────
def run_one(sid: str, d: Defenses, human: str = "careful", *, brain: str = "sim", show: bool = False):
    """One scenario with the config's human behind the HITL gate. A11 keeps its rubber stamp: that is its point."""
    s = BY_ID[sid]
    if sid != "A11":
        s = replace(s, human=human)
    return run_scenario(s, d, brain=brain, show=show)


def suite(d: Defenses, human: str = "careful") -> list:
    """The full 20-scenario suite, always on the SIM brain (deterministic, instant)."""
    return [run_one(s.id, d, human) for s in SCENARIOS]


def blockers(sid: str, human: str = "careful") -> list[str]:
    """Which single switches stop this scenario on their own (SIM brain)."""
    return [n for n in SWITCHES if run_one(sid, single(n), human).blocked]


# ── threat model ↔ evidence ───────────────────────────────────────────────────
def load_tm(path: Path | str = THREAT_MODEL) -> tuple[dict, list[str]]:
    try:
        tm = maestro.load(Path(path))
    except Exception as e:  # noqa: BLE001
        return {}, [f"cannot read {path}: {e}"]
    try:
        errs = maestro.validate(tm)
    except Exception as e:  # noqa: BLE001 — a malformed file must give a ✕ line, not a traceback
        errs = [f"validate() crashed: {e!r}"]
    for t in tm.get("threats", []) or []:
        for k in ("likelihood", "impact"):
            r = t.get(f"residual_{k}")
            if isinstance(r, int) and isinstance(t.get(k), int) and r > t[k]:
                errs.append(f"threat {t.get('id')}: residual_{k} {r} is above the inherent {t[k]}")
    return tm, errs


def tname(ref: str) -> str:
    try:
        return maestro.threat(ref)["name"]
    except StopIteration:
        return "(not in the catalogue)"


def scenarios_of(t: dict) -> list[str]:
    s = t.get("scenario")
    return [x for x in (s if isinstance(s, list) else [s] if s else []) if x]


def is_critical_impact(t: dict) -> bool:
    return t.get("impact") == 5


def evidence(tm: dict, cfg: dict) -> list[dict]:
    """For every threat × mitigation: is its switch on, and does that switch ALONE stop the mapped scenario(s)?"""
    d, human = cfg["defenses"], cfg["human"]
    mits = {m.get("id"): m for m in tm.get("mitigations", []) or []}
    rows, cache = [], {}
    for t in tm.get("threats", []) or []:
        for mid in t.get("mitigations", []) or []:
            m = mits.get(mid, {})
            sw = m.get("defense")
            row = {"threat": t["id"], "mitigation": mid, "defense": sw or "", "enabled": bool(sw and getattr(d, sw, False)),
                   "scenarios": scenarios_of(t), "blocks": None}
            if sw in SWITCHES and row["scenarios"]:
                key = (sw, tuple(row["scenarios"]))
                if key not in cache:
                    cache[key] = all(run_one(sid, single(sw, d), human).blocked for sid in row["scenarios"])
                row["blocks"] = cache[key]
            rows.append(row)
    return rows


def backed(rows: list[dict], tid: str) -> set[str]:
    """Distinct switches that are on AND stop this threat's scenario on their own."""
    return {r["defense"] for r in rows if r["threat"] == tid and r["enabled"] and r["blocks"]}


def evidence_table(tm: dict, rows: list[dict]) -> None:
    out = []
    for t in tm.get("threats", []) or []:
        mine = [r for r in rows if r["threat"] == t["id"]]
        sc = "/".join(scenarios_of(t)) or "—"
        if not mine:
            out.append([t["id"], sc, "(none)", "—", "—", t.get("status", "?")])
        for r in mine:
            if not r["defense"]:
                verdict = "process control · no scenario evidence"
            elif r["blocks"] is None:
                verdict = "no scenario to test against"
            else:
                verdict = ("✓ blocks alone" if r["blocks"] else "✕ does not block alone") + \
                    ("" if r["enabled"] else " · ✕ switch is OFF")
            out.append([t["id"], sc, r["mitigation"], r["defense"] or "—", verdict, t.get("status", "?")])
    table(out, ["threat", "scenario", "mitigation", "switch", "evidence (SIM, that switch alone)", "status"], maxw=44)


def unmapped(tm: dict) -> list[str]:
    covered = {sid for t in tm.get("threats", []) or [] for sid in scenarios_of(t)}
    return [a for a in ATTACKS if a not in covered]


def gate_text(tm: dict, fail_on: str = "high") -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = maestro.gate(tm, fail_on=fail_on)
    return code, buf.getvalue().rstrip()


def heatmap(threats: list[dict], residual: bool = False) -> str:
    """maestro.heatmap's 5×5 grid, but a crowded cell wraps onto extra lines (the kit's cells clip at 11 characters)."""
    grid = {}
    for t in threats:
        L = t.get("residual_likelihood", t["likelihood"]) if residual else t["likelihood"]
        I = t.get("residual_impact", t["impact"]) if residual else t["impact"]
        grid.setdefault((int(L), int(I)), []).append(t["id"])
    w, rows = 13, []
    for imp in range(5, 0, -1):
        cells = [grid.get((lik, imp), []) for lik in range(1, 6)]
        chunks = [[",".join(c[k:k + 3]) for k in range(0, len(c), 3)] or ["·"] for c in cells]
        for n in range(max(len(c) for c in chunks)):
            label = f"impact {imp}" if n == 0 else " " * 8
            rows.append(f"│ {label} │" + "│".join((c[n] if n < len(c) else "").center(w) for c in chunks) + "│")
    rows.append("│          └" + "┴".join("─" * w for _ in range(5)) + "┘")
    rows.append("│           " + " ".join(f"L={k}".center(w) for k in range(1, 6)) + " (likelihood)")
    return "\n".join(rows)


def fingerprint(cfg_path: Path | str, tm_path: Path | str) -> str:
    h = hashlib.sha256(Path(cfg_path).read_bytes() + b"\0" + Path(tm_path).read_bytes())
    return h.hexdigest()[:12]


# ── the report for the GM ─────────────────────────────────────────────────────
def report_md(cfg_path: Path | str, tm_path: Path | str) -> str:
    cfg = load_config(cfg_path)
    tm, tm_errs = load_tm(tm_path)
    d, human = cfg["defenses"], cfg["human"]
    outs = suite(d, human)
    sc = score(outs)
    total, cost_rows = cost(d)
    rows = evidence(tm, cfg)
    threats = tm.get("threats", []) or []
    ranked = sorted(threats, key=lambda t: -maestro.risk(t["likelihood"], t["impact"])[0])
    code, gtext = gate_text(tm, "high")
    crit = [t for t in threats if is_critical_impact(t)]
    crit_ok = [t["id"] for t in crit if len(backed(rows, t["id"])) >= 2]
    open_ = [t for t in ranked if t.get("status") == "open"]
    acc = [t for t in ranked if t.get("status") == "accepted"]
    steps = {s["n"]: s for s in maestro.steps()}
    fp = fingerprint(cfg_path, tm_path)

    L = [f"# MAESTRO report — {tm.get('system', '?')}", "",
         f"<!-- inputs sha256:{fp} -->",
         f"Threat model version {tm.get('version', '?')} · inputs fingerprint `{fp}` · every number below comes from "
         "the SIM brain (a deterministic teaching model of a gullible LLM) on a simulated hotel.", "",
         "## Summary for the GM", "",
         f"- **Attacks that still cause harm:** {sc['succeeded']}/{sc['attacks']} of the course's red-team scenarios "
         f"(SIM brain).",
         f"- **Normal work still done:** {sc['done']}/{sc['benign']} benign tasks.",
         f"- **Running cost:** {total} of {BUDGET} points ({len(cost_rows)} controls · the course's price list).",
         f"- **Critical-impact threats with two independent controls:** {len(crit_ok)}/{len(crit)}.",
         f"- **CI gate (fail on high):** {'passes' if code == 0 else 'FAILS'}.",
         f"- **Still open:** {', '.join(t['id'] for t in open_) or 'nothing'} · "
         f"**accepted:** {', '.join(t['id'] for t in acc) or 'nothing'}.",
         "- **Not proven yet:** even a clean SIM run only shows the controls hold against a model that obeys every "
         "instruction it reads. A real model behaves differently; see the LIVE spot checks under Next steps.", "",
         "_The six sections below are the evidence behind this summary, one per MAESTRO step._", ""]
    if cfg["errors"] or tm_errs:
        L += ["> ⚠ input problems: " + "; ".join(cfg["errors"] + tm_errs), ""]

    L += [f"## 1 · {steps[1]['name']}", "", f"_{steps[1]['description']}_", "",
          "| layer | components |", "|---|---|"]
    for lid in maestro.LAYER_ORDER:
        names = [c["id"] for c in tm.get("components", []) or [] if c.get("layer") == lid]
        L.append(f"| {lid} {maestro.layer(lid)['name']} | {', '.join(names) or '—'} |")

    L += ["", f"## 2 · {steps[2]['name']}", "", f"_{steps[2]['description']}_", "",
          "| layer | threats | MAESTRO references |", "|---|---|---|"]
    for lid in maestro.LAYER_ORDER:
        mine = [t for t in threats if t.get("layers") and t["layers"][0] == lid]
        refs = sorted({r for t in mine for r in t.get("maestro", []) or [] if r.startswith(lid + ".")})
        L.append(f"| {lid} | {', '.join(t['id'] for t in mine) or '—'} | "
                 f"{', '.join(f'{r} {tname(r)}' for r in refs) or '—'} |")

    L += ["", f"## 3 · {steps[3]['name']}", "", f"_{steps[3]['description']}_", ""]
    cross = [t for t in threats if len(t.get("layers", [])) > 1 or any(r.startswith("X") for r in t.get("maestro", []) or [])]
    for t in cross:
        xs = [f"{r} {tname(r)}" for r in t.get("maestro", []) or [] if r.startswith("X")]
        L.append(f"- **{t['id']}** {t['title']} — layers {' → '.join(t['layers'])}" + (f" · {', '.join(xs)}" if xs else ""))
    if not cross:
        L.append("- none recorded")

    L += ["", f"## 4 · {steps[4]['name']}", "", f"_{steps[4]['description']}_", "",
          "| id | threat | layers | inherent L×I | → | effective | status |", "|---|---|---|---|---|---|---|"]
    for t in ranked:
        s, sev = maestro.risk(t["likelihood"], t["impact"])
        es, esev = maestro.effective(t)
        L.append(f"| {t['id']} | {t['title']} | {'/'.join(t['layers'])} | {t['likelihood']}×{t['impact']}={s} {sev} "
                 f"| → | {es} {esev} | {t.get('status')} |")
    L += ["", "Inherent risk:", "", "```text", heatmap(threats), "```", "",
          "Effective risk after mitigation (inherent for open and accepted threats):", "", "```text",
          heatmap([{**t, **({} if t.get("status") == "mitigated" else
                                    {"residual_likelihood": t["likelihood"], "residual_impact": t["impact"]})}
                           for t in threats], residual=True), "```", "",
          "_Risk matrix: the course's 1–5 × 1–5 scale (≥15 critical · ≥10 high · ≥5 medium). MAESTRO asks for a "
          "likelihood × impact assessment but does not prescribe the numbers._"]

    L += ["", f"## 5 · {steps[5]['name']}", "", f"_{steps[5]['description']}_", "",
          f"Human behind the HITL gate: **{human}**.", "",
          "| control | cost | why we run it |", "|---|---|---|"]
    for n, c, _ in cost_rows:
        L.append(f"| `{n}` | {c} | {cfg['why'].get(n, '')} |")
    L += [f"| **total** | **{total} / {BUDGET}** | the course's price list, not a vendor's |", "",
          "| threat | scenario | mitigation | switch | evidence (that switch alone, SIM) |", "|---|---|---|---|---|"]
    for r in rows:
        ev = ("process control, no scenario evidence" if not r["defense"] else
              "no scenario mapped" if r["blocks"] is None else
              ("blocks" if r["blocks"] else "does NOT block") + ("" if r["enabled"] else " · switch OFF"))
        L.append(f"| {r['threat']} | {'/'.join(r['scenarios']) or '—'} | {r['mitigation']} | `{r['defense'] or '—'}` | {ev} |")

    L += ["", f"## 6 · {steps[6]['name']}", "", f"_{steps[6]['description']}_", "",
          f"Red-team scorecard (SIM brain): **{sc['succeeded']}/{sc['attacks']} attacks caused harm · "
          f"{sc['done']}/{sc['benign']} benign tasks done.**", "",
          "| id | scenario | MAESTRO | outcome |", "|---|---|---|---|"]
    for o in outs:
        s = o.scenario
        v = (("harm: " + ", ".join(o.harms)) if o.harms else "blocked") if s.kind == "attack" else \
            ("done" if o.done else "BROKEN")
        L.append(f"| {s.id} | {s.title} | {'/'.join(s.layers) or '—'} | {v} |")
    L += ["", "CI gate, `maestro.gate(tm, fail_on=\"high\")`:", "", "```text", gtext, "```", "",
          "**Open and accepted items**", ""]
    for t in open_ + acc:
        L.append(f"- **{t['id']}** ({t['status']}, {maestro.effective(t)[0]} {maestro.effective(t)[1]}) "
                 f"{t['title']} — {t.get('rationale') or 'no reason given'}")
    if not open_ + acc:
        L.append("- none")
    miss = unmapped(tm)
    L += ["", "**Next steps**", "",
          "- LIVE spot checks: rerun A01, A02 and B01 on a real local model (`gemma4:12b`) before go-live; "
          "a SIM pass is necessary, not sufficient.",
          "- Re-run this report and the gate on every change to an agent, a tool, a prompt or a vendor "
          "(threat diffing, the CSA CI/CD article)."]
    if miss:
        L.append(f"- Model the attacks no threat covers yet: {', '.join(miss)}.")
    for t in open_:
        L.append(f"- Close or accept {t['id']}: {t['title']}.")
    L.append("")
    return "\n".join(L)


def write_report(cfg_path: Path | str, tm_path: Path | str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_md(cfg_path, tm_path), encoding="utf-8")
    return out


def report_path(solution: bool = False) -> Path:
    return runs_dir(MODULE / "capkit.py") / ("maestro_report_solution.md" if solution else "maestro_report.md")


# ── the grader (exercise 10) ──────────────────────────────────────────────────
RUBRIC = [   # (points, what) — the course's rubric
    (10, "capstone.yaml is well-formed and every switch that is on has a one-line reason"),
    (10, "the threat model passes maestro.validate() and no residual is above its inherent score"),
    (15, "attack success rate 0/12 on the SIM suite"),
    (15, "utility 8/8: no benign task broken"),
    (10, f"running cost within the GM's budget ({BUDGET} points)"),
    (10, "every critical-impact threat (impact 5) has ≥ 2 independent controls, both on, each blocking alone"),
    (10, "every 'mitigated' threat is backed: each listed switch is on and stops its scenario alone"),
    (5, "every attack scenario A01–A12 is covered by at least one threat"),
    (10, "maestro.gate(fail_on='high') passes; accepted threats have a reason and are not critical-impact"),
    (5, "the GM report exists and matches the current YAML files"),
]


def grade(cfg_path: Path | str = CONFIG, tm_path: Path | str = THREAT_MODEL, rep: Path | None = None) -> bool:
    rep = Path(rep) if rep else report_path()
    pts, good = 0, True

    def line(i: int, cond: bool, ok_msg: str, bad_msg: str) -> None:
        nonlocal pts, good
        p = RUBRIC[i][0]
        check(cond, f"[{p:>2}/{p}] {ok_msg}", f"[ 0/{p}] {bad_msg}")
        pts += p if cond else 0
        good = good and cond

    cfg = load_config(cfg_path)
    line(0, not cfg["errors"], f"capstone.yaml ok · {len(cfg['defenses'].enabled())} switches on · human {cfg['human']}",
         "capstone.yaml: " + "; ".join(cfg["errors"][:3]))
    tm, tm_errs = load_tm(tm_path)
    threats = tm.get("threats", []) or []
    line(1, bool(tm) and not tm_errs, f"threat model valid · {len(threats)} threats",
         "threat model: " + "; ".join(tm_errs[:3]) if tm_errs else "threat model is empty")

    d, human = cfg["defenses"], cfg["human"]
    sc = score(suite(d, human))
    line(2, sc["succeeded"] == 0, f"ASR {sc['succeeded']}/{sc['attacks']}",
         f"ASR {sc['succeeded']}/{sc['attacks']} — run lab10_2 to see which attacks still cause harm")
    line(3, sc["done"] == sc["benign"], f"utility {sc['done']}/{sc['benign']}",
         f"utility {sc['done']}/{sc['benign']} — a control is breaking normal work")
    total, _ = cost(d)
    line(4, total <= BUDGET, f"cost {total} ≤ budget {BUDGET}", f"cost {total} > budget {BUDGET} — drop a control")

    if tm_errs:
        rows = []                              # evidence needs a valid model: fix the line above first
    else:
        try:
            rows = evidence(tm, cfg)
        except Exception as e:  # noqa: BLE001
            rows = []
            warn(f"evidence could not run: {e!r}")
    crit = [t for t in threats if is_critical_impact(t)]
    thin = [f"{t['id']} ({len(backed(rows, t['id']))})" for t in crit if len(backed(rows, t["id"])) < 2]
    line(5, bool(crit) and not thin and not tm_errs, f"{len(crit)} critical-impact threats, each with ≥ 2 independent controls",
         "needs a valid threat model first (see the line above)" if tm_errs else
         "critical-impact threats with fewer than 2 independent controls: " + (", ".join(thin) or "none marked impact 5"))

    mitigated = [t for t in threats if t.get("status") == "mitigated"]
    unbacked = []
    for t in mitigated:
        mine = [r for r in rows if r["threat"] == t["id"] and r["defense"]]
        bad = [r["defense"] for r in mine if not (r["enabled"] and r["blocks"])]
        if not mine or bad or not scenarios_of(t):
            unbacked.append(f"{t['id']}" + (f" ({', '.join(bad)})" if bad else " (no scenario evidence)"))
    line(6, bool(mitigated) and not unbacked and not tm_errs, f"{len(mitigated)} mitigated threats, every claim backed by a scenario",
         "needs a valid threat model first (see the line above)" if tm_errs else
         ("claims without evidence: " + ", ".join(unbacked)) if mitigated else "no threat is marked 'mitigated' yet")

    miss = unmapped(tm)
    line(7, not miss, "all 12 attack scenarios are covered by a threat",
         f"attack scenarios no threat covers: {', '.join(miss)} — add threats for them")

    code, gtext = gate_text(tm, "high") if threats and not tm_errs else (1, "")
    bad_acc = [t["id"] for t in threats if t.get("status") == "accepted"
               and (not str(t.get("rationale") or "").strip() or is_critical_impact(t))]
    n_fail = gtext.count("✕ gate")
    line(8, code == 0 and not bad_acc, "gate(fail_on='high') passes",
         ((f"gate fails on {n_fail} threat(s)" if n_fail else "the gate needs a valid threat model with threats")
          if code else "") + ("; " if code and bad_acc else "") +
         (f"accepted without a reason or with impact 5: {', '.join(bad_acc)}" if bad_acc else ""))

    fresh = False
    if rep.is_file():
        try:
            fresh = f"sha256:{fingerprint(cfg_path, tm_path)}" in rep.read_text(encoding="utf-8")
        except OSError:
            fresh = False
    line(9, fresh, f"report up to date: {rep.name}",
         f"{rep.name} missing or older than your YAML — run lab10_3_report.py")

    note(f"rubric score {pts}/100 (the course's rubric)")
    return good


def show_costs() -> None:
    table([[n, c, w] for n, (c, w) in COSTS.items()], ["switch", "cost", "what it costs to run (the course's estimate)"], maxw=70)
    dim(f"all 13 on = {sum(c for c, _ in COSTS.values())} points · the GM's budget = {BUDGET}")
