#!/usr/bin/env python3
"""maestro — the CSA MAESTRO framework as data, plus threat-model-as-code helpers.

MAESTRO (Multi-Agent Environment, Security, Threat, Risk, and Outcome) is Ken Huang's agentic-AI threat
modeling framework, published by the Cloud Security Alliance on 2025-02-06. This module PARSES the layer
descriptions, threats, cross-layer threats, mitigations, the six-step method and the architecture patterns
out of the saved page in `sources/maestro_framework_2025-02-06.txt`, so every name and description you see is
the source's own wording. The OWASP Top 10 for Agentic Applications (2026) list comes the same way from
`sources/owasp_agentic_top10_2026_giskard.txt`.

What is the course's OWN (not from either source), and labelled so wherever it is printed:
  • OWASP_TO_LAYERS — which MAESTRO layers each ASI risk mostly lives in
  • the 1–5 likelihood × impact matrix and its severity bands (MAESTRO says to "assess likelihood and impact
    … using the risk measurement and risk matrix" but does not prescribe numbers)
  • the threat-model YAML schema, report, diff and CI gate

    import maestro
    cat = maestro.catalog()                    # what the runner's 🗺 MAESTRO map shows
    tm = maestro.load("threat_model.yaml"); errs = maestro.validate(tm)
    print(maestro.report_md(tm)); code = maestro.gate(tm, fail_on="critical")
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

SOURCES = Path(__file__).resolve().parent / "sources"
MAESTRO_SRC = SOURCES / "maestro_framework_2025-02-06.txt"
OWASP_SRC = SOURCES / "owasp_agentic_top10_2026_giskard.txt"
MAESTRO_URL = "https://cloudsecurityalliance.org/blog/2025/02/06/agentic-ai-threat-modeling-framework-maestro"
OWASP_URL = "https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/"

LAYER_ORDER = ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]

# The course's mapping (NOT from MAESTRO or OWASP): where each ASI risk mostly lives in MAESTRO's layers.
OWASP_TO_LAYERS = {
    "ASI01": ["L1", "L7"], "ASI02": ["L3", "L7"], "ASI03": ["L6", "L7"], "ASI04": ["L3", "L4", "L7"],
    "ASI05": ["L3", "L4"], "ASI06": ["L2"], "ASI07": ["L3", "L7"], "ASI08": ["L7", "L5"],
    "ASI09": ["L6", "L7"], "ASI10": ["L7", "L5"],
}


def _pairs(lines: list[str]) -> list[tuple[str, str]]:
    """'Name:  description' lines (the page uses two spaces after the colon)."""
    out = []
    for ln in lines:
        m = re.match(r"^([^:]{3,80}?):\s{2}(.+)$", ln)
        if m:
            out.append((m.group(1).strip(), m.group(2).strip()))
    return out


@lru_cache(maxsize=1)
def _parsed() -> dict:
    lines = MAESTRO_SRC.read_text(encoding="utf-8").splitlines()
    layers, i = {}, 0
    while i < len(lines):
        m = re.match(r"^Layer (\d): (.+?)(?: \(Vertical Layer\))?$", lines[i])
        if m and i + 1 < len(lines) and lines[i + 1].startswith("Description:"):
            lid, name = f"L{m.group(1)}", m.group(2).strip()
            desc = lines[i + 1].split(":", 1)[1].strip()
            j, block = i + 3, []                         # skip "Threat Landscape:"
            while j < len(lines) and not re.match(r"^(Layer \d:|B\. Cross-Layer)", lines[j]):
                block.append(lines[j])
                j += 1
            threats = [{"id": f"{lid}.T{k + 1}", "name": n, "description": d}
                       for k, (n, d) in enumerate(_pairs(block))]
            layers[lid] = {"id": lid, "name": name, "description": desc, "threats": threats,
                           "vertical": lid == "L6"}
            i = j
            continue
        i += 1

    def section(start: str, stop: str) -> list[str]:
        a = next(k for k, ln in enumerate(lines) if ln.startswith(start))
        b = next(k for k in range(a + 1, len(lines)) if lines[k].startswith(stop))
        return lines[a + 1:b]

    cross = [{"id": f"X{k + 1}", "name": n, "description": d}
             for k, (n, d) in enumerate(_pairs(section("B. Cross-Layer Threats", "C. Mitigation")))]
    mit_lines = section("C. Mitigation Strategies", "D. Using MAESTRO")
    cut = next(k for k, ln in enumerate(mit_lines) if ln.startswith("AI-Specific"))
    mitigations = {
        "cross_layer": [{"name": n, "description": d} for n, d in _pairs(mit_lines[:cut])
                        if n not in ("Layer-Specific Mitigations",)],
        "ai_specific": [{"name": n, "description": d} for n, d in _pairs(mit_lines[cut:])],
    }
    steps = [{"n": k + 1, "name": n, "description": d}
             for k, (n, d) in enumerate(_pairs(section("D. Using MAESTRO", "4. Agentic Architecture")))]
    pat_lines = section("4. Agentic Architecture Patterns", "Conclusion")
    patterns, cur = [], None
    for ln in pat_lines[1:]:
        if ":" not in ln and ln.strip():
            cur = {"name": ln.strip(), "fields": {}}
            patterns.append(cur)
        elif cur is not None:
            k, _, v = ln.partition(":")
            if v.strip():
                cur["fields"][k.strip()] = v.strip()
    return {"layers": [layers[k] for k in LAYER_ORDER], "cross_layer": cross, "mitigations": mitigations,
            "steps": steps, "patterns": patterns}


@lru_cache(maxsize=1)
def _owasp() -> list[dict]:
    text = OWASP_SRC.read_text(encoding="utf-8").splitlines()
    out, seen = [], set()
    for k, ln in enumerate(text):
        m = re.match(r"^OWASP (ASI\d\d): (.+)$", ln)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            out.append({"id": m.group(1), "name": m.group(2).strip(), "summary": text[k + 1].strip(),
                        "layers": OWASP_TO_LAYERS.get(m.group(1), []), "layers_source": "course mapping"})
    return out


def layers() -> list[dict]:
    return _parsed()["layers"]


def layer(lid: str) -> dict:
    return next(x for x in layers() if x["id"] == lid)


def threat(tid: str) -> dict:
    """'L7.T5' → the fifth threat in Layer 7's landscape, or 'X3' → the third cross-layer threat."""
    if tid.startswith("X"):
        return next(x for x in _parsed()["cross_layer"] if x["id"] == tid)
    return next(t for t in layer(tid.split(".")[0])["threats"] if t["id"] == tid)


def cross_layer() -> list[dict]:
    return _parsed()["cross_layer"]


def mitigations() -> dict:
    return _parsed()["mitigations"]


def steps() -> list[dict]:
    return _parsed()["steps"]


def patterns() -> list[dict]:
    return _parsed()["patterns"]


def owasp() -> list[dict]:
    return _owasp()


def catalog() -> dict:
    """Everything, in the shape the runner's 🗺 MAESTRO map expects."""
    p = _parsed()
    return {"layers": p["layers"], "cross_layer": p["cross_layer"], "mitigations": p["mitigations"],
            "steps": p["steps"], "patterns": p["patterns"],
            "owasp": [{"id": o["id"], "name": o["name"], "layers": o["layers"]} for o in owasp()],
            "source": MAESTRO_URL, "owasp_source": OWASP_URL,
            "note": "Layer threats, cross-layer threats, mitigations, steps and patterns are parsed verbatim from "
                    "the CSA page. The OWASP→layer mapping is the course's own."}


# ── risk scoring (the course's matrix) ───────────────────────────────────────
SEVERITY_BANDS = [(15, "critical"), (10, "high"), (5, "medium"), (0, "low")]
SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def risk(likelihood: int, impact: int) -> tuple[int, str]:
    """1–5 × 1–5 → (score, severity). The course's matrix: ≥15 critical · ≥10 high · ≥5 medium · else low."""
    s = int(likelihood) * int(impact)
    return s, next(name for floor, name in SEVERITY_BANDS if s >= floor)


def heatmap(threats: list[dict], residual: bool = False, cell: int = 14) -> str:
    """A 5×5 text grid (impact up, likelihood across) with threat ids in each cell; '+n' when a cell overflows."""
    grid = {}
    for t in threats:
        L = t.get("residual_likelihood", t["likelihood"]) if residual else t["likelihood"]
        I = t.get("residual_impact", t["impact"]) if residual else t["impact"]
        grid.setdefault((int(L), int(I)), []).append(str(t["id"]))

    def fit(ids: list[str]) -> str:
        if not ids:
            return "·"
        shown = []
        for i, x in enumerate(ids):
            more = len(ids) - i - 1
            cand = ",".join(shown + [x]) + (f" +{more}" if more else "")
            if len(cand) > cell - 2:
                return ",".join(shown) + f" +{len(ids) - len(shown)}" if shown else x[:cell - 4] + f" +{len(ids) - 1}"
            shown.append(x)
        return ",".join(shown)

    rows = []
    for imp in range(5, 0, -1):
        rows.append(f"│ impact {imp} │" + "│".join(fit(grid.get((lik, imp), [])).center(cell) for lik in range(1, 6)) + "│")
    rows.append("│          └" + "┴".join("─" * cell for _ in range(5)) + "┘")
    rows.append("│ likelihood " + "".join(f"{k}".center(cell + 1) for k in range(1, 6)))
    return "\n".join(rows)


# ── threat model as code ─────────────────────────────────────────────────────
REQUIRED_THREAT = ("id", "title", "layers", "components", "likelihood", "impact", "mitigations", "status")
STATUSES = ("open", "mitigated", "accepted")


def load(path_or_text) -> dict:
    import yaml
    raw = str(path_or_text)
    try:
        is_file = "\n" not in raw and Path(raw).is_file()
    except OSError:                              # YAML text too long to be a file name
        is_file = False
    text = Path(raw).read_text(encoding="utf-8") if is_file else raw
    return yaml.safe_load(text) or {}


def validate(tm: dict) -> list[str]:
    """Schema + reference checks. An empty list means the threat model is well-formed."""
    errs = []
    comps = {c.get("id"): c for c in tm.get("components", []) or []}
    mits = {m.get("id") for m in tm.get("mitigations", []) or []}
    if not tm.get("system"):
        errs.append("missing 'system'")
    if not comps:
        errs.append("no components")
    for c in comps.values():
        if c.get("layer") not in LAYER_ORDER:
            errs.append(f"component {c.get('id')}: layer must be one of L1..L7, got {c.get('layer')!r}")
    ids = set()
    for t in tm.get("threats", []) or []:
        tid = t.get("id", "?")
        for k in REQUIRED_THREAT:
            if k not in t:
                errs.append(f"threat {tid}: missing '{k}'")
        if tid in ids:
            errs.append(f"threat {tid}: duplicate id")
        ids.add(tid)
        for lid in t.get("layers", []) or []:
            if lid not in LAYER_ORDER:
                errs.append(f"threat {tid}: unknown layer {lid}")
        for c in t.get("components", []) or []:
            if c not in comps:
                errs.append(f"threat {tid}: unknown component {c}")
        for m in t.get("mitigations", []) or []:
            if m not in mits:
                errs.append(f"threat {tid}: unknown mitigation {m}")
        for k in ("likelihood", "impact", "residual_likelihood", "residual_impact"):
            if k in t and not (isinstance(t[k], int) and 1 <= t[k] <= 5):
                errs.append(f"threat {tid}: {k} must be an integer 1..5")
        if t.get("status") not in STATUSES:
            errs.append(f"threat {tid}: status must be one of {STATUSES}")
        if t.get("status") == "mitigated" and not t.get("mitigations"):
            errs.append(f"threat {tid}: 'mitigated' but no mitigations listed")
        for ref in t.get("maestro", []) or []:
            try:
                threat(ref)
            except StopIteration:
                errs.append(f"threat {tid}: maestro ref {ref} is not in the catalogue")
    return errs


def effective(t: dict) -> tuple[int, str]:
    """Risk that still counts: residual for mitigated threats, inherent otherwise."""
    if t.get("status") == "mitigated":
        return risk(t.get("residual_likelihood", t["likelihood"]), t.get("residual_impact", t["impact"]))
    return risk(t["likelihood"], t["impact"])


def report_md(tm: dict) -> str:
    """A short markdown report: components by layer, threats ranked by risk, open items."""
    out = [f"# Threat model — {tm.get('system', '?')}", ""]
    if tm.get("version"):
        out += [f"Version {tm['version']}", ""]
    out += ["## Components by MAESTRO layer", "", "| layer | components |", "|---|---|"]
    for lid in LAYER_ORDER:
        names = [c["id"] for c in tm.get("components", []) if c.get("layer") == lid]
        out.append(f"| {lid} {layer(lid)['name']} | {', '.join(names) or '—'} |")
    out += ["", "## Threats, highest risk first", "",
            "| id | threat | layers | L×I | severity | status | effective |", "|---|---|---|---|---|---|---|"]
    ranked = sorted(tm.get("threats", []), key=lambda t: -risk(t["likelihood"], t["impact"])[0])
    for t in ranked:
        s, sev = risk(t["likelihood"], t["impact"])
        es, esev = effective(t)
        out.append(f"| {t['id']} | {t['title']} | {'/'.join(t['layers'])} | {t['likelihood']}×{t['impact']}={s} | "
                   f"{sev} | {t['status']} | {es} {esev} |")
    open_ = [t for t in ranked if t["status"] == "open"]
    out += ["", f"**Open threats:** {len(open_)}" + (f" — {', '.join(t['id'] for t in open_)}" if open_ else ""), "",
            "_Risk matrix: the course's 1–5 × 1–5 scale (≥15 critical · ≥10 high · ≥5 medium). MAESTRO asks for a "
            "likelihood × impact assessment but does not prescribe the numbers._"]
    return "\n".join(out)


def diff(old: dict, new: dict) -> dict:
    """Threat diffing (the idea from CSA's Feb 2026 CI/CD article): what a change added, removed or re-scored."""
    o = {t["id"]: t for t in old.get("threats", [])}
    n = {t["id"]: t for t in new.get("threats", [])}
    changed = [k for k in n if k in o and (effective(n[k]) != effective(o[k]) or n[k]["status"] != o[k]["status"])]
    return {"added": [k for k in n if k not in o], "removed": [k for k in o if k not in n], "changed": changed}


def gate(tm: dict, fail_on: str = "critical", only: list[str] | None = None) -> int:
    """CI gate: exit code 1 if any OPEN threat (or a mitigated one whose residual is still too high) is at or above
    `fail_on`. `only` restricts the check to some ids (e.g. the ones a diff says this change introduced)."""
    floor = SEVERITY_RANK[fail_on]
    bad = []
    for t in tm.get("threats", []):
        if only and t["id"] not in only:
            continue
        if t["status"] == "accepted":
            continue
        s, sev = effective(t)
        if SEVERITY_RANK[sev] >= floor:
            bad.append((t["id"], t["title"], s, sev))
    for tid, title, s, sev in bad:
        print(f"✕ gate: {tid} {title} — effective risk {s} ({sev}) ≥ {fail_on}")
    if not bad:
        print(f"✓ gate: no threat at or above '{fail_on}'")
    return 1 if bad else 0


if __name__ == "__main__":
    c = catalog()
    for L in reversed(c["layers"]):
        print(f"{L['id']} {L['name']:<32} {len(L['threats'])} threats{' (vertical)' if L['vertical'] else ''}")
    print(f"cross-layer: {', '.join(x['name'] for x in c['cross_layer'])}")
    print(f"steps: {' → '.join(s['name'] for s in c['steps'])}")
    print(f"patterns: {', '.join(p['name'] for p in c['patterns'])}")
    print(f"owasp: {', '.join(o['id'] + ' ' + o['name'] for o in c['owasp'])}")
    print(f"mitigations: {len(c['mitigations']['cross_layer'])} cross-layer · {len(c['mitigations']['ai_specific'])} AI-specific")
