#!/usr/bin/env python3
"""tmkit — Module 03's small threat-model helpers on top of common/maestro.py, plus a CI gate CLI.

    import tmkit
    tm2 = tmkit.merge(tm1, change)                  # apply a design change (new components/mitigations/threats)
    blocked, why = tmkit.verify(tm, threat)         # does the named defense block the threat's red-team scenario?
    tmkit.show_heatmap(tm["threats"], residual=True)

As a CLI (what a CI job runs — the course's own gate, not TITO):

    .venv/bin/python week28_maestro_mas_security/03_threat_model_as_code/tmkit.py --base base.yaml --head head.yaml --fail-on critical

It diffs base → head and gates ONLY the threats the change added or re-scored. Exit 0 = pass, 1 = fail.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import maestro  # noqa: E402

MODULE = Path(__file__).resolve().parent
V1 = MODULE / "data" / "threat_model.yaml"


def show_heatmap(threats: list[dict], residual: bool = False) -> None:
    """The same 5×5 grid as maestro.heatmap(), but each cell grows to list every id in it (the kit's
    fixed-width cells abbreviate a crowded cell as "T01,T06 +2")."""
    grid: dict[tuple[int, int], list[str]] = {}
    for t in threats:
        L = t.get("residual_likelihood", t["likelihood"]) if residual and t.get("status") == "mitigated" else t["likelihood"]
        I = t.get("residual_impact", t["impact"]) if residual and t.get("status") == "mitigated" else t["impact"]
        grid.setdefault((int(L), int(I)), []).append(t["id"])
    w = max([11] + [len(",".join(v)) + 2 for v in grid.values()])
    for imp in range(5, 0, -1):
        cells = [(",".join(grid.get((lik, imp), [])) or "·").center(w) for lik in range(1, 6)]
        print(f"│ impact {imp} │" + "│".join(cells) + "│")
    print("│          └" + "┴".join("─" * w for _ in range(5)) + "┘")
    print("│           " + "".join(f"L={k}".center(w + 1) for k in range(1, 6)) + " ← likelihood")


def merge(base: dict, change: dict) -> dict:
    """A design change as data: its components, mitigations and threats are appended to a copy of base."""
    out = copy.deepcopy(base)
    for key in ("components", "mitigations", "threats"):
        out.setdefault(key, [])
        out[key] += copy.deepcopy(change.get(key, []) or [])
    out["version"] = change.get("version", (base.get("version") or 0) + 1)
    return out


def defenses_for(tm: dict, t: dict):
    """The ensemble.Defenses switches named by a threat's mitigations (spotlight uses the per-run nonce)."""
    from ensemble import Defenses
    by_id = {m["id"]: m for m in tm.get("mitigations", [])}
    names = [by_id[m]["defense"] for m in t.get("mitigations", []) if by_id.get(m, {}).get("defense")]
    return Defenses(**{n: ("nonce" if n == "spotlight" else True) for n in names}), names


def verify(tm: dict, t: dict, brain: str = "sim") -> tuple[bool | None, str]:
    """Run the threat's red-team scenario with ONLY its mitigations' defenses on (SIM brain by default:
    evidence should be reproducible, and a full model costs a dozen scenarios).
    → (True, …) blocked · (False, …) still harmful · (None, …) nothing to run in the simulator."""
    from redteam import SCENARIOS, run_scenario
    sid = t.get("scenario")
    d, names = defenses_for(tm, t)
    if not sid:
        return None, "no red-team scenario: cannot be checked in the simulator"
    if not names:
        return None, "no mitigation names a Defenses switch"
    s = next(x for x in SCENARIOS if x.id == sid)
    out = run_scenario(s, d, brain=brain)
    if out.harms:
        return False, f"{sid} with {'+'.join(names)} → still harmful: {', '.join(out.harms)}"
    return True, f"{sid} with {'+'.join(names)} → blocked ({out.denials} denial{'s' if out.denials != 1 else ''})"


def gate_change(base: dict, head: dict, fail_on: str = "critical") -> int:
    """Threat diffing + gate: judge a change only by what it added or re-scored."""
    d = maestro.diff(base, head)
    print(f"◆ diff v{base.get('version', '?')} → v{head.get('version', '?')}: added {d['added'] or '—'} · "
          f"changed {d['changed'] or '—'} · removed {d['removed'] or '—'}")
    errs = maestro.validate(head)
    if errs:
        for e in errs:
            print(f"✕ validate: {e}")
        return 1
    ids = d["added"] + d["changed"]
    if not ids:
        print("✓ gate: this change adds or re-scores no threats")
        return 0
    return maestro.gate(head, fail_on=fail_on, only=ids)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Threat-model gate for a pull request (course tool).")
    ap.add_argument("--base", default=str(V1), help="threat model YAML on the base branch")
    ap.add_argument("--head", required=True, help="threat model YAML on the PR branch")
    ap.add_argument("--fail-on", default="critical", choices=list(maestro.SEVERITY_RANK))
    a = ap.parse_args(argv)
    return gate_change(maestro.load(a.base), maestro.load(a.head), a.fail_on)


if __name__ == "__main__":
    sys.exit(main())
