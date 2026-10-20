#!/usr/bin/env python3
"""Exercise 03 · New threats: threat-model a housekeeping robot agent that can unlock guest-room doors.

The design change (fictional, simulated hotel): a housekeeping robot joins the ensemble. It takes jobs from
the orchestrator ("clean room 1203") and has ONE dangerous tool, unlock_door(room). Model it the way
Lab 03-3 modelled the LINE channel — as a delta on data/threat_model.yaml.

Fill in the four TODOs in CHANGE_YAML, save, then run:
    .venv/bin/python week28_maestro_mas_security/03_threat_model_as_code/exercises/ex03_new_threats.py

The checker is offline and instant: it merges your delta into v1, then runs validate, the CI gate
(fail_on="high", only your new ids) and a few sanity rules. Stuck? Compare with exercises/solutions/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import maestro  # noqa: E402
import tmkit  # noqa: E402
import yaml  # noqa: E402
from maskit import banner, check, finish  # noqa: E402

CHANGE_YAML = """
version: 2

components:
  # ── TODO 1 ── which MAESTRO layer does a robot AGENT live on? ("L1" … "L7")
  - {id: housekeeping_robot, layer: "", kind: agent, trust: internal}

# ── TODO 3 ── at least two mitigations: id, title, control (one sentence), and defense: when an
#   ensemble.Defenses switch fits (principal_authz, hitl, tool_allowlist, param_bounds, audit_chain …).
mitigations: []

# ── TODO 2 ── three threats R01–R03: layers, components, maestro ids (at least one cross-layer X…),
#   owasp ids, likelihood and impact 1–5. Unlocking the wrong door is physical: think about impact.
# ── TODO 4 ── status + residuals so the gate passes: "mitigated" needs residual_likelihood and
#   residual_impact lower in total than the inherent score; "accepted" with impact 5 needs a reason:.
threats:
  - id: R01
    title: A guest message makes the robot unlock another guest's door
    layers: []
    components: [housekeeping_robot]
    maestro: []
    owasp: []
    likelihood: 0
    impact: 0
    mitigations: []
    status: open
  - id: R02
    title: ""
    layers: []
    components: [housekeeping_robot]
    maestro: []
    owasp: []
    likelihood: 0
    impact: 0
    mitigations: []
    status: open
  - id: R03
    title: ""
    layers: []
    components: [housekeeping_robot]
    maestro: []
    owasp: []
    likelihood: 0
    impact: 0
    mitigations: []
    status: open
"""


# ─────────────────────────── checker — no need to edit below ────────────────
def main() -> None:
    banner("Exercise 03 · new threats", "offline checker · instant", brain=False)
    v1 = maestro.load(Path(__file__).resolve().parents[1] / "data" / "threat_model.yaml")
    change = yaml.safe_load(CHANGE_YAML) or {}      # (maestro.load() treats long text as a file name)
    v2 = tmkit.merge(v1, change)
    new = [t for t in change.get("threats", []) or []]
    ids = [t.get("id") for t in new]
    good = True

    comp = (change.get("components") or [{}])[0]
    good &= check(comp.get("layer") == "L7", "TODO 1: housekeeping_robot is an agent → L7 Agent Ecosystem",
                  f"TODO 1: housekeeping_robot layer = {comp.get('layer')!r} — agents live on the ecosystem layer")

    errs = maestro.validate(v2)
    good &= check(not errs, "validate: the merged model is well-formed",
                  f"validate: {len(errs)} errors — " + "; ".join(errs[:4]) + (" …" if len(errs) > 4 else ""))

    titled = [t for t in new if str(t.get("title", "")).strip()]
    refs = [r for t in new for r in t.get("maestro", []) or []]
    good &= check(len(new) == 3 and len(set(ids)) == 3 and len(titled) == 3
                  and all(t.get("maestro") and t.get("layers") for t in new),
                  "TODO 2: three titled threats, each with layers and MAESTRO refs",
                  f"TODO 2: need R01–R03, each with a title, ≥1 layer and ≥1 maestro id (have {len(titled)} titled, "
                  f"{sum(1 for t in new if t.get('maestro') and t.get('layers'))} with layers + refs)")
    good &= check(any(r.startswith("X") for r in refs), "TODO 2: at least one cross-layer threat (X…)",
                  "TODO 2: add a cross-layer id — a robot that opens doors for the wrong principal is X3 territory")
    impacts = [t.get("impact") for t in new if isinstance(t.get("impact"), int)]
    good &= check(any(i >= 4 for i in impacts), "TODO 2: the door-unlock threat is scored as serious (impact ≥ 4)",
                  "TODO 2: an unlocked guest-room door is a physical-safety harm — no impact ≥ 4 yet")

    mits = change.get("mitigations", []) or []
    good &= check(len(mits) >= 2 and all(str(m.get("control", "")).strip() for m in mits),
                  f"TODO 3: {len(mits)} mitigations, each says what the control does",
                  f"TODO 3: need ≥ 2 mitigations with a control: sentence (have {len(mits)})")

    sane = []
    for t in new:
        L, I = t.get("likelihood"), t.get("impact")
        if t.get("status") == "accepted" and I == 5 and not str(t.get("reason", "")).strip():
            sane.append(f"{t.get('id')}: impact-5 threat accepted without a reason:")
        if t.get("status") == "mitigated":
            rl, ri = t.get("residual_likelihood"), t.get("residual_impact")
            if not (isinstance(rl, int) and isinstance(ri, int)):
                sane.append(f"{t.get('id')}: mitigated without residual_likelihood/residual_impact")
            elif isinstance(L, int) and isinstance(I, int) and rl * ri >= L * I:
                sane.append(f"{t.get('id')}: residual {rl}×{ri} is not lower than inherent {L}×{I}")
    good &= check(not sane and all(t.get("status") in ("open", "mitigated", "accepted") for t in new),
                  "TODO 4: statuses and residuals make sense",
                  "TODO 4: " + ("; ".join(sane) or "set a status on every threat"))

    if not errs:
        print("◆ gate fail_on=high, only your new threats:")
        code = maestro.gate(v2, fail_on="high", only=ids)
    else:
        code = 1
    good &= check(code == 0, "gate: nothing high or critical left open in your change",
                  "gate: the change still carries a high/critical threat (or does not validate yet)")
    finish(good, "Done. That YAML delta is what a pull request adding the robot would carry.")


if __name__ == "__main__":
    main()
