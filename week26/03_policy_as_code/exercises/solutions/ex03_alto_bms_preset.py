#!/usr/bin/env python3
"""Exercise 03 · Alto BMS preset — reference solution.

The research tutorial's Part 2 exercise 1. Alto Ops Claw must read points from, and write setpoints to, the
hotel's BMS REST API at bms.alto.local:8443 (it resolves to 10.20.0.15). Only /usr/bin/python3 may call it:
  GET  /api/v1/points/**      allowed        POST /api/v1/setpoints/*   allowed
  POST /api/v1/admin/**       DENIED, even if an allow rule would match — enforced, not just audited.

Fill in the four TODOs, save, then run:
    .venv/bin/python week26/03_policy_as_code/exercises/ex03_alto_bms_preset.py

The checker is free and offline: policykit.validate, a policykit.decide() test matrix (a teaching model), and —
if the laptop CLI is installed — the real OpenShell 0.0.111 parser. Stuck? Compare with exercises/solutions/.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3] / "common"))
sys.path.insert(0, str(HERE.parents[2]))
import yaml  # noqa: E402

import parsekit  # noqa: E402
import policykit as pk  # noqa: E402
from clawkit import banner, check, note  # noqa: E402

# ── TODO 1 ── the endpoint: add port, protocol (L7 REST), enforcement (block, don't just log) and allowed_ips
#   (pin the one address the host resolves to, as a /32).
# ── TODO 2 ── rules: allow GET /api/v1/points/** and POST /api/v1/setpoints/*;
#   deny_rules: POST /api/v1/admin/**. Remember: `allow` wraps its matchers, deny_rules list them directly.
# ── TODO 3 ── binaries: only /usr/bin/python3.
ALTO_BMS_YAML = """
version: 1
network_policies:
  alto-bms:
    name: alto-bms
    endpoints:
      - host: bms.alto.local
        port: 8443
        protocol: rest
        enforcement: enforce
        allowed_ips: [10.20.0.15/32]
        rules:
          - allow: { method: GET,  path: "/api/v1/points/**" }
          - allow: { method: POST, path: "/api/v1/setpoints/*" }
        deny_rules:
          - { method: POST, path: "/api/v1/admin/**" }
    binaries:
      - { path: /usr/bin/python3 }
"""

# ── TODO 4 ── NemoClaw's `policy add --from-file` wants a `preset:` header whose name is a lowercase, hyphenated
#   RFC 1123 label (the NemoClaw applications playbook). Pick one for this preset.
PRESET_NAME = "alto-bms"


# ─────────────────────────── checker — no need to edit below ────────────────
PY, CURL, HOST, PORT = "/usr/bin/python3", "/usr/bin/curl", "bms.alto.local", 8443
MATRIX = [   # (binary, method, path, expected, what it proves)
    (PY, "GET", "/api/v1/points/ahu-1/sat", "allow", "read a point"),
    (PY, "GET", "/api/v1/points/chiller-2/kw", "allow", "read another point"),
    (PY, "POST", "/api/v1/setpoints/ahu-1", "allow", "write one setpoint"),
    (PY, "POST", "/api/v1/admin/users", "deny", "admin write — stopped by a deny_rule"),
    (PY, "POST", "/api/v1/admin/reboot/now", "deny", "deeper admin path — still denied"),
    (PY, "GET", "/api/v1/admin/users", "deny", "admin read — no allow rule"),
    (PY, "DELETE", "/api/v1/points/ahu-1", "deny", "a method nobody allowed"),
    (CURL, "GET", "/api/v1/points/ahu-1/sat", "deny", "the wrong binary"),
]
RUNS = HERE.parents[2] / ".runs"


def main() -> None:
    banner("Exercise 03 · Alto BMS preset", "offline checker · policykit (teaching model) + the real CLI parser",
           status=False)
    good = True
    try:
        policy = yaml.safe_load(ALTO_BMS_YAML) or {}
    except yaml.YAMLError as e:
        print(f"✕ ALTO_BMS_YAML is not valid YAML: {str(e).splitlines()[0]}")
        sys.exit(1)
    errs, warns = pk.validate(policy)
    warns = [w for w in warns if not w.startswith("nothing is writable")]   # a preset merges into a full policy
    good &= check(not errs and not warns, "policykit.validate: no errors, no warnings",
                  "TODOs 1–3: policykit says → " + "; ".join((errs + warns)[:2]))

    groups = policy.get("network_policies") or {}
    g = next(iter(groups.values()), {}) if isinstance(groups, dict) and len(groups) == 1 else {}
    eps = g.get("endpoints") or []
    e = eps[0] if len(eps) == 1 and isinstance(eps[0], dict) else {}
    shape = (e.get("host") == HOST and e.get("port") == PORT and e.get("protocol") == "rest"
             and e.get("enforcement") == "enforce" and "10.20.0.15/32" in [str(x) for x in e.get("allowed_ips") or []])
    good &= check(shape, "endpoint: bms.alto.local:8443 · rest · enforce · allowed_ips [10.20.0.15/32]",
                  f"TODO 1: one group, one endpoint with host {HOST}, port {PORT} (an integer), protocol rest, "
                  "enforcement enforce, allowed_ips containing 10.20.0.15/32")
    bins = [b.get("path") for b in g.get("binaries") or [] if isinstance(b, dict)]
    good &= check(bins == [PY], "binaries: /usr/bin/python3 only",
                  f"TODO 3: binaries is {bins or '[]'} — exactly one entry, {{path: {PY}}}")

    fails = []
    for binary, method, path, want, why in MATRIX:
        d, reason = pk.decide(policy, {"op": "http", "host": HOST, "port": PORT, "binary": binary,
                                       "method": method, "path": path})
        if d != want:
            fails.append(f"{method} {path} as {binary.rsplit('/', 1)[-1]} → {d}, want {want} ({why})")
        elif path == "/api/v1/admin/users" and method == "POST" and "deny_rule" not in reason:
            fails.append("POST /api/v1/admin/users is denied, but not by a deny_rule — add deny_rules")
    ip_d, _ = pk.decide(policy, {"op": "connect", "host": "10.20.0.16", "port": PORT, "binary": PY})
    if ip_d != "deny":
        fails.append("10.20.0.16 is reachable — allowed_ips should be a /32, not a wide range")
    good &= check(not fails, f"decide() matrix: {len(MATRIX) + 1}/{len(MATRIX) + 1} as expected (policykit)",
                  "TODO 2: " + (fails[0] if fails else ""))
    for f in fails[1:]:
        print(f"  ✕ also: {f}")

    good &= check(bool(re.fullmatch(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?", PRESET_NAME or "")),
                  f"preset name {PRESET_NAME!r} is a lowercase, hyphenated RFC 1123 label",
                  f"TODO 4: PRESET_NAME={PRESET_NAME!r} — lowercase letters, digits and hyphens only (no _)")

    RUNS.mkdir(parents=True, exist_ok=True)
    pol_file, preset_file = RUNS / "alto-bms.yaml", RUNS / "alto-bms.preset.yaml"
    pol_file.write_text(ALTO_BMS_YAML.lstrip(), encoding="utf-8")
    body = {k: v for k, v in policy.items() if k != "version"} if isinstance(policy, dict) else {}
    preset_file.write_text(yaml.safe_dump({"preset": {"name": PRESET_NAME or "alto-bms",
                                                      "description": "Alto BMS points + setpoints (python3 only)"},
                                           **body}, sort_keys=False), encoding="utf-8")
    if parsekit.OPENSHELL.exists():
        v, msg = parsekit.cli_file(pol_file)
        good &= check(bool(v), "the real OpenShell 0.0.111 parser accepts alto-bms.yaml (policy set, no gateway)",
                      f"the real parser refused it: {msg}")
        v2, msg2 = parsekit.cli_file(preset_file)
        note(f"the preset form (`preset:` header, for `nemoclaw … policy add --from-file`) is NOT an OpenShell "
             f"policy: {'parsed' if v2 else msg2.split(', expected')[0]}")
    else:
        note("no laptop OpenShell CLI — skipped the real-parser check")
    if not good:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)
    print(f"\n═ Done. Files: {pol_file.relative_to(HERE.parents[3])} and "
          f"{preset_file.relative_to(HERE.parents[3])}. Preview on the Spark with --dry-run first.")


if __name__ == "__main__":
    main()
