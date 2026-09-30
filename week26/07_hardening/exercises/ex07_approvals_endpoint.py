#!/usr/bin/env python3
"""Exercise 07 · The approvals endpoint: the claw may open tickets and read them, never approve them.

Part 6, exercise 4 of the research tutorial. Alto Copilot's approvals service (approvals.alto.local:443) is the
only way a write ever happens: the claw files a ticket, a human approves it outside the sandbox. Write the
network_policies entry so the claw can

    create a ticket   POST /api/v1/tickets
    read a ticket     GET  /api/v1/tickets/<id>

and can NEVER approve one (/api/v1/tickets/<id>/approve, whatever the method).

Fill in the four TODOs in APPROVALS_YAML, save, then run:
    .venv/bin/python week26/07_hardening/exercises/ex07_approvals_endpoint.py

The checker is free and offline: policykit.validate + harden + a decide() matrix. policykit is the course's
teaching model, not OpenShell. Stuck? Compare with exercises/solutions/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import policykit as pk  # noqa: E402
from clawkit import banner, check  # noqa: E402

APPROVALS_YAML = """
approvals:
  name: approvals
  endpoints:
    - host: approvals.alto.local
      port: 443
      protocol: rest
      enforcement: audit            # TODO 3 — audit logs a violation and FORWARDS it. Is that what you want?
      rules: []                     # TODO 1 — two allow rules: create a ticket, read one ticket
      deny_rules: []                # TODO 2 — one deny rule that really matches every approve URL
  binaries: []                      # TODO 4 — exactly one binary: the Python that runs the NAT claw
"""


# ─────────────────────────── checker — no need to edit below ────────────────
PROD = next(p for p in Path(__file__).resolve().parents if (p / "policies").is_dir()) / "policies" / "prod.yaml"
PY, CURL = "/usr/bin/python3.12", "/usr/bin/curl"
H = {"host": "approvals.alto.local", "port": 443}


def req(method: str, path: str, binary: str = PY) -> dict:
    return {"op": "http", **H, "binary": binary, "method": method, "path": path}


MATRIX = [  # (action, expected decision, why it matters)
    (req("POST", "/api/v1/tickets"), "allow", "create a ticket"),
    (req("GET", "/api/v1/tickets/42"), "allow", "read ticket 42"),
    (req("POST", "/api/v1/tickets/42/approve"), "deny", "approve ticket 42"),
    (req("PUT", "/api/v1/tickets/42/approve"), "deny", "approve with PUT"),
    (req("PATCH", "/api/v1/tickets/42/approve"), "deny", "approve with PATCH"),
    (req("GET", "/api/v1/tickets/42/approve"), "deny", "approve with a GET link"),
    (req("DELETE", "/api/v1/tickets/42"), "deny", "delete a ticket"),
    (req("POST", "/api/v1/admin/users"), "deny", "anything else"),
    (req("POST", "/api/v1/tickets", CURL), "deny", "same call from curl"),
]


def main() -> None:
    banner("Exercise 07 · approvals endpoint", "offline checker · policykit teaching model · no Spark needed",
           status=False)
    good = True
    try:
        entry = pk.load(APPROVALS_YAML)
    except Exception as e:  # noqa: BLE001
        print(f"✕ APPROVALS_YAML is not valid YAML: {e}")
        sys.exit(1)
    policy = pk.load(PROD)
    policy["network_policies"].update(entry)
    errs, _ = pk.validate(policy)
    good &= check(not errs, "policykit.validate: the production policy + your entry has 0 errors",
                  f"validate: {errs[0] if errs else ''}")
    ep = ((entry.get("approvals") or {}).get("endpoints") or [{}])[0]
    bins = [b.get("path") for b in (entry.get("approvals") or {}).get("binaries") or []]

    good &= check(ep.get("enforcement") == "enforce",
                  "TODO 3: enforcement: enforce — a denied approve gets a 403, not a log line",
                  f"TODO 3: enforcement is {ep.get('enforcement')!r} — under audit, a denied approve is "
                  "logged and then FORWARDED to the service")
    good &= check(bins == [PY], f"TODO 4: one binary, {PY}",
                  f"TODO 4: binaries are {bins or '[]'} — list exactly one: the Python that runs the claw")

    approve = req("POST", "/api/v1/tickets/42/approve")
    deny_hits = [d for d in ep.get("deny_rules") or []
                 if pk._match("POST", (d.get("method") or "*").upper()) and pk._match(approve["path"], d.get("path", "*"))]
    good &= check(bool(deny_hits), "TODO 2: your deny_rule really matches POST /api/v1/tickets/42/approve",
                  "TODO 2: no deny_rule matches POST /api/v1/tickets/42/approve — a deny rule that never "
                  "matches a real URL is dead code (see the tutorial's `/api/v1/tickets//approve`)")

    wrong = []
    for action, want, why in MATRIX:
        if errs:
            break
        d, reason = pk.decide(policy, action)
        if d == "allow" and "audit" in reason:
            d = "allow (audit)"
        if d != want:
            wrong.append(f"{why}: {pk.fmt_action(action)} → {d}, want {want}")
    good &= check(not errs and not wrong,
                  f"TODO 1 + 2: all {len(MATRIX)} decisions right — create ✓ read ✓ approve ✕ (every method) "
                  "delete ✕ curl ✕",
                  "TODO 1/2: " + ("fix validate first" if errs else f"{len(wrong)} wrong: " + " · ".join(wrong[:3])))

    new = [f for f in pk.harden(policy) if f[1].startswith("approvals") and f[0] in ("HIGH", "MEDIUM")]
    good &= check(not new, "harden: no HIGH/MEDIUM finding on the approvals entry",
                  f"harden: {new[0][0]} {new[0][2]}" if new else "")
    if not good:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)
    print("\n═ Done. The claw can ask for a write; only a human, outside the sandbox, can grant one.")


if __name__ == "__main__":
    main()
