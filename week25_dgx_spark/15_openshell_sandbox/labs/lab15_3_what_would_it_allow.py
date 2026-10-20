#!/usr/bin/env python3
"""Lab 15-3 · "What would this policy allow?" — replay the NAT agent's actions against the policy.

A course-made TEACHING MODEL (policykit.decide), not OpenShell: it applies the semantics the playbooks
describe — Landlock read_only / read_write, default-deny egress, per-group binaries allow-lists,
inference.local routing, L7 method rules — to a list of things the Module 14 hotel agent (or a
prompt-injected version of it) might try. Then it previews a policy change the way
`openshell policy update --add-endpoint … --dry-run` would, and shows which decisions flip.
Finally it prints the commands that watch real decisions on the Spark (DRY unless connected).

Run: .venv/bin/python week25/15_openshell_sandbox/labs/lab15_3_what_would_it_allow.py
"""
import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE.parents[1]))
import policykit as pk  # noqa: E402
from sparkkit import banner, note, result, sh, step, table  # noqa: E402

MOD = HERE.parents[1]
POLICY = pk.load(MOD / "policies" / "hotel_agent_policy.yaml")
NAT_PY = "/sandbox/.venv-nat/bin/python3.12"

# (who, action) — the first block is the agent doing its job; the second is what a hijacked agent might try.
ACTIONS = [
    ("agent", {"op": "read", "path": "/sandbox/14_nat_agents/configs/hotel_agent.yml"}),
    ("agent", {"op": "write", "path": "/sandbox/.runs/tickets.jsonl"}),
    ("agent", {"op": "write", "path": "/tmp/nat_trace.jsonl"}),
    ("agent", {"op": "connect", "host": "inference.local", "port": 443, "binary": NAT_PY}),
    ("agent", {"op": "read", "path": "/etc/ssl/certs/ca-certificates.crt"}),
    ("setup", {"op": "connect", "host": "pypi.org", "port": 443, "binary": "/sandbox/.venv-nat/bin/pip"}),
    ("hijack", {"op": "connect", "host": "api.openai.com", "port": 443, "binary": NAT_PY}),
    ("hijack", {"op": "connect", "host": "pypi.org", "port": 443, "binary": "/usr/bin/curl"}),
    ("hijack", {"op": "connect", "host": "192.168.1.20", "port": 8000, "binary": NAT_PY}),
    ("hijack", {"op": "read", "path": "/home/sandbox/.ssh/id_ed25519"}),
    ("hijack", {"op": "write", "path": "/etc/cron.d/backdoor"}),
    ("hijack", {"op": "write", "path": "/usr/lib/python3/dist-packages/sitecustomize.py"}),
    ("hijack", {"op": "run_as", "user": "root"}),
]
GLYPH = {"allow": "✓ allow", "deny": "✕ deny", "inspect_for_inference": "→ inference"}


def run(policy: dict, actions) -> list[tuple[str, str]]:
    return [pk.decide(policy, a) for _, a in actions]


banner("Lab 15-3 · what would this policy allow?",
       "policykit.decide — a course-made teaching model of OpenShell's policy semantics, not OpenShell itself")

step(1, "the hotel agent's actions (and a hijacked agent's) against policies/hotel_agent_policy.yaml")
decisions = run(POLICY, ACTIONS)
table([[who, pk.fmt_action(a), GLYPH[d], why] for (who, a), (d, why) in zip(ACTIONS, decisions)],
      ["who", "action", "decision", "why (the rule that decided)"])
bad = [pk.fmt_action(a) for (who, a), (d, _) in zip(ACTIONS, decisions) if who == "hijack" and d != "deny"]
good = [pk.fmt_action(a) for (who, a), (d, _) in zip(ACTIONS, decisions) if who == "agent" and d == "deny"]
note(f"{sum(d == 'deny' for d, _ in decisions)} of {len(ACTIONS)} actions denied · hijack attempts that got through: "
     f"{bad or 'none'} · legitimate actions blocked: {good or 'none'}")

step(2, "preview a change: add GitHub read-only for curl (like `openshell policy update --add-endpoint … --dry-run`)")
print("$ openshell policy update hotel-agent --add-endpoint api.github.com:443:read-only:rest:enforce "
      "--binary /usr/bin/curl --dry-run   (the real preview needs a gateway; policykit models it below)")
changed = copy.deepcopy(POLICY)
changed["network_policies"]["github"] = pk.group(
    "github", [{"host": "api.github.com", "port": 443, "access": "read-only", "protocol": "rest",
                "enforcement": "enforce"}], ["/usr/bin/curl"])
probe = ACTIONS + [
    ("new?", {"op": "http", "host": "api.github.com", "port": 443, "binary": "/usr/bin/curl", "method": "GET",
              "path": "/repos/NVIDIA/OpenShell"}),
    ("new?", {"op": "http", "host": "api.github.com", "port": 443, "binary": "/usr/bin/curl", "method": "POST",
              "path": "/repos/NVIDIA/OpenShell/issues"}),
    ("new?", {"op": "http", "host": "api.github.com", "port": 443, "binary": NAT_PY, "method": "GET", "path": "/"}),
]
before, after = run(POLICY, probe), run(changed, probe)
rows = [[pk.fmt_action(a), GLYPH[b[0]], GLYPH[c[0]], "◆ CHANGED" if b[0] != c[0] else ""]
        for (_, a), b, c in zip(probe, before, after) if b != c or _ == "new?"]
table(rows, ["action", "before", "after", ""])
note("Only one new thing is possible after the change: curl may GET from api.github.com. POST is still denied "
     "(read-only), and the agent's own Python still cannot reach GitHub (binaries allow-list). OpenShell 0.1's "
     "prover does this kind of before/after check for real and holds risky changes for human review.")

step(3, "watch the real decisions on the Spark (playbook Step 11 + troubleshooting)")
sh("openshell logs hotel-agent -n 20 --source sandbox",
   example="… policy decision=deny dst=api.openai.com:443 binary=/sandbox/.venv-nat/bin/python3.12\n"
           "… policy decision=inspect_for_inference dst=inference.local:443", timeout=60)
note("Interactive: `openshell term` shows live allow / deny / inspect_for_inference decisions (f follow, s filter, "
     "q quit). Denied a host you need? Add it with `openshell policy update … --add-endpoint` or edit the YAML and "
     "`openshell policy set hotel-agent --policy <file> --wait` — network rules hot-reload; filesystem rules need "
     "a new sandbox.")

result("Every egress is denied unless a group names BOTH the endpoint and the binary; every write is denied "
       "outside read_write. Your agent keeps its tools and loses the ability to leak.")
