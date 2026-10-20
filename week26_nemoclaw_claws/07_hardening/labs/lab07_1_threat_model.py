#!/usr/bin/env python3
"""Lab 07-1 · Threat model: one prompt-injected agent, two policies, twelve attack steps.

Offline, runs anywhere. The research tutorial's §6.1 threat model (assets, adversaries, NemoClaw's documented
limitations) turned into something you can run: a prompt-injected Alto Ops Claw tries twelve concrete steps,
and the course's policykit teaching model decides each one twice, once against a COURSE-MADE first-draft
Balanced policy (policies/balanced_draft.yaml) and once against the tutorial's production policy
(policies/prod.yaml, Lab 6.1, verbatim).

policykit.decide() is NOT OpenShell. The real enforcement is Landlock, seccomp and the egress proxy on the
Spark. The model follows the semantics the docs describe so you can reason about a policy before you push it.

Run: .venv/bin/python week26/07_hardening/labs/lab07_1_threat_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import policykit as pk  # noqa: E402
from clawkit import banner, note, ok, result, step, table, warn  # noqa: E402

MOD = Path(__file__).resolve().parents[1]
PY, CURL, PIP = "/usr/bin/python3.12", "/usr/bin/curl", "/usr/bin/pip"

banner("Lab 07-1 · threat model", "offline · policykit teaching model · draft vs production policy", status=False)

step(1, "what is worth stealing, and who is trying (research tutorial §6.1)")
table([
    ["sandbox filesystem", "/sandbox, the agent's own config tree", "filesystem (Landlock)"],
    ["provider credentials", "model API keys", "gateway (credential providers, inference.local)"],
    ["channel tokens", "Telegram / Slack bot tokens", "gateway + network (each token is an outbound path)"],
    ["CSV / BMS data", "chiller exports, live points", "network (where can it be sent?)"],
    ["tool authority", "write_setpoint — the most valuable asset", "network L7 (MCP rules) + approvals service"],
], ["asset", "example", "layer that limits the damage"])
table([
    ["indirect prompt injection", "anything the agent reads: web pages, emails, MCP tool results, files"],
    ["malicious skill / plugin", "a hub download that runs inside the harness"],
    ["compromised dependency", "a package pulled through the npm / pypi presets"],
    ["insider with host access", "someone who can docker exec or edit the host"],
], ["adversary", "way in"])
note("Every adversary above except the insider starts the same way: the model reads text an attacker "
     "controls. So the question is never 'is the prompt good?' but 'what can the process do next?'")

step(2, "load the two policies")
policies = {}
for name, fname in (("draft", "balanced_draft.yaml"), ("prod", "prod.yaml")):
    p = pk.load(MOD / "policies" / fname)
    errs, warns = pk.validate(p)
    for e in errs:
        print(f"✕ {fname}: {e}")
    for w in warns:
        warn(f"{fname}: {w}")
    if not errs:
        ok(f"{fname}: valid · {len(p['network_policies'])} network groups · landlock "
           f"{p['landlock']['compatibility']}")
    policies[name] = p

step(3, "the attack: a prompt-injected agent tries twelve steps")
ATTACK = [
    ("phone home to a cloud model", {"op": "connect", "host": "api.openai.com", "port": 443, "binary": CURL}),
    ("read its own secrets file", {"op": "read", "path": "/sandbox/.hermes/.env"}),
    ("POST the secrets to an attacker host", {"op": "http", "host": "paste.attacker.example", "port": 443,
                                              "binary": PY, "method": "POST", "path": "/upload"}),
    ("hide the secrets in a GET URL", {"op": "http", "host": "pypi.org", "port": 443, "binary": PIP,
                                       "method": "GET", "path": "/simple/c2stbGl2ZS0xMjM0/"}),
    ("cloud metadata (SSRF)", {"op": "connect", "host": "169.254.169.254", "port": 80, "binary": CURL}),
    ("write_setpoint over MCP", {"op": "mcp", "host": "bms.alto.local", "port": 8443, "binary": PY,
                                 "tool": "write_setpoint"}),
    ("pip install a package", {"op": "http", "host": "files.pythonhosted.org", "port": 443, "binary": PIP,
                               "method": "GET", "path": "/packages/evil-1.0.tar.gz"}),
    ("write to /usr (plant a sitecustomize.py)", {"op": "write", "path": "/usr/lib/python3/sitecustomize.py"}),
    ("rewrite its own harness config", {"op": "write", "path": "/sandbox/.openclaw/openclaw.json"}),
    ("send data out as 'telemetry'", {"op": "http", "host": "otel.alto.local", "port": 4318, "binary": PY,
                                      "method": "POST", "path": "/v1/logs"}),
    ("become root", {"op": "run_as", "user": "root"}),
    ("call the BMS with curl instead", {"op": "mcp", "host": "bms.alto.local", "port": 8443, "binary": CURL,
                                        "tool": "read_point"}),
]


def verdict(d: str, why: str) -> str:
    if d == "allow" and "audit" in why:
        return "⚠ allowed (audit)"
    return {"allow": "✓ allowed", "deny": "✕ denied", "inspect_for_inference": "◆ inference"}[d]


rows, wins, seen = [], {"draft": 0, "prod": 0}, {}
for i, (what, action) in enumerate(ATTACK, 1):
    for name in ("draft", "prod"):
        seen[(i, name)] = pk.decide(policies[name], action)
        wins[name] += seen[(i, name)][0] == "allow"
    rows.append([f"A{i}", what, pk.fmt_action(action), verdict(*seen[(i, "draft")]), verdict(*seen[(i, "prod")])])
table(rows, ["#", "attacker step", "action", "draft", "prod"])
print("  (for the attacker, ✓ allowed means the step worked)")

step(4, "why: the reason policykit gives for each step that changed")
for i, (what, _) in enumerate(ATTACK, 1):
    if verdict(*seen[(i, "draft")]) != verdict(*seen[(i, "prod")]):
        print(f"→ A{i} {what}")
        print(f"    draft: {seen[(i, 'draft')][1]}")
        print(f"    prod : {seen[(i, 'prod')][1]}")

step(5, "the tally")
n = len(ATTACK)
table([[name, f"{wins[name]} / {n}", "█" * wins[name] + "░" * (n - wins[name])] for name in ("draft", "prod")],
      ["policy", "steps that worked", ""])
note("Two steps work under BOTH policies: reading /sandbox/.hermes/.env and rewriting /sandbox/.openclaw. "
     "include_workdir: true makes /sandbox writable, and the agent owns its config tree. The docs say the same: "
     "that tree is not an isolation boundary. The fix is not a filesystem rule. Keep secrets out of files (use "
     "OpenShell credential handles), and make sure nothing the agent reads can LEAVE: under prod there is no "
     "host left to send it to.")

step(6, "what a policy cannot fix — NemoClaw's documented limitations (research tutorial §6.1)")
table([
    ["bypassing managed gateway paths", "policy + inference auth not enforced for runtimes started elsewhere",
     "start agents only via the managed entrypoints; never docker exec a second agent in"],
    ["same-UID native lifecycle", "supervisor, gateway and agent share the sandbox UID",
     "put nothing in the sandbox you would not give the agent"],
    ["raw filesystem writes bypass scanners", "scanners see tool calls, not `echo secret > file`",
     "Landlock write scoping; keep secrets out of files"],
    ["encoded secrets undetected", "regex redaction misses Base64 / hex", "credential handles, not file secrets"],
], ["limitation", "why it matters", "mitigation"])
note("A4 above hides Base64 in a URL path. A regex redactor (like the redact-secrets middleware in prod.yaml, "
     "which only covers bms.alto.local) would likely miss it: limitation 4. What stops A4 under prod is that "
     "pypi.org is not in the policy at all.")
note("policykit.decide() models none of these four rows. They live outside the policy file: in how you start "
     "agents, what you put in the image, and where secrets are stored.")

result(f"draft: {wins['draft']}/{n} attacker steps work · prod: {wins['prod']}/{n}. The two that survive prod are "
       "filesystem reads/writes inside /sandbox, and prod leaves them nowhere to go.")
