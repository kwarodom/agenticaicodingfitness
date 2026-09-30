#!/usr/bin/env python3
"""Lab 05-4 · The policy plane: reading OpenShell like a trace.

Research tutorial Lab 4.4 (L4.4). Three parts:
  1. ON THIS LAPTOP, for real: the research tutorial's OpenShell log commands, parsed by the OpenShell 0.0.111 CLI
     against a dead gateway (argument errors come back at once; "Connection refused" = it parsed). This also
     shows that `logs --tail` FOLLOWS (streams live) — so on the Spark the lab always bounds it with `timeout 10`;
  2. ON THE SPARK, read-only via sh(): bounded logs, `settings get`, `policy get --full`, and the supervisor's
     docker logs grepped for the two healthy-start lines. DRY → EXAMPLE shapes, never invented output.
     `openshell term` is an interactive TUI: shown, never run by a lab;
  3. ON THIS LAPTOP: a parser for a decision stream. The stream is an EXAMPLE shape written for the course whose
     decisions come from the course's policykit teaching model (not from OpenShell); the parser counts
     allow / deny / inspect_for_inference and reports audit-mode violations as FINDINGS, not blocks.

Run: .venv/bin/python week26/05_tracing/labs/lab05_4_policy_plane.py
"""
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import policykit as pk  # noqa: E402
from clawkit import banner, note, ok, openshell_offline, parsed_ok, result, sandbox, sh, step, table, warn  # noqa: E402

SB = sandbox("alto-ops")                                 # the sandboxed NAT claw from research Lab 3.8

banner("Lab 05-4 · the policy plane", "OpenShell CLI offline for real · Spark read-only (or DRY) · a decision-stream parser")

step(1, "the research tutorial's log commands — parsed by the real OpenShell CLI (0.0.111, offline)")
CMDS = [
    (["logs", SB, "--tail", "--source", "sandbox"], "denied host, path, binary — FOLLOWS: bound it on the Spark"),
    (["logs", SB, "-n", "50", "--since", "10m", "--source", "sandbox"], "a bounded alternative: last 50 lines, 10 min"),
    (["settings", "get", SB], "effective policy source"),
    (["policy", "get", SB, "--full"], "what is really enforced now (incl. provider-composed entries)"),
]
rows = []
for args, why in CMDS:
    r = openshell_offline(args)
    rows.append(["openshell " + " ".join(args), "✓ parsed" if parsed_ok(r) else f"✕ {r.out.strip()[:60]}", why])
table(rows, ["command", "laptop CLI 0.0.111", "what it answers"])
r = openshell_offline(["logs", "--help"])
tail_help = next((ln.strip() for ln in r.out.splitlines() if ln.strip().startswith("--tail")), "")
if tail_help:
    note(f"`openshell logs --help` on this laptop: {tail_help!r} → it never returns on its own. The lab wraps it "
         "in `timeout 10` on the Spark, and the runner never follows a log in the foreground.")
print("$ openshell term   [interactive TUI — you run it in the ⌨ terminal; never run by a lab]")
note("the OpenShell playbook: the TUI's live log stream shows outbound connections, policy decisions (allow, deny, "
     "inspect_for_inference) and inference interceptions; press f to follow, s to filter by source, q to quit")

step(2, f"on the Spark: the policy plane of sandbox {SB} (read-only)")
sh(f"timeout 10 openshell logs {SB} --tail --source sandbox; echo \"(stopped after 10 s, exit $?)\"", timeout=30,
   example=("<ts> INFO  sandbox  CONNECT /usr/bin/python3.12 -> inference.local:443  inspect_for_inference\n"
            "<ts> WARN  sandbox  CONNECT /usr/bin/python3.12 -> api.open-meteo.com:443  deny (no matching policy)\n"
            "(stopped after 10 s, exit 124)"))
sh(f"openshell settings get {SB}", timeout=30,
   example="(effective settings for the sandbox — including where its policy comes from)")
sh(f"openshell policy get {SB} --full", timeout=30,
   example="(the current policy revision + the full effective YAML: filesystem_policy, process, network_policies …)")
sh(f"docker logs $(docker ps --filter name=openshell-{SB} --format '{{{{.Names}}}}') --tail 50 2>&1 "
   "| grep -E 'OpenShell Sandbox Supervisor success|Applying Landlock filesystem sandbox'", timeout=30,
   example="<ts> … Applying Landlock filesystem sandbox …\n<ts> … OpenShell Sandbox Supervisor success …")
note("the OpenShell playbook: look for `OpenShell Sandbox Supervisor success` and `Applying Landlock filesystem "
     "sandbox` in the supervisor's docker logs — both lines = a healthy start with Landlock applied")

step(3, "a decision stream, parsed — EXAMPLE shape, decisions from policykit (not from OpenShell)")
PY = "/usr/bin/python3.12"
policy = pk.load(f"""
version: 1
filesystem_policy:
  include_workdir: true
  read_only: [/usr, /lib, /etc]
  read_write: [/tmp]
landlock:
  compatibility: best_effort
process:
  run_as_user: sandbox
  run_as_group: sandbox
network_policies:
  otel_collector:
    name: otel_collector
    endpoints:
      - {{ host: otel.alto.local, port: 4318, protocol: rest, enforcement: audit, access: read-only }}
    binaries:
      - {{ path: {PY} }}
  bms_mcp:
    name: bms_mcp
    endpoints:
      - host: bms.alto.local
        port: 8443
        path: /mcp
        protocol: mcp
        enforcement: enforce
        rules:
          - allow: {{ method: tools/call, tool: {{ any: [read_point, list_alarms] }} }}
        deny_rules:
          - {{ method: tools/call, tool: write_setpoint }}
    binaries:
      - {{ path: {PY} }}
""")
errs, _ = pk.validate(policy)
print("✓ policy valid (policykit)" if not errs else f"✕ {errs}")
# One Alto Ops request's egress, in order: 2 LLM calls, 1 BMS read, 1 web fetch the agent was asked for,
# 1 attempted write, the trace export at the end, and an SSRF probe.
ACTIONS = [
    {"op": "http", "host": "inference.local", "port": 443, "binary": PY, "method": "POST", "path": "/v1/chat/completions"},
    {"op": "mcp", "host": "bms.alto.local", "port": 8443, "binary": PY, "method": "tools/call", "tool": "read_point"},
    {"op": "connect", "host": "api.open-meteo.com", "port": 443, "binary": PY},
    {"op": "mcp", "host": "bms.alto.local", "port": 8443, "binary": PY, "method": "tools/call", "tool": "write_setpoint"},
    {"op": "http", "host": "inference.local", "port": 443, "binary": PY, "method": "POST", "path": "/v1/chat/completions"},
    {"op": "http", "host": "otel.alto.local", "port": 4318, "binary": PY, "method": "POST", "path": "/v1/traces"},
    {"op": "connect", "host": "169.254.169.254", "port": 80, "binary": "/usr/bin/curl"},
]
lines = []
for i, a in enumerate(ACTIONS):
    d, why = pk.decide(policy, a)
    audit = "enforcement is audit" in why
    tag = "allow (audit violation)" if audit else d
    lines.append(f"t+{i * 1.7:04.1f}s sandbox {SB} {tag:24s} {pk.fmt_action(a)} · {why}")
lines.append(f"t+12.0s sandbox {SB} finding HIGH  DetectionFinding · Landlock best_effort skipped a path that does "
             "not exist in the image")
print("◈ EXAMPLE — illustrative shape only (course format; decisions from policykit, not OpenShell output):")
for ln in lines:
    print("  " + ln)

LINE = re.compile(r"^t\+(?P<t>[\d.]+)s sandbox (?P<sb>\S+) (?P<decision>allow \(audit violation\)|allow|deny|"
                  r"inspect_for_inference|finding)\s+(?P<rest>.*)$")
counts, findings = Counter(), []
for ln in lines:
    m = LINE.match(ln)
    if not m:
        continue
    d = m["decision"]
    if d == "allow (audit violation)":
        counts["allow"] += 1
        findings.append(("MEDIUM", "audit-mode violation — traffic was FORWARDED; fix the policy before enforce",
                         m["rest"].split(" · ")[0]))
    elif d == "finding":
        findings.append(("HIGH", "OCSF DetectionFinding", m["rest"].split(" · ")[-1]))
    else:
        counts[d] += 1
table([[k, counts.get(k, 0)] for k in ("allow", "deny", "inspect_for_inference")], ["decision", "count"])
for sev, what, where in findings:
    print(f"{'✕' if sev == 'HIGH' else '⚠'} {sev} · {what} · {where}")
note("audit vs enforce: an L7 violation under `enforcement: audit` is LOGGED and the request goes through — a "
     "finding to fix, not a block. The same rule under `enforce` returns 403 (lab 05-2, exercise 05 b).")
note("a Landlock path skipped under `best_effort` shows up as a High-severity OCSF DetectionFinding (OpenShell "
     "security best practices, cited in the research tutorial)")
n_inf = counts.get("inspect_for_inference", 0)
ok(f"{n_inf} inspect_for_inference events = the {n_inf} LLM calls of this EXAMPLE request — lab 05-5 checks that "
   "number against a real NAT trace")
if counts.get("deny"):
    warn(f"{counts['deny']} deny: the weather fetch, the write_setpoint call and the metadata-IP probe — in the agent "
         "plane you would see the matching tool span fail")
result("Policy plane = what the agent TRIED to reach. Read allow / deny / inspect_for_inference like spans.")
