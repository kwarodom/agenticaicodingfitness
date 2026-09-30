#!/usr/bin/env python3
"""Lab 07-2 · L6.1 — The production policy: validate, lint, break it on purpose, then apply it.

Laptop-real first. The research tutorial's Lab 6.1 policy (policies/prod.yaml, verbatim) goes through three
checkers that each see different things: the REAL OpenShell CLI parser on this laptop (0.0.111, against a dead
gateway), policykit.validate() (the course's schema + semantics check) and policykit.harden() (the course's
§6.2 checklist lint). Then twelve deliberately weakened copies show which checker catches which mistake.

Then the mock BMS starts on this laptop, and its REAL tool list is checked against the policy. A small
course-made gate forwards the allowed calls to the real server and refuses write_setpoint. Last, the Spark:
`openshell policy set alto-ops --policy prod.yaml --wait` through change() (only with 🔓, CLAW_APPLY=1),
with a read-only preview, and the deny-proof log check (EXAMPLE in DRY mode).

policykit is a TEACHING MODEL and a lint, NOT OpenShell and NOT OpenShell's prover.

Run: .venv/bin/python week26/07_hardening/labs/lab07_2_prod_policy.py
"""
import asyncio
import copy
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import policykit as pk  # noqa: E402
from clawkit import (COMMON, NAT_PY, background, banner, change, free_port, note, ok, openshell_offline,  # noqa: E402
                     parsed_ok, put, result, sandbox, sh, step, table, warn)

MOD = Path(__file__).resolve().parents[1]
RUNS = MOD / ".runs"
HOME = RUNS / "openshell-home"
PROD = MOD / "policies" / "prod.yaml"
SB = sandbox("alto-ops")
PY, CURL = "/usr/bin/python3.12", "/usr/bin/curl"
RUNS.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    r = os.path.relpath(path)
    return str(path) if r.startswith("..") else r


def cli_verdict(path: Path) -> str:
    """Ask the real laptop OpenShell CLI to parse the file (the dead gateway makes it stop right after)."""
    r = openshell_offline(["policy", "set", SB, "--policy", rel(path), "--wait"], home=HOME)
    if parsed_ok(r):
        return "parsed"
    msg = " ".join(r.out.split())
    return "✕ " + (msg.split("╰─▶", 1)[-1].strip()[:70] if "╰─▶" in msg else msg[:70])


banner("Lab 07-2 · the production policy (L6.1)", "laptop: real CLI parser + policykit · Spark: policy set via change()")

step(1, "three checkers on the tutorial's policy (policies/prod.yaml)")
policy = pk.load(PROD)
r = openshell_offline(["policy", "set", SB, "--policy", rel(PROD), "--wait"], home=HOME, quiet=False)
if parsed_ok(r):
    ok("real CLI 0.0.111: the YAML and every field name parsed; it stopped only because no gateway answers")
else:
    warn("the laptop CLI rejected the file — read the error above")
errs, warns = pk.validate(policy)
for e in errs:
    print(f"✕ validate: {e}")
for w in warns:
    warn(f"validate: {w}")
if not errs:
    ok(f"policykit.validate: 0 errors, {len(warns)} warnings")
findings = pk.harden(policy)
table([[sev, where, what] for sev, where, what in findings], ["severity", "where", "finding (course §6.2 lint)"])
note("The one LOW is honest: otel.alto.local has no allowed_ips. The docs block private RFC 1918 ranges only for "
     "wildcard or hostless entries, so an exact host is fine; the course lint is stricter on purpose. Pin "
     "otel.alto.local to its /32 if you know the IP.")

step(2, "the decision matrix — what this policy allows")
MATRIX = [
    {"op": "mcp", "host": "bms.alto.local", "port": 8443, "binary": PY, "method": "initialize"},
    {"op": "mcp", "host": "bms.alto.local", "port": 8443, "binary": PY, "method": "tools/list"},
    {"op": "mcp", "host": "bms.alto.local", "port": 8443, "binary": PY, "tool": "read_point"},
    {"op": "mcp", "host": "bms.alto.local", "port": 8443, "binary": PY, "tool": "get_trend"},
    {"op": "mcp", "host": "bms.alto.local", "port": 8443, "binary": PY, "tool": "write_setpoint"},
    {"op": "mcp", "host": "bms.alto.local", "port": 8443, "binary": PY, "tool": "override_schedule"},
    {"op": "mcp", "host": "bms.alto.local", "port": 8443, "binary": PY, "tool": "reboot_controller"},
    {"op": "http", "host": "otel.alto.local", "port": 4318, "binary": PY, "method": "POST", "path": "/v1/traces"},
    {"op": "http", "host": "otel.alto.local", "port": 4318, "binary": PY, "method": "GET", "path": "/v1/traces"},
    {"op": "http", "host": "inference.local", "port": 443, "binary": PY, "method": "POST",
     "path": "/v1/chat/completions"},
    {"op": "connect", "host": "api.openai.com", "port": 443, "binary": PY},
    {"op": "connect", "host": "169.254.169.254", "port": 80, "binary": CURL},
    {"op": "write", "path": "/sandbox/data/out/report.csv"},
    {"op": "write", "path": "/app/alto_ops/register.py"},
]
GLYPH = {"allow": "✓ allow", "deny": "✕ deny", "inspect_for_inference": "◆ inference"}
rows = []
for a in MATRIX:
    d, why = pk.decide(policy, a)
    rows.append([pk.fmt_action(a), GLYPH[d], why])
table(rows, ["action", "decision", "why (policykit)"])
note("reboot_controller is not named anywhere, and it is still denied: the MCP rules are an allow-list, so "
     "deny_rules are a second line, not the only one.")

step(3, "break it on purpose — twelve weakened copies, one §6.2 line each")


def ep(p, group, i=0):
    return p["network_policies"][group]["endpoints"][i]


def v_audit(p):
    ep(p, "bms_mcp")["enforcement"] = "audit"


def v_l4(p):
    e = ep(p, "bms_mcp")
    for k in ("protocol", "enforcement", "rules", "deny_rules", "mcp", "path"):
        e.pop(k, None)


def v_full(p):
    e = ep(p, "otel_collector")
    e.pop("rules")
    e["access"] = "full"


def v_wild(p):
    ep(p, "otel_collector")["host"] = "*.alto.local"


def v_openai(p):
    p["network_policies"]["openai"] = {"name": "openai", "endpoints": [
        {"host": "api.openai.com", "port": 443, "protocol": "rest", "enforcement": "enforce", "access": "read-write"}],
        "binaries": [{"path": PY}]}


def v_best(p):
    p["landlock"]["compatibility"] = "best_effort"


def v_curl(p):
    p["network_policies"]["bms_mcp"]["binaries"].append({"path": CURL})


def v_usr(p):
    p["filesystem_policy"]["read_write"].append("/usr")


def v_root(p):
    p["process"]["run_as_user"] = "root"


def v_tls(p):
    ep(p, "otel_collector")["tls"] = "skip"


def v_version(p):
    p["Version"] = p.pop("version")


def v_mcpkey(p):
    ep(p, "bms_mcp")["mcp"]["max_bytes"] = 1


VARIANTS = [
    ("bms enforcement: audit", "network: enforce", v_audit,
     {"op": "mcp", "host": "bms.alto.local", "port": 8443, "binary": PY, "tool": "write_setpoint"}),
    ("bms back to L4 (no protocol)", "network: protocol mcp", v_l4,
     {"op": "mcp", "host": "bms.alto.local", "port": 8443, "binary": PY, "tool": "write_setpoint"}),
    ("otel access: full", "network: explicit rules", v_full,
     {"op": "http", "host": "otel.alto.local", "port": 4318, "binary": PY, "method": "POST", "path": "/v1/logs"}),
    ("otel host *.alto.local", "network: no wildcards", v_wild,
     {"op": "http", "host": "nas.alto.local", "port": 4318, "binary": PY, "method": "POST", "path": "/v1/traces"}),
    ("add api.openai.com", "never inference hosts", v_openai,
     {"op": "http", "host": "api.openai.com", "port": 443, "binary": PY, "method": "POST",
      "path": "/v1/chat/completions"}),
    ("landlock best_effort", "fs: hard_requirement", v_best, None),
    ("curl on bms_mcp too", "one binary per endpoint", v_curl,
     {"op": "mcp", "host": "bms.alto.local", "port": 8443, "binary": CURL, "tool": "read_point"}),
    ("read_write += /usr", "fs: minimal read_write", v_usr, {"op": "write", "path": "/usr/lib/python3/x.py"}),
    ("run_as_user: root", "process: non-root", v_root, {"op": "run_as", "user": "root"}),
    ("otel tls: skip", "network: inspection on", v_tls, None),
    ("Version: 1 (--full header)", "(the playbook trap)", v_version, None),
    ("mcp: max_bytes (typo)", "(schema)", v_mcpkey, None),
]
base_findings = {(s, w, f) for s, w, f in findings}
checks, probes = [], []
for name, line, fn, probe in VARIANTS:
    p = copy.deepcopy(policy)
    fn(p)
    path = RUNS / f"prod.{fn.__name__}.yaml"
    path.write_text(pk.dump(p), encoding="utf-8")
    cli = cli_verdict(path)
    try:
        v_errs, _ = pk.validate(p)
    except Exception as e:  # noqa: BLE001 — a renamed key can trip the model; report it as an error
        v_errs = [f"{type(e).__name__}"]
    try:
        new = [f for f in pk.harden(p) if f not in base_findings and f[0] != "OK"]
    except Exception:  # noqa: BLE001
        new = []
    checks.append([name, line, cli[:40], ("✕ " + v_errs[0][:34]) if v_errs else "ok",
                   f"{new[0][0]} · {new[0][2][:34]}" if new else "—"])
    if probe and not v_errs:
        d, why = pk.decide(p, probe)
        probes.append([name, pk.fmt_action(probe),
                       "⚠ allow (audit)" if d == "allow" and "audit" in why else GLYPH[d]])
print()
table(checks, ["weakened copy", "§6.2 line", "real CLI 0.0.111", "policykit.validate", "policykit.harden (new)"])
print()
table(probes, ["weakened copy", "probe (was ✕ deny under prod.yaml)", "now"])
note("Read the columns top to bottom. The real CLI checks YAML and field NAMES (it caught `Version` and the "
     "mcp typo). policykit.validate checks values (root, enum strings). Only the lint sees the quiet "
     "weakenings, the ones that parse fine and still open a hole: audit, L4, access: full, wildcards, provider "
     "hosts, extra binaries.")

step(4, "the §6.2 lines no policy lint can see")
table([
    ["SHA256 TOFU binary pinning", "OpenShell, at first use", "keep binaries in the image, not installed at runtime"],
    ["seccomp, no_new_privs, RLIMIT_CORE=0", "OpenShell supervisor", "nothing to set; check the supervisor logs"],
    ["gateway policy_validation_failure_mode", "gateway config, not the policy",
     "keep fail_closed (the default), not retain_last_valid"],
    ["tier (Restricted / Balanced / Personal)", "NemoClaw onboarding", "Restricted for always-on claws"],
    ["channel tokens + device pairing", "NemoClaw / harness", "pair devices explicitly"],
    ["snapshot before change; recreate on suspicion", "your runbook", "Module 08"],
    ["the prover: what a change newly allows", "OpenShell (TUI approval)", "wait for human approval"],
], ["§6.2 line", "who enforces it", "what you do"])

step(5, "the real mock BMS on this laptop — its tool list vs the policy")
BMS = COMMON / "bms_mcp_server.py"
port = free_port(8443)
note(f"mock BMS on port {port} (documented 8443). The policy talks about bms.alto.local:8443; here the same "
     "server runs on localhost, and the course gate evaluates each call as if it went to bms.alto.local:8443.")


async def bms_session(url: str) -> list[list[str]]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    out = []
    async with streamablehttp_client(url) as (rd, wr, _):
        async with ClientSession(rd, wr) as s:
            await s.initialize()
            tools = [t.name for t in (await s.list_tools()).tools]
            ok(f"tools/list from the REAL mock server: {', '.join(tools)}")
            args = {"read_point": {"point": "PLANT.KW_PER_RT"}, "list_alarms": {}, "get_trend": {"hours": 2},
                    "write_setpoint": {"point": "CH2.CHWST", "value": 6.5}}
            for t in tools:
                d, why = pk.decide(policy, {"op": "mcp", "host": "bms.alto.local", "port": 8443, "binary": PY,
                                            "tool": t})
                if d == "allow":
                    res = await s.call_tool(t, args.get(t, {}))
                    text = " ".join(c.text for c in res.content if hasattr(c, "text"))
                    out.append([t, "✓ forwarded", text[:70]])
                else:
                    out.append([t, "✕ 403 (course gate)", why[:70]])
    return out


try:
    py = NAT_PY if NAT_PY.exists() else Path(sys.executable)
    with background([py, BMS], ready_url=f"http://localhost:{port}/mcp", log=RUNS / "bms_mcp.log",
                    env={"BMS_MCP_PORT": str(port)}, show=f"python week26/common/bms_mcp_server.py  # port {port}"):
        rows = asyncio.run(bms_session(f"http://localhost:{port}/mcp"))
    table(rows, ["tool (from the real server)", "course gate", "real answer / reason"])
    note("write_setpoint never reached the server: the gate refused it before forwarding. That is the shape of "
         "the real thing. On the Spark the OpenShell proxy does it and answers 403 with a JSON body. The gate "
         "here is course code with a course-made message, not OpenShell's.")
except Exception as e:  # noqa: BLE001
    warn(f"mock BMS step skipped: {type(e).__name__}: {str(e)[:160]}")

step(6, f"the Spark — apply it to `{SB}` (a change: needs 🔓 CLAW_APPLY=1)")
EXAMPLE_GET = """version: 1
filesystem_policy:
  include_workdir: true
  ...
network_policies:
  <the entries your sandbox has today>"""
put(PROD, "~/week26/07_hardening/prod.yaml")
change(f"cd ~/week26/07_hardening && openshell policy set {SB} --policy prod.yaml --wait",
       preview=f"openshell policy get {SB}", example=EXAMPLE_GET)
note("policy set changes the network sections live. filesystem_policy, landlock and process are fixed when the "
     "sandbox is created: to get hard_requirement and this read_write list in force, create the sandbox with "
     "`--policy prod.yaml` (Module 08). What policy set does with a changed static section is not stated in the "
     "course sources, so read `openshell policy get --full` afterwards.")
sh(f"openshell policy list {SB}", example="""<revision>  <timestamp>  <source>
3           …            policy set (prod.yaml)
2           …            …""")

step(7, "prove the deny path on the Spark")
print(f"→ in the claw, ask: \"set chiller 2 setpoint to 6.5°C\" — then read the sandbox log (no --tail: it streams)")
sh(f"openshell logs {SB} -n 50 --source sandbox", example="""… deny  bms.alto.local:8443  tools/call write_setpoint  /usr/bin/python3.12  403
… allow bms.alto.local:8443  tools/call read_point      /usr/bin/python3.12""")
note("The line format above is an EXAMPLE shape. The research tutorial says to 'confirm the 403 JSON in "
     "openshell logs and a failed tool span in Phoenix', but no course source prints the exact line. Record your "
     "own with SPARK_RECORD=1.")
result("prod.yaml parses (real CLI), validates (policykit), lints with one LOW, and denies write_setpoint by name "
       "and everything unnamed by default.")
