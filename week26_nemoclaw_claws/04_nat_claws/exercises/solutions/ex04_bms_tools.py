#!/usr/bin/env python3
"""Exercise 04 · BMS tools, two layers — reference solution.

What the agent is OFFERED (NAT) and what the sandbox ALLOWS (OpenShell).

Fill in the three TODOs, save, then run:
    .venv/bin/python week26/04_nat_claws/exercises/ex04_bms_tools.py

The checker is offline: it parses your YAML, asks NAT 1.9's own MCPClientConfig model whether it accepts your
block (no server needed), runs the course's policykit teaching model on MCP actions, and asks the laptop OpenShell
0.0.111 parser to read your policy (no gateway needed). Stuck? Compare with exercises/solutions/.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import policykit as pk  # noqa: E402
from clawkit import NAT_PY, OPENSHELL, banner, check, openshell_offline, parsed_ok  # noqa: E402

BMS_TOOLS = ("read_point", "list_alarms", "get_trend", "write_setpoint")   # what the mock BMS lists (lab 04-4)

# ── TODO 1 ── a NAT `function_groups` block named bms_tools that consumes the BMS MCP server at
#   http://bms.alto.local:8443/mcp over streamable-http and offers the agent ONLY read_point and list_alarms.
#   write_setpoint must never reach the agent. Add a tool_overrides description for read_point that says what
#   RT means (Lab 04-2's lesson). Remember: NAT 1.9 refuses include + exclude in the same group.
MCP_CLIENT_YAML = """
function_groups:
  bms_tools:
    _type: mcp_client
    server:
      transport: streamable-http
      url: "http://bms.alto.local:8443/mcp"
    include: [read_point, list_alarms]
    tool_call_timeout: 60
    reconnect_enabled: true
    reconnect_max_attempts: 3
    tool_overrides:
      read_point:
        description: "Read the latest value of one BMS point: PLANT.KW, PLANT.RT (refrigeration tons of cooling) or PLANT.KW_PER_RT."
"""

# ── TODO 2 ── the exact tool name the LLM sees for read_point once the group above is loaded.
TOOL_NAME = "bms_tools__read_point"

# ── TODO 3 ── the research tutorial's Part 3 exercise 4: a network_policies entry `bms_mcp` that lets the
#   sandboxed NAT process (/usr/bin/python3.12) reach bms.alto.local:8443 path /mcp with protocol mcp,
#   enforcement enforce, the MCP handshake (initialize, notifications/initialized, tools/list), tools/call ONLY
#   on read_point and list_alarms, and an explicit deny_rules entry for write_setpoint.
POLICY_YAML = """
version: 1
filesystem_policy:
  include_workdir: true
  read_only: [/usr, /lib, /etc, /app]
  read_write: [/tmp]
landlock:
  compatibility: best_effort
process:
  run_as_user: "1500"
  run_as_group: "1500"
network_policies:
  bms_mcp:
    name: bms_mcp
    endpoints:
      - host: bms.alto.local
        port: 8443
        path: /mcp
        protocol: mcp
        enforcement: enforce
        mcp: { max_body_bytes: 131072 }
        rules:
          - allow: { method: initialize }
          - allow: { method: notifications/initialized }
          - allow: { method: tools/list }
          - allow: { method: tools/call, tool: { any: [read_point, list_alarms] } }
        deny_rules:
          - { method: tools/call, tool: write_setpoint }
    binaries:
      - { path: /usr/bin/python3.12 }
"""


# ─────────────────────────── checker — no need to edit below ────────────────
NAT_CHECK = """
import json, sys, warnings
warnings.filterwarnings("ignore")
from nat.plugins.mcp.client.client_config import MCPClientConfig
blk = json.loads(sys.argv[1])
blk.pop("_type", None)
try:
    c = MCPClientConfig.model_validate(blk)
    print("OK", c.server.transport, c.include, c.exclude)
except Exception as e:
    print("ERR", str(e).splitlines()[0:3])
"""


def load(text: str, what: str):
    try:
        return yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        print(f"✕ {what}: not valid YAML — {str(e).splitlines()[0]}")
        return None


def main() -> None:
    banner("Exercise 04 · BMS tools", "offline checker · NAT's own config model + policykit + the OpenShell parser",
           status=False)
    good = True

    # TODO 1 ────────────────────────────────────────────────────────────────────
    doc = load(MCP_CLIENT_YAML, "TODO 1") or {}
    groups = doc.get("function_groups") or {}
    g = groups.get("bms_tools") if isinstance(groups, dict) else None
    g = g if isinstance(g, dict) else {}
    server = g.get("server") or {}
    inc, exc = set(g.get("include") or []), set(g.get("exclude") or [])
    offered = inc if inc else (set(BMS_TOOLS) - exc)
    shape = (g.get("_type") == "mcp_client" and server.get("transport") == "streamable-http"
             and str(server.get("url", "")).rstrip("/").endswith("bms.alto.local:8443/mcp"))
    good &= check(shape, "TODO 1: bms_tools is an mcp_client on streamable-http http://bms.alto.local:8443/mcp",
                  "TODO 1: need function_groups → bms_tools → _type: mcp_client, server: {transport: streamable-http, "
                  "url: http://bms.alto.local:8443/mcp}")
    good &= check(bool(g) and not (inc and exc), "TODO 1: include OR exclude, not both",
                  "TODO 1: NAT 1.9 refuses include and exclude together (lab 04-4 step B3)" if g else
                  "TODO 1: no bms_tools group yet — when you add one, use include OR exclude, not both")
    good &= check(bool(g) and offered == {"read_point", "list_alarms"},
                  "TODO 1: the agent is offered exactly read_point + list_alarms (write_setpoint never reaches it)",
                  f"TODO 1: the agent would be offered {sorted(offered) if g else '—'} — want read_point, list_alarms")
    ov = (g.get("tool_overrides") or {}).get("read_point") or {}
    good &= check("refrigeration ton" in str(ov.get("description", "")).lower(),
                  "TODO 1: read_point's tool_overrides description says RT = refrigeration tons",
                  "TODO 1: add tool_overrides → read_point → description mentioning 'refrigeration tons'")
    if g and NAT_PY.exists():
        p = subprocess.run([str(NAT_PY), "-c", NAT_CHECK, json.dumps(g)], capture_output=True, text=True, timeout=120,
                           env={"PYTHONWARNINGS": "ignore", "PATH": "/usr/bin:/bin"})
        line = (p.stdout.strip().splitlines() or ["ERR no output: " + p.stderr.strip()[-200:]])[-1]
        good &= check(line.startswith("OK"), f"TODO 1: NAT 1.9's MCPClientConfig accepts it ({line[3:]})",
                      f"TODO 1: NAT 1.9's MCPClientConfig rejects it: {line[4:]}")

    # TODO 2 ────────────────────────────────────────────────────────────────────
    good &= check(TOOL_NAME == "bms_tools__read_point", "TODO 2: the LLM sees bms_tools__read_point (<group>__<tool>)",
                  f"TODO 2: {TOOL_NAME!r} — how does NAT 1.9 join a group name and a tool name? (lab 04-4 step B2)")

    # TODO 3 ────────────────────────────────────────────────────────────────────
    pol = load(POLICY_YAML, "TODO 3") or {}
    errs, _ = pk.validate(pol) if pol else (["empty"], [])
    ep = (((pol.get("network_policies") or {}).get("bms_mcp") or {}).get("endpoints") or [{}])[0]
    good &= check(not errs and bool(ep.get("host")), "TODO 3: policykit validates the policy and bms_mcp has an endpoint",
                  f"TODO 3: {errs[0] if errs else 'add network_policies → bms_mcp → endpoints + binaries'}")
    good &= check(ep.get("protocol") == "mcp" and ep.get("enforcement") == "enforce",
                  "TODO 3: protocol mcp + enforcement enforce (a violation is blocked, not just logged)",
                  f"TODO 3: protocol={ep.get('protocol')!r} enforcement={ep.get('enforcement')!r} — want mcp + enforce")
    PY = "/usr/bin/python3.12"
    want = [("tools/call", "read_point", PY, "allow"), ("tools/call", "list_alarms", PY, "allow"),
            ("tools/call", "write_setpoint", PY, "deny"), ("tools/call", "get_trend", PY, "deny"),
            ("tools/list", "", PY, "allow"), ("initialize", "", PY, "allow"),
            ("tools/call", "read_point", "/usr/bin/curl", "deny")]
    bad = []
    for method, tool, binary, expect in want:
        a = {"op": "mcp", "host": "bms.alto.local", "port": 8443, "binary": binary, "method": method, "tool": tool}
        got, why = pk.decide(pol, a) if pol else ("deny", "no policy")
        if got != expect:
            bad.append(f"{method} {tool or ''} from {binary.rsplit('/', 1)[-1]} → {got} (want {expect}): {why}")
    good &= check(not bad, "TODO 3: read_point + list_alarms allowed · write_setpoint + get_trend denied · curl denied "
                  "· handshake allowed (policykit)", "TODO 3: " + ("; ".join(bad[:2]) if bad else ""))
    denies = [d for d in (ep.get("deny_rules") or []) if (d or {}).get("tool") == "write_setpoint"]
    good &= check(bool(denies), "TODO 3: an explicit deny_rules entry names write_setpoint",
                  "TODO 3: add deny_rules: [{method: tools/call, tool: write_setpoint}]")
    if pol and OPENSHELL.exists():
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(POLICY_YAML)
        r = openshell_offline(["policy", "set", "alto-ops", "--policy", f.name, "--wait"], quiet=True)
        Path(f.name).unlink(missing_ok=True)
        msg = " ".join(r.out.split())[-160:]
        good &= check(parsed_ok(r), "TODO 3: the real OpenShell 0.0.111 parser reads your policy (then finds no gateway)",
                      f"TODO 3: OpenShell's parser rejects it: {msg}")

    if not good:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)
    print("\n═ Done. Two layers: NAT decides what the agent is OFFERED; OpenShell decides what the process can DO.\n"
          "  Only the second one survives a prompt injection that rewrites the agent's config.")


if __name__ == "__main__":
    main()
