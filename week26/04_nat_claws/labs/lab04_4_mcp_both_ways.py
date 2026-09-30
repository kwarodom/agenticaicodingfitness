#!/usr/bin/env python3
"""Lab 04-4 · MCP both ways: NAT as an MCP server, and NAT as an MCP client of a (mock) BMS.

Research tutorial L3.6. Part A publishes Alto Ops tools with `nat mcp serve` (free_port(9901)) — first
everything, then `--tool_names chiller_kpi` only — and calls them with `nat mcp client` (no LLM). Part B starts
the course's mock BMS MCP server (week26/common/bms_mcp_server.py, free_port(8443)), shows which of its tools a
`function_groups: … _type: mcp_client` block with `include: [read_point, list_alarms]` hands to the agent (no
LLM), proves that `include` + `exclude` together is refused by `nat validate`, then asks the agent one question.
All servers are stopped at the end.

LLM budget: 1 agent question ≈ 2 laptop LLM calls.
Run: .venv/bin/python week26/04_nat_claws/labs/lab04_4_mcp_both_ways.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clawkit import NAT, NAT_PY, background, banner, free_port, laptop_models, note, ok, result, step, table, warn  # noqa: E402
from natkit import (BMS_SERVER, CONFIGS, LAPTOP_YML, RUNS, TOOLS, clean, error_line, get_json, nat, rel,  # noqa: E402
                    workflow_result)

banner("Lab 04-4 · MCP both ways", "nat mcp serve + nat mcp client · NAT consuming the mock BMS · all stopped at the end",
       status=False)
QUIET = {"PYTHONWARNINGS": "ignore"}
DIRECT = TOOLS / "nat_direct.py"
BMS_YML = CONFIGS / "mcp_client_bms.yml"


def tool_list(url: str) -> list[str]:
    r, _ = nat(["mcp", "client", "tool", "list", "--url", url], echo=False)
    names = [ln.strip() for ln in clean(r.out).splitlines()
             if re.fullmatch(r"[a-z_]+", ln.strip() or "-")]
    for n in names:
        print(f"  {n}")
    return names


try:
    # ── Part A ────────────────────────────────────────────────────────────────
    mport = free_port(9901)
    murl = f"http://localhost:{mport}/mcp"
    step("A1", f"nat mcp serve — publish the whole workflow's tools on :{mport}" +
         ("" if mport == 9901 else " (9901 was busy)"))
    with background([NAT, "mcp", "serve", "--config_file", LAPTOP_YML, "--name", "Alto Ops MCP", "--host", "localhost",
                     "--port", str(mport)], ready_url=f"http://localhost:{mport}/debug/tools/list",
                    log=RUNS / "lab04_4_mcp_all.log", env=QUIET,
                    show=f"week26/.venv-nat/bin/nat mcp serve --config_file {rel(LAPTOP_YML)} --name \"Alto Ops MCP\" "
                         f"--port {mport}"):
        d = get_json(f"http://localhost:{mport}/debug/tools/list")
        print(f"→ GET /debug/tools/list → server_name={d.get('server_name')!r} · count={d.get('count')}")
        table([[t["name"], t.get("is_workflow"), t.get("description", "")[:40]] for t in d.get("tools", [])],
              ["tool", "is_workflow", "description (as listed)"])
        warn("without --tool_names the WHOLE agent is published too (`tool_calling_agent`) — any MCP client could "
             "run your LLM loop. Publish only what a caller needs.")

    step("A2", "nat mcp serve --tool_names chiller_kpi — publish one tool only")
    with background([NAT, "mcp", "serve", "--config_file", LAPTOP_YML, "--name", "Alto Ops MCP", "--host", "localhost",
                     "--port", str(mport), "--tool_names", "chiller_kpi"],
                    ready_url=f"http://localhost:{mport}/debug/tools/list", log=RUNS / "lab04_4_mcp_one.log", env=QUIET,
                    show=f"week26/.venv-nat/bin/nat mcp serve --config_file {rel(LAPTOP_YML)} --port {mport} "
                         "--tool_names chiller_kpi"):
        listed = tool_list(murl)
        r, _ = nat(["mcp", "client", "tool", "call", "chiller_kpi", "--url", murl, "--json-args", '{"hours": 6}'],
                   echo=False,
                   show=f"week26/.venv-nat/bin/nat mcp client tool call chiller_kpi --url {murl} --json-args '{{\"hours\": 6}}'")
        out = next((ln for ln in clean(r.out).splitlines() if ln.startswith("window=")), error_line(r.out))
        print(f"  {out}")
        if listed == ["chiller_kpi"] and "kw_per_rt=0.901" in out:
            ok("one tool listed, one tool called over streamable-http /mcp — no LLM involved")

    # ── Part B ────────────────────────────────────────────────────────────────
    bport = free_port(8443)
    burl = f"http://localhost:{bport}/mcp"
    step("B1", f"start the mock BMS MCP server on :{bport} (synthetic data; write_setpoint never writes)")
    with background([NAT_PY, BMS_SERVER], ready_url=burl, log=RUNS / "lab04_4_bms.log",
                    env={**QUIET, "BMS_MCP_PORT": str(bport)},
                    show=f"BMS_MCP_PORT={bport} week26/.venv-nat/bin/python {rel(BMS_SERVER)}"):
        server_tools = tool_list(burl)

        step("B2", f"what the mcp_client block hands the agent — {rel(BMS_YML)} (no LLM)")
        blk = BMS_YML.read_text(encoding="utf-8")
        for ln in blk.split("function_groups:", 1)[1].split("llms:", 1)[0].rstrip().splitlines():
            print(f"│   {ln}")
        env_b = {**QUIET, "BMS_MCP_URL": burl}
        r, _ = nat(["group", BMS_YML, "bms_tools"], prefix=[NAT_PY, DIRECT], env=env_b,
                   show=f"BMS_MCP_URL={burl} week26/.venv-nat/bin/python {rel(DIRECT)} group {rel(BMS_YML)} bms_tools")
        sees = re.findall(r"AGENT SEES\s+(\S+)", r.out)
        hidden = re.findall(r"hidden\s+(\S+)", r.out)
        if sorted(sees) == ["bms_tools__list_alarms", "bms_tools__read_point"]:
            ok(f"the group holds {len(sees) + len(hidden)} tools; the agent gets 2, named <group>__<tool>")

        step("B3", "include + exclude in one group? NAT 1.9 refuses it at validate time")
        bad = RUNS / "mcp_client_include_and_exclude.yml"
        bad.write_text(blk.replace("    include: [read_point, list_alarms]",
                                   "    include: [read_point, list_alarms]\n    exclude: [write_setpoint]"),
                       encoding="utf-8")
        r, _ = nat(["validate", "--config_file", bad], env=env_b)
        if r.code != 0:
            ok("pick one: include (an allow-list — safer) or exclude (a deny-list)")

        if not any(m.startswith("nemotron-3-nano") for m in laptop_models()):
            warn("no nemotron-3-nano on this laptop's Ollama — skipping the agent question.")
        else:
            q = "Are there any active plant alarms right now, and what is the latest kW/RT reading?"
            step("B4", "the agent uses the BMS tools (LAPTOP STAND-IN)")
            r, secs = nat(["run", "--config_file", BMS_YML, "--input", q], env=env_b, log="lab04_4_agent.log")
            called = sorted(set(re.findall(r"bms_tools__\w+|chiller_kpi", " ".join(
                ln for ln in clean(r.out).splitlines() if "Calling tools" in ln))))
            ans = workflow_result(r.out) or error_line(r.out)
            print("· ANSWER  " + ans[:700].replace("\n", "\n          "))
            print(f"◆ LAPTOP STAND-IN · nemotron-3-nano · {secs:.1f}s · tools called: {', '.join(called) or 'none'}")
            if "bms_tools__write_setpoint" not in called:
                ok("write_setpoint was never offered, so it could not be called — but this is the AGENT's config. "
                   "Module 03/07's OpenShell policy (protocol: mcp) is what holds if the agent is tricked.")
            note("Check the answer against the tool responses above: any 'since …' or 'minutes ago' the model adds on "
                 "its own is not from the BMS.")
        (RUNS / "lab04_4_summary.json").write_text(json.dumps(
            {"server_tools": server_tools, "agent_sees": sees, "hidden": hidden}, indent=1), encoding="utf-8")
except RuntimeError as e:
    warn(f"a server did not start: {str(e)[:500]}")
    sys.exit(1)

result("MCP both ways: `nat mcp serve --tool_names …` publishes; `function_groups: {_type: mcp_client, include: …}` consumes.")
