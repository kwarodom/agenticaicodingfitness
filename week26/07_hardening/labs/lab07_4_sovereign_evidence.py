#!/usr/bin/env python3
"""Lab 07-4 · The evidence pack: "no data left the building last month" (Part 6, exercise 5).

A hotel owner or a regulator will not read your policy YAML. They want evidence. This lab turns the research
tutorial's answer to Part 6 exercise 5 into a runnable pack:

  1. the Alto Copilot tier table (§6.6): which tier makes the claim possible at all;
  2. every evidence item → the read-only command that produces it. The laptop OpenShell CLI (0.0.111) checks
     that each command and flag parses; on the Spark the commands run read-only through sh() (EXAMPLE in DRY);
  3. an offline check of the policy itself: every host in policies/prod.yaml sorted into "inside the building"
     and "outside";
  4. an offline log checker that counts `allow` decisions to hosts outside the building. It runs on two
     COURSE-MADE EXAMPLE logs (a clean month and a leaky one), and on your real log when a Spark is connected;
  5. a manifest (.runs/evidence/manifest.json) that says, for each item, whether it is live, example or laptop.

The log line format used here is an EXAMPLE shape. No course source prints a real `openshell logs` line, so
the checker tells you how many lines it could NOT parse. Zero parsed lines is never "zero leaks".

Run: .venv/bin/python week26/07_hardening/labs/lab07_4_sovereign_evidence.py
"""
import ipaddress
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import policykit as pk  # noqa: E402
from clawkit import (banner, mode, note, ok, openshell_offline, parsed_ok, result, sandbox, sh, step,  # noqa: E402
                     table, warn)

MOD = Path(__file__).resolve().parents[1]
RUNS = MOD / ".runs"
EVID = RUNS / "evidence"
EVID.mkdir(parents=True, exist_ok=True)
SB = sandbox("alto-ops")

# What "the building" means for THIS property. Write it down explicitly; never infer it from ".local".
INSIDE_SUFFIXES = (".alto.local",)
INSIDE_CIDRS = [ipaddress.ip_network("10.20.0.0/16")]
INSIDE_HOSTS = {"inference.local"}          # intercepted by the supervisor, routed to the on-prem provider

REF_LANDLOCK = "Look for `OpenShell Sandbox Supervisor success` and `Applying Landlock filesystem sandbox`."

banner("Lab 07-4 · the sovereign evidence pack", "laptop: CLI parse + offline checkers · Spark: read-only")


def inside(h: str) -> bool:
    if h in INSIDE_HOSTS or h.endswith(INSIDE_SUFFIXES):
        return True
    try:
        return any(ipaddress.ip_address(h) in n for n in INSIDE_CIDRS)
    except ValueError:
        return False


step(1, "which tier can make the claim at all (research tutorial §6.6)")
table([
    ["Cloud", "NAT behind Copilot's gateway; optional sandbox per tenant", "NVIDIA endpoints / Model Router",
     "no — inference leaves by design"],
    ["Sovereign (on-prem)", "NemoClaw on Spark/Station; NAT inside OpenShell", "local vLLM/Ollama via inference.local",
     "yes — the only inference route is on-prem"],
    ["Edge", "Restricted claw next to the BMS, read-only tools", "local NVFP4 model", "yes — plus writes via approvals"],
], ["Alto Copilot tier", "claw pattern", "inference", "'no data left'?"])
note("The owner's three talking points (§6.6): the only inference route is on-prem; every network decision is "
     "logged; write authority is structurally absent from the agent. Each item below backs one of them.")

step(2, "evidence item → command (the laptop CLI checks that each one parses)")
HOME = RUNS / "openshell-home"
EVIDENCE = [
    ("policy in force: no outside hosts", ["policy", "get", SB, "--full"], "network_policies lists only inside hosts"),
    ("every revision last month", ["policy", "list", SB], "each revision reviewed; none adds an outside host"),
    ("the only inference route", ["inference", "get"], "provider = your on-prem vLLM / Ollama"),
    ("a month of network decisions", ["logs", SB, "-n", "5000", "--since", "720h", "--source", "sandbox"],
     "0 allow decisions to outside hosts"),
]
rows = []
for what, args, good in EVIDENCE:
    r = openshell_offline(args, home=HOME)
    rows.append([what, "openshell " + " ".join(args), "✓ parses" if parsed_ok(r) else "✕ " + r.out.strip()[:40], good])
rows += [
    ["Landlock applied, no findings", f"docker logs <openshell-{SB} container> --tail 50", "(docker, not parsed)",
     "'Applying Landlock filesystem sandbox', no DetectionFinding"],
    ["traces stayed on-prem", "docker ps --filter name=otel", "(docker, not parsed)", "the OTel collector runs on the Spark"],
    ["the tier", f"nemoclaw {SB} policy list", "(nemoclaw, not on laptop)", "Restricted + custom presets only"],
]
table(rows, ["evidence", "command", "laptop CLI 0.0.111", "what good looks like"])
note("`logs -n 5000 --since 720h` parses on 0.0.111. Whether your gateway still HOLDS 30 days of logs is not in "
     "the course sources. Export daily (a cron job) and hand over the exports, not one call at month end.")

step(3, "the policy itself: every host, inside or outside the building (offline, prod.yaml)")
policy = pk.load(MOD / "policies" / "prod.yaml")
rows, outside_hosts = [], []
for gname, g in policy["network_policies"].items():
    for e in g["endpoints"]:
        h = str(e["host"])
        where = "inside" if inside(h) else "OUTSIDE"
        if where == "OUTSIDE":
            outside_hosts.append(h)
        rows.append([gname, f"{h}:{e['port']}", e.get("protocol", "L4"), e.get("enforcement", "audit"), where])
table(rows, ["group", "endpoint", "protocol", "enforcement", "building"])
(ok if not outside_hosts else warn)(f"{len(outside_hosts)} outside hosts in prod.yaml"
                                    + (f": {', '.join(outside_hosts)}" if outside_hosts else ""))
note("inference.local is not in the file: the supervisor intercepts it and the gateway routes it. That is why "
     "`openshell inference get` is its own evidence item.")

step(4, "the log checker — COURSE-MADE EXAMPLE logs (illustrative shape, not a real openshell line)")
EXAMPLE_CLEAN = """2026-09-02T08:14:03Z alto-ops inspect_for_inference inference.local:443 /usr/bin/python3.12 POST /v1/chat/completions
2026-09-02T08:14:05Z alto-ops allow bms.alto.local:8443 /usr/bin/python3.12 tools/call read_point
2026-09-02T08:14:06Z alto-ops deny bms.alto.local:8443 /usr/bin/python3.12 tools/call write_setpoint
2026-09-02T08:14:09Z alto-ops allow otel.alto.local:4318 /usr/bin/python3.12 POST /v1/traces
2026-09-03T02:00:00Z alto-ops deny api.openai.com:443 /usr/bin/curl CONNECT
2026-09-03T02:00:01Z alto-ops deny 169.254.169.254:80 /usr/bin/curl CONNECT
2026-09-14T11:30:00Z alto-ops allow 10.20.0.15:8443 /usr/bin/python3.12 tools/call get_trend"""
EXAMPLE_LEAKY = EXAMPLE_CLEAN + """
2026-09-21T19:02:11Z alto-ops allow api.openai.com:443 /usr/bin/python3.12 POST /v1/chat/completions audit-violation
2026-09-21T19:02:12Z alto-ops allow 203.0.113.7:443 /usr/bin/python3.12 CONNECT
supervisor: policy reloaded (revision 7)"""
LINE = re.compile(r"\b(allow|deny|inspect_for_inference)\s+(\[?[A-Za-z0-9_.:\-]+?\]?):(\d+)\b")


def check_log(text: str) -> dict:
    counts, leaks, unparsed = {"allow": 0, "deny": 0, "inspect_for_inference": 0}, [], 0
    for ln in text.splitlines():
        if not ln.strip():
            continue
        m = LINE.search(ln)
        if not m:
            unparsed += 1
            continue
        decision, h = m.group(1), m.group(2).strip("[]")
        counts[decision] += 1
        if decision == "allow" and not inside(h):
            leaks.append(ln.strip())
    return {"counts": counts, "leaks": leaks, "unparsed": unparsed}


def report(name: str, rep: dict) -> None:
    c = rep["counts"]
    print(f"◆ {name}: {c['allow']} allow · {c['deny']} deny · {c['inspect_for_inference']} inference · "
          f"{rep['unparsed']} line(s) not parsed")
    for ln in rep["leaks"]:
        print(f"✕ allow to an OUTSIDE host: {ln}")
    if not sum(c.values()):
        warn("0 lines matched the course pattern — this proves nothing. Adapt LINE to your real log format.")
    elif not rep["leaks"]:
        ok(f"{name}: 0 allow decisions to outside hosts")


reports = {"example clean month": check_log(EXAMPLE_CLEAN), "example leaky month": check_log(EXAMPLE_LEAKY)}
for name, rep in reports.items():
    report(name, rep)
note("The leaky month has an `allow` to api.openai.com marked audit-violation. That is `enforcement: audit` "
     "doing what the docs say it does: log the violation, forward the traffic. One audit entry is enough to break "
     "the owner's claim. That is why §6.2 says enforce.")

step(5, "the Spark — produce the real evidence (read-only)")
manifest = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "sandbox": SB, "mode": mode(), "items": []}
EX = {
    "policy get": "version: 1\n…\nnetwork_policies:\n  bms_mcp: …\n  otel_collector: …",
    "policy list": "<revision>  <timestamp>  <source>\n…",
    "inference get": "provider: <your on-prem provider>  model: <model handle>",
    "logs": EXAMPLE_CLEAN,
}
live_log = ""
for (what, args, _), key in zip(EVIDENCE, ("policy get", "policy list", "inference get", "logs")):
    r = sh("openshell " + " ".join(args), example=EX[key], quiet=key == "logs", timeout=120)
    if key == "logs" and r.source == "example":
        print("◈ EXAMPLE — the clean-month log from step 4 (not your machine)")
    if r.live:
        (EVID / f"{key.replace(' ', '_')}.txt").write_text(r.out, encoding="utf-8")
        if key == "logs":
            live_log = r.out
    manifest["items"].append({"evidence": what, "command": "openshell " + " ".join(args), "source": r.source,
                              "exit": r.code, "saved": r.live})
r = sh(f"docker logs $(docker ps --filter name=openshell-{SB} --format '{{{{.Names}}}}') --tail 50 2>&1 "
       "| grep -E 'Landlock|Supervisor|DetectionFinding'", reference=REF_LANDLOCK)
manifest["items"].append({"evidence": "Landlock applied", "command": "docker logs … | grep Landlock",
                          "source": r.source, "exit": r.code, "saved": r.live})
if live_log:
    report("YOUR sandbox log (live)", check_log(live_log))
else:
    note("DRY: nothing ran on a Spark, so there is no real log to check. The pack below is a template.")

step(6, "the manifest — what you would hand over, and where each item came from")
manifest["policy_outside_hosts"] = outside_hosts
manifest["example_checks"] = {k: {"allow_outside": len(v["leaks"]), "unparsed": v["unparsed"]}
                              for k, v in reports.items()}
out = EVID / "manifest.json"
out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
table([[i["evidence"], i["source"], "saved" if i["saved"] else "—"] for i in manifest["items"]],
      ["evidence", "source", "file"])
ok(f"wrote {out.relative_to(MOD)}")
if mode() == "dry":
    warn("every Spark item above is EXAMPLE/REFERENCE — this pack proves nothing until you run it LIVE")
result("Evidence = policy (no outside hosts) + inference route (on-prem) + a month of logs (0 allow outside) + "
       "Landlock applied + the tier. Each item names its source.")
