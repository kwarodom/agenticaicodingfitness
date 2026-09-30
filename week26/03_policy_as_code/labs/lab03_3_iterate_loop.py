#!/usr/bin/env python3
"""Lab 03-3 · The iterate loop (L2.3): deny → observe → allow → verify, and the spec grammars behind it.

Part 1 is offline and real: it feeds a table of `--add-endpoint` and `--add-allow` specs to the REAL OpenShell
CLI 0.0.111 on this laptop (dead gateway) and to policykit's parse_endpoint_spec / parse_rule_spec. The CLI
rejects some specs before it contacts any gateway; the rest "parse" and then fail to connect. Where the docs say
the GATEWAY rejects a spec the CLI lets through (the `::rest` trap), the lab says so.

Part 2 is the loop on your Spark. Reads use sh(); every change goes through clawkit.change() with a
`--dry-run` preview, so nothing changes unless you are LIVE and turned on 🔓 Allow changes (CLAW_APPLY=1).

Run: .venv/bin/python week26/03_policy_as_code/labs/lab03_3_iterate_loop.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE.parents[1]))
import parsekit  # noqa: E402
import policykit as pk  # noqa: E402
from clawkit import banner, change, note, ok, result, sandbox, sh, step, table  # noqa: E402

S = sandbox()

ENDPOINTS = [   # (spec, what to learn from it)
    ("api.github.com:443:read-only:rest:enforce", "the docs' example: L7, read-only, enforced"),
    ("api.github.com:443::rest", "THE TRAP: docs say the gateway rejects it"),
    ("pypi.org:443", "L4 only: host, port, binary"),
    ("timescale.alto.local:5432::tcp", "Part 2 ex. 4: empty access before tcp"),
    ("bms.alto.local:8443:read-only:rest:enforce:allowed-ip=10.20.0.15/32", "an option: pin a /32"),
    ("realtime.example.com:443:read-write:websocket:enforce:websocket-credential-rewrite,allowed-ip=10.0.0.0/8",
     "the CLI's own --help example"),
    ("api.github.com:443:readonly", "typo in the access segment"),
    ("api.github.com:443:read-only:http", "http is not a protocol"),
    ("mcp.example.com:443:read-only:mcp", "mcp is YAML-only in 0.0.111"),
    ("db.internal.example:5432::sql", "sql: the CLI knows it, policykit not"),
    ("api.github.com:443:read-only:rest:block", "enforcement is enforce or audit"),
    ("api.github.com:443:read-only::enforce", "enforcement needs a protocol"),
    ("api.github.com:443:read-write:tcp", "tcp takes no access mode"),
    ("api.github.com:443:read-only:rest:enforce:bogus-opt", "unknown option"),
    ("api.github.com", "no port"),
    ("api.github.com:99999", "port out of range"),
]
RULES = [
    ("api.github.com:443:POST:/repos/*/issues", "the docs' add-allow example"),
    ("api.github.com:443:GET:/repos/**", "the docs' dry-run example"),
    ("api.github.com:443:POST:/admin/**", "the docs' add-deny example"),
    ("realtime.example.com:443:WEBSOCKET_TEXT:/v1/realtime", "a WebSocket frame rule"),
    ("api.github.com:443:GET:**", "bare ** — the CLI allows it"),
    ("api.github.com:443:FETCH:/x", "a made-up method"),
    ("api.github.com:443:POST:repos", "path without /"),
    ("api.github.com:443:POST", "no path"),
    ("api.github.com:x:GET:/a", "port not a number"),
]


def pk_try(fn, spec: str) -> tuple[bool, str]:
    try:
        return True, str(fn(spec))
    except ValueError as e:
        return False, str(e)


banner("Lab 03-3 · the iterate loop", f"spec grammars on this laptop (real CLI + policykit) · the loop on "
       f"`{S}` via change()")

step(1, "--add-endpoint host:port[:access[:protocol[:enforcement[:options]]]] — who rejects what?")
rows, full = [], []
for spec, why in ENDPOINTS:
    v, msg = parsekit.cli(["policy", "update", S, "--add-endpoint", spec, "--dry-run"])
    pv, pmsg = pk_try(pk.parse_endpoint_spec, spec)
    rows.append([spec if len(spec) < 46 else spec[:44] + "…", "✕ rejects" if v is False else
                 ("· parses" if v else "· n/a"), "· parses" if pv else "✕ rejects", why])
    full.append((spec, v, msg, pv, pmsg))
table(rows, ["spec", "CLI 0.0.111", "policykit", "why it is in the table"])
for spec, v, msg, pv, pmsg in full:
    if v is False or not pv or "," in spec:
        print(f"▣ {spec}")
        print(f"  CLI:       {msg if v is False else 'parses (then: Connection refused — no gateway)'}")
        print(f"  policykit: {pmsg if not pv else 'parses → ' + pmsg}")
note("Three honest disagreements. (1) `::rest` parses in the CLI but the research tutorial, citing the OpenShell "
     "sandbox-policies guide, says it is rejected — by the gateway, when it merges. policykit rejects it early. "
     "(2) The CLI knows a `sql` protocol, validates options and refuses tcp + access; policykit does not. "
     "(3) policykit splits options on `:` only, so the CLI's comma list comes back without its options.")

step(2, "--add-allow / --add-deny host:port:METHOD:path_glob")
rows, full = [], []
for spec, why in RULES:
    v, msg = parsekit.cli(["policy", "update", S, "--add-allow", spec, "--dry-run"])
    pv, pmsg = pk_try(pk.parse_rule_spec, spec)
    rows.append([spec, "✕ rejects" if v is False else ("· parses" if v else "· n/a"),
                 "· parses" if pv else "✕ rejects", why])
    full.append((spec, v, msg, pv, pmsg))
table(rows, ["rule spec", "CLI 0.0.111", "policykit", "why it is in the table"])
for spec, v, msg, pv, pmsg in full:
    if v is False or not pv:
        print(f"▣ {spec}")
        print(f"  CLI:       {msg if v is False else 'parses (then: Connection refused — no gateway)'}")
        print(f"  policykit: {pmsg if not pv else 'parses'}")
note("Offline, the CLI checks the shape (4 segments, integer port, a path starting with / or `**`). It does not "
     "check the METHOD — `FETCH` parses. policykit refuses a bare `**`, the CLI accepts it.")

step(3, "flag rules the CLI enforces before it needs a gateway")
rows = []
for args, why in [
    (["--binary", "/usr/bin/gh"], "--binary alone"),
    ([], "no operation at all"),
    (["--add-endpoint", "a.example.com:443", "--add-endpoint", "b.example.com:443", "--rule-name", "x"],
     "--rule-name with two endpoints"),
    (["--remove-endpoint", "pypi.org"], "--remove-endpoint without a port"),
    (["--add-allow", "api.github.com:443:GET:/repos/**", "--dry-run", "--wait"], "--dry-run together with --wait"),
    (["--add-endpoint", "pypi.org:443", "--dry-run"], "a clean --dry-run"),
]:
    v, msg = parsekit.cli(["policy", "update", S, *args])
    rows.append([why, parsekit.glyph(v, msg, 70)])
table(rows, ["openshell policy update …", "what the laptop CLI said"])
note("The last row matters: `--dry-run` is NOT offline. It fetches the live policy from the gateway, merges your "
     "change into it and shows the result without sending it. That is why the loop below can use it as a safe "
     "preview on the Spark. Nothing is sent, so there is nothing to --wait for.")
ok("steps 1–3 captured on this Mac: the real OpenShell 0.0.111 parser, no gateway")

step(4, f"the loop on `{S}` — 1. watch denials (read-only)")
sh(f"openshell logs {S} -n 20 --source sandbox", timeout=60,
   example="… sandbox  deny   dst=api.github.com:443  binary=/usr/bin/gh  reason=no matching network_policies entry …")
note(f"To follow live, run `openshell logs {S} --tail --source sandbox` in the ⌨ terminal (it streams; a lab "
     "never runs it). `openshell term` shows the same decisions in a TUI.")

step(5, "2. additive fixes — each one previewed with --dry-run, applied with --wait")
PREVIEW_EX = "merged policy preview — nothing sent:\n  + {what}"
change(f"openshell policy update {S} --add-endpoint api.github.com:443:read-only:rest:enforce --binary /usr/bin/gh "
       "--wait",
       preview=f"openshell policy update {S} --add-endpoint api.github.com:443:read-only:rest:enforce "
               "--binary /usr/bin/gh --dry-run",
       example=PREVIEW_EX.format(what="api.github.com:443 read-only rest enforce · binaries: /usr/bin/gh"))
change(f"openshell policy update {S} --add-allow 'api.github.com:443:POST:/repos/*/issues' --wait",
       preview=f"openshell policy update {S} --add-allow 'api.github.com:443:POST:/repos/*/issues' --dry-run",
       example=PREVIEW_EX.format(what="rule allow POST /repos/*/issues on api.github.com:443"))
change(f"openshell policy update {S} --add-deny 'api.github.com:443:POST:/admin/**' --wait",
       preview=f"openshell policy update {S} --add-deny 'api.github.com:443:POST:/admin/**' --dry-run",
       example=PREVIEW_EX.format(what="deny_rule POST /admin/** on api.github.com:443"))
change(f"openshell policy update {S} --add-endpoint pypi.org:443 --add-endpoint files.pythonhosted.org:443 "
       "--binary /usr/bin/pip --binary /usr/local/bin/uv --wait",
       preview=f"openshell policy update {S} --add-endpoint pypi.org:443 --add-endpoint files.pythonhosted.org:443 "
               "--binary /usr/bin/pip --binary /usr/local/bin/uv --dry-run",
       example=PREVIEW_EX.format(what="pypi.org:443 + files.pythonhosted.org:443 · binaries: /usr/bin/pip, "
                                      "/usr/local/bin/uv"))

step(6, "3. preview a merge before sending it (read-only)")
sh(f"openshell policy update {S} --add-allow 'api.github.com:443:GET:/repos/**' --dry-run",
   example=PREVIEW_EX.format(what="rule allow GET /repos/** on api.github.com:443"), timeout=60)

step(7, "4. remove — an endpoint, then a named rule")
change(f"openshell policy update {S} --remove-endpoint pypi.org:443 --wait",
       preview=f"openshell policy update {S} --remove-endpoint pypi.org:443 --dry-run",
       example=PREVIEW_EX.format(what="(removed) pypi.org:443"))
change(f"openshell policy update {S} --remove-rule github_repos --wait",
       preview=f"openshell policy update {S} --remove-rule github_repos --dry-run",
       example=PREVIEW_EX.format(what="(removed) rule github_repos"))
note("`github_repos` is the research tutorial's example rule name. Read the real generated names in "
     f"`openshell policy get {S}` first (or set one with --rule-name when you add a single endpoint).")

step(8, "5. full replacement — push lab 03-1's export back (this is also your rollback)")
change(f"openshell policy set {S} --policy ~/week26/current-policy.yaml --wait",
       preview=f"openshell policy list {S}",
       example="  REV  STATUS   CREATED\n  7    loaded   <time>\n  …   (EXAMPLE shape — one revision per change above)")
sh(f"openshell policy list {S}", example="  REV  STATUS   CREATED\n  8    loaded   <time>   ← the file you set\n"
   "  7    loaded   <time>\n  …", timeout=60)

step(9, "the NemoClaw wrapper for presets — it knows the blueprint's baseline")
change(f"nemoclaw {S} policy add github --yes", preview=f"nemoclaw {S} policy add github --dry-run",
       example="preset github → endpoints for github.com / api.github.com · binary-scoped · "
               "nothing applied (--dry-run)")
change(f"nemoclaw {S} policy remove github --yes", preview=f"nemoclaw {S} policy list",
       example="  preset   status\n  github   applied\n  npm      applied")
note("NemoClaw refuses, for example, an `npm` change when the live baseline drifted from the reviewed GET-only "
     "entry. The DGX Spark playbooks spell these verbs `policy-add` / `policy-remove`; check "
     f"`nemoclaw {S} --help` on your unit.")
result("You can grow a policy one endpoint or rule at a time, preview every merge, and roll back with policy set.")
