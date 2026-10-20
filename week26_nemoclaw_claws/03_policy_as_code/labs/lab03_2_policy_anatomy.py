#!/usr/bin/env python3
"""Lab 03-2 · Anatomy of a policy file (L2.2): the annotated schema, every rule form, and fourteen ways to break it.

Offline, runs anywhere. Two checkers read the same files and you compare their answers:

  • policykit.validate() — the course's TEACHING MODEL of the rules the docs describe (semantics included);
  • the REAL OpenShell CLI 0.0.111 on this laptop — `openshell policy set <name> --policy <file>` parses the YAML
    before it contacts a gateway, so against a dead gateway it answers "parse error" or "parsed".

Neither of them is the gateway on your Spark, which checks the semantic rules when you push. Where the two
disagree, the lab says so and says which source backs which answer.

Run: .venv/bin/python week26/03_policy_as_code/labs/lab03_2_policy_anatomy.py
"""
import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE.parents[1]))
import parsekit  # noqa: E402
import policykit as pk  # noqa: E402
from clawkit import banner, note, ok, result, step, table, warn  # noqa: E402

MOD = HERE.parents[1]
RUNS = MOD / ".runs" / "variants"
ANATOMY = MOD / "policies" / "anatomy.yaml"
FORMS = MOD / "policies" / "rule_forms.yaml"


def both(policy_or_path, path: Path) -> tuple[str, str]:
    p = pk.load(policy_or_path) if isinstance(policy_or_path, (str, Path)) else policy_or_path
    errs, warns = pk.validate(p)
    pkv = "✓ valid" if not errs and not warns else (f"✕ {errs[0]}" if errs else f"⚠ {warns[0]}")
    v, msg = parsekit.cli_file(path)
    return pkv, parsekit.glyph(v, msg)


banner("Lab 03-2 · anatomy of a policy file", "offline · policykit (teaching model) + the real OpenShell CLI "
       "parser · no gateway", status=False)

step(1, "the annotated schema policy — static half, dynamic half")
anat = pk.load(ANATOMY)
table([
    ["version", "1", "must be lowercase `version: 1`"],
    ["filesystem_policy", f"ro {anat['filesystem_policy']['read_only']} · rw {anat['filesystem_policy']['read_write']}",
     "STATIC — Landlock, locked at creation"],
    ["landlock", anat["landlock"]["compatibility"], "STATIC — best_effort or hard_requirement"],
    ["process", "(commented out here)", "STATIC — run_as_user / run_as_group, never root"],
    ["network_policies", ", ".join(anat["network_policies"]), "DYNAMIC — hot-reloadable"],
    ["network_middlewares", ", ".join(anat["network_middlewares"]), "DYNAMIC — e.g. openshell/regex redaction"],
], ["top-level key", "in policies/anatomy.yaml", "what it is"])
pkv, osv = both(ANATOMY, ANATOMY)
table([["policies/anatomy.yaml", pkv, osv]], ["file", "policykit.validate", "openshell CLI parser"])

step(2, "every rule form in one policy — REST, WebSocket, GraphQL, MCP, TCP")
forms = pk.load(FORMS)
rows = []
for gname, g in forms["network_policies"].items():
    e = g["endpoints"][0]
    rows.append([gname, e.get("protocol", "(L4)"), f"{e['host']}:{e['port']}", len(e.get("rules") or []),
                 len(e.get("deny_rules") or []), g["binaries"][0]["path"]])
table(rows, ["group", "protocol", "endpoint", "allow", "deny", "binary"])
pkv, osv = both(FORMS, FORMS)
table([["policies/rule_forms.yaml", pkv, osv]], ["file", "policykit.validate", "openshell CLI parser"])
note("REST allow rules WRAP their matchers (`- allow: {method, path}`); deny_rules LIST matchers directly "
     "(`- {method, path}`). The real parser knows the matcher fields: method, path, command, query, "
     "operation_type, operation_name, fields, tool, params (captured from its error message in step 4).")

step(3, "what would these rules allow? (policykit.decide — REST, MCP and TCP only)")
GH, NODE, PY, PSQL, CURL = "/usr/bin/gh", "/usr/bin/node", "/usr/bin/python3", "/usr/bin/psql", "/usr/bin/curl"
CASES = [
    ("REST", {"op": "http", "host": "api.github.com", "port": 443, "binary": GH, "method": "GET",
              "path": "/repos/nvidia/nemoclaw"}),
    ("REST", {"op": "http", "host": "api.github.com", "port": 443, "binary": GH, "method": "GET",
              "path": "/repos/nvidia/nemoclaw/rulesets"}),
    ("REST", {"op": "http", "host": "api.github.com", "port": 443, "binary": GH, "method": "POST",
              "path": "/repos/nvidia/nemoclaw/issues"}),
    ("REST", {"op": "http", "host": "api.github.com", "port": 443, "binary": CURL, "method": "GET",
              "path": "/repos/nvidia/nemoclaw"}),
    ("MCP", {"op": "mcp", "host": "mcp.example.com", "port": 443, "binary": PY, "method": "initialize"}),
    ("MCP", {"op": "mcp", "host": "mcp.example.com", "port": 443, "binary": PY, "tool": "search_web"}),
    ("MCP", {"op": "mcp", "host": "mcp.example.com", "port": 443, "binary": PY, "tool": "send_email"}),
    ("MCP", {"op": "mcp", "host": "mcp.example.com", "port": 443, "binary": PY, "tool": "delete_repo"}),
    ("TCP", {"op": "connect", "host": "db.internal.example", "port": 5432, "binary": PSQL}),
    ("TCP", {"op": "connect", "host": "db.internal.example", "port": 5432, "binary": PY}),
]
rows = []
for kind, a in CASES:
    d, why = pk.decide(forms, a)
    rows.append([kind, pk.fmt_action(a), "✓ allow" if d == "allow" else "✕ deny", why])
table(rows, ["form", "action", "decision", "why (policykit)"])
note("policykit does not model WebSocket frames or GraphQL operations — for those groups trust only the parser "
     "and, on the Spark, `openshell logs <s> --source sandbox`. It also matches paths with Python's fnmatch, "
     "where `*` can cross a `/`; OpenShell writes multi-segment globs as `**`. Keep `*` for one segment.")

step(4, "fourteen broken variants — who catches what?")
RUNS.mkdir(parents=True, exist_ok=True)


def gh(p):
    return p["network_policies"]["github_rest_api"]["endpoints"][0]


def v_rw_root(p): p["filesystem_policy"]["read_write"] = ["/"]
def v_rest_noaccess(p): gh(p).pop("access")
def v_root(p): p["process"] = {"run_as_user": "root", "run_as_group": "root"}
def v_no_port(p): gh(p).pop("port")
def v_Version(p): p["Version"] = p.pop("version")
def v_list(p): p["network_policies"] = [{"host": "pypi.org", "port": 443}]
def v_desc(p): gh(p)["description"] = "GitHub, read-only"
def v_group_field(p): p["network_policies"]["github_rest_api"]["comment"] = "for gh"
def v_deny_wrapped(p): gh(p)["deny_rules"] = [{"deny": {"method": "POST", "path": "/admin/**"}}]
def v_bad_matcher(p): gh(p)["rules"] = [{"allow": {"method": "GET", "path": "/repos/**", "verb": "read"}}]
def v_port_str(p): gh(p)["port"] = "443"
def v_access_typo(p): gh(p)["access"] = "readonly"
def v_sql(p): p["network_policies"]["npm_registry"]["endpoints"][0]["protocol"] = "sql"
def v_landlock(p): p["landlock"]["compatibility"] = "strict"


VARIANTS = [   # (name, mutate, what a real Spark does, per which source)
    ("read_write: [/]", v_rw_root, "refused: INVALID_ARGUMENT (research tutorial)"),
    ("rest, no access or rules", v_rest_noaccess, "the `::rest` trap in YAML — verify on your unit"),
    ("run_as_user: root", v_root, "push fails validation (OpenShell playbook)"),
    ("endpoint without port", v_no_port, "push fails validation (OpenShell playbook)"),
    ("Version: (capital V)", v_Version, "unknown field 'Version' (OpenShell playbook)"),
    ("network_policies as a list", v_list, "expected a map (NemoClaw applications playbook)"),
    ("endpoint description:", v_desc, "same as the parser: unknown field"),
    ("group comment:", v_group_field, "same as the parser: unknown field"),
    ("deny_rules wrapped in deny:", v_deny_wrapped, "same as the parser: unknown field `deny`"),
    ("allow matcher `verb:`", v_bad_matcher, "same as the parser: unknown field"),
    ("port: \"443\" (a string)", v_port_str, "same as the parser: expected u16"),
    ("access: readonly (typo)", v_access_typo, "parser passes it; expect the gateway to refuse"),
    ("protocol: sql", v_sql, "the CLI grammar lists sql — policykit is behind"),
    ("landlock: strict", v_landlock, "parser passes it; expect the gateway to refuse"),
]
rows, pk_hits, os_hits, said = [], 0, 0, []
for i, (name, fn, spark) in enumerate(VARIANTS, 1):
    p = copy.deepcopy(anat)
    fn(p)
    f = RUNS / f"{i:02d}_{fn.__name__[2:]}.yaml"
    f.write_text(pk.dump(p), encoding="utf-8")
    errs, _ = pk.validate(p)
    v, msg = parsekit.cli_file(f)
    pk_hits += bool(errs)
    os_hits += v is False
    said.append((name, errs, msg))
    rows.append([name, "✕ caught" if errs else "· missed", "✕ caught" if v is False else
                 ("· parsed" if v else "· n/a"), spark])
table(rows, ["variant", "policykit", "CLI parser", "on a real gateway (source)"])
print(f"◆ policykit caught {pk_hits}/{len(VARIANTS)} · the real parser caught {os_hits}/{len(VARIANTS)} · "
      f"variant files in 03_policy_as_code/.runs/variants/")

step(5, "the messages, word for word")
for name, errs, msg in said:
    print(f"▣ {name}")
    print(f"  policykit:   {errs[0] if errs else '(no error — the course model does not check this)'}")
    print(f"  CLI 0.0.111: {msg}")

note("Read the columns together. The parser catches STRUCTURE (unknown field, list vs map, wrong type) on your "
     "laptop. SEMANTIC rules (no root, no `/` writable, host + port present, a known access mode) come back from "
     "the gateway when you push; policykit checks them before you push. policykit misses rule-shape mistakes the "
     "parser catches, and it does not know `sql` yet. Use both, then trust the gateway.")
if not parsekit.OPENSHELL.exists():
    warn("the CLI column is n/a — install openshell==0.0.111 into week26/.venv-openshell")
ok("captured on this Mac: policykit + the real OpenShell 0.0.111 parser, no gateway")
result("You can read every rule form, and you know which mistakes the laptop catches and which only the gateway "
       "catches.")
