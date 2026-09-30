#!/usr/bin/env python3
"""Lab 03-1 · Read the policy NemoClaw created for you (L2.1), without changing anything.

Read-only on the Spark: `nemoclaw <s> policy list` and `policy get`, then OpenShell's own views —
`openshell policy get <s> --base`, `--full`, and the revision list `openshell policy list <s>`. It looks for the
six baseline entries the NemoClaw network-policies reference names. In DRY mode every Spark output is an
EXAMPLE shape (labelled), and nothing is counted as found.

On THIS laptop it runs the real OpenShell CLI (0.0.111) as an offline parser to show the one export trap the
DGX Spark OpenShell playbook warns about: a `--full` dump carries a `Version` header that `policy set` refuses.

Run: .venv/bin/python week26/03_policy_as_code/labs/lab03_1_read_policy.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE.parents[1]))
import parsekit  # noqa: E402
import policykit as pk  # noqa: E402
from clawkit import banner, note, ok, result, sandbox, sh, step, table, warn  # noqa: E402

MOD = HERE.parents[1]
RUNS = MOD / ".runs"
S = sandbox()
BASELINE = {   # research tutorial §2.2, citing the NemoClaw network-policies reference
    "nvidia": "integrate.api.nvidia.com:443 · binary /usr/local/bin/openclaw · POST inference/embedding, GET models",
    "clawhub": "named in the baseline — read its hosts in your export",
    "openclaw_api": "named in the baseline — read its hosts in your export",
    "openclaw_docs": "named in the baseline — read its hosts in your export",
    "npm_registry": "GET only · openclaw binary only",
    "managed_inference": "the required inference route",
}

EX_LIST = """(EXAMPLE shape — your tier and presets decide the real list)
  preset       status
  npm          applied
  pypi         applied
  telegram     not applied"""
EX_GET = """version: 1
filesystem_policy:
  include_workdir: true
  read_only: [/usr, /lib, /proc, /dev/urandom, /app, /etc, /var/log, /var/lib/dpkg]
  read_write: [/sandbox, /tmp, /dev/null, /dev/pts]
network_policies:
  nvidia: {…}
  clawhub: {…}
  openclaw_api: {…}
  openclaw_docs: {…}
  npm_registry: {…}
  managed_inference: {…}"""
EX_BASE = """version: 1
network_policies:          (base only — no provider-composed entries)
  nvidia: {…}
  …"""
EX_FULL = """<metadata header: sandbox, revision, a Version field …>
---
version: 1
network_policies:
  … the base entries, plus entries composed from attached providers …"""
EX_REVS = """  REV  STATUS   CREATED
  2    loaded   <time>
  1    loaded   <time>"""

banner("Lab 03-1 · read the policy NemoClaw created", f"sandbox `{S}` · read-only on the Spark · the real CLI "
       "parser on this laptop")

step(1, f"NemoClaw's view — which presets are on `{S}`?")
note("The research tutorial spells it `policy list`; the DGX Spark NemoClaw playbook spells it `policy-list`. "
     "The command below tries one, then the other — `nemoclaw <s> --help` on your unit wins.")
sh(f"nemoclaw {S} policy list 2>/dev/null || nemoclaw {S} policy-list", example=EX_LIST, timeout=60)

step(2, "NemoClaw's export — the policy as YAML, credentials stripped")
res = sh(f"mkdir -p ~/week26 && nemoclaw {S} policy get > ~/week26/current-policy.yaml && "
         "head -40 ~/week26/current-policy.yaml", example=EX_GET, timeout=60)
note("`nemoclaw <s> policy get` strips metadata and replaces any literal credential with "
     "[STRIPPED_BY_MIGRATION] (needs OpenShell 0.0.72+). Lab 03-3 pushes this file back with `policy set`.")

step(3, "OpenShell's views — base, full (effective), and the revision history")
sh(f"openshell policy get {S} --base | head -30", example=EX_BASE, timeout=60)
sh(f"openshell policy get {S} --full | head -12", example=EX_FULL, timeout=60)
sh(f"openshell policy list {S}", example=EX_REVS, timeout=60)

step(4, "do you recognise the baseline? (the NemoClaw network-policies reference names six entries)")
exported = sh("cat ~/week26/current-policy.yaml", quiet=True, timeout=60) if res.live and res.ok else None
live = bool(exported and exported.ok)
rows = []
for name, what in BASELINE.items():
    status = ("✓ in your policy" if name in exported.out else "✕ not in the export") if live else "◈ DRY — not checked"
    rows.append([name, what, status])
table(rows, ["entry", "what the docs say it is", "on your Spark"])
note("All six are TLS-terminated on 443. `managed_inference` is required: it is how inference.local reaches the "
     "gateway, so never remove it.")
if live:
    RUNS.mkdir(parents=True, exist_ok=True)
    local = RUNS / "current-policy.from-spark.yaml"
    local.write_text(exported.out, encoding="utf-8")
    try:
        errs, _ = pk.validate(pk.load(exported.out))
    except Exception as e:  # noqa: BLE001
        errs = [f"not YAML: {e}"]
    v, msg = parsekit.cli_file(local)
    table([["policykit.validate (course model)", "✓ no errors" if not errs else f"✕ {len(errs)} error(s)"],
           ["openshell CLI parser (this laptop)", parsekit.glyph(v, msg)]], ["check on your export", "verdict"])
    for e in errs:
        print(f"✕ {e}")
else:
    warn("DRY: the table above is not your machine. Connect a Spark (🖥 Spark setup) and switch to ⚡ Live.")

step(5, "the export trap, for real on this laptop — `--full` is for reading, not for `policy set`")
RUNS.mkdir(parents=True, exist_ok=True)
body = (MOD / "policies" / "anatomy.yaml").read_text(encoding="utf-8")
dumped = RUNS / "full-dump-like.yaml"
dumped.write_text("Version: 3\n" + body, encoding="utf-8")        # a header key in front of the policy keys
fixed = RUNS / "full-dump-fixed.yaml"
fixed.write_text(body, encoding="utf-8")
v1, m1 = parsekit.cli_file(dumped)
v2, m2 = parsekit.cli_file(fixed)
v3, m3 = parsekit.cli(["policy", "get", S, "--base", "--full"])
table([["policy with a `Version:` header line", parsekit.glyph(v1, m1)],
       ["same file, header removed", parsekit.glyph(v2, m2)],
       ["policy get --base --full", parsekit.glyph(v3, m3)]],
      ["what the laptop CLI 0.0.111 was given", "what it said"])
print(f"✕ `Version:` header → {m1}")
print(f"✕ --base --full → {m3}")
note("The playbook's fix: export with `openshell policy get <s>` (no --full), or strip every line before the "
     "first `---`. `--base` and `--full` are two different views, so the CLI refuses both at once.")
ok("captured on this Mac: the real parser, no gateway")
result("You can read a claw's policy three ways (NemoClaw export, OpenShell base, OpenShell full) and you know "
       "which one to edit and push back.")
