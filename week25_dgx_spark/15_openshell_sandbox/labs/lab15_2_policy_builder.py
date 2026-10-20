#!/usr/bin/env python3
"""Lab 15-2 · Build an OpenShell policy for the NAT hotel agent, then validate it two ways.

1. Generate the policy from a short Python spec (policykit.build_policy) and check it matches the
   committed policies/hotel_agent_policy.yaml.
2. Validate it — and the policy NVIDIA ships with the healthcare-agent playbook, and eight broken
   variants — with policykit.validate(): the rules the playbooks document (course-made checker).
3. Ask the REAL OpenShell CLI parser about the same files. `openshell policy set` parses the YAML on
   your machine before it contacts a gateway, so pointing it at a closed port (127.0.0.1:9) tells you
   "parse error" vs "parsed, then could not connect" without any gateway. Semantic rules (path must
   start with /, no root user, host + port present) are checked later, by the gateway — which is why
   policykit exists.

Run: .venv/bin/python week25/15_openshell_sandbox/labs/lab15_2_policy_builder.py
"""
import copy
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE.parents[1]))
import policykit as pk  # noqa: E402
from sparkkit import ROOT, banner, check, note, result, step, table, warn  # noqa: E402

MOD = HERE.parents[1]
RUNS = MOD / ".runs"
OPENSHELL = Path(os.environ.get("OPENSHELL_BIN", MOD / ".venv-openshell" / "bin" / "openshell"))
SHIPPED = ROOT / "dgx-spark-playbooks" / "nvidia" / "playbook-healthcare-agent" / "assets" / "sandbox-policy.yaml"
ANSI = re.compile(r"\x1b\[[0-9;]*m")

PY = ["/sandbox/.venv-nat/bin/python*", "/usr/bin/python3*"]
SPEC = dict(
    read_only=["/usr", "/lib", "/proc", "/dev/urandom", "/etc"],
    read_write=["/sandbox", "/tmp", "/dev/null"],
    groups={
        "inference": pk.group("inference", [{"host": "inference.local", "port": 443}], PY + ["/usr/bin/curl"]),
        "pypi": pk.group("pypi", [{"host": "pypi.org", "port": 443, "access": "full", "tls": "skip"},
                                  {"host": "files.pythonhosted.org", "port": 443, "access": "full", "tls": "skip"}],
                         ["/sandbox/.venv-nat/bin/python*", "/sandbox/.venv-nat/bin/pip*", "/usr/bin/python3*"]),
    })


def openshell_parse(path: Path) -> str:
    """'parsed' | 'parse error: …' | 'n/a' — the real CLI parser, no gateway involved."""
    if not OPENSHELL.exists():
        return "n/a (no laptop CLI)"
    home = RUNS / "openshell-home"
    home.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "NO_COLOR": "1", "HOME": str(home), "OPENSHELL_GATEWAY_ENDPOINT": "http://127.0.0.1:9"}
    p = subprocess.run([str(OPENSHELL), "policy", "set", "parse-probe", "--policy", str(path)], capture_output=True,
                       text=True, timeout=60, env=env)
    out = " ".join(ANSI.sub("", p.stdout + p.stderr).split())
    if "failed to parse sandbox policy YAML" in out:
        return "✕ " + re.sub(r"^.*?policy YAML\s*[╰─▶]*\s*", "", out)[:70]
    if "transport error" in out or "failed to connect" in out:
        return "✓ parsed"
    return "? " + out[:60]


banner("Lab 15-2 · build and validate an OpenShell policy", "offline · policykit (course-made) + the real OpenShell "
       "CLI parser · no gateway", status=False)
RUNS.mkdir(exist_ok=True)

step(1, "generate the hotel-agent policy from a Python spec")
policy = pk.build_policy(**SPEC)
gen = RUNS / "hotel_agent_policy.generated.yaml"
gen.write_text(pk.dump(policy), encoding="utf-8")
print(pk.dump(policy).rstrip())
committed = pk.load(MOD / "policies" / "hotel_agent_policy.yaml")
check(committed == policy, "generated policy == policies/hotel_agent_policy.yaml (same keys, same values)",
      "the generated policy differs from policies/hotel_agent_policy.yaml")

step(2, "eight broken variants — each one is a row in the playbooks' troubleshooting tables")


def variant(name: str, fn) -> tuple[str, dict]:
    p = copy.deepcopy(policy)
    fn(p)
    return name, p


def _list_net(p):
    p["network_policies"] = [{"host": "pypi.org", "port": 443}]


def _no_port(p):
    del p["network_policies"]["inference"]["endpoints"][0]["port"]


def _description(p):
    p["network_policies"]["pypi"]["endpoints"][0]["description"] = "PyPI"


VARIANTS = [
    variant("Version (capital V)", lambda p: p.update({"Version": p.pop("version")})),
    variant("network_policies as a list", _list_net),
    variant("relative / .. path", lambda p: p["filesystem_policy"]["read_write"].extend(["sandbox", "/tmp/../etc"])),
    variant("run_as_user: root", lambda p: p["process"].update(run_as_user="root")),
    variant("endpoint without port", _no_port),
    variant("group without binaries", lambda p: p["network_policies"]["pypi"].pop("binaries")),
    variant("endpoint without access mode", lambda p: p["network_policies"]["pypi"]["endpoints"][0].pop("access")),
    variant("endpoint with description:", _description),
]
cases = [("policies/hotel_agent_policy.yaml", committed, MOD / "policies" / "hotel_agent_policy.yaml")]
if SHIPPED.exists():
    cases.append(("healthcare playbook (shipped)", pk.load(SHIPPED), SHIPPED))
for name, p in VARIANTS:
    f = RUNS / ("bad_" + re.sub(r"\W+", "_", name).strip("_").lower() + ".yaml")
    f.write_text(pk.dump(p) if isinstance(p, dict) else str(p), encoding="utf-8")
    cases.append((name, p, f))

rows, details = [], []
for name, p, path in cases:
    errs, warns = pk.validate(p)
    verdict = "✓ ok" if not errs and not warns else (f"✕ {len(errs)} error(s)" if errs else f"⚠ {len(warns)} warning(s)")
    rows.append([name, verdict, openshell_parse(path)])
    details += [(name, "✕", e) for e in errs] + [(name, "⚠", w) for w in warns]
table(rows, ["policy", "policykit.validate", "openshell CLI parser"])

step(3, "why each variant fails (policykit's messages)")
for name, glyph, msg in details:
    print(f"{glyph} {name}: {msg}")

note("Read the two columns together. The CLI parser rejects STRUCTURE (unknown field, list vs map, wrong type) "
     "on your laptop. SEMANTIC rules — absolute paths, no root, host + port — come back from the gateway when you "
     "push; policykit checks them before you push. A clean parse is not a working policy: 'no binaries' and "
     "'no access mode' apply fine and then deny every call.")
if not OPENSHELL.exists():
    warn("the OpenShell column is n/a — lab 15-1 prints the one-line laptop install")
result("policies/hotel_agent_policy.yaml (== the generated file) is the policy Module 20 uses; push it with "
       "`openshell sandbox create --policy …` (lab 15-4).")
