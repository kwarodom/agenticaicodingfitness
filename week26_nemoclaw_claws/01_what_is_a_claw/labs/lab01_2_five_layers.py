#!/usr/bin/env python3
"""Lab 01-2 · Five layers, two speeds: one allow/deny decision per policy layer.

Offline, runs anywhere. Loads the annotated policy from the research tutorial's Lab 2.2, adds the process
section NemoClaw's baseline uses (a dedicated non-root user), validates it, then asks the course's policykit
teaching model for one decision per layer: filesystem, process, network (L4 binary check + L7 method check),
inference, and credentials. Finally it sorts the layers into hot-reloadable vs locked at creation.

policykit.decide() is NOT OpenShell — the real enforcement is Landlock, seccomp and the proxy on the Spark.
It follows the semantics the docs describe so you can reason about a policy before you push it.

Run: .venv/bin/python week26/01_what_is_a_claw/labs/lab01_2_five_layers.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import policykit as pk  # noqa: E402
from clawkit import banner, note, ok, result, step, table, warn  # noqa: E402

# The research tutorial's Lab 2.2 policy (its comments trimmed), with the process block uncommented.
POLICY_YAML = """
version: 1
filesystem_policy:
  include_workdir: true
  read_only: [/usr, /lib, /etc]
  read_write: [/tmp]
landlock:
  compatibility: best_effort
process:
  run_as_user: "1500"
  run_as_group: "1500"
network_policies:
  github_rest_api:
    endpoints:
      - host: api.github.com
        port: 443
        protocol: rest
        enforcement: enforce
        access: read-only
    binaries:
      - path: /usr/bin/gh
  npm_registry:
    endpoints:
      - host: registry.npmjs.org
        port: 443
        protocol: rest
        enforcement: enforce
        access: read-only
        allow_encoded_slash: true
    binaries:
      - path: /usr/bin/node
"""

banner("Lab 01-2 · five layers, two speeds", "offline · policykit teaching model · no Spark needed", status=False)

step(1, "load and validate the policy")
policy = pk.load(POLICY_YAML)
errs, warns = pk.validate(policy)
for e in errs:
    print(f"✕ {e}")
for w in warns:
    warn(w)
if not errs:
    ok(f"valid: {len(policy['network_policies'])} network groups, "
       f"{len(policy['filesystem_policy']['read_only'])} read-only paths, runs as uid "
       f"{policy['process']['run_as_user']}")

step(2, "one decision per layer")
PY, GH, CURL = "/usr/bin/python3", "/usr/bin/gh", "/usr/bin/curl"
CASES = [
    ("filesystem", {"op": "write", "path": "/tmp/report.csv"}),
    ("filesystem", {"op": "write", "path": "/etc/hosts"}),
    ("process", {"op": "run_as", "user": "root"}),
    ("network · L4", {"op": "connect", "host": "api.github.com", "port": 443, "binary": CURL}),
    ("network · L7", {"op": "http", "host": "api.github.com", "port": 443, "binary": GH, "method": "GET",
                      "path": "/repos/nvidia/nemoclaw"}),
    ("network · L7", {"op": "http", "host": "api.github.com", "port": 443, "binary": GH, "method": "POST",
                      "path": "/repos/nvidia/nemoclaw/issues"}),
    ("network", {"op": "connect", "host": "api.openai.com", "port": 443, "binary": PY}),
    ("network · SSRF", {"op": "connect", "host": "169.254.169.254", "port": 80, "binary": CURL}),
    ("inference", {"op": "http", "host": "inference.local", "port": 443, "binary": PY, "method": "POST",
                   "path": "/v1/chat/completions"}),
]
rows = []
for layer, action in CASES:
    decision, why = pk.decide(policy, action)
    glyph = {"allow": "✓ allow", "deny": "✕ deny", "inspect_for_inference": "◆ inspect_for_inference"}[decision]
    rows.append([layer, pk.fmt_action(action), glyph, why])
table(rows, ["layer", "action", "decision", "why (policykit)"])
note("api.openai.com is denied even though the agent 'needs a model': inference goes through inference.local, "
     "where the gateway injects the real key. Never add provider hosts to a policy (Module 07).")

step(3, "credentials — a literal secret in a policy file is refused")
leaky = POLICY_YAML + "\n# oops\n#   api_key: sk-live-1234567890abcdefgh\n"
secret = re.search(r"(sk-|nvapi-|hf_)[A-Za-z0-9_\-]{12,}", leaky)
if secret:
    print(f"✕ literal credential in the policy text: {secret.group(0)[:8]}•••  → NemoClaw refuses it; the "
          "credential belongs in an OpenShell provider, the sandbox sees only a placeholder")
note("This check is the course's regex. The real rule (how-it-works page): literal credentials in policy "
     "files are refused, and `nemoclaw <s> policy get` replaces them with [STRIPPED_BY_MIGRATION].")

step(4, "two speeds — what you can change on a running sandbox")
table([
    ["filesystem", "filesystem_policy · landlock", "LOCKED at creation", "recreate the sandbox"],
    ["process", "process", "LOCKED at creation", "recreate the sandbox"],
    ["network", "network_policies", "HOT — reloadable", "openshell policy update / set --wait"],
    ["inference", "(gateway route)", "HOT — reloadable", "openshell inference set / nemoclaw inference set"],
    ["gateway auth", "(gateway, not this file)", "—", "tokens, device pairing"],
], ["layer", "where", "speed", "how you change it"])
result("Two layers you change live (network, inference); two you only change by recreating (filesystem, process).")
