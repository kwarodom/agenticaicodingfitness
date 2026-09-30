#!/usr/bin/env python3
"""Lab 03-4 · Presets, posture profiles, snapshots and operator approval (L2.4 · L2.5 · L2.6).

  • The maintained preset catalogue and the risk notes the NemoClaw security guide gives (as cited in the
    research tutorial), and the four posture profiles.
  • Read-only on the Spark: `nemoclaw <s> policy list` → which presets are applied, which profile that looks like.
  • Changes only via clawkit.change(): add a preset (previewed with --dry-run), create a snapshot.
  • Operator approval in `openshell term` is interactive, so the lab never runs it. It shows the real laptop
    CLI's `--approval-mode` help text, then SIMULATES the approval lifecycle with policykit (a teaching model):
    denied → approved revision → recreated sandbox, denied again.

Run: .venv/bin/python week26/03_policy_as_code/labs/lab03_4_presets_posture.py
"""
import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE.parents[1]))
import parsekit  # noqa: E402
import policykit as pk  # noqa: E402
from clawkit import banner, change, note, ok, result, sandbox, sh, step, table, warn  # noqa: E402

MOD = HERE.parents[1]
S = sandbox()
PRESETS = ["brave", "brew", "claude-code", "discord", "github", "gmail", "googlechat", "huggingface", "jira",
           "local-inference", "npm", "nous-*", "openclaw-pricing", "outlook", "public-reference", "pypi", "slack",
           "tavily", "teams", "telegram", "weather", "wechat", "whatsapp"]
RISK = {   # research tutorial §2.6 (NemoClaw security best practices) + the NemoClaw applications playbook
    "pypi": "GET/HEAD only — but allows installing arbitrary packages",
    "github": "read/write to repos via git only (binary-scoped to /usr/bin/git)",
    "slack": "WebSocket legs use access: full with no inspection",
    "discord": "WebSocket legs use access: full with no inspection",
    "whatsapp": "access: full + tls: skip pass-through (applications playbook)",
    "brew": "access: full + tls: skip pass-through (applications playbook)",
    "nous-*": "the Hermes presets",
    "personal-open-internet": "removes hostname, method, path and body limits on 80/443",
}
PROFILES = [
    ["Locked-Down", "Restricted", "none (no web search)", "NVIDIA Endpoints or local Ollama",
     "operator approval for everything else; watch TUI"],
    ["Development", "Balanced", "pypi, npm", "any", "keep binary restrictions; review with openshell term"],
    ["Personal", "Personal", "personal-open-internet", "any", "trusted single-user only; recreate as Balanced"],
    ["Integration Testing", "custom", "tight method/path entries, protocol: rest", "any",
     "clean up baseline after tests"],
]

banner("Lab 03-4 · presets, posture, snapshots, approval", f"sandbox `{S}` · read-only + change() on the Spark · "
       "policykit simulation on this laptop")

step(1, "the maintained presets (nemoclaw-blueprint/policies/presets/) — read the risk column first")
rows = [[p, RISK.get(p, "read its YAML before you apply it")] for p in PRESETS]
rows.append(["personal-open-internet", RISK["personal-open-internet"] + "  (the Personal tier)"])
table(rows, ["preset", "risk note"])

step(2, "posture profiles (NemoClaw security best practices, as the research tutorial tabulates them)")
table(PROFILES, ["profile", "tier", "presets", "inference", "notes"])

step(3, f"which presets are on `{S}` — and which profile does that look like? (read-only)")
res = sh(f"nemoclaw {S} policy list 2>/dev/null || nemoclaw {S} policy-list", timeout=60,
         example="  preset   status\n  npm      applied\n  pypi     applied")
if res.live and res.ok:
    on = [p for p in PRESETS + ["personal-open-internet"] if p.rstrip("*") in res.out]
    guess = ("Personal" if "personal-open-internet" in on else "Development" if {"pypi", "npm"} & set(on)
             else "Locked-Down" if not on else "custom — compare with the table")
    ok(f"presets named in the output: {', '.join(on) or 'none'} → closest profile: {guess}")
    note("A name in the output is not proof it is applied — read the status column yourself.")
else:
    warn("DRY: no Spark — the list above is an EXAMPLE. The closest-profile guess runs only on real output.")

step(4, "add a preset — preview first, then apply (Development profile: pypi)")
change(f"nemoclaw {S} policy add pypi --yes", preview=f"nemoclaw {S} policy add pypi --dry-run",
       example="preset pypi → pypi.org / files.pythonhosted.org · GET/HEAD · nothing applied (--dry-run)")
note("The DGX Spark playbooks spell it `policy-add pypi`. Network presets hot-reload: no rebuild.")

step(5, "snapshot before you change anything big (L2.6)")
change(f"nemoclaw {S} snapshot create --name before-change", preview=f"nemoclaw {S} status",
       example=f"  Sandbox: {S}\n  Phase:   Ready\n  … (status shape)")
print(f"→ after risky changes, rebuild in the ⌨ terminal when you mean it:  nemoclaw {S} rebuild")
note("Suspected compromise? Do not clean the sandbox — recreate it from trusted inputs. The agent can rewrite "
     "its own config tree, so a cleaned sandbox is not a trusted one.")

step(6, "operator approval in the TUI (L2.4) — what the real CLI says, then a simulation")
r = parsekit.run(["sandbox", "create", "--help"])
block, grab = [], False
for ln in r.out.splitlines():
    if "--approval-mode" in ln:
        grab = True
    elif grab and ln.strip().startswith("-") and "approval" not in ln:
        break
    if grab and ln.strip():
        block.append(ln.rstrip())
print("\n".join(block) if block else "⚠ this CLI build has no --approval-mode flag")
ok("captured on this Mac: `openshell sandbox create --help` (0.0.111)")
print("→ in the ⌨ terminal on the Spark: `openshell term`, then ask the assistant to "
      "\"fetch https://httpbin.org/get and show me the headers\", approve once, and read the new revision with "
      f"`openshell policy list {S}`")

base = pk.load(MOD / "policies" / "anatomy.yaml")
approved = copy.deepcopy(base)
approved["network_policies"]["httpbin"] = pk.group("httpbin", [{"host": "httpbin.org", "port": 443}],
                                                   ["/usr/local/bin/openclaw"])
FETCH = {"op": "connect", "host": "httpbin.org", "port": 443, "binary": "/usr/local/bin/openclaw"}
rows = []
for label, pol in [("rev N · before approval", base), ("rev N+1 · after one approval (simulated entry)", approved),
                   ("same instance, after a restart", approved), ("destroyed + recreated", base)]:
    d, why = pk.decide(pol, FETCH)
    rows.append([label, "✓ allow" if d == "allow" else "✕ deny", why])
table(rows, ["policy state", "openclaw → httpbin.org:443", "why (policykit)"])
note("SIMULATION: the entry the TUI really writes may differ — read it with `openshell policy get`. The rule it "
     "models is the docs': an approval becomes a durable revision for this sandbox instance and is gone when the "
     "sandbox is destroyed and recreated. Before approval, OpenShell's prover flags risky new access (a new host "
     "with credentials, a new API method) and waits for a human.")
result("You can pick a posture, add a preset safely, snapshot first, and you know an approval lives only as long "
       "as the sandbox instance.")
