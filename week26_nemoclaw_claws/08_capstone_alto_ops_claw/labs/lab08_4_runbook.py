#!/usr/bin/env python3
"""Lab 08-4 · The one-page runbook: rebuild, snapshot, rotate credential handles, upgrade OpenShell.

Writes .runs/RUNBOOK.md (capstone deliverable 6). The laptop half is measured for real: the NAT, OpenShell-parser,
Docker and Ollama versions on this Mac. The Spark half is one read-only version check (`openshell --version`,
`nemoclaw update --check`, `nat --version` inside the sandbox). In DRY it is an EXAMPLE and the runbook says
"not checked". Before anything goes into the runbook, every `openshell` command in it is run through the laptop's
0.0.111 CLI as an offline parser. A typo in a runbook is found now, not during an incident.

No LLM calls. Run: .venv/bin/python week26/08_capstone_alto_ops_claw/labs/lab08_4_runbook.py
"""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capkit as ck  # noqa: E402
from clawkit import (LAPTOP_OLLAMA, NAT, OPENSHELL, OPENSHELL_PINNED, ROOT, banner, free_port, http_json,  # noqa: E402
                     label, laptop, note, ok, openshell_offline, parsed_ok, result, sh, step, table, warn)

S, R = ck.SANDBOX, ck.REMOTE
LATEST = "0.1.2"
banner("Lab 08-4 · the one-page runbook", "laptop versions for real · Spark versions read-only (or DRY) · RUNBOOK.md")

# ── 1 · laptop versions ────────────────────────────────────────────────────────────────────────────────────
step(1, "this laptop — tool versions, measured")
env = {"PYTHONWARNINGS": "ignore"}
v = {}
r = laptop([NAT, "--version"], quiet=True, env=env, show="nat --version")
v["nat"] = next((ln.strip() for ln in r.out.splitlines() if ln.startswith("nat")), "missing")
r = laptop([OPENSHELL, "--version"], quiet=True, show="openshell --version")
v["openshell (parser)"] = (r.out.strip().splitlines() or ["missing"])[-1]
r = laptop([ROOT / "week26/.venv-nat/bin/python", "--version"], quiet=True, show="week26/.venv-nat/bin/python --version")
v["python (NAT venv)"] = r.out.strip() or "missing"
if shutil.which("docker"):
    r = laptop(["docker", "--version"], quiet=True, show="docker --version")
    v["docker client"] = r.out.strip().splitlines()[-1] if r.out.strip() else "?"
try:
    v["ollama"] = "ollama " + http_json("GET", LAPTOP_OLLAMA.replace("/v1", "") + "/api/version", timeout=3).get("version", "?")
    print(f"$ curl {LAPTOP_OLLAMA.replace('/v1', '')}/api/version   [this laptop]")
except Exception:  # noqa: BLE001
    v["ollama"] = "not running"
table([[k, val] for k, val in v.items()], ["laptop tool", "version (measured now)"])

# ── 2 · Spark versions (read-only) ─────────────────────────────────────────────────────────────────────────
step(2, "the Spark — versions, read-only")
EX_VERSIONS = f"""openshell {OPENSHELL_PINNED}
<nemoclaw update --check: current LKG, or 'update available'>
nat, version <x.y.z>"""
rs = sh(ck.CMD["versions"], example=EX_VERSIONS, timeout=120)
spark_v = rs.out.strip() if rs.source in ("live", "recorded") else ""
if spark_v:
    ok(f"Spark versions captured ({rs.source})")
    if OPENSHELL_PINNED not in spark_v:
        warn(f"the Spark does not report OpenShell {OPENSHELL_PINNED} — you are off NemoClaw's pin; read runbook §4")
else:
    note("DRY: the Spark versions are NOT checked. The runbook says so instead of copying the EXAMPLE.")

# ── 3 · parse every openshell command the runbook contains ─────────────────────────────────────────────────
step(3, "every openshell command in the runbook, through the 0.0.111 parser (offline)")
POLICY_FILE = str(ck.BUNDLE / "prod-policy.yaml") if (ck.BUNDLE / "prod-policy.yaml").is_file() else \
    str(ck.POLICIES / "prod-policy.yaml")
DATA_DIR = str(ck.BUNDLE / "data") if (ck.BUNDLE / "data").is_dir() else str(ck.CSV.parent)   # upload checks it exists
FP = free_port(8001)                    # the CLI checks the LOCAL forward port is free before it dials the gateway
RUNBOOK_OS = [
    ["--version"],
    ["status"],
    ["sandbox", "list"],
    ["policy", "get", S],
    ["policy", "list", S],
    ["policy", "set", S, "--policy", POLICY_FILE, "--wait"],
    ["inference", "get"],
    ["provider", "list"],
    ["provider", "update", "local-vllm", "--credential", "OPENAI_API_KEY"],
    ["logs", S, "--source", "sandbox", "-n", "200"],
    ["sandbox", "delete", S],
    ["sandbox", "upload", S, DATA_DIR, "/sandbox/data"],
    ["forward", "start", "--background", str(FP), S],
]
parse_rows, all_parsed = [], True
for argv in RUNBOOK_OS:
    if argv == ["--version"]:
        continue
    rr = openshell_offline(argv)
    good = parsed_ok(rr)
    all_parsed &= good
    shown = " ".join(argv).replace(POLICY_FILE, "./prod-policy.yaml").replace(DATA_DIR, "./data").replace(f"background {FP}", "background 8001")
    parse_rows.append([f"openshell {shown}"[:62], "✓ parsed" if good else "✕ " + rr.out.strip().splitlines()[0][:40]])
table(parse_rows, ["runbook command", "0.0.111 parser"])
print(("✓ " if all_parsed else "✕ ") + f"{len(parse_rows)} openshell commands parse on the laptop CLI "
      f"(the Spark runs {OPENSHELL_PINNED}: re-check any flag with --help there)")

# ── 4 · write RUNBOOK.md ───────────────────────────────────────────────────────────────────────────────────
step(4, "write the one-page runbook → week26/08_capstone_alto_ops_claw/.runs/RUNBOOK.md")
man = ck.load_json(ck.BUNDLE / "MANIFEST.json")
man_sha = ck.sha256(ck.BUNDLE / "MANIFEST.json")[:16] if man else "— (run lab 08-1 first)"
spark_line = (f"measured {time.strftime('%Y-%m-%d')} on {label()} ({rs.source}):\n\n```\n{spark_v}\n```" if spark_v else
              "**not checked**: no Spark was connected when this runbook was generated. Run §0 on the Spark "
              "and paste the output here.")
laptop_tbl = "\n".join(f"| {k} | {val} |" for k, val in v.items())
RUNBOOK = f"""# Alto Ops Claw v1: runbook (one page)

Sandbox `{S}` · bundle `{R}` · MANIFEST sha256 `{man_sha}` · generated {time.strftime('%Y-%m-%d %H:%M')} by lab 08-4.
Rules that never change: **snapshot before a change · recreate on suspicion, never clean in place · no key in a
file, a policy or a command line · the sandbox never holds a provider credential**.

## 0 · Versions (before and after every change)
```bash
openshell --version                                  # NemoClaw pins {OPENSHELL_PINNED}; latest OpenShell is {LATEST}
nemoclaw update --check                              # a newer NemoClaw LKG release?
nemoclaw upgrade-sandboxes --check                   # sandboxes built by an older release?
openshell sandbox exec -n {S} -- nat --version  # NAT inside the image
```
Spark: {spark_line}

| laptop tool (measured) | version |
|---|---|
{laptop_tbl}

## 1 · Snapshot (before any change)
```bash
nemoclaw {S} snapshot create --name before-change   # NemoClaw-managed sandboxes (onboard --from ./Dockerfile)
openshell policy get {S} > policy-$(date +%F).yaml  # raw OpenShell: keep the policy (no --full: policy set rejects its header)
openshell policy list {S}                           # and the revision number it came from
```
The image inputs are pinned by `MANIFEST.json` (SHA-256 per file). Keep the bundle tarball with the snapshot.

## 2 · Rebuild (new image, new tool code, or suspected compromise)
```bash
nemoclaw {S} rebuild                                # NemoClaw-managed
openshell sandbox delete {S}                        # raw OpenShell: recreate from TRUSTED inputs (the bundle)
cd {R} && openshell sandbox create --name {S} --from ./ --policy ./prod-policy.yaml \\
  --forward 8001 --keep -- nat serve --config_file /app/workflow.yml --host 0.0.0.0 --port 8001
openshell sandbox upload {S} ./data /sandbox/data
openshell forward start --background 8001 {S}
```
Then re-prove the write-deny path (lab 08-2 step 5): `openshell logs {S} --source sandbox -n 200` shows the deny.
Filesystem, Landlock and process rules are locked at creation. Changing them means this rebuild, not `policy set`.

## 3 · Rotate credential handles
Alto Ops Claw v1 holds **no** credential in the sandbox. The only credential is the inference provider's, held by
the gateway. Rotate it there, then prove the route still works:
```bash
nemoclaw credentials reset <KEY> && nemoclaw onboard             # NemoClaw (playbook "Manage later")
export OPENAI_API_KEY=<new-key>                                  # raw OpenShell: the value comes from the env, never argv
openshell provider update local-vllm --credential OPENAI_API_KEY # from `provider update --help` on 0.0.111: confirm on {OPENSHELL_PINNED}
unset OPENAI_API_KEY
openshell inference get                                          # provider + model still set
```
SaaS keys (e.g. Langfuse on a Hermes claw) follow the same pattern: an OpenShell credential of type
`langfuse-hermes-v1`, placeholders in the sandbox, then `nemohermes <s> rebuild`. Never write a key to `/sandbox/.hermes`.

## 4 · Upgrade OpenShell (pinned {OPENSHELL_PINNED} vs latest {LATEST})
1. Stay on NemoClaw's pin in production. Move only when NemoClaw moves: `nemoclaw update --check`, then
   `nemoclaw update --yes` (host CLI only), then `nemoclaw upgrade-sandboxes --check` and rebuild what it lists.
   The playbook's fix for an old OpenShell is to re-run the `nemoclaw.sh` install. Pin a release with
   `NEMOCLAW_INSTALL_TAG=vX.Y.Z`.
2. Try {LATEST} ahead of the pin only on a second, non-production Spark. There, re-run lab 08-1 against the new
   CLI's `--help`, re-prove deliverable 2 (the deny), and re-measure the sandbox tax. Publish the number with
   `openshell --version`.
3. External gateways pin it too: `blueprint.yaml` has `min_openshell_version` / `max_openshell_version` =
   {OPENSHELL_PINNED}.
4. Roll back: snapshot (§1) + reinstall the pinned tag + rebuild (§2).

## 5 · Evidence kept every month
`openshell policy list` + `policy get --full` · `openshell inference get` · `openshell logs` export · the on-prem
OTel collector file · the supervisor's Landlock lines under hard_requirement · the tier record (Exercise 08).
"""
out = ck.RUNS / "RUNBOOK.md"
out.write_text(RUNBOOK, encoding="utf-8")
lines, words = RUNBOOK.count("\n"), len(RUNBOOK.split())
required = {"versions": "## 0 · Versions", "snapshot": "## 1 · Snapshot", "rebuild": "## 2 · Rebuild",
            "rotate credential handles": "## 3 · Rotate credential handles",
            f"upgrade OpenShell ({OPENSHELL_PINNED} vs {LATEST})": f"## 4 · Upgrade OpenShell (pinned {OPENSHELL_PINNED} vs latest {LATEST})"}
have = {k: h in RUNBOOK for k, h in required.items()}
for k, got in have.items():
    print(("✓ " if got else "✕ ") + f"section: {k}")
one_page = lines <= 90
print(("✓ " if one_page else "⚠ ") + f"{lines} lines · {words} words — {'fits' if one_page else 'longer than'} one page")
(ck.RUNS / "runbook.json").write_text(json.dumps({"date": time.strftime("%Y-%m-%d %H:%M"), "sections": have,
                                                   "lines": lines, "words": words, "parsed_all": all_parsed,
                                                   "spark_versions": spark_v, "spark_source": rs.source,
                                                   "laptop_versions": v}, indent=1), encoding="utf-8")
ok(f"→ {out.relative_to(ROOT)}")
print("· RUNBOOK (first lines)")
for ln in RUNBOOK.splitlines()[:6]:
    print("  " + ln)
print()
if all(have.values()) and all_parsed:
    result("Runbook written: 5 sections, one page, every openshell command parsed. Spark versions: "
           + ("captured." if spark_v else "NOT checked (DRY) — the runbook says so."))
else:
    print("✕ the runbook is missing a section or a command failed to parse")
    sys.exit(1)
