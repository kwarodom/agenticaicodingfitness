#!/usr/bin/env python3
"""Lab 04-5 · NAT inside an OpenShell sandbox: build context, policy, and the Spark steps (+ L3.7 code execution).

Research tutorial L3.8 and L3.7. On THIS laptop, for real: generate the sandbox build context into
.runs/alto-ops-sandbox/ (the tutorial's Dockerfile.alto-ops, workflow.sandbox.yml, a deny-by-default
alto-ops-policy.yaml, the alto_ops package and the CSV), `nat validate` the sandbox workflow, check the policy
with the course's policykit model AND the real OpenShell 0.0.111 parser, and parse the tutorial's
`openshell sandbox create …` line offline. On the Spark: create / upload / forward / logs, gated by change().
Then L3.7: which code-execution sandbox NAT 1.9 really supports, and whether Docker is up here.

No LLM calls.
Run: .venv/bin/python week26/04_nat_claws/labs/lab04_5_nat_in_sandbox.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import policykit as pk  # noqa: E402
from clawkit import (NAT_PY, OPENSHELL_PINNED, WEEK, apply_enabled, banner, change, note, ok, openshell_offline,  # noqa: E402
                     parsed_ok, put, result, sh, step, table, warn, where)
from natkit import CONFIGS, CSV, RUNS, SANDBOX_YML, docker_daemon, nat, rel  # noqa: E402

banner("Lab 04-5 · NAT inside an OpenShell sandbox", "laptop: build context + policy checks for real · Spark: via "
       "change() (or DRY)")
CTX = RUNS / "alto-ops-sandbox"
SB = "alto-ops"

# The research tutorial's Dockerfile, verbatim. It says: "this tutorial's own composition — verify the NAT install
# line and non-root USER against your base image".
DOCKERFILE = """FROM ubuntu:24.04
RUN apt-get update && apt-get install -y python3.12 python3-pip curl && rm -rf /var/lib/apt/lists/*
RUN pip3 install --break-system-packages uv && uv pip install --system 'nvidia-nat[langchain,mcp,profiler,opentelemetry]'
COPY workflows/alto_ops /app/alto_ops
RUN uv pip install --system -e /app/alto_ops
COPY workflow.sandbox.yml /app/workflow.yml
USER 1500
WORKDIR /sandbox
"""
# Deny by default: no network_policies at all. Inference is NOT an entry you add — the supervisor intercepts
# https://inference.local and the gateway routes it (research tutorial L3.8).
POLICY = """version: 1
filesystem_policy:
  include_workdir: true          # /sandbox (WORKDIR) is writable — tickets, uploaded data
  read_only: [/usr, /lib, /etc, /app]
  read_write: [/tmp]
landlock:
  compatibility: best_effort
process:
  run_as_user: "1500"            # matches USER 1500 in the Dockerfile; OpenShell rejects root
  run_as_group: "1500"
network_policies: {}             # nothing leaves the sandbox except inference.local
"""

step(1, f"generate the build context → {rel(CTX)}/")
shutil.rmtree(CTX, ignore_errors=True)
(CTX / "data").mkdir(parents=True)
(CTX / "Dockerfile.alto-ops").write_text(DOCKERFILE, encoding="utf-8")
(CTX / "alto-ops-policy.yaml").write_text(POLICY, encoding="utf-8")
shutil.copyfile(SANDBOX_YML, CTX / "workflow.sandbox.yml")
shutil.copyfile(CSV, CTX / "data" / "chiller_plant.csv")
shutil.copytree(WEEK / "common" / "alto_ops", CTX / "workflows" / "alto_ops",
                ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
files = sorted(str(p.relative_to(CTX)) for p in CTX.rglob("*") if p.is_file())
table([[f, f"{(CTX / f).stat().st_size:,} B"] for f in files], ["file", "size"])
note("The Dockerfile is the research tutorial's own composition, not an NVIDIA image. It was NOT built here (see "
     "step 7): check the NAT install line and USER 1500 on your base image before you trust it.")

step(2, "workflow.sandbox.yml — the only change is where the LLM lives")
for ln in (CTX / "workflow.sandbox.yml").read_text(encoding="utf-8").splitlines():
    if "base_url" in ln or "model_name" in ln or "api_key" in ln or "csv_path" in ln:
        print(f"│ {ln.strip()}")
nat(["validate", "--config_file", CTX / "workflow.sandbox.yml"])
note("Validation does not connect to the LLM, so it passes here even though inference.local only exists inside a "
     "sandbox. model_name must equal the MODEL_HANDLE you set with `openshell inference set` (Module 03).")

step(3, "the policy — the course's policykit model (a TEACHING model, not OpenShell)")
policy = pk.load(POLICY)
errs, warns = pk.validate(policy)
for e in errs:
    print(f"✕ {e}")
for w in warns:
    warn(w)
if not errs:
    ok("policykit: valid")
PY = "/usr/bin/python3.12"
cases = [
    {"op": "read", "path": "/sandbox/data/chiller_plant.csv"},
    {"op": "write", "path": "/sandbox/tickets.jsonl"},
    {"op": "write", "path": "/app/workflow.yml"},
    {"op": "run_as", "user": "root"},
    {"op": "http", "host": "inference.local", "port": 443, "binary": PY, "method": "POST", "path": "/v1/chat/completions"},
    {"op": "connect", "host": "api.openai.com", "port": 443, "binary": PY},
    {"op": "mcp", "host": "bms.alto.local", "port": 8443, "binary": PY, "tool": "read_point"},
    {"op": "connect", "host": "169.254.169.254", "port": 80, "binary": PY},
]
rows = []
for a in cases:
    d, why = pk.decide(policy, a)
    rows.append([pk.fmt_action(a), {"allow": "✓ allow", "deny": "✕ deny"}.get(d, "◆ " + d), why])
table(rows, ["action (NAT process in the sandbox)", "decision", "why (policykit)"])
note("bms.alto.local is denied: this sandbox has no BMS entry yet. Exercise 04 writes one that allows only "
     "tools/call on read_point + list_alarms.")
for sev, w, f in pk.harden(policy):
    print(f"{'⚠' if sev in ('HIGH', 'MEDIUM') else '◆'} harden · {sev} · {w}: {f}")

step(4, "the policy — the real OpenShell parser (laptop CLI 0.0.111, no gateway)")
r = openshell_offline(["policy", "set", SB, "--policy", str(CTX / "alto-ops-policy.yaml"), "--wait"], quiet=True)
print("  " + (r.out.strip().splitlines() or [""])[-1])
if parsed_ok(r):
    ok("parsed: OpenShell read the YAML and only failed to reach a gateway (there is none on this laptop)")
else:
    print("✕ the parser rejected the policy:\n" + r.out.strip()[-600:])

step(5, "the tutorial's `sandbox create` line — does the 0.0.111 parser accept it?")
tutorial_args = ["sandbox", "create", "--name", SB, "--from", "./", "--policy", "./alto-ops-policy.yaml",
                 "--upload", "./data:/sandbox/data", "--forward", "8001", "--keep",
                 "--", "nat", "serve", "--config_file", "/app/workflow.yml", "--host", "0.0.0.0", "--port", "8001"]
r = openshell_offline(tutorial_args, quiet=True)
first = next((ln for ln in r.out.splitlines() if ln.startswith("error")), r.out.strip().splitlines()[-1:])
print(f"  {first}")
tut_ok = parsed_ok(r)
if not tut_ok:
    warn("OpenShell 0.0.111 refuses `--upload` together with a trailing `-- <command>`. NemoClaw pins "
         f"{OPENSHELL_PINNED} on the Spark — run `openshell sandbox create --help` there. The split form below "
         "parses on 0.0.111.")
split_create = ["sandbox", "create", "--name", SB, "--from", "./Dockerfile.alto-ops", "--policy",
                "./alto-ops-policy.yaml", "--forward", "8001", "--keep", "--detach",
                "--", "nat", "serve", "--config_file", "/app/workflow.yml", "--host", "0.0.0.0", "--port", "8001"]
split_upload = ["sandbox", "upload", SB, str(CTX / "data"), "/sandbox/data"]   # upload checks the local path
parse_rows = [["tutorial one-liner (--upload + -- nat serve)", "✓ parsed" if tut_ok else "✕ rejected"]]
for label, args in (("create … --detach -- nat serve …", split_create), ("sandbox upload alto-ops ./data …", split_upload),
                    ("forward start --background 8001 alto-ops", ["forward", "start", "--background", "8001", SB]),
                    ("logs alto-ops -n 50 --source sandbox", ["logs", SB, "-n", "50", "--source", "sandbox"])):
    rr = openshell_offline(args, quiet=True)
    parse_rows.append([label, "✓ parsed" if parsed_ok(rr) else "✕ " + rr.out.strip().splitlines()[0][:60]])
table(parse_rows, ["command (OpenShell 0.0.111 parser)", "result"])
note("--from ./Dockerfile.alto-ops names the file: the help says --from takes 'a path to a Dockerfile or directory "
     "containing one', and this Dockerfile is not called `Dockerfile`. --detach (listed by 0.0.111) starts the main "
     "process without attaching, so a lab never blocks on `nat serve`.")

step(6, "on the Spark — copy the context, create, upload, forward, look (changes need 🔓)")
remote = "~/works/alto-ops-claw/alto-ops-sandbox"
if where() != "dry" and apply_enabled():
    for f in files:
        put(CTX / f, f"{remote}/{f}")
else:
    print(f"$ scp -r {rel(CTX)} <spark>:~/works/alto-ops-claw/   [{'DRY' if where() == 'dry' else 'NOT RUN'}]")
sh("openshell inference get", example="Provider: <your provider>\nModel: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8")
change(f"cd {remote} && openshell sandbox create --name {SB} --from ./Dockerfile.alto-ops "
       "--policy ./alto-ops-policy.yaml --forward 8001 --keep --detach "
       "-- nat serve --config_file /app/workflow.yml --host 0.0.0.0 --port 8001",
       example="Building image from ./Dockerfile.alto-ops …\nCreating sandbox alto-ops …\nSandbox alto-ops is ready")
change(f"cd {remote} && openshell sandbox upload {SB} ./data /sandbox/data",
       example="Uploaded ./data → alto-ops:/sandbox/data")
change(f"openshell forward start --background 8001 {SB}", example="Forwarding 127.0.0.1:8001 → alto-ops:8001")
sh(f"openshell sandbox get {SB}", example="Name: alto-ops\nPhase: Ready\nPolicy: alto-ops-policy.yaml (revision 1)")
sh(f"openshell logs {SB} -n 50 --source sandbox",
   example="… OpenShell Sandbox Supervisor success …\n… Applying Landlock filesystem sandbox …\n"
           "… Uvicorn running on http://0.0.0.0:8001 …")
sh("curl -s http://localhost:8001/health", example='{"status":"healthy"}')
note("The /health shape is what `nat serve` answered on this laptop (lab 04-3). For the live log, run "
     "`openshell logs alto-ops --tail --source sandbox` yourself in the ⌨ terminal — a lab never follows a log.")

step(7, "L3.7 — sandboxed code execution: what NAT 1.9 really ships")
probe = subprocess.run([str(NAT_PY), "-c", "import nat.tool.code_execution as m, os; print(os.path.dirname(m.__file__))"],
                       capture_output=True, text=True, env={"PYTHONWARNINGS": "ignore"})
ce_dir = Path(probe.stdout.strip().splitlines()[-1]) if probe.returncode == 0 and probe.stdout.strip() else None
if ce_dir:
    present = sorted(p.name for p in ce_dir.iterdir() if not p.name.startswith("__"))
    print(f"│ nat/tool/code_execution/ → {', '.join(present)}")
    has_local = (ce_dir / "local_sandbox").exists()
    print(f"◆ local_sandbox/ (the tutorial's start_local_sandbox.sh, port 6000): {'present' if has_local else 'NOT in 1.9.0'}")
nat(["validate", "--config_file", CONFIGS / "code_execution.tutorial.yml"])
note("The tutorial's block validates (uri is just a URL) — but in 1.9.0 it would speak the Piston API to :6000.")
bad = RUNS / "code_execution.local.yml"
bad.write_text((CONFIGS / "code_execution.tutorial.yml").read_text(encoding="utf-8").replace(
    '    uri: "http://127.0.0.1:6000"', "    sandbox_type: local"), encoding="utf-8")
nat(["validate", "--config_file", bad])
nat(["validate", "--config_file", CONFIGS / "code_execution.piston.yml"])
why_not = docker_daemon()
if why_not:
    warn(f"{why_not} — no Piston server can run on this laptop right now, so the code_execution tool is config-only "
         "here. Start Docker Desktop, deploy Piston (see NAT's code_execution README), and point `uri` at it.")
else:
    ok("the Docker daemon answers — you can run a Piston server here (see NAT's code_execution README)")
note("Inside a claw, a Piston server is one more network endpoint: it needs its own network_policies entry, and the "
     "agent's code runs THERE, never on the Spark host.")
result("NAT runs inside OpenShell with one YAML change (base_url → inference.local) and a policy with no network entries.")
