#!/usr/bin/env python3
"""Lab 08-1 · Assemble Alto Ops Claw v1: generate the whole bundle and validate every file offline.

Writes .runs/alto-ops-claw-v1/ from the module's configs/ and policies/, plus a copy of the alto_ops NAT
package and an eval dataset computed from the SYNTHETIC chiller CSV. The dataset has 20 rows for the Spark.
The laptop uses the first 3. Then it checks the bundle with the real tools on THIS laptop:
PyYAML, `nat validate` (NAT 1.9.0), the policykit teaching model (validate + the §6.2 harden checklist), and
the OpenShell 0.0.111 CLI as an offline parser. It also cross-checks every network endpoint the workflow names
against the policy. The last step copies the bundle to your Spark. The sandbox create command is printed for
you to run in the ⌨ terminal: it attaches to a long-running `nat serve`, so a lab never runs it in the
foreground. The data upload and the port forward go through change().

No LLM calls. Run: .venv/bin/python week26/08_capstone_alto_ops_claw/labs/lab08_1_assemble.py
"""
import json
import shutil
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml  # noqa: E402

import capkit as ck  # noqa: E402
import policykit as pk  # noqa: E402
from clawkit import (NAT, NAT_PY, ROOT, background, banner, change, free_port, laptop, note, ok,  # noqa: E402
                     openshell_offline, parsed_ok, put, result, sh, step, table, warn)

T0 = time.time()
banner("Lab 08-1 · assemble Alto Ops Claw v1", "the bundle, validated offline · then copied to your Spark (or DRY)")

# ── 1 · generate the bundle ────────────────────────────────────────────────────────────────────────────────
step(1, "generate the bundle → week26/08_capstone_alto_ops_claw/.runs/alto-ops-claw-v1/")
B = ck.BUNDLE
if B.exists():
    shutil.rmtree(B)
(B / "data").mkdir(parents=True)
shutil.copyfile(ck.CONFIGS / "Dockerfile.alto-ops", B / "Dockerfile")
shutil.copyfile(ck.CONFIGS / "workflow.sandbox.yml", B / "workflow.sandbox.yml")
shutil.copyfile(ck.CONFIGS / "eval_config.yml", B / "eval_config.yml")
shutil.copyfile(ck.POLICIES / "prod-policy.yaml", B / "prod-policy.yaml")
pkg = B / "workflows" / "alto_ops"
shutil.copytree(ck.ALTO_OPS, pkg, ignore=shutil.ignore_patterns("*.egg-info", "__pycache__", "*.pyc"))
shutil.copyfile(ck.CSV, B / "data" / "chiller_plant.csv")
shutil.copyfile(ck.COMMON / "bms_mcp_server.py", B / "bms_mcp_server.py")     # the mock BMS, next to data/
shutil.copyfile(ck.CONFIGS / "bms_lan.py", B / "bms_lan.py")                  # …bound to the LAN for the sandbox
rows = ck.eval_rows()
ck.write_jsonl(B / "data" / "alto_ops_eval.jsonl", rows)
ck.write_jsonl(B / "data" / "alto_ops_eval.laptop.jsonl", rows[:ck.LAPTOP_ROWS])
man = ck.manifest(B)
(B / "MANIFEST.json").write_text(json.dumps({"bundle": "alto-ops-claw-v1", "files": man}, indent=1), encoding="utf-8")
table([[m["path"], m["bytes"], m["sha256"][:12]] for m in man], ["file", "bytes", "sha256"])
ok(f"{len(man)} files + MANIFEST.json · eval dataset {len(rows)} rows (Spark) · first {ck.LAPTOP_ROWS} for the laptop")
k6, k24 = ck.kpi(6), ck.kpi(24)
note(f"answers come from the CSV with chiller_kpi's own arithmetic: 6 h → {k6['kw_per_rt']} kW/RT {k6['status']}, "
     f"24 h → {k24['kw_per_rt']} kW/RT {k24['status']} (SYNTHETIC data)")
table([[r["id"], r["question"][:58], r["answer"][:50]] for r in rows[:5]] + [["…", f"{len(rows) - 5} more", ""]],
      ["id", "question", "reference answer"])

# ── 2 · YAML and Dockerfile ────────────────────────────────────────────────────────────────────────────────
step(2, "every YAML file parses · the Dockerfile has the Lab 3.8 essentials")
good = True
for p in [B / "workflow.sandbox.yml", B / "eval_config.yml", B / "prod-policy.yaml", ck.CONFIGS / "eval_config.laptop.yml"]:
    try:
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        ok(f"{p.name}: {len(d)} top-level keys ({', '.join(d)})")
    except yaml.YAMLError as e:
        print(f"✕ {p.name}: {e}")
        good = False
df = (B / "Dockerfile").read_text(encoding="utf-8")
checks = [
    ("non-root user", "USER 1500" in df and "USER root" not in df),
    ("NAT installed at build time (no runtime pip)", "nvidia-nat[" in df),
    ("the chiller tool package is copied and installed", "COPY workflows/alto_ops" in df and "-e /app/alto_ops" in df),
    ("the sandbox workflow becomes /app/workflow.yml", "COPY workflow.sandbox.yml /app/workflow.yml" in df),
    ("WORKDIR /sandbox (the data upload target)", "WORKDIR /sandbox" in df),
    ("chiller_kpi is registered by the package", "chiller_tool" in (pkg / "src/alto_ops/register.py").read_text()),
]
for what, cond in checks:
    print(("✓ " if cond else "✕ ") + what)
    good &= cond

# ── 3 · nat validate ───────────────────────────────────────────────────────────────────────────────────────
step(3, "nat validate — the real NAT 1.9.0 schema check, on this laptop")
nat_ok = True
for cfg in [B / "workflow.sandbox.yml", B / "eval_config.yml", ck.CONFIGS / "eval_config.laptop.yml"]:
    rel = cfg.relative_to(ROOT)
    r = laptop([NAT, "validate", "--config_file", rel], quiet=True, cwd=ROOT, timeout=120,
               show=f"nat validate --config_file {rel}")
    valid = "Configuration file is valid" in r.out
    summary = [ln.strip() for ln in r.out.splitlines() if ln.strip().startswith(("Workflow Type", "Number of Functions",
                                                                               "Number of Function Groups", "Number of LLMs"))]
    print(("✓ valid · " if valid else "✕ INVALID · ") + " · ".join(summary))
    if not valid:
        print("\n".join(r.out.splitlines()[-6:]))
    good &= valid
    nat_ok &= valid
note("`nat validate` checks the schema and the registered component types (chiller_kpi, request_setpoint_change, "
     "mcp_client, phoenix, otelcollector). It does not connect to inference.local, the BMS or Phoenix.")

# the bundle's mock BMS launcher: start it from the bundle root (on 127.0.0.1 here), read one point over MCP
bport = free_port(8443)
with background([NAT_PY, B / "bms_lan.py"], ready_url=f"http://localhost:{bport}/mcp", log=ck.RUNS / "bms_lan.log",
                env={"BMS_MCP_PORT": str(bport), "BMS_MCP_HOST": "127.0.0.1", "PYTHONWARNINGS": "ignore"}, cwd=B,
                timeout=60, show=f"BMS_MCP_PORT={bport} BMS_MCP_HOST=127.0.0.1 python bms_lan.py   # from the bundle"):
    rb = laptop([NAT, "mcp", "client", "tool", "call", "read_point", "--url", f"http://localhost:{bport}/mcp",
                 "--json-args", '{"point": "PLANT.KW_PER_RT"}'], quiet=True, env={"PYTHONWARNINGS": "ignore"},
                timeout=90, show=f"nat mcp client tool call read_point --url http://localhost:{bport}/mcp "
                                 "--json-args '{\"point\": \"PLANT.KW_PER_RT\"}'")
    reading = next((ln.strip() for ln in rb.out.splitlines() if ln.strip().startswith("PLANT.")), "")
    print(("✓ bundle BMS answers: " if reading else "✕ bundle BMS did not answer: ") + (reading or rb.out[-200:]))
    good &= bool(reading)

# ── 4 · workflow ↔ policy cross-check ──────────────────────────────────────────────────────────────────────
step(4, "every network endpoint the sandbox workflow names — is the prod policy allowing exactly that?")
wf = yaml.safe_load((B / "workflow.sandbox.yml").read_text(encoding="utf-8"))
prod = pk.load(str(B / "prod-policy.yaml"))
PY = "/usr/bin/python3.12"
need = []
for name, ex in wf["general"]["telemetry"]["tracing"].items():
    ep = ex.get("endpoint", "")
    if ep.startswith("http"):
        hostport, path = ep.split("//", 1)[1].split("/", 1)
        h, p = hostport.split(":")
        need.append((f"exporter {name}", {"op": "http", "host": h, "port": int(p), "binary": PY, "method": "POST",
                                          "path": "/" + path}))
bms_url = wf["function_groups"]["bms"]["server"]["url"]
bh, bp = bms_url.split("//", 1)[1].split("/", 1)[0].split(":")
need += [("mcp_client bms · tools/list", {"op": "mcp", "host": bh, "port": int(bp), "binary": PY, "method": "tools/list"}),
         ("mcp_client bms · read_point", {"op": "mcp", "host": bh, "port": int(bp), "binary": PY, "tool": "read_point"}),
         ("mcp_client bms · write_setpoint", {"op": "mcp", "host": bh, "port": int(bp), "binary": PY,
                                              "tool": "write_setpoint"}),
         ("llm routed", {"op": "http", "host": "inference.local", "port": 443, "binary": PY, "method": "POST",
                         "path": "/v1/chat/completions"})]
cross = []
for what, act in need:
    d, why = pk.decide(prod, act)
    want = "deny" if act.get("tool") == "write_setpoint" else ("inspect_for_inference" if act["host"] == "inference.local"
                                                                else "allow")
    cross.append([what, pk.fmt_action(act), ("✓ " if d == want else "✕ ") + d, why[:60]])
    good &= d == want
table(cross, ["workflow names", "action", "policykit", "why"])
ok("the workflow needs 3 network hosts plus inference.local, and the policy opens exactly those — write_setpoint "
   "stays denied")

# ── 5 · policykit: Lab 6.1 as printed vs the capstone policy ───────────────────────────────────────────────
step(5, "policykit validate + harden (§6.2) — Lab 6.1 as printed vs the capstone policy (a TEACHING MODEL)")
for label, pol in [("Lab 6.1 as printed", ck.lab61_policy(prod)), ("capstone prod-policy.yaml", prod)]:
    errs, warns = pk.validate(pol)
    findings = [f for f in pk.harden(pol, tier="restricted") if f[0] != "OK"]
    print(f"▣ {label}: {len(errs)} errors · {len(warns)} warnings · {len(findings)} non-OK findings")
    for sev, where, what in findings:
        print(f"  ⚠ {sev:6s} {where}: {what}")
    if label.startswith("capstone"):
        good &= not errs and not findings
ok("the capstone policy: landlock hard_requirement · every endpoint L7 + enforce · private hosts pinned to /32 · "
   "no inference-provider hosts")
note("policykit is the course's model, not OpenShell. The real check runs on the Spark when the sandbox is created "
     "with this file (Landlock hard_requirement refuses to start if it cannot apply).")

# ── 6 · the real OpenShell parser ──────────────────────────────────────────────────────────────────────────
step(6, "the OpenShell 0.0.111 CLI as an offline parser (no gateway behind it)")
rp = r = openshell_offline(["policy", "set", ck.SANDBOX, "--policy", str(B / "prod-policy.yaml"), "--wait"])
print(("✓ " if parsed_ok(r) else "✕ ") + "prod-policy.yaml parsed by the real CLI — it only failed to reach the "
      "gateway" if parsed_ok(r) else r.out[-400:])
good &= parsed_ok(r)
lab38 = ["sandbox", "create", "--name", ck.SANDBOX, "--from", str(B), "--policy", str(B / "prod-policy.yaml"),
         "--upload", str(B / "data") + ":/sandbox/data", "--forward", str(free_port(8001)), "--keep", "--",
         "nat", "serve", "--config_file", "/app/workflow.yml", "--host", "0.0.0.0", "--port", "8001"]
r38 = openshell_offline(lab38)
first_err = next((ln.strip() for ln in r38.out.splitlines() if ln.strip().startswith("error:")), "")
if parsed_ok(r38):
    ok("Lab 3.8's one-line create (with --upload AND a trailing command) parsed")
else:
    warn(f"Lab 3.8's one-line create, on the 0.0.111 parser: {first_err or r38.out.strip()[-200:]}")
    note("So the course splits it in two: create with the command, then `openshell sandbox upload`. chiller_kpi "
         "opens the CSV on every call, so uploading after start works. Check `sandbox create --help` on your "
         "0.0.116: it may accept the one-liner.")
fp = free_port(8001)          # the CLI checks that the LOCAL forward port is free before it dials the gateway
if fp != 8001:
    note(f"port 8001 is busy on this laptop (another lab?) — the offline parse uses {fp}; the Spark command keeps 8001")
two = [["sandbox", "create", "--name", ck.SANDBOX, "--from", str(B), "--policy", str(B / "prod-policy.yaml"),
        "--forward", str(fp), "--keep", "--", "nat", "serve", "--config_file", "/app/workflow.yml",
        "--host", "0.0.0.0", "--port", "8001"],
       ["sandbox", "upload", ck.SANDBOX, str(B / "data"), "/sandbox/data"],
       ["forward", "start", "--background", str(fp), ck.SANDBOX]]
for argv in two:
    rr = openshell_offline(argv)
    print(("✓ parsed · " if parsed_ok(rr) else "✕ rejected · ") + "openshell " + " ".join(argv[:3]) + " …")
    good &= parsed_ok(rr)

# ── 7 · to the Spark ───────────────────────────────────────────────────────────────────────────────────────
step(7, "copy the bundle to your Spark, then create the sandbox from it")
tgz = ck.RUNS / "alto-ops-claw-v1.tgz"
with tarfile.open(tgz, "w:gz") as tf:
    tf.add(B, arcname="alto-ops-claw-v1")
ok(f"{tgz.name}: {tgz.stat().st_size / 1024:.0f} KiB")
put(tgz, "~/alto-ops-claw-v1.tgz")
sh(f"mkdir -p {ck.REMOTE}/evidence && tar -xzf ~/alto-ops-claw-v1.tgz -C ~ && ls {ck.REMOTE}",
   example="Dockerfile\nMANIFEST.json\ndata\nevidence\neval_config.yml\nprod-policy.yaml\nworkflow.sandbox.yml\nworkflows")
print("→ run this one yourself in the ⌨ terminal (# on: spark). It builds the image, then attaches to nat serve:")
print(f"  cd {ck.REMOTE} && openshell sandbox create --name {ck.SANDBOX} --from ./ --policy ./prod-policy.yaml \\")
print("    --forward 8001 --keep -- nat serve --config_file /app/workflow.yml --host 0.0.0.0 --port 8001")
EX_LIST = f"""NAME        PHASE
{ck.SANDBOX}    Ready"""
change(f"openshell sandbox upload {ck.SANDBOX} {ck.REMOTE}/data /sandbox/data",
       preview=ck.CMD["sandbox_list"], example=EX_LIST)
change(f"openshell forward start --background 8001 {ck.SANDBOX}")

(ck.RUNS / "assemble.json").write_text(json.dumps({
    "date": time.strftime("%Y-%m-%d %H:%M"), "ok": bool(good), "files": len(man), "eval_rows": len(rows),
    "dockerfile_checks": {w: bool(c) for w, c in checks}, "nat_validate_ok": bool(nat_ok),
    "cross_check": [[c[0], c[2]] for c in cross], "harden_findings": len(findings),
    "parser_prod_policy": parsed_ok(rp), "manifest_sha256": ck.sha256(B / "MANIFEST.json")}, indent=1), encoding="utf-8")
print()
if good:
    result(f"Bundle assembled and validated offline in {time.time() - T0:.0f}s. Next: lab 08-2 proves the write boundary.")
else:
    print("✕ one or more checks failed — read the ✕ lines above")
    sys.exit(1)
