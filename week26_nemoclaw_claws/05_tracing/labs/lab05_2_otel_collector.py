#!/usr/bin/env python3
"""Lab 05-2 · Vendor-neutral traces: an OTel collector to a file, and the sandbox rule it needs.

Research tutorial Lab 4.2 (L4.2). On THIS laptop, for real:
  1. write the research tutorial's otelcollectorconfig.yaml (verbatim) and the `otelcollector` exporter block;
  2. `nat validate` the Alto Ops config that uses it;
  3. IF the Docker daemon answers: run otel/opentelemetry-collector-contrib:0.128.0 in the background on a free
     port, send one Alto Ops run to it, and count the spans it wrote to otellogs/llm_spans.json. If Docker is
     not running, the lab says so and stops that part — it never prints a collector file it did not receive;
  4. the sandbox-policy entry a SANDBOXED NAT needs to reach the collector (protocol rest, POST /v1/traces,
     audit first), checked with the course's policykit model and parsed by the real OpenShell CLI offline.

Run: .venv/bin/python week26/05_tracing/labs/lab05_2_otel_collector.py
"""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import policykit as pk  # noqa: E402
from clawkit import (NAT, ROOT, background, banner, change, free_port, laptop, note, ok,  # noqa: E402
                     openshell_offline, parsed_ok, result, sandbox, sh, step, table, warn)
from tracekit import CONFIGS, RUNS  # noqa: E402

QUESTION = "Plant status last 6 hours?"
IMAGE = "otel/opentelemetry-collector-contrib:0.128.0"
SB = sandbox("alto-ops")                      # the sandboxed NAT claw from research Lab 3.8
OTEL = RUNS / "otel"
LOGS = OTEL / "otellogs"
OTEL.mkdir(parents=True, exist_ok=True)

banner("Lab 05-2 · OTel collector", "config + nat validate for real · collector only if Docker answers · policy offline")

step(1, "the collector config — the research tutorial's, verbatim")
col_cfg = OTEL / "otelcollectorconfig.yaml"
shutil.copyfile(CONFIGS / "otelcollectorconfig.yaml", col_cfg)
print(col_cfg.read_text(encoding="utf-8").rstrip())
ok(f"written → {col_cfg.relative_to(ROOT)} (receiver: OTLP/HTTP on :4318 · exporter: file /otellogs/llm_spans.json)")

step(2, "the NAT side: an `otelcollector` exporter next to the file backup")
port = free_port(4318)
if port != 4318:
    note(f"4318 is taken on this laptop → this lab uses {port} (the Spark keeps 4318)")
nat_cfg_text = (CONFIGS / "workflow.otel.yml").read_text(encoding="utf-8")
nat_cfg_text = nat_cfg_text.replace("endpoint: http://0.0.0.0:4318/v1/traces", f"endpoint: http://localhost:{port}/v1/traces")
nat_cfg_text = nat_cfg_text.replace("week26/05_tracing/.runs/traces/alto_ops_trace.jsonl",
                                    "week26/05_tracing/.runs/traces/lab05_2_trace.jsonl")
nat_cfg = OTEL / "workflow.otel.yml"
nat_cfg.write_text(nat_cfg_text, encoding="utf-8")
block = nat_cfg_text.split("    tracing:\n", 1)[1].split("      file_backup:", 1)[0]
print("    tracing:\n" + block.rstrip())
note("the research tutorial's block says endpoint http://0.0.0.0:4318/v1/traces; this laptop copy sends to "
     f"localhost:{port} (the port the collector publishes here). Inside a sandbox you name the collector's host.")
r = laptop([NAT, "validate", "--config_file", nat_cfg], quiet=True, timeout=120,
           show="nat validate --config_file week26/05_tracing/.runs/otel/workflow.otel.yml")
print(("✓ " if r.ok else "✕ ") + f"nat validate → exit {r.code}")
note("NAT 1.9 sets the OTel resource attribute service.name = `project` (nat/plugins/opentelemetry/register.py). "
     "That is why Phoenix files otelcollector traces under 'default' — the quirk the research tutorial cites.")

step(3, "is the Docker daemon up on this laptop?")
docker = shutil.which("docker")
daemon = ""
if docker:
    r = laptop([docker, "info", "--format", "{{.ServerVersion}}"], quiet=True, timeout=20,
               show="docker info --format '{{.ServerVersion}}'")
    daemon = r.out.strip().splitlines()[-1] if r.ok and r.out.strip() else ""
collector_spans = None
if not daemon:
    warn("the Docker daemon is not running here (client only, or not installed) → the collector is NOT started, "
         "and no collector output is shown. Start Docker Desktop and run this lab again to see it for real.")
    print("→ the command this step would run (in the background, stopped when the lab ends):")
    print(f"$ docker run --rm -v $(pwd)/otelcollectorconfig.yaml:/etc/otelcol-contrib/config.yaml "
          f"-p {port}:4318 -v $(pwd)/otellogs:/otellogs/ {IMAGE}   [not run]")
else:
    ok(f"Docker daemon {daemon} → starting the collector for real")
    LOGS.mkdir(parents=True, exist_ok=True)
    out_file = LOGS / "llm_spans.json"
    if out_file.exists():
        out_file.unlink()
    argv = [docker, "run", "--rm", "--name", f"otel-nat-{port}", "-v", f"{col_cfg}:/etc/otelcol-contrib/config.yaml",
            "-p", f"{port}:4318", "-v", f"{LOGS}:/otellogs/", IMAGE]
    try:
        with background(argv, ready_url=f"http://localhost:{port}/v1/traces", log=RUNS / "lab05_2_collector.log",
                        timeout=300):
            r = laptop([NAT, "run", "--config_file", nat_cfg, "--input", QUESTION], cwd=ROOT, quiet=True,
                       timeout=400, show=f'nat run --config_file week26/05_tracing/.runs/otel/workflow.otel.yml '
                                         f'--input "{QUESTION}"')
            print(("✓ " if r.ok else "✕ ") + f"nat run exit {r.code} (LAPTOP STAND-IN model)")
            time.sleep(7)                    # the collector's file exporter and NAT's batch flush (5 s default)
    except RuntimeError as e:
        warn(f"the collector did not come up: {str(e)[:300]}")
    names, services = [], set()
    if out_file.is_file():
        for line in out_file.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            for rs in doc.get("resourceSpans", []):
                for a in rs.get("resource", {}).get("attributes", []):
                    if a.get("key") == "service.name":
                        services.add(a.get("value", {}).get("stringValue", "?"))
                for ss in rs.get("scopeSpans", []):
                    names += [s.get("name", "?") for s in ss.get("spans", [])]
    collector_spans = len(names)
    if names:
        ok(f"the collector wrote {len(names)} span(s) to otellogs/llm_spans.json · service.name = "
           f"{', '.join(sorted(services))}")
        table([[n, names.count(n)] for n in sorted(set(names))], ["span name (from the collector file)", "count"])
    else:
        warn("the collector file has no spans — see .runs/lab05_2_collector.log (a short `nat run` can exit before "
             "the batch exporter flushes; lab 05-5 uses a long-lived `nat serve`)")

note("on the Spark the collector runs next to NAT and vLLM (same image tag and mounts, detached). It is a change:")
change(f"docker run -d -v $(pwd)/otelcollectorconfig.yaml:/etc/otelcol-contrib/config.yaml \\\n"
       f"  -p 4318:4318 -v $(pwd)/otellogs:/otellogs/ {IMAGE}",
       preview=f"docker ps --filter ancestor={IMAGE} --format '{{{{.ID}}}} {{{{.Status}}}}'",
       example="3f9c2a1b7d4e Up 2 minutes")
sh("ls -l otellogs/ && head -c 300 otellogs/llm_spans.json", example=(
    "-rw-r--r-- 1 root root 48213 <date> llm_spans.json\n"
    '{"resourceSpans":[{"resource":{"attributes":[{"key":"service.name","value":{"stringValue":"alto-ops-claw"}}…'),
   timeout=30)

step(4, f"a SANDBOXED NAT ({SB}) exporting to the collector — the policy entry it needs")
OTEL_HOST = "otel.alto.local"
PY = "/usr/bin/python3.12"


def policy_with(endpoint: dict) -> dict:
    return pk.build_policy(read_only=["/usr", "/lib", "/etc"], read_write=["/tmp"],
                           groups={"otel_collector": pk.group("otel_collector", [endpoint], [PY])})


audit = {"host": OTEL_HOST, "port": 4318, "protocol": "rest", "enforcement": "audit", "access": "read-write"}
enforce_ro = {**audit, "enforcement": "enforce", "access": "read-only"}
enforce_rule = {"host": OTEL_HOST, "port": 4318, "protocol": "rest", "enforcement": "enforce",
                "rules": [{"allow": {"method": "POST", "path": "/v1/traces"}}]}
print("network_policies:" + pk.dump(policy_with(audit)).split("network_policies:", 1)[1].rstrip())
for name, ep in (("audit · read-write (start here)", audit), ("enforce · read-only (the 403 bug)", enforce_ro),
                 ("enforce · rule POST /v1/traces", enforce_rule)):
    errs, _ = pk.validate(policy_with(ep))
    if errs:
        print(f"✕ {name}: {errs[0]}")
rows = []
POST = {"op": "http", "host": OTEL_HOST, "port": 4318, "binary": PY, "method": "POST", "path": "/v1/traces"}
for name, ep in (("audit · read-write", audit), ("enforce · read-only", enforce_ro),
                 ("enforce · allow POST /v1/traces", enforce_rule)):
    d, why = pk.decide(policy_with(ep), POST)
    rows.append([name, "POST /v1/traces", {"allow": "✓ allow", "deny": "✕ deny"}.get(d, d), why])
d, why = pk.decide(policy_with(audit), {**POST, "op": "connect", "host": "0.0.0.0", "port": 4318})
rows.append(["any", "connect 0.0.0.0:4318", "✕ deny" if d == "deny" else d, why])
d, why = pk.decide(policy_with(audit), {**POST, "binary": "/usr/bin/curl"})
rows.append(["audit · read-write", "curl POST /v1/traces", "✕ deny" if d == "deny" else d, why])
table(rows, ["endpoint entry", "request (python3.12)", "policykit", "why"])
note("policykit is the course's TEACHING MODEL, not OpenShell. Two lessons: (1) OTLP/HTTP is a POST, so a "
     "read-only preset blocks it with 403 once you switch to enforce; (2) inside a sandbox the exporter must name "
     "the collector host — 0.0.0.0 (the research tutorial's laptop endpoint) is always blocked as SSRF.")

spec = f"{OTEL_HOST}:4318:read-write:rest:audit"
r = openshell_offline(["policy", "update", SB, "--add-endpoint", spec, "--binary", PY, "--dry-run"])
print(("✓ " if parsed_ok(r) else "✕ ") + f"openshell 0.0.111 parsed `--add-endpoint {spec} --binary {PY} --dry-run`"
      + ("" if parsed_ok(r) else f" → {r.out.strip()[:200]}"))
change(f"openshell policy update {SB} --add-endpoint {spec} --binary {PY} --wait",
       preview=f"openshell policy update {SB} --add-endpoint {spec} --binary {PY} --dry-run",
       example=f"(merged policy preview: network_policies gains an endpoint {OTEL_HOST}:4318 rest audit read-write "
               f"for {PY})")
result(("Collector run: " + (f"{collector_spans} span(s) received." if collector_spans is not None else
                             "skipped — Docker daemon not running (nothing faked).")) +
       " Sandbox rule: rest + POST /v1/traces, audit first, then enforce.")
