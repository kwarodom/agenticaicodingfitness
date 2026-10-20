# ▶ Reef Lab 05 — Tracing and observability: agent, policy and harness planes

> Part of Week 26 · NemoClaw on DGX Spark, beginner to expert. You type the commands, you see the real output. Laptop labs (the NAT and OpenShell CLIs, the policy model) run for real everywhere. Spark labs also run in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's docs or playbooks, or a clearly marked EXAMPLE.

**What you'll actually do**
- Learn the three telemetry planes of a claw, and which question each one answers.
- Trace a real Alto Ops Claw run with NAT 1.9's `file` exporter, and parse it into LLM spans, tool spans, tokens and durations.
- Add Phoenix as a second exporter when it is running, and see why the laptop run never pretends it is.
- Write the OTel-collector config and the sandbox policy entry a sandboxed NAT needs to reach it.
- Register Langfuse keys as an OpenShell credential handle, and scan Hermes config for leaked keys.
- Read OpenShell's policy decisions like a trace, then join all three planes for one request.

**Time** ~75 min · **Difficulty** intermediate · **Hardware** laptop (NAT + Ollama, for real) · 1 DGX Spark optional (DRY without)

**Sources:** the course's research tutorial *NemoClaw on DGX Spark — Beginner to Expert* (Part 4, labs L4.1–L4.5), which cites [NAT observe](https://docs.nvidia.com/nemo/agent-toolkit/latest/run-workflows/observe/observe.html) · [NAT profiler](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/profiler.html) · [Classmethod — NAT on DGX Spark](https://dev.classmethod.jp/en/articles/dgx-spark-nemo-agent-toolkit-local-intro/) · [NemoClaw Hermes quickstart](https://docs.nvidia.com/nemoclaw/user-guide/hermes/get-started/quickstart) · [NemoClaw network policies reference](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/reference/network-policies) · [OpenShell playbook](https://build.nvidia.com/playbooks/openshell/instructions) · [OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies) · [OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| The NAT laptop venv | `week26/.venv-nat/bin/nat --version` → 1.9.0 | every agent trace in this module comes from a real NAT run |
| Laptop Ollama with `nemotron-3-nano:latest` | `curl -s localhost:11434/v1/models` | the LAPTOP STAND-IN model behind Alto Ops Claw (about 10–20 s per LLM call) |
| The OpenShell laptop CLI | `week26/.venv-openshell/bin/openshell --version` → 0.0.111 | parses the policy-plane commands offline |
| Docker daemon (optional) | `docker info` | only for Phoenix and the OTel collector on the laptop; without it those parts are skipped, never faked |
| A DGX Spark (optional) | `ssh -o BatchMode=yes <spark> true` | the sandboxed claw's policy logs; without one they run DRY |

Module 04 built Alto Ops Claw: a NAT `tool_calling_agent` with a `chiller_kpi` tool. This module watches it work. Every lab writes its files to `week26/05_tracing/.runs/`.

```bash
# on: laptop
week26/.venv-nat/bin/nat --version
curl -s localhost:11434/v1/models | head -c 200
docker info --format '{{.ServerVersion}}'
```

✓ Checkpoint: NAT prints 1.9.0 and Ollama lists `nemotron-3-nano:latest`. You know whether your Docker daemon is running.

## 1 · Three planes, three questions

A claw produces three kinds of telemetry. They come from different processes and answer different questions:

| Plane | Source | Answers |
|---|---|---|
| **Agent traces** | NAT telemetry exporters (Phoenix, OTel collector, Langfuse, file …), the Hermes Langfuse plugin | What did the model decide, which tools ran, how many tokens, how long |
| **Policy / inference logs** | OpenShell supervisor and gateway: `openshell logs`, `openshell term`, OCSF findings | What the agent tried to reach, and whether it was allowed, denied or inspected; did Landlock apply |
| **Harness logs** | `nemoclaw <sandbox> logs --follow`, `/tmp/gateway.log` inside OpenClaw | Channel events, pairing, crashes |

The agent plane tells you what the agent *meant* to do. The policy plane tells you what it *tried* to do on the network. You need both, because a prompt-injected agent may trace a harmless story while its egress tells the truth.

NAT's observability runs off the hot path. An `IntermediateStepManager` publishes `IntermediateStep` events (function boundaries, LLM calls, tool calls) to a reactive stream. Exporters consume that stream asynchronously, and several can run at once. The research tutorial cites the NAT observe guide for this design. You will see one side effect of "asynchronously" in lab 05-1.

✓ Checkpoint: for "the agent called api.open-meteo.com", name the plane that proves it (policy) and the plane that shows why (agent).

## 2 · L4.1 — Agent traces: the file exporter, and Phoenix when it is there

Start with what NAT 1.9 has installed. Exporter names are the `_type` values you write in YAML:

```bash
# on: laptop
week26/.venv-nat/bin/nat info components -t tracing
```

**Expected output** (captured on this Mac, lab 05-1 step 1)

```
│ component_name (_type)  package                   version
│ ──────────────────────  ────────────────────────  ───────
│ phoenix                 nvidia-nat-phoenix        1.9.0  
│ file                    nvidia-nat-core           1.9.0  
│ langfuse                nvidia-nat-opentelemetry  1.9.0  
│ langsmith               nvidia-nat-opentelemetry  1.9.0  
│ otelcollector           nvidia-nat-opentelemetry  1.9.0  
│ patronus                nvidia-nat-opentelemetry  1.9.0  
│ galileo                 nvidia-nat-opentelemetry  1.9.0  
│ mlflow                  nvidia-nat-opentelemetry  1.9.0  
│ arize_ax                nvidia-nat-opentelemetry  1.9.0  
✓ 9 tracing exporters registered: phoenix, file, langfuse, langsmith, otelcollector, patronus, galileo, mlflow, arize_ax
```

The research tutorial's telemetry block adds a `phoenix` exporter and a `file_backup` exporter whose keys it leaves as `# path etc.`. NAT 1.9.0 is strict about that. Its `file` tracing exporter has two **required** fields, `output_path` and `project`. So the tutorial's block fails validation with a vague message:

**Expected output** (captured on this Mac, lab 05-1 step 2)

```
│ field            default                    
│ ───────────────  ───────────────────────────
│ type             'unknown'                  
│ output_path      REQUIRED                   
│ project          REQUIRED                   
│ mode             <FileMode.APPEND: 'append'>
│ enable_rolling   False                      
│ max_file_size    10485760                   
│ max_files        5                          
│ cleanup_on_init  False                      
◆ required in 1.9.0: output_path, project — the research tutorial's `file_backup: {_type: file}` has neither (its comment says only `# path etc.`)
$ nat validate --config_file week26/05_tracing/.runs/lab05_1_tutorial_block.yml   [this laptop]
✕ research tutorial's block → exit 1 · Invalid configuration: general: Field required; general: Field required
$ nat validate --config_file week26/05_tracing/configs/workflow.traced.yml   [this laptop]
✓ course block (output_path + project + mode) → exit 0
```

The course's block lives in `week26/05_tracing/configs/workflow.traced.yml`. It is the Module 04 laptop config plus this `general` section:

```yaml
general:
  telemetry:
    logging:
      console: { _type: console, level: WARN }
      file:    { _type: file, path: week26/05_tracing/.runs/alto_ops.log, level: DEBUG }
    tracing:
      file_backup:
        _type: file
        output_path: week26/05_tracing/.runs/traces/alto_ops_trace.jsonl
        project: alto-ops-claw
        mode: overwrite          # 1.9 default is append: one file would collect every run
```

`configs/workflow.phoenix.yml` adds the research tutorial's Phoenix exporter next to it (`endpoint: http://localhost:6006/v1/traces`, `project: alto-ops-claw`). Two exporters under `tracing:` run at the same time. Lab 05-1 uses that file **only if something answers on localhost:6006**. On this Mac nothing did, so Phoenix was skipped and no Phoenix trace appears anywhere in this module. To run Phoenix yourself (laptop with Docker, or the Spark):

```bash
# on: laptop
docker run -d --name phoenix-nat -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest
week26/.venv-nat/bin/nat run --config_file week26/05_tracing/configs/workflow.phoenix.yml --input "Plant status last 6 hours?"
# open http://localhost:6006 → project alto-ops-claw
```

Now trace one real run. The file exporter is the path that always works:

```bash
# on: laptop
week26/.venv-nat/bin/nat run --config_file week26/05_tracing/configs/workflow.traced.yml --input "Plant status last 6 hours?"
```

The file is **raw**. Each line is one `IntermediateStep` event, not a finished OpenTelemetry span. Most lines are `LLM_NEW_TOKEN` (one per streamed token). A span is a `*_START` and a `*_END` that share one UUID. The course's `tracekit.py` pairs them:

**Expected output** (captured on this Mac, lab 05-1 steps 4–5, excerpt · LAPTOP STAND-IN model)

```
✓ nat run exit 0 in 15.2s (LAPTOP STAND-IN: nemotron-3-nano on Ollama, not vLLM on a GB10)
│ event_type      count
│ ──────────────  ─────
│ LLM_NEW_TOKEN   429  
│ FUNCTION_START  11   
│ FUNCTION_END    7    
│ LLM_START       2    
│ LLM_END         2    
│ WORKFLOW_START  1    
│ TOOL_START      1    
│ TOOL_END        1    
│ kind      name                    duration       tokens (prompt + completion)
│ ────────  ──────────────────────  ─────────────  ────────────────────────────
│ WORKFLOW  tool_calling_agent      OPEN (no END)                              
│ FUNCTION  <workflow>              OPEN (no END)                              
│ FUNCTION  LangGraph               OPEN (no END)                              
│ LLM       nemotron-3-nano:latest  3.590 s        359 + 144 = 503             
│ TOOL      chiller_kpi             0.009 s                                    
│ FUNCTION  agent                   OPEN (no END)                              
│ FUNCTION  RunnableSequence        OPEN (no END)                              
│ LLM       nemotron-3-nano:latest  8.088 s        439 + 326 = 765             
│ measure (LAPTOP STAND-IN)             value                      
│ ────────────────────────────────────  ───────────────────────────
│ LLM calls started (LLM_START)         2                          
│ LLM spans closed (START + END)        2                          
│ TOOL spans                            1 (chiller_kpi)            
│ tokens (prompt / completion / total)  798 / 470 / 1268           
│ time in LLM spans                     11.68 s                    
│ time in TOOL spans                    0.009 s                    
│ workflow span                         no WORKFLOW_END in the file
│ open spans (START, no END)            5                          
⚠ 5 span(s) have no END line. NAT 1.9's exporter stop() does not wait for background export tasks, so a short-lived `nat run` can exit before the last events are written. The model did answer; the TAIL of the trace was lost. Lab 05-5 uses `nat serve` (a long-lived process) and compares.
```

Read it like a story. The first LLM call (359 prompt tokens) decides to call `chiller_kpi`. The tool takes 9 ms. The second LLM call writes the answer. The model's time is almost all of the run.

The five open spans are a real NAT 1.9 behaviour, not a parser bug. `nat run` exits as soon as it has the answer. In NAT 1.9 the exporter's `stop()` does not wait for its background write tasks, so the last events can be lost. Expect some variation: on another run on this Mac, even the second `LLM_END` was missing, so the file showed 1 closed LLM span for 2 real calls. **Never count LLM calls from a short-lived `nat run`'s trace file.** Count `LLM_START`, or use a long-lived server (Section 6).

| Research tutorial (NAT 1.8 docs) | NAT 1.9.0 on this laptop | What to do |
|---|---|---|
| `file_backup: {_type: file}` + `# path etc.` | `output_path` and `project` are required; `mode` defaults to `append` | set both, plus `mode: overwrite` for one-run files |
| file exporter = "a backup of the trace" | a RAW exporter: IntermediateStep events, mostly `LLM_NEW_TOKEN` | pair START/END by UUID to get spans |
| (not mentioned) | a short `nat run` can lose the trace's tail | count `LLM_START`, or trace a `nat serve` |
| exporter list: Phoenix, OTel collector, Langfuse, Weave, file | 9 registered here: phoenix, file, langfuse, langsmith, otelcollector, patronus, galileo, mlflow, arize_ax (no weave in this install) | check `nat info components -t tracing` on your unit |

✓ Checkpoint: you ran lab 05-1 and can say how many LLM calls one Alto Ops question makes (2), and why the trace file had open spans.

## 3 · L4.2 — Vendor-neutral: an OTel collector, and the sandbox rule it needs

An OpenTelemetry collector is a normal OTLP endpoint. NAT, OpenClaw diagnostics and anything else can send to it, and it forwards to files, Phoenix or a SaaS backend. The research tutorial's collector config writes every trace to a file:

```bash
# on: laptop
cat week26/05_tracing/configs/otelcollectorconfig.yaml
cd week26/05_tracing/.runs/otel
docker run -d -v $(pwd)/otelcollectorconfig.yaml:/etc/otelcol-contrib/config.yaml \
  -p 4318:4318 -v $(pwd)/otellogs:/otellogs/ otel/opentelemetry-collector-contrib:0.128.0
```

```yaml
receivers:
  otlp:
    protocols:
      http: { endpoint: 0.0.0.0:4318 }
exporters:
  file:
    path: /otellogs/llm_spans.json
service:
  pipelines:
    traces: { receivers: [otlp], exporters: [file] }
```

On the NAT side, `configs/workflow.otel.yml` adds the tutorial's exporter block:

```yaml
    tracing:
      otelcollector:
        _type: otelcollector
        endpoint: http://0.0.0.0:4318/v1/traces
        project: alto-ops-claw
```

In the NAT 1.9 source (`nat/plugins/opentelemetry/register.py`), this exporter sets the OTel resource attribute `service.name` to `project`. That explains the quirk the research tutorial cites from Classmethod: Phoenix files otelcollector traces under `default`, so the native `phoenix` exporter is the cleaner route to Phoenix.

Lab 05-2 validates this config for real. Then it checks the Docker daemon. On this Mac the daemon was not running, so the lab stopped that part and said so. It printed no collector file it never received:

**Expected output** (captured on this Mac, lab 05-2 steps 2–3)

```
$ nat validate --config_file week26/05_tracing/.runs/otel/workflow.otel.yml   [this laptop]
✓ nat validate → exit 0
$ docker info --format '{{.ServerVersion}}'   [this laptop]
⚠ the Docker daemon is not running here (client only, or not installed) → the collector is NOT started, and no collector output is shown. Start Docker Desktop and run this lab again to see it for real.
```

With Docker running, the lab starts the collector in the background on a free port, sends one Alto Ops run, counts the spans in `otellogs/llm_spans.json`, and stops the container.

**The sandbox rule.** A NAT inside an OpenShell sandbox (Module 04's `alto-ops`) can export to the collector only if the policy has an endpoint entry for the collector host, bound to the Python binary. OTLP/HTTP is a **POST** to `/v1/traces`. Start in `audit` mode, then move to `enforce`:

```bash
# on: spark
openshell policy update alto-ops --add-endpoint otel.alto.local:4318:read-write:rest:audit --binary /usr/bin/python3.12 --dry-run
openshell policy update alto-ops --add-endpoint otel.alto.local:4318:read-write:rest:audit --binary /usr/bin/python3.12 --wait
```

**Expected output** (captured on this Mac, lab 05-2 step 4, excerpt · policykit is the course's teaching model, not OpenShell)

```
│ endpoint entry                   request (python3.12)  policykit  why                                                 
│ ───────────────────────────────  ────────────────────  ─────────  ────────────────────────────────────────────────────
│ audit · read-write               POST /v1/traces       ✓ allow    network_policies.otel_collector: otel.alto.local:43…
│ enforce · read-only              POST /v1/traces       ✕ deny     otel_collector: access read-only allows only GET/HE…
│ enforce · allow POST /v1/traces  POST /v1/traces       ✓ allow    otel_collector: rule allow POST /v1/traces          
│ any                              connect 0.0.0.0:4318  ✕ deny     0.0.0.0 is loopback / link-local / 0.0.0.0 — always…
│ audit · read-write               curl POST /v1/traces  ✕ deny     otel.alto.local:4318 is listed, but not for binary …
✓ openshell 0.0.111 parsed `--add-endpoint otel.alto.local:4318:read-write:rest:audit --binary /usr/bin/python3.12 --dry-run`
```

Two lessons. First, a `read-only` preset allows only GET/HEAD/OPTIONS, so under `enforce` the export gets a 403 (Part 4 exercise 3). Second, the tutorial's laptop endpoint `http://0.0.0.0:4318` cannot work from inside a sandbox, because 0.0.0.0 is always blocked as SSRF. There you name the collector's host.

For OpenClaw claws, the research tutorial notes that enabling OpenClaw OTEL diagnostics with a local endpoint adds the `openclaw-diagnostics-otel-local` preset on Balanced/Open/Personal tiers. So one collector on the Spark host can collect both harness traces and NAT traces.

✓ Checkpoint: you can say why the collector logs 403 for a sandboxed NAT (read-only + enforce blocks POST), and name two fixes (`access: read-write`, or an explicit `allow: {method: POST, path: /v1/traces}`).

## 4 · L4.3 — Hermes traces to Langfuse without leaking keys

The Hermes harness has a Langfuse plugin. The research tutorial, citing the NemoClaw Hermes quickstart, shows how NemoClaw keeps the keys out of the sandbox. You register them as an OpenShell credential of type `langfuse-hermes-v1`. The sandbox receives only placeholders, and OpenShell substitutes the real values at egress.

Type this in the ⌨ terminal on the Spark yourself, with your real keys. The `export` lines hold secrets, so lab 05-3 never runs them. The lab runs `credentials add` through `change()` only when both variables are already set in its shell (it checks without printing them), and `rebuild` and `gateway restart` through `change()`.

```bash
# on: spark
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
nemohermes credentials add my-hermes-langfuse \
  --type langfuse-hermes-v1 \
  --credential LANGFUSE_PUBLIC_KEY \
  --credential LANGFUSE_SECRET_KEY
unset LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY
nemohermes my-hermes rebuild
```

```bash
# on: spark
nemohermes my-hermes connect
# inside the sandbox (nemohermes my-hermes connect)
hermes plugins enable observability/langfuse
exit
nemohermes my-hermes gateway restart
```

Do not put raw Langfuse keys, or copies of the placeholders, into `~/.hermes/.env`. Only the non-secret `HERMES_LANGFUSE_BASE_URL` belongs there. Copy this pattern for any SaaS observability backend: a credential handle in OpenShell, a placeholder in the sandbox, substitution at the proxy.

Lab 05-3's checker scans Hermes config text for raw `pk-lf-` / `sk-lf-` keys and for key variables that do not belong there. It prints redacted findings only. On this Mac it scanned four samples with fake keys (there is no `~/.hermes` here):

**Expected output** (captured on this Mac, lab 05-3 step 3)

```
✕ HIGH   sample-bad/.hermes/.env:2 — raw Langfuse key pk-lf-••• — exfiltrable by prompt injection; move it to an OpenShell credential (langfuse-hermes-v1)
✕ HIGH   sample-bad/.hermes/.env:3 — raw Langfuse key sk-lf-••• — exfiltrable by prompt injection; move it to an OpenShell credential (langfuse-hermes-v1)
✕ HIGH   sample-bad/.hermes/config.yaml:4 — raw Langfuse key sk-lf-••• — exfiltrable by prompt injection; move it to an OpenShell credential (langfuse-hermes-v1)
⚠ MEDIUM sample-copied-placeholder/.hermes/.env:2 — LANGFUSE_PUBLIC_KEY is set here (value not shown) — keys and copied placeholders do not belong in the agent's config
✓ OK     sample-good/.hermes/.env — no Langfuse keys; only non-secret settings
◆ 3 HIGH · 1 MEDIUM · 1 OK
```

Why is a key in `/sandbox/.hermes/config.yaml` worse than useless (Part 4 exercise 4)? The agent can read and rewrite its own config tree. The docs treat `/sandbox/.hermes` as mutable, agent-controlled state, not an isolation boundary. So a prompt-injected agent can send the key out. The raw key does not even help, because OpenShell expects the placeholder.

On the Spark, count matches inside the sandbox. Never print them:

```bash
# on: spark
# inside the sandbox (nemohermes my-hermes connect)
grep -cE 'pk-lf-|sk-lf-' ~/.hermes/.env ~/.hermes/config.yaml
```

NAT 1.9 also has its own `langfuse` exporter (fields `endpoint`, `public_key`, `secret_key`; empty keys fall back to the `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` environment variables). NAT builds the Basic auth header itself. Whether OpenShell's placeholder substitution reaches a header that NAT base64-encodes is not covered by this course's sources. Verify it on your unit before you rely on it.

✓ Checkpoint: you can explain, in one sentence each, where the Langfuse keys live (an OpenShell credential), what the sandbox sees (placeholders), and what belongs in `~/.hermes/.env` (only `HERMES_LANGFUSE_BASE_URL`).

## 5 · L4.4 — The policy plane: reading OpenShell like a trace

OpenShell's supervisor and gateway log every decision. These are the research tutorial's commands, with one change you must make: **bound anything that follows**. On the laptop CLI, `openshell logs --help` says `--tail` means "Stream live logs", so the command never returns on its own. Never run a follow in the foreground on the Spark. Wrap it in `timeout 10`, or use the bounded form:

```bash
# on: spark
timeout 10 openshell logs alto-ops --tail --source sandbox     # denied host, path, binary
openshell logs alto-ops -n 50 --since 10m --source sandbox      # bounded: last 50 lines
openshell settings get alto-ops                                 # effective policy source
openshell policy get alto-ops --full                            # what is really enforced now
docker logs $(docker ps --filter name=openshell-alto-ops --format '{{.Names}}') --tail 50
#   look for "OpenShell Sandbox Supervisor success" and "Applying Landlock filesystem sandbox"
```

```bash
# on: spark
openshell term      # live TUI: allow / deny / inspect_for_inference — f follow, s filter by source, q quit
```

**Expected output** (REFERENCE — quoted from the OpenShell playbook, Step 11)

```
- **Live log stream** — outbound connections, policy decisions (`allow`, `deny`, `inspect_for_inference`), and inference interceptions
```

Lab 05-4 parses all four commands with the real OpenShell 0.0.111 CLI against a dead gateway. All four parse. The Spark half is read-only; in DRY mode it prints EXAMPLE shapes.

Keep two facts in mind when you read these logs:

- An L7 violation under `enforcement: audit` is **logged, and the traffic is forwarded**. In audit mode a "violation" is a finding to fix, not a block.
- A Landlock path skipped under `landlock: best_effort` shows up as a High-severity OCSF `DetectionFinding`.

The lab's parser shows both on a decision stream. The stream is an **EXAMPLE shape** written for the course; its decisions come from `policykit`, not from OpenShell:

**Expected output** (captured on this Mac, lab 05-4 step 3, excerpt — the stream lines themselves are an EXAMPLE shape)

```
│ decision               count
│ ─────────────────────  ─────
│ allow                  2    
│ deny                   3    
│ inspect_for_inference  2    
⚠ MEDIUM · audit-mode violation — traffic was FORWARDED; fix the policy before enforce · python3.12 → otel.alto.local:4318 POST /v1/traces
✕ HIGH · OCSF DetectionFinding · Landlock best_effort skipped a path that does not exist in the image
✓ 2 inspect_for_inference events = the 2 LLM calls of this EXAMPLE request — lab 05-5 checks that number against a real NAT trace
```

✓ Checkpoint: you can say why `openshell logs --tail` needs `timeout 10` in a lab, and why an audit-mode violation is a finding and not a block.

## 6 · L4.5 — Correlate: one request, three planes

Now join the planes for one request. The research tutorial's recipe:

1. Send one request through `nat serve` (`/v1/workflow/full`) and save the intermediate steps.
2. Find the same run in the agent traces; note the LLM span count and total tokens.
3. In `openshell term`, filter (`s`) to the sandbox and count `inspect_for_inference`. It should equal the LLM span count, because every model call goes through `inference.local`.
4. Trigger a denied call (ask for "the latest weather from open-meteo"). A tool span fails in the agent plane at the same moment OpenShell logs `deny`.

```bash
# on: laptop
week26/.venv-nat/bin/nat serve --config_file week26/05_tracing/configs/workflow.traced.yml --port 8001
```

Leave it running, and in a second terminal send the request:

```bash
# on: laptop
curl -s -X POST 'http://localhost:8001/v1/workflow/full?filter_steps=LLM_END,TOOL_END' \
  -H 'Content-Type: application/json' -d '{"input_message": "Plant status last 6 hours?"}'
```

What the live NAT 1.9.0 server on this Mac does with that request (checked against its `/openapi.json` and with `curl`):

| Behaviour | Seen on this Mac |
|---|---|
| routes | `POST /v1/workflow/full`, plus the legacy `POST /generate/full`; both take the `filter_steps` query parameter |
| body | `{"input_message": "…"}` (or `messages`); anything else → `422` "Either messages or input_message must be provided" |
| `filter_steps=LLM_END,TOOL_END` | 3 `intermediate_data:` lines (2 × `LLM_END`, 1 × `TOOL_END`) and 88 `data:` lines (the streamed answer) |
| `filter_steps=none` | 0 `intermediate_data:` lines, the same answer lines |
| `… \| grep -c '"type":"LLM_END"'` | `2` — so the Spark check below works as written |

`nat serve` needs the `greenlet` package in `week26/.venv-nat`: NAT 1.9.0's FastAPI front end imports its async job store (SQLAlchemy asyncio) at start-up. Without it the server exits before it is ready. Install it once with `uv pip install --python week26/.venv-nat/bin/python greenlet`. If `nat serve` still fails, lab 05-5 says why and falls back to calling the same route function (`generate_streaming_response_full`) in-process through `week26/05_tracing/fullstream.py`, labelled as such.

**Expected output** (captured on this Mac, lab 05-5 steps 1–3, excerpt · LAPTOP STAND-IN model)

```
$ nat serve --config_file week26/05_tracing/.runs/lab05_5_workflow.yml --port 8001 &   [this laptop, background → lab05_5_nat_serve.log]
✓ ready in 7.4s → http://localhost:8001/docs
$ curl -s -X POST 'http://localhost:8001/v1/workflow/full?filter_steps=LLM_END,TOOL_END' -H 'Content-Type: application/json' -d '{"input_message": "Plant status last 6 hours?"}'   [this laptop]
■ stopped nat (pid 12931)
✓ nat serve · HTTP: 3 intermediate step(s) after the filter in 25.2s → LLM_END ×2, TOOL_END ×1
│ event (stream)  name                    tokens  duration
│ ──────────────  ──────────────────────  ──────  ────────
│ LLM_END         nemotron-3-nano:latest  503     7.327 s 
│ TOOL_END        chiller_kpi                     0.005 s 
│ LLM_END         nemotron-3-nano:latest  765     15.483 s
│ measure (LAPTOP STAND-IN)  /v1/workflow/full stream  file exporter
│ ─────────────────────────  ────────────────────────  ─────────────
│ LLM spans                  2                         2            
│ TOOL spans                 1                         1            
│ total tokens (LLM)         1268                      1268         
│ open spans                 —                         0            
│ workflow span              —                         22.918 s     
✓ both views of the run agree: 2 LLM span(s), 1 TOOL span(s)
✓ no open spans: the process stayed alive after the run, so the exporter finished writing (compare lab 05-1)
```

Three things to notice. The stream and the file agree: 2 LLM spans, 1 tool span, 1268 tokens. This time the file has **no** open spans, because the server process stayed alive after the run. And these LLM calls were slower than in lab 05-1 (7.3 s and 15.5 s against 3.6 s and 8.1 s), because other labs were sharing the laptop's Ollama. Laptop timings are a stand-in: never compare them with Spark numbers (Module 06 measures the Spark).

So the prediction for the sandboxed run of this request is **2 `inspect_for_inference` events**, one per LLM span. Check it on the Spark against the sandboxed `nat serve` from Module 04 (reachable on port 8001), with bounded, read-only commands:

```bash
# on: spark
curl -s -X POST 'http://localhost:8001/v1/workflow/full?filter_steps=LLM_END,TOOL_END' -H 'Content-Type: application/json' -d '{"input_message": "Plant status last 6 hours?"}' | grep -c '"type":"LLM_END"'
openshell logs alto-ops -n 200 --since 2m --source sandbox | grep -c inspect_for_inference
```

Alto Ops Claw has no web tool, so on the laptop it cannot even try the open-meteo call. A harness that does try shows both halves: a failed tool span and a `deny` line.

Write the mapping down. It is the start of the audit story you will tell a hotel owner, or a Thai regulator, in Module 07:

| Plane | Source | Join key | For this request (LAPTOP STAND-IN) |
|---|---|---|---|
| agent traces | NAT exporters (file · Phoenix · OTel · Langfuse) | `workflow_run_id` / trace id, input text, timestamps | 2 LLM spans, 1 TOOL span, 1268 tokens |
| policy logs | `openshell logs` / `openshell term`, OCSF findings | sandbox name + timestamp window | 2 `inspect_for_inference` (predicted); a `deny` = a failed tool span |
| harness logs | `nemoclaw <sandbox> logs`, `/tmp/gateway.log` (OpenClaw) | timestamp, channel / session id | channel events, pairing, crashes |

✓ Checkpoint: you can predict the `inspect_for_inference` count for a request from its trace (one per LLM span), and you know which command checks it on the Spark.

## Labs — run them here

**labs/lab05_1_file_and_phoenix.py** — List NAT 1.9's tracing exporters, check the file exporter's real keys, trace one Alto Ops run to a file (and to Phoenix only if it answers), and parse the spans.

**labs/lab05_2_otel_collector.py** — Write the OTel-collector config and exporter block, validate them, run the collector only if Docker is up, and check the sandbox rule for POST /v1/traces.

**labs/lab05_3_langfuse_handles.py** — Walk the Hermes Langfuse credential-handle flow with placeholders only, and scan Hermes config text for leaked keys.

**labs/lab05_4_policy_plane.py** — Parse the OpenShell log commands offline, read the policy plane on the Spark (bounded), and count decisions in an EXAMPLE stream.

**labs/lab05_5_correlate.py** — Send one request to /v1/workflow/full, compare it with the file trace, and predict the sandbox's inspect_for_inference count.

## Try it yourself

`exercises/ex05_telemetry.py` has three TODOs:

1. Write a `general.telemetry.tracing` block with a Phoenix exporter **and** a file exporter, using the keys NAT 1.9 really needs. The checker runs the real `nat validate` on it.
2. Fix the collector endpoint entry that returns 403, and keep `enforcement: enforce`.
3. Name the NAT decorator that traces a plain Python function, by its full import path, and its three event types. The checker confirms them in the installed NAT source.

```bash
# on: laptop
.venv/bin/python week26/05_tracing/exercises/ex05_telemetry.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ telemetry: 2 exporters — phoenix + file (output_path, project)
$ nat validate --config_file <your block + workflow.laptop.yml>   [this laptop]
✓ the real `nat validate` (NAT 1.9.0) accepts your block
✓ collector: POST /v1/traces allowed under enforce (rest, otel.alto.local, not `full`)
✓ decorator: @track_function from nat.plugins.profiler.decorators.function_tracking · SPAN_START / SPAN_CHUNK (generators) / SPAN_END — found in the installed NAT 1.9.0
```

<details><summary>Hint — the file exporter</summary>

Lab 05-1 step 2 printed the fields. Two have no default. Add `mode: overwrite` too if you want one run per file.

</details>

<details><summary>Hint — the 403</summary>

OTLP/HTTP is a POST. A `read-only` access preset allows GET, HEAD and OPTIONS. Either change the access mode, or replace it with a single allow rule for `POST /v1/traces`. Switching to `audit` makes the 403 go away, but only because nothing is enforced any more.

</details>

<details><summary>Hint — the decorator</summary>

The research tutorial names `@track_function` in `nat.plugins.profiler.decorators.function_tracking`. In NAT 1.9.0 that path is still right. Look in `week26/.venv-nat/lib/python3.12/site-packages/nat/plugins/profiler/decorators/function_tracking.py` for the `IntermediateStepType.` values it pushes. The same file also has `track_unregistered_function`, which adds scope management.

</details>

✓ Checkpoint: all three checker lines are ✓.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `nat validate`: `Invalid configuration: general: Field required` | the `file` tracing exporter is missing `output_path` and/or `project` (NAT 1.9) | add both keys (Section 2) |
| the trace file has START lines with no END, or fewer `LLM_END` than LLM calls | a short-lived `nat run` exited before NAT's exporter finished writing | count `LLM_START`, or trace a long-lived `nat serve` (lab 05-5) |
| one trace file keeps growing | the file exporter's `mode` defaults to `append` | set `mode: overwrite`, or one `output_path` per run |
| `nat serve`: `The SQLAlchemy asyncio module requires that the Python 'greenlet' library is installed` | `greenlet` is missing from `week26/.venv-nat` (NAT 1.9.0's FastAPI front end imports SQLAlchemy asyncio at start-up) | install it: `uv pip install --python week26/.venv-nat/bin/python greenlet`, then run lab 05-5 again (until then it falls back to the in-process twin, labelled as such) |
| `/v1/workflow/full` returns `422` "Either messages or input_message must be provided" | the JSON body uses another key | send `{"input_message": "…"}` |
| lab 05-1 says Phoenix SKIPPED | nothing answers on localhost:6006 | start Phoenix (Section 2) and run the lab again |
| lab 05-2 says the Docker daemon is not running | Docker Desktop is closed (the client alone cannot run containers) | start Docker Desktop, or read the rest of the lab offline |
| otelcollector traces land in Phoenix project `default` | NAT sends `project` as `service.name` | use the native `phoenix` exporter for Phoenix |
| the collector logs 403 for a sandboxed NAT | a `read-only` preset under `enforce` blocks the POST | `access: read-write`, or `allow: {method: POST, path: /v1/traces}` (exercise 05, TODO 2) |
| `openshell logs … --tail` never returns | `--tail` streams live logs | `timeout 10 …`, or `-n 50 --since 10m` |

## Next

[Lab 06 — Performance benchmarking on DGX Spark](../06_benchmarking/TUTORIAL.md): measure the engine, the workflow, the quality and the sandbox tax separately, using the same traces you just learned to read.
