# Week 26 · NemoClaw on DGX Spark — beginner to expert

Build, sandbox, trace, benchmark and harden **claws**: always-on, tool-using agents inside NVIDIA
OpenShell sandboxes, installed with NemoClaw, with purpose-built agent logic written in NeMo Agent Toolkit (NAT).
The running example is **Alto Ops Claw**, a hotel chiller-plant assistant.

The course is the research tutorial [`NemoClaw on DGX Spark — Beginner to Expert Tutorial with NAT.md`](NemoClaw%20on%20DGX%20Spark%20—%20Beginner%20to%20Expert%20Tutorial%20with%20NAT.md)
turned into eight runnable modules, served by the **🪸 Reef Lab Runner**. The runner's design follows
[`Alto Reef — Web Lab Runner Spec`](Alto%20Reef%20—%20Web%20Lab%20Runner%20Spec%20and%20Frontier-Model%20Build%20Brief.md)
(honesty rules §4.5, security §4.4, lab mapping §6). The full visual React app from that spec is in [`alto-reef/`](alto-reef/).

## Start

```bash
# once — laptop tools (NAT 1.9 + the Alto Ops package; the OpenShell CLI used as an offline policy parser)
uv venv --python 3.12 week26/.venv-nat
uv pip install --python week26/.venv-nat/bin/python 'nvidia-nat[langchain,mcp,profiler,phoenix,opentelemetry,eval,ragas]==1.9.0' pandas greenlet   # greenlet: nat serve needs it
uv pip install --python week26/.venv-nat/bin/python -e week26/common/alto_ops
uv venv --python 3.12 week26/.venv-openshell
uv pip install --python week26/.venv-openshell/bin/python openshell==0.0.111

# every time
.venv/bin/python week26/00_reef_lab_runner/tutorial_server.py      # → http://127.0.0.1:8126
```

Point it at your Spark with **🖥 Spark setup** (`SPARK_HOST`, key-based ssh; a host saved in week25 is reused).
Without a Spark everything runs **DRY**: Spark commands are shown with labelled RECORDED / REFERENCE / EXAMPLE
output. Laptop labs (NAT agents on your Ollama, `nat serve`, MCP, eval, the policy parser) run for real either way.

## Modules

| # | module | research tutorial | you build |
|---|---|---|---|
| 01 | [What is a claw?](01_what_is_a_claw/TUTORIAL.md) | Part 0 | the mental model: stack, five layers, three harnesses |
| 02 | [Your first claw](02_first_claw/TUTORIAL.md) | Part 1 · L1.1–L1.7 | a verified Spark, a NemoClaw install, proof that inference is local |
| 03 | [Policy as code](03_policy_as_code/TUTORIAL.md) | Part 2 · L2.1–L2.7 | OpenShell policies you read, write, iterate on; bring-your-own vLLM |
| 04 | [NAT claws](04_nat_claws/TUTORIAL.md) | Part 3 · L3.1–L3.8 | Alto Ops Claw in NAT: custom tool, REST, MCP both ways, in a sandbox |
| 05 | [Tracing](05_tracing/TUTORIAL.md) | Part 4 · L4.1–L4.5 | agent traces + policy logs + harness logs, correlated |
| 06 | [Benchmarking](06_benchmarking/TUTORIAL.md) | Part 5 · L5.1–L5.5 | engine, workflow, quality and sandbox-tax numbers |
| 07 | [Hardening](07_hardening/TUTORIAL.md) | Part 6 · L6.1–L6.3 | threat model, production policy, blueprints, remote gateways |
| 08 | [Capstone](08_capstone_alto_ops_claw/TUTORIAL.md) | Capstone | Alto Ops Claw v1, graded on evidence |

## What is where

```text
week26/
  00_reef_lab_runner/    tutorial_server.py (FastAPI, :8126) + static/guide.html — course, ▶ labs, ⌨ terminal,
                         🔓 Allow changes (CLAW_APPLY), 🪸 Reef (/api/reef: read-only stack status with provenance)
  common/clawkit.py      the lab helper: sh() LIVE/DRY, change() opt-in mutations, laptop(), background(), chat_any()
  common/policykit.py    OpenShell policy teaching model: validate · decide · spec grammars · harden (§6.2)
  common/alto_ops/       NAT plugin package: chiller_kpi, request_setpoint_change; laptop/spark/sandbox configs
  common/bms_mcp_server.py   mock BMS over MCP (read_point, list_alarms, get_trend, write_setpoint — never writes)
  common/data/           synthetic chiller-plant CSV (+ its generator)
  common/audit_references.py · check_translation.py   maintainer checks
  NN_*/                  TUTORIAL.md (+ .th.md), labs/, exercises/, diagrams.json
  alto-reef/             the Alto Reef React + FastAPI app (M6 reef-map preview) from the spec
```

## Honesty rules (short version)

- Spark output is never invented. RECORDED = a real Spark transcript; REFERENCE = verbatim from an NVIDIA
  playbook or the research tutorial (audited); EXAMPLE = an illustrative shape.
- Laptop results are labelled **LAPTOP STAND-IN** and are never compared with Spark numbers.
- Nothing changes your Spark unless you turn on **🔓 Allow changes**; the NemoClaw installer is never run by a lab.
- Versions: NAT **1.9.0** (the research tutorial cites the 1.8 docs) · OpenShell **0.0.116** pinned by NemoClaw,
  **0.1.2** latest, **0.0.111** laptop parser (the last PyPI wheel with a macOS CLI).

Maintainers: see [AUTHORING.md](AUTHORING.md).
