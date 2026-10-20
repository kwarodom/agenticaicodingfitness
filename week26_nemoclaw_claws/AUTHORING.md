# Week 26 · authoring guide (for maintainers and agents adding modules)

Week 26 turns the research tutorial **`week26/NemoClaw on DGX Spark — Beginner to Expert Tutorial with NAT.md`**
(Parts 0–6 + Capstone, lab ids L1.1 … L6.3) into one hands-on course run by the **Reef Lab Runner**
(`week26/00_reef_lab_runner/`, port 8126), in the style of Week 25. The runner's design comes from
`week26/Alto Reef — Web Lab Runner Spec and Frontier-Model Build Brief.md` (§4.4 security, §4.5 honesty, §6 lab
mapping). **Module 01 (`01_what_is_a_claw/`) is the reference implementation — copy its shape exactly.**

## The module list (folder · title · research tutorial part · lab ids)

| # | folder | title | part | lab ids |
|---|---|---|---|---|
| 01 | `01_what_is_a_claw` | What is a claw? The stack, the sandbox, and the three harnesses | Part 0 | — |
| 02 | `02_first_claw` | Your first claw on DGX Spark: verify, install, onboard, prove inference is local | Part 1 | L1.1–L1.7 |
| 03 | `03_policy_as_code` | OpenShell sandboxes and policy as code | Part 2 | L2.1–L2.7 |
| 04 | `04_nat_claws` | Build a claw with NeMo Agent Toolkit: tools, serving, MCP, sandboxed NAT | Part 3 | L3.1–L3.8 |
| 05 | `05_tracing` | Tracing and observability: agent, policy and harness planes | Part 4 | L4.1–L4.5 |
| 06 | `06_benchmarking` | Performance benchmarking on DGX Spark: engine, workflow, quality, sandbox tax | Part 5 | L5.1–L5.5 |
| 07 | `07_hardening` | Expert: threat model, hardening, custom blueprints, remote gateways | Part 6 | L6.1–L6.3 |
| 08 | `08_capstone_alto_ops_claw` | Capstone: Alto Ops Claw v1 | Capstone | all |

Each module's `## Next` links to the next folder: `[Lab NN — title](../NN_folder/TUTORIAL.md)`. Module 08's
Next points back to `../01_what_is_a_claw/TUTORIAL.md` or to "what to build next".

## Local resources (already installed — do NOT pip install anything into these)

| resource | path | what it is |
|---|---|---|
| NAT 1.9.0 + extras | `week26/.venv-nat/` (`bin/nat`, `bin/python`) | `nvidia-nat[langchain,mcp,profiler,phoenix,opentelemetry,eval,ragas]` + pandas + greenlet (nat serve needs it). Evaluators: ragas, trajectory, avg_llm_latency, avg_num_llm_calls, avg_tokens_per_llm_end, avg_workflow_runtime. Tracing exporters: phoenix, otelcollector, file, langfuse … `nat sizing calc`, `nat mcp serve/client`, `nat serve`, `nat eval` all exist. |
| Alto Ops package | `week26/common/alto_ops/` (installed editable in .venv-nat) | NAT tools `chiller_kpi` (research Lab 3.4) and `request_setpoint_change` (ticket only, validates `work_order_id`), configs in `src/alto_ops/configs/`: `workflow.laptop.yml` (laptop Ollama, **verified working**, ~20 s per question with `nemotron-3-nano:latest`), `workflow.spark.yml` (vLLM on the Spark), `workflow.sandbox.yml` (`https://inference.local/v1`) |
| data | `week26/common/data/chiller_plant.csv` | SYNTHETIC, 672 rows, 15-min `ts,kw,rt`; last 6 h degraded → `chiller_kpi(hours=6)` = `kw_per_rt=0.901 status=ALARM`; 24 h = 0.703 OK |
| mock BMS over MCP | `week26/common/bms_mcp_server.py` | FastMCP streamable-http on `http://localhost:${BMS_MCP_PORT:-8443}/mcp`; tools `read_point`, `list_alarms`, `get_trend`, `write_setpoint` (never writes) |
| OpenShell CLI | `week26/.venv-openshell/bin/openshell` (0.0.111) | the last PyPI release with a macOS binary. **Offline parser only**: `clawkit.openshell_offline(args)` runs it against a dead gateway; argument/YAML errors come back immediately, anything that parsed fails with "Connection refused" (`clawkit.parsed_ok(r)`). NemoClaw pins 0.0.116 on the Spark; latest is 0.1.2 (SDK-only wheels on PyPI). |
| policy model | `week26/common/policykit.py` | `validate`, `decide` (fs / process / L4 / REST / **MCP** / SSRF / audit vs enforce), `parse_endpoint_spec`, `parse_rule_spec`, `harden` (§6.2 checklist). A TEACHING MODEL — say so. |
| playbooks | repo-root `dgx-spark-playbooks/nvidia/` | `playbook-nemoclaw`, `playbook-openshell`, `playbook-openclaw`, `playbook-hermes-agent`, `playbook-nemoclaw-applications`, `station-*` |
| laptop Ollama | `http://localhost:11434/v1` | `nemotron-3-nano:latest` (tool calling works with NAT), `gemma4:12b`, `gemma3:4b` |
| Docker | the Docker **client** is installed; the daemon is usually **not running** on this Mac | labs that want Phoenix/OTel containers must detect that and fall back (NAT `file` exporter) |

Versions to state honestly: the research tutorial cites NAT docs 1.8 — the course runs **1.9.0**. OpenShell pinned 0.0.116 / latest 0.1.2 / laptop parser 0.0.111.

## Files per module

```text
NN_folder/
  TUTORIAL.md            the course text (the runner splits it on "## " headings)
  TUTORIAL.th.md         Thai translation — structurally identical (check_translation.py)
  diagrams.json          architecture + sequence + charts (schema: see 01's file)
  labs/labNN_k_<name>.py 2–4 runnable labs; docstring line 1 = "Lab NN-k · <title>."
  exercises/exNN_<name>.py              starter with 2–4 TODOs + an offline checker (exits 1 with ✕ TODO lines)
  exercises/solutions/exNN_<name>.py    reference solution (sys.path uses parents[3])
  (optional) configs/ policies/ data/   module-owned YAML the labs use
```

## TUTORIAL.md contract (the parser depends on it)

- H1: `# ▶ Reef Lab NN — <title>`; then the `> Part of Week 26 …` blockquote (copy 01's).
- `**What you'll actually do**` bullets, then one meta line exactly:
  `**Time** ~NN min · **Difficulty** beginner|intermediate|advanced|expert · **Hardware** <…>`
- `**Sources:**` line: the research tutorial part + the doc links it cites for this part.
- Sections, in order: `## 0 · Before you start`, `## 1 · …` … `## N · …`, `## Labs — run them here`,
  `## Try it yourself`, `## Troubleshooting`, `## Next`.
- Every numbered section ends with one `✓ Checkpoint: …` line.
- Section headings that teach a research lab carry its id: `## 2 · L1.5 — Prove inference is local`.
- In "Labs — run them here", one line per lab: `**labs/labNN_k_x.py** — One-sentence title.`
- Bash blocks start with a target hint so the ⌨ terminal picks the machine: `# on: laptop` or `# on: spark`.
  Inside-the-sandbox commands: `# on: spark` + a comment `# inside the sandbox (nemoclaw <s> connect)`.
- `**Expected output**` precedes an output block, with provenance in parentheses:
  `(captured on this Mac)`, `(captured on this Mac, DRY mode)`, `(REFERENCE — quoted from <doc/playbook>)`,
  `(EXAMPLE — illustrative shape, not a measurement)`. The runner hides ▶ run on those blocks.
- Optional inline live blocks — a fenced block with language `spark` holding JSON
  `{"target": "vllm|ollama|nat|hermes", "model": "…", "messages": [...], "max_tokens": 256, "tools": [...]?}`.
  `target: "nat"` hits a `nat serve` on the Spark, else one a lab started on this laptop (:8001), else Ollama.
- `<details><summary>Hint — …</summary> … </details>` in "Try it yourself".
- Write the course in your own words: short sentences, active voice, why before how, tables for comparisons.
  Do not paste the research tutorial wholesale; every command a learner must type IS in the tutorial.

## Honesty rules (non-negotiable)

1. **Never invent Spark output.** No module has run on a real Spark yet (the Sparks were not reachable with a
   key while this was built). Spark-only output is either quoted verbatim (`reference=` → REFERENCE; the audit
   checks it against the playbooks + the research tutorial) or an illustrative shape (`example=` → EXAMPLE).
   Never put invented tok/s, latencies, versions or sandbox listings in a REFERENCE.
2. **Everything that CAN run on the laptop, you run, and you paste the real output**: policy parsing (the real
   CLI + policykit), spec grammars, config generators and validators, NAT agents against laptop Ollama,
   `nat serve` / `nat mcp serve` / `nat eval` / profiler on the laptop, the mock BMS, every exercise checker.
   Run every lab, exercise and solution before you finish; every lab and solution exits 0 (a starter exercise
   exits 1 with ✕ TODO lines).
3. Commands, flags, env vars, ports, image tags and model ids are copied from the research tutorial (or a
   playbook), not from memory. Where the tutorial itself says "verify on your unit" (e.g. Part 1 exercise 2's
   env-var combination, Phoenix project routing, NAT import paths), keep that caveat.
4. Third-party numbers (Part 5's tok/s table, CVE scores) are quoted **with their source** as "per <source>, cited
   in the research tutorial". Laptop numbers are labelled `LAPTOP STAND-IN` and never compared with Spark numbers.
5. **Changes go through `clawkit.change()`.** Anything that installs, onboards, adds/removes presets, sets a
   policy, snapshots, rebuilds, creates or deletes a sandbox/provider: `change(cmd, preview=<--dry-run or
   read-only cmd>, example=…)`. It runs only in LIVE mode with CLAW_APPLY=1 (the runner's 🔓 toggle). The
   one-command installer (`curl … | bash`) is never run by a lab — show it and tell the learner to run it in
   the ⌨ terminal themselves (the spec: "runner does not run it silently"). Read-only commands (`status`,
   `list`, `get`, `logs` without --follow, `--version`) use plain `sh()`.
6. Long or streaming commands (`logs --follow`, `openshell term`, `nat serve`) are never run in the foreground on
   the Spark. Laptop servers run inside `with clawkit.background([...], ready_url=…, log=RUNS/…):` and pick
   their port with `free_port(<documented port>)` (other labs may be running) — print the port you got.
   The runner kills a lab after 900 s: keep laptop LLM calls few (≤ 6 per lab, eval datasets ≤ 5 rows).
7. **Self-audit before you report:** `.venv/bin/python week26/common/audit_references.py <module>` must print
   `0 not found verbatim`, and `.venv/bin/python week26/common/check_translation.py <module>` must pass.
8. Secrets: never print or interpolate tokens into a command line (`sh()` echoes commands). Credential flows
   (Langfuse, Telegram, provider keys) are shown with placeholders (`pk-lf-...`, `<your-key>`) and the lesson
   is that the runner and the sandbox never see the raw key.

## clawkit (week26/common/clawkit.py) — read it, do not edit it

Week 25's sparkkit API (`banner(title, sub, status=True)` · `step` · `ok/warn/note/result` · `table` · `bar` ·
`check` · `sh(cmd, reference=, example=, timeout=)` → `Result(code, out, source)` · `put` · `url(kind)` · `up` ·
`models` · `chat` · `chat_any(kind, model, messages, tools=, reference=)` · `show_chat` · `weights_gb` ·
`kv_cache_gb` · `decode_ceiling_tok_s` · `SPEC` · `cfg` · `host` · `mode()` · `on_spark()` · `LAPTOP_OLLAMA` ·
`pick_laptop_model()`), plus the claw stack:
`PORTS` (vllm 8000 · ollama 11434 · nat 8001 · nat_mcp 9901 · phoenix 6006 · otel 4318 · openclaw 18789 ·
hermes 8642 · gateway 8080) · `sandbox()` (CLAW_SANDBOX, default `my-assistant`) · `change(cmd, preview=,
example=, reference=)` · `apply_enabled()` · `laptop(argv, quiet=, show=, env=, cwd=, timeout=)` →
`Result(source="laptop")` · `openshell_offline(args)` · `parsed_ok(r)` · `free_port(p)` ·
`background(argv, ready_url=, log=, env=, cwd=, show=)` · `NAT`, `NAT_PY`, `OPENSHELL`, `OPENSHELL_PINNED`,
`RESEARCH`, `PLAYBOOKS`, `ROOT`, `WEEK`, `SANDBOX_RE`, `ANSI`.
Output glyphs the runner colours: `━` banner · `▣` step · `✓` pass · `✕` fail · `⚠` warn · `◈` dry · `◆` metric ·
`═` result · `→` action · `│` table · `$` command · `■` stopped · `~ REASON` · `· ANSWER`.
If you need a helper that is missing, write it inside your lab file (or a module-local `NN_folder/<name>kit.py`).
Labs import it with `sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))`; write run
artefacts to `NN_folder/.runs/` (gitignored).

## diagrams.json

`{"architecture": {title, caption, lanes:[{id,label}], nodes:[{id,label,lane,sub,accent}],
edges:[{from,to,label,animated?}]}, "sequence": {title, caption, actors:[{id,label}],
steps:[{from,to,label,kind:"call"|"return"|"note"}]}, "charts": [{title, type:"bars"|"hbars"|"line"|"donut",
unit, labels:[…], series:[{name, values:[…], color}], caption}]}` — accents/colors: green cyan amber violet red.
Chart numbers come from a lab you ran, from arithmetic, or from a cited source, and the caption says which.

## Thai translation (TUTORIAL.th.md)

Same `## ` sections in the same order, same number of `✓ Checkpoint` lines per section (keep the literal
`✓ Checkpoint:` prefix, Thai after it), every fenced code block byte-for-byte identical, the same
`**labs/…**` / `**exercises/…**` blurbs (the path in bold stays, the sentence is Thai), the literal
`**Time** … · **Difficulty** … · **Hardware** …` labels and `**Expected output**` markers (Thai may follow in
brackets). Keep technical terms (sandbox, policy, preset, gateway, harness, claw) in English where Thai
engineers would. Natural Thai, not word-for-word.
