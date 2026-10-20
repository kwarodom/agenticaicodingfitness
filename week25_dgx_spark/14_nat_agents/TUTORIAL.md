# ▶ Spark Lab 14 — NeMo Agent Toolkit: agents on your Spark's models

> Part of Week 25 · DGX Spark: fine-tune, serve, and build sandboxed agents. You type the commands, you see the real output. Every lab also runs in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's playbook, or a clearly marked EXAMPLE.

**What you'll actually do**
- Install NVIDIA NeMo Agent Toolkit (NAT) 1.9 in its own venv, on the laptop and on the Spark.
- Read a NAT workflow file: three sections (`llms`, `functions`, `workflow`) turn a model server into an agent.
- Write two hotel-operations tools as a small NAT plugin package, and watch NAT discover them.
- Run the agent with `nat run` against the Spark's vLLM (or the laptop stand-in), and read the trace it writes.
- Serve the same agent as an OpenAI-compatible web service (`nat serve`) and its tools as an MCP server (`nat mcp serve`).
- Score the agent with `nat eval` and an evaluator that needs no LLM judge.

**Time** ~50 min · **Difficulty** intermediate · **Hardware** 1 Spark (or none: the laptop stand-in runs every lab for real)

**Official playbooks covered:** none. NAT has no DGX Spark playbook, so this module is **course-original**: every command, field name and output below was checked against the installed `nvidia-nat` 1.9.0 package (`nat --help`, `nat info components`, the package source). It builds on the [vLLM](https://build.nvidia.com/spark/vllm) playbook (Module 05) for the model server.

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| This repo's Python | `.venv/bin/python --version` → 3.13 | runs the labs |
| `uv` on the laptop | `uv --version` | creates the NAT venv (Python 3.12) |
| A model server with tool calling | Spark: vLLM from Module 05 on `:8000` · laptop: Ollama on `:11434` | the agent's brain |
| A tool-calling model on the laptop | `curl -s localhost:11434/v1/models` lists `nemotron-3-nano:latest` | the laptop stand-in |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
.venv/bin/python --version
uv --version
curl -s localhost:11434/v1/models | head -c 200
```

**Expected output** (captured on this Mac)

```
Python 3.13.13
uv 0.11.19 (7b2cff1c3 2026-06-03 aarch64-apple-darwin)
{"object":"list","data":[{"id":"nemotron-3.5-lightning:latest","object":"model","created":…
```

> 💡 NAT runs in **its own venv** (`week25/.venv-nat`, Python 3.12), not in the repo `.venv`. The labs run with the repo Python and call the `nat` CLI as a subprocess, so the two never clash.

✓ Checkpoint: `uv --version` prints a version, and either the Spark's vLLM or the laptop's Ollama lists a model.

## 1 · What NAT adds on top of a model server

Module 05 gave you an OpenAI-compatible server. A server answers one request at a time. An **agent** needs a loop around it: send the question and the tool list, run the tool the model asks for, send the result back, repeat until the model answers. NAT is that loop, plus the parts you would otherwise write by hand.

| Layer | You write | NAT gives you |
|---|---|---|
| Model | nothing (vLLM, Ollama, NIM, LiteLLM) | an LLM client for any OpenAI-compatible URL |
| Tools | plain async Python functions | schema from the type hints, registration, a tool wrapper for LangChain |
| Agent loop | one line: `_type: tool_calling_agent` | ReAct, tool-calling, ReWOO, router agents |
| Front ends | nothing | console (`nat run`), REST + OpenAI API (`nat serve`), MCP (`nat mcp serve`) |
| Quality | a small dataset | `nat eval`, tracing to a file or OpenTelemetry |

```text
   your laptop or the Spark                                      the Spark
 ┌───────────────────────────────────────────────┐          ┌─────────────────────┐
 │ configs/hotel_agent.yml                        │          │ vLLM :8000          │
 │   llms:      hotel_llm  (_type: openai) ───────┼── HTTP ─►│ Qwen3.6-35B-A3B     │
 │   functions: room_temperature                  │          │ (tool calling on)   │
 │              create_maintenance_ticket  ◄─ hotel_ops_nat └─────────────────────┘
 │              current_datetime           ◄─ built into NAT
 │   workflow:  tool_calling_agent                │
 │ front end:  nat run · nat serve · nat mcp serve│
 └───────────────────────────────────────────────┘
```

NAT is pure Python. Nothing in it needs the GPU; the model server does. So you can run NAT on the Spark next to vLLM, or on your laptop pointing at the Spark over the tailnet. Module 15 puts it inside an OpenShell sandbox on the Spark.

✓ Checkpoint: you can say which part of an agent NAT runs and which part vLLM runs.

## 2 · Install NAT and register the course tools

On the laptop, the course keeps NAT in `week25/.venv-nat`. Three packages: `nvidia-nat[langchain]` (the agents), `greenlet` (NAT 1.9's `nat serve` imports SQLAlchemy's asyncio module, which needs it, and it is not pulled in automatically), and `nvidia-nat-mcp` (the MCP client and server):

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
uv venv -p 3.12 week25/.venv-nat
uv pip install --python week25/.venv-nat/bin/python "nvidia-nat[langchain]~=1.9" greenlet "nvidia-nat-mcp~=1.9"
uv pip install --python week25/.venv-nat/bin/python --no-deps -e week25/14_nat_agents/hotel_ops_nat
week25/.venv-nat/bin/nat --version
```

On the Spark (aarch64, DGX OS), the same install goes in `~/w25`. Clone this repo on the Spark first (`git clone <this repo> ~/agenticaicodingfitness`), because Module 15 uploads the module folder into a sandbox from there:

```bash
# on: spark
mkdir -p ~/w25 && cd ~/w25 && python3 -m venv .venv-nat
.venv-nat/bin/pip install -q 'nvidia-nat[langchain]~=1.9' greenlet 'nvidia-nat-mcp~=1.9'
.venv-nat/bin/pip install -q --no-deps -e ~/agenticaicodingfitness/week25/14_nat_agents/hotel_ops_nat
.venv-nat/bin/nat --version
```

Lab 14-1 checks the install, registers the tools if needed, lists what NAT can see, and validates every workflow file in this module:

```bash
# on: laptop
.venv/bin/python week25/14_nat_agents/labs/lab14_1_install_and_register.py
```

**Expected output** (captured on this Mac, trimmed)

```
▣ STEP 1 · find the NAT CLI
$ week25/.venv-nat/bin/nat --version
✓ nat, version 1.9.0

▣ STEP 2 · register the course plugin (hotel_ops_nat) — the NAT way: a package with a 'nat.components' entry point
✓ hotel_ops_nat is already installed in week25/.venv-nat (entry point nat.components → hotel_ops_nat.register)

▣ STEP 3 · what can NAT see? `nat info components` (only the kinds this module uses)
│ type            name                       package
│ ──────────────  ─────────────────────────  ────────────────────────
│ evaluator       langsmith_custom           nvidia-nat-langchain
│ front_end       console                    nvidia-nat-core
│ front_end       fastapi                    nvidia-nat-core
│ front_end       mcp                        nvidia-nat-mcp
│ function        create_maintenance_ticket  hotel_ops_nat
│ function        current_datetime           nvidia-nat-core
│ function        react_agent                nvidia-nat-langchain
│ function        room_temperature           hotel_ops_nat
│ function        tool_calling_agent         nvidia-nat-langchain
│ function_group  mcp_client                 nvidia-nat-mcp
│ llm_provider    openai                     nvidia-nat-core
│ tracing         file                       nvidia-nat-core
│ tracing         otelcollector              nvidia-nat-opentelemetry
…
◆ 134 components registered in total. The two hotel tools come from YOUR package, the agents from nvidia-nat-langchain, mcp_client from nvidia-nat-mcp.

▣ STEP 4 · validate the workflow files (offline — NAT parses and type-checks, no model is called)
│ config                         nat validate  workflow
│ ─────────────────────────────  ────────────  ──────────────────
│ configs/hotel_agent.yml        ✓ valid       tool_calling_agent
│ configs/hotel_agent_spark.yml  ✓ valid       tool_calling_agent
│ configs/hotel_agent_mcp.yml    ✓ valid       tool_calling_agent
│ configs/hotel_eval.yml         ✓ valid       tool_calling_agent
```

✓ Checkpoint: `room_temperature` and `create_maintenance_ticket` appear in the component table with package `hotel_ops_nat`, and all four configs are ✓ valid.

## 3 · The workflow file: `llms`, `functions`, `workflow`

This is the whole agent (`configs/hotel_agent.yml`, trimmed):

```yaml
llms:
  hotel_llm:
    _type: openai                                   # any OpenAI-compatible server
    base_url: ${HOTEL_LLM_BASE_URL:-http://localhost:11434/v1}
    model_name: ${HOTEL_LLM_MODEL:-nemotron-3-nano:latest}
    api_key: ${HOTEL_LLM_API_KEY:-not-needed}
    temperature: 0.0
    max_tokens: 512
    reasoning_effort: ${HOTEL_LLM_REASONING:-none}  # thinking off (Ollama)
    verify_ssl: ${HOTEL_LLM_VERIFY_SSL:-true}

functions:
  room_temperature:          { _type: room_temperature, comfort_band_c: 1.5 }
  create_maintenance_ticket: { _type: create_maintenance_ticket, ticket_log: ${HOTEL_TICKET_LOG:-.runs/tickets.jsonl} }
  current_datetime:          { _type: current_datetime }

workflow:
  _type: tool_calling_agent
  llm_name: hotel_llm
  tool_names: [room_temperature, create_maintenance_ticket, current_datetime]
  max_iterations: 6
  system_prompt: |
    You are the night-shift operations assistant of a small hotel. …
```

Four rules explain every NAT config you will read:

1. **`_type` picks a registered component.** `openai`, `tool_calling_agent` and `current_datetime` ship with NAT; `room_temperature` comes from your plugin. `nat info components` lists them all.
2. **Names connect sections.** `llm_name: hotel_llm` points at a key under `llms`; each `tool_names` entry points at a key under `functions` (or `function_groups`).
3. **`${VAR:-default}` is expanded when NAT loads the file.** One file serves the laptop and the Spark; you change the URL with an environment variable, or with `--override llms.hotel_llm.base_url …`.
4. **`base: other.yml` inherits.** NAT loads the other file first and merges this one on top. The Spark variant and the eval config use it.

**Thinking models need a switch.** `nemotron-3-nano`, `gemma4` and the playbook's Qwen3.6 all "think" before answering, and the thinking can use the whole `max_tokens` budget. The switch differs per server:

| Server | How to turn thinking off | Where |
|---|---|---|
| Ollama (laptop stand-in) | `reasoning_effort: none` | `hotel_agent.yml` (tested on this Mac) |
| vLLM + Qwen3.6 (the Spark) | `extra_body.chat_template_kwargs.enable_thinking: false`, and `reasoning_effort: null` to drop the Ollama field | `hotel_agent_spark.yml` (checked with `nat validate` and by building the client; not yet run against a Spark) |

The OpenAI LLM config in NAT accepts extra fields and passes them to LangChain's `ChatOpenAI`, which is why both work. The Spark variant is only this:

```yaml
base: hotel_agent.yml
llms:
  hotel_llm:
    model_name: ${HOTEL_LLM_MODEL:-nvidia/Qwen3.6-35B-A3B-NVFP4}
    max_tokens: 1024
    reasoning_effort: null
    extra_body:
      chat_template_kwargs:
        enable_thinking: false
```

✓ Checkpoint: you can point at the line that decides which server the agent calls, and explain why the Spark variant sets `reasoning_effort: null`.

## 4 · Write your own tool: a NAT plugin package

NAT finds components through Python **entry points**, not by importing files you name in the YAML. So a custom tool lives in a small package. `nat workflow create <name>` generates one; `hotel_ops_nat/` has the same shape:

```text
hotel_ops_nat/
  pyproject.toml                 [project.entry-points.'nat.components']  hotel_ops_nat = "hotel_ops_nat.register"
  src/hotel_ops_nat/register.py  imports tools.py, so the decorators run
  src/hotel_ops_nat/tools.py     two @register_function tools (fake, deterministic)
  src/hotel_ops_nat/evals.py     an offline evaluator for nat eval (Section 7)
```

One tool, from `tools.py`:

```python
from nat.plugin_api import Builder, FunctionBaseConfig, FunctionInfo, LLMFrameworkEnum, register_function

class RoomTemperatureConfig(FunctionBaseConfig, name="room_temperature"):     # name = the YAML _type
    """Read the current temperature, setpoint and HVAC status of one hotel room (fake data)."""
    comfort_band_c: float = Field(default=1.5, description="±°C around the setpoint that counts as comfortable.")

@register_function(config_type=RoomTemperatureConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def room_temperature_function(config: RoomTemperatureConfig, builder: Builder):
    async def _room_temperature(room: str) -> str:
        """Get the current temperature, setpoint and HVAC status of a hotel room. …"""
        return read_room(room, config.comfort_band_c)
    yield FunctionInfo.from_fn(_room_temperature, description=_room_temperature.__doc__)
```

Three things to notice:

- **Config class = YAML block.** Fields on `RoomTemperatureConfig` become keys the YAML may set (`comfort_band_c`). NAT validates them.
- **Type hints and docstring become the tool schema** the model sees. Write them for the model: say what the argument looks like ("808").
- **The tools are safe by design.** `room_temperature` reads a four-room table; `create_maintenance_ticket` returns `MT-` plus a hash of room and issue, so the same fault always gets the same id, and it only writes a local JSONL file you choose. An agent can call them as often as it likes. Swap the bodies for your building system's API later; the YAML does not change.

Install it once with `uv pip install … --no-deps -e week25/14_nat_agents/hotel_ops_nat` (Section 2). It is editable: change `tools.py`, and the next `nat run` uses the new code.

✓ Checkpoint: you can name the three places a new tool must appear: a config class with a `name=`, a `@register_function`, and an import in `register.py`.

## 5 · Run the agent, and read its trace

`nat run` is the console front end: one question in, one answer out. On the Spark, point the Spark variant at vLLM on `:8000`:

```bash
# on: spark
cd ~/agenticaicodingfitness/week25/14_nat_agents
HOTEL_LLM_BASE_URL=http://localhost:8000/v1 ~/w25/.venv-nat/bin/nat run \
  --config_file configs/hotel_agent_spark.yml --input "Room 808 feels hot. Check it and open a ticket if something is wrong."
```

Lab 14-2 does the same from the laptop. It picks vLLM on the Spark if it answers, then Ollama on the Spark, then the laptop stand-in, and prints which one it used:

```bash
# on: laptop
.venv/bin/python week25/14_nat_agents/labs/lab14_2_run_agent.py
```

**Expected output** (captured on this Mac — LAPTOP STAND-IN, `nemotron-3-nano:latest` on Ollama)

```
▣ STEP 1 · where does the model run?
◆ Ollama on THIS laptop — LAPTOP STAND-IN, not the Spark · base_url=http://localhost:11434/v1 · model=nemotron-3-nano:latest · config=configs/hotel_agent.yml

▣ STEP 2 · nat run · “Room 808 feels hot. Check it and open a ticket if something is wrong.”
→ tool_call room_temperature({"room": "808"})
  ← Room 808: 27.9 °C, setpoint 23.0 °C (+4.9 °C, too warm). HVAC status: fault: fan coil unit not responding.
→ tool_call create_maintenance_ticket({"room": "808", "issue": "HVAC fault: fan coil unit not responding", "priority": "high"})
  ← Created ticket MT-A83AD2 for room 808 (priority high): HVAC fault: fan coil unit not responding
· ANSWER  Room 808 is too warm (27.9 °C vs. 23 °C setpoint) with a fan coil unit fault. Created maintenance ticket MT-A83AD2 (high priority).

▣ STEP 3 · the trace NAT wrote — one JSON line per event (general.telemetry.tracing.local_file)
│ t        event           name                       input (START) / output (END)
│ ───────  ──────────────  ─────────────────────────  ────────────────────────────────────────────────────
│   0.00s  WORKFLOW_START  tool_calling_agent         Room 808 feels hot. Check it and open a ticket if s…
│   0.00s  FUNCTION_START  <workflow>                 Room 808 feels hot. Check it and open a ticket if s…
│   8.33s  FUNCTION_START  room_temperature           {'room': '808'}
│   8.33s  FUNCTION_END    room_temperature           Room 808: 27.9 °C, setpoint 23.0 °C (+4.9 °C, too w…
│  22.98s  FUNCTION_START  create_maintenance_ticket  {'room': '808', 'issue': 'HVAC fault: fan coil unit…
│  22.99s  FUNCTION_END    create_maintenance_ticket  Created ticket MT-A83AD2 for room 808 (priority hig…
│  36.71s  FUNCTION_END    <workflow>                 Room 808 is too warm (27.9 °C vs. 23 °C setpoint) w…
```

The model made two tool calls on its own: it read the room first, saw the fault, then opened a ticket, exactly as the system prompt asked. The timings are this laptop's (another agent was using the same Ollama); do not compare them with a Spark.

**Observability** is one block at the top of the workflow file. The `file` exporter writes the trace you just read; the other exporters installed with NAT (`otelcollector`, `langfuse`, and more with extras such as `nvidia-nat[phoenix]`) send the same events to a tracing UI:

```yaml
general:
  telemetry:
    tracing:
      local_file:
        _type: file
        output_path: ${HOTEL_TRACE_FILE:-.runs/nat_trace.jsonl}
        project: hotel-ops
        mode: overwrite
```

✓ Checkpoint: your run shows at least one `→ tool_call` line, and the trace table shows the tool's START and END events.

## 6 · Serve it: a REST / OpenAI endpoint, and an MCP server

The same YAML file runs behind two more front ends.

**`nat serve`** starts a FastAPI server. It exposes `/generate`, `/chat`, streaming variants, and an OpenAI-compatible **`/v1/chat/completions`**, so Open WebUI or LiteLLM (Module 08) can treat the whole agent as if it were a model. The course uses port **8400**, because `:8000` is vLLM's port on the Spark. On the Spark, bind to `0.0.0.0` only if you want the tailnet to reach it; `nat serve` has no authentication by default.

```bash
# on: spark
cd ~/agenticaicodingfitness/week25/14_nat_agents
HOTEL_LLM_BASE_URL=http://localhost:8000/v1 ~/w25/.venv-nat/bin/nat serve \
  --config_file configs/hotel_agent_spark.yml --host 127.0.0.1 --port 8400
```

**`nat mcp serve`** publishes functions as **MCP tools** (streamable HTTP on `:9901/mcp` by default). `--tool_names` limits it to the two hotel tools. Any MCP client can then list and call them; `configs/hotel_agent_mcp.yml` is a NAT agent whose tools come from that server through `function_groups: {_type: mcp_client}`.

Lab 14-3 starts both servers, calls them, and stops them:

```bash
# on: laptop
.venv/bin/python week25/14_nat_agents/labs/lab14_3_serve_and_mcp.py
```

**Expected output** (captured on this Mac — LAPTOP STAND-IN, trimmed)

```
▣ STEP 1 · nat serve — the agent as a web service on 127.0.0.1:8400
→ GET /health → {"status": "healthy"}
│ route NAT created     methods
│ ────────────────────  ───────
│ /generate             POST
│ /generate/stream      POST
│ /v1/chat/completions  POST
│ /chat                 POST
│ /v1/workflow          POST
│ /evaluate/item        POST
│ /health               GET
◆ 19 routes in total — see http://127.0.0.1:8400/docs while it runs.

▣ STEP 2 · call it like a model: POST /v1/chat/completions (one request)
→ POST http://127.0.0.1:8400/v1/chat/completions  {"model": "hotel-agent", "messages": [{"role": "user", "content": "Room 1510 guest says it is cold. Check it."}]}
· ANSWER  Room 1510 is indeed too cold (20.1 °C vs. 22 °C setpoint) with the HVAC idle due to guest‑away mode. A high‑priority maintenance ticket **MT-64DDF0** has been created for this issue.
◆ LAPTOP STAND-IN (not Spark numbers) · object=chat.completion · finish_reason=stop · 12.7s for the whole agent loop (LLM calls + tool calls)

▣ STEP 4 · any MCP client can list and call them — here NAT's own client, no LLM involved
$ ../.venv-nat/bin/nat mcp client tool list --url http://localhost:9901/mcp
create_maintenance_ticket
room_temperature
$ ../.venv-nat/bin/nat mcp client tool call room_temperature --url http://localhost:9901/mcp --json-args {"room": "808"}
Room 808: 27.9 °C, setpoint 23.0 °C (+4.9 °C, too warm). HVAC status: fault: fan coil unit not responding.
```

Look at the 1510 answer: the room is in "guest away mode", which is arguably fine, yet this small model opened a ticket anyway. That is a judgement call the prompt did not settle, and exactly the kind of behaviour Section 7 measures.

✓ Checkpoint: `/v1/chat/completions` returned `object=chat.completion`, and `tool list` showed both hotel tools.

## 7 · Evaluate it: `nat eval`

An agent eval has three parts: a dataset, the workflow, and one or more evaluators. `configs/hotel_eval.yml` inherits the agent (`base: hotel_agent.yml`) and adds:

```yaml
eval:
  general:
    output_dir: .runs/eval
    max_concurrency: 1
    dataset:
      _type: json
      file_path: configs/eval_dataset.json
  evaluators:
    mentions_expected:
      _type: langsmith_custom
      evaluator: hotel_ops_nat.evals.mentions_expected
```

Each dataset row has a `question` and an `answer` that lists facts the reply must contain (`"808|MT-"`). `mentions_expected` scores the fraction of those facts found: deterministic, free, no LLM judge. NAT 1.9 also ships `trajectory` (an LLM judges the tool-call path) and `langsmith` (`exact_match`, `levenshtein_distance`); add them under `evaluators` when you have a judge model on the Spark.

```bash
# on: laptop
.venv/bin/python week25/14_nat_agents/labs/lab14_4_eval.py
```

**Expected output** (captured on this Mac — LAPTOP STAND-IN)

```
=== EVALUATION SUMMARY ===
Workflow Status: COMPLETED (workflow_output.json)
Total Runtime: 128.22s
Per evaluator results:
| Evaluator         |   Avg Score | Output File                   |
|-------------------|-------------|-------------------------------|
| mentions_expected |           1 | mentions_expected_output.json |

▣ STEP 3 · per question: what the agent said, and what the evaluator found
◆ hot_808: found ['808', 'mt-'] · missing []
◆ ok_1203: found ['1203', '23.5'] · missing []
◆ cold_1510: found ['1510', '20.1'] · missing []
```

A perfect score on three questions proves the plumbing, not the agent. The value comes when you change one thing and rerun: the base model against your fine-tune (Module 13), `tool_calling_agent` against `react_agent`, a longer system prompt. NAT saves `config_effective.yml` next to the results, so each score records exactly what produced it.

✓ Checkpoint: `mentions_expected_output.json` exists under `week25/14_nat_agents/.runs/eval/`, with one score per question.

## Labs — run them here

**labs/lab14_1_install_and_register.py** — Find the NAT CLI, register the hotel_ops_nat plugin, list NAT's components and validate all four workflow files.

**labs/lab14_2_run_agent.py** — Run the hotel agent with `nat run` on the Spark's vLLM or the laptop stand-in, stream its tool calls and print the trace.

**labs/lab14_3_serve_and_mcp.py** — Serve the agent over REST and the OpenAI API with `nat serve`, then publish its tools with `nat mcp serve` and call them.

**labs/lab14_4_eval.py** — Score the agent on a three-question dataset with `nat eval` and a deterministic evaluator.

All four run on the laptop against the laptop Ollama when no Spark answers, and label that output LAPTOP STAND-IN. With a Spark connected and vLLM up, labs 14-2 to 14-4 use it automatically.

## Try it yourself

**Exercise 14 — wire the agent yourself.** Open `week25/14_nat_agents/exercises/ex14_wire_the_agent.py`. Each of its three `TODO`s returns one section of a NAT workflow as a Python dict:

1. `llm_block(target, base_url, model)`: the `llms.hotel_llm` block, with thinking turned off the right way for Ollama and for vLLM.
2. `functions_block(ticket_log)`: the three tools, each `_type` equal to its registered name.
3. `workflow_block(llm_name, functions)`: a `tool_calling_agent` that uses every function.

The checker is offline. It lints your config, writes it as YAML, and then asks the real `nat validate` whether NAT accepts it (no model is called).

```bash
# on: laptop
.venv/bin/python week25/14_nat_agents/exercises/ex14_wire_the_agent.py
```

**Expected output** (once all three TODOs are done, captured on this Mac)

```
✓ llm_block: openai type · Ollama thinks off via reasoning_effort · vLLM via chat_template_kwargs
✓ functions_block: 3 tools, each `_type` = its registered name, ticket_log passed through
✓ workflow_block: tool_calling_agent · every function as a tool · max_iterations ≤ 6 · system prompt

▣ your config, assembled and linted for both targets
✓ lint ollama: no problems → week25/14_nat_agents/.runs/ex14_ollama.yml
✓ lint vllm  : no problems → week25/14_nat_agents/.runs/ex14_vllm.yml

▣ the real test: does NeMo Agent Toolkit accept it? (`nat validate`, offline)
✓ nat validate ex14_ollama.yml: valid
✓ nat validate ex14_vllm.yml: valid
```

<details><summary>Hint — why does the vLLM block not have reasoning_effort?</summary>

`reasoning_effort` is an OpenAI parameter that Ollama understands. vLLM with a Qwen3-family chat template turns thinking off through the template instead: `chat_template_kwargs: {enable_thinking: false}`, which the OpenAI client sends in `extra_body`. Sending a parameter the server does not expect can fail the request, so each block carries only its own switch.

</details>

<details><summary>Stretch — a third tool, and a different agent</summary>

Add `guest_request(room, request)` to `hotel_ops_nat/src/hotel_ops_nat/tools.py` (config class, `@register_function`, import in `register.py`), add it under `functions`, and rerun lab 14-2 with "Room 402 needs extra towels". Then change `workflow._type` to `react_agent` and compare the trace: a ReAct agent writes Thought / Action text that NAT parses, while the tool-calling agent uses the server's native tool calls.

</details>

✓ Checkpoint: all checker lines are ✓, including both `nat validate` lines.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `nat serve` fails with `The SQLAlchemy asyncio module requires that the Python 'greenlet' library is installed` | `uv pip install --python week25/.venv-nat/bin/python greenlet` (Section 2 installs it) |
| `nat mcp` is not a command, or `mcp_client` is unknown | Install the MCP extra: `nvidia-nat-mcp~=1.9` in the same venv |
| `ValidationError … _type 'room_temperature'` | The plugin is not installed in the venv that runs `nat`. Rerun lab 14-1, or `uv pip install --no-deps -e week25/14_nat_agents/hotel_ops_nat` |
| Tool changes do not show up | Install with `-e` (editable). A non-editable install copies the code once |
| The agent answers without calling a tool, or returns an empty answer | A thinking model spent `max_tokens` on reasoning. Check the switch in Section 3, or raise `max_tokens` |
| `Connection refused` to `:8000` from the laptop | vLLM is on the Spark: use `http://spark-a:8000/v1`, or an `ssh -L 8000:localhost:8000` tunnel (Module 01) |
| AuthlibDeprecationWarning on every command | Harmless in NAT 1.9.0; the labs set `PYTHONWARNINGS=ignore` and filter it |
| Relative paths (`.runs/…`, `configs/eval_dataset.json`) not found | Run `nat` from `week25/14_nat_agents`, as every command here does |

## Next

Continue to [Lab 15 — OpenShell: sandbox and govern AI agents](../15_openshell_sandbox/TUTORIAL.md): put this NAT agent inside an OpenShell sandbox on the Spark, where a policy decides which files it can touch and which hosts it can reach.
