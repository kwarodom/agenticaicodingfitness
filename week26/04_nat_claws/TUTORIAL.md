# ▶ Reef Lab 04 — Build a claw with NeMo Agent Toolkit: tools, serving, MCP, sandboxed NAT

> Part of Week 26 · NemoClaw on DGX Spark, beginner to expert. You type the commands, you see the real output. Laptop labs (the NAT and OpenShell CLIs, the policy model) run for real everywhere. Spark labs also run in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's docs or playbooks, or a clearly marked EXAMPLE.

**What you'll actually do**
- Run a NeMo Agent Toolkit (NAT) 1.9 workflow for real on this laptop, against a local Nemotron model in Ollama.
- Read, test and call a custom NAT tool (`chiller_kpi`) without an LLM, then watch one sentence in its description change what the agent says.
- Serve the same YAML as a REST API, an OpenAI-compatible endpoint and a step-by-step stream, and count the LLM and tool calls behind one answer.
- Publish tools over MCP, and consume a (mock) building-management system over MCP with only the safe tools switched on.
- Build the context for NAT inside an OpenShell sandbox: a Dockerfile, a workflow that calls `inference.local`, and a deny-by-default policy the real parser accepts.

**Time** ~120 min · **Difficulty** intermediate · **Hardware** laptop (NAT 1.9 + Ollama) · 1 DGX Spark optional

**Sources:** the course's research tutorial *NemoClaw on DGX Spark — Beginner to Expert* (Part 3, labs L3.1–L3.8), which cites [NeMo Agent Toolkit docs](https://docs.nvidia.com/nemo/agent-toolkit/latest/) · [NAT writing custom functions](https://docs.nvidia.com/nemo/agent-toolkit/latest/extend/functions.html) · [NAT tool calling agent](https://docs.nvidia.com/nemo/agent-toolkit/latest/components/agents/tool-calling-agent/tool-calling-agent.html) · [NAT API server endpoints](https://docs.nvidia.com/nemo/agent-toolkit/latest/reference/rest-api/api-server-endpoints.html) · [NAT MCP server](https://docs.nvidia.com/nemo/agent-toolkit/latest/run-workflows/mcp-server.html) · [NAT MCP client](https://docs.nvidia.com/nemo/agent-toolkit/latest/build-workflows/mcp-client.html) · [NAT code execution](https://docs.nvidia.com/nemo/agent-toolkit/latest/components/functions/code-execution.html) · [Classmethod — NAT on DGX Spark](https://dev.classmethod.jp/en/articles/dgx-spark-nemo-agent-toolkit-local-intro/) · [OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies) · [OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| NAT 1.9 on this laptop | `week26/.venv-nat/bin/nat --version` | every lab runs the real NAT CLI |
| A tool-calling model in laptop Ollama | `nemotron-3-nano:latest` in `curl -s localhost:11434/v1/models` | the LAPTOP STAND-IN for vLLM on the Spark (~10–20 s per agent question) |
| `greenlet` in the NAT venv | `week26/.venv-nat/bin/python -c "import greenlet"` | `nat serve` on NAT 1.9.0 needs it, and a fresh install can miss it (Section 5) |
| A DGX Spark (optional) | `ssh -o BatchMode=yes <spark> true` | the vLLM and sandbox steps; without one they run DRY |

```bash
# on: laptop
week26/.venv-nat/bin/nat --version 2>/dev/null
curl -s http://localhost:11434/v1/models | grep -o '"nemotron-3-nano[^"]*"'
week26/.venv-nat/bin/python -c "import greenlet; print('greenlet', greenlet.__version__)" 2>&1 | tail -1
```

**Expected output** (captured on this Mac)

```
nat, version 1.9.0
"nemotron-3-nano:latest"
greenlet 3.5.6
```

If the last line says `ModuleNotFoundError: No module named 'greenlet'`, run `uv pip install --python week26/.venv-nat/bin/python greenlet` once. This course's venv first shipped without it (Section 5 explains why). Only `nat serve` needs it.

> 📌 **Versions.** The research tutorial was written against the NAT **1.8** docs. This course runs NAT **1.9.0**, and this module checked every command, YAML key and endpoint against the installed package. Where 1.9 differs, the section says so in a **1.9 differs** note. Run one lab at a time if you can: other labs share the same Ollama, so times vary.

✓ Checkpoint: `nat --version` prints 1.9.0 and Ollama lists `nemotron-3-nano:latest`.

## 1 · L3.1 — NAT in one page, and installing it on the Spark

NAT (`nvidia-nat`) is a framework-agnostic layer for agents. Every agent, tool and workflow is a **function**. One YAML file wires them together, and `_type` picks the implementation:

| YAML section | Holds | Example in this module |
|---|---|---|
| `functions` | tools, and agents used as tools | `chiller_kpi`, `current_datetime` |
| `function_groups` | a bundle of tools from one source | `bms_tools` (`_type: mcp_client`) |
| `llms` | model endpoints | `_type: openai` + `base_url` |
| `workflow` | the entry point | `_type: tool_calling_agent` or `react_agent` |

Why NAT in a NemoClaw course? OpenClaw, Hermes and Deep Agents are general assistants. A purpose-built claw, such as **Alto Ops Claw**, needs explicit tools, deterministic evaluation and profiling. NAT gives you those, and the same workflow runs on the Spark host (calling vLLM) or inside an OpenShell sandbox (calling `inference.local`). Built-in agent types include ReAct, Reasoning, ReWOO, Router and Tool Calling.

On the Spark, NAT is a pure-Python wheel, so it installs on arm64 without compiling anything. The research tutorial follows Classmethod's DGX Spark write-up: Python 3.12 in a uv venv.

```bash
# on: spark
mkdir -p ~/works/alto-ops-claw && cd ~/works/alto-ops-claw
uv venv --python 3.12 && source .venv/bin/activate
uv pip install 'nvidia-nat[langchain,mcp,profiler,phoenix,opentelemetry]'
nat --version
```

The extras you will meet: `langchain` (the ReAct and tool-calling agents), `mcp` (client and server), `profiler` (for `nat eval` profiling and the sizing calculator), `phoenix` and `opentelemetry` (tracing, Module 05). This laptop's venv also has `eval` and `ragas` (Module 06).

✓ Checkpoint: you can name the four YAML sections and say which one a MCP server's tools go into (`function_groups`).

## 2 · L3.2 — A local vLLM for NAT on the Spark

On the Spark, NAT talks to a local vLLM server. The research tutorial starts the Nemotron 3 Nano model that Classmethod validated on the Spark:

```bash
# on: spark
docker run -d --name vllm-nat --gpus all --shm-size=16g -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  nvcr.io/nvidia/vllm:26.01-py3 \
  vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
  --trust-remote-code --max-model-len 8192 --gpu-memory-utilization 0.85
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

Nemotron 3 Nano uses the hybrid Mamba–Transformer `nemotron_h` architecture, so it needs `--trust-remote-code`. Without that flag, the server crashes on a pydantic `ValidationError`. Per Classmethod, cited in the research tutorial, loading the 30.5 GiB model took about four minutes, plus two minutes of `torch.compile`, and 34.79 GiB went to KV cache at 8K context.

This laptop has no GPU for that, so the labs use **Ollama as a LAPTOP STAND-IN**. Only two lines of the YAML change:

| Key | On the Spark (vLLM) | On this laptop (Ollama) |
|---|---|---|
| `base_url` | `http://localhost:8000/v1` | `http://localhost:11434/v1` |
| `model_name` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` | `nemotron-3-nano:latest` |

Laptop timings in this module describe the laptop only. They are never Spark numbers.

✓ Checkpoint: you can explain why the vLLM command needs `--trust-remote-code`, and which two YAML keys move a workflow from the Spark to the laptop.

## 3 · L3.3 — Hello, local workflow

Start with a scaffold. On NAT 1.9, `--workflow-dir` must already exist, so create it first:

```bash
# on: laptop
mkdir -p week26/04_nat_claws/.runs/workflows
week26/.venv-nat/bin/nat workflow create --no-install --workflow-dir week26/04_nat_claws/.runs/workflows alto_ops --description "Alto Ops Claw" 2>/dev/null
find week26/04_nat_claws/.runs/workflows/alto_ops -type f | sort
grep '^from nat' week26/04_nat_claws/.runs/workflows/alto_ops/src/alto_ops/alto_ops.py
```

**Expected output** (captured on this Mac)

```
Workflow 'alto_ops' created successfully in '/Users/altodev/Desktop/agenticaicodingfitness/week26/04_nat_claws/.runs/workflows/alto_ops'.
week26/04_nat_claws/.runs/workflows/alto_ops/pyproject.toml
week26/04_nat_claws/.runs/workflows/alto_ops/src/alto_ops/__init__.py
week26/04_nat_claws/.runs/workflows/alto_ops/src/alto_ops/alto_ops.py
week26/04_nat_claws/.runs/workflows/alto_ops/src/alto_ops/configs/config.yml
week26/04_nat_claws/.runs/workflows/alto_ops/src/alto_ops/register.py
from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import LLMFrameworkEnum
from nat.plugin_api import register_function
```

> **1.9 differs.** The research tutorial's `--workflow-dir ./workflows` fails on a fresh folder with `Invalid workflow directory specified. … does not exist.` The 1.9 scaffold imports everything from one facade, `nat.plugin_api`, and its `config.yml` uses `_type: nim` (a hosted NIM). You replace that `llms` block with a local server.

Now the minimal workflow. `week26/04_nat_claws/configs/hello.laptop.yml` is the research tutorial's YAML with the two laptop lines from Section 2. The `api_key` field is required by validation even for keyless servers, so it holds a dummy value, `EMPTY`. Validate before you run: it costs no LLM call.

```bash
# on: laptop
cat week26/04_nat_claws/configs/hello.laptop.yml
week26/.venv-nat/bin/nat validate --config_file week26/04_nat_claws/configs/hello.laptop.yml 2>/dev/null | head -2
week26/.venv-nat/bin/nat run --config_file week26/04_nat_claws/configs/hello.laptop.yml --input "What time is it in Bangkok right now?" 2>&1 | tail -1
```

**Expected output** (captured on this Mac, the last two commands)

```
Validating configuration file: week26/04_nat_claws/configs/hello.laptop.yml
✓ Configuration file is valid!
Error: ReActAgentParsingFailedError: Failed to parse agent output after 1 attempts. Error: Invalid Format: Missing 'Action:' after 'Thought:'. LLM output: ''
```

The YAML is fine. The model is the issue. `nemotron-3-nano` is a **thinking model**: on Ollama its thoughts go to a separate `reasoning` field, so the ReAct agent, which parses `Thought:` and `Action:` out of the answer text, gets an empty string. `hello.laptop.nothink.yml` adds one line under the LLM, `reasoning_effort: none`. NAT's `openai` LLM config accepts extra keys and passes them to the server.

```bash
# on: laptop
week26/.venv-nat/bin/nat run --config_file week26/04_nat_claws/configs/hello.laptop.nothink.yml --input "What time is it in Bangkok right now?" 2>&1 | grep -A1 -E "Action:|Tool's response|Workflow Result"
```

**Expected output** (captured on this Mac)

```
Action: current_datetime
Action Input: {"unused": "FieldInfo(annotation=str, required=True)"}
--
Tool's response: 
The current time of day is 2026-09-30 03:58:21 +0000
--
Workflow Result:
The current time in Bangkok is 2026-09-30 03:58:21 +0000.
```

NAT writes its agent log and the result to **stderr**, so the pipe needs `2>&1`.

Look at the offset. `current_datetime` returns UTC (`+0000`), and Bangkok is UTC+7. The agent passed the tool's answer straight through as "Bangkok time". The loop worked, and the answer is still wrong. Keep that in mind for every claw you build.

The `tool_calling_agent` in the next sections does not have this problem: it uses the model's native function calling instead of parsing text.

✓ Checkpoint: you can say why the first run failed (a thinking model gave the ReAct parser empty text) and what is still wrong with the second answer (UTC reported as Bangkok time).

## 4 · L3.4 — A custom tool: chiller-plant CSV analytics

Alto Ops Claw's first real tool reads a chiller-plant CSV export and returns the plant's kW per refrigeration ton (kW/RT). Lower is better. The course package is already installed in the NAT venv: `week26/common/alto_ops/src/alto_ops/chiller_tool.py`. It is the research tutorial's Lab 3.4 code with one change in the description.

Three NAT pieces make a function:

| Piece | What it does |
|---|---|
| `class ChillerKpiConfig(FunctionBaseConfig, name="chiller_kpi")` | the YAML `_type:` and its typed fields (`csv_path`, `kw_per_rt_alarm`) |
| `@register_function(config_type=ChillerKpiConfig)` | registers the type when the module is imported |
| `yield FunctionInfo.from_fn(_kpi, description=…)` | the callable the agent gets, and the description it reads |

NAT finds the package through the `nat.components` entry point in `pyproject.toml`, which imports `alto_ops.register`. The research tutorial asks you to confirm the import paths on your version. On 1.9.0, both the long paths and the new `nat.plugin_api` facade work, and they are the same objects:

```bash
# on: laptop
week26/.venv-nat/bin/python week26/04_nat_claws/tools/nat_direct.py imports
```

**Expected output** (captured on this Mac)

```
nat.plugin_api.Builder is nat.builder.builder.Builder: True
nat.plugin_api.FunctionInfo is nat.builder.function_info.FunctionInfo: True
nat.plugin_api.register_function is nat.cli.register_workflow.register_function: True
nat.plugin_api.FunctionBaseConfig is nat.data_models.function.FunctionBaseConfig: True
```

**Get the ground truth before you ask an agent.** `nat_direct.py` builds just the tool from the workflow YAML and calls it, with no LLM:

```bash
# on: laptop
week26/.venv-nat/bin/python week26/04_nat_claws/tools/nat_direct.py tool week26/common/alto_ops/src/alto_ops/configs/workflow.laptop.yml chiller_kpi '{"hours": 6}' '{"hours": 24}' 2>/dev/null
```

**Expected output** (captured on this Mac)

```
description: Average chiller plant kW, RT (refrigeration tons of cooling) and kW/RT over the last N hours; flags efficiency alarms.
input schema: {"hours": {"default": 24, "title": "Hours", "type": "integer"}}
chiller_kpi({"hours": 6}) -> window=6h avg_kw=473.6 avg_rt=525.9 kw_per_rt=0.901 status=ALARM
chiller_kpi({"hours": 24}) -> window=24h avg_kw=442.5 avg_rt=629.2 kw_per_rt=0.703 status=OK
```

The data is synthetic: 672 rows at 15 minutes, with the last six hours degraded on purpose. The YAML sets the alarm at 0.80 kW/RT, so 6 h is an ALARM and 24 h is OK.

**A tool description is a prompt.** Lab 04-2 asks the `tool_calling_agent` the same question twice. The first run uses `tools/chiller_v0.py`, the research tutorial's tool with its original description: *"Average chiller plant kW, RT and kW/RT…"*. The second uses the course's description, which adds *"RT (refrigeration tons of cooling)"*.

**Expected output** (captured on this Mac, lab 04-2 step 7 — LAPTOP STAND-IN answers)

```
│ tool description      kW/RT in answer  what RT means                       verdict  s (LAPTOP STAND-IN)
│ ────────────────────  ───────────────  ──────────────────────────────────  ───────  ───────────────────
│ tutorial description  ✓ 0.901          ✕ 'refrigerant/return temperature'  ✓ alarm  14.1
│ course description    ✓ 0.901          ✓ refrigeration tons                ✓ alarm  15.2
✓ same model, same data, same number — only the description changed, and so did the meaning of RT
```

The first answer said *"an average refrigerant temperature (RT) of 525.9"*. The second said *"525.9 refrigeration tons (RT) of cooling"*. The model only sees a tool's name, description and input schema, and `avg_rt=525.9` means nothing to it without context. This build hit that exact bug, which is why the course package carries the longer description. Model answers are stochastic, so a rerun may differ. The fix does not.

One more pattern from the research tutorial: an agent can be a tool of another agent, in YAML only. A `functions:` entry with `_type: tool_calling_agent`, its own `tool_names` and a `description` becomes a callable tool.

✓ Checkpoint: you can compute the 6-hour kW/RT without an LLM, and explain why one clause in a description changed the agent's answer.

## 5 · L3.5 — Serve it: REST and OpenAI-compatible endpoints

`nat serve` turns the same YAML into a FastAPI server. Its default port is 8000, which is vLLM's port on the Spark, so pass `--port 8001` (the research tutorial's curls use 8001 too).

> **1.9 differs.** NAT 1.9.0 pulls in SQLAlchemy 2.1, which no longer installs `greenlet` by default, yet `nat serve` imports SQLAlchemy's asyncio extension at start-up. Without `greenlet` it stops with *"The SQLAlchemy asyncio module requires that the Python 'greenlet' library is installed"*. This course's venv hit exactly that; the fix is one line, `uv pip install --python week26/.venv-nat/bin/python greenlet`, and it is now installed. If it is missing on your machine, lab 04-3 borrows the `greenlet` from `week25/.venv-nat` through `PYTHONPATH` (and says so), or prints the fix. `nat run` and `nat mcp serve` do not need it.

```bash
# on: laptop
week26/.venv-nat/bin/nat serve --config_file week26/common/alto_ops/src/alto_ops/configs/workflow.laptop.yml --host 127.0.0.1 --port 8001
```

In a second terminal:

```bash
# on: laptop
curl -s http://localhost:8001/health
curl -s -X POST http://localhost:8001/v1/workflow -H 'Content-Type: application/json' \
  -d '{"input_message":"Is the chiller plant efficient over the last 6 hours?"}'
curl -s -X POST http://localhost:8001/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"alto-ops","messages":[{"role":"user","content":"Plant status?"}],"stream":false}'
curl -s -X POST 'http://localhost:8001/v1/workflow/full?filter_steps=LLM_END,TOOL_END' \
  -H 'Content-Type: application/json' -d '{"input_message":"Is the chiller plant efficient over the last 6 hours?"}'
```

Lab 04-3 makes the same calls with Python's `urllib`, in the request shape the OpenAI client sends. These are the response shapes on 1.9.0:

| Endpoint | Response shape (captured on this Mac) |
|---|---|
| `GET /health` | `{"status": "healthy"}` |
| `POST /v1/workflow` | `{"value": "<answer>"}` |
| `POST /v1/chat/completions` | `object=chat.completion`, `model='unknown-model'` (the `model` you send is not echoed), `usage.prompt_tokens` 0 |
| `POST /v1/workflow/full` | a server-sent-event stream: `data: {"value": …}` answer chunks plus `intermediate_data: {…}` steps |

**Expected output** (captured on this Mac, lab 04-3 steps 2 and 5)

```
◆ in the tutorial's list but NOT served here: /v1/workflow/async, /feedback, /monitor/users
◆ served here but not in the tutorial's list: /executions/{execution_id}, /executions/{execution_id}/interactions/{interaction_id}/response, /health, /v1/workflow/atif, /generate/stream, /generate/full, /chat/stream, /evaluate/item, /mcp/client/tool/list, /mcp/client/tool/list/per_user
◆ it is a stream (server-sent events): 93 `data:` answer chunks + 3 `intermediate_data:` steps
│ step      name                    s     prompt tok  completion tok  output (start)
│ ────────  ──────────────────────  ────  ──────────  ──────────────  ────────────────────────────────────────────────────
│ LLM_END   nemotron-3-nano:latest  4.04  365         154                Tool calls: [{'name': 'chiller_kpi', 'args': {'h…
│ TOOL_END  chiller_kpi             0.01  0           0               {'content': 'window=6h avg_kw=473.6 avg_rt=525.9 kw…
│ LLM_END   nemotron-3-nano:latest  5.99  445         256             The chiller plant's efficiency over the last 6 hour…
◆ LAPTOP STAND-IN · 10.1s · LLM_END × 2 · TOOL_END × 1 · 1220 LLM tokens
✓ one question = 2 LLM call(s) + 1 tool call(s): plan → chiller_kpi → summarise
```

> **1.9 differs.** `/v1/workflow/async` needs the optional `dask` extra, which is not installed. `/monitor/users` appears only with `general.enable_per_user_monitoring: true`. `/feedback` is not registered by this config. New in 1.9: `/v1/workflow/atif`, `/executions/…` and `/mcp/client/tool/list`. Open `http://localhost:8001/docs` on your install to see its real list.

The step table is the most useful thing on this page. One question took two LLM calls (plan, then summarise) and one tool call. Nearly all the time is in the LLM, and the tool took 10 ms. Module 05 turns these steps into traces. The lab saves them to `.runs/lab04_3_full_steps.json`.

✓ Checkpoint: you can say how many LLM calls and tool calls one chiller question costs, and which endpoint told you.

## 6 · L3.6 — MCP both ways

**NAT as an MCP server** publishes Alto Ops tools to any MCP client: OpenClaw, Hermes, Claude Code, or another NAT. The default transport is streamable-HTTP on `/mcp`, port 9901.

```bash
# on: laptop
week26/.venv-nat/bin/nat mcp serve --config_file week26/common/alto_ops/src/alto_ops/configs/workflow.laptop.yml --name "Alto Ops MCP" --port 9901
```

In a second terminal:

```bash
# on: laptop
curl -s http://localhost:9901/debug/tools/list
week26/.venv-nat/bin/nat mcp client tool list --url http://localhost:9901/mcp 2>/dev/null
```

**Expected output** (captured on this Mac)

```
{"count":3,"tools":[{"name":"chiller_kpi","description":"Tool Calling Agent Workflow","is_workflow":true},{"name":"current_datetime","description":"Tool Calling Agent Workflow","is_workflow":true},{"name":"tool_calling_agent","description":"Tool Calling Agent Workflow","is_workflow":true}],"server_name":"Alto Ops MCP"}
current_datetime
chiller_kpi
tool_calling_agent
```

The list order changes from run to run.

Two things to notice. First, without `--tool_names` the **whole agent** is published too (`tool_calling_agent`), so any MCP client could run your LLM loop. Second, on 1.9.0 `/debug/tools/list` shows the workflow's description for every tool. Publish only what a caller needs:

```bash
# on: laptop
week26/.venv-nat/bin/nat mcp serve --config_file week26/common/alto_ops/src/alto_ops/configs/workflow.laptop.yml --port 9901 --tool_names chiller_kpi
```

In a second terminal:

```bash
# on: laptop
week26/.venv-nat/bin/nat mcp client tool call chiller_kpi --url http://localhost:9901/mcp --json-args '{"hours": 6}' 2>/dev/null | grep window
```

**Expected output** (captured on this Mac)

```
window=6h avg_kw=473.6 avg_rt=525.9 kw_per_rt=0.901 status=ALARM
```

**NAT as an MCP client** consumes someone else's tools. The course ships a mock BMS, `week26/common/bms_mcp_server.py`, with four tools: `read_point`, `list_alarms`, `get_trend` and `write_setpoint`. The data is synthetic, and `write_setpoint` never writes. `week26/04_nat_claws/configs/mcp_client_bms.yml` switches on only the two read tools:

```yaml
function_groups:
  bms_tools:
    _type: mcp_client
    server:
      transport: streamable-http
      url: ${BMS_MCP_URL:-http://localhost:8443/mcp}
    include: [read_point, list_alarms]      # write_setpoint and get_trend never reach the agent
    tool_call_timeout: 60
    reconnect_enabled: true
    reconnect_max_attempts: 3
    tool_overrides:
      read_point:
        description: "Read the latest value of one BMS point. Valid points: PLANT.KW, PLANT.RT (refrigeration tons of cooling), PLANT.KW_PER_RT."
```

Every key was checked against the installed 1.9.0 source (`nat/plugins/mcp/client/client_config.py`, `nat/data_models/function.py`, `nat/builder/function.py`):

| Key | On NAT 1.9.0 |
|---|---|
| `server.transport` | `stdio`, `sse` or `streamable-http` (the default) |
| `server.url` | required for `sse` and `streamable-http`; `auth_provider` and `custom_headers` also live under `server` |
| `include` / `exclude` | inherited from every function group; **using both is a validation error** |
| `tool_overrides.<tool>` | `alias` and `description`, keyed by the server's tool name |
| `tool_call_timeout` | seconds (a timedelta), default 60 |
| `reconnect_max_attempts` | default 2 (the research tutorial sets 3) |
| tool names | `<group>__<tool>`, such as `bms_tools__read_point` (the legacy separator was `.`) |

Start the mock BMS and ask NAT which tools the agent would get. No LLM is involved:

```bash
# on: laptop
BMS_MCP_PORT=8443 week26/.venv-nat/bin/python week26/common/bms_mcp_server.py
```

In a second terminal:

```bash
# on: laptop
BMS_MCP_URL=http://localhost:8443/mcp week26/.venv-nat/bin/python week26/04_nat_claws/tools/nat_direct.py group week26/04_nat_claws/configs/mcp_client_bms.yml bms_tools 2>/dev/null
```

**Expected output** (captured on this Mac)

```
config: include=['list_alarms', 'read_point'] exclude=[] tool_overrides=['read_point']
hidden      bms_tools__get_trend  | Hourly averages of PLANT.KW_PER_RT for the last N hours.
AGENT SEES  bms_tools__list_alarms  | List active plant alarms (kW/RT above 0.85 for the last hour).
AGENT SEES  bms_tools__read_point  | Read the latest value of one BMS point. Valid points: PLANT.KW, PLANT.RT (refrigeration tons of cooling), PLANT.KW_PER_RT.
hidden      bms_tools__write_setpoint  | WRITE a setpoint to equipment. Dangerous — policies should deny this tool. (Mock: never writes.)
```

The group still connects to all four tools. `include` decides which ones the agent is **offered**. Then one agent question:

```bash
# on: laptop
BMS_MCP_URL=http://localhost:8443/mcp week26/.venv-nat/bin/nat run --config_file week26/04_nat_claws/configs/mcp_client_bms.yml --input "Are there any active plant alarms right now, and what is the latest kW/RT reading?" 2>&1 | grep -A1 -E "Tool's response|Workflow Result"
```

**Expected output** (captured on this Mac — LAPTOP STAND-IN answer)

```
Tool's response: 
ALARM plant efficiency 0.901 kW/RT > 0.85 since 2026-09-27T23:00
--
Tool's response: 
PLANT.KW_PER_RT=0.897 at 2026-09-27T23:45 (mock BMS, synthetic data)
--
Workflow Result:
There is an active plant alarm: **plant efficiency 0.901 kW/RT > 0.85** (active since 2026-09-27T23:00).
```

The agent called `bms_tools__list_alarms` and `bms_tools__read_point` in one turn. In an earlier run the same model added *"(1 hour and 45 minutes ago)"*, a time the BMS never returned. Check an answer against its tool responses, always.

`include` is agent configuration. A prompt injection that rewrites the agent's config, or a different agent, is not bound by it. The layer that holds is OpenShell policy with `protocol: mcp`, which you write in Exercise 04.

✓ Checkpoint: you can publish one tool with `--tool_names`, and explain why `include: [read_point, list_alarms]` is useful but is not a security boundary.

## 7 · L3.7 — Sandboxed code execution for the agent

If Alto Ops Claw should write and run Python for ad-hoc analysis, it must not run it on the Spark host. NAT's `code_execution` function sends the code to a separate sandbox server and returns `stdout`, `stderr` and a status.

> **1.9 differs.** The research tutorial (citing the NAT docs) describes a Docker `local_sandbox` on `http://127.0.0.1:6000`, started with `start_local_sandbox.sh`. NAT 1.9.0 ships no `local_sandbox`. Its `code_execution` accepts only `sandbox_type: piston`, a [Piston](https://github.com/engineer-man/piston) server whose default URI is `http://127.0.0.1:2000/api/v2/`.

```bash
# on: laptop
week26/.venv-nat/bin/nat validate --config_file week26/04_nat_claws/configs/code_execution.tutorial.yml 2>/dev/null | head -2
week26/.venv-nat/bin/nat validate --config_file week26/04_nat_claws/configs/code_execution.piston.yml 2>/dev/null | head -2
docker info --format '{{.ServerVersion}}' 2>&1 | tail -1
```

**Expected output** (captured on this Mac)

```
Validating configuration file: week26/04_nat_claws/configs/code_execution.tutorial.yml
✓ Configuration file is valid!
Validating configuration file: week26/04_nat_claws/configs/code_execution.piston.yml
✓ Configuration file is valid!
failed to connect to the docker API at unix:///Users/altodev/.docker/run/docker.sock; check if the path is correct and if the daemon is running: dial unix /Users/altodev/.docker/run/docker.sock: connect: no such file or directory
```

The tutorial's block validates, because `uri` is just a URL, but on 1.9.0 it would speak the Piston API to port 6000. Lab 04-5 also shows that `sandbox_type: local` is refused (`Input should be 'piston'`). The Docker daemon is off on this Mac, so no Piston server can run here, and the tool stays config-only. To try it, start Docker, deploy Piston as NAT's `code_execution` README describes, and point `uri` at it.

Inside a claw, a Piston server is one more network endpoint. It needs its own `network_policies` entry, and the agent's code runs there, never on the host.

✓ Checkpoint: you can say which code-execution backend NAT 1.9 supports and why this laptop cannot run it right now.

## 8 · L3.8 — Run NAT inside an OpenShell sandbox

This is where the two halves of the course meet: the NAT workflow from Sections 3–6, inside the OpenShell boundary from Module 03. The plan has three files. Lab 04-5 generates all of them into `week26/04_nat_claws/.runs/alto-ops-sandbox/`, together with the `alto_ops` package and the CSV.

**1 · `Dockerfile.alto-ops`.** This is the research tutorial's own composition, not an NVIDIA image. It was not built on this Mac (the Docker daemon is off). Verify the NAT install line and the non-root `USER` against your base image. OpenShell requires a non-root identity and rejects root.

```dockerfile
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y python3.12 python3-pip curl && rm -rf /var/lib/apt/lists/*
RUN pip3 install --break-system-packages uv && uv pip install --system 'nvidia-nat[langchain,mcp,profiler,opentelemetry]'
COPY workflows/alto_ops /app/alto_ops
RUN uv pip install --system -e /app/alto_ops
COPY workflow.sandbox.yml /app/workflow.yml
USER 1500
WORKDIR /sandbox
```

**2 · `workflow.sandbox.yml`.** It differs from the laptop workflow in the LLM block only (plus the CSV path inside the sandbox). `base_url` is `https://inference.local/v1`, the managed route, and `api_key` stays `EMPTY`: the gateway injects the real credential. `model_name` must equal the model you set with `openshell inference set`. Because NAT's Python process is the caller, the proxy's TLS trust is already injected through `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE`, so `https://inference.local/v1` works for `openai`-type clients.

**3 · `alto-ops-policy.yaml`.** It has **no** network entries. Inference is not a `network_policies` entry you add: the supervisor intercepts `inference.local`.

```yaml
version: 1
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
```

Lab 04-5 checks this policy twice: once with the course's `policykit` teaching model, and once with the real OpenShell 0.0.111 parser.

**Expected output** (captured on this Mac, lab 04-5 steps 3–5)

```
│ read  /sandbox/data/chiller_plant.csv                 ✓ allow                  under read_write /sandbox
│ write /app/workflow.yml                               ✕ deny                   /app is read_only (Landlock: Permission denied)
│ run as root                                           ✕ deny                   the sandbox runs as '1500'; it cannot become 'root'
│ python3.12 → inference.local:443 POST /v1/chat/comp…  ◆ inspect_for_inference  inference.local is handled by the proxy's inference…
│ python3.12 → api.openai.com:443                       ✕ deny                   api.openai.com:443 is not in network_policies (defa…
│ python3.12 → bms.alto.local:8443 MCP tools/call rea…  ✕ deny                   bms.alto.local:8443 is not in network_policies (def…
✓ parsed: OpenShell read the YAML and only failed to reach a gateway (there is none on this laptop)
  error: the argument '--upload <UPLOAD>' cannot be used with '[COMMAND]...'
│ tutorial one-liner (--upload + -- nat serve)  ✕ rejected
│ create … --detach -- nat serve …              ✓ parsed
│ sandbox upload alto-ops ./data …              ✓ parsed
│ forward start --background 8001 alto-ops      ✓ parsed
│ logs alto-ops -n 50 --source sandbox          ✓ parsed
```

`policykit` is a teaching model. The real enforcement is Landlock, seccomp and the proxy on the Spark. The parser check is real: OpenShell read the YAML, then failed only because this laptop has no gateway.

> **Parser differs.** The research tutorial creates the sandbox in one line, with `--upload ./data:/sandbox/data` and a trailing `-- nat serve …`. The laptop's OpenShell 0.0.111 parser rejects that combination. NemoClaw pins 0.0.116 on the Spark, so check `openshell sandbox create --help` there. The split form below parses on 0.0.111. `chiller_kpi` opens the CSV on every call, so uploading the data after the server starts is fine.

On the Spark, copy the folder over (`scp -r week26/04_nat_claws/.runs/alto-ops-sandbox <spark>:~/works/alto-ops-claw/`), then:

```bash
# on: spark
cd ~/works/alto-ops-claw/alto-ops-sandbox
openshell inference get
openshell sandbox create --name alto-ops --from ./Dockerfile.alto-ops --policy ./alto-ops-policy.yaml --forward 8001 --keep --detach -- nat serve --config_file /app/workflow.yml --host 0.0.0.0 --port 8001
openshell sandbox upload alto-ops ./data /sandbox/data
openshell forward start --background 8001 alto-ops
openshell logs alto-ops --tail --source sandbox
```

Three choices differ from the tutorial's line, and each has a reason:

| Choice | Why |
|---|---|
| `--from ./Dockerfile.alto-ops` instead of `--from ./` | the help says `--from` takes "a path to a Dockerfile or directory containing one", and this file is not named `Dockerfile` |
| `--detach` | listed by 0.0.111: it starts the main process without attaching, so your terminal (and a lab) does not block on `nat serve` |
| `sandbox upload` as its own step | 0.0.111 refuses `--upload` together with a trailing command |

The `--tail` log follows forever, so run it yourself in the ⌨ terminal. Lab 04-5 reads `-n 50` instead. From the Spark host, `curl -s http://localhost:8001/health` should then answer `{"status":"healthy"}`, the same shape lab 04-3 got on this laptop. The NemoClaw equivalent for a custom image is `nemoclaw onboard --from <Dockerfile>`.

✓ Checkpoint: you can say which single YAML line moves a NAT workflow into a sandbox (`base_url: https://inference.local/v1`), and why the policy has no network entries.

## Labs — run them here

**labs/lab04_1_hello_workflow.py** — NAT on this laptop, a 1.9 scaffold, and the minimal ReAct workflow, first as written and then with thinking turned off.

**labs/lab04_2_custom_tool.py** — The chiller_kpi tool: its ground truth without an LLM, then the agent with the original and the fixed tool description.

**labs/lab04_3_serve.py** — nat serve on a free port: real routes, /v1/workflow, /v1/chat/completions and the /full step stream, counted.

**labs/lab04_4_mcp_both_ways.py** — nat mcp serve and nat mcp client, then NAT consuming the mock BMS with only the read tools switched on.

**labs/lab04_5_nat_in_sandbox.py** — The sandbox build context and policy, checked by policykit and the real OpenShell parser, the Spark steps, and NAT 1.9's code-execution backend.

## Try it yourself

`exercises/ex04_bms_tools.py` has three TODOs. Together they give Alto Ops Claw BMS access in two layers:

1. Write the `function_groups` block: `bms_tools`, `_type: mcp_client`, streamable-HTTP to `http://bms.alto.local:8443/mcp`, and only `read_point` + `list_alarms` offered. Add a `tool_overrides` description for `read_point` that says what RT means.
2. Write the exact tool name the LLM sees for `read_point`.
3. Write the `bms_mcp` `network_policies` entry from the research tutorial's Part 3 exercise 4: `protocol: mcp`, `enforcement: enforce`, the MCP handshake, `tools/call` only on `read_point` and `list_alarms`, and a `deny_rules` entry for `write_setpoint`.

The checker is offline. It asks NAT 1.9's own `MCPClientConfig` model to validate your block, runs `policykit.decide` on seven MCP actions, and asks the OpenShell 0.0.111 parser to read your policy.

```bash
# on: laptop
.venv/bin/python week26/04_nat_claws/exercises/ex04_bms_tools.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ TODO 1: bms_tools is an mcp_client on streamable-http http://bms.alto.local:8443/mcp
✓ TODO 1: include OR exclude, not both
✓ TODO 1: the agent is offered exactly read_point + list_alarms (write_setpoint never reaches it)
✓ TODO 1: read_point's tool_overrides description says RT = refrigeration tons
✓ TODO 1: NAT 1.9's MCPClientConfig accepts it (streamable-http ['list_alarms', 'read_point'] [])
✓ TODO 2: the LLM sees bms_tools__read_point (<group>__<tool>)
✓ TODO 3: policykit validates the policy and bms_mcp has an endpoint
✓ TODO 3: protocol mcp + enforcement enforce (a violation is blocked, not just logged)
✓ TODO 3: read_point + list_alarms allowed · write_setpoint + get_trend denied · curl denied · handshake allowed (policykit)
✓ TODO 3: an explicit deny_rules entry names write_setpoint
✓ TODO 3: the real OpenShell 0.0.111 parser reads your policy (then finds no gateway)
```

<details><summary>Hint — TODO 1, include or exclude?</summary>

Use `include: [read_point, list_alarms]`. An allow-list stays safe when the BMS adds a new tool tomorrow; an `exclude` list would quietly offer it. You cannot use both in one group on NAT 1.9.

</details>

<details><summary>Hint — TODO 3, the MCP rules</summary>

MCP rules match a JSON-RPC `method` and, for `tools/call`, a `tool`. A client must `initialize`, send `notifications/initialized` and call `tools/list` before any `tools/call`, so allow those three too. `tool: { any: [read_point, list_alarms] }` matches either name. The binary is the sandboxed NAT process, `/usr/bin/python3.12`.

</details>

<details><summary>Hint — going further (research tutorial Part 3, exercises 1 and 2)</summary>

Change `_type: tool_calling_agent` to `react_agent` (keep `reasoning_effort: none`) and count LLM_END steps with `/v1/workflow/full` for "give me the 6-hour and 24-hour kW/RT". Measure, don't assume. Then read `week26/common/alto_ops/src/alto_ops/setpoint_tool.py`: its Pydantic `input_schema` refuses a ticket without a `work_order_id`. That is layer (a) of the research tutorial's three-layer answer; Exercise 04's policy is layer (c), the only one that survives prompt injection.

</details>

✓ Checkpoint: all eleven checker lines are ✓.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ReActAgentParsingFailedError … LLM output: ''` | a thinking model put its whole answer in the reasoning field | add `reasoning_effort: none` under the LLM, or use `tool_calling_agent` |
| `Invalid workflow directory specified. … does not exist.` | NAT 1.9 wants `--workflow-dir` to exist | `mkdir -p` it first |
| `nat serve` exits: `…requires that the Python 'greenlet' library is installed` | NAT 1.9.0 + SQLAlchemy 2.1 without greenlet | `uv pip install --python week26/.venv-nat/bin/python greenlet` |
| `Address already in use` on 8001, 9901 or 8443 | another lab (or a module author) is running a server | the labs pick the next free port with `free_port()` and print it; stop your own foreground servers with Ctrl-C |
| `Invalid configuration: function_groups: Value error, include and exclude cannot be used together` | both lists in one group | keep only `include` |
| `error: the argument '--upload <UPLOAD>' cannot be used with '[COMMAND]...'` | OpenShell 0.0.111 argument rules | create first, then `openshell sandbox upload` |
| every AI answer takes 40–70 s instead of 10–15 s | other labs share the laptop's Ollama | run one lab at a time; laptop times are LAPTOP STAND-IN anyway |
| an `AuthlibDeprecationWarning` on every `nat` command | a NAT 1.9.0 dependency warning | harmless; the labs filter it, and `2>/dev/null` hides it in the terminal |

## Next

[Lab 05 — Tracing and observability: agent, policy and harness planes](../05_tracing/TUTORIAL.md): turn the LLM_END and TOOL_END steps you counted here into Phoenix and OpenTelemetry traces, and line them up with OpenShell's policy log for the same request.
