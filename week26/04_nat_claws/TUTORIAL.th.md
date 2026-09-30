# ▶ Reef Lab 04 — สร้าง claw ด้วย NeMo Agent Toolkit: tools, serving, MCP และ NAT ใน sandbox

> ส่วนหนึ่งของ Week 26 · NemoClaw บน DGX Spark ตั้งแต่ระดับเริ่มต้นจนถึงระดับผู้เชี่ยวชาญ คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง แล็บฝั่งแล็ปท็อป (NAT CLI, OpenShell CLI และ policy model) รันจริงได้ทุกที่ ส่วนแล็บฝั่ง Spark รันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): ระบบจะแสดงคำสั่ง และผลลัพธ์จะเป็นแบบ RECORDED ที่บันทึกจาก Spark จริง, REFERENCE ที่อ้างอิงจากเอกสารหรือ playbook ของ NVIDIA หรือ EXAMPLE ที่ติดป้ายไว้ชัดเจน

**สิ่งที่คุณจะได้ลงมือทำจริง**
- รัน workflow ของ NeMo Agent Toolkit (NAT) 1.9 บนแล็ปท็อปเครื่องนี้จริง ๆ โดยใช้โมเดล Nemotron ใน Ollama บนเครื่อง
- อ่าน ทดสอบ และเรียก tool ที่เขียนเอง (`chiller_kpi`) โดยไม่ใช้ LLM แล้วดูว่าประโยคเดียวใน description ของมันเปลี่ยนคำตอบของ agent ได้อย่างไร
- serve YAML ไฟล์เดียวกันเป็น REST API, endpoint ที่เข้ากันได้กับ OpenAI และ stream แบบทีละขั้น แล้วนับจำนวน LLM call กับ tool call ที่อยู่เบื้องหลังคำตอบหนึ่งคำตอบ
- เผยแพร่ tools ผ่าน MCP และใช้งานระบบ BMS (จำลอง) ผ่าน MCP โดยเปิดเฉพาะ tool ที่ปลอดภัย
- เตรียม build context สำหรับรัน NAT ใน OpenShell sandbox: Dockerfile, workflow ที่เรียก `inference.local` และ policy แบบ deny-by-default ที่ parser ตัวจริงยอมรับ

**Time** ~120 นาที · **Difficulty** intermediate · **Hardware** แล็ปท็อป (NAT 1.9 + Ollama) · DGX Spark 1 เครื่อง (ไม่บังคับ)

**Sources:** research tutorial ของคอร์ส *NemoClaw on DGX Spark — Beginner to Expert* (Part 3, แล็บ L3.1–L3.8) ซึ่งอ้างอิง [NeMo Agent Toolkit docs](https://docs.nvidia.com/nemo/agent-toolkit/latest/) · [NAT writing custom functions](https://docs.nvidia.com/nemo/agent-toolkit/latest/extend/functions.html) · [NAT tool calling agent](https://docs.nvidia.com/nemo/agent-toolkit/latest/components/agents/tool-calling-agent/tool-calling-agent.html) · [NAT API server endpoints](https://docs.nvidia.com/nemo/agent-toolkit/latest/reference/rest-api/api-server-endpoints.html) · [NAT MCP server](https://docs.nvidia.com/nemo/agent-toolkit/latest/run-workflows/mcp-server.html) · [NAT MCP client](https://docs.nvidia.com/nemo/agent-toolkit/latest/build-workflows/mcp-client.html) · [NAT code execution](https://docs.nvidia.com/nemo/agent-toolkit/latest/components/functions/code-execution.html) · [Classmethod — NAT on DGX Spark](https://dev.classmethod.jp/en/articles/dgx-spark-nemo-agent-toolkit-local-intro/) · [OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies) · [OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ใช้ทำอะไร |
|---|---|---|
| NAT 1.9 บนแล็ปท็อป | `week26/.venv-nat/bin/nat --version` | ทุกแล็บรัน NAT CLI ตัวจริง |
| โมเดลที่เรียก tool ได้ใน Ollama บนแล็ปท็อป | `nemotron-3-nano:latest` ใน `curl -s localhost:11434/v1/models` | ใช้เป็น LAPTOP STAND-IN แทน vLLM บน Spark (~10–20 วินาทีต่อคำถาม) |
| `greenlet` ใน NAT venv | `week26/.venv-nat/bin/python -c "import greenlet"` | `nat serve` ของ NAT 1.9.0 ต้องใช้ และการติดตั้งใหม่อาจไม่มีมาให้ (ดูหัวข้อ 5) |
| DGX Spark (ไม่บังคับ) | `ssh -o BatchMode=yes <spark> true` | ขั้นตอน vLLM และ sandbox; ถ้าไม่มีจะรันแบบ DRY |

```bash
# on: laptop
week26/.venv-nat/bin/nat --version 2>/dev/null
curl -s http://localhost:11434/v1/models | grep -o '"nemotron-3-nano[^"]*"'
week26/.venv-nat/bin/python -c "import greenlet; print('greenlet', greenlet.__version__)" 2>&1 | tail -1
```

**Expected output** (บันทึกจากเครื่อง Mac นี้)

```
nat, version 1.9.0
"nemotron-3-nano:latest"
greenlet 3.5.6
```

ถ้าบรรทัดสุดท้ายขึ้นว่า `ModuleNotFoundError: No module named 'greenlet'` ให้รัน `uv pip install --python week26/.venv-nat/bin/python greenlet` หนึ่งครั้ง venv ของคอร์สนี้เคยไม่มีแพ็กเกจนี้มาก่อน (หัวข้อ 5 อธิบายสาเหตุ) และมีแค่ `nat serve` ที่ต้องใช้

> 📌 **เรื่องเวอร์ชัน** research tutorial เขียนโดยอ้างอิงเอกสาร NAT **1.8** แต่คอร์สนี้รัน NAT **1.9.0** และโมดูลนี้ตรวจทุกคำสั่ง ทุก YAML key และทุก endpoint กับแพ็กเกจที่ติดตั้งจริงแล้ว จุดไหนที่ 1.9 ต่างออกไป หัวข้อนั้นจะมีกล่อง **1.9 differs** บอกไว้ ถ้าทำได้ให้รันทีละแล็บ เพราะแล็บอื่นใช้ Ollama ตัวเดียวกัน เวลาที่วัดได้จึงแกว่ง

✓ Checkpoint: `nat --version` แสดง 1.9.0 และ Ollama มี `nemotron-3-nano:latest`

## 1 · L3.1 — NAT ในหน้าเดียว และการติดตั้งบน Spark

NAT (`nvidia-nat`) เป็นชั้นสำหรับ agent ที่ไม่ผูกกับ framework ใด agent, tool และ workflow ทุกตัวคือ **function** ไฟล์ YAML ไฟล์เดียวเชื่อมทุกอย่างเข้าด้วยกัน และ `_type` เป็นตัวเลือก implementation:

| ส่วนใน YAML | เก็บอะไร | ตัวอย่างในโมดูลนี้ |
|---|---|---|
| `functions` | tools และ agent ที่ถูกใช้เป็น tool | `chiller_kpi`, `current_datetime` |
| `function_groups` | ชุดของ tools จากแหล่งเดียวกัน | `bms_tools` (`_type: mcp_client`) |
| `llms` | endpoint ของโมเดล | `_type: openai` + `base_url` |
| `workflow` | จุดเริ่มต้นของงาน | `_type: tool_calling_agent` หรือ `react_agent` |

ทำไมต้องมี NAT ในคอร์ส NemoClaw? เพราะ OpenClaw, Hermes และ Deep Agents เป็นผู้ช่วยแบบอเนกประสงค์ ส่วน claw ที่สร้างมาเพื่องานเฉพาะอย่าง **Alto Ops Claw** ต้องการ tools ที่ระบุชัด การประเมินผลที่ทำซ้ำได้ และการ profile ซึ่ง NAT มีให้ครบ และ workflow ตัวเดียวกันรันได้ทั้งบน Spark host (เรียก vLLM) และใน OpenShell sandbox (เรียก `inference.local`) agent type ที่มีมาในตัวได้แก่ ReAct, Reasoning, ReWOO, Router และ Tool Calling

บน Spark นั้น NAT เป็น wheel แบบ pure-Python จึงติดตั้งบน arm64 ได้โดยไม่ต้องคอมไพล์อะไร research tutorial ทำตามบทความของ Classmethod ที่ใช้ DGX Spark: Python 3.12 ใน uv venv

```bash
# on: spark
mkdir -p ~/works/alto-ops-claw && cd ~/works/alto-ops-claw
uv venv --python 3.12 && source .venv/bin/activate
uv pip install 'nvidia-nat[langchain,mcp,profiler,phoenix,opentelemetry]'
nat --version
```

extras ที่จะเจอ: `langchain` (ReAct agent และ tool-calling agent), `mcp` (client และ server), `profiler` (สำหรับ profiling ของ `nat eval` และ sizing calculator), `phoenix` และ `opentelemetry` (tracing ในโมดูล 05) venv บนแล็ปท็อปนี้มี `eval` และ `ragas` ด้วย (โมดูล 06)

✓ Checkpoint: คุณบอกชื่อทั้งสี่ส่วนของ YAML ได้ และบอกได้ว่า tools จาก MCP server ต้องใส่ไว้ส่วนไหน (`function_groups`)

## 2 · L3.2 — vLLM บนเครื่องสำหรับ NAT บน Spark

บน Spark นั้น NAT คุยกับ vLLM server บนเครื่อง research tutorial เริ่มโมเดล Nemotron 3 Nano ที่ Classmethod ทดสอบแล้วว่าใช้ได้บน Spark:

```bash
# on: spark
docker run -d --name vllm-nat --gpus all --shm-size=16g -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  nvcr.io/nvidia/vllm:26.01-py3 \
  vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
  --trust-remote-code --max-model-len 8192 --gpu-memory-utilization 0.85
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

Nemotron 3 Nano ใช้สถาปัตยกรรมไฮบริด Mamba–Transformer ชื่อ `nemotron_h` จึงต้องใส่ `--trust-remote-code` ถ้าไม่ใส่ server จะพังด้วย pydantic `ValidationError` ตามที่ Classmethod รายงาน (อ้างถึงใน research tutorial) การโหลดโมเดลขนาด 30.5 GiB ใช้เวลาประมาณสี่นาที บวก `torch.compile` อีกสองนาที และใช้ KV cache 34.79 GiB ที่ context 8K

แล็ปท็อปนี้ไม่มี GPU สำหรับงานนั้น แล็บจึงใช้ **Ollama เป็น LAPTOP STAND-IN** โดยเปลี่ยน YAML แค่สองบรรทัด:

| Key | บน Spark (vLLM) | บนแล็ปท็อปนี้ (Ollama) |
|---|---|---|
| `base_url` | `http://localhost:8000/v1` | `http://localhost:11434/v1` |
| `model_name` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` | `nemotron-3-nano:latest` |

ตัวเลขเวลาในโมดูลนี้เป็นของแล็ปท็อปเท่านั้น ไม่ใช่ตัวเลขของ Spark

✓ Checkpoint: คุณอธิบายได้ว่าทำไมคำสั่ง vLLM ต้องมี `--trust-remote-code` และ YAML key สองตัวไหนที่ย้าย workflow จาก Spark มาแล็ปท็อป

## 3 · L3.3 — Hello, local workflow

เริ่มจาก scaffold ก่อน บน NAT 1.9 โฟลเดอร์ `--workflow-dir` ต้องมีอยู่แล้ว จึงต้องสร้างก่อน:

```bash
# on: laptop
mkdir -p week26/04_nat_claws/.runs/workflows
week26/.venv-nat/bin/nat workflow create --no-install --workflow-dir week26/04_nat_claws/.runs/workflows alto_ops --description "Alto Ops Claw" 2>/dev/null
find week26/04_nat_claws/.runs/workflows/alto_ops -type f | sort
grep '^from nat' week26/04_nat_claws/.runs/workflows/alto_ops/src/alto_ops/alto_ops.py
```

**Expected output** (บันทึกจากเครื่อง Mac นี้)

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

> **1.9 differs.** คำสั่ง `--workflow-dir ./workflows` ของ research tutorial ล้มเหลวในโฟลเดอร์ใหม่ด้วยข้อความ `Invalid workflow directory specified. … does not exist.` scaffold ของ 1.9 import ทุกอย่างจาก facade เดียวคือ `nat.plugin_api` และ `config.yml` ใช้ `_type: nim` (NIM แบบ hosted) คุณต้องเปลี่ยนบล็อก `llms` นั้นเป็น server บนเครื่อง

ต่อด้วย workflow ขั้นต่ำ `week26/04_nat_claws/configs/hello.laptop.yml` คือ YAML ของ research tutorial ที่เปลี่ยนสองบรรทัดตามหัวข้อ 2 ฟิลด์ `api_key` ต้องมีเพราะ validation บังคับ แม้ server จะไม่ใช้ key จึงใส่ค่าหลอกไว้คือ `EMPTY` ให้ validate ก่อนรันเสมอ เพราะไม่เสีย LLM call เลย

```bash
# on: laptop
cat week26/04_nat_claws/configs/hello.laptop.yml
week26/.venv-nat/bin/nat validate --config_file week26/04_nat_claws/configs/hello.laptop.yml 2>/dev/null | head -2
week26/.venv-nat/bin/nat run --config_file week26/04_nat_claws/configs/hello.laptop.yml --input "What time is it in Bangkok right now?" 2>&1 | tail -1
```

**Expected output** (บันทึกจากเครื่อง Mac นี้ เฉพาะสองคำสั่งสุดท้าย)

```
Validating configuration file: week26/04_nat_claws/configs/hello.laptop.yml
✓ Configuration file is valid!
Error: ReActAgentParsingFailedError: Failed to parse agent output after 1 attempts. Error: Invalid Format: Missing 'Action:' after 'Thought:'. LLM output: ''
```

YAML ไม่ผิด ปัญหาอยู่ที่โมเดล `nemotron-3-nano` เป็น **thinking model** บน Ollama ความคิดของมันไปอยู่ในฟิลด์ `reasoning` แยกต่างหาก ReAct agent ซึ่งแยก `Thought:` กับ `Action:` ออกจากข้อความคำตอบ จึงได้สตริงว่าง ไฟล์ `hello.laptop.nothink.yml` เพิ่มบรรทัดเดียวใต้ LLM คือ `reasoning_effort: none` config แบบ `openai` ของ NAT ยอมรับ key เพิ่มเติมและส่งต่อไปให้ server

```bash
# on: laptop
week26/.venv-nat/bin/nat run --config_file week26/04_nat_claws/configs/hello.laptop.nothink.yml --input "What time is it in Bangkok right now?" 2>&1 | grep -A1 -E "Action:|Tool's response|Workflow Result"
```

**Expected output** (บันทึกจากเครื่อง Mac นี้)

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

NAT เขียน agent log และผลลัพธ์ออกทาง **stderr** ดังนั้น pipe ต้องมี `2>&1`

สังเกต offset ให้ดี `current_datetime` คืนค่าเป็น UTC (`+0000`) แต่กรุงเทพฯ คือ UTC+7 agent ส่งคำตอบของ tool ต่อมาตรง ๆ ว่าเป็น "เวลากรุงเทพฯ" loop ทำงานถูก แต่คำตอบยังผิด จำเรื่องนี้ไว้ทุกครั้งที่สร้าง claw

`tool_calling_agent` ในหัวข้อถัดไปไม่มีปัญหานี้ เพราะใช้ native function calling ของโมเดล แทนการแยกข้อความ

✓ Checkpoint: คุณบอกได้ว่าการรันครั้งแรกล้มเหลวเพราะอะไร (thinking model ส่งข้อความว่างให้ ReAct parser) และคำตอบครั้งที่สองยังผิดตรงไหน (รายงานเวลา UTC ว่าเป็นเวลากรุงเทพฯ)

## 4 · L3.4 — tool ที่เขียนเอง: วิเคราะห์ CSV ของ chiller plant

tool จริงตัวแรกของ Alto Ops Claw อ่านไฟล์ CSV ที่ export จาก chiller plant แล้วคืนค่า kW ต่อ refrigeration ton (kW/RT) ของทั้งโรงงาน ยิ่งต่ำยิ่งดี แพ็กเกจของคอร์สติดตั้งอยู่ใน NAT venv แล้ว: `week26/common/alto_ops/src/alto_ops/chiller_tool.py` เป็นโค้ด Lab 3.4 ของ research tutorial ที่แก้ description ไปหนึ่งจุด

function หนึ่งตัวประกอบด้วยสามส่วนของ NAT:

| ส่วน | หน้าที่ |
|---|---|
| `class ChillerKpiConfig(FunctionBaseConfig, name="chiller_kpi")` | `_type:` ใน YAML และฟิลด์ที่มี type กำกับ (`csv_path`, `kw_per_rt_alarm`) |
| `@register_function(config_type=ChillerKpiConfig)` | ลงทะเบียน type ตอนที่ module ถูก import |
| `yield FunctionInfo.from_fn(_kpi, description=…)` | callable ที่ agent ได้รับ และ description ที่ agent อ่าน |

NAT หาแพ็กเกจเจอผ่าน entry point `nat.components` ใน `pyproject.toml` ซึ่ง import `alto_ops.register` research tutorial ขอให้ยืนยัน import path กับเวอร์ชันของคุณ บน 1.9.0 ใช้ได้ทั้ง path แบบยาวและ facade ใหม่ `nat.plugin_api` และเป็น object ตัวเดียวกัน:

```bash
# on: laptop
week26/.venv-nat/bin/python week26/04_nat_claws/tools/nat_direct.py imports
```

**Expected output** (บันทึกจากเครื่อง Mac นี้)

```
nat.plugin_api.Builder is nat.builder.builder.Builder: True
nat.plugin_api.FunctionInfo is nat.builder.function_info.FunctionInfo: True
nat.plugin_api.register_function is nat.cli.register_workflow.register_function: True
nat.plugin_api.FunctionBaseConfig is nat.data_models.function.FunctionBaseConfig: True
```

**หาคำตอบที่ถูกต้องก่อนถาม agent** `nat_direct.py` สร้างเฉพาะ tool จาก workflow YAML แล้วเรียกมันตรง ๆ โดยไม่ใช้ LLM:

```bash
# on: laptop
week26/.venv-nat/bin/python week26/04_nat_claws/tools/nat_direct.py tool week26/common/alto_ops/src/alto_ops/configs/workflow.laptop.yml chiller_kpi '{"hours": 6}' '{"hours": 24}' 2>/dev/null
```

**Expected output** (บันทึกจากเครื่อง Mac นี้)

```
description: Average chiller plant kW, RT (refrigeration tons of cooling) and kW/RT over the last N hours; flags efficiency alarms.
input schema: {"hours": {"default": 24, "title": "Hours", "type": "integer"}}
chiller_kpi({"hours": 6}) -> window=6h avg_kw=473.6 avg_rt=525.9 kw_per_rt=0.901 status=ALARM
chiller_kpi({"hours": 24}) -> window=24h avg_kw=442.5 avg_rt=629.2 kw_per_rt=0.703 status=OK
```

ข้อมูลเป็นข้อมูลสังเคราะห์ (synthetic): 672 แถว ทุก 15 นาที และจงใจทำให้หกชั่วโมงสุดท้ายแย่ลง YAML ตั้ง alarm ไว้ที่ 0.80 kW/RT ดังนั้น 6 ชั่วโมงจึงเป็น ALARM และ 24 ชั่วโมงเป็น OK

**description ของ tool ก็คือ prompt** Lab 04-2 ถาม `tool_calling_agent` ด้วยคำถามเดียวกันสองครั้ง ครั้งแรกใช้ `tools/chiller_v0.py` ซึ่งเป็น tool ของ research tutorial พร้อม description เดิม: *"Average chiller plant kW, RT and kW/RT…"* ครั้งที่สองใช้ description ของคอร์ส ที่เติม *"RT (refrigeration tons of cooling)"*

**Expected output** (บันทึกจากเครื่อง Mac นี้, lab 04-2 step 7 — คำตอบแบบ LAPTOP STAND-IN)

```
│ tool description      kW/RT in answer  what RT means                       verdict  s (LAPTOP STAND-IN)
│ ────────────────────  ───────────────  ──────────────────────────────────  ───────  ───────────────────
│ tutorial description  ✓ 0.901          ✕ 'refrigerant/return temperature'  ✓ alarm  14.1
│ course description    ✓ 0.901          ✓ refrigeration tons                ✓ alarm  15.2
✓ same model, same data, same number — only the description changed, and so did the meaning of RT
```

คำตอบแรกบอกว่า *"an average refrigerant temperature (RT) of 525.9"* ส่วนคำตอบที่สองบอกว่า *"525.9 refrigeration tons (RT) of cooling"* โมเดลเห็นแค่ชื่อ description และ input schema ของ tool และ `avg_rt=525.9` ไม่มีความหมายอะไรสำหรับมันถ้าไม่มีบริบท การ build ครั้งนี้เจอบั๊กนี้จริง ๆ แพ็กเกจของคอร์สจึงใช้ description ที่ยาวกว่า คำตอบของโมเดลมีความสุ่ม รันใหม่อาจได้ต่างไป แต่วิธีแก้ไม่เปลี่ยน

อีกหนึ่ง pattern จาก research tutorial: agent หนึ่งเป็น tool ของอีก agent ได้ด้วย YAML อย่างเดียว entry ใน `functions:` ที่มี `_type: tool_calling_agent`, `tool_names` ของตัวเอง และ `description` จะกลายเป็น tool ที่เรียกใช้ได้

✓ Checkpoint: คุณคำนวณ kW/RT ของ 6 ชั่วโมงได้โดยไม่ใช้ LLM และอธิบายได้ว่าทำไมวลีเดียวใน description ถึงเปลี่ยนคำตอบของ agent

## 5 · L3.5 — serve มัน: REST และ endpoint ที่เข้ากันได้กับ OpenAI

`nat serve` เปลี่ยน YAML เดิมให้เป็น FastAPI server พอร์ตเริ่มต้นคือ 8000 ซึ่งเป็นพอร์ตของ vLLM บน Spark จึงต้องใส่ `--port 8001` (คำสั่ง curl ใน research tutorial ก็ใช้ 8001)

> **1.9 differs.** NAT 1.9.0 ดึง SQLAlchemy 2.1 มาด้วย ซึ่งไม่ได้ติดตั้ง `greenlet` ให้โดยอัตโนมัติแล้ว แต่ `nat serve` import ส่วน asyncio ของ SQLAlchemy ตอนเริ่มทำงาน ถ้าไม่มี `greenlet` จะหยุดพร้อมข้อความ *"The SQLAlchemy asyncio module requires that the Python 'greenlet' library is installed"* venv ของคอร์สนี้เจอแบบนั้นจริง วิธีแก้มีบรรทัดเดียวคือ `uv pip install --python week26/.venv-nat/bin/python greenlet` และตอนนี้ติดตั้งแล้ว ถ้าเครื่องของคุณยังไม่มี lab 04-3 จะยืม `greenlet` จาก `week25/.venv-nat` ผ่าน `PYTHONPATH` (และบอกไว้ชัด) หรือพิมพ์วิธีแก้ให้ `nat run` และ `nat mcp serve` ไม่ต้องใช้แพ็กเกจนี้

```bash
# on: laptop
week26/.venv-nat/bin/nat serve --config_file week26/common/alto_ops/src/alto_ops/configs/workflow.laptop.yml --host 127.0.0.1 --port 8001
```

ในเทอร์มินัลที่สอง:

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

Lab 04-3 เรียกแบบเดียวกันด้วย `urllib` ของ Python ในรูปแบบ request ที่ OpenAI client ส่ง รูปแบบ response บน 1.9.0 เป็นดังนี้:

| Endpoint | รูปแบบ response (บันทึกจากเครื่อง Mac นี้) |
|---|---|
| `GET /health` | `{"status": "healthy"}` |
| `POST /v1/workflow` | `{"value": "<answer>"}` |
| `POST /v1/chat/completions` | `object=chat.completion`, `model='unknown-model'` (ไม่ส่ง `model` ที่คุณส่งไปกลับมา), `usage.prompt_tokens` เป็น 0 |
| `POST /v1/workflow/full` | stream แบบ server-sent events: ชิ้นคำตอบ `data: {"value": …}` และขั้นตอน `intermediate_data: {…}` |

**Expected output** (บันทึกจากเครื่อง Mac นี้, lab 04-3 steps 2 และ 5)

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

> **1.9 differs.** `/v1/workflow/async` ต้องใช้ extra `dask` ซึ่งไม่ได้ติดตั้ง `/monitor/users` จะมีเฉพาะเมื่อตั้ง `general.enable_per_user_monitoring: true` และ config นี้ไม่ได้ลงทะเบียน `/feedback` ของใหม่ใน 1.9 คือ `/v1/workflow/atif`, `/executions/…` และ `/mcp/client/tool/list` เปิด `http://localhost:8001/docs` บนเครื่องของคุณเพื่อดูรายการจริง

ตารางขั้นตอนคือสิ่งที่มีประโยชน์ที่สุดในหน้านี้ คำถามหนึ่งข้อใช้ LLM call สองครั้ง (วางแผน แล้วสรุป) และ tool call หนึ่งครั้ง เวลาเกือบทั้งหมดอยู่ที่ LLM ส่วน tool ใช้แค่ 10 ms โมดูล 05 จะเปลี่ยนขั้นตอนเหล่านี้เป็น trace แล็บบันทึกไว้ที่ `.runs/lab04_3_full_steps.json`

✓ Checkpoint: คุณบอกได้ว่าคำถามเรื่อง chiller หนึ่งข้อใช้ LLM call กี่ครั้งและ tool call กี่ครั้ง และรู้จาก endpoint ไหน

## 6 · L3.6 — MCP ทั้งสองทาง

**NAT ในบทบาท MCP server** เผยแพร่ tools ของ Alto Ops ให้ MCP client ตัวไหนก็ได้ เช่น OpenClaw, Hermes, Claude Code หรือ NAT อีกตัว transport เริ่มต้นคือ streamable-HTTP บน `/mcp` พอร์ต 9901

```bash
# on: laptop
week26/.venv-nat/bin/nat mcp serve --config_file week26/common/alto_ops/src/alto_ops/configs/workflow.laptop.yml --name "Alto Ops MCP" --port 9901
```

ในเทอร์มินัลที่สอง:

```bash
# on: laptop
curl -s http://localhost:9901/debug/tools/list
week26/.venv-nat/bin/nat mcp client tool list --url http://localhost:9901/mcp 2>/dev/null
```

**Expected output** (บันทึกจากเครื่อง Mac นี้)

```
{"count":3,"tools":[{"name":"chiller_kpi","description":"Tool Calling Agent Workflow","is_workflow":true},{"name":"current_datetime","description":"Tool Calling Agent Workflow","is_workflow":true},{"name":"tool_calling_agent","description":"Tool Calling Agent Workflow","is_workflow":true}],"server_name":"Alto Ops MCP"}
current_datetime
chiller_kpi
tool_calling_agent
```

ลำดับในรายการเปลี่ยนไปในแต่ละครั้งที่รัน

มีสองเรื่องที่ต้องสังเกต เรื่องแรก ถ้าไม่ใส่ `--tool_names` **agent ทั้งตัว** จะถูกเผยแพร่ด้วย (`tool_calling_agent`) ทำให้ MCP client ใดก็ได้สั่งรัน LLM loop ของคุณ เรื่องที่สอง บน 1.9.0 `/debug/tools/list` แสดง description ของ workflow ให้ทุก tool เผยแพร่เฉพาะสิ่งที่ผู้เรียกต้องใช้:

```bash
# on: laptop
week26/.venv-nat/bin/nat mcp serve --config_file week26/common/alto_ops/src/alto_ops/configs/workflow.laptop.yml --port 9901 --tool_names chiller_kpi
```

ในเทอร์มินัลที่สอง:

```bash
# on: laptop
week26/.venv-nat/bin/nat mcp client tool call chiller_kpi --url http://localhost:9901/mcp --json-args '{"hours": 6}' 2>/dev/null | grep window
```

**Expected output** (บันทึกจากเครื่อง Mac นี้)

```
window=6h avg_kw=473.6 avg_rt=525.9 kw_per_rt=0.901 status=ALARM
```

**NAT ในบทบาท MCP client** ใช้ tools ของคนอื่น คอร์สมี BMS จำลองให้ คือ `week26/common/bms_mcp_server.py` ซึ่งมีสี่ tool: `read_point`, `list_alarms`, `get_trend` และ `write_setpoint` ข้อมูลเป็นข้อมูลสังเคราะห์ และ `write_setpoint` ไม่เคยเขียนอะไรจริง ไฟล์ `week26/04_nat_claws/configs/mcp_client_bms.yml` เปิดเฉพาะ tool อ่านค่าสองตัว:

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

ทุก key ตรวจกับซอร์สของ 1.9.0 ที่ติดตั้งแล้ว (`nat/plugins/mcp/client/client_config.py`, `nat/data_models/function.py`, `nat/builder/function.py`):

| Key | บน NAT 1.9.0 |
|---|---|
| `server.transport` | `stdio`, `sse` หรือ `streamable-http` (ค่าเริ่มต้น) |
| `server.url` | จำเป็นสำหรับ `sse` และ `streamable-http`; `auth_provider` และ `custom_headers` ก็อยู่ใต้ `server` เช่นกัน |
| `include` / `exclude` | สืบทอดมาจาก function group ทุกตัว; **ใช้พร้อมกันทั้งสองตัวจะ validate ไม่ผ่าน** |
| `tool_overrides.<tool>` | `alias` และ `description` โดยใช้ชื่อ tool ของ server เป็น key |
| `tool_call_timeout` | วินาที (timedelta) ค่าเริ่มต้น 60 |
| `reconnect_max_attempts` | ค่าเริ่มต้น 2 (research tutorial ตั้งเป็น 3) |
| ชื่อ tool | `<group>__<tool>` เช่น `bms_tools__read_point` (ตัวคั่นแบบเก่าคือ `.`) |

เริ่ม BMS จำลอง แล้วถาม NAT ว่า agent จะได้ tool ตัวไหนบ้าง โดยไม่มี LLM เกี่ยวข้อง:

```bash
# on: laptop
BMS_MCP_PORT=8443 week26/.venv-nat/bin/python week26/common/bms_mcp_server.py
```

ในเทอร์มินัลที่สอง:

```bash
# on: laptop
BMS_MCP_URL=http://localhost:8443/mcp week26/.venv-nat/bin/python week26/04_nat_claws/tools/nat_direct.py group week26/04_nat_claws/configs/mcp_client_bms.yml bms_tools 2>/dev/null
```

**Expected output** (บันทึกจากเครื่อง Mac นี้)

```
config: include=['list_alarms', 'read_point'] exclude=[] tool_overrides=['read_point']
hidden      bms_tools__get_trend  | Hourly averages of PLANT.KW_PER_RT for the last N hours.
AGENT SEES  bms_tools__list_alarms  | List active plant alarms (kW/RT above 0.85 for the last hour).
AGENT SEES  bms_tools__read_point  | Read the latest value of one BMS point. Valid points: PLANT.KW, PLANT.RT (refrigeration tons of cooling), PLANT.KW_PER_RT.
hidden      bms_tools__write_setpoint  | WRITE a setpoint to equipment. Dangerous — policies should deny this tool. (Mock: never writes.)
```

group ยังเชื่อมต่อกับ tool ครบทั้งสี่ตัว `include` เป็นตัวกำหนดว่า agent จะถูก **เสนอ** tool ตัวไหน จากนั้นถาม agent หนึ่งคำถาม:

```bash
# on: laptop
BMS_MCP_URL=http://localhost:8443/mcp week26/.venv-nat/bin/nat run --config_file week26/04_nat_claws/configs/mcp_client_bms.yml --input "Are there any active plant alarms right now, and what is the latest kW/RT reading?" 2>&1 | grep -A1 -E "Tool's response|Workflow Result"
```

**Expected output** (บันทึกจากเครื่อง Mac นี้ — คำตอบแบบ LAPTOP STAND-IN)

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

agent เรียก `bms_tools__list_alarms` และ `bms_tools__read_point` ในรอบเดียว ในการรันครั้งก่อน โมเดลตัวเดียวกันเติม *"(1 hour and 45 minutes ago)"* ซึ่งเป็นเวลาที่ BMS ไม่เคยส่งกลับมา ตรวจคำตอบกับ tool response ทุกครั้ง

`include` เป็น config ของ agent ถ้า prompt injection เขียนทับ config ของ agent หรือเป็น agent ตัวอื่น ก็ไม่ถูกผูกด้วย `include` ชั้นที่กันได้จริงคือ OpenShell policy ที่ใช้ `protocol: mcp` ซึ่งคุณจะเขียนใน Exercise 04

✓ Checkpoint: คุณเผยแพร่ tool ตัวเดียวด้วย `--tool_names` ได้ และอธิบายได้ว่าทำไม `include: [read_point, list_alarms]` มีประโยชน์แต่ไม่ใช่ขอบเขตด้านความปลอดภัย

## 7 · L3.7 — การรันโค้ดแบบ sandbox สำหรับ agent

ถ้า Alto Ops Claw ต้องเขียนและรัน Python เพื่อวิเคราะห์เฉพาะกิจ ห้ามรันบน Spark host function `code_execution` ของ NAT ส่งโค้ดไปยัง sandbox server แยกต่างหาก แล้วคืน `stdout`, `stderr` และสถานะ

> **1.9 differs.** research tutorial (อ้างอิงเอกสาร NAT) อธิบาย `local_sandbox` บน Docker ที่ `http://127.0.0.1:6000` ซึ่งเริ่มด้วย `start_local_sandbox.sh` แต่ NAT 1.9.0 ไม่มี `local_sandbox` แล้ว `code_execution` ของมันรับได้แค่ `sandbox_type: piston` คือ server [Piston](https://github.com/engineer-man/piston) ซึ่งมี URI เริ่มต้นเป็น `http://127.0.0.1:2000/api/v2/`

```bash
# on: laptop
week26/.venv-nat/bin/nat validate --config_file week26/04_nat_claws/configs/code_execution.tutorial.yml 2>/dev/null | head -2
week26/.venv-nat/bin/nat validate --config_file week26/04_nat_claws/configs/code_execution.piston.yml 2>/dev/null | head -2
docker info --format '{{.ServerVersion}}' 2>&1 | tail -1
```

**Expected output** (บันทึกจากเครื่อง Mac นี้)

```
Validating configuration file: week26/04_nat_claws/configs/code_execution.tutorial.yml
✓ Configuration file is valid!
Validating configuration file: week26/04_nat_claws/configs/code_execution.piston.yml
✓ Configuration file is valid!
failed to connect to the docker API at unix:///Users/altodev/.docker/run/docker.sock; check if the path is correct and if the daemon is running: dial unix /Users/altodev/.docker/run/docker.sock: connect: no such file or directory
```

บล็อกของ tutorial ผ่าน validate เพราะ `uri` เป็นแค่ URL แต่บน 1.9.0 มันจะคุย Piston API ไปที่พอร์ต 6000 Lab 04-5 ยังแสดงด้วยว่า `sandbox_type: local` ถูกปฏิเสธ (`Input should be 'piston'`) Docker daemon บน Mac เครื่องนี้ปิดอยู่ จึงรัน Piston server ไม่ได้ tool นี้จึงมีแค่ config หากอยากลอง ให้เปิด Docker, deploy Piston ตามที่ README ของ `code_execution` ใน NAT อธิบาย แล้วชี้ `uri` ไปที่มัน

ภายใน claw นั้น Piston server คือ network endpoint อีกหนึ่งตัว ต้องมี entry ใน `network_policies` ของตัวเอง และโค้ดของ agent จะรันที่นั่น ไม่ใช่บน host

✓ Checkpoint: คุณบอกได้ว่า NAT 1.9 รองรับ backend สำหรับรันโค้ดแบบไหน และทำไมแล็ปท็อปนี้ยังรันไม่ได้ตอนนี้

## 8 · L3.8 — รัน NAT ใน OpenShell sandbox

ตรงนี้คือจุดที่สองครึ่งของคอร์สมาบรรจบกัน: workflow ของ NAT จากหัวข้อ 3–6 เข้าไปอยู่ในขอบเขตของ OpenShell จากโมดูล 03 แผนนี้มีสามไฟล์ Lab 04-5 สร้างทั้งหมดไว้ใน `week26/04_nat_claws/.runs/alto-ops-sandbox/` พร้อมแพ็กเกจ `alto_ops` และไฟล์ CSV

**1 · `Dockerfile.alto-ops`** ไฟล์นี้ research tutorial เขียนขึ้นเอง ไม่ใช่ image ของ NVIDIA และยังไม่เคย build บน Mac เครื่องนี้ (Docker daemon ปิดอยู่) ให้ตรวจบรรทัดติดตั้ง NAT และ `USER` ที่ไม่ใช่ root กับ base image ของคุณเอง OpenShell ต้องการ identity ที่ไม่ใช่ root และปฏิเสธ root

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

**2 · `workflow.sandbox.yml`** ต่างจาก workflow บนแล็ปท็อปแค่บล็อก LLM (บวก path ของ CSV ภายใน sandbox) `base_url` คือ `https://inference.local/v1` ซึ่งเป็น managed route และ `api_key` ยังเป็น `EMPTY` เพราะ gateway จะใส่ credential จริงให้ `model_name` ต้องตรงกับโมเดลที่ตั้งด้วย `openshell inference set` เนื่องจาก process Python ของ NAT เป็นผู้เรียก TLS trust ของ proxy จึงถูกใส่มาให้แล้วผ่าน `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` ทำให้ `https://inference.local/v1` ใช้ได้ทันทีกับ client แบบ `openai`

**3 · `alto-ops-policy.yaml`** **ไม่มี** network entry เลย inference ไม่ใช่ entry ใน `network_policies` ที่คุณต้องเพิ่ม เพราะ supervisor ดักจับ `inference.local` ให้

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

Lab 04-5 ตรวจ policy นี้สองรอบ: รอบแรกด้วย teaching model `policykit` ของคอร์ส และรอบที่สองด้วย parser ของ OpenShell 0.0.111 ตัวจริง

**Expected output** (บันทึกจากเครื่อง Mac นี้, lab 04-5 steps 3–5)

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

`policykit` เป็น teaching model การบังคับใช้จริงคือ Landlock, seccomp และ proxy บน Spark ส่วนการตรวจด้วย parser เป็นของจริง: OpenShell อ่าน YAML ผ่าน แล้วล้มเหลวเพียงเพราะแล็ปท็อปนี้ไม่มี gateway

> **Parser differs.** research tutorial สร้าง sandbox ในบรรทัดเดียว ด้วย `--upload ./data:/sandbox/data` และ `-- nat serve …` ต่อท้าย parser ของ OpenShell 0.0.111 บนแล็ปท็อปปฏิเสธการใช้สองอย่างนี้ร่วมกัน NemoClaw ปักเวอร์ชัน 0.0.116 บน Spark จึงควรเช็ก `openshell sandbox create --help` ที่นั่น รูปแบบที่แยกคำสั่งด้านล่าง parse ผ่านบน 0.0.111 และเพราะ `chiller_kpi` เปิดไฟล์ CSV ใหม่ทุกครั้งที่ถูกเรียก การอัปโหลดข้อมูลหลัง server เริ่มแล้วจึงไม่มีปัญหา

บน Spark ให้คัดลอกโฟลเดอร์ไปก่อน (`scp -r week26/04_nat_claws/.runs/alto-ops-sandbox <spark>:~/works/alto-ops-claw/`) แล้วรัน:

```bash
# on: spark
cd ~/works/alto-ops-claw/alto-ops-sandbox
openshell inference get
openshell sandbox create --name alto-ops --from ./Dockerfile.alto-ops --policy ./alto-ops-policy.yaml --forward 8001 --keep --detach -- nat serve --config_file /app/workflow.yml --host 0.0.0.0 --port 8001
openshell sandbox upload alto-ops ./data /sandbox/data
openshell forward start --background 8001 alto-ops
openshell logs alto-ops --tail --source sandbox
```

มีสามจุดที่ต่างจากบรรทัดของ tutorial และแต่ละจุดมีเหตุผล:

| ทางเลือก | เหตุผล |
|---|---|
| `--from ./Dockerfile.alto-ops` แทน `--from ./` | help บอกว่า `--from` รับ "a path to a Dockerfile or directory containing one" และไฟล์นี้ไม่ได้ชื่อ `Dockerfile` |
| `--detach` | มีใน 0.0.111: เริ่ม main process โดยไม่ attach เทอร์มินัลของคุณ (และแล็บ) จึงไม่ค้างอยู่ที่ `nat serve` |
| `sandbox upload` เป็นขั้นแยก | 0.0.111 ปฏิเสธ `--upload` ที่ใช้คู่กับคำสั่งต่อท้าย |

log แบบ `--tail` จะตามไปเรื่อย ๆ ไม่หยุด ให้รันเองในเทอร์มินัล ⌨ ส่วน Lab 04-5 อ่านแบบ `-n 50` แทน จากนั้นบน Spark host คำสั่ง `curl -s http://localhost:8001/health` ควรตอบ `{"status":"healthy"}` ในรูปแบบเดียวกับที่ lab 04-3 ได้บนแล็ปท็อปนี้ ส่วนใน NemoClaw คำสั่งที่เทียบเท่าสำหรับ image ที่สร้างเองคือ `nemoclaw onboard --from <Dockerfile>`

✓ Checkpoint: คุณบอกได้ว่า YAML บรรทัดไหนบรรทัดเดียวที่ย้าย NAT workflow เข้า sandbox (`base_url: https://inference.local/v1`) และทำไม policy ถึงไม่มี network entry เลย

## Labs — run them here

**labs/lab04_1_hello_workflow.py** — NAT บนแล็ปท็อปนี้, scaffold ของ 1.9 และ ReAct workflow ขั้นต่ำ รันแบบต้นฉบับก่อน แล้วรันอีกครั้งแบบปิด thinking

**labs/lab04_2_custom_tool.py** — tool chiller_kpi: หาคำตอบที่ถูกต้องโดยไม่ใช้ LLM แล้วให้ agent ตอบด้วย description เดิมและ description ที่แก้แล้ว

**labs/lab04_3_serve.py** — nat serve บนพอร์ตที่ว่าง: route จริง, /v1/workflow, /v1/chat/completions และ stream ของขั้นตอนจาก /full พร้อมนับจำนวน

**labs/lab04_4_mcp_both_ways.py** — nat mcp serve และ nat mcp client จากนั้นให้ NAT ใช้ BMS จำลองโดยเปิดเฉพาะ tool อ่านค่า

**labs/lab04_5_nat_in_sandbox.py** — build context และ policy ของ sandbox ตรวจด้วย policykit และ parser ของ OpenShell ตัวจริง ขั้นตอนบน Spark และ backend สำหรับรันโค้ดของ NAT 1.9

## Try it yourself

`exercises/ex04_bms_tools.py` มีสาม TODO เมื่อรวมกันแล้วจะให้ Alto Ops Claw เข้าถึง BMS แบบสองชั้น:

1. เขียนบล็อก `function_groups`: `bms_tools`, `_type: mcp_client`, streamable-HTTP ไปที่ `http://bms.alto.local:8443/mcp` และเสนอเฉพาะ `read_point` + `list_alarms` เพิ่ม description ใน `tool_overrides` ของ `read_point` ที่บอกว่า RT หมายถึงอะไร
2. เขียนชื่อ tool ที่ LLM เห็นสำหรับ `read_point` ให้ถูกต้องทุกตัวอักษร
3. เขียน entry `bms_mcp` ใน `network_policies` ตาม exercise ข้อ 4 ของ Part 3 ใน research tutorial: `protocol: mcp`, `enforcement: enforce`, MCP handshake, `tools/call` เฉพาะ `read_point` และ `list_alarms` และ entry ใน `deny_rules` สำหรับ `write_setpoint`

checker ทำงานแบบ offline มันให้ model `MCPClientConfig` ของ NAT 1.9 validate บล็อกของคุณ รัน `policykit.decide` กับ MCP action เจ็ดแบบ และให้ parser ของ OpenShell 0.0.111 อ่าน policy ของคุณ

```bash
# on: laptop
.venv/bin/python week26/04_nat_claws/exercises/ex04_bms_tools.py
```

**Expected output** (บันทึกจากเครื่อง Mac นี้ เมื่อทำครบทุก TODO)

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

<details><summary>Hint — TODO 1 ใช้ include หรือ exclude?</summary>

ใช้ `include: [read_point, list_alarms]` allow-list ยังปลอดภัยแม้พรุ่งนี้ BMS จะเพิ่ม tool ใหม่ ขณะที่รายการ `exclude` จะเสนอ tool ใหม่นั้นให้ agent แบบเงียบ ๆ และบน NAT 1.9 ใช้ทั้งสองอย่างใน group เดียวกันไม่ได้

</details>

<details><summary>Hint — TODO 3 กฎของ MCP</summary>

กฎ MCP จับคู่กับ `method` ของ JSON-RPC และสำหรับ `tools/call` ก็จับคู่กับ `tool` ด้วย client ต้อง `initialize` ส่ง `notifications/initialized` และเรียก `tools/list` ก่อน `tools/call` ใด ๆ จึงต้องอนุญาตสามอย่างนี้ด้วย `tool: { any: [read_point, list_alarms] }` จับได้ทั้งสองชื่อ และ binary คือ process ของ NAT ใน sandbox คือ `/usr/bin/python3.12`

</details>

<details><summary>Hint — ไปต่ออีกขั้น (research tutorial Part 3, exercise ข้อ 1 และ 2)</summary>

เปลี่ยน `_type: tool_calling_agent` เป็น `react_agent` (คง `reasoning_effort: none` ไว้) แล้วนับขั้น LLM_END ด้วย `/v1/workflow/full` สำหรับคำถาม "give me the 6-hour and 24-hour kW/RT" ให้วัดจริง อย่าเดา จากนั้นอ่าน `week26/common/alto_ops/src/alto_ops/setpoint_tool.py`: `input_schema` แบบ Pydantic ของมันปฏิเสธการสร้าง ticket ถ้าไม่มี `work_order_id` นั่นคือชั้น (a) ในคำตอบสามชั้นของ research tutorial ส่วน policy ใน Exercise 04 คือชั้น (c) ซึ่งเป็นชั้นเดียวที่รอดจาก prompt injection

</details>

✓ Checkpoint: checker ทั้งสิบเอ็ดบรรทัดขึ้น ✓

## Troubleshooting

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `ReActAgentParsingFailedError … LLM output: ''` | thinking model ใส่คำตอบทั้งหมดไว้ในฟิลด์ reasoning | เพิ่ม `reasoning_effort: none` ใต้ LLM หรือใช้ `tool_calling_agent` |
| `Invalid workflow directory specified. … does not exist.` | NAT 1.9 ต้องการให้ `--workflow-dir` มีอยู่แล้ว | สร้างด้วย `mkdir -p` ก่อน |
| `nat serve` หยุดพร้อมข้อความ `…requires that the Python 'greenlet' library is installed` | NAT 1.9.0 + SQLAlchemy 2.1 ที่ไม่มี greenlet | `uv pip install --python week26/.venv-nat/bin/python greenlet` |
| `Address already in use` บน 8001, 9901 หรือ 8443 | มีแล็บอื่น (หรือผู้เขียนโมดูลอื่น) รัน server อยู่ | แล็บจะเลือกพอร์ตว่างถัดไปด้วย `free_port()` และพิมพ์บอก ส่วน server ที่คุณรันเองแบบ foreground ให้หยุดด้วย Ctrl-C |
| `Invalid configuration: function_groups: Value error, include and exclude cannot be used together` | ใส่ทั้งสองรายการใน group เดียว | เก็บไว้แค่ `include` |
| `error: the argument '--upload <UPLOAD>' cannot be used with '[COMMAND]...'` | กฎของ argument ใน OpenShell 0.0.111 | สร้าง sandbox ก่อน แล้วค่อย `openshell sandbox upload` |
| คำตอบของ AI ทุกข้อใช้ 40–70 วินาที แทนที่จะเป็น 10–15 วินาที | แล็บอื่นใช้ Ollama บนแล็ปท็อปร่วมกันอยู่ | รันทีละแล็บ อย่างไรก็ตามเวลาบนแล็ปท็อปเป็น LAPTOP STAND-IN อยู่แล้ว |
| มี `AuthlibDeprecationWarning` ทุกครั้งที่รัน `nat` | คำเตือนจาก dependency ของ NAT 1.9.0 | ไม่เป็นอันตราย แล็บกรองออกให้ และ `2>/dev/null` ซ่อนไว้ในเทอร์มินัล |

## Next

[Lab 05 — Tracing and observability: agent, policy and harness planes](../05_tracing/TUTORIAL.md): เปลี่ยนขั้น LLM_END และ TOOL_END ที่คุณนับไว้ที่นี่ให้เป็น trace ใน Phoenix และ OpenTelemetry แล้ววางเทียบกับ policy log ของ OpenShell สำหรับ request เดียวกัน
