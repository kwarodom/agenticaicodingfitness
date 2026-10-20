# ▶ Spark Lab 14 — NeMo Agent Toolkit: เอเจนต์บนโมเดลใน Spark ของคุณ

> ส่วนหนึ่งของ Week 25 · DGX Spark: fine-tune, serve และสร้างเอเจนต์ที่รันใน sandbox คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง ทุกแล็บรันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (อ้างอิงคำต่อคำจาก playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- ติดตั้ง NVIDIA NeMo Agent Toolkit (NAT) 1.9 ใน venv ของมันเอง ทั้งบนแล็ปท็อปและบน Spark
- อ่านไฟล์ workflow ของ NAT: สามส่วน (`llms`, `functions`, `workflow`) เปลี่ยน model server ให้กลายเป็นเอเจนต์
- เขียน tool สำหรับงานปฏิบัติการโรงแรมสองตัวเป็น plugin package ขนาดเล็กของ NAT แล้วดู NAT ค้นพบมัน
- รันเอเจนต์ด้วย `nat run` กับ vLLM บน Spark (หรือตัวแทนบนแล็ปท็อป) แล้วอ่าน trace ที่มันเขียนไว้
- serve เอเจนต์ตัวเดียวกันเป็นเว็บเซอร์วิสที่เข้ากันได้กับ OpenAI (`nat serve`) และ serve tool ของมันเป็น MCP server (`nat mcp serve`)
- ให้คะแนนเอเจนต์ด้วย `nat eval` และ evaluator ที่ไม่ต้องใช้ LLM เป็นกรรมการ (judge)

**Time** ~50 นาที · **Difficulty** ระดับกลาง · **Hardware** Spark 1 เครื่อง (หรือไม่มีเลยก็ได้: ตัวแทนบนแล็ปท็อปรันทุกแล็บได้จริง)

**Playbook ทางการที่ครอบคลุม:** ไม่มี NAT ไม่มี DGX Spark playbook โมดูลนี้จึงเป็น **เนื้อหาเฉพาะของคอร์ส (course-original)**: ทุกคำสั่ง ชื่อฟิลด์ และผลลัพธ์ด้านล่างถูกตรวจกับแพ็กเกจ `nvidia-nat` 1.9.0 ที่ติดตั้งจริง (`nat --help`, `nat info components`, source ของแพ็กเกจ) โดยต่อยอดจาก playbook [vLLM](https://build.nvidia.com/spark/vllm) (Module 05) ในส่วนของ model server

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| Python ของ repo นี้ | `.venv/bin/python --version` → 3.13 | ใช้รันแล็บ |
| `uv` บนแล็ปท็อป | `uv --version` | ใช้สร้าง venv ของ NAT (Python 3.12) |
| model server ที่รองรับ tool calling | Spark: vLLM จาก Module 05 ที่ `:8000` · แล็ปท็อป: Ollama ที่ `:11434` | สมองของเอเจนต์ |
| โมเดลที่ทำ tool calling ได้บนแล็ปท็อป | `curl -s localhost:11434/v1/models` แสดง `nemotron-3-nano:latest` | ตัวแทนบนแล็ปท็อป |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
.venv/bin/python --version
uv --version
curl -s localhost:11434/v1/models | head -c 200
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
Python 3.13.13
uv 0.11.19 (7b2cff1c3 2026-06-03 aarch64-apple-darwin)
{"object":"list","data":[{"id":"nemotron-3.5-lightning:latest","object":"model","created":…
```

> 💡 NAT รันใน **venv ของมันเอง** (`week25/.venv-nat`, Python 3.12) ไม่ใช่ใน `.venv` ของ repo แล็บรันด้วย Python ของ repo แล้วเรียก CLI `nat` เป็น subprocess ทั้งสองจึงไม่ชนกันเลย

✓ Checkpoint: `uv --version` พิมพ์เวอร์ชันออกมา และ vLLM บน Spark หรือ Ollama บนแล็ปท็อปแสดงรายชื่อโมเดลอย่างน้อยหนึ่งตัว

## 1 · NAT เพิ่มอะไรให้บน model server

Module 05 ให้เซิร์ฟเวอร์ที่เข้ากันได้กับ OpenAI แก่คุณ เซิร์ฟเวอร์ตอบทีละคำขอ แต่ **เอเจนต์** ต้องมีลูปครอบมันไว้: ส่งคำถามและรายการ tool รัน tool ที่โมเดลขอ ส่งผลกลับไป แล้ววนซ้ำจนโมเดลตอบ NAT คือลูปนั้น บวกกับส่วนที่ปกติคุณต้องเขียนเอง

| ชั้น | สิ่งที่คุณเขียน | สิ่งที่ NAT ให้ |
|---|---|---|
| โมเดล | ไม่ต้องเขียน (vLLM, Ollama, NIM, LiteLLM) | LLM client สำหรับ URL ใดก็ได้ที่เข้ากันได้กับ OpenAI |
| Tool | ฟังก์ชัน async ธรรมดาของ Python | schema จาก type hint, การลงทะเบียน, ตัวครอบ tool สำหรับ LangChain |
| ลูปของเอเจนต์ | บรรทัดเดียว: `_type: tool_calling_agent` | เอเจนต์แบบ ReAct, tool-calling, ReWOO, router |
| หน้าบ้าน (front end) | ไม่ต้องเขียน | console (`nat run`), REST + OpenAI API (`nat serve`), MCP (`nat mcp serve`) |
| คุณภาพ | ชุดข้อมูลเล็ก ๆ | `nat eval`, tracing ลงไฟล์หรือ OpenTelemetry |

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

NAT เป็น Python ล้วน ไม่มีส่วนไหนต้องใช้ GPU คนที่ใช้คือ model server ดังนั้นคุณรัน NAT บน Spark ข้าง vLLM ได้ หรือรันบนแล็ปท็อปแล้วชี้ไปที่ Spark ผ่าน tailnet ก็ได้ Module 15 จะนำมันไปไว้ใน OpenShell sandbox บน Spark

✓ Checkpoint: คุณบอกได้ว่าส่วนไหนของเอเจนต์ที่ NAT รัน และส่วนไหนที่ vLLM รัน

## 2 · ติดตั้ง NAT และลงทะเบียน tool ของคอร์ส

บนแล็ปท็อป คอร์สเก็บ NAT ไว้ใน `week25/.venv-nat` มีสามแพ็กเกจ: `nvidia-nat[langchain]` (ตัวเอเจนต์), `greenlet` (`nat serve` ของ NAT 1.9 import โมดูล asyncio ของ SQLAlchemy ซึ่งต้องใช้มัน และมันไม่ถูกดึงมาให้อัตโนมัติ) และ `nvidia-nat-mcp` (MCP client และ server):

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
uv venv -p 3.12 week25/.venv-nat
uv pip install --python week25/.venv-nat/bin/python "nvidia-nat[langchain]~=1.9" greenlet "nvidia-nat-mcp~=1.9"
uv pip install --python week25/.venv-nat/bin/python --no-deps -e week25/14_nat_agents/hotel_ops_nat
week25/.venv-nat/bin/nat --version
```

บน Spark (aarch64, DGX OS) ติดตั้งแบบเดียวกันไว้ใน `~/w25` ให้ clone repo นี้บน Spark ก่อน (`git clone <this repo> ~/agenticaicodingfitness`) เพราะ Module 15 จะอัปโหลดโฟลเดอร์โมดูลจากที่นั่นเข้าไปใน sandbox:

```bash
# on: spark
mkdir -p ~/w25 && cd ~/w25 && python3 -m venv .venv-nat
.venv-nat/bin/pip install -q 'nvidia-nat[langchain]~=1.9' greenlet 'nvidia-nat-mcp~=1.9'
.venv-nat/bin/pip install -q --no-deps -e ~/agenticaicodingfitness/week25/14_nat_agents/hotel_ops_nat
.venv-nat/bin/nat --version
```

Lab 14-1 ตรวจการติดตั้ง ลงทะเบียน tool ถ้ายังไม่ได้ทำ แสดงรายการสิ่งที่ NAT มองเห็น และตรวจความถูกต้องของไฟล์ workflow ทุกไฟล์ในโมดูลนี้:

```bash
# on: laptop
.venv/bin/python week25/14_nat_agents/labs/lab14_1_install_and_register.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ ตัดให้สั้นลง)

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

✓ Checkpoint: `room_temperature` และ `create_maintenance_ticket` ปรากฏในตาราง component โดยมี package เป็น `hotel_ops_nat` และ config ทั้งสี่ไฟล์เป็น ✓ valid

## 3 · ไฟล์ workflow: `llms`, `functions`, `workflow`

นี่คือเอเจนต์ทั้งตัว (`configs/hotel_agent.yml` แบบตัดให้สั้นลง):

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

กฎสี่ข้อที่อธิบาย config ของ NAT ทุกไฟล์ที่คุณจะได้อ่าน:

1. **`_type` เลือก component ที่ลงทะเบียนไว้** `openai`, `tool_calling_agent` และ `current_datetime` มาพร้อม NAT ส่วน `room_temperature` มาจาก plugin ของคุณ `nat info components` แสดงรายการทั้งหมด
2. **ชื่อเป็นตัวเชื่อมระหว่างส่วน** `llm_name: hotel_llm` ชี้ไปที่คีย์ใต้ `llms` แต่ละรายการใน `tool_names` ชี้ไปที่คีย์ใต้ `functions` (หรือ `function_groups`)
3. **`${VAR:-default}` ถูกขยายค่าตอนที่ NAT โหลดไฟล์** ไฟล์เดียวใช้ได้ทั้งแล็ปท็อปและ Spark คุณเปลี่ยน URL ด้วย environment variable หรือด้วย `--override llms.hotel_llm.base_url …`
4. **`base: other.yml` คือการสืบทอด** NAT โหลดไฟล์อื่นก่อนแล้ว merge ไฟล์นี้ทับด้านบน รุ่นสำหรับ Spark และ config สำหรับ eval ใช้วิธีนี้

**โมเดลแบบ thinking ต้องมีสวิตช์** `nemotron-3-nano`, `gemma4` และ Qwen3.6 ของ playbook ล้วน "คิด" ก่อนตอบ และการคิดอาจใช้งบ `max_tokens` จนหมด สวิตช์ต่างกันไปตามเซิร์ฟเวอร์:

| เซิร์ฟเวอร์ | วิธีปิด thinking | อยู่ที่ไหน |
|---|---|---|
| Ollama (ตัวแทนบนแล็ปท็อป) | `reasoning_effort: none` | `hotel_agent.yml` (ทดสอบแล้วบน Mac เครื่องนี้) |
| vLLM + Qwen3.6 (Spark) | `extra_body.chat_template_kwargs.enable_thinking: false` และ `reasoning_effort: null` เพื่อตัดฟิลด์ของ Ollama ทิ้ง | `hotel_agent_spark.yml` (ตรวจด้วย `nat validate` และด้วยการสร้าง client แล้ว แต่ยังไม่ได้รันกับ Spark จริง) |

LLM config แบบ OpenAI ใน NAT รับฟิลด์เพิ่มเติมได้และส่งต่อให้ `ChatOpenAI` ของ LangChain ทั้งสองแบบจึงใช้ได้ รุ่นสำหรับ Spark มีแค่นี้:

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

✓ Checkpoint: คุณชี้ได้ว่าบรรทัดไหนเป็นตัวตัดสินว่าเอเจนต์เรียกเซิร์ฟเวอร์ตัวไหน และอธิบายได้ว่าทำไมรุ่นสำหรับ Spark ตั้ง `reasoning_effort: null`

## 4 · เขียน tool ของคุณเอง: plugin package ของ NAT

NAT หา component ผ่าน **entry point** ของ Python ไม่ใช่ด้วยการ import ไฟล์ที่คุณระบุชื่อใน YAML ดังนั้น tool ที่เขียนเองจะอยู่ใน package เล็ก ๆ `nat workflow create <name>` สร้างให้ได้ และ `hotel_ops_nat/` มีโครงสร้างแบบเดียวกัน:

```text
hotel_ops_nat/
  pyproject.toml                 [project.entry-points.'nat.components']  hotel_ops_nat = "hotel_ops_nat.register"
  src/hotel_ops_nat/register.py  imports tools.py, so the decorators run
  src/hotel_ops_nat/tools.py     two @register_function tools (fake, deterministic)
  src/hotel_ops_nat/evals.py     an offline evaluator for nat eval (Section 7)
```

tool หนึ่งตัวจาก `tools.py`:

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

สามสิ่งที่ควรสังเกต:

- **Config class = บล็อกใน YAML** ฟิลด์ใน `RoomTemperatureConfig` กลายเป็นคีย์ที่ YAML ตั้งค่าได้ (`comfort_band_c`) และ NAT ตรวจความถูกต้องให้
- **Type hint และ docstring กลายเป็น schema ของ tool** ที่โมเดลเห็น จงเขียนให้โมเดลอ่าน: บอกว่า argument หน้าตาเป็นอย่างไร ("808")
- **Tool ถูกออกแบบให้ปลอดภัย** `room_temperature` อ่านตารางสี่ห้อง `create_maintenance_ticket` คืน `MT-` ตามด้วย hash ของห้องและปัญหา ความเสียหายเดียวกันจึงได้ id เดียวกันเสมอ และมันเขียนแค่ไฟล์ JSONL ในเครื่องที่คุณเลือก เอเจนต์เรียกได้บ่อยเท่าที่ต้องการ ภายหลังค่อยเปลี่ยนเนื้อในฟังก์ชันเป็น API ของระบบอาคารของคุณ โดย YAML ไม่ต้องเปลี่ยน

ติดตั้งครั้งเดียวด้วย `uv pip install … --no-deps -e week25/14_nat_agents/hotel_ops_nat` (ส่วนที่ 2) เป็นแบบ editable: แก้ `tools.py` แล้ว `nat run` ครั้งถัดไปจะใช้โค้ดใหม่

✓ Checkpoint: คุณบอกได้สามจุดที่ tool ใหม่ต้องปรากฏ: config class ที่มี `name=`, `@register_function` และการ import ใน `register.py`

## 5 · รันเอเจนต์ แล้วอ่าน trace ของมัน

`nat run` คือหน้าบ้านแบบ console: คำถามเข้าหนึ่งข้อ คำตอบออกหนึ่งข้อ บน Spark ให้ชี้รุ่นสำหรับ Spark ไปที่ vLLM บน `:8000`:

```bash
# on: spark
cd ~/agenticaicodingfitness/week25/14_nat_agents
HOTEL_LLM_BASE_URL=http://localhost:8000/v1 ~/w25/.venv-nat/bin/nat run \
  --config_file configs/hotel_agent_spark.yml --input "Room 808 feels hot. Check it and open a ticket if something is wrong."
```

Lab 14-2 ทำแบบเดียวกันจากแล็ปท็อป มันเลือก vLLM บน Spark ถ้าตอบ ถัดไปคือ Ollama บน Spark แล้วจึงเป็นตัวแทนบนแล็ปท็อป และพิมพ์บอกว่าใช้ตัวไหน:

```bash
# on: laptop
.venv/bin/python week25/14_nat_agents/labs/lab14_2_run_agent.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ — LAPTOP STAND-IN, `nemotron-3-nano:latest` บน Ollama)

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

โมเดลเรียก tool สองครั้งด้วยตัวเอง: อ่านค่าห้องก่อน เห็นความผิดปกติ แล้วเปิดตั๋วแจ้งซ่อม ตรงตามที่ system prompt ขอ เวลาที่แสดงเป็นของแล็ปท็อปนี้ (มีเอเจนต์อีกตัวใช้ Ollama ตัวเดียวกันอยู่) อย่านำไปเทียบกับ Spark

**Observability** (การสังเกตการทำงาน) คือบล็อกเดียวที่ด้านบนของไฟล์ workflow exporter แบบ `file` เขียน trace ที่คุณเพิ่งอ่าน exporter อื่นที่ติดตั้งมากับ NAT (`otelcollector`, `langfuse` และอื่น ๆ ผ่าน extra เช่น `nvidia-nat[phoenix]`) ส่งเหตุการณ์ชุดเดียวกันไปยัง UI สำหรับ tracing:

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

✓ Checkpoint: การรันของคุณแสดงบรรทัด `→ tool_call` อย่างน้อยหนึ่งบรรทัด และตาราง trace แสดงเหตุการณ์ START และ END ของ tool

## 6 · Serve มัน: endpoint แบบ REST / OpenAI และ MCP server

ไฟล์ YAML เดียวกันรันอยู่หลังหน้าบ้านได้อีกสองแบบ

**`nat serve`** เปิดเซิร์ฟเวอร์ FastAPI มันเปิด `/generate`, `/chat`, รุ่นที่สตรีม และ **`/v1/chat/completions`** ที่เข้ากันได้กับ OpenAI ดังนั้น Open WebUI หรือ LiteLLM (Module 08) จึงมองเอเจนต์ทั้งตัวเหมือนเป็นโมเดลตัวหนึ่งได้ คอร์สใช้พอร์ต **8400** เพราะ `:8000` เป็นพอร์ตของ vLLM บน Spark บน Spark ให้ bind กับ `0.0.0.0` เฉพาะเมื่อคุณต้องการให้ tailnet เข้าถึงได้เท่านั้น `nat serve` ไม่มีการยืนยันตัวตนตามค่าเริ่มต้น

```bash
# on: spark
cd ~/agenticaicodingfitness/week25/14_nat_agents
HOTEL_LLM_BASE_URL=http://localhost:8000/v1 ~/w25/.venv-nat/bin/nat serve \
  --config_file configs/hotel_agent_spark.yml --host 127.0.0.1 --port 8400
```

**`nat mcp serve`** เผยแพร่ฟังก์ชันเป็น **MCP tool** (streamable HTTP ที่ `:9901/mcp` ตามค่าเริ่มต้น) `--tool_names` จำกัดให้เหลือแค่ tool โรงแรมสองตัว จากนั้น MCP client ใดก็ได้จะแสดงรายการและเรียกมันได้ ส่วน `configs/hotel_agent_mcp.yml` คือเอเจนต์ NAT ที่ tool มาจากเซิร์ฟเวอร์นั้นผ่าน `function_groups: {_type: mcp_client}`

Lab 14-3 เปิดเซิร์ฟเวอร์ทั้งสอง เรียกใช้ แล้วปิด:

```bash
# on: laptop
.venv/bin/python week25/14_nat_agents/labs/lab14_3_serve_and_mcp.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ — LAPTOP STAND-IN ตัดให้สั้นลง)

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

ดูคำตอบของห้อง 1510: ห้องอยู่ใน "guest away mode" ซึ่งอาจถือว่าปกติก็ได้ แต่โมเดลเล็กตัวนี้ก็ยังเปิดตั๋วอยู่ดี นี่คือการใช้วิจารณญาณที่ prompt ไม่ได้กำหนดไว้ชัด และเป็นพฤติกรรมแบบที่ส่วนที่ 7 วัดพอดี

✓ Checkpoint: `/v1/chat/completions` คืน `object=chat.completion` และ `tool list` แสดง tool โรงแรมทั้งสองตัว

## 7 · ประเมินผล: `nat eval`

การประเมินเอเจนต์มีสามส่วน: ชุดข้อมูล workflow และ evaluator หนึ่งตัวขึ้นไป `configs/hotel_eval.yml` สืบทอดเอเจนต์ (`base: hotel_agent.yml`) แล้วเพิ่ม:

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

แต่ละแถวของชุดข้อมูลมี `question` และ `answer` ที่ระบุข้อเท็จจริงที่คำตอบต้องมี (`"808|MT-"`) `mentions_expected` ให้คะแนนเป็นสัดส่วนของข้อเท็จจริงที่พบ: ผลแน่นอนตายตัว (deterministic) ฟรี และไม่มี LLM เป็นกรรมการ NAT 1.9 ยังมี `trajectory` (LLM ตัดสินเส้นทางการเรียก tool) และ `langsmith` (`exact_match`, `levenshtein_distance`) ให้ด้วย เพิ่มพวกมันไว้ใต้ `evaluators` เมื่อคุณมีโมเดลกรรมการบน Spark

```bash
# on: laptop
.venv/bin/python week25/14_nat_agents/labs/lab14_4_eval.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ — LAPTOP STAND-IN)

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

คะแนนเต็มบนคำถามสามข้อพิสูจน์ว่าท่อต่อถึงกัน ไม่ได้พิสูจน์ว่าเอเจนต์ดี คุณค่าจะมาเมื่อคุณเปลี่ยนสิ่งหนึ่งแล้วรันใหม่: base model เทียบกับ fine-tune ของคุณ (Module 13), `tool_calling_agent` เทียบกับ `react_agent`, system prompt ที่ยาวขึ้น NAT บันทึก `config_effective.yml` ไว้ข้างผลลัพธ์ คะแนนแต่ละครั้งจึงบันทึกไว้ชัดว่าเกิดจากอะไร

✓ Checkpoint: มี `mentions_expected_output.json` อยู่ใต้ `week25/14_nat_agents/.runs/eval/` โดยมีคะแนนหนึ่งค่าต่อคำถาม

## Labs — รันแล็บได้ที่นี่

**labs/lab14_1_install_and_register.py** — หา CLI ของ NAT ลงทะเบียน plugin hotel_ops_nat แสดงรายการ component ของ NAT และตรวจความถูกต้องของไฟล์ workflow ทั้งสี่

**labs/lab14_2_run_agent.py** — รันเอเจนต์โรงแรมด้วย `nat run` บน vLLM ของ Spark หรือตัวแทนบนแล็ปท็อป สตรีม tool call ของมัน และพิมพ์ trace

**labs/lab14_3_serve_and_mcp.py** — serve เอเจนต์ผ่าน REST และ OpenAI API ด้วย `nat serve` แล้วเผยแพร่ tool ของมันด้วย `nat mcp serve` และเรียกใช้

**labs/lab14_4_eval.py** — ให้คะแนนเอเจนต์บนชุดข้อมูลสามคำถามด้วย `nat eval` และ evaluator แบบ deterministic

ทั้งสี่แล็บรันบนแล็ปท็อปกับ Ollama ของแล็ปท็อปเมื่อไม่มี Spark ตอบ และติดป้ายผลลัพธ์นั้นว่า LAPTOP STAND-IN เมื่อเชื่อม Spark และ vLLM ทำงานอยู่ lab 14-2 ถึง 14-4 จะใช้มันอัตโนมัติ

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 14 — ต่อสายเอเจนต์ด้วยตัวเอง** เปิด `week25/14_nat_agents/exercises/ex14_wire_the_agent.py` `TODO` แต่ละจุดในสามจุดคืนส่วนหนึ่งของ workflow ของ NAT เป็น dict ของ Python:

1. `llm_block(target, base_url, model)`: บล็อก `llms.hotel_llm` โดยปิด thinking ด้วยวิธีที่ถูกต้องสำหรับ Ollama และสำหรับ vLLM
2. `functions_block(ticket_log)`: tool ทั้งสามตัว แต่ละตัวมี `_type` เท่ากับชื่อที่ลงทะเบียนไว้
3. `workflow_block(llm_name, functions)`: `tool_calling_agent` ที่ใช้ทุกฟังก์ชัน

ตัวตรวจทำงานแบบออฟไลน์ มัน lint config ของคุณ เขียนเป็น YAML แล้วถาม `nat validate` ตัวจริงว่า NAT ยอมรับหรือไม่ (ไม่มีการเรียกโมเดล)

```bash
# on: laptop
.venv/bin/python week25/14_nat_agents/exercises/ex14_wire_the_agent.py
```

**Expected output** (เมื่อทำ TODO ครบทั้งสามจุด บันทึกจาก Mac เครื่องนี้)

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

<details><summary>คำใบ้ — ทำไมบล็อกของ vLLM ถึงไม่มี reasoning_effort?</summary>

`reasoning_effort` เป็นพารามิเตอร์ของ OpenAI ที่ Ollama เข้าใจ ส่วน vLLM ที่ใช้ chat template ของตระกูล Qwen3 ปิด thinking ผ่าน template แทน: `chat_template_kwargs: {enable_thinking: false}` ซึ่ง OpenAI client ส่งไปใน `extra_body` การส่งพารามิเตอร์ที่เซิร์ฟเวอร์ไม่คาดหวังอาจทำให้คำขอล้มเหลวได้ แต่ละบล็อกจึงมีเฉพาะสวิตช์ของตัวเอง

</details>

<details><summary>ท้าทายเพิ่ม — tool ตัวที่สาม และเอเจนต์อีกแบบ</summary>

เพิ่ม `guest_request(room, request)` ลงใน `hotel_ops_nat/src/hotel_ops_nat/tools.py` (config class, `@register_function`, import ใน `register.py`) เพิ่มมันใต้ `functions` แล้วรัน lab 14-2 ใหม่ด้วย "Room 402 needs extra towels" จากนั้นเปลี่ยน `workflow._type` เป็น `react_agent` แล้วเปรียบเทียบ trace: เอเจนต์แบบ ReAct เขียนข้อความ Thought / Action ที่ NAT ต้อง parse ขณะที่เอเจนต์แบบ tool-calling ใช้ tool call แบบ native ของเซิร์ฟเวอร์

</details>

✓ Checkpoint: บรรทัดตรวจทั้งหมดเป็น ✓ รวมถึงบรรทัด `nat validate` ทั้งสองบรรทัด

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `nat serve` ล้มเหลวด้วย `The SQLAlchemy asyncio module requires that the Python 'greenlet' library is installed` | `uv pip install --python week25/.venv-nat/bin/python greenlet` (ส่วนที่ 2 ติดตั้งให้แล้ว) |
| `nat mcp` ไม่ใช่คำสั่ง หรือไม่รู้จัก `mcp_client` | ติดตั้ง MCP extra: `nvidia-nat-mcp~=1.9` ใน venv เดียวกัน |
| `ValidationError … _type 'room_temperature'` | plugin ไม่ได้ติดตั้งใน venv ที่รัน `nat` รัน lab 14-1 ใหม่ หรือ `uv pip install --no-deps -e week25/14_nat_agents/hotel_ops_nat` |
| แก้ tool แล้วไม่เห็นผล | ติดตั้งด้วย `-e` (editable) การติดตั้งแบบไม่ใช่ editable คัดลอกโค้ดไปครั้งเดียว |
| เอเจนต์ตอบโดยไม่เรียก tool หรือคืนคำตอบว่างเปล่า | โมเดลแบบ thinking ใช้ `max_tokens` ไปกับการคิด ตรวจสวิตช์ในส่วนที่ 3 หรือเพิ่ม `max_tokens` |
| `Connection refused` ไปยัง `:8000` จากแล็ปท็อป | vLLM อยู่บน Spark: ใช้ `http://spark-a:8000/v1` หรือ tunnel `ssh -L 8000:localhost:8000` (Module 01) |
| AuthlibDeprecationWarning ทุกครั้งที่รันคำสั่ง | ไม่มีผลเสียใน NAT 1.9.0 แล็บตั้ง `PYTHONWARNINGS=ignore` และกรองมันออก |
| หา path แบบ relative (`.runs/…`, `configs/eval_dataset.json`) ไม่เจอ | รัน `nat` จาก `week25/14_nat_agents` เหมือนที่ทุกคำสั่งในที่นี้ทำ |

## Next — บทถัดไป

ไปต่อที่ [Lab 15 — OpenShell: sandbox และกำกับดูแล AI agent](../15_openshell_sandbox/TUTORIAL.md): นำเอเจนต์ NAT ตัวนี้ไปไว้ใน OpenShell sandbox บน Spark ที่ policy เป็นผู้ตัดสินว่ามันแตะไฟล์ไหนได้และเข้าถึง host ไหนได้
