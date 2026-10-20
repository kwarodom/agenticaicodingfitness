# ▶ Reef Lab 05 — Tracing และ observability: สามระนาบของ agent, policy และ harness

> ส่วนหนึ่งของ Week 26 · NemoClaw บน DGX Spark ตั้งแต่ระดับเริ่มต้นจนถึงระดับผู้เชี่ยวชาญ คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง แล็บบนแล็ปท็อป (CLI ของ NAT และ OpenShell รวมถึงโมเดล policy) รันได้จริงทุกที่ ส่วนแล็บบน Spark รันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (ยกมาจากเอกสารหรือ playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- รู้จักสามระนาบของ telemetry ใน claw และรู้ว่าแต่ละระนาบตอบคำถามอะไร
- trace การรันจริงของ Alto Ops Claw ด้วย exporter แบบ `file` ของ NAT 1.9 แล้วแยกออกมาเป็น LLM span, tool span, จำนวน token และระยะเวลา
- เพิ่ม Phoenix เป็น exporter ตัวที่สองเมื่อมันรันอยู่จริง และเข้าใจว่าทำไมการรันบนแล็ปท็อปจะไม่แกล้งทำเป็นว่ามี Phoenix
- เขียน config ของ OTel collector และ entry ใน sandbox policy ที่ NAT ใน sandbox ต้องมีเพื่อส่งข้อมูลไปหามันได้
- ลงทะเบียน key ของ Langfuse เป็น credential handle ของ OpenShell แล้วสแกน config ของ Hermes หา key ที่รั่ว
- อ่านการตัดสินใจของ policy ใน OpenShell เหมือนอ่าน trace แล้วเชื่อมทั้งสามระนาบเข้าด้วยกันสำหรับหนึ่ง request

**Time** ~75 นาที · **Difficulty** ระดับกลาง · **Hardware** แล็ปท็อป (NAT + Ollama รันจริง) · DGX Spark 1 เครื่อง (ไม่บังคับ ถ้าไม่มีก็รันแบบ DRY)

**แหล่งอ้างอิง:** research tutorial ของคอร์ส *NemoClaw on DGX Spark — Beginner to Expert* (Part 4, แล็บ L4.1–L4.5) ซึ่งอ้างถึง [NAT observe](https://docs.nvidia.com/nemo/agent-toolkit/latest/run-workflows/observe/observe.html) · [NAT profiler](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/profiler.html) · [Classmethod — NAT on DGX Spark](https://dev.classmethod.jp/en/articles/dgx-spark-nemo-agent-toolkit-local-intro/) · [NemoClaw Hermes quickstart](https://docs.nvidia.com/nemoclaw/user-guide/hermes/get-started/quickstart) · [NemoClaw network policies reference](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/reference/network-policies) · [OpenShell playbook](https://build.nvidia.com/playbooks/openshell/instructions) · [OpenShell sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies) · [OpenShell security best practices](https://docs.nvidia.com/openshell/security/best-practices)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| venv ของ NAT บนแล็ปท็อป | `week26/.venv-nat/bin/nat --version` → 1.9.0 | agent trace ทุกอันในโมดูลนี้มาจากการรัน NAT จริง |
| Ollama บนแล็ปท็อปที่มี `nemotron-3-nano:latest` | `curl -s localhost:11434/v1/models` | โมเดล LAPTOP STAND-IN ที่อยู่เบื้องหลัง Alto Ops Claw (ประมาณ 10–20 วินาทีต่อการเรียก LLM หนึ่งครั้ง) |
| CLI ของ OpenShell บนแล็ปท็อป | `week26/.venv-openshell/bin/openshell --version` → 0.0.111 | ใช้ parse คำสั่งของระนาบ policy แบบออฟไลน์ |
| Docker daemon (ไม่บังคับ) | `docker info` | ใช้เฉพาะ Phoenix และ OTel collector บนแล็ปท็อป ถ้าไม่มี ส่วนนั้นจะถูกข้ามไป ไม่มีการปลอมผลลัพธ์ |
| DGX Spark (ไม่บังคับ) | `ssh -o BatchMode=yes <spark> true` | ดู policy log ของ claw ที่อยู่ใน sandbox ถ้าไม่มี Spark ก็รันแบบ DRY |

Module 04 สร้าง Alto Ops Claw ไว้แล้ว: เป็น `tool_calling_agent` ของ NAT ที่มี tool ชื่อ `chiller_kpi` โมดูลนี้คือการเฝ้าดูมันทำงาน ทุกแล็บเขียนไฟล์ไว้ที่ `week26/05_tracing/.runs/`

```bash
# on: laptop
week26/.venv-nat/bin/nat --version
curl -s localhost:11434/v1/models | head -c 200
docker info --format '{{.ServerVersion}}'
```

✓ Checkpoint: NAT พิมพ์ 1.9.0 และ Ollama แสดง `nemotron-3-nano:latest` ในรายการ และคุณรู้แล้วว่า Docker daemon ของคุณรันอยู่หรือไม่

## 1 · สามระนาบ สามคำถาม

claw หนึ่งตัวสร้าง telemetry ออกมาสามแบบ มาจากคนละ process และตอบคนละคำถาม:

| ระนาบ | มาจาก | ตอบคำถาม |
|---|---|---|
| **Agent traces** | telemetry exporter ของ NAT (Phoenix, OTel collector, Langfuse, file …) และ plugin Langfuse ของ Hermes | โมเดลตัดสินใจอะไร tool ไหนรันบ้าง ใช้ไปกี่ token นานแค่ไหน |
| **Policy / inference logs** | supervisor และ gateway ของ OpenShell: `openshell logs`, `openshell term`, OCSF findings | เอเจนต์พยายามติดต่อที่ไหน และถูก allow, deny หรือ inspect; Landlock ถูกใช้งานหรือไม่ |
| **Harness logs** | `nemoclaw <sandbox> logs --follow`, `/tmp/gateway.log` ภายใน OpenClaw | event ของ channel, การ pairing, การ crash |

ระนาบ agent บอกว่าเอเจนต์ *ตั้งใจ* จะทำอะไร ระนาบ policy บอกว่ามัน *พยายาม* ทำอะไรบนเครือข่าย คุณต้องใช้ทั้งสองอย่าง เพราะเอเจนต์ที่โดน prompt injection อาจเขียน trace เป็นเรื่องที่ดูไม่มีพิษภัย ในขณะที่ egress ของมันเผยความจริง

observability ของ NAT ทำงานนอก hot path: `IntermediateStepManager` ส่ง event แบบ `IntermediateStep` (ขอบเขตของ function, การเรียก LLM, การเรียก tool) ออกไปใน reactive stream แล้ว exporter ก็ดึง stream นั้นไปใช้แบบ asynchronous และรันพร้อมกันได้หลายตัว research tutorial อ้างอิงการออกแบบนี้จากคู่มือ NAT observe คุณจะได้เห็นผลข้างเคียงหนึ่งของคำว่า "asynchronous" ใน lab 05-1

✓ Checkpoint: สำหรับเหตุการณ์ "เอเจนต์เรียก api.open-meteo.com" บอกได้ว่าระนาบไหนพิสูจน์ว่ามันเกิดขึ้น (policy) และระนาบไหนบอกเหตุผล (agent)

## 2 · L4.1 — Agent traces: exporter แบบ file และ Phoenix เมื่อมันมีอยู่จริง

เริ่มจากดูว่า NAT 1.9 ติดตั้งอะไรไว้บ้าง ชื่อ exporter คือค่า `_type` ที่คุณเขียนใน YAML:

```bash
# on: laptop
week26/.venv-nat/bin/nat info components -t tracing
```

**Expected output** (บันทึกจาก Mac เครื่องนี้, lab 05-1 ขั้นที่ 1)

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

telemetry block ใน research tutorial เพิ่ม exporter `phoenix` กับ exporter `file_backup` ที่ปล่อย key ไว้แค่ `# path etc.` แต่ NAT 1.9.0 เข้มงวดเรื่องนี้: exporter แบบ `file` สำหรับ tracing มีฟิลด์ **บังคับ** สองตัวคือ `output_path` และ `project` ดังนั้น block ของ tutorial จึงไม่ผ่าน validation และได้ข้อความ error ที่ไม่ค่อยชัดเจน:

**Expected output** (บันทึกจาก Mac เครื่องนี้, lab 05-1 ขั้นที่ 2)

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

block ของคอร์สอยู่ที่ `week26/05_tracing/configs/workflow.traced.yml` คือ config แล็ปท็อปจาก Module 04 บวกกับส่วน `general` นี้:

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

`configs/workflow.phoenix.yml` เพิ่ม exporter ของ Phoenix จาก research tutorial ไว้ข้างกัน (`endpoint: http://localhost:6006/v1/traces`, `project: alto-ops-claw`) exporter สองตัวใต้ `tracing:` จะทำงานพร้อมกัน lab 05-1 ใช้ไฟล์นั้น **เฉพาะเมื่อมีอะไรตอบอยู่ที่ localhost:6006** บน Mac เครื่องนี้ไม่มีอะไรตอบ Phoenix จึงถูกข้าม และไม่มี Phoenix trace ปรากฏที่ไหนเลยในโมดูลนี้ ถ้าจะรัน Phoenix เอง (แล็ปท็อปที่มี Docker หรือบน Spark):

```bash
# on: laptop
docker run -d --name phoenix-nat -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest
week26/.venv-nat/bin/nat run --config_file week26/05_tracing/configs/workflow.phoenix.yml --input "Plant status last 6 hours?"
# open http://localhost:6006 → project alto-ops-claw
```

ต่อไป trace การรันจริงหนึ่งครั้ง exporter แบบ file คือเส้นทางที่ใช้ได้เสมอ:

```bash
# on: laptop
week26/.venv-nat/bin/nat run --config_file week26/05_tracing/configs/workflow.traced.yml --input "Plant status last 6 hours?"
```

ไฟล์นี้เป็นแบบ **raw**: แต่ละบรรทัดคือ event `IntermediateStep` หนึ่งตัว ไม่ใช่ OpenTelemetry span ที่เสร็จสมบูรณ์ บรรทัดส่วนใหญ่เป็น `LLM_NEW_TOKEN` (หนึ่งบรรทัดต่อหนึ่ง token ที่ stream ออกมา) span หนึ่งตัวคือ `*_START` กับ `*_END` ที่ใช้ UUID เดียวกัน `tracekit.py` ของคอร์สจับคู่ให้:

**Expected output** (บันทึกจาก Mac เครื่องนี้, lab 05-1 ขั้นที่ 4–5 บางส่วน · โมเดล LAPTOP STAND-IN)

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

อ่านมันเหมือนอ่านเรื่องเล่า: การเรียก LLM ครั้งแรก (prompt 359 token) ตัดสินใจเรียก `chiller_kpi` tool ใช้เวลา 9 ms แล้วการเรียก LLM ครั้งที่สองก็เขียนคำตอบ เวลาเกือบทั้งหมดของการรันอยู่ที่โมเดล

span ที่เปิดค้างห้าตัวเป็นพฤติกรรมจริงของ NAT 1.9 ไม่ใช่บั๊กของตัว parse: `nat run` จบการทำงานทันทีที่ได้คำตอบ และใน NAT 1.9 เมธอด `stop()` ของ exporter ไม่รอ task เขียนไฟล์ที่ทำงานอยู่เบื้องหลัง event ท้าย ๆ จึงอาจหายไปได้ และผลแต่ละครั้งก็ไม่เท่ากัน: ในการรันอีกครั้งบน Mac เครื่องนี้ แม้แต่ `LLM_END` ตัวที่สองก็หายไป ไฟล์จึงแสดง LLM span ที่ปิดแล้วเพียง 1 ตัว ทั้งที่เรียกจริง 2 ครั้ง **อย่านับจำนวนการเรียก LLM จากไฟล์ trace ของ `nat run` ที่รันแป๊บเดียว** ให้นับ `LLM_START` แทน หรือใช้เซิร์ฟเวอร์ที่รันค้างไว้ (Section 6)

| research tutorial (เอกสาร NAT 1.8) | NAT 1.9.0 บนแล็ปท็อปนี้ | ควรทำอย่างไร |
|---|---|---|
| `file_backup: {_type: file}` + `# path etc.` | `output_path` และ `project` เป็นฟิลด์บังคับ; `mode` มีค่าเริ่มต้นเป็น `append` | ใส่ทั้งสองตัว และใส่ `mode: overwrite` ถ้าต้องการไฟล์ละหนึ่งการรัน |
| exporter แบบ file = "สำเนาสำรองของ trace" | เป็น RAW exporter: event แบบ IntermediateStep ส่วนใหญ่เป็น `LLM_NEW_TOKEN` | จับคู่ START/END ด้วย UUID เพื่อให้ได้ span |
| (ไม่ได้กล่าวถึง) | `nat run` ที่รันสั้น ๆ อาจทำส่วนท้ายของ trace หาย | นับ `LLM_START` หรือ trace จาก `nat serve` |
| รายชื่อ exporter: Phoenix, OTel collector, Langfuse, Weave, file | ที่นี่ลงทะเบียนไว้ 9 ตัว: phoenix, file, langfuse, langsmith, otelcollector, patronus, galileo, mlflow, arize_ax (การติดตั้งนี้ไม่มี weave) | ตรวจด้วย `nat info components -t tracing` บนเครื่องของคุณ |

✓ Checkpoint: คุณรัน lab 05-1 แล้ว และบอกได้ว่าคำถามหนึ่งข้อของ Alto Ops เรียก LLM กี่ครั้ง (2 ครั้ง) และทำไมไฟล์ trace ถึงมี span ที่เปิดค้าง

## 3 · L4.2 — แบบไม่ผูกกับ vendor: OTel collector และกฎใน sandbox ที่มันต้องการ

OpenTelemetry collector คือ OTLP endpoint ธรรมดา NAT, OpenClaw diagnostics หรืออะไรก็ตามส่งข้อมูลไปหามันได้ แล้วมันก็ส่งต่อไปยังไฟล์ Phoenix หรือ backend แบบ SaaS config ของ collector ใน research tutorial เขียนทุก trace ลงไฟล์:

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

ฝั่ง NAT ไฟล์ `configs/workflow.otel.yml` เพิ่ม exporter block จาก tutorial:

```yaml
    tracing:
      otelcollector:
        _type: otelcollector
        endpoint: http://0.0.0.0:4318/v1/traces
        project: alto-ops-claw
```

ในซอร์สของ NAT 1.9 (`nat/plugins/opentelemetry/register.py`) exporter ตัวนี้ตั้ง resource attribute ของ OTel ชื่อ `service.name` ให้เท่ากับ `project` นี่คือที่มาของพฤติกรรมแปลก ๆ ที่ research tutorial อ้างจาก Classmethod: Phoenix เก็บ trace ที่มาจาก otelcollector ไว้ใต้โปรเจกต์ `default` ดังนั้นถ้าจะส่งไป Phoenix ใช้ exporter `phoenix` ตัวจริงจะตรงไปตรงมากว่า

lab 05-2 validate config นี้จริง แล้วตรวจว่า Docker daemon รันอยู่หรือไม่ บน Mac เครื่องนี้ daemon ไม่ได้รัน แล็บจึงหยุดส่วนนั้นและบอกตรง ๆ โดยไม่พิมพ์ไฟล์ของ collector ที่ไม่เคยได้รับจริง:

**Expected output** (บันทึกจาก Mac เครื่องนี้, lab 05-2 ขั้นที่ 2–3)

```
$ nat validate --config_file week26/05_tracing/.runs/otel/workflow.otel.yml   [this laptop]
✓ nat validate → exit 0
$ docker info --format '{{.ServerVersion}}'   [this laptop]
⚠ the Docker daemon is not running here (client only, or not installed) → the collector is NOT started, and no collector output is shown. Start Docker Desktop and run this lab again to see it for real.
```

ถ้า Docker รันอยู่ แล็บจะเปิด collector เบื้องหลังบนพอร์ตที่ว่าง ส่งการรัน Alto Ops หนึ่งครั้ง นับ span ใน `otellogs/llm_spans.json` แล้วหยุด container

**กฎใน sandbox** NAT ที่อยู่ใน sandbox ของ OpenShell (`alto-ops` จาก Module 04) จะส่งข้อมูลไปหา collector ได้ก็ต่อเมื่อ policy มี endpoint entry ของโฮสต์ collector ที่ผูกกับ binary ของ Python เท่านั้น OTLP/HTTP คือการ **POST** ไปที่ `/v1/traces` ให้เริ่มด้วยโหมด `audit` ก่อน แล้วค่อยเปลี่ยนเป็น `enforce`:

```bash
# on: spark
openshell policy update alto-ops --add-endpoint otel.alto.local:4318:read-write:rest:audit --binary /usr/bin/python3.12 --dry-run
openshell policy update alto-ops --add-endpoint otel.alto.local:4318:read-write:rest:audit --binary /usr/bin/python3.12 --wait
```

**Expected output** (บันทึกจาก Mac เครื่องนี้, lab 05-2 ขั้นที่ 4 บางส่วน · policykit เป็นโมเดลสอนของคอร์ส ไม่ใช่ OpenShell)

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

บทเรียนสองข้อ ข้อแรก preset แบบ `read-only` อนุญาตแค่ GET/HEAD/OPTIONS ดังนั้นเมื่อเปลี่ยนเป็น `enforce` การส่ง trace จะโดน 403 (แบบฝึกหัดข้อ 3 ของ Part 4) ข้อสอง endpoint ของ tutorial บนแล็ปท็อป `http://0.0.0.0:4318` ใช้จากใน sandbox ไม่ได้ เพราะ 0.0.0.0 ถูกบล็อกเสมอในฐานะ SSRF ใน sandbox คุณต้องระบุชื่อโฮสต์ของ collector

สำหรับ claw แบบ OpenClaw research tutorial ระบุว่าการเปิด OpenClaw OTEL diagnostics กับ endpoint ภายในเครื่องจะเพิ่ม preset `openclaw-diagnostics-otel-local` ใน tier Balanced/Open/Personal ดังนั้น collector ตัวเดียวบนโฮสต์ Spark ก็รับได้ทั้ง trace ของ harness และ trace ของ NAT

✓ Checkpoint: คุณอธิบายได้ว่าทำไม collector ถึงบันทึก 403 สำหรับ NAT ใน sandbox (read-only + enforce บล็อก POST) และบอกวิธีแก้ได้สองวิธี (`access: read-write` หรือกฎ `allow: {method: POST, path: /v1/traces}` แบบระบุชัด)

## 4 · L4.3 — ส่ง trace ของ Hermes ไป Langfuse โดยไม่ให้ key รั่ว

harness Hermes มี plugin สำหรับ Langfuse research tutorial อ้างจาก NemoClaw Hermes quickstart ว่า NemoClaw กัน key ไว้นอก sandbox อย่างไร: คุณลงทะเบียน key เป็น credential ของ OpenShell ชนิด `langfuse-hermes-v1` sandbox จะได้รับแค่ placeholder แล้ว OpenShell จะแทนค่าจริงให้ตอนข้อมูลออกไป (egress)

ให้พิมพ์คำสั่งเหล่านี้เองในเทอร์มินัล ⌨ บน Spark ด้วย key จริงของคุณ บรรทัด `export` มีความลับอยู่ lab 05-3 จึงไม่รันมันเด็ดขาด แล็บจะรัน `credentials add` ผ่าน `change()` เฉพาะเมื่อตัวแปรทั้งสองถูกตั้งไว้แล้วใน shell ของมันเท่านั้น (ตรวจโดยไม่พิมพ์ค่าออกมา) ส่วน `rebuild` และ `gateway restart` ก็รันผ่าน `change()`

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

อย่าใส่ key ดิบของ Langfuse หรือสำเนาของ placeholder ลงใน `~/.hermes/.env` สิ่งเดียวที่ควรอยู่ในนั้นคือ `HERMES_LANGFUSE_BASE_URL` ซึ่งไม่ใช่ความลับ รูปแบบนี้ใช้ได้กับ backend observability แบบ SaaS ทุกตัว: credential handle อยู่ใน OpenShell, placeholder อยู่ใน sandbox, การแทนค่าเกิดที่ proxy

ตัวตรวจของ lab 05-3 สแกนข้อความ config ของ Hermes หา key ดิบแบบ `pk-lf-` / `sk-lf-` และหาตัวแปร key ที่ไม่ควรอยู่ตรงนั้น แล้วพิมพ์เฉพาะผลที่ปิดบังค่าแล้ว บน Mac เครื่องนี้มันสแกนตัวอย่างสี่ไฟล์ที่ใช้ key ปลอม (เครื่องนี้ไม่มี `~/.hermes`):

**Expected output** (บันทึกจาก Mac เครื่องนี้, lab 05-3 ขั้นที่ 3)

```
✕ HIGH   sample-bad/.hermes/.env:2 — raw Langfuse key pk-lf-••• — exfiltrable by prompt injection; move it to an OpenShell credential (langfuse-hermes-v1)
✕ HIGH   sample-bad/.hermes/.env:3 — raw Langfuse key sk-lf-••• — exfiltrable by prompt injection; move it to an OpenShell credential (langfuse-hermes-v1)
✕ HIGH   sample-bad/.hermes/config.yaml:4 — raw Langfuse key sk-lf-••• — exfiltrable by prompt injection; move it to an OpenShell credential (langfuse-hermes-v1)
⚠ MEDIUM sample-copied-placeholder/.hermes/.env:2 — LANGFUSE_PUBLIC_KEY is set here (value not shown) — keys and copied placeholders do not belong in the agent's config
✓ OK     sample-good/.hermes/.env — no Langfuse keys; only non-secret settings
◆ 3 HIGH · 1 MEDIUM · 1 OK
```

ทำไม key ใน `/sandbox/.hermes/config.yaml` ถึงแย่ยิ่งกว่าไม่มีประโยชน์ (แบบฝึกหัดข้อ 4 ของ Part 4)? เพราะเอเจนต์อ่านและแก้ไข config ของตัวเองได้ เอกสารถือว่า `/sandbox/.hermes` เป็นสถานะที่เปลี่ยนได้และเอเจนต์ควบคุม ไม่ใช่ขอบเขตการแยก (isolation) เอเจนต์ที่โดน prompt injection จึงส่ง key ออกไปได้ แถม key ดิบก็ใช้ไม่ได้ด้วยซ้ำ เพราะ OpenShell คาดหวัง placeholder

บน Spark ให้นับจำนวนที่เจอภายใน sandbox อย่าพิมพ์ค่าออกมา:

```bash
# on: spark
# inside the sandbox (nemohermes my-hermes connect)
grep -cE 'pk-lf-|sk-lf-' ~/.hermes/.env ~/.hermes/config.yaml
```

NAT 1.9 ก็มี exporter `langfuse` ของตัวเองด้วย (ฟิลด์ `endpoint`, `public_key`, `secret_key` ถ้าเว้น key ว่างไว้จะไปอ่านจากตัวแปรสภาพแวดล้อม `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`) NAT สร้าง header แบบ Basic auth เอง ส่วนที่ว่าการแทนค่า placeholder ของ OpenShell จะไปถึง header ที่ NAT เข้ารหัส base64 เองหรือไม่ แหล่งอ้างอิงของคอร์สนี้ไม่ได้ครอบคลุม ให้ทดสอบบนเครื่องของคุณก่อนจะพึ่งพามัน

✓ Checkpoint: คุณอธิบายได้ข้อละหนึ่งประโยค ว่า key ของ Langfuse อยู่ที่ไหน (credential ของ OpenShell) sandbox เห็นอะไร (placeholder) และอะไรควรอยู่ใน `~/.hermes/.env` (เฉพาะ `HERMES_LANGFUSE_BASE_URL`)

## 5 · L4.4 — ระนาบ policy: อ่าน OpenShell เหมือนอ่าน trace

supervisor และ gateway ของ OpenShell บันทึกทุกการตัดสินใจ นี่คือคำสั่งจาก research tutorial โดยมีสิ่งหนึ่งที่คุณต้องเปลี่ยน: **จำกัดเวลาให้ทุกคำสั่งที่ follow log** บน CLI ของแล็ปท็อป `openshell logs --help` บอกว่า `--tail` หมายถึง "Stream live logs" คำสั่งจึงไม่จบเอง อย่ารันการ follow แบบ foreground บน Spark ให้ครอบด้วย `timeout 10` หรือใช้รูปแบบที่มีขอบเขต:

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

**Expected output** (REFERENCE — ยกมาจาก OpenShell playbook, Step 11)

```
- **Live log stream** — outbound connections, policy decisions (`allow`, `deny`, `inspect_for_inference`), and inference interceptions
```

lab 05-4 parse คำสั่งทั้งสี่ด้วย CLI ของ OpenShell 0.0.111 ตัวจริงกับ gateway ที่ไม่มีอยู่ ทั้งสี่คำสั่ง parse ผ่าน ส่วนของ Spark เป็นแบบอ่านอย่างเดียว ในโหมด DRY จะพิมพ์รูปแบบ EXAMPLE

จำข้อเท็จจริงสองข้อนี้ไว้ตอนอ่าน log:

- การละเมิดในชั้น L7 ภายใต้ `enforcement: audit` จะถูก **บันทึก และ traffic ยังถูกส่งต่อไป** ในโหมด audit "การละเมิด" คือ finding ที่ต้องแก้ ไม่ใช่การบล็อก
- path ของ Landlock ที่ถูกข้ามภายใต้ `landlock: best_effort` จะแสดงเป็น OCSF `DetectionFinding` ระดับความรุนแรง High

ตัว parse ของแล็บแสดงทั้งสองกรณีบน stream ของการตัดสินใจ stream นี้เป็น **รูปแบบ EXAMPLE** ที่เขียนขึ้นสำหรับคอร์ส การตัดสินใจในนั้นมาจาก `policykit` ไม่ได้มาจาก OpenShell:

**Expected output** (บันทึกจาก Mac เครื่องนี้, lab 05-4 ขั้นที่ 3 บางส่วน — บรรทัดของ stream เองเป็นรูปแบบ EXAMPLE)

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

✓ Checkpoint: คุณบอกได้ว่าทำไม `openshell logs --tail` ต้องมี `timeout 10` เมื่ออยู่ในแล็บ และทำไมการละเมิดในโหมด audit จึงเป็น finding ไม่ใช่การบล็อก

## 6 · L4.5 — เชื่อมโยง: หนึ่ง request สามระนาบ

ตอนนี้มาเชื่อมทั้งสามระนาบสำหรับ request เดียว สูตรจาก research tutorial:

1. ส่ง request หนึ่งครั้งผ่าน `nat serve` (`/v1/workflow/full`) แล้วเก็บ intermediate step ไว้
2. หาการรันเดียวกันใน agent trace จดจำนวน LLM span และ token ทั้งหมด
3. ใน `openshell term` กรอง (`s`) ให้เหลือ sandbox นั้น แล้วนับ `inspect_for_inference` มันควรเท่ากับจำนวน LLM span เพราะการเรียกโมเดลทุกครั้งผ่าน `inference.local`
4. ทำให้เกิดการเรียกที่ถูกปฏิเสธ (ขอ "the latest weather from open-meteo") tool span จะล้มเหลวในระนาบ agent ในเวลาเดียวกับที่ OpenShell บันทึก `deny`

```bash
# on: laptop
week26/.venv-nat/bin/nat serve --config_file week26/05_tracing/configs/workflow.traced.yml --port 8001
```

ปล่อยให้มันรันค้างไว้ แล้วส่ง request จากเทอร์มินัลที่สอง:

```bash
# on: laptop
curl -s -X POST 'http://localhost:8001/v1/workflow/full?filter_steps=LLM_END,TOOL_END' \
  -H 'Content-Type: application/json' -d '{"input_message": "Plant status last 6 hours?"}'
```

สิ่งที่เซิร์ฟเวอร์ NAT 1.9.0 ตัวจริงบน Mac เครื่องนี้ทำกับ request นั้น (ตรวจกับ `/openapi.json` ของมันและด้วย `curl`):

| พฤติกรรม | ที่เห็นบน Mac เครื่องนี้ |
|---|---|
| route | `POST /v1/workflow/full` และ route เดิม `POST /generate/full` ทั้งสองรับ query parameter `filter_steps` |
| body | `{"input_message": "…"}` (หรือ `messages`) ถ้าเป็นแบบอื่นจะได้ `422` "Either messages or input_message must be provided" |
| `filter_steps=LLM_END,TOOL_END` | บรรทัด `intermediate_data:` 3 บรรทัด (`LLM_END` 2 ตัว, `TOOL_END` 1 ตัว) และบรรทัด `data:` 88 บรรทัด (คำตอบที่ stream ออกมา) |
| `filter_steps=none` | ไม่มีบรรทัด `intermediate_data:` เลย ส่วนบรรทัดคำตอบเหมือนเดิม |
| `… \| grep -c '"type":"LLM_END"'` | `2` คำสั่งตรวจบน Spark ด้านล่างจึงใช้ได้ตามที่เขียนไว้ |

`nat serve` ต้องมีแพ็กเกจ `greenlet` ใน `week26/.venv-nat` เพราะ FastAPI front end ของ NAT 1.9.0 import async job store (SQLAlchemy asyncio) ตั้งแต่ตอนเริ่ม ถ้าไม่มี เซิร์ฟเวอร์จะปิดตัวก่อนพร้อมใช้งาน ติดตั้งครั้งเดียวด้วย `uv pip install --python week26/.venv-nat/bin/python greenlet` ถ้า `nat serve` ยังเปิดไม่ขึ้น lab 05-5 จะบอกสาเหตุ แล้วเปลี่ยนไปเรียก function ของ route ตัวเดียวกัน (`generate_streaming_response_full`) แบบ in-process ผ่าน `week26/05_tracing/fullstream.py` และติดป้ายผลลัพธ์ไว้ตามนั้น

**Expected output** (บันทึกจาก Mac เครื่องนี้, lab 05-5 ขั้นที่ 1–3 บางส่วน · โมเดล LAPTOP STAND-IN)

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

สังเกตสามอย่าง: stream กับไฟล์ตรงกัน คือ 2 LLM span, 1 tool span, 1268 token รอบนี้ไฟล์ **ไม่มี** span เปิดค้างเลย เพราะ process ของเซิร์ฟเวอร์ยังอยู่ต่อหลังการรัน และการเรียก LLM รอบนี้ช้ากว่าใน lab 05-1 (7.3 วินาทีและ 15.5 วินาที เทียบกับ 3.6 และ 8.1 วินาที) เพราะแล็บอื่นใช้ Ollama บนแล็ปท็อปร่วมกันอยู่ เวลาบนแล็ปท็อปเป็นแค่ตัวแทน อย่านำไปเทียบกับตัวเลขของ Spark (Module 06 วัดบน Spark)

ดังนั้นคำทำนายสำหรับการรัน request นี้ใน sandbox คือ **`inspect_for_inference` 2 event** หนึ่งตัวต่อหนึ่ง LLM span ตรวจสอบบน Spark กับ `nat serve` ใน sandbox จาก Module 04 (เข้าถึงได้ที่พอร์ต 8001) ด้วยคำสั่งแบบอ่านอย่างเดียวที่มีขอบเขต:

```bash
# on: spark
curl -s -X POST 'http://localhost:8001/v1/workflow/full?filter_steps=LLM_END,TOOL_END' -H 'Content-Type: application/json' -d '{"input_message": "Plant status last 6 hours?"}' | grep -c '"type":"LLM_END"'
openshell logs alto-ops -n 200 --since 2m --source sandbox | grep -c inspect_for_inference
```

Alto Ops Claw ไม่มี tool สำหรับเว็บ บนแล็ปท็อปมันจึงไม่แม้แต่จะพยายามเรียก open-meteo ส่วน harness ที่พยายามเรียกจะแสดงทั้งสองฝั่ง: tool span ที่ล้มเหลว และบรรทัด `deny`

จดตารางนี้ไว้ มันคือจุดเริ่มต้นของเรื่องราวด้าน audit ที่คุณจะเล่าให้เจ้าของโรงแรมหรือหน่วยงานกำกับดูแลของไทยฟังใน Module 07:

| ระนาบ | มาจาก | คีย์สำหรับเชื่อม | สำหรับ request นี้ (LAPTOP STAND-IN) |
|---|---|---|---|
| agent traces | exporter ของ NAT (file · Phoenix · OTel · Langfuse) | `workflow_run_id` / trace id, ข้อความ input, timestamp | 2 LLM span, 1 TOOL span, 1268 token |
| policy logs | `openshell logs` / `openshell term`, OCSF findings | ชื่อ sandbox + ช่วงเวลา | `inspect_for_inference` 2 ครั้ง (ทำนาย); `deny` หนึ่งครั้ง = tool span ที่ล้มเหลว |
| harness logs | `nemoclaw <sandbox> logs`, `/tmp/gateway.log` (OpenClaw) | timestamp, channel / session id | event ของ channel, การ pairing, การ crash |

✓ Checkpoint: คุณทำนายจำนวน `inspect_for_inference` ของ request หนึ่งได้จาก trace ของมัน (หนึ่งตัวต่อหนึ่ง LLM span) และรู้ว่าคำสั่งไหนใช้ตรวจบน Spark

## Labs — รันแล็บได้ที่นี่

**labs/lab05_1_file_and_phoenix.py** — แสดง tracing exporter ของ NAT 1.9 ตรวจ key จริงของ exporter แบบ file แล้ว trace การรัน Alto Ops หนึ่งครั้งลงไฟล์ (และส่งไป Phoenix เฉพาะเมื่อมันตอบ) พร้อมแยก span

**labs/lab05_2_otel_collector.py** — เขียน config ของ OTel collector และ exporter block ตรวจความถูกต้อง รัน collector เฉพาะเมื่อ Docker ทำงานอยู่ และตรวจกฎใน sandbox สำหรับ POST /v1/traces

**labs/lab05_3_langfuse_handles.py** — ไล่ขั้นตอน credential handle ของ Langfuse สำหรับ Hermes ด้วย placeholder เท่านั้น และสแกนข้อความ config ของ Hermes หา key ที่รั่ว

**labs/lab05_4_policy_plane.py** — parse คำสั่ง log ของ OpenShell แบบออฟไลน์ อ่านระนาบ policy บน Spark (แบบมีขอบเขต) และนับการตัดสินใจใน stream แบบ EXAMPLE

**labs/lab05_5_correlate.py** — ส่ง request หนึ่งครั้งไปที่ /v1/workflow/full เทียบกับ trace ในไฟล์ แล้วทำนายจำนวน inspect_for_inference ของ sandbox

## Try it yourself — ลองทำเอง

`exercises/ex05_telemetry.py` มี TODO สามข้อ:

1. เขียน block `general.telemetry.tracing` ที่มีทั้ง exporter ของ Phoenix **และ** exporter แบบ file โดยใช้ key ที่ NAT 1.9 ต้องการจริง ตัวตรวจจะรัน `nat validate` ตัวจริงกับ block ของคุณ
2. แก้ endpoint entry ของ collector ที่ได้ 403 โดยต้องคง `enforcement: enforce` ไว้
3. บอกชื่อ decorator ของ NAT ที่ใช้ trace function Python ธรรมดา ด้วย import path แบบเต็ม และบอกชนิดของ event ทั้งสามที่มันส่งออกมา ตัวตรวจจะยืนยันกับซอร์สของ NAT ที่ติดตั้งอยู่

```bash
# on: laptop
.venv/bin/python week26/05_tracing/exercises/ex05_telemetry.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ เมื่อทำ TODO ครบทุกจุด)

```
✓ telemetry: 2 exporters — phoenix + file (output_path, project)
$ nat validate --config_file <your block + workflow.laptop.yml>   [this laptop]
✓ the real `nat validate` (NAT 1.9.0) accepts your block
✓ collector: POST /v1/traces allowed under enforce (rest, otel.alto.local, not `full`)
✓ decorator: @track_function from nat.plugins.profiler.decorators.function_tracking · SPAN_START / SPAN_CHUNK (generators) / SPAN_END — found in the installed NAT 1.9.0
```

<details><summary>Hint — exporter แบบ file</summary>

lab 05-1 ขั้นที่ 2 พิมพ์รายชื่อฟิลด์ไว้แล้ว มีสองตัวที่ไม่มีค่าเริ่มต้น ใส่ `mode: overwrite` ด้วยถ้าต้องการไฟล์ละหนึ่งการรัน

</details>

<details><summary>Hint — 403</summary>

OTLP/HTTP คือการ POST ส่วน preset แบบ `read-only` อนุญาต GET, HEAD และ OPTIONS ให้เปลี่ยน access mode หรือแทนที่ด้วยกฎ allow ตัวเดียวสำหรับ `POST /v1/traces` การเปลี่ยนไปเป็น `audit` ทำให้ 403 หายไปก็จริง แต่นั่นเป็นเพราะไม่มีอะไรถูกบังคับใช้แล้ว

</details>

<details><summary>Hint — decorator</summary>

research tutorial ระบุ `@track_function` ใน `nat.plugins.profiler.decorators.function_tracking` ซึ่งใน NAT 1.9.0 path นี้ยังถูกต้อง ลองเปิด `week26/.venv-nat/lib/python3.12/site-packages/nat/plugins/profiler/decorators/function_tracking.py` แล้วหาค่า `IntermediateStepType.` ที่มันส่งออกมา ไฟล์เดียวกันยังมี `track_unregistered_function` ซึ่งเพิ่มการจัดการ scope ให้ด้วย

</details>

✓ Checkpoint: ตัวตรวจขึ้น ✓ ครบทั้งสามข้อ

## Troubleshooting — แก้ปัญหา

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `nat validate`: `Invalid configuration: general: Field required` | exporter แบบ `file` สำหรับ tracing ขาด `output_path` และ/หรือ `project` (NAT 1.9) | ใส่ทั้งสอง key (Section 2) |
| ไฟล์ trace มีบรรทัด START ที่ไม่มี END หรือมี `LLM_END` น้อยกว่าจำนวนการเรียก LLM | `nat run` ที่รันสั้น ๆ จบก่อนที่ exporter ของ NAT จะเขียนเสร็จ | นับ `LLM_START` หรือ trace จาก `nat serve` ที่รันค้างไว้ (lab 05-5) |
| ไฟล์ trace ไฟล์เดียวโตขึ้นเรื่อย ๆ | `mode` ของ exporter แบบ file มีค่าเริ่มต้นเป็น `append` | ตั้ง `mode: overwrite` หรือใช้ `output_path` แยกต่อการรัน |
| `nat serve`: `The SQLAlchemy asyncio module requires that the Python 'greenlet' library is installed` | ไม่มี `greenlet` ใน `week26/.venv-nat` (FastAPI front end ของ NAT 1.9.0 import SQLAlchemy asyncio ตั้งแต่ตอนเริ่ม) | ติดตั้ง: `uv pip install --python week26/.venv-nat/bin/python greenlet` แล้วรัน lab 05-5 อีกครั้ง (ระหว่างนี้แล็บจะใช้ตัวแทนแบบ in-process ที่ติดป้ายไว้) |
| `/v1/workflow/full` ตอบ `422` "Either messages or input_message must be provided" | JSON body ใช้ key อื่น | ส่ง `{"input_message": "…"}` |
| lab 05-1 บอกว่า Phoenix SKIPPED | ไม่มีอะไรตอบที่ localhost:6006 | เปิด Phoenix (Section 2) แล้วรันแล็บอีกครั้ง |
| lab 05-2 บอกว่า Docker daemon ไม่ได้รัน | Docker Desktop ปิดอยู่ (มีแค่ client รัน container ไม่ได้) | เปิด Docker Desktop หรืออ่านส่วนที่เหลือของแล็บแบบออฟไลน์ |
| trace จาก otelcollector ไปอยู่ในโปรเจกต์ `default` ของ Phoenix | NAT ส่ง `project` เป็น `service.name` | ใช้ exporter `phoenix` ตัวจริงเมื่อส่งไป Phoenix |
| collector บันทึก 403 สำหรับ NAT ใน sandbox | preset แบบ `read-only` ภายใต้ `enforce` บล็อก POST | `access: read-write` หรือ `allow: {method: POST, path: /v1/traces}` (แบบฝึกหัด 05, TODO 2) |
| `openshell logs … --tail` ไม่จบเสียที | `--tail` คือการ stream log แบบสด | `timeout 10 …` หรือ `-n 50 --since 10m` |

## Next — บทถัดไป

[Lab 06 — Performance benchmarking on DGX Spark](../06_benchmarking/TUTORIAL.md): วัด engine, workflow, คุณภาพ และค่าใช้จ่ายของ sandbox แยกจากกัน โดยใช้ trace แบบเดียวกับที่คุณเพิ่งหัดอ่าน
