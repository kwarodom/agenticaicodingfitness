# ▶ Spark Lab 08 — LiteLLM: gateway เดียวสำหรับทุก engine และ Spark ทั้งสองเครื่อง

> ส่วนหนึ่งของ Week 25 · DGX Spark: fine-tune, serve และสร้างเอเจนต์ที่รันใน sandbox คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง ทุกแล็บรันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (อ้างอิงคำต่อคำจาก playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- เข้าใจว่าทำไม engine สี่ตัวบน Spark แต่ละเครื่องจากสองเครื่องจึงต้องมีประตูหน้าบานเดียว: OpenAI endpoint เดียว, model alias, fallback, retry, load balancing และ master key
- สร้างไฟล์ `config.yaml` ของ LiteLLM หนึ่งไฟล์จากการตั้งค่า Spark ของคุณ: Ollama :11434, vLLM :8000, SGLang :30000 และ TensorRT-LLM :8355 บน Spark A และ Spark B พร้อม fallback ไปที่แล็ปท็อป
- เปิด LiteLLM proxy **ของจริงบนแล็ปท็อปเครื่องนี้** เรียกผ่าน alias ด้วย OpenAI SDK และดู master key ปฏิเสธคำขอที่ไม่มีคีย์
- ชี้ alias ไปที่ Spark ที่ล่มอยู่ แล้วดู fallback, retry และ cooldown ใน response header
- ตัดสินใจว่า gateway ควรอยู่ที่ไหน (แล็ปท็อปหรือ Spark A) แล้วย้ายไปไว้ที่นั่น

**Time** ~45 นาที · **Difficulty** ระดับกลาง · **Hardware** ไม่ต้องใช้ (ทุกแล็บรันบนแล็ปท็อปด้วย Ollama ถ้ามี Spark alias ชุดเดียวกันจะส่งต่อไปที่ Spark)

**Playbook ทางการที่ครอบคลุม:** *เนื้อหาของคอร์สเอง (course-original)* NVIDIA ไม่มี playbook สำหรับ LiteLLM โมดูลนี้วาง gateway เดียวไว้หน้าเซิร์ฟเวอร์จาก [Open WebUI with Ollama](https://build.nvidia.com/spark/open-webui) · [vLLM](https://build.nvidia.com/spark/vllm) · [SGLang](https://build.nvidia.com/spark/sglang) · [TRT-LLM](https://build.nvidia.com/spark/trt-llm) ทุก flag และทุก config key ของ LiteLLM ในบทนี้ตรวจเทียบกับแพ็กเกจที่ติดตั้งจริงแล้ว (LiteLLM 1.89.0: `litellm --help` และ signature ของ `litellm.Router`)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| LiteLLM proxy 1.89 ใน `week25/.venv-litellm` | `week25/.venv-litellm/bin/litellm --version` | `.venv` ของ repo มีไลบรารี LiteLLM แต่ไม่มี extras `[proxy]` ที่เซิร์ฟเวอร์ต้องใช้ |
| Ollama บนแล็ปท็อป พร้อมโมเดลเล็กสองตัว | `ollama list` แสดง `gemma3:4b` และ `nemotron-3-nano:latest` | เป็น fallback บนแล็ปท็อปที่ทุกแล็บเรียกใช้ |
| พอร์ต 4000 ว่าง | `lsof -iTCP:4000 -sTCP:LISTEN` ไม่พิมพ์อะไรออกมา | ถ้า 4000 ไม่ว่าง แล็บจะใช้พอร์ตว่างถัดไปและแจ้งให้ทราบ |
| Module 03–07 (ไม่บังคับ) | `curl http://spark-a:8000/v1/models` | ถ้ามี engine รันอยู่ alias ของ Spark จะตอบจริง |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
week25/.venv-litellm/bin/litellm --version
ollama list | grep -E "gemma3:4b|nemotron-3-nano"
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
LiteLLM: Current Version = 1.89.0

nemotron-3-nano:latest           b725f1117407    24 GB     8 weeks ago
gemma3:4b                        a2af6cc3eb7f    3.3 GB    8 weeks ago
```

ถ้ายังไม่มี `week25/.venv-litellm` ให้สร้างขึ้นมา (เป็น venv เฉพาะของ week25 ไม่แตะอะไรในระดับ global):

```bash
# on: laptop
uv venv week25/.venv-litellm --python 3.13
uv pip install --python week25/.venv-litellm/bin/python 'litellm[proxy]==1.89.0' openai pyyaml
```

> ⚠ `.venv/bin/litellm` (venv ของ repo) จะปิดตัวพร้อม `ImportError: Missing dependency No module named 'backoff'. Run pip install 'litellm[proxy]'` นี่คือเหตุผลที่โมดูลนี้ใช้ venv แยกสำหรับ proxy ส่วนตัวแล็บเองยังรันด้วย `.venv/bin/python`

✓ Checkpoint: `week25/.venv-litellm/bin/litellm --version` พิมพ์ 1.89.0 และมีโมเดลบนแล็ปท็อปครบทั้งสองตัว

## 1 · ทำไมต้องมี gateway?

พอถึง Module 07 คุณอาจมีเซิร์ฟเวอร์ที่เข้ากันได้กับ OpenAI มากถึงแปดตัว: engine สี่ตัวบน Spark แต่ละเครื่อง ทุก client (notebook, เอเจนต์, Open WebUI) จะต้องรู้ทุก URL ทุก model id และต้องมีโค้ด failover ของตัวเอง gateway ย้ายทั้งหมดนั้นมารวมไว้ที่เดียว:

| ไม่มี gateway | มี LiteLLM ที่ :4000 |
|---|---|
| `http://spark-a:8000/v1`, โมเดล `nvidia/Llama-3.1-8B-Instruct-FP8` | `http://gateway:4000/v1`, โมเดล **`chat`** |
| สลับ vLLM เป็น SGLang → แก้ทุก client | แก้บรรทัดเดียวใน `config.yaml` |
| Spark A รีบูต → ทุกแอปเจอ error | router ลองซ้ำกับ Spark อีกเครื่อง แล้ว fallback ไปที่แล็ปท็อป |
| Spark สองเครื่อง → client ต้องเลือกเอง | alias เดียว สอง deployment กระจายโหลดให้ |
| engine **ไม่มีการยืนยันตัวตน** | มี **master key** หนึ่งตัวอยู่ด้านหน้า (engine ยังต้องเก็บไว้เป็นส่วนตัว: ส่วนที่ 3) |
| ไม่มี log ของคำขอ | ทุกคำตอบมี header `x-litellm-*`: deployment ไหนตอบ, retry, fallback และเวลา |

client จะเห็นแค่ **alias** (`model_name` ใน config) เท่านั้น แต่ละ alias ชี้ไปที่ **deployment** หนึ่งตัวหรือมากกว่า (`litellm_params`: model id ที่มี prefix ของ provider บวก `api_base`) prefix ของ provider บอก LiteLLM ว่าจะคุยกับเซิร์ฟเวอร์อย่างไร:

| Engine (พอร์ต) | `model:` ของ LiteLLM | ทำไมใช้ prefix นั้น |
|---|---|---|
| vLLM (:8000) | `hosted_vllm/<served model id>` | provider ของ LiteLLM สำหรับ vLLM ที่โฮสต์เอง |
| Ollama (:11434), SGLang (:30000), TensorRT-LLM (:8355) | `openai/<served model id>` | ทั้งสามตัว serve OpenAI API ที่ `/v1` |

`api_key: none` เป็นค่า placeholder: engine ไม่ได้ตรวจค่านี้ แต่โค้ด OpenAI client ภายใน LiteLLM ต้องการให้มีค่าอะไรสักอย่าง

✓ Checkpoint: คุณอธิบายได้ว่า alias คืออะไร deployment คืออะไร และทำไม `chat` จึงมีสอง deployment ได้

## 2 · สร้าง config จากการตั้งค่า Spark ของคุณ: lab 01

Lab 01 อ่านการตั้งค่าชุดเดียวกับทุกแล็บของ Week 25 (`SPARK_HOST`, `SPARK_API_HOST`, Module 01) มันถามแต่ละ engine ที่ตอบสนองว่า serve โมเดลอะไรอยู่ (`GET /v1/models`) และถ้าไม่ได้คำตอบจะใช้ model id ตั้งต้นของคอร์สแทน จากนั้นเขียนไฟล์หนึ่งไฟล์คือ `week25/08_litellm_gateway/.runs/litellm.config.yaml` (อยู่ใน gitignore) แล้ว validate ด้วย PyYAML **และ** ด้วย `Router` ของ LiteLLM เอง

```bash
# on: laptop
.venv/bin/python week25/08_litellm_gateway/labs/lab01_build_config.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ที่ยังไม่ได้ตั้งค่า Spark host จึงเป็น placeholder `spark-a` / `spark-b`)

```
▣ STEP 1 · where each alias points
│ Spark A HTTP host: (not set → placeholder spark-a) · Spark B: (not set → placeholder spark-b)
│ alias            LiteLLM model (provider/id)                   api_base                   model id from
│ ───────────────  ────────────────────────────────────────────  ─────────────────────────  ───────────────────
│ spark-a/ollama   openai/gpt-oss:20b                            http://spark-a:11434/v1    course default
│ spark-a/vllm     hosted_vllm/nvidia/Llama-3.1-8B-Instruct-FP8  http://spark-a:8000/v1     course default
│ spark-a/sglang   openai/Qwen/Qwen3-8B                          http://spark-a:30000/v1    course default
│ spark-a/trtllm   openai/nvidia/Llama-3.1-8B-Instruct-FP4       http://spark-a:8355/v1     course default
│ spark-b/ollama   openai/gpt-oss:20b                            http://spark-b:11434/v1    course default
│ …
│ chat             hosted_vllm/nvidia/Llama-3.1-8B-Instruct-FP8  http://spark-a:8000/v1     load-balanced A + B
│ chat             hosted_vllm/nvidia/Llama-3.1-8B-Instruct-FP8  http://spark-b:8000/v1     load-balanced A + B
│ laptop-fast      openai/gemma3:4b                              http://localhost:11434/v1  fallback
│ laptop-nemotron  openai/nemotron-3-nano:latest                 http://localhost:11434/v1  fallback

▣ STEP 3 · validate — PyYAML, then LiteLLM's own Router
✓ PyYAML: the file parses back to exactly the generated config
✓ every fallback names aliases that exist
✓ master_key is read from the environment (os.environ/LITELLM_MASTER_KEY), not stored in the file
✓ litellm 1.89.0 Router accepted 12 deployments in 11 aliases · strategy simple-shuffle
```

ไฟล์ที่สร้างขึ้น ตัดให้เหลือหนึ่ง deployment ต่อประเภท **นี่คือรูปแบบที่ capstone (Module 20) นำกลับมาใช้:**

```yaml
model_list:
- model_name: spark-a/vllm                  # one alias per engine per Spark
  litellm_params:
    model: hosted_vllm/nvidia/Llama-3.1-8B-Instruct-FP8
    api_base: http://spark-a:8000/v1
    api_key: none
- model_name: chat                          # SAME alias twice → load-balanced across the Sparks
  litellm_params: {model: hosted_vllm/nvidia/Llama-3.1-8B-Instruct-FP8, api_base: http://spark-a:8000/v1, api_key: none}
- model_name: chat
  litellm_params: {model: hosted_vllm/nvidia/Llama-3.1-8B-Instruct-FP8, api_base: http://spark-b:8000/v1, api_key: none}
- model_name: laptop-fast                   # the fallback that is always there
  litellm_params: {model: openai/gemma3:4b, api_base: http://localhost:11434/v1, api_key: none}
- model_name: laptop-nemotron               # a thinking model, told not to think
  litellm_params:
    model: openai/nemotron-3-nano:latest
    api_base: http://localhost:11434/v1
    api_key: none
    extra_body: {reasoning_effort: none}
router_settings:
  routing_strategy: simple-shuffle          # also: least-busy, latency-based-routing, usage-based-routing-v2 …
  num_retries: 1                            # retry once on another deployment of the same alias
  timeout: 120                              # seconds per call
  allowed_fails: 1                          # failures before a deployment is benched …
  cooldown_time: 30                         # … for this many seconds
  fallbacks:
  - chat: [laptop-fast]                     # whole alias down → try this alias instead
  - spark-a/vllm: [laptop-fast]
litellm_settings:
  drop_params: true                         # drop OpenAI params an engine does not support
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY # read the key from the environment; never write it here
```

รายละเอียดสองข้อที่จะเสียเวลาถ้าเดาเอา:

- **`extra_body: {reasoning_effort: none}`** คือวิธีที่ alias ของ nemotron ปิดการคิด (thinking) ใน Ollama ถ้าใส่ `reasoning_effort: none` ลงใน `litellm_params` ตรง ๆ จะล้มเหลวในเวอร์ชันนี้ด้วย `UnsupportedParamsError: openai does not support parameters: ['reasoning_effort']` (บันทึกจาก Mac เครื่องนี้) `extra_body` ส่งค่าไปถึง engine โดยไม่แตะต้อง
- **key ใน `router_settings` ต้องเป็นอาร์กิวเมนต์ของ `Router()`** proxy ส่งต่อเฉพาะตัวที่ถูกต้อง และสำหรับตัวที่เหลือจะ log ว่า `Key '…' is not a valid argument for Router.__init__(). Ignoring this key.` (อ่านจาก `proxy_server.py` ที่ติดตั้งอยู่) ตอนเริ่มทำงาน การพิมพ์ผิดจึงเป็นแค่บรรทัดหนึ่งใน log Lab 01 ส่ง `router_settings` เข้า `litellm.Router(...)` โดยตรง ซึ่งเข้มงวดกว่า บน Mac เครื่องนี้ `num_retrys: 1` ล้มเหลวที่นั่นด้วย `TypeError: … unexpected keyword argument 'num_retrys'. Did you mean 'num_retries'?` และ `routing_strategy: fastest` ล้มเหลวด้วย `ValueError: Invalid routing_strategy` ให้รัน lab 01 (หรือตัวตรวจของแบบฝึกหัด) ทุกครั้งหลังแก้ไฟล์ด้วยมือ

✓ Checkpoint: มีไฟล์ `.runs/litellm.config.yaml` และการตรวจทั้งสี่ข้อของ lab 01 เป็น ✓

## 3 · เปิดใช้งานจริง: alias และ master key (lab 02)

Lab 02 เปิด proxy เป็น child process เหมือนกับที่คุณจะทำด้วยมือทุกประการ:

```bash
# on: laptop
export LITELLM_MASTER_KEY="sk-$(openssl rand -hex 16)"      # a fresh key; keep it out of files and shell history
week25/.venv-litellm/bin/litellm --config week25/08_litellm_gateway/.runs/litellm.config.yaml \
  --host 127.0.0.1 --port 4000 --telemetry False
```

`--config`, `--host`, `--port` และ `--telemetry` เป็น flag จาก `litellm --help` ในเวอร์ชัน 1.89.0 แล็บยังตั้ง `LITELLM_LOCAL_MODEL_COST_MAP=True` เพื่อให้ proxy ใช้รายการราคาที่มากับแพ็กเกจแทนการดาวน์โหลด แล็บสร้าง master key แบบสุ่มใหม่ทุกครั้งที่รัน และส่งผ่าน environment ไม่เคยส่งผ่านบรรทัดคำสั่งและไม่เคยพิมพ์ออกมา แล็บจะหยุด proxy เมื่อจบการทำงาน แม้จะกด Ctrl-C ก็ตาม

```bash
# on: laptop
.venv/bin/python week25/08_litellm_gateway/labs/lab02_gateway_live.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้: LiteLLM proxy ของจริง เรียก Ollama บนแล็ปท็อปจริง ติดป้าย LAPTOP STAND-IN)

```
▣ STEP 1 · start LiteLLM with lab 01's config
$ LITELLM_MASTER_KEY=sk-w25-… week25/.venv-litellm/bin/litellm --config week25/08_litellm_gateway/.runs/litellm.config.yaml --host 127.0.0.1 --port 4000 --telemetry False
◆ gateway up on http://127.0.0.1:4000 after 2.1 s (pid 41369, log 08_litellm_gateway/.runs/litellm-4000.log)

▣ STEP 2 · GET /v1/models — the aliases, not the engines' model ids
│ spark-a/ollama · spark-a/vllm · spark-a/sglang · spark-a/trtllm · spark-b/ollama · spark-b/vllm · spark-b/sglang · spark-b/trtllm · chat · laptop-fast · laptop-nemotron

▣ STEP 3 · call two aliases through the OpenAI SDK (LAPTOP STAND-IN: Ollama on this Mac)
→ POST http://127.0.0.1:4000/v1/chat/completions · model=laptop-fast
· ANSWER  NVIDIA Tesla GPUs.
→ POST http://127.0.0.1:4000/v1/chat/completions · model=laptop-nemotron
· ANSWER  An API gateway acts as a single entry point that routes client requests to the appropriate backend services, handling tasks like request routing, authentication, rate limiting, and response aggregation.
│ alias asked for  model in reply   x-litellm-model-api-base   tokens  time
│ ───────────────  ───────────────  ─────────────────────────  ──────  ─────
│ laptop-fast      laptop-fast      http://localhost:11434/v1  6       0.5 s
│ laptop-nemotron  laptop-nemotron  http://localhost:11434/v1  35      9.1 s

▣ STEP 4 · streaming works through the gateway too
· STREAM  1, 2, 3, 4, 5, 6, 7, 8

▣ STEP 5 · the master key: three requests to GET /v1/models
│ Authorization   HTTP  type              message
│ ──────────────  ────  ────────────────  ───────────────────────────────────────────
│ no key          401   auth_error        Authentication Error, No api key passed in.
│ a wrong key     400   no_db_connection  No connected db.
│ the master key  200   11 aliases
✓ no key → 401: the gateway refuses anonymous callers
✓ wrong key → 400: refused (no database, so LiteLLM cannot look up virtual keys and says so)
✓ master key → 200

▣ STEP 6 · …but the engine behind it has no lock
│ GET http://localhost:11434/v1/models with NO key → HTTP 200, 7 models
◆ gateway stopped (pid 41369); port 4000 is free again: True
```

สามสิ่งที่ควรสังเกต:

1. **คำตอบระบุชื่อ alias ไม่ใช่ชื่อ engine** ฟิลด์ `model` บอกว่า `laptop-fast` ส่วน header `x-litellm-model-api-base` บอกว่าเซิร์ฟเวอร์ไหนเป็นผู้ตอบจริง (อีกอย่าง คำตอบของ gemma3:4b ผิด DGX Spark ใช้ GB10 ไม่ใช่ Tesla โมเดล 4B เป็นแค่ fallback ไม่ใช่ผู้รู้ทุกเรื่อง)
2. **คีย์ผิดได้ 400 ไม่ใช่ 401** ถ้าไม่มีฐานข้อมูล LiteLLM จะค้นหา *virtual key* (คีย์แยกตามทีมพร้อมงบประมาณ) ไม่ได้ คีย์ใดก็ตามที่ไม่ใช่ master key จึงล้มเหลวด้วย `no_db_connection` แต่ก็ยังถูกปฏิเสธอยู่ดี การเพิ่ม `DATABASE_URL` ของ Postgres จะเปิดใช้ virtual key ซึ่งเกินขอบเขตของโมดูลนี้
3. **Step 6 คือจุดที่ต้องระวัง** gateway ปกป้องเฉพาะผู้เรียกที่ผ่านมันเท่านั้น Ollama, vLLM, SGLang และ TensorRT-LLM ไม่มีการยืนยันตัวตน บน Spark ให้เก็บพอร์ตของ engine ไว้เป็นส่วนตัว (bind กับ `127.0.0.1` เมื่อ gateway รันบน Spark เครื่องเดียวกัน หรือให้เข้าถึงได้เฉพาะผ่านลิงก์ Spark-to-Spark) และเปิดเผยแค่ :4000 playbook เปิดเซิร์ฟเวอร์ด้วย `--host 0.0.0.0` หรือ `--network host` ซึ่งเปิดให้เข้าถึงได้ทุก interface รวมถึง tailnet ของคุณ (Module 01 ส่วนที่ 3)

OpenAI client ตัวไหนก็ใช้ได้แบบเดียวกัน รวมถึง `curl`:

```bash
# on: laptop
curl -s http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"model": "laptop-fast", "messages": [{"role": "user", "content": "Say hello in three words."}], "max_tokens": 20}'
```

✓ Checkpoint: lab 02 จบด้วยบรรทัด ✓ สามบรรทัดใน Step 5 และ `port 4000 is free again: True`

## 4 · เมื่อ Spark ล่ม: fallback, retry, cooldown และ load balancing (lab 03)

ฟีเจอร์สามอย่างของ router เป็นตัวกำหนดว่า client จะเห็นอะไรเมื่อ Spark ปิดอยู่:

| ฟีเจอร์ | Config | ทำงานกับ |
|---|---|---|
| **retry** | `num_retries: 1` | คำขอที่ล้มเหลวจะถูกส่งอีกครั้ง ไปยัง deployment อื่นของ **alias เดียวกัน** ถ้ามี |
| **cooldown** | `allowed_fails: 1`, `cooldown_time: 30` | deployment ที่ล้มเหลวซ้ำ ๆ จะถูกพักไว้ 30 วินาที ผู้เรียกจึงไม่ต้องรอมันอีก |
| **fallback** | `fallbacks: [{chat: [laptop-fast]}]` | เมื่อ alias ล้มเหลวทั้งกลุ่ม router จะลอง **alias อื่น** |

Lab 03 เพิ่ม alias สาธิตสองตัวเข้าไปใน config ของ lab 01: `one-spark-down` (vLLM ของ Spark A บวกแล็ปท็อป เป็นสอง deployment ใน alias เดียว) และ `laptop-pair` (Ollama บนแล็ปท็อปภายใต้สองชื่อ แทน Spark สองเครื่อง)

```bash
# on: laptop
.venv/bin/python week25/08_litellm_gateway/labs/lab03_fallback_and_balance.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ที่ยังไม่ได้ตั้งค่า Spark: deployment ของ "Spark" ล่ม แล็ปท็อปเป็นผู้ตอบ, LAPTOP STAND-IN)

```
▣ STEP 1 · spark-a/vllm, twice — one deployment on Spark A, fallback on the laptop
│ alias asked   HTTP  answered by (group)  api_base                   retries  fallbacks  time   reply
│ ────────────  ────  ───────────────────  ─────────────────────────  ───────  ─────────  ─────  ──────
│ spark-a/vllm  200   laptop-fast          http://localhost:11434/v1  0        1          4.1 s  Ready.
│ spark-a/vllm  200   laptop-fast          http://localhost:11434/v1  0        1          1.5 s  Ready.
✓ Spark A's vLLM did not answer → both calls fell back to laptop-fast (Ollama on this Mac, LAPTOP STAND-IN)

▣ STEP 2 · chat — load-balanced across Spark A and Spark B vLLM
│ chat         200   laptop-fast          http://localhost:11434/v1  0        1          3.2 s  Ready.

▣ STEP 3 · one-spark-down × 6 — retries inside the group, then cooldown
│ call  HTTP  api_base that answered     retries  time
│ ────  ────  ─────────────────────────  ───────  ──────
│ 1     200   http://localhost:11434/v1  1        6.5 s
│ 2     200   http://localhost:11434/v1  1        13.6 s
│ 3     200   http://localhost:11434/v1  0        0.1 s
│ 4     200   http://localhost:11434/v1  0        0.2 s
│ 5     200   http://localhost:11434/v1  0        0.2 s
│ 6     200   http://localhost:11434/v1  0        0.2 s
│ calls that first landed on the dead Spark and were retried on the laptop: [1, 2]

▣ STEP 4 · laptop-pair × 8 — the balancer at work (two names for the SAME laptop Ollama)
│ http://127.0.0.1:11434/v1      ████████████ 4
│ http://localhost:11434/v1      ████████████ 4
✓ simple-shuffle spread the calls over both deployments

▣ STEP 5 · the gateway's own log
│ INFO:     127.0.0.1:55921 - "POST /v1/chat/completions HTTP/1.1" 200 OK
│ … 17 requests logged, 17 of them HTTP 200 — the failed Spark attempts are not in this log
```

อ่านทีละแถว:

- **Step 1** `spark-a/vllm` มีแค่ deployment เดียว จึงไม่มีอะไรให้ retry *ภายใน* alias ทุกคำขอลอง Spark A ล้มเหลว แล้ว fallback (`fallbacks 1`) ในการรันแบบ `--detailed_debug` บน Mac เครื่องนี้ router ลอง `spark-a:8000` อีกครั้งในคำขอที่สอง alias ที่มี deployment เดียวจะไม่ถูกข้าม ในที่นี้ความล้มเหลวเกิดขึ้นทันที เพราะ host ที่เป็น placeholder resolve ไม่ได้ แต่ Spark ที่ resolve ได้แต่ปิดเครื่องอยู่จะเสียเวลา connection timeout ทุกคำขอ ให้ alias ที่สำคัญมีสอง deployment
- **Step 3** คือส่วนที่น่าสนใจ คำขอที่ 1 และ 2 ไปตกที่ Spark ที่ล่ม และถูก **retry** ไปที่แล็ปท็อป (`retries 1`) client ยังเห็น HTTP 200 อยู่ ถึงตอนนี้ Spark ล้มเหลวเกิน `allowed_fails: 1` แล้ว จึงเข้าสู่ **cooldown** (การรันแบบ `--detailed_debug` บน Mac เครื่องนี้ log ว่ามันอยู่ในรายการ cooldown ของ router หลังความล้มเหลวครั้งที่สอง) และคำขอที่ 3–6 ไปที่แล็ปท็อปโดยตรง (`retries 0`) การ shuffle เป็นแบบสุ่ม ลำดับคำขอของคุณจึงจะต่างออกไป
- **Step 4** `simple-shuffle` กระจาย 8 คำขอเป็น 4 และ 4 ในการรันนี้ ถ้ามี Spark จริงสองเครื่อง โหลดของแต่ละเครื่องจะลดลงครึ่งหนึ่ง router ยังรองรับ `least-busy` และ `latency-based-routing` ด้วย (เป็นค่าที่ใช้ได้ของ `routing_strategy` ในเวอร์ชัน 1.89.0)
- **เวลา** ส่วนใหญ่เป็นเวลาที่ Ollama บนแล็ปท็อปใช้สร้างคำตอบคำเดียว บนเครื่องที่ใช้ร่วมกับงานอื่น ไม่ใช่ต้นทุนของ router และแน่นอนว่าไม่ใช่ตัวเลขของ Spark

log ตั้งต้นมีหนึ่งบรรทัดต่อคำขอ ทั้งหมดเป็น `200` ความล้มเหลวเกิดขึ้น *ภายใน* gateway ถ้าต้องการเห็นทุกการตัดสินใจเรื่อง routing ให้เพิ่ม `--detailed_debug` ในคำสั่ง `litellm`

✓ Checkpoint: ในการรัน Step 3 ของคุณ มีบางคำขอแสดง `retries 1` และคำขอหลัง ๆ แสดง `retries 0` และคุณอธิบายเหตุผลได้โดยใช้ `allowed_fails` และ `cooldown_time`

## 5 · gateway ควรรันที่ไหน?

| | บนแล็ปท็อป (แล็บของโมดูลนี้) | บน Spark A (แนะนำเมื่อจะแชร์ และสำหรับ capstone) |
|---|---|---|
| ใครใช้ได้บ้าง | แอปบนแล็ปท็อปเครื่องนี้ | ใครก็ได้ใน tailnet ของคุณที่มีคีย์ |
| URL ของ engine | `http://spark-a:8000/v1` engine จึงต้องเข้าถึงได้จากแล็ปท็อป | `http://localhost:8000/v1` สำหรับ Spark A engine อยู่บน `127.0.0.1` ต่อไปได้ |
| Spark B | ผ่าน tailnet | ผ่านลิงก์ Spark-to-Spark หรือ tailnet |
| Fallback บนแล็ปท็อป | `localhost:11434` | ได้เฉพาะเมื่อ Ollama บนแล็ปท็อป listen บน tailnet (`OLLAMA_HOST=0.0.0.0`) ไม่อย่างนั้นให้ fallback ไปที่ Ollama ของ Spark A |
| ทำงานได้นานเท่ากับ | ตราบที่แล็ปท็อปของคุณยังเปิดอยู่ | ตราบที่ Spark เปิดอยู่ |

Lab 01 เขียนไฟล์รูปแบบสำหรับ Spark A ให้ด้วย engine ของ Spark A จะกลายเป็น `localhost` ส่วน fallback บนแล็ปท็อปจะถูกตัดออก หรือชี้ไปที่ชื่อ tailnet ของแล็ปท็อปถ้าคุณส่งค่านั้นมา เมื่อตั้งค่า Spark ไว้ ไฟล์จะถูกคัดลอกไปที่ `~/w25/litellm/config.yaml` บน Spark A:

```bash
# on: laptop
.venv/bin/python week25/08_litellm_gateway/labs/lab01_build_config.py --on-spark
# or keep the laptop fallback, via the tailnet (the laptop must run: OLLAMA_HOST=0.0.0.0 ollama serve)
.venv/bin/python week25/08_litellm_gateway/labs/lab01_build_config.py --on-spark --laptop-url http://<laptop-tailnet-name>:11434/v1
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ในโหมด DRY: ไม่มีการคัดลอกอะไร)

```
▣ STEP 4 · the variant for a gateway ON Spark A
│ alias           api_base
│ ──────────────  ─────────────────────────
│ spark-a/ollama  http://localhost:11434/v1
│ spark-a/vllm    http://localhost:8000/v1
│ …
│ chat            http://localhost:8000/v1
│ chat            http://spark-b:8000/v1
│ fallbacks: [{'chat': ['spark-a/ollama']}] …
◆ no --laptop-url: the laptop fallbacks are dropped (localhost on a Spark is the Spark), and `chat` falls back to Spark A's Ollama instead.
$ scp litellm.spark.yaml <spark>:~/w25/litellm/config.yaml   [DRY]
```

จากนั้นบน Spark A ให้ติดตั้ง LiteLLM เวอร์ชันเดียวกันใน venv แล้วเปิดในเบื้องหลัง นี่คือคำสั่งที่ใช้บน Mac เครื่องนี้ ปรับให้เข้ากับ path ของ Spark คอร์สยังไม่ได้รันคำสั่งเหล่านี้บน Spark ถ้า wheel ตัวไหน build ไม่ผ่านบน aarch64 ข้อความ error จะระบุชื่อแพ็กเกจ:

```bash
# on: spark
sudo apt install -y python3-venv                    # only if the next line says ensurepip is not available
python3 -m venv ~/w25/litellm-venv
~/w25/litellm-venv/bin/pip install 'litellm[proxy]==1.89.0'
mkdir -p ~/w25/litellm ~/w25/logs
( umask 077; echo "sk-$(openssl rand -hex 16)" > ~/w25/litellm/master.key )
LITELLM_MASTER_KEY="$(cat ~/w25/litellm/master.key)" LITELLM_LOCAL_MODEL_COST_MAP=True \
  nohup ~/w25/litellm-venv/bin/litellm --config ~/w25/litellm/config.yaml \
  --host 0.0.0.0 --port 4000 --telemetry False > ~/w25/logs/litellm.log 2>&1 &
curl -s http://localhost:4000/health/liveliness
```

`/health/liveliness` ตอบ `"I'm alive!"` โดยไม่ต้องใช้คีย์ (บันทึกจาก Mac เครื่องนี้) บนแล็ปท็อป ให้อ่านคีย์ผ่าน SSH มาไว้ในเชลล์ของคุณเท่านั้น ห้ามเขียนลงไฟล์ใน repo:

```bash
# on: laptop
export LITELLM_MASTER_KEY="$(ssh spark-a cat ~/w25/litellm/master.key)"
curl -s http://spark-a:4000/v1/models -H "Authorization: Bearer $LITELLM_MASTER_KEY"
```

> 💡 LiteLLM มี container image เผยแพร่ด้วย (ดูหน้า Docker ในเอกสารของ LiteLLM) ถ้าคุณอยากใช้ Docker บน Spark ให้ตรวจว่า tag ที่ pull มามี build สำหรับ **arm64** และตรงกับเวอร์ชัน 1.89 คอร์สยังไม่ได้ทดสอบวิธีนี้

**จาก Lab Runner** บล็อก ⚡ ด้านล่างส่งไปที่ `litellm` บน Spark A (:4000) runner ไม่เคยเก็บ master key ของคุณ จึงไม่ส่ง header `Authorization` gateway ที่ตั้งคีย์ไว้จะตอบ 401 และบล็อกจะแสดงข้อความอ้างอิงแทน ถ้าไม่มี gateway รันบน Spark runner จะ fallback ไปที่ Ollama บนแล็ปท็อปโดยตรง (ติดป้าย LAPTOP STAND-IN) ไม่ได้ผ่าน gateway สำหรับการเรียกที่ต้องใช้คีย์ ให้ใช้แล็บ, `curl` หรือ OpenAI SDK พร้อมคีย์ของคุณ

```spark
{"target": "litellm", "which": "a", "model": "chat", "messages": [{"role": "user", "content": "In one sentence: why put a gateway in front of two DGX Sparks?"}], "max_tokens": 120}
```

```spark
{"target": "litellm", "which": "a", "model": "laptop-fast", "messages": [{"role": "user", "content": "Say which alias you are, in five words."}], "max_tokens": 40}
```

✓ Checkpoint: คุณเลือกที่ตั้งของ gateway แล้ว คือ lab 01 `--on-spark` เขียน `litellm.spark.yaml` ออกมาแล้ว หรือคุณตัดสินใจเก็บ gateway ไว้บนแล็ปท็อปไปก่อน

## Labs — รันแล็บได้ที่นี่

**labs/lab01_build_config.py** — สร้าง config ของ LiteLLM หนึ่งไฟล์สำหรับทุก engine บน Spark ทั้งสองเครื่อง พร้อม fallback ไปที่แล็ปท็อป แล้ว validate ด้วย PyYAML และ Router ของ LiteLLM (`--on-spark` เขียนรูปแบบสำหรับ Spark A)

**labs/lab02_gateway_live.py** — เปิด LiteLLM proxy บนแล็ปท็อปเครื่องนี้ เรียกผ่าน alias ด้วย OpenAI SDK (ทั้งแบบปกติและแบบ streaming) และดู master key ปฏิเสธคำขอ

**labs/lab03_fallback_and_balance.py** — ชี้ alias ไปที่ Spark ที่ล่มอยู่ แล้วดู fallback, retry, cooldown และ load balancing ใน header `x-litellm-*`

ทั้งสามแล็บรันบนแล็ปท็อป Lab 02 และ 03 เปิด proxy ของจริงและหยุดมันทุกครั้งเมื่อจบ `labs/_gateway.py` เป็น helper ที่ใช้ร่วมกัน (ตัวสร้าง config และ process ของ proxy) ไม่ใช่แล็บ

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 08 — ทำ gateway ให้สมบูรณ์** เปิด `week25/08_litellm_gateway/exercises/ex08_complete_the_gateway.py` YAML ที่ต้นไฟล์มี deployment `chat` หนึ่งตัวบน Spark A และมี `TODO` สี่จุด:

1. deployment `chat` ตัวที่สอง: โมเดลเดียวกันบน Spark B (load balancing)
2. `fallbacks`: `chat` → `laptop-fast`
3. `num_retries: 1`, `timeout: 120`, `allowed_fails: 1`, `cooldown_time: 30`
4. master key ที่อ่านจาก environment คือ `os.environ/LITELLM_MASTER_KEY`

```bash
# on: laptop
.venv/bin/python week25/08_litellm_gateway/exercises/ex08_complete_the_gateway.py
```

**Expected output** (เมื่อทำ TODO ครบทั้งสี่จุด บันทึกจาก Mac เครื่องนี้ด้วยเฉลย)

```
✓ TODO 1: `chat` has two deployments — vLLM on spark-a:8000 and spark-b:8000, same model
✓ TODO 2: chat falls back to laptop-fast
✓ TODO 3: num_retries 1 · timeout 120 · allowed_fails 1 · cooldown_time 30
✓ TODO 4: master_key comes from the environment, not from the file
✓ LiteLLM Router loaded it: 3 deployments, aliases ['chat', 'laptop-fast']

═ saved to week25/08_litellm_gateway/.runs/ex08.config.yaml. Try it live (it will fall back to your laptop):
  export LITELLM_MASTER_KEY=sk-$(openssl rand -hex 12)
  week25/.venv-litellm/bin/litellm --config week25/08_litellm_gateway/.runs/ex08.config.yaml --port 4000
```

เมื่อเปิดไฟล์ที่บันทึกไว้ด้วยวิธีนั้นแล้วขอ `chat` บน Mac เครื่องนี้ได้ HTTP 200 พร้อม `x-litellm-model-group: laptop-fast` และ `x-litellm-attempted-fallbacks: 1`

<details><summary>คำใบ้ — key ของ TODO 2 และ 3 ต้องใส่ที่ไหน?</summary>

ทั้งห้าตัวเป็น key ของ `router_settings` อยู่ที่ระดับการย่อหน้าเดียวกับ `routing_strategy` ส่วน `fallbacks` เป็น list ของ map ที่มี key เดียว: `fallbacks: [{"chat": ["laptop-fast"]}]`

</details>

<details><summary>คำใบ้ — ทำไมไม่วางคีย์ลงในไฟล์ไปเลย?</summary>

ไฟล์ config มักถูกคัดลอก commit และแชร์ต่อ `os.environ/LITELLM_MASTER_KEY` บอก LiteLLM ให้อ่าน environment variable ตอนเริ่มทำงาน ไฟล์จึงแชร์ได้อย่างปลอดภัย และคีย์ยังอยู่ในเชลล์ของคุณ

</details>

<details><summary>ท้าทายเพิ่ม — ส่ง Ollama ของ Spark A ไปที่ Spark B</summary>

เพิ่ม fallback `spark-a/ollama: [spark-b/ollama, laptop-fast]` ใน config ของ lab 01 fallback จะถูกลองตามลำดับ alias ไหนเป็นผู้ตอบเมื่อมีแค่ Spark B ที่เปิดอยู่?

</details>

✓ Checkpoint: บรรทัด TODO ทั้งสี่และบรรทัด Router เป็น ✓

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `ImportError: Missing dependency No module named 'backoff'` | คุณรัน `.venv/bin/litellm` ให้ใช้ `week25/.venv-litellm/bin/litellm` (ส่วนที่ 0) |
| `port 4000 is taken on this laptop — using 4001 instead` | มีอย่างอื่น listen ที่ 4000 อยู่ (อาจเป็น gateway อีกตัว) แล็บจะเลือกพอร์ตว่างถัดไป ถ้าทำด้วยมือ: `lsof -iTCP:4000 -sTCP:LISTEN` |
| `Authentication Error, No api key passed in.` (401) | ส่ง `Authorization: Bearer $LITELLM_MASTER_KEY` |
| `No connected db.` (400) | คีย์นั้นไม่ใช่ master key ถ้าไม่มีฐานข้อมูล จะใช้ได้เฉพาะ master key |
| `UnsupportedParamsError: openai does not support parameters: ['reasoning_effort']` | ย้ายไปไว้ใน `extra_body: {reasoning_effort: none}` (ส่วนที่ 2) หรือตั้ง `litellm_settings: drop_params: true` เพื่อทิ้งพารามิเตอร์นั้น |
| การตั้งค่าใน `router_settings` ไม่มีผล | log ขึ้นว่า `Key '…' is not a valid argument for Router.__init__(). Ignoring this key.` ให้รัน lab 01: การตรวจด้วย Router จะบอกชื่อที่พิมพ์ผิด |
| ทุกคำขอไปที่ alias ของ Spark ใช้เวลาหลายวินาที แล้วจึง fallback | alias ที่มี deployment เดียวจะ retry Spark ที่ล่มทุกครั้ง และ host ที่ resolve ได้แต่ปิดเครื่องอยู่จะเสียเวลา connect timeout ให้เพิ่ม deployment ตัวที่สองให้ alias หรือลด `timeout` |
| gateway ยังรันอยู่หลังจากแล็บแครช | `pgrep -fl "litellm --config"` แล้ว `kill <pid>` ปกติแล็บจะหยุดมันเองเมื่อจบตามปกติและเมื่อกด Ctrl-C |

## Next — บทถัดไป

ไปต่อที่ [Lab 09 — fine-tune ด้วย LLaMA Factory](../09_llama_factory/TUTORIAL.md): เทรน LoRA, QLoRA และ full fine-tune บน Spark ของคุณ จากนั้น Module 13 จะ serve ผลลัพธ์และส่งผ่าน gateway ตัวนี้
