# ▶ Spark Lab 03 — Ollama + Open WebUI: model server ตัวแรกของคุณ

> ส่วนหนึ่งของ Week 25 · DGX Spark: fine-tune, serve และสร้างเอเจนต์ที่รันใน sandbox คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง ทุกแล็บรันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (อ้างอิงคำต่อคำจาก playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- ติดตั้ง Ollama บน Spark ดึงโมเดลตัวแรกตาม playbook (`gpt-oss:20b`) และดูว่าโมเดลถูกเก็บไว้ที่ไหน
- ตัดสินใจว่าใครเข้าถึงพอร์ต 11434 ได้บ้าง: localhost เท่านั้น ผ่าน SSH tunnel หรือ `OLLAMA_HOST=0.0.0.0`
- เรียกเซิร์ฟเวอร์เดียวกันสองแบบ: `/api/chat` แบบ native ของ Ollama และ `/v1/chat/completions` ที่เข้ากันได้กับ OpenAI
- จัดการโมเดลที่คิดก่อนตอบ (thinking model): อ่านฟิลด์ `reasoning` และปิดการคิดด้วย `reasoning_effort: "none"`
- วัด time-to-first-token (TTFT) และ tokens/second แล้วเทียบกับเพดาน bandwidth จาก Module 01
- รันวงรอบ tool calling เต็มรูปแบบ แล้ววางหน้าแชต (chat UI) ไว้ด้านบนด้วย Open WebUI

**Time** ~50 นาที · **Difficulty** ระดับเริ่มต้น · **Hardware** DGX Spark 1 เครื่อง (หรือไม่มีเลยก็ได้: ใช้โหมด DRY + Ollama บนแล็ปท็อปเป็นตัวแทนที่ติดป้ายไว้)

**Playbook ทางการที่ครอบคลุม:** [Open WebUI with Ollama](https://build.nvidia.com/spark/open-webui) · [Ollama](https://build.nvidia.com/spark/ollama) · [Run local LLMs](https://build.nvidia.com/spark/llms)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| ทำ Module 01 เสร็จแล้ว | `ssh -o BatchMode=yes spark-a true` | แล็บส่งคำสั่งไปที่ Spark ผ่าน SSH |
| Docker บน Spark โดยไม่ต้อง sudo | `docker ps > /dev/null` ไม่พิมพ์อะไรออกมา | Open WebUI รันเป็น container (ส่วนที่ 7) |
| ดิสก์บน Spark ว่าง ~40 GB | `df -h /` | ~15 GB สำหรับ `gpt-oss:20b`, ~7 GB สำหรับ image ของ Open WebUI และเผื่อโมเดลสำหรับเอเจนต์อีกตัว |
| ไม่บังคับ: Ollama บนแล็ปท็อป | `curl -s localhost:11434/api/version` | ทำให้แล็บ HTTP รันได้จริงโดยไม่ต้องมี Spark โดยติดป้าย LAPTOP STAND-IN |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
curl -s localhost:11434/api/version; echo
.venv/bin/python week25/common/sparkkit.py | tail -2
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ ซึ่งมี Ollama แต่ยังไม่ได้ตั้งค่า Spark)

```
{"version":"0.34.4"}
│ litellm   (no host)                                  ○ down
│ laptop    http://localhost:11434/v1                  nemotron-3.5-lightning:latest, nemotron-3-nano:latest, gemma3:4b, kimi-k2.7-code:cloud
```

> 💡 **ตัวแทนบนแล็ปท็อป (laptop stand-in) และสิ่งที่มันไม่ใช่** เมื่อเข้าถึง Ollama ของ Spark ไม่ได้ lab 02–04 และบล็อก ⚡ จะใช้ Ollama บนแล็ปท็อปของคุณแทน คำตอบ จำนวน token และ tool call เป็นของจริง แต่ *ความเร็ว* เป็นของแล็ปท็อป ทุกบรรทัดแบบนั้นจึงเขียนว่า `LAPTOP STAND-IN` ห้ามนำไปเทียบกับตัวเลขของ Spark เด็ดขาด

✓ Checkpoint: `ssh spark-a true` ใช้ได้ (หรือคุณตัดสินใจแล้วว่าจะเรียนตามแบบ DRY) และคุณรู้ว่าแล็ปท็อปของคุณรัน Ollama อยู่หรือไม่

## 1 · ติดตั้ง Ollama บน Spark และดึงโมเดล

Ollama คือ model server ที่สร้างบน GGML engine ของ llama.cpp (Module 04) มันดาวน์โหลดโมเดลจากคลังของตัวเองในรูปไฟล์ GGUF ที่พร้อมรัน เก็บไว้ในคลังเดียว โหลดขึ้นหน่วยความจำเมื่อมีคนเรียก และ serve บนพอร์ต **11434** มันคือทางที่เร็วที่สุดจาก "Spark ว่างเปล่า" ไปสู่ "ฉันเรียกโมเดลได้แล้ว"

[Ollama playbook](https://build.nvidia.com/spark/ollama) ตรวจก่อนว่ามีการติดตั้งอยู่แล้วหรือไม่ แล้วจึงใช้สคริปต์ทางการ:

```bash
# on: spark
ollama --version
curl -fsSL https://ollama.com/install.sh | sh
```

สคริปต์จะติดตั้ง binary `ollama` และ **systemd service** ชื่อ `ollama` ที่เริ่มทำงานตอนบูต (ดูได้จากขั้นตอนถอนการติดตั้งใน playbook: `sudo systemctl stop ollama`, `sudo rm /usr/local/bin/ollama`, `sudo userdel ollama`)

โมเดลไหนก่อน? playbook ระบุไว้สามตัว และคอร์สนี้ใช้สองตัว:

| โมเดล | playbook ใช้ที่ไหน | ขนาดบนดิสก์ | คอร์สนี้ |
|---|---|---|---|
| `gpt-oss:20b` | Open WebUI playbook โมเดลแรก | ประมาณ 15 GB | โมเดลเริ่มต้นในแล็บของโมดูลนี้ |
| `qwen3.6:35b-a3b` | Open WebUI playbook ตัวเลือก "agent-ready" สำหรับ DGX Spark | — | tool calling บน Spark (lab 04) |
| `qwen2.5:32b` | Ollama playbook ฉบับเก่า | 18 GB | ไม่ได้ใช้ |

```bash
# on: spark
ollama pull gpt-oss:20b
ollama list
```

playbook แสดงให้เห็นว่าการ pull หน้าตาเป็นอย่างไรสำหรับโมเดลตัวอย่างของมันเอง:

**Expected output** (REFERENCE — ยกมาจาก Ollama playbook สำหรับ `ollama pull qwen2.5:32b`)

```
pulling manifest
pulling 58574f2e94b9: 100% ████████████████████████████  18 GB
pulling 53e4ea15e8f5: 100% ████████████████████████████ 1.5 KB
pulling d18a5cc71b84: 100% ████████████████████████████  11 KB
pulling cff3f395ef37: 100% ████████████████████████████  120 B
pulling 3cdc64c2b371: 100% ████████████████████████████  494 B
verifying sha256 digest
writing manifest
success
```

layer ใหญ่หนึ่งชั้นคือ weights แบบ GGUF ส่วน layer เล็ก ๆ คือ chat template, license และพารามิเตอร์เริ่มต้น Lab 01 รันการตรวจเหล่านี้ให้คุณ (และเริ่ม pull เบื้องหลังด้วย `--pull` เพื่อไม่ให้ขีดจำกัด 15 นาทีของ Lab Runner ตัดมันทิ้ง):

```bash
# on: laptop
.venv/bin/python week25/03_ollama_open_webui/labs/lab01_ollama_models.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้: ขั้นตอนบน Spark เป็น DRY ส่วนขั้นที่ 5 ถาม Ollama ของแล็ปท็อป)

```
▣ STEP 2 · which models has it pulled?
$ ollama list   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
NAME           ID              SIZE     MODIFIED
gpt-oss:20b    <12-hex-id>     <size>   <when>
→ add --pull to start `ollama pull gpt-oss:20b` in the background (~15 GB download).
…
▣ STEP 5 · ask the native API what each model really is
◆ the Spark's Ollama is not reachable over HTTP (no host); open a tunnel or set OLLAMA_HOST on the Spark (Section 2).

→ GET http://localhost:11434/api/tags  +  POST /api/show for each model   [THIS laptop — LAPTOP STAND-IN, not the Spark]
│ model                          on disk  params  quant   context    kind      capabilities
│ ─────────────────────────────  ───────  ──────  ──────  ─────────  ────────  ──────────────────────────────
│ nemotron-3.5-lightning:latest  25.4 GB  32.9B   Q4_K_M  1,048,576  MoE ×128  tools, thinking
│ nemotron-3-nano:latest         24.3 GB  31.6B   Q4_K_M  1,048,576  MoE ×128  tools, thinking
│ gemma3:4b                      3.3 GB   4.3B    Q4_K_M  131,072    dense     vision
│ gemma4:12b                     7.6 GB   11.9B   Q4_K_M  262,144    dense     vision, audio, tools, thinking
│ gemma4:latest                  9.6 GB   8.0B    Q4_K_M  131,072    dense     vision, audio, tools, thinking
```

อ่านตารางสุดท้ายเหมือนอ่านสเปกชีต **quant** `Q4_K_M` คือฟอร์แมต GGUF ที่ Module 04 จะแยกชิ้นส่วนให้ดู **capabilities** บอกว่าโมเดลไหนเรียก tool ได้ (`tools`) และโมเดลไหนคิดออกมาให้เห็น (`thinking`) **MoE ×128** หมายถึงมี expert 128 ตัว แต่ทำงานจริงแค่ไม่กี่ตัวต่อ token ซึ่งเป็นเหตุผลที่โมเดล 31.6B ยังเร็วได้

✓ Checkpoint: `ollama list` บน Spark แสดง `gpt-oss:20b` หรือคุณอธิบายได้ว่า `--pull` ใน lab 01 ทำอะไร และ log ของมันไปอยู่ที่ไหน (`~/w25/logs/ollama-pull.log`)

## 2 · ใครเข้าถึงพอร์ต 11434 ได้: localhost, tunnel หรือ OLLAMA_HOST

โดยค่าเริ่มต้น Ollama listen ที่ **127.0.0.1:11434**: มีแค่โปรแกรมบน Spark เองเท่านั้นที่เรียกได้ binary ของ Ollama ระบุเรื่องนี้ไว้ในข้อความ help ของมันเอง:

```bash
# on: laptop
ollama serve --help | grep -E 'OLLAMA_(HOST|KEEP_ALIVE|NUM_PARALLEL|MAX_LOADED|CONTEXT_LENGTH)'
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ Ollama 0.34.4 เวอร์ชันเดียวกันบน Spark จะพิมพ์รายการเดียวกัน)

```
      OLLAMA_HOST                   IP Address for the ollama server (default 127.0.0.1:11434)
      OLLAMA_CONTEXT_LENGTH         Context length to use unless otherwise specified (default: 4k/32k/256k based on VRAM)
      OLLAMA_KEEP_ALIVE             The duration that models stay loaded in memory (default "5m")
      OLLAMA_MAX_LOADED_MODELS      Maximum number of loaded models per GPU
      OLLAMA_NUM_PARALLEL           Maximum number of parallel requests
```

คุณมีสามวิธีในการเรียก Ollama ของ Spark จากแล็ปท็อป:

| วิธี | ทำอย่างไร | ใครอื่นเข้าถึงได้บ้าง | ใช้เมื่อ |
|---|---|---|---|
| **SSH tunnel** (แนะนำ) | `ssh -N -L 21434:localhost:11434 spark-a` | ไม่มีใคร | ใช้เสมอสำหรับแล็บ |
| **NVIDIA Sync custom app** | ตาม Ollama playbook: Name `Ollama Server`, Port `11434`, ไม่มี launch script | ไม่มีใคร | คุณใช้ NVIDIA Sync อยู่แล้ว |
| **Bind กับทุก interface** | `OLLAMA_HOST=0.0.0.0:11434` บน Spark | ทุกอุปกรณ์ใน LAN และ tailnet ของคุณ โดย **ไม่มีการยืนยันตัวตน** | เฉพาะเครือข่ายแล็บที่ไว้ใจได้ |

**Tunnel** แล็ปท็อปของคุณอาจรัน Ollama ที่ 11434 อยู่แล้ว ดังนั้นให้ forward พอร์ตของ Spark มาที่ **21434** (เทคนิคเดียวกับ Module 01, lab 03) แล้วชี้แล็บไปที่นั่น:

```bash
# on: laptop
ssh -N -L 21434:localhost:11434 spark-a &
curl -s http://localhost:21434/api/version; echo
export SPARK_URL_OLLAMA=http://localhost:21434/v1     # or set it in 🖥 Spark setup
```

`SPARK_URL_OLLAMA` จะถูกใช้ก็ต่อเมื่อตั้ง `SPARK_HOST` ไว้และเชื่อมต่อได้ด้วย (โหมด LIVE) ไม่อย่างนั้นแล็บจะถอยไปใช้ตัวแทนบนแล็ปท็อป และบอกให้รู้

**Bind กับทุก interface** วิธีนี้ไม่อยู่ใน playbook ของ NVIDIA แต่เป็น systemd override มาตรฐานจาก FAQ ของ Ollama เอง service อ่าน `OLLAMA_HOST` จาก unit file ของมัน การ export ในเชลล์ของคุณจึงไม่มีผลอะไร:

```bash
# on: spark
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="OLLAMA_HOST=0.0.0.0:11434"\n' | \
  sudo tee /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
ss -tln | grep 11434
```

บรรทัดสุดท้ายควรแสดง `*:11434` หรือ `0.0.0.0:11434` แทน `127.0.0.1:11434` ถ้าจะย้อนกลับ ให้ลบ `override.conf` แล้วรันสองคำสั่งสุดท้ายอีกครั้ง

> ⚠ Ollama ไม่มี API key ใครก็ตามที่เข้าถึง 0.0.0.0:11434 ได้จะใช้โมเดลของคุณ ดึงโมเดลใหม่ และลบโมเดลได้ ให้ใช้ tunnel ต่อไป หรือวาง LiteLLM ที่มีคีย์ไว้ด้านหน้า (Module 08)

ไฟล์ override เดียวกันนี้คือที่สำหรับตัวแปรอื่น ๆ ด้วย: `OLLAMA_KEEP_ALIVE=30m` ทำให้โมเดลค้างอยู่ในหน่วยความจำระหว่างแล็บ และ `OLLAMA_NUM_PARALLEL` ให้หลายคำขอใช้โมเดลที่โหลดไว้ตัวเดียวร่วมกันได้ ส่วนที่ 5 จะแสดงว่าทำไมตัวแรกจึงสำคัญ

✓ Checkpoint: `curl -s http://localhost:21434/api/version` ตอบผ่าน tunnel ของคุณ และคุณบอกได้ว่าทำไม `export OLLAMA_HOST=…` ในเชลล์จึงไม่เปลี่ยน service ที่รันอยู่

## 3 · สอง API เซิร์ฟเวอร์เดียว

Ollama พูด HTTP สองภาษาบนพอร์ตเดียวกัน:

| | Native API | API ที่เข้ากันได้กับ OpenAI |
|---|---|---|
| แชต | `POST /api/chat` | `POST /v1/chat/completions` |
| แสดงรายการโมเดล | `GET /api/tags` | `GET /v1/models` |
| รายละเอียดโมเดล | `POST /api/show` (template, capabilities, context length) | — |
| ดึง / ลบ | `POST /api/pull`, `DELETE /api/delete` | — |
| จำกัดความยาวผลลัพธ์ | `"options": {"num_predict": 200}` | `"max_tokens": 200` |
| สวิตช์การคิด | `"think": false` | `"reasoning_effort": "none"` |
| ข้อมูลเวลา | `eval_count`, `eval_duration`, `load_duration` (หน่วยนาโนวินาที) | มีแค่จำนวน token ใน `usage` |
| Streaming | JSON คั่นด้วยการขึ้นบรรทัดใหม่ เปิดเป็นค่าเริ่มต้น | Server-Sent Events (`data: …`) เมื่อใส่ `"stream": true` |
| ใครใช้ | CLI `ollama`, Open WebUI | LiteLLM, NAT, OpenClaw และ OpenAI SDK ทุกตัว — ส่วนที่เหลือทั้งหมดของสัปดาห์นี้ |

การทดสอบของ playbook เองเรียก native API ผ่าน tunnel:

```bash
# on: laptop
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5:32b",
  "messages": [{"role": "user", "content": "Write me a haiku about GPUs and AI."}],
  "stream": false
}'
```

(นั่นคือคำสั่งของ playbook ซึ่งถือว่า NVIDIA Sync forward Spark มาที่พอร์ต 11434 บนเครื่อง ถ้าใช้ tunnel ของคอร์ส ให้ใช้พอร์ต 21434 และ `gpt-oss:20b`)

**Expected output** (REFERENCE — ยกมาจาก Ollama playbook ซึ่งเรียกสิ่งนี้ว่า "expected response format")

```json
{
  "model": "qwen2.5:32b",
  "created_at": "2024-01-15T12:30:45.123Z",
  "message": {
    "role": "assistant",
    "content": "Silicon power flows\nThrough circuits, dreams become real\nAI awakens"
  },
  "done": true
}
```

คำตอบจริงมีมากกว่านั้น: มีฟิลด์เวลาที่ฟอร์แมตของ OpenAI ไม่มี Lab 02 ส่งคำถามเดียวทั้งสองแบบแล้วพิมพ์ฟิลด์เหล่านั้นออกมา:

```bash
# on: laptop
.venv/bin/python week25/03_ollama_open_webui/labs/lab02_native_vs_openai.py
```

**Expected output** (LAPTOP STAND-IN — บันทึกจาก Mac เครื่องนี้ด้วย nemotron-3-nano ตัวเลขของคุณจะต่างออกไป)

```
▣ STEP 1 · native API, thinking on (the model's default)
→ POST http://localhost:11434/api/chat  {"model": "nemotron-3-nano:latest", "stream": false, "options": {"num_predict": 200}, "think": true}
· content   '51'
~ thinking  73 chars: 'We need to answer with the number only. 17 * 3 = 51. So output just "51".'…
│ native field          value    meaning
│ ────────────────────  ───────  ─────────────────────────────────────────────────
│ load_duration         36.86 s  loading weights into memory (0 if already loaded)
│ prompt_eval_count     31       input tokens
│ prompt_eval_duration  0.212 s  prefill time
│ eval_count            32       output tokens, thinking included
│ eval_duration         0.607 s  decode time → 52.7 tok/s server-side
│ total_duration        42.22 s  (wall clock seen by this script: 42.23 s)

▣ STEP 2 · OpenAI-compatible API, same question
→ POST http://localhost:11434/v1/chat/completions  {"model": "nemotron-3-nano:latest", "max_tokens": 200}
· content    '51'
~ reasoning  139 chars (Ollama's /v1 puts thinking in a `reasoning` field)
◆ usage: prompt_tokens=31 completion_tokens=52 · 7.89 s wall · no server timings in the OpenAI format
```

ดูที่ `load_duration`: **36.9 จาก 42.2 วินาที** หมดไปกับการโหลดโมเดล 24 GB ขึ้นหน่วยความจำ เพราะระหว่างนั้นโปรแกรมอื่นบนแล็ปท็อปนี้โหลดโมเดลตัวอื่นไป บน Spark ก็เกิดเรื่องเดียวกันเมื่อคุณสลับโมเดล ส่วนที่ 5 จะกลับมาพูดถึงเรื่องนี้

✓ Checkpoint: คุณบอกชื่อฟิลด์ native ที่ให้ค่า tok/s ของเซิร์ฟเวอร์เอง (`eval_count ÷ eval_duration`) และ endpoint ที่เครื่องมือตัวอื่น ๆ ตลอดสัปดาห์ใช้ (`/v1/chat/completions`) ได้

## 4 · Thinking model: ฟิลด์ reasoning และ `reasoning_effort: "none"`

โมเดลอย่าง `gpt-oss`, Qwen3.x, Nemotron 3 และ Gemma 4 สามารถ **คิด** ก่อนตอบได้: มันเขียนลำดับการให้เหตุผลที่ซ่อนไว้ก่อน แล้วจึงเขียนคำตอบ Ollama แยกสองส่วนนี้ออกจากกัน:

- native API: `message.thinking` อยู่ข้าง `message.content`;
- OpenAI API: `message.reasoning` (หรือ `delta.reasoning` ตอน streaming) อยู่ข้าง `content` ส่วน vLLM และ SGLang เรียกฟิลด์เดียวกันนี้ว่า `reasoning_content` (Module 05–06) sparkkit อ่านได้ทั้งสองแบบ

การคิดใช้ token และ token คือเวลา ขั้นที่ 3 ของ lab 02 ปิดการคิดทั้งสองแบบ:

**Expected output** (LAPTOP STAND-IN — รันเดียวกับด้านบน)

```
▣ STEP 3 · thinking off, both ways
→ POST http://localhost:11434/api/chat  {"model": "nemotron-3-nano:latest", "stream": false, "options": {"num_predict": 200}, "think": false}
→ POST http://localhost:11434/v1/chat/completions  {"model": "nemotron-3-nano:latest", "max_tokens": 200, "reasoning_effort": "none"}
│ API         setting                     output tokens  answer
│ ──────────  ──────────────────────────  ─────────────  ──────
│ native      "think": true               32             '51'
│ native      "think": false              3              '51'
│ OpenAI /v1  (default)                   52             '51'
│ OpenAI /v1  "reasoning_effort": "none"  3              '51'
```

คำตอบเหมือนเดิม แต่ใช้ 3 token แทน 32–52 ความยาวของการคิดยังเปลี่ยนไปในแต่ละรันด้วย: ในการรันแล็บเดียวกันอีกสองครั้งบนแล็ปท็อปนี้ ครั้งแรกคือการเรียก native แบบเปิดการคิด และอีกครั้งคือการเรียก `/v1` แบบค่าเริ่มต้น ใช้ **ครบทั้ง 200 token** ไปกับการให้เหตุผล แล้วคืน `content` ที่ **ว่างเปล่า** (`completion_tokens=200`, คำตอบ `''`) นี่คือบั๊กคลาสสิกของ thinking model: `max_tokens` ถูกใช้ร่วมกันระหว่างการคิดและคำตอบ

กฎของคอร์สนี้:

| คุณต้องการ | Native API | OpenAI API (สิ่งที่ LiteLLM, NAT และเอเจนต์ส่ง) |
|---|---|---|
| คำตอบตรง ๆ และเร็ว | `"think": false` | `"reasoning_effort": "none"` |
| ให้โมเดลให้เหตุผล (คณิตศาสตร์ การวางแผน) | `"think": true` และ `num_predict` ที่มากขึ้น | ไม่ต้องใส่ `reasoning_effort` และเพิ่ม `max_tokens` |

`sparkkit.chat_any()` ส่ง `reasoning_effort: "none"` ให้ Ollama เว้นแต่แล็บจะส่ง `think=True` บล็อก ⚡ ด้านล่างก็ทำแบบเดียวกัน ลองดูเลย: มันจะถาม Ollama ของ Spark หรือตัวแทนบนแล็ปท็อปถ้ายังไม่ได้เชื่อมต่อ Spark

```spark
{"target": "ollama", "which": "a", "model": "gpt-oss:20b",
 "messages": [{"role": "system", "content": "Answer in one short sentence."},
              {"role": "user", "content": "Why does memory bandwidth limit how fast one conversation generates tokens?"}],
 "max_tokens": 120}
```

✓ Checkpoint: คุณอธิบายได้ว่าทำไม thinking model จึงคืนคำตอบว่างได้ และฟิลด์ไหนที่ปิดการคิดผ่าน `/v1`

## 5 · Streaming: วัด TTFT และ tokens ต่อวินาที

chat UI ใช้การ stream: token จะปรากฏขึ้นทันทีที่ถูกสร้าง มีตัวเลขสองตัวที่อธิบายสิ่งที่ผู้ใช้รู้สึก:

- **TTFT, time to first token**: เวลาโหลดโมเดล (ถ้ายังไม่อยู่ในหน่วยความจำ) บวก **prefill** ซึ่งอ่าน prompt ทั้งหมดในรอบเดียวแบบขนาน
- **ความเร็ว decode, tokens/second**: ทีละหนึ่ง token แต่ละ token ต้องอ่าน weight ที่ทำงานอยู่ทั้งหมดจากหน่วยความจำซ้ำ สูตรของ Module 01 จึงเป็นเพดานของมัน: `tok/s ≤ bandwidth ÷ bytes read per token`

Lab 03 stream prompt เดียวสี่ครั้งผ่าน `/v1` ด้วย `"stream": true` และ `stream_options: {"include_usage": true}` (ถ้าไม่มี Ollama จะไม่ส่งจำนวน token มาใน stream) จากนั้นถาม native API เพื่อดูตัวเลขของเซิร์ฟเวอร์เอง แล้วเทียบกับเพดาน:

```bash
# on: laptop
LAPTOP_BW_GBS=410 .venv/bin/python week25/03_ollama_open_webui/labs/lab03_speed_benchmark.py
```

`LAPTOP_BW_GBS` คือ memory bandwidth ของแล็ปท็อปคุณจากสเปกชีต (410 GB/s สำหรับ Mac เครื่องนี้ซึ่งเป็น Apple M4 Max) บน Spark แล็บจะใช้ 273 GB/s และไม่สนใจค่านี้

**Expected output** (LAPTOP STAND-IN — บันทึกจาก Mac เครื่องนี้ด้วย gemma3:4b ซึ่งเป็นโมเดล dense ขนาดเล็ก)

```
▣ STEP 1 · what are we timing?
◆ gemma3:4b: 3.34 GB on disk · dense (every weight read per token) · ≈ 3.34 GB read per token

▣ STEP 2 · stream the same prompt 4 times (run 0 may include loading the model)
→ POST http://localhost:11434/v1/chat/completions  model=gemma3:4b  stream=True
│ run              TTFT   tokens  total   tok/s
│ ───────────────  ─────  ──────  ──────  ─────  ────────────────────
│ run 0 (warm-up)  73 ms  160     1.70 s  98.4   █████████████░░░░░░░
│ run 1            38 ms  160     1.71 s  95.5   █████████████░░░░░░░
│ run 2            48 ms  160     1.71 s  96.1   █████████████░░░░░░░
│ run 3            42 ms  156     1.86 s  85.6   ███████████░░░░░░░░░
◆ median over runs 1–3: TTFT 42 ms · decode 95.5 tok/s (client-side: tokens after the first ÷ time after the first). Source: LAPTOP STAND-IN (Ollama on this laptop, not the Spark).

▣ STEP 3 · the server's own count (native /api/chat: eval_count ÷ eval_duration)
│ phase             tokens  time     rate
│ ────────────────  ──────  ───────  ─────────────────────────────────
│ decode            160     1.82 s   87.9 tok/s
│ prefill (prompt)  31      0.039 s  805 tok/s
│ model load        —       0.00 s   0 s when it was already in memory

▣ STEP 4 · compare with the bandwidth ceiling (Module 01)
│ model      bandwidth                 read/token  ceiling    measured    of ceiling
│ ─────────  ────────────────────────  ──────────  ─────────  ──────────  ──────────
│ gemma3:4b  410 GB/s (LAPTOP_BW_GBS)  3.34 GB     123 tok/s  95.5 tok/s  78%
◆ The file size includes weights that text generation never reads (gemma3:4b carries a 0.84 GB vision encoder — Module 04, lab 02), so the true ceiling is higher. Kernel overheads, sampling and other requests sharing the server keep real speed below it.
◆ on the Spark, gpt-oss:20b (~15 GB, 3.6B of 21B active) reads ≈ 2.6 GB per token → ceiling ≈ 106 tok/s (arithmetic). Run this lab on the Spark to measure it.
◆ Do not compare the laptop's tok/s with the Spark's: different chips, different models.
```

สิ่งที่ควรได้จากผลนี้:

1. **ฝั่ง client และ server ได้ค่าใกล้เคียงกัน** สคริปต์วัดได้ 95.5 tok/s ส่วน Ollama รายงาน 87.9 tok/s สำหรับอีกคำขอหนึ่ง วัดที่ฝั่ง client เมื่อคุณมองไม่เห็นเซิร์ฟเวอร์ (vLLM, gateway) และอ่านตัวเลขของเซิร์ฟเวอร์เมื่อทำได้
2. **เซิร์ฟเวอร์ที่ใช้ร่วมกันทำให้ benchmark เพี้ยน** การรันแล็บเดียวกันนี้ครั้งก่อน ขณะที่โปรแกรมอื่นกำลังใช้ Ollama ของแล็ปท็อป วัดได้ตั้งแต่ **6.8 ถึง 95 tok/s** และ TTFT สูงถึง 6 วินาที ให้ benchmark บนเซิร์ฟเวอร์ที่ว่าง รันหลายครั้ง แล้วรายงานค่ามัธยฐาน (median)
3. **TTFT น้อยมาก จนกว่าจะต้องโหลดโมเดล** 38–73 ms เมื่อโมเดลอุ่นอยู่แล้ว แบบฝึกหัดจะเล่น stream ซ้ำที่มี TTFT **18.6 วินาที** บน Spark ให้ตั้ง `OLLAMA_KEEP_ALIVE` (ส่วนที่ 2) เพื่อให้โมเดลค้างอยู่ในหน่วยความจำ
4. **เพดานคือไม้บรรทัด** `gpt-oss:20b` เป็นโมเดล Mixture-of-Experts: จาก ~15 GB บนดิสก์ มีแค่ ~2.6 GB ที่ถูกอ่านต่อ token เพดานบน Spark จึงอยู่ที่ประมาณ **106 tok/s** เมื่อคุณรัน lab 03 บน Spark คอลัมน์ "of ceiling" จะบอกว่า Ollama ทำได้ใกล้แค่ไหน

✓ Checkpoint: คุณรัน lab 03 แล้ว ชี้คอลัมน์ TTFT และ tok/s ได้ และบอกได้ว่าทำไม 95.5 tok/s ของแล็ปท็อปไม่ได้บอกอะไรเกี่ยวกับ Spark เลย

## 6 · Tool calling: โมเดลขอ โค้ดของคุณตอบ

เอเจนต์ (Module 14–18) สร้างบนกลไกเดียว คุณส่งรายการ **tool** (ชื่อ คำอธิบาย และพารามิเตอร์แบบ JSON-schema) โมเดลไม่ได้รันอะไรเลย: มันตอบกลับด้วย **`tool_calls`** คือชื่อฟังก์ชันและ argument แบบ JSON โปรแกรมของคุณรันฟังก์ชันนั้น แล้วส่งผลกลับเป็นข้อความ `role: "tool"` จากนั้นโมเดลจึงเขียนคำตอบ

Lab 04 ให้ tool จริงสองตัวแก่โมเดล คือสูตรของ Module 01 จาก sparkkit แล้วถามคำถามที่มันตอบได้ด้วย tool เหล่านี้เท่านั้น มันตรวจ `/api/show` ก่อน เพราะมีแค่โมเดลที่มี capability `tools` เท่านั้นที่ทำแบบนี้ได้ บน Spark จะใช้ `qwen3.6:35b-a3b` ซึ่งเป็นโมเดล agent-ready ตาม playbook

```bash
# on: laptop
.venv/bin/python week25/03_ollama_open_webui/labs/lab04_tool_calling.py
```

**Expected output** (LAPTOP STAND-IN — บันทึกจาก Mac เครื่องนี้ด้วย nemotron-3-nano)

```
▣ STEP 1 · can this model call tools? (POST /api/show → capabilities)
◆ nemotron-3-nano:latest: completion, tools, thinking

▣ STEP 2 · ask, with two tools on offer
→ POST http://localhost:11434/v1/chat/completions · tools=[weights_gb, decode_ceiling_tok_s]
· ANSWER
→ tool_call weights_gb({"params_b":117,"fmt":"mxfp4"})
→ tool_call decode_ceiling_tok_s({"active_params_b":5.1,"fmt":"mxfp4"})
◆ LAPTOP STAND-IN (not Spark numbers) · nemotron-3-nano:latest · TTFT 11370 ms · 85 tok in 11.4s · 7.5 tok/s

▣ STEP 3 · run 2 tool call(s) here, send the results back as role=tool
│ weights_gb({"params_b":117,"fmt":"mxfp4"}) → 62.2
│ decode_ceiling_tok_s({"active_params_b":5.1,"fmt":"mxfp4"}) → 100.8
· ANSWER  The weights of the **gpt-oss-120b** model (117B total parameters in **mxfp4** format) occupy **62.2 GB** of storage.

          On a **DGX Spark** (with 273 GB/s memory bandwidth), the decode ceiling for this model is **100.8 tokens per second**.
          …
◆ ground truth from sparkkit: weights 62.2 GB · ceiling 100.8 tok/s — does the final answer quote these numbers?
```

รายละเอียดสามข้อที่ควรสังเกต:

- คำตอบแรกมี **คำตอบว่าง** และ tool call สองตัว **แบบขนาน**: โมเดลขอตัวเลขทั้งสองพร้อมกัน
- คำขอที่มี tool calling ถูกส่ง **แบบไม่ stream** (sparkkit ปิด `stream` เมื่อมี `tools`) ดังนั้น "TTFT" ตรงนั้นคือเวลาของคำตอบทั้งหมด และในที่นี้รวมเวลาโหลดโมเดลด้วย
- ตัวเลขสุดท้ายถูกต้องเพราะ *โค้ดของคุณ* เป็นผู้คำนวณ โมเดลแค่เลือกว่าจะเรียกฟังก์ชันไหนด้วย argument อะไร Module 15–16 วาง sandbox และ policy ไว้ตรงช่องว่างนั้นพอดี

✓ Checkpoint: คุณไล่ข้อความทั้งสี่ในวงรอบได้ (user → assistant ที่มี `tool_calls` → `tool` → assistant) และบอกได้ว่าฝั่งไหนเป็นผู้รันฟังก์ชัน

## 7 · Open WebUI: chat UI ด้วย docker run คำสั่งเดียว

[Open WebUI](https://build.nvidia.com/spark/open-webui) คือเว็บแอปสไตล์ ChatGPT ที่โฮสต์เอง playbook รัน image **`ghcr.io/open-webui/open-webui:ollama`** ซึ่งรวม Ollama **ของตัวเอง** ไว้ใน container มันไม่ได้ใช้ Ollama ที่คุณติดตั้งในส่วนที่ 1

```bash
# on: spark
docker ps > /dev/null                       # blank output = Docker works without sudo
docker pull ghcr.io/open-webui/open-webui:ollama
docker run -d -p 12000:8080 --gpus=all \
  -v open-webui:/app/backend/data \
  -v open-webui-ollama:/root/.ollama \
  --name open-webui ghcr.io/open-webui/open-webui:ollama
```

นี่คือบรรทัด `docker run` จาก launch script ของ NVIDIA Sync ใน playbook เส้นทาง **desktop** ของ playbook ใช้ `-p 8080:8080` แทน ส่วนคอร์สนี้ใช้ **12000** (พอร์ตใน `sparkkit.PORTS` และใน tunnel ของ Module 01) เพื่อไม่ให้ชนกับบริการอื่นบน 8080

| Flag | ความหมาย |
|---|---|
| `-p 12000:8080` | เว็บแอป listen ที่ 8080 ภายใน container และ 12000 บน Spark |
| `--gpus=all` | Ollama ที่รวมมาได้ใช้ GPU GB10 (ถ้าไม่มี: "GPU not detected" และ inference บน CPU ช้ามาก) |
| `-v open-webui:/app/backend/data` | บัญชีผู้ใช้และประวัติแชตคงอยู่หลังรีสตาร์ต |
| `-v open-webui-ollama:/root/.ollama` | โมเดลของ Ollama ที่รวมมา: เป็นคลังโมเดล **ชุดที่สอง** แยกจากของส่วนที่ 1 |

เปิดผ่าน tunnel (หรือเพิ่มเป็น NVIDIA Sync custom app บนพอร์ต 12000 แบบที่ playbook ทำ):

```bash
# on: laptop
ssh -N -L 12000:localhost:12000 spark-a
# open http://localhost:12000
```

จากนั้นทำตาม playbook ในเบราว์เซอร์:

1. **Get Started** → สร้างบัญชี admin ซึ่งเก็บไว้บน Spark เท่านั้น
2. **Select a model** → พิมพ์ `gpt-oss:20b` → **Pull "gpt-oss:20b" from Ollama.com** container มาโดยไม่มีโมเดลติดมาเลย
3. เลือกโมเดลแล้วส่ง **Write me a haiku about GPUs** คำตอบแรกอาจใช้เวลาถึง 30 วินาทีระหว่างที่โมเดลโหลดขึ้น GPU
4. สำหรับเอเจนต์ playbook แนะนำ `qwen3.6:35b-a3b` และ context window **อย่างน้อย 32K token** หรือ 64K ขึ้นไปถ้าหน่วยความจำพอ

เนื่องจาก Ollama ที่รวมมาไม่ได้เปิดพอร์ตออกมา `ollama list` บน Spark จึงไม่แสดงโมเดลที่คุณดึงผ่าน UI ให้ดูภายใน container แทน (lab 01 ขั้นที่ 3 ทำสิ่งนี้ให้):

```bash
# on: spark
docker exec open-webui ollama list
```

> 💡 Ollama สองตัวที่ถือโมเดล 15 GB ตัวเดียวกันหมายถึงดิสก์ 30 GB และถ้าโหลดขึ้นทั้งคู่ ก็ใช้หน่วยความจำเป็นสองเท่า บนเครื่อง unified memory 128 GB เรื่องนี้สำคัญ หยุด container เมื่อใช้เสร็จ: `docker stop open-webui` ขั้นตอนเก็บกวาดของ playbook (`docker rm`, `docker rmi`, `docker volume rm open-webui open-webui-ollama`) จะลบประวัติแชตและโมเดลอย่างถาวร

✓ Checkpoint: `http://localhost:12000` แสดง Open WebUI ผ่าน tunnel ของคุณและตอบ prompt ไฮกุได้ และคุณอธิบายได้ว่าทำไม `ollama list` บน host จึงไม่แสดงโมเดลของมัน

## Labs — รันแล็บได้ที่นี่

**labs/lab01_ollama_models.py** — ตรวจการติดตั้ง Ollama บน Spark โมเดล และการ bind พอร์ต แล้วอธิบายทุกโมเดลผ่าน native API

**labs/lab02_native_vs_openai.py** — ส่งคำถามเดียวไปที่ `/api/chat` แบบ native และ API `/v1` แบบ OpenAI แล้วนับว่าการคิดมีต้นทุนเท่าไร

**labs/lab03_speed_benchmark.py** — stream prompt หนึ่งอัน วัด TTFT และ tok/s แล้วเทียบกับเพดาน bandwidth

**labs/lab04_tool_calling.py** — รันวงรอบ tool calling เต็มรูปแบบ โดยใช้สูตรหน่วยความจำและความเร็วของ Module 01 เป็น tool

Lab 01 รันขั้นตอนเชลล์แบบ LIVE บน Spark ของคุณหรือแบบ DRY ส่วน Lab 02–04 เรียก Ollama ของ Spark เมื่อมันตอบ ไม่อย่างนั้นใช้ตัวแทนบนแล็ปท็อป (ติดป้ายไว้) และถ้าไม่มีทั้งคู่จะหยุดพร้อมหมายเหตุ DRY แต่ละคำขอขอ token ไม่เกิน 200 ตัว

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 03 — มาตรวัด stream** เปิด `week25/03_ollama_open_webui/exercises/ex03_stream_meter.py` มันเล่นซ้ำ stream **จริง** ที่บันทึกจาก endpoint `/v1` ของ Ollama บนแล็ปท็อปของคอร์ส (nemotron-3-nano เปิดการคิด) พร้อมเวลาที่แต่ละบรรทัดมาถึง มี `TODO` สี่จุด:

1. `parse_sse(lines)`: แปลง Server-Sent Events ให้เป็นก้อน JSON (chunk)
2. `split_text(chunks)`: ต่อชิ้นส่วน `content` และชิ้นส่วน `reasoning` แยกกัน
3. `speed(timed)`: TTFT จำนวน token และ tok/s ด้วยกฎเดียวกับที่ `sparkkit.chat()` ใช้
4. `request_body(...)`: คำขอแบบ streaming พร้อม `include_usage` และปิดการคิด

```bash
# on: laptop
.venv/bin/python week25/03_ollama_open_webui/exercises/ex03_stream_meter.py
```

**Expected output** (เมื่อทำ TODO ครบทั้งสี่จุด ตัวตรวจทำงานแบบออฟไลน์ ผลจึงเหมือนกันทุกเครื่อง)

```
✓ parse_sse: 27 JSON chunks, the last one carries usage, [DONE] not included
✓ split_text: answer '51' · reasoning 'We need to answer with the number only: 17 * 3 = 51 …'
✓ speed: TTFT 18649 ms · 31 tokens · ≈ 82.7 tok/s after the first token
✓ request_body: stream + include_usage · reasoning_effort 'none' only when think=False

▣ your meter, applied to the recorded stream (LAPTOP STAND-IN)
│ answer '51' · reasoning 68 chars · 31 tokens billed
│ TTFT 18.6 s · then 82.7 tok/s
```

<details><summary>คำใบ้ — SSE หนึ่งบรรทัดหน้าตาเป็นอย่างไร?</summary>

`data: {"id":"chatcmpl-667","object":"chat.completion.chunk",…,"choices":[{"index":0,"delta":{"content":"","reasoning":" need"},"finish_reason":null}]}` ตัด `data:` ออก แล้ว `json.loads` ส่วนที่เหลือ บรรทัดว่างคั่นระหว่าง event และ stream จบด้วย `data: [DONE]` ซึ่งไม่ใช่ JSON

</details>

<details><summary>ท้าทายเพิ่ม — ชี้มาตรวัดของคุณไปที่ Spark</summary>

ส่ง `request_body("gpt-oss:20b", …)` ไปที่ `http://localhost:21434/v1/chat/completions` ผ่าน tunnel บันทึกคู่ `(time, line)` ขณะอ่านคำตอบ แล้วป้อนให้ `speed()` เทียบกับตัวเลขของ lab 03 สำหรับโมเดลเดียวกัน และลอง `think=True` เพื่อดูว่าการให้เหตุผลเพิ่ม TTFT ไปเท่าไรก่อนจะถึง token แรกของ *คำตอบ*

</details>

✓ Checkpoint: บรรทัดตรวจทั้งสี่เป็น ✓

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `curl: (7) Failed to connect to localhost port 11434` จากแล็ปท็อป | tunnel ไม่ได้รันอยู่ หรือ forward ผิดพอร์ต เริ่ม `ssh -N -L 21434:localhost:11434 spark-a` แล้วเรียกพอร์ต **21434** |
| แล็บขึ้นว่า `LAPTOP STAND-IN` ทั้งที่ tunnel ใช้ได้ | ยังไม่ได้ตั้ง `SPARK_HOST` หรือเชื่อมต่อไม่ได้ แล็บจึงอยู่ในโหมด DRY ตั้งค่าใน 🖥 Spark setup และตั้ง `SPARK_URL_OLLAMA=http://localhost:21434/v1` |
| `model "gpt-oss:20b" not found` | โมเดลถูกดึงใน Open WebUI (Ollama ใน container) ไม่ใช่บน host รัน `ollama pull gpt-oss:20b` บน Spark หรือตรวจด้วย `docker exec open-webui ollama list` |
| คำตอบว่าง และ output tokens == `max_tokens` | thinking model ใช้งบไปกับการให้เหตุผลหมด ส่ง `reasoning_effort: "none"` (หรือ `think: false`) หรือเพิ่ม `max_tokens` |
| คำขอแรกใช้เวลา 10–40 วินาที คำขอถัดไปเร็ว | โมเดลกำลังโหลด (`load_duration` ใน native API) ตั้ง `OLLAMA_KEEP_ALIVE` ใน service override และหลีกเลี่ยงการสลับโมเดลระหว่างคำขอ |
| error `does not support tools` | โมเดลนั้นไม่มี capability `tools` (ดูตารางของ lab 01) ใช้ `qwen3.6:35b-a3b` หรือ `gpt-oss:20b` บน Spark |
| Open WebUI: "GPU not detected" หรือตอบช้ามาก | container ถูกเปิดโดยไม่มี `--gpus=all` รัน `docker rm -f open-webui` แล้วรันบรรทัด `docker run` ใหม่ (volume ยังเก็บข้อมูลของคุณไว้) |
| `Port 12000 already in use` | มีบริการอื่นจองพอร์ตอยู่ ใช้ `ss -tlnp \| grep 12000` หรือ map พอร์ตฝั่ง host อื่น: `-p 12001:8080` |
| หน่วยความจำตึงทั้งที่โมเดลควรใส่ได้ | หมายเหตุเรื่อง UMA ของ playbook: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` และตรวจด้วยว่าไม่มี Ollama สองตัวถือโมเดลพร้อมกันอยู่ |

## Next — บทถัดไป

ไปต่อที่ [Lab 04 — llama.cpp + LM Studio](../04_llama_cpp_lm_studio/TUTORIAL.md): คอมไพล์ engine ที่อยู่ข้างใน Ollama ด้วยตัวเอง เปิดไฟล์ GGUF เพื่อดูว่า `Q4_K_M` หมายถึงอะไรจริง ๆ และ serve มันด้วย `llama-server` และ LM Studio
