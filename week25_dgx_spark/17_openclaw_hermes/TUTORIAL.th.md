# ▶ Spark Lab 17 — OpenClaw และ Hermes Agent กับ LLM บนเครื่อง

> ส่วนหนึ่งของ Week 25 · DGX Spark: fine-tune, serve และสร้างเอเจนต์ที่รันใน sandbox คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง ทุกแล็บรันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (อ้างอิงคำต่อคำจาก playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- serve โมเดลที่พร้อมสำหรับเอเจนต์ (agent-ready) ตามที่ playbook แนะนำสำหรับ DGX Spark (`nvidia/Qwen3.6-35B-A3B-NVFP4`) ด้วย vLLM และให้มันเข้าถึงได้เฉพาะภายใน Spark
- รัน **preflight** แบบอ่านอย่างเดียวบน Spark: เครื่องมือที่จำเป็น รายการโมเดลบน `:8000` ใครเข้าถึงพอร์ตนั้นได้บ้าง และสิ่งที่ติดตั้งไว้แล้ว
- **ทดสอบ tool calling** ก่อนติดตั้งอะไรทั้งนั้น: การทดสอบเล็ก ๆ ห้าข้อที่แยกโมเดลแชตออกจากโมเดลสำหรับเอเจนต์
- ติดตั้ง **OpenClaw** และ **Hermes Agent** ตามที่ playbook ของแต่ละตัวบอก แล้วชี้ทั้งสองไปที่โมเดลบนเครื่องตัวเดียวกัน
- สร้างและตรวจ config ของเอเจนต์ทั้งสองบนแล็ปท็อป และเปรียบเทียบว่าเอเจนต์ทั้งสองตั้งค่าและเรียก tool ต่างกันอย่างไร

**Time** ~50 นาที · **Difficulty** ระดับกลาง · **Hardware** Spark 1 เครื่อง (หรือไม่มีเลยก็ได้: ใช้โหมด DRY + แล็บบนแล็ปท็อป)

**Playbook ทางการที่ครอบคลุม:** [Run OpenClaw with a Local LLM](https://build.nvidia.com/spark/openclaw) · [Run Hermes Agent with a Local LLM](https://build.nvidia.com/spark/hermes-agent)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| ทำ Module 01 เสร็จแล้ว: `ssh spark-a` ใช้ได้ด้วยคีย์ | `ssh -o BatchMode=yes spark-a true` | lab 17-1 รันผ่าน SSH |
| อ่าน Module 16 แล้ว | — | โมดูลนั้นรัน OpenClaw *ภายใน* sandbox ส่วนที่นี่รัน **โดยไม่มี** sandbox |
| Spark ที่คุณใช้เป็นเครื่องทดลอง | คุณตัดสินใจเอง | playbook ทั้งสองบอกว่า: ให้รันเอเจนต์บนระบบที่แยกไว้เฉพาะหรือถูกกันออกจากระบบอื่น |
| ล็อกอิน Hugging Face **บน Spark** | `hf auth whoami` บน Spark | vLLM ดาวน์โหลดโมเดล |
| Python ของ repo นี้ + Ollama บนแล็ปท็อป | `.venv/bin/python --version`, `curl -s localhost:11434/api/tags` | lab 17-2 และ 17-3 รันบนแล็ปท็อป |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
.venv/bin/python --version
curl -s localhost:11434/api/tags | head -c 120; echo
```

> 🔐 **โมดูลนี้ไม่มี sandbox** ใน Module 16 มี OpenShell คั่นอยู่ระหว่างเอเจนต์กับเครื่องของคุณ แต่ที่นี่ OpenClaw และ Hermes รันเป็น **ผู้ใช้ของคุณเอง** บน Spark: tool ที่รันคำสั่งเชลล์ก็รันจริง playbook ของ OpenClaw ให้ระดับความเสี่ยง **Medium to High** ส่วน playbook ของ Hermes ให้ระดับ **Medium** แล็บในโมดูลนี้ไม่เคยติดตั้ง messaging integration เปิดใช้ tool หรือแก้ไฟล์ config ของเอเจนต์ ห้ามวางโทเค็นจริงลงในแชต prompt ไฟล์ config ที่คุณแชร์ หรือบรรทัดคำสั่งเด็ดขาด

✓ Checkpoint: คุณบอกได้ในประโยคเดียวว่าการรันเอเจนต์ที่นี่ต่างจาก Module 16 อย่างไร

## 1 · สองเอเจนต์ โมเดลเดียวบนเครื่อง

**OpenClaw** ตามคำของ playbook คือเอเจนต์แบบ **local-first**: มัน "remembers conversations, adapts to your usage, runs continuously, uses context from your files and apps, and can be extended with community **skills**" มันคือเอเจนต์ที่ NemoClaw นำไปใส่ใน sandbox ใน Module 16

**Hermes Agent** ของ Nous Research คือเอเจนต์แบบ **พัฒนาตัวเองได้ (self-improving)** มันรันเป็น TUI ในเทอร์มินัล "creates skills from experience, improves them during use, persists memory across sessions, and can run scheduled tasks via its built-in cron" gateway ในตัวทำให้เข้าถึงมันได้จาก Telegram, Discord หรือ Slack

ทั้งสองต้องการสิ่งเดียวกันจาก Spark: **endpoint ที่เข้ากันได้กับ OpenAI (OpenAI-compatible)** ซึ่งโมเดลเรียก tool ได้ playbook ทั้งสองแนะนำโมเดลและเซิร์ฟเวอร์เดียวกันสำหรับ DGX Spark:

| ฮาร์ดแวร์ | โมเดล agent-ready ที่แนะนำ | serve ด้วย |
|---|---|---|
| DGX Spark | Agent-ready Qwen3.6-35B-A3B (NVFP4) — `nvidia/Qwen3.6-35B-A3B-NVFP4` | vLLM ที่ `http://localhost:8000/v1` |

(REFERENCE — จากแท็บ Agent-ready Models ของ playbook ทั้งสอง) Qwen3.6-35B-A3B เป็นโมเดล Mixture-of-Experts: มีพารามิเตอร์ทำงานอยู่ราว 3B ต่อ token ซึ่งเป็นสูตรจาก Module 01 สำหรับการรักษาความเร็วบน 273 GB/s

คุณรันเอเจนต์ตัวไหนก็ได้ *ภายใน* sandbox ของ NemoClaw ด้วย: `NEMOCLAW_AGENT=hermes` (หรือ `nemohermes onboard`) ของ Module 16 ทำแบบนั้นให้ Hermes โมดูลนี้ติดตั้งทั้งสองตัวตรง ๆ เพื่อให้คุณเห็นว่าแต่ละเอเจนต์เป็นอย่างไรเมื่ออยู่ตัวเดียว

✓ Checkpoint: คุณบอกชื่อ model handle หนึ่งตัวและ URL หนึ่งเส้นที่เอเจนต์ทั้งสองจะใช้ได้

## 2 · serve โมเดล agent-ready บน Spark

playbook ของเอเจนต์ทั้งสองส่งคุณไปที่ [vLLM playbook](https://build.nvidia.com/spark/vllm) เพื่อดูคำสั่งเปิดเซิร์ฟเวอร์ และ Module 05 อธิบาย vLLM อย่างละเอียด vLLM playbook ฉบับล่าสุดระบุแค่ชื่อโมเดลและลิงก์ไปยังสูตร (recipe) บน `recipes.vllm.ai` คำสั่งฉบับเต็มซึ่งรวม flag สามตัวที่เอเจนต์ต้องใช้ อยู่ใน vLLM playbook **ฉบับเก่า** ในหัวข้อ "Run Agent Ready Qwen3.6 35B Model with vLLM" (`dgx-spark-playbooks/nvidia/vllm/README.md`) Module 05 ก็ใช้แหล่งเดียวกัน นี่คือคำสั่งนั้น โดยมีการเปลี่ยนแปลงสามจุดที่อธิบายไว้ด้านล่าง:

```bash
# on: spark
docker pull vllm/vllm-openai:latest
read -rs HF_TOKEN && export HF_TOKEN        # paste your token; it is not echoed or kept in shell history
docker run -d --name vllm-server --gpus all -p 127.0.0.1:8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --moe-backend marlin \
  --gpu-memory-utilization 0.4 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
  --load-format fastsafetensors \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice
```

สามบรรทัดสุดท้ายคือสิ่งที่ทำให้นี่เป็นเซิร์ฟเวอร์สำหรับ **เอเจนต์** `--enable-auto-tool-choice` ให้โมเดลตัดสินใจเองว่าจะเรียก tool เมื่อไร `--tool-call-parser qwen3_xml` แปลงผลลัพธ์ tool call ของ Qwen3.6 ให้เป็น `tool_calls` แบบ OpenAI และ `--reasoning-parser qwen3` ย้ายส่วนการคิด (thinking) ไปไว้ในฟิลด์ `reasoning` แยกต่างหาก ถ้าไม่มี flag ด้าน tool เซิร์ฟเวอร์ยังแชตได้ แต่จะไม่ส่ง `tool_calls` กลับมา และ lab 17-2 จะให้คะแนนว่า "chat only" ส่วน `--max-model-len 262144` คือขนาด context ที่คุณจะตั้งให้เอเจนต์ทั้งสองในส่วนที่ 5 และ 6

จุดที่ต่างจากคำสั่งของ playbook ซึ่งตั้งใจทำทั้งหมด:

- **`-p 127.0.0.1:8000:8000`** แทน `-p 8000:8000` แบบใน playbook จะ listen บนทุก interface ทำให้ใครก็ได้ใน LAN หรือ tailnet ของคุณใช้โมเดลของคุณได้ playbook ของเอเจนต์ทั้งสองบอกให้ bind endpoint ไว้กับ Spark และเอเจนต์เหล่านี้รันบน Spark อยู่แล้ว loopback จึงเพียงพอ `--host 0.0.0.0` ยังคงไว้: มันคือที่อยู่ *ภายใน* container (ถ้าคุณใช้ตัวเลือก "Existing vLLM" ของ Module 16 ด้วย ให้รันการตรวจ `inference.local` ของมันใหม่หลังเปลี่ยนการ bind)
- **`read -rs HF_TOKEN`** แทน `export HF_TOKEN="your_huggingface_token"` เพื่อไม่ให้โทเค็นไปตกอยู่ใน `~/.bash_history`
- **`-d --name vllm-server`** แทน `-it` เพื่อให้เซิร์ฟเวอร์ทำงานต่อหลังจากปิดเทอร์มินัล และคุณอ่าน log ของมันได้ด้วยชื่อ

รอจนเห็น `Application startup complete` (`docker logs -f vllm-server`) แล้วตรวจรายการโมเดล:

```bash
# on: spark
curl -sS http://localhost:8000/v1/models
```

"You should see your served model handle (for example `nvidia/Qwen3.6-35B-A3B-NVFP4`) in the returned list." (REFERENCE — ยกมาจาก Hermes playbook)

สำหรับเอเจนต์ context สำคัญ: playbook ของ OpenClaw ขอ **อย่างน้อย 32K token** และ 64K ขึ้นไปถ้าหน่วยความจำพอ ตั้ง context window ของเอเจนต์ให้ตรงกับ `--max-model-len` ของเซิร์ฟเวอร์

✓ Checkpoint: `curl -sS http://localhost:8000/v1/models` บน Spark คืน JSON ที่มี array `"data"` ซึ่งมี model handle ของคุณอยู่

## 3 · Preflight: lab 17-1

Lab 17-1 รัน Step 1 ของ Hermes playbook (`uname -a`, `curl --version`, `git --version`) พร้อมการตรวจอีกสี่ข้อที่เอเจนต์ทั้งสองต้องพึ่ง เป็นแบบอ่านอย่างเดียว

```bash
# on: laptop
.venv/bin/python week25/17_openclaw_hermes/labs/lab17_1_preflight.py
```

**Expected output** (โหมด DRY บันทึกจาก Mac เครื่องนี้ ถ้ามี Spark คุณจะได้ค่าของเครื่องคุณเองพร้อม ✓ / ✕)

```
│ check                     result     last line of output
│ ────────────────────────  ─────────  ────────────────────────────────────────────────────
│ Linux                     ◈ example  Linux spark-abcd 6.11.0-1016-nvidia #16-Ubuntu SMP …
│ curl                      ◈ example  curl 8.5.0 (aarch64-unknown-linux-gnu) …
│ git                       ◈ example  git version 2.43.0
│ vLLM model list           ◈ example  {"object":"list","data":[{"id":"nvidia/Qwen3.6-35B-A
│ Who can reach :8000       ◈ example  LISTEN 0      4096         0.0.0.0:8000       0.0.0.
│ Existing installs         ◈ example  (end of list)
│ OpenClaw gateway process  ◈ example  no openclaw process

▣ STEP 8 · what the answers mean
◆ DRY: the EXAMPLE shows 0.0.0.0:8000 — what `docker run -p 8000:8000` gives you. See Section 2 for the fix.
```

แต่ละบรรทัดป้องกันปัญหาอะไร:

| การตรวจ | ถ้าไม่ผ่าน จะพังภายหลังแบบ… |
|---|---|
| vLLM model list | ตัวติดตั้ง Hermes "can't list any models at the model-selection prompt" ส่วน OpenClaw บอกว่า "no model available" |
| Who can reach `:8000` | model endpoint ที่ใครในเครือข่ายก็ใช้ได้ โหมด LIVE จะเตือนเมื่อเห็น `0.0.0.0:8000` |
| Existing installs | ตัวติดตั้ง Hermes เสนอให้ **import/migrate** จาก OpenClaw ซึ่ง playbook บอกให้ตอบ `n` |
| OpenClaw gateway process | การเปลี่ยน config ที่ไม่เคยมีผล เพราะไม่ได้รีสตาร์ต gateway ที่รันอยู่ |

✓ Checkpoint: ในโหมด LIVE ทุกแถวเป็น ✓ และ `:8000` listen บน `127.0.0.1` เท่านั้น

## 4 · ทดสอบ tool calling ก่อนติดตั้ง: lab 17-2

เอเจนต์ทำงานเป็นวงรอบ (loop): ส่งบทสนทนาพร้อมนิยามของ tool ได้ `tool_call` กลับมา (ชื่อและ argument แบบ JSON) **รัน tool นั้นเอง** ส่งผลลัพธ์กลับเป็นข้อความ `tool` แล้วได้คำตอบสุดท้าย โมเดลทำหน้าที่แค่เสนอ ส่วนเอเจนต์เป็นผู้ลงมือ ถ้าโมเดลไม่เคยเสนอ skill และ tool ของเอเจนต์ก็จะไม่เคยทำงาน

Lab 17-2 ทดสอบวงรอบนั้นด้วย probe เล็ก ๆ ห้าข้อ มันรันกับ vLLM บน Spark เมื่อ `:8000` ตอบ ไม่อย่างนั้นจะรันจริงกับ Ollama บนแล็ปท็อปของคุณ โดยติดป้ายไว้

| Probe | คำถาม | ทำไมเอเจนต์ต้องการ |
|---|---|---|
| P1 call | เมื่อถูกถามสิ่งที่มีแต่ tool ตอบได้ มันส่ง tool call ไหม? | ไม่มี call ก็ไม่มี tool |
| P2 choose | เมื่อเสนอ tool สามตัว มันเลือกถูกตัวไหม? | เอเจนต์เสนอ tool หลายตัว |
| P3 schema | JSON ถูกต้อง มี key ที่จำเป็น ชนิดข้อมูลถูก (`amount` เป็นตัวเลข) ไหม? | เอเจนต์ต้อง parse argument |
| P4 round trip | เมื่อได้ผลลัพธ์จาก tool คำตอบนำผลนั้นไปใช้ไหม? | งานหลายขั้นตอน |
| P5 restraint | เมื่อถูกถาม "2 + 2" มันตอบโดยไม่ใช้ tool ไหม? | ลดการเรียกที่ไม่จำเป็น (และเสี่ยง) |

```bash
# on: laptop
.venv/bin/python week25/17_openclaw_hermes/labs/lab17_2_tool_probe.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้: LAPTOP STAND-IN นี่คือโมเดลของแล็ปท็อป ไม่ใช่ของ Spark)

```
◆ probing Ollama on THIS laptop · http://localhost:11434/v1 · LAPTOP STAND-IN — tool behaviour is the model's, speed is not the Spark's

▣ STEP 1 · probe gemma4:12b
│ P1 call ✓ · P2 choose ✓ · P3 schema ✓ · P4 round trip ✓ · P5 restraint ✓
· first tool call: convert_currency({"amount":120,"from_ccy":"USD","to_ccy":"THB"})

▣ STEP 2 · probe nemotron-3-nano:latest
│ P1 call ✓ · P2 choose ✓ · P3 schema ✓ · P4 round trip ✓ · P5 restraint ✓
· first tool call: convert_currency({"from_ccy":"USD","to_ccy":"THB","amount":120})

▣ STEP 3 · probe gemma3:4b
│ P1 call ✕ · P2 choose ✕ · P3 schema ✕ · P4 round trip ✕ · P5 restraint ✕
· why: HTTP 400: registry.ollama.ai/library/gemma3:4b does not support tools

│ model                   P1  P2  P3  P4  P5  verdict      wall time
│ ──────────────────────  ──  ──  ──  ──  ──  ───────────  ─────────
│ gemma4:12b              ✓   ✓   ✓   ✓   ✓   agent-ready  52s
│ nemotron-3-nano:latest  ✓   ✓   ✓   ✓   ✓   agent-ready  26s
│ gemma3:4b               ✕   ✕   ✕   ✕   ✕   chat only    0s
◆ P1–P4 must pass for OpenClaw/Hermes to use tools. P5 failing means the agent calls tools when it should not (slower, and riskier once the tools can run commands).
```

`gemma3:4b` คือบทเรียน: มันตอบคำถามแชตได้ดี แต่ Ollama ปฏิเสธคำขอทันทีที่แนบ tool เข้าไป ถ้าชี้เอเจนต์ไปที่มัน คุณจะได้ error ไม่ใช่ผู้ช่วย บน Spark vLLM ต้องใช้ `--enable-auto-tool-choice --tool-call-parser qwen3_xml` (ส่วนที่ 2) ด้วยเหตุผลเดียวกัน: ถ้าไม่มี เซิร์ฟเวอร์จะไม่ส่ง `tool_calls` กลับมา wall time เป็นของแล็ปท็อปและไม่ได้บอกอะไรเกี่ยวกับ Spark

✓ Checkpoint: คุณรัน lab 17-2 กับโมเดลที่ตั้งใจจะใช้แล้ว และ P1–P4 เป็น ✓

## 5 · ติดตั้งและรัน OpenClaw

รันสคริปต์ติดตั้งทางการบน Spark (คำแนะนำ "อ่านก่อน" เดียวกับ Module 16 ใช้ได้ที่นี่: คุณบันทึกด้วย `-o ~/openclaw-install.sh` อ่าน แล้วจึงรันด้วย `bash` ได้):

```bash
# on: spark
curl -fsSL https://openclaw.ai/install.sh | bash
```

หลังดาวน์โหลด dependency เสร็จ OpenClaw จะแสดง **คำเตือนด้านความปลอดภัย** อ่านให้ดี เลือก **Yes** เฉพาะเมื่อคุณยอมรับความเสี่ยงได้ จากนั้นตอบคำถาม onboarding:

| คำถาม | ตัวเลือก | ทำไม |
|---|---|---|
| Quickstart vs Manual | **Quickstart** | |
| Model provider | **Skip for now** (ล่างสุดของรายการ) หรือ backend ของคุณถ้ามีอยู่ในรายการ | คุณจะตั้งค่าโมเดลใน `openclaw.json` ในขั้นถัดไป |
| Filtering models by provider | **All Providers** แล้ว **Keep Current** | |
| Communication channel | **Skip for Now** | เพิ่ม channel หลังจากพื้นฐานใช้งานได้แล้วเท่านั้น |
| Skills | **No** | skill เพิ่มทั้งความสามารถ *และ* ความเสี่ยง |
| Homebrew | **No** | ใช้กับ macOS เท่านั้น ไม่จำเป็นบน Spark |
| Hooks | playbook แนะนำให้เลือกทั้งสามตัว | หมายเหตุ: hook อาจบันทึกข้อมูลไว้บนเครื่อง |
| Dashboard URL | **บันทึก URL และโทเค็นไว้** | ต้องใช้ทั้งคู่เพื่อเปิด web UI ให้ถือว่าโทเค็นเป็นรหัสผ่าน |
| Finish | **Yes** | |

**ชี้ OpenClaw ไปที่ vLLM** ถ้าคุณข้ามขั้นเลือก provider ให้เปิด `~/.openclaw/openclaw.json` (`nano ~/.openclaw/openclaw.json`) แล้วเพิ่มส่วน `models` นี่คือตัวอย่างของ playbook สำหรับ vLLM server:

```json
"models": {
  "mode": "merge",
  "providers": {
    "vllm": {
      "baseUrl": "http://localhost:8000/v1",
      "apiKey": "vllm",
      "api": "openai-responses",
      "models": [
        {
          "id": "nvidia/Qwen3.6-35B-A3B-NVFP4",
          "name": "nvidia/Qwen3.6-35B-A3B-NVFP4",
          "reasoning": true,
          "input": ["text"],
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          },
          "contextWindow": 262144,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

`id` และ `name` ต้องตรงกับ handle ที่ serve อยู่ทุกตัวอักษร `contextWindow` ต้องตรงกับ `--max-model-len` ของเซิร์ฟเวอร์ `apiKey` เป็นแค่ค่าจองที่ (placeholder): vLLM ไม่ต้องใช้คีย์ "any non-empty placeholder works" ถ้า OpenClaw รายงานว่า endpoint ไม่รองรับ Responses API playbook บอกให้เปลี่ยน `"api"` เป็นแบบ chat-completions ที่ตรงกับเวอร์ชัน OpenClaw ของคุณ (vLLM เปิด `/v1/chat/completions` ไว้เสมอ)

รีสตาร์ต OpenClaw gateway เพื่อให้มันโหลดไฟล์ใหม่ แล้วตรวจ:

1. forward พอร์ตของ dashboard มาที่แล็ปท็อป ใช้พอร์ตใน URL ที่ตัวติดตั้งพิมพ์ออกมา (playbook ไม่ได้กำหนดพอร์ตตายตัว):

```bash
# on: laptop
ssh -N -L <port>:127.0.0.1:<port> spark-a
```

2. เปิด URL ของ dashboard (พร้อมโทเค็น) ในเบราว์เซอร์ เริ่มบทสนทนา **ใหม่** แล้วส่งข้อความสั้น ๆ ถ้าได้คำตอบแปลว่าการตั้งค่าใช้งานได้
3. ถามว่ามันใช้โมเดลอะไรอยู่ ใน chat UI คำสั่ง `/model MODEL_NAME` ใช้สลับโมเดล

> ⚠ กฎสำคัญที่สุดของ playbook: web UI ของ OpenClaw และ messaging channel ใด ๆ ต้อง **ไม่ถูกเปิด** สู่อินเทอร์เน็ตสาธารณะเด็ดขาด ถ้าไม่มีการยืนยันตัวตนที่แข็งแรง ให้ใช้ SSH tunnel หรือ VPN เปิดใช้เฉพาะ **skill ที่คุณไว้ใจ**; skill ที่เข้าถึงเทอร์มินัลหรือ file system "increase risk significantly"

✓ Checkpoint: dashboard ของ OpenClaw ตอบผ่าน tunnel ของคุณ และรายงานว่าใช้โมเดล `nvidia/Qwen3.6-35B-A3B-NVFP4` (หรือ handle ของคุณ)

## 6 · ติดตั้งและรัน Hermes Agent

Hermes playbook ผ่านการตรวจสอบกับ **Hermes Agent v0.18.0 (2026.7.1)**; ตัวติดตั้งรุ่นใหม่กว่าอาจใช้ถ้อยคำในคำถามต่างออกไป รันตัวติดตั้งจากเซสชัน SSH แบบ **interactive** (`ssh -t spark-a`) โดยที่ vLLM serve อยู่แล้ว:

```bash
# on: spark
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

playbook ใช้เส้นทาง **Blank Slate** ซึ่งทำให้ตัวช่วยตั้งค่าสั้นลง:

| คำถาม | ตัวเลือก | ทำไม |
|---|---|---|
| Install ripgrep … ffmpeg? | **Enter** (yes) | ค้นหาไฟล์เร็วขึ้น, TTS ต้องใช้ sudo |
| Import/migrate from OpenClaw? | **`n`** | การผสม migration อาจทำให้สถานะของ gateway หรือ messaging ไม่สอดคล้องกัน |
| How would you like to set up Hermes? | **Blank Slate** | Quick Setup จะล็อกอินเข้า Nous Portal แทนที่จะใช้โมเดลบนเครื่องของคุณ |
| Select provider | **Custom endpoint (enter URL manually)** | Spark ของคุณ ไม่ใช่ provider บนคลาวด์ |
| API base URL | `http://localhost:8000/v1` | vLLM จากส่วนที่ 2 |
| API key [optional] | เว้นว่าง | vLLM ไม่ต้องใช้คีย์ |
| Model selection | `nvidia/Qwen3.6-35B-A3B-NVFP4` | Hermes แสดงรายการตามที่ `/v1/models` ส่งกลับมา |
| Context length | Enter (ตรวจหาอัตโนมัติ) | สูตรนี้ serve ที่ 262144 |
| Display name | Enter | |
| Select terminal backend | **Keep current (local)** / **Local** | ดูคำเตือนด้านล่าง |
| What next? | **Start with everything disabled — finish now** | ค่อยเปิด tool ทีละตัวในภายหลัง |

โหลดเชลล์ใหม่และตรวจคำสั่ง:

```bash
# on: spark
source ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
which hermes
```

รัน `hermes` พิมพ์ `hello` แล้วกด **Enter** เมื่อคุณ `/exit` Hermes จะพิมพ์ `hermes --resume <sessionId>` ให้บันทึกไว้เพื่อคุยต่อในภายหลัง

**ไม่มีเทอร์มินัล? ใช้คำสั่ง config** ถ้าตัวติดตั้งพิมพ์ว่า "Setup wizard skipped (no terminal available)" ทางสำรองของ playbook จะตั้งค่า endpoint โดยไม่ต้องตอบคำถาม:

```bash
# on: spark
export PATH="$HOME/.local/bin:$PATH"
hermes config set model.provider custom
hermes config set model.base_url http://localhost:8000/v1
hermes config set model.default nvidia/Qwen3.6-35B-A3B-NVFP4
hermes -z "Reply exactly HERMES_OK"
```

"The last command should return `HERMES_OK`, confirming that Hermes can call the local vLLM model without opening the TUI." (REFERENCE — ยกมาจาก playbook)

**tool มาทีหลัง ทีละตัว** Blank Slate เริ่มต้นโดยปิด tool ของเอเจนต์ไว้ทั้งหมด `hermes tools` เปิดรายการขึ้นมา (web search, browser automation, terminal, file operations, code execution และอื่น ๆ; **SPACE** สลับเปิด/ปิด, **ENTER** ยืนยัน)

> ⚠ terminal backend แบบ **local** "runs them directly on the hardware platform": เมื่อคุณเปิด terminal tool ของ Hermes คำสั่งของโมเดลจะรันในฐานะผู้ใช้ของคุณบน Spark เปิดใช้เฉพาะบน Spark ที่ไม่มีอะไรที่คุณจะเดือดร้อนถ้าเอเจนต์อ่านหรือลบไป

คำสั่งใช้งานประจำวันอื่น ๆ จาก playbook: `hermes model` (สลับโมเดล), `hermes --resume <sessionId>`, `hermes update` และ `/reasoning show` ใน TUI เพื่อดูการให้เหตุผลระหว่างทางของโมเดล

✓ Checkpoint: `hermes -z "Reply exactly HERMES_OK"` พิมพ์ `HERMES_OK` บน Spark

## 7 · config เทียบกันแบบเคียงข้าง: lab 17-3

เอเจนต์ทั้งสองเก็บข้อเท็จจริงสามอย่างเดียวกัน (endpoint, model handle, ขนาด context) ไว้คนละที่ Lab 17-3 เขียน config ของทั้งคู่จากโมเดลที่ serve อยู่ตัวเดียว ตรวจความถูกต้อง และทดสอบ endpoint เบื้องต้น (smoke test) มันไม่เคยแก้ `~/.openclaw/openclaw.json`; ถ้ามี Spark มันจะคัดลอกไฟล์ทั้งสองไปไว้ที่ `~/w25/agents/` ให้คุณ merge เองด้วยมือ

```bash
# on: laptop
.venv/bin/python week25/17_openclaw_hermes/labs/lab17_3_agent_configs.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ ยังไม่ได้ตั้งค่า Spark; smoke test เป็น LAPTOP STAND-IN)

```
▣ STEP 1 · which model does the endpoint serve?
◆ no Spark vLLM reachable → using the playbooks' DGX Spark default: nvidia/Qwen3.6-35B-A3B-NVFP4
◆ the validation below cannot check 'is it served?' without the Spark; the smoke test uses the laptop stand-in

▣ STEP 2 · generate the two configs
▣ openclaw.models.json  (merge into ~/.openclaw/openclaw.json)
│ {
│   "models": {
│     "mode": "merge",
│     "providers": {
│       "vllm": {
│         "baseUrl": "http://localhost:8000/v1",
│         "apiKey": "vllm",
│         "api": "openai-responses",
…
▣ hermes-config.sh
│ #!/usr/bin/env bash
│ # Hermes: point the agent at the local vLLM endpoint (Hermes playbook, non-interactive fallback)
│ set -e
│ export PATH="$HOME/.local/bin:$PATH"
│ hermes config set model.provider custom
│ hermes config set model.base_url http://localhost:8000/v1
│ hermes config set model.default nvidia/Qwen3.6-35B-A3B-NVFP4
│ hermes -z "Reply exactly HERMES_OK"
◆ written to week25/17_openclaw_hermes/.runs/

▣ STEP 3 · validate
✓ openclaw.models.json: local /v1 endpoint, placeholder key, ≥32K context
│ broken config     validator  finding
│ ────────────────  ─────────  ────────────────────────────────────────────────────
│ real-looking key  ✓ caught   apiKey must be a non-empty placeholder (vLLM needs …
│ LAN address       ✓ caught   baseUrl host '192.168.1.42' is not local — the agen…
│ 8K context        ✓ caught   contextWindow 8192 < 32768 (the playbook's minimum …
│ missing /v1       ✓ caught   baseUrl 'http://localhost:8000' must end in /v1

▣ STEP 4 · smoke test: does this (endpoint, model) pair answer?
→ POST http://localhost:11434/v1/chat/completions · model=gemma4:12b · Ollama on THIS laptop (stand-in, not the Spark)
· ANSWER  HERMES_OK
◆ LAPTOP STAND-IN (not Spark numbers) · gemma4:12b · TTFT 10847 ms · 5 tok in 11.2s · 15.4 tok/s

✓ configs are valid for a local vLLM
✓ every broken config was caught
✓ smoke test answered (laptop)
```

เมื่อเอเจนต์ทั้งสองทำงานแล้ว นี่คือการเปรียบเทียบ:

| | OpenClaw | Hermes Agent |
|---|---|---|
| ผู้สร้าง | โปรเจกต์ OpenClaw ([openclaw.ai](https://openclaw.ai)) | Nous Research |
| สร้างมาเพื่อ | ผู้ช่วยแบบ local-first ที่ทำงานตลอดเวลา ขยายความสามารถด้วย skill จากชุมชน | เอเจนต์ที่พัฒนาตัวเองได้ เขียนและปรับปรุง skill ของตัวเอง |
| หน้าจอหลัก | web dashboard (URL + โทเค็น), แชตผ่าน gateway, `/model` | TUI ในเทอร์มินัล `hermes`; แบบครั้งเดียว `hermes -z "…"`; `--resume` |
| การติดตั้ง | `curl -fsSL https://openclaw.ai/install.sh \| bash` | `curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh \| bash` |
| config ของโมเดล | `~/.openclaw/openclaw.json` → `models.providers.vllm` {`baseUrl`, `apiKey`, `api`, `models[]`} | `hermes config set model.provider custom` / `model.base_url` / `model.default` หรือ `hermes model` |
| API ที่ใช้ | `"api": "openai-responses"` (Responses API) หรือแบบ chat-completions ถ้าไม่รองรับ | custom endpoint ที่เข้ากันได้กับ OpenAI (`…/v1`) |
| การทำให้การเปลี่ยนแปลงมีผล | รีสตาร์ต gateway เพื่อให้โหลด `openclaw.json` ใหม่ | มีผลในเซสชันถัดไป |
| tool มาจากไหน | skill: แถบด้านข้างของ web UI, [Clawhub](https://docs.openclaw.ai/tools/clawhub) หรือขอให้ OpenClaw หาให้; และ hook | `hermes tools` (web search, browser, terminal, file ops, code execution); และ skill ที่มันสร้างจากประสบการณ์ |
| tool รันที่ไหน | บน host ในฐานะผู้ใช้ของคุณ (อยู่ใน sandbox เฉพาะเมื่อผ่าน NemoClaw) | terminal backend: `local` = บน host; Docker, Modal, SSH, Daytona, Singularity อยู่นอกขอบเขตของ playbook |
| หน่วยความจำและการตั้งเวลา | จำบทสนทนาได้; cron ของ OpenClaw (Module 16 ใช้ `openclaw cron add`) | หน่วยความจำถาวรข้ามเซสชัน; cron ในตัว |
| การส่งข้อความ | channel ระหว่าง onboarding หรือเพิ่มภายหลัง | `hermes gateway setup` (Telegram, Discord, Slack) + systemd service |
| อัปเดต / ถอนการติดตั้ง | รันสคริปต์ติดตั้งใหม่; ลบไดเรกทอรีของมัน; หยุด gateway | `hermes update`; `hermes uninstall`; `~/.hermes` อาจยังค้างอยู่ |

ตัววงรอบของ tool เหมือนกันในทั้งสองตัว และเป็นสิ่งที่ lab 17-2 ทดสอบ: เอเจนต์ส่งนิยามของ tool โมเดลส่ง `tool_call` กลับมา เอเจนต์รันมัน แล้วผลลัพธ์ถูกส่งกลับไปเป็นข้อความ `tool`

✓ Checkpoint: คุณชี้ได้ว่าเอเจนต์แต่ละตัวเก็บ model handle ไว้ที่ไหน และบอกได้ว่าต้องเปลี่ยนอะไรในแต่ละตัวเพื่อสลับโมเดล

## 8 · เสริมความปลอดภัย อัปเดต และถอนการติดตั้ง

playbook ทั้งสองจบส่วนความเสี่ยงแบบเดียวกัน: คุณขจัดความเสี่ยงทั้งหมดไม่ได้ จึงต้องลดมันลง

| มาตรการ | OpenClaw playbook | Hermes playbook | คอร์สนี้ |
|---|---|---|---|
| เครื่องที่แยกออกมา มีเฉพาะข้อมูลที่เอเจนต์ต้องใช้ | แนะนำอย่างยิ่ง | แนะนำ | Spark สำหรับทดลอง |
| บัญชีเฉพาะ สิทธิ์ให้น้อยที่สุด | ใช่ | — | ห้ามใช้บัญชีหลักของคุณ |
| Web UI / endpoint ไม่เปิดสู่สาธารณะเด็ดขาด | สำคัญมาก | bind vLLM ไว้กับ Spark | SSH tunnel; `-p 127.0.0.1:8000:8000` |
| เฉพาะ skill / tool ที่ไว้ใจได้ | ใช่ | เปิด tool อย่างตั้งใจ (`hermes tools`) | Blank Slate ทีละ tool |
| จำกัดการเข้าอินเทอร์เน็ตของเอเจนต์ | firewall หรือการแยกเครือข่าย | — | หรือรันใน NemoClaw (Module 16) |
| จำกัดการส่งข้อความให้เฉพาะคุณ | — | ใส่ user ID ตัวเลขของคุณที่ "Allowed user IDs" | ถ้าเว้นว่าง ใครก็ตามที่เจอบอตก็ใช้ได้ |
| เฝ้าดูกิจกรรม | ตรวจ log และคำสั่งที่ถูกรัน | ตรวจเซสชัน; `sudo journalctl -u <hermes-gateway-unit> -e` | ตรวจทุกครั้งหลังเพิ่ม tool ใหม่ |

**การส่งข้อความสำหรับ Hermes (ไม่บังคับ)** ถ้าคุณต้องการใช้ Hermes จากมือถือ playbook จะตรวจก่อนว่า Spark เข้าถึง Telegram ได้ แล้วรันตัวช่วยตั้งค่า gateway ซึ่งจะถาม bot token (ซ่อนไว้ขณะวาง) และ user ID ที่อนุญาต:

```bash
# on: spark
curl -sS --connect-timeout 10 -o /dev/null -w "HTTP %{http_code}\n" https://api.telegram.org/
hermes gateway setup
```

บรรทัดสถานะ HTTP ใด ๆ (`HTTP 404`, `HTTP 200`, `HTTP 302`) แปลว่าการเชื่อมต่อผ่าน TLS สำเร็จ ถ้าหมดเวลา (timeout) แปลว่าเครือข่ายของคุณบล็อก Telegram ไม่มีแล็บใดรันคำสั่งเหล่านี้

**อัปเดตและถอนการติดตั้ง**

```bash
# on: spark
hermes update
hermes uninstall             # interactive; add sudo "$(which hermes)" if you installed a system gateway service
ls -la ~/.hermes             # configuration, sessions and skills may still be here
```

สำหรับ OpenClaw วิธีย้อนกลับของ playbook คือหยุด gateway แล้วถอนการติดตั้งด้วยสคริปต์ติดตั้งหรือลบไดเรกทอรีของมัน หยุด vLLM แยกต่างหากด้วย `docker stop vllm-server && docker rm vllm-server` การลบ `~/.hermes` (`rm -rf ~/.hermes`) ย้อนกลับไม่ได้ ทำเฉพาะเมื่อต้องการรีเซ็ตทั้งหมด

✓ Checkpoint: พอร์ต vLLM ของคุณเป็น loopback เท่านั้น ไม่มี messaging channel ใดเปิดอยู่โดยไม่มีรายการผู้ใช้ที่อนุญาต และคุณรู้ว่าเอเจนต์แต่ละตัวเก็บข้อมูลไว้ที่ไหน

## Labs — รันแล็บได้ที่นี่

**labs/lab17_1_preflight.py** — การตรวจแบบอ่านอย่างเดียวบน Spark: Linux, curl และ git, รายการโมเดลของ vLLM, ใครเข้าถึง `:8000` ได้ และสิ่งที่ติดตั้งไว้แล้ว

**labs/lab17_2_tool_probe.py** — probe ด้าน tool calling ห้าข้อ (call, choose, schema, round trip, restraint) กับ vLLM บน Spark หรือตัวแทนบนแล็ปท็อป

**labs/lab17_3_agent_configs.py** — สร้างส่วน `models` ของ OpenClaw และคำสั่ง config ของ Hermes จากโมเดลที่ serve อยู่ตัวเดียว ตรวจทั้งสอง และทำ smoke test กับ endpoint

Lab 17-1 รันแบบ LIVE บน Spark ของคุณหรือแบบ DRY ส่วน Lab 17-2 และ 17-3 รันจริงบนแล็ปท็อป (ใช้ vLLM บน Spark เมื่อมันตอบ) โดย 17-3 จะคัดลอกไฟล์ไปที่ Spark เมื่อตั้งค่าไว้

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 17 — โมเดลเดียว สองเอเจนต์** เปิด `week25/17_openclaw_hermes/exercises/ex17_agent_config.py` ในไฟล์มี `TODO` สามจุด:

1. `openclaw_provider(base_url, model_id, context_window)`: entry `models.providers.vllm` ในรูปแบบของ playbook
2. `hermes_commands(base_url, model_id)`: สี่คำสั่งของการตั้งค่าแบบ non-interactive ใน Hermes playbook
3. `review(provider, served)`: แจ้งเตือนเมื่อไม่มี `/v1`, host ไม่ใช่ local, คีย์ที่หน้าตาเหมือนของจริง, โมเดลที่ไม่ได้ serve อยู่ และ context ที่ต่ำกว่า 32K

```bash
# on: laptop
.venv/bin/python week25/17_openclaw_hermes/exercises/ex17_agent_config.py
```

**Expected output** (เมื่อทำ TODO ครบทั้งสามจุด บันทึกจาก Mac เครื่องนี้ด้วยเฉลยอ้างอิง)

```
✓ openclaw_provider: baseUrl · placeholder key · openai-responses · id = name · 262144 context
✓ hermes_commands: provider custom · base_url · default · hermes -z check
✓ review: good → [] · catches /v1 · local · key · served · context

▣ ~/.openclaw/openclaw.json → models.providers.vllm
│ {
│   "baseUrl": "http://localhost:8000/v1",
│   "apiKey": "vllm",
│   "api": "openai-responses",
│   "models": [
│     {
│       "id": "nvidia/Qwen3.6-35B-A3B-NVFP4",
│       "name": "nvidia/Qwen3.6-35B-A3B-NVFP4",
│ …

▣ Hermes, on the Spark
$ hermes config set model.provider custom
$ hermes config set model.base_url http://localhost:8000/v1
$ hermes config set model.default nvidia/Qwen3.6-35B-A3B-NVFP4
$ hermes -z "Reply exactly HERMES_OK"
```

<details><summary>คำใบ้ — ทำไม baseUrl ที่ไม่ใช่ local จึงนับเป็นปัญหา?</summary>

เอเจนต์เหล่านี้รันบน Spark อยู่ข้าง ๆ vLLM ดังนั้น `localhost` จึงถูกต้องเสมอสำหรับพวกมัน ที่อยู่ LAN ใน config มักหมายความว่าพอร์ตของโมเดลเปิดให้ LAN ด้วย ซึ่ง playbook ทั้งสองเตือนไว้ และถ้า config ชี้ไปที่เครื่องของคนอื่นเมื่อไร prompt และไฟล์ของคุณก็จะไปอยู่ที่นั่น

</details>

<details><summary>ท้าทายเพิ่ม — probe ก่อนเลือก</summary>

รัน `lab17_2_tool_probe.py --models gemma4:12b,gemma3:4b` แล้วใช้ `openclaw_provider()` ของแบบฝึกหัดกับโมเดลที่ผ่าน ถ้าคุณตั้งค่าเป็นตัวที่ไม่ผ่าน จะเกิดอะไรขึ้นในแชตของ OpenClaw?

</details>

✓ Checkpoint: บรรทัดตรวจทั้งสามเป็น ✓

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| URL ของ dashboard OpenClaw โหลดไม่ขึ้น | รีสตาร์ต gateway เพื่อให้โหลด `~/.openclaw/openclaw.json` ใหม่; ตรวจว่ามันรันอยู่ด้วย `pgrep -f openclaw`; หา URL และโทเค็นในผลลัพธ์ของตัวติดตั้งหรือใน `~/.openclaw/logs/` |
| "Connection refused" ไปที่ `localhost:8000` | vLLM ไม่ได้รัน ยังโหลดอยู่ หรืออยู่คนละพอร์ต: `docker ps` แล้ว `curl http://localhost:8000/v1/models` |
| OpenClaw บอกว่า no model available | เพิ่ม provider ลงใน `openclaw.json`; `id`/`name` ต้องตรงกับ handle ที่ serve อยู่ทุกตัวอักษร |
| OpenClaw รายงานว่า endpoint ไม่รองรับ Responses API | เปลี่ยน `"api": "openai-responses"` เป็นแบบ chat-completions ที่ตรงกับเวอร์ชัน OpenClaw ของคุณ |
| การเปลี่ยน config ไม่มีผล | รีสตาร์ต OpenClaw gateway |
| `hermes: command not found` | `source ~/.bashrc`; ในสคริปต์ใช้ `export PATH="$HOME/.local/bin:$PATH"` หรือเรียก `~/.local/bin/hermes` |
| `sudo: hermes: command not found` | `sudo "$(which hermes)" …` (sudo รีเซ็ต PATH) |
| "Setup wizard skipped (no terminal available)" | รัน `hermes setup` ใหม่ในเทอร์มินัลแบบ interactive หรือใช้ทางสำรอง `hermes config set` (ส่วนที่ 6) |
| ตัวติดตั้ง Hermes ไม่แสดงโมเดลใดเลย | vLLM ยังไม่พร้อม: รอ `Application startup complete` ตรวจ `curl http://localhost:8000/v1/models` แล้วรันตัวติดตั้งใหม่ |
| ตัวติดตั้งถามเรื่อง import/migration จาก OpenClaw | ตอบ `n`; ถ้า migrate ไปแล้ว ให้ถอนการติดตั้ง `rm -rf ~/.hermes` แล้วติดตั้งใหม่ |
| เอเจนต์แชตได้แต่ไม่เคยใช้ tool | โมเดลหรือเซิร์ฟเวอร์ไม่ส่ง `tool_calls` กลับมา: รัน lab 17-2; บน Spark ให้เปิด vLLM ด้วย `--enable-auto-tool-choice --tool-call-parser qwen3_xml` (ส่วนที่ 2) |
| หน่วยความจำไม่พอหรือช้ามาก | ตรวจ `nvidia-smi`; ลด `--gpu-memory-utilization` / `--max-model-len` หรือใช้โมเดลที่เล็กลง; playbook ล้าง page cache ด้วย `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| บอต Telegram ไม่เคยตอบ | รันการตรวจ HTTPS ไปที่ `api.telegram.org` (ส่วนที่ 8); ตรวจ gateway unit ด้วย `systemctl` / `journalctl`; ยืนยันว่า user ID ของคุณอยู่ในรายการที่อนุญาต |

## Next — บทถัดไป

ไปต่อที่ [Lab 18 — coding agent บน inference ในเครื่อง](../18_coding_agents/TUTORIAL.md): ชี้ coding agent แบบ command-line ไปที่โมเดลบนเครื่องชุดเดียวกัน และดูว่า probe ข้อไหนในโมดูลนี้ที่ coding agent ต้องการมากที่สุด
