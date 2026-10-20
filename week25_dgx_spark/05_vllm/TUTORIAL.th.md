# ▶ Spark Lab 05 — vLLM: serving ที่ throughput สูง, tool calling และ tensor parallel บน Spark สองเครื่อง

> ส่วนหนึ่งของ Week 25 · DGX Spark: fine-tune, serve และสร้างเอเจนต์ที่รันใน sandbox คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง ทุกแล็บรันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (อ้างอิงคำต่อคำจาก playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- เข้าใจว่าทำไม vLLM จึงมีอยู่: PagedAttention, continuous batching และ API ที่เข้ากันได้กับ OpenAI
- เลือก container ของ vLLM ให้ถูกกับงาน (upstream `vllm/vllm-openai` หรือ NGC `nvcr.io/nvidia/vllm`) และเปิดเซิร์ฟเวอร์ตัวแรกที่พอร์ต 8000
- กำหนดขนาด flag หน่วยความจำสามตัวที่สำคัญบน unified memory 128 GB: `--gpu-memory-utilization`, `--max-model-len`, `--max-num-seqs`
- serve โมเดล Qwen3.6-35B-A3B แบบ **agent-ready** ตาม playbook และรัน tool calling ครบหนึ่งรอบ (round trip)
- วัด continuous batching: tok/s รวม เทียบกับ tok/s ต่อสาย (per-stream) ที่คำขอขนาน 1, 2, 4 และ 8 คำขอ
- รันโมเดล 70B ตัวเดียวข้าม **Spark สองเครื่อง** ด้วย Ray และ tensor parallelism จากนั้น (ไม่บังคับ) รัน **Nemotron 3 Super 120B** ของ NVIDIA แบบ FP8 และ serve LoRA adapter คู่กับ base model ของมัน

**Time** ~55 นาที · **Difficulty** ระดับกลาง · **Hardware** Spark 1 เครื่อง (2 เครื่องสำหรับส่วนที่ 6) หรือไม่มีเลยก็ได้: ใช้โหมด DRY + ตัวแทนบนแล็ปท็อป

**Playbook ทางการที่ครอบคลุม:** [vLLM](https://build.nvidia.com/spark/vllm) (ทั้ง `playbook-vllm` ฉบับปัจจุบัน และ `vllm` ฉบับเก่าซึ่งมีส่วน 405B บน Spark สองเครื่องและส่วน Qwen3.6 แบบ agent-ready)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| ทำ Module 01 เสร็จแล้ว | `ssh -o BatchMode=yes spark-a true` | แล็บส่งคำสั่งไปที่ Spark ผ่าน SSH |
| Docker โดยไม่ต้อง sudo บน Spark | `docker ps` บน Spark | ทุกเส้นทางของ vLLM ใน playbook เป็น container |
| ดิสก์ว่าง ~60 GB บน Spark | `df -h /` | สำหรับ container image และโมเดลหนึ่งถึงสองตัว |
| ล็อกอิน Hugging Face **บน Spark** | `hf auth login` บน Spark ครั้งเดียว | โมเดลที่ต้องขอสิทธิ์ (gated เช่น Llama) ดาวน์โหลดด้วยโทเค็นของ Spark เอง |
| ไม่บังคับ: Ollama บนแล็ปท็อป | `curl -s localhost:11434/v1/models` | ตัวแทนบนแล็ปท็อปสำหรับแล็บ HTTP |

```bash
# on: spark
docker ps
hf auth whoami
df -h / | tail -1
```

> 🔐 playbook ส่ง `-e HF_TOKEN="$HF_TOKEN"` ให้ container แต่คอร์สนี้ mount ทั้ง `~/.cache/huggingface` แทน ทำให้พบโทเค็นที่ `hf auth login` บันทึกไว้บน Spark โดยไม่ต้องพิมพ์ลงในบรรทัดคำสั่งเลย ใช้ได้ทั้งสองแบบ

✓ Checkpoint: `docker ps` รันได้โดยไม่ต้อง `sudo` บน Spark และ `hf auth whoami` พิมพ์ชื่อผู้ใช้ Hugging Face ของคุณ

## 1 · ทำไมต้อง vLLM: PagedAttention และ continuous batching

Ollama (Module 03) และ llama.cpp (Module 04) เหมาะกับการใช้งานคนเดียวที่โต๊ะ ส่วน vLLM สร้างมาเพื่อ **คำขอจำนวนมากพร้อมกัน** playbook ระบุแนวคิดไว้สามข้อ:

| แนวคิด | ทำอะไร | ทำไมสำคัญบน Spark |
|---|---|---|
| **PagedAttention** | เก็บ KV cache เป็นหน้า (page) เล็ก ๆ แบบเดียวกับ virtual memory แทนที่จะเป็นก้อนใหญ่ก้อนเดียวต่อคำขอ | ไม่เปลืองหน่วยความจำกับ context ที่คำขอไม่ได้ใช้ ผู้ใช้จึงใส่ใน 128 GB ได้มากขึ้น |
| **Continuous batching** | เพิ่มคำขอใหม่เข้าไปใน batch ที่กำลังรันอยู่ ทีละ token | การอ่าน weights จากหน่วยความจำครั้งเดียวให้บริการผู้ใช้ทุกคนใน batch (ส่วนที่ 5) |
| **API ที่เข้ากันได้กับ OpenAI** | `/v1/models`, `/v1/chat/completions`, `/v1/completions` | curl, `openai` SDK, LiteLLM (Module 08) และ NAT (Module 14) ใช้งานได้โดยไม่ต้องแก้ |

Module 01 แสดงให้เห็นแล้วว่าความเร็วของบทสนทนาหนึ่งสายถูกจำกัดด้วย memory bandwidth: ทุก token ต้องอ่าน weight ที่ทำงานอยู่ (active) ทุกตัวหนึ่งครั้ง continuous batching ไม่ได้ยกเพดานนั้นสำหรับผู้ใช้คนเดียว แต่แบ่งการอ่าน weights ครั้งเดียวกันให้ผู้ใช้หลายคน token ต่อวินาที **รวม** จึงเพิ่มขึ้นตามจำนวนผู้ใช้พร้อมกัน

```text
  requests ──►  ┌─────────── vLLM (one container, port 8000) ────────────┐
  (any client)  │  scheduler: joins new requests into the running batch  │
                │  ┌──────────────┐   ┌───────────────────────────────┐  │
                │  │ model weights│   │ KV cache pages (PagedAttention)│  │
                │  └──────────────┘   └───────────────────────────────┘  │
                │     both live inside --gpu-memory-utilization × 128 GB │
                └────────────────────────────────────────────────────────┘
```

✓ Checkpoint: คุณบอกได้ว่าแนวคิดข้อไหนในสามข้อทำให้ vLLM ให้บริการผู้ใช้ต่อ Spark ได้มากขึ้น และข้อไหนทำให้ OpenAI client ที่คุณมีอยู่คุยกับมันได้

## 2 · เลือก container และเปิดเซิร์ฟเวอร์ตัวแรก

playbook ใช้ image ของ vLLM หลายตัว และใช้แทนกันไม่ได้ ให้ใช้ตัวที่สูตรระบุ:

| Image (tag ตรงตาม playbook) | ใช้สำหรับ |
|---|---|
| `vllm/vllm-openai:latest` | การตั้งค่าพื้นฐานของ playbook ฉบับปัจจุบัน และสูตร Qwen3.6 แบบ agent-ready (Spark เครื่องเดียว) |
| `nvcr.io/nvidia/vllm:26.05-py3` | build ของ NVIDIA บน NGC ใช้กับ **Spark สองเครื่อง** (Ray + tensor parallel, ส่วนที่ 6) |
| `nvcr.io/nvidia/vllm:26.02-py3` | Spark สี่เครื่องขึ้นไปผ่าน QSFP switch (อย่าใช้ปนกับ tag สำหรับสองโหนด) |
| `vllm/vllm-openai:gemma4-cu130` | ตระกูล Gemma 4 |
| `vllm/vllm-openai:v0.20.0` · `vllm/vllm-openai:cu130-nightly` | Nemotron Nano · Nemotron Super (Module 06) |

> 💡 playbook ฉบับเก่าบอกให้หา build ล่าสุดของ NGC ที่ <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm> (ตัวอย่างของมันคือ `26.05.post1-py3`) ส่วน playbook ฉบับปัจจุบันชี้ไปที่ [vLLM Recipes for DGX Spark](https://recipes.vllm.ai/browse?panel=open&hw=dgx_spark_gb10) ซึ่งระบุ image และคำสั่งที่ผ่านการทดสอบแล้วสำหรับแต่ละโมเดล

นี่คือ **การตั้งค่าพื้นฐาน (base configuration)** ของ playbook โมเดลคือ `nvidia/Llama-3.1-8B-Instruct-FP8` จากตารางโมเดลที่รองรับ (support matrix) ของ playbook และ context เป็น 32K แทนที่จะเป็น 131072 ตาม playbook (ส่วนที่ 3 อธิบายการแลกเปลี่ยนนี้):

```bash
# on: spark
docker pull vllm/vllm-openai:latest
docker run -d \
  --name vllm-server \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --entrypoint "" \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai:latest \
  vllm serve nvidia/Llama-3.1-8B-Instruct-FP8 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.8
```

ดูมันเริ่มทำงาน ครั้งแรกจะดาวน์โหลดโมเดลก่อน แล้วจึงโหลดเข้าหน่วยความจำ:

```bash
# on: spark
docker logs -f vllm-server
```

**Expected output** (REFERENCE — ยกมาจาก playbook: บรรทัดที่หมายความว่า "พร้อมแล้ว")

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

หรือรอ health endpoint แบบที่ playbook ทำ แล้วส่งคำขอทดสอบของมัน:

```bash
# on: spark
timeout 900 bash -c 'until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 10; done' \
  || { echo "Server failed to start within 900s"; docker logs vllm-server | tail -50; exit 1; }
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "nvidia/Llama-3.1-8B-Instruct-FP8", "messages": [{"role": "user", "content": "12*17"}], "max_tokens": 500}'
```

playbook บอกว่าคำตอบควรมี `"content": "204"` หรือการคำนวณที่คล้ายกัน ลองจากตรงนี้ได้เลย บล็อกด้านล่างจะส่งไปที่ vLLM บน Spark ของคุณ หรือไปที่ตัวแทนบนแล็ปท็อปที่ติดป้ายไว้:

```spark
{"target": "vllm", "which": "a", "model": "nvidia/Llama-3.1-8B-Instruct-FP8",
 "messages": [{"role": "user", "content": "12*17"}], "max_tokens": 150}
```

Lab 05-2 ทำทั้งหมดนี้ในรอบเดียว มันจะเปิด container ก็ต่อเมื่อคุณใส่ `--yes`:

```bash
# on: laptop
.venv/bin/python week25/05_vllm/labs/lab02_first_server.py          # add --yes to really start it
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ในโหมด DRY: ยังไม่ได้ตั้งค่า Spark ขั้นที่ 4 จึงใช้ตัวแทนบนแล็ปท็อป)

```
▣ STEP 3 · wait for /health (model loading can take several minutes)
$ timeout 600 bash -c 'until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 10; done'; docker logs vllm-server 2>&1 | grep -E 'Application startup complete|Uvicorn running' | tail -2   [DRY]
◈ REFERENCE — expected output from the NVIDIA playbook (not your machine):
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

▣ STEP 4 · the OpenAI API: GET /v1/models, then POST /v1/chat/completions
◆ no vLLM answering at (no Spark host) — the chat below falls back to the laptop stand-in
→ POST http://localhost:11434/v1/chat/completions · model=nemotron-3-nano:latest · Ollama on THIS laptop (stand-in, not the Spark)
· ANSWER  The product of 12 and 17 is **204**.
◆ LAPTOP STAND-IN (not Spark numbers) · nemotron-3-nano:latest · TTFT 9446 ms · 78 tok in 10.6s · 69.7 tok/s
✓ the answer contains 204, as the playbook expects
```

> ⚠ `-p 8000:8000` listen บนทุก interface รวมถึง tailnet ของคุณ (Module 01 ส่วนที่ 3) vLLM ไม่มีรหัสผ่านโดยค่าเริ่มต้น Module 08 จะวาง LiteLLM gateway ที่มีคีย์ไว้ด้านหน้า

หยุดมันเมื่อใช้เสร็จ โมเดลที่ดาวน์โหลดมาแล้วยังอยู่ใน cache:

```bash
# on: spark
docker rm -f vllm-server 2>/dev/null || true
```

✓ Checkpoint: `curl http://localhost:8000/v1/models` บน Spark แสดงโมเดลของคุณ และคำขอ `12*17` ตอบกลับเป็น 204

## 3 · flag ที่สำคัญบน unified memory 128 GB

ตอนเริ่มต้น vLLM จะจองหน่วยความจำไว้ส่วนหนึ่งแบบคงที่ โหลด weights ลงไป แล้วเปลี่ยน **ทุกอย่างที่เหลือ** เป็นหน้า KV cache บน Spark ส่วนนั้นมาจาก 128 GB ก้อนเดียวกับที่ OS และโปรเซสอื่น ๆ ของคุณใช้ มีสาม flag ที่ตัดสินผลรวมนี้:

| Flag | ค่าใน playbook | ควบคุมอะไร |
|---|---|---|
| `--gpu-memory-utilization` | `0.8` (base) · `0.4` (Qwen3.6 แบบ agent-ready) · `0.90` (Nemotron Super) | ส่วนที่ vLLM ใช้ได้สำหรับ weights + KV cache |
| `--max-model-len` | `131072` (base) · `262144` (Qwen3.6) · `2048` (70B บน Spark สองเครื่อง) | prompt + output ที่ยาวที่สุดของคำขอหนึ่งรายการ |
| `--max-num-seqs` | `4` (Qwen3.6, Nemotron Super) · `8` (Nemotron Nano) · `1` (405B) | จำนวน sequence ที่รันใน batch เดียว |
| `--kv-cache-dtype fp8` | สูตรของ Qwen3.6 และ Nemotron Super | ลด KV cache ต่อ token ลงครึ่งหนึ่ง |

```text
KV room            = gpu-memory-utilization × 128 GB − weights − runtime (~4 GB, a course assumption)
KV per sequence    = 2 × layers × KV heads × head dim × max-model-len × bytes (2 for bf16, 1 for fp8)
full-length seqs   = KV room ÷ KV per sequence          → a safe --max-num-seqs
```

Lab 05-1 คำนวณผลรวมนี้ให้โมเดลจาก support matrix ของ playbook และพิมพ์คำสั่ง `docker run` ที่พร้อมใช้:

```bash
# on: laptop
.venv/bin/python week25/05_vllm/labs/lab01_launch_builder.py
```

**Expected output** (เป็นการคำนวณ ผลเหมือนกันทุกเครื่อง)

```
▣ STEP 1 · what fits in --gpu-memory-utilization 0.8 × 128 GB, --max-model-len 32768, --kv-cache-dtype auto
│ #  model (HF handle)                    format  weights   KV room   KV / seq  fits at full context
│ ─  ───────────────────────────────────  ──────  ────────  ────────  ────────  ────────────────────────────────
│ 1  nvidia/Llama-3.1-8B-Instruct-FP8     fp8       8.0 GB   90.4 GB   4.29 GB    21.0 full-length seqs
│ 2  nvidia/Qwen3-14B-NVFP4               nvfp4     8.3 GB   90.1 GB   5.37 GB    16.8 full-length seqs
│ 3  nvidia/Qwen3-32B-NVFP4               nvfp4    18.4 GB   80.0 GB   8.59 GB     9.3 full-length seqs
│ 4  nvidia/Llama-3.3-70B-Instruct-NVFP4  nvfp4    39.7 GB   58.7 GB  10.74 GB     5.5 full-length seqs
│ 5  meta-llama/Llama-3.3-70B-Instruct    bf16    141.2 GB    0.0 GB  10.74 GB  ✕ weights alone exceed the slice

▣ STEP 2 · the three levers — Qwen3-32B NVFP4 at 32K context
│ util 0.5 · kv auto    4.8 × 32K seqs  █████░░░░░░░░░░░░░░░░░░░░░░░
│ util 0.5 · kv fp8     9.7 × 32K seqs  █████████░░░░░░░░░░░░░░░░░░░
│ util 0.8 · kv auto    9.3 × 32K seqs  █████████░░░░░░░░░░░░░░░░░░░
│ util 0.8 · kv fp8    18.6 × 32K seqs  █████████████████░░░░░░░░░░░
│ util 0.9 · kv auto   10.8 × 32K seqs  ██████████░░░░░░░░░░░░░░░░░░
│ util 0.9 · kv fp8    21.6 × 32K seqs  ████████████████████░░░░░░░░
│ util 0.8 · kv auto · ctx   8K →   37.2 full-length seqs
│ util 0.8 · kv auto · ctx 128K →    2.3 full-length seqs
```

บทเรียนสี่ข้อ:

1. **context มีราคาแพง** Qwen3-32B ที่ 0.8 รองรับผู้ใช้ 37 คนที่ context 8K แต่รองรับแค่ 2 คนที่ 128K ค่าเริ่มต้น `MAX_MODEL_LEN=131072` ของ playbook ถือว่าใจกว้าง ให้กำหนดขนาดตามงานของคุณ อย่างที่ playbook บอก
2. **KV แบบ fp8 เพิ่มจำนวนผู้ใช้เป็นสองเท่า** ที่หน่วยความจำเท่าเดิม นี่คือเหตุผลที่สูตรเอเจนต์แบบ context ยาวทั้งสองสูตรใช้ `--kv-cache-dtype fp8`
3. **utilization คือหน่วยความจำที่ใช้ร่วมกัน** การเพิ่ม `--gpu-memory-utilization` เป็น 0.95 ตามที่ playbook แนะนำสำหรับ GPU *ที่ใช้เฉพาะงานนี้* จะไปดึงหน่วยความจำจาก OS บน Spark เมื่อ playbook ของ Nemotron เจอปัญหาหน่วยความจำไม่พอ วิธีแก้แรกของมันคือ *ลด* ค่านี้ลงเป็น 0.70
4. **โมเดล 70B แบบ bf16 ใส่ใน Spark เครื่องเดียวไม่ได้เลย** ให้ใช้ checkpoint แบบ NVFP4 (แถวที่ 4) หรือแบ่งข้าม Spark สองเครื่อง (ส่วนที่ 6)

> 💡 ตัวเลขเหล่านี้เป็นค่าประมาณ checkpoint แบบ NVFP4 และ FP8 เก็บบาง layer ไว้ที่ความละเอียดสูงกว่า และ log ตอนเริ่มต้นของ vLLM เองจะรายงานขนาด KV cache จริงและจำนวนผู้ใช้พร้อมกันสูงสุด เชื่อ log มากกว่าแล็บ เปลี่ยนค่าที่ป้อนได้ด้วย `--pick 4 --ctx 131072 --kv fp8`

✓ Checkpoint: คุณใช้ lab 05-1 เลือก `--max-model-len` และ `--max-num-seqs` สำหรับ Llama 3.3 70B NVFP4 ที่ใส่ได้ที่ `--gpu-memory-utilization 0.8`

## 4 · Agent-ready: Qwen3.6-35B-A3B พร้อม tool calling

เอเจนต์ต้องการสามอย่างจาก model server: ต้องส่ง **tool call** กลับมาในฟิลด์ `tool_calls` แบบ OpenAI ต้องแยก **การให้เหตุผล (reasoning)** ออกจากคำตอบ และต้องรองรับ context แบบ **หลายรอบ (multi-turn) ที่ยาว** โมเดล agent-ready ที่ playbook แนะนำสำหรับ DGX Spark คือ `nvidia/Qwen3.6-35B-A3B-NVFP4`: พารามิเตอร์ 35B ทำงานอยู่ราว 3B ต่อ token (เป็น Mixture-of-Experts ซึ่งเป็นจุดที่เหมาะที่สุดของ Spark ตาม Module 01)

**flag เหล่านี้มาจากไหน (โมดูลนี้เป็นแหล่งอ้างอิงหลักของคอร์สสำหรับ flag ชุดนี้):**

| แหล่งที่มา | ให้อะไร |
|---|---|
| playbook ฉบับเก่า `nvidia/vllm/README.md` หัวข้อ "Run Agent Ready Qwen3.6 35B Model with vLLM" | **ทุก flag ในคำสั่งด้านล่าง** image `vllm/vllm-openai:latest` และการทดสอบ `12*17` |
| playbook ฉบับปัจจุบัน `nvidia/playbook-vllm/README.md` แท็บ "Agent-ready Models" | แค่ model handle `nvidia/Qwen3.6-35B-A3B-NVFP4` และลิงก์ไปยัง [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B?hardware=dgx_spark_gb10&features=tool_calling%2Creasoning) — ไม่มี flag |
| recipes.vllm.ai | **ไม่มีอะไรในคอร์สนี้** ตอนที่เขียนโมดูลนี้ (2026-09-29) หน้า recipe ไม่ได้แสดงคำสั่งสำหรับ DGX Spark ที่ยกมาอ้างอิงได้ ให้เทียบ recipe กับคำสั่งด้านล่างก่อนจะพึ่งตัวใดตัวหนึ่ง |

ดังนั้น flag สำหรับเอเจนต์สามตัว ได้แก่ `--reasoning-parser qwen3`, `--tool-call-parser qwen3_xml` และ `--enable-auto-tool-choice` จึงยกมาจาก playbook ฉบับเก่า โมดูลหลังจากนี้ (08, 14, 17, 20) นำไปใช้ต่อจากที่นี่

นี่คือคำสั่งเปิดเซิร์ฟเวอร์ของ playbook นั้นโดยไม่แก้ไข entrypoint ของ image `vllm/vllm-openai` เป็น `vllm serve` อยู่แล้ว model handle และ flag จึงส่งให้ container ได้โดยตรง:

```bash
# on: spark
docker pull vllm/vllm-openai:latest
export HF_TOKEN="your_huggingface_token"      # or rely on `hf auth login` and drop the -e line
docker run -it --gpus all -p 8000:8000 \
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

แบ่ง flag ออกเป็นสามกลุ่ม:

| กลุ่ม | Flag | ทำไม |
|---|---|---|
| หน่วยความจำ | `--gpu-memory-utilization 0.4` · `--max-model-len 262144` · `--max-num-seqs 4` · `--kv-cache-dtype fp8` | context ยาวสำหรับเอเจนต์ 4 ตัวพร้อมกัน playbook ไม่ได้อธิบายค่า 0.4 ผลอย่างหนึ่งคือ Spark ยังเหลือหน่วยความจำว่าง ~60% สำหรับงานอื่น |
| ความเร็ว | `--enable-chunked-prefill` · `--async-scheduling` · `--enable-prefix-caching` · `--speculative-config` (MTP, Module 07) · `--moe-backend marlin` · `--attention-backend flashinfer` | prompt ยาว ๆ ของเอเจนต์ใช้ส่วนต้น (prefix) ร่วมกันทุกรอบ prefix caching จึงช่วยประหยัดงาน |
| เอเจนต์ | `--reasoning-parser qwen3` · `--tool-call-parser qwen3_xml` · `--enable-auto-tool-choice` | แปลงไวยากรณ์เฉพาะของโมเดลให้เป็นฟิลด์ `reasoning` และ `tool_calls` แบบ OpenAI |

**parser ต้องตรงกับตระกูลโมเดล** จาก playbook: Qwen3.6 → `qwen3_xml`; Nemotron Nano และ Super → `qwen3_coder`; Gemma 4 → `gemma4` (คู่กับ `--reasoning-parser gemma4`) parser ที่ผิดไม่ได้ทำให้เซิร์ฟเวอร์ล่ม: tool call จะกลับมาเป็นข้อความธรรมดาใน `content` และเอเจนต์ของคุณจะไม่เคยรัน tool เลยโดยไม่มีอะไรแจ้งเตือน

> ⚠ reasoning model ใช้ `max_tokens` ส่วนหนึ่งไปกับการคิดก่อนตอบ การทดสอบ API ของ playbook ใช้ `max_tokens: 4096` ถ้างบน้อย คุณอาจเห็น `finish_reason: length` ข้อความการคิดอยู่ใต้ `reasoning` และ `content: null`

ลองเรียก tool จากตรงนี้ Lab Runner จะส่ง list `tools` ไปพร้อมกับ messages:

```spark
{"target": "vllm", "which": "a", "model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
 "messages": [{"role": "user", "content": "Guest in room 1204 says it feels warm. What is the temperature there right now?"}],
 "max_tokens": 150,
 "tools": [{"type": "function", "function": {"name": "get_room_temperature",
   "description": "Current air temperature of a hotel room, in degrees Celsius.",
   "parameters": {"type": "object", "properties": {"room": {"type": "string"}}, "required": ["room"]}}}]}
```

Lab 05-4 รันวงรอบเอเจนต์ครบหนึ่งรอบ: โมเดลขอใช้ tool แล็บรัน tool นั้น (เซนเซอร์ปลอมที่รันบนเครื่องอย่างชัดเจน) ส่งผลลัพธ์กลับเป็นข้อความ `role: tool` แล้วโมเดลก็ตอบ:

```bash
# on: laptop
.venv/bin/python week25/05_vllm/labs/lab04_tool_calling.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้: ไม่มี Spark จึงใช้โมเดลบนแล็ปท็อปที่รองรับ tool เป็นตัวแทน)

```
▣ STEP 2 · turn 1 — the model decides to call a tool
→ POST http://localhost:11434/v1/chat/completions · model=gemma4:12b · Ollama on THIS laptop (stand-in, not the Spark)
· ANSWER
→ tool_call get_room_temperature({"room":"1204"})
◆ LAPTOP STAND-IN (not Spark numbers) · gemma4:12b · TTFT 13340 ms · 20 tok in 13.3s · 1.5 tok/s
✓ 1 tool call(s) — parsed by the server into OpenAI `tool_calls`, not by this script

▣ STEP 3 · this script runs the tool locally and sends the result back as role=tool
│ get_room_temperature({"room":"1204"}) → {"room": "1204", "celsius": 26.5, "source": "fake sensor in lab04"}

▣ STEP 4 · turn 2 — the model answers from the tool result
→ POST http://localhost:11434/v1/chat/completions · model=gemma4:12b · Ollama on THIS laptop (stand-in, not the Spark)
· ANSWER  The current temperature in room 1204 is 26.5°C.
◆ LAPTOP STAND-IN (not Spark numbers) · gemma4:12b · TTFT 27577 ms · 24 tok in 27.6s · 0.9 tok/s
✓ the final answer quotes the tool's 26.5 °C — grounded in the tool result, not guessed
```

ตัวเลขที่ช้าของแล็ปท็อปรวมเวลาโหลดโมเดลและโปรแกรมอื่นที่ใช้แล็ปท็อปร่วมกันอยู่ด้วย มันไม่ได้บอกอะไรเกี่ยวกับ Spark เลย

✓ Checkpoint: คุณบอกชื่อ flag สามตัวที่ทำให้ vLLM ส่ง `tool_calls` กลับมาได้ และ lab 05-4 พิมพ์บรรทัด `→ tool_call` และคำตอบสุดท้ายที่อ้าง 26.5 °C

## 5 · วัด continuous batching

Lab 05-3 ส่งคำขอแบบเดียวกันทีละ 1, 2, 4 และ 8 คำขอพร้อมกัน แล้ววัดตัวเลขสองตัว:

- **tok/s รวม (total)**: token ทั้งหมดที่สร้าง ÷ เวลาจริงที่ผ่านไป บอกว่าเครื่องหนึ่งเครื่องให้บริการผู้ใช้ได้กี่คน
- **tok/s ต่อสาย (per-stream)**: ความเร็ว decode ของผู้ใช้หนึ่งคน บอกว่าตัวอักษรปรากฏในหน้าแชต *ของเขา* เร็วแค่ไหน

```bash
# on: laptop
.venv/bin/python week25/05_vllm/labs/lab03_continuous_batching.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้กับ Ollama: LAPTOP STAND-IN ไม่ใช่ Spark และไม่ใช่ vLLM)

```
◆ endpoint: LAPTOP STAND-IN · Ollama on this Mac (not Spark numbers) · model gemma3:4b · max_tokens 64

▣ STEP 3 · total vs per-stream throughput
│ parallel  ok   total tok/s  per-stream tok/s  mean TTFT   vs 1   total
│ ────────  ───  ───────────  ────────────────  ──────────  ─────  ────────────────────
│ 1         1/1    90.4         95.4                 37 ms  ×1.00  ████████████████████
│ 2         2/2    80.9         88.4                401 ms  ×0.90  ██████████████████░░
│ 4         4/4    41.4         46.6               2695 ms  ×0.46  █████████░░░░░░░░░░░
│ 8         8/8    47.8         51.7               4661 ms  ×0.53  ███████████░░░░░░░░░
```

อ่านอย่างตรงไปตรงมา บนแล็ปท็อปเครื่องนี้ค่ารวม **ไม่ได้** เพิ่มขึ้น: ตั้งแต่ 2 คำขอขึ้นไป ผู้ใช้แต่ละคนช้าลง และเวลาจนถึง token แรก (TTFT) ไต่จาก 37 ms ไปถึง 4.7 วินาที คำขอต้องรอคิวแทนที่จะอยู่ใน batch เดียวกัน ตอนนั้นมีโปรแกรมอื่นใช้ Ollama ตัวเดียวกันอยู่ด้วย และการรันซ้ำให้ตัวเลขต่างออกไป แล็บจึงเตือนคุณเมื่อคำขอเดี่ยวเองก็ต้องรอคิว

นั่นคือความต่างที่แล็บนี้ต้องการแสดง ให้รันกับ vLLM บน Spark ของคุณ (ทำ lab 05-2 ก่อน) แล้วเทียบ **รูปร่าง** ของกราฟคุณกับกราฟนี้: เมื่อมี continuous batching คอลัมน์ค่ารวมควรไต่ขึ้นเรื่อย ๆ ขณะที่ความเร็วต่อสายลดลงเพียงช้า ๆ จนกว่า batch จะใช้ compute หรือ KV cache หมด (`--max-num-seqs`) อย่าเทียบตัวเลขของแล็ปท็อปกับของ Spark เด็ดขาด

✓ Checkpoint: คุณรัน lab 05-3 แล้ว และอธิบายจากตารางของคุณเองได้ว่าเซิร์ฟเวอร์รวมคำขอเป็น batch หรือให้รอคิว

## 6 · โมเดลเดียวข้าม Spark สองเครื่อง: Ray + tensor parallel

Llama 3.3 70B แบบ bf16 ต้องใช้ weights 141 GB มากกว่าที่ Spark เครื่องเดียวมี **Tensor parallelism** (TP) แบ่งทุก layer ข้าม GPU: เมื่อใช้ `--tensor-parallel-size 2` Spark แต่ละเครื่องจะถือ weights ครึ่งหนึ่งและ KV head ครึ่งหนึ่ง และทั้งสองแลกเปลี่ยนผลลัพธ์บางส่วนกันผ่านลิงก์ QSFP ในทุก layer ของทุก token vLLM รันสองส่วนนี้เป็นคลัสเตอร์ **Ray** เดียว

```text
  Spark A (head)                      QSFP 200 Gb/s                  Spark B (worker)
 ┌───────────────────────────┐   all-reduce every layer    ┌───────────────────────────┐
 │ container node-NNNN       │◄───────────────────────────►│ container node-NNNN       │
 │ ray head · vllm serve     │                             │ ray worker                │
 │ half of every layer (TP 0)│                             │ half of every layer (TP 1)│
 │ API :8000 · Ray UI :8265  │                             │                           │
 └───────────────────────────┘                             └───────────────────────────┘
```

ขั้นที่ 3 ของ Lab 05-1 คำนวณหน่วยความจำสำหรับการตั้งค่าของ playbook:

```
▣ STEP 3 · two Sparks, tensor parallel 2 — the playbook's Llama 3.3 70B (bf16)
│ setup           weights           vLLM slice         at --max-model-len 2048
│ ──────────────  ────────────────  ─────────────────  ───────────────────────
│ 1 Spark         141.2 GB          102.4 GB           ✕ does not fit
│ 2 Sparks, TP=2  70.6 GB per node  102.4 GB per node  83 × 2K seqs
│ TP=2 · ctx  8K →  20.7 full-length seqs
│ TP=2 · ctx 32K →   5.2 full-length seqs
│ nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 · ctx 256K · kv fp8
│   1 Spark         120.0 GB          ✕ does not fit
│   2 Sparks, TP=2   60.0 GB per node  0.54 GB KV per 256K seq → 71.5 full-length seqs
```

สามบรรทัดสุดท้ายใช้กับการรัน Nemotron (ไม่บังคับ) ท้ายส่วนนี้

ทำ Module 02 ให้เสร็จก่อน (สาย QSFP, IP, SSH แบบไม่ใช้รหัสผ่านระหว่าง Spark) หรือรัน Cluster Assistant ของ NVIDIA Sync ซึ่ง playbook ยอมรับให้ใช้แทนได้ จากนั้นทำตาม playbook บน **Spark ทั้งสองเครื่อง**

**ขั้นที่ 1 — สคริปต์คลัสเตอร์และ NGC image** (บน Spark ทั้งสองเครื่อง) สคริปต์ถูกตรึง (pin) ไว้กับ commit ของ vLLM ที่รู้ว่าใช้ได้ และถูกแก้ให้ติดตั้ง Ray ภายใน container เพราะ image `26.05-py3` ไม่มี Ray มาให้:

```bash
# on: spark
wget https://raw.githubusercontent.com/vllm-project/vllm/51c1ee9b7c8acbba4899a8ebffd390685d171946/examples/ray_serving/run_cluster.sh
sed -i 's|^RAY_START_CMD="ray start|RAY_START_CMD="pip install -q --root-user-action=ignore '\''ray[default]>=2.9'\'' \&\& ray start|' run_cluster.sh
chmod +x run_cluster.sh
docker pull nvcr.io/nvidia/vllm:26.05-py3
```

ทำสี่คำสั่งเดียวกันซ้ำบน Spark B (`ssh spark-b`)

**ขั้นที่ 2 — Ray head บน Spark A** รันภายใน `tmux`: `run_cluster.sh` จะหยุด container เมื่อเทอร์มินัลของมันปิด `enp1s0f1np1` คือ interface QSFP ตัวอย่างของ playbook ให้ใช้ตัวที่ขึ้น `(Up)` ใน `ibdev2netdev` (Module 02):

```bash
# on: spark
tmux new -s ray
export MN_IF_NAME=enp1s0f1np1
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
export VLLM_IMAGE=nvcr.io/nvidia/vllm:26.05-py3
echo "Using interface $MN_IF_NAME with IP $VLLM_HOST_IP"
bash run_cluster.sh $VLLM_IMAGE $VLLM_HOST_IP --head ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR=$VLLM_HOST_IP
```

**ขั้นที่ 3 — Ray worker บน Spark B** แทน `<NODE_1_IP_ADDRESS>` ด้วย IP ที่ Spark A เพิ่งพิมพ์ออกมา:

```bash
# on: spark-b
tmux new -s ray
export MN_IF_NAME=enp1s0f1np1
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
export HEAD_NODE_IP=<NODE_1_IP_ADDRESS>
export VLLM_IMAGE=nvcr.io/nvidia/vllm:26.05-py3
echo "Worker IP: $VLLM_HOST_IP, connecting to head node at: $HEAD_NODE_IP"
bash run_cluster.sh $VLLM_IMAGE $HEAD_NODE_IP --worker ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR=$HEAD_NODE_IP
```

**ขั้นที่ 4 — ตรวจคลัสเตอร์ ดาวน์โหลด แล้ว serve** (บน Spark A ในเทอร์มินัลที่สอง) playbook บอกว่า `ray status` ควรแสดง 2 โหนดที่มีทรัพยากร GPU Llama 3.3 70B เป็นโมเดลที่ต้องขอสิทธิ์: ยอมรับ license บน Hugging Face ก่อน:

```bash
# on: spark
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
echo "Found container: $VLLM_CONTAINER"
docker exec $VLLM_CONTAINER ray status
docker exec -it $VLLM_CONTAINER /bin/bash -c '
  hf auth login
  hf download meta-llama/Llama-3.3-70B-Instruct'
docker exec -it $VLLM_CONTAINER /bin/bash -c '
  vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 2 --max-model-len 2048 \
    --gpu-memory-utilization 0.8 --distributed-executor-backend ray'
```

(`--gpu-memory-utilization 0.8` มาจาก playbook ฉบับเก่า ส่วนฉบับปัจจุบันปล่อยไว้เป็นค่าเริ่มต้น) เมื่อ log ขึ้นว่า `Application startup complete.` ให้ทดสอบบน Spark A ด้วยคำขอของ playbook:

```bash
# on: spark
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.3-70B-Instruct", "prompt": "Write a haiku about a GPU", "max_tokens": 32, "temperature": 0.7}'
```

Ray dashboard รันอยู่ที่พอร์ต 8265 ของ Spark A: `ssh -N -L 8265:localhost:8265 spark-a` แล้วเปิด `http://localhost:8265`

| อยากไปให้ใหญ่กว่านี้? | จาก playbook |
|---|---|
| Llama 3.1 405B บน Spark สองเครื่อง | `hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4` พร้อม `--max-model-len 64 --gpu-memory-utilization 0.9 --max-num-seqs 1 --max-num-batched-tokens 64` playbook เตือนว่ามัน "has insufficient memory headroom for production use" |
| Spark สี่เครื่องขึ้นไปผ่าน switch | image `nvcr.io/nvidia/vllm:26.02-py3`, `run_cluster.sh` แบบไม่ตรึง commit จาก branch main ของ vLLM, `MiniMaxAI/MiniMax-M2.5` พร้อม `--tensor-parallel-size 4` (= จำนวนโหนด) |

> 💡 Spark สองเครื่องแบบ TP เพิ่มหน่วยความจำ ไม่ได้เพิ่มความเร็วต่อสาย ทุก token ต้องรอ all-reduce ผ่านลิงก์ วัด tok/s ของคุณเองด้วย lab 05-3 แล้วเทียบกับแถวที่ 4 ของ lab 05-1: 70B แบบ NVFP4 ใส่ใน Spark **เครื่องเดียว** ได้ Module 07 จะให้คุณ quantize โมเดลเอง

### ไม่บังคับ: Nemotron 3 Super 120B แบบ FP8 ข้าม Spark ทั้งสองเครื่อง

> ⚠ **ส่วนที่คอร์สเพิ่มเอง รันจริงบน Spark สองเครื่องเมื่อ 2026-10-03** ไม่มี playbook ของ NVIDIA สำหรับ Spark ที่ serve checkpoint นี้บน Spark สองเครื่อง flag มาจาก [model card ของ FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8) (เขียนไว้สำหรับ H100 4 ตัว) และแก้ตามที่ตารางด้านล่างบอก output ทั้งหมดในส่วนนี้บันทึกจาก Spark สองเครื่องของเราด้วย `nvcr.io/nvidia/vllm:26.05-py3` (vLLM 0.20.1)

Module 06 serve Nemotron 3 Super แบบ **NVFP4** (80 GB) บน Spark เครื่องเดียว checkpoint แบบ **FP8** มีขนาด 128.4 GB ใหญ่เกิน Spark เครื่องเดียว แต่สบาย ๆ บนสองเครื่อง: weight 57.57 GiB ต่อโหนด FP8 เก็บความแม่นยำได้มากกว่า NVFP4

มันยังเป็นโมเดลอีกแบบหนึ่งด้วย Nemotron 3 Super เป็น hybrid ของ Mamba-2/MoE: จาก 88 ชั้น มีเพียง **8 ชั้นที่เป็น attention** และมีแค่ชั้นเหล่านั้นที่เก็บ KV cache หนึ่ง token ใช้ KV 2 × 8 ชั้น × 2 KV heads × 128 × 1 byte (fp8) = **4 KB** ส่วน Llama 3.3 70B แบบ bf16 ใช้ 2 × 80 × 8 × 128 × 2 = 328 KB ชั้น Mamba 40 ชั้นเก็บ state ขนาดคงที่ต่อ sequence แทน

| Model card (H100 4 ตัว) | Spark สองเครื่อง | เหตุผล |
|---|---|---|
| `--tensor-parallel-size 4` | `--tensor-parallel-size 2` | GPU หนึ่งตัวต่อ Spark |
| เครื่องเดียว backend ค่าเริ่มต้น | `--distributed-executor-backend ray` | คลัสเตอร์ Ray จากขั้นที่ 1–3 |
| `--async-scheduling` | เอาออก | vLLM 0.20.1 ไม่ยอมเมื่อใช้ Ray: `` `ray` does not support async scheduling yet `` |
| `--swap-space 0` | เอาออก | flag นี้ไม่มีแล้วใน vLLM 0.20.1: `unrecognized arguments: --swap-space 0` |
| `--gpu-memory-utilization 0.9` | `0.8` | ค่าเดียวกับ 70B ด้านบน บน Spark หน่วยความจำ GPU คือหน่วยความจำของระบบ |
| `--served-model-name nvidia/nemotron-3-super` | `nemotron-3-super` | ชื่อที่ Module 06 ใช้ บล็อก ⚡ เดิมจึงใช้ได้ |

**ขั้นที่ 5 — weight บน Spark ทั้งสองเครื่อง** Spark แต่ละเครื่องโหลดครึ่งของตัวเองจาก `~/.cache/huggingface` **ของเครื่องนั้นเอง** ทั้งสองเครื่องจึงต้องมีครบ 128 GB ดาวน์โหลดบน Spark A:

```bash
# on: spark
hf download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8
```

จากนั้นจะรันคำสั่งเดียวกันบน Spark B ก็ได้ หรือคัดลอกผ่านลิงก์ QSFP ซึ่งเร็วกว่าดาวน์โหลดรอบสองมาก ให้คัดลอกทั้งโฟลเดอร์ `hub` ไม่ใช่แค่โฟลเดอร์ของโมเดล: `huggingface_hub` รุ่นใหม่เก็บข้อมูลไว้ในที่เก็บรวม `hub/blobs/` และโฟลเดอร์ของโมเดลมีแค่ลิงก์ชี้เข้าไป `rsync` จะข้ามไฟล์ที่ Spark B มีอยู่แล้ว:

```bash
# on: spark
rsync -a --info=progress2 ~/.cache/huggingface/hub/ <SPARK_B_QSFP_IP>:.cache/huggingface/hub/
```

**ขั้นที่ 6 — serve** ให้ Ray head และ worker จากขั้นที่ 2–3 รันต่อไป (`ray status`: 2 โหนด 2 GPU) หยุดเซิร์ฟเวอร์ 70B ก่อนถ้ายังรันอยู่ `--enable-expert-parallel` กระจาย MoE expert ไปบน GPU สองตัวแทนการหั่นแต่ละ expert ส่วน `HF_HUB_OFFLINE=1` ทำให้ทั้งสองโหนดโหลดจาก cache ในเครื่อง:

```bash
# on: spark
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec -e HF_HUB_OFFLINE=1 $VLLM_CONTAINER vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \
  --served-model-name nemotron-3-super \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --distributed-executor-backend ray \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --trust-remote-code \
  --gpu-memory-utilization 0.8 \
  --max-cudagraph-capture-size 128 \
  --enable-chunked-prefill \
  --mamba-ssm-cache-dtype float32 \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

**Expected output** (RECORDED — Spark A + Spark B, 2026-10-03; ตัด prefix และเวลาของ log ออก; `ip=192.168.100.96` คือครึ่งของ Spark B)

```
(RayWorkerWrapper pid=406, ip=192.168.100.96) Model loading took 57.57 GiB memory and 283.078022 seconds
(RayWorkerWrapper pid=2786) Model loading took 57.57 GiB memory and 284.023674 seconds
GPU KV cache size: 15,948,274 tokens
Maximum concurrency for 262,144 tokens per request: 60.84x
init engine (profile, create kv cache, warmup model) took 62.15 s (compilation: 14.25 s)
(APIServer pid=1608) INFO:     Application startup complete.
```

ใช้เวลาราว 6 นาทีจากเริ่มจนพร้อม ส่วนใหญ่คือการอ่าน 120 GB จากดิสก์ KV cache 15.9 ล้าน token พอสำหรับ **บทสนทนายาว 262K token 60 รายการพร้อมกัน** การคำนวณของ Lab 05-1 ได้ 71.5: ค่าประมาณ weight 60 GB ต่ำกว่า 61.8 GB ที่ vLLM โหลดจริง และไม่ได้นับ state ของ Mamba เชื่อ log

**ขั้นที่ 7 — คุยกับมัน** นี่คือ reasoning model: มันคิดก่อนตอบ จึงต้องให้ `max_tokens` มาก ๆ เมื่อให้ 400 มันใช้ทุก token ไปกับการคิดและคืน `content: null`:

```bash
# on: spark
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "nemotron-3-super", "messages": [{"role": "user", "content": "Write a haiku about a GPU"}], "max_tokens": 2000, "temperature": 0.7}' \
  | python3 -c 'import sys,json; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"]); print(r["usage"])'
```

**Expected output** (RECORDED — Spark A + Spark B, 2026-10-03; haiku ของคุณจะต่างไปที่ temperature 0.7)

```
Silicon heart beats
Pixels dance in parallel
Dreams render fast now
{'prompt_tokens': 23, 'total_tokens': 454, 'completion_tokens': 431, 'prompt_tokens_details': None}
```

```spark
{"target": "vllm", "which": "a", "model": "nemotron-3-super",
 "messages": [{"role": "user", "content": "In three sentences: why does tensor parallelism need a fast link between the GPUs?"}], "max_tokens": 2000}
```

tool calling ใช้คำขอแบบเดียวกับส่วนที่ 4 ได้ เมื่อถามว่า "What's the weather in Bangkok right now?" พร้อม tool `get_weather` มันคืน `get_weather({"city": "Bangkok"})` พร้อม `finish_reason: tool_calls` ใน 2.5 วินาที ผลที่เราวัดจากฝั่ง Spark B ผ่าน LAN:

| คำขอพร้อมกัน | tok/s รวม | tok/s ต่อสาย |
|---|---|---|
| 1 (haiku เปิดการคิด: 740 token ใน 40.9 วินาที) | 18.1 | 18.1 |
| 1 (ปิดการคิด) | 16.6 | 16.6 |
| 4 (ปิดการคิด) | 34.6 | 9.8 |

(RECORDED — 2026-10-03 "ปิดการคิด" คือส่ง `"chat_template_kwargs": {"enable_thinking": false}`) ผู้ใช้สี่คนได้ throughput รวมเป็นสองเท่าของคนเดียว: continuous batching (ส่วนที่ 5) ทำงานข้าม Spark ทั้งสองเครื่อง แต่ละสายช้าลง เพราะทุก token ต้องรอ all-reduce ผ่านลิงก์

> 💡 vLLM ฟังทุก interface แล็ปท็อปใน tailnet ของคุณจึงเรียกได้ที่ `http://<ชื่อ tailnet ของ spark-a>:8000/v1` ด้วยโมเดล `nemotron-3-super` และ `/docs` เปิดหน้าสำรวจ API ในเบราว์เซอร์ได้ ไม่มีรหัสผ่าน: Module 08 วาง LiteLLM gateway ที่มี key ไว้ข้างหน้า

✓ Checkpoint: `docker exec $VLLM_CONTAINER ray status` แสดง 2 โหนด และคำขอ haiku ได้ข้อความตอบกลับจากโมเดล 70B ที่ serve ข้าม Spark ทั้งสองเครื่อง (ไม่บังคับ: ได้จาก `nemotron-3-super` ด้วย โดยมี `Model loading took 57.57 GiB` ใน log ของทั้งสองโหนด)

## 7 · serve LoRA fine-tune คู่กับ base model

Module 09–11 fine-tune โมเดลด้วย **LoRA**: adapter ขนาดเล็ก (หลักสิบ MB) ที่ฝึกซ้อนบน base model ที่ถูกแช่แข็งไว้ vLLM serve ได้ทั้ง base model **และ** adapter หนึ่งตัวหรือมากกว่าจากเซิร์ฟเวอร์เดียว โดย adapter แต่ละตัวจะปรากฏเป็น model id ของตัวเองใน `/v1/models` คุณจึง serve fine-tune ได้โดยไม่ต้อง merge

> ⚠ **ส่วนที่คอร์สเพิ่มเข้ามา** vLLM playbook ของ NVIDIA ไม่ได้ครอบคลุม LoRA flag ด้านล่างเป็นของ vLLM เอง (`--enable-lora`, `--lora-modules`, `--max-lora-rank`) ยืนยันบน Spark ของคุณก่อนจะพึ่งมัน เพราะ flag เปลี่ยนไปตามเวอร์ชันของ vLLM:

```bash
# on: spark
docker run --rm --entrypoint vllm vllm/vllm-openai:latest serve --help=all 2>/dev/null | grep -i -- '--.*lora' \
  || docker run --rm --entrypoint vllm vllm/vllm-openai:latest serve --help | grep -i -- '--.*lora'
```

คำสั่งด้านล่าง serve `Qwen/Qwen3-8B` (base model ที่ไม่ต้องขอสิทธิ์ ซึ่ง SGLang playbook ใช้เป็นค่าเริ่มต้น) พร้อม adapter ที่บันทึกไว้ใน `~/w25/adapters/hotel-ft` บน Spark ให้แทน base model ด้วย `base_model_name_or_path` จาก `adapter_config.json` ของ adapter คุณ และตั้ง `--max-lora-rank` อย่างน้อยเท่ากับ `r` ของ adapter:

```bash
# on: spark
docker run -d \
  --name vllm-server \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/w25/adapters:/adapters" \
  --entrypoint '' \
  vllm/vllm-openai:latest \
  vllm serve Qwen/Qwen3-8B \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.5 \
    --max-num-seqs 9 \
    --enable-lora \
    --lora-modules hotel-ft=/adapters/hotel-ft \
    --max-lora-rank 16
curl -s http://localhost:8000/v1/models | python3 -m json.tool | grep '"id"'
```

**Expected output** (EXAMPLE — รูปแบบเพื่อประกอบคำอธิบาย ไม่ใช่ค่าที่วัดจริง: id หนึ่งตัวสำหรับ base และอีกตัวสำหรับ adapter)

```
            "id": "Qwen/Qwen3-8B",
            "id": "hotel-ft",
```

client เลือก adapter ด้วย `"model": "hotel-ft"` และเลือก base ด้วย `"model": "Qwen/Qwen3-8B"` LiteLLM gateway ของ Module 08 ส่งต่อไปยังตัวไหนก็ได้ และ Module 13 ประเมิน adapter เทียบกับ base ก่อนที่คุณจะตัดสินใจ merge คุณจะสร้างคำสั่งนี้เองในแบบฝึกหัดด้านล่าง

✓ Checkpoint: คุณอธิบายได้ว่าทำไม `--max-lora-rank` ต้องไม่น้อยกว่า rank ของ adapter และ `--lora-modules` ต้องใช้ path ไหน (path **ภายใน** container)

## Labs — รันแล็บได้ที่นี่

**labs/lab01_launch_builder.py** — กำหนดขนาด flag หน่วยความจำของ vLLM ด้วยการคำนวณ (Spark หนึ่งและสองเครื่อง) แล้วพิมพ์คำสั่ง `docker run` ที่พร้อมใช้

**labs/lab02_first_server.py** — เปิดเซิร์ฟเวอร์พื้นฐานของ playbook (เฉพาะเมื่อใส่ `--yes`) รอ `/health` แล้วส่งการทดสอบ `12*17`

**labs/lab03_continuous_batching.py** — ส่งคำขอขนาน 1, 2, 4 และ 8 คำขอ แล้วเทียบ tok/s รวมกับ tok/s ต่อสาย

**labs/lab04_tool_calling.py** — tool calling ครบหนึ่งรอบผ่าน OpenAI API จบด้วยคำตอบที่อิงผลจาก tool

Lab 01 เป็นการคำนวณและรันได้ทุกที่ Lab 02 สั่งงาน Spark ผ่าน SSH (เป็น DRY ถ้าไม่มี Spark) ส่วน Lab 03 และ 04 เรียก vLLM บน Spark หรือตัวแทนบนแล็ปท็อปที่ติดป้ายไว้

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 05 — เขียนคำสั่งเปิด vLLM ให้ถูกต้อง** เปิด `week25/05_vllm/exercises/ex05_launch_command.py` ในไฟล์มี `TODO` สี่จุด แต่ละจุดคืนค่าเป็น list ของ token บรรทัดคำสั่ง (หรือตัวเลข):

1. `docker_flags()`: flag ของ container จากการตั้งค่าพื้นฐาน: GPU, `--ipc host`, พอร์ต 8000 และ HF cache
2. `max_full_length_seqs(weights_gb, kv_gb_per_seq, util)`: จำนวน sequence แบบ context เต็มที่ใส่ได้ ซึ่งจะกลายเป็น `--max-num-seqs`
3. `tool_flags(family)`: auto tool choice พร้อม `--tool-call-parser` ที่ถูกต้องสำหรับ Qwen3.6, Nemotron และ Gemma 4
4. `lora_flags(name, path, rank)`: serve LoRA adapter คู่กับ base model

ตัวตรวจแบบออฟไลน์ทดสอบแต่ละฟังก์ชัน จากนั้นประกอบคำสั่งเต็มสองชุด (เซิร์ฟเวอร์ agent-ready และเซิร์ฟเวอร์ base + LoRA) แล้ว parse แบบเดียวกับที่เชลล์ทำ

```bash
# on: laptop
.venv/bin/python week25/05_vllm/exercises/ex05_launch_command.py
```

**Expected output** (เมื่อทำ TODO ครบทั้งสี่จุด บันทึกจาก Mac เครื่องนี้ คำสั่งสองชุดที่พิมพ์ออกมาถูกตัดให้สั้นลงที่นี่)

```
✓ docker_flags: --gpus all · --ipc host · -p 8000:8000 · HF cache → /root/.cache/huggingface
✓ max_full_length_seqs: Qwen3-32B@0.8 → 9 · Llama-70B-NVFP4@0.8 → 5 · 70B-bf16 → 0 · 20 GB@0.4 → 13
✓ tool_flags: auto tool choice on · qwen3.6→qwen3_xml · nemotron→qwen3_coder · gemma4→gemma4
✓ lora_flags: --enable-lora · --lora-modules hotel-ft=/adapters/hotel-ft · --max-lora-rank ≥ 16
✓ both assembled commands parse: docker flags, IMAGE, then `vllm serve <model>` flags

▣ A · agent-ready server (assembled from your functions; the full recipe adds speed flags)
$ docker run -d \
  --name vllm-server \
  --gpus all \
  …
  vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.4 \
    --max-num-seqs 4 \
    --kv-cache-dtype fp8 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml
```

<details><summary>คำใบ้ — ทำไมกรณี 70B bf16 จึงคืนค่า 0?</summary>

ส่วนที่จองไว้คือ 0.8 × 128 = 102.4 GB แต่ weights อย่างเดียวก็ 141.2 GB แล้ว พื้นที่ KV จึงติดลบ ให้คืนค่า 0 แทนตัวเลขติดลบ: ไม่มี sequence แบบ context เต็มใดใส่ได้ และ vLLM จะไม่ยอมเริ่มทำงาน

</details>

<details><summary>คำใบ้ — อะไรต้องตามหลัง <code>--lora-modules</code>?</summary>

คู่ `NAME=PATH` หนึ่งคู่ต่อหนึ่ง adapter `NAME` คือ model id ที่ client ส่งมา `PATH` คือตำแหน่งของ adapter **ภายใน container**: `/adapters/hotel-ft` เพราะคำสั่ง mount `$HOME/w25/adapters` ไว้ที่ `/adapters`

</details>

<details><summary>ท้าทายเพิ่ม — สอง adapter หนึ่งเซิร์ฟเวอร์</summary>

เพิ่ม adapter ตัวที่สองลงใน `lora_flags` (vLLM รับคู่ `NAME=PATH` หลายคู่หลัง `--lora-modules`) adapter ทำงานพร้อมกันใน batch เดียวได้กี่ตัว? ค้นหา `--max-loras` ใน `vllm serve --help=all` บน Spark ของคุณ

</details>

✓ Checkpoint: บรรทัดตรวจทั้งห้าเป็น ✓

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `CUDA out of memory` ตอนเริ่มต้น | ลด `--max-model-len` และ `--max-num-seqs` หรือลด `--gpu-memory-utilization` (playbook) ใช้ lab 05-1 ดูว่าพจน์ไหนใหญ่เกินไป |
| หน่วยความจำไม่พอทั้งที่โมเดลควรจะใส่ได้ | unified memory: page cache ของ OS ยังถือไฟล์ไว้ playbook ล้างมันด้วย: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| `nvidia-smi --query-gpu` แสดงหน่วยความจำเป็น `N/A` | เป็นเรื่องปกติบน unified memory (playbook) ใช้ `nvidia-smi` เปล่า ๆ หรือ `free -g` |
| `content: null` พร้อม `finish_reason: length` | reasoning model ใช้งบไปกับการคิดหมด เพิ่ม `max_tokens` (playbook ใช้ 4096) |
| tool call มาเป็นข้อความใน `content` และ `tool_calls` ว่าง | เซิร์ฟเวอร์ถูกเปิดโดยไม่มี `--enable-auto-tool-choice` หรือใช้ `--tool-call-parser` ของตระกูลโมเดลอื่น (ส่วนที่ 4) |
| error บอกว่า auto tool choice ต้องใช้ `--enable-auto-tool-choice` และ `--tool-call-parser` | client ของคุณส่ง `tools` มา แต่เซิร์ฟเวอร์ไม่ได้เปิดสำหรับ tool calling รีสตาร์ตพร้อม flag ทั้งสองตัว |
| `Server not responding on port 8000` | พอร์ตถูกใช้อยู่ (NIM ก็ใช้ 8000, Module 06) `lsof -i :8000` หรือ map เป็น `-p 8001:8000` |
| `exec format error` / ไม่มี image ARM64 | ใช้ image จากส่วนที่ 2 Spark เป็น aarch64 |
| Gated repo / 401 จาก Hugging Face | ยอมรับ license ของโมเดลบนหน้า Hugging Face ของมัน แล้ว `hf auth login` บน Spark |
| `rm: cannot remove …/models--…: Permission denied` | container ดาวน์โหลดในฐานะ root: `sudo rm -rf $HOME/.cache/huggingface/hub/<model>` (playbook) |
| โหนด 2 ไม่ปรากฏใน `ray status` | ลิงก์ QSFP หรือ IP มีปัญหา: ทำการตรวจของ Module 02 ใหม่ และตรวจว่า Spark ทั้งสองใช้ `MN_IF_NAME` และ image tag เดียวกัน |
| คลัสเตอร์ Ray หายไปหลัง SSH หลุด | `run_cluster.sh` มี EXIT trap: ให้เปิดมันภายใน `tmux` เสมอ |
| `run_cluster.sh` ไม่พิมพ์อะไรเลยนานหลายนาที และ `ray status` บอกว่ายังไม่ได้ติดตั้ง Ray | `pip install ray` ที่สคริปต์ติดตั้งตอนเริ่มค้างไป (เจอครั้งหนึ่งเมื่อ 2026-10-03 รอบที่สองเสร็จในไม่กี่วินาที) `docker rm -f node-NNNN` แล้วเริ่มใหม่ |
| `unrecognized arguments: --swap-space 0` หรือ `` `ray` does not support async scheduling yet `` | flag จาก model card ที่เขียนไว้สำหรับ vLLM รุ่นอื่นหรือเครื่องเดียว เอาออก ตามคำสั่ง Nemotron ในส่วนที่ 6 |
| โมเดลที่คัดลอกมาจาก Spark อีกเครื่องเล็กผิดปกติ: `du -shL ~/.cache/huggingface/hub/models--<org>--<name>/snapshots` แสดงเป็น KB ไม่ใช่ GB | คัดลอกมาแค่โฟลเดอร์ลิงก์ของโมเดล ไม่ได้คัดลอกที่เก็บรวม `hub/blobs/` ให้คัดลอกทั้งโฟลเดอร์ `~/.cache/huggingface/hub/` (ส่วนที่ 6 ขั้นที่ 5) |

## Next — บทถัดไป

ไปต่อที่ [Lab 06 — SGLang, TensorRT-LLM, NIM และ Nemotron: ประลอง engine](../06_sglang_trtllm_nim/TUTORIAL.md): เปิด serving engine อีกสามตัว serve Nemotron Nano และ Super แล้วเปรียบเทียบทั้งหมดด้วย benchmark ที่ยุติธรรมชุดเดียว
