# ▶ Spark Lab 20 — Capstone: เอเจนต์โรงแรมแบบอธิปไตยข้อมูล (sovereign) — fine-tune → serve → gateway → NAT agent ใน sandbox

> ส่วนหนึ่งของ Week 25 · DGX Spark: fine-tune, serve และสร้างเอเจนต์ที่รันใน sandbox คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง ทุกแล็บรันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (อ้างอิงคำต่อคำจาก playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- รวมชิ้นส่วนทั้งหมดของสัปดาห์นี้เป็นระบบเดียวบน **Spark สองเครื่อง**: router ที่ fine-tune แล้วอยู่บน Spark B ส่วน "สมอง" ของเอเจนต์ gateway และ sandbox อยู่บน Spark A
- เปลี่ยน **router ที่ fine-tune แล้วให้เป็น tool** ที่เอเจนต์ของ NeMo Agent Toolkit เรียกก่อนลงมือทำ
- รัน **สถานการณ์ทดสอบการยอมรับ (acceptance scenario)** สี่แบบตั้งแต่ต้นจนจบ (กลิ่นควัน ห้องร้อน คำถามเรื่องร้านอาหาร และข้อความภาษาไทย) แล้วให้โค้ดเป็นผู้ตัดสินแต่ละกรณี
- ล้อมเอเจนต์ไว้ด้วย **OpenShell policy** ที่อนุญาตทางออกแค่สองทางพอดี และตรวจกับความพยายามหลบหนีสิบแบบ
- อ่าน **scorecard** หนึ่งใบที่บอกอย่างตรงไปตรงมาว่าอะไรพิสูจน์แล้ว และอะไรยังต้องรอ Spark

**Time** ~75 นาที · **Difficulty** ระดับสูง · **Hardware** Spark 2 เครื่อง (เครื่องเดียวก็ได้โดยรัน vLLM สองตัวบนเครื่องเดียว; ไม่มีเลย: การเชื่อมต่อทั้งหมดรันบนตัวแทนบนแล็ปท็อป)

**Playbook ทางการที่ครอบคลุม:** [vLLM](https://build.nvidia.com/spark/vllm) (สูตร agent-ready) · [OpenShell](https://build.nvidia.com/spark/openshell) · [LLaMA Factory](https://build.nvidia.com/spark/llama-factory) · [Connect two Sparks](https://build.nvidia.com/spark/connect-two-sparks) ทั้งหมดผ่านโมดูลที่สอนเรื่องนั้นมาแล้ว ส่วนที่เป็นตัวเชื่อม (router-as-a-tool, gateway alias, acceptance judge, capstone policy) เป็นงานต้นฉบับของคอร์สนี้

## 0 · ก่อนเริ่ม

capstone นำโมดูลก่อนหน้ากลับมาใช้ ไม่ได้ทำซ้ำ ให้รู้ว่าแต่ละชิ้นมาจากไหน:

| ชิ้นส่วน | Module | สิ่งที่ capstone นำมาใช้ |
|---|---|---|
| การ fine-tune router ของโรงแรม | 09 | LoRA adapter `~/w25/m09/saves/qwen3-4b-hotel/lora/sft` และ system prompt ของมัน |
| การให้คะแนนและ serve โมเดลที่ fine-tune | 13 | `_hotel_eval.py` (ด่านตรวจก่อนปล่อยใช้งาน หรือ ship gate), vLLM พร้อม `--lora-modules hotel-ft=…` |
| vLLM แบบ agent-ready | 05 | คำสั่งของ Qwen3.6-35B-A3B-NVFP4 และ flag สามตัวสำหรับเอเจนต์ |
| Gateway | 08 | `_gateway.py` (ตัวเขียน config + ตัวรัน proxy), การติดตั้งบน Spark |
| เอเจนต์ | 14 | NAT 1.9, plugin `hotel_ops_nat` (`room_temperature`, `create_maintenance_ticket`) |
| Sandbox | 15 | `policykit`, ลำดับคำสั่ง OpenShell จาก lab 15-4 |
| Spark สองเครื่อง | 01, 02 | `SPARK_HOST` และ `SPARK_HOST2` ที่ตอบ `ssh -o BatchMode=yes` ได้ทั้งคู่ |

บนแล็ปท็อปคุณต้องมี venv ของ NAT ที่ติดตั้ง plugin ทั้งสองแล้ว และ venv ของ LiteLLM:

```bash
# on: laptop
uv pip install --python week25/.venv-nat/bin/python --no-deps \
  -e week25/14_nat_agents/hotel_ops_nat -e week25/20_capstone_sovereign_agent/hotel_capstone_nat
week25/.venv-nat/bin/nat info components -t function | grep -E "route_guest|room_temp|maintenance"
ls week25/.venv-litellm/bin/litellm
```

✓ Checkpoint: `nat info components` แสดง `route_guest_message`, `room_temperature` และ `create_maintenance_ticket` และมีไฟล์ binary ของ LiteLLM อยู่

## 1 · สิ่งที่คุณกำลังสร้าง

เอเจนต์ปฏิบัติการกะดึกสำหรับโรงแรมขนาดเล็ก แขกส่งข้อความมาเป็นภาษาอังกฤษหรือภาษาไทย เอเจนต์จัดประเภทข้อความด้วย **โมเดลที่คุณ fine-tune มาเพื่องานนี้โดยเฉพาะ** อ่านอุณหภูมิห้องถ้าเป็นเรื่องความร้อนเย็น เปิด ticket เมื่อฝ่ายช่าง (engineering) หรือฝ่ายรักษาความปลอดภัย (security) ต้องลงมือ แล้วจึงตอบแขก ไม่มีข้อมูลใดออกไปนอก Spark สองเครื่องของคุณ

```text
                    Spark A — agent side                                   Spark B — router side
 ┌──────────────────────────────────────────────────────────┐   ┌───────────────────────────────────┐
 │  OpenShell sandbox (policy: 2 ways out)                   │   │  vLLM :8000                        │
 │   NAT tool_calling_agent                                  │   │   Qwen3-4B-Instruct-2507           │
 │     tools: route_guest_message · room_temperature ·       │   │   + LoRA hotel-ft (Module 09)      │
 │            create_maintenance_ticket                      │   │                                   │
 │        │ brain: inference.local      │ router tool        │   │  LLaMA Factory: retrain the next   │
 │        ▼                             ▼                    │   │  adapter here while this one serves│
 │  LiteLLM gateway :4000 (master key)                        │   └───────────────▲───────────────────┘
 │     agent-brain  → vLLM 127.0.0.1:8000 (Qwen3.6 NVFP4) ───┼───── hotel-router ─┘ (LAN / ConnectX-7)
 └──────────────────────────────────────────────────────────┘
```

การตัดสินใจเชิงออกแบบสามข้อ และเหตุผล:

1. **router เป็น tool ไม่ใช่หน้าที่ของเอเจนต์** สมอง (MoE ขนาด 35B) เดาแผนกได้ก็จริง แต่คุณวัดไว้แล้วใน Module 13 ว่าการเดาพลาดเหตุฉุกเฉินและติดป้ายคำขอภาษาไทยผิด โมเดล 4B ที่เทรนมาเพื่องานนี้ เมื่อเรียกใช้เป็น tool จะถูกกว่าและตรวจสอบได้
2. **การเรียกโมเดลทุกครั้งผ่าน gateway เดียว** gateway คือจุดเดียวที่ใช้เก็บคีย์ บันทึกการ fallback และสลับโมเดล ฝั่งผู้เรียก (รวมถึงเอเจนต์) ไม่รู้เลยว่า Spark เครื่องไหนเป็นผู้ตอบ
3. **sandbox ถูกล้อมให้ไปได้แค่สองปลายทางพอดี** เอเจนต์เข้าถึงสมองของมันและ router ได้ แค่นั้น: ไม่ได้ออกอินเทอร์เน็ต ไม่ได้ต่อ Spark B ตรง ๆ และไม่ได้เข้า home directory ของคุณ

✓ Checkpoint: คุณบอกได้ว่า Spark เครื่องไหน serve โมเดลไหน และทำไม router จึงคุยกับ Spark B ผ่าน gateway เท่านั้น

## 2 · แผนงานและ preflight: lab 20-1

```bash
# on: laptop
.venv/bin/python week25/20_capstone_sovereign_agent/labs/lab01_plan_and_preflight.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ในโหมด DRY: แผนงานเป็นการคำนวณ แถวของ Spark เป็นค่า EXAMPLE)

```
▣ STEP 1 · the plan: what runs where (arithmetic, not a measurement)
│ where    service                        port  model / role                                  weights  module
│ ───────  ─────────────────────────────  ────  ────────────────────────────────────────────  ───────  ───────
│ Spark A  vLLM · agent brain             8000  nvidia/Qwen3.6-35B-A3B-NVFP4                  20.0 GB  05
│ Spark A  LiteLLM gateway                4000  agent-brain · hotel-router aliases            0.5 GB   08
│ Spark A  OpenShell sandbox + NAT agent  —     route_guest_message · room_temperature · tic  1.0 GB   14 · 15
│ Spark B  vLLM · base + LoRA hotel-ft    8000  Qwen/Qwen3-4B-Instruct-2507 + hotel-ft        8.1 GB   09 · 13
│ Spark A: vLLM reserves 40% of 128 GB =  51.2 GB  ███████████░░░░░░░░░░░░░░░░░
│ Spark B: vLLM reserves 30% of 128 GB =  38.4 GB  ████████░░░░░░░░░░░░░░░░░░░░
◆ Spark A keeps ~77 GB free for the gateway, the sandbox and the OS. Spark B keeps ~90 GB free: enough to retrain the next hotel adapter (Module 09, LoRA on 4B) while the current one serves.
◆ Why two Sparks: the brain and the router scale and fail separately, and retraining never touches the agent's Spark. One Spark also works: run both vLLMs on A with --gpu-memory-utilization 0.4 + 0.3 on ports 8000/8001.
…
│ Spark    GPU   free mem  free disk  needs ports  busy  ready
│ ───────  ────  ────────  ─────────  ───────────  ────  ─────────
│ Spark A  GB10  112 GB    3100 GB    8000 + 4000  none  ◈ example
│ Spark B  GB10  112 GB    3100 GB    8000         none  ◈ example
═ Plan fits: 20 GB brain + 8 GB router on two 128 GB Sparks, each well under its reservation. Next: lab 20-2.
```

20 GB มาจากการคำนวณแบบ NVFP4 ใน Module 01 (35B × ~4.5 บิต) ส่วน 8.1 GB คือ Qwen3-4B ที่ bf16 ทั้งสองอยู่ในพื้นที่ที่จองไว้อย่างสบาย ๆ และพื้นที่จองที่เหลือจะกลายเป็น KV cache

✓ Checkpoint: ในโหมด LIVE Spark ทั้งสองแสดง GB10 พอร์ตว่าง และ ✓; ในโหมด DRY คุณอธิบายแต่ละคอลัมน์ได้

## 3 · เปิดระบบขึ้นมา: lab 20-2

Lab 20-2 พิมพ์ลำดับคำสั่งเปิดเซิร์ฟเวอร์สามชุดโดยเติม host ของคุณให้แล้ว และเขียน config ของ gateway สำหรับ Spark A ถ้าใส่ `--launch` มันจะเปิดทีละตัวตามลำดับ และรอจนแต่ละตัวตอบ `/v1/models`:

```bash
# on: laptop
.venv/bin/python week25/20_capstone_sovereign_agent/labs/lab02_bring_up.py            # plan + config
.venv/bin/python week25/20_capstone_sovereign_agent/labs/lab02_bring_up.py --launch   # start all three
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ในโหมด DRY บรรทัดแรก ๆ)

```
▣ STEP 1 · the gateway config the Spark will run (no laptop fallbacks: the sandbox must stay on the Sparks)
│ alias         backend                                   api_base (as seen from Spark A)
│ ────────────  ────────────────────────────────────────  ───────────────────────────────
│ agent-brain   hosted_vllm/nvidia/Qwen3.6-35B-A3B-NVFP4  http://localhost:8000/v1
│ hotel-router  hosted_vllm/hotel-ft                      http://spark-b:8000/v1
```

แต่ละลำดับคำสั่งเปลี่ยนอะไรจากโมดูลต้นทาง (พิมพ์ไว้ข้าง ๆ กัน):

| บริการ | ต้นทาง | สิ่งที่คอร์สเปลี่ยน (และเหตุผล) |
|---|---|---|
| Router บน B | Module 13, path A | ไม่มี; flag ของ LoRA เป็น CLI ของ vLLM เอง (ส่วนที่คอร์สเพิ่มเข้ามา ดู Module 05 §7) |
| สมองบน A | Module 05 §4 (vLLM playbook ฉบับเก่า) ยกมาตรง ๆ | `-d --name` แทน `-it`; `-p 127.0.0.1:8000:8000` เพื่อให้มีแค่ gateway บน A ที่เข้าถึงได้; ไม่มี `-e HF_TOKEN` (โมเดลนี้ไม่ได้ล็อกสิทธิ์ไว้) |
| Gateway บน A | Module 08 §7 | config **ไม่มี fallback ไปยังแล็ปท็อป**: บน Spark การ fallback แบบเงียบ ๆ ไปยังแล็ปท็อปของใครสักคนจะทำให้ข้อความของแขกรั่วออกนอก Spark |

gateway bind ที่ `0.0.0.0:4000` โดยตั้งใจ ส่วน troubleshooting ของ OpenShell playbook บอกว่า URL ของ provider ต้องใช้ IP ของ Spark ไม่ใช่ `localhost` เพราะ gateway ของ OpenShell รันอยู่ใน Docker สิ่งที่ปกป้องมันคือ master key

> 💡 **การเสริมความปลอดภัย (ไม่บังคับ):** vLLM รับ `--api-key <key>` ได้ ซึ่งทำให้ router บน Spark B ปฏิเสธผู้เรียกทุกรายที่ไม่ใช่ gateway ของคุณ ใส่คีย์เดียวกันใน config ของ gateway เป็น `api_key` ของ deployment นั้น นี่คือ flag ของ vLLM เอง ไม่ได้อยู่ใน playbook

✓ Checkpoint: ถ้ามี Spark สองเครื่อง `curl http://spark-b:8000/v1/models` แสดง `hotel-ft` และบน Spark A `curl -H "Authorization: Bearer $(cat ~/w25/litellm/master.key)" localhost:4000/v1/models` แสดง alias ทั้งสองตัว

## 4 · router กลายเป็น tool

`hotel_capstone_nat/` คือแพ็กเกจ plugin ของ NAT ที่มี tool ตัวเดียว สร้างแบบเดียวกับของ Module 14:

```text
route_guest_message(message)                       → JSON the agent reads
  POST {gateway}/chat/completions
    model: hotel-router                              (the gateway alias, never a hostname)
    system: Module 09's exact training prompt        (a different prompt is a different task)
    temperature: 0
  parse strictly → {department, priority, reply}     (invalid JSON → an error, never a guess)
  + served_by, x-litellm-attempted-fallbacks        (so a dead fine-tune is visible)
```

รายละเอียดสองข้อที่สำคัญกว่าตัวโค้ด:

- **system prompt ต้องเป็นของ Module 09 ทุกไบต์** LoRA adapter เรียนรู้งานนี้ภายใต้ prompt นั้น ถ้าส่ง prompt อื่นไป เท่ากับคุณกำลังวัดโมเดลอีกตัวหนึ่ง
- **ไม่มีการซ่อมแซม** ถ้า router ส่ง `"priority": urgent` กลับมาโดยไม่มีเครื่องหมายคำพูด (ความล้มเหลวที่เจอใน Module 13) tool จะคืนค่า error และ system prompt ของเอเจนต์จะสั่งให้ส่งต่อแขกให้พนักงานดูแล แทนที่จะเดาเอง

`configs/capstone_agent.yml` เป็นตัวเชื่อมทุกอย่างเข้าด้วยกัน ส่วนสำคัญคือ:

```text
llms.brain            base_url ${CAPSTONE_BRAIN_URL}, model agent-brain, api_key ${HOTEL_GATEWAY_KEY}
functions             route_guest_message (base_url ${CAPSTONE_GATEWAY}, model hotel-router)
                      room_temperature · create_maintenance_ticket   (Module 14's plugin)
workflow              tool_calling_agent, max_iterations 6, and the rules:
                        1. route first  2. read the room for temperature  3. ticket only for engineering/security
                        4. two sentences: what you did (quote the ticket id) + the router's reply
```

คีย์อยู่ใน environment เท่านั้นเสมอ ไฟล์ YAML ระบุแค่ชื่อตัวแปร ไม่เคยระบุค่า

✓ Checkpoint: คุณอธิบายได้ว่าทำไม tool จึงปฏิเสธที่จะ "แก้" JSON ที่ไม่ถูกต้อง และทำไมมันต้องส่ง system prompt ของ Module 09 แบบตรงทุกตัวอักษร

## 5 · ตั้งแต่ต้นจนจบ: lab 20-3

Lab 20-3 เปิด gateway (alias ทั้งสองตัวพร้อม fallback ไปยังแล็ปท็อป) แล้วรันเอเจนต์ NAT หนึ่งครั้งต่อหนึ่งสถานการณ์ จากนั้นตัดสินแต่ละรอบจาก **trace ของ NAT เอง** (tool ไหนรัน ตามลำดับใด ได้ผลลัพธ์อะไร) และจาก log ของ ticket มันตัดสินจากสิ่งที่เอเจนต์ทำจริง ไม่ใช่สิ่งที่เอเจนต์บอกว่าทำ:

```bash
# on: laptop
.venv/bin/python week25/20_capstone_sovereign_agent/labs/lab03_agent_end_to_end.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้: NAT 1.9 และ LiteLLM 1.89 ของจริง; ไม่มี Spark alias ทั้งสองจึง fallback ไปที่ gemma4:12b เป็น LAPTOP STAND-IN รันเต็มสองรอบได้ผลตัดสินเหมือนกัน)

```
▣ STEP 3 · scenario 'smoke': Hello, there's a strong smell of smoke in the corridor near room 522. Please hurry.
→ route_guest_message({'message': "Hello, there's a strong smell of smoke in the corridor ne)
  ← {"ok": true, "department": "security", "priority": "urgent", "reply": "We have alerted our security team to investigate the smoke smell immediately.",
→ create_maintenance_ticket({'room': '522', 'issue': 'Strong smell of smoke in the corridor near r)
  ← Created ticket MT-9C47E8 for room 522 (priority urgent): Strong smell of smoke in the corridor near room 522.
· ANSWER  I have opened an urgent maintenance ticket (MT-9C47E8). We have alerted our security team to investigate the smoke smell immediately.
✓ routed first
✓ router said security
✓ ticket for room 522
✓ ticket priority urgent

▣ STEP 3 · scenario 'hot-room': Guest in room 808: it's really hot in here and the air conditioning just blows air.
→ route_guest_message({'message': "it's really hot in here and the air conditioning just blo)
  ← {"ok": true, "department": "engineering", "priority": "urgent", "reply": "I'm sorry for the discomfort; I have notified our engineering team to inspec
→ room_temperature({'room': '808'})
  ← Room 808: 27.9 °C, setpoint 23.0 °C (+4.9 °C, too warm). HVAC status: fault: fan coil unit not responding.
→ create_maintenance_ticket({'room': '808', 'issue': 'Air conditioning blowing air but not cooling)
  ← Created ticket MT-8E1E86 for room 808 (priority urgent): Air conditioning blowing air but not cooling; fan coil unit not responding.
✓ routed first
✓ router said engineering
✓ called room_temperature
✓ ticket for room 808

▣ STEP 3 · scenario 'restaurant': Could you recommend a good seafood restaurant within walking distance? Thank you.
→ route_guest_message({'message': 'Could you recommend a good seafood restaurant within walk)
  ← {"ok": true, "department": "concierge", "priority": "normal", "reply": "I would be happy to provide some excellent local seafood restaurant recommenda
✓ routed first
✓ router said concierge
✓ no ticket opened

▣ STEP 3 · scenario 'thai-tv': สวัสดีครับ ทีวีในห้อง 1203 เปิดไม่ติด
→ route_guest_message({'message': 'สวัสดีครับ ทีวีในห้อง 1203 เปิดไม่ติด'})
  ← {"ok": true, "department": "engineering", "priority": "urgent", "reply": "รับทราบครับ ช่างจะรีบดำเนินการตรวจสอบทีวีในห้อง 1203 ให้โดยเร็วที่สุดครับ",
→ create_maintenance_ticket({'room': '1203', 'issue': 'ทีวีในห้อง 1203 เปิดไม่ติด', 'priority': 'u)
  ← Created ticket MT-19010C for room 1203 (priority urgent): ทีวีในห้อง 1203 เปิดไม่ติด
✓ routed first
✓ router said engineering
✓ ticket for room 1203
✕ ticket priority normal
◆ gateway stopped (pid 57879); port 4000 is free again: True

▣ STEP 4 · acceptance summary
│ scenario    checks passed  time
│ ──────────  ─────────────  ────  ─
│ smoke       4/4            9s    ✓
│ hot-room    4/4            10s   ✓
│ restaurant  3/3            7s    ✓
│ thai-tv     3/4            10s   ✕
═ NOT accepted yet — read the ✕ lines: a routing miss is the router's job (retrain, Module 09/13), a tool-order or ticket miss is the agent's (prompt, Module 14).
```

วิธีอ่านผล:

- **การเชื่อมต่อทั้งระบบพิสูจน์แล้ว** gateway, NAT, tool ที่เขียนเอง, แพ็กเกจ plugin สองตัว และลำดับการเรียก tool ทำงานได้จริงบนแล็ปท็อป เอเจนต์เรียก router ก่อนทุกครั้ง อ่านอุณหภูมิห้อง 808 ก่อนเปิด ticket และไม่เปิด ticket สำหรับคำถามเรื่องร้านอาหาร
- **ความล้มเหลวข้อเดียวคือสิ่งที่คุณรู้อยู่แล้ว** Module 13 วัดไว้แล้วว่าโมเดลที่ใช้แค่ prompt มักตีคำขอภาษาไทยธรรมดา ๆ ว่าเป็นเรื่องด่วน และที่นี่มันก็ทำแบบเดิมอีก คราวนี้ภายในเอเจนต์ที่ทำงานได้ครบทุกส่วน: แขกแจ้งแค่ว่าทีวีเปิดไม่ติด แต่ระบบกลับเปิด ticket ด่วน (urgent) ตอนตีสาม ซึ่งหมายถึงการปลุกช่างเวรกลางดึกโดยไม่จำเป็น ตัวตัดสินการยอมรับ (acceptance judge) จับได้ และ ✕ บอกคุณว่าต้องแก้ที่ชั้นไหน: ที่ **router** โดยเทรนใหม่ใน Module 09 ด้วยตัวอย่างภาษาไทยที่มีความสำคัญระดับปกติ (normal) ให้มากขึ้น prompt ของเอเจนต์ไม่ใช่ต้นเหตุ
- **ดู input ของกรณีห้องร้อน** เอเจนต์ส่ง `"it's really hot in here…"` ให้ router โดยตัด `"Guest in room 808:"` ทิ้ง ทั้งที่ prompt บอกให้ใช้ "the guest's exact words" ผลออกมายังใช้ได้ แต่เอเจนต์มักถอดความ (paraphrase) input ของ tool ถ้าต้องการระบบที่เข้มงวดขึ้น ให้ส่งข้อความดิบเข้า tool จากโค้ดโดยตรง แทนที่จะไว้ใจให้โมเดลคัดลอกเอง

เมื่อ Spark ทั้งสองเครื่อง serve อยู่ แล็บเดียวกันจะใช้ router ที่ fine-tune แล้ว และแถวภาษาไทยคือแถวที่ต้องจับตาดู

✓ Checkpoint: คุณรัน lab 20-3 แล้ว และบอกได้สำหรับทุก ✕ ว่าต้องแก้ที่ router ที่ prompt ของเอเจนต์ หรือที่ tool

## 6 · รั้วกั้น: lab 20-4 ขั้นที่ 1–3

sandbox policy สร้างด้วย `policykit` ของ Module 15 และเขียนลง `policies/capstone_policy.yaml`:

| กลุ่มเครือข่าย | Endpoint | Binaries | เมื่อไร |
|---|---|---|---|
| `inference` | `inference.local:443` | Python ของ venv, `/usr/bin/python3*`, `curl` | ตลอดเวลา: สมอง ผ่าน `openshell inference set` → `agent-brain` ของ gateway |
| `gateway` | `host.openshell.internal:4000` (`access: full`) | Python ของ venv เท่านั้น | ตลอดเวลา: tool router |
| `pypi` | `pypi.org`, `files.pythonhosted.org` | Python, pip | ช่วงติดตั้งเท่านั้น แล้วลบออกด้วย `openshell policy update … --remove-rule pypi` |

```bash
# on: laptop
.venv/bin/python week25/20_capstone_sovereign_agent/labs/lab04_sandbox_and_scorecard.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้; ขั้นที่ 2 เป็นแบบจำลองเพื่อการสอนของ policykit ไม่ใช่ OpenShell)

```
▣ STEP 1 · build and validate the capstone policy (Module 15's policykit)
✓ policy valid · 3 network groups (inference, gateway, pypi) · 0 warning(s)

▣ STEP 2 · what would it allow? (policykit's teaching model — not OpenShell)
│ attempted action (after setup)  decision                  why
│ ──────────────────────────────  ─────────────────────  ─  ────────────────────────────────────────────────────
│ router tool → gateway           allow                  ✓  network_policies.gateway: host.openshell.internal:4…
│ brain → inference.local         inspect_for_inference  ✓  inference.local is handled by the proxy's inference…
│ router tool → Spark B directly  deny                   ✓  spark-b:8000 is not in network_policies (default de…
│ curl → gateway                  deny                   ✓  host.openshell.internal:4000 is listed, but not for…
│ agent → api.openai.com          deny                   ✓  api.openai.com:443 is not in network_policies (defa…
│ write tickets.jsonl             allow                  ✓  under read_write /sandbox
│ write /etc/hosts                deny                   ✓  /etc is read_only (Landlock: Permission denied)
│ read ~/.ssh on the host         deny                   ✓  not in read_only or read_write (Landlock: Permissio…
│ run as root                     deny                   ✓  the sandbox runs as 'sandbox'; it cannot become 'ro…
│ pip install (after setup)       deny                   ✓  pypi.org:443 is not in network_policies (default de…
✓ every attempt lands where the plan says: two ways out, nothing else
```

บั๊กจริงที่ขั้นนี้จับได้ระหว่างเขียน capstone: กฎ `gateway` เวอร์ชันแรกไม่มี `access` mode policykit เตือนว่า *"no access mode … the proxy denies it with CONNECT 403"* ซึ่งเป็นกฎที่มันนำมาจาก NemoClaw applications playbook ถ้าไม่ได้ตรวจ tool router จะถูกบล็อกภายใน sandbox ทั้งที่การทดสอบบนแล็ปท็อปผ่านหมดทุกข้อ ตรวจความถูกต้องก่อน push เสมอ

ขั้นที่ 3 พิมพ์ลำดับคำสั่ง OpenShell บน Spark A มันคือลำดับเดียวกับ lab 4 ของ Module 15 แต่ชี้ไปที่ gateway แทน: สร้าง provider สำหรับ gateway, `inference set --model agent-brain`, สร้าง sandbox พร้อม policy, อัปโหลดทั้งสองโฟลเดอร์, ติดตั้ง NAT ขณะที่ `pypi` ยังเปิดอยู่, ลบ `pypi` ออก, รันเอเจนต์ แล้วพิสูจน์รั้วด้วย `curl https://example.com` (ต้องล้มเหลว) ทุกคำสั่งที่เปลี่ยนแปลง Spark จะรันเฉพาะเมื่อใส่ `--yes`

> ⚠ **สมมติฐานของคอร์ส ให้ตรวจครั้งแรกบน Spark** (1) NemoClaw applications playbook แสดงว่า `host.openshell.internal` เข้าถึง vLLM ของ host บน `:8000` ได้ capstone ใช้ชื่อเดียวกันนี้สำหรับ `:4000` ของ gateway (2) คีย์ของ gateway ถูกส่งด้วย `--env` บนบรรทัดคำสั่ง ซึ่งผู้ใช้คนอื่นบน Spark อาจเห็นได้ในรายการโปรเซส บน Spark ที่ใช้ร่วมกัน ให้ใช้ credential ของ OpenShell provider แทน ในการรันครั้งแรกให้ดู `openshell logs hotel-capstone --source sandbox` เพื่อหาบรรทัด `decision=deny`

✓ Checkpoint: ขั้นที่ 2 แสดง ✓ ครบทั้งสิบความพยายาม และคุณบอกได้ว่าทำไม "router tool → Spark B directly" ต้องถูกปฏิเสธ แม้ว่า Spark B จะเป็นเครื่องของคุณเอง

## 7 · Scorecard: "เสร็จ" หมายความว่าอะไร

ขั้นที่ 4 ของ lab 20-4 รวบรวมทุกด่านตรวจไว้ในตารางเดียว:

**Expected output** (บันทึกจาก Mac เครื่องนี้หลังรัน lab 20-3 เต็มรอบตามด้านบน)

```
▣ STEP 4 · the capstone scorecard
│ gate                                             evidence
│ ────────────────────────────────────────────  ─  ────────────────────────────────────────────────────
│ router ship gate (Module 13)                  —  no fine-tune predictions yet (needs Module 09 on a …
│ agent scenarios (lab 20-3)                    ✕  3/4 accepted · LAPTOP STAND-IN · 2026-09-29 13:32  …
│ sandbox policy valid + fence (this lab)       ✓  2 runtime groups after setup · 10 attempts checked …
│ sandbox applied on Spark A (this lab, --yes)  —  DRY: not applied
═ Not complete yet: every — needs a Spark, every ✕ names the module to revisit. That is the honest state of a stack built on a laptop: the wiring is proven, the fine-tune and the sandbox are waiting for their Sparks.
```

capstone จะเสร็จสมบูรณ์เมื่อทุกแถวเป็น ✓ **บน Spark จริง** การรันแบบ LIVE รอบแรก ตามลำดับ:

1. Module 09 บน Spark B: เทรน แล้ว predict → Module 13 lab 02 ให้คะแนน (แถวที่ 1)
2. Lab 20-2 `--launch`: router บน B, สมองและ gateway บน A
3. Lab 20-3 กับ gateway บน Spark (แถวที่ 2) แถวภาษาไทยคือแถวที่ต้องจับตาดู
4. Lab 20-4 `--yes`: sandbox บน Spark A (แถวที่ 4) แล้วใช้ `openshell logs` ยืนยันทั้งสองเส้นทางและการปฏิเสธต่าง ๆ
5. รันแล็บใดก็ได้ซ้ำด้วย `SPARK_RECORD=1` เพื่อให้โหมด DRY เล่นบันทึกจริงของคุณซ้ำได้นับจากนั้น

✓ Checkpoint: คุณอ่าน scorecard ของตัวเองได้ และบอกคำสั่งถัดไปที่ต้องรันสำหรับทุกแถวที่ยังไม่เป็น ✓

## Labs — รันแล็บได้ที่นี่

**labs/lab01_plan_and_preflight.py** — อะไรรันบน Spark เครื่องไหน แต่ละตัวจองหน่วยความจำเท่าไร และการตรวจความพร้อมแบบอ่านอย่างเดียวบน Spark ทั้งสอง

**labs/lab02_bring_up.py** — ลำดับคำสั่งเปิดเซิร์ฟเวอร์สามชุดพร้อม host ของคุณ config ของ gateway ฝั่ง Spark และตัวเลือกเปิดระบบที่รอจนแต่ละเซิร์ฟเวอร์พร้อม

**labs/lab03_agent_end_to_end.py** — สถานการณ์ของแขกสี่แบบผ่าน gateway, เอเจนต์ NAT, tool router และ tool ของโรงแรม ตัดสินจาก trace ของ NAT เอง

**labs/lab04_sandbox_and_scorecard.py** — OpenShell policy ของ capstone ความพยายามหลบหนีสิบแบบ ลำดับคำสั่ง sandbox ฝั่ง Spark และ scorecard สุดท้าย

`labs/_capstone.py` (แผนงาน config ของ gateway สถานการณ์ทดสอบ และตัวตัดสิน) และ `hotel_capstone_nat/` (tool router) เป็นส่วนที่ใช้ร่วมกัน ไม่ใช่แล็บ

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 20 — ตัวตัดสินการยอมรับ (acceptance judge)** เปิด `week25/20_capstone_sovereign_agent/exercises/ex20_acceptance.py` ในไฟล์มี `TODO` สามจุด:

1. `judge(scenario, calls, tickets)`: เรียก router ก่อน แผนกถูกต้อง กฎเรื่อง ticket (เปิด ticket เฉพาะงานของฝ่ายช่างและฝ่ายรักษาความปลอดภัย) และระดับความสำคัญเมื่อควรมี ticket
2. `THAI_SCENARIO`: ข้อความภาษาไทยเรื่องปัญหางานช่างทั่วไปที่ไม่ด่วน ซึ่งเป็นสถานการณ์ที่จับบั๊กที่ lab 20-3 พบได้
3. `verdict(results)`: การตรวจที่ไม่ผ่านแม้แต่ข้อเดียวที่ใดก็ตามจะขวางการยอมรับ และต้องระบุชื่อตัวเองออกมา

```bash
# on: laptop
.venv/bin/python week25/20_capstone_sovereign_agent/exercises/ex20_acceptance.py
```

**Expected output** (เมื่อทำ TODO ครบทั้งสามจุด)

```
✓ judge: 6 recorded runs · smoke ✓ · restaurant ✓ · stray ticket ✕ · Thai urgency ✕ · acted first ✕ · misroute ✕
✓ Thai scenario: "สวัสดีค่ะ ไฟในห้องน้ำห้อง 402 กะพริบ" → engineering · normal
✓ verdict: one failed check anywhere blocks acceptance, and names it (thai: priority)

═ Add THAI_SCENARIO to _capstone.SCENARIOS and re-run lab 20-3: the laptop stand-in may well fail it;
  the fine-tuned router (trained on Thai examples) is what has to pass it.
```

<details><summary>คำใบ้ — ทำไมต้องตัดสินจาก trace แทนที่จะดูจากข้อความคำตอบ?</summary>

คำตอบของเอเจนต์เป็นร้อยแก้ว และมันอ้างว่า "เปิด ticket แล้ว" ได้โดยไม่เคยเรียก tool เลย trace ของ NAT บันทึกการเรียก tool ทุกครั้งพร้อม input และ output และ log ของ ticket บันทึกทุก ticket ให้ตัดสินจากสิ่งที่เกิดขึ้นจริง ไม่ใช่สิ่งที่ถูกพูด

</details>

<details><summary>ท้าทายเพิ่ม — ปิดวงรอบให้ครบจริง</summary>

เพิ่มข้อความภาษาไทยที่มีความสำคัญระดับปกติสิบข้อลงในข้อมูลเทรนของ Module 09 (`week25/09_llama_factory/data/hotel_ops.json`) เทรนใหม่บน Spark B ให้คะแนนด้วย Module 13 lab 02 และถ้าผ่านด่านตรวจ ให้สลับโฟลเดอร์ adapter แล้วรีสตาร์ต container ของ router จากนั้นรัน lab 20-3 ใหม่ นี่คือทั้งสัปดาห์ในวงรอบเดียว: data → fine-tune → gate → serve → agent → acceptance

</details>

✓ Checkpoint: บรรทัดตรวจทั้งสามเป็น ✓

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| lab 20-3: `tool not registered: route_guest_message` | ติดตั้ง plugin ทั้งสองลงใน venv ของ NAT (ส่วนที่ 0); `nat info components -t function` ต้องแสดง tool ครบทั้งสามตัว |
| ทุกสถานการณ์: `(no answer) … Connection error` | gateway ไม่ได้เปิด หรือไม่มี `HOTEL_GATEWAY_KEY` แล็บตั้งค่าทั้งสองให้; ถ้ารัน `nat run` เอง ให้ export ทั้งสองก่อน |
| tool router คืนค่า `router returned invalid JSON` | นั่นคือ tool ทำงานถูกต้อง: router เขียนสิ่งที่โค้ด parse ไม่ได้ (`urgent` ที่ไม่มีเครื่องหมายคำพูดแบบใน Module 13) ให้แก้ที่ router ไม่ใช่ที่ tool |
| การเรียก tool ทุกครั้งแสดง `fallbacks: 1` บน Spark | `hotel-router` หรือ `agent-brain` ไม่ได้ถูก serve อยู่ ตรวจ `curl http://spark-b:8000/v1/models` และ `docker logs w25-router` / `w25-brain` |
| ภายใน sandbox: `CONNECT tunnel failed, response 403` บนพอร์ต 4000 | กฎ `gateway` ต้องมี access mode (`access: full`) และ binary ที่เป็นผู้เรียกต้องอยู่ในรายการ เทียบกับ `policies/capstone_policy.yaml` |
| ภายใน sandbox: `inference.local` ใช้ไม่ได้ | `openshell inference get` ต้องแสดง provider `capstone-gateway` และ `agent-brain`; URL ของ provider ต้องใช้ IP ของ Spark ไม่ใช่ localhost |
| เอเจนต์ตอบโดยไม่เรียก `route_guest_message` | โมเดลแบบ thinking หรือโมเดลเล็กบางครั้งข้าม tool ตรวจว่า `agent-brain` เป็นสูตร Qwen3.6 ที่มี `--enable-auto-tool-choice` หรือใช้ตัวแทนบนแล็ปท็อปที่แข็งแรงกว่า |

## Next — บทถัดไป

ไปต่อที่ [Lab 21 — แผนที่รวม playbook (playbook atlas)](../21_playbook_atlas/TUTORIAL.md): playbook ทางการของ DGX Spark อื่น ๆ ทั้งหมด จัดกลุ่มตามสิ่งที่คุณอยากสร้างต่อไป
