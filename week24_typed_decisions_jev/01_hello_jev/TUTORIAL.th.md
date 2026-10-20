# ▶ Jev Lab 01 — สวัสดี Jev: การตัดสินใจแบบมีชนิด (typed decision) ครั้งแรกของคุณ

> ส่วนหนึ่งของ Week 24 · การตัดสินใจของ AI แบบมีชนิดด้วย Jev คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง ทุกแล็บรันในโหมด **DRY** ได้ด้วย (ไม่ต้องมีคีย์ ไม่ใช้เครือข่าย $0) โดยเล่นคำตอบที่บันทึกไว้ซ้ำ

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- เข้าใจจากภาพเดียวว่า Jev คืออะไร และ *ไม่ใช่* อะไร
- เรียก Jev ครั้งแรกด้วยสามวิธี: `curl`, Python ธรรมดา และตัวช่วย `jevkit`
- อ่านทุกฟิลด์ของคำตอบจาก Jev: `choice`, `probabilities`, `confidence`, `noul`, `score`, `usage`
- แก้ไขคำขอจริงในหน้านี้ แล้วดูค่าความน่าจะเป็น (probability) ขยับตาม

**Time** ~20 นาที · **Difficulty** ระดับเริ่มต้น · **Cost** ≈ $0.0001 เมื่อรันจริง · $0 ในโหมด dry

## 0 · ก่อนเริ่ม

คุณต้องมีสามอย่าง Lab Runner ตรวจให้แล้ว ดูได้ที่ป้ายสถานะ (status pill) บนแถบด้านบน

| สิ่งที่ต้องมี | วิธีตรวจ | พร้อมแล้วหรือยัง |
|---|---|---|
| Python 3.10 ขึ้นไป | `.venv/bin/python --version` | `.venv` ของ repo เป็น 3.13 |
| TypeSafe API key | `grep -c '^TYPESAFE_API_KEY=' .env` พิมพ์ `1` | พร้อม — อยู่ใน `.env` ที่ root ของ repo |
| ไม่ต้อง `pip install` | แล็บใช้แค่ standard library | — |

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
.venv/bin/python --version
grep -c '^TYPESAFE_API_KEY=' .env
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
Python 3.13.13
1
```

> 🔐 คีย์อยู่บนเครื่องนี้เท่านั้น Lab Runner อ่านคีย์ฝั่งเซิร์ฟเวอร์แล้วส่งให้สคริปต์แล็บ — ไม่เคยส่งไปที่เบราว์เซอร์ ไม่พิมพ์ออกมา และไม่เขียนลงไฟล์ผลลัพธ์ ห้ามวางคีย์ลงในโค้ด โน้ตบุ๊ก หรือภาพหน้าจอเด็ดขาด

✓ Checkpoint: ป้ายสถานะบนแถบด้านบนขึ้นว่า **LIVE · jev-1.13.0** (ถ้าขึ้นว่า DRY ทุกอย่างยังใช้ได้ — คุณจะเห็นคำตอบที่บันทึกไว้แทนคำตอบใหม่)

## 1 · Jev คืออะไร ในภาพเดียว

โมเดล AI ส่วนใหญ่ที่คุณเคยใช้ *สร้างข้อความ*: คุณถาม มันเขียนย่อหน้าตอบ **Jev ต่างออกไป** Jev เป็นโมเดลแบบ *System One* — มันให้ **การตัดสินที่รวดเร็วและมีชนิดข้อมูลชัดเจน (typed judgment)** ที่โค้ดของคุณนำไปใช้ได้ทันที เหมือนเรียกฟังก์ชัน

คุณส่งให้มันสองอย่าง:

- **`state`** — ข้อเท็จจริงที่ให้พิจารณา (ข้อความ อีเมล หรือระเบียน JSON)
- **`questions`** — แผนผัง (map) ของคำถามแบบมีชนิด ที่ตั้งชื่อไว้ เกี่ยวกับ state นั้น

มันส่งกลับ **คำตอบแบบมีชนิดหนึ่งคำตอบต่อหนึ่งคำถาม** และมีคำถามแค่สามชนิด:

| ชนิด | ความหมายแบบภาษาคน | สิ่งที่ได้กลับมา |
|---|---|---|
| `choice` | "ตัวเลือกไหน *หนึ่งเดียว* ที่เข้ากันที่สุด?" | ตัวเลือกที่ถูกเลือก + probability ของทุกตัวเลือก + confidence |
| `noul` | "ข้อความนี้มีโอกาสเป็นจริงแค่ไหน?" | ตัวเลขหนึ่งตัวจาก 0 (ไม่ใช่) ถึง 1 (ใช่) |
| `score` | "สิ่งนี้อยู่ตรงไหนบนสเกลที่ฉันเรียงลำดับไว้?" | ตัวเลขเช่น 1.3 บนระดับ 0…n + probabilities + confidence |

ให้นึกถึง Jev ว่าเป็น **เพื่อนร่วมงานที่เร็วมาก แต่ทำได้แค่ติ๊กช่องที่คุณออกแบบไว้** — ไม่เคยเขียนจดหมายให้คุณ ข้อจำกัดนี้คือจุดประสงค์: คำตอบเครื่องอ่านได้เสมอ *โค้ดของคุณจึงยังเป็นผู้ควบคุม*

```text
your code ──►  state + typed questions  ──►  Jev  ──►  typed answers  ──►  your code decides what to do
               "Room 1203 is too hot"                   intent = hvac (0.97)      route to the HVAC queue
               intent? urgent? how bad?                 urgent = 0.88             page the duty engineer? (policy!)
```

**สิ่งที่ Jev จะไม่ทำ** (เก็บไว้ในโค้ด หรือให้โมเดลอื่นทำ):

- ✗ เขียนคำตอบ สรุป หรือรายงาน → ใช้ LLM แบบสร้างข้อความ (generative) *หลังจาก* Jev ส่งต่อคำขอแล้ว
- ✗ คำนวณเลข นับจำนวน เปรียบเทียบวันที่ → คำนวณใน Python แล้วส่งผลให้ Jev
- ✗ ดึงข้อมูล ท่องเว็บ หรือ scrape → ให้ส่วนประกอบอื่นเก็บข้อมูลที่ได้รับอนุญาตมาก่อน
- ✗ ให้สิทธิ์หรือลงมือทำ (action) → การอนุญาตและการอนุมัติอยู่ในโค้ดของคุณ

✓ Checkpoint: คุณอธิบายให้เพื่อนร่วมงานฟังได้ว่า `state` กับ `questions` ต่างกันอย่างไร และบอกชื่อคำถามทั้งสามชนิดได้

## 2 · เรียกครั้งแรกด้วย curl

นี่คือ API *ทั้งหมด*: `POST` หนึ่งครั้งพร้อม JSON body โหลดคีย์เข้าเชลล์ (⌨ Terminal ในตัวมีคีย์ให้แล้ว) แล้วส่งคำถามหนึ่งข้อ

```bash
set -a; source <(grep '^TYPESAFE_API_KEY=' .env); set +a
curl -s https://api.typesafe.ai/v1/systemone \
  -H "Authorization: Bearer $TYPESAFE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"jev-1.13.0",
       "state":"Room 1203 is too warm and the AC is blowing warm air.",
       "questions":{"is_hvac":{"type":"noul","instructions":"Is this message about air conditioning or cooling?"}}}'
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
{"model":"jev-1.13.0","answers":{"is_hvac":{"type":"noul","noul":0.99}},"usage":{"input_tokens":292,"output_tokens":22}}
```

อ่านจากซ้ายไปขวา:

- `model` — เวอร์ชันที่ตอบจริง ให้บันทึก (log) ไว้เสมอ เพราะชื่อแฝง `jev-latest` อาจย้ายไปชี้เวอร์ชันใหม่ในภายหลัง
- `answers.is_hvac` — กลับมาใต้ **คีย์ที่คุณตั้งเอง** คีย์เป็นแค่ป้ายสำหรับโค้ดของคุณ Jev ไม่เคยเห็นมัน
- `noul: 0.99` — Jev มั่นใจ 99% ว่าข้อความนั้นเป็นจริง
- `usage.input_tokens` — สิ่งที่คุณจ่ายเงิน: 292 token × $0.042 ต่อล้าน ≈ **$0.000012** token ขาออกฟรี (จำนวนจริงเปลี่ยนเล็กน้อยตามถ้อยคำของคำขอ)

✓ Checkpoint: คุณได้ JSON กลับมาพร้อม HTTP 200 และ `"noul"` ใกล้ 1

## 3 · เรียกแบบเดียวกันด้วย Python ธรรมดา

Lab 01 ทำแบบเดียวกับ curl ด้านบนเป๊ะ ๆ โดยใช้แค่ standard library ของ Python — ไม่มีตัวช่วย ไม่มี SDK — และพิมพ์ทั้งคำขอและคำตอบดิบ ให้คุณเห็นทุกไบต์ เปิดดูได้ด้วย **view source** ในส่วน 🧪 Labs หรือรันตรงนี้:

```bash
.venv/bin/python week24/01_hello_jev/labs/lab01_first_call_raw.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
━━ STEP 1 · the request we send (this is ALL of it)
{
  "model": "jev-1.13.0",
  "state": {
    "message": "Room 1203 is too warm and the AC is blowing warm air."
  },
  "questions": {
    "is_hvac": {
      "type": "noul",
      "instructions": "Is `message` about air conditioning or cooling?"
    }
  }
}

━━ STEP 2 · POST https://api.typesafe.ai/v1/systemone
✓ HTTP 200 in 462 ms

━━ STEP 3 · the raw response
{"model": "jev-1.13.0", "answers": {"is_hvac": {"type": "noul", "noul": 0.99}}, "usage": {"input_tokens": 299, "output_tokens": 22}}

═ is_hvac = 0.99 → the model is 99% sure the message is about air conditioning.
◆ you paid for 299 input tokens × $0.042/M = $0.0000126 (output is free)
◆ answered by jev-1.13.0 — log this so you know which version made each decision
```

สังเกตเครื่องหมาย backtick ใน ``Is `message` about…`` เมื่อ `state` เป็นออบเจกต์ JSON คุณชี้คำถามไปที่ฟิลด์ได้ด้วยการเขียนชื่อฟิลด์ใน backtick — นั่นบอก Jev ว่าให้ตัดสิน *ส่วนไหน* ของ state

✓ Checkpoint: คุณรัน lab 01 แล้ว และชี้ได้ว่าคำขอมีสามส่วนคือ `model`, `state`, `questions`

## 4 · คำถามทั้งสามชนิดในการเรียกครั้งเดียว

งานจริงมักถามหลายคำถามพร้อมกัน แต่ละคำถามถูกประเมิน **แยกกันและขนานกัน** บน state เดียวกัน — คำถามหนึ่งมองไม่เห็นคำตอบของอีกคำถาม คุณจ่ายค่า state แค่ครั้งเดียว การรวมคำถาม (batching) จึงถูกกว่า *และ* เร็วกว่าการเรียกแยก

ลองตรงนี้เลย บล็อกด้านล่าง **ใช้งานได้จริง**: แก้อะไรก็ได้ — ข้อความ ตัวเลือก หรือระดับ — แล้วกด **⚡ Ask Jev**

```jev
{
  "state": {"message": "The lobby has been freezing since this morning and guests are complaining. Please fix it before the 6pm wedding!"},
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle `message`?",
      "criteria": {
        "hvac": "Heating, cooling, ventilation or air-conditioning problems",
        "housekeeping": "Cleaning, towels, linen, room tidiness",
        "front_desk": "Bookings, billing, check-in and general requests",
        "unknown": "No clear request"
      }
    },
    "has_deadline": {
      "type": "noul",
      "instructions": "Does `message` state an explicit deadline or time limit?"
    },
    "severity": {
      "type": "score",
      "instructions": "How severe is the impact described in `message`?",
      "criteria": ["No impact stated", "Minor inconvenience for one guest", "Many guests or an event affected", "Safety risk"]
    }
  }
}
```

วิธีอ่านสิ่งที่ได้กลับมา:

- **choice** → `department` คือผู้ชนะ แถบแสดงว่า probability ถูกแบ่งกันอย่างไร ถ้าสองแถบใกล้กัน แปลว่าโมเดลลังเล — โค้ดของคุณควรสังเกตเรื่องนี้
- **noul** → `has_deadline` ใกล้ 1 แปลว่า "ใช่ ชัดเจน" ใกล้ **0.5 แปลว่าไม่แน่ใจ** ไม่ใช่ "มีกำหนดเวลาครึ่งหนึ่ง"
- **score** → `severity` เช่น `2.1` แปลว่า "ส่วนใหญ่เป็นระดับ 2 เอียงไปทาง 3 นิดหน่อย" เลขระดับเริ่มที่ **0**
- **confidence** (เฉพาะ choice และ score) → probability กระจุกตัวแค่ไหน: 1.0 = รวมอยู่ที่ตัวเลือกเดียว 0.0 = แบนราบเท่ากันหมด มัน *ไม่ใช่* คำรับประกันว่าถูกต้อง

ทีนี้ลองทดลอง — เปลี่ยนทีละอย่าง แล้วถามใหม่:

1. ลบ `"before the 6pm wedding"` ออกจากข้อความ `has_deadline` เปลี่ยนไปอย่างไร?
2. เปลี่ยนข้อความเป็น `"Can I get extra towels?"` แผนกไหนชนะ และมั่นใจแค่ไหน?
3. ลบตัวเลือก `"unknown"` ออก แล้วส่ง `"hello"` ตอนนี้ probability ไปอยู่ที่ไหน? (บทเรียน: ให้ทางโมเดลบอกว่า *ไม่ตรงกับข้อไหนเลย* เสมอ)

Jev รับข้อความภาษาไทยได้เช่นกัน (แม้ภาษาอังกฤษจะแม่นที่สุด) ลองคำขอเดียวกันด้วยข้อความภาษาไทย:

```jev
{
  "state": {"message": "ล็อบบี้เย็นจัดตั้งแต่เช้า แขกบ่นกันเยอะมาก ช่วยแก้ให้เสร็จก่อนงานแต่งงานหกโมงเย็นด้วย!"},
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle `message`?",
      "criteria": {
        "hvac": "Heating, cooling, ventilation or air-conditioning problems",
        "housekeeping": "Cleaning, towels, linen, room tidiness",
        "front_desk": "Bookings, billing, check-in and general requests",
        "unknown": "No clear request"
      }
    },
    "has_deadline": {
      "type": "noul",
      "instructions": "Does `message` state an explicit deadline or time limit?"
    },
    "severity": {
      "type": "score",
      "instructions": "How severe is the impact described in `message`?",
      "criteria": ["No impact stated", "Minor inconvenience for one guest", "Many guests or an event affected", "Safety risk"]
    }
  }
}
```

เทียบกับคำตอบภาษาอังกฤษด้านบน: แผนกเหมือนกันไหม? ค่า probability ต่างกันแค่ไหน? งานภาษาไทยจริงควรวัดผลแยกเสมอ (Lab 04 และ 09 จะวัดให้ดู)

✓ Checkpoint: คุณเปลี่ยน state อย่างน้อยสองครั้ง และเห็น `choice`, `noul` และ `score` ตอบสนองทั้งหมด

## 5 · อ่านคำตอบแบบที่โปรแกรมอ่าน — lab 02

Lab 02 เรียกแบบสามคำถามเดียวกันผ่าน `jevkit` (ตัวช่วยเล็ก ๆ ที่แล็บถัด ๆ ไปใช้ทั้งหมด) พิมพ์ผลพร้อมแถบ probability แล้วแสดงให้เห็นว่า **โค้ดของคุณ** เปลี่ยนคำตอบแบบมีชนิดให้เป็นการตัดสินใจได้อย่างไร — เป็นกฎการส่งต่อ (routing rule) ที่คุณอ่าน ทดสอบ และแก้ได้โดยไม่ต้องแตะโมเดล

```bash
.venv/bin/python week24/01_hello_jev/labs/lab02_read_the_answer.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 1 · ask three typed questions about one guest message
» department             choice  → hvac   (confidence 1.00)
      hvac                       1.00  ████████████████████████ ◀
      unknown                    0.00  ░░░░░░░░░░░░░░░░░░░░░░░░
      housekeeping               0.00  ░░░░░░░░░░░░░░░░░░░░░░░░
      front_desk                 0.00  ░░░░░░░░░░░░░░░░░░░░░░░░
» has_deadline           noul    0.98  ████████████████████████  likely YES
» severity               score   2.00 on 0…3   (confidence 1.00)
      0 No impact stated                        0.00  ░░░░░░░░░░░░░░░░
      1 Minor inconvenience for one guest       0.00  ░░░░░░░░░░░░░░░░
      2 Many guests or an event affected        1.00  ████████████████
      3 Safety risk                             0.00  ░░░░░░░░░░░░░░░░
◆ jev-1.13.0 · LIVE · 478 input tok · $0.000020 · 454 ms
▣ STEP 2 · your code decides — Jev only supplied the judgments
→ route to hvac queue · priority P1 (event affected + explicit deadline)
═ execute: False — this lab never pages anyone; it prints the proposal.
```

`jevkit` คือ Python ธรรมดาประมาณ 350 บรรทัดใน `week24/common/jevkit.py`: `ask()` ส่งคำขอ `show()` พิมพ์แถบ และ `choice()/noul()/score()` สร้างคำถาม ไม่มีอะไรลึกลับ — เปิดอ่านได้เลย

✓ Checkpoint: คุณชี้ได้ว่าคำสั่ง `if` บรรทัดไหนใน lab 02 ที่เปลี่ยนคำตอบของ Jev ให้เป็นระดับความสำคัญ (priority)

## แล็บ — รันได้ที่นี่

**labs/lab01_first_call_raw.py** — การเรียกครั้งแรกด้วย standard library ล้วน ๆ: คำขอ คำตอบดิบ และตัวเลขหนึ่งตัว

**labs/lab02_read_the_answer.py** — คำถามพื้นฐานทั้งสามชนิดในการเรียกครั้งเดียว พิมพ์ให้อ่านง่าย แล้วโค้ดธรรมดาแปลงเป็นการตัดสินใจส่งต่อ

ทั้งสองรันแบบ LIVE เมื่อมีคีย์ หรือแบบ DRY (คำตอบที่บันทึกไว้) ด้วยสวิตช์โหมดบนแถบด้านบน

## ลองทำเอง

**แบบฝึกหัด 01 — คำถามชุดแรกของคุณ** เปิด `week24/01_hello_jev/exercises/ex01_my_first_questions.py` ในเอดิเตอร์ ในไฟล์มี `TODO` สามจุด:

1. เขียน `noul` ถามว่าข้อความจากแขกเป็น **คำร้องเรียน (complaint)** หรือไม่
2. เขียน `choice` ที่มี **บริเวณในห้องพัก** อย่างน้อยสามแห่ง (`bathroom`, `bedroom`, `balcony` และตัวเลือกทางออก (escape option))
3. เขียน `score` วัดว่าข้อความ **สุภาพแค่ไหน** โดยมีระดับเรียงลำดับอย่างน้อยสามระดับ

บันทึกไฟล์ แล้วกด **▶ Run** บนการ์ดแบบฝึกหัดด้านล่าง (หรือรันคำสั่ง) ตัวตรวจในตัวจะตรวจคำถามของคุณ *ก่อน* เสียเงินแม้แต่เซ็นต์เดียว จากนั้นถาม Jev เกี่ยวกับข้อความทดสอบสามข้อและพิมพ์คำตอบ

```bash
.venv/bin/python week24/01_hello_jev/exercises/ex01_my_first_questions.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
✓ is_complaint is a noul with instructions
✓ room_area is a choice with 4 options (includes an escape option)
✓ politeness is a score with 4 ordered levels

▣ asking Jev about 3 test messages …

━━ “What time does breakfast start?”
» is_complaint           noul    0.02  ░░░░░░░░░░░░░░░░░░░░░░░░  likely NO
» room_area              choice  → not_stated   (confidence 1.00)
» politeness             score   2.05 on 0…3   (confidence 0.92)
```

<details><summary>คำใบ้ — ตัวเลือกทางออกที่ดีเป็นแบบไหน?</summary>

ตั้งชื่อว่า `unknown` หรือ `not_stated` และอธิบายว่า *เมื่อไร* ควรเลือก: "The message does not mention a specific area of the room." ถ้าไม่มีตัวเลือกนี้ Jev ต้องบังคับจัดทุกข้อความเข้าบริเวณจริงสักแห่ง — และมันจะทำอย่างมั่นใจด้วย

</details>

<details><summary>ท้าทายเพิ่ม — รวมคำถาม vs เรียกแยก</summary>

เรียก `ask()` สามครั้ง ครั้งละหนึ่งคำถาม แล้วเรียกอีกครั้งด้วยคำถามทั้งสามพร้อมกัน เทียบ `input_tokens` รวม ทำไมการเรียกแบบรวมถึงถูกกว่า? (คำใบ้: state ถูกอ่านเข้าไปครั้งเดียว)

</details>

✓ Checkpoint: บรรทัดตรวจทั้งสามเป็น ✓ และ Jev ตอบครบทุกข้อความทดสอบ

## แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `HTTP 401` | ไม่มีคีย์หรือคีย์ผิด ตรวจด้วย `grep -c '^TYPESAFE_API_KEY=' .env` และดูว่า source คีย์ในเชลล์ *นี้* แล้ว |
| `HTTP 422` | รูปแบบ JSON ผิด — เช่น `choice` ที่ไม่มี `criteria` หรือ `score` ที่ `criteria` ไม่ใช่ array เนื้อหาข้อผิดพลาดจะบอกชื่อฟิลด์ |
| `HTTP 429` / `529` | ถูกจำกัดอัตรา (rate limit) หรือระบบรับภาระเกิน — `jevkit` จะรอแล้วลองใหม่สูงสุด 3 ครั้ง ให้รอสักนาที |
| `◈ DRY … placeholder` | คุณอยู่ในโหมด DRY และแก้คำขอ จึงไม่มีคำตอบที่บันทึกไว้ สลับแถบด้านบนเป็น ⚡ Live |
| `curl: command not found` ในเทอร์มินัล | ใช้ lab 01 แทน — เป็นการเรียกแบบเดียวกันด้วย Python |

## ถัดไป

ไปต่อที่ [Lab 02 — คำถามพื้นฐานสามชนิดแบบเจาะลึก](../02_three_primitives/TUTORIAL.th.md): เมื่อไรควรใช้ `choice` vs `noul` vs `score` และ `confidence` วัดอะไรจริง ๆ
