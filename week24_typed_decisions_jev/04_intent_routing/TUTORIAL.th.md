# ▶ Jev Lab 04 — การจัดเส้นทางตามเจตนา (intent routing) สำหรับ Alto Copilot

> ส่วนหนึ่งของ Week 24 · การตัดสินใจแบบมีชนิดข้อมูล (typed decisions) ด้วย Jev นี่คือ workflow จริงชิ้นแรกของคุณ: จัดประเภทคำขอแต่ละรายการที่ส่งเข้า Copilot ตัดสินว่าผู้เชี่ยวชาญคนไหน (หรือทีมผู้เชี่ยวชาญชุดไหน) ควรเป็นคนตอบ แล้วเลือก prompt ที่ผ่านการอนุมัติแล้ว โดยที่ Jev ไม่เคยเขียนคำตอบเองและไม่เคยให้สิทธิ์ใด ๆ

**พูดง่าย ๆ คือ:** ผู้ใช้พิมพ์คำขอสารพัดแบบเข้ามาใน copilot ของอาคาร เช่น *"ทำไม AHU-3 ไม่เย็น?"*, *"คำนวณผลการประหยัดของเดือนที่แล้ว"*, *"เขียนฟังก์ชัน Python ให้หน่อย"* ก่อนที่ AI ตัวใดจะเขียนคำตอบ ต้องมีบางอย่างตัดสินก่อนว่า **ใครควรเป็นคนตอบ**: ผู้เชี่ยวชาญ HVAC, นักวิเคราะห์การประหยัดพลังงาน, ผู้ช่วยเขียนโค้ด หรือส่งให้คน เพราะคำขอกำกวมเกินไป Jev ตัดสินเรื่องนี้ได้ในเวลาไม่ถึงวินาที จากนั้นโค้ดธรรมดาจะใช้กฎที่เขียนไว้และเลือกชุดคำสั่งที่อนุมัติไว้ล่วงหน้า (ที่เรียกว่า *prompt*) ให้ผู้เชี่ยวชาญคนนั้น

**คำศัพท์ใหม่:**

| คำ | ความหมาย |
|---|---|
| intent / route | คำขอนี้เป็นคำขอประเภทไหน และจึงควรส่งให้ตัวจัดการใด |
| MAS | *multi-agent system*: AI ผู้เชี่ยวชาญหลายตัวทำงานร่วมกันกับคำขอเดียว |
| confidence gate | กฎประมาณว่า "ส่งต่ออัตโนมัติเฉพาะเมื่อ Jev มั่นใจชัดเจน ไม่อย่างนั้นให้ถามคน" |
| compute tier | ใช้โมเดลที่เร็วและถูกกับคำของ่าย ๆ หรือโมเดลที่ใช้การให้เหตุผล (reasoning) ซึ่งช้ากว่ากับคำขอยาก |
| prompt bundle | ชุดคำสั่งระบบ (system instructions) ที่อนุมัติแล้วของผู้เชี่ยวชาญแต่ละคน เก็บไว้ในโค้ด ไม่มีการแต่งขึ้นใหม่ตอนรันจริง |

**สิ่งที่คุณจะได้ลงมือทำจริง**
- จัดเส้นทางคำขอภาษาอังกฤษ/ไทย 9 รายการ แต่ละรายการใช้การเรียก 1 ครั้งที่มี 7 คำถาม โดยใช้ `jev_lab.py` ต้นแบบ
- อ่าน policy ที่ทำงานแบบกำหนดแน่นอน (deterministic): confidence gate → route, `needs_mas` → การส่งงาน (dispatch), ความซับซ้อน → compute tier
- แปลง route ให้เป็น **prompt bundle** ที่อนุมัติแล้ว จาก registry ในโค้ด
- วัดผลภาษาอังกฤษเทียบกับภาษาไทยด้วยคำขอที่จับคู่กัน แทนการเดาเอาเอง
- เพิ่ม route ใหม่เอี่ยม (`iot_connectivity`) และพิสูจน์ว่ามันใช้งานได้

**Time** ~40 นาที · **Difficulty** ระดับกลาง · **Cost** ≈ $0.0008 เมื่อ live · $0 เมื่อ dry

## 0 · แพทเทิร์น: ตัวจัดประเภทแบบมีชนิดข้อมูลวางไว้หน้าผู้เชี่ยวชาญ

TypeSafe เรียกแพทเทิร์นนี้ว่า **intent routing**: ตัวจัดประเภท (classifier) แบบมีชนิดข้อมูลที่ทำงานเร็ว ตัดสินว่าคำขอควรไป *ที่ไหน* ส่วนงานจริงให้ตัวจัดการแบบ deterministic, LLM ผู้เชี่ยวชาญ หรือคนเป็นผู้ทำ

```text
user request (EN / TH)
  → code: authenticate user, resolve tenant/project permissions, strip secrets
  → Jev: topic + needs_retrieval + needs_live_data + needs_mas + expertise + complexity   (ONE call)
  → code: confidence gate + policy  → route · dispatch · compute tier
  → registry: approved system prompt + allowed tools for that route
  → specialist LLM / multi-agent dispatcher / human — each re-checks authorization
```

มีกฎสองข้อที่ทำให้แพทเทิร์นนี้ปลอดภัย:

1. **Jev เสนอ โค้ดเป็นผู้ตัดสิน** ทุกผลลัพธ์ในแล็บนี้มี `execute: false` และ `authorization: must_be_checked_separately`
2. **label ไม่เคยให้สิทธิ์ใด ๆ** การถูกส่งไปที่ `energy_mv` ไม่ได้ทำให้ผู้ใช้อ่านมิเตอร์ของ tenant อื่นได้

label หัวข้อด้านล่างเป็นข้อเสนอของบทเรียนนี้ ไม่จำเป็นต้องตรงกับ schema จริงของ Alto Copilot บน production:

| Label | ความหมาย |
|---|---|
| `hvac` | HVAC, ชิลเลอร์, AHU, VRF, ประสิทธิภาพการทำความเย็นหรือการวินิจฉัยปัญหา |
| `energy_mv` | baseline พลังงาน การวัดหรือการตรวจสอบผลการประหยัด (M&V) |
| `facility_ops` | งานซ่อมบำรุง การปฏิบัติงานหน้างาน ใบสั่งงาน (work order) |
| `sustainability` | คาร์บอน การปล่อยก๊าซ หลักฐานหรือรายงานด้านความยั่งยืน |
| `coding` | การพัฒนาซอฟต์แวร์ การดีบัก หรือการเชื่อมต่อ API |
| `general` | คำถามทั่วไปที่ไม่เกี่ยวกับอาคาร การเขียน การสนทนา |
| `unknown` | กำกวมเกินไป ไม่รองรับ หรือไม่มีคำขอที่ระบุได้ |

✓ Checkpoint: คุณบอกได้ว่าส่วนไหนเป็นผู้ตัดสิน route (โค้ด) และส่วนไหนเป็นผู้ให้การตัดสินใจเชิงความหมาย (Jev)

## 1 · จัดเส้นทางคำขอเก้ารายการ

ตัวต้นแบบ (reference implementation) จากคู่มือต้นทางอยู่ที่ `week24/jev_lab/jev_lab.py` (ใช้แค่ standard library) แล็บ 01 นำเข้า `QUESTIONS["intent"]` (choice 1 ข้อ + noul 5 ข้อ + score 1 ข้อ), `prepare()` (ส่งเฉพาะฟิลด์ `message` ไม่เคยส่ง label ที่ใช้สอน) และ `policy()` ของมัน

```bash
.venv/bin/python week24/04_intent_routing/labs/lab01_route_requests.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 1 · look at ONE request in full — 7 questions, 1 call
» intent                 choice  → hvac   (confidence 1.00)
» needs_retrieval        noul    0.91  ██████████████████████░░  likely YES
» needs_live_data        noul    0.92  ██████████████████████░░  likely YES
» needs_mas              noul    0.04  █░░░░░░░░░░░░░░░░░░░░░░░  likely NO
» hvac_expertise         noul    0.80  ███████████████████░░░░░  likely YES
» mv_expertise           noul    0.42  ██████████░░░░░░░░░░░░░░  uncertain
» complexity             score   1.09 on 0…2   (confidence 0.46)
◆ jev-1.13.0 · LIVE · 858 input tok · $0.000036 · 450 ms

▣ STEP 2 · route all nine requests (9 calls)
│ id  message                                           teaching label  jev             conf  route              dispatch         tier
│ i1  Why is AHU-3 not cooling the hotel lobby? Check…  hvac            hvac            1.00  hvac               single_agent     reasoning_candidate
│ i2  ช่วยตรวจสอบว่า AHU-3 ไม่เย็นเพราะอะไร             hvac            hvac            1.00  hvac               single_agent     reasoning_candidate
│ i3  Compare an HVAC engineer's diagnosis with an M&…  hvac            energy_mv       0.83  energy_mv          mas_candidate    reasoning_candidate
│ i4  Write a Python function to validate a JSON payl…  coding          coding          1.00  coding             single_agent     reasoning_candidate
│ i5  Tell me a short story about a cat.                general         general         1.00  general            single_agent     fast_candidate
│ …
│ i9  Please do that thing.                             unknown         unknown         0.96  clarify_or_review  review_dispatch  fast_candidate
◆ Jev agreed with the teaching label on 8/9 rows — read the disagreements, don't just count them
```

การรันแบบเดียวกันนี้มีให้ใช้เป็น CLI ที่พิมพ์ผลเป็น JSON ทีละบรรทัดด้วย ซึ่งเป็นสคริปต์ตัวเดียวกับในคู่มือต้นทาง:

```bash
.venv/bin/python week24/jev_lab/jev_lab.py run intent --live --limit 9
```

### เรื่องของ i3 — เมื่อ label "เฉลย" ก็ยังถกเถียงได้

คำขอ i3 เขียนว่า *"Compare an HVAC engineer's diagnosis with an M&V analyst's savings assessment and resolve their disagreement."* (เปรียบเทียบผลวินิจฉัยของวิศวกร HVAC กับผลประเมินการประหยัดของนักวิเคราะห์ M&V แล้วหาข้อยุติเมื่อทั้งสองเห็นไม่ตรงกัน) ไฟล์สำหรับสอนติด label ไว้ว่า `hvac` แต่ Jev ตอบว่า **`energy_mv`** (probability 0.87, confidence 0.83) โดย `hvac` ได้ 0.12, `needs_mas` = **0.98** และทั้ง `hvac_expertise` (0.82) กับ `mv_expertise` (0.92) สูง *ทั้งคู่* (ตัวเลขของคุณอาจต่างไปหนึ่งถึงสองจุดระหว่างการเรียกแบบ live แต่ละครั้ง)

ใครถูก? พูดได้ว่าไม่มีหัวข้อไหนถูกเพียงลำพัง นี่คืองานที่ต้องใช้ผู้เชี่ยวชาญสองคน และ policy จัดการได้ดี: `dispatch = mas_candidate` คู่มือต้นทางเตือนเรื่องนี้ไว้ตรง ๆ ว่า *แม้แต่ label เฉลยก็อาจต้องมีการตัดสินชี้ขาด* เมื่อผลประเมินบอกว่า "ผิด" ให้ดูให้ดีก่อนจะไป "แก้" โมเดล

✓ Checkpoint: คุณรันแล็บ 01 แล้ว และอธิบายได้ว่าทำไม i3 จึงไปที่ `mas_candidate`

## 2 · อ่าน policy — โค้ดธรรมดาที่ทดสอบได้

ทุกอย่างหลังการเรียก Jev คือ Python แบบ deterministic ใน `jev_lab.py` ไม่มีโมเดลใดเปลี่ยนมันได้

| การตัดสินใจ | กฎ (ตัวอย่าง ยังไม่ได้ปรับเทียบ) | เหตุผล |
|---|---|---|
| **route** | ใช้หัวข้อนั้น เฉพาะเมื่อ `accepted()` ผ่าน: confidence ≥ .75, ค่าสูงสุด (peak) ≥ .80, ส่วนต่าง (margin) ≥ .20 และไม่ใช่ `unknown` ไม่อย่างนั้นเป็น `clarify_or_review` | คำขอที่ไม่แน่ชัดต้องไม่เดาผู้เชี่ยวชาญ |
| **dispatch** | `needs_mas` ≤ .20 → `single_agent` · ≥ .80 → `mas_candidate` · นอกนั้น `review_dispatch` | 0.5 คือความไม่แน่ใจ ไม่ใช่ "ครึ่งทีม" |
| **specialists** | noul ด้านความเชี่ยวชาญแต่ละข้อ ≥ .80 | ข้อเสนอสำหรับตัวส่งงานแบบหลายเอเจนต์ |
| **compute tier** | complexity ≥ 1.5 **หรือ** confidence ของมัน < .75 → `reasoning_candidate` นอกนั้น `fast_candidate` | เมื่อไม่แน่ใจว่างานยากแค่ไหน ยอมจ่ายเพื่อใช้โมเดลที่ใหญ่กว่า |

สังเกตในตารางด้านบนว่าคำขอส่วนใหญ่ได้ `reasoning_candidate` (มีเพียง i5 และ i9 ที่ได้ `fast_candidate`) ไม่ใช่เพราะมันยาก แต่เพราะ **confidence ของ score** ความซับซ้อนต่ำกว่า .75 (i1: 0.46) นี่คือทางเลือกเชิง policy ที่คุณอาจกลับมาทบทวน: มันยอมจ่ายเงินเพื่อความปลอดภัย และเป็น threshold (เกณฑ์) แบบที่คุณปรับจูนกับทราฟฟิกจริงได้ *โดยไม่ต้องถามโมเดลใหม่*

ลองใช้ router เต็มรูปแบบกับคำขอของคุณเอง:

```jev
{
  "state": {"message": "Compare an HVAC engineer's diagnosis with an M&V analyst's savings assessment and resolve their disagreement."},
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Classify the primary topic of `message`.",
      "criteria": {
        "hvac": "HVAC, chiller, AHU, VRF, cooling performance or diagnosis.",
        "energy_mv": "Energy baselines, savings measurement or verification.",
        "facility_ops": "Maintenance workflow, site operations, work orders.",
        "sustainability": "Carbon, emissions, sustainability evidence or reporting.",
        "coding": "Software implementation, debugging or API integration.",
        "general": "Ordinary non-building questions, writing or conversation.",
        "unknown": "Too vague, unsupported topic, or no identifiable request."
      }
    },
    "needs_mas": {"type": "noul", "instructions": "Does `message` explicitly require combining distinct expert analyses, comparing independent judgments, or resolving their disagreement? Merely having multiple steps is not enough."},
    "needs_live_data": {"type": "noul", "instructions": "Does answering `message` require current or historical operational records from connected systems?"},
    "complexity": {"type": "score", "instructions": "How much reasoning does the request require?", "criteria": ["Simple answer or lookup.", "One specialist with a few steps.", "Multiple specialist analyses or substantial investigation."]}
  }
}
```

ลองทดลอง:

1. เปลี่ยนข้อความเป็น `"Schedule preventive maintenance for the chiller next week."` ในการรันของเรา คำขอนี้ไปที่ **`hvac` (0.73)** ไม่ใช่ `facility_ops` คำว่า *chiller* ดึงมันไป แบบนี้ผิดไหม? ตัดสินใจก่อน แล้วเขียนเส้นแบ่งลงในคำอธิบายของ `facility_ops` ("maintenance *scheduling* for any equipment, including chillers")
2. ลอง `"Please do that thing."` — `unknown` ควรชนะ แล้ว policy จะขอให้ผู้ใช้อธิบายเพิ่ม

✓ Checkpoint: คุณชี้ได้ว่าบรรทัดไหนของ policy ที่ส่งคำขอกำกวมไปที่ `clarify_or_review`

## 3 · จาก route สู่ prompt bundle ที่อนุมัติแล้ว

Jev เลือก **กุญแจ (key)** ส่วน registry ของคุณเป็นผู้ให้ **prompt** registry คือโค้ดที่คุณรีวิวเหมือนโค้ดอื่น ๆ ตัวจัดประเภทไม่มีทางแทรก prompt หรือเครื่องมือใหม่เข้ามาได้

```bash
.venv/bin/python week24/04_intent_routing/labs/lab02_prompt_bundle.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 1 · “Why is AHU-3 not cooling the hotel lobby? Check yesterday's trends.”
» intent hvac (confidence 1.00) → route hvac
{
  "agent_candidate": "hvac_expert",
  "system_prompt_candidate": "Separate observed symptoms from hypotheses. Use authorized building data. State missing evidence. Propose inspections only; do not change controls.",
  "compute_tier_candidate": "reasoning_candidate",
  "dispatch_candidate": "single_agent",
  "specialists_suggested": ["hvac_expert"],
  "tools": "resolve separately from the authorized registry",
  "execute": false
}
…
▣ STEP 3 · “Please do that thing.”
» intent unknown (confidence 0.95) → route clarify_or_review
→ no prompt selected automatically — ask the user to clarify, or send to review
```

(คำขอกำกวมได้ confidence 0.96 ในแล็บ 01 และ 0.95 ที่นี่: เป็นการเรียกแบบ live สองครั้งด้วยคำขอเดียวกัน Jev สม่ำเสมอมากแต่ไม่ได้ให้ผลเหมือนเดิมทุกบิต อีกเหตุผลหนึ่งว่าทำไม threshold ไม่ควรวางไว้บนเส้นบาง ๆ)

ลำดับการเชื่อมต่อ Alto Copilot ที่เสนอไว้ในคู่มือต้นทาง:

1. ยืนยันตัวตนผู้ใช้และตรวจสิทธิ์ระดับ tenant/project ในโค้ดฝั่งเซิร์ฟเวอร์
2. ตัดบริบทที่ไม่เกี่ยวข้องและข้อมูลลับออก ก่อนส่ง input ให้ตัวจัดประเภท
3. รับสัญญาณหัวข้อ การค้นเอกสาร ข้อมูล live ความซับซ้อน และ MAS
4. ใช้ policy แบบ deterministic และ uncertainty gate ที่ผ่านการทดสอบแล้ว
5. จับคู่ระดับโมเดลเชิงตรรกะกับ registry ของโมเดลที่อนุญาต งบประมาณ และข้อจำกัดเรื่องที่ตั้งของข้อมูล (data residency)
6. ส่งงานให้ผู้เชี่ยวชาญคนเดียว หรือส่ง MAS candidate ให้ตัวประสานงานหลายเอเจนต์ (multi-agent coordinator) แยกต่างหากที่มีสิทธิ์และขั้นตอนอนุมัติของตัวเอง
7. ตรวจสิทธิ์ซ้ำทุกครั้งที่ค้นข้อมูลหรือเรียกเครื่องมือ label จากการจัดประเภทไม่เคยให้สิทธิ์
8. บันทึก route, เวอร์ชันโมเดล, เวอร์ชัน rubric, latency และผลลัพธ์ที่ถูกแก้ไขแล้ว

✓ Checkpoint: คุณอธิบายได้ว่าทำไมข้อความ prompt ถึงอยู่ใน registry ไม่ใช่อยู่ในคำตอบของ Jev

## 4 · ภาษาไทยเทียบภาษาอังกฤษ — วัดผล อย่าเดา

ภาษาอังกฤษคือภาษาหลักที่ใช้ฝึก Jev ผู้ใช้ Alto Copilot เขียนทั้งไทย อังกฤษ และปนกัน แล็บ 03 ส่งคำขอเดียวกันในทั้งสองภาษา แล้วเปรียบเทียบหัวข้อ confidence และ **total variation distance** (TVD) ระหว่างการแจกแจงความน่าจะเป็นสองชุด (0 = เหมือนกัน, 1 = ไม่ทับซ้อนกันเลย)

```bash
.venv/bin/python week24/04_intent_routing/labs/lab03_thai_vs_english.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
│ Thai / mixed message                              EN topic        TH topic        confidence EN→TH  TVD   live-data EN→TH
│ ทำไม AHU-3 ไม่เย็นที่ล็อบบี้โรงแรม                hvac            hvac            1.00 → 1.00       0.00  0.84 → 0.75
│ คำนวณผลการประหยัดไฟฟ้าที่ผ่านการตรวจสอบ เทียบกั…  energy_mv       energy_mv       1.00 → 1.00       0.00  0.87 → 0.85
│ นัดบำรุงรักษาเชิงป้องกันชิลเลอร์สัปดาห์หน้า       hvac            hvac            0.73 → 0.84       0.09  0.52 → 0.43
│ สรุปการปล่อยคาร์บอนของโรงแรมสำหรับรายงาน ESG      sustainability  sustainability  1.00 → 1.00       0.00  0.91 → 0.85
│ ช่วย check chiller plant kW/ton ของเมื่อวานหน่อย  hvac            hvac            1.00 → 0.99       0.00  0.94 → 0.93
◆ same topic in 5/5 pairs
```

ผลน่าพอใจ: หัวข้อตรงกัน 5/5 และแม้แต่ข้อความที่ปนไทยกับอังกฤษก็ยังนิ่ง แต่สังเกตว่า **ค่า noul ขยับลงเล็กน้อย** ในภาษาไทย (`needs_live_data` 0.84 → 0.75 สำหรับคำถามเรื่อง AHU) ถ้าตั้ง threshold ไว้ที่ 0.80 การขยับเท่านี้ก็พลิกการตัดสินใจได้ ห้าคู่ไม่ได้พิสูจน์อะไรในทางสถิติ มันบอกคุณแค่ว่า *ควรวัดอะไร* กับข้อความจริงที่ตัดข้อมูลระบุตัวตนออกแล้วหลายร้อยข้อความ

```jev
{
  "state": {"message": "ช่วยตรวจสอบว่า AHU-3 ไม่เย็นเพราะอะไร ดูเทรนด์เมื่อวานด้วย"},
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Classify the primary topic of `message`.",
      "criteria": {
        "hvac": "HVAC, chiller, AHU, VRF, cooling performance or diagnosis.",
        "energy_mv": "Energy baselines, savings measurement or verification.",
        "facility_ops": "Maintenance workflow, site operations, work orders.",
        "unknown": "Too vague, unsupported topic, or no identifiable request."
      }
    },
    "needs_live_data": {"type": "noul", "instructions": "Does answering `message` require current or historical operational records from connected systems?"}
  }
}
```

ลองทดลอง: ลบ "ดูเทรนด์เมื่อวานด้วย" ออก `needs_live_data` ลดลงไหม? จากนั้นเขียนเวอร์ชันภาษาอังกฤษแล้วเปรียบเทียบกัน

✓ Checkpoint: คุณบอกชื่อตัวชี้วัดที่บอกว่าการแจกแจงสองชุดต่างกัน (TVD) ได้ และอธิบายได้ว่าทำไมการขยับเล็กน้อยจึงสำคัญเมื่ออยู่ใกล้ threshold

## 5 · Shadow mode — เส้นทางสู่ production

อย่าเปิดใช้การจัดเส้นทางเพียงเพราะผลใน notebook ดูดี การใช้งานจริงครั้งแรกที่คู่มือต้นทางแนะนำคือ **shadow mode**: รัน router ของ Jev คู่ไปกับ router เดิม บันทึกข้อเสนอของทั้งสองฝั่ง ไม่เปลี่ยนอะไรในระบบจริง แล้วเปรียบเทียบแยกตามคลาสและตามภาษา

| ระยะ | สิ่งที่เปลี่ยนสำหรับผู้ใช้ | เงื่อนไขในการผ่านไปขั้นต่อไป |
|---|---|---|
| แล็บ (โมดูลนี้) | ไม่มี | ทุกคนอธิบายฟิลด์ ความไม่แน่นอน และสิทธิ์ได้ |
| Shadow | ไม่มี — router ทั้งสองแค่บันทึก | ผลวัดแยกตามคลาสทั้ง EN/TH นิ่ง และข้อผิดพลาดทุกกรณีปลอดภัยเพราะผ่านการรีวิว |
| Assisted | แสดงข้อเสนอให้ผู้ปฏิบัติงานเห็น | มีข้อมูลการที่คนเข้าไปแก้ (override) และ audit trail |
| Router integration | Jev ทำงานคู่กับ router เดิมและตัวประสานงานหลายเอเจนต์ | ได้ตามเป้าคุณภาพ ต้นทุน และ latency บนชุดข้อมูลที่กันไว้ทดสอบ |

✓ Checkpoint: คุณอธิบายได้ว่าทำไม shadow mode ไม่เปลี่ยนอะไรสำหรับผู้ใช้เลย แต่ยังได้หลักฐานที่คุณต้องการ

## Labs — รันได้ที่นี่

**labs/lab01_route_requests.py** — คำขอ EN/TH เก้ารายการผ่าน schema intent 7 คำถามและ policy ของตัวต้นแบบ

**labs/lab02_prompt_bundle.py** — Route → system prompt ที่อนุมัติแล้วจาก registry ในโค้ด คำขอกำกวมจะตกไปที่การขอให้อธิบายเพิ่ม

**labs/lab03_thai_vs_english.py** — คำขอภาษาอังกฤษ/ไทย/ปนกันแบบจับคู่: ความตรงกันของหัวข้อ confidence และระยะห่างของการแจกแจง

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 04 — เพิ่ม route** Alto Copilot ได้รับคำถามแนว "gateway ออฟไลน์ / เซนเซอร์ไม่ส่งข้อมูล" เยอะมาก เปิดไฟล์ `week24/04_intent_routing/exercises/ex04_add_a_route.py`:

1. **ส่วน A (ออฟไลน์ ไม่เสียค่าใช้จ่าย):** เขียนฟังก์ชัน `accepted(a)` ซึ่งก็คือ confidence gate ให้ผ่าน fixture ตายตัวทั้งห้าชุด
2. **ส่วน B:** เขียนคำอธิบายของตัวเลือกใหม่ `iot_connectivity` (อะไรเข้าข่าย และอะไร **ไม่** เข้าข่าย) พร้อมข้อความทดสอบของคุณเองสามข้อความ ลองเขียนหนึ่งข้อความเป็นภาษาไทย
3. **ส่วน C (อัตโนมัติ):** ตัวตรวจจะส่งข้อความของคุณ ข้อความเรื่อง gateway ที่ซ่อนไว้สองข้อความ และข้อความควบคุมสองข้อความที่ต้องอยู่ที่ `hvac` และ `coding` เหมือนเดิม

```bash
.venv/bin/python week24/04_intent_routing/exercises/ex04_add_a_route.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP A · your accepted() gate against 5 fixed fixtures (offline, $0)
✓ case 1: accepted → True
…
✓ case 5: accepted → True
▣ STEP C · route 7 messages (one call each)
│ kind     message                                           want              jev               conf  accepted?
│ yours    The Modbus gateway on the chiller plant keeps d…  iot_connectivity  iot_connectivity  0.98  yes        ✓
│ yours    เกตเวย์ที่โรงแรมสาขาภูเก็ตออฟไลน์ ข้อมูลไม่เข้า…  iot_connectivity  iot_connectivity  1.00  yes        ✓
│ hidden   The LoRaWAN gateway at Siam Tower has been offl…  iot_connectivity  iot_connectivity  1.00  yes        ✓
│ control  Why is AHU-3 not cooling the hotel lobby?         hvac              hvac              1.00  yes        ✓
✓ hidden gateway messages route to iot_connectivity, controls stay put
```

<details><summary>คำใบ้ — คำอธิบายที่ไม่แย่งปัญหา HVAC ไป</summary>

"Gateways, IoT sensors or meters that are offline, not reporting, or sending stale or missing data. **Not** for equipment that is reporting normally but performing badly (that is hvac)." ประโยคที่สองคือสิ่งที่ทำให้ "AHU-3 is not cooling" ยังอยู่ที่ `hvac`

</details>

<details><summary>คำใบ้ — กฎเรื่องส่วนต่าง (margin)</summary>

ถ้าความน่าจะเป็นรวมกันได้ 1 และค่าสูงสุด ≥ 0.80 ตัวเลือกอันดับสองจะได้ไม่เกิน 0.20 ส่วนต่างจึงอย่างน้อย 0.60 กฎเรื่องส่วนต่างจะเริ่มมีผลก็ต่อเมื่อคุณ *ลด* threshold ของค่าสูงสุดลง แต่ให้เก็บไว้อยู่ดี มันบันทึกเจตนาไว้ และช่วยคุณเมื่อมีการเปลี่ยน threshold

</details>

✓ Checkpoint: gate ผ่านครบทั้งห้ากรณี และแถว hidden กับ control เป็น ✓ ทั้งหมด

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `ModuleNotFoundError: jev_lab` | รันจากโฟลเดอร์รากของ repo แล็บจะเพิ่ม `week24/jev_lab` เข้า `sys.path` ให้เอง |
| ทุกคำขอไปที่ `clarify_or_review` หมด | คุณอยู่ในโหมด DRY และแก้คำขอไปแล้ว (คำตอบ placeholder มี confidence เป็น 0) ให้สลับเป็น ⚡ Live |
| `jev_lab.py run … --live` บอกว่า "Set TYPESAFE_API_KEY" | โหลดคีย์เข้า shell: `set -a; source <(grep '^TYPESAFE_API_KEY=' .env); set +a` (terminal ที่อยู่ในแอปมีคีย์อยู่แล้ว) |
| `Output file already exists` | CLI ต้นแบบไม่ยอมเขียนทับไฟล์ผลลัพธ์ ให้ตั้งชื่อ `--out` ใหม่ |
| ผลภาษาไทยดูแย่กว่าเมื่อใช้ข้อมูลของคุณเอง | วัดผลแยกตามคลาส ปรับ rubric และเปรียบเทียบกับตัวจัดประเภทอื่น (แล็บ 10 พูดถึง Laya) |

## Next — ต่อไป

ไปต่อที่ [แล็บ 05 — คัดแยกอีเมลโดยไม่ยกการควบคุมกล่องจดหมายให้ AI](../05_email_triage/TUTORIAL.th.md)
