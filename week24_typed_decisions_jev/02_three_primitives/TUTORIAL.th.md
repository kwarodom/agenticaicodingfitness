# ▶ Jev Lab 02 — คำถามพื้นฐานสามชนิดแบบเจาะลึก: choice, noul, score

> ส่วนหนึ่งของ Week 24 · การตัดสินใจของ AI แบบมีชนิดด้วย Jev Lab 01 แสดงให้เห็น *ว่า* Jev ตอบกลับเป็นคำตอบแบบมีชนิด แล็บนี้สอน *วิธีอ่าน* คำตอบเหล่านั้น — และวิธีเลือกชนิดคำถามให้ถูกกับทุกความต้องการ

**สิ่งที่คุณจะได้ลงมือทำ**
- ดู `choice` แบ่ง probability เมื่อข้อความมีสองความต้องการ — และเรียนรู้วิธีจับสัญญาณนี้
- ใช้ `noul` กับข้อความแบบใช่/ไม่ใช่ รวมถึงหลายป้ายกำกับ (label) พร้อมกัน
- สร้าง `score` ที่มีระดับเป็นรูปธรรม แล้วแปลงเป็น threshold (เกณฑ์) ในโค้ด
- คำนวณ `confidence` ด้วยตัวเอง และพบว่า confidence ของ Score ต่างไปตรงไหน
- ตัดสินใจได้ว่าความต้องการใหม่แต่ละแบบ ควรใช้คำถามพื้นฐาน (primitive) ชนิดไหนในสามชนิด

**Time** ~30 นาที · **Difficulty** ระดับเริ่มต้น · **Cost** ≈ $0.0003 เมื่อรันจริง · $0 ในโหมด dry

## 0 · ทบทวนใน 30 วินาที

ทุกคำขอไปยัง Jev คือ `state` (ข้อเท็จจริง) + `questions` (map ที่ตั้งชื่อไว้) ทุกคำตอบมี `type` ตรงกับคำถามของมัน:

| Primitive | ถามว่า | ฟิลด์ในคำตอบ | มี `confidence` ไหม? |
|---|---|---|---|
| `choice` | ตัวเลือกไหน *หนึ่งเดียว* ที่เข้ากัน? | `choice`, `probabilities` (ต่อตัวเลือก รวม ≈ 1) | มี |
| `noul` | ข้อความนี้จริงไหม? | `noul` (0 → ไม่ใช่, 1 → ใช่) | ไม่มี |
| `score` | อยู่ตรงไหนบนสเกลที่เรียงไว้? | `score` (ระดับคาดหมาย), `probabilities` (ต่อระดับ), `legend` | มี |

กฎสามข้อที่ใช้กับทุกชนิด:

- **id ของคำถามเป็นของคุณ** `"department"` เป็นป้ายสำหรับโค้ดของคุณ — Jev ไม่เคยเห็นมัน ใส่ความหมายทั้งหมดไว้ใน `instructions` และ `criteria`
- **คำถามเป็นอิสระต่อกัน** ในคำขอเดียว แต่ละคำถามถูกตอบแยกกันบน state เดียวกัน คำถามหนึ่งอ่านคำตอบของอีกคำถามไม่ได้
- **มีชนิด ≠ ถูกต้อง** คำตอบที่รูปแบบสมบูรณ์ก็ยังผิดได้ โค้ดของคุณเป็นผู้ตรวจ กั้น และตัดสินใจ

✓ Checkpoint: คุณบอกได้โดยไม่ต้องดูว่า primitive ไหนไม่มีฟิลด์ `confidence` (noul)

## 1 · choice — ผู้ชนะหนึ่งเดียว แต่ให้อ่านการกระจายทั้งหมด

Choice เลือกหนึ่งตัวเลือกจากชุดที่คุณกำหนด (สูงสุด 255) คุณได้ทั้งผู้ชนะ **และ** probability ของทุกตัวเลือก การดูแค่ผู้ชนะจะซ่อนข้อมูลที่มีประโยชน์ที่สุดไว้

Lab 01 ถามคำถามเรื่องแผนกเดียวกันกับข้อความสามข้อ:

```bash
.venv/bin/python week24/02_three_primitives/labs/lab01_choice.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 2 · two_needs: “Our room is too hot and we also need fresh towels, please.”
» department             choice  → hvac   (confidence 0.34)
      hvac                       0.51  ████████████░░░░░░░░░░░░ ◀
      housekeeping               0.46  ███████████░░░░░░░░░░░░░
      unknown                    0.02  ░░░░░░░░░░░░░░░░░░░░░░░░
…
▣ STEP 4 · side by side — the margin between the top two options is the tell
│ message     winner   top p  2nd p  margin  confidence
│ clear       hvac     1.00   0.00   1.00    1.00
│ two_needs   hvac     0.51   0.46   0.05    0.34
│ no_request  unknown  0.99   0.01   0.98    0.99
```

ดูที่ `two_needs`: HVAC "ชนะ" ด้วย 0.51 แต่ housekeeping ได้ 0.46 นี่ไม่ใช่ความผิดพลาด — ข้อความนี้ต้องการทั้งสองทีมจริง ๆ โปรแกรมที่อ่านแค่ `choice` จะทิ้งเรื่องผ้าเช็ดตัวไปเงียบ ๆ **เมื่อส่วนต่าง (margin) ระหว่างสองอันดับแรกน้อย โค้ดของคุณควรสังเกต** (ส่วนที่ 2 แสดงเครื่องมือที่ดีกว่าสำหรับกรณี "จริงได้หลายข้อ")

ลองเอง — แก้ข้อความหรือตัวเลือก แล้วกด **⚡ Ask Jev**:

```jev
{
  "state": {"message": "Our room is too hot and we also need fresh towels, please."},
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which hotel team should handle `message`?",
      "criteria": {
        "hvac": "Heating, cooling, ventilation or air-conditioning problems",
        "housekeeping": "Cleaning, towels, linen, amenities, room tidiness",
        "front_desk": "Bookings, billing, check-in, check-out and general questions",
        "unknown": "The message contains no request for any of these teams"
      }
    }
  }
}
```

การทดลอง:

1. เพิ่มตัวเลือก `"multiple": "The message needs two or more different teams"` มันชนะไหม?
2. เปลี่ยนข้อความเป็น `"Thanks, we had a lovely stay!"` — ตัวเลือกทางออก `unknown` ควรได้เกือบทั้งหมด
3. ลบตัวเลือก `unknown` แล้วถามเรื่องข้อความขอบคุณอีกครั้ง ตอนนี้ Jev ต้องเลือกทีมจริงสักทีม — ดูมันทำแบบนั้น

✓ Checkpoint: คุณรัน lab 01 แล้ว และอธิบายได้ว่าทำไม `hvac 0.51 / housekeeping 0.46` เป็นสัญญาณ ไม่ใช่สัญญาณรบกวน

## 2 · noul — probability ที่ข้อความหนึ่งเป็นจริง

Noul (อ่านว่า "นู-อัล") ส่งตัวเลขกลับมาหนึ่งตัว: probability ที่ข้อความแบบใช่/ไม่ใช่ของคุณเป็นจริง คุณจะอธิบายเพิ่มด้วยก็ได้ว่า *จริง* และ *ไม่จริง* หมายถึงอะไร ผ่าน `criteria`

สามเรื่องที่มือใหม่มักเข้าใจผิด:

| ความเชื่อผิด | ความจริง |
|---|---|
| "0.5 แปลว่าปานกลาง" | 0.5 แปลว่า **ไม่แน่ใจ** — โมเดลแยกใช่กับไม่ใช่ไม่ออก |
| "noul หลายตัวเกี่ยวกับข้อความเดียวรวมกันได้ 1" | noul แต่ละตัวเป็นอิสระ สี่ป้ายกำกับอาจได้ 0.9 ทั้งหมดก็ได้ |
| "p(X) + p(ไม่ X) = 1" | ไม่รับประกัน ให้ถามแต่ละการตัดสินใจ **ทางเดียว** แล้วบังคับตรรกะในโค้ด |

```bash
.venv/bin/python week24/02_three_primitives/labs/lab02_noul.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 1 · same question, three messages: clear yes, clear no, unsure
│ The TV remote is not working.                     0.98  yes
│ What time is breakfast?                           0.01  no
│ The room is fine I guess, but it smells a bit d…  0.50  UNSURE → review

▣ STEP 2 · multi-label — a message can need SEVERAL teams: one noul per label
» needs_hvac             noul    0.97  ███████████████████████░  likely YES
» needs_housekeeping     noul    0.87  █████████████████████░░░  likely YES
» needs_billing          noul    0.98  ████████████████████████  likely YES
» needs_security         noul    0.22  █████░░░░░░░░░░░░░░░░░░░  uncertain
◆ sum of the four nouls = 3.04 — independent yes/no questions do NOT sum to 1

▣ STEP 3 · ask a statement AND its negation — are they complementary?
» leaving_early          noul    0.36  █████████░░░░░░░░░░░░░░░  uncertain
» not_leaving_early      noul    0.33  ████████░░░░░░░░░░░░░░░░  uncertain
◆ p(yes) + p(negation) = 0.69 — nothing guarantees exactly 1.00
```

มีสองเรื่องน่าประหลาดใจจริงจากการรันนี้ที่ควรจำ:

- "The room is fine I guess, but it smells a bit different" ได้ **0.50 พอดี** นั่นคือความไม่แน่ใจอย่างตรงไปตรงมา — ส่งให้คนตรวจ อย่าปัดเป็นใช่หรือไม่ใช่
- สำหรับ "I might be checking out a day early, not sure yet" ข้อความได้ 0.36 **และข้อความปฏิเสธของมันได้ 0.33** ทั้งคู่ "ไม่แน่ใจ" และรวมกันได้ 0.69 ไม่ใช่ 1 ถ้าโค้ดของคุณต้องใช้ `p(no) = 1 − p(yes)` ให้คำนวณในโค้ดจากคำถาม *เดียว*

หลายป้ายกำกับ ใช้งานจริง — หนึ่ง noul ต่อหนึ่งป้าย:

```jev
{
  "state": {"message": "The AC is dripping water onto the carpet and the minibar bill looks wrong."},
  "questions": {
    "needs_hvac": {"type": "noul", "instructions": "Does `message` describe a heating or air-conditioning problem?"},
    "needs_housekeeping": {"type": "noul", "instructions": "Does `message` require cleaning or drying something in the room?"},
    "needs_billing": {"type": "noul", "instructions": "Does `message` raise a question or complaint about a bill or charge?"},
    "needs_security": {"type": "noul", "instructions": "Does `message` report a safety or security threat?"}
  }
}
```

การทดลอง:

1. ลบประโยคเรื่อง minibar ออก noul ตัวไหนลดลง และตัวอื่นขยับไหม?
2. เพิ่ม `"criteria": {"true": "…", "false": "…"}` ให้ `needs_security` โดยบอกว่าน้ำใกล้ระบบไฟฟ้า **เป็น** ภัยด้านความปลอดภัย ตัวเลขขยับไหม?

✓ Checkpoint: คุณอธิบายได้ว่าทำไมปัญหาหลายป้ายกำกับจึงต้องใช้หนึ่ง noul ต่อหนึ่งป้าย แทนที่จะใช้ choice เดียว

## 3 · score — ตำแหน่งบนสเกลที่คุณเรียงลำดับไว้

Score ให้คะแนน state ตามระดับที่คุณเรียง **จากต่ำสุดก่อน** (2 ถึง 10 ระดับ) คุณจะได้:

- `score` — ระดับคาดหมาย (expected level) เช่น `1.6` = "ส่วนใหญ่อยู่ระหว่างระดับ 1 กับระดับ 2" ระดับเริ่มที่ **0**
- `probabilities` — หนึ่งค่าต่อระดับ (`"0"`, `"1"`, …)
- `legend` — ข้อความระดับของคุณที่ส่งกลับมา ทำให้ log อ่านเข้าใจได้ในตัว

เขียนระดับเป็น **สถานการณ์ที่เป็นรูปธรรม** ที่คนจำได้ ("แขกใช้ห้องบางส่วนไม่ได้") ไม่ใช่คำกำกวม ("ปานกลาง") และใช้ตัวเลขเพื่อ **threshold** ("≥ 2.5 → เรียกผู้จัดการ") ห้ามใช้เพื่อย้อนหาขนาดจริง — 1.6 ไม่ใช่ "1.6 ชั่วโมง"

```bash
.venv/bin/python week24/02_three_primitives/labs/lab03_score.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 1 · the full answer for one message
» urgency                score   2.00 on 0…3   (confidence 1.00)
      2 Needs action within the hour: guest c…  1.00  ████████████████
◆ Σ level×p = 2.00  vs API score 2.00  (same idea, rounded)

▣ STEP 2 · four messages across the scale
│ message                                           score  top level  conf  → policy (in code)
│ Could you recommend a restaurant for tomorrow n…  0.49   0          0.51  normal queue
│ The bedside lamp bulb is out, whenever you get …  0.11   0          0.89  normal queue
│ There's no hot water and I have a meeting in 40…  2.00   2          1.00  same-hour ticket
│ Water is pouring from the ceiling onto the elec…  3.00   3          1.00  page duty manager
```

สังเกตคำถามเรื่องร้านอาหาร: `0.49` กับ confidence `0.51` — แบ่งกันระหว่างระดับ 0 ("ไม่มีแรงกดดันด้านเวลา") กับระดับ 1 ("จัดการภายในวันนี้") เพราะ "tomorrow night" เป็นการอ้างถึงเวลาจริง นโยบายยังทำถูกเพราะ **threshold อยู่ในโค้ด**

```jev
{
  "state": {"message": "There's no hot water and I have a meeting in 40 minutes."},
  "questions": {
    "urgency": {
      "type": "score",
      "instructions": "How urgent is the request in `message`?",
      "criteria": [
        "No time pressure: a question or a request for later",
        "Should be handled today, guest is mildly inconvenienced",
        "Needs action within the hour: guest cannot use part of the room",
        "Emergency now: safety risk, flooding, fire, or someone hurt"
      ]
    }
  }
}
```

การทดลอง:

1. เปลี่ยนสี่ระดับเป็น `["low", "medium", "high"]` คำตอบมั่นใจน้อยลงไหม? ระดับที่กำกวมมักทำให้ probability กระจาย
2. เปลี่ยนข้อความเป็น `"Water is pouring from the ceiling onto the electrical sockets!"` — ควรกระโดดไปที่ระดับ 3

✓ Checkpoint: คุณคำนวณ score ด้วยมือได้: Σ (ระดับ × probability)

## 4 · `confidence` วัดอะไรจริง ๆ

`confidence` (เฉพาะ choice และ score) สรุปว่า probability **กระจุกตัว** แค่ไหน: 1.0 = รวมอยู่ที่ตัวเลือกเดียว 0.0 = แบนราบ สำหรับ **Choice** เอกสารของ TypeSafe ใช้สูตร:

```text
confidence = (n · peak − 1) / (n − 1)        n = number of options, peak = highest probability
```

ดังนั้นค่ายอด (peak) เท่ากันจะมีความหมายมากขึ้นเมื่อมีตัวเลือกมากขึ้น: 0.70 จาก 4 ตัวเลือก → 0.60; 0.70 จาก 2 ตัวเลือก → 0.40

สำหรับ **Score** TypeSafe ใช้สถิติอีกแบบที่ไม่ได้เปิดเผย Lab 04 วัดทั้งสองแบบ:

```bash
.venv/bin/python week24/02_three_primitives/labs/lab04_confidence_math.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
│ message                                           type    answer  peak  API conf  peak formula  match    score p(0 1 2)
│ The chiller tripped twice last night.             choice  hvac    0.88  0.84      0.84          ✓ same   —
│ The chiller tripped twice last night.             score   0.88    0.38  0.07      0.07          ✓ same   0.37 0.38 0.25
│ Why did our electricity bill jump after the chi…  choice  energy  0.94  0.92      0.92          ✓ same   —
│ Why did our electricity bill jump after the chi…  score   0.94    0.62  0.44      0.43          ✓ same   0.22 0.62 0.16
│ Can you look into it?                             choice  other   0.96  0.95      0.95          ✓ same   —
│ Can you look into it?                             score   0.52    0.66  0.21      0.49          differs  0.66 0.17 0.17
◆ choice: the peak formula reproduces the API confidence
◆ score : often close, but NOT the same statistic — look at the rows marked 'differs'
```

เราพบเรื่องนี้ตอนสร้างคอร์สด้วยการเรียกจริง: สำหรับ "Can you look into it?" probability ของ score คือ `0.66 / 0.17 / 0.17` ให้ค่าจากสูตร peak เป็น 0.49 แต่ API บอกว่า **0.21** สถิติของ Score ดูเหมือนจะสนใจ *ลำดับ*: probability ที่อยู่บนระดับ **ไกล** (ระดับ 2 ห่างจากยอดสองขั้น) ทำให้ confidence ลดลงมากกว่า probability บนระดับที่ติดกัน (เทียบกับแถวร้านอาหารใน lab 03 ด้านบน: `0.51 / 0.49 / 0 / 0` — แบ่งกันในระดับติดกัน — ได้ confidence 0.51 สูงกว่าสูตร peak ที่ได้ 0.35)

บทเรียนใหญ่กว่าสูตร: **probability คือความจริงพื้นฐาน ส่วน confidence เป็นแค่ตัวสรุปที่ใช้สะดวก** และทั้งสองอย่างไม่ใช่ "โอกาสที่คำตอบจะถูกบนข้อมูลของคุณ" — มีแต่การประเมินผล (evaluation) เท่านั้นที่บอกได้ (Lab 09)

✓ Checkpoint: คุณคำนวณ confidence ของ Choice ด้วยมือ และตรงกับ API

## 5 · ใช้ primitive ไหนดี? ตารางช่วยตัดสินใจ

| ความต้องการของคุณฟังดูเหมือน… | ใช้ | ทำไม |
|---|---|---|
| "อันไหนหนึ่งเดียวในนี้ …?" (ชุดที่เลือกได้อย่างเดียว) | `choice` | ผู้ชนะหนึ่งเดียว + การกระจาย; เพิ่มตัวเลือกทางออก |
| "จริงไหมว่า …?" | `noul` | probability ค่าเดียว ไม่มีตัวเลือกบังคับ |
| "แท็กไหนบ้างที่เข้าข่าย?" (จริงได้หลายข้อ) | หนึ่ง `noul` ต่อแท็ก | choice จะบังคับให้มีผู้ชนะหนึ่งเดียว |
| "มากแค่ไหน / รุนแรงแค่ไหน / มีแนวโน้มจะซื้อแค่ไหน …?" | `score` | ระดับที่เรียงลำดับ + threshold ในโค้ด |
| "กี่อัน …?" / "วันไหนมาก่อน?" / "28 − 24 ได้เท่าไร?" | **โค้ด** | การนับ วันที่ และคณิตศาสตร์ ไม่ใช่การตัดสิน (Lab 03) |
| "เขียนคำตอบให้หน่อย" | LLM แบบสร้างข้อความ | Jev ไม่สร้างข้อความ |

✓ Checkpoint: คุณอธิบายแต่ละแถวของตารางนี้ด้วยคำพูดของตัวเองได้

## แล็บ — รันได้ที่นี่

**labs/lab01_choice.py** — Choice เดียว สามข้อความ: ชัดเจน ลังเลระหว่างสองตัวเลือก และไม่เข้าข้อไหนเลย

**labs/lab02_noul.py** — ใช่/ไม่ใช่/ไม่แน่ใจ หลายป้ายกำกับด้วยหนึ่ง noul ต่อป้าย และข้อความกับข้อความปฏิเสธของมัน

**labs/lab03_score.py** — ระดับที่เรียงลำดับเป็นรูปธรรม ค่าคาดหมายคำนวณด้วยมือ และนโยบาย threshold ในโค้ด

**labs/lab04_confidence_math.py** — คำนวณ confidence เอง และดูว่า confidence ของ Score ต่างจากสูตรของ Choice ตรงไหน

## ลองทำเอง

**แบบฝึกหัด 02 — เลือก primitive ให้ถูก** เปิด `week24/02_three_primitives/exercises/ex02_pick_the_primitive.py`

- **ส่วน A (ออฟไลน์ ฟรี):** ควิซหกข้อ — แต่ละความต้องการให้เขียน `"choice"`, `"noul"` หรือ `"score"`
- **ส่วน B:** เขียนคำถามสามข้อสำหรับข้อความจากแขก — `department` (choice พร้อมตัวเลือกทางออก), `mentions_refund` (noul) และ `anger` (score เริ่มจากระดับที่สงบที่สุด)
- **ส่วน C (อัตโนมัติ):** ตัวตรวจจะถาม Jev เกี่ยวกับข้อความทดสอบสามข้อ

```bash
.venv/bin/python week24/02_three_primitives/exercises/ex02_pick_the_primitive.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ PART A · quiz (offline, $0)
✓ q1_which_language: choice
…
◆ quiz: 6/6
▣ PART B · question shapes (offline, $0)
✓ department is a choice with 4 options incl. an escape option
✓ mentions_refund is a noul
✓ anger is a score with 4 levels
▣ PART C · asking Jev about 3 messages (one batched call each)
━━ “I've been waiting 45 minutes for room service. If it isn't here in 10 I want my money back.”
» department             choice  → room_service   (confidence 1.00)
» mentions_refund        noul    0.99  ████████████████████████  likely YES
» anger                  score   2.93 on 0…3   (confidence 0.93)
```

<details><summary>คำใบ้ — ข้อควิซที่หลอกตา (q4)</summary>

"รีวิวหนึ่งชมได้ทั้งอาหาร พนักงาน และทำเล" — หลายแท็กเป็นจริงพร้อมกันได้ `choice` จะบังคับให้มีผู้ชนะหนึ่งเดียวพอดี คำตอบจึงเป็น **หนึ่ง noul ต่อแท็ก**

</details>

<details><summary>คำใบ้ — การขอเงินคืนที่เป็นแค่คำขู่</summary>

"If it isn't here in 10 I want my money back" เป็นข้อเรียกร้อง *แบบมีเงื่อนไข* ตัดสินใจว่าจะนับหรือไม่ แล้วเขียนไว้ใน `criteria` ของ noul — เช่น `true: "A refund is requested or demanded, even conditionally"` Jev อ่านตามตัวอักษร criteria คือที่ที่ใส่เจตนาของคุณ

</details>

✓ Checkpoint: ควิซขึ้นว่า 6/6 และการตรวจรูปแบบทั้งสามเป็น ✓

## แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `HTTP 422` กับ score | `criteria` ต้องเป็น **list** (ระดับที่เรียงลำดับ) ไม่ใช่ dict — และมี 2–10 ระดับ |
| `HTTP 422` กับ choice | `criteria` ต้องเป็น **dict** `{option: description}`; ใช้ `null` ถ้าตัวเลือกไม่ต้องมีคำอธิบาย |
| คีย์ของ Score เป็น string | `probabilities` และ `legend` ใช้ `"0"`, `"1"`… — เข้าถึงด้วย `str(i)` |
| noul อยู่ใกล้ 0.5 | นั่นคือโมเดลบอกว่า "ไม่แน่ใจ" ทำข้อความให้คมขึ้น เพิ่ม `criteria` หรือส่งให้คนตรวจ |
| confidence ของ Choice ดูต่ำแต่ผู้ชนะชัดเจน | อาจมีหลายตัวเลือกที่ยอมรับได้จริง — ดูสองอันดับแรกก่อนทิ้งคำตอบ |

## ถัดไป

ไปต่อที่ [Lab 03 — เขียนคำถามให้ Jev ตอบได้ดี](../03_question_design/TUTORIAL.th.md): การอ่านตามตัวอักษร ตัวเลือกทางออก fan-out, prompt injection และทำไมคณิตศาสตร์ต้องอยู่ในโค้ด
