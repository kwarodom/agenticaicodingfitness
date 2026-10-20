# ▶ Jev Lab 07 — คัดแยกเหตุขัดข้องระบบ HVAC: โค้ดคำนวณ Jev ตัดสิน

> ส่วนหนึ่งของ Week 24 · การตัดสินใจของ AI แบบมีชนิดด้วย Jev ระบบตรวจจับและวินิจฉัยเหตุขัดข้องอัตโนมัติ (AFDD — automated fault detection and diagnostics) สร้างบันทึกของผู้ปฏิบัติงานและข้อมูล telemetry ออกมาต่อเนื่อง งานของ Jev ในที่นี้แคบมาก: จัดแต่ละเหตุการณ์เข้า **คิวการตรวจสอบ (investigation queue)** ที่ถูกต้องให้วิศวกร มันไม่เคยวินิจฉัยต้นเหตุ และไม่เคยแตะต้องอาคาร

**สิ่งที่คุณจะได้ลงมือทำ**
- คำนวณข้อเท็จจริงเชิงตัวเลขใน Python ก่อน — ความเก่าของข้อมูล คุณภาพข้อมูล และส่วนเบี่ยงเบนจากค่าตั้ง (setpoint)
- ส่งข้อเท็จจริงเหล่านั้นพร้อมบันทึกของผู้ปฏิบัติงานให้ Jev แล้วได้คิว สัญญาณความปลอดภัย และระดับความรุนแรงกลับมา
- ดูนโยบายแบบกำหนดแน่นอน (deterministic policy) **ทับ (override)** คำตอบของ Jev เมื่อข้อมูลเชื่อถือไม่ได้
- เจอกับดักการอ่านตามตัวอักษรคลาสสิก: *"No smoke reported"* vs *"smoke reported"*
- เขียนการคำนวณ flag เอง ทดสอบแบบออฟไลน์ละเอียดถึงทศนิยม

**Time** ~30 นาที · **Difficulty** ระดับกลาง · **Cost** ≈ $0.0003 เมื่อรันจริง · $0 ในโหมด dry

## 0 · ทำไมโค้ดต้องทำเรื่องตัวเลข

เอกสารของ TypeSafe เองระบุไว้ว่านี่คือจุดอ่อนที่รู้กันของ Jev 1.13: *มันไม่ใช่เครื่องคิดเลข* มันอ่านตัวเลขและวันที่เป็นข้อความ เราจึงแบ่งงานกัน:

| งาน | ใครทำ | ทำไม |
|---|---|---|
| ค่าที่อ่านได้เก่ากว่า 10 นาทีไหม? | **โค้ด** | เปรียบเทียบ timestamp ที่เชื่อถือได้อย่างแม่นยำ |
| โซนสูงกว่า setpoint เกิน 2 °C ไหม? | **โค้ด** | คำนวณเลข |
| flag คุณภาพเซนเซอร์เสียไหม? | **โค้ด** | เป็นค่า boolean จาก BMS |
| คิวการตรวจสอบไหนเข้ากับบันทึกนี้ + ข้อเท็จจริงเหล่านี้? | **Jev** | เป็นการตัดสินเรื่องภาษาและบริบท |
| บันทึกรายงานข้อกังวลด้านความปลอดภัยไหม? | **Jev + ตัวสำรองในโค้ด** | ห้ามพึ่งสัญญาณเดียวเรื่องความปลอดภัย |
| เปลี่ยน setpoint เขียน BACnet เปิดใบสั่งงาน | **ไม่มีใครในที่นี้** | เป็นระบบที่อนุมัติแยกต่างหาก โดยมีคนดูแล |

```text
BMS reading ──► compute flags (code) ──┐
                                        ├──► Jev: queue · safety · severity ──► policy (code) ──► engineer_review
operator note ─────────────────────────┘                                           │
                                                   stale / bad data overrides the queue · safety is OR'd with a keyword check
```

threshold (เกณฑ์) ในแล็บนี้ (**600 s** ถือว่าเก่า, **2 °C** ถือว่าร้อนเกิน) เป็น **ค่าคงที่เพื่อการสอน** ไม่ใช่ขีดจำกัดของ AltoTech ที่ผ่านการรับรอง หรือมาตรฐานความสบาย ไซต์จริงจะโหลดเกณฑ์ที่อนุมัติแล้วแยกตามไซต์ อุปกรณ์ โหมดการใช้งาน และตารางเวลา

✓ Checkpoint: คุณบอกได้ว่าโค้ดคำนวณข้อเท็จจริงสามอย่างไหนก่อนที่ Jev จะเห็นอะไรเลย และเพราะอะไร

## 1 · ให้ข้อเท็จจริงกับ Jev ไม่ใช่ให้มันคำนวณ

นี่คือ state ที่ Jev ได้รับสำหรับหนึ่งเหตุการณ์ — บันทึก บวกกับบล็อก `computed` ที่โค้ดเติมให้ ลองใช้งานจริง:

```jev
{
  "state": {
    "operator_note": "Guests report warm rooms on floor 3. AHU-3 is running.",
    "computed": {"stale": false, "bad_quality": false, "warm_deviation": true, "deviation_c": 5.1}
  },
  "questions": {
    "queue": {
      "type": "choice",
      "instructions": "Using `operator_note` and the `computed` flags, select the next investigation queue. Do not claim a proven root cause.",
      "criteria": {
        "cooling": "Comfort or cooling-performance investigation.",
        "sensor": "Sensor plausibility or calibration investigation.",
        "connectivity": "Offline gateway, missing or stale telemetry investigation.",
        "other": "Insufficient or conflicting evidence; engineer triage."
      }
    },
    "severity": {
      "type": "score",
      "instructions": "How severe is the reported operational impact?",
      "criteria": ["Informational or no impact stated.", "Localized discomfort or limited degradation.", "Significant service disruption.", "Possible immediate safety incident."]
    }
  }
}
```

การทดลอง:

1. ตั้งค่า `"stale": true` คิวขยับไปทาง `connectivity` ไหม? (อาจใช่ — แต่ใน step 2 คุณจะเห็นว่านโยบายไม่ได้ *พึ่ง* เรื่องนั้น)
2. ตั้งค่า `"bad_quality": true` และเปลี่ยนบันทึกเป็น `"Room 214 sensor reads 35 C but the guest says the room feels cold."` ตอนนี้ได้คิวไหน?
3. เปลี่ยนบันทึกเป็น `"Water is dripping from the ceiling near the AHU."` ดูค่า `severity`

ลองแบบภาษาไทยด้วย — บันทึกของผู้ปฏิบัติงานเป็นภาษาไทย แต่ flag ที่โค้ดคำนวณยังเหมือนเดิม:

```jev
{
  "state": {
    "operator_note": "แขกแจ้งว่าห้องชั้น 3 ร้อน เครื่อง AHU-3 ยังทำงานอยู่",
    "computed": {"stale": false, "bad_quality": false, "warm_deviation": true, "deviation_c": 5.1}
  },
  "questions": {
    "queue": {
      "type": "choice",
      "instructions": "Using `operator_note` and the `computed` flags, select the next investigation queue. Do not claim a proven root cause.",
      "criteria": {
        "cooling": "Comfort or cooling-performance investigation.",
        "sensor": "Sensor plausibility or calibration investigation.",
        "connectivity": "Offline gateway, missing or stale telemetry investigation.",
        "other": "Insufficient or conflicting evidence; engineer triage."
      }
    },
    "severity": {
      "type": "score",
      "instructions": "How severe is the reported operational impact?",
      "criteria": ["Informational or no impact stated.", "Localized discomfort or limited degradation.", "Significant service disruption.", "Possible immediate safety incident."]
    }
  }
}
```

✓ Checkpoint: คุณเปลี่ยน flag ที่คำนวณไว้แล้วเห็น Jev นำไปใช้ — โดยที่ Jev ไม่ได้คำนวณเอง

## 2 · คัดแยกห้าเหตุการณ์ — lab 01

```bash
.venv/bin/python week24/07_afdd_triage/labs/lab01_triage_incidents.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 1 · what code computes BEFORE the model sees anything
   raw reading: zone 29.1 °C, setpoint 24.0 °C, age 120 s, quality_ok True
✓ computed:   {"stale": false, "bad_quality": false, "warm_deviation": true, "deviation_c": 5.1}
…
▣ STEP 3 · triage all 5 incidents
│ id       Δ°C    code flags                  jev queue     safety  sev   proposed queue                safety rev
│ fault-1  +5.1   warm_deviation              cooling       0.03    1.01  cooling                       -
│ fault-2  -1.0   stale                       connectivity  0.03    1.82  connectivity_or_data_quality  -
│ fault-3  +0.4   -                           other         0.73    2.79  other                         YES
│ fault-4  +12.0  bad_quality,warm_deviation  sensor        0.03    1.03  connectivity_or_data_quality  -
│ fault-5  +1.2   -                           other         0.02    0.60  other                         -
═ execute: False · route: engineer_review · no_bacnet_write: True · no_setpoint_change: True
```

นโยบายคือ Python ธรรมดาสิบบรรทัด:

```python
def policy(state, a):
    flags = state["computed"]
    data_problem = flags["stale"] or flags["bad_quality"]
    keyword_hit = bool(SAFETY_WORDS.search(state["operator_note"]))
    return {
        "route": "engineer_review",
        "proposed_queue": "connectivity_or_data_quality" if data_problem else a["queue"]["choice"],
        "safety_review": a["safety_concern"]["noul"] >= 0.20 or keyword_hit,   # OR — never AND
        "execute": False, "no_bacnet_write": True, "no_setpoint_change": True,
    }
```

อ่านแถวที่น่าสนใจ:

- **fault-4** — เซนเซอร์อ่านได้ 35 °C (+12 °C!) แต่แขกบอกว่าห้อง *เย็น* และ `quality_ok` เป็น False Jev เลือก `sensor` อย่างสมเหตุสมผล นโยบายยังทับไปที่คิวคุณภาพข้อมูล: **ห้ามวินิจฉัยเรื่องความสบายจากข้อมูลที่รู้อยู่ว่าเสีย**
- **fault-3** — *"Burning smell reported near the AHU-2 plant room."* probability ด้านความปลอดภัยของ Jev ได้แค่ **0.73** ไม่ใช่ 0.99 นโยบายยังส่งตรวจด้านความปลอดภัยอยู่ดี — ทั้งเพราะ 0.73 ≥ 0.20 และเพราะตัวสำรองแบบคำสำคัญ (keyword backstop) จับคำว่า `burning` ได้ **probability ต่ำจากโมเดลต้องไม่มีวันยับยั้งการตอบสนองด้านความปลอดภัย**
- **fault-2** — Jev ตอบ `connectivity` เอง ดี แต่การ override ไม่ได้พึ่งมัน

> ⚠ ขั้นตอนด้านอัคคีภัย ความปลอดภัย และสัญญาณเตือนฉุกเฉินที่มีอยู่ ต้องทำงาน **แยกอิสระ** จากตัวจำแนกนี้ API หมดเวลาหรือ probability ต่ำต้องไม่มีวันทำให้มันล่าช้า

✓ Checkpoint: คุณอธิบายได้ว่าทำไม fault-4 ถูก override แม้คำตอบ `sensor` ของ Jev จะดูสมเหตุสมผล

## 3 · ถ้าตัวเลขเปลี่ยนล่ะ? — lab 02

Lab 02 ถาม Jev เกี่ยวกับบันทึกของผู้ปฏิบัติงานหนึ่งฉบับ **ครั้งเดียว** แล้วนำสถานการณ์ telemetry สี่แบบผ่านนโยบายเดียวกัน เรียกโมเดลครั้งเดียว ได้ผลลัพธ์สี่แบบ

```bash
.venv/bin/python week24/07_afdd_triage/labs/lab02_what_if.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 2 · same answers, four telemetry situations → the policy decides
│ situation           Δ°C   stale  bad_q  warm   proposed queue
│ fresh, 4.2 °C warm  +4.2  False  False  True   cooling
│ stale (2 h old)     +4.2  True   False  True   connectivity_or_data_quality
│ bad quality flag    +4.2  False  True   True   connectivity_or_data_quality
│ fresh, on setpoint  +0.3  False  False  False  cooling
◆ 1 model call · 4 outcomes — the override is plain Python you can unit-test
▣ STEP 3 · a literal-reading trap: 'No smoke' vs 'smoke'
» no_smoke  jev safety 0.02 · keyword backstop HIT   “Lobby is warm. AHU fan is running. No smoke or unusual noise reported.”
» smoke     jev safety 0.89 · keyword backstop HIT   “Lobby is warm. AHU fan is running. Smoke smell reported near the AHU.”
◆ compare STEP 1: lab 01 asks whether the note 'explicitly MENTIONS smoke…' → 0.61 on the SAME 'No smoke' note.
```

ทำไม keyword backstop จึงขึ้นว่า **HIT** กับ *"No smoke…"*? เพราะมันเป็น regex ที่ตั้งใจให้ทำงานแบบตรงตัว คือเห็นคำว่า `smoke` ก็ทำงานทันที ไม่ว่าจะมีคำปฏิเสธหรือไม่ และนั่นคือจุดประสงค์ backstop ได้รับอนุญาตให้เตือนเกินจริง เพราะเตือนผิดหนึ่งครั้งเสียแค่เวลาวิศวกรเหลือบดู แต่พลาดเหตุไฟไหม้จริงเสียหายมากกว่ามาก ค่า 0.02 ของ Jev แสดงว่ามันอ่านคำปฏิเสธได้ถูก แต่ policy ยังส่งเข้าคิวตรวจความปลอดภัย เพราะสัญญาณ *ใดสัญญาณหนึ่ง* ก็เพียงพอแล้ว

นี่คือข้อค้นพบที่สำคัญที่สุดของแล็บนี้ บันทึกเขียนว่า **"No smoke or unusual noise reported."**

- คำถามของ Lab 01 ถามว่าบันทึก *"explicitly **mention**s smoke, electrical burning, fire…"* ไหม → **0.61** Jev อ่านตามตัวอักษร: "No smoke" *ก็เป็น* การกล่าวถึงควันจริง ๆ
- Lab 02 ถามว่ามีข้อกังวล *"**currently present**"* (เกิดอยู่ตอนนี้) ไหม → **0.02**

บันทึกเดียวกัน โมเดลเดียวกัน ถ้อยคำต่างกัน ได้คำตอบต่างกันโดยสิ้นเชิง **ข้อความคำถามเป็นส่วนหนึ่งของโปรแกรมคุณ** — ทดสอบประโยคปฏิเสธโดยตรงเสมอ ("do not turn off the chiller" vs "turn off the chiller")

ลองดู:

```jev
{
  "state": {"note": "Lobby is warm. AHU fan is running. No smoke or unusual noise reported."},
  "questions": {
    "mentions": {"type": "noul", "instructions": "Does `note` explicitly mention smoke, burning, fire or another safety concern?"},
    "present": {"type": "noul", "instructions": "Does `note` report smoke, burning, fire or another safety concern that is currently present?"}
  }
}
```

ตอนเรารันบล็อกนี้: `mentions` **0.72**, `present` **0.02** การทดลอง: เปลี่ยนบันทึกเป็น `"Smoke smell near the AHU."` ทั้งสองควรพุ่งขึ้น จากนั้นลอง `"The smoke alarm test passed this morning."` — ถ้อยคำแบบไหนรับมือได้ดีกว่า?

✓ Checkpoint: คุณเห็นบันทึก "No smoke" เดียวกันได้คะแนนสูงในถ้อยคำหนึ่งและต่ำในอีกถ้อยคำ และอธิบายได้ว่าทำไมตัวสำรองแบบคำสำคัญยังทำงานอยู่

## 4 · เชื่อมกับอาคารจริง (แบบปลอดภัย)

ลำดับงานที่เสนอสำหรับระบบใช้งานจริงคงขอบเขตทุกข้อจากแล็บนี้ไว้:

1. query ข้อมูลอนุกรมเวลาที่ **ได้รับอนุญาต** สำหรับไซต์ที่ผู้ใช้มีสิทธิ์ดู
2. ตรวจ timestamp และคุณภาพใน **โค้ดฝั่งเซิร์ฟเวอร์** — ห้ามให้ผู้ใช้หรือ LLM เป็นผู้ยืนยันว่าข้อมูลยังใหม่
3. คำนวณ feature ของ AFDD ในโค้ด (ส่วนเบี่ยงเบน ชั่วโมงการทำงาน จำนวนรอบการเปิดปิด)
4. Jev คัดแยก → เสนอคิวการตรวจสอบ
5. ผู้เชี่ยวชาญ HVAC (คน หรือโมเดลสร้างข้อความที่มีหลักฐานประกอบ) เขียนคำอธิบาย
6. ข้อเสนอการซ่อมบำรุงที่ **คนตรวจทานแล้ว** Copilot ทำได้แค่เสนอเท่านั้น

reference implementation ในสคริปต์ต้นฉบับใช้คำถามและการ override แบบเดียวกัน:

```bash
.venv/bin/python week24/jev_lab/jev_lab.py run afdd --limit 2 --live
```

✓ Checkpoint: คุณชี้ได้ว่าขั้นไหนในลำดับงานนี้ที่คนเป็นผู้อนุมัติก่อนจะมีอะไรเปลี่ยนที่ไซต์

## แล็บ — รันได้ที่นี่

**labs/lab01_triage_incidents.py** — ห้าเหตุการณ์: โค้ดคำนวณ flag Jev เลือกคิว นโยบาย override เมื่อข้อมูลเสีย และรวมสัญญาณความปลอดภัยแบบ OR

**labs/lab02_what_if.py** — เรียก Jev ครั้งเดียว สี่สถานการณ์ telemetry และกับดักการอ่านตามตัวอักษร "No smoke"

## ลองทำเอง

**แบบฝึกหัด 07 — คำนวณ flag เอง** เปิด `week24/07_afdd_triage/exercises/ex07_compute_flags.py` แล้วเขียน `compute_flags(reading)` ให้คืนค่าตามนี้ทุกประการ:

| key | กฎ |
|---|---|
| `stale` | `age_seconds > 600` |
| `bad_quality` | `quality_ok is False` |
| `warm_deviation` | `zone_c − setpoint_c > 2.0` |
| `cold_deviation` | `setpoint_c − zone_c > 2.0` *(ใหม่)* |
| `humidity_high` | `humidity_pct > 70` *(ใหม่)* |
| `deviation_c` | `zone_c − setpoint_c` ปัดเป็นทศนิยม 1 ตำแหน่ง |

assert ออฟไลน์ที่แม่นยำห้าข้อจะรันก่อน — รวมถึงกรณีขอบ "เท่ากับ 2.0 °C พอดี **ไม่ถือ** ว่าเบี่ยงเบน" จากนั้น flag ของคุณจะป้อนเข้าการคัดแยกจริงของห้าเหตุการณ์

```bash
.venv/bin/python week24/07_afdd_triage/exercises/ex07_compute_flags.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
✓ warm and fresh
✓ cold room
✓ exactly 2.0 warm is NOT a deviation
✓ stale + humid
✓ bad quality
▣ STEP 2 · your flags feed a live Jev triage
│ fault-4  bad_quality,warm_deviation  sensor        0.03
```

<details><summary>คำใบ้ — กรณีขอบ 2.0 °C</summary>

กฎคือ "มากกว่า 2.0" จึงใช้ `>` ไม่ใช่ `>=` นี่คือรายละเอียดแบบที่ Jev ทำได้ไม่น่าเชื่อถือ — และเป็นเหตุผลที่มันต้องอยู่ในโค้ดพร้อมการทดสอบ

</details>

<details><summary>ท้าทายเพิ่ม — คิวเรื่องความชื้น</summary>

เพิ่มตัวเลือก `humidity` ให้ choice `queue` ("Dehumidification or latent-load investigation") ในสำเนา `QUESTIONS` ของ lab 01 แล้วส่ง flag `humidity_high` ของคุณเข้าไป เหตุการณ์ไหนที่ย้ายคิว?

</details>

✓ Checkpoint: assert ผ่าน 5/5 และตารางผลจริงแสดง flag ของคุณคู่กับคิวของ Jev

## แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `invalid numeric field` | ค่าที่อ่านได้ขาดหายหรือไม่ใช่ตัวเลข แก้ข้อมูลที่ต้นทาง; ห้ามให้โมเดลเดาตัวเลข |
| probability ด้านความปลอดภัยดูต่ำสำหรับอันตรายที่ชัดเจน | เกิดได้บางครั้ง (0.73 สำหรับ "burning smell") นั่นคือเหตุผลที่นโยบายรวมกับตัวสำรองแบบคำสำคัญด้วย OR และใช้เกณฑ์ต่ำที่ 0.20 |
| "No smoke" ถูกจัดเป็นข้อกังวลด้านความปลอดภัย | instruction ของคุณถามว่า "mentions" ให้ถามว่าข้อกังวล "currently present" (เกิดอยู่ตอนนี้) — และคงตัวสำรองไว้อยู่ดี |
| คิวดูถูกแต่นโยบาย override | ทำงานตามที่ออกแบบ: ข้อมูลที่เก่าหรือคุณภาพเสียจะไปคิวคุณภาพข้อมูลก่อนเสมอ |
| แบบฝึกหัดขึ้น `0/5 pass` | `compute_flags()` ยังคืน `{}` อยู่ — เขียนให้ครบทั้งหก key |

## ถัดไป

ไปต่อที่ [Lab 08 — ลีดการขาย การกรอง passage สำหรับ RAG และการจับคู่จุด BMS](../08_leads_rag_points/TUTORIAL.th.md)
