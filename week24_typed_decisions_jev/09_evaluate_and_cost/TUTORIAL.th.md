# ▶ Jev Lab 09 — ประเมินผลก่อนทำให้เป็นอัตโนมัติ (และรู้ว่าต้องจ่ายเท่าไร)

> ส่วนหนึ่งของ Week 24 · การตัดสินใจแบบมีชนิดข้อมูล (typed decisions) ด้วย Jev คำตอบที่มีรูปแบบถูกต้องไม่ได้แปลว่าเป็นคำตอบที่ถูก แล็บนี้เปลี่ยนความรู้สึกว่า "ดูเหมือนจะถูก" ให้เป็นตัวเลขที่คุณยืนยันได้: accuracy, coverage, wrong-accepted, latency และต้นทุน

**สิ่งที่คุณจะได้ลงมือทำจริง**
- รันการประเมินผลแบบ smoke test ที่มี label 12 แถว (ภาษาอังกฤษ + ไทย) กับ router จัดเส้นทางตามเจตนาจากแล็บ 04
- อ่านตัวเลขสี่ตัวที่ตัดสินว่า router ทำงานเองได้หรือไม่ และตัวที่สำคัญที่สุด
- ไล่ปรับ threshold (เกณฑ์) บนคำตอบชุด **เดียวกัน** โดยไม่เรียก API ใหม่เลย
- ประเมินราคาตั้งแต่ 1 → 1,000,000 คำขอ จากจำนวน token ที่คุณวัดได้จริง และตรวจ rate limit
- สร้างชุดทดสอบที่มี label ของคุณเอง และเขียนฟังก์ชัน macro-F1

**Time** ~40 นาที · **Difficulty** ระดับกลาง · **Cost** ≈ $0.001 เมื่อ live · $0 เมื่อ dry

## 0 · ทำไมต้องประเมินผลด้วย?

Jev ส่งคืน JSON ที่ถูกต้องเสมอ พร้อมความน่าจะเป็นที่รวมกันได้ 1 สิ่งนี้รับประกัน **interface** ไม่ได้รับประกัน **ความจริง** วิธีเดียวที่จะรู้ว่า router ดีพอจะทำงานเองได้หรือไม่ คือรันมันกับตัวอย่างที่ *คุณรู้คำตอบที่ถูกอยู่แล้ว* แล้วนับผล

คำสี่คำที่คุณจะใช้ตลอดแล็บนี้:

| คำ | ความหมาย | ทำไมต้องสนใจ |
|---|---|---|
| **gold label** | คำตอบที่คนตัดสินแล้วว่าถูกต้อง | ใช้เป็นเฉลยในการให้คะแนน |
| **coverage** | สัดส่วนของแถวทั้งหมดที่ router จัดการเองอัตโนมัติ | คุณประหยัดงานได้เท่าไร |
| **accepted accuracy** | สัดส่วนของแถวที่จัดการอัตโนมัติแล้วถูก | ระบบอัตโนมัติปลอดภัยแค่ไหน |
| **wrong accepted** | แถวที่จัดการอัตโนมัติ **และ** ผิด | ความเสียหาย ต้องกดให้เป็น 0 |

router ที่ส่งทุกอย่างให้คนมี coverage 0% และ wrong-accepted 0: ปลอดภัยแต่ไร้ประโยชน์ router ที่จัดการเองทุกอย่างมี coverage 100% และทุกความผิดพลาดหลุดผ่านไปหมด หน้าที่ของคุณคือเลือกจุดที่อยู่ระหว่างสองขั้วนี้

✓ Checkpoint: คุณอธิบายได้ว่าทำไม "coverage" กับ "accepted accuracy" จึงดึงไปคนละทิศ

## 1 · รันการประเมินแบบ smoke test (แล็บ 01)

ข้อมูลอยู่ที่ `week24/09_evaluate_and_cost/data/intent_eval.jsonl` ประกอบด้วยแถวสำหรับสอน 9 แถวจาก `jev_lab.py` บวกแถวภาษาไทยอีก 3 แถว แต่ละแถวเป็น JSON หนึ่งบรรทัด:

```json
{"id": "t3", "language": "th", "state": {"message": "นัดช่างมาบำรุงรักษาเครื่องทำน้ำเย็นสัปดาห์หน้า"}, "expected": {"intent": "facility_ops"}}
```

ฟิลด์ `expected` ไม่เคยไปถึง Jev: `jev_lab.prepare()` คัดลอกเฉพาะ `message` ลงใน state การปล่อยให้เฉลยรั่วเข้าไปใน input คือบั๊กของการประเมินผลที่พบบ่อยที่สุด

```bash
.venv/bin/python week24/09_evaluate_and_cost/labs/lab01_smoke_eval.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 2 · ask Jev — 12 calls, one per row
│ id  lang  gold            predicted       top p  match  gate
│ i1  en    hvac            hvac            1.00   ✓      auto
│ i3  en    hvac            energy_mv       0.86   ✕      auto
│ i9  en    unknown         unknown         0.97   ✓      review
│ t3  th    facility_ops    hvac            0.78   ✕      review
│ …
▣ STEP 3 · score it — reference gate: confidence ≥ .75 AND top ≥ .80 AND margin ≥ .20
◆ raw accuracy (successes only) : 83%
◆ coverage (auto-routed / all)  : 83%  (10 of 12)
◆ accepted accuracy             : 90%
◆ wrong accepted                : 1   ← routed automatically AND wrong
◆ accuracy [en]                 : 7/8
◆ accuracy [th]                 : 3/4
▣ STEP 5 · latency and cost (successful calls)
◆ p50 460 ms · p95 559 ms   (nearest-rank, n=12)
◆ 10354 input tokens total · avg 863/call · $0.000435 for the whole eval
```

มีสองแถวที่ผิด และสอนบทเรียนต่างกัน:

- **t3** (ภาษาไทย: "นัดช่างมาบำรุงรักษาเครื่องทำน้ำเย็นสัปดาห์หน้า") → Jev เลือก `hvac` ด้วย p ≈ 0.78–0.80 เราเห็นทั้งสองค่านี้จากการรันแบบ live คนละครั้งด้วยคำขอ *เดียวกัน* ที่ 0.78 gate **จับได้** (ต่ำกว่าเส้น 0.80 → ส่งรีวิว) แต่ที่ 0.80 มัน **หลุดผ่าน** และกลายเป็น "wrong accepted" แถวที่สอง นั่นคือบทเรียนจริง: threshold ที่วางไว้ตรงขอบของข้อมูลพอดี ทำให้ผลลัพธ์ขึ้นกับการขยับเล็ก ๆ ระหว่างการรันแต่ละครั้ง ให้เลือก threshold ที่มีระยะเผื่อ (แล็บ 09-2 แสดงว่า 0.90 กำจัด wrong-accepted ได้ทั้งสองแถว) และตรวจซ้ำทุกครั้งที่มีโมเดลเวอร์ชันใหม่
- **i3** ("เปรียบเทียบผลวินิจฉัย HVAC กับผลประเมิน M&V แล้วหาข้อยุติ") → Jev เลือก `energy_mv` ที่ 0.86 และถูก **ยอมรับ** แต่ Jev ผิดจริงหรือ? คำขอนี้ครึ่งหนึ่งเป็น HVAC อีกครึ่งเป็น M&V และในการเรียกครั้งเดียวกัน Jev ตอบ `needs_mas = 0.98` ("งานนี้ต้องใช้ผู้เชี่ยวชาญหลายคน") *label* เองต่างหากที่ถกเถียงได้ การตัดสินว่าอะไรถูกเรียกว่า **adjudication** (การตัดสินชี้ขาด) และคุณต้องทำก่อนจะเชื่อตัวเลข accuracy ใด ๆ

> ตัวเลขของคุณอาจต่างจากตัวอย่างด้านบนเล็กน้อย: คำตอบแบบ live ขยับไปไม่กี่ส่วนร้อยระหว่างการรัน `t3` คือแถวที่มีโอกาสสลับระหว่าง `review` กับ `auto` มากที่สุด

✓ Checkpoint: คุณรันแล็บ 01 แล้ว บอกได้ว่าแถวไหนคือ "wrong accepted" และให้เหตุผลได้ว่าความผิดอยู่ที่โมเดลหรือที่ label

## 2 · สัมผัสความกำกวมด้วยตัวเอง

นี่คือแถว i3 ที่มีแค่คำถามเรื่องหัวข้อ กด **⚡ Ask Jev** แล้วลองแก้ตามด้านล่าง เพื่อดูความน่าจะเป็นขยับไปมาระหว่าง `hvac` กับ `energy_mv`

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
        "unknown": "Too vague, unsupported topic, or no identifiable request."
      }
    },
    "needs_several_experts": {
      "type": "noul",
      "instructions": "Does `message` explicitly require combining distinct expert analyses or resolving their disagreement?"
    }
  }
}
```

1. ลบ `"and resolve their disagreement"` ออก หัวข้อชัดขึ้นไหม?
2. เปลี่ยนข้อความเป็น `"Why is the HVAC diagnosis disagreeing with the M&V baseline?"`
3. เพิ่มตัวเลือก `"mixed_specialists"` ลงใน `criteria` ตอนนี้ชุด label เองยอมรับความกำกวมแล้ว ซึ่งมักเป็นวิธีแก้ที่ตรงไปตรงมาที่สุด

✓ Checkpoint: คุณหาการแก้ได้อย่างน้อยหนึ่งแบบที่ทำให้ `intent` ขยับมากกว่า 0.2

## 3 · ไล่ปรับ threshold — ฟรี (แล็บ 02)

threshold อยู่ในโค้ด **ของคุณ** ไม่ได้อยู่ในโมเดล ดังนั้นเมื่อได้คำตอบมาแล้ว คุณลองทุก threshold ได้โดยไม่ต้องเรียก Jev อีก แล็บ 02 อ่านคำตอบที่แล็บ 01 บันทึกไว้ใน `.runs/intent_answers.json`

```bash
.venv/bin/python week24/09_evaluate_and_cost/labs/lab02_threshold_sweep.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
✓ 12 saved answers · 0 new API calls · $0
│ threshold  coverage                accepted acc  wrong accepted
│ 0.50       92%       ███████████░  82%           2
│ 0.70       92%       ███████████░  82%           2
│ 0.80       83%       ██████████░░  90%           1
│ 0.90       75%       █████████░░░  100%          0
│ 0.95       75%       █████████░░░  100%          0
│ 0.99       75%       █████████░░░  100%          0
│ ref gate   83%       ██████████░░  90%           1
│ t3  top p 0.78  pred hvac       gold facility_ops  ✕  → auto only while threshold ≤ 0.78
│ i3  top p 0.86  pred energy_mv  gold hvac          ✕  → auto only while threshold ≤ 0.86
```

ที่ 0.90 ความผิดพลาดทั้งสองไปที่การรีวิว และ wrong-accepted ลดลงเป็น 0 โดยแลกกับแถวที่คนต้องดูเพิ่มหนึ่งแถว **แต่:** เราเลือก 0.90 *จากการดู 12 แถวนี้* ถ้ารายงานผลบนแถวชุดเดียวกันนี้ ก็เท่ากับตรวจการบ้านตัวเอง workflow ที่ซื่อตรงคือแบ่งข้อมูลเป็นสามส่วน:

| ส่วน | ใช้ทำอะไร |
|---|---|
| development | เขียนและแก้คำถาม |
| calibration | เลือก threshold แล้ว **ตรึง** ไว้ |
| test (กันไว้ต่างหาก) | ตัวเลขที่คุณรายงาน แตะแค่ครั้งเดียว |

ความผิดพลาดที่เกิดขึ้นที่ top p = 1.00 ไม่มี threshold ใดจับได้ จับได้ด้วยคำถามที่ดีขึ้น label ที่ดีขึ้น หรือการตรวจซ้ำอีกชั้นเท่านั้น

✓ Checkpoint: คุณบอกได้ว่า threshold ไหนให้ wrong-accepted เป็น 0 บนชุดนี้ และอธิบายได้ว่าทำไมคุณยังไม่ควรนำไปใช้จริงโดยอิงจากชุดนี้

## 4 · ตัวประเมินผลต้นแบบ

`week24/jev_lab/jev_lab.py` (สคริปต์ต้นฉบับของบทเรียน) มีตัวประเมินผลแบบเดียวกันอยู่ในตัว รันกับไฟล์เดียวกันเพื่อดูว่าตัวเลขตรงกัน ต้องมีคีย์ใน shell ของคุณ ส่วน ⌨ Terminal ที่อยู่ในแอปมีคีย์อยู่แล้ว

```bash
set -a; source <(grep '^TYPESAFE_API_KEY=' .env); set +a
.venv/bin/python week24/jev_lab/jev_lab.py evaluate intent \
  --input week24/09_evaluate_and_cost/data/intent_eval.jsonl --limit 12 --live
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
{
  "rows": 12,
  "api_or_validation_failures": 0,
  "raw_accuracy_successes_only": 0.8333333333333334,
  "recommendation_coverage_all_rows": 0.8333333333333334,
  "accepted_route_accuracy": 0.9,
  "wrong_accepted_routes": 1,
  "p50_ms_successes_only": 493.27,
  "p95_ms_successes_only": 744.47,
  "reported_cost_usd_successes_only": null,
  "note": "Toy dataset, not a production benchmark. Failures count against coverage. …"
}
```

`reported_cost_usd` เป็น `null` เพราะ API ของ TypeSafe โดยตรงส่งคืนจำนวน token แต่ไม่ส่งตัวเลขเป็นดอลลาร์ (OpenRouter ส่งคืน `usage.cost`) ต้นทุนที่ไม่มีข้อมูลคือ **ไม่ทราบ** ไม่ใช่ศูนย์ ให้คำนวณจาก token แบบที่แล็บ 03 ทำ

✓ Checkpoint: ผลรันต้นแบบของคุณตรงกับ accuracy, coverage และ wrong-accepted ของแล็บ 01

## 5 · ต้องจ่ายเท่าไร? (แล็บ 03)

Jev คิดเงิน **เฉพาะ input token**: **$0.042 ต่อล้าน token** ส่วน output ฟรี แล็บ 03 เป็นการคำนวณล้วน ๆ จึงไม่ต้องใช้คีย์

```bash
.venv/bin/python week24/09_evaluate_and_cost/labs/lab03_cost_calculator.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
│ requests   input tokens @1,000 each  classifier cost
│ 1          1,000                     $0.000042
│ 10,000     10,000,000                $0.42
│ 100,000    100,000,000               $4.20
│ 1,000,000  1,000,000,000             $42.00
◆ measured: 12 intent calls, avg 863 input tokens (7 questions + guard text + state)
│ 1,000,000  862,833,333   $36.24
│ one batched call  860               $36.1 per million msgs
│ 7 separate calls  2,240             $94.1 per million msgs
◆ batching saves 62% here — and one round-trip instead of seven.
✓    10,000 req/day → peak ≈ 35 req/min, 499 tok/s fits
⚠   500,000 req/day → peak ≈ 1,736 req/min, 24,966 tok/s → queue, batch, or ask for a higher limit
```

ข้อสรุปสามข้อ:

1. **คำถามของคุณแพงกว่าข้อความของคุณ** ในที่นี้ข้อความสั้น ๆ ใช้ประมาณ 30 token ส่วนอีกประมาณ 830 token ส่วนใหญ่เป็นคำถาม 7 ข้อพร้อม criteria rubric ที่ยาวต้องจ่ายทุกครั้งที่เรียก
2. **รวมคำถามที่ใช้ state เดียวกันไว้ในการเรียกเดียว (batch)** state และส่วนนำจ่ายแค่ครั้งเดียวต่อการเรียก
3. **rate limit ด้านจำนวนคำขอ (1,200 ต่อนาที) มาถึงก่อน limit ด้าน token** สำหรับข้อความสั้น ๆ งานที่มีช่วงพุ่งสูงระดับ 500k ต่อวันต้องมีคิว

การจัดประเภทหนึ่งล้านครั้งในราคาประมาณ $36 ถือว่าถูก นั่นคือเหตุผลที่ตัวจัดประเภทแทบไม่เคยเป็นตัวที่ทำให้ต้นทุนสูง: โมเดล generative การลองใหม่ (retry) งานวิศวกรรม และการรีวิวโดยคนแพงกว่ามาก ให้ประเมินราคาทั้ง workflow ไม่ใช่แค่ API ตัวเดียว

✓ Checkpoint: คุณประเมินค่าใช้จ่าย Jev รายเดือนสำหรับอีเมลเข้ากล่อง 20,000 ฉบับต่อวันได้ จากจำนวน token ต่อการเรียกที่คุณวัดเอง

## Labs — รันได้ที่นี่

**labs/lab01_smoke_eval.py** — 12 แถวที่มี label ทั้ง EN/TH → accuracy, coverage, wrong-accepted, confusion matrix, latency และต้นทุน

**labs/lab02_threshold_sweep.py** — ตั้ง threshold ใหม่ให้คำตอบที่บันทึกไว้ ตั้งแต่ 0.5 ถึง 0.99 โดยไม่เรียกใหม่เลย

**labs/lab03_cost_calculator.py** — คำนวณล้วน ๆ: ตารางราคา การประหยัดจากการ batch และช่องว่างก่อนชน rate limit

ให้รันแล็บ 01 ก่อน แล็บ 02 และ 03 นำคำตอบที่มันบันทึกไว้ (`.runs/intent_answers.json`) มาใช้ต่อ

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 09 — ชุดทดสอบของคุณเอง ตัวชี้วัดของคุณเอง** เปิดไฟล์ `week24/09_evaluate_and_cost/exercises/ex09_build_a_testset.py`:

1. **TODO 1** — เขียนแถวที่มี label อย่างน้อย 6 แถวสำหรับ router ของ Alto Copilot: อย่างน้อย 2 แถวเป็นภาษาไทย และอย่างน้อย 1 แถวที่คำตอบที่ถูกคือ `unknown` ให้เขียนคำขอที่ *คุณ* จะได้รับจริง ๆ
2. **TODO 2** — เขียน `macro_f1(pairs)`: ค่าเฉลี่ยของ F1 ในทุกคลาส คอมเมนต์ในไฟล์อธิบาย TP, FP, FN ไว้ให้แล้ว

ตัวตรวจจะตรวจแถวของคุณและทดสอบ `macro_f1()` กับ fixture สี่ชุด **ก่อน** เรียก API ใด ๆ เมื่อทุกอย่างเป็น ✓ แล้ว มันจะประเมินแถวของคุณแบบ live

```bash
.venv/bin/python week24/09_evaluate_and_cost/exercises/ex09_build_a_testset.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
✓ 7 rows
✓ 2 Thai rows
✓ includes an 'unknown' row
✓ macro_f1 fixture → 0.667
▣ STEP 2 · ask Jev about your 7 rows
│ my-5  en     facility_ops    facility_ops    0.63   ✓      review
│ my-7  en     unknown         unknown         0.90   ✓      review
◆ raw accuracy 100% · macro-F1 1.00 · coverage 71% · wrong accepted 0
```

<details><summary>คำใบ้ — macro_f1 ในห้าบรรทัด</summary>

รวบรวม `classes = {g for g, _ in pairs} | {p for _, p in pairs}` สำหรับแต่ละคลาส นับ TP, FP และ FN ด้วย `sum(...)` บน pairs คำนวณ `2*tp / (2*tp + fp + fn)` แล้วหาค่าเฉลี่ย กันการหารด้วยศูนย์ในกรณีที่คลาสนั้นไม่มี TP, FP หรือ FN เลย

</details>

<details><summary>ท้าทายเพิ่ม — ทำให้ router พัง</summary>

เขียน 3 แถวที่ *ตั้งใจ* ให้มันพัง: ประโยคปฏิเสธ ("don't schedule maintenance, just tell me the chiller COP"), คำย่อปนไทย-อังกฤษ ("AHU-2 ช่วยเช็ค SAT หน่อย") และคำถามต่อเนื่องที่ไม่มีบริบท ("and the other building?") ชุดทดสอบที่ไม่มีกรณียาก ๆ ไม่ได้วัดอะไรเลย

</details>

✓ Checkpoint: บรรทัด fixture ทั้งหมดเป็น ✓ และคุณติด label ให้อย่างน้อยหนึ่งแถวที่คุณ *ไม่* แน่ใจ พร้อมจดเหตุผลไว้

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| แล็บ 02 บอกว่า `.runs/intent_answers.json not found` | รันแล็บ 01 ก่อน จากนั้นแล็บ 02 จะเรียก 12 ครั้งเองหนึ่งรอบ |
| latency ขึ้นว่า `no latency in DRY mode` | คำตอบที่เล่นซ้ำไม่มีการจับเวลา ให้สลับเป็น ⚡ Live เพื่อวัดผล |
| accuracy ต่างจากผลลัพธ์ที่ควรเห็นเล็กน้อย | เป็นเรื่องปกติ: อาจเป็นโมเดลเวอร์ชันอื่น (ดูฟิลด์ `model`) หรือเป็นแถวที่ก้ำกึ่ง ให้ดูที่แถว ไม่ใช่ที่เปอร์เซ็นต์ |
| `expected.intent must be one of …` | สะกด label ในไฟล์ข้อมูลผิด label ต้องตรงกับ key ของ `criteria` ทุกตัวอักษร |
| `reported_cost_usd: null` | API ของ TypeSafe ส่งคืนจำนวน token ไม่ใช่ดอลลาร์ ให้คูณ `input_tokens` ด้วย $0.042 / 1,000,000 |

## Next — ต่อไป

ไปต่อที่ [แล็บ 10 — Jev เทียบ Laya: API ที่โฮสต์ไว้ หรือ open weights](../10_jev_vs_laya/TUTORIAL.th.md) แล็บนี้ถามว่าเมื่อไรที่ตัวจัดประเภทที่โฮสต์เองจึงสมเหตุสมผล และจะเปรียบเทียบสองโมเดลอย่างยุติธรรมได้อย่างไร
