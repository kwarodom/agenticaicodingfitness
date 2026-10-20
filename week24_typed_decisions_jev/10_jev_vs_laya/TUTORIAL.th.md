# ▶ Jev Lab 10 — Jev กับ Laya: ใช้ API แบบ hosted หรือ open weights?

> ส่วนหนึ่งของ Week 24 · การตัดสินใจแบบมี type ด้วย Jev Laya เป็นโมเดลแบบ open weights ที่มี primitive สามแบบเหมือนกัน (`choice`, `noul`, `score`) แล็บนี้ว่าด้วย *การเลือกรูปแบบการ deploy* และ *การเปรียบเทียบสองโมเดลอย่างยุติธรรม* เรารันฝั่ง Jev แบบ live ส่วนฝั่ง Laya เป็นทางเลือกเสริม

**สิ่งที่คุณจะได้ลงมือทำ**
- เปรียบเทียบ managed inference (Jev) กับ inference ที่คุณดูแลเอง (Laya) ทีละมิติ
- เรียนรู้ว่าทำไมผลเปรียบเทียบสองชิ้นที่ตีพิมพ์ออกมาดูเหมือนขัดกัน และทำไมมันไม่ใช่การทดลองเดียวกัน
- ตรวจสอบ benchmark harness ที่ใช้ร่วมกันแบบออฟไลน์ (46 + 32 การตรวจ) แล้ววางแผนการรันโดยไม่เสียเงินเลย
- รัน benchmark 28 แถวฝั่ง Jev แบบ live และอ่านรายงานแบบผู้รีวิว
- ทดสอบการปฏิเสธ ("do NOT turn off the chiller") ทั้งภาษาอังกฤษและภาษาไทย
- เขียน "กรรมการ" ที่ตัดสินว่าผลรัน benchmark สองชุดเปรียบเทียบกันได้หรือไม่

**Time** ~35 นาที · **Difficulty** ระดับกลาง · **Cost** ≈ $0.001 แบบ live (28 + 8 ครั้ง) · $0 แบบ dry · Laya เป็นทางเลือกเสริม

## 0 · ความต่างในประโยคเดียว

**Jev** เป็น *managed API*: คุณส่ง `state` + `questions` ไปยังเซิร์ฟเวอร์ของ TypeSafe และจ่ายตาม input token **Laya** เผยแพร่ *model weights* (Apache-2.0) และ Python runtime ให้คุณรันบน CPU/GPU ของตัวเอง ชนิดคำถามเหมือนกัน แต่ความรับผิดชอบต่างกัน

| มิติ | Jev | Laya |
|---|---|---|
| รูปแบบการให้บริการ | TypeSafe API แบบ hosted (ใช้ผ่าน OpenRouter ได้ด้วย) | checkpoint ที่ดาวน์โหลดมารันเอง |
| Primitive | `choice`, `noul`, `score` | สามแบบเดียวกัน ใช้รูปแบบ state/questions เหมือนกัน |
| การล็อกเวอร์ชัน | ล็อก API id เช่น `jev-1.13.0` | ล็อก weights + runtime ที่แน่นอน (`laya==0.3.20`) ด้วยตัวเอง |
| Context | 64k ต่อ request; 32k สำหรับ state + คำถามที่ยาวที่สุด | ค่าเริ่มต้น 512 หรือ 1,024 token (multilingual ได้ถึง 8,192 ถ้าตั้งค่า) |
| การเทรนเพิ่ม | ไม่มี — ปรับแต่งผ่าน state, instructions, criteria | มีเอกสารเรื่อง fine-tuning และ temperature fitting |
| ภาษา | เน้นภาษาอังกฤษ; ต้องทดสอบภาษาไทยเอง | checkpoint multilingual อ้างว่ารองรับ 100+ ภาษา; ต้องทดสอบภาษาไทยเอง |
| ขอบเขตความเป็นส่วนตัว | state ที่เลือกจะถูกส่งไปยังผู้ให้บริการ | inference อยู่บนฮาร์ดแวร์ของคุณได้; แต่ log และการเรียก cloud ภายหลังยังต้องมีการควบคุม |
| ค่าใช้จ่าย | $0.042 ต่อล้าน input token, output ฟรี | ไม่มีบิลรายครั้ง; แต่จ่ายค่าฮาร์ดแวร์ กำลังเครื่องที่ว่าง งานวิศวกรรม และการดูแลรักษา |
| สิ่งที่คุณต้องรับผิดชอบ | คุณภาพ input, การประเมิน, การจัดการความล้มเหลว, นโยบายการกระทำ | ทั้งหมดนั้น **บวก** ฮาร์ดแวร์, dependency, วงจรชีวิตของ checkpoint, capacity, ความปลอดภัยของบริการ |

เริ่มจาก **ขอบเขตของข้อมูลและเป้าหมายด้านคุณภาพ** แล้วค่อยเลือกโครงสร้างพื้นฐาน อย่าเริ่มจากตัวเลขมิลลิวินาทีที่โฆษณาว่าต่ำที่สุด

✓ Checkpoint: คุณยกตัวอย่างงานหนึ่งของ AltoTech ที่ขอบเขตความเป็นส่วนตัวสนับสนุนให้ทำ inference ในเครื่อง และอีกงานหนึ่งที่ context ขนาดใหญ่ของ Jev สนับสนุนให้ใช้ API แบบ hosted ได้

## 1 · Laya คือสาม checkpoint ไม่ใช่โมเดลเดียว

สเปกเหล่านี้ **รายงานโดยโปรเจกต์ Laya** เราไม่ได้วัดเอง:

| Checkpoint | Backbone | ความยาวเริ่มต้น | ควรทดสอบกับ |
|---|---|---|---|
| `english` | ModernBERT-large, 421M | 512 token | label ภาษาอังกฤษสั้น ๆ, การคัดแยก inbox |
| `multilingual` | mmBERT-base, 322M | 1,024 (ได้ถึง 8,192 ถ้าตั้งค่า) | คำขอภาษาไทย/อังกฤษ |
| `typed-decisions` | ModernBERT-large, 421M | 1,024 token | งานใบแจ้งหนี้ ซัพพอร์ต ความปลอดภัย และ agent-trace ที่คล้ายข้อมูลที่ใช้เทรน |

**กับดักการตัดข้อความแบบเงียบ ๆ** Laya สงวนไว้ราว 192–256 token สำหรับคำถามและตัวเลือก ส่วนที่เหลือ (~320–768 token) เป็นของ state ตัว predictor ปกติจะ *ตัดส่วนที่เกินทิ้งเงียบ ๆ* label ด้าน HR หรือเหตุขัดข้องที่ดูมั่นใจ อาจมาจากโมเดลที่ไม่เคยเห็นย่อหน้าที่เกี่ยวข้องเลย benchmark runner ที่ใช้ร่วมกันจึง **ปฏิเสธ** ที่จะรันแถวที่จะถูกตัด (`laya_head_would_truncate`) แทนที่จะให้คะแนน input ที่ไม่ครบ

✓ Checkpoint: คุณอธิบายได้ว่าทำไมคำตอบที่มั่นใจแต่มาจาก input ที่ถูกตัด จึงแย่กว่า error

## 2 · ทำไมผลเปรียบเทียบสองชิ้นที่ตีพิมพ์จึง "ขัดกัน"

**พูดง่าย ๆ:** บทความสองชิ้นดูเหมือนประกาศผู้ชนะคนละคน แต่จริง ๆ แล้ววิ่งกันคนละสนาม ชิ้นหนึ่งเปรียบเทียบ Laya แบบยังไม่ปรับแต่งกับ Jev บน benchmark เดียว อีกชิ้นเปรียบเทียบ Laya ที่ถูกเทรนมาเฉพาะงาน กับตัวเลขของ Jev ที่คัดลอกมาจากที่อื่น โดยใช้ prompt และขนาดชุดทดสอบต่างกัน เหมือนเอาเวลาของนักวิ่งคนหนึ่งบนลู่เรียบ ไปเทียบกับเวลาของอีกคนที่วิ่งขึ้นเนิน ไม่มีผลไหนบอกได้ว่าโมเดลไหนดีกว่า *สำหรับ AltoTech* มีแต่การทดสอบที่ยุติธรรมบนข้อมูลของคุณเอง (ตัวตัดสินความยุติธรรมของการทดลองนำร่อง ที่คุณเขียนในแบบฝึกหัดของบทนี้) เท่านั้นที่บอกได้


| แหล่งที่มา | รายงานว่า | สิ่งที่ถูกเปรียบเทียบจริง |
|---|---|---|
| บล็อกชุมชน Hugging Face | Jev 74.4 vs Laya ที่ยังไม่ปรับแต่ง 54.4 (JevBench composite, 534 การตัดสินใจ) | เป็นคะแนนรวม ไม่ใช่ % ที่ถูก; เป็นแหล่งทุติยภูมิ เราไม่ได้ทำซ้ำ |
| บทความของ ZimaSpace | Laya ที่ fine-tune แล้ว 0.766 vs Jev 0.727 accuracy | model card ของ Laya ระบุว่าโปรเจกต์นั้น **ไม่ได้รัน** Jev เอง; prompt และขนาดตัวอย่างต่างกัน |

ข้อมูลเพิ่มจาก typed-decisions card ของ Laya (400 test case, 2,000 การตัดสินใจ, fine-tune บนชุดแยก 1,200 case):

| Metric | Laya ที่ fine-tune แล้ว | Jev (ค่าอ้างอิงที่ตีพิมพ์) | หมายเหตุ |
|---|---:|---:|---|
| Accuracy ↑ | 0.766 | 0.727 | prompt/ขนาดตัวอย่างต่างกัน |
| Soft accuracy ↑ | 0.471 | 0.580 | เป็น metric คนละตัวกับ hard accuracy |
| Brier ↓ | 0.062 | 0.148 | ความคลาดเคลื่อนของความน่าจะเป็น |
| ECE ↓ | 0.213 | 0.144 | แม่นกว่า ≠ calibrate ดีกว่า |
| Score MAE ↓ | 0.242 | 0.391 | ความคลาดเคลื่อนเชิงลำดับ |

checkpoint ภาษาอังกฤษของ Laya ที่ *ยังไม่ปรับแต่ง* ได้คะแนน **0.362** บนชุดทดสอบเดียวกันนั้น ห้ามนำผลของโมเดลที่ถูกปรับแต่งเฉพาะทางไปใช้แทนทุก checkpoint สิ่งเหล่านี้คือ **การทดลองคนละแบบ** ไม่ใช่การศึกษาแบบควบคุมสองชิ้นที่ได้ผู้ชนะตรงข้ามกัน ในแบบฝึกหัด คุณจะได้เขียนกรรมการที่บอกเรื่องนี้

✓ Checkpoint: คุณบอกได้สามเหตุผลที่ตัวเลข 0.766 vs 0.727 ไม่ใช่ชัยชนะแบบตัวต่อตัว

## 3 · ทดสอบ harness ก่อนเชื่อตัวเลขของมัน (แล็บ 01)

benchmark ก็คือซอฟต์แวร์ และซอฟต์แวร์มีบั๊ก ก่อนเปรียบเทียบโมเดล ให้พิสูจน์แบบออฟไลน์ว่า harness ส่ง **input ที่เหมือนกันทุกไบต์** ให้ทั้งสองโมเดล นับความล้มเหลว และปฏิเสธการตัดข้อความ

```bash
.venv/bin/python week24/10_jev_vs_laya/labs/lab01_benchmark_offline_tests.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 1 · jev_lab.py selftest — request/response contracts for all 7 labs
✓ 46 contract checks passed · live inference tested: False
▣ STEP 2 · python -m unittest test_jev_laya_benchmark -v
✓ gold never in state
✓ mocked end to end equal inputs and outputs
✓ preflight state overflow
✓ setup failure prevents jev billing
✓ record failure no secret
…
◆ Ran 32 tests in 0.186s · 32 passed · 0 failed
⚠ NOT tested: real model quality, Thai accuracy, latency. Tests use mocked answers.
```

✓ Checkpoint: contract check 46 ข้อ และ unit test 32 ข้อผ่านทั้งหมด โดยไม่ต้องใช้ key และไม่ต้องใช้เครือข่าย

## 4 · วางแผนก่อนจ่ายเงิน (แล็บ 02)

`plan` สร้าง corpus คำนวณ hash นับจำนวนการเรียก และแสดงตัวอย่าง request หนึ่งรายการ โดยไม่ส่งอะไรออกไปเลย

```bash
.venv/bin/python week24/10_jev_vs_laya/labs/lab02_benchmark_plan.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
◆ 28 rows · by lab {'intent': 11, 'email': 6, 'hr': 2, 'afdd': 2, 'lead': 2, 'rag': 2, 'point': 3}
◆ languages {'en': 23, 'th': 4, 'mixed': 1} · 43 gold labels (partial on purpose — unlabeled answers are never scored)
◆ mode dry_run_no_inference · providers ['jev', 'laya'] · split smoke · rows 28
◆ max logical calls per provider: 28 (each Jev call may retry up to 4 HTTP attempts)
◆ corpus hash 1ddbb974afc82c82… · 7 question schemas hashed
» state     {"message":"Why is AHU-3 not cooling the hotel lobby? Check yesterday's trends."}
◆ providers ['jev'] · 28 logical Jev calls · est. ≈ $0.0011 at ~900 tok/call
```

สังเกตว่า `state` เป็น **JSON string** ไม่ใช่ object: runner แปลง state เป็นข้อความครั้งเดียวแล้วส่งข้อความเดียวกันให้ทั้งสองโมเดล จึงไม่มีโมเดลไหนได้รูปแบบที่ง่ายกว่า hash ของ corpus และ schema ช่วยให้ใครก็ตามตรวจสอบภายหลังได้ว่าการรันสองครั้งใช้ข้อมูลและคำถามชุดเดียวกัน

✓ Checkpoint: คุณบอกได้ว่าการรัน Jev อย่างเดียว 28 แถวต้องจ่ายค่าเรียกกี่ครั้ง และทำไมจำนวน HTTP attempt จริงอาจสูงกว่านั้น

## 5 · รันฝั่ง Jev แบบ live (แล็บ 03)

นี่คือ harness จริงบน API จริง: 28 แถวสมมติที่ครอบคลุม schema ของทั้งเจ็ดแล็บ ทั้ง EN, TH และภาษาผสม ใช้เวลาประมาณ 15 วินาที และเสียค่าใช้จ่ายราวหนึ่งในสิบของหนึ่งเซนต์ ผลลัพธ์จะถูกเขียนลงโฟลเดอร์ **ใหม่** ทุกครั้ง runner ไม่ยอมเขียนทับหลักฐานเดิม

```bash
.venv/bin/python week24/10_jev_vs_laya/labs/lab03_jev_only_benchmark.py
```

การรันแบบเดียวกันด้วยมือ ในเทอร์มินัลของคุณเอง:

```bash
set -a; source <(grep '^TYPESAFE_API_KEY=' .env); set +a
cd week24/jev_lab
../../.venv/bin/python jev_laya_benchmark.py init --dir /tmp/bench_data
../../.venv/bin/python jev_laya_benchmark.py run --input /tmp/bench_data/smoke.jsonl \
  --providers jev --live-jev --limit 28 --out /tmp/bench-run-01
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
◆ 28 calls · 28 ok · 0 failed · p50 442 ms · p95 1304 ms
◆ 20,142 input tokens · ≈ $0.00085
│ task/question            type    labels  accuracy  coverage  acc. accepted  wrong acc.
│ email/category           choice  6       1.00      1.00      1.00           0
│ hr/python_evidence       choice  2       0.50      1.00      0.50           1
│ intent/complexity        score   1       0.00      0.00      n/a            0
│ intent/intent            choice  11      0.91      0.91      0.90           1
│ point/ambiguous          noul    3       1.00      0.33      1.00           0
│ rag/injection            noul    2       1.00      1.00      1.00           0
│ …
⚠ candidate-demo-2 [hr/python_evidence, en] gold 'unclear' → Jev 'not_stated' (top p 0.99)
⚠ i3 [intent/intent, en] gold 'hvac' → Jev 'energy_mv' (top p 0.87)
⚠ b-mixed-code [intent/complexity, mixed] gold 1 → Jev 0 (top p 0.66)
```

อ่านจุดที่ไม่ตรงกันทั้งสามจุด ไม่ใช่ค่าเฉลี่ย:

- **candidate-demo-2** — ข้อความเขียนว่า "Lists Python as an interest." gold บอก `unclear` แต่ Jev บอก `not_stated` ด้วย 0.99 ทั้งสองการตีความมีเหตุผลรองรับ และ label 0.99 บนกรณีที่ถกเถียงได้ คือเหตุผลที่ผลลัพธ์ด้าน HR ต้องส่งให้ recruiter และห้ามใช้จัดอันดับ
- **i3** — คำขอครึ่ง HVAC ครึ่ง M&V ตัวเดียวกับที่แล็บ 09 ทักไว้ ผลคงที่ทุกครั้งที่รัน ซึ่งมีประโยชน์: label นี้ต้องมีคนตัดสินชี้ขาด
- **b-mixed-code** — คำขอเขียนโค้ดแบบภาษาไทยผสมอังกฤษ ถูกให้คะแนนว่า "ง่าย" แทนที่จะเป็น "ผู้เชี่ยวชาญหนึ่งคน" เมื่อมี label แค่ตัวเดียว accuracy ของคำถามนี้จึงเป็น 0% ตัวอย่างเดียวพิสูจน์อะไรไม่ได้เลยทั้งสองทาง

สังเกตด้วยว่า **p95 = 1,304 ms** ขณะที่ p50 = 442 ms ส่วนหางมีผลต่อประสบการณ์ผู้ใช้ และเป็นจุดที่เครือข่ายและการ retry แสดงผลออกมา นี่เป็นอีกเหตุผลหนึ่งที่ต้องวัด latency แบบ end-to-end ไม่ใช่วัดแค่ความเร็วของโมเดล

✓ Checkpoint: คุณรัน benchmark แบบ live แล้ว และอธิบายได้ว่าทำไม "hr/python_evidence accuracy 0.50" แทบไม่บอกอะไรเลยเกี่ยวกับคุณภาพด้าน HR ของ Jev

## 6 · ทดสอบการปฏิเสธ (แล็บ 04)

repository ของ Laya บันทึกกรณีคำขอยกเลิกแบบปฏิเสธ ที่ยังถูกจัดเป็น "cancel" ด้วยความมั่นใจสูง ส่วน TypeSafe บันทึกว่า Jev อ่าน instruction *ตามตัวอักษร* ไม่ว่าแบบไหน คุณต้องทดสอบเองก่อนให้ตัวจัดประเภทเข้าใกล้อุปกรณ์จริง

```bash
.venv/bin/python week24/10_jev_vs_laya/labs/lab04_negation_probe.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
│ message                                       shutdown?  action        conf  outage?  human agrees
│ Turn off chiller 2 now.                       0.98       turn_off      0.97  0.04     ✓
│ Do NOT turn off chiller 2.                    0.02       keep_running  0.92  0.03     ✓
│ Don't turn off chiller 2 unless the high-pr…  0.03       conditional   0.83  0.07     ✓
│ There is no outage. Please send pricing for…  0.02       no_action     0.96  0.03     ✓
│ ปิดชิลเลอร์ 2 ตอนนี้เลย                       0.97       turn_off      0.96  0.07     ✓
│ ห้ามปิดชิลเลอร์ 2 เด็ดขาด                     0.02       keep_running  0.98  0.04     ✓
◆ 8/8 messages match the human reading on all three questions
```

jev-1.13.0 ผ่านทั้งแปดข้อเมื่อ 2026-09-27 เป็นข่าวดี แต่ไม่ใช่ใบอนุญาต การผ่านขึ้นอยู่กับการใช้ถ้อยคำ: Noul ระบุชัดว่าคำขอแบบ *มีเงื่อนไข* นับเป็น "ไม่" และ Choice มีตัวเลือก `conditional` แยกไว้ ลองใช้ถ้อยคำที่อ่อนกว่าด้านล่าง แล้วดูว่ากรณีมีเงื่อนไขยังตกไปอยู่ตรงที่คุณต้องการหรือไม่

```jev
{
  "state": {"message": "Don't turn off chiller 2 unless the high-pressure alarm clears."},
  "questions": {
    "requests_shutdown": {
      "type": "noul",
      "instructions": "Does `message` mention turning off the chiller?"
    },
    "action": {
      "type": "choice",
      "instructions": "Which equipment action does `message` request?",
      "criteria": {
        "turn_off": "Turn the equipment off",
        "keep_running": "Keep the equipment running"
      }
    }
  }
}
```

ตอนที่เรารัน `requests_shutdown` ได้ **0.99** Jev ไม่ได้ผิด: ข้อความ *กล่าวถึง* การปิดชิลเลอร์จริงตามตัวอักษร และนั่นคือสิ่งที่คำถามถาม policy ที่อ่าน noul นี้ว่า "ผู้ปฏิบัติงานต้องการให้ปิด" จะผิดพลาดอย่างอันตราย Choice ที่อ่อนยังเลือก `keep_running` แต่ก็เพียงเพราะไม่มีตัวเลือก `conditional` ให้แก้ instruction กลับเป็น "ask for the chiller to be turned off now or unconditionally" เพิ่ม `conditional` และ `no_action` แล้วถามอีกครั้ง

✓ Checkpoint: คุณพบถ้อยคำที่ทำให้ข้อความแบบมีเงื่อนไขดูเหมือนคำขอให้ปิดเครื่อง และแก้ไขด้วย instruction หรือตัวเลือกที่ดีขึ้น ไม่ใช่ด้วยการเชื่อคะแนน

## 7 · แต่ละโมเดลเหมาะกับงานไหนที่ AltoTech — และกฎของระบบไฮบริด

สิ่งเหล่านี้เป็นการทดลองที่เสนอ ไม่ใช่ความสามารถที่ deploy แล้ว:

| เวิร์กโฟลว์ | การทดลองแรก | ขอบเขต |
|---|---|---|
| Router ของ Alto Copilot EN/TH | Jev แบบ hosted vs Laya `multilingual` บน input สั้นที่ได้รับอนุญาตชุดเดียวกัน | ไม่แน่ใจ → ขอคำชี้แจง; ห้ามคัดลอก threshold ข้ามโมเดล |
| HR และ inbox ภายใน | ถ้านโยบายห้ามประมวลผลภายนอก ให้ทดสอบ Laya ในเครื่องด้วยหลักฐานที่ลดทอนแล้ว | มีคนตรวจทุกข้อเสนอ |
| การคัดแยกเหตุขัดข้องที่ site gateway | Laya ในกรณีที่ต้องทำงานแบบออฟไลน์ | alarm แบบ deterministic ที่มีอยู่ยังทำงานแยกอิสระ |
| เอกสารยาว | context ขนาดใหญ่ของ Jev ในกรณีที่อนุญาตให้ประมวลผลแบบ hosted | บังคับใช้ขีดจำกัดจริงของผู้ให้บริการ |
| งานขาย / inbox ที่ใช้ร่วมกัน | Jev เมื่อความเร็วในการส่งมอบและภาระ model-ops ที่ต่ำสำคัญที่สุด | คิดราคาทั้งเวิร์กโฟลว์ก่อนซื้อฮาร์ดแวร์ |

ระบบไฮบริดที่คำนึงถึงความเป็นส่วนตัว:

```text
identity + tenant policy resolved in code
  → data minimization + deterministic calculations
  → local Laya proposal
     → accepted only under separately tested policy
     → uncertain:
        → cloud ALLOWED for this data?  → approved minimal context to Jev
        → cloud PROHIBITED?             → local specialist or human review
  → authorization and human approval stay independent
```

**ห้ามส่งข้อมูลให้ Jev เพียงเพราะ Laya ไม่แน่ใจ** ข้อมูลส่งขึ้น cloud ได้หรือไม่ เป็นการตัดสินใจเชิงนโยบายที่ทำ *ก่อน* การยกระดับ ไม่ใช่สิ่งที่ความไม่แน่ใจของโมเดลเป็นคนตัดสิน

✓ Checkpoint: คุณอธิบายได้ว่าทำไม "โมเดลในเครื่องไม่แน่ใจ ก็เลยถาม cloud" เป็นบั๊กด้าน data governance

## 8 · ทางเลือกเสริม — รัน Laya ในเครื่อง

Laya ต้องใช้ environment แยก, PyTorch runtime และต้องดาวน์โหลดโมเดลตอนใช้งานครั้งแรก จึงเป็นทางเลือกเสริม สคริปต์เดโมบันทึกไว้ที่ `week24/10_jev_vs_laya/optional/laya_demo.py` และโหมดเริ่มต้นเป็น dry run ที่ไม่ import อะไรเลย:

```bash
.venv/bin/python week24/10_jev_vs_laya/optional/laya_demo.py
```

ถ้าจะรันจริง ให้ใช้โฟลเดอร์แยก เพื่อให้ `.venv` ของคอร์สสะอาดอยู่เสมอ:

```bash
mkdir -p ~/laya-tutorial && cd ~/laya-tutorial
python3 -m venv .venv && source .venv/bin/activate
python -m pip install "laya==0.3.20" && python -m pip check
cp /Users/altodev/Desktop/agenticaicodingfitness/week24/10_jev_vs_laya/optional/laya_demo.py .
python laya_demo.py --run            # first run downloads the multilingual checkpoint
python laya_demo.py --run --offline  # proves it works from the local cache
```

label ที่แนะนำ (โดยคน) สำหรับตัวอย่างทั้งสาม: ใบแจ้งหนี้ซ้ำ → `finance`, ไม่มี outage; ภาษาไทย "แอร์โรงแรมหยุดทำงาน" → `support`, มีการแจ้ง outage; "There is no outage…" → `sales`, ไม่มี outage การเรียกครั้งแรกรวมเวลาโหลดโมเดลด้วย จึงห้ามเทียบ `call_ms_including_any_lazy_load` ของมันกับการเรียก Jev ที่อุ่นเครื่องแล้ว เมื่อติดตั้ง Laya แล้ว runner ที่ใช้ร่วมกันจะเปรียบเทียบทั้งสองได้: `jev_laya_benchmark.py run --limit 1 --live-jev --live-laya --out run-smoke-01`

✓ Checkpoint (ทางเลือกเสริม): `laya_demo.py` พิมพ์ `dry_run_no_inference` ใน venv ของคอร์ส — หรือถ้าคุณติดตั้ง Laya แล้ว จะได้สามรายการที่ติดป้าย `human_review: true, execute: false`

## แล็บ — รันที่นี่

**labs/lab01_benchmark_offline_tests.py** — รัน contract check 46 ข้อ และ unit test ของ benchmark 32 ข้อแบบออฟไลน์: ไม่ใช้ key ไม่ใช้โมเดล

**labs/lab02_benchmark_plan.py** — สร้าง corpus 28 แถวในโฟลเดอร์ชั่วคราว แล้วพิมพ์แผนแบบ dry run: hash, เพดานจำนวนการเรียก, request ที่ใช้ร่วมกัน

**labs/lab03_jev_only_benchmark.py** — benchmark จริงที่ใช้ร่วมกัน ฝั่ง Jev: เรียก live 28 ครั้ง, metric รายคำถาม, ทุกจุดที่ไม่ตรงกัน

**labs/lab04_negation_probe.py** — ทดสอบการปฏิเสธและเงื่อนไขแปดแบบ ทั้งภาษาอังกฤษและภาษาไทย; ไม่มีการสั่งการอุปกรณ์ใด ๆ

แล็บ 01 และ 02 ไม่เรียก API เลย แล็บ 03 เล่นซ้ำผลการรันจริงที่บันทึกไว้เมื่ออยู่ในโหมด DRY

## ลองทำเอง

**แบบฝึกหัด 10 — เป็นกรรมการ** เปิด `week24/10_jev_vs_laya/exercises/ex10_fair_pilot.py` แบบฝึกหัดนี้ทำงานออฟไลน์ทั้งหมด:

1. **TODO 1** — เติม `PILOT_PLAN`: split, ภาษา, metric, ตำแหน่งที่ปรับจูน threshold, จะเริ่มด้วย shadow mode (รันคู่ขนานเงียบ ๆ) หรือไม่ และกฎการ fallback ของระบบไฮบริด
2. **TODO 2** — เขียน `is_fair_comparison(run_a, run_b)` ให้คืนข้อความปัญหาหนึ่งข้อต่อหนึ่งกฎที่ถูกละเมิด (R1–R7 ใน docstring ของไฟล์) และคืน list ว่างเมื่อผลรันทั้งสองเปรียบเทียบกันได้

```bash
.venv/bin/python week24/10_jev_vs_laya/exercises/ex10_fair_pilot.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
✓ no automatic cloud fallback just because the local model was unsure
│ ✓  identical fair setup                              0                  0
│ ✓  different corpora (the two articles!)             1                  1      R1 different corpus
│ ✓  fine-tuned Laya vs zero-shot Jev                  1                  1      R5 fine-tuned vs zero-shot
│ ✓  smoke split + different prompts + failures hidd…  3                  3      R2 not both on the held-out…
```

<details><summary>คำใบ้ — โครงสร้างของกรรมการ</summary>

ใช้ `if` เจ็ดชุดที่เป็นอิสระต่อกัน หนึ่งชุดต่อหนึ่งกฎ แต่ละชุดเพิ่มข้อความของตัวเอง อย่า `return` ก่อนเวลา: ผู้รีวิวต้องการเห็นปัญหา **ทุกข้อ** พร้อมกัน ไม่ใช่แค่ข้อแรก

</details>

<details><summary>ท้าทายเพิ่ม — เพิ่ม R8</summary>

เพิ่มกฎเรื่องความยุติธรรมของ latency: เมื่อเปรียบเทียบ latency ผลรันทั้งสองต้องบันทึก `hardware` และ `warmup_calls` และห้ามเทียบการเรียก Laya ครั้งแรกแบบเย็น กับการเรียก Jev ที่อุ่นเครื่องแล้ว เพิ่ม fixture ที่ละเมิดกฎนี้

</details>

✓ Checkpoint: fixture ทั้งหกเป็น ✓ และแผนของคุณระบุ `cloud_fallback_when_local_uncertain: False`

## การแก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `Missing selected Jev provider key` | runner อ่าน `TYPESAFE_API_KEY` จาก environment เท่านั้น แล็บ 03 ส่งให้คุณแล้ว; ถ้ารันด้วยมือ ให้ `source` ก่อน |
| `Output directory exists` | ตั้งใจให้เป็นแบบนี้ — ผลลัพธ์คือหลักฐาน ใช้โฟลเดอร์ `--out` ใหม่ |
| `Explicit --live-jev / --live-laya required` | runner จะไม่ใช้เงินเลยถ้าไม่มี flag ของผู้ให้บริการแต่ละราย |
| `ModuleNotFoundError: laya` ตอน `--run` | เป็นเรื่องปกติใน venv ของคอร์ส — Laya เป็นทางเลือกเสริม; ใช้ venv แยกที่ `~/laya-tutorial` |
| `laya_head_would_truncate` | เป็นตัวป้องกันความยุติธรรม ไม่ใช่คำตอบของโมเดล: ย่อ rubric ให้สั้นลงสำหรับ **ทั้งสอง** โมเดล ในการทดลองใหม่ที่มีเวอร์ชันกำกับ |
| ตัวเลขของแล็บ 03 ต่างจาก expected output | ปกติ เป็นความต่างระหว่างการรันและระหว่างเวอร์ชันบน label 1–11 ตัวต่อคำถาม — ให้อ่านรายการจุดที่ไม่ตรงกัน |

## ถัดไป

ไปต่อที่ [Lab 11 — Jev + LLM ที่คุณเลือก](../11_jev_plus_llm/TUTORIAL.th.md): Jev ตัดสินใจ แล้ว Claude / ChatGPT / Gemini / DeepSeek / Kimi / GLM เป็นคนเขียน จากนั้นไปที่ capstone: [Lab 12](../12_capstone_hotel_copilot/TUTORIAL.th.md) คุณจะได้นำทุกอย่างมารวมกัน: คำถามแบบ batch, การดึงข้อมูลด้วยโค้ด, uncertainty gate, safety override และการเปรียบเทียบแบบ shadow mode
