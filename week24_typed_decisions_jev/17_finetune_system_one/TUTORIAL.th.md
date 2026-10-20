# ▶ Jev Lab 17 — ปรับจูน (fine-tune) โมเดล System One ในเครื่อง: Mac หรือ DGX Spark และจะรู้ได้อย่างไรว่าได้ผลจริง

> ส่วนหนึ่งของ Week 24 · การตัดสินใจแบบมีชนิด (typed decision) ด้วย Jev — ในบทที่ 16 Laya ตัวตั้งต้นจัดคิว ticket ถูก 12 จาก 18 ใบ และส่งเรื่องกลิ่นไหม้ภาษาไทยไปฝ่ายประปาด้วยความมั่นใจ 0.95 ผู้สร้างเองเรียกมันว่า "a fast base to specialise, not a zero-shot decision engine" บทนี้จึงทำให้มันเชี่ยวชาญงานของเรา คุณจะ fine-tune **Laya** (fine-tune ทั้งโมเดล บน GPU ของ Apple) และ **GLiNER2.5-multi-Decide** (ทั้งแบบ LoRA adapter และ fine-tune ทั้งโมเดล) ด้วย ticket โรงแรม 456 ใบ บน Mac หรือ DGX Spark แล้วทำส่วนที่บทเรียนส่วนใหญ่ข้ามไป คือกำหนด **ก่อน** เทรนว่าความสำเร็จหมายถึงอะไร และให้คะแนนทุกโมเดลด้วย ticket ที่มันไม่เคยเห็น

**สิ่งที่คุณจะได้ลงมือทำ**
- สร้างชุดข้อมูลจากสถานการณ์ที่เขียนด้วยมือ 72 แบบ และจับได้ว่าการแบ่งแบบสุ่มทำให้ validation รั่วเข้าไปในข้อมูลเทรนถึง 85 %
- วัดหน่วยความจำและความเร็วบนเครื่องของคุณ แล้วตัดสินใจเลือก Mac หรือ DGX Spark จากตัวเลข ไม่ใช่จากความรู้สึก
- ล็อกเกณฑ์ความสำเร็จเจ็ดข้อไว้ในไฟล์ **ก่อน** การเทรนครั้งแรก และพบว่า keyword สิบบรรทัดชนะโมเดลในเครื่องแบบ zero-shot ทุกตัว
- fine-tune Laya ในเวลาราว 4 นาทีบน M4 Max แล้วดูมันแม่นขึ้น *และ* มั่นใจเกินจริงไปพร้อมกัน
- เทรน LoRA adapter ของ GLiNER ขนาด 5 MB และ fine-tune ทั้งโมเดลขนาด 1.1 GB แล้วดู early stopping เก็บ epoch ที่ถูกต้องไว้
- ให้คะแนนทุกอย่างด้วย ticket held-out 66 ใบ: ช่วงความคลาดเคลื่อน สิ่งที่แก้ได้เทียบกับสิ่งที่พัง calibration การลืม (forgetting) และความเร็ว
- ให้ด่านตรวจ (gate) ไม่ใช่อารมณ์ของคุณ เป็นผู้ตัดสิน SHIP หรือ HOLD และเขียน model card

**Time** ~60 นาที (+ การเทรนราว 20 นาทีที่รันเบื้องหลังได้) · **Difficulty** ระดับกลางถึงสูง · **Cost** ≈ $0.002 เมื่อรันจริง (Jev เป็น baseline และครู; การเทรนเสีย $0) · $0 ในโหมด dry

## 0 · ก่อน fine-tune: ใช่เครื่องมือที่ถูกหรือเปล่า?

fine-tune คือการเปลี่ยน weights ของโมเดล เป็นวิธีแก้ที่แพงที่สุด ดังนั้นลองวิธีที่ถูกกว่าก่อน:

| อาการ | วิธีแก้ที่ถูกที่สุด | บทที่ |
|---|---|---|
| คำตอบเปลี่ยนเมื่อเรียบเรียงคำถามใหม่ | เขียน instructions และ criteria ใหม่ | 03 |
| ตอบถูก แต่ความมั่นใจผิด | fit threshold หรือ temperature ใหม่บน validation | 09 และ §6 ของบทนี้ |
| ภาษาหนึ่งอ่อน | ใช้ checkpoint หลายภาษาของผู้ผลิต | 16 |
| ผิดสม่ำเสมอกับคำศัพท์ **ของคุณ** และข้อมูลต้องอยู่ในเครื่อง | **Fine-tune** | 17 |

บทนี้ใช้ environment `week24_typed_decisions_jev/.venv-local` จากบทที่ 16 (Laya 0.4.8, gliner2 2.1.0, PEFT 0.21) ไม่ต้องติดตั้งอะไรเพิ่ม ถ้าคุณข้ามบทที่ 16 ให้ทำ §0 ของบทนั้นก่อน ทุกแล็บจะรันตัวเองใหม่ภายใน `.venv-local` ถ้าไม่มี แล็บจะเล่นซ้ำผลที่บันทึกไว้เมื่อ 2026-10-17 (ติดป้าย `RECORDED`)

แล็บเทรนสองตัวใช้เวลานานกว่าขีดจำกัด 150 วินาทีของปุ่ม ▶ ของ runner ให้เริ่มจาก ⌨ terminal (จำกัด 600 วินาที) หรือ terminal ของคุณเอง แล้วอ่านต่อระหว่างที่มันรัน:

```bash
week24_typed_decisions_jev/.venv-local/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab04_finetune_laya.py --train
week24_typed_decisions_jev/.venv-local/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab05_finetune_gliner.py --train lora
week24_typed_decisions_jev/.venv-local/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab05_finetune_gliner.py --train full
```

checkpoint จะถูกเขียนไปที่ `17_finetune_system_one/.runs/` (gitignore ไว้แล้ว): ราว 1.3 GB สำหรับ Laya, 1.1 GB สำหรับ GLiNER แบบเต็ม และ 5 MB สำหรับ LoRA adapter

✓ Checkpoint: คุณบอกได้หนึ่งปัญหาที่ fine-tune ไม่ใช่ขั้นแรกที่ถูกต้อง และรู้ว่า checkpoint จะถูกเขียนไว้ที่ไหน

## 1 · สร้างชุดข้อมูลที่ไว้ใจได้

```bash
.venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab01_build_dataset.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
» 72 situations × 8 variants = 576 tickets (336 English, 240 Thai)
│ split                       train  val  val with a near-copy (≥0.5)  closest pair
│ random (shuffle tickets)    461    115  85%                          1.00
│ by situation (group split)  456    120  0%                           0.44
…
│ test (48, hand-written)  0/48                0.22
│ Module 16 (18)           1/18                0.52
…
» Jev (DRY) agrees with the generator: queue 23/24 · safety 22/24 · urgency 22/24   cost ≈ $0.0005
```

ข้อมูลถูกสร้างอย่างไร (`datagen.py`):

- **สถานการณ์ ไม่ใช่ประโยค** สถานการณ์แต่ละแบบจาก 72 แบบ ("ชักโครกน้ำล้น", "ปลั๊กมีประกายไฟ", "เช็คเอาท์ช้า") เขียนไว้ 2–3 แบบเป็นภาษาอังกฤษหรือไทย และติดป้ายกำกับสามป้ายเพียงครั้งเดียว คำขึ้นต้น คำลงท้าย และหมายเลขห้องคูณมันออกมาเป็น 576 ticket ดังนั้นป้าย 576 ชุดใช้การตัดสินใจติดป้ายเพียงราว 70 ครั้ง คำลงท้ายเลือกตามความเร่งด่วน ดังนั้น "No rush." จะไม่ไปอยู่กับเรื่องน้ำท่วม
- **การแบ่งแบบสุ่มคือกับดัก** การสลับ ticket แบบสุ่มทำให้สำเนาที่เรียบเรียงใหม่ของสถานการณ์เดียวกันไปอยู่ทั้งสองฝั่ง: 85 % ของ ticket ใน validation มีสำเนาใกล้เคียงอยู่ในข้อมูลเทรน validation จึงกลายเป็นการวัดความจำ **การแบ่งตามสถานการณ์** ให้สำเนาใกล้เคียง 0 % ทำให้ validation ถามคำถามจริง: โมเดลรับมือสถานการณ์ที่ไม่เคยเห็นได้หรือไม่?
- **ข้อสอบเขียนแยกต่างหาก** `data/test_handwritten.json` มี 48 ticket ที่เขียนในสไตล์อื่น: ภาษาวิทยุสั้น ๆ ของพนักงาน ("RM 1418 - guest says aircon dead"), อีเมลยาว, ไทยปนอังกฤษ, ประโยคปฏิเสธ ("the burning smell was just my toast") และแก๊สรั่วที่นิยามความปลอดภัยที่เขียนไว้ไม่ครอบคลุม มีห้า ticket ที่มี `note` บอกว่าทำไมจึงยาก รวมกับ 18 ticket ของบทที่ 16 จึงได้ **ticket held-out 66 ใบ** ที่ไม่เคยถูกใช้เทรน แล็บพิสูจน์ให้เห็น: ticket ทดสอบใหม่ที่ใกล้ที่สุดคล้ายกับข้อมูลเทรนเพียง 0.22 มีคำขอผ้าเช็ดตัวหนึ่งใบของบทที่ 16 ที่คล้าย 0.52 และแล็บบอกตรง ๆ แทนที่จะซ่อนไว้
- **เทรนแบบเดียวกับที่ใช้งานจริง** ไฟล์เทรนทั้งสองสร้างจากชุดคำถามชุดเดียว `ftkit.Q` ชุดเดียวกับที่ Lab 16-5 ใช้ record ของ GLiNER ผ่าน adapter `to_gliner_schema` ของบทที่ 16 เอง ดังนั้นการเทรนจะเห็น prompt และคำอธิบายป้ายแบบเดียวกับที่ `classify_text()` ส่ง
- **ไม่มีป้าย? ใช้ครู** Jev ติดป้าย 24 ticket ในข้อมูลเทรนด้วยเงิน $0.0005 และเห็นตรงกับ generator 23 จาก 24 ในเรื่องคิว ข้อที่เห็นต่างคือจุดที่คนต้องดู หนึ่งในนั้นคือกระเป๋าสตางค์หาย: front desk หรือ housekeeping? ถ้ามี ticket จริงแต่ไม่มี generator การใช้ Jev เป็นครูแล้วให้คนตรวจคำตอบที่มันไม่แน่ใจ คือวิธีราคาถูกในการสร้างชุดข้อมูลเทรน

✓ Checkpoint: คุณอธิบายได้ว่าทำไมการแบ่งแบบสุ่มจะทำให้ความแม่นยำบน validation สูงเกินจริง และแล็บพิสูจน์อย่างไรว่าชุดทดสอบที่เขียนด้วยมือไม่ได้ซ่อนอยู่ในข้อมูลเทรน

## 2 · Mac หรือ DGX Spark?

```bash
.venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab02_mac_or_spark.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น — Apple M4 Max, 36 GiB)

```
│ Laya trainer uses     mps  (order: cuda → mps → cpu)
│ GLiNER2 trainer uses  cpu  (cuda, else cpu — never mps)
│ mixed precision       CUDA only — on a Mac both train in fp32
…
│ model                   parameters  weights fp32  full fine-tune state  LoRA r=8 state
│ Laya multilingual       322 M       1.3 GB        ≈ 5.2 GB              —
│ GLiNER2.5-multi-Decide  287 M       1.1 GB        ≈ 4.6 GB              ≈ 1.2 GB
…
│ dataset         Laya full · 3 epochs  GLiNER LoRA · 6  GLiNER full · 4  DGX Spark (Laya)
│ 456 tickets     4 min                 7 min            7 min            not measured yet
```

อะไรเป็นตัวตัดสิน:

- **นี่คือ encoder ไม่ใช่ LLM** โมเดล 322M พารามิเตอร์ต้องใช้ราว 16 ไบต์ต่อ weight สำหรับการ fine-tune ทั้งโมเดล (weights แบบ fp32, gradients และ Adam moment สองตัว): ราว 5 GB ซึ่งพอดีกับ MacBook 16 GB ส่วน LLM ขนาด 8B ต้องใช้ราว 130 GB สำหรับงานเดียวกัน ซึ่งเป็นงานที่ DGX Spark 128 GB มีไว้ทำ (week 25)
- **อ่านว่า trainer ทำอะไรจริง ๆ** trainer ของ Laya เลือก `cuda → mps → cpu` จึงได้ใช้ GPU ของ Apple ส่วน `ExtractorTrainer` ของ GLiNER2 ตรวจหา CUDA ถ้าไม่มีก็ใช้ **CPU**: มันไม่มีเส้นทาง MPS mixed precision (bf16) ใช้ได้เฉพาะ CUDA ทั้งสองตัว บน Mac Laya เทรนบน GPU แบบ fp32 และ GLiNER เทรนบน CPU แบบ fp32 บน Spark ทั้งคู่ได้ GPU และ bf16
- **วัดก่อน แล้วค่อยคาดการณ์** แล็บจับเวลาการเทรนจริงหนึ่งรอบบน 48 ticket เมื่อมีผลการเทรนเต็มจาก §4–§5 แล้ว แล็บจะคาดการณ์จากผลนั้นแทน เพราะรอบสั้นรวมช่วง warm-up ไว้ด้วย สำหรับ 456 ticket Mac ใช้เวลาไม่กี่นาที ที่ 50,000 ticket ใช้เวลาเกือบทั้งวันต่อรอบ และนั่นคือตอนที่ Spark คุ้มค่า
- **คอลัมน์ Spark เขียนว่า `not measured yet`** สเปกของ NVIDIA บอกหน่วยความจำ (128 GB unified LPDDR5x), แบนด์วิดท์ (273 GB/s) และ GPU แบบ CUDA ไม่มีสเปกชีตไหนบอกว่าโมเดล *เหล่านี้* เทรนเร็วแค่ไหน และคอร์สนี้ไม่พิมพ์ตัวเลขที่เดาเอาเองว่าเป็นผลการวัด Lab 17-8 จะเติมคอลัมน์นี้เมื่อคุณรันบน Spark ของคุณ

| สถานการณ์ | เทรนบน |
|---|---|
| fine-tune ครั้งแรก ต่ำกว่า ~5k ticket โมเดลเดียว | **Mac**: ไม่กี่นาที และเป็นเครื่องเดียวกับที่ใช้ให้บริการ |
| GLiNER2 มากกว่า ~5k ticket หรือรันหลายรอบ | **Spark**: trainer ของมันได้ใช้ CUDA ที่นั่น |
| sweep, หลาย seed, เทรนใหม่ทุกคืน | **Spark**: รันได้โดยไม่ต้องเฝ้า, bf16 และแล็ปท็อปของคุณว่าง |
| ข้อมูลออกนอกอาคารไม่ได้ | **ตัวไหนก็ได้**: ทั้งคู่เป็น on-prem ดังนั้นนี่ไม่ใช่เหตุผลให้เลือก cloud เลย |

✓ Checkpoint: คุณบอกได้ว่า trainer แต่ละตัวใช้อุปกรณ์อะไรบนเครื่องของคุณ และทำไมแถวของ GLiNER จึงช้ากว่าแถวของ Laya บน Mac

## 3 · ล็อกเกณฑ์ แล้ววัด baseline

```bash
.venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab03_baselines_and_criteria.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
» locked: sha256 e0b13dfbde8db0f8 · 2026-10-17 23:17:41  (.runs/criteria.lock)
…
│ engine       queue  95% CI    macro-F1  EN     TH     safety recall  false alarms  source
│ majority     15/66  14%–33%    0.07      9/42   6/24   0/14           0             RULE
│ keywords     52/66  68%–88%    0.77      31/42  21/24  8/14           2             RULE
│ laya_router  40/66  48%–73%    0.60      30/42  10/24  9/14           2             LOCAL
│ laya_base    36/66  42%–67%    0.52      26/42  10/24  8/14           1             LOCAL
│ gliner_base  40/66  48%–73%    0.57      23/42  17/24  14/14          31            LOCAL
│ jev          66/66  100%–100%  1.00      42/42  24/24  13/14          7             RECORDED
```

`criteria.json` ถูกเขียนก่อนที่จะเทรนโมเดลใด ๆ Lab 17-3 เก็บค่า hash ของไฟล์ไว้ใน `.runs/criteria.lock` Lab 17-7 จะไม่ยอมให้คะแนนถ้าไฟล์ถูกแก้ภายหลัง หรือถ้า lock ใหม่กว่า checkpoint เกณฑ์เจ็ดข้อบน ticket held-out 66 ใบ:

| | ต้องเป็นจริง | ทำไม |
|---|---|---|
| C1 | ความแม่นยำของคิวชนะ baseline แบบ zero-shot ≥ 10 จุด | คุ้มกับความพยายาม |
| C2 | McNemar p < 0.05 | ผลที่ได้ไม่ใช่โชค |
| C3 | Safety recall 100 % | ห้ามพลาดเหตุไฟไหม้ |
| C4 | ภาษาไทยห่างจากอังกฤษไม่เกิน 10 จุด | ไม่ทิ้งภาษาใดไว้ข้างหลัง |
| C5 | ทำพัง ≤ 2 ticket ที่ baseline ตอบถูก | ไม่มีการถดถอยแบบเงียบ ๆ |
| C6 | forgetting probe ลดลง ≤ 2 จาก 12 | ทักษะทั่วไปยังอยู่ |
| C7 | latency มัธยฐาน ≤ 1.5 × baseline | เหตุผลที่ใช้โมเดลในเครื่องยังอยู่ |

สิ่งที่ baseline สอน:

- **keyword สิบบรรทัดชนะโมเดลในเครื่องแบบ zero-shot ทุกตัว**: 52/66 เทียบกับ 40/66 ของ Router ของ Laya และ GLiNER2.5-multi-Decide กฎ keyword คือพื้นขั้นต่ำ และโมเดลที่ชนะมันไม่ได้ก็ไม่คุ้มที่จะนำไปใช้
- **Jev คือเพดาน**: 66/66 สำหรับคิว แม้ว่าจะยังพลาด ticket อันตราย 1 จาก 14 ใบ (แก๊สรั่ว ซึ่งนิยามความปลอดภัยที่เขียนไว้ไม่ได้กล่าวถึง) การ fine-tune โมเดลในเครื่องมีเหตุผลเมื่อข้อมูลต้องอยู่ในเครื่องหรือทุกมิลลิวินาทีมีค่า ไม่ใช่เพื่อชนะ Jev
- **baseline ของ Laya สองตัว โดยตั้งใจ** `laya_router` (checkpoint ภาษาอังกฤษสำหรับภาษาอังกฤษ หลายภาษาสำหรับไทย) คือสิ่งที่คุณจะ ship วันนี้ เกณฑ์จึงเทียบกับมัน ส่วน `laya_base` (หลายภาษาอย่างเดียว) คือ weights ตัวเดียวกับที่คุณ fine-tune forgetting probe จึงเทียบกับมัน ตาราง before/after ของ Laya เอง (§4) ใช้ `laya_base` ซึ่งทำให้ผลดูดีเกินจริงถ้าคุณอ้างว่ามันคือการพัฒนา "ที่แท้จริง"
- **safety recall 14/14 ของ GLiNER ไม่ใช่ชัยชนะ** มันส่งสัญญาณเตือนผิด 31 ครั้งจาก ticket ที่ปลอดภัย 52 ใบ: มันตอบ "yes" แทบทุกอย่าง อ่าน recall คู่กับ false alarm เสมอ
- **ช่วง 95 % กว้างมาก** 40/66 หมายถึง "อยู่ระหว่าง 48 % ถึง 73 %" ด้วย ticket 66 ใบ ความต่างไม่กี่จุดพิสูจน์อะไรไม่ได้ นั่นคือเหตุผลที่มี C2

✓ Checkpoint: มีไฟล์ `.runs/criteria.lock` แล้ว และคุณบอกได้ว่าเกณฑ์แต่ละข้อเทียบกับ baseline ตัวไหนและทำไม

## 4 · Fine-tune Laya

```bash
week24_typed_decisions_jev/.venv-local/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab04_finetune_laya.py --train   # ≈ 4 min, ⌨ terminal
.venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab04_finetune_laya.py                         # ▶ reads the result
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
train items 1204, calibration items 164, eval items 360 (…laya_val.jsonl), skipped {}, device mps
optimizer budget: 1204 train item(s), effective batch up to 16 (micro_batch 8 x grad_accum 2), 228 optimizer updates over 3 epoch(s)
» 1204 train items + 164 calibration items · mps · 3.9 min · source: LOCAL (just trained)
   epoch 1  loss 0.4041  ██████████████████████████████
   epoch 2  loss 0.1941  ██████████████
   epoch 3  loss 0.0478  ████
   ⚠ laya.train: choice calibration: the fit came back unchanged at 1.0 (accuracy 1.00, mean confidence 1.000 …
…
│ accuracy (all 3 questions)  0.614              0.889       +0.275
│ loss                        1.364              2.080       +0.717
│ mean confidence             0.818              0.997       +0.179
│ choice (queue)   0.48 → 0.80  0.84 → 0.99      0.36 → 0.19
…
│ train (seen)             114      114/114        100%
│ val (unseen situations)  120      96/120         80%
│ held-out (hand-written)  66       55/66          83%
```

ทั้งหมดคือการเรียกฟังก์ชันเดียว: `laya.train.finetune(train.jsonl, "multilingual", out_dir, TrainConfig(…, eval_data=val.jsonl))` มันให้คะแนน checkpoint ตั้งต้นบน validation, เทรน, fit temperature บน calibration slice, ให้คะแนนอีกครั้ง แล้วบันทึก checkpoint ที่ `laya.load(out_dir)` อ่านได้ สูตรและเหตุผล:

- **checkpoint เดียวสำหรับสองภาษา** Router ต้อง fine-tune สองครั้ง `multilingual` (mmBERT, 322M) เรียนภาษาอังกฤษและไทยไปพร้อมกัน
- **batch 8 × 2 = 16 ไม่ใช่ค่าเริ่มต้น 64 ของ Laya** ด้วย 1,204 item ค่า 64 ให้การอัปเดตเพียงราว 19 ครั้งต่อ epoch Laya เตือนเรื่อง "collapse to the class prior" เมื่อข้อมูลน้อย และการอัปเดตที่มากขึ้นช่วยหลีกเลี่ยงได้
- **3 epoch, `shuffle_options=("choice",)`** รอบน้อย เพราะมีสถานการณ์ให้จำเพียง 72 แบบ การสลับตัวเลือกทำให้ลำดับตัวเลือกของคิวเปลี่ยนทุก epoch โมเดลจึงเรียนแบบ "ตัวเลือกแรกมักถูก" ไม่ได้
- **`eval_data=` ทำให้ตารางซื่อตรง** ถ้าไม่มี Laya จะให้คะแนน calibration slice ที่มาจากข้อมูลเทรนเอง และบอกไว้ในผลลัพธ์ ("not held-out") การรันทดสอบครั้งหนึ่งที่ทำแบบนี้บนสำเนาข้อมูลของมันเองพิมพ์ความแม่นยำ 0.98 และตัวเลขนั้นไม่มีความหมายอะไรเลย

สิ่งที่การรันนี้สอน:

- **มันได้ผล** ความแม่นยำของคิวบน held-out ขึ้นจาก 40/66 (Router) เป็น 55/66 ในเวลาไม่ถึงสี่นาทีบนแล็ปท็อป ความเร่งด่วนดีขึ้นมากที่สุด (0.47 → 0.89 บน validation) เพราะ "วันนี้" กับ "ทันที" เป็นกฎของบ้าน และกฎของบ้านคือสิ่งที่ตัวอย่างสอนได้
- **ช่องว่างของการ generalise คือบทสรุป** 100 % บนสถานการณ์ที่เทรน, 80 % บนสถานการณ์ที่สร้างขึ้นแต่ไม่เคยเห็น, 83 % บน ticket held-out ที่เขียนด้วยมือ ความแม่นยำบนข้อมูลเทรนบอกได้แค่ว่าโมเดลทำการบ้านได้
- **ความแม่นยำขึ้น loss บน validation ก็ขึ้น** loss เพิ่มจาก 1.36 เป็น 2.08 ขณะที่ความแม่นยำเพิ่มขึ้น โมเดลผิดน้อยลง แต่ผิดแต่ละครั้งด้วยความมั่นใจเกือบเต็มที่: ความมั่นใจเฉลี่ยคือ 0.997
- **Laya บอกคุณแล้ว** การ fit temperature "came back unchanged at 1.0" เพราะ calibration slice มาจากสถานการณ์ในข้อมูลเทรนที่โมเดลตอบถูก 100 % แล้ว จึงไม่มีอะไรให้แก้ cut สำหรับการงดตอบ (abstention) จึงไปอยู่ที่ราว 1.000 ด้วยเหตุผลเดียวกัน วิธีแก้อยู่ใน §6: fit cut บน validation
- **loss ติดลบได้** objective `rlcd` ที่เป็นค่าเริ่มต้นไม่ใช่ cross-entropy (มีพจน์ reward อยู่ด้วย) loss ของบาง step ที่ติดลบจึงเป็นเรื่องปกติ อย่าอ่านมันเป็นความแม่นยำ

✓ Checkpoint: คุณอธิบายได้ว่าทำไม loss บน validation จึงเพิ่มขึ้นขณะที่ความแม่นยำบน validation เพิ่มขึ้น และทำไม temperature ที่ fit ได้จึงยังอยู่ที่ 1.0

## 5 · Fine-tune GLiNER2.5-multi-Decide: LoRA adapter 5 MB เทียบกับ fine-tune ทั้งโมเดล

```bash
week24_typed_decisions_jev/.venv-local/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab05_finetune_gliner.py --train lora   # ≈ 7 min, CPU
week24_typed_decisions_jev/.venv-local/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab05_finetune_gliner.py --train full   # ≈ 7 min, CPU
.venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab05_finetune_gliner.py                              # ▶ reads both
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
│ run   trainable          device  time     tickets  validation loss per epoch                      source
│ lora  1.36 M (0.5%)      cpu     7.2 min  6.3/s    0.306 → 0.160 → 0.240 → 0.154 → 0.246 → 0.480  LOCAL
│ full  287.36 M (100.0%)  cpu     6.7 min  4.5/s    0.246 → 0.167 → 0.189 → 0.169                  LOCAL
» lora: lowest validation loss at epoch 4 of 6 → that checkpoint is kept as 'best/'  (later epochs were already overfitting)
   early stopping ended it after epoch 6 of 8: two epochs in a row without a new best
» full: lowest validation loss at epoch 2 of 4 → that checkpoint is kept as 'best/'  (later epochs were already overfitting)
…
│ lora  5.5 MB           needs the 1.1 GB base next to it
│ full  1,165.5 MB       stands alone
…
│ gliner_base  65/120   45%–62%  0.53      38/80  27/40  24/24          52            …
│ gliner_lora  108/120  84%–95%  0.90      71/80  37/40  24/24          0             …
│ gliner_full  101/120  78%–90%  0.84      64/80  37/40  24/24          0             …
…
   adapter ON  → safety       1.00
   adapter OFF → safety       0.35   (the untouched base model)
```

ทั้งสองการรันใช้ `ExtractorTrainer` ของ GLiNER2 เองบน `data/gliner_train.jsonl` และให้คะแนน `data/gliner_val.jsonl` หลังทุก epoch:

- **LoRA แช่แข็ง weights 287M ไว้** แล้วเทรนเมทริกซ์ rank-8 เล็ก ๆ ข้างแต่ละชั้นของ encoder บวกกับ task head: 1.36M weights หรือ 0.47 % คู่มือของ GLiNER2 แนะนำ rank 4 เมื่อมีตัวอย่างต่ำกว่า ~1k และ rank 8 สำหรับ 1k–10k ตัวอย่าง 456 ticket × 3 งานคือราว 1.4k
- **fine-tune ทั้งโมเดลขยับทุก weight** จึงได้ epoch น้อยกว่า (4) และ learning rate ของ encoder ต่ำ (1e-5)
- **`save_best` + `early_stopping`** ตรวจ loss บน validation ทุก epoch ค่าต่ำสุดถูกเก็บไว้เป็น `best/` และการเทรนหยุดหลังสอง epoch ที่ไม่มีค่าดีที่สุดใหม่ ลองอ่านกราฟของ LoRA: มันดีขึ้นจนถึง epoch 4 แล้วขึ้นไปถึง 0.48 เมื่อเริ่มจำข้อมูล ถ้าไม่มี `save_best` คุณจะ ship epoch 6 (0.48) ถ้าไม่มี early stopping คุณต้องจ่ายค่า 8 epoch การรันครั้งแรกที่บันทึกไว้ไม่มี early stopping และใช้เวลา 10.1 นาที นานกว่าขีดจำกัด 600 วินาทีของ ⌨ terminal นั่นคือเหตุผลที่สูตรตอนนี้หยุดเร็ว
- **บน Mac นี่คืองานของ CPU** trainer ของ GLiNER2 ไม่มีเส้นทาง MPS (§2) การรัน LoRA ประมวลผลราว 6 ticket ต่อวินาทีที่นี่ และ Spark ให้โค้ดเดียวกันได้ใช้ GPU แบบ CUDA (§8)

สิ่งที่การรันนี้สอน:

- **adapter ขนาด 5 MB ชนะ** LoRA ได้ 108/120 บน validation, fine-tune ทั้งโมเดลได้ 101/120 และโมเดลตั้งต้นได้ 65/120 ด้วยตัวอย่าง 456 ใบ การเปลี่ยน weights น้อยกว่า generalise ได้ดีกว่า
- **ขนาดเปลี่ยนรูปแบบการ deploy** base 1.1 GB หนึ่งตัวบวก adapter 5.5 MB ต่อโรงแรม รวมราว 1.1 GB สำหรับห้าโรงแรม ส่วน fine-tune ทั้งโมเดลห้าตัวคือราว 5.5 GB
- **โมเดลตั้งต้นยังอยู่** เมื่อใช้ `pm.disable_adapter()` object เดียวกันจะตอบเหมือน base ที่ไม่ถูกแตะต้อง (0.35 บน ticket ควันภาษาไทย เทียบกับ 1.00 เมื่อเปิด adapter) checkpoint แบบ boundary อย่าง `GLiNER2.5-multi-Decide` โหลด adapter ผ่าน PEFT (`PeftModel.from_pretrained`) ส่วน `.load_adapter()` แบบเก่าใช้ได้เฉพาะ checkpoint แบบ span

✓ Checkpoint: คุณอ่านกราฟ loss บน validation ของ LoRA ได้ บอกได้ว่า epoch ไหนถูกเก็บไว้และทำไม และอธิบายได้ว่า `disable_adapter()` ให้อะไรคุณ

## 6 · การสอบ: ได้ผลหรือเปล่า?

```bash
.venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab06_evaluate.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
│ engine       queue  95% CI     macro-F1  EN     TH     safety recall  false alarms  source
│ keywords     52/66  68%–88%    0.77      31/42  21/24  8/14           2             RULE
│ laya_router  40/66  48%–73%    0.60      30/42  10/24  9/14           2             …
│ laya_ft      55/66  74%–92%    0.82      34/42  21/24  10/14          2             …
│ gliner_base  40/66  48%–73%    0.57      23/42  17/24  14/14          31            …
│ gliner_lora  60/66  83%–97%    0.90      37/42  23/24  13/14          3             …
│ gliner_full  57/66  77%–94%    0.86      35/42  22/24  12/14          2             …
│ jev          66/66  100%–100%  1.00      42/42  24/24  13/14          7             RECORDED
…
│ laya_router → laya_ft      40/66 → 55/66  +23 pts  19     4      0.0026
│ gliner_base → gliner_lora  40/66 → 60/66  +30 pts  20     0      0.0000
│ gliner_base → gliner_full  40/66 → 57/66  +26 pts  18     1      0.0001
…
   gliner_lora  missed: t_en03 “Engineering: the basement car park…”
…
│ laya_ft      0.17                  1.000                     94%                15%
│ gliner_lora  0.09                  0.947                     95%                8%
…
│ gliner_base  10/12  · gliner_lora  8/12  · gliner_full  10/12      (forgetting probe)
│ laya_router → laya_ft      46 ms → 31 ms  × 0.67
```

หกคำถาม เรียงตามลำดับที่ผู้ตรวจควรถาม:

1. **ดีขึ้นไหม?** fine-tune ทุกตัวชนะ baseline แบบ zero-shot ของตัวเอง 23–30 จุดบน ticket 66 ใบที่ไม่เคยเห็น GLiNER + LoRA (60/66) ตอนนี้ชนะกฎ keyword (52/66) และช่วงของมัน (83–97 %) ไม่ทับกับของโมเดลตั้งต้น (48–73 %) อีกแล้ว
2. **เป็นโชคหรือเปล่า?** McNemar ดูเฉพาะ ticket ที่ก่อนและหลังตอบไม่ตรงกัน LoRA แก้ได้ 20 และทำพัง 0 จึงได้ p ≈ 0.000002 Laya แก้ได้ 19 และทำพัง 4 จึงได้ p = 0.003
3. **มันทำอะไรพัง?** Laya ทำพัง ticket ของ housekeeping/front desk สามใบที่เขียนเป็นคำขอยาว ๆ อย่างสุภาพ ("we arrived late last night…", "can the hotel store our luggage…") บวกกับตัวหลอกเรื่องเทียน ซึ่งตอนนี้มันเรียกว่า `safety` นั่นคือสไตล์ที่ generator ไม่เคยใช้ รายการแบบจับคู่บอกคุณได้ว่าต้องเขียนข้อมูลอะไรต่อ ตัวเลขความแม่นยำอย่างเดียวบอกไม่ได้
4. **ปลอดภัยไหม?** ไม่มี fine-tune ตัวไหนจับ ticket อันตรายได้ครบ ทุกตัวพลาด **t_en03** "the basement car park drain is backing up and water is now reaching the main switchboard room door" นั่นคือน้ำใกล้ไฟฟ้าในถ้อยคำที่ไม่มีสถานการณ์ในข้อมูลเทรนใช้เลย Laya ยังพลาดเรื่องแก๊สรั่ว เครื่องทำน้ำแข็งรั่วลงปลั๊กที่พื้น และมอเตอร์ลิฟต์ร้อนเกิน
5. **เชื่อความมั่นใจของมันได้ไหม?** fit cut บน **validation** (เป้าหมาย: ผิด ≤ 5 % ในบรรดา ticket ที่ตอบอัตโนมัติ) แล้วตรวจบน held-out สำหรับ LoRA cut นี้ตอบ ticket held-out 95 % ด้วยความผิด 8 %: ใกล้ แต่ไม่ใช่ 5 % ตามที่สัญญา สำหรับ Laya cut คือ 1.000 เพราะหลัง fine-tune แทบทุกอย่างได้ 1.000 มันตอบ 94 % ด้วยความผิด 15 % ความมั่นใจเกินจริงจาก §4 กลายเป็นต้นทุนจริงตรงนี้ และวิธีแก้คือให้คนอยู่ในวงจร หรือ fit calibration บนข้อมูลที่คล้าย held-out
6. **อะไรแย่ลงอีก?** บน forgetting probe (คำถาม sentiment ของรีวิว 12 ข้อที่โมเดลไม่เคยเทรน) Laya และ GLiNER ที่ fine-tune ทั้งโมเดลยังได้คะแนนเท่าเดิม แต่ GLiNER **ที่เปิด adapter** ลดจาก 10/12 เป็น 8/12 adapter เปลี่ยนทุกงานที่โมเดลรันขณะที่มันถูกโหลดอยู่ ให้บริการงานอื่นที่ไม่เกี่ยวข้องโดยปิด adapter ด้วย 12 ข้อ ±1 คือสัญญาณรบกวน และการลดลง 3 จึงจะเป็นสัญญาณ

ความเร็วไม่ได้แย่ลง: Laya เร็วขึ้น (46 → 31 ms) เพราะสำหรับภาษาอังกฤษ Router ใช้ checkpoint ภาษาอังกฤษขนาด 421M ที่ใหญ่กว่า และ LoRA ที่ไม่ได้ merge เพิ่มเวลาราว 9 % (74 → 80 ms)

✓ Checkpoint: คุณบอกได้ว่า ticket ไหนที่ fine-tune ทุกตัวพลาด และอธิบายได้ว่าทำไม cut ที่ fit บน validation จึงให้ความผิด 15 % บน held-out สำหรับ Laya

## 7 · Ship หรือ hold: ให้เกณฑ์ที่ล็อกไว้เป็นผู้ตัดสิน

```bash
.venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab07_ship_or_hold.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
✓ criteria.json matches the lock (sha256 e0b13dfbde8db0f8, locked 2026-10-17 23:17:41)
✓ laya_ft: weights written 2026-10-17 23:24 — after the criteria were locked
…
   gliner_lora  →  🔴 HOLD
│ C1  PASS    queue_gain_points = 30.3 (needs ≥ 10)
│ C2  PASS    queue_mcnemar_p = 1.90735e-06 (needs ≤ 0.05)
│ C3  FAIL    safety_recall = 0.928571 (needs ≥ 1.0)
│ C4  PASS    th_minus_en_points = 7.7 (needs ≥ -10)
│ C5  PASS    queue_broke = 0 (needs ≤ 2)
│ C6  PASS    probe_drop = 2 (needs ≤ 2)
│ C7  PASS    latency_ratio = 1.09 (needs ≤ 1.5)
…
   ✓ .runs/model_card_gliner_lora.json  (HOLD)
```

**ผู้สมัครทั้งสามตัวได้ HOLD** และนั่นคือผลลัพธ์ที่มีประโยชน์ที่สุดของบทนี้:

| | Laya fine-tune | GLiNER + LoRA | GLiNER แบบเต็ม |
|---|---|---|---|
| ไม่ผ่าน | C3 (ticket อันตราย 10/14), C5 (ทำพัง 4) | C3 (13/14) | C3 (12/14) |
| ผ่าน | C1, C2, C4, C6, C7 | C1, C2, C4, C5, C6, C7 | C1, C2, C4, C5, C6, C7 |

- **ด่านตรวจเป็นผู้ตัดสิน ไม่ใช่บรรยากาศในห้อง** "+30 จุด!" ฟังดูน่า ship แต่ C3 บอกว่าน้ำท่วมใกล้ตู้สวิตช์บอร์ดที่พลาดไปหนึ่งครั้งก็มากเกินไปแล้ว และ C3 ถูกเขียนก่อนที่ใครจะเห็นตัวเลข Lab 17-7 ยังตรวจว่า `criteria.json` ยังตรงกับ lock และ weights ของทุก checkpoint ถูกเขียน *หลัง* lock การเลื่อนเสาประตูหลังเห็นผลคือวิธีที่พบบ่อยที่สุดที่ทำให้การ fine-tune "สำเร็จ"
- **HOLD บอกขั้นต่อไป** เขียนสถานการณ์เรื่อง *น้ำไหลเข้าห้องไฟฟ้าและปลั๊กที่พื้น* และ *คำขอ housekeeping แบบยาวและสุภาพ* แล้วเทรนใหม่และให้คะแนนอีกครั้งภายใต้เกณฑ์เดิม เขียนสำหรับ **หมวดหมู่** ไม่ใช่สำหรับประโยค การคัดลอก t_en03 ไปไว้ในข้อมูลเทรนก็แค่ทำให้ข้อสอบรั่ว
- **ข้อสอบชุดนี้ใช้หมดแล้ว** คุณศึกษาข้อที่มันผิดเพื่อตัดสินใจว่าจะเขียนอะไรต่อ ดังนั้น SHIP ครั้งต่อไปต้องใช้ชุด held-out **ใหม่** เก็บ ticket ที่เขียนด้วยมือไว้สำรองสักไม่กี่สิบใบสำหรับเรื่องนี้
- **model card เดินทางไปกับ checkpoint** `.runs/model_card_*.json` บันทึก base, hash ของข้อมูล, hash ของเกณฑ์, ผลทุกข้อ, ความผิดพลาดที่รู้แล้ว และการใช้งานที่ตั้งใจ ("NOT validated on real guests")
- **การ ship เมื่อถึงเวลา ทำแบบค่อยเป็นค่อยไป** รันโมเดลในโหมด shadow ข้างตัวปัจจุบัน ใช้ cut ที่ fit บน validation เป็นด่าน ส่งคำตอบที่ความมั่นใจต่ำไปให้คน (ห้ามส่งไป cloud แบบเงียบ ๆ) และเก็บ checkpoint ก่อนหน้าไว้สำหรับ rollback

✓ Checkpoint: คุณบอกได้ว่าทำไม GLiNER + LoRA จึงได้ HOLD แม้จะดีขึ้น +30 จุด และทำไม SHIP ครั้งต่อไปต้องใช้ ticket held-out ชุดใหม่

## 8 · fine-tune แบบเดียวกันบน DGX Spark

```bash
.venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab08_run_on_spark.py                         # ▶ plan + status
SPARK_HOST=you@spark-abcd .venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab08_run_on_spark.py --run   # ⌨ terminal
```

**Expected output** (ผลลัพธ์ที่ควรเห็น — ยังไม่ได้ตั้งค่า Spark เหมือนตอนที่บันทึกบทนี้)

```
» SPARK_HOST is not set (env or week25_dgx_spark/.env.local). Set it to your ssh target, e.g.
…
│ 1  copy code + data          rsync -az --exclude .runs … $SPARK_HOST:~/agentic-week24/ (keeps paths)
│ 2  train inside NGC PyTorch  ssh $SPARK_HOST docker run --rm --gpus all --ipc=host -v ~/agentic-week24:/w …
│ 3  bring the run files back  scp $SPARK_HOST:…/lab0[45]_run.json → recorded/lab08_spark.json
…
│ Laya full fine-tune, 456 tickets × 3 epochs       3.9 min (mps)       NOT RUN
│ GLiNER LoRA, 456 tickets × 6 epochs (early sto…   7.2 min (cpu)       NOT RUN
```

ปุ่ม ▶ ไม่เคยเริ่มงานบนเครื่องปลายทาง มันหา Spark ของคุณ (`SPARK_HOST` หรือ `week25_dgx_spark/.env.local` จากคอร์ส week 25) ทดสอบด้วย `ssh -o BatchMode=yes` แล้วพิมพ์แผนออกมา `--run` จะทำตามแผน:

1. `rsync` คัดลอก `week24_typed_decisions_jev/common`, `localkit.py` ของบทที่ 16 และบทนี้ (ยกเว้น `.runs/`) ไปที่ Spark
2. `docker run --gpus all` หนึ่งครั้งใน **`nvcr.io/nvidia/pytorch:25.11-py3`** ซึ่งเป็น image ที่ playbook ของ NVIDIA ใน week 25 ใช้ ติดตั้ง `laya` และ `gliner2[train]` *โดยยึด PyTorch ของ container เอง* แล้วรัน `lab04 --train` และ `lab05 --train lora` Spark เป็น aarch64 + CUDA และ PyTorch ของ container ตรงกับมัน โค้ดของแล็บจึงไม่ต้องเปลี่ยน
3. ไฟล์ผลการรันกลับมาที่ `recorded/lab08_spark.json` จากนั้น Lab 17-2 และ 17-8 จะแสดงเวลาของ Spark ที่ **วัดจริง** ข้างเวลาของ Mac

บน Spark trainer ของ Laya เลือก CUDA และเปิด bf16 autocast และ gradient checkpointing trainer ของ GLiNER2 ได้ GPU ที่ไม่เคยได้บน Mac และ Lab 17-5 สลับมันไปใช้ bf16 ที่นั่น จนกว่าคุณจะรัน คอลัมน์ Spark จะเขียนว่า `NOT RUN` คอร์สนี้ไม่พิมพ์ตัวเลขการเร่งความเร็วที่ยังไม่ได้วัด

✓ Checkpoint: คุณบอกได้สามขั้นตอนที่ `--run` ทำ และอธิบายได้ว่าทำไมการเทรนจึงรันใน container ของ NVIDIA แทนที่จะใช้ PyTorch ที่ติดตั้งด้วย pip

## Labs — รันได้ที่นี่

**labs/lab01_build_dataset.py** — สร้าง 576 ticket จาก 72 สถานการณ์ เทียบการแบ่งแบบสุ่มกับการแบ่งตามสถานการณ์ ตรวจการรั่วของชุดทดสอบที่เขียนด้วยมือ และเขียนไฟล์เทรนของ Laya และ GLiNER

**labs/lab02_mac_or_spark.py** — trainer แต่ละตัวใช้อุปกรณ์อะไร หน่วยความจำที่นับจากไฟล์ weights การจับเวลาการเทรนหนึ่งรอบ และการคาดการณ์สำหรับ 456, 5k และ 50k ticket

**labs/lab03_baselines_and_criteria.py** — ล็อกเกณฑ์ความสำเร็จ แล้ววัด majority label, กฎ keyword, Laya และ GLiNER แบบ zero-shot และ Jev บน validation และ held-out

**labs/lab04_finetune_laya.py** — fine-tune checkpoint หลายภาษาของ Laya (`--train`, ⌨ terminal) หรืออ่านผลการรัน: loss, ตาราง before/after ของ Laya, ช่องว่างของการ generalise และ temperature

**labs/lab05_finetune_gliner.py** — GLiNER2.5-multi-Decide แบบ LoRA adapter และแบบ fine-tune ทั้งโมเดล (`--train lora|full`): กราฟ loss บน validation, early stopping, ขนาด และการปิด adapter

**labs/lab06_evaluate.py** — การสอบบน ticket held-out 66 ใบ: ช่วงความเชื่อมั่น สิ่งที่แก้ได้เทียบกับสิ่งที่พัง ticket อันตรายที่พลาด cut ความมั่นใจที่ fit บน validation การลืม และความเร็ว

**labs/lab07_ship_or_hold.py** — ตรวจ lock ให้คะแนน C1–C7 เขียน model card ต่อผู้สมัครแต่ละตัว และดูว่าการ ship และ rollback หน้าตาเป็นอย่างไร

**labs/lab08_run_on_spark.py** — หา DGX Spark ของคุณ พิมพ์แผน และเมื่อใช้ `--run` จะเทรนที่นั่นภายใน container PyTorch ของ NVIDIA แล้วนำเวลาที่วัดได้กลับมา

การเทรน (`--train`) รันบนเครื่องนี้ภายใน `week24_typed_decisions_jev/.venv-local` ส่วนอื่นทั้งหมดรันจาก checkpoint ของคุณ หรือถ้าไม่มี ก็จากผลที่บันทึกไว้เมื่อ 2026-10-17 ฝั่ง Jev รันจริง (LIVE) ด้วย key ของคุณ หรือแบบ DRY จากผลที่บันทึกไว้

## Try it yourself — ลองทำเอง

**Exercise 17 — การแบ่งข้อมูลที่ไม่รั่ว และด่านตรวจการ ship** เปิด `week24_typed_decisions_jev/17_finetune_system_one/exercises/ex17_split_and_ship.py` มี TODO สองข้อ:

1. เขียน `group_split(tickets, val_groups)` — ticket ทุกใบของสถานการณ์เดียวกันไปอยู่ฝั่งเดียวกัน คงลำดับไว้ ไม่แก้ไข input และ id ของสถานการณ์ที่ไม่รู้จักต้อง raise `ValueError` (พิมพ์ผิดตัวเดียวจะทำให้ validation หดลงแบบเงียบ ๆ)
2. เขียน `ship(metrics, criteria)` — เกณฑ์แต่ละข้อเป็น PASS, FAIL หรือ MISSING ค่าที่เท่ากับขอบเขตพอดีถือว่า PASS MISSING หยุดการปล่อยเหมือน FAIL และรายการเกณฑ์ว่างเปล่าคือ HOLD

การทดสอบแบบออฟไลน์สิบสองข้อรันก่อน เมื่อผ่านทั้งหมด ด่านตรวจของคุณจะให้คะแนน fine-tune ของวันที่ 2026-10-17 และเทียบผลกับ Lab 17-7

```bash
.venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/exercises/ex17_split_and_ship.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
✓ val holds every ticket of situation b, in order
✓ train holds everything else, in order
✓ no situation on both sides
…
✓ no criteria → HOLD

✓ all offline tests pass — grading the 2026-10-17 fine-tunes with YOUR gate
   laya_ft      your gate: HOLD  Lab 17-7: HOLD  ✓   not passing: C3, C5
   gliner_lora  your gate: HOLD  Lab 17-7: HOLD  ✓   not passing: C3
   gliner_full  your gate: HOLD  Lab 17-7: HOLD  ✓   not passing: C3
```

<details><summary>คำใบ้ — ทำไม MISSING ต้องหยุดการปล่อย</summary>

ถ้า metric หนึ่งไม่ได้ถูกวัด เช่น forgetting probe ไม่ได้รันเพราะโหลดโมเดลไม่สำเร็จ ด่านตรวจที่ข้ามมันไปจะผ่านโดยปริยาย "เราไม่ได้ตรวจ" จะกลายเป็น "มันผ่าน" แบบเงียบ ๆ ให้ถือว่า MISSING เหมือน FAIL ทุกประการ แล้ววิธีเดียวที่จะ SHIP ได้คือวัดทุกอย่างที่คุณสัญญาว่าจะวัด

</details>

<details><summary>Stretch — รอบที่สอง ทำอย่างซื่อตรง</summary>

เพิ่มสถานการณ์สามหรือสี่แบบใน `datagen.py` สำหรับ *หมวดหมู่* ของความผิดพลาดจาก §6: น้ำไหลเข้าห้องไฟฟ้าหรือปลั๊กที่พื้น และคำขอ housekeeping แบบยาวและสุภาพ เขียนด้วยคำของคุณเอง โดยไม่คัดลอก ticket held-out ใบใด รัน Lab 17-1 ใหม่ (ตรวจว่าการตรวจรั่วยังได้ 0/48) เทรน LoRA ใหม่ และรัน Lab 17-6 และ 17-7 ใหม่ จากนั้นเขียน ticket held-out ใหม่ 20 ใบก่อนที่จะเชื่อผล SHIP เพราะคุณได้ดูชุดเก่าไปแล้ว

</details>

✓ Checkpoint: การทดสอบทั้งสิบสองข้อผ่าน และด่านตรวจของคุณให้ผลสามข้อตรงกับ Lab 17-7

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `✗ data/train.json is missing` | รัน Lab 17-1 ก่อน — มันเขียน `data/` |
| `--train needs week24_typed_decisions_jev/.venv-local` | ตั้งค่า §0 ของบทที่ 16 (คำสั่ง `uv` สองชุดและ Lab 16-0) |
| การรัน `--train` ถูกหยุดหลัง 150 วินาที | นั่นคือขีดจำกัดของปุ่ม ▶ — รันการเทรนจาก ⌨ terminal (600 วินาที) หรือ terminal ของคุณเอง |
| `⚠ criteria.json CHANGED since it was locked` | ถ้ายังไม่ได้เทรนอะไร ลบ `.runs/criteria.lock` แล้วรัน Lab 17-3 ใหม่ ถ้าเทรนไปแล้ว ให้คืนไฟล์เดิม — นั่นคือจุดประสงค์ของ lock |
| Laya เตือน `fit came back unchanged at 1.0` | เป็นเรื่องปกติที่นี่: calibration slice มาจากสถานการณ์ในข้อมูลเทรนที่โมเดลตอบถูกแล้ว — ใช้ cut ที่ fit บน validation จาก Lab 17-6 |
| GLiNER เทรนช้าบน Mac | trainer ของมันไม่มีเส้นทาง MPS จึงรันบน CPU — ใช้ LoRA, ลด epoch หรือใช้ Spark (§8) |
| `AttributeError: 'BoundaryExtractor' object has no attribute 'has_adapter'` | checkpoint แบบ boundary ใช้ PEFT: `PeftModel.from_pretrained(base, path)` และ `with pm.disable_adapter():` |
| ตัวเลขของคุณต่างจากที่บันทึกไว้เล็กน้อย | kernel ของ MPS และลำดับข้อมูลไม่ได้ตรงกันทุกบิตข้ามเครื่อง — คาดว่าจะคลาดเล็กน้อย และให้อ่านช่วงความเชื่อมั่น |

## Next — ต่อไป

กลับไปที่ [ภาพรวมของคอร์ส](../README.md) — หรือจริงจังกับผล HOLD: ทำ Stretch exercise แล้วชี้ `pick_engine()` ของบทที่ 16 ([บทที่ 16](../16_local_system_one_models/TUTORIAL.th.md)) ไปที่ checkpoint ของคุณเมื่อชุด held-out ชุดใหม่บอกว่า SHIP คอร์ส week 25 ([DGX Spark](../../week25_dgx_spark/README.md)) ครอบคลุมการ fine-tune ฝั่ง *generative* (LLaMA Factory, Unsloth, NeMo) เมื่อโมเดล System One ยังไม่พอ
