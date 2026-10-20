# ▶ Jev Lab 16 — โมเดล System One แบบรันในเครื่อง: Laya, GLiNER และ GLiNER2.5-Decide เทียบกับ Jev

> ส่วนหนึ่งของ Week 24 · การตัดสินใจแบบมีชนิด (typed decision) ด้วย Jev — หนึ่งเดือนหลังจาก Jev เปิดตัว หมวด **System One** มีโมเดลมากกว่า 25 ตัวแล้ว และเก้าตัวเปิดให้ดาวน์โหลด weights ไปรันเองได้ (systemonemodels.org ตรวจสอบเมื่อ 2026-10-17) บทนี้รันสามตัว **บนแล็ปท็อปของคุณ** — **Laya** (Convai Innovations), **GLiNER2.5-Decide** (Fastino Labs) และ **GLiNER แบบดั้งเดิม** ซึ่งเป็นตัวดึง span ที่ตระกูล GLiNER เริ่มต้นมา — แล้วดูว่าแต่ละตัวเหมาะกับงานไหนเมื่อวางข้าง Jev และวัดผลทุกตัวอย่างยุติธรรมบน ticket ชุดเดียวกัน

**สิ่งที่คุณจะได้ลงมือทำ**
- อ่านภาพรวมของวงการในเดือนตุลาคม 2026 — และเข้าใจว่าทำไมกระดานคะแนนสามกระดานดูเหมือนขัดแย้งกัน
- รัน Laya แบบออฟไลน์: JSON เดียวกับ Jev ตอบได้ในหลักสิบมิลลิวินาที — แล้วจับจุดที่มันพลาดกับภาษาไทยและการตัดข้อความ
- ใช้ GLiNER แบบดั้งเดิม **ดึง** หมายเลขห้องพร้อมตำแหน่ง (offset) แล้วให้ Jev **ตัดสิน** ว่าห้องไหนมีปัญหา
- ป้อนคำถามชุดเดียวกับ Jev ให้ GLiNER2.5-Decide ผ่าน adapter — เทียบโมเดลภาษาอังกฤษกับโมเดลหลายภาษา
- จัดการแข่งขันแบบยุติธรรมบน 18 ticket: ความแม่นยำ ความเร็ว ต้นทุน และ calibration
- เขียน adapter และ policy `pick_engine()` สำหรับบริการที่ใช้โมเดลในเครื่องเป็นหลัก

**Time** ~45 นาที (+ ตั้งค่าครั้งเดียว ~1 นาที และดาวน์โหลด ~3 GB) · **Difficulty** ระดับกลาง · **Cost** ≈ $0.001 เมื่อรันจริง (เฉพาะฝั่ง Jev — โมเดลในเครื่องเสีย $0) · $0 ในโหมด dry

## 0 · ตั้งค่าโมเดลในเครื่องครั้งเดียว

`.venv` ของคอร์สยังเบาเหมือนเดิม (ตัวเรียก Jev เป็น Python ธรรมดา) แต่โมเดลในเครื่องต้องใช้ `torch` และ `transformers` และ `gliner` แบบดั้งเดิมล็อก `transformers < 5.17` ไว้ จึงต้องแยกไปอยู่ในสภาพแวดล้อมของตัวเองชื่อ `week24_typed_decisions_jev/.venv-local` — ทุกแล็บของบทที่ 16 ตรวจพบและรันตัวเองซ้ำในนั้นให้อัตโนมัติ รันคำสั่งสองชุดนี้ **ครั้งเดียว** จาก ⌨ terminal (การดาวน์โหลดใช้เวลานานกว่าขีดจำกัด 150 วินาทีของ runner):

```bash
uv venv --python 3.13 week24_typed_decisions_jev/.venv-local
uv pip install --python week24_typed_decisions_jev/.venv-local/bin/python laya "gliner2[local]" gliner tiktoken sentencepiece pillow
week24_typed_decisions_jev/.venv-local/bin/python week24_typed_decisions_jev/16_local_system_one_models/labs/lab00_download_models.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ LOCAL models · python 3.13.13 in week24_typed_decisions_jev/.venv-local · Laya on mps (picks the best device itself) · GLiNER on cpu (its default)
✓ convaiinnovations/laya                      3.8 s
✓ convaiinnovations/laya/multilingual        25.9 s
✓ fastino/GLiNER2.5-Decide                    9.4 s
✓ fastino/GLiNER2.5-multi-Decide              8.5 s
✓ urchade/gliner_multi-v2.1                   8.5 s
◆ done in 56 s — the Module 16 labs now run offline.
```

หลังจากนี้แล็บจะโหลด weights ด้วย `HF_HUB_OFFLINE=1` แล็บจึงไม่มีทางเริ่มดาวน์โหลดโดยไม่ตั้งใจ — และฝั่งโมเดลในเครื่อง **ไม่มีข้อความ ticket ใดออกจากเครื่องของคุณ** ถ้าข้ามการตั้งค่า ทุกแล็บก็ยังรันได้: คำตอบฝั่งโมเดลในเครื่องจะเล่นซ้ำจากการบันทึกเมื่อ 2026-10-17 (ติดป้าย `RECORDED`)

✓ Checkpoint: Lab 16-0 แสดงเครื่องหมาย ✓ ห้าบรรทัด (หรือคุณเลือกเรียนจากการบันทึก) และคุณอธิบายได้ว่าทำไมโมเดลในเครื่องต้องอยู่ใน venv ตัวที่สอง

## 1 · ภาพรวมของวงการ และกระดานคะแนนสามกระดานที่ "ขัดแย้งกัน"

```bash
.venv/bin/python week24_typed_decisions_jev/16_local_system_one_models/labs/lab01_landscape.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
│ board                                     Jev    GLiNER2.5-Decide  Laya
│ Jev Decision Index (indep.) — score       60.11  9.24              4.43
│ Jev Decision Index — calibration error ↓  0.074  0.0884            0.1402
│ JevBench capability score                 77.1   —                 32.5
│ Fastino Fast Decisions (vendor-run) %     57.6   60.2              46.6
│ Banking77, Laya fine-tuned (Ollman) %     80.0   —                 79.4
```

ทั้งสามแถวเป็นความจริง:

- **Jev Decision Index** (อิสระ กว้าง zero-shot): Jev นำห่าง ส่วน Laya รุ่นพื้นฐานใกล้เคียงการเดาสุ่ม
- **Fast Decisions ของ Fastino** (17 โดเมนภาษาอังกฤษที่ Fastino เลือกเอง): GLiNER2.5-Decide ชนะ Jev เล็กน้อย
- **Banking77 หลังจาก fine-tune Laya ด้วย 1,001 ตัวอย่าง**: Laya ตามหลัง Jev แค่ 0.6 คะแนน

แต่ละกระดานวัดงานคนละแบบ ให้มองตารางที่ผู้ขายรันเองแบบที่บทที่ 10 สอนไว้ และเชื่อเฉพาะการเปรียบเทียบที่คุณรันเองบนข้อมูลของคุณ โดยให้คะแนนทุกโมเดลด้วยวิธีเดียวกัน แล็บจบด้วยตาราง "ตัวไหนเหมาะกับงานอะไร" สำหรับ AltoTech

โมเดลในบทนี้ ตามข้อมูลจาก model card ของแต่ละตัว:

| | Laya | GLiNER2.5-Decide | GLiNER (ดั้งเดิม) | Jev |
|---|---|---|---|---|
| หน้าที่ | การตัดสินใจแบบมีชนิด (choice/score/noul) | หัวจำแนกแบบมีชนิด + span | **span** พร้อม offset (NER) | การตัดสินใจแบบมีชนิด |
| ขนาด | 421M EN · 322M หลายภาษา | 340M EN · 287M multi | small/medium/large | ไม่เปิดเผย |
| context | 512 EN · 1,024 หลายภาษา (8,192 เมื่อตั้ง `max_len`) | ไม่เปิดเผย | ข้อความสั้น | 64k (262k แบบ long context) |
| รันบน | GPU/CPU ของคุณ (มีพอร์ต MLX, Core ML, ONNX, laya.cpp) | CPU/GPU ของคุณ | CPU/GPU ของคุณ | API ของ TypeSafe |
| สัญญาอนุญาต | Apache-2.0 | Apache-2.0 | Apache-2.0 | ปิด, $0.042 ต่อล้าน input token |

✓ Checkpoint: คุณอธิบายได้ว่า GLiNER2.5-Decide "ชนะ Jev" และได้ 9 คะแนนเทียบกับ 60 ของ Jev ได้อย่างไรโดยไม่มีใครโกหก

## 2 · Laya บนแล็ปท็อปของคุณ

Laya รับ **JSON คำขอแบบเดียวกับ Jev** (`laya-serve` ของมันยังเปิด `POST /v1/systemone` ให้ client ของ TypeSafe ชี้มาใช้ได้ด้วย) `Router` จะเลือก checkpoint จากชนิดอักษรของข้อความก่อนประมวลผล

```bash
.venv/bin/python week24_typed_decisions_jev/16_local_system_one_models/labs/lab02_laya_local.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
» queue                  choice  → hvac   (confidence 0.48)
◆ laya/english · LOCAL (this machine) · $0 per call · 43 ms
…
│ question    Laya (local)                Jev (hosted)
│ queue       hvac (peak 0.77)            front_desk (peak 0.60)
│ wants_move  0.93                        0.99
│ latency     43 ms (no network, mps)     448 ms (internet)
…
│ Router (auto)   multilingual  plumbing    0.95       15 ms
│ forced english  english       front_desk  0.28       27 ms
» Router (auto)   ✕ wrong AND confident — the dangerous kind
…
│ laya english (512 tok)           hvac      0.86
│ laya multilingual, max_len=8192  plumbing  0.96
│ Jev (64k context)                safety    1.00
```

สิ่งที่รอบนี้สอน:

- **ความเร็วเป็นของจริง** — 43 ms บน GPU ของ Apple (15 ms บน checkpoint หลายภาษา) เทียบกับ ~450 ms ของ Jev ผ่านอินเทอร์เน็ต ส่วนต่างส่วนใหญ่มาจากเวลาเดินทางของเครือข่าย
- **`confidence` มีความหมายต่างกัน** — `confidence` ของ Laya เป็นค่าสถิติของมันเอง (และยังคืน `answer_confidence` = ค่าสูงสุดให้ด้วย) ให้เทียบ *ความน่าจะเป็น* ข้ามโมเดล อย่าเทียบช่อง `confidence`
- **ภาษาไทยพลาดแบบมั่นใจ** — กลิ่นไหม้ใกล้ตู้ไฟถูกส่งไป `plumbing` ด้วยค่า 0.95 การบังคับใช้ checkpoint ภาษาอังกฤษก็ผิดเช่นกัน แต่ *ไม่มั่นใจ* ซึ่งเกณฑ์กรองจะจับได้ ความมั่นใจอย่างเดียวปกป้องคุณไม่ได้ — ต้องทดสอบภาษาที่คุณให้บริการจริง
- **กับดักการตัดข้อความ** — checkpoint ภาษาอังกฤษเก็บ state ได้ราว 320 โทเคนแล้วทิ้งส่วนที่เหลือเงียบ ๆ ประโยคเรื่องควันจึงไปไม่ถึงโมเดล แม้แต่รอบหลายภาษาแบบ 8k ที่เห็นข้อความทั้งหมดก็ยังพลาด Laya รุ่นพื้นฐานคือ "ฐานที่เร็วสำหรับนำไปปรับให้เชี่ยวชาญ" (คำของ Convai) ไม่ใช่เครื่องมือ zero-shot — ต้อง fine-tune ก่อนพึ่งพา

✓ Checkpoint: คุณบอกได้สองแบบที่ Laya พลาดในแล็บนี้ และบอกได้ว่าแบบไหนที่เกณฑ์กรองความมั่นใจจะจับได้

## 3 · ดึงด้วย GLiNER แล้วค่อยตัดสิน

GLiNER แบบดั้งเดิมตอบคำถามคนละแบบ: "หมายเลขห้อง" "อุปกรณ์" หรือ "เวลา" อยู่ *ตรงไหน* ในข้อความ — มันคืน span พร้อมตำแหน่งเริ่ม/จบ ซึ่ง Jev ไม่เคยทำ รูปแบบคือ **ดึงก่อน แล้วค่อยตัดสิน**: GLiNER เสนอผู้สมัคร โมเดลตัดสินใจเลือกหนึ่งตัว และโค้ดเก็บตำแหน่งไว้

```bash
.venv/bin/python week24_typed_decisions_jev/16_local_system_one_models/labs/lab03_gliner_extract_then_decide.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP TH-1 · extract — GLiNER over: ได้กลิ่นไหม้แถวตู้ไฟชั้น 3 ใกล้ห้อง 1203 รีบมาดูด่วนค่ะ…
│ room number  3     25–26    0.59
│ room number  1203  36–40    0.64
▣ STEP TH-2 · decide — Jev chooses among the candidates
» problem_room           choice  → room_1203   (confidence 0.88)
      room_1203                  0.92  ██████████████████████░░ ◀
      none_of_these              0.06  █░░░░░░░░░░░░░░░░░░░░░░░
      room_3                     0.02  ░░░░░░░░░░░░░░░░░░░░░░░░
→ code: room 1203, mentioned at offsets 36–40
   ได้กลิ่นไหม้แถวตู้ไฟชั้น 3 ใกล้ห้อง ⟦1203⟧ รีบมาดูด่วนค่ะ
```

GLiNER ทำเครื่องหมาย `3` ว่าเป็นหมายเลขห้อง แต่ในภาษาไทยคือ *ชั้น* 3 โมเดล span หาผู้สมัครได้ แต่ไม่ได้ตัดสิน ใน ticket ภาษาอังกฤษ ห้อง 1203 ปรากฏสองครั้งและ 1408 หนึ่งครั้ง — Jev เลือก 1203 ด้วยค่า 1.00 และโค้ดไฮไลต์ **ทั้งสอง** ตำแหน่ง ราวกันตกสองข้อยังอยู่ในโค้ด: ไม่มีผู้สมัคร → ไม่เดา (ส่งให้คน) และมีตัวเลือก `none_of_these` เสมอ

✓ Checkpoint: คุณอธิบายได้ว่าทำไมตัวเลือกห้องต้องมาจาก GLiNER ไม่ใช่จาก Jev และโค้ดทำอะไรเมื่อ GLiNER หาอะไรไม่เจอ

## 4 · GLiNER2.5-Decide กับคำถามชุดเดิม

GLiNER2.5-Decide ตอบคำถามจำแนกแบบมีชนิดได้ในรอบเดียวบน CPU มันไม่มีชื่อ `choice` / `noul` / `score` ดังนั้น `localkit.to_gliner_schema()` จึงแปลให้: choice → labels พร้อมคำอธิบาย, noul → `["yes", "no"]` + prompt, score → labels ที่เรียงลำดับ

```bash
.venv/bin/python week24_typed_decisions_jev/16_local_system_one_models/labs/lab04_gliner_decide.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
│ question  GLiNER2.5-Decide      Jev
│ queue     safety (0.56)         safety (p 1.00)
│ safety    yes (0.95)            noul 0.98
│ urgency   routine (0.34)        score 2.00 / 2
│ time      160 ms · LOCAL · CPU  518 ms · LIVE
…
│ GLiNER2.5-Decide (English)  plumbing (0.29)  yes (0.61)  149 ms · LOCAL
│ GLiNER2.5-multi-Decide      safety (0.30)    yes (0.86)  56 ms · LOCAL
│ Jev 1.13                    safety (1.00)    noul 0.94   hosted
…
» every topic's score: {'hvac': 0.97, 'billing': 0.42, 'housekeeping': 0.23, 'room_change': 0.39, 'safety': 0.01} → sum 2.02
```

สี่บทเรียน:

1. **ภาษาไทยต้องใช้โมเดล multi** — โมเดลภาษาอังกฤษส่งกลิ่นไหม้ไป `plumbing` ส่วน `GLiNER2.5-multi-Decide` ตอบ `safety` ได้ถูก และเร็วกว่า
2. **ควันถูกจัดเป็น `routine`** — หัวลำดับความเร่งด่วนอ่อน model card ของ Fastino เองเตือนว่าหัวแบบลำดับ "ไม่รู้ว่า 6 ใกล้ 7 มากกว่า 0"
3. **ทำได้ในสิ่งที่ Jev ทำไม่ได้** — หัวเดียวคืนหัวข้อได้หลายป้าย (`hvac` + `billing`) และ **คืน span จากโมเดลเดียวกัน** (`extract_entities`)
4. **คะแนนของมันไม่ใช่การแจกแจงความน่าจะเป็น** — การเรียกหัวข้อหนึ่งครั้งรวมได้ 2.02 เป็นคะแนนอิสระรายป้าย อย่าใช้เกณฑ์เหมือนความน่าจะเป็นของ Jev และอย่าเฉลี่ยรวมกับของ Jev

✓ Checkpoint: คุณบอกได้ว่าจะใช้โมเดล GLiNER2.5-Decide ตัวไหนกับ ticket ภาษาไทย และทำไม 0.95 ของมันเทียบกับ 0.98 ของ Jev ไม่ได้

## 5 · แข่งขันแบบยุติธรรมบน 18 ticket

"ยุติธรรม" หมายถึง: ข้อความเดียวกัน คำถามสามข้อเดียวกัน (ผ่าน adapter ตัวเดียว) ป้ายคำตอบเดียวกัน วิธีให้คะแนนเดียวกัน ทุกเครื่องมือเป็น zero-shot และทำตามคำแนะนำเรื่องหลายภาษาของผู้ผลิตแต่ละรายสำหรับ ticket ภาษาไทย

```bash
.venv/bin/python week24_typed_decisions_jev/16_local_system_one_models/labs/lab05_bakeoff.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
│ question         Jev 1.13  Laya (Router)  GLiNER2.5-Decide (+multi for TH)
│ queue            18/18     12/18          16/18
│ safety           17/18     17/18          16/18
│ urgency          15/18     10/18          10/18
│ queue · EN only  12/12     9/12           11/12
│ queue · TH only  6/6       3/6            5/6
» urgency mean absolute error (0 = perfect, levels 0–2): jev 0.17 · laya 0.44 · decide 0.44
…
│ Jev 1.13          476 ms       internet round trip    $0.020              LIVE
│ Laya              62 ms        this machine, mps      $0 + your hardware  LOCAL
│ GLiNER2.5-Decide  148 ms       this machine, CPU      $0 + your hardware  LOCAL
…
» jev   Brier 0.026 · confidently wrong 0/18
» laya  Brier 0.063 · confidently wrong 1/18
```

อ่านแบบผู้ตรวจสอบ:

- **บอกทิศทาง ไม่ใช่คำตัดสิน** — 18 ticket เป็นตัวอย่างในห้องเรียน ลำดับที่ได้สอดคล้องในทิศทางเดียวกับ Jev Decision Index ที่เป็นอิสระ ถ้าจะตัดสินใจจริง ให้ใช้วิธีของบทที่ 09 กับ ticket *ของคุณ* หลายร้อยรายการ
- **ต้นทุนไม่ได้มีแค่ต่อการเรียก** — Jev ≈ $0.02 ต่อ 1,000 ticket โมเดลในเครื่องเสีย $0 ต่อการเรียก แต่คุณจ่ายค่าฮาร์ดแวร์ การอัปเดต และงานวิศวกรรม (แถว "สิ่งที่คุณต้องดูแลเอง" ในบทที่ 10)
- **วัด calibration เฉพาะที่มีให้วัด** — Jev และ Laya คืน P(yes) สำหรับ `safety` จึงใช้ Brier score ได้อย่างยุติธรรม GLiNER2.5-Decide คืนคะแนนเดียวของป้ายที่ชนะ จึงตั้งใจไม่นำมาเทียบ แทนที่จะฝืนใส่
- **ช่องว่างปิดได้** — Laya ที่ fine-tune ด้วย Banking77 1,001 ตัวอย่างได้ 79.4 % เทียบกับ 80.0 % ของ Jev ถ้าโมเดลในเครื่องใกล้เคียงบนข้อมูลของคุณ *และ* ข้อมูลต้องอยู่ในเครื่อง การ fine-tune คุ้มค่าแก่การทดลอง

✓ Checkpoint: คุณอธิบายได้ว่าทำไม GLiNER2.5-Decide ไม่มี Brier score และบอกต้นทุนสองอย่างของโมเดลในเครื่องที่ช่อง "$0" ซ่อนไว้

## Labs — รันได้ที่นี่

**labs/lab00_download_models.py** — ตั้งค่าครั้งเดียว: ดาวน์โหลด Laya (EN + หลายภาษา), GLiNER2.5-Decide (EN + multi) และ GLiNER multi — รันจาก ⌨ terminal

**labs/lab01_landscape.py** — ทัวร์ภาพรวม System One เดือนตุลาคม 2026 แบบออฟไลน์ กระดานคะแนน และตารางตัวไหนเหมาะกับงานอะไร

**labs/lab02_laya_local.py** — Laya เทียบกับ Jev บนคำขอเดียวกัน; ticket ภาษาไทยแบบ Router เลือกเองกับแบบบังคับภาษาอังกฤษ; กับดักการตัดข้อความ

**labs/lab03_gliner_extract_then_decide.py** — GLiNER แบบดั้งเดิมดึงหมายเลขห้องพร้อมตำแหน่ง; Jev เลือกห้องที่มีปัญหา; โค้ดไฮไลต์ให้

**labs/lab04_gliner_decide.py** — adapter จาก Jev ไป GLiNER, Decide ภาษาอังกฤษเทียบหลายภาษาบนข้อความไทย, หัวข้อหลายป้าย และทำไมคะแนนรวมไม่เป็น 1

**labs/lab05_bakeoff.py** — 18 ticket EN/TH ผ่าน Jev, Laya และ GLiNER2.5-Decide: ความแม่นยำ ความคลาดเคลื่อนของความเร่งด่วน ความเร็ว ต้นทุน และ calibration

โมเดลในเครื่องจะรันบนเครื่องนี้เมื่อตั้งค่า `week24_typed_decisions_jev/.venv-local` แล้ว มิฉะนั้นเล่นซ้ำจากการบันทึก ฝั่ง Jev รัน LIVE ด้วย key ของคุณ หรือ DRY จากการบันทึก

## Try it yourself — ลองทำเอง

**Exercise 16 — adapter และตัวเลือกเครื่องมือ** เปิด `week24_typed_decisions_jev/16_local_system_one_models/exercises/ex16_local_or_hosted.py` มี TODO สองข้อ:

1. เขียน `to_gliner(questions)` — แปลง choice, noul และ score เป็นงานของ GLiNER2.5-Decide คงลำดับระดับของ score ลบ backtick ออก และ `ValueError` สำหรับชนิดอื่นทั้งหมด (คำถาม `box` ต้องไม่หายไปเงียบ ๆ)
2. เขียน `pick_engine(task)` — ต้องการ offset → GLiNER; ข้อมูลลับ → เครื่องมือในเครื่องเท่านั้น (ยาวเกิน → คน, ไทย → Laya, อังกฤษ → Decide); งบเวลาต่ำกว่า 100 ms → Laya; นอกนั้น → Jev

การทดสอบแบบออฟไลน์ 11 ข้อรันก่อนและฟรี เมื่อผ่านทั้งหมด ticket หนึ่งรายการจะถูกส่งผ่าน adapter **ของคุณ** ไปที่ GLiNER2.5-Decide บนเครื่องนี้

```bash
.venv/bin/python week24_typed_decisions_jev/16_local_system_one_models/exercises/ex16_local_or_hosted.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
✓ to_gliner maps choice, noul and score exactly
✓ to_gliner raises ValueError for a box question
✓ score levels keep their order
✓ pick default → jev
✓ pick {'needs_offsets': True, 'data': 'confidential'} → gliner_ner
…
✓ all offline tests pass — one ticket through YOUR adapter on this machine
» {'queue': {'label': 'safety', 'confidence': 0.64…}, 'safety': {'label': 'yes', …}, 'urgency': {'label': 'routine', …}}
```

<details><summary>คำใบ้ — ทำไม "ข้อมูลลับ + ไม่มั่นใจ" ต้องไม่ถอยไปใช้ Jev</summary>

กฎแบบ "ถ้าโมเดลในเครื่องไม่แน่ใจ ให้ถามคลาวด์" จะส่งข้อความลับที่ยากที่สุดออกนอกเครื่องพอดี บทที่ 10 เรียกสิ่งนี้ว่าบั๊กด้านธรรมาภิบาลข้อมูล ใน `pick_engine` กิ่งของข้อมูลลับต้องคืนเฉพาะเครื่องมือในเครื่องหรือ `"human"` เท่านั้น

</details>

<details><summary>ท้าทายเพิ่ม — Kev และ laya-serve หลัง client ตัวเดียวกัน</summary>

`laya-serve` (`pip install "laya[serve]"`) และเซิร์ฟเวอร์ของ Kev ต่างเปิด `POST /v1/systemone` ลองคัดลอก `jevkit.ENDPOINT` ให้ชี้ไปที่ `http://127.0.0.1:8000/v1/systemone` แล้วรันแล็บของบทที่ 04 กับ Laya — จดว่าคำตอบไหนเปลี่ยน และสิ่งนั้นส่งผลอย่างไรกับเกณฑ์ที่คุณปรับไว้บน Jev

</details>

✓ Checkpoint: การทดสอบทั้งสิบเอ็ดข้อผ่าน และตัวเลือกของคุณไม่เคยคืน `"jev"` สำหรับข้อมูลลับ

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `◇ Laya not available here → replaying recordings` | ไม่มี `week24_typed_decisions_jev/.venv-local` หรือติดตั้งไม่ครบ — รันคำสั่งตั้งค่าสองชุดใน §0 |
| `… LocalEntryNotFoundError` / ข้อผิดพลาดโหมดออฟไลน์ | ยังไม่ได้ดาวน์โหลด weights — รัน Lab 16-0 จาก ⌨ terminal |
| Lab 16-0 หมดเวลาเมื่อกด ▶ Run | เป็นเรื่องปกติ การดาวน์โหลดครั้งแรกอาจใช้หลายนาที — ใช้ terminal |
| `tiktoken is required to read a tiktoken file` | `uv pip install --python week24_typed_decisions_jev/.venv-local/bin/python tiktoken sentencepiece` |
| `laya.load()` ค้าง | model card ของ Laya: transformers + TensorFlow อาจ deadlock — `localkit` ตั้ง `USE_TF=0` ให้แล้ว อย่าติดตั้ง TF ใน `.venv-local` |
| คำตอบภาษาไทยดูสุ่ม | คุณใช้ checkpoint ภาษาอังกฤษอย่างเดียว (Laya `model="english"` หรือ `GLiNER2.5-Decide`) — ใช้ Router หรือ `GLiNER2.5-multi-Decide` |
| แล็บในเครื่องช้าในครั้งแรก | การโหลด checkpoint ใช้ 5–25 วินาที การเรียกครั้งต่อไปใช้เพียงมิลลิวินาที |

## Next — ต่อไป

**[บทที่ 17 — ปรับจูน (fine-tune) โมเดล System One ในเครื่อง](../17_finetune_system_one/TUTORIAL.th.md)**: Laya ตัวตั้งต้นได้ 12/18 ในบทนี้ และตอบภาษาไทยผิดอย่างมั่นใจ บทที่ 17 จะปรับจูน Laya และ GLiNER2.5-multi-Decide ด้วยตั๋ว 456 ใบ (บน Mac หรือ DGX Spark) แล้วให้คะแนนอย่างตรงไปตรงมากับตั๋วทดสอบ 66 ใบที่ไม่เคยใช้ฝึก ซึ่งรวมตั๋ว 18 ใบของบทนี้ด้วย

กลับไปที่ [ภาพรวมของคอร์ส](../README.md) — หรือต่อยอดเรื่องราวของบทที่ 15: ตอนนี้ Jev 1.14 อ่านภาพได้ (และ `pplx-decider` กับ Clef-omni ก็อ่านได้เช่นกัน) ลองวางโมเดลข้อความในเครื่อง (บทนี้) ไว้หลังด่านกรองภาพ (บทที่ 15) แล้วถามว่าการตัดสินใจไหนบ้างที่จำเป็นต้องออกนอกอาคารจริง ๆ
