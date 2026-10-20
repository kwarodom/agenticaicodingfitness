# ▶ Jev Lab 15 — Jev มองเห็นภาพ: รูปถ่าย Box และ long context ใน Jev 1.14

> ส่วนหนึ่งของ Week 24 · การตัดสินใจแบบมีชนิด (typed decision) ด้วย Jev — วันที่ 2026-10-14 TypeSafe ปล่อย **Jev 1.14.0** พร้อมฟีเจอร์ "danger zone" (โซนทดลอง) สามอย่างสำหรับนักสร้างที่อยากลองของใหม่: **ภาพ** ในคำขอใดก็ได้, คำถามชนิดใหม่ **`box`** ที่บอกว่าสิ่งนั้น *อยู่ตรงไหน* ในภาพ และ **long context** (262k โทเคน) ในบทที่ 13 ต้องมี vision LLM บรรยายรูปเป็นข้อความก่อน Jev จึงจะตัดสินได้ — บทนี้ Jev ดูรูปเองโดยตรง และคุณจะวัดผลว่าเมื่อไรวิธีนี้ดีกว่า สื่อทั้งหมดเป็นข้อมูลสังเคราะห์ (synthetic)

**สิ่งที่คุณจะได้ลงมือทำ**
- ส่งรูปถ่ายเข้า Jev โดยตรง — และเห็น HTTP 400 ที่ได้เมื่อไม่ใส่ header ยืนยันการใช้งาน
- วางภาพในสามตำแหน่งที่ข้อความวางได้: `state`, `instructions` และ `criteria` — แล้วเทียบเฟรม CCTV สองเฟรมตามเวลา
- หาแอ่งน้ำด้วย **Box** ครอปภาพ และไม่ยอมเชื่อกรอบที่วาดรอบสิ่งที่ไม่มีอยู่จริง
- แข่งเส้นทางตรงกับ pipeline แบบ vision LLM ของบทที่ 13 บนรูปชุดเดียวกัน
- เทียบ Jev 1.13 กับ 1.14 เรื่องภาพ Box และภาษาไทย — และส่ง log ขนาด 47k โทเคนผ่าน long context

**Time** ~40 นาที · **Difficulty** ระดับกลาง · **Cost** ≈ $0.004 เมื่อรันจริง (ภาพละ ≈ $0.0001, long context หนึ่งครั้ง ≈ $0.002) · $0 ในโหมด dry

## 0 · Jev 1.14 เพิ่มอะไร และ "danger zone" หมายถึงอะไร

บทความประกาศของ TypeSafe ("Jev: For the builders AND dreamers", 2026-10-14) มีการปรับปรุงแบบเสถียรหนึ่งอย่าง และของทดลองสามอย่าง:

| ฟีเจอร์ | วิธีเปิดใช้ | ใช้ได้กับ | สถานะ |
|---|---|---|---|
| **jev-1.14.0** — ดีขึ้นเล็กน้อยทุกด้าน มากที่สุดในงานหลายภาษา | `"model": "jev-1.14.0"` หรือชื่อแฝง `jev-preview` | 1.14 | รุ่นทางการ; `jev-latest` จะตามมา "ภายในหนึ่งสัปดาห์" |
| **ภาพ** ใน `state`, `instructions` หรือ `criteria` | header `TypeSafe-Danger-Zone-Images: true` + map ชื่อ `assets` | 1.13 **และ** 1.14 ("เป็น multimodal มาตั้งแต่แรก") | dreamer-ready — ยังไม่ใช่ระดับ production |
| คำถาม **`box`** — คืนกรอบสี่เหลี่ยมหนึ่งกรอบ | header ภาพตัวเดียวกัน | **1.14 เท่านั้น** | dreamer-ready |
| **Long context** (262k แทน 64k ต่อคำขอ / 32k ต่อคำถาม) | header `TypeSafe-Danger-Zone-Long-Context: true` | 1.13 และ 1.14 | ช้ากว่าและแม่นยำน้อยกว่า ราคาเท่าเดิม |

"Danger zone" เป็นคำของ TypeSafe เอง ฟีเจอร์เหล่านี้จะอยู่ต่อและดีขึ้นเรื่อย ๆ แต่ยังมีความ *ไม่สม่ำเสมอ (jagged)* มากเกินกว่าที่ TypeSafe จะรับรองสำหรับ production บทนี้จึงยังใช้กฎเดิมตั้งแต่บทที่ 01: **Jev ตัดสิน โค้ดเป็นผู้ตัดสินใจ** และมีคนอยู่ในวงจรเสมอสำหรับของใหม่

```text
photo ── image() ──► {state: {"photo": "img-…"},  assets: {"img-…": {jpg, base64}},  questions}
                                │  header TypeSafe-Danger-Zone-Images: true
                                ▼
                     Jev 1.14  →  choice · noul · score · box      ~1 s · ~2.7k tokens · ≈ $0.0001
                                ▼
                     your code →  gate · crop · policy · execute: False
```

เรื่องเวอร์ชัน: ณ วันที่ 2026-10-17 `jev-preview` ชี้ไปที่ `jev-1.14.0` ส่วน `jev-latest` ยังชี้ไปที่ `jev-1.13.0` คอร์สนี้ตรึงค่าเริ่มต้นไว้ที่ **`jev-1.13.0`** เพื่อให้การเล่นซ้ำในโหมด DRY ตรงทุกไบต์ ส่วนบทที่ 15 ตรึงไว้ที่ **`jev-1.14.0`** เพราะ Box ต้องใช้รุ่นนี้ — ตัวเลือกโมเดลใน runner มีให้เลือกทั้งสองรุ่นแล้ว

✓ Checkpoint: คุณบอกชื่อ header ทั้งสองได้ บอกได้ว่าฟีเจอร์ไหนต้องใช้ 1.14 และอธิบายได้ว่าทำไมคอร์สยังตรึงเวอร์ชันแทนการใช้ชื่อแฝง

## 1 · การเรียกด้วยภาพครั้งแรก

ภาพแต่ละภาพถูกระบุ **ครั้งเดียว** ใน map `assets` ระดับบนสุด โดยใช้ *ข้อความอ้างอิง (reference)* เป็น key — แล้วเขียนข้อความอ้างอิงนั้นไว้ตรงไหนก็ได้ที่ต้องการให้ภาพไปอยู่ `jevkit.image()` เตรียมไฟล์ตามที่เอกสารกำหนด: หมุนพิกเซลให้ตั้งตรง (API ลบ EXIF ทิ้งและไม่ใช้ค่าการหมุนในนั้น), ย่อด้านยาวให้ไม่เกิน 1568 px, เติมขอบภาพที่เล็กกว่า 280 px และบันทึกเป็น JPEG มาตรฐาน — และทำให้ข้อความอ้างอิงเป็น content hash (`img-` + เลขฐานสิบหก 16 ตัว) รูปเดียวกันจึงได้ข้อความอ้างอิงเดิมเสมอ ซึ่งทำให้โหมด DRY เล่นซ้ำคำขอที่มีภาพได้

```bash
.venv/bin/python week24_typed_decisions_jev/15_jev_sees_images/labs/lab01_first_image_call.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 1 · prepare media/leak_under_ahu.jpg
» file: 1152×768 px · 123 KB
» reference: img-bdb430c290c344ad   (sha256 of the file → the same photo always gets the same reference)
…
▣ STEP 3 · send it WITHOUT the danger-zone header
✕ HTTP 400 … "Images are experimental. Send the header TypeSafe-Danger-Zone-Images: true to use them."
▣ STEP 4 · send it WITH the header (jevkit adds it when assets are present)
» leak                   noul    0.98  ████████████████████████  likely YES
» queue                  choice  → water_leak   (confidence 1.00)
◆ jev-1.14.0 · LIVE · 2784 input tok · $0.000117 · 1083 ms
```

ภาพคิดเงินเป็น **input token** ธรรมดา — ไม่มีราคาแยกสำหรับภาพ รูปขนาด 1152×768 หนึ่งรูปใช้ประมาณ 2.6k โทเคน ราว 15 เท่าของการเรียกด้วยข้อความสั้น ๆ แต่ยังแค่ประมาณ $0.0001 ขีดจำกัดที่ต้องจำ: PNG หรือ JPEG เท่านั้น แต่ละด้าน 280–1568 px ไฟล์ละไม่เกิน 16 MiB และทั้งคำขอไม่เกิน 32 MiB ถ้ามีภาพ **ใดภาพหนึ่ง** ผิดขีดจำกัด คำขอทั้งหมดจะล้มเหลว — ไม่มีคำตอบบางส่วน

✓ Checkpoint: คุณเห็น 400 เมื่อไม่ใส่ header แล้ว และอธิบายได้ว่าทำไมข้อความอ้างอิงแบบ content hash ปลอดภัยกว่าชื่อสั้น ๆ อย่าง `cat`

## 2 · ภาพไปอยู่ได้ทุกที่ที่ข้อความไปได้

| ตำแหน่ง | ความหมาย | ตัวอย่างใน Lab 15-2 |
|---|---|---|
| `state` | หลักฐานที่ทุกคำถามถามถึง | รูปน้ำรั่ว คู่กับช่องข้อความ `location` |
| `instructions` | ภาพที่เป็นของคำถาม **ข้อเดียว** | "`photo` เป็นห้องเดียวกับภาพอ้างอิงนี้หรือไม่: img-…?" |
| `criteria` | ตัวเลือกที่นิยาม **ด้วยภาพตัวอย่าง** แทนคำพูด | `normal` / `haze` / `water` แต่ละตัวมีภาพประกอบ |
| รายการใน `state` | ประวัติสั้น ๆ เรียงจากเก่าไปใหม่ | เฟรม CCTV สองเฟรม: "มีหมอกควันเกิดขึ้นระหว่างสองเฟรมหรือไม่?" |

```bash
.venv/bin/python week24_typed_decisions_jev/15_jev_sees_images/labs/lab02_where_images_go.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 2 · INSTRUCTIONS — compare a live CCTV frame against a known reference photo
» same_room              noul    0.53  █████████████░░░░░░░░░░░  uncertain
» differs_from_normal    noul    0.98  ████████████████████████  likely YES
◆ jev-1.14.0 · LIVE · 6702 input tok · $0.000281 · 804 ms
▣ STEP 3 · CRITERIA — options defined by example pictures (and one by words)
   photo = media/cctv_t1s.jpg   → normal (confidence 0.99)
   photo = media/cctv_t7s.jpg   → haze   (confidence 1.00)
▣ STEP 4 · FRAMES — a short history, oldest first: what changed?
» haze_appeared          noul    0.99  ████████████████████████  likely YES
» haze_before            noul    0.02  ░░░░░░░░░░░░░░░░░░░░░░░░  likely NO
```

ดูที่ `same_room = 0.53` — เฟรม CCTV กับรูปอ้างอิงเป็นห้องเดียวกัน แต่เฟรมมีแถบข้อความ มุมครอปต่างกัน และมีหมอกควัน Jev ไม่แน่ใจ และ *บอกออกมาว่าไม่แน่ใจ* นี่คือความไม่สม่ำเสมอที่ TypeSafe เตือนไว้ คำถามที่แยกย่อยและเรียบง่าย (`differs_from_normal`) ทำได้ดีกว่าคำถามแบบภาพรวมมาก อีกสองกฎจากเอกสารที่จำนวนโทเคนยืนยัน: **ทุกครั้งที่ใช้ข้อความอ้างอิง ภาพจะถูกเพิ่มเข้าไปอีกหนึ่งชุด** (6.7k โทเคนเมื่อมีภาพอ้างอิงหนึ่งภาพ, 9.1k เมื่อมีภาพตัวอย่างสามภาพ) และ **ชื่อตัวเลือกเป็น key ของ object** จึงไม่ถูกแทนที่ด้วยภาพ — โค้ดของคุณยังได้รับค่า `normal` / `haze` เหมือนเดิม

✓ Checkpoint: คุณอธิบายได้ว่าทำไม `same_room` ไม่แน่ใจ ขณะที่ `differs_from_normal` มั่นใจ และบอกได้ว่าควรวางภาพที่สามคำถามต้องใช้ร่วมกันไว้ตรงไหน

## 3 · Box — มันอยู่ตรงไหน?

คำถามชนิด `box` คืนกรอบสี่เหลี่ยมที่เล็กที่สุดรอบเป้าหมายหนึ่งชิ้น เป็นขอบสี่ด้านในรูปสัดส่วนของภาพ (`top`, `bottom`, `left`, `right` ค่า 0–1) พร้อม `confidence` คูณด้วยขนาดพิกเซลเพื่อครอป — กรอบเดียวกันใช้ได้กับภาพทุกความละเอียด

```bash
.venv/bin/python week24_typed_decisions_jev/15_jev_sees_images/labs/lab03_box_and_crop.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 1 · media/leak_under_ahu.jpg — where is the puddle?
» puddle_visible         noul    0.97  ███████████████████████░  likely YES
» puddle                 box     top 0.67 · bottom 0.95 · left 0.15 · right 0.76   (17% of the image, confidence 0.69)
→ gate 0.97 · box confidence 0.69
» saved .runs/puddle_crop.jpg and .runs/puddle_box.jpg
▣ STEP 2 · ask for something that is not in the photo
» extinguisher_visible   noul    0.09  ██░░░░░░░░░░░░░░░░░░░░░░  likely NO
» extinguisher           box     top 0.02 · bottom 0.07 · left 0.64 · right 0.69   (0% of the image, confidence 0.34)
→ gate 0.09 < 0.8 → target probably absent, ignore the box
▣ STEP 4 · media/thermostat_room1203.png — find the reading, then ask about it
» band                   choice  → above_28   (confidence 1.00)
» above_setpoint         noul    0.97  ███████████████████████░  likely YES
```

![กรอบ Box รอบแอ่งน้ำที่ Lab 15-3 วาดบนรูป (สำเนาของ .runs/puddle_box.jpg)](media/puddle_box_example.jpg)

สามนิสัยที่แล็บนี้สร้าง:

1. **คำตอบ Box เป็นกรอบเสมอ** — ไม่มีคำตอบ "หาไม่เจอ" ให้ถาม Noul *ในคำขอเดียวกัน* ("มี … ใน `photo` หรือไม่?") แล้วให้โค้ดเพิกเฉยต่อกรอบเมื่อ Noul ต่ำ ขั้นที่ 2 คือกรอบที่วาดรอบความว่างเปล่าด้วย confidence 0.34
2. **confidence เป็นอีกแกนหนึ่งแยกกัน** — ถ้าต่ำกว่าเกณฑ์ ให้ขยายกรอบ (`V.pad` ด้านละ 5 %) หรือถามใหม่บนภาพครอปที่ใหญ่ขึ้น เอกสารไม่มีเกณฑ์แนะนำ — ปรับจูนบนภาพของคุณเอง ค่า 0.80 / 0.60 ในแล็บเป็นเพียงตัวอย่าง
3. **ตรวจภาพครอปด้วยคำถามรายขอบ** — Noul หนึ่งข้อต่อหนึ่งขอบ ("ต่อเลยขอบซ้ายออกไปหรือไม่?") ในการทดสอบของ TypeSafe วิธีนี้จับการตัดขาดได้ 30 จาก 36 ครั้ง เทียบกับ 26 ครั้งของคำถามเดียว "อยู่ในกรอบทั้งหมดหรือไม่?" ในรอบของเรา ขอบซ้ายได้ 0.76 ทั้งที่ดูจากภาพกรอบก็ดีอยู่ — คำถามรายขอบคือ *สัญญาณ* ให้คนดู ไม่ใช่ข้อพิสูจน์

ขั้นที่ 4 ขีดเส้นเดียวกับบทที่ 13: Jev บอกได้ว่าเทอร์โมสแตต **สูงกว่า 28 °C** และ **สูงกว่าค่าตั้งเกิน 3 °C** และ Box หาตำแหน่งตัวเลขเจอ — แต่ Jev ไม่คืนข้อความ จึงบอกคุณไม่ได้ว่า "29.1" ค่าที่แม่นยำยังต้องมาจาก vision LLM หรือ OCR แล้วให้โค้ดตรวจสอบ

✓ Checkpoint: คุณเปิด `.runs/puddle_box.jpg` แล้ว และอธิบายได้ว่าทำไมกรอบถังดับเพลิงถูกเพิกเฉย ทั้งที่ Jev คืนขอบสี่ด้านที่ถูกต้องตามรูปแบบ

## 4 · ตรง หรือ pipeline? วัดผลเลย

Lab 15-4 ส่งรูปสามรูปเดียวกันผ่านทั้งสองเส้นทาง โดยใช้คำถามของบทที่ 13 แบบไม่แก้ไข:

- **pipeline**: `gemini-3.5-flash` บรรยายรูป → Jev 1.13 ตัดสิน `description`
- **ตรง (direct)**: Jev 1.14 ตัดสิน `photo`

```bash
.venv/bin/python week24_typed_decisions_jev/15_jev_sees_images/labs/lab04_direct_vs_pipeline.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
│ photo                 pipeline (text)              direct (image)               agree  pipe time  direct time
│ leak_under_ahu.jpg    water_leak · safety 0.93     water_leak · safety 0.96     ✓      4.2 s      1.1 s
│ plantroom_smoke.jpg   smoke_or_fire · safety 0.97  smoke_or_fire · safety 0.91  ✓      3.7 s      0.7 s
│ plantroom_normal.jpg  none · safety 0.47           none · safety 0.05           ✓      4.1 s      0.8 s
» 3/3 photos: same queue and same side of 0.5 on safety
```

บนรูปชุดนี้ เส้นทางตรงเร็วกว่าราว 4 เท่า จ่ายเงินครั้งเดียวแทนสองครั้ง และตัดสินห้องปกติได้ชัดเจนกว่า (คำบรรยายของ pipeline ยังทำให้ `safety` ลอยอยู่ที่ 0.47) แต่สามรูปคือการสาธิต ไม่ใช่การประเมินผล — ทดสอบกับรูปของคุณเองก่อนเปลี่ยนอะไร ใช้ตารางนี้ในการเลือก:

| สถานการณ์ | เลือก |
|---|---|
| ต้องการตัวเลขหรือข้อความที่แม่นยำ (29.1 °C, ยอดรวมใบแจ้งหนี้) | **pipeline** — Jev ไม่เขียนอะไรออกมา |
| ต้องการร่องรอยที่คนอ่านได้ ("โมเดลเห็นอะไร?") | **pipeline** — คำบรรยาย *คือ* log |
| เสียงหรือวิดีโอ | **pipeline** — Jev รับเฉพาะภาพนิ่ง PNG/JPEG (ดึงเฟรมออกมา เหมือน Lab 15-2) |
| การตัดสินแบบมีชนิดบนรูปอย่างรวดเร็ว เรียกครั้งเดียว จ่ายครั้งเดียว | **ตรง** |
| เทียบกับภาพที่รู้จัก หรือนิยามตัวเลือกด้วยภาพตัวอย่าง | **ตรง** — ข้อความส่งต่อสิ่งนี้ไม่ได้ |
| production ในวันนี้ | pipeline หรือเส้นทางตรงที่มีคนตรวจ — ภาพยังเป็น "dreamer-ready" |

✓ Checkpoint: คุณบอกได้หนึ่งงานของ AltoTech ที่จะคง pipeline ไว้ และหนึ่งงานที่เส้นทางตรงคุ้มค่าแก่การทดลอง

## 5 · 1.13 กับ 1.14 ภาษาไทย และ long context

```bash
.venv/bin/python week24_typed_decisions_jev/15_jev_sees_images/labs/lab05_versions_and_long_context.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
│ model       leak noul  input tok  Box question
│ jev-1.13.0  1.00       2652       ✕ HTTP 400 "Invalid request."
│ jev-1.14.0  0.98       2652       ✓ conf 0.71
…
│ ไม่ต้องส่งช่างมาแล้วนะคะ แอร์ก  no_action     ✓ no_action 0.70     ✓ no_action 0.86
» correct: jev-1.13.0 6/6 · jev-1.14.0 6/6 — six messages prove nothing; they show you HOW to check.
▣ STEP 4 · a 47k-token BMS log with one alarm buried in it
✕ without header: HTTP 400 … {"detail":{"error_type":"max_tokens_exceeded"}}
» with TypeSafe-Danger-Zone-Long-Context: true → alarm noul 0.99
◆ jev-1.13.0 · LIVE · 46997 input tok · $0.001974 · 907 ms
```

สามเรื่องที่ควรสังเกต:

- **ภาพใช้ได้ทั้งสองรุ่น Box ใช้ได้เฉพาะ 1.14** — 1.13 ตอบคำขอ Box ด้วย `Invalid request.` เฉย ๆ
- **ภาษาไทย:** ทั้งสองรุ่นตอบถูกทั้งหกข้อ แต่ 1.14 มั่นใจกว่าในข้อยาก — คำขอแบบ *ปฏิเสธ* ("ไม่ต้องส่งช่างมาแล้ว") 0.86 เทียบกับ 0.70 สอดคล้องกับที่ TypeSafe บอกว่างานหลายภาษาดีขึ้นมากที่สุด — แต่ก็ยังเป็นแค่หกข้อความ ใช้วิธีประเมินผลของบทที่ 09 ก่อนเปลี่ยนเวอร์ชันที่ตรึงไว้
- **Long context:** log ขนาด 47k โทเคนเกินขีดจำกัด 32k ของ state + คำถาม จึงได้ 400 เมื่อไม่มี header และได้คำตอบเมื่อมี header ในราคาต่อโทเคนเท่าเดิม แต่ทางที่ดีกว่ายังคือกรอง log ด้วยโค้ดก่อน (grep หา `ALARM`) แล้วถาม Jev เฉพาะ 20 บรรทัดที่สำคัญ — ถูกกว่า เร็วกว่า และแม่นยำกว่า

ลองคำขอรูปแบบ long context ในหน้านี้ได้เลย (state สั้น จึงใช้ได้ทั้งมีและไม่มี header — บล็อกนี้แสดงแค่รูปร่างของคำขอ):

```jev
{
  "state": {"log": "2026-10-03 03:10 AHU-3 SAT 13.4 C status OK\n2026-10-03 03:12 AHU-3 SMOKE DETECTOR ALARM duct smoke detected, unit tripped\n2026-10-03 03:14 AHU-3 SAT 18.9 C status FAULT"},
  "questions": {
    "alarm": {"type": "noul", "instructions": "Does `log` contain a smoke or fire alarm?"},
    "unit": {"type": "choice", "instructions": "Which unit in `log` raised the alarm?", "criteria": {"AHU-1": "AHU-1", "AHU-2": "AHU-2", "AHU-3": "AHU-3", "none": "no alarm"}}
  }
}
```

✓ Checkpoint: คุณอธิบายได้ว่าทำไมการตอบถูก 6/6 จากข้อความไทยหกข้อความไม่ใช่เหตุผลในการเปลี่ยนเวอร์ชัน และทำไม grep-แล้ว-Jev ดีกว่า long context สำหรับ log ของ BMS

## Labs — รันได้ที่นี่

**labs/lab01_first_image_call.py** — เตรียมรูป ดูคำขอดิบ เห็น 400 เมื่อไม่มี header แล้วได้คำตอบแบบมีชนิดพร้อมค่าโทเคน

**labs/lab02_where_images_go.py** — ภาพใน state, ใน instructions (เทียบกับภาพอ้างอิง), ใน criteria (ตัวเลือกด้วยภาพตัวอย่าง) และเป็นประวัติสองเฟรม

**labs/lab03_box_and_crop.py** — Box รอบแอ่งน้ำ ครอปลง `.runs/` ไม่เชื่อกรอบรอบความว่างเปล่า ตรวจขอบ และหาค่าบนเทอร์โมสแตต

**labs/lab04_direct_vs_pipeline.py** — รูปสามรูปเดียวกัน ผ่าน pipeline แบบ vision LLM ของบทที่ 13 และผ่าน Jev โดยตรง — คำตอบ เวลา และโทเคน

**labs/lab05_versions_and_long_context.py** — 1.13 กับ 1.14 เรื่องภาพ Box และภาษาไทย; log 47k โทเคนแบบมีและไม่มี header long context

ทั้งห้าแล็บรัน LIVE ด้วย TypeSafe key ของคุณ (Lab 15-4 ใช้ผู้ให้บริการ vision ด้วย) หรือ DRY จากการบันทึกเมื่อ 2026-10-17

## Try it yourself — ลองทำเอง

**Exercise 15 — ตรวจคำขอภาพก่อนส่ง แล้วกรองคำตอบ Box** เปิด `week24_typed_decisions_jev/15_jev_sees_images/exercises/ex15_image_preflight.py` มี TODO สองข้อ:

1. เขียน `preflight(width, height, fmt, file_bytes)` — คืนข้อความปัญหาหนึ่งข้อต่อหนึ่งกฎของ API ที่ผิด (format, side, pixels, size) และ `[]` เมื่อภาพส่งได้อย่างปลอดภัย
2. เขียน `gate(answers)` — Noul ต่ำกว่า 0.80 → เพิกเฉยต่อกรอบ; confidence ของกรอบต่ำกว่า 0.60 → ขยายกรอบ 5 %; นอกนั้น → ใช้กรอบตามที่ได้

การทดสอบแบบออฟไลน์ 12 ข้อรันก่อนและฟรี เมื่อผ่านทั้งหมด ตัวตรวจจะส่งเฟรม CCTV หนึ่งเฟรมไปที่ `jev-1.14.0` รัน gate **ของคุณ** กับคำตอบ และบันทึกภาพที่วาดกรอบลง `.runs/`

```bash
.venv/bin/python week24_typed_decisions_jev/15_jev_sees_images/exercises/ex15_image_preflight.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
✓ preflight (1152, 768, 'jpg', 125582) → []
✓ preflight (2000, 1500, 'jpg', 900000) → ['side 2000×1500 — …', 'pixels 3,000,000 — …']
…
✓ gate visible=0.95 conf=0.8 → act=True
✓ all offline tests pass — now one real frame (media/cctv_t7s.jpg) through jev-1.14.0
» visible                noul    0.96  ███████████████████████░  likely YES
» where                  box     top 0.05 · bottom 0.48 · left 0.19 · right 0.43   (10% of the image, confidence 0.33)
→ your gate: {'act': True, 'edges': {…padded…}, 'reason': 'padded — box confidence 0.33'}
```

<details><summary>คำใบ้ — รูปหนึ่งรูปผิดได้สองกฎพร้อมกัน</summary>

ตรวจแต่ละกฎด้วย `if` ของตัวเองแล้ว append — อย่าใช้ `elif` รูปขนาด 2000×1500 มีด้านหนึ่งยาวเกิน **และ** มีพิกเซลมากเกิน ตัวตรวจคาดหวังปัญหาทั้งสองข้อ

</details>

<details><summary>ท้าทายเพิ่ม — ค้นหาในภาพใหญ่ด้วยการแบ่งช่อง</summary>

เอกสารแนะนำรูปแบบสำหรับเป้าหมายเล็กในภาพใหญ่: แบ่งภาพเป็น 4 ช่อง ถาม Choice หนึ่งข้อโดยให้ตัวเลือกเป็นช่องต่าง ๆ (`"criteria": {"tl": "<ref>", "tr": "<ref>", …}`) แล้วทำซ้ำบนช่องที่ถูกเลือกจนเป้าหมายกินพื้นที่ส่วนใหญ่ของช่อง จากนั้นถาม Box บนช่องนั้น และแปลงขอบกลับเป็นสัดส่วนของภาพเต็ม

</details>

✓ Checkpoint: การทดสอบทั้งสิบสองข้อผ่าน และกรอบสีแดงใน `.runs/ex15_haze_box.jpg` อยู่บนกลุ่มหมอกควัน

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `HTTP 400 … Images are experimental` | ส่ง `TypeSafe-Danger-Zone-Images: true` — `jevkit.ask(assets=…)` ใส่ให้อัตโนมัติ เว้นแต่คุณส่ง `images_header=False` |
| `HTTP 400 "Invalid request."` เมื่อถาม Box | Box ต้องใช้ `jev-1.14.0` หรือ `jev-preview` — 1.13 ปฏิเสธ |
| `HTTP 413` | ภาพหรือคำขอเกินขีดจำกัด: ย่อให้ไม่เกิน 1568 px ส่งภาพน้อยลง หรือวางภาพที่ใช้ร่วมกันไว้ใน `state` ครั้งเดียว |
| `HTTP 400 … max_tokens_exceeded` | ภาพรวมกับข้อความเกิน context: ย้ายภาพออกจาก `instructions` หรือเปิด long context สำหรับข้อความยาว |
| รูปมาถึงแบบตะแคง | API ไม่ใช้ค่าการหมุนใน EXIF — `image()` หมุนพิกเซลให้ก่อน โค้ดของคุณเองก็ต้องทำเช่นกัน |
| `◈ DRY — no recording` | คุณเปลี่ยนคำถาม ภาพ หรือโมเดล — สลับเป็น ⚡ Live หรือย้อนการเปลี่ยนแปลง |

## Next — ต่อไป

ไปต่อที่ [Lab 16 — โมเดล System One แบบรันในเครื่อง](../16_local_system_one_models/TUTORIAL.md): Laya, GLiNER และ GLiNER2.5-Decide รันบนเครื่องของคุณเองด้วยแนวคิดการตัดสินใจแบบมีชนิดเดียวกัน — วัดผลเทียบกับ Jev บน ticket ชุดเดียวกัน
