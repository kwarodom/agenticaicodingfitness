# ▶ Jev Lab 06 — เช็กลิสต์หลักฐานสำหรับงาน HR อย่างรับผิดชอบ

> ส่วนหนึ่งของ Week 24 · การตัดสินใจแบบมีชนิดข้อมูล (typed decisions) ด้วย Jev งานสรรหาบุคลากรคือจุดที่ AI ที่ใช้อย่างไม่ระวังสร้างความเสียหายได้จริง แล็บนี้ใช้ Jev กับงานเดียวที่แคบและตรวจสอบย้อนหลังได้: **หาหลักฐานที่ระบุชัดในข้อความเชิงวิชาชีพ** เพื่อให้เจ้าหน้าที่สรรหา (recruiter) ตรวจยืนยัน ไม่ใช่เพื่อจัดอันดับ ให้คะแนน หรือคัดใครออก

**สิ่งที่คุณจะได้ลงมือทำจริง**
- เรียนรู้กติกาพื้นฐานก่อน: ข้อมูลมาจากไหนได้บ้าง และอะไรบ้างที่โมเดลต้องไม่เป็นผู้ตัดสิน
- สร้างเช็กลิสต์หลักฐาน: สำหรับเกณฑ์งานแต่ละข้อ ตอบ `evidenced` / `not_stated` / `unclear`
- ก้าวไปอีกขั้น: ให้ Jev **เลือกประโยคที่เป็นหลักฐานจริง** แล้วให้โค้ดยกประโยคนั้นมาแบบคำต่อคำ
- เพิ่มเกณฑ์ของคุณเอง และ policy ของเช็กลิสต์ที่ไม่สามารถจัดอันดับหรือคัดออกได้

**Time** ~35 นาที · **Difficulty** ระดับกลาง · **Cost** ≈ $0.0003 เมื่อ live · $0 เมื่อ dry

## 0 · กติกาพื้นฐานก่อนเขียนโค้ดใด ๆ

อ่านส่วนนี้ก่อน นี่คือกฎการออกแบบของแล็บนี้ ไม่ใช่ความสามารถของโมเดล และสำคัญกว่าตัวโค้ด

**ข้อความมาจากไหน**

- ✓ ผู้สมัครส่งเรซูเม่หรือสรุปประวัติการทำงานให้คุณ **โดยตรง**
- ✓ ข้อมูลที่องค์กรของคุณถือครองอยู่แล้วอย่างถูกกฎหมายในระบบ ATS ตามเงื่อนไขการประมวลผลขององค์กร
- ✓ ข้อมูลที่มาจาก LinkedIn **เฉพาะ** ผ่านการเชื่อมต่อที่ LinkedIn อนุมัติ และเฉพาะการดำเนินการที่ข้อตกลงของคุณอนุญาต
- ✗ **ห้าม scrape** LinkedIn ห้ามใช้เครื่องมือ scrape บอท และส่วนขยายเบราว์เซอร์ Jev ไม่ท่องเว็บและไม่ scrape และคอร์สนี้จะไม่สอนวิธีทำ "มองเห็นได้สาธารณะ" ไม่ได้แปลว่า "ได้รับอนุญาตให้ประมวลผล"

**ข้อความมีอะไรได้บ้าง**

เฉพาะข้อความเชิงวิชาชีพที่เกี่ยวกับงานและลดทอนข้อมูลแล้ว (minimised) ให้ลบชื่อ ข้อมูลติดต่อ รูปถ่าย อายุ เพศ สัญชาติ และทุกอย่างที่ไม่เกี่ยวกับงานออก **ก่อน** ที่ข้อความจะไปถึงโมเดลใด ๆ ข้อมูลตัวอย่างใน `data/profiles.csv` เป็นข้อมูลสมมติและลดทอนแล้ว มีแค่ `candidate_id` กับ `professional_text`

**โมเดลตัดสินอะไรได้บ้าง**

| อนุญาต (เป็นข้อเสนอที่ recruiter ตรวจยืนยัน) | ห้ามเด็ดขาด |
|---|---|
| "ข้อความนี้อธิบายงานพัฒนาด้วย Python อย่างชัดเจนหรือไม่?" | "ผู้สมัครคนนี้ดีไหม?" |
| "ประโยคไหนคือหลักฐาน?" | คะแนนรวม การจัดอันดับ รายชื่อผู้ผ่านเข้ารอบ (shortlist) |
| "เกณฑ์ข้อนี้ไม่ชัดเจนหรือไม่?" | การคัดออกอัตโนมัติ |
| | บุคลิกภาพ "ความเข้ากันได้กับวัฒนธรรมองค์กร (culture fit)" หรือสิ่งใดก็ตามที่ไม่อยู่ในเกณฑ์งาน |

> ⚠ **`not_stated` หมายถึง "ข้อความนี้ไม่ได้แสดงให้เห็น" ไม่ได้หมายถึง "คนคนนี้ทำไม่ได้" เด็ดขาด** แต่ละคนเล่าถึงงานของตัวเองต่างกันมาก วิธีตอบสนองที่ถูกต้องต่อ `not_stated` คือ *ถามผู้สมัคร*

✓ Checkpoint: คุณบอกแหล่งที่มาของข้อความเชิงวิชาชีพที่ถูกกฎหมายได้สองแหล่ง และอธิบายได้ว่าทำไม `not_stated` ต้องไม่กลายเป็นการคัดออก

## 1 · ออกแบบคำถามหาหลักฐานที่แคบและชัด

ตำแหน่งตัวอย่างคือ **วิศวกรเชื่อมต่อระบบ AI/IoT (AI/IoT integration engineer)** แทนที่จะถามกว้าง ๆ ข้อเดียวว่า "คนนี้มีคุณสมบัติไหม?" เราถามสามคำถามที่ตรงตามตัวอักษรและเกี่ยวกับงาน แต่ละข้อมีคำตอบสามแบบเหมือนกัน:

| Label | ความหมาย |
|---|---|
| `evidenced` | มีการอธิบายโปรเจกต์ งานพัฒนา หรือหน้าที่รับผิดชอบที่เจาะจง |
| `not_stated` | ข้อความไม่ได้อธิบายงานลักษณะนั้น |
| `unclear` | มีการเอ่ยถึงเทคโนโลยีหรือสาขานั้น แต่ไม่ชัดว่าลงมือทำจริงหรือไม่ |

ลองดูผลของการอ่านตามตัวอักษรด้วยตัวเอง ข้อความด้านล่างพูดถึง Python สองครั้ง แต่เป็นแค่ความสนใจและคอร์สเรียน:

```jev
{
  "state": {"professional_text": "Worked six years as a facility technician on chiller plants and AHUs. Completed an online Python course last year."},
  "questions": {
    "python_evidence": {
      "type": "choice",
      "instructions": "Does `professional_text` provide explicit evidence of hands-on Python implementation work? A course or an interest is not hands-on work.",
      "criteria": {
        "evidenced": "A specific Python project, implementation or work responsibility is described.",
        "not_stated": "No concrete Python implementation work is described.",
        "unclear": "Python is named but practical work is ambiguous."
      }
    },
    "building_evidence": {
      "type": "choice",
      "instructions": "Does `professional_text` state work with HVAC, BMS or building energy systems?",
      "criteria": {
        "evidenced": "Specific relevant building-system work is described.",
        "not_stated": "No such work is described.",
        "unclear": "A possible connection is mentioned without concrete work."
      }
    }
  }
}
```

ลองทดลอง:

1. ลบ `"A course or an interest is not hands-on work."` ออกจาก instruction แล้ว `python_evidence` ขยับไปทาง `unclear` หรือ `evidenced` ไหม? (ตอนที่เรารัน: **มี** ประโยคนี้ได้ `not_stated` ที่ confidence **1.00** ส่วนแล็บ 01 ถามคำถามเดียวกันแต่ *ไม่มี* ประโยคนี้ ได้ `not_stated` ที่ confidence แค่ **0.75** และมี 14% อยู่ที่ `unclear` rubric เพิ่มแค่ประโยคเดียวก็ขจัดความกำกวมได้)
2. เปลี่ยนประโยคที่สองเป็น `"Wrote Python scripts to export chiller trend logs every night."` ควรเกิดอะไรขึ้น และเกิดขึ้นจริงไหม?
3. เพิ่มประโยคที่ไม่เกี่ยวกับงาน เช่น `"Enjoys marathon running."` ไม่ควรมีอะไรเปลี่ยน (ถ้างานอดิเรกทำให้ผลเกณฑ์งานขยับ นั่นคือปัญหาอคติ (bias) ที่ต้องตรวจสอบ)

✓ Checkpoint: คุณเห็นแล้วว่าถ้อยคำใน instruction เป็นตัวตัดสินว่าคอร์สเรียนนับเป็นหลักฐานหรือไม่

## 2 · สร้างเช็กลิสต์ — แล็บ 01

แล็บ 01 อ่านโปรไฟล์สมมติสี่รายการจาก `data/profiles.csv` แล้วถามคำถามหาหลักฐานสามข้อกับแต่ละรายการ ทุกรายการไม่ว่าจะได้ label อะไร จะไปที่ `recruiter_review`

```bash
.venv/bin/python week24/06_hr_evidence/labs/lab01_evidence_checklist.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 2 · the checklist for all 4 profiles
│ id             python             integration        building           route
│ candidate-001  evidenced (1.00)   evidenced (1.00)   evidenced (1.00)   recruiter_review
│ candidate-002  not_stated (0.88)  not_stated (1.00)  not_stated (1.00)  recruiter_review
│ candidate-003  not_stated (0.75)  not_stated (1.00)  evidenced (0.97)   recruiter_review
│ candidate-004  not_stated (0.46)  evidenced (0.99)   evidenced (0.97)   recruiter_review
▣ STEP 3 · what the recruiter does next
→ 'not_stated' = missing from this text. Ask the candidate; do not assume they lack it
═ execute: False · route: recruiter_review · no_ranking: True · no_auto_rejection: True
```

ดู **candidate-004** ให้ดี: *"Reviewed Python pull requests for the analytics team."* (รีวิว pull request ที่เป็น Python ให้ทีม analytics) Jev ตอบว่า `not_stated` แต่ด้วย **confidence 0.46** การรีวิวโค้ด Python นับเป็น "งานพัฒนาด้วย Python ที่ลงมือทำจริง" ไหม? คนที่มีเหตุผลก็เห็นต่างกันได้ confidence ต่ำคือโมเดลกำลังบอกคุณว่า *รายการนี้ต้องให้คนดู* ซึ่งก็คือจุดประสงค์ของเช็กลิสต์นี้พอดี

สังเกตด้วยว่าอะไรที่ **ไม่มี** ในตาราง: ไม่มีผลรวม ไม่มีอันดับ ไม่มี "ผู้สมัครอันดับหนึ่ง" การเอาสามคอลัมน์มารวมกันจะเปลี่ยนเช็กลิสต์หลักฐานให้กลายเป็นการจัดอันดับไปแบบเงียบ ๆ โค้ดจึงไม่ทำแบบนั้นเลย

✓ Checkpoint: คุณอธิบายได้ว่าทำไมแถว Python ของ candidate-004 มี confidence ต่ำ และทำไมตารางถึงไม่มีคอลัมน์ผลรวม

## 3 · เลือก ไม่ใช่แต่ง — แล็บ 02

label อย่าง `evidenced` ยังตรวจสอบย้อนหลังได้ยาก: หลักฐานอยู่ *ตรงไหน*? คุณอาจอยากให้โมเดลแบบสร้างข้อความ (generative model) "ยกหลักฐานมาให้ดู" อย่าทำแบบนั้น เพราะมันอาจแต่งคำพูดที่ฟังดูถูกขึ้นมาเอง

ให้ใช้ `choice` ของ Jev เป็น **ตัวเลือก (selector)** แทน:

1. **โค้ด** แบ่งข้อความเป็นประโยคที่มีหมายเลข: `s1`, `s2`, `s3`
2. **Jev** เลือกว่าประโยคหมายเลขใดเป็นหลักฐานที่ชัดที่สุด หรือเลือก `not_stated`
3. **โค้ด** พิมพ์ประโยคนั้นออกมา **แบบคำต่อคำ** จากข้อความต้นฉบับ

เพราะตัวเลือกคือตัวประโยคเอง Jev จึง *ไม่มีทาง* ส่งคืนถ้อยคำที่ไม่อยู่ในข้อความ

```jev
{
  "state": {"sentences": {
    "s1": "Led a team that wrote BACnet drivers in C++.",
    "s2": "Reviewed Python pull requests for the analytics team.",
    "s3": "Presented at a smart-building meetup."
  }},
  "questions": {
    "integration": {
      "type": "choice",
      "instructions": "Which ONE sentence in `sentences` is the clearest explicit evidence of hands-on REST API, BACnet or Modbus integration work?",
      "criteria": {
        "s1": "Sentence s1: Led a team that wrote BACnet drivers in C++.",
        "s2": "Sentence s2: Reviewed Python pull requests for the analytics team.",
        "s3": "Sentence s3: Presented at a smart-building meetup.",
        "not_stated": "No sentence describes integration work."
      }
    }
  }
}
```

ลองทดลอง: เปลี่ยนคำถามเป็นเรื่องงาน *ระบบอาคาร* มันเลือก `s1` (ไดรเวอร์ BACnet) หรือ `s3` (งานมีตอัปเรื่องอาคารอัจฉริยะ)? แล้ว *คุณ* จะนับข้อไหนเป็นหลักฐานของการลงมือทำจริง?

```bash
.venv/bin/python week24/06_hr_evidence/labs/lab02_sentence_evidence.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
▣ STEP 1 · candidate-001 — 3 sentences
   s1: Implemented Python ETL services for hotel energy meters.
   s2: Integrated Modbus gateways with REST APIs.
   s3: Maintained BMS telemetry pipelines for 40 buildings.
» python       → s1         conf 1.00   “Implemented Python ETL services for hotel energy meters.”
» integration  → s2         conf 1.00   “Integrated Modbus gateways with REST APIs.”
» building     → s3         conf 1.00   “Maintained BMS telemetry pipelines for 40 buildings.”
…
▣ STEP 4 · candidate-004 — 3 sentences
» python       → not_stated conf 0.64   (ask the candidate)
» integration  → s1         conf 0.99   “Led a team that wrote BACnet drivers in C++.”
» building     → s1         conf 1.00   “Led a team that wrote BACnet drivers in C++.”
◆ every quoted sentence above was copied from the original text by code — Jev only chose an id
═ execute: False · route: recruiter_review · no_ranking: True · no_auto_rejection: True
```

ตอนนี้ recruiter เห็น **ถ้อยคำจริงที่ตรวจได้ในไม่กี่วินาที** แพทเทิร์น "เลือก ไม่ใช่แต่ง" นี้เป็นหนึ่งในสิ่งที่มีประโยชน์ที่สุดที่ Jev ทำได้ คุณจะเจออีกครั้งกับ passage ของ RAG (แล็บ 08)

✓ Checkpoint: คุณอธิบายได้ว่าทำไม Choice บนหมายเลขประโยคจึงแต่งคำพูดขึ้นมาไม่ได้ ในขณะที่การสั่งโมเดล generative ให้ "ยกหลักฐานมา" ทำได้

## 4 · ก่อนใช้ข้อมูลผู้สมัครจริง

- ต้องมีการตัดสินใจอย่างชัดเจนเรื่อง **การประมวลผลที่ได้รับอนุมัติ ระยะเวลาเก็บรักษา และการส่งข้อมูลข้ามพรมแดน** ก่อนส่งข้อความของผู้สมัครจริงไปยัง API ที่โฮสต์อยู่ภายนอก การที่ผู้สมัครยินยอมสมัครงาน ไม่ได้แปลว่ายินยอมให้บริการภายนอกทุกรายประมวลผลโดยอัตโนมัติ
- เก็บ **ที่มาของข้อมูล (provenance)** อำนาจในการประมวลผล และกำหนดวันหมดอายุการเก็บรักษาไว้ใน ATS
- **ตรวจสอบ (audit)** rubric: รันคำถามชุดเดียวกันกับข้อความที่จับคู่กันซึ่งต่างกันเฉพาะส่วนที่ไม่เกี่ยวข้อง (สำนวนการเขียน งานอดิเรก การใช้ถ้อยคำที่ไม่ใช่ภาษาอังกฤษ) แล้วตรวจว่า label ไม่ขยับ
- ใช้ **เกณฑ์ที่เกี่ยวกับงานชุดเดียวกันกับทุกคน** และบันทึกเวอร์ชันโมเดลกับเวอร์ชัน rubric ไว้กับทุกเช็กลิสต์

ตัวต้นแบบเต็ม (สามคำถามเดียวกัน policy เดียวกัน) อยู่ในสคริปต์ดั้งเดิม:

```bash
.venv/bin/python week24/jev_lab/jev_lab.py run hr --limit 2 --live
```

✓ Checkpoint: คุณระบุการตรวจสอบสี่ข้อที่ต้องทำก่อนประมวลผลข้อความของผู้สมัครจริงได้

## Labs — รันได้ที่นี่

**labs/lab01_evidence_checklist.py** — โปรไฟล์สมมติสี่รายการ → evidenced / not_stated / unclear ต่อเกณฑ์ → ส่งให้ recruiter รีวิว ไม่มีการจัดอันดับ

**labs/lab02_sentence_evidence.py** — โค้ดใส่หมายเลขประโยค Jev เลือกหมายเลขที่เป็นหลักฐาน โค้ดยกประโยคนั้นมาแบบคำต่อคำ

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 06 — เพิ่มเกณฑ์และ policy ที่เป็นธรรม** เปิดไฟล์ `week24/06_hr_evidence/exercises/ex06_new_criterion.py`

1. **TODO 1** — เขียน `cloud_evidence`: `choice` ที่มีตัวเลือก `evidenced` / `not_stated` / `unclear` พอดี เกี่ยวกับการ deploy บนคลาวด์ที่ลงมือทำจริง (AWS, Azure, GCP…) ระบุใน instruction ว่าความสนใจหรือคอร์สเรียนไม่นับเป็นการลงมือทำจริง
2. **TODO 2** — เขียน `checklist_row(answers)` ที่คืนค่า `{"cloud": …, "follow_up": [criteria not evidenced], "route": "recruiter_review"}` ต้องไม่มีคะแนน อันดับ ผลรวม หรือการคัดออกอยู่ในนั้นเด็ดขาด

ตัวตรวจจะรันการทดสอบรูปแบบและ policy แบบออฟไลน์ก่อน จากนั้นจึงถาม Jev เกี่ยวกับข้อความทดสอบสามข้อความ

```bash
.venv/bin/python week24/06_hr_evidence/exercises/ex06_new_criterion.py
```

**Expected output** (ผลลัพธ์ที่ควรเห็น)

```
✓ cloud_evidence is a choice
✓ labels are exactly evidenced / not_stated / unclear
✓ every record goes to recruiter_review
✓ follow_up lists the criteria that are not evidenced
✓ no score, rank or rejection in the output
▣ STEP 3 · live — does Jev agree with the expected labels?
│ ✓  Deployed Python microservices to AWS ECS with T…  evidenced           evidenced   1.00
│ ✓  Interested in learning Azure. Completed a cloud…  not_stated|unclear  not_stated  1.00
│ ✓  Built Excel reports for the finance team.         not_stated          not_stated  1.00
```

<details><summary>คำใบ้ — ทำไมเรียกว่า "follow_up" ไม่ใช่ "missing"?</summary>

ถ้อยคำกำหนดพฤติกรรม คอลัมน์ที่ชื่อ `missing` หรือ `gaps` ชวนให้ recruiter มองว่าเป็นข้อบกพร่อง ส่วน `follow_up` บอกว่าต้องทำอะไรจริง ๆ: ถามผู้สมัครเรื่องนั้น

</details>

<details><summary>ท้าทายเพิ่ม — หลักฐานเรื่องคลาวด์ระดับประโยค</summary>

คัดลอก `evidence_questions()` จากแล็บ 02 แล้วเพิ่ม `"cloud": "hands-on cloud deployment"` ลงใน `CRITERIA` รันกับข้อความทดสอบสามข้อความ แล้วพิมพ์ประโยคแบบคำต่อคำของแต่ละข้อความ

</details>

✓ Checkpoint: การตรวจแบบออฟไลน์เป็น ✓ ทั้งหมด และตาราง live แสดงแถว ✓ สามแถว (หรือมี ⚠ ที่คุณอธิบายได้)

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| คอร์สเรียนหรือความสนใจถูกตัดสินเป็น `evidenced` | ระบุให้ชัดใน instruction ว่าความสนใจและคอร์สเรียนไม่นับเป็นการลงมือทำจริง Jev อ่านตามตัวอักษร |
| confidence ต่ำในบางแถว | นั่นคือข้อมูลที่มีประโยชน์ มันบอก recruiter ว่ากรณีนี้อยู่ก้ำกึ่ง อย่า "แก้" ด้วยการลด threshold (เกณฑ์) |
| แล็บ 02 เลือกประโยคที่คุณไม่เห็นด้วย | พูดคุยกันแล้วปรับถ้อยคำของเกณฑ์ให้คมขึ้น จดบันทึกไว้ มันจะกลายเป็นตัวอย่างสำหรับการประเมินผล |
| `every row needs candidate_id and professional_text` | มีแถวใน CSV ว่างอยู่ ให้แก้ข้อมูล ห้ามข้ามผู้สมัครไปแบบเงียบ ๆ |
| มีคนขอให้ "เพิ่มคะแนนรวมหน่อย" | อย่าทำ นั่นเปลี่ยนเช็กลิสต์หลักฐานให้เป็นการจัดอันดับอัตโนมัติ ซึ่งเป็นระบบคนละแบบที่มีความเสี่ยงสูงกว่ามาก และต้องผ่านการทบทวนด้านกฎหมายและความเป็นธรรมของตัวเอง |

## Next — ต่อไป

ไปต่อที่ [แล็บ 07 — คัดแยกปัญหา HVAC: โค้ดคำนวณ Jev ตัดสิน](../07_afdd_triage/TUTORIAL.th.md)
