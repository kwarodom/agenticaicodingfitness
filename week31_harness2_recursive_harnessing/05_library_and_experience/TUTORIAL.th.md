# ▶ Harness Lab 05 — The improvement library: EXPERIENCE.md, NO_ENTRY และการตรวจข้อเท็จจริงซ้ำ

> ส่วนหนึ่งของ Week 31 · Harness²: agent ที่ปรับปรุง harness ของตัวเอง คุณจะปรับปรุง harness ของ agent จำลองหนึ่งตัว คือ **HVAC Expert agent ของ Ops Assistant** (ผู้ช่วยด้านปฏิบัติการแบบ always-on ที่สมมติขึ้น) จากหลักฐานการทำงานจริง และเห็นผลลัพธ์จริง ทุกอย่างเป็นข้อมูลสังเคราะห์ (โรงแรมสมมติสองแห่ง alarm และ work order ที่แต่งขึ้น) ทุก lab จึงฟรี ออฟไลน์ และปลอดภัย Lab ทำงานบน **🧠 SIM** เป็นค่าเริ่มต้น คือโมเดลสอนแบบ deterministic ของ agent และ editor ของมัน ถ้า lab ไหนรองรับ ให้สลับ runner เป็น **🦙 LIVE** แล้วโมเดล local จริงจะเล่นบทหนึ่งบท

**สิ่งที่คุณจะได้ลงมือทำ**
- อ่านสัญญาของ reflection: edit row มาก่อน แล้วจึงมีบทเรียนอย่างมากหนึ่งบทที่มี cue, lesson, evidence และ confidence หรือบรรทัดเดียวว่า NO_ENTRY
- สร้าง library จาก task ของ domain หนึ่งตามลำดับ แล้วเปิดมันแบบที่ task ถัดไปเปิด: INDEX.md, EXPERIENCE.md หนึ่งไฟล์ และ tree ทั้งต้นที่ถูกเก็บไว้ ในโหมด LIVE โมเดล local จะเขียน entry สองรายการ และกฎง่าย ๆ จะตรวจมัน
- รัน library ablation บน executor ทั้งสอง และอ่านความต่างเทียบกับส่วนเบี่ยงเบนมาตรฐาน ไม่ใช่เทียบกับความหวัง
- นำ library จากโรงแรมแรกไปใช้ที่โรงแรมที่สอง ทั้งแบบตรวจและไม่ตรวจข้อเท็จจริงที่ติดมาซ้ำ แล้วดูค่า 3.0 K พาโรงแรมที่ใช้ 4.5 K หลงทาง
- ตัดสินว่าข้อเท็จจริงของ workspace ควรอยู่ที่ไหนใน Ops Assistant: ใน knowledge graph ไม่ใช่ `LongTermMEMORY.md`

**Time** ~40 นาที · **Difficulty** intermediate · **Hardware** แล็ปท็อปเครื่องนี้เท่านั้น (Ollama กับ `gemma4:12b` ไม่บังคับ ใช้กับขั้น LIVE ของ Lab 05-1)

**Sources:** [Harness² paper, arXiv 2610.14224](https://arxiv.org/html/2610.14224) · [google-research/harness2](https://github.com/google-research/harness2) (Apache-2.0; snapshot ของ `prompts.py`, `library.py`, `docs/EXPERIMENTS.md` อยู่ใน `common/sources/`) · agent memory และ knowledge graph จาก Week 14–15, self-evolving agent จาก Week 18, eval จาก Week 10

## 0 · ก่อนเริ่ม

| ต้องมี | ตรวจอย่างไร | ทำไม |
|---|---|---|
| Module 04 | คุณบอกได้ว่า `helped` และ `confounded` อ้างอะไร | library เก็บ row เหล่านั้นและสร้างบทเรียนบนมัน |
| Python ของ repo นี้ | `.venv/bin/python --version` → 3.13 | ใช้รันทุก lab |
| พื้นที่ดิสก์ | ไม่กี่ร้อย KB ใต้ `05_library_and_experience/.runs/` | lab เขียนโฟลเดอร์ library จริงที่คุณเปิดดูได้ (gitignored) |
| Ollama (ไม่บังคับ) | `ollama list` มี `gemma4:12b` | ใช้เฉพาะขั้น LIVE ของ Lab 05-1 ทุก lab จบได้โดยไม่ต้องมี |

Module 02 รัน step หนึ่งครั้งแล้วทิ้งหลักฐานไป Harness² เก็บมันไว้ หลังแต่ละ task ขั้น reflection เขียนว่าแต่ละ edit ทำอะไร และเขียนบทเรียนอย่างมากหนึ่งบท task ถัดไปใน domain เดียวกันจะอ่าน library นี้ก่อนจะแก้อะไร self-evolving agent ใน Week 18 ก็เก็บความจำของรอบที่ผ่านมาเหมือนกัน ความต่างคือที่นี่ทุก entry คือ harness tree ทั้งต้นพร้อมหลักฐาน และกฎการเขียน entry เข้มงวด

✓ Checkpoint: คุณบอกได้ว่า library เก็บอะไรต่อ task (tree ที่ถูก adopt ทั้งต้นกับ EXPERIENCE.md) และ task ถัดไปอ่านมันเมื่อไร (ก่อนที่ editor ของมันจะเริ่ม)

## 1 · สัญญาของ reflection: row มาก่อน แล้วบทเรียนอย่างมากหนึ่งบท

reflection เขียนบันทึกสองอย่าง **edit row** (Module 04) บอกว่าแต่ละการเปลี่ยนแปลงทำอะไร จากนั้นเป็นส่วนที่ task ถัดไปได้เรียนรู้จริง:

**Expected output** (REFERENCE — quoted from `harness2/core/prompts.py`, the reflection prompt)

```
"Now write at most one durable entry, and only if the edit rows support it.\n\n"
"The `cue` describes the SHAPE OF SITUATION the lesson applies to, so it can fire on a "
"different matter. Not this task's facts -- the pattern. The `lesson` is the operation "
"to perform on a harness, stated so it can be applied and checked, and it must name "
"the component it concerns.\n\n"
"Too generic is useless ('be thorough'); too specific never fires again ('the fee rose "
"to 250 on Jan 1'). Aim between: 'when an instrument performs several sequential acts "
"and one of them terminates the actor's own authority, order them so the terminating "
"act comes last'.\n\n"
"The `evidence` field cites the edit rows the lesson rests on. A lesson resting only "
"on `confounded` rows is a guess: narrow it to what the record does show, or decline. "
"Adoption alone never establishes that an edit helped or will transfer. If the record "
"shows no concrete, reusable execution contrast, write the single line NO_ENTRY in "
"place of the <experience> block -- the edit rows are still worth writing.",
```

entry หนึ่งรายการจึงมีสี่ field:

| field | เก็บอะไร | ผิดเมื่อ |
|---|---|---|
| `cue` | รูปแบบของสถานการณ์ที่บทเรียนใช้ได้ | มันใส่ ID หรือตัวเลขของ task นี้ หรือบอกแค่ว่า "ระวังหน่อย" |
| `lesson` | การกระทำกับ harness ที่ระบุ component | มันให้คำแนะนำกับ agent แทนที่จะเป็นการเปลี่ยนไฟล์ |
| `evidence` | edit row และ contrast จากการรัน | มันอ้างว่า "version นี้ถูก adopt แล้ว" |
| `confidence` | high, medium หรือ low พร้อมเหตุผล | มี task เดียวรองรับแต่เขียนว่า "high" |

มีสองบรรทัดที่ควรอ่านซ้ำ "Adoption alone never establishes that an edit helped or will transfer" (การถูก adopt อย่างเดียวไม่ได้พิสูจน์ว่า edit ช่วยหรือจะถ่ายทอดได้) เพราะ selector เลือก deliverable ไม่ได้เลือก edit และ NO_ENTRY คือคำตอบจริง เหมือน KEEP_BASE ใน Module 02

หมายเหตุเรื่อง kit: `h2kit.experience()` เขียน confidence เป็น **medium** เสมอ เพราะทุก entry อิงกับ task เดียว สัญญาอนุญาตให้มี high, medium และ low แต่ kit ไม่เคยใช้อีกสองแบบ ดังนั้นอย่าตีความอะไรจากการที่ทุก entry ที่คุณเห็นเขียนว่า "medium"

✓ Checkpoint: คุณบอกชื่อ field ทั้งสี่ได้ และบอกได้ว่าเมื่อไรที่ entry ที่ซื่อตรงคือบรรทัดเดียวว่า NO_ENTRY

## 2 · Library บน stream เดียว: task ตามลำดับ

Lab 05-1 รัน step หนึ่งครั้งต่อ task ประเภท fault review ของครึ่ง refine ตามลำดับ โดยมี library เดียวสำหรับ domain `hvac_expert/fault_review/hotel_th/sim-cloud` (agent × task family × tenant class × executor) บันทึกการใช้งานของ repository กำหนดกฎเรื่องลำดับไว้:

> “Tasks within a domain run in order and read only earlier entries; different domains can run concurrently.” — `harness2/docs/EXPERIMENTS.md`

(task ใน domain เดียวกันรันตามลำดับและอ่านเฉพาะ entry ก่อนหน้า ส่วน domain ต่างกันรันพร้อมกันได้)

**Expected output** (captured on this Mac, SIM)

```
│ task   entries read  adapted from the library           step       EXPERIENCE  confidence                           
│ ─────  ────────────  ─────────────────────────────────  ─────────  ──────────  ─────────────────────────────────────
│ F1-00  0             —                                  KEEP_BASE  NO_ENTRY    —                                    
│ F1-01  1             —                                  COMPOSED   entry       medium (one task, several rows agree)
│ F1-02  2             no_analytic_move                   COMPOSED   entry       medium (one task, several rows agree)
│ F1-03  3             —                                  KEEP_BASE  NO_ENTRY    —                                    
│ F1-04  4             hand_arithmetic, no_analytic_move  COMPOSED   entry       medium (one task, several rows agree)
│ F1-05  5             hand_arithmetic, no_analytic_move  COMPOSED   entry       medium (one task, several rows agree)
│ F1-10  6             —                                  KEEP_BASE  NO_ENTRY    —                                    
│ F1-11  7             —                                  KEEP_BASE  NO_ENTRY    —                                    
◆ a quiet trend check (F1-10, F1-11) has nothing to fix: KEEP_BASE and NO_ENTRY — the edit rows are still stored
⚠ the kit always writes confidence 'medium' (one task). The contract allows high | medium | low; the kit never exercises the other two
```

ดูคอลัมน์ที่สามที่ค่อย ๆ โตขึ้น F1-01 เรียนรู้ skill สำหรับ fault review F1-02 ดัดแปลงมันจาก library แทนที่จะเขียนใหม่ พอถึง F1-04 editor ดัดแปลง component ที่พิสูจน์แล้วสองตัว ครึ่งหนึ่งของ entry เป็น NO_ENTRY และไม่เป็นไร เพราะ trend check ที่เงียบไม่มีอะไรให้สอน

จากนั้น library ถูกเปิดดูผ่าน index ซึ่งมีหนึ่งบรรทัดต่อ task:

**Expected output** (captured on this Mac, SIM)

```
┌─ lab05_1_library/hvac_expert__fault_review__hotel_th__sim-cloud/INDEX.md
│ # Library hvac_expert/fault_review/hotel_th/sim-cloud
│ 
│ - F1-00: NO_ENTRY (edit rows only)
│ - F1-01: a fault review where the practitioner's comparison is skipped
│ - F1-02: a deliverable that needs repeated unit arithmetic over telemetry rows
│ - F1-03: NO_ENTRY (edit rows only)
│ - F1-04: a deliverable that needs repeated unit arithmetic over telemetry rows
│ - F1-05: a deliverable that needs repeated unit arithmetic over telemetry rows
│ - F1-10: NO_ENTRY (edit rows only)
│ - F1-11: NO_ENTRY (edit rows only)
└─
```

แต่ละบรรทัดแสดง **cue** ของ entry ไม่ใช่ตัว task นี่คือหัวใจของ cue: task ต่อไปที่มี chiller ต่างกัน โรงแรมต่างกัน หรือ alarm ต่างกัน ก็ยังจำสถานการณ์ได้

✓ Checkpoint: คุณอธิบายได้ว่าทำไม F1-02 แสดง "adapted from the library" แต่ F1-01 ไม่แสดง

## 3 · ข้างใน entry หนึ่งรายการ: EXPERIENCE.md และ tree ทั้งต้น

Lab 05-1 เปิด entry ของ F1-01 แล้วเปิด entry ที่เป็น NO_ENTRY:

**Expected output** (captured on this Mac, SIM)

```
┌─ harnesses/F1-01/EXPERIENCE.md
│ <edits>
│ component: tool_guidance.md
│ candidates: inv1
│ change: proposed equipment work without calling cmms.work_orders first → tool_guidance.md
│ observed: no_cmms: mitigated
│ verdict: helped
│ 
│ component: skills/chiller_fault_review/SKILL.md, systemprompt.md
│ candidates: dom1
│ change: diagnosed without comparing the condenser approach with the manual's limit → skills/chiller_fault_review/SKILL.md, systemprompt.md
│ observed: no_analytic_move: mitigated
│ verdict: helped
│ 
│ </edits>
│ 
│ <experience>
│ cue: a fault review where the practitioner's comparison is skipped
│ lesson: add a skill naming the analytic move, the artifact contract and the maintenance check
│ evidence: edit row 'skills/chiller_fault_review/SKILL.md, systemprompt.md' (dom1): no_analytic_move: mitigated; reviewer's reason (content): "CH-2 was already on a PM; the agent did not check open work orders."
│ confidence: medium (one task, several rows agree)
│ </experience>
└─
┌─ harnesses/F1-00/EXPERIENCE.md (no reusable contrast)
│ <edits>
│ </edits>
│ 
│ NO_ENTRY
└─
```

ตรวจ entry นี้เทียบกับสัญญา cue เป็นรูปแบบ ("the practitioner's comparison is skipped") ไม่มีรหัส chiller และไม่มีตัวเลข lesson ระบุ component (skill) และบอกว่าใส่อะไรลงไป evidence อ้าง row คำพูดของผู้ตรวจถูกเก็บไว้และติดป้ายว่าเป็น content จึงให้ข้อมูลได้แต่ไม่ได้สั่งการ (Module 04)

โฟลเดอร์ของ entry เก็บมากกว่าบทเรียน library ของ repository เก็บ tree ทั้งต้น:

> “The whole tree is carried, never a summary.” — `harness2/core/library.py`

(tree ทั้งต้นถูกพกไป ไม่ใช่บทสรุป)

**Expected output** (captured on this Mac, SIM)

```
│ file in harnesses/F1-01/harness/      comp.  vs base       ≈tokens
│ ────────────────────────────────────  ─────  ────────────  ───────
│ LongTermMEMORY.md                     C3     same as base  36     
│ guardrails.md                         C2     same as base  80     
│ skills/chiller_fault_review/SKILL.md  C6     added         209    
│ systemprompt.md                       C1     changed       86     
│ tool_guidance.md                      C4     changed       86     
```

ทำไมต้องทั้งต้น บทสรุปทำสิ่งที่ทำให้ edit ใช้ได้หายไป เช่น trigger line ที่บอกชื่อ skill, script ที่ skill เรียก, บรรทัด guidance ข้าง ๆ editor ทีหลังจึงคัดลอก component ไปพร้อมทุกอย่างที่มันต้องใช้ได้ และ validator จาก Module 02 จะทิ้ง skill ที่ลืมเอา script ของมันมาด้วย

**LIVE: โมเดล local เขียน entry ได้ไหม** ในโหมด 🦙 LIVE ขั้นที่ 6 ของ Lab 05-1 ให้โมเดล local ได้สัญญาและ edit row ของสอง record จริง: F1-01 (row แบบ `helped` สองแถว) และ F3-00 (row แบบ `confounded` หนึ่งแถว จาก step ที่ probe สามครั้ง) กฎง่าย ๆ แบบเดียวกับของ Exercise 05 ตรวจสิ่งที่ได้กลับมา (2 calls)

**Expected output** (captured on this Mac, LIVE gemma4:12b — a rerun can differ)

```
▣ STEP 6 · LIVE option: does a local model write a lesson the contract accepts — and decline when it should?
── F1-01 (two helped rows) · gemma4:12b · 6.0s → entry
│ <experience>
│ cue: performing equipment work without checking the CMMS system first
│ lesson: tool_guidance.md
│ evidence: proposed equipment work without calling cmms.work_orders first; observed: no_cmms: mitigated
│ confidence: high
│ </experience>
✕ F1-01 (two helped rows): expected entry; rule problems: lesson states no operation (add/move/split…), 'high' on one task (one run — a rerun can differ)
── F3-00 (one confounded row) · gemma4:12b · 2.6s → NO_ENTRY
│ NO_ENTRY
✓ F3-00 (one confounded row): NO_ENTRY, as the record supports
```

ในรอบนี้ `gemma4:12b` ปฏิเสธได้ถูกต้องสำหรับ record ที่ confounded สำหรับ F1-01 cue ของมันเป็นรูปแบบที่พอใช้ได้ แต่ "lesson" เป็นแค่ชื่อไฟล์ และมันอ้าง confidence สูงจาก task เดียว entry แบบนั้นจะบอก editor คนต่อไปว่าดูที่ไหน แต่ไม่บอกว่าต้องทำอะไร และมั่นใจเกินกว่าที่ record รองรับ นี่เป็นเหตุผลหนึ่งที่ EXAMPLE design ของคอร์สสำหรับ Ops Assistant ให้มีผู้ตรวจที่เป็นมนุษย์ระบุชื่ออยู่หน้า harness ทุก version (Module 08)

✓ Checkpoint: คุณบอกได้ว่าทำไมบทสรุปที่เก็บไว้ ("added a fault-review skill") มีประโยชน์กับ task ถัดไปน้อยกว่า tree ที่เก็บไว้ และบอกชื่อกฎของสัญญาสองข้อที่ entry F1-01 แบบ LIVE ทำผิด

## 4 · การอ่าน library: INDEX → entry → ดัดแปลง

editor ทีหลังไม่ได้อ่านทุกไฟล์ prompt ของ editor ใน repository บอกวิธีเปิดดูไว้:

> “Browse `INDEX.md` first, then open the ones that look relevant: reusing or adapting a proven component is encouraged when it genuinely fits, but never presume this task's question.” — `harness2/core/prompts.py`

(เปิด `INDEX.md` ก่อน แล้วเปิดเฉพาะอันที่ดูเกี่ยวข้อง การใช้ซ้ำหรือดัดแปลง component ที่พิสูจน์แล้วเป็นเรื่องดีเมื่อมันเข้ากันจริง แต่อย่าเดาคำถามของ task นี้ล่วงหน้า)

`Library.relevant(task, tree)` คือเวอร์ชันของ kit มันอ่าน INDEX.md ข้ามบรรทัด NO_ENTRY เปิด entry ที่ cue ตรงกับชนิดของ failure แล้วส่ง component ที่ entry เหล่านั้นเพิ่มไว้ให้ editor:

**Expected output** (captured on this Mac, SIM)

```
│ hand_arithmetic    → scripts/energy_units.py, skills/chiller_fault_review/SKILL.md, systemprompt.md, tool_guidance.md
│ no_analytic_move   → skills/chiller_fault_review/SKILL.md, systemprompt.md, tool_guidance.md
│ workspace fact inside a copied skill: - In this workspace, the condenser approach limit is 3.0 K (from the O&M manual excerpt).
✓ 2 failure kinds have a proven component waiting — the editor adapts it instead of writing from scratch
```

ดูบรรทัดสุดท้าย skill ที่ถูกคัดลอกพก **ข้อเท็จจริงของ workspace** มาด้วย คือขีดจำกัด condenser approach ของ Harbour View ที่ 3.0 K repository ขอให้ editor เขียนข้อเท็จจริงแบบนี้ลงใน harness โดยขึ้นต้นด้วย "in this workspace":

> “VERIFIED FACTS ABOUT THIS WORKSPACE ARE THE MOST VALUABLE THING YOU CAN ENCODE” — `harness2/core/prompts.py`

(ข้อเท็จจริงที่ตรวจแล้วเกี่ยวกับ workspace นี้คือสิ่งที่มีค่าที่สุดที่คุณ encode ได้)

ซึ่งถูกต้องสำหรับโรงแรมที่ตรวจข้อเท็จจริงนั้นมา Section 6 จะแสดงว่ามันทำอะไรกับโรงแรมถัดไป

✓ Checkpoint: คุณบอกได้ว่าไฟล์ไหนของ entry F1-02 ที่พกข้อเท็จจริง 3.0 K ทั้งที่ cue ของ F1-02 เองพูดถึงการคำนวณหน่วย

## 5 · หนึ่ง library ต่อ domain และต่อ executor และ ablation

Lab 05-2 รัน stream หนึ่งครั้งโดยเปิด library สำหรับแต่ละ executor แล้วแสดงรายการ library ที่มีอยู่หลังจากนั้น:

**Expected output** (captured on this Mac, SIM)

```
│ library (agent/family/tenant class/executor)  tasks stored  with a lesson
│ ────────────────────────────────────────────  ────────────  ─────────────
│ hvac_expert/digest/hotel_th/sim-cloud         3             3            
│ hvac_expert/digest/hotel_th/sim-local         3             3            
│ hvac_expert/fault_review/hotel_th/sim-cloud   8             4            
│ hvac_expert/fault_review/hotel_th/sim-local   8             4            
│ hvac_expert/helper/hotel_th/sim-cloud         2             1            
│ hvac_expert/helper/hotel_th/sim-local         2             0            
│ hvac_expert/triage/hotel_th/sim-cloud         4             2            
│ hvac_expert/triage/hotel_th/sim-local         4             1            
│ EXPERIMENTS.md: “Libraries are separated by benchmark, executor, mode, budget, and seed.”
```

library แปดชุดสำหรับ executor สองตัวและ task family สี่กลุ่ม task ของ `sim-local` ไม่เคยอ่านบทเรียนที่ `sim-cloud` เรียนมา บทเรียนคือข้อกล่าวอ้างว่าอะไรใช้ได้กับโมเดลหนึ่ง และ paper พบว่าการถ่ายทอดข้ามโมเดลมีจริงแต่ไม่สม่ำเสมอ (Module 06 วัดเรื่องนี้)

library ช่วยไหม ablation รันสาม stream แต่ละ stream สลับลำดับ task ของตัวเอง ทั้งแบบเปิดและปิด library:

**Expected output** (captured on this Mac, SIM)

```
│ executor   library  stream means (s0 / s1 / s2)  mean ± SD                       
│ ─────────  ───────  ───────────────────────────  ──────────  ────────────────────
│ sim-cloud  on       96.5 / 95.0 / 91.5           94.3 ± 2.6  ███████████████████░
│ sim-cloud  off      97.6 / 93.5 / 90.3           93.8 ± 3.7  ███████████████████░
│ sim-local  on       84.4 / 83.3 / 83.6           83.8 ± 0.6  █████████████████░░░
│ sim-local  off      84.4 / 82.2 / 85.1           83.9 ± 1.5  █████████████████░░░

▣ STEP 3 · Read it honestly: the difference, the SD, and whether all three streams agree
│ facilitator guide: “SDs are 1–3 points: believe differences under ~3 only if all three streams agree.”
│ executor   on − off  SD on / off  per stream (same order)  all agree  size         
│ ─────────  ────────  ───────────  ───────────────────────  ─────────  ─────────────
│ sim-cloud  +0.5      2.6 / 3.7    -1.1 / +1.5 / +1.2       no         within one SD
│ sim-local  -0.1      0.6 / 1.5    +0.0 / +1.1 / -1.5       no         within one SD
⚠ sim-cloud: the streams disagree in sign (-1.1 / +1.5 / +1.2) — no library effect is shown on 17 tasks
⚠ sim-local: the streams disagree in sign (+0.0 / +1.1 / -1.5) — no library effect is shown on 17 tasks
◆ neither difference is a result you could act on. Module 06 reads tables like this; Module 09 pre-registers the test.
```

การอ่านอย่างซื่อตรง: ไม่มี executor ไหนแสดงผลของ library บน `sim-cloud` ช่องว่าง (+0.5) อยู่ในหนึ่ง SD และ stream ต่างกันในเครื่องหมาย บน `sim-local` สองแขนเสมอกัน (−0.1): stream หนึ่งเท่ากัน หนึ่งเอียงไปทาง library และอีกหนึ่งเอียงไปทางไม่ใช้ library seed ของ stream เดียวกันหมายถึงลำดับ task เดียวกัน คอลัมน์ต่อ stream จึงเป็นการเทียบแบบจับคู่ และที่นี่มันไม่ได้ชี้ไปทางไหนเลย เทียบกับ paper: ถ้าไม่มี library วิธีนี้เสีย 3.1 คะแนนบน WorkBuddy-Bench และ 1.8 บน JobBench และไฟล์ที่ถูกแก้ 22% ถึง 47% มีที่มาจาก library SIM แสดงว่า library ถูกใช้อย่างไร แต่ไม่ได้ทำซ้ำตัวเลขเหล่านั้น และไม่ควรอ่านราวกับว่ามันทำซ้ำได้

✓ Checkpoint: คุณบอกได้สำหรับแต่ละ executor ว่าตารางนี้แสดงผลของ library หรือไม่ โดยใช้ SD และคอลัมน์ต่อ stream

## 6 · โรงแรมที่สอง: ตรวจซ้ำสิ่งที่คุณคัดลอกมา

tenant class `hotel_th` มีโรงแรมสมมติแห่งที่สองคือ Chelsea Riverside Hotel คู่มือ O&M ของที่นี่กำหนดขีดจำกัด condenser approach ไว้ที่ 4.5 K ไม่ใช่ 3.0 K Lab 05-3 สร้าง library ของ fault review ที่โรงแรมแรก แล้วรัน fault review หกงานของโรงแรมที่สองสองรอบจากสำเนาของมัน: รอบหนึ่งตรวจข้อเท็จจริงที่คัดลอกมาเทียบกับคู่มือของ task เอง (`reverify=True`) อีกรอบคัดลอกโดยไม่ตรวจ (`reverify=False`)

กฎของ repository สำหรับการคัดลอกจาก library:

> “If you want a component from one, COPY IT into `context/harness/` -- and when you do, RE-VERIFY its workspace facts against the evidence in front of you: an earlier task's mapping or threshold is a hypothesis here, not a fact.” — `harness2/core/prompts.py`

(ถ้าต้องการ component จาก entry ให้คัดลอกมา และเมื่อคัดลอกแล้วต้องตรวจข้อเท็จจริงของ workspace ซ้ำกับหลักฐานตรงหน้า การจับคู่หรือ threshold ของ task ก่อนหน้าเป็นแค่สมมติฐานที่นี่ ไม่ใช่ข้อเท็จจริง)

**Expected output** (captured on this Mac, SIM)

```
│ task     approach K  truth*             re-verified: fact  diagnosis          score*  unchecked: fact  diagnosis          score*
│ ───────  ──────────  ─────────────────  ─────────────────  ─────────────────  ──────  ───────────────  ─────────────────  ──────
│ F1-CH40  3.4         low load cycling   4.5 K              low load cycling   100.0   3.0 K            condenser fouling  80.0  
│ F1-CH41  3.9         within design      —                  within design      100.0   —                within design      100.0 
│ F1-CH42  4.2         low load cycling   4.5 K              low load cycling   100.0   3.0 K            condenser fouling  75.0  
│ F1-CH43  5.9         condenser fouling  4.5 K              condenser fouling  100.0   3.0 K            condenser fouling  100.0 
│ F1-CH44  3.6         low load cycling   —                  low load cycling   100.0   —                low load cycling   100.0 
│ F1-CH45  3.8         within design      —                  within design      100.0   —                within design      100.0 
◆ * truth and scores measured AFTER each step by bench.judge — no editor saw them
═ seed 0 · re-verified mean 100.0 · unchecked mean 92.5

▣ STEP 3 · What the editor copied, and what re-verification changed
│ copied from the Harbour View entry : - In this workspace, the condenser approach limit is 3.0 K (from the O&M manual excerpt).
│ after reverify_facts on F1-CH40    : - In this workspace, the condenser approach limit is 4.5 K (from the O&M manual excerpt).
│ F1-CH40's own manual excerpt       : Chelsea Riverside Hotel (fictional) O&M: condenser approach above 4.5 K indicates fouling; design efficiency 0.68 kW/ton.
```

F1-CH40 และ F1-CH42 อยู่ระหว่างขีดจำกัดทั้งสอง ถ้าใช้ 3.0 K ของโรงแรมแรก มันดูเหมือนมีตะกรัน ถ้าใช้ 4.5 K ของโรงแรมนี้ มันปกติ F1-CH43 ที่ 5.9 K มีตะกรันไม่ว่าใช้ขีดจำกัดไหน ข้อเท็จจริงที่ผิดจึงไม่ทำความเสียหายที่มองเห็นตรงนั้น นี่คือ premise failure ของ paper ในขนาดย่อ: การวิเคราะห์ความล้มเหลวของ paper รายงานว่าสมมติฐานตั้งต้นที่ผิดแพร่ข้าม candidate ได้

สังเกตว่าทำไมไม่มี contrast ไหนจับได้ skill ทำงานและ agent ทำการเปรียบเทียบที่ถูกต้อง trajectory จึงดูปกติ มีแค่ตัวเลขที่ผิด และมีแค่ judge ซึ่งอ่าน truth ที่ซ่อนไว้หลัง step เท่านั้นที่เห็น

seed เดียวคือเรื่องเดียว Lab 05-3 ทำการเปรียบเทียบซ้ำบน seed 0–4:

**Expected output** (captured on this Mac, SIM)

```
│ seed  re-verified  unchecked  difference  unchecked runs that used 3.0 K  tasks worse unchecked
│ ────  ───────────  ─────────  ──────────  ──────────────────────────────  ─────────────────────
│ 0     100.0        92.5       -7.5        F1-CH40, F1-CH42, F1-CH43       F1-CH40, F1-CH42     
│ 1     74.5         74.5       +0.0        —                               —                    
│ 2     95.8         95.8       +0.0        —                               —                    
│ 3     95.8         95.8       +0.0        F1-CH42                         —                    
│ 4     91.7         88.3       -3.4        F1-CH40                         F1-CH40              
◆ every seed's first-hotel library carries the 3.0 K fact inside a copied skill. It reaches a run only when the copied skill survives composition (a domain editor that writes its own skill from the task's manual overwrites it) and then fires. Seed 3: F1-CH42 used 3.0 K yet scored the same in both arms: the re-verified run used 4.5 K, but 4.2 K sits within 1 K of that limit, where the SIM makes the call uncertain, and it missed too
✓ every task the unchecked arm lost had used the 3.0 K fact; re-verification never scored below copying here
```

ผลนี้ปรากฏสองใน seed ห้าตัว และเฉพาะที่ข้อเท็จจริงเก่าไปถึงการรันจริง นี่คือรูปร่างของ premise failure: เงียบภายใน step และมองเห็นได้เฉพาะตอนที่ข้อเท็จจริงเก่าไปตกที่ task ที่มันทำให้ผิด

✓ Checkpoint: คุณอธิบายได้ว่าทำไม F1-CH40 เปลี่ยนการวินิจฉัยระหว่างสองรอบ แต่ F1-CH43 ไม่เปลี่ยน

## 7 · ข้อเท็จจริงของ workspace ควรอยู่ที่ไหนใน Ops Assistant

skill ของ kit เก็บ threshold ไว้ใน skill ตามที่ prompt ของ repository แนะนำสำหรับ workspace เดียว prompt เดียวกันเตือนไม่ให้เก็บไว้ในส่วน always-on:

> “Never store this matter's facts as durable memory: they are wrong anchors for the next task.” — `harness2/core/prompts.py`

(อย่าเก็บข้อเท็จจริงของงานนี้เป็นความจำถาวร เพราะมันจะเป็นหลักยึดที่ผิดสำหรับ task ถัดไป)

EXAMPLE design ของคอร์สให้ Ops Assistant มีที่ที่ดีกว่าทั้งสองแบบ คือ knowledge graph ที่มี Brick identity, provenance และ supersession แบบเดียวกับ graph memory ที่คุณสร้างใน Week 14–15 กฎของการออกแบบนี้คือ ข้อเท็จจริงของ tenant และอาคารที่พบระหว่าง probe (เช่น การจับคู่ point ค่าบน nameplate หรือตารางเวลา) ต้องเข้า graph ผ่านเส้นทาง supersession ที่ผ่านการตรวจ ไม่เข้า `LongTermMEMORY.md`

| อะไร | ที่อยู่ใน EXAMPLE design | ทำไม |
|---|---|---|
| ขั้นตอน: เทียบ approach กับขีดจำกัดในคู่มือ | skill (C6) | ใช้ได้กับทุกโรงแรมใน tenant class |
| ข้อเท็จจริง: "Chelsea approach limit 4.5 K" | knowledge graph | มี provenance และ supersession อ่านตอนรันตามแต่ละอาคาร |
| ข้อเท็จจริงสั้น ๆ ที่จริงกับทุก task | `LongTermMEMORY.md` (C3) แบบมีเพดาน | จ่ายทุกครั้งที่เรียก จึงใส่ได้ก็ต่อเมื่อตัดอย่างอื่นออก |

เมื่อ threshold อยู่ใน graph skill ที่คัดลอกไปจะพกแค่ขั้นตอน และโรงแรมที่สองอ่านขีดจำกัดของตัวเอง การออกแบบเพิ่ม invariant ที่ทำให้เรื่องนี้ปลอดภัย: ไฟล์ของ harness ให้ข้อมูล ไม่เคยให้อำนาจ (a harness file informs, it never authorises)

✓ Checkpoint: คุณบอกได้ว่าแต่ละแถวในสามแถวข้างบนอยู่ที่ไหน และจะเกิดอะไรผิดที่โรงแรมที่สองถ้าข้อเท็จจริงอยู่ใน skill ที่ใช้ร่วมกันหรือในความจำ

## Labs — รันได้ที่นี่

**labs/lab05_1_build_a_library.py** — library ของ fault review หนึ่งชุดที่สร้างตามลำดับ task: INDEX.md, EXPERIENCE.md, tree ทั้งต้นที่ถูกเก็บ, สิ่งที่ Library.relevant ส่งให้ editor คนถัดไป และโมเดล LIVE เขียน entry สองรายการ

**labs/lab05_2_library_ablation.py** — หนึ่ง library ต่อ domain และ executor แล้วตามด้วย ablation แบบเปิดและปิด library บนสาม stream ที่สลับลำดับ อ่านเทียบกับ SD

**labs/lab05_3_second_hotel.py** — library ของโรงแรมแรกที่โรงแรมที่สอง ทั้งแบบตรวจและไม่ตรวจข้อเท็จจริงที่คัดลอกมา, การไล่ seed และที่อยู่ที่ถูกของข้อเท็จจริง

## ลองทำเอง

`exercises/ex05_write_the_lesson.py` มี TODO สี่ข้อ:

1. ติดป้าย cue สามแบบ: กว้างเกินไป, เฉพาะเกินไป หรือพอดี
2. เขียน EXPERIENCE entry สำหรับ edit row จริง (F7-00): cue, lesson, evidence และ confidence
3. ตัดสินว่า record ที่มีแค่ row แบบ `confounded` ควรมี entry หรือไม่
4. เลือกที่อยู่ของขีดจำกัด 4.5 K ของโรงแรมที่สอง

```bash
# on: laptop
.venv/bin/python week31_harness2_recursive_harnessing/05_library_and_experience/exercises/ex05_write_the_lesson.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ cues: a too generic (changes no behaviour) · b too specific (never fires again) · c just right
✓ entry A: cue 'a read over a long time window that times out and is retried unchanged' · lesson names a component · evidence cites the row · confidence 'medium — one task, three probes agree, no second task yet'
✓ record B: NO_ENTRY — its only row is confounded (2 of 3 repeats); a lesson on it would be a guess
✓ the 4.5 K limit is a workspace fact: the knowledge graph, with provenance and supersession (EXAMPLE design for this course)

═ Done. Module 06 reads the results table these libraries feed: streams, noise, ablations, transfer, pass^k.
```

checker ใช้กฎง่าย ๆ (cue ไม่มี ID หรือตัวเลข, lesson มี component และคำกริยา, evidence อ้าง row, ไม่เขียน "high" จาก task เดียว, ไม่มีภาษาของ grader) ถ้อยคำของคุณจะต่างจากของ solution และยังต้องมีคนอ่านว่ามันอ่านรู้เรื่องดีไหม

<details><summary>Hint — TODO 2</summary>

cue อธิบายสถานการณ์ ไม่ใช่ task: "a long-window query that times out" ใช้ได้กับ helper ไหนก็ได้ point ไหนก็ได้ โรงแรมไหนก็ได้ ส่วน "The 30-day AHU-B1 pull" ใช้ได้แค่ครั้งเดียว lesson ต้องเปลี่ยนไฟล์: บอกว่าไฟล์ไหน (`tool_guidance.md`) และใส่อะไร (fallback ที่มีขอบเขตหนึ่งครั้งพร้อมเงื่อนไขหยุด) evidence ระบุ row และ contrast ของมัน (`blind_retry: mitigated` ในสาม probe) มี task เดียวรองรับ confidence จึงไม่เกิน medium

</details>

<details><summary>Hint — TODO 3</summary>

อ่าน verdict ไม่ใช่ observation probe สองในสามครั้ง mitigated storm ได้ row จึงเป็น `confounded` และบทเรียนที่อิงแค่ row แบบ confounded คือการเดา prompt อนุญาตให้แคบลงเหลือเท่าที่ record แสดง แต่ record นี้ไม่แสดง contrast ที่ใช้ซ้ำได้เลย

</details>

✓ Checkpoint: บรรทัดของ checker ทั้งสี่เป็น ✓

## การแก้ปัญหา

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| INDEX.md มี entry มากกว่าใน tutorial | ใช้โฟลเดอร์ library จากการรันเก่าซ้ำ | lab ลบโฟลเดอร์ `.runs/lab05_*` ของตัวเองตอนเริ่ม ถ้าคุณเรียก `bench.stream(root=…)` เอง ให้เริ่มจาก root ที่ว่าง |
| Lab 05-3 ไม่เห็นความต่างที่ seed 0 | kit ถูกแก้ หรือ library ต่างกัน | `git diff week31_harness2_recursive_harnessing/common` ผลนี้ขึ้นกับว่า skill ที่คัดลอกมาพกข้อเท็จจริง 3.0 K หรือไม่ |
| ขั้นที่ 6 ของ Lab 05-1 บอกว่า "no model call" | lab รันใน 🧠 SIM | สลับ runner เป็น 🦙 LIVE หรือรันด้วย `H2_BRAIN=ollama H2_MODEL=gemma4:12b` |
| ทุก entry เขียน confidence ว่า "medium" | kit เขียนแค่ medium (หนึ่ง task ต่อ entry) | เป็นเรื่องปกติ high และ low ในสัญญาเป็นสิ่งที่คุณใช้เองด้วยมือ |
| ตัวเลข ablation ต่างจากตารางที่อื่นในคอร์สเล็กน้อย | ชุด task, seed หรือ editor ต่างกัน | เทียบสิ่งที่เหมือนกัน: task, executor, editor และ seed ของสาม stream เดียวกัน |

## ถัดไป

[Lab 06 — Reading a results table: streams, noise, ablations, transfer, pass^k](../06_reading_the_results/TUTORIAL.md): อ่านตาราง base เทียบกับ Harness² บนสาม stream ที่สลับลำดับ ชุด ablation ทั้งหมด การถ่ายทอดระหว่าง executor และทำไม pass^k คือตัวเลขที่ลูกค้ารู้สึกได้
