# ▶ Harness Lab 03 — สอง scope และ component ที่ถูกที่สุดซึ่งได้ผล

> ส่วนหนึ่งของ Week 31 · Harness²: agent ที่ปรับปรุง harness ของตัวเอง คุณจะปรับปรุง harness ของ agent จำลองหนึ่งตัว คือ **HVAC Expert agent ของ Ops Assistant** (ผู้ช่วยงานปฏิบัติการแบบ always-on ที่สมมติขึ้น) จากหลักฐานการทำงานจริง และเห็นผลลัพธ์จริง ทุกอย่างเป็นข้อมูลสังเคราะห์ (โรงแรมสมมติสองแห่ง alarm และ work order ที่แต่งขึ้น) ทุก lab จึงฟรี ออฟไลน์ และปลอดภัย Lab ทำงานบน **🧠 SIM** เป็นค่าเริ่มต้น คือโมเดลสอนแบบ deterministic ของ agent และ editor ของมัน ถ้า lab ไหนรองรับ ให้สลับ runner เป็น **🦙 LIVE** แล้วโมเดล local จริงจะเล่นบทหนึ่งบท

**สิ่งที่คุณจะได้ลงมือทำ**
- อ่าน template ของ editor สองตัวใน repository เทียบกัน และเรียนรู้การทดสอบข้อเดียวที่ตัดสินว่า editor ตัวไหนเป็นเจ้าของความล้มเหลว
- เรียนรู้บันได "prefer, in this order" ของ repository: skill, script, subagent, plugin และ always-on text เป็นอันดับสุดท้าย
- จับคู่ทุกความล้มเหลวที่ agent จำลองแสดงบน refine half กับ scope, บรรทัดใน template และ component ที่ editor ของ kit เขียน พร้อมต้นทุน always-on ของแต่ละอัน
- รัน generic-editor ablation บนสาม stream แล้วดูผลสองอย่าง: score และ always-on text ที่แต่ละแบบเพิ่มเข้ามา
- ดู editor ไต่จากกฎแบบ prose ไปเป็น plugin เฉพาะหลังจาก prose ถูกข้ามอีกครั้ง

**Time** ~45 นาที · **Difficulty** intermediate · **Hardware** แล็ปท็อปเครื่องนี้เท่านั้น (ไม่ต้องใช้โมเดล)

**Sources:** [Harness² paper, arXiv 2610.14224](https://arxiv.org/html/2610.14224) · [google-research/harness2](https://github.com/google-research/harness2) (Apache-2.0; snapshot ของ `prompts.py` อยู่ใน `common/sources/`) · skill จาก Week 7, hook และ mod จาก Week 29, prompt caching จาก Week 2

## 0 · ก่อนเริ่ม

| ต้องมี | ตรวจอย่างไร | ทำไม |
|---|---|---|
| ทำ Module 02 แล้ว | คุณอ่าน `rec.show()` ได้ และบอกชื่อหกขั้นได้ | module นี้ถามว่าการแก้แต่ละอันที่ถูกเก็บไว้ควรอยู่ที่ไหน |
| Python ของ repo นี้ | `.venv/bin/python --version` → 3.13 | ใช้รันทุก lab |
| ไม่ต้องมีโมเดล | — | ทั้งสาม lab เป็น SIM และ generic ablation รัน loop 204 ก้าวในเวลาไม่ถึงหนึ่งวินาที |

```bash
# on: laptop
.venv/bin/python -c "import sys; sys.path.insert(0, 'week31_harness2_recursive_harnessing/common'); from h2kit import observe; import inspect; print(len(inspect.getsource(observe).splitlines()), 'lines decide what an editor can see')"
```

**Expected output** (captured on this Mac, SIM)

```
28 lines decide what an editor can see
```

เปิด `common/h2kit.py` แล้วอ่าน `observe()` ความล้มเหลวทุกชนิดใน module นี้มาจากบรรทัดเหล่านั้น และแต่ละชนิดติดป้าย `inv` หรือ `dom`

✓ Checkpoint: คุณอ่าน `observe()` แล้ว และบอกชนิดที่เป็น `inv` ได้หนึ่งชนิด และ `dom` ได้หนึ่งชนิด

## 1 · Editor สองตัว สองมิติ

ก้าวของ paper เสนอ candidate **สองตัว** ไม่ใช่ตัวเดียว แต่ละตัวมาจาก editor ที่มีหน้าที่ของตัวเอง Lab 03-1 พิมพ์บรรทัดสำคัญของ template ทั้งสอง

> “You own procedure, tool use and completion -- not subject matter.” — `harness2/core/prompts.py`, the INV template (คุณเป็นเจ้าของ procedure การใช้ tool และการทำงานให้เสร็จ ไม่ใช่เนื้อหาเฉพาะทาง)

> “You own the practitioner model the agent lacks:” — `harness2/core/prompts.py`, the DOM template (คุณเป็นเจ้าของความรู้ของผู้ปฏิบัติงานที่ agent ขาด)

editor INV ระบุรูปแบบความล้มเหลวห้าแบบ พร้อม component สำหรับแต่ละแบบ editor DOM เปลี่ยนความรู้ของผู้ปฏิบัติงานให้เป็นกฎที่ตรวจได้สี่ชนิด

| INV — agent ทำงานอย่างไร | DOM — ผู้ปฏิบัติงานรู้อะไร |
|---|---|
| TOOL ERROR → fallback ที่มีขอบเขตหนึ่งทาง พร้อมเงื่อนไขหยุด | ANALYTIC MOVE → ระบุการกระทำ input และผลที่ต้องได้ |
| MECHANICAL REPETITION → script | ARTIFACT CONTRACT → ส่วนที่ต้องมีและการตรวจของมัน |
| TRIGGERED WORKFLOW → skill แบบ on-demand | DECISION RULE → trigger, หลักฐาน, ข้อขัดแย้ง, ความไม่แน่นอน |
| IGNORED INVARIANT → plugin เชิงป้องกัน เฉพาะเมื่อมีหลักฐานซ้ำ | ENCODING → เลือกรูปแบบ โดยค่าเริ่มต้นคือ skill |
| ORDINARY SEQUENCING → always-on guidance แบบกระชับ | |

แต่ละ template ปิดท้ายด้วยการกันอีกฝ่ายออกไป:

> “Do not add professional rules, required domain sections or matter facts. A parallel candidate owns them; stay focused on how the agent works.” — the INV template (อย่าเพิ่มกฎวิชาชีพ ส่วนที่ต้องมีตามสาขา หรือข้อเท็จจริงของงาน candidate อีกตัวเป็นเจ้าของ)

> “Do not diagnose commands, retries, reading order, output paths, budgeting, scripts or hooks. A parallel candidate owns procedure; stay on practitioner knowledge.” — the DOM template (อย่าวินิจฉัยคำสั่ง การ retry ลำดับการอ่าน path ของ output budget script หรือ hook)

prompt pack ชุดที่สองของ repository บอกการทดสอบไว้ในบรรทัดเดียวของแต่ละฝั่ง:

> “The test: if your edit stops making sense when the subject matter changes, it belongs to them, not you.” — `harness2/core/prompts.py`, the INVARIANT editor (ถ้าการแก้ของคุณไม่สมเหตุสมผลเมื่อเปลี่ยนเนื้อหา มันเป็นของอีกฝ่าย)

> “The test: if your edit would help just as much on a task from a completely different profession, it belongs to them.” — `harness2/core/prompts.py`, the DOMAIN editor (ถ้าการแก้ช่วยได้เท่ากันในงานของวิชาชีพอื่นโดยสิ้นเชิง มันเป็นของอีกฝ่าย)

ข้อเท็จจริงเดียวกันลงได้ทั้งสอง scope ในสองรูปแบบ "เรียก `cmms.work_orders` ก่อนเสนองานกับอุปกรณ์ใด ๆ" เป็นกฎลำดับงานที่ใช้กับอุปกรณ์ทุกชนิด: INV "กระทบยอด alarm กับ work order ที่เปิดอยู่ และบอกว่า PM ครอบคลุมอะไรแล้ว" เป็น analytic move ของผู้ปฏิบัติงาน: DOM ใน Lab 02-2 candidate ทั้งสองตัวทำการตรวจการซ่อมบำรุง แต่ละตัวในรูปแบบของตัวเอง

แยกไปทำไม? editor ตัวเดียวที่เห็นทุกอย่างมักเขียนออกมาอย่างเดียว คือ prose หนึ่งย่อหน้า editor สองตัวที่มีหน้าที่ถูกกั้นไว้จะเขียนการแก้สองชนิดที่ต่างกัน และ composer ตัดสินแต่ละอันได้จาก probe ของมันเอง Section 4 วัดว่าจะเกิดอะไรขึ้นเมื่อเอาการแยกออก

✓ Checkpoint: คุณใช้การทดสอบบรรทัดเดียวกับ "call cmms.work_orders before any equipment proposal" (INV) และกับ "reconcile the alarm with the open work orders" (DOM) ได้ และบอกได้ว่าทำไมสองอันนี้ต่างกัน

## 2 · บันได: prefer, in this order

editor ทั้งสองได้รับภาพของ harness แบบเดียวกัน และคำเตือนเรื่องต้นทุนแบบเดียวกัน Module 01 วัดต้นทุนนั้นไปแล้ว prompt พูดไว้ตรง ๆ:

> “ALWAYS-ON PROSE IS THE EXPENSIVE OPTION. Every word of it is in context on every run, competing with the agent's own output budget” — `harness2/core/prompts.py` (always-on prose คือทางเลือกที่แพง ทุกคำอยู่ใน context ทุก run และแย่ง output budget ของ agent)

จากนั้นมันให้ลำดับการลองใช้ component:

> “So prefer, in this order:” — `harness2/core/prompts.py` (จึงควรเลือกตามลำดับนี้)

| ขั้น | Component | เมื่อไร (จาก prompt) |
|---|---|---|
| 1 | **skill** (C6) | "anything with a recognisable trigger" (อะไรก็ตามที่มี trigger ชัดเจน) เป็นรูปแบบเริ่มต้นของคำสั่งใหม่ |
| 2 | **script** (C7) | "a deterministic operation the runs hand-rolled and got wrong, or did slowly" (งานแบบ deterministic ที่ run ทำเองแล้วผิด หรือทำช้า) |
| 3 | **subagent** (C5) | "a bounded investigation whose input is large and whose answer is small" (การตรวจสอบที่มีขอบเขต input ใหญ่ คำตอบเล็ก) |
| 4 | **plugin** (C8) | ปัญหาเชิงกลไกที่เกิดซ้ำใน tool loop |
| 5 | **always-on text** | "last, and only for a short rule that genuinely applies to every run" (เป็นอันดับสุดท้าย และใช้กับกฎสั้น ๆ ที่ใช้ได้กับทุก run จริง ๆ เท่านั้น) |

อ่านขั้นที่ 5 ให้ดี มันไม่ได้แปลว่า "ห้ามเด็ดขาด" template ของ INV ส่ง ORDINARY SEQUENCING ไปที่ "concise always-on guidance" โดยตั้งใจ: "ตรวจ work order ก่อนเสนอ" เป็นบรรทัดสั้นบรรทัดเดียวที่ใช้กับทุก fault review กฎนี้คือกฎเดียวกับที่ Module 01 สอนสำหรับ Ops Assistant จ่ายค่า always-on text เฉพาะเมื่อมันคุ้ม และถ้าเพิ่มก็ต้องตัดอย่างอื่นออก

บันไดนี้ยังตรงกับผล ablation ที่ชัดที่สุดของ paper ด้วย ตามที่ paper รายงาน การเอา skill ออกจาก harness ที่ปรับแล้วทำให้เสียเฉลี่ย 5.3 คะแนน เอา script ออกเสีย 3.7 และเอา prompt text ออกเสียแค่ 1.5

✓ Checkpoint: คุณท่องห้าขั้นตามลำดับได้ และบอกได้ว่าขั้นไหนที่ prompt เรียกว่า "the default form for new instruction" (skill)

## 3 · ทุกความล้มเหลว scope และ component ของมัน

Lab 03-1 รัน base harness บน refine task ทั้ง 17 ตัว (สาม seed, executor ทั้งสองแบบ) เก็บความล้มเหลวทุกชนิดที่ editor มองเห็นได้ แล้วถาม editor ของ kit ว่าจะเขียนอะไรให้แต่ละชนิด

**Expected output** (captured on this Mac, SIM)

```
│ failure kind      scope  template line (our reading)                  kit writes  ladder rung  Δ always-on  tasks
│ ────────────────  ─────  ───────────────────────────────────────────  ──────────  ───────────  ───────────  ─────
│ digest_contract   DOM    ARTIFACT CONTRACT                            C1 + C6     1 skill      +19          3    
│ no_analytic_move  DOM    ANALYTIC MOVE                                C1 + C6     1 skill      +19          6    
│ nuisance_notify   DOM    DECISION RULE                                C1 + C6     1 skill      +13          1    
│ storm_spam        DOM    DECISION RULE                                C1 + C6     1 skill      +13          1    
│ thai_register     DOM    ENCODING                                     C1 + C6     1 skill      +18          1    
│ blind_retry       INV    TOOL ERROR                                   C4          5 always-on  +38          1    
│ hand_arithmetic   INV    MECHANICAL REPETITION                        C4 + C7     2 script     +23          4    
│ no_cmms           INV    ORDINARY SEQUENCING                          C4          5 always-on  +24          6    
│ no_deliverable    INV    completion: reserve room for every artifact  —           — (no edit)  +0           1    
│ no_evidence       INV    ORDINARY SEQUENCING → IGNORED INVARIANT      C4          5 always-on  +25          6    
│ no_reopen         INV    completion: reopen or check the deliverable  C4 + C7     2 script     +29          3    
```

คอลัมน์ "template line" เป็นการอ่านของคอร์ส ไม่ใช่ของ repository ส่วนที่เหลืออ่านแบบนี้

- **ทุกแถวของ DOM เป็น skill บวก trigger line หนึ่งบรรทัด** `C1` ใน "C1 + C6" คือบรรทัดใน `systemprompt.md` ที่ระบุชื่อ skill ถ้าไม่มีบรรทัดนี้ skill จะไม่ถูกโหลดเลย surface prompt ของ repository เรียก trigger line ที่หายไปว่า "the most violated rule here by a wide margin" (กฎที่ถูกละเมิดบ่อยที่สุดโดยทิ้งห่าง) always-on 13–19 token คือบรรทัดนั้นอย่างเดียว เนื้อหาของ skill โหลดเฉพาะเมื่อ trigger ของมันทำงาน
- **แถวของ INV แยกเป็นสามทาง** script เมื่อ run ทำอะไรเองด้วยมือ (kW/ton, การตรวจ digest) always-on หนึ่งบรรทัดสั้น ๆ สำหรับลำดับงานทั่วไป และสำหรับ fallback ที่มีขอบเขตของ tool error และไม่เขียนอะไรเลยสำหรับ `no_deliverable`: kit ไม่เคยเขียน prose แบบ "พยายามให้มากขึ้น" มันแก้สาเหตุที่มองเห็นแทน เช่น การ retry แบบไม่คิด
- **`no_evidence` เริ่มที่ขั้นที่ 5** การแก้ครั้งแรกของกฎที่ถูกข้ามคือการเขียนกฎให้ชัด Section 5 แสดงว่าเมื่อไรมันไต่ขึ้นไปขั้นที่ 4 ได้

ทุกแถวคือสิ่งที่ trajectory แสดงให้เห็น คอลัมน์ "tasks" นับจำนวน refine task ที่มันปรากฏอย่างน้อยหนึ่งครั้ง นี่คือคำสั่งของ template INV ในรูปที่จับต้องได้: "match each failure to the cheapest effective component" (จับคู่แต่ละความล้มเหลวกับ component ที่ถูกที่สุดซึ่งได้ผล)

✓ Checkpoint: คุณบอกได้ว่าทำไม `hand_arithmetic` ได้ script แต่ `no_cmms` ได้ always-on หนึ่งบรรทัด

## 4 · Generic-editor ablation

ถ้ามี editor ตัวเดียวทำทุกอย่างจะเป็นอย่างไร `scoped=False` ของ kit เปลี่ยน editor INV และ DOM เป็น **generic editor ตัวเดียว** ที่เขียนแบบที่ "editor ตัวเดียวเขียน prompt ใหม่" มักเขียน: prose ยาว ไม่มี scope ใน `systemprompt.md` Lab 03-2 แสดงการแก้ของ editor ทั้งสองแบบสำหรับความล้มเหลวเดียวกันก่อน

**Expected output** (captured on this Mac, SIM)

```
┌─ scoped INV editor → tool_guidance.md  (+24 tokens)
│ - Before any proposal about equipment, check open and recent work orders with cmms.work_orders.
└─
┌─ generic editor → systemprompt.md  (+68 tokens)
│ When working, always be careful and thorough about no cmms. Before any proposal about equipment, check open and recent work orders, because maintenance may already be planned. Think step by step, double-check your work, and explain your reasoning clearly to the reader.
└─
```

จากนั้นมันรัน open-world stream สาม stream (สลับลำดับ) บน refine task 17 ตัว สามแบบ โดยปิด library ในทั้งสองแบบที่เป็น loop เพื่อให้ต่างกันแค่เรื่อง scope (editor คือ `sim-cloud-editor` ในทุกแบบ เหมือนตารางใน `AUTHORING.md` ส่วน Lab 02-3 จับคู่ executor แต่ละตัวกับ editor ของมันเอง)

**Expected output** (captured on this Mac, SIM)

```
│ executor   arm                                score* mean ± SD  per stream                              
│ ─────────  ─────────────────────────────────  ────────────────  ──────────────────  ────────────────────
│ sim-cloud  base harness                       80.3 ± 4.1        81.2 / 83.8 / 75.8  ████████████████░░░░
│ sim-cloud  Harness² k=1 · scoped INV + DOM    93.8 ± 3.7        97.6 / 93.5 / 90.3  ███████████████████░
│ sim-cloud  Harness² k=1 · one generic editor  91.1 ± 2.9        94.1 / 90.9 / 88.3  ██████████████████░░
│ sim-local  base harness                       72.2 ± 3.7        76.3 / 69.2 / 71.0  ██████████████░░░░░░
│ sim-local  Harness² k=1 · scoped INV + DOM    81.8 ± 4.5        84.1 / 76.6 / 84.8  ████████████████░░░░
│ sim-local  Harness² k=1 · one generic editor  77.3 ± 3.6        80.4 / 73.3 / 78.2  ███████████████░░░░░
◆ * measured after the run by bench.judge, which reads the hidden truth — nothing in the loop ever sees it
◆ sim-cloud: scoped − generic = +2.7 points · all three streams agree in direction: yes · within one SD — the size is not shown on 17 tasks
◆ sim-local: scoped − generic = +4.5 points · all three streams agree in direction: yes · larger than one SD
```

อ่าน score อย่างซื่อตรง แบบมี scope ชนะ generic บน executor ทั้งสองแบบ ทิศทางเดียวกับ paper ซึ่งรายงานว่า ablation ที่ไม่มี scope ทำให้ลดลง 2.9 และ 3.7 บน WorkBuddy-Bench และ JobBench บน executor ทั้งสองแบบ ทั้งสาม stream เห็นทิศทางตรงกัน กฎของ facilitator guide (เชื่อความต่างเล็ก ๆ เฉพาะเมื่อทั้งสาม stream เห็นตรงกัน) จึงให้คุณเชื่อทิศทางได้ ส่วนขนาดเป็นอีกเรื่อง บน sim-cloud ช่องว่าง (+2.7) อยู่ภายในหนึ่ง standard deviation บน 17 task ขนาดของมันจึง **ยังไม่ได้แสดง** บน sim-local ช่องว่าง (+4.5) ใหญ่กว่าหนึ่ง SD แต่ขนาดนั้นมาจากสมมติฐานของ SIM (ดูหมายเหตุด้านล่าง) มันจึงแสดงกลไก และไม่ได้วัดอะไรเลย Module 06 อ่านตารางผลแบบนี้อย่างละเอียด

ตารางที่สองคือตารางที่ไม่ได้ซ่อนอยู่ในสัญญาณรบกวน

**Expected output** (captured on this Mac, SIM)

```
│ executor   arm                                composed  Δ always-on mean  max   components changed          nothing delivered
│ ─────────  ─────────────────────────────────  ────────  ────────────────  ────  ──────────────────────────  ─────────────────
│ sim-cloud  Harness² k=1 · scoped INV + DOM    27/51     +38.3             +68   C1×20, C4×19, C6×22, C7×12  0                
│ sim-cloud  Harness² k=1 · one generic editor  17/51     +100.9            +140  C1×17                       0                
│ sim-local  Harness² k=1 · scoped INV + DOM    25/51     +36.3             +72   C1×16, C4×19, C6×17, C7×13  2                
│ sim-local  Harness² k=1 · one generic editor  14/51     +105.7            +201  C1×14                       2                
✓ sim-cloud: every generic edit landed in C1 (systemprompt.md) — always-on prose, paid on every call
✓ sim-local: every generic edit landed in C1 (systemprompt.md) — always-on prose, paid on every call
```

- generic editor เพิ่ม **always-on ประมาณ 100 token** ต่อก้าวที่ compose เทียบกับประมาณ 40 ของคู่ที่มี scope ก้าวที่แย่ที่สุดเพิ่ม 201
- การแก้ของ generic ทุกอันลงที่ **C1** คู่ที่มี scope กระจายการแก้ไปที่ skill, script, tool guidance และ trigger line สั้น ๆ
- แบบ generic ยัง compose น้อยกว่าด้วย (17 เทียบกับ 27 ก้าวบน sim-cloud และ 14 เทียบกับ 25 บน sim-local) prose ของมันไม่ระบุ trigger ไม่มี script และไม่มี tool probe ของมันจึงมักไม่เห็นความเปลี่ยนแปลง และ composer ทิ้งไป KEEP_BASE ช่วยปกป้องแบบ generic จากตัวมันเองได้นิดหน่อย

สำหรับ agent แบบ always-on คอลัมน์ token คือคอลัมน์ที่แพง always-on token ทุกตัวต้องจ่ายในทุกการเรียก และทุกการเปลี่ยนแปลงคือ prompt cache เย็นหนึ่งครั้ง นี่คือเหตุผลที่ EXAMPLE design ของคอร์สสำหรับ Ops Assistant ให้ invariant editor เอนไปทาง component แบบ on-demand และให้ review packet ทุกชุดแสดง always-on delta

> 📌 **SIM ไม่ใช่หลักฐาน** agent ของ SIM ทำตาม prose ด้วยความน่าจะเป็นคงที่ และทำตาม skill ที่ถูกเรียกได้สม่ำเสมอกว่า สิ่งเหล่านี้เป็นสมมติฐานที่เขียนไว้ใน `ops_sim.py` เพื่อแสดงกลไกที่ paper รายงาน มันทำให้รูปร่างของตารางนี้มีแนวโน้มจะเกิด แต่ไม่ได้วัดมัน

✓ Checkpoint: คุณบอกได้ว่าผลไหนของ ablation อยู่ในระดับสัญญาณรบกวนในที่นี้ (ช่องว่างของ score บน sim-cloud) ผลไหนพ้นหนึ่ง SD ได้ภายใต้สมมติฐานของ SIM เท่านั้น (ช่องว่างของ score บน sim-local) และผลไหนไม่ใช่สัญญาณรบกวนบน executor ใดเลย (always-on ที่โตขึ้น ซึ่งทั้งหมดอยู่ใน C1)

## 5 · Plugin เฉพาะหลังจากมีหลักฐานซ้ำ

plugin เป็น component เดียวที่บังคับได้ Module 01 ยกเหตุผลไว้แล้ว: "Prose can be stopped reading; a hook cannot." แล้วทำไมไม่หยิบมันทุกครั้งที่กฎถูกข้าม? เพราะ template ของ INV ตั้งเกณฑ์ไว้:

> “- IGNORED INVARIANT: only repeated evidence that clear prose is skipped can justify a defensive” — `harness2/core/prompts.py` (the template names the plugin hook here) (มีแต่หลักฐานซ้ำว่า prose ที่ชัดเจนถูกข้าม จึงจะอ้างการใช้ plugin เชิงป้องกันได้)

> “It must not block valid completion.” — `harness2/core/prompts.py` (มันต้องไม่ขวางการทำงานที่เสร็จอย่างถูกต้อง)

kit เข้ารหัสเกณฑ์นี้ไว้ใน `inv_edits("no_evidence", tree)` Lab 03-3 เรียกมันสองครั้ง

**Expected output** (captured on this Mac, SIM)

```
✎ tree = v1 base       → files — · always-on line: '- Every ActionProposal carries evidence links: the alarm id, the telemetry window, the document.'
✎ tree = v1 + that line → files ['plugin/require_evidence.ts'] · always-on line: ''
```

บน base tree editor เขียนกฎเป็นบรรทัดที่ชัดเจนหนึ่งบรรทัด บน tree ที่มีบรรทัดนั้นแล้ว และยังได้ proposal ที่ไม่มี evidence มันจะเขียน `plugin/require_evidence.ts` แทน สิ่งที่ kit ใช้แทน "หลักฐานซ้ำ" คือแบบนั้นพอดี: บรรทัดนั้นมีอยู่แล้ว (การแก้ครั้งก่อนใส่ไว้) และ run นี้ยังข้ามมันอีก เป็นการสังเกตสองครั้ง ไม่ใช่ครั้งเดียว

แต่ละขั้นได้อะไรกลับมา? lab นับ evidence list ที่ว่างเปล่าจาก 40 proposal ต่อแถว:

**Expected output** (captured on this Mac, SIM)

```
│ executor   harness           proposals  evidence empty  plugin blocked + agent fixed
│ ─────────  ────────────────  ─────────  ──────────────  ────────────────────────────
│ sim-cloud  v1 base           40         16              0                           
│ sim-cloud  + prose line      40         2               0                           
│ sim-cloud  + prose + plugin  40         0               2                           
│ sim-local  v1 base           40         27              0                           
│ sim-local  + prose line      40         11              0                           
│ sim-local  + prose + plugin  40         0               11                          
```

บรรทัด prose ตัดความล้มเหลวไปได้เกือบหมด แต่ตัดได้ไม่หมด และ executor ที่อ่อนกว่าข้ามมันบ่อยกว่า (ยังว่าง 11 จาก 40 เทียบกับ 2 บน sim-cloud) hook จับกรณีที่เหลือได้ทุกกรณี และเพิ่ม always-on เป็นศูนย์ token เพราะมันอยู่ใน tool loop อัตราเหล่านี้เป็นสมมติฐานของ SIM ไม่ใช่การวัดโมเดลจริงตัวใด

จากนั้น lab แสดงการไต่แบบเดียวกันภายในก้าวเดียว ก้าวแบบ **sequential** ที่ k = 2 แก้ checkpoint ของตัวเองในรอบที่ 2 (Module 06 เปรียบเทียบสองโหมดนี้) lab ใช้ seed 19 ของ F1-02 บน sim-local ซึ่งเลือกมาเพราะความล้มเหลวกลับมาบน checkpoint ที่มีบรรทัดนั้นอยู่แล้ว ใน seed ส่วนใหญ่ของ task นี้ความล้มเหลวไม่กลับมา และไม่มีอะไรไต่ขึ้นไป: จาก seed 0–29 มีแค่ seed 19 ที่ไปถึง plugin นี่คือเกณฑ์ที่ทำงานตามที่ตั้งใจ:

**Expected output** (captured on this Mac, SIM)

```
✎ inv1.r1 [inv] ActionProposal sent with an empty evidence list → tool_guidance.md
✎ inv1.r2 [inv] ActionProposal sent with an empty evidence list → plugin/require_evidence.ts
◆ round 1 (editing v1): no_evidence → ['tool_guidance.md'] · round 2 (editing the round-1 checkpoint, which has the line): no_evidence → ['plugin/require_evidence.ts']
✓ the plugin appears only in round 2, after the prose line was in place and the failure came back
```

ส่วนที่ตัดมาเก็บบรรทัด edit ของ `no_evidence` สองบรรทัดจาก `rec.show()` ฉบับเต็ม และบทสรุปของ lab ท้ายสุด lab ตรวจประโยคที่สองของ template บน deliverable สี่ตัวที่ไม่ใช่ proposal plugin ทำงานศูนย์ครั้ง และทั้งสี่ตัวส่งงานได้ ใน EXAMPLE design ของ Ops Assistant hook นี้คือ auto-review ของ `send_to_user` มันยังไม่เคยแทนที่การอนุมัติโดยคนที่ระบุชื่อ ซึ่งอยู่ใน platform code (Module 01)

✓ Checkpoint: คุณบอกได้ว่าต้องเกิดอะไรก่อน editor จึงจะเขียน plugin ได้ และ plugin ต้องไม่ทำอะไรเด็ดขาด

## Labs — รันได้ที่นี่

**labs/lab03_1_two_scopes_and_the_ladder.py** — template ของ INV และ DOM กับบันได แบบคำต่อคำ แล้วจับคู่ทุกความล้มเหลวใน refine half กับ scope, บรรทัดใน template, component และต้นทุน always-on

**labs/lab03_2_generic_editor_ablation.py** — base เทียบกับแบบมี scope และ generic editor ตัวเดียว บนสาม stream ต่อ executor: score และ always-on text ที่แต่ละแบบเพิ่ม

**labs/lab03_3_plugin_after_repeated_evidence.py** — กฎของ no_evidence เมื่อมีและไม่มีบรรทัด prose แต่ละขั้นได้อะไร ก้าวแบบ sequential ที่ไต่ไปถึง plugin และการตรวจว่า plugin ไม่ขวางงาน

## ลองทำเอง

`exercises/ex03_scope_and_component.py` แสดงชนิดความล้มเหลวแบบ EXAMPLE เก้าข้อของ Ops Assistant ซึ่งเป็น design ของคอร์สเอง มี TODO สามข้อ:

1. scope ของแต่ละความล้มเหลว: INV หรือ DOM
2. บรรทัดใน template ที่แต่ละข้อตกอยู่
3. component ที่ถูกที่สุดซึ่งได้ผลของแต่ละข้อ

```bash
# on: laptop
.venv/bin/python week31_harness2_recursive_harnessing/03_two_scopes_cheapest_component/exercises/ex03_scope_and_component.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ scopes right: five INV (how the agent works), four DOM (what a practitioner knows)
✓ every failure sits on the right line of its editor's template
✓ cheapest components right: one bounded fallback, two scripts, five skills, one plugin
✓ only failure 4 earns a plugin, and only because clear prose was already skipped repeatedly

═ Done. Module 04 asks how the composer knows an edit helped: contrasts, verdicts and the evidence boundary.
```

ในที่ที่ความล้มเหลวหนึ่งข้อใช้ได้สอง component (guidance line *และ* script) แบบฝึกหัดนี้ถามหาตัวที่ถูกที่สุด

<details><summary>คำใบ้ — TODO 1</summary>

ใช้การทดสอบบรรทัดเดียวจาก section 1 "retry query ที่ timeout หนึ่งครั้งแบบแบ่งช่วง" สมเหตุสมผลไม่ว่าอาคารจะทำอะไร: INV "escalate alarm แรงดัน condenser สูง เฉพาะเมื่อ…" ไม่สมเหตุสมผลนอก HVAC: DOM มีสองข้อที่หลอกได้ procedure ของ chiller trip เป็น workflow หลายขั้นที่ตายตัว คอร์สจึงจัดไว้ใต้ TRIGGERED WORKFLOW ของ INV การกระทบยอดรายการ BMS alarm กับ work order ใน CMMS เป็นการเปรียบเทียบของผู้ปฏิบัติงาน คอร์สจึงจัดไว้ใต้ ANALYTIC MOVE ของ DOM แม้จะมีการเรียก tool อยู่ด้วย

</details>

<details><summary>คำใบ้ — TODO 3</summary>

ไล่บันไดจากบนลงล่าง trigger ที่ชัดเจนแปลว่า skill การคำนวณด้วยมือหรือการตรวจเชิงกลไกแปลว่า script tool error แปลว่า fallback ที่มีขอบเขตหนึ่งทางใน tool guidance มีแค่ความล้มเหลวที่บอกว่า "despite instructions" (ทั้งที่มีคำสั่งแล้ว) ที่มีหลักฐานซ้ำซึ่ง plugin ต้องการ

</details>

✓ Checkpoint: ตัวตรวจขึ้น ✓ ครบทั้งสี่บรรทัด

## การแก้ปัญหา

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| ตัวเลขของ Lab 03-2 ต่างจากตารางใน `AUTHORING.md` เล็กน้อย | ตารางนั้นรันโดยเปิด library ส่วน lab นี้ปิดในทั้งสองแบบที่เป็น loop | เทียบให้ตรงกัน: แบบ "scoped" ในที่นี้คือคอลัมน์ "no library" ในตารางนั้น |
| ขั้น 4 ของ Lab 03-3 พิมพ์ `⚠ the escalation did not happen` | kit ถูกแก้ (ตัวอย่าง sequential ขึ้นกับ `sim-local`, seed 19) | `git diff week31_harness2_recursive_harnessing/common` |

## ถัดไป

[Lab 04 — Evidence, not scores: contrasts, verdicts and the evidence boundary](../04_evidence_not_scores/TUTORIAL.md): เรียนรู้ว่า composer แยกการแก้ที่ช่วยจริงออกจากการแก้ที่แค่ดูเหมือนช่วยได้อย่างไร และทำไมหลักฐานที่มันอ่านจึงเป็นข้อมูล ไม่เคยเป็นคำสั่ง
