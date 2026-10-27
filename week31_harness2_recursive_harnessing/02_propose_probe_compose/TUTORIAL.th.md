# ▶ Harness Lab 02 — Propose–Probe–Compose หนึ่งก้าว ทั้งด้วยมือและด้วย loop

> ส่วนหนึ่งของ Week 31 · Harness²: agent ที่ปรับปรุง harness ของตัวเอง คุณจะปรับปรุง harness ของ agent จำลองหนึ่งตัว คือ **HVAC Expert agent ของ Ops Assistant** (ผู้ช่วยด้านปฏิบัติการแบบ always-on ที่สมมติขึ้น) จากหลักฐานการทำงานจริง และเห็นผลลัพธ์จริง ทุกอย่างเป็นข้อมูลสังเคราะห์ (โรงแรมสมมติสองแห่ง alarm และ work order ที่แต่งขึ้น) ทุก lab จึงฟรี ออฟไลน์ และปลอดภัย Lab ทำงานบน **🧠 SIM** เป็นค่าเริ่มต้น คือโมเดลสอนแบบ deterministic ของ agent และ editor ของมัน ถ้า lab ไหนรองรับ ให้สลับ runner เป็น **🦙 LIVE** แล้วโมเดล local จริงจะเล่นบทหนึ่งบท

**สิ่งที่คุณจะได้ลงมือทำ**
- เรียนรู้หกขั้นของ Harness² หนึ่งก้าว และแต่ละขั้นอ่านอะไรได้บ้าง
- จำลอง demo แบบออฟไลน์ของ repository ใหม่ด้วย script สั้น ๆ ของคุณเอง: ledger หนึ่งไฟล์, script ที่มี bug กรองเฉพาะค่าบวก, editor ที่ลบตัวกรองออก, probe, การตัดสินใจ compose และ entry ใน `EXPERIENCE.md`
- รัน loop ของคอร์สหนึ่งก้าวเต็มบน task F1-01 ซึ่งคือ chiller proposal ที่ถูกปฏิเสธจากตัวอย่างที่คอร์สใช้เดินเรื่อง แล้วอ่านทุกบรรทัดที่มันพิมพ์
- รันหนึ่งก้าวบน quiet task และ trap task ทุกตัว นับว่าคำตอบที่ซื่อตรงคือ KEEP_BASE บ่อยแค่ไหน และดูว่าทำไม trap task บางครั้งก็ได้ KEEP_BASE
- เป็น composer เอง: keep, drop หรือ KEEP_BASE สำหรับตาราง contrast ห้าตาราง

**Time** ~45 นาที · **Difficulty** intermediate · **Hardware** แล็ปท็อปเครื่องนี้เท่านั้น (ไม่ต้องใช้โมเดล)

**Sources:** [Harness² paper, arXiv 2610.14224](https://arxiv.org/html/2610.14224) · [google-research/harness2](https://github.com/google-research/harness2) (Apache-2.0; snapshot ของ `prompts.py` และ `demo.py` อยู่ใน `common/sources/`) · self-evolving agent จาก Week 18, eval จาก Week 10, prompt caching จาก Week 2

## 0 · ก่อนเริ่ม

| ต้องมี | ตรวจอย่างไร | ทำไม |
|---|---|---|
| ทำ Module 01 แล้ว | คุณบอกชื่อ C1–C8 ได้ และรู้ว่าสี่ตัวไหนเป็น always-on | ทุกการแก้ใน module นี้ลงที่ component ใดตัวหนึ่ง |
| Python ของ repo นี้ | `.venv/bin/python --version` → 3.13 | ใช้รันทุก lab |
| ไม่ต้องมีโมเดล | — | ทั้งสาม lab เป็น SIM และ editor ในที่นี้เป็นตารางกฎ |

```bash
# on: laptop
.venv/bin/python week31_harness2_recursive_harnessing/common/h2kit.py | tail -3
```

**Expected output** (captured on this Mac, SIM)

```
⚖ tool_guidance.md → helped
⚖ skills/chiller_fault_review/SKILL.md, systemprompt → helped
═ COMPOSED · selected composed · EXPERIENCE entry written · always-on 245 → 288 tokens · 4 executions, 4 editor sessions
```

บรรทัดเดียวนี้คือ Harness² หนึ่งก้าวเต็ม ๆ เมื่อจบ module นี้ คุณจะอธิบายได้ทุกคำในบรรทัดนั้น

✓ Checkpoint: การรัน kit ด้วยตัวเองพิมพ์บรรทัด `COMPOSED` ที่มี always-on 245 → 288 tokens

## 1 · หนึ่งก้าว หกขั้น

Week 18 แสดง agent ที่เขียน prompt ของตัวเองใหม่ Harness² ยังมีความทะเยอทะยานแบบนั้น แต่เพิ่มวินัยเข้าไป: **ทุกการแก้ต้องมาจากสิ่งที่ execution แสดงให้เห็น** ก้าวหนึ่งของ paper มีหกขั้น: reference, propose, probe, compose, select และ reflect และทุกการตัดสินใจผูกกับสิ่งที่สังเกตได้จริง ไม่ใช่ score ที่หวังว่าจะเพิ่ม

| ขั้น | เกิดอะไรขึ้น | อ่านอะไรได้ |
|---|---|---|
| Reference | รัน harness ปัจจุบันหนึ่งครั้งบน task | task และ harness |
| Propose | editor **INV** และ editor **DOM** เขียน candidate harness คนละหนึ่งตัว | trajectory, deliverable, request และ source, library |
| Probe | รัน candidate แต่ละตัวบน task เดิมด้วยโมเดลเดิม | — |
| Compose | merge การแก้ที่ probe ของมันรองรับ หรือตอบ KEEP_BASE | reference, candidate, probe ของ candidate และ diff |
| Select | เฉพาะเมื่อ k > 1: เลือกหนึ่งใน k final แบบไม่รู้ที่มา | deliverable ที่ไม่ระบุตัวตน, request, source |
| Reflect | เขียน edit row พร้อม verdict และ entry ใน `EXPERIENCE.md` ไม่เกินหนึ่งรายการ | บันทึกทั้งหมด |

กฎสองข้อจาก paper ทำให้ทั้งหมดนี้ทำงานได้โดยไม่มี ground truth ข้อแรก reference แสดงพฤติกรรม ไม่ได้แสดงความถูกต้อง และ composer ทำงานจาก contrast ของพฤติกรรมเท่านั้น:

> “compose a refined harness from behavioral contrasts between executions, without ground-truth feedback” — `harness2/README.md` (compose harness ที่ปรับแล้วจาก contrast ของพฤติกรรมระหว่าง execution โดยไม่มี feedback จาก ground truth)

ข้อสอง ไม่มีขั้นไหนเห็น score เลย ในคอร์สนี้มีแค่ `bench.judge` ที่อ่านความจริงที่ซ่อนไว้ของ task และมันรันหลังจากก้าวนั้น commit แล้ว Module 04 อธิบายว่าทำไมเส้นแบ่งนี้สำคัญมาก

ที่ k = 1 มี reference หนึ่งตัว candidate ต่อ scope หนึ่งตัว และไม่มีขั้น select นี่คือค่าที่ EXAMPLE design ของคอร์สใช้กับ routine ทุกคืนของ Ops Assistant และทุก lab ในที่นี้ใช้ค่านี้

✓ Checkpoint: คุณบอกชื่อหกขั้นได้ และบอกได้ว่าขั้นไหนทำงานเฉพาะเมื่อ k > 1 (select)

## 2 · demo ของ repository ทำใหม่ด้วยมือ

repository ของ Harness² มี demo ออฟไลน์ขนาดเล็กมาด้วย docstring ของมันบอกว่ามีไว้ทำอะไร:

> “A deterministic local adapter for exercising the real adaptation workflow.” — `harness2/demo.py` (adapter แบบ local ที่ deterministic สำหรับฝึก workflow การปรับจริง)

> “Scripted edits and selection; no model service or benchmark judge is used.” — `harness2/demo.py`, `DemoProposer` (การแก้และการเลือกเขียนเป็น script ไว้ ไม่ใช้บริการโมเดลหรือ judge)

โลกทั้งหมดของมันคือ ledger CSV หนึ่งไฟล์ และ base harness ที่มี script หนึ่งตัว script นั้นมี bug ที่มองข้ามได้ง่าย: มันเก็บเฉพาะยอดที่เป็นบวก

> “total = sum((amount for amount in amounts if amount > 0), Decimal(0))” — `harness2/demo.py`, `BASE_SCRIPT` (บรรทัดที่รวมเฉพาะยอดบวก)

editor ที่เขียนเป็น script แก้ด้วยการ replace บรรทัดเดียว และ reflection ที่เขียนเป็น script ก็เขียน edit row:

> “script.write_text(script.read_text().replace(" if amount > 0", ""), encoding="utf-8")” — `harness2/demo.py` (ลบตัวกรอง if amount > 0 ออก)

> “change: dropped the positive-only filter from the amount sum” — `harness2/demo.py` (edit row: ลบตัวกรองค่าบวกออกจากผลรวม)

> 📌 **คอร์สนี้ไม่ได้รัน demo** การรัน repository ที่ clone มาถูกปฏิเสธบนเครื่องนี้ จึงไม่มีผลลัพธ์ของ demo ที่บันทึกไว้ในที่นี้ facilitator guide ขอให้ผู้เรียนรัน `harness2-demo --output runs/demo --k 2` เองใน clone ของตัวเอง เป็นขั้นตอนเสริมก่อนเข้าเรียน Lab 02-1 ทำแนวคิดเดียวกันใหม่ด้วยโค้ดของตัวเอง

Lab 02-1 สร้าง ledger แบบเดียวกัน script ที่มี bug แบบเดียวกัน และ request แบบเดียวกัน แล้วเดินตาม loop สิ่งที่ editor มองเห็นคือ request ("Sum every signed amount"), แถวใน source และ trajectory มันไม่เคยเห็นเฉลย

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 2 · Reference run on ledger-1: evidence of BEHAVIOUR, not of correctness
│ [ACTION] Run scripts/summarize.py on ledger.csv
│ [OBSERVATION] Total: 105.25
◆ observed: credits_omitted: the source has 1 negative row(s); the delivered total equals the positive rows only

▣ STEP 3 · Propose: an INV editor edits the script (the demo's scripted editor does the same one-line replace)
✎ inv1 scripts/summarize.py (C7) modified — removed the positive-only filter
│ - total = sum((amount for amount in amounts if amount > 0), Decimal(0))
│ + total = sum((amount for amount in amounts), Decimal(0))

▣ STEP 4 · Probe the candidate on the SAME ledger and contrast it with the reference
│ reference: Total: 105.25   probe: Total: 85.15
◆ contrast: credits_omitted mitigated

▣ STEP 5 · Repeat the contrast on a second ledger (a repeated execution separates an edit from luck)
│ ledger-2 reference: Total: 35.50   probe: Total: 20.25
◆ contrast: credits_omitted mitigated

▣ STEP 6 · Compose: keep an edit only if its probe mitigated the failure, delivered, and regressed nothing
→ compose: keep scripts/summarize.py — credits_omitted mitigated on 2/2 ledgers, nothing regressed
```

สังเกตสามอย่าง

1. **การแก้อยู่ใน script (C7) ไม่ใช่ prose** always-on text ยังเป็น 14 token และ prefix hash ไม่ขยับ prompt cache จาก Week 2 จึงยังอุ่นอยู่
2. **contrast เป็นเรื่องของพฤติกรรม** "ยอดรวมที่ส่งมาเท่ากับแถวที่เป็นบวกเท่านั้น" เป็นสิ่งที่เห็นได้จากการอ่าน source คู่กับ trajectory selector ที่เขียนเป็น script ของ demo เองก็ตรวจเรื่องเดียวกันกับ ledger ต้นทาง
3. **ledger ที่สามไม่มียอดติดลบ จึงไม่มีหลักฐาน** bug ยังอยู่ใน script แต่ trajectory แสดงมันไม่ได้ คำตอบที่ซื่อตรงจึงเป็น KEEP_BASE และ `NO_ENTRY`

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 8 · The same loop on ledger-3, which has no credits: nothing to see, so KEEP_BASE
│ ledger-3 reference: Total: 20.50
◆ observed: nothing — the filter bug is real but this trajectory gives no evidence of it
→ compose: KEEP_BASE — the evidence does not argue for any change
✓ NO_ENTRY written: no edit rows, no lesson — the loop only fixes what an execution shows
```

✓ Checkpoint: คุณบอกได้ว่าทำไม ledger-3 ได้ KEEP_BASE ทั้งที่ script ของมันยังมี bug อยู่

## 3 · หนึ่งก้าวบน F1-01 ตัวอย่างที่คอร์สใช้เดินเรื่อง

EXAMPLE design ของคอร์สตาม proposal ที่ถูกปฏิเสธหนึ่งรายการผ่าน loop chief engineer ปฏิเสธ work order ของ chiller ด้วยเหตุผล "CH-2 was already on a PM; the agent did not check open work orders." (CH-2 อยู่ระหว่าง PM อยู่แล้ว agent ไม่ได้ตรวจ work order ที่เปิดอยู่) task F1-01 คือกรณีนั้นใน simulator Lab 02-2 รัน `step(task("F1-01"), base_harness())` แล้วพิมพ์ทุกอย่างด้วย `rec.show(trajectories=True)`

**Expected output** (captured on this Mac, SIM)

```
▣ task F1-01 · fault_review · signal rejected_proposal · reviewer: "CH-2 was already on a PM; the agent did not check open work orders."
◆ reference 0: delivered=True · observed: hand_arithmetic, no_cmms, no_analytic_move
┌─ run F1-01 · sim-cloud · seed 0 · harness 773bf3d26eee · prefix ee3bb0d97ccb
│ [ACTION] alarms.list(window='overnight')
│ [OBSERVATION] AL-901 CH-2.kw_per_ton: high kW/ton
│ [ACTION] telemetry.query('CH-2.kw', days=1)
│ [OBSERVATION] 6 hourly rows (kw, tons)
│ [ACTION] docs.search('CH-2 high kW/ton')
│ [OBSERVATION] Harbour View Hotel (fictional) O&M: condenser approach above 3.0 K indicates fouling; design efficiency 0.62 kW/ton.
│ [REASONING] summed kW and tons by hand across six rows
│ [DELIVER] send_to_user(outcome=done)
│ ── deliverable ──
│ diagnosis: condenser fouling
│ kw_per_ton: 0.201
│ proposal: {"type": "ActionProposal", "action": "create work order: CH-2 condenser fouling", "approver": "chief engineer", "evidence": ["AL-901", "CH-2.kw telemetry"]}
└─
✎ inv1 [inv] proposed equipment work without calling cmms.work_orders first → tool_guidance.md
◆ probe inv1#0: hand_arithmetic unchanged, no_analytic_move mitigated, no_cmms mitigated
✎ dom1 [dom] diagnosed without comparing the condenser approach with the manual's limit → skills/chiller_fault_review/SKILL.md, systemprompt.md
◆ probe dom1#0: hand_arithmetic unchanged, no_analytic_move mitigated, no_cmms mitigated
→ compose: inv1: keep tool_guidance.md — no_cmms mitigated in its probe
→ compose: dom1: keep skills/chiller_fault_review/SKILL.md, systemprompt.md — no_analytic_move mitigated in its probe
```

อ่านจากบนลงล่าง แบบที่ composer อ่าน

- **Reference** trajectory ไม่มีการเรียก `cmms.work_orders` agent จึงไม่เห็น PM ที่เปิดอยู่ มันยังรวม kW และ ton ด้วยมือด้วย kW/ton ที่ได้คือ 0.201 ซึ่งผิด (มันหารด้วย 3.517 เกินไปหนึ่งครั้ง) แต่ไม่มี editor ไหนรู้ได้ เพราะไม่มีเฉลย สิ่งที่ editor *เห็น* ได้คือบรรทัด `summed kW and tons by hand`
- **Propose** editor INV ดูแลว่า agent ทำงานอย่างไร จึงเพิ่ม tool guidance หนึ่งบรรทัด: ตรวจ work order ก่อนเสนองานกับอุปกรณ์ใด ๆ editor DOM ดูแลสิ่งที่ผู้ปฏิบัติงานรู้ จึงเพิ่ม skill `chiller_fault_review` พร้อม trigger line หนึ่งบรรทัดใน `systemprompt.md`
- **Probe** ตอนนี้ probe ทั้งสองตัวเรียก `cmms.work_orders` เหมือนใน EXAMPLE walk-through ที่ candidate ทั้งสองตัวทำการตรวจประวัติการซ่อมบำรุง skill เองก็ระบุการตรวจนี้ไว้ด้วย
- **Compose** ความล้มเหลวที่การแก้แต่ละอันตั้งใจแก้ ถูก mitigate ใน probe ของมันเอง และไม่มีอะไร regress จึงเก็บไว้ทั้งสองอัน

มีสองบรรทัดที่ควรดูซ้ำ `hand_arithmetic unchanged` ปรากฏเพราะ **ไม่มี editor ไหนเล็งมัน** editor ของ SIM พลาดความล้มเหลวไป 25% ของครั้ง และครั้งนี้เป็นหนึ่งในนั้น "unchanged" ไม่ใช่ "regressed" จึงไม่ขวางการ merge ความล้มเหลวนั้นแค่ยังไม่ถูกแก้ และ `inv1` "mitigated" `no_analytic_move` ทั้งที่ไม่ได้แตะมัน: เมื่อ agent เจอ work order ที่เปิดอยู่ มันก็ไม่เสนออะไรใหม่ และการสังเกตนั้นเกิดเฉพาะเมื่อมี proposal contrast บันทึกพฤติกรรม ไม่ได้บันทึกสาเหตุ Module 04 เปลี่ยนเรื่องนี้เป็น verdict `confounded`

จากนั้น lab วาง EXAMPLE walk-through ไว้ข้างก้าวของ SIM

**Expected output** (captured on this Mac, SIM)

```
│ stage          EXAMPLE walk-through                         this lab (SIM, seed 0)                          
│ ─────────────  ───────────────────────────────────────────  ────────────────────────────────────────────────
│ reference      no CMMS call                                 no cmms.work_orders call; kW/ton summed by hand 
│ INV candidate  guidance line + scripts/open_work_orders.py  guidance line only (missed hand arithmetic)     
│ DOM candidate  skills/chiller_fault_review                  skills/chiller_fault_review + trigger line      
│ probes         both add the maintenance check               both call cmms.work_orders                      
│ composer       merges both                                  keeps both                                      
│ cue            maintenance state can contradict the alarm   a fault review where the practitioner's compari…
│ always-on Δ    +0 (a sentence was replaced)                 +43 (the kit only appends)                      
```

ขั้นตอนตรงกัน รายละเอียดไม่ตรง และนั่นไม่เป็นไร walk-through เป็น EXAMPLE design ส่วน lab เป็นการรัน SIM editor ของ kit แค่ต่อบรรทัดเพิ่ม always-on text จึงโตขึ้น 43 token ในที่นี้ ขณะที่ editor ใน EXAMPLE จ่ายค่าบรรทัดใหม่ด้วยการแทนประโยคเก่า กฎของ prompt ที่ว่า "if you add always-on text, cut something else" (ถ้าเพิ่ม always-on text ให้ตัดอย่างอื่นออก) คือกฎที่ kit ไม่ได้จำลอง

✓ Checkpoint: คุณชี้ได้ว่าการแก้แต่ละอันที่ถูกเก็บไว้ตอบบรรทัดไหนของ trajectory และบอกได้ว่าทำไม `hand_arithmetic unchanged` ไม่ได้หยุดการ merge

## 4 · Edit row และ entry ใน EXPERIENCE.md

หลังจบก้าว reflection เขียนบันทึกสองอย่าง อย่างแรกคือ **edit row** หนึ่งแถวต่อหนึ่งการแก้ พร้อม verdict: `helped`, `hurt`, `no observed effect` หรือ `confounded` จากนั้นคือบทเรียน **ไม่เกินหนึ่งรายการ**:

> “at most one durable entry, and only if the edit rows support it” — `harness2/core/prompts.py` (entry ถาวรไม่เกินหนึ่งรายการ และต้องมี edit row รองรับ)

entry มีสี่ field `cue` ระบุ "the SHAPE OF SITUATION the lesson applies to, so it can fire on a different matter" (รูปแบบของสถานการณ์ที่บทเรียนใช้ได้ เพื่อให้ใช้ได้กับเรื่องอื่น — `prompts.py`) `lesson` ระบุการกระทำและ component `evidence` อ้าง edit row `confidence` เป็น high, medium หรือ low

**Expected output** (captured on this Mac, SIM)

```
┌─ EXPERIENCE.md
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
✓ wrote 02_propose_probe_compose/.runs/lab02_2/EXPERIENCE.md — no grader verdict or measurement word in it
```

- kit เลือก row ของ skill เป็นบทเรียน เพราะมันให้ความสำคัญกับ component ที่มีชื่อ (C6, C7, C8) มากกว่า always-on text ส่วน EXAMPLE walk-through เลือก cue เรื่องการตรวจการซ่อมบำรุง ทั้งสองแบบมี row ที่เป็น `helped` รองรับ
- เหตุผลของ reviewer ปรากฏในฐานะ **เนื้อหา** ที่ยกมา ไม่เคยเป็นคำสั่ง Module 08 ต่อยอดจากไฟร์วอลล์นี้
- confidence เป็น "medium" เพราะเป็น task เดียว kit เขียน medium เสมอ Lab 02-1 เขียน "high" ด้วยมือ เพราะ script ให้ผลแน่นอน และ contrast เกิดซ้ำบน ledger สองไฟล์ Module 05 อ่านสัญญาทั้งหมด
- คำเตือนของ paper ใช้กับทุก entry: "Adoption alone never establishes that an edit helped or will transfer." (การถูกเลือกใช้อย่างเดียวไม่ได้พิสูจน์ว่าการแก้ช่วย หรือจะใช้ได้ที่อื่น — `prompts.py`)

ตอนท้ายสุด lab จึงถาม `bench.judge`: reference ได้ 40 และ run ที่ compose แล้วได้ 100 task เดียว seed เดียวคือการสาธิต ไม่ใช่การวัด

✓ Checkpoint: คุณบอกชื่อสี่ field ของ `EXPERIENCE.md` ได้ และบอกได้ว่าอะไรตัดสินระหว่าง entry กับ `NO_ENTRY`

## 5 · KEEP_BASE เป็นคำตอบจริง

repository บอก composer ของมันตรง ๆ ว่าการไม่ทำอะไรอาจเป็นสิ่งที่ถูก:

> “KEEP_BASE is a real answer, not a failure. Adaptation that the evidence does not argue for is a regression: it costs context and buys nothing.” — `harness2/core/prompts.py` (KEEP_BASE เป็นคำตอบจริง ไม่ใช่ความล้มเหลว การปรับที่หลักฐานไม่รองรับคือการถอยหลัง เสีย context และไม่ได้อะไรกลับมา)

Lab 02-3 ทดสอบว่า loop ของคอร์สเชื่อฟังกฎนี้หรือไม่ มันรันหนึ่งก้าวบน **quiet** task ทุกตัวของ refine half (ไม่มีกับดัก base harness จัดการได้อยู่แล้ว) และบน **trap** task ทุกตัว สาม seed และ executor ทั้งสองแบบ loop ไม่เคยอ่านว่าตัวไหนเป็นแบบไหน `quiet()` เป็น metadata ของ kit ไว้ใช้นับเท่านั้น

**Expected output** (captured on this Mac, SIM)

```
│ executor (+ its editor)  seed  KEEP_BASE on quiet  KEEP_BASE on trap
│ ───────────────────────  ────  ──────────────────  ─────────────────
│ sim-cloud                0     5/5                 2/12             
│ sim-cloud                1     5/5                 3/12             
│ sim-cloud                2     5/5                 4/12             
│ sim-local                0     5/5                 7/12             
│ sim-local                1     5/5                 4/12             
│ sim-local                2     5/5                 2/12             
◆ KEEP_BASE rate: quiet 30/30 = 100% · trap 22/72 = 31%
✓ on quiet tasks the loop never edits: no observation, no proposal, no new version
```

quiet task ได้ KEEP_BASE ทุกครั้ง: ไม่มีอะไรผิดใน trajectory จึงไม่มี editor ไหนเสนออะไร แต่ก้าวบน **trap** task ก็ได้ KEEP_BASE ถึง 31% lab จึงถามว่าทำไม

**Expected output** (captured on this Mac, SIM)

```
│ why a trap task got KEEP_BASE (3 seeds)  sim-cloud  sim-local
│ ───────────────────────────────────────  ─────────  ─────────
│ reference showed nothing                 7          5        
│ editors proposed nothing                 1          6        
│ edits proposed, none supported           1          2        
```

- **Reference ไม่แสดงอะไร** บน F1-00 และ F1-03 ที่ seed 0 agent บังเอิญเรียก `cmms.work_orders` (sim-cloud ทำแบบนั้น 45% ของครั้งบน v1) reference หนึ่งตัวคือการสุ่มหนึ่งครั้ง จุดอ่อนยังอยู่ แค่ไม่ได้แสดงออกมา
- **Editor ไม่เสนออะไร** editor พลาดสิ่งที่ trajectory แสดง editor ของ sim-local ที่อ่อนกว่าพลาดบ่อยกว่า (40% เทียบกับ 25%)
- **ไม่มีอันไหนได้รับการรองรับ** probe ไม่เปลี่ยนอะไร หรือทำให้บางอย่าง regress composer จึงทิ้งการแก้นั้น

ดังนั้น KEEP_BASE ตอบคำถามว่า "หลักฐาน *ชุดนี้* สนับสนุนการเปลี่ยนแปลงไหม" ไม่ใช่ "harness สมบูรณ์แล้วหรือยัง" reference ที่มากขึ้น (k > 1) หรือการรันซ้ำจะเผยกรณีที่ซ่อนอยู่ Module 04 และ 06 วัดเรื่องนี้

lab จบด้วยการบังคับใส่ skill ที่ไม่มีหลักฐานรองรับลงใน quiet task F1-10 composer ทิ้งมัน และตารางแสดงว่าถ้าปล่อยออกไปจะเสียอะไร:

**Expected output** (captured on this Mac, SIM)

```
◆ reference observed: nothing wrong · probe contrast: no difference
→ compose: dom1: drop skills/chiller_fault_review/SKILL.md, systemprompt.md — no_analytic_move: no observed change
→ compose: KEEP_BASE — the evidence does not argue for any change
│ outcome                         always-on ≈tokens  prefix hash   files
│ ──────────────────────────────  ─────────────────  ────────────  ─────
│ KEEP_BASE (composer)            245                ee3bb0d97ccb  4    
│ if the forced edit had shipped  264                efa3d6dfa285  5    
✓ the composer dropped it: an edit whose probe changed nothing buys nothing, and would still cost +19 always-on tokens and one cold prompt cache
```

สำหรับ agent แบบ always-on อย่าง Ops Assistant นี่คือความต่างระหว่างคืนที่เงียบกับคืนที่วุ่นวาย ใน EXAMPLE design ของคอร์ส ผล KEEP_BASE ไม่สร้าง draft version และไม่แจ้งเตือน loop ที่แก้ทุกครั้งจะปลุก reviewer ทุกคืน และทำ prompt cache พังโดยไม่ได้อะไร

✓ Checkpoint: คุณให้เหตุผลได้สองแบบว่าทำไม trap task จึงจบด้วย KEEP_BASE ได้ และบอกได้ว่าทำไมทั้งสองแบบไม่ใช่ bug ของ KEEP_BASE

## Labs — รันได้ที่นี่

**labs/lab02_1_the_demo_by_hand.py** — แนวคิดของ demo ใน repository ทำใหม่ด้วยโค้ดของคุณเอง: ledger, bug ตัวกรอง, reference, การแก้, probe, compose, EXPERIENCE.md และ ledger ที่ได้ KEEP_BASE

**labs/lab02_2_one_step_on_f1_01.py** — หนึ่งก้าวเต็มที่ k = 1 บน F1-01 แสดงทุก trajectory, หลักฐานที่ composer ใช้, edit row และ EXPERIENCE.md ที่เขียนไว้ใต้ .runs/

**labs/lab02_3_keep_base_honesty.py** — หนึ่งก้าวบน quiet task และ trap task ทุกตัว (สาม seed, สอง executor): อัตรา KEEP_BASE, ทำไม trap task ได้มัน และการแก้แบบบังคับจะเสียอะไร

## ลองทำเอง

`exercises/ex02_compose_by_hand.py` ให้ตาราง contrast ห้าตาราง และให้คุณเป็น composer มี TODO สามข้อ:

1. สำหรับ candidate ทุกตัว เขียน `("keep" | "drop", reason)`
2. สำหรับทุกกรณี เขียนผลของก้าวนั้น: `COMPOSED` หรือ `KEEP_BASE`
3. ในกรณี D บอกชื่อความล้มเหลวที่ regress ซึ่ง composer แค่ *บันทึก* ไว้ว่าอาจเป็นความแปรปรวนของการ execution

```bash
# on: laptop
.venv/bin/python week31_harness2_recursive_harnessing/02_propose_probe_compose/exercises/ex02_compose_by_hand.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ every keep/drop decision and its reason match the composer's rule
✓ outcomes right: B and E are KEEP_BASE — no edit earned its place
✓ no_evidence in D is noted as possible variance: stochastic, and no edit touched it

═ Done. Module 03 asks WHERE each kept edit should live: the cheapest effective component.
```

<details><summary>คำใบ้ — TODO 1</summary>

ตรวจสามอย่างตามลำดับ probe ส่งงานไหม ถ้าไม่: drop, "not delivered" (composer prompt ของ repository: "Never merge an edit from a non-delivering rollout without stating why") เป้าหมาย *ของการแก้นั้นเอง* เป็น "mitigated" ไหม ถ้าไม่: drop, "no observed change" candidate ทำให้ความล้มเหลวที่ถูกเล็ง หรือความล้มเหลวแบบ deterministic (hand_arithmetic, no_reopen, blind_retry, no_deliverable) regress ไหม ถ้าใช่: drop, "regressed" ถ้าผ่านหมด: keep, "mitigated"

</details>

<details><summary>คำใบ้ — TODO 3</summary>

กรณี D มีสองรายการที่เป็น "regressed" อันหนึ่งคือ `hand_arithmetic` ซึ่งเป็นความล้มเหลวแบบ deterministic ถ้ามันปรากฏ แปลว่ามีอะไรเปลี่ยนจริง อีกอันเป็นความล้มเหลวแบบสุ่มที่ไม่มีการแก้ไหนใน candidate นั้นเล็งไว้ probe เดียวแยกผลของการแก้ออกจากโชคไม่ได้ composer จึงเขียนหมายเหตุไว้และเก็บการแก้ไว้ Module 04 แสดงว่าการรันซ้ำตัดสินเรื่องนี้อย่างไร

</details>

อยากลองต่อ? รัน Lab 02-2 ใหม่โดยใส่ `seed=2` ในการเรียก `h2step(...)` editor INV จะจับการคำนวณด้วยมือได้ด้วย และเพิ่ม `scripts/energy_units.py` แล้ว edit row หนึ่งแถวจะออกมาเป็น `confounded` อ่านว่าทำไมก่อนที่ Module 04 จะอธิบาย

✓ Checkpoint: ตัวตรวจขึ้น ✓ ครบทั้งสามบรรทัด

## การแก้ปัญหา

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `harness2-demo: command not found` | คอร์สนี้ไม่เคยติดตั้ง repository | รัน demo ใน clone ของคุณเองตามที่ facilitator guide อธิบาย module นี้ไม่ต้องใช้มันเลย |
| Lab 02-1 ล้มด้วย `FileNotFoundError` ที่ `.runs/` | โฟลเดอร์ของ module เป็นแบบอ่านอย่างเดียว | รันจาก checkout ที่เขียนได้ lab เขียนเฉพาะใต้ `02_propose_probe_compose/.runs/` |
| จำนวน KEEP_BASE ต่างจากหน้านี้ | kit ถูกแก้ หรือ seed ต่างกัน | `git diff week31_harness2_recursive_harnessing/common` tree + task + seed เดียวกันให้ก้าวเดียวกัน |

## ถัดไป

[Lab 03 — Two scopes and the cheapest effective component](../03_two_scopes_cheapest_component/TUTORIAL.md): ดูว่าทำไม loop ใช้ editor สองตัวแทนตัวเดียว และทำไมแต่ละตัวจึงหยิบ skill, script หรือ plugin ก่อนจะเขียน always-on prose แม้แต่บรรทัดเดียว
