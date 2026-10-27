# ▶ Harness Lab 01 — Agent = Model + Harness: ส่วนประกอบทั้งแปดและต้นทุนของแต่ละส่วน

> ส่วนหนึ่งของ Week 31 · Harness²: agent ที่ปรับปรุง harness ของตัวเอง คุณจะปรับปรุง harness ของ agent จำลองหนึ่งตัว คือ **HVAC Expert agent ของ Ops Assistant** (ผู้ช่วยด้านปฏิบัติการแบบ always-on ที่สมมติขึ้น) จากหลักฐานการทำงานจริง และเห็นผลลัพธ์จริง ทุกอย่างเป็นข้อมูลสังเคราะห์ (โรงแรมสมมติสองแห่ง alarm และ work order ที่แต่งขึ้น) ทุก lab จึงฟรี ออฟไลน์ และปลอดภัย Lab ทำงานบน **🧠 SIM** เป็นค่าเริ่มต้น คือโมเดลสอนแบบ deterministic ของ agent และ editor ของมัน ถ้า lab ไหนรองรับ ให้สลับ runner เป็น **🦙 LIVE** แล้วโมเดล local จริงจะเล่นบทหนึ่งบท

**สิ่งที่คุณจะได้ลงมือทำ**
- อ่านนิยามของ paper ว่า "Agent = Model + Harness" และดูว่านิยามนี้ตั้งใจเว้นอะไรไว้ นั่นคือ weights
- จัดกลุ่ม component ทั้งแปดของ Harness² ตามวิธีที่มันเข้าถึงโมเดล และรู้ว่าตัวไหนต้องจ่ายทุกครั้งที่เรียกโมเดล
- วัดบน tree จริงว่ากฎหนึ่งข้อมีต้นทุนเท่าไรเมื่อเขียนเป็น always-on prose เทียบกับเมื่อเป็น skill และทำไมการแก้ทั้งสองแบบทำให้ prompt cache เย็นหนึ่งครั้ง
- รันโมเดลจำลองตัวเดียวกันกับ harness สามแบบ แล้วดูพฤติกรรมเปลี่ยน ทั้งที่โมเดลไม่เปลี่ยน
- จัดทุกชิ้นของ harness ของ Ops Assistant ลงใน component และหากฎหนึ่งข้อที่ต้องอยู่นอก component ทั้งหมด

**Time** ~35 นาที · **Difficulty** beginner · **Hardware** แล็ปท็อปเครื่องนี้เท่านั้น (Ollama ไม่บังคับ)

**Sources:** [Harness² paper, arXiv 2610.14224](https://arxiv.org/html/2610.14224) · [google-research/harness2](https://github.com/google-research/harness2) (Apache-2.0; snapshot ของ `prompts.py`, `tree.py`, `README.md` อยู่ใน `common/sources/`) · prompt caching จาก Week 2, skill จาก Week 7, always-on agent จาก Week 17

## 0 · ก่อนเริ่ม

| ต้องมี | ตรวจอย่างไร | ทำไม |
|---|---|---|
| Python ของ repo นี้ | `.venv/bin/python --version` → 3.13 | ใช้รันทุก lab และ Harness Lab Runner |
| ไม่ต้องติดตั้ง package เพิ่ม | kit ใช้แค่ standard library | loop, agent จำลอง และ judge เป็น Python ธรรมดา |
| Ollama (ไม่บังคับ) | `ollama list` มี `gemma4:12b` | ใช้เฉพาะ lab แบบ 🦙 LIVE (Module 04, 05, 08) ทุก lab ทำงานได้โดยไม่ต้องมี |

```bash
# on: laptop
.venv/bin/python week31_harness2_recursive_harnessing/common/h2out.py
```

**Expected output** (captured on this Mac, SIM)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
━━ h2out self-check
   which mode, which local models
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
◈ 🧠 SIM · deterministic teaching model of an agent and its editors (switch to 🦙 LIVE in the runner)
◆ H2_BRAIN=(unset → sim) · model=gemma4:12b
◆ Ollama local models: scb10x/typhoon-ocr1.5-3b:latest, hf.co/ggml-org/embeddinggemma-2-GGUF:Q8_0, nemotron-3.5-lightning:latest, nemotron-3-nano:latest, gemma3:4b, gemma4:12b, gemma4:latest
```

> 📌 **SIM คืออะไร และไม่ใช่อะไร** คอร์สนี้ไม่เคยรันโค้ดของ google-research/harness2 โฟลเดอร์ `common/` คือการเขียน method ขึ้นใหม่ของคอร์สเองด้วย Python ธรรมดา บน agent ของ Ops Assistant ที่ **จำลอง** ขึ้นและมีจุดอ่อนที่รู้อยู่แล้ว ตัวเลขของมัน (agent ทำตาม prose บ่อยแค่ไหน skill ถูกเรียกใช้บ่อยแค่ไหน) เป็น **สมมติฐานที่เขียนไว้เพื่อแสดงกลไก** เมื่อ lab แสดงว่า skill ดีกว่า prose มันแสดงว่ากลไกทำงานอย่างไร ไม่ได้พิสูจน์ว่ามันได้ผลจริง หลักฐานเป็นของ paper และ Module 06 จะอ่านมัน

✓ Checkpoint: self-check พิมพ์บรรทัด SIM ออกมา และคุณรู้ว่ามี Ollama ให้ใช้กับ lab แบบ LIVE หรือไม่

## 1 · Agent = Model + Harness

Paper เริ่มจากสมการเดียว README ของ repository นิยามชั้นที่สมการนี้ตั้งชื่อไว้:

> “the system around a model that enables it to act as an agent: prompts, tools, memory, skills, orchestration, and more” — `harness2/README.md` (ระบบรอบโมเดลที่ทำให้โมเดลทำงานเป็น agent ได้: prompt, tool, memory, skill, การประสานงาน และอื่น ๆ)

ทุกอย่างที่อยู่รอบโมเดลที่ถูกแช่แข็งไว้คือ harness weights ของโมเดลไม่เคยเปลี่ยน **Harnessing** คืองานปรับปรุงชั้นนี้ Harness² ถามว่างานนี้ปรับปรุงตัวเองได้ไหม ทีละ task โดยไม่มีใครบอกคำตอบที่ถูก

คุณแก้ harness มาตลอดคอร์สโดยไม่รู้ชื่อ skill ที่คุณเขียนใน Week 7 คือ component ของ harness กฎ retry ของ Week 17 ก็ใช่ policy ที่ลงนามใน Week 23 ก็ใช่ hook จาก Week 29 ก็ใช่ สิ่งที่ใหม่ในสัปดาห์นี้คือการมองทั้งหมดเป็น **object เดียวที่มี version** ซึ่ง loop แก้ไขได้ โดยทุกการแก้ต้องมีหลักฐาน

| | Fine-tune โมเดล | แก้ harness |
|---|---|---|
| อะไรเปลี่ยน | weights | ไฟล์ข้อความและโค้ดรอบโมเดล |
| การรีวิว | ยาก: diff ของตัวเลข | ง่าย: diff ของไฟล์ที่คนอ่านได้ |
| การย้อนกลับ | train ใหม่ หรือเก็บ checkpoint เก่าไว้ | ชี้กลับไปที่ parent version |
| ใช้กับโมเดลอื่นได้ไหม | ไม่ได้ | มักได้ แต่ไม่สม่ำเสมอ (Module 06) |
| ใครทำได้ | training pipeline | reviewer หรือ loop ที่มีด่านอนุมัติโดยคน |

✓ Checkpoint: คุณบอกได้ว่าอะไรคงที่ใน "Agent = Model + Harness" (weights ของโมเดล) และ loop ของการ harness ได้รับอนุญาตให้เปลี่ยนอะไร (ไฟล์รอบโมเดล)

## 2 · Component ทั้งแปด แบ่งตามวิธีที่เข้าถึงโมเดล

Harness² ตั้งชื่อ component ที่แก้ไขได้ไว้แปดตัว repository จัดวางเป็นไฟล์ (layout แบบ OpenCode) สิ่งที่สำคัญคือ **แต่ละตัวเข้าถึงโมเดลอย่างไร** เพราะนั่นคือสิ่งที่กำหนดต้นทุน

```
                 ┌──────────────── always on: in context on EVERY call ────────────────┐
  frozen model ◀─┤ C1 systemprompt.md · C2 guardrails.md · C3 LongTermMEMORY.md · C4 tool_guidance.md │
                 └──────────────────────────────────────────────────────────────────────┘
                 ┌──────────────── on demand: cost only when used ─────────────────────┐
                 │ C6 skills/<name>/SKILL.md   loaded by the skill tool when its trigger fires │
                 │ C7 scripts/<name>.py        run via bash; the model reads only the output   │
                 │ C5 sub_agents/<name>/AGENT.md  a worker in its OWN context; returns a result │
                 └──────────────────────────────────────────────────────────────────────┘
                 ┌──────────────── in the tool loop: code, not prose ──────────────────┐
                 │ C8 plugin/<name>.ts         a hook around tool calls; enforces, never asks   │
                 └──────────────────────────────────────────────────────────────────────┘
```

สองบรรทัดจาก editor prompt ใน repository อธิบายปลายทั้งสองข้างของภาพนี้:

> “An on-demand component costs nothing until the moment it is invoked.” — `harness2/core/prompts.py` (component แบบ on-demand ไม่มีต้นทุนจนกว่าจะถูกเรียกใช้)

> “Prose can be stopped reading; a hook cannot.” — `harness2/core/prompts.py` (prose ถูกอ่านข้ามได้ แต่ hook ข้ามไม่ได้)

ดังนั้น plugin จึงเป็น component เดียวที่ **บังคับ** พฤติกรรม component อื่นทุกตัว **สั่ง** โมเดล แล้วหวังให้โมเดลทำตาม

✓ Checkpoint: คุณบอกชื่อ component แบบ always-on ทั้งสี่ได้ และบอกได้ว่า component ตัวเดียวที่บังคับพฤติกรรมแทนการขอร้องคือ C8 (plugin)

## 3 · always-on มีต้นทุนเท่าไร: token และ prompt cache

Lab 01-1 เอากฎความยาว 50 คำข้อเดียวกันไปไว้สองที่ แล้ววัดทั้งสองแบบ

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 3 · One rule, two homes: always-on prose versus an on-demand skill
│ harness                           always-on ≈tokens  prefix hash   vs base
│ ────────────────────────────────  ─────────────────  ────────────  ───────
│ v1 (base)                         245                ee3bb0d97ccb  same   
│ rule in systemprompt.md           321                6d2bba4898ca  CHANGED
│ rule in a skill + 1 trigger line  260                9006e0d07ecd  CHANGED
◆ prose costs +76 tokens on EVERY call; the skill costs +15 always-on (its trigger line) plus 91 tokens only on runs that load it
⚠ both edits change the prefix hash: any harness change is a cold prompt cache once — so ship versions at a deployment boundary, never mid-day
```

ตารางนี้ซ่อนต้นทุนไว้สองอย่าง

1. **Token ทุกครั้งที่เรียก** กฎแบบ prose เพิ่ม 76 token ให้ทุกการเรียกโมเดล รวมถึงการเรียกจำนวนมากที่ไม่เกี่ยวกับ chiller เลย ส่วน skill เพิ่มแค่ trigger line 15 token และเนื้อหาของมันโหลดเฉพาะตอนที่ chiller alarm เกิดขึ้น จำนวน token เป็น **ESTIMATE** (จำนวนตัวอักษร ÷ 4) แม่นพอสำหรับเปรียบเทียบ version แต่ไม่พอสำหรับคิดเงิน
2. **Prompt cache** cache ใช้ always-on prefix ที่ตรงกันทุกตัวอักษรเป็น key Week 2 สอนว่า cache miss มีต้นทุนเป็นเงินจริง agent แบบ always-on จึงควรปฏิบัติต่อ cache miss เหมือน bug บนฮาร์ดแวร์ local prefix ที่ cache เย็นต้อง prefill ใหม่ก่อนจะได้ token แรก และ prefix ยาว ๆ ใช้เวลาหลายวินาที การแก้ always-on text **ทุกครั้ง** เปลี่ยน prefix hash ดังนั้น cache จะเย็นหนึ่งครั้ง นี่คือเหตุผลที่ EXAMPLE design ของคอร์สนี้ปล่อย harness version เฉพาะที่ deployment boundary

editor prompt ของ repository ชี้เรื่องเดียวกันจากฝั่งคุณภาพ:

> “a paragraph buried in the always-on text is paid for on every run and travels badly” — `harness2/core/prompts.py` (ย่อหน้าที่ฝังอยู่ใน always-on text ต้องจ่ายทุก run และพกไปใช้ที่อื่นได้ไม่ดี)

✓ Checkpoint: คุณอธิบายได้ว่าทำไม skill บวก trigger line หนึ่งบรรทัดถูกกว่ากฎเดียวกันที่เขียนเป็น prose และทำไมทั้งสองแบบยังทำให้ cache เย็นหนึ่งครั้ง

## 4 · Ops Assistant agent จำลอง: โมเดลเดิม harness ต่างกัน

Lab 01-2 รัน agent จำลองกับ proposal ที่ chief engineer ปฏิเสธพร้อมเหตุผล โมเดลยังเป็นตัวเดิม มีแค่ไฟล์รอบตัวมันที่เปลี่ยน

**Expected output** (captured on this Mac, SIM)

```
│ harness              always-on  cmms checked  by hand  score*  failed checks*                              
│ ───────────────────  ─────────  ────────────  ───────  ──────  ────────────────────────────────────────────
│ v1 base              245        False         True     40.0    no duplicate work order, names the open wor…
│ v1 + skill           264        True          True     100.0   —                                           
│ v1 + skill + script  287        True          False    100.0   —                                           
◆ * measured AFTER the run by bench.judge, which reads the hidden truth — nothing in the loop ever sees it
```

อ่านสามแถวจากซ้ายไปขวา

- **v1 base** ไม่เคยเรียก `cmms.work_orders` จึงเสนอ work order ใหม่ให้ chiller ที่อยู่ระหว่าง PM อยู่แล้ว ตรงกับเหตุผลที่ chief engineer ให้ไว้ทุกประการ
- **+ skill** (C6) บอกลำดับงานของผู้ปฏิบัติงาน: ตรวจประวัติการซ่อมบำรุงก่อนเสนออะไร ตอนนี้ agent เจอ work order ที่เปิดอยู่ แต่ยังรวม kW และ ton ด้วยมือ และรอบนี้ได้หน่วยถูกเพราะโชคช่วย
- **+ script** (C7) ย้ายการคำนวณไปไว้ใน `scripts/energy_units.py` agent รัน script แทนการคิดตัวเลขทีละ token คำตอบจึงไม่ขึ้นกับโชคอีกต่อไป

คอลัมน์ score มีดอกจันด้วยเหตุผล `bench.judge` อ่านความจริงที่ซ่อนไว้ของแต่ละ task **หลังจาก** run เสร็จ ไม่มีสิ่งใดที่ editor หรือ agent เห็นมี score อยู่ Module 04 อธิบายว่าทำไม loop ต้องไม่เห็น score เลย

ตารางที่สองใน lab รัน executor สองแบบบน harness v1 `sim-local` แทนโมเดล open-weight ที่เล็กกว่าและอ่อนกว่าซึ่งรันบนฮาร์ดแวร์ local มันทำตาม prose น้อยกว่า และคำนวณเองบ่อยกว่า

**Expected output** (captured on this Mac, SIM)

```
│ executor   mean score*  min  max  
│ ─────────  ───────────  ───  ─────
│ sim-cloud  73.0         0.0  100.0
│ sim-local  69.6         0.0  100.0
```

ในการรันนี้ช่องว่างเล็ก (73.0 เทียบกับ 69.6 จาก 15 run ต่อ executor) จึงควรอ่านเป็นทิศทาง ไม่ใช่ขนาด ผลได้ที่มากที่สุดใน paper เกิดกับ executor ที่อ่อนกว่า Module 06 แสดงรูปแบบเดียวกันใน simulator

✓ Checkpoint: คุณชี้ได้ว่าบรรทัดไหนใน trajectory เปลี่ยนระหว่าง v1 กับ v1 + skill (`cmms.work_orders('CH-2')` ปรากฏขึ้น) และบอกได้ว่า component ไหนเป็นสาเหตุ

## 5 · Harness ของ agent แบบ always-on วันนี้ จัดลงใน component ทั้งแปด

agent ที่ใช้งานจริงอย่าง Ops Assistant มักมี harness อยู่แล้วเกือบทั้งหมด เพียงแต่ชิ้นส่วนกระจายอยู่ในหลายระบบที่ไม่รู้ว่าตัวเองรวมกันเป็น object เดียว ตารางข้างล่างคือ EXAMPLE design ของคอร์สว่าชิ้นส่วนเหล่านั้นอยู่ที่ไหน และแต่ละชิ้นยังขาดอะไร

| Component | สิ่งที่ Ops Assistant มีวันนี้ (EXAMPLE) | ช่องว่าง |
|---|---|---|
| C1 System prompt | base prompt ต่อ agent (HVAC Expert, Facility Ops, Energy Analyst) | ไม่ได้ทำ version เป็น object เดียวที่มี hash; developer แก้เอง ไม่เคยแก้จากหลักฐาน |
| C2 Guardrails | ขอบเขต propose-only, connector trust plane, budget | ขอบเขต **แข็ง** อยู่ในโค้ด ซึ่งถูกต้อง แต่ task constraint ไม่มีที่อยู่ |
| C3 Memory | semantic และ episodic memory ที่ประกาศไว้; knowledge graph | เขียนโดย sweep ไม่ได้คัดตามหลัก "จ่ายทุก run" |
| C4 Tool guidance | MCP manifest ที่ pin ไว้ | ไม่มี "เมื่อไรและอย่างไร" ต่อ tool ที่ปรับปรุงจาก trace |
| C5 Subagents | supervisor และ review jury; helper workflow ที่วางแผนไว้ | helper เป็นโค้ด ไม่ใช่ไฟล์ `AGENT.md` ที่แก้ได้ |
| C6 Skills | skill ที่มี version พร้อม `definition_hash` และ publish lock | คนเขียน ไม่มี loop ที่เสนอ skill จากความล้มเหลว |
| C7 Scripts | watcher, การเรียก classifier, physics-in-code, การแปลงหน่วย | ถูกมองเป็น platform code ไม่มีเจ้าของเรื่อง "รูปแบบที่ซ้ำกลายเป็น tool" |
| C8 Plugins | auto-review ที่วางแผนไว้, envelope ของ `send_to_user`, compaction hook | ยังไม่ได้ประกาศเป็น component |

Lab 01-1 จบด้วยการลองแก้กฎการอนุมัติใน `guardrails.md` tree validator ปฏิเสธ ในคอร์สนี้ block ที่ platform เป็นเจ้าของเป็นส่วนหนึ่งของไฟล์เพื่อให้คุณมองเห็น และ `validate_tree` คุมให้มันเหมือนเดิมทุก byte ในระบบที่ใช้งานจริง มันอยู่ในโค้ด Module 08 แสดงว่าจะเกิดอะไรขึ้นเมื่อ loop *เอื้อมถึง* กฎแบบนี้ได้

**Expected output** (captured on this Mac, SIM)

```
✕ validate_tree: guardrails.md: the platform-owned block was changed — owned by the platform, not editable
✓ the approval boundary is platform-owned: validate_tree refuses any tree that edits it
═ v1 = 4 files · 245 always-on tokens · definition hash 773bf3d26eee
```

✓ Checkpoint: คุณบอกได้ว่าทำไม "คนที่ระบุชื่ออนุมัติทุก physical action" ไม่ใช่หนึ่งใน component ที่แก้ได้ทั้งแปด

## Labs — รันได้ที่นี่

**labs/lab01_1_meet_the_harness.py** — base harness ในรูป tree: component ทั้งแปด, always-on token, prefix hash และกฎหนึ่งข้อในสองที่

**labs/lab01_2_harness_shapes_behaviour.py** — โมเดลจำลองตัวเดียวกันกับ harness สามแบบบน chiller proposal ที่ถูกปฏิเสธ แล้วตามด้วย executor สองแบบบน harness v1

## ลองทำเอง

`exercises/ex01_place_the_component.py` มี TODO สามข้อ:

1. จัดแปดชิ้นของ harness ของ Ops Assistant ลงใน C1 … C8
2. ตัดสินว่า "คนที่ระบุชื่ออนุมัติทุก physical action" ต้องอยู่ที่ไหน
3. ย้าย procedure แบบ always-on ไปเป็น skill ที่มี trigger เพื่อให้ always-on token ลดลงและ tree ยัง valid

```bash
# on: laptop
.venv/bin/python week31_harness2_recursive_harnessing/01_agent_model_harness/exercises/ex01_place_the_component.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ all eight pieces placed: C1 … C8, one each
✓ the approval boundary lives in platform code — an editable file could be edited away by the loop
✓ procedure moved into skills/chw_digest · always-on 291 → 260 tokens · tree valid

═ Done. Module 02 lets a loop make edits like TODO 3 from execution evidence instead of by hand.
```

<details><summary>คำใบ้ — TODO 1</summary>

ถามว่าแต่ละชิ้นเข้าถึงโมเดลอย่างไร ข้อความที่อยู่ในทุกการเรียกคือ C1–C4 (persona และ procedure: C1, constraint: C2, ข้อเท็จจริงสั้น ๆ ที่ใช้ได้ทั่วไป: C3, วิธีใช้ tool: C4) worker ที่มี context ของตัวเองคือ C5 procedure ที่โหลดตาม trigger คือ C6 โค้ด deterministic ที่ agent รันคือ C7 hook ใน tool loop คือ C8

</details>

<details><summary>คำใบ้ — TODO 3</summary>

`H.skills(tree)` หาไฟล์ที่ `skills/<name>/SKILL.md` ใส่บรรทัด front-matter `trigger: chilled-water digest` ให้ไฟล์ ใส่ procedure ไว้ในเนื้อหา สร้าง `systemprompt.md` ใหม่จาก `H.BASE_SYSTEMPROMPT` บวกหนึ่งบรรทัดที่ระบุชื่อ skill แล้วตรวจ `H.always_on_tokens(MOVED)`

</details>

✓ Checkpoint: ตัวตรวจขึ้น ✓ ครบทั้งสามบรรทัด

## การแก้ปัญหา

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `ModuleNotFoundError: harness` | lab ถูกรันจากสำเนาที่อยู่นอกโฟลเดอร์ของสัปดาห์ | รันจาก root ของ repo ด้วย path ที่แสดงไว้ หรือรันจาก runner |
| จำนวน token ต่างจาก tutorial | คุณแก้ base file ใน `common/harness.py` | `git diff week31_harness2_recursive_harnessing/common` ผลลัพธ์ที่บันทึกไว้ของทุก module ขึ้นกับ kit |
| score ต่างจาก tutorial | seed ต่างกัน หรือ kit ถูกแก้ | lab ใช้ seed คงที่ tree + task + seed เดียวกันให้ run เดียวกัน |

## ถัดไป

[Lab 02 — One Propose–Probe–Compose step, by hand and by loop](../02_propose_probe_compose/TUTORIAL.md): ให้ loop หาช่องว่างที่คุณเติมด้วยมือใน Lab 01-2 จาก trajectory อย่างเดียว และเรียนรู้ว่าเมื่อไรคำตอบที่ถูกคือ KEEP_BASE
