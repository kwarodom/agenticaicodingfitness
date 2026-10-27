# ▶ Harness Lab 06 — อ่านตารางผลลัพธ์ให้เป็น: streams, noise, ablations, transfer, pass^k

> ส่วนหนึ่งของ Week 31 · Harness²: agent ที่ปรับปรุง harness ของตัวเอง คุณจะปรับปรุง harness ของ agent จำลองหนึ่งตัว คือ **HVAC Expert agent ของ Ops Assistant** (ผู้ช่วยงานปฏิบัติการแบบ always-on ที่สมมติขึ้น) จากหลักฐานการทำงานจริง และเห็นผลลัพธ์จริง ทุกอย่างเป็นข้อมูลสังเคราะห์ (โรงแรมสมมติสองแห่ง alarm และใบสั่งงานที่แต่งขึ้น) ทุก lab จึงฟรี ออฟไลน์ และปลอดภัย Lab ทั้งหมดรันบน **🧠 SIM** เป็นค่าเริ่มต้น คือโมเดลจำลองเพื่อการสอนแบบ deterministic ของ agent และ editor ของมัน สลับ runner เป็น **🦙 LIVE** ได้ใน lab ที่รองรับ แล้วโมเดลจริงบนเครื่องจะรับบทหนึ่งบท

**สิ่งที่คุณจะได้ลงมือทำ**
- อ่านตัวเลขหลักของเปเปอร์แบบที่เปเปอร์รายงาน: ค่าเฉลี่ยของ stream ที่สลับลำดับสาม stream พร้อมค่าเบี่ยงเบนมาตรฐาน (SD)
- เรียนกฎการอ่านหนึ่งข้อ (ทั้งสาม stream ไปทางเดียวกันไหม และช่องว่างกว้างกว่าหนึ่ง SD ไหม) แล้วใช้กับทุกแถว
- รัน base เทียบ Harness² บน executor จำลองทั้งสองตัว แล้วตรวจว่าตัวที่อ่อนกว่าได้ประโยชน์มากกว่าหรือไม่
- สร้างตาราง ablation ของ SIM เอง (ไม่มี library, generic editor, ไม่มี composition, k = 3, k = 5, sequential) แล้วบอกอย่างซื่อตรงว่าแถวไหนตารางรองรับได้และแถวไหนรองรับไม่ได้
- Promote harness version บน executor หนึ่งแล้ววัดบนอีกตัว จากนั้นดูว่าทำไม pass^3 ไม่ใช่ pass@3 ที่ใช้ promote version

**Time** ~45 นาที · **Difficulty** intermediate · **Hardware** แล็ปท็อปเครื่องนี้เท่านั้น (ไม่ต้องใช้โมเดล)

**Sources:** [Harness² paper, arXiv 2610.14224](https://arxiv.org/html/2610.14224) · [google-research/harness2](https://github.com/google-research/harness2) (Apache-2.0; snapshot ของ `README.md`, `EXPERIMENTS.md` อยู่ใน `common/sources/`) · [Anthropic, Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) · evals และ observability ของ Week 10

## 0 · ก่อนเริ่ม

| ต้องมี | ตรวจ | ทำไม |
|---|---|---|
| Python ของ repo นี้ | `.venv/bin/python --version` → 3.13 | ใช้รันทุก lab และ Harness Lab Runner |
| Module 02–05 | รู้จัก step, KEEP_BASE, verdict ของ edit row และ library | โมดูลนี้วัดว่าชิ้นส่วนเหล่านั้นเพิ่มอะไร |
| ไม่ต้องใช้โมเดล | ทุก lab ในโมดูลนี้เป็น SIM อย่างเดียว | ตารางผลลัพธ์เป็นเรื่องของ protocol ไม่ใช่ผลของโมเดลตัวใดตัวหนึ่ง |

ตัวเลขทุกตัวในโมดูลนี้มีป้ายกำกับหนึ่งในสองแบบ **reported in the paper** (รายงานในเปเปอร์) หมายถึงตัวเลขหรือข้อค้นพบจากเปเปอร์ Harness² (arXiv 2610.14224) หรือ repository ของมัน ซึ่ง VERIFIED กับต้นฉบับแล้ว ที่นี่ยกคำพูดแบบคำต่อคำเฉพาะจาก snapshot ของ repository ใน `common/sources/` ข้อค้นพบของเปเปอร์จึงเขียนเป็นข้อความธรรมดา ไม่ใช่คำพูดที่ยกมา **SIM** หมายถึงคุณสร้างตัวเลขนั้นบนแล็ปท็อปนี้ด้วยโมเดลสอนของคอร์ส SIM ฝังข้อค้นพบของเปเปอร์ไว้เป็นสมมติฐาน ตาราง SIM จึงสอนวิธี *อ่าน* ผลลัพธ์ แต่ไม่เคยยืนยันผลของเปเปอร์

```bash
# on: laptop
.venv/bin/python week31_harness2_recursive_harnessing/06_reading_the_results/labs/lab06_1_three_streams.py
```

✓ Checkpoint: Lab 06-1 รันเสร็จในราวหนึ่งวินาทีและพิมพ์หกขั้นตอน

## 1 · เปเปอร์รายงานอะไร และรายงานอย่างไร

ข้อสรุปหลักของเปเปอร์คือ Harness² หนึ่ง step ช่วยทุก configuration ที่ทดลอง ค่าเฉลี่ยมีดังนี้ (as reported in the paper):

| Benchmark | Executor | Base | One step | Gain |
|---|---|---|---|---|
| WorkBuddy-Bench average | Gemini 3.5 Flash / OpenCode | 72.4 | 77.5 | +5.1 |
| WorkBuddy-Bench average | GLM-5 / OpenCode | 64.2 | 75.7 | +11.5 |
| JobBench | Gemini 3.5 Flash | 31.5 | 36.0 | +4.5 |
| LAB | GLM-5 | 59.6 | 71.4 | +11.8 |

ก่อนจะเชื่อตัวเลขที่ดีขึ้น ให้ถามก่อนว่าตัวเลขนั้นได้มาอย่างไร เปเปอร์รายงานคะแนนแต่ละตัวเป็นค่าเฉลี่ยของสาม stream ที่เป็นอิสระต่อกัน และคิด SD จากสามรอบที่สลับลำดับงาน (as reported in the paper)

หนึ่ง stream คือรายการงานทั้งหมดในหนึ่งลำดับ โดยอ่าน library ก่อนแต่ละงานและเพิ่มเข้า library หลังงาน (Module 05) สาม stream ในสามลำดับให้ค่าเฉลี่ยสามค่า การกระจายของค่าเหล่านี้คือ noise ของ protocol คอร์สนี้แปลง SD ของเปเปอร์เองเป็นหลักคิดง่าย ๆ (EXAMPLE): SD ของ base harness บน JobBench ในเปเปอร์คือ 1.1 และ 2.6 คะแนน (as reported in the paper) ดังนั้นความต่างที่น้อยกว่าราว 3 คะแนนบนชุดงาน 30 งานถือเป็น noise เว้นแต่ทั้งสาม stream ไปทางเดียวกัน คอร์สนี้จึงอ่านทุกแถวด้วยสองคำถาม:

1. **ทั้งสาม stream ไปทางเดียวกันไหม?** จับคู่ stream *s* ของ arm หนึ่งกับ stream *s* ของอีก arm (ลำดับงานเดียวกัน) ความต่างทั้งสามมีเครื่องหมายเดียวกันหรือไม่
2. **ช่องว่างกว้างกว่าหนึ่ง SD ไหม?** ผลต่างของค่าเฉลี่ยมากกว่า SD ที่ใหญ่กว่าของสอง arm หรือไม่

ใช่ทั้งคู่: **beyond noise** (เกิน noise) stream ไปทางเดียวกันแต่ช่องว่างอยู่ในหนึ่ง SD: **direction only** (เห็นแค่ทิศทาง ต้องการ stream หรืองานมากขึ้น) stream ไม่ไปทางเดียวกัน: **within noise** (อยู่ใน noise) เกณฑ์ที่คอร์สลงทะเบียนไว้ล่วงหน้า (EXAMPLE design, Module 09) ถามสองคำถามเดียวกันนี้ โมดูลนี้จึงใช้มัน `readkit.py` ในโฟลเดอร์นี้เก็บกฎนี้ไว้ และ Exercise 06 ให้คุณเขียนเอง

ผลอีกสองข้อช่วยวางฉาก (as reported in the paper) executor ที่อ่อนกว่าได้ประโยชน์มากกว่า โมเดลที่อ่อนกว่าแต่มี harness ที่ปรับปรุงแล้วจึงแซงโมเดลที่แข็งกว่าซึ่งใช้ harness เดิมได้ และ editor ไม่จำเป็นต้องเป็นโมเดลที่แข็งกว่า เมื่อใช้โมเดลเดียวกันเป็นทั้ง executor และ editor ทั้งห้าโมเดลที่ทดลอง ตั้งแต่ GLM-5 ถึง Claude Opus 5 ดีขึ้นในทุก benchmark

✓ Checkpoint: คุณบอกได้ว่า "77.5" หมายถึงอะไร (ค่าเฉลี่ยของค่าเฉลี่ยสาม stream) และพูดกฎการอ่านได้ในประโยคเดียว

## 2 · สาม stream ใน SIM: base เทียบ Harness² บน executor ทั้งสอง

Lab 06-1 รัน protocol ของเปเปอร์บน refine task 17 งานของคอร์ส editor ถูกตรึงไว้ (`sim-cloud-editor`) สำหรับ executor ทั้งสอง เหมือนที่เปเปอร์ใช้ default editor ตัวเดียว คะแนนมาจาก `bench.judge` หลังแต่ละ step และ loop ไม่เคยเห็นคะแนน

**วัดบนเครื่อง Mac นี้ — protocol: open-world streams** (deliverable สุดท้ายของแต่ละงานหลัง step ของมันเอง, refine half, อ่าน library ก่อนและเพิ่มหลังภายใน stream, สาม stream ที่สลับลำดับ, mean ± SD)

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 2 · The SIM, same protocol: 17 refine tasks × 3 shuffled streams, base vs one Harness² step per task
│ executor   arm              stream 1 / 2 / 3    mean ± SD*
│ ─────────  ───────────────  ──────────────────  ──────────
│ sim-cloud  base harness v1  81.2 / 83.8 / 75.8  80.3 ± 4.1
│ sim-cloud  Harness² k=1     96.5 / 95.0 / 91.5  94.3 ± 2.6
│ sim-local  base harness v1  76.3 / 69.2 / 71.0  72.2 ± 3.7
│ sim-local  Harness² k=1     86.5 / 80.1 / 88.3  85.0 ± 4.3
◆ * measured after each step by bench.judge (hidden truth); the loop never saw a score

▣ STEP 3 · Read each difference stream by stream: do all three agree, and is the gap larger than one SD?
│ executor   per-stream Δ           Δ mean  larger SD  all agree  |Δ| > SD  reading     
│ ─────────  ─────────────────────  ──────  ─────────  ─────────  ────────  ────────────
│ sim-cloud  +15.3 / +11.2 / +15.7  +14.1   4.1        yes        yes       beyond noise
│ sim-local  +10.2 / +10.9 / +17.3  +12.8   4.3        yes        yes       beyond noise
✓ one step helps both SIM executors beyond stream noise — the same shape as the paper's 'every configuration'
```

SD ของ SIM (2.6 ถึง 4.3) ใหญ่กว่า SD ของ JobBench ในเปเปอร์ งาน 17 งานเป็นชุดเล็ก และความพลาดครั้งเดียวใน stream สั้น ๆ ขยับค่าเฉลี่ยได้หลายคะแนน นี่คือบทเรียนซื่อตรงข้อแรก: **ชุดงานที่เล็กกว่าต้องการช่องว่างที่กว้างกว่า ก่อนที่คุณจะเชื่อมัน**

ต่อไปคือคำถามที่สอง: executor ที่อ่อนกว่าได้ประโยชน์มากกว่าไหม ให้เทียบ gain ไม่ใช่เทียบคะแนน

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 4 · Does the weaker executor gain more? Compare the GAINS, stream by stream
│ gain per stream  sim-cloud +15.3 / +11.2 / +15.7   sim-local +10.2 / +10.9 / +17.3
│ local gain − cloud gain: -1.3 · per stream -5.1 / -0.3 / +1.6 · larger SD of the gains 3.9 → within noise
◆ sim-local with Harness² (85.0) vs sim-cloud on its base (80.3): +4.7 · per stream +5.3 / -3.7 / +12.5 → within noise (the paper reports the weaker executor with an improved harness ahead, for GLM-5)
⚠ the SIM ENCODES 'weaker executors follow prose less and hand-roll more' as an assumption, so this shows the mechanism; it is not evidence for the paper's finding
```

ตารางนี้ไม่แสดงเรื่องนั้น ค่าเฉลี่ย gain ของ local *น้อยกว่า* 1.3 คะแนน และ stream ไม่ไปทางเดียวกัน (−5.1 / −0.3 / +1.6): within noise SIM ฝังกลไกเบื้องหลังข้อค้นพบของเปเปอร์ไว้ แต่งาน 17 งานกับสาม stream มองไม่เห็นมัน และนี่ก็ไม่ใช่หลักฐานค้านเปเปอร์ด้วย ประโยคที่ซื่อตรงคือ "ตารางนี้บอกไม่ได้ว่า executor ไหนได้ประโยชน์มากกว่า"

บรรทัดที่สองถามอีกคำถามของเปเปอร์: executor ที่อ่อนกว่าซึ่งมี harness ที่ปรับแล้ว ชนะ executor ที่แข็งกว่าบน base ของมันไหม ค่าเฉลี่ยนำอยู่ (85.0 เทียบกับ 80.3) แต่ stream 2 ไปอีกทาง (80.1 เทียบกับ 83.8) ตามกฎเดียวกันจึง within noise เช่นกัน

✓ Checkpoint: คุณอธิบายได้ว่าทำไม "−1.3, ต่อ stream −5.1 / −0.3 / +1.6" ได้ป้าย *within noise* และทำไมการนำ 4.7 คะแนนในค่าเฉลี่ยแต่มี stream หนึ่งค้าน จึงได้ป้ายเดียวกัน

## 3 · Same-model editing, การใช้ tool และราคาของ gain

ขั้นที่ 5 ของ Lab 06-1 ให้ `sim-local` แก้ harness ของตัวเอง SIM ตั้งให้ local editor พลาดความล้มเหลว 40% ของครั้ง เทียบกับ 25% ของ cloud editor

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 5 · Same-model editing: sim-local as its own editor versus the fixed cloud editor
│ executor   editor            stream 1 / 2 / 3    mean ± SD 
│ ─────────  ────────────────  ──────────────────  ──────────
│ sim-local  sim-cloud-editor  86.5 / 80.1 / 88.3  85.0 ± 4.3
│ sim-local  sim-local-editor  84.4 / 83.3 / 83.6  83.8 ± 0.6
◆ self-editing − cloud editor: -1.2 · per stream -2.1 / +3.2 / -4.7 → within noise
◆ the SIM local editor is set to miss a failure 40% of the time (cloud editor 25%), yet over 17 tasks and three streams that difference does not show: the table cannot tell the two editors apart
```

"within noise" ไม่ได้แปลว่า "เท่ากัน" มันแปลว่าตารางนี้แยก editor สองตัวไม่ออก สำหรับ production agent ที่มี local tier คำถามนี้สำคัญ: ถ้าโมเดล open-weight ที่เล็กกว่าบนฮาร์ดแวร์ในเครื่องแก้ harness ของตัวเองได้ trace ของมันก็ไม่ต้องออกจากเครื่องนั้นเลย EXAMPLE design ของคอร์สจึงลงทะเบียนเรื่องนี้ไว้ล่วงหน้าเป็น arm แยก (arm F, Module 09) พร้อมเกณฑ์ของตัวเอง แทนที่จะอ่านจากตารางแบบนี้

ขั้นที่ 6 นับสิ่งที่ run สุดท้ายทำ เปเปอร์มีตาราง tool-use ด้วย (as reported in the paper): จำนวน tool call ต่องานเพิ่มจาก 27.4 เป็น 32.8 ขณะที่อัตรา tool error ลดจาก 14.4% เป็น 9.1% ตัวเลขสองตัวนี้ไม่อยู่ใน snapshot ของ repository ใน `common/sources/` ที่นี่จึงไม่ได้ตรวจซ้ำ แผนการสอนย่อประเด็นไว้ในบรรทัดเดียว:

> “fewer errors, more useful calls, at a compute cost” — facilitator guide, Part 3

(error น้อยลง call ที่มีประโยชน์มากขึ้น โดยแลกกับ compute)

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 6 · Tool use and cost: what the final runs did, and what the loop spent to get them
│ executor   tool calls / task  tool-error rate  executions / task  editor sessions / task
│ ─────────  ─────────────────  ───────────────  ─────────────────  ──────────────────────
│ sim-cloud  2.67 → 3.24        6.6% → 1.8%      1 → 2.4            0 → 4.0               
│ sim-local  2.59 → 3.20        6.8% → 5.5%      1 → 2.5            0 → 4.0               
◆ more calls, fewer errors: scripts, validators and the maintenance check ADD useful work; refinement need not shorten a run
⚠ the base row costs one execution per task; Harness² adds references, probes and editor sessions — the paper publishes no token totals, only that it 'requires additional executions and editor sessions'
```

SIM แสดงรูปแบบเดียวกัน: run ที่ปรับแล้วเรียก tool มากขึ้น (การตรวจ `cmms.work_orders`, script, validator) และล้มเหลวน้อยลง call ที่มากขึ้นไม่ได้แย่ที่นี่ เพราะ call ที่เพิ่มแต่ละครั้งทำงานที่ base ข้ามไป คอลัมน์ขวาคือข้อควรระวังเรื่องต้นทุน เปเปอร์พูดไว้ตรง ๆ (as reported in the paper): Harness² ต้องใช้ execution และ editor session เพิ่มเพื่อ propose, probe และ compose การแก้ harness และเมื่อเทียบ Harness² กับ production harness แบบตายตัว เปเปอร์เสริมว่าการปรับใช้ test-time compute เพิ่ม

✓ Checkpoint: คุณบอกได้ว่าทำไม "tool call มากขึ้น" ไม่ใช่ regression ในตารางนี้ และบอกได้ว่าตารางผลลัพธ์ต้องรายงานต้นทุนอะไรคู่กับ gain

## 4 · ตาราง ablation: ส่วนไหนเห็นได้เกิน noise

ablation คือการตัดส่วนหนึ่งของวิธีออกแล้ววัดว่าเสียไปเท่าไร ablation ของเปเปอร์ที่ k = 1 (as reported in the paper):

| Ablation | WorkBuddy-Bench | JobBench |
|---|---|---|
| ไม่มี library | −3.1 | −1.8 |
| ไม่มี scoping (generic editors) | −2.9 | −3.7 |
| ไม่มี composition | −1.6 | −1.9 |

และการถอด component ออกจาก harness ที่ปรับแล้ว: skills −5.3 โดยเฉลี่ย (6.9 บน Web), scripts −3.7 (7.4 บน Office), prompt text −1.5 เปเปอร์เสริมว่า skill และ script คิดเป็นราวครึ่งหนึ่งของ edit ในแต่ละ benchmark

อ่านตัวเลขเหล่านี้ด้วยกฎของส่วนที่ 1 เมื่อเทียบกับ SD ของ JobBench ที่ 1.1 และ 2.6 แถว composition (1.6 และ 1.9) คือแถวที่ต้องอ่านอย่างระวังที่สุด

Lab 06-2 สร้างตารางของ SIM เอง ทุก arm เป็นสาม stream อ่านเทียบกับ Harness² k = 1 **วัดบนเครื่อง Mac นี้ — protocol: open-world streams** (เหมือนส่วนที่ 2)

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 2 · The SIM's ablation table on sim-cloud (3 shuffled streams each, read against k=1)
│ arm             stream 1 / 2 / 3     mean ± SD*  Δ vs k=1  reading         execs/task  sessions/task  always-on
│ ──────────────  ───────────────────  ──────────  ────────  ──────────────  ──────────  ─────────────  ─────────
│ Harness² k=1    96.5 / 95.0 / 91.5   94.3 ± 2.6  —         —               2.4         4.0            271      
│ no library      97.6 / 93.5 / 90.3   93.8 ± 3.7  -0.5      within noise    2.3         4.0            265      
│ generic editor  94.1 / 90.9 / 88.3   91.1 ± 2.9  -3.2      beyond noise    1.8         3.0            279      
│ no composition  97.6 / 92.1 / 92.7   94.1 ± 3.0  -0.2      within noise    2.4         3.0            265      
│ parallel k=3    98.8 / 100.0 / 95.4  98.1 ± 2.4  +3.7      beyond noise    7.3         11.0           281      
│ parallel k=5    97.6 / 95.7 / 92.3   95.2 ± 2.7  +0.9      direction only  12.5        17.0           284      
│ sequential k=3  98.5 / 92.7 / 97.4   96.2 ± 3.1  +1.9      within noise    4.8         10.0           279      
│ sequential k=5  92.5 / 98.8 / 98.0   96.4 ± 3.4  +2.1      within noise    7.0         16.0           282      
◆ * measured after each step by bench.judge · always-on = mean always-on tokens of the adopted harnesses (base 245)

▣ STEP 3 · The SIM's ablation table on sim-local (3 shuffled streams each, read against k=1)
│ arm             stream 1 / 2 / 3     mean ± SD*  Δ vs k=1  reading         execs/task  sessions/task  always-on
│ ──────────────  ───────────────────  ──────────  ────────  ──────────────  ──────────  ─────────────  ─────────
│ Harness² k=1    86.5 / 80.1 / 88.3   85.0 ± 4.3  —         —               2.5         4.0            270      
│ no library      84.1 / 76.6 / 84.8   81.8 ± 4.5  -3.1      direction only  2.3         4.0            263      
│ generic editor  81.9 / 73.3 / 78.2   77.8 ± 4.3  -7.2      beyond noise    1.8         3.0            275      
│ no composition  86.8 / 78.6 / 81.9   82.4 ± 4.1  -2.5      within noise    2.5         3.0            264      
│ parallel k=3    94.2 / 95.4 / 95.7   95.1 ± 0.8  +10.1     beyond noise    8.0         11.0           286      
│ parallel k=5    97.6 / 92.4 / 95.7   95.2 ± 2.6  +10.3     beyond noise    13.4        17.0           288      
│ sequential k=3  100.0 / 97.1 / 95.9  97.7 ± 2.1  +12.7     beyond noise    5.2         10.0           282      
│ sequential k=5  91.2 / 90.0 / 88.3   89.8 ± 1.5  +4.9      within noise    7.4         16.0           284      
```

อ่านทีละแถว

- **Generic editor** บน sim-cloud เสีย 3.2 คะแนนในทั้งสาม stream: beyond noise บน sim-local เสีย 7.2 (−4.6 / −6.8 / −10.1): beyond noise เช่นกัน บน executor ทั้งสอง harness ที่มันรับมามีข้อความ always-on มากกว่า (279 เทียบกับ 271 tokens บน sim-cloud, 275 เทียบกับ 270 บน sim-local) ทั้งที่คะแนนต่ำกว่า นี่คือ ablation เดียวที่ SIM แสดงได้บน executor ทั้งสอง
- **ไม่มี library, ไม่มี composition** บน sim-cloud ทั้งคู่อยู่ในหนึ่ง SD และ stream ไม่ไปทางเดียวกัน: within noise บน sim-local การไม่มี library เสีย 3.1 คะแนนในทั้งสาม stream (−2.4 / −3.5 / −3.5) แต่ SD คือ 4.5: direction only การไม่มี composition บน sim-local คือ −2.5 โดยมี stream หนึ่งเป็น +0.3: within noise 1.6 ถึง 3.1 คะแนนของเปเปอร์มองไม่เห็นเกิน noise บนงาน 17 งาน ประโยคที่ซื่อตรงคือ "**ไม่เห็นในตารางนี้**" (สำหรับแถว library บน sim-local คือ "ทิศทางสม่ำเสมอ แต่ยังบอกขนาดไม่ได้") ซึ่งไม่ใช่ทั้ง "ไม่มีผล" และ "เห็นผล" เกณฑ์ที่คอร์สลงทะเบียนล่วงหน้าสำหรับ Ops Assistant (EXAMPLE design, Module 09) กำหนดผลที่ตามมาไว้ก่อน: รับ library ก็ต่อเมื่อ arm C (มี library) สูงกว่า arm B (ไม่มี library) เกิน SD ของ stream มิฉะนั้นให้ส่งแค่ task-level harnessing และเลื่อน library ออกไป
- **k = 3 และ k = 5** ดูคอลัมน์ต้นทุนก่อน: k = 3 ใช้ execution สามเท่า k = 5 ราวห้าเท่า บน sim-cloud k = 3 beyond noise แต่ k = 5 สูงกว่า k = 1 ไม่ถึงหนึ่ง SD บน sim-local ซึ่งเริ่มจากแถว k = 1 ที่ต่ำกว่า parallel k = 3, parallel k = 5 และ sequential k = 3 beyond noise ทั้งหมด (+10.1 ถึง +12.7)
- **Sequential k = 5 บน sim-local** ค่าเฉลี่ยขึ้น 4.9 คะแนน มากกว่า SD ที่ 4.3 แต่ stream ที่สามเป็น +0.0 พอดี (88.3 เทียบกับ 88.3) stream จึงไม่ไปทางเดียวกัน: *within noise* ค่าเฉลี่ยที่ขยับมากแต่มี stream หนึ่งไม่ไปด้วย คือกรณีที่กฎนี้มีไว้รับมือ

ขั้นที่ 4 เทียบเรื่อง k ให้ชัด:

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 4 · Two comparisons the table invites: k=5 against k=3, and sequential against parallel at k=5
│ executor   comparison                     per-stream Δ        Δ mean  larger SD  reading     
│ ─────────  ─────────────────────────────  ──────────────────  ──────  ─────────  ────────────
│ sim-cloud  parallel k=5 − parallel k=3    -1.2 / -4.3 / -3.1  -2.9    2.7        beyond noise
│ sim-cloud  sequential k=5 − parallel k=5  -5.1 / +3.1 / +5.7  +1.2    3.4        within noise
│ sim-local  parallel k=5 − parallel k=3    +3.4 / -3.0 / +0.0  +0.1    2.6        within noise
│ sim-local  sequential k=5 − parallel k=5  -6.4 / -2.4 / -7.4  -5.4    2.6        beyond noise
```

เปเปอร์รายงานว่าการ recursion ทั้งสองโหมดดีขึ้นต่อเนื่องเมื่อ k เพิ่มจาก 1 ถึง 5

SIM ไม่เป็นเช่นนั้น บน sim-cloud k = 5 *ต่ำกว่า* k = 3 ในทั้งสาม stream เกินหนึ่ง SD เล็กน้อย: beyond noise บน sim-local ทั้งสองเท่ากัน (+0.1, ต่อ stream +3.4 / −3.0 / +0.0): within noise ไม่มี executor ไหนแสดงว่า k = 5 สูงกว่า k = 3 ให้พูดตามนั้น และบอกว่าทำไมอาจคาดได้: เมื่อมี final ห้าชิ้น selector ที่มองไม่เห็นคะแนนมีโอกาสส่งชิ้นที่แย่กว่ามากขึ้น Module 07 นับเรื่องนี้ คำแนะนำ EXAMPLE ของคอร์สสำหรับ Ops Assistant ไม่ขึ้นกับว่าเส้นโค้งไหนถูก: อย่าคาดหวังประโยชน์ทันทีจาก k มากกว่า 1 gain จาก recursion ที่ k = 5 ของเปเปอร์ (2.9 ถึง 3.9 คะแนนบนค่าเฉลี่ย WorkBuddy, reported in the paper) แลกกับ probe ราวห้าเท่า agent ที่ทำงาน always-on จึงควรพิสูจน์ทีละ family

แถว sequential ตั้งใจให้เป็นโจทย์ บน sim-local sequential k = 3 คือแถวที่ดีที่สุดของทั้งตาราง (97.7 ± 2.1) แต่โหมดเดียวกันที่ k = 5 ตกลงเหลือ 89.8: ต่ำกว่า sequential k = 3 ในทุก stream และต่ำกว่า parallel k = 5 แบบ beyond noise (−5.4, ขั้นที่ 4) บน sim-cloud sequential เทียบ parallel ที่ k = 5 อยู่ within noise แถวเดียวที่ beyond noise บน executor เดียว คือผลที่ต้องรันซ้ำด้วยงานมากขึ้น ไม่ใช่นโยบาย

✓ Checkpoint: สำหรับแต่ละแถว ablation คุณบอกได้ว่า *beyond noise*, *direction only* หรือ *within noise* และอธิบายได้ว่าทำไม "within noise" ไม่ได้แปลว่า "library ไม่ทำอะไรเลย"

## 5 · Transfer: version ที่ปรับบน executor หนึ่ง วัดบนอีกตัว

EXAMPLE design ของ Ops Assistant มีสอง tier: โมเดลบน cloud และโมเดล open-weight ที่เล็กกว่าบนฮาร์ดแวร์ในเครื่อง harness tree เดียวสำหรับทั้งสองจะดูแลถูกกว่า เปเปอร์บอกว่า transfer เกิดขึ้นได้ แต่ไม่สม่ำเสมอ (as reported in the paper): harness ที่ปรับสำหรับ Gemini 3.5 Flash ให้ GLM-5 +17.3 คะแนนบน Code (58.5 → 75.8), Gemini 3.7 Flash +0.3 บน Code และ +6.4 บน Office, Opus 4.8 +2.6 และ +1.1 ประโยชน์นั้นต่างกันมากตาม executor และ domain

Lab 06-3 วัดเรื่องนี้ มัน promote หนึ่ง **version** ต่อ executor จาก stream record ของ executor นั้นด้วย reviewer's merge (`bench.promote`, `min_support=2`, EXAMPLE) ตรึงไว้ แล้วประเมินบน **test half** ซึ่งไม่เคยเข้า loop **วัดบนเครื่อง Mac นี้ — protocol: version** (version ที่ promote แล้วและตรึงไว้ บน test half ที่แช่แข็ง, สาม trial ต่องาน, pass^3) ทำซ้ำภายใต้ evaluation seed สามค่าเพื่อได้ mean ± SD นี่เป็นคนละ protocol กับส่วนที่ 2–4 อย่าเทียบตัวเลขข้ามกัน

**Expected output** (captured on this Mac, SIM)

```
▣ STEP 3 · Evaluate every version on the frozen TEST half (17 tasks × 3 trials) on both executors, 3 seeds
│ executor   version               seed 0 / 1 / 2      mean ± SD*  Δ vs base             pass^3*  unsafe
│ ─────────  ────────────────────  ──────────────────  ──────────  ────────────────────  ───────  ──────
│ sim-cloud  base v1               81.8 / 86.0 / 83.8  83.9 ± 2.1  —                     0.49     0     
│ sim-cloud  refined on sim-cloud  97.4 / 96.7 / 98.4  97.5 ± 0.9  +13.6 · beyond noise  0.92     0     
│ sim-cloud  refined on sim-local  93.5 / 95.7 / 96.5  95.2 ± 1.6  +11.4 · beyond noise  0.88     0     
│ sim-local  base v1               70.9 / 71.9 / 73.0  71.9 ± 1.1  —                     0.22     0     
│ sim-local  refined on sim-cloud  94.4 / 91.6 / 92.9  93.0 ± 1.4  +21.0 · beyond noise  0.71     0     
│ sim-local  refined on sim-local  88.5 / 89.6 / 89.9  89.3 ± 0.7  +17.4 · beyond noise  0.69     0     
◆ * measured on the test half by bench.evaluate; nothing in the loop ever read these tasks

▣ STEP 4 · Read transfer per direction
│ sim-cloud version on sim-local: 93.0 vs sim-local's own version 89.3 → +3.6 · beyond noise
│ sim-local version on sim-cloud: 95.2 vs sim-cloud's own version 97.5 → -2.3 · beyond noise
⚠ in the SIM both executors share the same weaknesses and the same editors, so a version moves almost intact. The SIM cannot reproduce the paper's uneven transfer (+17.3 vs +0.3); it can only show how to measure it
```

อ่านสามเรื่องจากตารางนี้

1. **ทั้งสอง version transfer ได้ใน SIM** ทั้งสองทิศ เกิน noise ชัดเจน เพราะ executor จำลองทั้งสองล้มเหลวแบบเดียวกัน โมเดลจริงต่างกันมากกว่านี้ ซึ่งคือประเด็นของเปเปอร์
2. **version ที่ปรับบน executor ที่อ่อนกว่าคือ version ที่อ่อนกว่า บน executor ทั้งสอง** บน sim-cloud มันตาม version ที่ปรับบน cloud อยู่ 2.3 คะแนน beyond noise บน sim-local ซึ่งเป็น executor ของมันเอง มันก็ตามอยู่: version ที่ปรับบน cloud ได้มากกว่า 3.6 คะแนนที่นั่น beyond noise ขั้นที่ 2 บอกเหตุผลข้อหนึ่ง: stream ของ local รองรับ edit ที่ถูกรับ 11 รายการเทียบกับ 12 จึงผ่าน reviewer's merge ได้น้อยกว่า
3. **Transfer เป็นการวัดแยกตามทิศทาง** arm E ที่คอร์สลงทะเบียนล่วงหน้า (Module 09) ทำแบบนี้เป๊ะ: version ที่ปรับบน cloud ใช้กับ local executor *และ* version ที่ปรับบน local ใช้กับ cloud executor เก็บ library แยกต่อ executor และวัดก่อนจะใช้ version ร่วมกัน

✓ Checkpoint: คุณบอกได้ว่าทำไมตารางนี้วางคู่กับตาราง stream ของส่วนที่ 2–4 ไม่ได้ และ arm E วัดอะไร

## 6 · pass@k เทียบ pass^k

หัวหน้าช่างอ่าน digest ทุกเช้า digest ที่ถูกสองเช้าจากสามเช้าคือ digest ที่ไม่มีใครเชื่อ นี่คือความต่างระหว่างตัวเลขสองตัวที่ดูคล้ายกัน คู่มือ evals ของ Anthropic แบ่งไว้ว่า pass@k เหมาะกับ tool ที่สำเร็จครั้งเดียวก็พอ และ pass^k เหมาะกับ agent ที่ต้องสม่ำเสมอ

pass@k คือโอกาสที่อย่างน้อยหนึ่งใน k trial สำเร็จ pass^k คือโอกาสที่ทั้ง k trial สำเร็จ ถ้า trial เป็นอิสระต่อกันและสำเร็จต่อ trial ด้วยโอกาส p จะได้ pass@k = 1 − (1 − p)^k และ pass^k = p^k

**Expected output** (captured on this Mac, SIM)

```
│ per-trial success p  k  pass@k = 1 − (1 − p)^k  pass^k = p^k
│ ───────────────────  ─  ──────────────────────  ────────────
│ 0.75                 3  98.4%                   42.2%       
◆ worked example: 75% per trial is about 42% across three trials
│ version               executor   per-trial p  pass@3 if independent  p^3 if independent  pass^3 observed
│ ────────────────────  ─────────  ───────────  ─────────────────────  ──────────────────  ───────────────
│ base v1               sim-cloud  0.69         0.97                   0.32                0.41           
│ base v1               sim-local  0.51         0.88                   0.13                0.23           
│ refined on sim-cloud  sim-cloud  0.98         1.00                   0.94                0.94           
│ refined on sim-cloud  sim-local  0.92         1.00                   0.78                0.77           
│ refined on sim-local  sim-cloud  0.94         1.00                   0.83                0.88           
│ refined on sim-local  sim-local  0.86         1.00                   0.64                0.71           
◆ observed pass^3 is not p^3 (at or above it on 5 of 6 rows): trials of one task are not independent — some tasks pass every time, some never. Measure pass^3 per task; do not derive it from the pass rate
```

75% ต่อ trial ฟังดูดี แต่เป็น 98% ในแบบ pass@3 และ 42% ในแบบ pass^3 base harness บน sim-cloud ผ่านแต่ละ trial 69% ของครั้ง และผ่านครบสาม trial เพียง 41% ของงานใน test half (seed 0) pass^3 ที่สังเกตได้ไม่เท่ากับ p^3 ห้าจากหกแถวมีค่าเท่ากับหรือสูงกว่า p^3 เพราะ trial ของงานเดียวกันสัมพันธ์กัน: บางงานผ่านทุกครั้ง บางงานไม่เคยผ่าน ในอีกแถวหนึ่ง (version ที่ปรับบน cloud บน sim-local) มันต่ำกว่าเล็กน้อย (0.77 เทียบกับ 0.78) ไม่ว่าทางไหน pass rate ก็ไม่ได้บอก pass^3 ให้คำนวณ pass^3 ต่องานจาก trial จริง (`bench.evaluate` ทำให้) ใน EXAMPLE design ของคอร์ส family ที่ลูกค้าเห็น (digest, การแจ้งเตือน, ร่างภาษาไทย) promote ด้วย pass^3 ส่วน pass@k เป็นแค่ตัวช่วยวินิจฉัย

✓ Checkpoint: คุณคิดในใจได้ว่า pass^3 ที่ p = 0.9 คือราว 73% และบอกได้ว่าระหว่าง pass@3 กับ pass^3 ตัวไหนใช้ promote version ของ digest

## Labs — run them here

**labs/lab06_1_three_streams.py** — สาม stream ที่สลับลำดับ: base เทียบ Harness² บน executor ทั้งสอง, same-model editing, การใช้ tool และต้นทุน

**labs/lab06_2_ablation_table.py** — ตาราง ablation ของ SIM บน executor ทั้งสอง ทุกแถวอ่านเทียบ k = 1 พร้อมต้นทุน

**labs/lab06_3_transfer_and_pass_k.py** — version ที่ promote บนแต่ละ executor วัดบนทั้งสองตัวใน test half ที่แช่แข็ง แล้วเทียบ pass@k กับ pass^k

## Try it yourself

`exercises/ex06_read_the_table.py` มีสาม TODO:

1. เขียนกฎการอ่าน `reading_of(a, b)`: beyond noise, direction only หรือ within noise
2. ตัดสินว่าข้ออ้างแปดข้อไหนที่หลักฐานรองรับ (ห้าข้อเกี่ยวกับตาราง SIM, สองข้อเกี่ยวกับเปเปอร์, หนึ่งข้อเกี่ยวกับ pass^3)
3. เขียน `pass_hat(p, k)`

```bash
# on: laptop
.venv/bin/python week31_harness2_recursive_harnessing/06_reading_the_results/exercises/ex06_read_the_table.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ reading rule matches on all 6 comparisons (e.g. no library vs k=1: within noise)
✓ pass^3 at p = 0.90 is 0.729; at p = 0.75 it is 0.422 (the worked example's ≈ 42%)
✓ all eight claims judged: 1, 3, 6 supported; 2, 4, 5, 7, 8 not supported by this evidence

═ Done. Module 07 asks why a table can mislead even when you read it carefully: three failure modes.
```

<details><summary>Hint — TODO 1</summary>

จับคู่ list ตามตำแหน่ง: `diffs = [y - x for x, y in zip(a, b)]` "ไปทางเดียวกัน" หมายถึงความต่างทุกตัวเป็นบวกหรือเป็นลบทั้งหมด (ความต่างที่เป็นศูนย์ทำให้ไม่นับว่าไปทางเดียวกัน) เทียบ `abs(mean(b) - mean(a))` กับ `max(stdev(a), stdev(b))` จากโมดูล `statistics`

</details>

<details><summary>Hint — TODO 2</summary>

"not supported" ครอบคลุมสองกรณี: ตารางไม่แสดงอะไรเกิน noise (ข้อ 2 และ 5) หรือแสดงผลตรงกันข้าม (ข้อ 4: k = 5 ต่ำกว่า k = 3) สำหรับข้อ 7 ให้เทียบ +17.3 ของ GLM-5 กับ +0.3 ของ Gemini 3.7 Flash จากส่วนที่ 5 สำหรับข้อ 8 ใช้ TODO 3

</details>

✓ Checkpoint: บรรทัดตรวจทั้งสามเป็น ✓

## Troubleshooting

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| ค่าเฉลี่ย stream ของคุณไม่ตรงกับ tutorial | kit ถูกแก้ หรือใช้ editor คนละตัว | `git diff week31_harness2_recursive_harnessing/common`; lab ตรึง `editor="sim-cloud-editor"` ให้ executor ทั้งสอง |
| `streams(..., mode="sequential")` ได้ตัวเลขแบบ parallel | `stream(mode=)` คือ base หรือ harness2; โหมด recursion คือ `recursion=` | ส่ง `recursion="sequential"` อย่างที่ Lab 06-2 ทำ |
| แถวหนึ่งอ่านว่า "within noise" ทั้งที่ค่าเฉลี่ยขยับมาก | มี stream หนึ่งไปอีกทาง หรือไม่ขยับ | นั่นคือกฎทำงานถูกต้อง; รัน stream หรืองานเพิ่มก่อนจะเขียนประโยคเกี่ยวกับมัน |
| ตัวเลข Lab 06-3 ไม่ตรงกับ Lab 06-2 | คนละ protocol: version ที่ตรึงบน test half เทียบกับ open-world stream บน refine half | อย่าใส่ทั้งสองในตารางเดียวกัน |

## Next

[Lab 07 — Three ways harnessing fails, and measurement hygiene](../07_failure_modes/TUTORIAL.md): blind selection, premise ที่ทุก candidate มีร่วมกัน, attribution noise และ rubric ที่ executor ไม่ควรดึงมาได้เลย
