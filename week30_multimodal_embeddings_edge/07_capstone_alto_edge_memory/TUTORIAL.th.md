# ▶ Embed Lab 07 — Capstone: Alto Edge Memory หน่วยความจำของโรงแรมบนเครื่อง edge

> ส่วนหนึ่งของ Week 30 · multimodal embedding ด้วย EmbeddingGemma 2 บนฮาร์ดแวร์แบบ local, sovereign และ edge ทุกแล็บรันบนเครื่องของคุณเอง ตัวเลขทุกตัวในคอร์สนี้วัดจากเครื่องที่รันจริง หรือไม่ก็อ้างอิงพร้อมแหล่งที่มา

**สิ่งที่คุณจะได้ลงมือทำจริง**
- สร้าง **Alto Edge Memory**: index แบบออฟไลน์ชุดเดียว ที่รวมโน้ต รูปถ่าย ข้อความเสียง CCTV และบันทึกเหตุการณ์ของโรงแรมไว้ด้วยกัน
- ส่งให้ agent ใช้: Gemma 4 ที่รันบนเครื่อง พร้อม tool ตัวเดียวคือ `search_memory` ซึ่งจัดอันดับแยกตามชนิดข้อมูล เพื่อเลี่ยงปัญหา modality gap
- ถามคำถามกะดึกสองข้อ เป็นภาษาอังกฤษและภาษาไทย ภายใต้ egress guard
- ให้คะแนนผลลัพธ์จากหลักฐานเท่านั้น แล้วหาว่า grader มองข้ามอะไรไป

**Time** ~60 นาที · **Difficulty** ขั้นสูง · **Hardware** แล็ปท็อปเครื่องไหนก็ได้ · RAM ว่าง ~5 GB · Ollama พร้อมโมเดลแชตบนเครื่อง

**แหล่งอ้างอิง:** ทุกอย่างใน Module 01–06 · [model card](https://huggingface.co/google/embeddinggemma-2) · [Ollama tool calling](https://ollama.com) · [`week30_multimodal_embeddings_edge/RESEARCH.md`](../RESEARCH.md)

## 0 · ก่อนเริ่ม

capstone นี้มีสามแล็บ รันตามลำดับ: **07-1 สร้าง**, **07-2 รัน agent**, **07-3 ให้คะแนน** แต่ละแล็บเขียนหลักฐานไว้ที่ `week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/.runs/` คุณต้องมีโมเดลแคชไว้แล้ว (Lab 02) และมี Ollama พร้อมโมเดลแชตบนเครื่อง (Lab 05)

```bash
# on: laptop
ls week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/.runs/ 2>/dev/null || echo "empty — Lab 07-1 creates it"
```

✓ Checkpoint: คุณเคยรัน Lab 02-1 และ 05-3 สำเร็จมาแล้วอย่างน้อยหนึ่งครั้ง

## 1 · สร้างหน่วยความจำ

การออกแบบของ capstone นี้มาจากผลการวัดในบทก่อน ๆ:

| ตัวเลือก | ค่า | เหตุผล |
|---|---|---|
| encoder | ทั้งหมด (740M) | รูปถ่าย เสียง และ CCTV ต้องค้นหาได้ (Module 04) |
| มิติ | **512** | คุณภาพแบบ multimodal ตกเร็วกว่าข้อความเมื่อต่ำกว่า 512 (ตารางใน model card; Lab 03) |
| การจัดเก็บ | float16 ในไฟล์ SQLite ไฟล์เดียว | เล็ก พกพาง่าย ตรวจสอบได้ (Lab 05-2) |
| เหตุการณ์ | บันทึกแบบ interleaved: ข้อความ + รูป (+ CCTV) | หนึ่งเวกเตอร์ต่อหนึ่งเรื่อง (Lab 04-2) |
| เครือข่าย | อยู่ใน egress guard | เป็นหลักฐาน ไม่ใช่แค่คำกล่าวอ้าง (Lab 05-1) |

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/labs/lab07_1_build_memory.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
▣ STEP 4 · summary
│ kind    items
│ ──────  ─────
│ audio   3
│ image   6
│ note    18
│ record  2
│ video   1
✓ 30 items · 52 KiB · built in 19.8s · left this machine: 0 connections
═ week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/.runs/edge_memory.sqlite is the whole memory — Lab 07-2 gives it to an agent.
```

✓ Checkpoint: มีไฟล์ `.runs/edge_memory.sqlite` และ `.runs/manifest.json` และ manifest แสดงข้อมูลครบห้าชนิด

## 2 · agent และ tool ตัวเดียวของมัน

agent ในที่นี้คือลูป: LLM อ่านคำถาม ตัดสินใจเรียก `search_memory` อ่านผลลัพธ์ แล้วตอบ tool ตัวนี้ใช้เทคนิคหนึ่งจาก Module 04 และ 05 คือค้นหา **แยกทีละชนิด** (โน้ต รูปถ่าย เสียง วิดีโอ บันทึกเหตุการณ์) แล้วคืนผลที่ดีที่สุดของทุกชนิด แต่ละรายการมี `source` ให้อ้างอิง ผลคือ agent อ้างถึงข้อความเสียงได้ แม้โน้ตจะได้คะแนนรวมสูงกว่า

```python
def search_memory(model, store, query, kinds=None, per_kind=2):
    q = model.encode(query, prompt_name="SearchQuery")
    hits = []
    for k in kinds or ["note", "image", "audio", "video", "record"]:   # per-kind: the modality-gap fix
        hits += store.search(q, k=per_kind, kinds={k})
    return hits
```

system prompt สั่งให้โมเดลค้นหาก่อนเสมอ ตอบจากผลค้นหาเท่านั้น ใส่ citation ในวงเล็บเหลี่ยม และตอบเป็นภาษาเดียวกับคำถาม แล็บนี้อนุญาตให้เรียก LLM ได้ไม่เกิน 6 ครั้ง และมีปลายทางเครือข่ายได้แค่ที่เดียว คือ Ollama บน localhost

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/labs/lab07_2_memory_agent.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้, gemma4:12b)

```
▣ STEP 1 · Night shift handover: what happened in plant room B2, and what is our procedure for it?
  → search_memory({"query": "plant room B2 incident and procedure"})
      0.798  note   Smoke in the plant room  [hotel_kb.json#2]
      0.707  note   Chiller plant efficiency alarm  [hotel_kb.json#0]
      0.632  image  Photo: water pooling under AHU-3  [media/leak_under_ahu.jpg]
      0.615  image  Photo: plant room B2, haze  [media/plantroom_smoke.jpg]
      0.628  audio  Voicemail: towels for room 812, no rush  [media/request_towels_en.wav]
      0.614  audio  Voicemail: room 1507 bathroom flooding  [media/voicemail_flood_en.wav]
      0.692  video  CCTV: plant room B2, haze appears after ~4 s  [media/cctv_plantroom.mp4]
      0.776  record Incident 0412 · plant room B2 haze  [incident log]
      0.718  record Incident 0413 · AHU-3 condensate leak  [incident log]
  · ANSWER A haze was observed in plant room B2 at 02:10 [incident log]. If smoke or haze is detected in any plant room, it is treated as a fire event [hotel_kb.json#2]. The procedure is to press the nearest call point, evacuate the area, and contact security on extension 2222 [hotel_kb.json#2]. Do not attempt to locate the source of the smoke yourself [hotel_kb.json#2].

▣ STEP 2 · แขกห้อง 1203 แจ้งเรื่องอะไรเข้ามาบ้าง และควรทำอะไรต่อ
  → search_memory({"query": "room 1203 report issue"})
      0.698  note   Smoke in the plant room  [hotel_kb.json#2]
      0.691  note   Noise complaint at night  [hotel_kb.json#13]
      0.649  image  Photo: thermostat in room 1203 reads 29.1 °C  [media/thermostat_room1203.png]
      0.632  image  Screenshot: BMS trend for AHU-3  [media/bms_trend_ahu3.png]
      0.704  audio  Voicemail: towels for room 812, no rush  [media/request_towels_en.wav]
      0.655  audio  Voicemail (Thai): room 1203, AC not cold, burning smell  [media/voicemail_burning_th.wav]
      0.597  video  CCTV: plant room B2, haze appears after ~4 s  [media/cctv_plantroom.mp4]
      0.678  record Incident 0413 · AHU-3 condensate leak  [incident log]
      0.653  record Incident 0412 · plant room B2 haze  [incident log]
  · ANSWER แขกห้อง 1203 แจ้งว่าเครื่องปรับอากาศไม่เย็นและมีกลิ่นไหม้ [media/voicemail_burning_th.wav] คุณควรตรวจสอบระบบปรับอากาศและตรวจสอบกลิ่นไหม้ทันที หากพบกลุ่มควันหรือควันในห้องเครื่อง ให้กดปุ่มแจ้งเหตุเพลิงไหมฯ อพยพคน และแจ้งฝ่ายรักษาความปลอดภัยที่เบอร์ 2222 ทันที [hotel_kb.json#2]

▣ STEP 3 · evidence
│ allowed  connect  ::1:11434  ×4
│ allowed  connect  127.0.0.1:11434  ×4
✓ 4 LLM calls · left this machine: 0 connections · log → .runs/agent_run.json
```

ดูว่าการค้นหาแยกตามชนิดช่วยอะไรได้บ้าง ในคำถามที่ 2 ข้อความเสียงภาษาไทยได้อันดับต่ำกว่าข้อความเสียงเรื่องผ้าเช็ดตัวและโน้ตอีกสองรายการ แต่ก็ยังไปถึงโมเดลได้ เพราะรายการเสียงมีที่ว่างให้เสมอ สังเกตด้วยว่า LLM รู้อะไรจริง ๆ: มันไม่เคยได้ยินเสียงเลย EmbeddingGemma 2 เป็นตัวจับคู่กับไฟล์เสียง ส่วน LLM อ่านแค่ **ชื่อเรื่อง** ที่เก็บไว้คู่กัน ("AC not cold, burning smell") metadata ที่คุณเขียนตอน ingest คือสิ่งที่ agent พูดออกมาได้ แต่ก่อนจะไปดูคะแนน ให้อ่านคำตอบภาษาไทยให้ละเอียดก่อน

✓ Checkpoint: ผลรันของคุณแสดงคำตอบทั้งสองข้อพร้อม citation และมีบรรทัด `left this machine: 0 connections`

## 3 · ให้คะแนนจากหลักฐาน

Lab 07-3 อ่านเฉพาะไฟล์ใน `.runs/` และรันการทดสอบ retrieval ชุดคงที่ซ้ำอีกรอบ ทุกคะแนนย้อนกลับไปหาไฟล์หรือผลการวัดได้:

| หมวด | คะแนน | หลักฐาน |
|---|---|---|
| index | 20 | มี manifest; มีข้อมูลครบทั้งห้าชนิด |
| retrieval | 25 | คำถามคงที่ 8 ข้อ, ค้นหาแยกตามชนิด, รายการที่คาดไว้ต้องมาเป็นอันดับแรก |
| sovereignty | 20 | ไม่มีการเชื่อมต่อออกจากเครื่องเลย ทั้งตอน build **และ** ตอนรัน agent |
| agent | 35 | ตอบครบทั้งสองข้อ; ทุก citation เป็น source ที่มีอยู่จริง; คำตอบเรื่อง B2 อ้างถึงขั้นตอนปฏิบัติ + กล้อง/รูป/เหตุการณ์; คำตอบภาษาไทยอ้างถึงข้อความเสียงภาษาไทย |

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/labs/lab07_3_grade.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
▣ STEP 2 · retrieval (25) — per-kind search, expected item first
  ✓ [note  ] procedure for smoke in a plant room        → Smoke in the plant room
  ✓ [video ] haze seen on the plant room camera         → CCTV: plant room B2, haze appears after ~4 s
  ✓ [image ] photo of water under an air handler        → Photo: water pooling under AHU-3
  ✓ [image ] thermostat display showing a hot room      → Photo: thermostat in room 1203 reads 29.1 °C
  ✓ [audio ] แขกแจ้งว่าแอร์มีกลิ่นไหม้                  → Voicemail (Thai): room 1203, AC not cold, burn
  ✓ [audio ] guest bathroom is flooding                 → Voicemail: room 1507 bathroom flooding
  ✓ [record] incident report about haze in B2           → Incident 0412 · plant room B2 haze
  ✓ [note  ] supplier wants payment to a new bank accou → Supplier bank detail changes
✓ 8/8 · +25

▣ STEP 3 · sovereignty (20)
✓ connections that left the machine — build: 0 · agent: 0 · +20

▣ STEP 4 · agent (35)
  citations: 3 · real sources: 3 · not in the index: none
  ✓ plant-room answer cites the smoke procedure AND a camera/photo/incident item
  ✓ Thai answer cites the Thai voicemail
✓ +35

▣ STEP 5 · result
  ████████████████████████████████████████ 100/100
═ 100/100 — distinction. Every point above came from a file in .runs/ or a measurement just now.
```

✓ Checkpoint: คะแนนของคุณพิมพ์ออกมาแล้ว และคุณชี้ได้ว่าแต่ละหมวดทั้งสี่มาจากไฟล์ไหน

## 4 · สิ่งที่ grader มองข้ามไป

ได้ 100/100 แต่ลองอ่านคำตอบภาษาไทยอีกครั้ง มันอ้างถึงข้อความเสียงถูกต้อง แต่กลับบอกช่างให้กดปุ่มแจ้งเหตุเพลิงไหม้และอพยพคน ซึ่งเป็นขั้นตอนของ **ควันในห้องเครื่อง** โน้ตที่ถูกต้องคือ "แอร์มีกลิ่นไหม้" (ปิดเครื่อง ย้ายแขก ส่งช่างไฟฟ้า) แต่ไม่เคยถูกดึงขึ้นมาเลย agent ค้นด้วยคำกว้าง ๆ ว่า "room 1203 report issue" แทนที่จะค้นตัวปัญหาจริง ขั้นตอนที่ใกล้ที่สุดที่มันเห็นจึงเป็นของห้องเครื่อง

grader ตรวจแค่ว่า citation **มีอยู่และเป็นของจริง** แต่ไม่ได้ตรวจว่าขั้นตอนนั้น **ตรงกับปัญหา** หรือเปล่า นี่คือบทเรียนที่มีประโยชน์ที่สุดของ capstone นี้ มีสามวิธีปิดช่องโหว่:

| วิธีแก้ | แก้ที่ไหน |
|---|---|
| ค้นสองขั้น: หารายงานก่อน แล้วค้นอีกครั้งด้วย **เนื้อหา** ของรายงาน ("AC burning smell procedure") | ใน system prompt หรือเรียก tool ครั้งที่สอง |
| เพิ่มการตรวจใน grader: คำตอบภาษาไทยต้องอ้างถึง `hotel_kb.json#16` | `lab07_3_grade.py` |
| ให้คนอยู่ในลูปเสมอสำหรับขั้นตอนด้านความปลอดภัย | งานปฏิบัติการ ไม่ใช่โค้ด |

✓ Checkpoint: คุณบอกจุดบกพร่องได้ในประโยคเดียว และบอกได้ว่าการตรวจแบบไหนใน grader จะจับได้ (Exercise 07 ถามข้อนี้ตรง ๆ)

## Labs — รันแล็บได้ที่นี่

**labs/lab07_1_build_memory.py** — รวมโน้ต รูปถ่าย ข้อความเสียง CCTV และบันทึกเหตุการณ์แบบ interleaved ไว้ในหน่วยความจำ SQLite ขนาด 512 มิติชุดเดียว สร้างภายใต้ egress guard

**labs/lab07_2_memory_agent.py** — Gemma 4 บนเครื่อง พร้อม tool `search_memory` ที่ค้นแยกตามชนิด ตอบคำถามกะดึกสองข้อ เป็นภาษาอังกฤษและภาษาไทย

**labs/lab07_3_grade.py** — ให้คะแนนเต็ม 100 จากไฟล์หลักฐานและการทดสอบ retrieval ชุดคงที่ ไม่มีคะแนนส่วนไหนมาจากความเห็น

## Try it yourself — ลองทำเอง

`exercises/ex07_memory_tool.py` มี TODO สามจุด:

1. `per_kind_top(hits, per_kind)`: แก้ปัญหา modality gap ด้วยมือ
2. `cited(answer)` และ `unknown_citations(answer, sources)`: จับ citation ที่แต่งขึ้นมาเอง
3. เขียน 1–3 ประโยค: Lab 07-3 ให้คะแนน 100 กับอะไรที่ไม่ควรได้?

จากนั้นลองต่อยอด: เพิ่มการตรวจจุดบอดของคุณลงใน `lab07_3_grade.py` แก้ system prompt ใน `lab07_2_memory_agent.py` ให้ agent ค้นหาตัวปัญหาจริง แล้วรัน 07-2 และ 07-3 ใหม่อีกครั้ง

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/exercises/ex07_memory_tool.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ เมื่อทำ TODO ครบทุกจุด)

```
✓ per_kind_top: the best hit OF EACH KIND, then sorted by score (the modality-gap fix)
✓ cited / unknown_citations: every [source] in order · flag the ones the index never returned
✓ your note names what Lab 07-3 scored 100 but should not have
```

<details><summary>คำใบ้ — จุดบอด</summary>

เทียบคำแนะนำในคำตอบภาษาไทยกับโน้ต #16 ใน `week30_multimodal_embeddings_edge/common/data/hotel_kb.json` แล้วดู query ที่ agent ส่งไปให้ `search_memory`

</details>

✓ Checkpoint: ตัวตรวจขึ้น ✓ ครบทั้งสามบรรทัด และ grader ที่คุณปรับปรุงแล้วจับคำตอบภาษาไทยได้

## Troubleshooting — แก้ปัญหา

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| Lab 07-2: "run Lab 07-1 first" | ไม่มี `.runs/edge_memory.sqlite` | รัน Lab 07-1 |
| agent ตอบโดยไม่เรียก tool | โมเดลเล็กบางครั้งข้ามการเรียก tool | ใช้ `gemma4:12b`; system prompt เขียนไว้ว่า "always call search_memory first" |
| อาร์กิวเมนต์ของ `tool_calls` มาเป็น string | บางโมเดลคืนค่าเป็นข้อความ JSON | แล็บ parse ได้ทั้งสองแบบ |
| คะแนนเปลี่ยนไปในแต่ละรอบ | ถ้อยคำและการตัดสินใจของ LLM ไม่เหมือนเดิมทุกครั้ง | ส่วน retrieval และ sovereignty ให้ผลคงที่; รัน 07-2 ใหม่สำหรับส่วน agent |

## Next — บทถัดไป

[Lab 08 — Mac ของคุณคือเครื่อง edge](../08_mac_edge_box/TUTORIAL.md): รันโมเดลเดียวกันด้วย MLX และ ONNX บน Apple silicon แล้วเปลี่ยน Mac mini ให้เป็นเซิร์ฟเวอร์ embedding ประจำโรงแรมที่เปิดตลอดเวลา

จากนั้น สิ่งที่น่าสร้างต่อ:
- รัน Lab 06-1 บนเครื่อง edge จริง (Raspberry Pi 5, Jetson, DGX Spark) แล้วแทนบรรทัด ESTIMATE ด้วยผลการวัดจริง
- อัปเกรด Ollama แล้วรัน Lab 03-3 ใหม่: GGUF แบบข้อความล้วนจะขึ้น ✓ และคุณเทียบกับโมเดล PyTorch ได้
- ชี้ Alto Edge Memory ไปที่ข้อมูลของคุณเอง ที่ **ได้รับอนุมัติ เป็นข้อมูลสังเคราะห์ หรือได้รับความยินยอมแล้ว**

หรือเริ่มใหม่ตั้งแต่ [Lab 01 — Local, sovereign, edge](../01_local_sovereign_edge/TUTORIAL.md)
