# ▶ Embed Lab 04 — พื้นที่เดียวสำหรับข้อความ รูปภาพ เสียง และวิดีโอ

> ส่วนหนึ่งของ Week 30 · Multimodal embedding ด้วย EmbeddingGemma 2 บนฮาร์ดแวร์ในเครื่อง แบบ sovereign และ edge ทุกแล็บรันบนเครื่องของคุณเอง ตัวเลขทุกตัวในคอร์สนี้วัดจากเครื่องที่รันจริง หรือไม่ก็ยกมาพร้อมแหล่งที่มา

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- นำรูปภาพ ข้อความเสียง (voicemail) และคลิป CCTV ไปไว้ใน**พื้นที่เดียวกัน**กับข้อความ คือพื้นที่ 768 ตัวเลข โดยไม่ต้องมีคำบรรยายภาพและไม่ต้องถอดเสียง
- ค้นหาด้วยคำถามภาษาอังกฤษธรรมดา และด้วยคำถามภาษาไทย
- สร้าง vector เดียวสำหรับเหตุการณ์ทั้งเหตุการณ์ ทั้งข้อความ รูปภาพ และคลิปวิดีโอ ใน input เดียว
- วางงบ context 8,192 token ที่ทุก modality ใช้ร่วมกัน
- รู้จักข้อจำกัดสองข้อของการค้นหาด้วย embedding คือ **modality gap** และการที่มันมองไม่เห็นสิ่งที่**ไม่มี**อยู่

**Time** ~50 นาที · **Difficulty** ระดับกลาง · **Hardware** แล็ปท็อปเครื่องไหนก็ได้ · RAM ว่างราว 5 GB สำหรับ encoder ทั้งหมด

**แหล่งอ้างอิง:** [model card — Multimodal Input, Context Limits](https://huggingface.co/google/embeddinggemma-2) · [บล็อกเปิดตัว](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/) · สื่อ: `week30_multimodal_embeddings_edge/common/media/` (สังเคราะห์ขึ้น จาก Week 24 Module 13) · [`week30_multimodal_embeddings_edge/RESEARCH.md`](../RESEARCH.md) §1

## 0 · ก่อนเริ่ม

วิดีโอต้องมีตัวถอดรหัส (decoder) ตอนนี้ `torchvision` เลิกอ่านวิดีโอแล้ว sentence-transformers จึงใช้ **torchcodec** แทน และ torchcodec ต้องมี ffmpeg อยู่บนเครื่อง

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python -c "import torchcodec; print('torchcodec', torchcodec.__version__)"
ffmpeg -version | head -1
```

สื่อทั้งหมดใน module นี้เป็น**ข้อมูลสังเคราะห์ (synthetic)** รูปภาพสร้างด้วยโมเดลสร้างภาพ เสียงพูดสร้างด้วย text-to-speech และคลิปก็สร้างขึ้นเช่นกัน ข้อความเสียงเป็นแบบ mono 16 kHz ตามรูปแบบที่ model card กำหนด

✓ Checkpoint: ทั้งสองคำสั่งพิมพ์เวอร์ชันออกมา

## 1 · การค้นหาแบบ cross-modal: ประโยคหนึ่งหารูปภาพเจอ

เมื่อโหลด encoder ครบทุกตัว (740M) รูปภาพหรือคลิปเสียงจะเข้าไปอยู่ในพื้นที่เดียวกับข้อความ คำถามเป็นข้อความเพียงอย่างเดียว ส่วนรูปภาพและข้อความเสียง**ไม่ต้องใส่ prefix** (model card: "Prefixes apply to text only")

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/04_multimodal_search/labs/lab04_1_cross_modal.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
▣ STEP 2 · embed the media — no captions, no transcripts
✓ 6 images in 1.48s (247 ms each) · 3 voicemails (8.8–12.1 s each) in 0.33s (111 ms each)

▣ STEP 3 · seven questions, ranked against all nine items
  ✓ water leaking on the floor under equipment                 → leak_under_ahu.jpg 0.725  plantroom_smoke.jpg 0.630
  ✓ smoke or haze in the plant room                            → plantroom_smoke.jpg 0.658  leak_under_ahu.jpg 0.632
  ✓ a hotel room thermostat that is too hot                    → thermostat_room1203.png 0.724  voicemail_flood_en.wav 0.648
  ✓ a suspicious invoice asking to change bank details         → invoice_bank_change.png 0.763  request_towels_en.wav 0.561
  ✓ a guest reports flooding in the bathroom                   → voicemail_flood_en.wav 0.785  request_towels_en.wav 0.672
  ✓ a guest needs more towels                                  → request_towels_en.wav 0.788  voicemail_flood_en.wav 0.671
  ✓ a Thai speaker reports a burning smell from the air condit → voicemail_burning_th.wav 0.692  invoice_bank_change.png 0.637
✓ 7/7 questions found the right item first

▣ STEP 4 · image ↔ image: which photo is closest to the normal plant room?
  0.969  plantroom_smoke.jpg
  0.848  leak_under_ahu.jpg
  0.664  thermostat_room1203.png
```

โมเดลอ่านค่าบนหน้าจอ thermostat ได้ จำใบแจ้งหนี้ที่ขอเปลี่ยนบัญชีธนาคารได้ และเข้าใจคำขอผ้าเช็ดตัวที่พูดออกมาได้ โดยไม่มีขั้นตอนแปลงเสียงเป็นข้อความเลย Step 4 บอกข้อเท็จจริงที่เงียบกว่านั้น ห้อง plant room ที่มีควันคล้ายกับห้องปกติถึง 0.969 สำหรับ embedding แล้ว ทั้งสองคือห้องเดียวกัน และหมอกควันเป็นแค่ความต่างเล็กน้อย จำข้อนี้ไว้สำหรับ Section 4

✓ Checkpoint: บนเครื่องของคุณ คำถามทั้งเจ็ดข้อเจอ item ที่ถูกต้องเป็นอันดับแรก

## 2 · วิดีโอและ record แบบ interleaved

คลิป CCTV ถูกอ่านเป็นเฟรม ค่าเริ่มต้นคือ **1 เฟรมต่อวินาที** และ **140 token ต่อเฟรม** input เดียวยังผสมหลาย modality ได้ด้วย placeholder ในข้อความ (`<|image|>`, `<|video|>`, `<|audio|>`) จะถูกเติมจาก key ที่ตรงกันตามลำดับ แล้วโมเดลจะคืน vector **เดียว** สำหรับทั้ง record:

```python
emb = model.encode({
    "text": "Incident 0412, plant room B2, 02:10. Night shift saw this: <|image|> CCTV from the same camera: <|video|>",
    "image": ["plantroom_smoke.jpg"],
    "video": "cctv_plantroom.mp4",
})
```

ทุก modality ใช้ context **8,192 token** ร่วมกัน ต้นทุนตาม model card: ข้อความ 1 token ต่อ subword · รูปภาพ 280 · วิดีโอ 140 ต่อเฟรม · เสียง 25 ต่อวินาที

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/04_multimodal_search/labs/lab04_2_video_and_interleaved.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
▣ STEP 1 · budget the 8,192-token context (model card token costs)
│ part                       tokens
│ ─────────────────────────  ────────────
│ text (≈ 20 words)          ≈ 30
│ 1 photo                    280
│ CCTV clip 10s → 10 frames  1400
│ total                      1710 of 8192
◆ room left: ≈ 46 more frames, or 259 s of audio, in the same record

▣ STEP 3 · which record answers which question?
  ✓ haze appearing on the plant room camera      → incident: B2 haze (text + photo + CCTV) 0.719   (2nd: cctv_plantroom.mp4 (video  0.686)
  ✓ water leaking from an air handler            → incident: AHU-3 leak (text + photo) 0.681   (2nd: incident: B2 haze (text +  0.620)
  ✓ incident report about smoke at night         → incident: B2 haze (text + photo + CCTV) 0.711   (2nd: incident: AHU-3 leak (text 0.659)
  ✕ what does the plant room normally look like  → incident: B2 haze (text + photo + CCTV) 0.658   (2nd: cctv_plantroom.mp4 (video  0.645)
⚠ 3/4 as expected
◆ an embedding scores what IS in the input. 'Normally' asks about smoke being ABSENT — the haze record is still a plant room, plus text, so it wins. Absence and negation need a rule or a classifier.
```

vector เดียวต่อหนึ่งเหตุการณ์ช่วยให้ index มีขนาดเล็ก และให้คำถามเดียวเจอเรื่องราวทั้งหมดได้ บรรทัด ✕ คือบรรทัดที่สำคัญ embedding วัดสิ่งที่**มีอยู่**ใน input คำว่า "normally" หรือ "without smoke" จึงไม่ได้ดัน record ที่มีหมอกควันให้ออกห่างไป คำถามเรื่องการไม่มีอยู่ (absence) หรือการปฏิเสธ (negation) ต้องมี filter, rule หรือ classifier ซ้อนอยู่ด้านบน

✓ Checkpoint: คุณคำนวณได้ว่าวิดีโอยาวกี่วินาทีจะใส่ได้พอดีคู่กับรูปภาพหนึ่งรูปและข้อความ 30 token (8,192 − 310 = 7,882 token, ÷ 140 ≈ 56 วินาที)

## 3 · ไทยและอังกฤษในพื้นที่เดียว

โมเดลรองรับมากกว่า 100 ภาษา คำถามภาษาไทยควรหาโน้ตภาษาอังกฤษเจอ และกลับกันก็ได้เช่นกัน โดยไม่ต้องแปล

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/04_multimodal_search/labs/lab04_3_cross_lingual.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
▣ STEP 1 · Thai questions → English notes
  ✓ ปั๊มสระว่ายน้ำตัดบ่อย ทำยังไงดี    → Pool pump trip (0.773)
  ✓ มีคนติดอยู่ในลิฟต์                 → Lift entrapment (0.766)
  ✓ ใบแจ้งหนี้ขอเปลี่ยนเลขบัญชีธนาคาร  → Supplier bank detail changes (0.737)
  ✓ ห้องพักร้อนมาก แขกบ่น              → Guest room too hot (0.677)

▣ STEP 2 · English questions → Thai notes
  ✓ the air conditioner smells like something is burning → แอร์มีกลิ่นไหม้ (0.699)
  ✓ water is dripping from the ceiling                   → น้ำรั่วจากเพดาน (0.721)
✓ 6/6 cross-lingual questions found the right note first

▣ STEP 3 · the Thai voicemail, asked in Thai and in English
  ✓ แขกโทรมาบอกว่าแอร์มีกลิ่นไหม้                    → voicemail_burning_th.wav  flood 0.575 · burning 0.707 · towels 0.588
  ✕ a guest called about a burning smell from the AC → voicemail_flood_en.wav  flood 0.654 · burning 0.628 · towels 0.651

▣ STEP 4 · does the Thai voicemail sit near its Thai note or its English twin?
│ pair                                          cosine
│ ────────────────────────────────────────────  ──────
│ Thai voicemail ↔ Thai note                    0.691
│ Thai voicemail ↔ English note (same meaning)  0.644
│ Thai voicemail ↔ towel request voicemail      0.702
```

ข้อความกับข้อความทำงานได้ดีทั้งสองภาษา ได้ 6 จาก 6 ส่วนเสียงพูดยากกว่า ถ้าถามเป็นภาษาไทย ข้อความเสียงภาษาไทยจะถูกพบเป็นอันดับแรก แต่ถ้าถามเป็นภาษาอังกฤษจะไม่เจอ ถึงแม้คำถามใน Lab 04-1 จะหาเจอเพราะระบุว่า "a Thai speaker" ก็ตาม

✓ Checkpoint: คุณบอกได้ว่าคำถามแบบไหนของผู้ใช้คุณที่ข้ามทั้งขอบเขตภาษาและขอบเขต modality ไปพร้อมกัน

## 4 · Modality gap

Step 4 ด้านบนแสดงรูปแบบนี้ให้เห็น ข้อความเสียงภาษาไทยอยู่ใกล้ข้อความเสียงเรื่อง**ผ้าเช็ดตัว**ที่ไม่เกี่ยวข้องกันเลย (0.702) มากกว่าโน้ตภาษาอังกฤษที่มีความหมายเหมือนกันทุกประการ (0.644) item ที่เป็น modality เดียวกันจะจับกลุ่มกัน ไม่ว่าความหมายจะเป็นอะไร **modality gap** นี้พบได้ทั่วไปในโมเดล multimodal embedding ไม่ใช่บั๊กของโมเดลนี้ ผลในทางปฏิบัติคือ ถ้าคุณจัดอันดับโน้ต รูปภาพ และข้อความเสียงรวมในรายการเดียว โน้ตมักจะเบียดสื่ออื่นตกอันดับไป

มีวิธีแก้สองแบบ และทั้งสองแบบจะใช้ในบทต่อ ๆ ไปของสัปดาห์นี้:

| วิธีแก้ | ทำอย่างไร | ใช้ที่ไหน |
|---|---|---|
| กรองตามชนิด | ค้นหาเฉพาะ `audio` เมื่อมีคนถามว่า "ข้อความเสียงไหน…" | `kinds=` ใน Lab 05-2 |
| จัดอันดับแยกตามชนิด แล้วค่อยรวม | เลือกผลลัพธ์อันดับต้น ๆ **ของแต่ละชนิด** แล้วแสดงทั้งหมด | tool `search_memory` ใน Lab 07-2 |

✓ Checkpoint: คุณอธิบายได้ว่าทำไมข้อความเสียงของแขกอาจได้อันดับต่ำกว่าโน้ตที่เกี่ยวข้องแค่หลวม ๆ และบอกวิธีแก้ได้หนึ่งวิธี

## Labs — รันแล็บได้ที่นี่

**labs/lab04_1_cross_modal.py** — คำถามภาษาอังกฤษธรรมดาเจ็ดข้อ ค้นหาในรูปภาพหกรูปและข้อความเสียงสามรายการที่อยู่ในพื้นที่เดียวกัน พร้อมความคล้ายแบบรูปภาพกับรูปภาพ

**labs/lab04_2_video_and_interleaved.py** — งบ token, คลิป CCTV และ record เหตุการณ์แบบ interleaved สองรายการ พร้อมกรณีหนึ่งที่มันตอบผิดและเหตุผลว่าทำไม

**labs/lab04_3_cross_lingual.py** — ไทย ⇄ อังกฤษ ข้ามโน้ตและข้อความเสียงภาษาไทย พร้อมวัด modality gap

## Try it yourself — ลองทำเอง

`exercises/ex04_token_budget.py` คือด่านป้องกันฝั่ง ingest สำหรับ record แบบ interleaved:

1. `record_tokens(...)`: ต้นทุน token ของ record หนึ่งรายการ
2. `fits(tokens)` และ `seconds_of_video_left(tokens)`
3. `check_record(record)`: placeholder หนึ่งตัวต่อสื่อหนึ่งรายการ แยกตาม modality ถ้าไม่ตรวจ `<|image|>` ที่หายไปจะพังแบบเงียบ ๆ
4. task prefix ตัวไหนที่ต้องใส่หน้ารูปภาพ?

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/04_multimodal_search/exercises/ex04_token_budget.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ เมื่อทำ TODO ครบทุกจุด)

```
✓ record_tokens: 30+280+10×140 = 1710 · 60 s audio = 1500 · 100+560+300 = 960
✓ fits(): ≤ 8,192 tokens · 1,710 used leaves room for 46 more seconds of video
✓ check_record: one placeholder per media item, per modality — problems listed, none for a good record
✓ media items get NO task prefix (prefixes are for text only)
```

<details><summary>คำใบ้ — check_record()</summary>

สำหรับแต่ละ `image`, `video`, `audio` ที่มีอยู่ใน record ให้นับ `record["text"].count("<|image|>")` (และตัวอื่น ๆ ในทำนองเดียวกัน) string นับเป็นหนึ่งรายการ ส่วน list นับเป็น `len(list)` รายงานทุก modality ที่ตัวเลขสองตัวนี้ไม่ตรงกัน

</details>

✓ Checkpoint: ตัวตรวจขึ้น ✓ ครบทั้งสี่บรรทัด

## Troubleshooting — แก้ปัญหา

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `torchcodec is not installed` | การถอดรหัสวิดีโอต้องใช้ torchcodec | `uv pip install --python week30_multimodal_embeddings_edge/.venv-embed/bin/python torchcodec` |
| torchcodec หาไลบรารีของ ffmpeg ไม่เจอ | ไม่มี ffmpeg บนเครื่อง | `brew install ffmpeg` (macOS) หรือ ffmpeg ของ distro ที่คุณใช้ |
| vector ของเสียงดูเหมือนค่าสุ่ม | เสียงเป็น stereo หรือ 44.1 kHz | แปลงเป็น mono 16 kHz: `ffmpeg -i in.wav -ac 1 -ar 16000 out.wav` |
| หน่วยความจำไม่พอเมื่อโหลด encoder ทั้งหมด | 740M แบบ float32 ใช้หน่วยความจำสูงสุดราว 4.7 GB (Lab 02-3) | โหลดเฉพาะ encoder ที่ต้องใช้ หรือใช้ bfloat16 บน GPU |

## Next — บทถัดไป

[Lab 05 — Sovereign RAG: nothing leaves the building](../05_sovereign_rag/TUTORIAL.md): พิสูจน์ด้วย network log, เก็บทุกอย่างไว้ในไฟล์เดียวบนเครื่อง และให้ Gemma 4 ที่รันในเครื่องตอบคำถามจากไฟล์นั้น
