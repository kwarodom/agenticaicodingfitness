# ▶ Embed Lab 02 — รู้จัก EmbeddingGemma 2: vector, ความคล้าย และ task prefix

> ส่วนหนึ่งของ Week 30 · Multimodal embedding ด้วย EmbeddingGemma 2 บนฮาร์ดแวร์ local, sovereign และ edge ทุกแล็บรันบนเครื่องของคุณเอง ทุกตัวเลขในคอร์สนี้วัดจากเครื่องที่รันจริง หรือไม่ก็อ้างอิงพร้อมแหล่งที่มา

**สิ่งที่คุณจะได้ทำจริง**
- ดาวน์โหลด EmbeddingGemma 2 ครั้งเดียว (1.5 GB) หลังจากนั้นทุกแล็บรันแบบออฟไลน์จาก cache ในเครื่อง
- แปลงประโยคหนึ่งให้เป็นตัวเลข 768 ตัว และดูว่าทำไมความยาวของ vector นั้นเท่ากับ 1
- จัดอันดับบันทึกงานโรงแรม 18 รายการตาม **ความหมาย** โดยไม่ต้องมีคำซ้ำกันเลย
- วัดว่า **task prefix** ของ model ช่วยได้แค่ไหน ด้วยคำถามจริง 18 ข้อ
- โหลดเฉพาะ encoder ที่ต้องใช้ และวัดว่าแต่ละตัวกินหน่วยความจำเท่าไร

**Time** ~45 นาที · **Difficulty** ระดับเริ่มต้น · **Hardware** แล็ปท็อปเครื่องไหนก็ได้ · ดิสก์ 1.5 GB, RAM ว่าง ~2 GB

**แหล่งอ้างอิง:** [EmbeddingGemma 2 model card](https://huggingface.co/google/embeddinggemma-2) (snapshot: `week30_multimodal_embeddings_edge/common/sources/embeddinggemma-2-model-card.md`) · [บล็อกเปิดตัว](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/) · [sentence-transformers](https://sbert.net) · [`week30_multimodal_embeddings_edge/RESEARCH.md`](../RESEARCH.md) §1

## 0 · ก่อนเริ่ม

model นี้ใช้ไลเซนส์ Apache 2.0 และ **ไม่ gated**: ไม่ต้องล็อกอิน Hugging Face Lab 02-1 จะดาวน์โหลดให้ในครั้งแรกที่รัน checkpoint คือไฟล์ `model.safetensors` ขนาด 1,488,915,288 ไบต์ หลังจากนั้น `embedkit` จะตั้ง `HF_HUB_OFFLINE=1` ทันทีที่เจอ model ใน cache ของคุณ จึงไม่มีแล็บไหนต่อเน็ตอีก

```bash
# on: laptop
ls ~/.cache/huggingface/hub/ | grep embeddinggemma-2 || echo "not downloaded yet — Lab 02-1 fetches it"
```

ข้อมูลสัปดาห์นี้เป็นข้อมูล **สังเคราะห์**: `week30_multimodal_embeddings_edge/common/data/hotel_kb.json` มีบันทึกงานโรงแรมที่แต่งขึ้น 18 รายการ (เป็นภาษาไทย 2 รายการ) และคำถาม 18 ข้อ แต่ละข้อมีคำตอบที่ถูกกำกับไว้ ส่วน `week30_multimodal_embeddings_edge/common/media/` มีรูปถ่าย ข้อความเสียง และคลิป CCTV แบบสังเคราะห์ ที่ทำไว้ตั้งแต่ Week 24

✓ Checkpoint: คุณรู้ว่า model จะถูก cache ไว้ที่ไหน และรู้ว่าไม่ต้องใช้บัญชีหรือ token ใดๆ

## 1 · embedding คือจุดหนึ่งในพื้นที่ของความหมาย

embedding model อ่าน input แล้วคืนรายการตัวเลขที่ยาวคงที่ เรียกว่า **vector** input ที่มีความหมายใกล้กันจะได้ vector ที่ชี้ไปทิศทางใกล้กัน เราเปรียบเทียบ vector สองตัวด้วย **cosine similarity**: 1 คือทิศเดียวกัน 0 คือไม่เกี่ยวกัน −1 คือตรงข้ามกัน

module สุดท้ายของ EmbeddingGemma 2 จะ normalise ทุก vector ให้ยาวเท่ากับ 1 ทำให้ cosine similarity กลายเป็นแค่ dot product ธรรมดา นี่คือเหตุผลที่ vector database จัดอันดับข้อมูลเป็นล้านรายการได้อย่างรวดเร็ว

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/02_first_embeddings/labs/lab02_1_first_vectors.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
▣ STEP 1 · load the text-only model (vision and audio encoders left out)
✓ loaded google/embeddinggemma-2 · encoders text · 271.0M params · mps · 1.0s

▣ STEP 2 · one sentence → one vector
│ property        value
│ ──────────────  ─────────────────────────────────────────────────
│ shape           (768,)
│ dtype           float32
│ first 6 values  [-0.0221  0.0083  0.0347 -0.0084  0.0117  0.014 ]
│ L2 norm         0.9996
✓ 768 numbers, length ≈ 1 — the model's last module normalises every vector

▣ STEP 3 · the model card's quick start — similarity is a dot product of unit vectors
│                               lovely  sunny  stadium
│ ────────────────────────────  ──────  ─────  ───────
│ The weather is lovely today.  1.000   0.851  0.672
│ It's so sunny outside!        0.851   1.000  0.681
│ He drove to the stadium.      0.672   0.681  1.000
✓ 'lovely' vs 'sunny' = 0.851 > 'lovely' vs 'stadium' = 0.672
```

ค่า norm แสดงเป็น 0.9996 ไม่ใช่ 1 พอดี เพราะ model รันแบบ bfloat16 บน GPU ของ Apple ซึ่งเก็บความแม่นยำได้ราวสามหลัก

✓ Checkpoint: คุณบอกได้ว่า vector มี shape เท่าไร (768) และทำไมในกรณีนี้ cosine similarity จึงเท่ากับ dot product

## 2 · การค้นหาตามความหมายครั้งแรก

STEP 4 ของแล็บเดียวกันจะ embed บันทึกโรงแรม 18 รายการ แล้วถามสามคำถามที่แทบไม่มีคำซ้ำกับคำตอบที่ถูกเลย

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
▣ STEP 4 · first semantic search over the hotel knowledge base (SYNTHETIC)
◆ embedded 18 documents in 0.24s
  'water on the floor under the air handler'
     1. 0.821  AHU condensate leak
     2. 0.659  Bathroom flooding
  'a guest wants more towels'
     1. 0.780  Towel and amenity requests
     2. 0.692  Guest room too hot
  'someone is stuck in the lift'
     1. 0.819  Lift entrapment
     2. 0.661  Pool pump trip
```

"Air handler" เจอ "AHU" และ "lift" เจอ "Lift entrapment" ทั้งที่ในบันทึกไม่มีคำว่า "elevator" เลย keyword search หาคำ ส่วน embedding search หาความหมาย ลองดูช่องว่างระหว่างผลอันดับหนึ่งกับอันดับสองด้วย (0.821 กับ 0.659) ถ้าห่างกันชัด แปลว่า model มั่นใจในผลที่เจอ

✓ Checkpoint: คุณรันแล็บแล้ว และผลอันดับหนึ่งของทุกคำถามตรงกับบันทึกที่คุณเองก็จะเลือก

## 3 · task prefix — วัดจริง ไม่ใช่เดาเอา

EmbeddingGemma 2 ถูกเทรนโดยมีคำสั่งสั้นๆ นำหน้าข้อความทุกชิ้น คำถาม (query) ใช้ `task: search result | query: …` เอกสารใช้ `title: … | text: …` ส่วนสื่ออย่างรูปหรือเสียงไม่ต้องมี prefix model card บอกว่าถ้าไม่ใส่ก็ "ยังใช้ได้ แต่ความแม่นยำลดลง" Lab 02-2 จะวัดข้อนี้กับข้อมูลของสัปดาห์นี้

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/02_first_embeddings/labs/lab02_2_task_prefixes.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
▣ STEP 1 · the prompt strings this checkpoint ships with (config_sentence_transformers.json)
│ prompt_name         prepended text
│ ──────────────────  ─────────────────────────────────────
│ SearchQuery         'task: search result | query: '
│ Document            'title: none | text: '
│ QuestionAnswering   'task: question answering | query: '
│ Classification      'task: classification | query: '
│ Clustering          'task: clustering | query: '
│ SentenceSimilarity  'task: sentence similarity | query: '

▣ STEP 2 · three ways to embed the same data
│ how                                   recall@1  MRR
│ ────────────────────────────────────  ────────  ─────  ────────────────────
│ A · no prefixes at all                0.944     0.972  ███████████████████░
│ B · prompts (SearchQuery / Document)  1.000     1.000  ████████████████████
│ C · prompts + real titles             1.000     1.000  ████████████████████

▣ STEP 3 · where they disagree
  guest says room 1203 is boiling hot                  A→✕ แอร์มีกลิ่นไหม้             C→✓ Guest room too hot
```

ถ้าไม่ใส่ prefix คำถาม 1 ใน 18 ข้อไปเจอบันทึกผิด ถ้าใส่ ถูกครบทั้ง 18 ข้อ คำถาม 18 ข้อถือเป็นการทดสอบเล็กๆ หลักฐานที่หนักแน่นกว่าคือ benchmark ใน model card เอง แต่นิสัยนี้ทำง่ายมาก: ใส่ query prefix ทุกคำถาม และใส่ `title: … | text: …` ทุกเอกสาร ใน sentence-transformers ใช้ `prompt_name="SearchQuery"` สำหรับ query ส่วน `prompt_name="Document"` จะเขียน `title: none` ให้ ถ้ามี title จริงก็จัดรูปแบบเอง

✓ Checkpoint: คุณเขียน string ของ query และเอกสารสำหรับงาน question answering ได้ถูกต้องเป๊ะโดยไม่ต้องเปิดดู

## 4 · โหลดเฉพาะสิ่งที่ใช้

พารามิเตอร์ 740M แบ่งเป็นสามส่วนที่เลือกได้: text (270M), vision (170M) และ audio (300M) sentence-transformers จะไม่โหลด encoder ตัวไหนก็ตามที่ config เป็น `None`:

| modality ที่เปิดใช้ | `config_kwargs` | ขนาด (model card) |
|---|---|---|
| text อย่างเดียว | `{"vision_config": None, "audio_config": None}` | 270M |
| text + รูปภาพ | `{"audio_config": None}` | 440M |
| text + เสียง | `{"vision_config": None}` | 570M |
| ครบทุกอย่าง | `{}` | 740M |

Lab 02-3 โหลดแต่ละ configuration ใน process ใหม่ บน CPU แบบ float32 เหมือนที่กล่อง edge Linux ขนาดเล็กจะทำ แล้ววัดหน่วยความจำ

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/02_first_embeddings/labs/lab02_3_selective_loading.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
▣ STEP 5 · compare
│ configuration  params  weights fp32  RSS after encode  peak RSS  load
│ ─────────────  ──────  ────────────  ────────────────  ────────  ────  ────────────────────
│ text only      271M    1034 MB       1760 MB           2249 MB   2.9s  ██████████░░░░░░░░░░
│ text + image   439M    1674 MB       2405 MB           2929 MB   3.0s  █████████████░░░░░░░
│ text + audio   577M    2200 MB       1466 MB           3721 MB   3.3s  █████████████████░░░
│ all encoders   744M    2840 MB       1223 MB           4686 MB   3.6s  ████████████████████
✓ text vector with and without media encoders: cosine 1.000000 on the first 16 dims — identical
◆ size an edge box by the PEAK: macOS compresses idle pages, so RSS after encode can read below the weights
```

มีบทเรียนสองข้อ ข้อแรก หน่วยความจำ peak เพิ่มขึ้นกว่าเท่าตัว จาก text อย่างเดียว (2.2 GB) เป็นครบทุกอย่าง (4.7 GB) ใน float32 ดังนั้นบนกล่องเล็กๆ ให้โหลด text model ก่อน แล้วค่อยเพิ่ม encoder เฉพาะจุดที่ต้องค้นหารูปหรือเสียงได้ ข้อสอง vector ของ text **เหมือนกันทุกประการ** ไม่ว่าจะโหลดแบบไหน คุณจึง index บันทึกด้วย model เล็กวันนี้ แล้วเพิ่มรูปทีหลังได้ โดยไม่ต้อง embed อะไรใหม่เลย

✓ Checkpoint: คุณบอกได้ว่าจะ ship `config_kwargs` แบบไหนบน gateway ที่ค้นหาแค่บันทึกงานซ่อมบำรุง

## Labs — รันแล็บได้ที่นี่

**labs/lab02_1_first_vectors.py** — โหลด text model ดูข้างใน vector หนึ่งตัว ทำ quick start ของ model card ซ้ำ และค้นหาบันทึกโรงแรมตามความหมาย

**labs/lab02_2_task_prefixes.py** — คำถาม 18 ข้อชุดเดิม embed สามแบบ พร้อม recall@1 และ MRR เพื่อให้เห็นว่า task prefix ช่วยได้แค่ไหน

**labs/lab02_3_selective_loading.py** — แต่ละ configuration ของ encoder รันใน process ของตัวเอง พร้อมวัดขนาด weights, หน่วยความจำ resident และหน่วยความจำ peak

## Try it yourself — ลองทำเอง

`exercises/ex02_cosine_and_rank.py` มี TODO สี่จุด รวมกันแล้วคือแกนของทุกการค้นหาในสัปดาห์นี้ เขียนเองด้วย numpy:

1. `cosine(a, b)`
2. `rank(q, docs, k)`: k แถวที่ดีที่สุด เรียงจากคะแนนสูงไปต่ำ
3. `query_text(text, task)`: query prefix ตาม model card
4. `doc_text(title, text)`: รูปแบบเอกสาร โดยใช้ `title: none` เมื่อไม่มี title

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/02_first_embeddings/exercises/ex02_cosine_and_rank.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ เมื่อทำ TODO ครบทุกจุด)

```
✓ cosine: (3,4)·(4,3) = 0.96 · self = 1 · opposite = −1
✓ rank: best three documents in order [1, 2, 0]
✓ query_text: the model card's search and question-answering prefixes, exactly
✓ doc_text: 'title: … | text: …', and 'title: none' when there is no title
```

<details><summary>คำใบ้ — rank()</summary>

ให้คะแนนทุกแถวด้วย `cosine` ของคุณ เรียงคู่ `(index, score)` ตามคะแนนจากสูงไปต่ำ แล้วเก็บไว้ k ตัวแรก

</details>

✓ Checkpoint: ตัวตรวจขึ้น ✓ ครบทั้งสี่บรรทัด

## Troubleshooting — แก้ปัญหา

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `ValueError: … model type embedding_gemma2 …` | transformers เก่ากว่า 5.19 (`.venv` กลางของ repo เป็น 5.10) | รันแล็บด้วย `week30_multimodal_embeddings_edge/.venv-embed/bin/python` |
| รันครั้งแรกค้างที่ "loading" | กำลังดาวน์โหลด 1.5 GB | รอแค่ครั้งเดียว ครั้งต่อไปโหลดเสร็จในราว 2 วินาที |
| `OfflineModeIsEnabled` | ตั้ง `HF_HUB_OFFLINE=1` ไว้ แต่ model ยังไม่อยู่ใน cache | `unset HF_HUB_OFFLINE` หนึ่งครั้ง รัน Lab 02-1 แล้วค่อยกลับไปโหมดออฟไลน์ |
| vector เป็น NaN | มีอะไรสักอย่างโหลด model เป็น float16 | ใช้ bfloat16 หรือ float32 ห้ามใช้ float16 (ตาม model card; Module 06 ทดสอบเรื่องนี้) |

## Next — บทถัดไป

[Lab 03 — vector เล็กลง model เล็กลง](../03_matryoshka_and_quantization/TUTORIAL.md): ตัด vector เหลือตัวเลข 128 ตัว เก็บแต่ละค่าด้วย 1 ไบต์หรือ 1 บิต และย่อขนาดตัว model เอง
