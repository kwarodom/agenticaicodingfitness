# ▶ Embed Lab 01 — local, sovereign และ edge: บันไดอุปกรณ์

> ส่วนหนึ่งของ Week 30 · multimodal embedding ด้วย EmbeddingGemma 2 บนฮาร์ดแวร์แบบ local, sovereign และ edge ทุกแล็บรันบนเครื่องของคุณเอง ตัวเลขทุกตัวในคอร์สนี้วัดจากเครื่องที่รันจริง หรือไม่ก็ระบุแหล่งที่มาไว้ชัดเจน

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- แยกให้ออกว่า AI แบบ **local**, **sovereign** และ **edge** ต่างกันอย่างไร และทำไมทั้งสามคำไม่ได้สัญญาเรื่องเดียวกัน
- ไต่บันไดอุปกรณ์ของ Tina Huang ตั้งแต่ Arduino ที่มีหน่วยความจำ 32 KB ไปจนถึง GPU H100 แปดตัว พร้อมสูตรคร่าว ๆ ของเธอว่าโมเดลขนาดไหนใส่ได้
- เรียนรู้ไอเดียเดียวที่อธิบายบันไดทั้งหมด: **capacity** ตัดสินว่าอะไรใส่ได้ ส่วน **bandwidth** ตัดสินว่ารันได้เร็วแค่ไหน
- ดูว่า **embedding model** อยู่ตรงไหนบนบันได คำตอบคือต่ำกว่า chat model ทุกตัวหลายขั้น
- วัดเครื่องของคุณเอง: RAM, memory bandwidth ของ CPU และ memory bandwidth ของ GPU

**Time** ~35 นาที · **Difficulty** ระดับเริ่มต้น · **Hardware** แล็ปท็อปเครื่องไหนก็ได้ · ไม่ต้องดาวน์โหลด model

**แหล่งอ้างอิง:** Tina Huang, [*Every Size Local AI In 24 Minutes*](https://youtu.be/rPGJhrunbxo) (28 ก.ย. 2026) · [บล็อกเปิดตัว EmbeddingGemma 2](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/) · [model card](https://huggingface.co/google/embeddinggemma-2) · [Sector88 — What is sovereign AI?](https://www.sector88.co/blog/what-is-sovereign-ai) · [`week30_multimodal_embeddings_edge/RESEARCH.md`](../RESEARCH.md)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| Python ของ Week 30 | `week30_multimodal_embeddings_edge/.venv-embed/bin/python --version` → 3.12 | ทุกแล็บรันในนี้ (torch, transformers 5.19, sentence-transformers 6.1) |
| Embed Lab Runner | `.venv/bin/python week30_multimodal_embeddings_edge/00_embed_lab_runner/tutorial_server.py` | เนื้อหาคอร์ส ปุ่ม ▶ run และ ⌨ terminal ที่ http://127.0.0.1:8130 |

ถ้ายังไม่มี `.venv-embed` ใน `week30_multimodal_embeddings_edge/README.md` มีคำสั่ง `uv` สองบรรทัดสำหรับสร้างไว้ให้แล้ว

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python -c "import torch, transformers, sentence_transformers as s; print(torch.__version__, transformers.__version__, s.__version__)"
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
2.14.1 5.19.0 6.1.0
```

✓ Checkpoint: คำสั่งพิมพ์เวอร์ชันออกมาสามตัว โดย transformers เป็น 5.19 ขึ้นไป และ sentence-transformers เป็น 6.1 ขึ้นไป

## 1 · local, sovereign, edge — คำสัญญาสามแบบที่ไม่เหมือนกัน

หลายคนใช้สามคำนี้เหมือนเป็นความหมายเดียวกัน แต่จริง ๆ ไม่ใช่

| คำ | คำสัญญา | วิธีทดสอบ |
|---|---|---|
| **local** | model รันบนฮาร์ดแวร์ที่อยู่ตรงหน้าคุณ | ถอดสาย LAN ออก แล้วมันยังตอบได้อยู่ไหม? |
| **sovereign** | คุณควบคุมฮาร์ดแวร์ เวอร์ชันของ model และข้อมูลเอง ภายใต้เขตอำนาจกฎหมายของคุณ และไม่มีอะไรแอบส่งข้อมูลกลับออกไป | ใครบ้างที่อ่าน แก้ หรือปิดระบบนี้ได้ และเขาอยู่ประเทศไหน? |
| **edge** | inference เกิดขึ้นตรงจุดที่ข้อมูลถูกสร้าง เช่น โทรศัพท์ gateway หรือกล่องในห้องเครื่อง | ข้อมูลต้องเดินทางไปที่อื่นก่อนถึงจะถูกเข้าใจหรือเปล่า? |

นิยามหนึ่งของ sovereign AI ในปี 2026 คือ "การสร้างและรันปัญญาประดิษฐ์บนโครงสร้างพื้นฐานที่คุณเป็นเจ้าของหรือควบคุมโดยตรง ภายในเขตอำนาจกฎหมายของคุณเอง โดยใช้ model และข้อมูลที่ไม่เคยออกนอกขอบเขตของคุณ" (Sector88) ส่วนคลาวด์ของ vendor ที่ตั้ง region ไว้ในประเทศของคุณ นั่นคือ **data residency** ไม่ใช่ sovereignty เพราะ vendor ยังเป็นคนดูแลระบบอยู่

edge AI คือจุดที่ sovereignty กลายเป็นเรื่องของฟิสิกส์ ถ้า model อยู่บนอุปกรณ์ ข้อมูลก็ไม่จำเป็นต้องออกไปไหนเลย

สัปดาห์นี้เพิ่มอีกประเด็นที่การพูดคุยส่วนใหญ่มองข้าม: **embedding คือข้อมูลอ่อนไหว** vector หนึ่งตัวคือสำเนาแบบบีบอัดของ "ความหมาย" ของข้อความเสียงจากแขก เฟรมจากกล้อง CCTV หรือบันทึกงานซ่อมบำรุง งานวิจัยแสดงให้เห็นว่าข้อความส่วนใหญ่สร้างกลับคืนได้จาก embedding ของมัน (Morris et al., EMNLP 2023) การส่งเอกสารไปที่ embedding API บนคลาวด์จึงนับเป็นการส่งข้อมูลออก ถึงแม้คุณจะเก็บไว้แค่ vector ก็ตาม

✓ Checkpoint: คุณอธิบายได้ว่าทำไม "vendor ของเราโฮสต์ไว้ในประเทศไทย" จึงเป็นแค่ residency และต้องทำอะไรเพิ่มถึงจะเป็น sovereign

## 2 · บันไดอุปกรณ์

ในคลิป *Every Size Local AI In 24 Minutes* Tina Huang ลองรัน AI บนฮาร์ดแวร์ทุกขนาดที่เธอหามาได้ แล้วจัดกลุ่มเป็นห้าประเภท:

| ประเภท | อุปกรณ์ของเธอ | หน่วยความจำ (ตามที่เธอบอก) | รันอะไรได้ |
|---|---|---|---|
| ชิป (ไม่มี OS) | Arduino Uno R4 | 32 KB | ไม่ได้เลย — แต่เรียกใช้เครื่องที่ใหญ่กว่าผ่าน Wi-Fi ได้ |
| ชิป | ESP32-S3 (~$20) | 8 MB | TinyStories 260K และ 3M |
| มินิคอมพิวเตอร์ | Raspberry Pi 5 | 8 GB | Whisper → Qwen3 1.7B → Piper เป็น voice loop ครบวง; Moondream กับเฟรมจากเว็บแคม |
| มินิคอมพิวเตอร์ | iPhone 16 | 8 GB (แอปได้ใช้ ~4–5 GB) | Gemma 4 (AI Edge Gallery), Stable Diffusion 1.5, Whisper Large, YOLO11n |
| คอมพิวเตอร์ส่วนตัว | MacBook Pro | 32 GB | "แทบทุกประเภท": Qwen3 14B–32B, Qwen3 Coder 30B, Flux Dev |
| เซิร์ฟเวอร์ในบ้าน | Mac Studio · AMD Ryzen AI Halo | 64 GB · 128 GB | LLM ขนาด 27–32B · Llama 3.3 70B, GPT-OSS 120B |
| GPU | RTX 4090 · 8× H100 | 24 GB · 640 GB | สร้างภาพและวิดีโอได้เร็ว · model ขนาดใหญ่มาก |

สูตรคร่าว ๆ ของเธอสำหรับดูว่า model ไหนใส่ได้: เอา RAM มาหัก 25 % เผื่อไว้ให้อย่างอื่น แล้วหารด้วย 0.6 ผลที่ได้คือขนาด model ใหญ่สุดในหน่วย **พันล้านพารามิเตอร์ (B)** ตัวอย่างของเธอเอง: 32 GB → ใช้ได้ 24 GB → 24 ÷ 0.6 = **40B** ตัวเลข 0.6 GB ต่อพันล้านพารามิเตอร์นี้ใกล้เคียงกับ chat model แบบ 4-bit บวก overhead

Lab 01-1 ไล่ทุกขั้นของบันไดด้วยสูตรนี้

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/01_local_sovereign_edge/labs/lab01_1_device_ladder.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
▣ STEP 1 · her rule of thumb, rung by rung
│ class              device                        memory  usable (−25 %)  max params ≈ usable ÷ 0.6
│ ─────────────────  ────────────────────────────  ──────  ──────────────  ─────────────────────────
│ chip               Arduino Uno R4                32 KB   24 KB           —
│ chip               ESP32-S3                      8 MB    6 MB            9.8M
│ mini computer      Raspberry Pi 5                8 GB    6 GB            10.0B
│ mini computer      iPhone 16 (apps get ~4–5 GB)  4.5 GB  3.4 GB          5.6B
│ personal computer  MacBook Pro (the video's)     32 GB   24 GB           40.0B
│ home server        Mac Studio                    64 GB   48 GB           80.0B
│ home server        AMD Ryzen AI Halo             128 GB  96 GB           160.0B
│ GPU                RTX 4090                      24 GB   18 GB           30.0B
│ GPU                8× H100                       640 GB  480 GB          800.0B
◆ worked example from the video: 32 GB → 24 GB usable → 24 ÷ 0.6 = 40B parameters
✓ our function agrees: max_params_b(32) = 40.0
```

✓ Checkpoint: ใช้สูตรนี้แล้ว คุณบอกได้ว่าแล็ปท็อป 16 GB ใส่ model ได้กี่พันล้านพารามิเตอร์ (≈ 20B)

## 3 · capacity กับ bandwidth — เปรียบกับห้องครัว

ไอเดียที่ดีที่สุดในคลิปคือการเปรียบเทียบ ให้คิดว่าคอมพิวเตอร์ของคุณคือครัวร้านอาหาร:

| ห้องครัว | คอมพิวเตอร์ | หน้าที่ |
|---|---|---|
| ตู้แช่แข็งชั้นล่าง | storage (SSD) | ที่เก็บ model ตอนที่ไม่ได้รัน |
| เคาน์เตอร์เตรียมของ | RAM | ที่วาง model ระหว่างรัน ขนาดของมันคือ **capacity** |
| สายพานลำเลียง | memory bus | ส่งวัตถุดิบไปให้พ่อครัว ความเร็วของมันคือ **bandwidth** |
| หัวหน้าเชฟ | CPU | ทำได้ทุกอย่าง แต่มีอยู่คนเดียว |
| ทีมพ่อครัวในไลน์ | GPU | คนงานที่ทำงานง่าย ๆ จำนวนมากพร้อมกัน |
| เครื่องหั่นหัวหอม | NPU | ทำงานเดียว ทำได้เร็ว และแทบไม่กินไฟ |

Raspberry Pi 5 กับ iPhone 16 มี RAM 8 GB เท่ากัน คือเคาน์เตอร์ขนาดเดียวกัน แต่ iPhone เร็วกว่ามาก เพราะสายพานกว้างกว่า และมีทั้ง GPU และ NPU capacity ตัดสินว่า **อะไรใส่ได้** ส่วน bandwidth ตัดสินว่ารันได้ **เร็วแค่ไหน** chat model เขียนทีละ token และต้องดึง weight ทั้งหมดผ่านสายพานทุกครั้งที่สร้าง token ใหม่ ดังนั้นสำหรับงาน chat สายพานมักเป็นคอขวด Module 06 จะแสดงให้เห็นว่า embedding model มีพฤติกรรมต่างออกไป

Lab 01-2 วัดสายพานของเครื่องคุณเอง

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/01_local_sovereign_edge/labs/lab01_2_measure_this_machine.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
▣ STEP 2 · bandwidth — CPU memory copy (numpy, one thread)
✓ CPU copy: 139 GB/s (best of 5, 1 GiB array)

▣ STEP 3 · bandwidth — GPU memory copy (PyTorch)
✓ GPU copy on mps: 314 GB/s (best of 10, 1 GiB tensor)

▣ STEP 4 · where this machine sits on the ladder
│ measure                        value     meaning
│ ─────────────────────────────  ────────  ──────────────────────────
│ capacity                       36 GB     personal computer
│ CPU copy bandwidth (measured)  139 GB/s  one core's view of the bus
│ GPU copy bandwidth (measured)  314 GB/s  closer to the vendor peak
◆ vendor peaks for comparison (RESEARCH.md §4): Raspberry Pi 5 ~17 GB/s · Apple M4 Max (14-core) 410 GB/s · AMD Ryzen AI Max+ 395 256 GB/s · RTX 4090 1,008 GB/s · H100 SXM 3,350 GB/s
```

การ copy บน GPU ทำได้ 314 GB/s จากค่าสูงสุด 410 GB/s ของ M4 Max ความเร็วที่วัดได้จริงจะต่ำกว่าค่าสูงสุดใน spec sheet เสมอ และควรรู้ช่องว่างนี้ไว้ก่อนจะไปสัญญาเรื่อง latency กับใคร

✓ Checkpoint: คุณรู้ capacity ของเครื่องตัวเอง รู้ bandwidth ของการ copy บน CPU และ GPU ที่วัดได้ และรู้ว่าเครื่องอยู่ขั้นไหนของบันได

## 4 · embedding model อยู่ตรงไหนบนบันได

คลิปพูดถึงแค่ model แบบ **generative** คือ chat ภาพ วิดีโอ และเสียง แต่ข้าม model ประเภทที่ขับเคลื่อนการค้นหา RAG และหน่วยความจำของ agent ไป นั่นคือ **embedding model** embedding model อ่าน input แล้วคืนรายการตัวเลขความยาวคงที่ออกมา เรียกว่า vector มันไม่เคยเขียนข้อความ จึงมีขนาดเล็กได้

EmbeddingGemma 2 (Google DeepMind เปิดตัววันที่ 6 ตุลาคม 2026) มีพารามิเตอร์รวม 740M และออกแบบเป็นโมดูล: text model ขนาด 270M บวก vision encoder 170M และ audio encoder 300M ซึ่งโหลดเฉพาะตอนที่ต้องใช้ ขั้นที่ 2 ของ Lab 01-1 แปลงตัวเลขนี้เป็นขนาดหน่วยความจำ:

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
│ configuration           params  float32  bfloat16  int8 / Q8_0  4-bit / Q4
│ ──────────────────────  ──────  ───────  ────────  ───────────  ──────────
│ text only               270M    1080 MB  540 MB    270 MB       135 MB
│ text + image            440M    1760 MB  880 MB    440 MB       220 MB
│ text + audio            570M    2280 MB  1140 MB   570 MB       285 MB
│ all four (incl. video)  740M    2960 MB  1480 MB   740 MB       370 MB
◆ weights only — activations, the tokenizer and the runtime add more on top
◆ REFERENCE (launch blog): quantized on a Pixel 11 Pro, text-only uses ~191MB active RAM, full multimodal ~567MB
```

Google ระบุว่า text model ที่ผ่าน quantization แล้วใช้ RAM ประมาณ 191 MB บนโทรศัพท์ ซึ่งใส่ได้ทุกขั้นตั้งแต่ Raspberry Pi ขึ้นไป ESP32-S3 ยังใส่ไม่ได้ เพราะใช้ได้แค่ 6 MB แต่มันก็ยังใช้ embedding ได้อยู่ดี โดยส่งประโยคไปให้เครื่องที่ใหญ่กว่าใน Wi-Fi วงเดียวกัน แล้วรับ vector กลับมา เป็นเทคนิคเดียวกับที่ Arduino ในคลิปใช้กับ Mac Studio

✓ Checkpoint: คุณอธิบายได้ว่าทำไม Raspberry Pi จึงรัน search index ของโรงแรมได้ทั้งวัน ทั้งที่รัน chat model ขนาด 7B แทบไม่ไหว

## แล็บ — รันได้ที่นี่

**labs/lab01_1_device_ladder.py** — บันไดอุปกรณ์ของ Tina Huang พร้อมสูตรคำนวณว่าอะไรใส่ได้ และตำแหน่งของ EmbeddingGemma 2 บนบันได

**labs/lab01_2_measure_this_machine.py** — อ่าน RAM ของเครื่องจาก OS แล้ววัด memory bandwidth ของ CPU และ GPU ด้วยการ copy ขนาด 1 GiB

## ลองทำเอง

ใน `exercises/ex01_fit_formula.py` มี TODO อยู่สี่ข้อ:

1. `usable_gb(ram)`: "หัก 25 %" ตามสูตรของ Tina
2. `max_params_b(ram)`: GB ที่ใช้ได้ ÷ 0.6
3. `weights_mb(params_m, dtype)`: จำนวนพารามิเตอร์ × จำนวน byte ต่อพารามิเตอร์
4. ติดป้ายให้การ deploy สี่แบบว่าเป็น `sovereign`, `residency only` หรือ `not sovereign`

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/01_local_sovereign_edge/exercises/ex01_fit_formula.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ หลังเติม TODO ครบทุกข้อ)

```
✓ usable_gb(32) = 24
✓ max_params_b: 32 GB → 40B · 64 GB → 80B · 8 GB → 10B
✓ weights_mb: text bf16 540 MB · full fp32 2960 MB · text q4 135 MB
✓ sovereignty: cloud API ✕ · vendor region = residency only · own server, no egress ✓ · offline phone ✓
```

<details><summary>คำใบ้ — region ของ vendor</summary>

ถามว่าใครเป็นคนดูแลเครื่อง และใครอาจถูกบังคับให้ส่งมอบข้อมูล ถ้าคำตอบคือ "vendor" แปลว่าข้อมูลอยู่ในประเทศของคุณ แต่อำนาจควบคุมไม่ได้อยู่ นั่นคือ residency

</details>

✓ Checkpoint: ตัวตรวจขึ้น ✓ ครบทั้งสี่บรรทัด

## การแก้ปัญหา

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `week30_multimodal_embeddings_edge/.venv-embed/bin/python: No such file or directory` | ยังไม่ได้สร้าง venv ของ Week 30 | รันคำสั่ง `uv` ใน `week30_multimodal_embeddings_edge/README.md` |
| Lab 01-2 แจ้งว่า "no GPU backend here" | ไม่มี Apple GPU (MPS) หรือ NVIDIA GPU (CUDA) | ไม่เป็นไร เครื่องของคุณใช้ CPU อย่างเดียว เหมือน Raspberry Pi ในคลิป |
| bandwidth ที่วัดได้ไม่เท่ากันในแต่ละรอบ | แอปอื่นใช้ memory bus ร่วมกันอยู่ | ปิดแอปที่หนัก ๆ แล็บเลือกค่าที่ดีที่สุดจากหลายรอบให้อยู่แล้ว |

## ถัดไป

[Lab 02 — รู้จัก EmbeddingGemma 2](../02_first_embeddings/TUTORIAL.md): ดาวน์โหลด model ครั้งเดียว แปลงประโยคเป็น vector แล้วลองค้นหาด้วยความหมายเป็นครั้งแรก
