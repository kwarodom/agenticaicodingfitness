# ▶ Embed Lab 05 — Sovereign RAG: ไม่มีข้อมูลไหนออกนอกอาคาร

> ส่วนหนึ่งของ Week 30 · multimodal embedding ด้วย EmbeddingGemma 2 บนฮาร์ดแวร์ในเครื่อง แบบ sovereign และบน edge ทุกแล็บรันบนเครื่องของคุณเอง ตัวเลขทุกตัวในคอร์สนี้วัดจากเครื่องที่รันจริง หรือไม่ก็ระบุแหล่งที่มาไว้

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- เปลี่ยนคำว่า "มันรันในเครื่อง" ให้เป็น **หลักฐาน**: log ของทุก network connection ที่ pipeline พยายามเปิด
- ดูการรั่วไหลที่จงใจสร้างขึ้นสองครั้งถูกบล็อก
- เก็บโน้ต รูปถ่าย ข้อความเสียง และ CCTV ของโรงแรมไว้ใน **ไฟล์ SQLite ไฟล์เดียว** ที่ 256 มิติ
- รัน retrieval-augmented generation ครบวงจรบนเครื่องของคุณ: EmbeddingGemma 2 หาโน้ตที่เกี่ยวข้อง แล้ว Gemma 4 ที่รันในเครื่องตอบเป็นภาษาอังกฤษและภาษาไทย พร้อม citation อ้างถึงโน้ตเหล่านั้น

**Time** ~50 นาที · **Difficulty** ระดับกลาง · **Hardware** แล็ปท็อปเครื่องไหนก็ได้ · Ollama พร้อม chat model ในเครื่องสำหรับ Lab 05-3

**แหล่งอ้างอิง:** [Sector88 — What is sovereign AI?](https://www.sector88.co/blog/what-is-sovereign-ai) · [VDF AI — data sovereignty risks](https://vdf.ai/blog/data-sovereignty-risks-2026-regulated-industries/) · Morris et al., [*Text Embeddings Reveal (Almost) As Much As Text*](https://arxiv.org/abs/2310.06816), EMNLP 2023 · พ.ร.บ. คุ้มครองข้อมูลส่วนบุคคล (PDPA) พ.ศ. 2562 · [`week30_multimodal_embeddings_edge/RESEARCH.md`](../RESEARCH.md) §3

## 0 · ก่อนเริ่ม

Lab 05-3 ต้องมี chat model ในเครื่องที่รันผ่าน Ollama แล็บจะเลือก `gemma4:12b` ถ้าคุณมี ถ้าไม่มีก็จะเลือก model อื่นที่อยู่ **ในเครื่อง** แล็บจะไม่เลือก model ที่ลงท้ายด้วย `:cloud` เด็ดขาด เพราะการส่งต่อไปคลาวด์ไม่ใช่ sovereign

```bash
# on: laptop
curl -s localhost:11434/api/tags | python3 -c "import json,sys; print([m['name'] for m in json.load(sys.stdin)['models'] if 'cloud' not in m['name']])"
```

✓ Checkpoint: คุณเห็น model ในเครื่องอย่างน้อยหนึ่งตัว หรือวางแผนว่าจะรัน `ollama pull gemma4:12b`

## 1 · ทำไมขั้น embedding จึงตัดสินว่า sovereign หรือไม่

RAG pipeline บนคลาวด์ส่งของสามอย่างออกนอกอาคาร: **เอกสาร** ทุกชิ้น (เพื่อทำ embedding), **คำถาม** ทุกข้อ (เพื่อทำ embedding และตอบ) และยังเก็บ **vector** ไว้บนดิสก์ของคนอื่นด้วย สำหรับโรงแรม เอกสารเหล่านี้คือข้อความเสียงของแขก CCTV รูปห้องพัก และบันทึกงานซ่อมบำรุง ซึ่งล้วนเป็นข้อมูลส่วนบุคคลตาม PDPA ของไทย

vector ไม่ใช่บทสรุปที่ปลอดภัย "embedding เข้ารหัสข้อมูลเกี่ยวกับเนื้อหาที่อ่อนไหวไว้ ถ้าการสร้าง embedding หรือการเก็บ vector เกิดขึ้นนอกการควบคุมขององค์กร ก็ถือเป็นความเสี่ยงด้านการโอนข้อมูลที่มีนัยสำคัญ" (VDF AI, 2026) นักวิจัยเคยสร้างข้อความต้นฉบับกลับคืนมาได้เป็นส่วนใหญ่จาก embedding อย่างเดียว (Morris et al., EMNLP 2023)

embedding ยังเป็นขั้นที่ **หนักที่สุด** ด้วย เพราะต้องอ่านทุกรูปและเสียงทุกวินาที ทำขั้นนี้ในเครื่องก็แก้ปัญหาการโอนข้อมูลไปได้เกือบหมดในคราวเดียว จากนั้นค่อยให้ LLM รันในเครื่องด้วย (Section 4)

✓ Checkpoint: คุณบอกได้ว่า RAG pipeline บนคลาวด์ส่งอะไรออกไปบ้างสามอย่าง และทำไม "เราส่งแค่ vector" จึงไม่ใช่การรับประกันความเป็นส่วนตัว

## 2 · หลักฐาน ไม่ใช่คำกล่าวอ้าง: egress guard

`week30_multimodal_embeddings_edge/common/egress.py` ครอบโค้ดส่วนหนึ่งไว้ ให้ไปถึงได้แค่ **loopback** (เครื่องนี้เท่านั้น) และบันทึกทุกครั้งที่มีการพยายามเชื่อมต่อหรือ lookup ชื่อ host Lab 05-1 รัน pipeline ทั้งหมดอยู่ข้างใน: โหลด encoder ทุกตัว ทำ embedding โน้ตกับรูปหนึ่งรูป แล้วค้นหา จากนั้นจงใจลองทำให้ข้อมูลรั่วสองครั้ง

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/05_sovereign_rag/labs/lab05_1_egress_proof.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
▣ STEP 2 · load + embed + search inside the guard
✓ loaded google/embeddinggemma-2 · encoders audio+image+text · 744.4M params · mps · 2.7s
✓ pipeline finished · top hit: AHU condensate leak (0.808)
✓ connection attempts during the pipeline: 0 · blocked 0 · left this machine: 0

▣ STEP 3 · now try to leak — the guard must block both
✓ fetch the model card from huggingface.co: blocked — egress guard: blocked DNS lookup of huggingface.co
✓ POST a query vector to a cloud embedding API: blocked — egress guard: blocked DNS lookup of api.example-embeddings.com
│ BLOCKED  dns      huggingface.co:443  ×1
│ BLOCKED  dns      api.example-embeddings.com:443  ×1
```

ไม่มีการพยายามเชื่อมต่อเลยสักครั้ง: เมื่อ model ถูก cache ไว้แล้ว การโหลด การทำ embedding และการค้นหาไม่แตะ network เลย `embedkit` ตั้ง `HF_HUB_OFFLINE=1` ไว้ก็เพราะเหตุผลนี้

ต้องเข้าใจให้ชัดว่า guard นี้คืออะไร มันเป็น **สัญญาณเตือน (tripwire) ภายใน Python process เดียว** เหมาะกับการทดสอบและการ audit แต่ subprocess, โปรแกรมอื่น หรือ system call ดิบ ๆ ก็เลี่ยงผ่านมันไปได้ ถ้าจะบังคับ sovereignty จริง ต้องทำ **นอก** แอป ด้วย firewall ของ host, network policy (policy ของ OpenShell ใน Week 26 ก็เป็นทางหนึ่ง) หรือ air gap

✓ Checkpoint: คุณบอกได้ว่า guard พิสูจน์อะไรได้ และพิสูจน์อะไรไม่ได้

## 3 · ไฟล์เดียวเก็บความจำทั้งหมดของโรงแรม

`week30_multimodal_embeddings_edge/common/vectorstore.py` คือ vector store ขนาดจิ๋ว: ใช้ SQLite เก็บ record ใช้ numpy คำนวณ ไม่ต้องมี server แต่ละแถวเก็บสิ่งที่คนต้องใช้เพื่อเชื่อผลการค้นหา (ประเภท ชื่อ ข้อความ แหล่งที่มา) ไว้คู่กับ vector แบบ float16 ขนาด 256-d Lab 03 แสดงให้เห็นแล้วว่า 256-d รักษาคุณภาพไว้ได้ที่ขนาดแค่หนึ่งในสาม

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/05_sovereign_rag/labs/lab05_2_local_vector_store.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
▣ STEP 1 · embed and store everything
✓ 28 items · {'audio': 3, 'image': 6, 'note': 18, 'video': 1} · 4.0s · 32 KiB on disk

▣ STEP 2 · reopen the file, as tomorrow's process would, and search
  ✓ water leaking under an air handler           [all kinds  ] → note:AHU condensate leak 0.836 · image:Water pooling under AHU-3 0.728
  ✓ water leaking under an air handler           [image      ] → image:Water pooling under AHU-3 0.728 · image:Plant room B2, haze 0.673
  ✕ guest called about a burning smell from the  [all kinds  ] → note:Guest room too hot 0.789 · note:แอร์มีกลิ่นไหม้ 0.747
  ✕ guest called about a burning smell from the  [audio      ] → audio:Voicemail: bathroom flooding 0.682 · audio:Voicemail: towels for 812 0.680
  ✓ แขกโทรมาบอกว่าแอร์มีกลิ่นไหม้                [audio      ] → audio:Voicemail (Thai): AC burning 0.735 · audio:Voicemail: towels for 812 0.609
  ✓ haze on the plant room camera                [image+video] → video:CCTV: plant room B2, 10 s 0.718 · image:Plant room B2, haze 0.666
⚠ 4/6 found the expected item first
◆ the English 'burning smell' question lands on an English AC note before the Thai one, and misses the Thai voicemail even inside the audio filter — a filter removes the notes, it does not close the language + modality gap. Asked in Thai, the voicemail is found first.
✓ left this machine: 0 connections (attempts 0, blocked 0)
```

ความจำแบบ multimodal ของทั้งโรงแรม ทั้งโน้ต รูปถ่าย เสียง และวิดีโอ มีขนาดแค่ **ไฟล์ 32 KiB** คุณสำรองมันได้ audit ได้ คัดลอกไปยัง edge box เครื่องอื่นได้ และลบทิ้งได้ บรรทัด ✕ สองบรรทัดคือ modality gap จาก Lab 04 อีกครั้ง ตัวกรอง `kinds` ช่วยไม่ให้โน้ตเบียดสื่ออื่นตกอันดับ แต่ปิดช่องว่างแบบ **ภาษา + modality** ไม่ได้: คำถามภาษาอังกฤษยังหาข้อความเสียงภาษาไทยไม่เจอ แต่ถ้าถามเป็นภาษาไทย ก็เจอเป็นอันดับแรก เอเจนต์ใน Module 07 แก้ปัญหานี้ด้วยการค้นทุกประเภท แล้วให้ LLM อ่านผลลัพธ์เอง

✓ Checkpoint: คุณรู้ว่าไฟล์ index อยู่ที่ไหน และคัดลอกไปค้นบนเครื่องอื่นได้

## 4 · Private RAG: model ในเครื่องเขียนคำตอบ

RAG มีสองขั้น **Retrieve**: ทำ embedding คำถามด้วย prefix `QuestionAnswering` แล้วเอาโน้ต 3 อันดับแรก **Generate**: ส่งโน้ตเหล่านั้นให้ LLM แล้วสั่งให้ตอบ **จากโน้ตเท่านั้น** พร้อม citation ทุกข้อ model ทั้งสองตัวรันที่นี่ และ egress guard อนุญาตปลายทางเดียวเท่านั้น คือ Ollama ที่ `localhost:11434`

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/05_sovereign_rag/labs/lab05_3_private_rag.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้, gemma4:12b)

```
▣ STEP 1 · There's water on the floor under an air handler in the basement. What do I do first?
  [1] 0.817  AHU condensate leak
  [2] 0.670  Bathroom flooding
  [3] 0.664  Chiller plant efficiency alarm
  · ANSWER You should isolate the unit and put out wet-floor signs. Then, clear the drain before restarting the unit [1].
◆ 26 tokens in 27.9s on this machine (includes loading the model into memory)

▣ STEP 2 · แขกห้อง 1203 บอกว่าแอร์มีกลิ่นไหม้ ต้องทำอะไรบ้าง
  [1] 0.707  แอร์มีกลิ่นไหม้
  [2] 0.687  Smoke in the plant room
  [3] 0.668  Guest room too hot
  · ANSWER หากแขกแจ้งว่าแอร์มีกลิ่นไหม้ ให้ปิดเครื่องทันทีและย้ายแขกออกจากห้อง [1] จากนั้นต้องส่งช่างไฟฟ้ามาตรวจเช็คพัดลมและมอเตอร์ก่อนที่จะเปิดใช้งานอีกครั้ง [1]
◆ 54 tokens in 3.7s on this machine

▣ STEP 3 · A supplier emailed asking us to pay their invoice into a new bank account. Can I just update it?
  [1] 0.809  Supplier bank detail changes
  [2] 0.585  Bathroom flooding
  [3] 0.553  Guest room too hot
  · ANSWER No, you must not pay an invoice that asks to change the supplier's bank account. You must call the supplier using the number already on file and obtain approval from the finance director first [1].

▣ STEP 4 · what crossed the network
│ allowed  connect  ::1:11434  ×3
│ allowed  connect  127.0.0.1:11434  ×3
✓ left this machine: 0 connections
```

คำถามภาษาไทยได้คำตอบภาษาไทยจากโน้ตภาษาไทย คำตอบเรื่องเปลี่ยนบัญชีธนาคารปฏิเสธรูปแบบการโกงด้วยการยกขั้นตอนที่กำหนดไว้มาอ้าง connection ทั้งหมดไปที่ Ollama ของเครื่องนี้เท่านั้น คำตอบแรกใช้เวลา 27.9 วินาที เพราะ Ollama ต้องโหลด Gemma 4 เข้าหน่วยความจำ หลังจากนั้นแต่ละคำตอบใช้เวลา 3–4 วินาที นี่คือความต่างระหว่าง "โฮสต์อยู่ใน region ของเรา" กับ sovereign: คุณเป็นเจ้าของทุกขั้นตอน

✓ Checkpoint: ผลการรันของคุณแสดงคำตอบที่มี citation แบบ `[n]` และขึ้นว่า `left this machine: 0 connections`

## Labs — รันแล็บได้ที่นี่

**labs/lab05_1_egress_proof.py** — รัน embed pipeline เต็มรูปแบบภายใน guard ที่อนุญาตแค่ loopback บันทึกทุกการพยายามเชื่อมต่อ และบล็อกการรั่วไหลที่จงใจสร้างขึ้นสองครั้ง

**labs/lab05_2_local_vector_store.py** — เก็บโน้ต รูปถ่าย ข้อความเสียง และ CCTV ไว้ในไฟล์ SQLite ขนาด 32 KiB ไฟล์เดียว ที่ 256-d float16 แล้วเปิดไฟล์ใหม่และค้นหาแยกตามประเภท

**labs/lab05_3_private_rag.py** — retrieve ในเครื่อง แล้วให้ Gemma 4 ในเครื่องตอบเป็นภาษาอังกฤษและภาษาไทยพร้อม citation โดยมีหลักฐานจาก network ยืนยัน

## Try it yourself — ลองทำเอง

`exercises/ex05_egress_policy.py` ให้คุณเขียนกฎที่ guard บังคับใช้:

1. `allowed(host, port)`: อนุญาตแค่ loopback และเฉพาะพอร์ตของ Ollama
2. `leaked(log)`: connection ที่ได้รับอนุญาตแต่ไปยัง host ที่ไม่ใช่ loopback
3. ใน sovereign RAG ข้อมูลชิ้นไหนบ้างที่ต้องอยู่ในเครื่อง?
4. อธิบายด้วยคำพูดของคุณเอง: ทำไม guard ที่อยู่ใน process จึงเป็นแค่ tripwire ไม่ใช่ firewall?

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/05_sovereign_rag/exercises/ex05_egress_policy.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ เมื่อทำ TODO ครบทุกจุด)

```
✓ allowed(): loopback (127.0.0.0/8, ::1, localhost) on port 11434 only
✓ leaked(): only ALLOWED connections to non-loopback hosts count
✓ in sovereign RAG every artefact stays: audio, questions, vectors, prompts
✓ your note says what the in-process guard cannot see AND what enforces sovereignty for real
```

<details><summary>คำใบ้ — loopback</summary>

`ipaddress.ip_address("127.8.8.8").is_loopback` ได้ค่า `True`: ทั้งช่วง 127.0.0.0/8 เป็น loopback ไม่ใช่แค่ 127.0.0.1 ส่วนชื่อ `localhost` จะทำให้เกิด `ValueError` ตรงนั้น จึงต้องจัดการกรณีนี้ก่อน

</details>

✓ Checkpoint: ตัวตรวจขึ้น ✓ ครบทั้งสี่บรรทัด

## Troubleshooting — แก้ปัญหา

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| Lab 05-3 จบก่อนเวลาพร้อมข้อความ "Ollama is not running" | ไม่มี Ollama ที่ localhost:11434 | รัน `ollama serve` แล้วตามด้วย `ollama pull gemma4:12b` |
| คำตอบว่างเปล่า | Gemma 4 เป็น thinking model และใช้ token หมดไปกับการคิด | แล็บส่ง `think: false` และ 1,536 token ไปแล้ว ถ้าต้องการคำตอบยาวให้เพิ่ม `max_tokens` |
| เจอ `EgressBlocked: … huggingface.co` ตอนโหลด | model ยังไม่ถูก cache | รัน Lab 02-1 หนึ่งครั้งโดยต่อ network แล้วรันใหม่ |
| คำตอบแรกช้า | Ollama โหลด chat model เข้าหน่วยความจำในการเรียกครั้งแรก | เป็นเรื่องปกติ ครั้งต่อไปใช้เวลาแค่ไม่กี่วินาที |

## Next — บทถัดไป

[Lab 06 — Benchmark like an edge engineer](../06_edge_benchmarking/TUTORIAL.md): วัด p50 และ p95 ของทุก modality, ทำไม float16 จึงถูกห้าม และอะไรคือข้อจำกัดที่แท้จริงของ embedding model
