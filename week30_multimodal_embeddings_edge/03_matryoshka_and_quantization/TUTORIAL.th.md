# ▶ Embed Lab 03 — vector เล็กลง model เล็กลง: Matryoshka, int8 และ GGUF

> ส่วนหนึ่งของ Week 30 · Multimodal embedding ด้วย EmbeddingGemma 2 บนฮาร์ดแวร์ local, sovereign และ edge ทุกแล็บรันบนเครื่องของคุณเอง ทุกตัวเลขในคอร์สนี้วัดจากเครื่องที่รันจริง หรือไม่ก็อ้างอิงแหล่งที่มาไว้

**สิ่งที่คุณจะได้ลงมือทำ**
- ตัด vector 768 ตัวเลขให้เหลือ 512, 256 หรือ 128 ตัวเลข แล้ววัดว่าเสียอะไรไปบ้าง
- ทำซ้ำกับดักเงียบ ๆ ที่ model card เตือนไว้: vector ที่ถูกตัดแล้วจะไม่ยาว 1 อีกต่อไป
- เก็บแต่ละตัวเลขด้วย 1 byte (int8) หรือ 1 bit (binary) แล้ววัด recall ในทุกขนาด
- ย่อตัว **model** เอง: เส้นทาง GGUF สำหรับ llama.cpp และ Ollama และ model int8 บน PyTorch ที่วัดผลได้วันนี้

**Time** ~45 นาที · **Difficulty** ระดับกลาง · **Hardware** แล็ปท็อปเครื่องไหนก็ได้ · model จาก Lab 02

**แหล่งอ้างอิง:** [model card — Matryoshka, ตารางการตัด](https://huggingface.co/google/embeddinggemma-2) · [ggml-org GGUF](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF) · [คู่มือรันบนเครื่องของ Unsloth](https://unsloth.ai/docs/models/embeddinggemma-2) · [quantization ใน sentence-transformers](https://sbert.net) · [`week30_multimodal_embeddings_edge/RESEARCH.md`](../RESEARCH.md) §1

## 0 · ก่อนเริ่ม

คุณต้องมี model ที่แคชไว้จาก Lab 02 ส่วนหนึ่งของโมดูลนี้ไม่บังคับ: Ollama บน `localhost:11434` (`ollama --version`) Lab 03-3 จะถาม Ollama ว่ารัน GGUF แบบ text-only ได้หรือไม่

```bash
# on: laptop
ollama --version
```

✓ Checkpoint: คุณรู้ว่าเครื่องคุณมี Ollama เวอร์ชันไหน (Lab 03-3 จะบอกว่าใหม่พอหรือยัง)

## 1 · Matryoshka: ตัวเลขที่สำคัญมาก่อน

EmbeddingGemma 2 ถูกเทรนด้วย **Matryoshka Representation Learning** เหมือนตุ๊กตาแม่ลูกดก ตัวเลข 512, 256 หรือ 128 ตัวแรกของ vector ก็เป็น vector ที่เล็กกว่าและใช้งานได้ในตัวเอง แค่ตัดส่วนท้ายทิ้งแล้วใช้ต่อได้เลย model card วัดต้นทุนนี้ไว้บน benchmark ขนาดใหญ่:

| dims | พื้นที่เก็บ | MTEB multilingual (model card) |
|---|---|---|
| 768 | 1× | 61.36 |
| 512 | เล็กลง 1.5× | 61.17 |
| 256 | เล็กลง 3× | 60.41 |
| 128 | เล็กลง 6× | 57.89 |

คุณภาพแทบไม่ต่างกันจนถึง 256 พอถึง 128 คุณภาพจะลดลง และ model card เตือนว่าคุณภาพแบบ multimodal ลดลงมากที่สุด

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/03_matryoshka_and_quantization/labs/lab03_1_matryoshka.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
▣ STEP 1 · quality and size at each Matryoshka dimension (cut + re-normalise)
│ dims  recall@1  MRR    bytes/vector (fp32)  1M vectors  smaller  MRR
│ ────  ────────  ─────  ───────────────────  ──────────  ───────  ────────────────
│ 768   1.000     1.000  3072 B               2.86 GiB    1.0×     ████████████████
│ 512   1.000     1.000  2048 B               1.91 GiB    1.5×     ████████████████
│ 256   1.000     1.000  1024 B               0.95 GiB    3.0×     ████████████████
│ 128   1.000     1.000  512 B                0.48 GiB    6.0×     ████████████████
◆ 18 easy questions saturate at 1.000 — the model card's large benchmark shows the real slope:
◆ REFERENCE (model card, MTEB multilingual v2): 768d 61.36 · 512d 61.17 · 256d 60.41 · 128d 57.89
```

กับคำถามง่าย ๆ 18 ข้อ ทุกขนาดได้คะแนนเต็ม ชุดทดสอบเล็กจึงมองไม่เห็นความชัน นี่คือบทเรียนในตัวมันเอง: **ทดสอบด้วยข้อมูลที่ยากพอ ๆ กับ query จริงของคุณ** ก่อนจะเลือก 128

✓ Checkpoint: คุณบอกได้ว่า vector 1 ล้านตัวใช้พื้นที่เท่าไรที่ 768 และ 256 มิติ (2.86 GiB เทียบกับ 0.95 GiB แบบ float32)

## 2 · กับดัก: ต้อง re-normalise หลังตัด

unit vector ที่ถูกตัดครึ่งไม่ใช่ unit vector อีกต่อไป ความยาวของมันหดลง และ dot product ทุกค่าก็หดตาม โดยหดไม่เท่ากันในแต่ละ vector sentence-transformers จัดการเรื่องนี้ให้: `encode(…, truncate_dim=128, normalize_embeddings=True)` จะตัดแล้ว re-normalise ให้ แต่ถ้าเขียน `v[:128]` เองจะไม่ได้ทำ

**Expected output** (บันทึกจาก Mac เครื่องนี้ — Lab 03-1 ขั้นที่ 2–3)

```
▣ STEP 2 · the trap — a hand-cut vector is shorter than 1
│ vector                                               L2 norm
│ ───────────────────────────────────────────────────  ───────
│ full 768-d vector                                    1.0021
│ encode(truncate_dim=128, normalize_embeddings=True)  1.0013
│ manual slice [:128]                                  0.5985
│ slice + ek.renorm()                                  1.0000
◆ the 18 document vectors cut to 128-d have lengths 0.566 … 0.610 — not 1

▣ STEP 3 · does skipping re-normalisation change the ranking? (dot product on cut vectors)
◆ 256-d: 4 of 54 top-3 slots differ · scores off by up to 0.477
◆ 128-d: 27 of 54 top-3 slots differ · scores off by up to 0.626
```

ที่ 128 มิติ ผลลัพธ์ top-3 ครึ่งหนึ่งสลับลำดับ โดยไม่มี error ใด ๆ โผล่มา model card เรียกปรากฏการณ์นี้ตรง ๆ ว่ามัน "degrades ranking quality silently" จึงได้กฎสองข้อ ข้อแรก re-normalise ทุก vector ที่ตัด ข้อสอง ให้ query และ document อยู่ในมิติเดียวกัน เพราะ query 128-d ให้คะแนนกับ index 768-d ไม่ได้เลย

✓ Checkpoint: คุณอธิบายได้ว่าทำไม index แบบ dot product ถึงจัดอันดับผิด โดยไม่มี error ใด ๆ ถ้าข้ามการ re-normalise

## 3 · int8 และ binary: ใช้ bit น้อยลงต่อตัวเลข

Matryoshka ทำให้ vector สั้นลง ส่วน **quantization** ทำให้แต่ละตัวเลขเล็กลง int8 ใช้ 1 byte แทน 4 byte ส่วน binary เก็บแค่เครื่องหมาย คือ 1 bit vector แบบ binary เปรียบเทียบกันด้วย **Hamming distance** คือจำนวน bit ที่ต่างกัน ซึ่งเร็วมากบน CPU ทุกตัว

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/03_matryoshka_and_quantization/labs/lab03_2_int8_and_binary.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
│ dims  precision  bytes/vector  smaller than 768 fp32  recall@1
│ ────  ─────────  ────────────  ─────────────────────  ────────  ────────────────
│ 768   float32    3072 B        1×                     1.000     ████████████████
│ 768   int8       768 B         4×                     1.000     ████████████████
│ 768   binary     96 B          32×                    1.000     ████████████████
│ 512   float32    2048 B        2×                     1.000     ████████████████
│ 512   int8       512 B         6×                     1.000     ████████████████
│ 512   binary     64 B          48×                    1.000     ████████████████
│ 256   float32    1024 B        3×                     1.000     ████████████████
│ 256   int8       256 B         12×                    1.000     ████████████████
│ 256   binary     32 B          96×                    1.000     ████████████████
│ 128   float32    512 B         6×                     1.000     ████████████████
│ 128   int8       128 B         24×                    1.000     ████████████████
│ 128   binary     16 B          192×                   0.667     ███████████░░░░░

▣ STEP 2 · what that means for an edge box
◆ 1,000,000 notes/photos/clips at 768-d float32: 2,930 MiB
◆ 1,000,000 notes/photos/clips at 256-d int8: 244 MiB
◆ 1,000,000 notes/photos/clips at 128-d binary: 15 MiB
```

มีแค่ชุดที่สุดโต่งที่สุดที่พัง: 128 มิติ × 1 bit = ลายนิ้วมือขนาด 16 byte ซึ่งตอบผิดไปหนึ่งในสามของคำถาม ส่วน 256-d int8 ยังได้ recall เต็ม ทั้งที่เล็กลง 12× ข้อมูลล้านรายการในพื้นที่ 244 MiB ใส่ใน Raspberry Pi ได้สบาย ๆ รูปแบบที่ใช้ใน production คือ **rescoring**: ค้นใน binary index ขนาดจิ๋วก่อน แล้วค่อยให้คะแนนใหม่กับ 50 อันดับแรกด้วย vector เต็ม

✓ Checkpoint: คุณเลือกรูปแบบ index สำหรับ gateway ที่มีพื้นที่ว่าง 512 MB และข้อมูล 1 ล้านรายการได้ พร้อมให้เหตุผลประกอบ

## 4 · model เล็กลง: GGUF และ weight แบบ int8

vector เป็นแค่ครึ่งเดียวของเรื่อง ตัว **model** ก็ย่อได้เหมือนกัน model ข้อความถูกแจกจ่ายเป็น **GGUF** ซึ่งเป็นฟอร์แมตของ llama.cpp, Ollama และ LM Studio:

| ไฟล์ | ขนาด | แหล่งที่มา |
|---|---|---|
| `embeddinggemma-2-UD-Q4_K_XL.gguf` | 176 MB | Unsloth |
| `embeddinggemma-2-GGUF:Q8_0` | 310 MB | ggml-org |
| `embeddinggemma-2-GGUF:BF16` | 558 MB | ggml-org |

**GGUF เหล่านี้เป็นแบบ text-only** ไม่มี encoder สำหรับภาพหรือเสียง ถ้าจะใช้กับรูปถ่ายและเสียงพูด ต้องใช้ model เต็ม (โมดูล 04–07) นอกจากนี้ สถาปัตยกรรมใหม่เอี่ยมยังต้องการ runtime ที่รู้จักมันด้วย Lab 03-3 จะถาม runtime ของคุณก่อน

```bash
# on: laptop
ollama pull hf.co/ggml-org/embeddinggemma-2-GGUF:Q8_0
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/03_matryoshka_and_quantization/labs/lab03_3_quantized_weights.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ ด้วย Ollama 0.34.4 — เก่าเกินไปสำหรับสถาปัตยกรรมใหม่)

```
▣ STEP 1 · can your GGUF runtimes load gemma-embedding2? (text-only GGUF)
⚠ Ollama has the GGUF but cannot run it: unknown model architecture 'gemma-embedding2' — upgrade Ollama (support arrived after 0.34.x)
◆ Ollama here: ollama version is 0.34.4
◆ llama.cpp here: version: 0.5.0 (build 11146, commit 7fe450e19)
◆ REFERENCE (Unsloth guide): llama-server --model embeddinggemma-2-UD-Q4_K_XL.gguf --embeddings --pooling mean

▣ STEP 2 · int8 dynamic quantization of the text model — PyTorch, CPU
◆ quantized engine: qnnpack (available: qnnpack)
│ weights               serialized size
│ ────────────────────  ───────────────
│ float32               1034 MB
│ int8 (Linear layers)  644 MB

▣ STEP 3 · speed and agreement on the hotel knowledge base
│ weights  36 texts  per text  recall@1
│ ───────  ────────  ────────  ────────
│ float32  0.44s     12 ms     1.000
│ int8     0.87s     24 ms     1.000
✓ int8 vs float32 document vectors: mean cosine 0.9700 (min 0.9617)
```

นี่คือหน้าตาของงาน edge ในสัปดาห์ที่ model เพิ่งเปิดตัว GGUF ดาวน์โหลดมาได้ แต่ Ollama 0.34.4 และ llama.cpp 0.5.0 บน Mac เครื่องนี้ยังไม่รู้จัก `gemma-embedding2` release notes ของ Ollama 0.40.0 พูดถึง EmbeddingGemma 2 ไว้ อัปเกรด รันใหม่ แล้วขั้นนี้จะกลายเป็น ✓

model int8 ของจริงและรันได้วันนี้ เล็กลง 38 % ยังได้ recall เต็ม และ vector ที่ได้ตรงกับ float32 ที่ cosine 0.97 แต่บน CPU ARM ตัวนี้มัน **ช้ากว่า** เพราะ kernel int8 ของ `qnnpack` สู้การคำนวณ float32 ที่เร็วมากของ Apple ไม่ได้ ให้วัดความเร็วบนอุปกรณ์ปลายทางจริง อย่าเหมาเอาว่า quantized แล้วจะเร็วเสมอ

✓ Checkpoint: คุณบอกได้สองเหตุผลที่ GGUF อาจรันบนเครื่องหนึ่งไม่ได้ (runtime เก่าเกินกว่าจะรู้จักสถาปัตยกรรม หรือ GGUF ไม่มี modality ที่คุณต้องการ)

## Labs — รันแล็บได้ที่นี่

**labs/lab03_1_matryoshka.py** — วัด recall, MRR และพื้นที่เก็บที่ 768/512/256/128 มิติ พร้อมทำซ้ำกับดักเรื่อง re-normalise

**labs/lab03_2_int8_and_binary.py** — ทุกชุดผสมของมิติและ precision ตั้งแต่ vector float ขนาด 3,072 byte ลงไปจนถึงลายนิ้วมือขนาด 16 byte

**labs/lab03_3_quantized_weights.py** — ตรวจว่า Ollama และ llama.cpp ของคุณรัน GGUF แบบ text-only ได้หรือไม่ จากนั้นวัด weight int8 บน PyTorch ทั้งขนาด ความเร็ว และความตรงกันของผลลัพธ์

## Try it yourself — ลองทำเอง

`exercises/ex03_truncate_renorm.py` สร้างชั้นจัดเก็บข้อมูลของ vector index บน edge:

1. `truncate(v, dim)`: ตัดแล้ว re-normalise และปฏิเสธมิติที่ model ไม่ได้ถูกเทรนมา
2. `storage_bytes(n, dim, precision)`
3. `pack_bits(v)` และ `hamming(a, b)`: ทำ binary quantization ด้วยมือ

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/03_matryoshka_and_quantization/exercises/ex03_truncate_renorm.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ เมื่อทำ TODO ครบทุกจุด)

```
✓ truncate(v, 128): the first 128 numbers, re-normalised to length 1
✓ truncate(v, 300) raises ValueError — only 768/512/256/128 were trained
✓ storage_bytes: 1M × 768 fp32 = 3.07 GB · 256 int8 = 256 MB · 128 binary = 16 MB
✓ pack_bits: positive → 1, else 0, 8 per byte · hamming counts differing bits
```

<details><summary>คำใบ้ — hamming()</summary>

`np.bitwise_xor(a, b)` ให้ bit ที่เป็น 1 ตรงตำแหน่งที่ a กับ b ต่างกันพอดี แล้ว `np.unpackbits(...).sum()` จะนับจำนวน bit เหล่านั้น

</details>

✓ Checkpoint: ตัวตรวจขึ้น ✓ ครบทั้งสี่บรรทัด

## Troubleshooting — แก้ปัญหา

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `unknown model architecture: 'gemma-embedding2'` | Ollama หรือ llama.cpp เก่ากว่า model | อัปเกรด (Ollama 0.40.0 ระบุว่ารองรับ EmbeddingGemma 2) หรือใช้เส้นทาง PyTorch ต่อไป |
| `Didn't find engine for operation quantized::linear_prepack NoQEngine` | ยังไม่ได้เลือก engine สำหรับ int8 | แล็บจะเลือก `qnnpack` บน ARM ส่วนบน x86 จะใช้ `x86`/`fbgemm` |
| `DeprecationWarning: … sentence_transformers.quantization` | path การ import แบบเก่า | ให้ import จาก `sentence_transformers.util.quantization` |

## Next — บทถัดไป

[Lab 04 — พื้นที่เดียวสำหรับข้อความ รูปถ่าย เสียง และวิดีโอ](../04_multimodal_search/TUTORIAL.md): โหลด encoder ทุกตัว แล้วให้ประโยคเดียวค้นหารูปถ่าย ข้อความเสียง และคลิปจากกล้อง CCTV
