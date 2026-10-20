# ▶ Embed Lab 06 — benchmark แบบวิศวกร edge

> ส่วนหนึ่งของ Week 30 · multimodal embedding ด้วย EmbeddingGemma 2 บนฮาร์ดแวร์ในเครื่อง ฮาร์ดแวร์แบบ sovereign และอุปกรณ์ edge ทุกแล็บรันบนเครื่องของคุณเอง ทุกตัวเลขในคอร์สนี้วัดจากเครื่องที่รันจริง หรือไม่ก็อ้างอิงพร้อมแหล่งที่มา

**สิ่งที่คุณจะได้ลงมือทำจริง**
- วัด latency ระดับ p50 และ p95 ของข้อความ 1 ข้อความ, batch ข้อความ 32 ข้อความ, รูปถ่าย, ข้อความเสียง (voicemail) และคลิป CCTV ทั้งบน GPU และบน CPU
- ทดสอบคำเตือนเรื่อง float16 ใน model card ด้วยตัวเอง แล้วดูว่าผลทดสอบบอกอะไรได้ และบอกอะไรไม่ได้
- คำนวณ **bandwidth floor** ของอุปกรณ์ทุกขั้นในบันไดอุปกรณ์ แล้วค้นพบว่ามันไม่ใช่ข้อจำกัดที่แท้จริงของ embedding model
- วางแผนการสร้าง index จากตัวเลขที่วัดได้จริง และติดป้ายทุกค่าที่เป็นการประมาณว่าเป็นการประมาณ

**Time** ~45 นาที · **Difficulty** ระดับสูง · **Hardware** แล็ปท็อปเครื่องไหนก็ได้ · RAM ว่าง ~5 GB · มี GPU (MPS หรือ CUDA) ก็ดี แต่ไม่บังคับ

**แหล่งอ้างอิง:** [model card — Numerical Precision](https://huggingface.co/google/embeddinggemma-2) · Tina Huang, [*Every Size Local AI In 24 Minutes*](https://youtu.be/rPGJhrunbxo) (ความจุเทียบกับ bandwidth) · สเปก bandwidth จากผู้ผลิตใน [`week30_multimodal_embeddings_edge/RESEARCH.md`](../RESEARCH.md) §4

## 0 · ก่อนเริ่ม

benchmark จะโกหกเมื่อเครื่องกำลังยุ่ง ปิดแอปหนัก ๆ เสียบสายชาร์จแล็ปท็อป แล้วรันแต่ละแล็บสองรอบ เก็บผลรอบที่สองไว้ ทุกตัวเลขในบทนี้เป็น **LAPTOP STAND-IN** คือบอกถึงเครื่องนี้ ไม่ใช่ Raspberry Pi หรือโทรศัพท์

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python -c "import torch; print('MPS', torch.backends.mps.is_available(), '· CUDA', torch.cuda.is_available())"
```

✓ Checkpoint: คุณรู้แล้วว่าแล็บของคุณจะวัด GPU, CPU หรือทั้งสองอย่าง

## 1 · latency แยกตามประเภทข้อมูล: p50 และ p95

ให้รายงานเป็น **percentile** เสมอ อย่าใช้ผลจากรอบเดียวที่บังเอิญเร็ว p50 คือกรณีปกติ ส่วน p95 คือเวลาที่ผู้ใช้ 1 ใน 20 คนต้องรอ Lab 06-1 จะ warm up โมเดลก่อน (การเรียกครั้งแรกต้อง compile kernel และเติม cache) จากนั้นจึงจับเวลาแต่ละงานซ้ำหลายรอบ

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/06_edge_benchmarking/labs/lab06_1_latency_by_modality.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ — Apple M4 Max, LAPTOP STAND-IN)

```
▣ STEP 1 · mps · bfloat16
│ work            runs  p50      p95      per item
│ ──────────────  ────  ───────  ───────  ─────────  ──────────────────
│ text ×1         20    22 ms    24 ms    22.2 ms    ░░░░░░░░░░░░░░░░░░
│ text ×32        8     135 ms   152 ms   4.2 ms     █░░░░░░░░░░░░░░░░░
│ photo           8     259 ms   264 ms   258.9 ms   ██░░░░░░░░░░░░░░░░
│ voicemail 12 s  6     105 ms   107 ms   105.3 ms   █░░░░░░░░░░░░░░░░░
│ CCTV 10 s       4     1247 ms  1251 ms  1247.3 ms  ███████░░░░░░░░░░░

▣ STEP 3 · GPU vs CPU
│ work            MPS p50  CPU p50  CPU / MPS
│ ──────────────  ───────  ───────  ─────────
│ text ×1         22 ms    34 ms    1.5×
│ text ×32        135 ms   495 ms   3.7×
│ photo           259 ms   920 ms   3.6×
│ voicemail 12 s  105 ms   1091 ms  10.4×
│ CCTV 10 s       1247 ms  4191 ms  3.4×
```

อ่านผลแบบวิศวกร edge:

- **การทำ batch** คือคันโยกที่ได้ผลมากที่สุด ข้อความเดียวใช้ 22 ms แต่ถ้าอยู่ใน batch ขนาด 32 จะเหลือ 4.2 ms ต่อข้อความ
- **สื่อกินโทเค็น** รูปถ่ายหนึ่งรูปคือ 280 โทเค็น วิดีโอ 10 วินาทีคือ 1,400 โทเค็น คลิป CCTV จึงช้าที่สุด
- **CPU ไม่ได้แย่ขนาดนั้น** 34 ms ต่อคำค้นหาหนึ่งครั้งถือว่าพอสำหรับงานค้นหา งานที่ GPU หรือ NPU คุ้มค่าจริง ๆ คือการทำ index รูปภาพและคลิป
- **p95 สำคัญบนฮาร์ดแวร์ที่ใช้ร่วมกัน** p95 ของข้อความบน CPU อยู่ที่ 152 ms ขณะที่ p50 อยู่ที่ 34 ms เพราะมีงานอื่นบนเครื่องมาแทรก

✓ Checkpoint: คุณบอกค่า p50 และ p95 ของคำค้นหาข้อความหนึ่งครั้งบนเครื่องของคุณได้ และบอกต้นทุนต่อชิ้นเมื่ออยู่ใน batch ขนาด 32 ได้

## 2 · ความแม่นยำเชิงตัวเลข: ทำไม model card ห้ามใช้ float16

ลดจำนวนไบต์ต่อ weight ลงครึ่งหนึ่ง ปริมาณข้อมูลที่วิ่งผ่านหน่วยความจำก็ลดลงครึ่งหนึ่งด้วย **bfloat16** คงช่วง exponent ไว้เท่า float32 แต่ใช้ขนาดแค่ครึ่งเดียว **float16** ละเอียดกว่าแต่มีช่วงค่าแคบกว่ามาก model card เตือนว่า activation ของ EmbeddingGemma 2 เกินช่วงของ float16 ผลคือมัน "returns NaN or silently degraded embeddings rather than raising an error" (คืนค่า NaN หรือ embedding ที่คุณภาพตกลงแบบเงียบ ๆ แทนที่จะแจ้ง error)

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/06_edge_benchmarking/labs/lab06_2_precision.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้, CPU)

```
▣ STEP 3 · float16
✓ loaded google/embeddinggemma-2 · encoders text · 271.0M params · cpu · 1.6s
◆ 517 MB of weights · 36 short texts in 1.90s, NaN 0/36, recall@1 1.000 · long document in 63.7s, NaN: YES

▣ STEP 4 · agreement with float32
│ dtype     weights  short NaN  recall@1  long doc  short cos vs fp32  long cos vs fp32
│ ────────  ───────  ─────────  ────────  ────────  ─────────────────  ────────────────
│ float32   1034 MB  0/36       1.000     ok        1.0000             1.0000
│ bfloat16  517 MB   0/36       1.000     ok        1.0000             1.0000
│ float16   517 MB   0/36       1.000     NaN       1.0000             NaN
⚠ float16 returned a NaN vector — no exception, no warning. Exactly the failure the model card describes; a test set of short texts would never have shown it.
```

บนข้อความสั้น 36 ข้อความ ทั้งสามรูปแบบผ่านหมด แต่เอกสารยาว (3,502 token) คือจุดที่ float16 พัง: มันคืน vector ที่เป็น NaN โดยไม่มี exception และไม่มีคำเตือนใด ๆ ตรงกับที่ model card บอกไว้ทุกประการ การค้นหาใดที่แตะ vector นี้จะได้ผลขยะแบบเงียบ ๆ ชุดทดสอบที่มีแต่ข้อความสั้นจะไม่มีทางจับปัญหานี้ได้ จึงต้องทำตามกฎของผู้ผลิต (ใช้ bfloat16 ถ้าฮาร์ดแวร์รองรับ นอกนั้นใช้ float32) และใส่ input ยาว ๆ ไว้ใน regression test เสมอ นอกจากนี้ half-precision ยัง**ช้ากว่ามาก**บน CPU เครื่องนี้ (67 วินาที เทียบกับ 2.0 วินาทีสำหรับเอกสารยาว) เพราะ CPU จำนวนมากไม่มีชุดคำสั่ง bf16/fp16 ที่เร็ว ความแม่นยำเชิงตัวเลขจึงเป็นคันโยกของ GPU/NPU ไม่ใช่ของฟรีบน CPU

✓ Checkpoint: คุณอธิบายได้ว่าทำไม "มันใช้ได้กับชุดทดสอบของฉัน" ยังไม่พอจะลบล้างคำเตือนใน model card

## 3 · อะไรคือข้อจำกัดของ embedding model? ไม่ใช่สายพาน

สายพานลำเลียงจาก Module 01 บอกว่า forward pass หนึ่งรอบต้องอ่าน weight ทั้งหมดอย่างน้อยหนึ่งครั้ง จึงไม่มีทางเร็วกว่า **ขนาด weight เป็นไบต์ ÷ bandwidth** ค่านี้คือ **bandwidth floor** สำหรับ LLM ที่เขียนทีละโทเค็น ค่า floor นี้ใกล้เคียงความจริงมาก Lab 06-3 จะตรวจดูว่าสำหรับ embedding model เป็นอย่างไร

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/06_edge_benchmarking/labs/lab06_3_tier_projection.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้; อุปกรณ์อื่น = ESTIMATE)

```
▣ STEP 1 · bandwidth floor per forward pass (ms) — weights cross the bus at least once
│ device                                 peak bandwidth  text Q4 GGUF  text Q8_0 GGUF  text bf16  full bf16
│ ─────────────────────────────────────  ──────────────  ────────────  ──────────────  ─────────  ─────────
│ Raspberry Pi 5 (CPU only)              17 GB/s         10.1          17.8            31.0       85.0
│ DGX Spark (GB10)                       273 GB/s        0.6           1.1             1.9        5.3
│ AMD Ryzen AI Max+ 395                  256 GB/s        0.7           1.2             2.1        5.6
│ Apple M4 Max 14-core (this Mac, peak)  410 GB/s        0.4           0.7             1.3        3.5
│ RTX 4090                               1008 GB/s       0.2           0.3             0.5        1.4
│ H100 SXM                               3350 GB/s       0.1           0.1             0.2        0.4
◆ ESTIMATE — a lower bound from vendor peak bandwidth, never a measurement of that device

▣ STEP 2 · this machine: measured time vs the bandwidth floor vs the work
│ work (MPS bf16)  tokens  measured p50  bandwidth floor  measured ÷ floor  work         achieved
│ ───────────────  ──────  ────────────  ───────────────  ────────────────  ───────────  ────────────
│ text ×1          14      22.2 ms       1.3 ms           17×               3.6 GFLOP    0.16 TFLOP/s
│ text ×32         1381    135.5 ms      1.3 ms           105×              359.1 GFLOP  2.65 TFLOP/s
```

เวลาที่วัดได้สูงกว่า floor 17× ถึง 105× แปลว่าสายพาน**ไม่ใช่**ตัวจำกัดของ embedding model encoder อ่านโทเค็น input ทั้งหมดใน**รอบเดียว** weight วิ่งผ่าน bus แค่ครั้งเดียว แต่ปริมาณการคำนวณเพิ่มตามจำนวนโทเค็น ที่ batch 1 ต้นทุนคงที่เป็นตัวหลัก (การสั่ง kernel การตัดโทเค็น) ส่วนที่ batch 32 การคำนวณเป็นตัวหลัก (359 GFLOP) ดังนั้นเวลาเลือกขนาดอุปกรณ์ edge สำหรับงาน embedding ให้ดูที่ **กำลังประมวลผล** (CPU, GPU, NPU) และมอง bandwidth เป็นแค่ค่าต่ำสุดเท่านั้น บทเรียนเรื่องความจุเทียบกับ bandwidth จากวิดีโอยังใช้ได้กับ chat model ที่ทำงานอยู่ข้าง ๆ

✓ Checkpoint: คุณอธิบายได้ว่าทำไม Raspberry Pi เครื่องเดียวกันถึงติดข้อจำกัดที่ bandwidth เมื่อรัน chat model แต่ติดที่กำลังประมวลผลเมื่อรัน embedding model

## 4 · วางแผนสร้าง index จากตัวเลขที่วัดได้จริง

Step 3 ของ Lab 06-3 วางแผนงานที่โรงแรมต้องทำจริง คือ embed บันทึก 50,000 รายการให้เสร็จภายในคืนเดียว

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
▣ STEP 3 · index plan for 50,000 hotel notes — from MEASURED per-item times on this machine
│ where                                             per note  50,000 notes
│ ────────────────────────────────────────────────  ────────  ──────────────────────
│ this machine · mps (measured)                     4.2 ms    3.5 min
│ this machine · cpu (measured)                     15.5 ms   12.9 min
│ Raspberry Pi 5 · bandwidth floor only (ESTIMATE)  —         ≥ 28 s, real: far more
```

แม้บน CPU บันทึกของทั้งโรงแรมก็ embed เสร็จในไม่กี่นาที บนอุปกรณ์ที่เล็กกว่าอาจใช้เวลาเป็นชั่วโมง ซึ่งยังเป็นงาน batch กลางคืนได้สบาย คือ embed ใหม่เฉพาะส่วนที่เปลี่ยน แล้วค้นหาได้ทั้งวัน ก่อนจะรับปากตัวเลขสำหรับ Pi โทรศัพท์ หรือ Jetson ให้รัน Lab 06-1 **บนอุปกรณ์นั้น**ก่อน แล็บนี้พิมพ์เฉพาะสิ่งที่วัดได้บนเครื่องนี้ และติดป้ายทุกอย่างที่เหลือว่าเป็นการประมาณ

✓ Checkpoint: คุณมีแผนที่ใช้เวลาต่อชิ้นจากการวัดจริง และบอกได้ว่าตัวเลขไหนในแผนเป็นการประมาณ

## Labs — รันแล็บได้ที่นี่

**labs/lab06_1_latency_by_modality.py** — วัด p50/p95 ของข้อความ, batch, รูปถ่าย, voicemail และ CCTV บน GPU (bfloat16) และ CPU (float32)

**labs/lab06_2_precision.py** — เทียบ float32, bfloat16 และ float16 ทั้งด้านหน่วยความจำ ความเร็ว NaN และความสอดคล้องของผลลัพธ์ พร้อมทดสอบคำเตือนใน model card อย่างตรงไปตรงมา

**labs/lab06_3_tier_projection.py** — bandwidth floor ของอุปกรณ์ทุกขั้น เทียบกับเวลาที่วัดได้และจำนวน FLOP พร้อมแผนสร้าง index จากตัวเลขที่วัดจริง

## Try it yourself — ลองทำเอง

`exercises/ex06_benchmark_math.py` มี TODO สี่จุด ซึ่งก็คือคณิตศาสตร์เบื้องหลังตาราง benchmark ทุกตาราง

1. `percentile(values, p)` แบบ linear interpolation
2. `throughput(items, ms)`
3. `bandwidth_floor_ms(weights_mb, gbps)`
4. `limited_by(measured_ms, floor_ms)`: คืน "bandwidth" ถ้าอยู่ภายใน 2× ของ floor ไม่เช่นนั้นคืน "compute or overhead"

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/06_edge_benchmarking/exercises/ex06_benchmark_math.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ เมื่อทำ TODO ครบทุกจุด)

```
✓ percentile: p50 22.3 ms · p95 48.75 ms (linear interpolation) · one value → itself
✓ throughput: 32 items in 135.5 ms = 236.2 items/s · 1 in 22.2 ms = 45.0/s
✓ bandwidth_floor_ms: 540 MB at 410 GB/s = 1.29 ms · 310 MB at 17 GB/s = 17.8 ms
✓ limited_by: far above the floor → compute or overhead · within 2× of it → bandwidth
```

<details><summary>คำใบ้ — percentile()</summary>

เรียงค่าจากน้อยไปมาก ตำแหน่งคือ `i = (n − 1) · p / 100` หยิบค่าที่ตำแหน่ง `floor(i)` และ `ceil(i)` แล้ว interpolate ด้วยส่วนทศนิยมของ `i` นี่คือวิธีมาตรฐานของ numpy

</details>

✓ Checkpoint: ตัวตรวจขึ้น ✓ ครบทั้งสี่บรรทัด

## Troubleshooting — แก้ปัญหา

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| p95 สูงกว่า p50 มาก | มีงานอื่นรันอยู่บนเครื่อง หรือเครื่องร้อนจนลดความเร็ว (thermal throttling) | ปิดแอป เสียบสายชาร์จ รันสองรอบแล้วเก็บผลรอบที่สอง |
| Lab 06-3 แจ้งว่า "run Lab 06-1 first" | แล็บนี้อ่านไฟล์ `.runs/latency.json` | รัน Lab 06-1 ก่อน |
| bf16 บน CPU ช้ากว่า fp32 | CPU เครื่องนี้ไม่มีชุดคำสั่ง bf16 ที่เร็ว | เป็นเรื่องปกติ — ใช้ float32 บน CPU แบบนี้ (ตาม model card) |
| MPS หน่วยความจำไม่พอ | มีแอปอื่นใช้ GPU อยู่ หรือรัน encoder ทุกตัวเป็น fp32 | ใช้ bfloat16 บน MPS (แล็บทำแบบนี้อยู่แล้ว) |

## Next — บทถัดไป

[Lab 07 — Capstone: Alto Edge Memory](../07_capstone_alto_edge_memory/TUTORIAL.md): index แบบออฟไลน์ชุดเดียวที่รวมทุกอย่าง เอเจนต์ที่ค้นหาใน index นั้น และการให้คะแนนที่อิงหลักฐาน
