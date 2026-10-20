# ▶ Embed Lab 08 — ใช้ Mac ของคุณเป็น edge box: MLX, ONNX และ embedding server ที่เปิดไว้ตลอดเวลา

> ส่วนหนึ่งของ Week 30 · multimodal embedding ด้วย EmbeddingGemma 2 บนฮาร์ดแวร์ในเครื่อง แบบ sovereign และบน edge ทุกแล็บรันบนเครื่องของคุณเอง ตัวเลขทุกตัวในคอร์สนี้วัดจากเครื่องที่รันจริง หรือไม่ก็ระบุแหล่งที่มาไว้

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- มอง Mac ของคุณเป็นอุปกรณ์ edge ตัวจริง ไม่ใช่แค่ตัวแทนชั่วคราว ในการแบ่งระดับของ Tina Huang นั้น MacBook คือ "personal computer" ส่วน Mac mini หรือ Mac Studio คือ "home server"
- รัน EmbeddingGemma 2 บน Mac เครื่องนี้ห้าแบบ: PyTorch บน GPU ของ Apple, **MLX** (ML framework ของ Apple เอง) แบบ bf16 และ 4-bit และ **ONNX Runtime** แบบ q4 และ fp16 แล้ววัดผลเทียบกันแบบเคียงข้าง
- ลองเส้นทาง **Neural Engine** ของ Apple ผ่าน CoreML และดูว่าทำไมวันนี้ยังรัน model นี้ไม่ได้
- รัน multimodal model เต็มตัวใน MLX: รูปถ่ายและข้อความเสียง เทียบ bf16 กับ 4-bit พร้อมตรวจความแม่นยำแยกตาม modality
- เปลี่ยน Mac ให้เป็น **embedding server ที่เปิดไว้ตลอดเวลา** ของโรงแรม: HTTP service ในเครื่องที่อุปกรณ์เล็ก ๆ เรียกใช้ได้ พร้อม load test 60 วินาที การอ่านค่ากำลังไฟ และไฟล์ launchd ที่ทำให้มันรันอยู่ตลอด

**Time** ~60 นาที · **Difficulty** ระดับสูง · **Hardware** Mac ที่ใช้ Apple silicon (M1 ขึ้นไป) · ดิสก์ว่างเพิ่มอีก ~4 GB สำหรับ build ของ MLX และ ONNX

**แหล่งอ้างอิง:** [mlx-community/embeddinggemma-2-bf16](https://huggingface.co/mlx-community/embeddinggemma-2-bf16) · [mlx-community/embeddinggemma-2-4bit](https://huggingface.co/mlx-community/embeddinggemma-2-4bit) · [onnx-community/embeddinggemma-2-ONNX](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX) · [MLX](https://github.com/ml-explore/mlx) · [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) · [บล็อกเปิดตัว](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/) (มี MLX อยู่ในรายการ) · Tina Huang, [*Every Size Local AI In 24 Minutes*](https://youtu.be/rPGJhrunbxo) · [`week30_multimodal_embeddings_edge/RESEARCH.md`](../RESEARCH.md) §5

## 0 · ก่อนเริ่ม

MLX มี environment แยกของตัวเองคือ `week30_multimodal_embeddings_edge/.venv-mlx` เพราะ build ของ MLX ถูกแปลงด้วย commit เฉพาะของ `mlx-vlm` และ model card ระบุให้ติดตั้ง commit นั้นเท่านั้น แล็บจะเริ่มใน `.venv-embed` เหมือนแล็บอื่นทุกแล็บ แล้วสลับไปใช้ `.venv-mlx` เองเมื่อต้องใช้ MLX ตอนนั้นคุณจะเห็นบรรทัด `→ MLX lives in week30_multimodal_embeddings_edge/.venv-mlx`

```bash
# on: laptop
uv venv --python 3.12 week30_multimodal_embeddings_edge/.venv-mlx
uv pip install --python week30_multimodal_embeddings_edge/.venv-mlx/bin/python "git+https://github.com/Blaizzy/mlx-vlm.git@3d87e88402f307efbf68e568971aa887ee7d9ed0" "mlx>=0.32.3" "transformers>=5.18.0" torch torchvision torchcodec soundfile librosa pillow numpy fastapi uvicorn
uv pip install --python week30_multimodal_embeddings_edge/.venv-embed/bin/python onnxruntime
week30_multimodal_embeddings_edge/.venv-mlx/bin/python -c "import mlx.core as mx; print('mlx', mx.__version__, mx.default_device())"
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
mlx 0.32.3 Device(gpu, 0)
```

ใน environment ของ MLX มี `torch` อยู่ด้วยเพียงเพราะ processor ของ model ต้องใช้มันแปลงรูปถ่ายและเสียงให้เป็น input ของ model ส่วนงาน embedding ทั้งหมด MLX เป็นคนทำ แล็บจะดาวน์โหลดแต่ละ build แค่ครั้งเดียว: MLX bf16 ขนาด 1.49 GB, MLX 4-bit 1.10 GB และ ONNX text model 175 MB (q4) กับ 543 MB (fp16) ไฟล์ ONNX จะเก็บไว้ที่ `week30_multimodal_embeddings_edge/.models/` (git ไม่ track) เพราะ ONNX Runtime ไม่ยอมรับไฟล์แบบ symlink ใน cache ของ Hugging Face

✓ Checkpoint: `mlx` พิมพ์เลขเวอร์ชันและ `Device(gpu, 0)` ออกมา

## 1 · ทำไม Mac จึงเป็น edge box ที่ดี

Mac ที่ใช้ Apple silicon มีสิ่งที่อุปกรณ์ edge ส่วนใหญ่ไม่มี คือ **unified memory** CPU, GPU และ Neural Engine ใช้ RAM ก้อนเดียวกัน ถ้าเทียบกับครัวใน Module 01 ก็คือโต๊ะเตรียมอาหารตัวใหญ่ตัวเดียวที่หัวหน้าเชฟ พ่อครัว และเครื่องหั่นหัวหอมเอื้อมถึงได้หมด ไม่ต้องคัดลอกอะไรไปการ์ดจอแยก Mac ที่มี 36 GB จึงถือ embedding model, vector index และ Gemma 4 ในเครื่องไว้พร้อมกันได้

| บน Mac เครื่องนี้ | ค่า | ที่มา |
|---|---|---|
| ชิป · หน่วยความจำ | Apple M4 Max · 36 GB unified | Lab 01-2 |
| ความเร็วคัดลอกหน่วยความจำของ GPU | 314 GB/s (จาก peak 410 GB/s) | Lab 01-2, สเปกของ Apple |
| ขั้นในบันได | personal computer (Mac mini หรือ Mac Studio = home server) | การแบ่งระดับของ Tina Huang |

วิธีรัน EmbeddingGemma 2 บน Mac มีสามแบบ:

| runtime | คืออะไร | build ที่ใช้ในแล็บนี้ |
|---|---|---|
| PyTorch + MPS | stack อ้างอิง; MPS คือ backend ของ PyTorch สำหรับ GPU ของ Apple | `google/embeddinggemma-2` (Labs 02–07) |
| **MLX** | array framework แบบ open-source ของ Apple ออกแบบมาเพื่อ Apple silicon และ unified memory | `mlx-community/embeddinggemma-2-bf16`, `-4bit` |
| **ONNX Runtime** | inference engine ที่พกพาไปใช้ได้หลายแพลตฟอร์ม; บน Mac ส่งงานต่อให้ CoreML และ Neural Engine ได้ | `onnx-community/embeddinggemma-2-ONNX` (q4, fp16 text model) |

✓ Checkpoint: คุณอธิบายได้ว่าทำไม unified memory จึงสำคัญ เมื่อ Mac เครื่องเดียวรันทั้ง embedder, index และ chat model พร้อมกัน

## 2 · สาม runtime บน Mac เครื่องเดียว

Lab 08-1 รันงานข้อความชุดเดียวกันในแต่ละ runtime โดยแยก process ใหม่ทุกครั้ง เวลาโหลดและหน่วยความจำจึงเป็นของ runtime นั้นจริง ๆ งานคือคำถาม 18 ข้อกับโน้ต 18 ชิ้น โดยพิมพ์ prefix ที่ถูกต้องใส่เอง เพราะมีแค่ sentence-transformers ที่เติมให้อัตโนมัติ จากนั้นเทียบทุกผลกับ PyTorch float32 บน CPU

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/08_mac_edge_box/labs/lab08_1_three_runtimes.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
▣ STEP 6 · ONNX Runtime · q4 · CoreML
◆ CoreML could take 238 of 1066 graph nodes (22 %) — the rest would stay on the CPU
⚠ ONNX Runtime · q4 · CoreML did not run: Input (video_features) has a dynamic shape ({-1,512}) but the runtime shape ({0,512}) has zero elements. This is not supported by the CoreML EP.

▣ STEP 7 · agreement with PyTorch float32 on the CPU (the reference)
│ runtime                    weights on disk  load  1 query  per note in ×32  recall@1  min cos vs fp32  peak memory
│ ─────────────────────────  ───────────────  ────  ───────  ───────────────  ────────  ───────────────  ──────────────────
│ PyTorch + MPS · bf16       1489 MB          3.7s  22.5 ms  3.7 ms           1.000     0.9999           1232 MB
│ MLX · bf16                 1489 MB          0.8s  7.3 ms   2.1 ms           1.000     0.9998           2644 MB (MLX 2029)
│ MLX · 4-bit                1098 MB          0.9s  6.1 ms   2.1 ms           1.000     0.9598           2274 MB (MLX 1690)
│ ONNX Runtime · q4 · CPU    175 MB           1.0s  15.5 ms  26.8 ms          1.000     0.9703           1278 MB
│ ONNX Runtime · fp16 · CPU  543 MB           1.3s  17.2 ms  32.4 ms          1.000     1.0000           1500 MB
```

ตารางนี้บอกอะไร:

- **MLX เร็วที่สุดบน Mac เครื่องนี้** คำถามหนึ่งข้อใช้ 7.3 ms ใน MLX bf16 เทียบกับ 22.5 ms ใน PyTorch เร็วกว่าสามเท่า และได้คำตอบเหมือนกัน (cosine 0.9998) MLX ยังโหลดเสร็จในไม่ถึงวินาทีด้วย
- **MLX แบบ 4-bit ได้ขนาดที่เล็กลง แต่ไม่ได้ความเร็วเพิ่มมากนักในกรณีนี้** ไฟล์เล็กลง 26 % และเร็วขึ้นเล็กน้อยสำหรับคำถามเดียว (6.1 เทียบกับ 7.3 ms) ส่วนแบบ batch ก็ยัง 2.1 ms ต่อโน้ตเท่าเดิม vector ของมันยังเพี้ยนไปมากกว่าด้วย (cosine แย่สุด 0.96) text model ขนาดเล็กบน GPU ใหญ่ไม่ได้ติดคอขวดที่หน่วยความจำ (Module 06) การลดจำนวนไบต์จึงแทบไม่ช่วยเรื่องความเร็ว
- **ONNX q4 เล็กที่สุดแบบทิ้งห่าง** ขนาด 175 MB รองรับแค่ข้อความและรันบน CPU จึงเหมาะกับเครื่องที่ดิสก์น้อย คำถามเดียวทำได้เร็ว (15.5 ms) แต่แบบ batch ช้า (26.8 ms ต่อโน้ต)
- **ทั้งห้าแบบจัดอันดับถูกทุกคำถาม** (recall@1 = 1.000) ความต่างอยู่ที่ความเร็ว ขนาด และการเพี้ยนของ vector ไม่ใช่คำตอบ เพราะชุดคำถามนี้ง่าย

และนี่คือความจริงจาก step 6: เส้นทาง **Neural Engine** ใช้กับ model นี้ไม่ได้ในวันนี้ CoreML รับไปได้แค่ 22 % ของ graph และไม่ยอมรับ video input ที่ว่างของ model (tensor ที่ไม่มีแถวเลย ซึ่งการเรียกแบบข้อความอย่างเดียวจะส่งเข้าไปแบบนั้น) คำว่า "Mac มี NPU" ไม่ได้แปลว่า "model นี้ใช้ NPU ได้" ตรวจสอบก่อนจะไปสัญญากับใคร

✓ Checkpoint: คุณบอกได้ว่า runtime ไหนเร็วที่สุดบน Mac ของคุณ ตัวไหนเล็กที่สุด และตัวไหนควรใช้เป็นตัวอ้างอิง

## 3 · MLX กับรูปถ่ายและข้อความเสียง

MLX รัน multimodal model ได้ **ทั้งตัว** ไม่ใช่แค่ข้อความ Lab 08-2 ทำคำถามข้าม modality เจ็ดข้อจาก Module 04 ซ้ำอีกครั้งใน MLX bf16 และ MLX 4-bit แล้ววัดว่าแต่ละ build เพี้ยนจาก PyTorch float32 แค่ไหนในแต่ละ modality model card ของรุ่น 4-bit เตือนไว้ว่าเสียงคือ modality ที่ได้รับผลจาก quantization มากที่สุด

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/08_mac_edge_box/labs/lab08_2_mlx_multimodal.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
▣ STEP 4 · compare
│ build     weights  questions right  min cos text  min cos photo  min cos voice  per photo  per voicemail
│ ────────  ───────  ───────────────  ────────────  ─────────────  ─────────────  ─────────  ─────────────
│ MLX bf16  1489 MB  7/7              0.9999        0.9999         1.0000         151 ms     54 ms
│ MLX 4bit  1098 MB  7/7              0.9865        0.9831         0.9794         150 ms     59 ms
◆ REFERENCE (mlx-community 4-bit card): minimum cosine vs FP32 — text 0.980877 · image 0.990386 · audio 0.974435
```

ทั้งสอง build ตอบถูกครบเจ็ดข้อ และ MLX เร็วกว่า PyTorch ใน Module 04: 151 ms ต่อรูปเทียบกับ 247 ms และ 54 ms ต่อข้อความเสียงเทียบกับ 111 ms ลองเทียบแถว 4-bit กับ model card ดู ค่าของข้อความและเสียงใกล้เคียงกับตัวเลขใน card แต่ค่าของ **รูปถ่าย** ต่ำกว่า (0.983 เทียบกับ 0.990 ใน card) เพราะเป็นรูปคนละชุดกับชุดทดสอบของ card นี่คือเหตุผลที่คุณต้องวัดกับข้อมูลของตัวเอง ถ้าจะทำ index ข้อความเสียง ให้เลือก bf16: มันใกล้กับต้นฉบับที่สุด และบน Mac เครื่องนี้ 4-bit ก็ไม่ได้เร็วกว่า

✓ Checkpoint: คุณเลือกได้ระหว่าง MLX bf16 กับ 4-bit สำหรับ index ข้อความเสียง และให้เหตุผลจากคอลัมน์ voice ได้

## 4 · Mac mini เป็น embedding server ของโรงแรม

ระดับ home server ของ Tina Huang คือ "เครื่องเฉพาะที่เปิดไว้ตลอดเวลา" Arduino ของเธอเล็กเกินกว่าจะรัน model ใดได้ จึงเรียกใช้ Mac Studio ผ่าน Wi-Fi `week30_multimodal_embeddings_edge/common/embed_service.py` นำรูปแบบเดียวกันนี้มาใช้กับ embedding มันคือ FastAPI service ตัวเล็กที่รันบน MLX 4-bit (คำถามเดี่ยวเร็วที่สุด) ทำ embedding โน้ตของโรงแรมไว้ล่วงหน้า และตอบ `/embed` กับ `/search` ด้วย JSON ธรรมดา gateway, Pi หรือ microcontroller ส่งคำเข้าไป แล้วได้ความหมายกลับมา

ความปลอดภัยถูกสร้างไว้ในตัว โดยค่าเริ่มต้น service จะฟังแค่ที่ **loopback** (`127.0.0.1`) และจะไม่ยอมฟังที่ network address ใด ๆ เลย จนกว่าจะตั้ง `EMBED_TOKEN` ยาว 24 ตัวอักษรขึ้นไป เมื่อตั้งแล้ว ทุก request ต้องแนบ token นั้นมาด้วย ใครก็ตามที่เข้าถึง `/search` ได้ก็อ่านคลังความรู้ได้ ดังนั้นให้เปิดบน network ที่ไว้ใจได้เท่านั้น

Lab 08-3 เปิด service แล้วสวมบทเป็นอุปกรณ์เล็ก รัน load test 60 วินาที และตรวจกำลังไฟ:

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/08_mac_edge_box/labs/lab08_3_always_on_service.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ ขณะเสียบไฟ AC)

```
▣ STEP 2 · be the small device: plain JSON over HTTP
  POST /search {"q": "someone is stuck in the lift"}  →  [{'title': 'Lift entrapment', 'score': 0.832}, {'title': 'Pool pump trip', 'score': 0.651}]
│ dim  request  response (JSON)  round trip
│ ───  ───────  ───────────────  ──────────
│ 768  54 B     16,253 B         9.1 ms
│ 256  54 B     5,377 B          8.8 ms
│ 128  54 B     2,702 B          8.6 ms
◆ a microcontroller with 8 MB of RAM can parse a 128-d answer (≈ 2.7 KB of JSON); send /search instead and it never handles a vector at all

▣ STEP 3 · round-trip latency: 30 searches
✓ p50 9.0 ms · p95 9.8 ms · over loopback HTTP + JSON

▣ STEP 4 · sustained load: 60 s of 32-note batches — does the Mac slow down?
  window 1 (15 s):    409 notes/s  ████████████████████████
  window 2 (15 s):    387 notes/s  ███████████████████████░
  window 3 (15 s):    383 notes/s  ██████████████████████░░
✓ 23,264 notes in 60 s · last window vs first: -6.3 %

▣ STEP 5 · power — no sudo needed
◆ power source: AC — the battery is not discharging, so its draw cannot be read without sudo
◆ unplug and rerun for whole-Mac watts, or run Apple's tool yourself in the ⌨ terminal:
  $ sudo powermetrics --samplers cpu_power,gpu_power -i 1000 -n 10
```

การค้นหาหนึ่งครั้งใช้เวลา 9 ms ตั้งแต่ต้นจนจบ รวม HTTP และ JSON แล้ว Mac ทำ embedding ได้ราว 390 โน้ตต่อวินาทีต่อเนื่องหนึ่งนาทีเต็ม และช้าลง 6 % เมื่อเครื่องร้อนขึ้น ด้วยอัตรานี้ โน้ตโรงแรม 50,000 ชิ้นจาก Module 06 ใช้เวลาประมาณสองนาที

เรื่องกำลังไฟ แล็บจะอ่านค่าการดึงไฟจากแบตเตอรี่ตอนที่ Mac ไม่ได้เสียบสาย ซึ่งไม่ต้องใช้รหัสผ่าน admin ถ้าเสียบไฟ AC อยู่ แล็บจะบอกตรง ๆ และให้คำสั่ง `powermetrics` ของ Apple ไว้ให้คุณรันเอง เพราะคำสั่งนั้นต้องใช้ `sudo`

ขั้นสุดท้ายจะเขียนไฟล์ launchd ที่สั่งให้ service เริ่มทำงานตอน login และรีสตาร์ตเมื่อมันหยุด แต่จะ **ไม่** ติดตั้งให้:

```text
week30_multimodal_embeddings_edge/08_mac_edge_box/.runs/com.alto.edge-memory.plist   (RunAtLoad · KeepAlive · --host 127.0.0.1 · HF_HUB_OFFLINE=1)
```

```bash
# on: laptop
# only if you want it running permanently — this starts it at every login
cp week30_multimodal_embeddings_edge/08_mac_edge_box/.runs/com.alto.edge-memory.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.alto.edge-memory.plist
launchctl unload ~/Library/LaunchAgents/com.alto.edge-memory.plist   # stop it
```

✓ Checkpoint: คุณรู้ค่า latency p50 ของ service, throughput ต่อเนื่องของมัน และ environment variable ตัวเดียวที่ต้องตั้งก่อนจะให้มันฟังบน network ได้

## Labs — รันแล็บได้ที่นี่

**labs/lab08_1_three_runtimes.py** — วัด PyTorch MPS, MLX bf16/4-bit และ ONNX Runtime q4/fp16 เทียบกันบน Mac เครื่องนี้ พร้อมลองใช้ Neural Engine ผ่าน CoreML

**labs/lab08_2_mlx_multimodal.py** — รูปถ่ายและข้อความเสียงใน MLX bf16 และ 4-bit พร้อมคำถามข้าม modality เจ็ดข้อ และความแม่นยำเทียบกับ PyTorch แยกตาม modality

**labs/lab08_3_always_on_service.py** — Mac ในฐานะ embedding server ในเครื่อง: JSON round trip, load test 60 วินาที, วัดกำลังไฟโดยไม่ต้องใช้ sudo และไฟล์ launchd

## Try it yourself — ลองทำเอง

`exercises/ex08_mac_edge_plan.py` เปลี่ยนผลการวัดให้เป็นแผน deploy:

1. เลือก runtime สำหรับงานสี่อย่างจากตาราง **ที่วัดจริง**
2. `safe_bind(host, token)`: กฎ network ของ service
3. `joules_per_1000(watts, notes_per_s)`: คำนวณพลังงานจากค่ากำลังไฟที่อ่านได้
4. แก้ไฟล์ launchd ที่เสีย

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/08_mac_edge_box/exercises/ex08_mac_edge_plan.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ เมื่อทำ TODO ครบทุกจุด)

```
✓ runtimes: voicemails → MLX bf16 · smallest text-only → ONNX q4 · fastest query → MLX 4-bit · reference → PyTorch
✓ safe_bind: loopback always · a network address only with a token of ≥ 24 characters
✓ joules_per_1000: 18 W at 400 notes/s = 45.0 J · 9.5 W at 120/s = 79.2 J
✓ launchd plist: starts at login, restarts if it dies, loopback only, offline
```

<details><summary>คำใบ้ — วัดกำลังไฟด้วยตัวเอง</summary>

ถอดสายชาร์จ Mac แล้วรัน Lab 08-3: step 5 จะอ่านค่าการดึงไฟจากแบตเตอรี่และพิมพ์จำนวนจูลต่อ 1,000 โน้ต หรือถ้าเสียบสายอยู่ ให้รัน `sudo powermetrics --samplers cpu_power,gpu_power -i 1000 -n 10` ในเทอร์มินัลอีกหน้าต่างระหว่างที่ load test กำลังรัน แล้วเอาค่าวัตต์ไปใส่ใน `joules_per_1000` ของคุณ

</details>

✓ Checkpoint: ตัวตรวจขึ้น ✓ ครบทั้งสี่บรรทัด

## Troubleshooting — แก้ปัญหา

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `✕ week30_multimodal_embeddings_edge/.venv-mlx is missing` | ยังไม่ได้สร้าง environment ของ MLX | รันคำสั่งใน Section 0 |
| `Could not import module 'EmbeddingGemma2Processor'` | processor ต้องใช้ PyTorch แต่ `.venv-mlx` ไม่มี | ติดตั้ง `torch torchvision` ลงใน `.venv-mlx` (คำสั่งใน Section 0 ทำให้แล้ว) |
| `External data path escapes model directory` | ONNX Runtime ไม่ยอมรับ blob แบบ symlink ใน cache ของ Hugging Face | แล็บจึงดาวน์โหลด ONNX ไปไว้ที่ `week30_multimodal_embeddings_edge/.models/` แทน |
| step ของ CoreML ล้มเหลว | ไม่รองรับ media input ที่ว่างของ model และรันบนนั้นได้แค่ 22 % ของ graph | เป็นเรื่องปกติในวันนี้ ให้ใช้ MLX หรือ ONNX บน CPU |
| `refusing to listen on a network address` | ไม่มี `EMBED_TOKEN` หรือมีแต่สั้นกว่า 24 ตัวอักษร | `export EMBED_TOKEN=$(openssl rand -hex 24)` และใช้บน network ที่ไว้ใจได้เท่านั้น |
| service ของ launchd ไม่เริ่มทำงาน | path ใน plist ผิด หรือ model ยังไม่ถูก cache | ดู `/tmp/alto-edge-memory.log` และรันแล็บสักครั้งเพื่อ cache model |

## Next — บทถัดไป

คุณเปลี่ยน Mac จากตัวแทนชั่วคราวให้กลายเป็น edge box ตัวจริงแล้ว กลับไปที่คอร์ส: [Lab 01 — Local, sovereign, edge](../01_local_sovereign_edge/TUTORIAL.md) หรือรัน [Lab 07 — the capstone](../07_capstone_alto_edge_memory/TUTORIAL.md) อีกครั้ง โดยให้ `search_memory` ของเอเจนต์เรียก service นี้แทนการโหลด model เอง
