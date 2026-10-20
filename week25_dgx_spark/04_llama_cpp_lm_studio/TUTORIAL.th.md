# ▶ Spark Lab 04 — llama.cpp + LM Studio: GGUF และโมเดลแบบ quantized

> ส่วนหนึ่งของ Week 25 · DGX Spark: fine-tune, serve และสร้างเอเจนต์ที่รันใน sandbox คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง ทุกแล็บรันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (อ้างอิงคำต่อคำจาก playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- คอมไพล์ llama.cpp จากซอร์สโค้ดพร้อม CUDA สำหรับ GB10 (`CMAKE_CUDA_ARCHITECTURES=121a-real`) ตามที่ playbook ทำทุกขั้น
- เปิดไฟล์ GGUF จริง แล้วดูว่า `Q4_K_M` เก็บอะไรไว้บ้าง ทีละ tensor ลงไปถึงระดับไบต์
- quantize weights หนึ่งล้านตัวด้วยตัวเอง แล้ววัดว่าทุกบิตที่ตัดออกต้องแลกกับอะไร
- serve GGUF ของ `Qwen3.6-35B-A3B` ตาม playbook ด้วย `llama-server` บนพอร์ต **30080** แล้วเรียกผ่าน OpenAI API
- รัน LM Studio แบบ headless (`llmster` และ CLI `lms`) บนพอร์ต 1234
- ตัดสินใจว่าเมื่อไรควรใช้ llama.cpp, Ollama, LM Studio หรือ vLLM

**Time** ~50 นาที · **Difficulty** ระดับกลาง · **Hardware** DGX Spark 1 เครื่อง (หรือไม่มีเลยก็ได้: ใช้โหมด DRY + แล็บที่เป็นการคำนวณ + llama-server บนแล็ปท็อป (ไม่บังคับ))

**Playbook ทางการที่ครอบคลุม:** [llama.cpp](https://build.nvidia.com/spark/llama-cpp) · [LM Studio](https://build.nvidia.com/spark/lm-studio)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| ทำ Module 03 เสร็จแล้ว | คุณเรียก Ollama ผ่าน `/v1` ได้ | โมดูลนี้ใช้แนวคิด OpenAI client ชุดเดิมซ้ำ |
| เครื่องมือ build บน Spark | `git --version`, `cmake --version` (3.14+), `nvcc --version` | playbook คอมไพล์ llama.cpp จากซอร์สโค้ด |
| ดิสก์ว่าง ~40 GB บน Spark | `df -h /` | playbook บอกว่า: "at least ~40 GB free disk for the example download plus build artifacts" |
| หน่วยความจำว่าง ~30 GB บน Spark | `free -g` | playbook บอกว่า: "about 30 GB free for the example model" |
| numpy ใน `.venv` ของ repo | `.venv/bin/python -c "import numpy"` | lab 03 quantize weights หนึ่งล้านตัว |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
.venv/bin/python -c "import numpy; print('numpy', numpy.__version__)"
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
numpy 2.4.2
```

> ⚠ **เปลี่ยนพอร์ต** llama.cpp playbook serve บน **พอร์ต 30000** แต่ SGLang (Module 06) ก็ใช้ 30000 เป็นค่าเริ่มต้นเช่นกัน คอร์สนี้จึงรัน `llama-server` บน **30080** (`sparkkit.PORTS["llamacpp"]`) เพื่อให้สอง engine รันคู่กันได้ ทุกคำสั่งด้านล่างใช้ 30080 เวลาอ่าน playbook ให้แทน 30000 ด้วย 30080

✓ Checkpoint: import numpy ได้ และคุณรู้ว่าคอร์สนี้ใช้พอร์ตไหนสำหรับ llama-server และเพราะอะไร

## 1 · llama.cpp, Ollama และ LM Studio: engine เดียว สามเปลือกหุ้ม

[llama.cpp](https://github.com/ggml-org/llama.cpp) คือ inference engine ที่เขียนด้วย C/C++ ไลบรารี tensor ของมันชื่อ **GGML** เป็นผู้กำหนดฟอร์แมตไฟล์ **GGUF**: ไฟล์เดียวที่เก็บ weights (ส่วนใหญ่เป็นแบบ quantized) tokenizer, chat template และการตั้งค่าของโมเดล playbook เขียนไว้ว่า: "a lightweight C/C++ inference stack … load GGUF weights and expose chat through `llama-server`'s OpenAI-compatible HTTP API."

Ollama (Module 03) และ LM Studio รันโมเดล GGUF ได้ทั้งคู่ และทั้งคู่เพิ่มชั้นที่ใช้งานง่ายกว่าครอบไว้ด้านบน:

| | llama.cpp (`llama-server`) | Ollama | LM Studio (`llmster` / `lms`) |
|---|---|---|---|
| วิธีติดตั้ง | คอมไพล์จากซอร์ส (โมดูลนี้) | สคริปต์เดียว | สคริปต์เดียว |
| โมเดลมาจาก | GGUF ใดก็ได้บน Hugging Face (`-hf org/repo:QUANT`) | คลังของ Ollama (`ollama pull`) | แคตตาล็อกของ LM Studio (`lms get`) |
| ไฟล์โมเดล | `.gguf` ธรรมดา อยู่ใน `~/.cache/huggingface/hub` | blob แบบ GGUF ในที่เก็บของ Ollama เอง | ไฟล์ใต้ `~/.lmstudio/models/` |
| เซิร์ฟเวอร์หนึ่งตัวถือ | หนึ่งโมเดลต่อหนึ่งโปรเซส (ค่าเริ่มต้น) | หลายโมเดล โหลดและปลดตามต้องการ | โมเดลที่คุณ `lms load` |
| ปุ่มปรับ (knob) | ทุก flag: context, speculative decoding, GPU layers, slots | environment variable ไม่กี่ตัวและ Modelfile | flag ของ CLI และ GUI |
| พอร์ตเริ่มต้น | 8080 (playbook: 30000, **คอร์ส: 30080**) | 11434 | 1234 |
| API | OpenAI `/v1` + `/health` + `timings` | `/api` แบบ native + OpenAI `/v1` | OpenAI `/v1` + LM Studio REST `/api/v1` |

GGUF จากเครื่องมือหนึ่งไม่ได้โหลดในอีกเครื่องมือหนึ่งได้เสมอไป สำเนา `gemma3:4b` ของ Ollama บน Mac เครื่องนี้เป็น GGUF ที่ถูกต้อง (lab 02 อ่านมันได้) แต่เมื่อเราชี้ `llama-server` จาก Homebrew ไปที่ไฟล์นี้ มันหยุดพร้อมข้อความ:

```
E llama_model_load: error loading model: error loading model hyperparameters: key not found in model: gemma3.attention.layer_norm_rms_epsilon
```

Ollama แพ็กบางโมเดลด้วยวิธีของตัวเอง ให้ดาวน์โหลด GGUF สำหรับ llama.cpp จาก Hugging Face ตามที่ playbook ทำ

✓ Checkpoint: คุณบอกได้ว่าไฟล์ GGUF มีอะไรอยู่ข้างใน และทำไมโมเดลเดียวกันจึงมีไฟล์ต่างกันใน Ollama และบน Hugging Face

## 2 · คอมไพล์ llama.cpp สำหรับ CUDA บน GB10

ขั้นตอนที่ 1–3 ของ playbook บน **Spark**:

```bash
# on: spark
sudo apt update
sudo apt install -y git clang cmake libcurl4-openssl-dev libssl-dev
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp
cmake -B build -DGGML_NATIVE=ON -DGGML_CUDA=ON -DGGML_CURL=ON -DGGML_RPC=ON -DCMAKE_CUDA_ARCHITECTURES=121a-real
cmake --build build --config Release --target llama-server -j
```

ความหมายของแต่ละ flag:

| Flag | ทำไม |
|---|---|
| `-DGGML_CUDA=ON` | คอมไพล์ CUDA backend เพื่อให้ GPU ของ GB10 เป็นผู้ทำงาน |
| `-DCMAKE_CUDA_ARCHITECTURES=121a-real` | คอมไพล์สำหรับ compute capability 12.1 (sm_121) ของ GB10 ส่วน troubleshooting ของ playbook บอกว่า: "Build errors mentioning wrong GPU arch → use `-DCMAKE_CUDA_ARCHITECTURES=121a-real`" |
| `-DGGML_NATIVE=ON` | ปรับโค้ดฝั่ง CPU ให้เข้ากับคอร์ Arm ของเครื่องนี้ |
| `-DGGML_CURL=ON` | ทำให้ `-hf` ดาวน์โหลดโมเดลจาก Hugging Face ได้ |
| `-DGGML_RPC=ON` | คอมไพล์ RPC backend ของ llama.cpp (สำหรับกระจายโมเดลข้ามหลายเครื่อง; คอร์สนี้ไม่ได้ใช้) |
| `--target llama-server` | คอมไพล์เฉพาะเซิร์ฟเวอร์ ไม่ใช่ทุกเครื่องมือ |

playbook บอกว่าการคอมไพล์ "usually takes on the order of 5–10 minutes" Lab 01 ตรวจเครื่องมือ แล้ว (เมื่อใส่ `--build`) จะรันการ clone และคอมไพล์ **อยู่เบื้องหลัง** พร้อม log เพื่อไม่ให้ขีดจำกัด 15 นาทีของ Lab Runner ฆ่ามันทิ้ง บรรทัด `sudo apt` จะพิมพ์ออกมาให้คุณพิมพ์เอง แล็บไม่เคยใช้ sudo

```bash
# on: laptop
.venv/bin/python week25/04_llama_cpp_lm_studio/labs/lab01_build_llama_cpp.py            # read-only checks
.venv/bin/python week25/04_llama_cpp_lm_studio/labs/lab01_build_llama_cpp.py --build    # start the build
```

**Expected output** (โหมด DRY บันทึกจาก Mac เครื่องนี้โดยไม่ใส่ `--build`)

```
▣ STEP 1 · build tools: git, cmake ≥ 3.14, the CUDA compiler, an Arm CPU
$ uname -m && git --version && cmake --version | head -1 && (nvcc --version || /usr/local/cuda/bin/nvcc --version) | tail -1   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
aarch64
git version 2.43.0
cmake version 3.28.3
Build cuda_13.0.r13.0/compiler.xxxxxxxx_0
◆ nvcc not found? The playbook's fix: export PATH=/usr/local/cuda/bin:$PATH, then run CMake again from a clean build dir.
…
▣ STEP 3 · clone and build llama-server with CUDA for GB10 (sm_121)
→ add --build to run, in the background:
  git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp && cd ~/llama.cpp
  cmake -B build -DGGML_NATIVE=ON -DGGML_CUDA=ON -DGGML_CURL=ON -DGGML_RPC=ON -DCMAKE_CUDA_ARCHITECTURES=121a-real
  cmake --build build --config Release --target llama-server -j
```

ติดตามการคอมไพล์ที่กำลังรันจากแล็ปท็อป:

```bash
# on: laptop
ssh spark-a tail -f ~/w25/logs/llama-build.log
```

เมื่อเสร็จแล้ว `llama-server` จะอยู่ที่ `~/llama.cpp/build/bin/llama-server`

✓ Checkpoint: `ssh spark-a ls ~/llama.cpp/build/bin/llama-server` เจอไฟล์ binary หรือในโหมด DRY คุณอธิบายได้ว่า `121a-real` เลือกอะไร

## 3 · GGUF จากข้างใน: `Q4_K_M` เก็บอะไรจริง ๆ

GGML quantize weights เป็น **บล็อก (block)** แต่ละบล็อกเก็บจำนวนเต็มไม่กี่บิตต่อ weight บวก scale ขนาดเล็กหนึ่งหรือสองตัว ต้นทุนจริงต่อ weight จึงสูงกว่าความกว้างของจำนวนเต็มเล็กน้อย:

| ชนิด GGML | weights ต่อบล็อก | ไบต์ต่อบล็อก | บิตต่อ weight | ในบล็อกมีอะไร |
|---|---|---|---|---|
| `F16` / `BF16` | 1 | 2 | 16 | ไม่ quantize |
| `Q8_0` | 32 | 34 | 8.5 | 32 × int8 + fp16 scale หนึ่งตัว |
| `Q6_K` | 256 | 210 | 6.5625 | ค่า 6 บิต + sub-scale 8 บิต + fp16 scale |
| `Q5_K` | 256 | 176 | 5.5 | ค่า 5 บิต + sub-scale 6 บิต + fp16 scale สองตัว |
| `Q4_K` | 256 | 144 | 4.5 | ค่า 4 บิต + sub-scale 6 บิต + fp16 scale สองตัว |
| `Q4_0` | 32 | 18 | 4.5 | 32 × 4 บิต + fp16 scale หนึ่งตัว |
| `Q3_K` | 256 | 110 | 3.4375 | ค่า 3 บิต + sub-scale |
| `Q2_K` | 256 | 84 | 2.625 | ค่า 2 บิต + sub-scale 4 บิต |

ชื่ออย่าง **`Q4_K_M` คือสูตร (recipe) ไม่ใช่ชนิดข้อมูล**: tensor ส่วนใหญ่ได้ `Q4_K` ตัวที่อ่อนไหวที่สุด (เช่น ครึ่งหนึ่งของชั้น `ffn_down` และ embeddings) ได้ `Q6_K` และตัวที่เล็กมากยังคงเป็น `F32` คุณไม่ต้องเชื่อตามนี้เฉย ๆ Lab 02 อ่าน header ของ GGUF จริง รวมไบต์ของทุก tensor จากตารางด้านบน แล้วเทียบผลรวมกับขนาดไฟล์:

```bash
# on: laptop
.venv/bin/python week25/04_llama_cpp_lm_studio/labs/lab02_gguf_inside.py
```

ถ้าไม่ใส่ argument มันจะเปิด GGUF ที่เล็กที่สุดที่ Ollama ดาวน์โหลดไว้บนเครื่องคุณ มันอ่านแค่ header (ไม่กี่ MB) ไม่เคยอ่าน weights

**Expected output** (บันทึกจาก Mac เครื่องนี้ โดยอ่านไฟล์ `gemma3:4b` ของ Ollama)

```
▣ STEP 1 · read the header of gemma3:4b
│ GGUF version  3 · 35 metadata keys · 883 tensors · header 15.8 MB (tokenizer included)
│ architecture  gemma3 · layers 34 · context 131072
│ file_type     15 → Q4_K_M   (what the file calls itself)

▣ STEP 2 · what each tensor is actually stored as
│ type  tensors  weights  bytes     bits/weight  share of file         e.g.
│ ────  ───────  ───────  ────────  ───────────  ────────────────────  ─────────────────────────────
│ Q4_K  205      2.721 B  1.531 GB  4.500        46.1% ██████░░░░░░░░  blk.0.attn_k.weight
│ Q6_K  34       1.159 B  0.950 GB  6.562        28.6% ████░░░░░░░░░░  token_embd.weight
│ F16   165      0.419 B  0.839 GB  16.000       25.2% ████░░░░░░░░░░  mm.mm_input_projection.weight
│ F32   479      0.001 B  0.003 GB  32.000        0.1% ░░░░░░░░░░░░░░  mm.mm_soft_emb_norm.weight

▣ STEP 3 · does the arithmetic match the file?
│ sum of tensor bytes from block sizes     3,322,976,704
│ file size − header                       3,322,976,731   (includes ≤ 32-byte alignment padding per tensor)
✓ block sizes × tensor shapes = the file, to within alignment padding
◆ effective bits per weight for the whole file: 6.18 (4.30 B weights, 3.32 GB)

▣ STEP 4 · where the bits go: transformer layers vs embeddings vs vision
│ group                       weights  bytes     bits/weight
│ ──────────────────────────  ───────  ────────  ───────────
│ transformer layers (blk.*)  3.209 B  1.932 GB  4.82
│ vision tower (v.*, mm.*)    0.420 B  0.840 GB  16.02
│ embeddings, output, norms   0.671 B  0.551 GB  6.56
◆ A K-quant recipe keeps most layer weights at its base type (Q4_K = 4.5 bits for Q4_K_M) and gives the most sensitive ones more (Q6_K = 6.56 bits). Tiny tensors (norms) stay F32.
◆ This file also carries a vision encoder, stored at F16: it counts toward the download and memory, but text generation does not read it.
```

สามสิ่งที่ไฟล์จริงนี้สอน:

1. **ตารางแม่นยำ** 883 tensor และไบต์ที่คำนวณได้ต่างจากขนาดไฟล์เพียง 27 ไบต์ ซึ่งคือ alignment padding
2. **"Q4" คือ 4.82 บิตในชั้น layer และ 6.18 บิตทั้งไฟล์** ส่วน layer ใกล้เคียงค่าเฉลี่ย 4.85 บิตของ sparkkit สำหรับ `Q4_K_M` ทั้งไฟล์หนักกว่าเพราะ gemma3 ของ Ollama แพ็ก vision encoder แบบ F16 ขนาด 0.84 GB มาในไฟล์ด้วย ประเมินขนาดดาวน์โหลดจากทั้งไฟล์ ประเมินคุณภาพจากส่วน layer
3. **โมเดลเล็กทำให้สูตรเพี้ยน** เรายังชี้ lab 02 ไปที่ `qwen2.5-0.5b-instruct-q4_k_m.gguf` ของ Qwen เอง (`--path FILE.gguf`) ในไฟล์นั้น 133 จาก 170 tensor ที่ถูก quantize เป็น `Q5_0` ไม่ใช่ `Q4_K` K-quant แพ็ก 256 weights ต่อ super-block แต่แถว (row) ของโมเดลนั้นกว้าง 896 (ไม่ใช่พหุคูณของ 256) ตัว quantize จึงถอยไปใช้ชนิดที่มี 32 weights ต่อบล็อก แล็บจะบอกเมื่อเกิดกรณีนี้

บน Spark ให้รันกับไฟล์ที่ดาวน์โหลดตาม playbook เมื่อ lab 01 `--serve` ดึงมาแล้ว:

```bash
# on: spark
cd ~/agenticaicodingfitness      # wherever you cloned this repo on the Spark
python3 week25/04_llama_cpp_lm_studio/labs/lab02_gguf_inside.py \
  --path '~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/*/*.gguf'
```

✓ Checkpoint: step 3 ของ lab 02 พิมพ์ ✓ และคุณอธิบายได้ว่าทำไมไฟล์ `Q4_K_M` จึงเฉลี่ยมากกว่า 4.5 บิตต่อ weight

## 4 · กี่บิตดี? ขนาดเทียบกับความคลาดเคลื่อน

บิตน้อยลงหมายถึงไฟล์เล็กลง มีที่ว่างสำหรับ KV cache มากขึ้น และ decode เร็วขึ้น (อ่านไบต์ต่อ token น้อยลง, Module 01) ราคาที่ต้องจ่ายคือความคลาดเคลื่อนจากการปัดเศษ (rounding error) Lab 03 วัดราคานั้นด้วยสูตร `Q8_0`, `Q4_0` และ `Q4_1` ของ ggml เอง บน weights สังเคราะห์หนึ่งล้านตัว (แจกแจงแบบ Gaussian มี outlier 0.1% เหมือน checkpoint จริง):

```bash
# on: laptop
.venv/bin/python week25/04_llama_cpp_lm_studio/labs/lab03_quant_tradeoffs.py
```

**Expected output** (เป็นการคำนวณที่ใช้ random seed คงที่: ได้ผลเหมือนกันทุกเครื่อง)

```
▣ STEP 1 · quantize 1,048,576 weights with each recipe and measure the error
│ recipe  bits/w  size      rel. error  SNR                           how
│ ──────  ──────  ────────  ──────────  ────────  ──────────────────  ─────────────────────────────────────────────
│ F16     16       2.10 MB    0.02%      73.7 dB  ░░░░░░░░░░░░░░░░░░  half precision, no blocks
│ Q8_0    8.5      1.11 MB    0.62%      44.1 dB  ░░░░░░░░░░░░░░░░░░  ggml Q8_0: 32 × int8 + fp16 scale
│ sym-6   6.5      0.85 MB    2.56%      31.9 dB  █░░░░░░░░░░░░░░░░░  same recipe, 6-bit values
│ Q4_1    5        0.66 MB    8.21%      21.7 dB  ██░░░░░░░░░░░░░░░░  ggml Q4_1: 32 × 4-bit + fp16 scale + fp16 min
│ Q4_0    4.5      0.59 MB    9.91%      20.1 dB  ███░░░░░░░░░░░░░░░  ggml Q4_0: 32 × 4-bit + fp16 scale
│ sym-3   3.5      0.46 MB   24.14%      12.3 dB  ███████░░░░░░░░░░░  same recipe, 3-bit values
│ sym-2   2.5      0.33 MB   63.88%       3.9 dB  ██████████████████  same recipe, 2-bit values

▣ STEP 2 · why blocks: one outlier ruins the scale for everyone who shares it
│ weights per scale  bits/w  rel. error  SNR
│ ─────────────────  ──────  ──────────  ────────  ──────────────────
│ 32                 4.500    11.18%      19.0 dB  ███░░░░░░░░░░░░░░░
│ 256                4.062    20.97%      13.6 dB  ██████░░░░░░░░░░░░
│ 4,096              4.004    55.15%       5.2 dB  █████████████████░
│ whole tensor       4.000    92.33%       0.7 dB  ██████████████████

▣ STEP 3 · what it means for the playbook's model: Qwen3.6-35B-A3B on one Spark
│ quant   bits/w  weights   left of 128 GB  decode ceiling  share of memory
│ ──────  ──────  ────────  ──────────────  ──────────────  ────────────────
│ BF16    16       70.0 GB   58.0 GB           46 tok/s     █████████░░░░░░░
│ Q8_0    8.5      37.2 GB   90.8 GB           86 tok/s     █████░░░░░░░░░░░
│ Q6_K    6.6      28.9 GB   99.1 GB          110 tok/s     ████░░░░░░░░░░░░
│ Q5_K_M  5.7      24.9 GB  103.1 GB          128 tok/s     ███░░░░░░░░░░░░░
│ Q4_K_M  4.85     21.2 GB  106.8 GB          150 tok/s     ███░░░░░░░░░░░░░
│ Q3_K_M  3.9      17.1 GB  110.9 GB          187 tok/s     ██░░░░░░░░░░░░░░
│ Q2_K    2.625    11.5 GB  116.5 GB          277 tok/s     █░░░░░░░░░░░░░░░
◆ The playbook's pick is unsloth's UD-Q4_K_XL (an Unsloth 'dynamic' ~4-bit recipe that keeps more tensors at higher precision). The playbook budgets about 30 GB of memory and a ~35 GB-order download for it.
◆ On a Spark, even BF16 of this 35B model fits. You quantize here for SPEED (fewer bytes per token) and for room: KV cache for long contexts and several models loaded side by side.
```

วิธีอ่าน:

- **ประมาณ 6 dB ต่อบิต** ทุกบิตที่ตัดออกทำให้ความคลาดเคลื่อนเพิ่มขึ้นราวเท่าตัว จาก 8 เหลือ 4 บิต ความคลาดเคลื่อนเพิ่มจาก 0.6% เป็นราว 10% และต่ำกว่า 4 บิตจะพุ่งขึ้นเร็วมาก
- **บล็อกคือเหตุผลที่ 4 บิตใช้ได้เลย** ถ้ามี scale ตัวเดียวทั้ง tensor outlier จะยืด scale ออกไปจนสัญญาณหายไป 92% บล็อกขนาด 32 weights มีต้นทุนแค่ครึ่งบิต แต่ลดความคลาดเคลื่อนลงเหลือ 11%
- **บน Spark ที่มี 128 GB การ quantize เป็นเรื่องความเร็วและที่ว่าง ไม่ใช่เรื่องใส่ได้หรือไม่** สำหรับโมเดล MoE 35B ของ playbook แบบ BF16 ก็ใส่ได้ แต่ Q4_K_M อ่านไบต์ต่อ token น้อยกว่าราว 3.3× เพดาน decode จึงอยู่ที่ ~150 tok/s แทนที่จะเป็น ~46

> ⚠ แล็บนี้วัดว่า **ตัวเลข** รอดมาได้ดีแค่ไหน ไม่ได้วัดว่า **โมเดล** ตอบได้ดีแค่ไหน คุณภาพขึ้นกับโมเดลและงาน [Run local LLMs playbook](https://build.nvidia.com/spark/llms) เขียนไว้ว่า: "Aggressive quantization can reduce response quality. NVFP4 or Q4_K_M are a good balance of throughput, accuracy, and memory." Module 07 ครอบคลุม NVFP4 และ Module 13 วัดคุณภาพกับโมเดลจริง

✓ Checkpoint: จากตาราง คุณบอกได้ว่าทำไม Q8_0 จึงเป็นค่าเริ่มต้นที่ปลอดภัยเมื่อหน่วยความจำพอ และทำไม Q2_K จึงเป็นทางเลือกสุดท้าย

## 5 · serve GGUF ด้วย llama-server บน :30080

playbook serve GGUF ของ Qwen3.6-35B-A3B ที่รองรับ MTP `-hf org/repo:QUANT` จะดาวน์โหลดไฟล์ลง `~/.cache/huggingface/hub` ในการใช้ครั้งแรก (และโหลดไฟล์ vision `mmproj` ให้อัตโนมัติถ้าโมเดลมี) จาก `~/llama.cpp/build` ด้วยพอร์ตของคอร์ส:

```bash
# on: spark
cd ~/llama.cpp/build
./bin/llama-server \
  -hf unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL \
  --host 0.0.0.0 \
  --port 30080
```

รุ่นที่เร็วกว่าของ playbook เพิ่ม **MTP speculative decoding** (Multi-Token Prediction: โมเดลร่าง token หลายตัวแล้วตรวจสอบในรอบเดียว; Module 07 อธิบายว่าทำไมจึงเร่ง decode ได้) และเก็บ thinking block ของ Qwen ไว้ในประวัติสำหรับเซสชันของเอเจนต์:

```bash
# on: spark
cd ~/llama.cpp/build
./bin/llama-server \
  -hf unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL \
  --host 0.0.0.0 \
  --port 30080 \
  --chat-template-kwargs '{"preserve_thinking": true}' \
  --spec-type draft-mtp \
  --spec-draft-n-max 3
```

เปิดเทอร์มินัลนั้นทิ้งไว้ หรือให้ lab 01 เปิดเซิร์ฟเวอร์อยู่เบื้องหลัง (`--serve`, log อยู่ที่ `~/w25/logs/llama-server.log`) เซิร์ฟเวอร์พร้อมเมื่อ log บอกว่ากำลัง listen:

**Expected output** (REFERENCE — ยกมาจาก llama.cpp playbook; การรันใน playbook ใช้พอร์ต 30000 ของคุณจะขึ้นว่า 30080)

```
0.14.322.968 I srv    load_model: speculative decoding context initialized
0.14.322.970 I slot   load_model: id  0 | task -1 | new slot, n_ctx = 262144
0.14.322.972 I slot   load_model: id  1 | task -1 | new slot, n_ctx = 262144
0.14.322.972 I slot   load_model: id  2 | task -1 | new slot, n_ctx = 262144
0.14.322.973 I slot   load_model: id  3 | task -1 | new slot, n_ctx = 262144
0.14.323.063 I srv    load_model: prompt cache is enabled, size limit: 8192 MiB

...
0.14.342.935 I srv  llama_server: model loaded
0.14.342.939 I srv  llama_server: server is listening on http://0.0.0.0:30000
0.14.342.944 I srv  update_slots: all slots are idle
```

**slot** สี่ตัวหมายความว่ารันได้พร้อมกันสี่คำขอ แต่ละคำขอมี context 262,144 token: playbook ระบุว่า llama-server "tries to fit the full model context with the ability to serve 4 concurrent requests" สำหรับเอเจนต์และงานเขียนโค้ด playbook แนะนำอย่างน้อย 32,768 token (`--ctx-size` / `-c`) และดีที่สุดคือ 100,000 ขึ้นไป ลด `-c` ลงถ้าเจอหน่วยความจำไม่พอ (out-of-memory)

ทดสอบแบบที่ playbook ทำ (บน Spark ด้วยพอร์ตของคอร์ส):

```bash
# on: spark
timeout 900 bash -c 'until curl -sf http://127.0.0.1:30080/health > /dev/null 2>&1; do sleep 5; done' || exit 1
curl -X POST http://127.0.0.1:30080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL",
    "messages": [{"role": "user", "content": "New York is a great city because..."}],
    "max_tokens": 100
  }'
```

`--host 0.0.0.0` ทำให้เซิร์ฟเวอร์เข้าถึงได้จาก LAN และ tailnet ของคุณ (`http://spark-a:30080/v1`) โดยไม่มีการยืนยันตัวตน จากแล็ปท็อปคุณทำ tunnel ได้ด้วย: `ssh -N -L 30080:localhost:30080 spark-a`

Lab 04 เรียกแบบเดียวกันจาก Python มันใช้ llama-server บน Spark เมื่อมันตอบ ไม่อย่างนั้นใช้ตัวที่รันบนแล็ปท็อปของคุณที่พอร์ต 30080 และถ้าไม่มีอีก ก็ใช้ Ollama ของแล็ปท็อป เพื่อลองทางเลือกตรงกลาง เรารัน `llama-server` จาก Homebrew บน Mac เครื่องนี้ด้วย GGUF ขนาดเล็กจาก Hugging Face (`qwen2.5-0.5b-instruct-q4_k_m.gguf` เปิดด้วย `--alias qwen2.5-0.5b-instruct-q4_k_m --port 30080 -c 4096`):

```bash
# on: laptop
.venv/bin/python week25/04_llama_cpp_lm_studio/labs/lab04_llama_server_lm_studio.py
```

**Expected output** (LAPTOP STAND-IN — โมเดล 0.5B บน Mac เครื่องนี้ ความเร็วเหล่านี้ไม่ได้บอกอะไรเกี่ยวกับ Spark)

```
▣ STEP 1 · llama-server — where does :30080 answer?
● http://localhost:30080/v1 · LAPTOP STAND-IN (llama-server on this laptop, not the Spark)
→ GET http://localhost:30080/health → {'status': 'ok'}
→ GET http://localhost:30080/v1/models → ['qwen2.5-0.5b-instruct-q4_k_m']

▣ STEP 2 · one chat completion, and the server's own `timings`
→ POST http://localhost:30080/v1/chat/completions  {"model": "qwen2.5-0.5b-instruct-q4_k_m", "max_tokens": 100}
· ANSWER  New York is a great city for many reasons, but one of the most significant and enduring is its vibrant cultural scene and rich history. The city boasts a diverse array of neighborhoods and cultural institutions, including museums, theaters, and galleries, all of which attract both locals and tourist
◆ usage {'completion_tokens': 100, 'prompt_tokens': 37, 'total_tokens': 137, 'prompt_tokens_details': {'cached_tokens': 0}} · finish_reason=length · 0.37 s wall
│ timings (llama.cpp only)  tokens  time    rate
│ ────────────────────────  ──────  ──────  ───────────
│ prefill                   37      29 ms   1265 tok/s
│ decode                    100     320 ms  309.4 tok/s

▣ STEP 3 · the same request, streamed (sparkkit.chat measures TTFT on the client)
→ POST http://localhost:30080/v1/chat/completions  model=qwen2.5-0.5b-instruct-q4_k_m  stream=True
· ANSWER  New York is a great city because it offers a unique blend of history, culture, and innovation. …
◆ LAPTOP STAND-IN (not Spark numbers) · qwen2.5-0.5b-instruct-q4_k_m · TTFT 11 ms · 100 tok in 0.4s · 241.2 tok/s
```

คำตอบมีรูปแบบของ OpenAI บวกบล็อก **`timings`** ที่มีเฉพาะใน llama.cpp: ความเร็ว prefill และ decode ตามที่เซิร์ฟเวอร์วัดเอง (playbook บอกว่า: "fields vary by llama.cpp version") `finish_reason=length` แปลว่ามันหยุดเพราะถึง `max_tokens` เหมือนในตัวอย่างคำตอบของ playbook ลองแบบ inline ได้: บล็อก ⚡ นี้จะส่งไปที่ llama-server บน Spark หรือไปที่ตัวแทนบนแล็ปท็อปเมื่อยังไม่ได้เชื่อมต่อ Spark

```spark
{"target": "llamacpp", "which": "a", "model": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL",
 "messages": [{"role": "user", "content": "New York is a great city because..."}],
 "max_tokens": 100}
```

✓ Checkpoint: `curl -sf http://127.0.0.1:30080/health` บน Spark ตอบ OK และคุณชี้ได้ว่าบล็อก `timings` อยู่ตรงไหนในคำตอบ

## 6 · LM Studio แบบ headless: llmster และ `lms` บนพอร์ต 1234

ปกติ LM Studio เป็นแอปเดสก์ท็อป [LM Studio playbook](https://build.nvidia.com/spark/lm-studio) ติดตั้ง **llmster** ซึ่งเป็น daemon แบบ headless ของมันลงบน Spark แล้วสั่งงานด้วย CLI `lms` มัน serve ทั้ง API ที่เข้ากันได้กับ OpenAI และ REST API ของ LM Studio เองบนพอร์ต **1234**

```bash
# on: spark
curl -fsSL https://lmstudio.ai/install.sh | bash      # then follow the printed hint to add lms to your PATH
lms server start --bind 0.0.0.0 --port 1234
lms get nvidia/nemotron-3-nano-omni                   # the playbook's example model
lms ls
lms load nvidia/nemotron-3-nano-omni
```

playbook ต้องการ "minimum 65 GB memory for model inference (70 GB or above recommended for the example model)" และพื้นที่ดิสก์เท่ากัน ให้ตรวจ `free -g` ก่อน หรือปลดเซิร์ฟเวอร์อื่นออก โมเดลที่ผ่านการตรวจสอบแล้วของ playbook:

| โมเดล | path ของ `lms` | หมายเหตุ |
|---|---|---|
| Nemotron 3 Nano Omni | `nvidia/nemotron-3-nano-omni` | ตัวอย่างของ playbook |
| Qwen3.6-35B-A3B | `qwen/qwen3.6-35b-a3b` | ตัวเลือก "agent-ready" ของ playbook; โหลดด้วย `--context-length 65536` สำหรับเซสชันเอเจนต์ที่ยาว |
| GPT-OSS-120B | `openai/gpt-oss-120b` | |

ตรวจจากแล็ปท็อป (คำสั่งของ playbook; หาที่อยู่ของ Spark ด้วย `hostname -I` บน Spark หรือใช้ชื่อใน tailnet):

```bash
# on: laptop
curl http://spark-a:1234/api/v1/models
```

Lab 04 step 4 ทำการตรวจแบบเดียวกันและคุยกับโมเดลตัวแรกที่โหลดไว้ ถ้าไม่มี LM Studio รันอยู่ที่ไหนเลย มันจะพิมพ์คำสั่งสำหรับเปิดให้:

**Expected output** (บันทึกจาก Mac เครื่องนี้ ซึ่งไม่ได้รัน LM Studio)

```
▣ STEP 4 · LM Studio (llmster) — where does :1234 answer?
○ no LM Studio server on the Spark or on localhost:1234
$ lms ls   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
(the models you downloaded with `lms get …`, with their sizes)
→ start it on the Spark (playbook step 3): lms server start --bind 0.0.0.0 --port 1234
```

**LM Link** (ไม่บังคับ มีอยู่ใน playbook) เชื่อม Spark กับแล็ปท็อปของคุณผ่านเครือข่ายที่เข้ารหัสแบบ end-to-end บนพื้นฐานของ Tailscale ทำให้โมเดลบน Spark ปรากฏที่ `localhost:1234` บนแล็ปท็อปโดยไม่ต้อง bind กับ `0.0.0.0` ต้องมีบัญชีที่ lmstudio.ai/link และใช้ฟรีในช่วง preview สำหรับผู้ใช้สูงสุด 2 คน

การเก็บกวาดตาม playbook: หยุดเซิร์ฟเวอร์ ลบ `~/.lmstudio/llmster` เพื่อถอนการติดตั้ง และลบ `~/.lmstudio/models/` เพื่อคืนพื้นที่ดิสก์

✓ Checkpoint: `curl http://spark-a:1234/api/v1/models` แสดงโมเดลที่คุณโหลดไว้ หรือคุณอธิบายได้ว่าทำไม playbook จึงต้องใช้ 65 GB สำหรับตัวอย่าง

## 7 · ใช้ engine ไหนเมื่อไร?

ตอนนี้คุณมีสามวิธีในการ serve GGUF และ Module 05 จะเพิ่ม vLLM เข้ามา นี่คือหลักคร่าว ๆ ของคอร์ส (ไม่ใช่การจัดอันดับของ NVIDIA):

| คุณต้องการ | เลือก | เพราะ |
|---|---|---|
| โมเดลที่รันได้ในสองนาที และมีหลายโมเดลให้ลอง | **Ollama** | คำสั่ง pull คำสั่งเดียว โหลด/ปลดอัตโนมัติ แล็ปท็อปกับ Spark ทำงานเหมือนกัน |
| GGUF เฉพาะตัวจาก Hugging Face พร้อมปุ่มปรับทุกตัว | **llama.cpp** | quant ใดก็ได้, `-c`, MTP speculative decoding, slot, `timings`; binary เดียวที่คุณคอมไพล์เอง |
| GUI แคตตาล็อกโมเดล หรือใช้จากระยะไกลโดยไม่ต้องเปิดพอร์ต | **LM Studio** | `lms` บวกลิงก์เข้ารหัสของ LM Link |
| ผู้ใช้หลายคนพร้อมกัน, weights แบบ FP8 / NVFP4, Spark สองเครื่อง | **vLLM** (Module 05) | continuous batching และ paged KV cache; tensor parallel ข้าม Spark |
| ความเร็วหยดสุดท้ายสำหรับโมเดลเดียว | **TensorRT-LLM / SGLang** (Module 06) | kernel ที่คอมไพล์แล้ว การแข่งขันระหว่าง engine |

ทุกตัวพูดภาษา `/v1/chat/completions` โค้ดฝั่ง client จาก Module 03 จึงไม่ต้องเปลี่ยน มีแค่ base URL และ model id ที่เปลี่ยน Module 08 จะวาง LiteLLM ไว้ด้านหน้าเพื่อให้แม้สองอย่างนั้นก็ไม่สำคัญอีกต่อไป

✓ Checkpoint: สำหรับแต่ละกรณีใช้งานของคุณเอง (ผู้ใช้แชตหนึ่งคน, เอเจนต์ที่มี tool, ทีมสิบคน) คุณบอกชื่อ engine และเหตุผลได้

## Labs — รันแล็บได้ที่นี่

**labs/lab01_build_llama_cpp.py** — ตรวจเครื่องมือ build แล้วคอมไพล์ llama.cpp สำหรับ CUDA และ serve บน :30080 (ทั้งสองอย่างต้องสั่งเปิดเอง)

**labs/lab02_gguf_inside.py** — อ่าน header ของ GGUF จริง นับชนิด quant และพิสูจน์ขนาดไฟล์จากขนาดบล็อกของ GGML

**labs/lab03_quant_tradeoffs.py** — quantize weights หนึ่งล้านตัว วัดความคลาดเคลื่อนต่อบิต และประเมินขนาดโมเดลของ playbook ในแต่ละ quant

**labs/lab04_llama_server_lm_studio.py** — เรียก llama-server และ LM Studio ผ่าน OpenAI API ของมัน และอ่าน `timings` ของ llama.cpp

Lab 01 รันแบบ LIVE บน Spark ของคุณหรือแบบ DRY ส่วน Lab 02 และ 03 รันบนแล็ปท็อป (lab 02 รันบน Spark ได้ด้วยเมื่อใส่ `--path`) Lab 04 เรียกเซิร์ฟเวอร์บน Spark ถ้าไม่มีก็ใช้เซิร์ฟเวอร์บนแล็ปท็อป (ติดป้ายไว้) และถ้ายังไม่มีอีกก็ใช้ Ollama ของแล็ปท็อป

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 04 — เลือก quant ที่ใส่ได้** เปิด `week25/04_llama_cpp_lm_studio/exercises/ex04_pick_a_quant.py` ในไฟล์มี `TODO` สามจุด:

1. `bits_per_weight(block_bytes, block_weights)`: ต้นทุนของบล็อก GGML หนึ่งชนิด
2. `gguf_gb(params_b, bpw)`: ขนาดของไฟล์ GGUF
3. `pick_quant(params_b, budget_gb)`: quant ที่ precision สูงที่สุดซึ่งใส่ในงบหน่วยความจำได้

```bash
# on: laptop
.venv/bin/python week25/04_llama_cpp_lm_studio/exercises/ex04_pick_a_quant.py
```

**Expected output** (เมื่อทำ TODO ครบทั้งสามจุด)

```
✓ bits_per_weight: Q8_0 8.5 · Q6_K 6.5625 · Q4_K 4.5 · Q2_K 2.625
✓ gguf_gb: Qwen3.6-35B-A3B at Q4_K_M ≈ 21.2 GB · 8B at BF16 = 16 GB
✓ pick_quant: 35B→Q8_0 · 70.6B→Q6_K · 405B→None · 8B→BF16

▣ your picker, applied: 1 Spark keeps 60 GB free for KV cache and a second model; 2 Sparks keep 10 GB
│ Qwen3.6-35B-A3B  on 1 Spark   budget  68 GB → Q8_0           37.2 GB
│ Llama 3.3 70B    on 1 Spark   budget  68 GB → Q6_K           57.9 GB
│ Llama 3.1 405B   on 1 Spark   budget  68 GB → does not fit    —
│ Llama 3.1 405B   on 2 Sparks  budget 246 GB → Q4_K_M        245.5 GB
```

<details><summary>คำใบ้ — เรียงตามคุณภาพ</summary>

"precision สูงที่สุด" หมายถึงบิตต่อ weight มากที่สุด เรียง `quants.items()` ตามค่า จากมากไปน้อย แล้วคืนชื่อแรกที่ `gguf_gb()` ไม่เกินงบ

</details>

<details><summary>ท้าทายเพิ่ม — รวม KV cache เข้าไปด้วย</summary>

import `kv_cache_gb` จาก sparkkit แล้วหัก KV cache ที่ context 32K ของ Llama 3.3 70B (80 layers, 8 KV heads, head dim 128) ออกจากงบก่อนเลือก 70B ยังได้ Q6_K บน Spark เครื่องเดียวอยู่ไหมเมื่อมีผู้ใช้พร้อมกัน 4 คน? นี่คือคำถามที่ slot และ `-c` ของ llama-server ตอบให้คุณ

</details>

✓ Checkpoint: บรรทัดตรวจทั้งสามเป็น ✓

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `cmake` ล้มเหลวด้วย "CUDA not found" | `export PATH=/usr/local/cuda/bin:$PATH` แล้วรัน CMake ใหม่จากไดเรกทอรี build ที่สะอาด (`rm -rf build` ก่อน) |
| build error ที่พูดถึงสถาปัตยกรรม GPU | ใช้ `-DCMAKE_CUDA_ARCHITECTURES=121a-real` ตามส่วนที่ 2 |
| `curl: (7) Failed to connect` บนพอร์ต 30080 | ยังโหลดอยู่ ล่มไปแล้ว หรือ host ผิด รอจนเห็น "listening" รัน `ss -tln \| grep 30080` และอ่าน `~/w25/logs/llama-server.log` หา error เรื่อง OOM หรือ path |
| ทำตาม playbook แล้วไม่มีอะไรตอบบน 30080 | playbook ใช้ `--port 30000` ในคอร์สนี้ให้ใช้ `--port 30080` หรือตั้ง `SPARK_URL_LLAMACPP=http://spark-a:30000/v1` |
| "CUDA out of memory" ตอน llama-server เริ่มทำงาน | ลด context (`-c 32768` หรือ `-c 4096` เพื่อทดสอบ) หรือเลือก quant ที่เล็กลงจาก repo เดียวกัน หยุดโมเดลของ Ollama หรือ LM Studio ที่ไม่ได้ใช้ |
| การดาวน์โหลด GGUF ค้าง | รันคำสั่ง `-hf` เดิมซ้ำ: มันจะดาวน์โหลดต่อจากไฟล์ที่ค้างไว้ |
| `key not found in model` ตอนโหลด blob ของ Ollama | Ollama แพ็กบางโมเดลด้วยวิธีของตัวเอง ให้ดาวน์โหลด GGUF จาก Hugging Face แทน (ส่วนที่ 1) |
| `lms: command not found` | `source ~/.bashrc` (หรือไฟล์ที่ตัวติดตั้งระบุ) ตามที่ playbook บอก |
| LM Studio ขึ้น "model not found" | `lms ls` เพื่อตรวจการดาวน์โหลด แล้ว `lms load <model>` |
| หน่วยความจำตึงทั้งที่โมเดลควรใส่ได้ | หมายเหตุเรื่อง UMA ของ playbook: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |

## Next — บทถัดไป

ไปต่อที่ [Lab 05 — vLLM: serving ที่ throughput สูง](../05_vllm/TUTORIAL.md): serve ผู้ใช้หลายคนพร้อมกันด้วย continuous batching เรียก tool ผ่าน vLLM และแบ่งโมเดลเดียวไปรันบน Spark สองเครื่อง
