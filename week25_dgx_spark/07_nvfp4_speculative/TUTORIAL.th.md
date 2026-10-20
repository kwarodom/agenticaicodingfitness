# ▶ Spark Lab 07 — NVFP4 quantization และ speculative decoding

> ส่วนหนึ่งของ Week 25 · DGX Spark: fine-tune, serve และสร้างเอเจนต์ที่รันใน sandbox คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง ทุกแล็บรันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (อ้างอิงคำต่อคำจาก playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง ข้อความที่ยกมาจาก playbook คงไว้เป็นภาษาอังกฤษตามต้นฉบับ พร้อมคำแปลกำกับ

**สิ่งที่คุณจะได้ลงมือทำ**
- แยกชิ้นส่วน NVFP4: ค่าแบบ E2M1 ขนาด 4 บิต, scale แบบ FP8 หนึ่งตัวต่อ 16 ค่า และเหตุผลที่ทำให้ได้ **4.5 บิต** ต่อ weight
- quantize ตัวเลข 65,536 ตัวด้วยห้าวิธีใน Python ล้วน ๆ แล้ววัดความคลาดเคลื่อนจริงของแต่ละฟอร์แมต
- quantize DeepSeek-R1-Distill-Llama-8B เป็น NVFP4 ด้วย **NVIDIA Model Optimizer** บน Spark ของคุณ แล้ว serve ด้วย `trtllm-serve`
- วัดขนาดและคุณภาพ: ขนาด checkpoint บนดิสก์ และคำถามที่ตรวจคำตอบได้หกข้อ ถามทั้งเซิร์ฟเวอร์ NVFP4 และ BF16
- หาคำตอบว่าทำไม **speculative decoding** จึงเร่งความเร็ว Spark ที่ติดคอขวดด้านหน่วยความจำ (memory-bound) ได้ ทำนาย speed-up จาก acceptance rate แล้วรัน EAGLE-3 และ Draft-Target ตาม playbook

**Time** ~55 นาที · **Difficulty** ระดับกลาง · **Hardware** DGX Spark 1 เครื่อง (หรือไม่มีเลยก็ได้: lab 01 และ 04 เป็นการคำนวณล้วน ส่วน lab 02–03 รันแบบ DRY)

**Playbook ทางการที่ครอบคลุม:** [NVFP4 Quantization](https://build.nvidia.com/spark/nvfp4-quantization) · [Speculative Decoding](https://build.nvidia.com/spark/speculative-decoding)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| ทำ Module 01 เสร็จแล้ว | `ssh -o BatchMode=yes spark-a true` | lab 02–03 สั่งงาน Spark ผ่าน SSH |
| Docker โดยไม่ต้อง sudo บน Spark | `docker ps` บน Spark | ทุกขั้นตอนในบทนี้เป็น container |
| ดิสก์ว่าง ~60 GB (~160 GB ถ้าทำส่วนที่ 6) | `df -h ~` บน Spark | โมเดล 8B (~16 GB), สำเนา NVFP4 ของมัน และ image ของ TensorRT-LLM ส่วนที่ 6 เพิ่ม gpt-oss-120b (~62 GB ตามการคำนวณแบบ 4 บิต) หรือ Llama 3.3 70B FP4 (~40 GB) |
| ล็อกอิน Hugging Face **บน Spark** | `hf auth login` ครั้งเดียว บน Spark | โมเดล Llama FP4 ในส่วนที่ 6 เป็นแบบ gated แล็บไม่เคยเห็นโทเค็นของคุณ |
| สิทธิ์ NGC สำหรับ image บน `nvcr.io` | `docker login nvcr.io` ครั้งเดียว บน Spark ถ้า pull แล้วขึ้น *unauthorized* | container ของ TensorRT-LLM |

```bash
# on: spark
docker ps
df -h ~
hf auth whoami
```

> 💡 Lab 01 และ 04 ไม่ต้องใช้ Spark เลย ถ้าคุณเรียนตามในโหมด DRY ให้ทำสองแล็บนี้ก่อน: มันอธิบายทุกตัวเลขที่แล็บบน Spark พิมพ์ออกมา

✓ Checkpoint: `docker ps` ใช้ได้บน Spark โดยไม่ต้อง `sudo` และ `df -h ~` แสดงพื้นที่พอสำหรับส่วนที่คุณวางแผนจะรัน

## 1 · ทำไมต้อง 4 บิต และทำไม NVFP4 ไม่ใช่ INT4

Module 01 แสดงให้เห็นว่า Spark สร้างแต่ละ token ด้วยการอ่านทุก weight ที่ active จากหน่วยความจำหนึ่งครั้ง ที่ 273 GB/s ไบต์ต่อ weight ยิ่งน้อย ก็ยิ่งได้ token ต่อวินาทีมากขึ้น และใส่โมเดลที่ใหญ่ขึ้นใน Spark เครื่องเดียวได้ playbook สรุปคำมั่นของ NVFP4 ไว้ว่า *"Cut memory use ~3.5× vs FP16 and ~1.8× vs FP8"* (ลดการใช้หน่วยความจำ ~3.5 เท่าเมื่อเทียบกับ FP16 และ ~1.8 เท่าเมื่อเทียบกับ FP8) โดยยังคง *"accuracy close to FP8 (usually <1% loss)"* (ความแม่นยำใกล้เคียง FP8 ปกติสูญเสียไม่ถึง 1%)

playbook อธิบาย NVFP4 ว่าเป็น *"a 4-bit floating-point format for NVIDIA Blackwell GPUs"* (ฟอร์แมตทศนิยมลอยตัว 4 บิตสำหรับ GPU Blackwell) ซึ่ง *"unlike uniform INT4 quantization, … keeps floating-point semantics with a shared exponent and a compact mantissa"* (ต่างจาก INT4 แบบ uniform ตรงที่ยังคงความหมายแบบ floating-point ไว้ ด้วย exponent ที่ใช้ร่วมกันและ mantissa ขนาดกะทัดรัด) ในรายละเอียด (โครงสร้างนี้มาจากเอกสารอธิบายฟอร์แมต NVFP4 ของ NVIDIA ไม่ใช่จาก playbook):

| ชิ้นส่วน | ฟอร์แมต | ใช้ร่วมกันโดย | ต้นทุนต่อค่า |
|---|---|---|---|
| แต่ละค่า | **E2M1**: sign 1 บิต, exponent 2 บิต, mantissa 1 บิต | ตัวมันเอง | 4 บิต |
| block scale | **FP8 E4M3** | 16 ค่า | 8 ÷ 16 = 0.5 บิต |
| tensor scale | FP32 | ทั้ง tensor | ≈ 0 |
| **รวม** | | | **4.5 บิต** |

ตัวเลข 4.5 นี้คือเหตุผลที่ `sparkkit.BITS["nvfp4"]` มีค่า 4.5 และทำไม 16 ÷ 4.5 = 3.56× ("~3.5×" ของ playbook) และ 8 ÷ 4.5 = 1.78× ("~1.8×")

ค่าแบบ E2M1 เก็บขนาด (magnitude) ได้เพียงแปดค่า: **0, 0.5, 1, 1.5, 2, 3, 4, 6** การ quantize หนึ่ง block ทำงานแบบนี้:

```text
block scale  = largest |value| in the 16  ÷ 6        → the largest value lands on ±6
stored scale = that, rounded to FP8 (E4M3)            → 8 bits per block
code         = nearest E2M1 value to (value ÷ scale)  → 4 bits per value
decoded      = code × scale
```

มีการตัดสินใจเชิงออกแบบสองข้อที่สำคัญ **block เล็ก (16):** ค่าผิดปกติ (outlier) หนึ่งตัวจะทำให้ความละเอียดเสียเฉพาะเพื่อนบ้าน 15 ตัวใน block ของมันเท่านั้น **scale แบบ FP8** แทนที่จะเป็นกำลังของสอง: MXFP4 ซึ่งเป็นฟอร์แมตทศนิยม 4 บิตอีกตัว ใช้ block ขนาด 32 ค่าและ scale แบบกำลังของสอง scale ของมันจึงพอดีกับแต่ละ block น้อยกว่า Lab 01 วัดผลของทั้งสองข้อนี้

```bash
# on: laptop
.venv/bin/python week25/07_nvfp4_speculative/labs/lab01_nvfp4_simulator.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้: Python ล้วนที่ใช้ seed คงที่ ทุกเครื่องจึงพิมพ์ตัวเลขเดียวกัน)

```
▣ STEP 1 · the E2M1 grid: every magnitude a 4-bit NVFP4 value can hold
│ 0  0.5  1  1.5  2  3  4  6   (and their negatives: 15 distinct values, ±0 share a code)

▣ STEP 2 · one 16-value block by hand
│ amax of block = 0.0913  →  block scale = amax / 6 = 0.015217
│ per-tensor FP32 scale = tensor amax 0.25 / (6 × 448) = 9.301e-05
│ block scale in FP8 units = 163.61 → rounded to E4M3 = 160  (FP8 has 3 mantissa bits)
│ value    ÷ scale  E2M1 code  decoded  abs error
│ ───────  ───────  ─────────  ───────  ─────────
│ -0.0051  -0.343   -0.5       -0.0074  0.0023
│ +0.0102  +0.685   +0.5       +0.0074  0.0028
│ -0.0186  -1.250   -1         -0.0149  0.0037
│ +0.0913  +6.135   +6         +0.0893  0.0020
│ +0.0222  +1.492   +1.5       +0.0223  0.0001
…
▣ STEP 3 · 65,536 weight-like numbers, five formats
│ format                     bits/value  rel. err (gauss)  rel. err (outliers)  SQNR gauss  SQNR outliers
│ ─────────────────────────  ──────────  ────────────────  ───────────────────  ──────────  ─────────────
│ FP8 E4M3 (per-tensor)      8.000        2.68 %            2.69 %               31.4 dB     31.4 dB
│ NVFP4 (16-blk FP8 scale)   4.500        9.51 %            9.43 %               20.4 dB     20.5 dB
│ MXFP4 (32-blk 2^k scale)   4.250       11.44 %           14.62 %               18.8 dB     16.7 dB
│ INT4 group-128 (FP16 sc.)  4.125       11.79 %           26.48 %               18.6 dB     11.5 dB
│ INT4 per-tensor            4.000       20.17 %           76.27 %               13.9 dB      2.4 dB

▣ STEP 4 · what 4.5 bits buys: an 8B model's weights
│ BF16 (no quantization)      16.1 GB  ████████████████████████████
│ FP8 E4M3 (per-tensor)        8.0 GB  ██████████████░░░░░░░░░░░░░░
│ NVFP4 (16-blk FP8 scale)     4.5 GB  ████████░░░░░░░░░░░░░░░░░░░░
│ BF16 ÷ NVFP4 = 3.56×   FP8 ÷ NVFP4 = 1.78×   (the playbook says ~3.5× and ~1.8×)
```

ดูที่คอลัมน์ "outliers" ข้อมูลทดสอบคือ Gaussian ชุดเดิมที่มี 0.5 % ของค่าถูกคูณด้วย 12 ซึ่งใกล้เคียงกับหน้าตาของแถว weight จริง ความคลาดเคลื่อนของ NVFP4 ไม่ขยับเลย INT4 ที่ใช้ scale หนึ่งตัวต่อ 128 ค่าเสียไป 7 dB และ INT4 ที่ใช้ scale ตัวเดียวทั้ง tensor พังไปเลย SQNR (signal-to-quantization-noise ratio) เป็นตัวเลขระดับ layer: ประมาณ 6 dB เท่ากับความละเอียดหนึ่งบิต มันไม่ใช่ความแม่นยำของโมเดล ส่วนที่ 4 จะวัดตัวโมเดล

> 💡 playbook เสริมว่า *"Blackwell Tensor Cores support mixed-precision execution across FP16, FP8, and FP4, so models can use FP4 for weights and activations while accumulating in higher precision"* (Tensor Core ของ Blackwell รองรับการคำนวณแบบ mixed-precision ทั้ง FP16, FP8 และ FP4 โมเดลจึงใช้ FP4 กับ weights และ activations ได้ ขณะที่สะสมผลรวมด้วยความละเอียดที่สูงกว่า) นี่คือตัวเลข 1 PFLOP ที่ FP4 จาก Module 01: GB10 คูณเลข FP4 ได้โดยตรง NVFP4 จึงประหยัดทั้งหน่วยความจำ *และ* การประมวลผล

✓ Checkpoint: คุณอธิบายได้ว่า 0.5 บิตที่เพิ่มมาเกิดจากอะไร และทำไมความคลาดเคลื่อนของ NVFP4 จึงคงที่เมื่อมี outlier ขณะที่ INT4 แบบ per-tensor ไม่คงที่

## 2 · quantize ด้วย Model Optimizer บน Spark ของคุณ

playbook quantize **`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`** ภายใน container ของ TensorRT-LLM จาก NVIDIA ตัว container จะ clone **NVIDIA Model Optimizer** ที่ tag `0.35.0` แล้วรันสคริปต์ post-training quantization (PTQ) ของมัน สคริปต์จะ calibrate ด้วยข้อความตัวอย่างเพื่อวัดช่วงค่าของแต่ละ layer เลือก scale แล้ว export ออกมาเป็น checkpoint แบบ Hugging Face

| การตั้งค่า | ค่าสำหรับ DGX Spark (playbook, Step 4) |
|---|---|
| Container | `nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev` |
| GPU flag | `--gpus all` |
| Model Optimizer | `NVIDIA/Model-Optimizer` @ `0.35.0` |
| โฟลเดอร์ผลลัพธ์ | `./output_models/saved_models_DeepSeek-R1-Distill-Llama-8B_nvfp4_hf/` |

นี่คือคำสั่ง Step 5 ของ playbook สำหรับ DGX Spark ตรงตามที่เผยแพร่ ให้รันจากไดเรกทอรีทำงานที่ว่างเปล่า:

```bash
# on: spark
mkdir -p ~/w25/nvfp4/output_models && cd ~/w25/nvfp4
docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "./output_models:/workspace/output_models" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN=$HF_TOKEN \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c "
    git clone -b 0.35.0 --single-branch https://github.com/NVIDIA/Model-Optimizer.git /app/Model-Optimizer && \
    cd /app/Model-Optimizer && pip install -e '.[dev]' && \
    export ROOT_SAVE_PATH='/workspace/output_models' && \
    /app/Model-Optimizer/examples/llm_ptq/scripts/huggingface_example.sh \
    --model 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B' \
    --quant nvfp4 \
    --tp 1 \
    --export_fmt hf
  "
```

ใช้เวลา 45–90 นาที (ตามที่ playbook ประมาณไว้) นานกว่าที่ Lab Runner ยอมให้แล็บแบบ foreground ทำงาน (900 s) lab 02 จึงเปิดคำสั่งเดียวกันในเบื้องหลัง (background) โดยมีการเปลี่ยนแปลงของคอร์สสามจุด ซึ่งพิมพ์ไว้ในแล็บ: ครอบคำสั่งด้วย `nohup … > ~/w25/logs/nvfp4_quant.log &`, ไม่มี `-it` (งานเบื้องหลังไม่มีเทอร์มินัล) และใช้ `-e HF_TOKEN` โดยไม่มี `=$HF_TOKEN` ซึ่งจะส่งตัวแปรเข้าไปเฉพาะเมื่อมีการตั้งค่าไว้ โมเดลนี้เป็นแบบสาธารณะ สำหรับโมเดลแบบ gated โทเค็นที่ `hf auth login` เก็บไว้ใน `~/.cache/huggingface` จะเข้าถึง container ได้ผ่าน cache ที่ mount ไว้

```bash
# on: laptop
.venv/bin/python week25/07_nvfp4_speculative/labs/lab02_quantize_nvfp4.py --start   # launch (LIVE only)
.venv/bin/python week25/07_nvfp4_speculative/labs/lab02_quantize_nvfp4.py           # status, any time
```

**Expected output** (โหมด DRY บันทึกจาก Mac เครื่องนี้ ผลลัพธ์เป็นรูปแบบ EXAMPLE ไม่ใช่ของ Spark)

```
▣ STEP 2 · the quantization job (the playbook's Step 5, run in the background)
$ mkdir -p ~/w25/nvfp4/output_models ~/w25/logs && cd ~/w25/nvfp4 && \   [DRY]
  nohup docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  …
  echo "started quantization job (pid $!) → ~/w25/logs/nvfp4_quant.log"
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
started quantization job (pid <pid>) → ~/w25/logs/nvfp4_quant.log

▣ STEP 3 · status — the log tail and the exported checkpoint
$ ls ~/w25/nvfp4/output_models/saved_models_DeepSeek-R1-Distill-Llama-8B_nvfp4_hf/ 2>/dev/null || echo 'not exported yet'   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
config.json
generation_config.json
hf_quant_config.json
model-00001-of-00002.safetensors
…
```

หมายเหตุของ playbook สำหรับขั้นตอนนี้ ควรรู้ไว้ก่อนตกใจ: *"You can safely ignore `No module named 'mpi4py'`"* (เพิกเฉยข้อความนี้ได้อย่างปลอดภัย) และ *"`pynvml.NVMLError_NotSupported: Not Supported` can appear … and does not affect results."* (อาจปรากฏขึ้น และไม่มีผลต่อผลลัพธ์) ตรวจสอบด้วย Step 7 ของ playbook:

```bash
# on: spark
cd ~/w25/nvfp4
ls -la ./output_models/
find ./output_models/ \( -name "*.bin" -o -name "*.safetensors" -o -name "*.json" -o -name "*.jinja" \)
```

`hf_quant_config.json` คือไฟล์ที่บอก serving engine ว่า checkpoint นี้สร้างโดย Model Optimizer และถูก quantize แบบไหน Lab 02 พิมพ์ไฟล์นี้ออกมา

> 💡 **อีกรูปแบบหนึ่งของ playbook** repository ยังมี playbook คู่แฝด (`nvidia/nvfp4-quantization/`) ที่ quantize โมเดล MoE `Qwen/Qwen3.6-35B-A3B` ภายใน container **vLLM** ของ NGC ด้วย **recipe** ของ Model Optimizer (`hf_ptq.py --recipe …`) มีให้เลือกสองแบบ: `w4a16_nvfp4-fp8_attn-kv_fp8_cast` (เป็น NVFP4 เฉพาะ *weights* ส่วน activations ยังเป็น 16 บิต *"recommended on DGX Spark"* (แนะนำบน DGX Spark) สำหรับการใช้งานแบบโต้ตอบ) และ `nvfp4_experts_only-kv_fp8_cast` (W4A4: ทั้ง weights *และ* activations เป็น NVFP4 สำหรับ *"higher-concurrency, compute-bound serving"* (การ serve ที่มีผู้ใช้พร้อมกันมากและติดคอขวดที่การประมวลผล)) ฟอร์แมตเดียวกัน แต่แลกกันคนละแบบ: ที่ผู้ใช้หนึ่งคน Spark ติดคอขวดที่หน่วยความจำ weights แบบ 4 บิตจึงเป็นสิ่งที่มีผล

✓ Checkpoint: `ls` ที่โฟลเดอร์ผลลัพธ์แสดงไฟล์ `.safetensors` และ `hf_quant_config.json` หรือในโหมด DRY คุณบอกได้ว่า lab 02 เปลี่ยนอะไรสามอย่างจากคำสั่งของ playbook และเพราะอะไร

## 3 · serve checkpoint แบบ NVFP4

playbook serve ผลลัพธ์ด้วย **`trtllm-serve`** จาก container เดียวกัน นี่คือคำสั่ง Step 8 (DGX Spark):

```bash
# on: spark
export MODEL_PATH="$HOME/w25/nvfp4/output_models/saved_models_DeepSeek-R1-Distill-Llama-8B_nvfp4_hf/"
docker run \
  -e HF_TOKEN=$HF_TOKEN \
  -v "$MODEL_PATH:/workspace/model" \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  trtllm-serve /workspace/model \
    --backend pytorch \
    --max_batch_size 4 \
    --port 8000
```

**การเปลี่ยนแปลงของคอร์ส: ใช้พอร์ต 8355 ไม่ใช่ 8000** พอร์ต 8000 เป็นของ vLLM จาก Module 05 playbook ของ TensorRT-LLM ใช้ 8355 และ `sparkkit.PORTS["trtllm"]` ก็เช่นกัน Lab 03 รันคำสั่งข้างบนด้วย `--port 8355` ตั้งชื่อ container (`--name w25-trtllm` เพื่อให้ `--stop` หาเจอ) และใช้ `nohup` พร้อม log จากนั้นทดสอบด้วยคำขอของ playbook บนพอร์ตของคอร์ส:

```bash
# on: spark
curl -X POST http://localhost:8355/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [{"role": "user", "content": "What is artificial intelligence?"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false
  }'
```

หรือจาก Lab Runner (ส่งไปที่ `trtllm` บน Spark A, :8355; ถ้าไม่มี Spark ตัว laptop stand-in จะตอบแทนและบอกไว้ชัดเจน):

```spark
{"target": "trtllm", "which": "a", "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "messages": [{"role": "user", "content": "In two sentences: what does NVFP4 store for every block of 16 weights?"}], "max_tokens": 150}
```

DeepSeek-R1-Distill เป็นโมเดลแบบ reasoning: มันเขียนส่วน `<think>` ก่อนตอบ จึงควรเผื่อ token ไว้ (`max_tokens` หลายร้อย) เมื่อคุณต้องการคำตอบเต็ม

Lab 03 เปิดเซิร์ฟเวอร์และจับเวลาคำขอหนึ่งครั้ง:

```bash
# on: laptop
.venv/bin/python week25/07_nvfp4_speculative/labs/lab03_serve_and_measure.py --start nvfp4
.venv/bin/python week25/07_nvfp4_speculative/labs/lab03_serve_and_measure.py --label nvfp4
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ที่ไม่มี Spark: เวลาที่วัดได้เป็นของ Ollama บนแล็ปท็อปเครื่องนี้ ติดป้ายไว้ และไม่ได้บอกอะไรเกี่ยวกับ Spark)

```
▣ STEP 2 · the endpoint — GET /v1/models on :8355
│ (no Spark host) → ○ down
◆ No TensorRT-LLM on the Spark, so the measurement below uses Ollama on THIS laptop (gemma3:4b). It shows how the measurement works; its speed says nothing about a Spark.

▣ STEP 3 · one timed request — the speculative-decoding playbook's prompt, temperature 0
→ POST http://localhost:11434/v1/completions · model=gemma3:4b · max_tokens=120 · LAPTOP STAND-IN (not Spark numbers)
· ANSWER  Here's how to solve the problem step-by-step:
          **Step 1: Calculate the initial speed**
          …
◆ LAPTOP STAND-IN (not Spark numbers) · gemma3:4b · 120 tokens in 12.5 s · 9.6 tok/s (end to end, incl. prefill)

▣ STEP 4 · your measurements so far (grouped by where they ran — never compare across groups)
│ where   label            model      tokens  seconds  tok/s  when
│ ──────  ───────────────  ─────────  ──────  ───────  ─────  ────────────────
│ laptop  laptop-stand-in  gemma3:4b  120     12.5     9.6    2026-09-29 11:37
```

บน Spark ให้เทียบ tok/s ที่วัดได้กับเพดานจาก Module 01: โมเดล 8B ที่ 4.5 บิตอ่าน ~4.5 GB ต่อ token ดังนั้น 273 GB/s ÷ 4.5 GB ≈ 60 tok/s คือขอบบนสำหรับหนึ่งสาย ตัวเลขของคุณจะต่ำกว่านั้น ส่วนต่างคือ overhead และ embeddings ที่ยังเป็น 16 บิต

✓ Checkpoint: `curl http://localhost:8355/v1/models` บน Spark แสดงรายชื่อโมเดล และ lab 03 พิมพ์ tok/s แบบ LIVE ในโหมด DRY lab 03 จะพิมพ์ตัวเลขจาก laptop stand-in ที่ติดป้ายไว้ชัดเจนแทน

## 4 · วัดผล: ขนาดบนดิสก์ และคุณภาพ

โมเดลที่ถูก quantize จะมีประโยชน์ก็ต่อเมื่อมันเล็กลง **และ** ยังตอบถูก วัดทั้งสองอย่าง

**ขนาด** Step 4 ของ lab 02 รัน `du -sh` กับโฟลเดอร์ NVFP4 และกับต้นฉบับ BF16 ใน cache ของ Hugging Face แล้วพิมพ์ผลการคำนวณไว้ข้าง ๆ:

```
│ arithmetic: BF16 weights 16.1 GB · NVFP4 weights 4.5 GB (every weight at 4.5 bits)
│ the embeddings (128,256 × 4,096 ≈ 0.53 B params) and lm_head usually stay BF16, so expect the real checkpoint above 4.5 GB
```

(พิมพ์โดย lab 02 บน Mac เครื่องนี้: เป็นการคำนวณ ไม่ใช่การวัด) อัตราส่วนที่คุณวัดได้จะต่ำกว่า 3.56× เพราะเครื่องมือ quantize มักเก็บ layer ที่อ่อนไหวไว้ที่ 16 บิต (output head และบ่อยครั้งรวมถึง embeddings) `hf_quant_config.json` ระบุว่ามีอะไรถูกยกเว้นบ้าง

**คุณภาพ** SQNR ของ lab 01 เป็นค่าต่อ layer โมเดลยังอาจตอบผิดได้ playbook พูดตรง ๆ ว่า *"Quantization can change model quality. Run evaluations for your use case before deploying."* (quantization อาจเปลี่ยนคุณภาพของโมเดล ให้รัน evaluation สำหรับกรณีใช้งานของคุณก่อน deploy) โหมด `--quality` ของ lab 03 คือเวอร์ชันที่เล็กที่สุดที่ยังตรงไปตรงมา: คำถามหกข้อ แต่ละข้อมีคำตอบที่ตรวจได้หนึ่งคำตอบ ที่ temperature 0 รันกับเซิร์ฟเวอร์ NVFP4 แล้วรันกับต้นฉบับ BF16 ที่ serve แบบเดียวกัน (รูปแบบของคอร์ส: ใช้ id บน Hugging Face แทนโฟลเดอร์ที่ export):

```bash
# on: laptop
.venv/bin/python week25/07_nvfp4_speculative/labs/lab03_serve_and_measure.py --quality --label nvfp4
.venv/bin/python week25/07_nvfp4_speculative/labs/lab03_serve_and_measure.py --start bf16
.venv/bin/python week25/07_nvfp4_speculative/labs/lab03_serve_and_measure.py --quality --label bf16
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ที่ไม่มี Spark: LAPTOP STAND-IN รันคำถามหกข้อชุดเดียวกันบน gemma3:4b เพื่อแสดงการทำงานของชุดทดสอบ)

```
▣ STEP 3 · quality — six questions with one right answer each, temperature 0
│    question                                              answer (thinking removed)
│ ─  ────────────────────────────────────────────────────  ─────────────────────────
│ ✓  What is 17 × 23? Answer with the number only.         391
│ ✓  What is the capital of Australia? Answer in one word  Canberra
│ ✓  Which planet is known as the Red Planet? Answer in o  Mars
│ ✓  What is the chemical symbol for gold? Answer with th  Au
│ ✓  A train travels 180 km in 3 hours. What is its speed  60
│ ✕  Spell the word 'spark' backwards. Answer with the wo  krasp
◆ LAPTOP STAND-IN (not the Spark's model) · gemma3:4b · score 5/6
```

(โมเดล 4B สะกด "kraps" ผิด เป็นผลจริงจาก Mac เครื่องนี้ งานระดับตัวอักษรยากสำหรับทุกโมเดลที่ใช้ tokenizer) คำถามหกข้อจะจับ quantization ที่ **พัง** ได้ (scale ผิด หรือ layer ที่ควรยกเว้นแต่ไม่ได้ยกเว้น) แต่จะไม่เห็นความแม่นยำที่ลดลง 1 % ถ้าต้องการวัดระดับนั้น ให้รันชุด evaluation จริงกับงานของคุณเอง Module 13 และ 20 สร้างชุดนั้น

| สิ่งที่คุณเปรียบเทียบ | เครื่องมือ | จับอะไรได้ |
|---|---|---|
| ความคลาดเคลื่อนต่อ layer | lab 01 (SQNR) | การเลือกฟอร์แมตที่ไม่ดี |
| จำนวนไบต์บนดิสก์ | lab 02 (`du -sh`) | layer ที่ไม่ได้ถูก quantize |
| คำถามตายตัวหกข้อ | lab 03 `--quality` | checkpoint ที่พัง |
| ชุด eval ของงานคุณ | Module 13 | การเปลี่ยนแปลงความแม่นยำที่สำคัญกับคุณ |

✓ Checkpoint: คุณมีอัตราส่วนจาก `du -sh` และคะแนน `--quality` สองชุด (nvfp4, bf16) จาก Spark เครื่องเดียวกัน หรือคุณอธิบายได้ว่าทำไมคะแนนของ laptop stand-in จึงไม่ใช่ผลลัพธ์เกี่ยวกับ NVFP4

## 5 · Speculative decoding: ทำไมการเดาล่วงหน้าจึงแทบไม่มีต้นทุนบน Spark

สรุปของ playbook speculative decoding: *"a small, fast draft path … propose[s] several tokens ahead, then … the larger target model verif[ies] or correct[s] them in parallel."* (เส้นทางร่างที่เล็กและเร็วเสนอ token ล่วงหน้าหลายตัว จากนั้นโมเดลเป้าหมายที่ใหญ่กว่าจะตรวจหรือแก้ไขพร้อมกันแบบขนาน) ทำไมการตรวจห้า token จึงใช้ต้นทุนน้อยกว่าการสร้างห้า token?

เพราะการ decode แบบสายเดียวบน Spark นั้น **ติดคอขวดที่หน่วยความจำ (memory-bound)** forward pass หนึ่งครั้งอ่านทุก weight จากหน่วยความจำหนึ่งรอบ ไม่ว่าจะให้คะแนนกี่ token ก็ตาม Lab 04 ใส่ตัวเลขให้ดูกับ Llama 3.3 70B ที่ NVFP4:

```bash
# on: laptop
.venv/bin/python week25/07_nvfp4_speculative/labs/lab04_speculative_calculator.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้: เป็นการคำนวณและการจำลองที่ใช้ seed คงที่ ได้ผลเหมือนกันทุกที่)

```
▣ STEP 1 · why checking 6 tokens costs about the same as generating 1 (Llama 3.3 70B, NVFP4, one Spark)
│ tokens in one pass  weights read  time to read  math          time at peak  bound by
│ ──────────────────  ────────────  ────────────  ────────────  ────────────  ────────
│ 1 token             39.7 GB        145.5 ms       0.14 TFLOP    0.14 ms     memory
│ 6 tokens            39.7 GB        145.5 ms       0.85 TFLOP    0.85 ms     memory
│ 32 tokens           39.7 GB        145.5 ms       4.52 TFLOP    4.52 ms     memory

▣ STEP 2 · tokens per target step: E = (1 − α^(k+1)) / (1 − α)
│ acceptance  k=1   k=2   k=3   k=4   k=5   k=8
│ ──────────  ────  ────  ────  ────  ────  ────
│ α = 0.5     1.50  1.75  1.88  1.94  1.97  2.00
│ α = 0.6     1.60  1.96  2.18  2.31  2.38  2.47
│ α = 0.7     1.70  2.19  2.53  2.77  2.94  3.20
│ α = 0.8     1.80  2.44  2.95  3.36  3.69  4.33
│ α = 0.9     1.90  2.71  3.44  4.10  4.69  6.13

▣ STEP 3 · check the formula: simulate 20,000 steps per cell (seed 25)
│ case       formula  simulated  difference
│ ─────────  ───────  ─────────  ──────────
│ α=0.6 k=3  2.176    2.168      0.37 %
│ α=0.7 k=4  2.773    2.771      0.06 %
│ α=0.8 k=5  3.689    3.708      0.50 %
│ α=0.9 k=5  4.686    4.703      0.36 %
```

สูตรนี้มาจากไหน draft เสนอ **k** token ตัว target ตรวจจากซ้ายไปขวา และยอมรับแต่ละตัวด้วยความน่าจะเป็น **α** (*acceptance rate*) โดยหยุดที่ตัวแรกที่ไม่ผ่าน จากนั้นมันจะเพิ่ม token ของตัวเองอีกหนึ่งตัวเสมอ: เป็นตัวแก้ไข หรือเป็น token โบนัสถ้าผ่านครบทั้ง k ตัว จำนวน token ที่คาดหวังต่อ target pass หนึ่งครั้งคือ

```text
E = 1 + α + α² + … + α^k = (1 − α^(k+1)) / (1 − α)
```

แบบจำลองนี้สมมติว่าการยอมรับแต่ละ token เป็นอิสระต่อกัน ซึ่งข้อความจริงเป็นเช่นนั้นเพียงคร่าว ๆ แต่ก็ยังทำนายได้ดีพอสำหรับการเลือก `k`

**draft ไม่ได้ฟรี** ทุก token ที่ร่างต้องใช้ draft pass หนึ่งครั้ง ถ้า draft pass หนึ่งครั้งมีต้นทุนเท่ากับ **c** target pass แล้ว speed-up คือ `E ÷ (1 + k·c)`:

```
│ Draft-Target: 8B FP4 drafts for 70B FP4 · c = 0.113 · α = 0.7 · playbook max_draft_len = 4
│   k=2  E=2.19  speed-up 1.79×  ██████████████░░░░░░░░░░░░░░
│   k=4  E=2.77  speed-up 1.91×  ███████████████░░░░░░░░░░░░░  ← best · playbook
│   k=8  E=3.20  speed-up 1.68×  █████████████░░░░░░░░░░░░░░░

│ EAGLE-3 head on gpt-oss-120b (assumed c) · c = 0.050 · α = 0.7 · playbook max_draft_len = 5
│   k=5  E=2.94  speed-up 2.35×  ███████████████████░░░░░░░░░  ← playbook
│   k=6  E=3.06  speed-up 2.35×  ███████████████████░░░░░░░░░  ← best
```

ค่า `c` ของ Draft-Target มาจากการคำนวณ (draft ขนาด 8B อ่าน 8 ÷ 70.6 ของจำนวนไบต์ของ target) ส่วน `c = 0.05` ของ EAGLE-3 เป็น **สมมติฐาน** เพื่อใช้อธิบาย เปลี่ยนได้ด้วย `--cost` และเปลี่ยน acceptance rate ด้วย `--alpha`

| | EAGLE-3 | Draft-Target |
|---|---|---|
| Draft | head ขนาดเล็กที่เทรนบน hidden state ของ target เอง | โมเดลเล็กแยกอีกตัวที่ใช้ tokenizer เดียวกัน |
| คู่ใน playbook | `openai/gpt-oss-120b` + `nvidia/gpt-oss-120b-Eagle3-long-context` | `nvidia/Llama-3.3-70B-Instruct-FP4` + `nvidia/Llama-3.1-8B-Instruct-FP4` |
| `max_draft_len` | 5 | 4 |
| จุดขายใน playbook | *"features fused from multiple layers improve draft-token acceptance"* (feature ที่รวมจากหลาย layer ช่วยเพิ่มการยอมรับ draft token) | *"an 8B draft model accelerates a 70B target model"* (draft model ขนาด 8B เร่งความเร็ว target model ขนาด 70B) |

> 💡 speculation ช่วยได้มากที่สุดที่ **batch 1** นี่คือเหตุผลที่ config ทั้งสองของ playbook ตั้ง `max_batch_size: 1` เมื่อมีผู้ใช้หลายคน batch จะใช้พลังประมวลผลที่ว่างอยู่ซึ่ง speculation จะได้ใช้ไปจนเต็มแล้ว

✓ Checkpoint: จากตารางใน Step 2 คุณบอกได้ว่า α = 0.8, k = 4 ให้กี่ token ต่อ pass (3.36) และทำไมการเพิ่ม k เกิน ~6 แทบไม่ช่วยที่ α = 0.7

## 6 · รัน EAGLE-3 และ Draft-Target บน Spark ของคุณ

ทั้งสองตัวเลือกใช้ `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc12` และไฟล์ `extra-llm-api-config.yml` ของ TensorRT-LLM ที่มีบล็อก `speculative_config` playbook ระบุว่า *"Run one option at a time on a free port."* (รันทีละตัวเลือกบนพอร์ตที่ว่าง) นี่คือ Option 1 ตรงตามต้นฉบับ:

```bash
# on: spark
export TRTLLM_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc12"
docker run \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface/:/root/.cache/huggingface/" \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  "$TRTLLM_IMAGE" \
  bash -c '
    hf download openai/gpt-oss-120b && \
    hf download nvidia/gpt-oss-120b-Eagle3-long-context \
        --local-dir /opt/gpt-oss-120b-Eagle3/ && \
    cat > /tmp/extra-llm-api-config.yml <<EOF
enable_attention_dp: false
disable_overlap_scheduler: false
enable_autotuner: false
cuda_graph_config:
    max_batch_size: 1
speculative_config:
    decoding_type: Eagle
    max_draft_len: 5
    speculative_model_dir: /opt/gpt-oss-120b-Eagle3/

kv_cache_config:
    free_gpu_memory_fraction: 0.9
    enable_block_reuse: false
EOF
    export TIKTOKEN_ENCODINGS_BASE="/tmp/harmony-reqs" && \
    mkdir -p $TIKTOKEN_ENCODINGS_BASE && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
    trtllm-serve openai/gpt-oss-120b \
      --backend pytorch --tp_size 1 \
      --max_batch_size 1 \
      --extra_llm_api_options /tmp/extra-llm-api-config.yml'
```

คำสั่งของ playbook ไม่มี `--port` จึง serve ที่ค่าตั้งต้นของ `trtllm-serve` คือ **8000** และ playbook ทดสอบที่ `http://localhost:8000/v1/completions` Lab 03 ต่อท้ายด้วย `--port 8355` (พอร์ต TensorRT-LLM ของคอร์ส) และรันในเบื้องหลัง Option 2 (Draft-Target) อยู่ใน lab 03 ตรงตามต้นฉบับ พร้อมการเปลี่ยนแปลงแบบเดียวกัน ถ้าจะเห็น speed-up คุณต้องมี baseline เซิร์ฟเวอร์ `baseline` ของ lab 03 เป็นรูปแบบของคอร์ส: **คำสั่งเดียวกันแต่ตัดบล็อก `speculative_config` และการดาวน์โหลด draft ออก** จึงต่างกันเฉพาะเรื่อง speculation เท่านั้น

```bash
# on: laptop
.venv/bin/python week25/07_nvfp4_speculative/labs/lab03_serve_and_measure.py --start baseline
.venv/bin/python week25/07_nvfp4_speculative/labs/lab03_serve_and_measure.py --label baseline
.venv/bin/python week25/07_nvfp4_speculative/labs/lab03_serve_and_measure.py --start eagle3
.venv/bin/python week25/07_nvfp4_speculative/labs/lab03_serve_and_measure.py --label eagle3
.venv/bin/python week25/07_nvfp4_speculative/labs/lab03_serve_and_measure.py --stop
```

การวัดแต่ละครั้งส่ง prompt ทดสอบ EAGLE-3 ของ playbook เอง (โจทย์ปัญหาเรื่องรถไฟ) ด้วย `max_tokens: 300` ที่ temperature 0 แล้วบันทึก tok/s ไว้ภายใต้ label ของมัน หาร eagle3 ด้วย baseline เพื่อได้ speed-up ของคุณ แล้วให้ lab 04 คำนวณย้อนกลับหา acceptance rate ที่สอดคล้องกัน:

```bash
# on: laptop
.venv/bin/python week25/07_nvfp4_speculative/labs/lab04_speculative_calculator.py --measured 1.8
```

**Expected output** (บันทึกจาก Mac เครื่องนี้สำหรับค่าตัวอย่าง 1.8× ไม่ใช่ค่าที่วัดจาก Spark: ใส่ค่าของคุณเอง)

```
▣ STEP 6 · back out α from YOUR measured speed-up (1.8× = eagle3 tok/s ÷ baseline tok/s, from lab 03)
│ Draft-Target: 8B FP4 drafts for 70B FP4      k=4  α ≈ 0.67
│ EAGLE-3 head on gpt-oss-120b (assumed c)     k=5  α ≈ 0.57
```

ขั้นตอนถัดไปตาม playbook: *"Experiment with different `max_draft_len` values (1, 2, 3, 4, 8)"* (ทดลองค่า `max_draft_len` ต่าง ๆ) และ *"Monitor token acceptance rates and throughput improvements."* (ติดตาม acceptance rate ของ token และ throughput ที่ดีขึ้น) แก้ `max_draft_len` ใน heredoc ของ lab 03 เพื่อทดลอง

**Spark สองเครื่อง** โมเดลที่ใหญ่เกินกว่า Spark เครื่องเดียว เช่น `nvidia/Qwen3-235B-A22B-FP4` รันด้วย tensor parallelism (`TP_SIZE=2`) บวก EAGLE-3 head (`nvidia/Qwen3-235B-A22B-Eagle3`, `max_draft_len: 3`) บนพอร์ต **8355** และเปิดด้วย `mpirun` ข้าม Spark ทั้งสองเครื่อง แท็บ *Multi-node serving* ของ playbook มีครบทุกขั้นตอน โดยต่อยอดจากการเดินสายและ SSH แบบไม่ใช้รหัสผ่านใน Module 02 และยังแนะนำ *NVIDIA Sync Cluster Assistant* สำหรับการตั้งค่านี้ด้วย

✓ Checkpoint: คุณมีตัวเลข tok/s สองค่าจาก Spark เครื่องเดียวกัน (baseline และ eagle3) และค่า speed-up หรือในโหมด DRY คุณทำนาย speed-up สำหรับ α = 0.8 ด้วย `max_draft_len: 5` ของ playbook ได้

## Labs — รันแล็บได้ที่นี่

**labs/lab01_nvfp4_simulator.py** — quantize ตัวเลขคล้าย weight 65,536 ตัวเป็น FP8, NVFP4, MXFP4 และ INT4 ด้วย Python ล้วน แล้ววัดความคลาดเคลื่อนจริงของแต่ละแบบ

**labs/lab02_quantize_nvfp4.py** — รัน NVFP4 quantization ด้วย Model Optimizer ตาม playbook บน Spark ของคุณในเบื้องหลัง (`--start`) แล้วตรวจ log, ไฟล์, `hf_quant_config.json` และขนาด

**labs/lab03_serve_and_measure.py** — เปิด `trtllm-serve` สำหรับ NVFP4, BF16, EAGLE-3, Draft-Target หรือ baseline (`--start …`) แล้วจับเวลาคำขอ หรือให้คะแนนคำถามหกข้อ (`--quality`)

**labs/lab04_speculative_calculator.py** — แปลง acceptance rate และความยาว draft เป็น token ต่อ step และ speed-up ตรวจสอบด้วยการจำลอง พร้อมคำนวณ α ย้อนกลับจาก speed-up ที่วัดได้

Lab 01 และ 04 รันบนแล็ปท็อป Lab 02 และ 03 รันแบบ LIVE บน Spark ของคุณหรือแบบ DRY Lab 03 จะใช้ laptop stand-in ที่ติดป้ายไว้แทนสำหรับคำขอของมัน

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 07 — ตัว quantize 4 บิต และตัววางแผนความยาว draft** เปิด `week25/07_nvfp4_speculative/exercises/ex07_fp4_and_speculation.py` ในไฟล์มี `TODO` สี่จุด:

1. `e2m1_round(x)`: ค่า E2M1 ที่ใกล้ที่สุด พร้อมเครื่องหมาย และอิ่มตัว (saturate) ที่ ±6
2. `quantize_block(block)`: scale = max |value| ÷ 6 แล้วคืนค่าที่ decode แล้ว
3. `expected_tokens(alpha, k)`: สูตรจากส่วนที่ 5
4. `best_draft_len(alpha, c, kmax)`: ค่า k ที่ให้ `E ÷ (1 + k·c)` ดีที่สุด

```bash
# on: laptop
.venv/bin/python week25/07_nvfp4_speculative/exercises/ex07_fp4_and_speculation.py
```

**Expected output** (เมื่อทำ TODO ครบทั้งสี่จุด บันทึกจาก Mac เครื่องนี้ด้วยเฉลย)

```
✓ e2m1_round: 2.6→3 · -0.3→-0.5 · 100→6 · 0.1→0 · -4.9→-4 · 5.2→6 · 1.2→1
✓ quantize_block: scale 0.06/6 = 0.01 → [0.01, -0.03, 0.005, 0.06, -0.02, 0, 0.03, -0.06]
✓ expected_tokens: α=0.8 k=4 → 3.36 · α=0.5 k=1 → 1.5 · α=1 k=5 → 6
✓ best_draft_len: α=0.7 c=0.113 → 4 · free drafts → kmax · weak draft → 1

▣ your quantizer, applied to 4,096 weight-like numbers (seed 7, 0.5 % outliers)
│ one scale per    16 values   SQNR  20.7 dB
│ one scale per    64 values   SQNR  17.4 dB
│ one scale per  4096 values   SQNR   6.7 dB

▣ your planner, applied (Draft-Target c = 0.113, from lab 04)
│ α = 0.6  best k = 3  E = 2.18 tokens per target pass
│ α = 0.7  best k = 4  E = 2.77 tokens per target pass
│ α = 0.8  best k = 5  E = 3.69 tokens per target pass
│ α = 0.9  best k = 9  E = 6.51 tokens per target pass
```

<details><summary>คำใบ้ — e2m1_round ในบรรทัดเดียว</summary>

เอาขนาด (magnitude) มา หาค่าใน grid ที่ใกล้ที่สุด (`min(E2M1_GRID, key=lambda g: abs(abs(x) - g))`) แล้วใส่เครื่องหมายกลับด้วย `math.copysign` ค่าที่มากกว่า 5 จะใกล้ 6 มากกว่า 4 การอิ่มตัวจึงได้มาฟรี ๆ

</details>

<details><summary>คำใบ้ — ทำไม block ขนาด 16 จึงดีกว่า scale ตัวเดียวสำหรับทุกค่า?</summary>

scale ถูกกำหนดด้วยค่าที่ใหญ่ที่สุดใน block ถ้าใช้ scale ตัวเดียวกับ 4,096 ค่า outlier เพียงตัวเดียวจะดันค่าเล็ก ๆ ทั้งหมดลงเป็น 0 หรือ ±0.5 แต่ถ้าใช้ block ละ 16 จะมีแค่เพื่อนบ้าน 15 ตัวที่รับผลกระทบ

</details>

<details><summary>ท้าทายเพิ่ม — ปัด scale เป็น FP8</summary>

NVFP4 ของจริงเก็บ block scale เป็น FP8 E4M3 คัดลอก `minifloat_round` จาก lab 01 มาไว้ในแบบฝึกหัดของคุณ แล้วปัด `scale` (ในหน่วยของ per-tensor scale) ก่อนนำไปใช้ SQNR ลดลงเท่าไร? แถว NVFP4 ของ lab 01 (20.5 dB) คือคำตอบที่ใช้เทียบ

</details>

✓ Checkpoint: บรรทัดตรวจทั้งสี่เป็น ✓

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `No module named 'mpi4py'` หรือ `pynvml.NVMLError_NotSupported` ระหว่าง quantize | เป็นเรื่องปกติบน DGX Spark ตามที่ playbook ระบุ ให้ตรวจว่ามีไฟล์ผลลัพธ์อยู่ใต้ `./output_models` |
| container ที่ quantize ปิดตัวพร้อม CUDA out of memory | playbook: quantize โมเดลที่เล็กกว่า หรือคืนหน่วยความจำ บน unified memory ให้ล้าง page cache ด้วย: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| `Cannot access gated repo` | ขอสิทธิ์เข้าถึงโมเดลบน Hugging Face แล้ว `hf auth login` บน Spark |
| `permission denied` ตอนลบ `./output_models` | container เขียนไฟล์ในฐานะ root playbook: ลองใหม่ด้วย `sudo rm -rf ./output_models` |
| `the input device is not a TTY` | คุณรันคำสั่งของ playbook ที่มี `-it` จากสคริปต์หรือ `nohup` ให้เอา `-it` ออก (lab 02/03 ทำให้แล้ว) |
| เซิร์ฟเวอร์ไม่ตอบที่ :8355 | ยังโหลดอยู่ หรือถูกเปิดด้วยพอร์ตตั้งต้น 8000 ของ playbook ตรวจ `tail ~/w25/logs/w25-trtllm.log` และ `docker ps` |
| `bind: address already in use` | มีเซิร์ฟเวอร์อื่นใช้พอร์ตอยู่ (vLLM บน 8000 หรือ `w25-trtllm` ตัวก่อนหน้า) รัน lab 03 `--stop` ก่อน playbook: *"Run one option at a time"* |
| speculative decoding ช้ากว่า baseline | acceptance ต่ำ (prompt เชิงสร้างสรรค์) หรือ `max_batch_size` มากกว่า 1 ลด `max_draft_len` และใช้ lab 04 ดูค่า α ที่จุดคุ้มทุน |

## Next — บทถัดไป

ไปต่อที่ [Lab 08 — LiteLLM: gateway เดียวสำหรับทุก engine และ Spark ทั้งสองเครื่อง](../08_litellm_gateway/TUTORIAL.md): วาง OpenAI endpoint เดียวที่มีการยืนยันตัวตนไว้หน้า Ollama, vLLM, SGLang และเซิร์ฟเวอร์ TensorRT-LLM ที่คุณเพิ่งเปิด พร้อม fallback เมื่อ Spark เครื่องใดล่ม
