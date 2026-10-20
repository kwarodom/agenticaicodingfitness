# ▶ Spark Lab 09 — fine-tune ด้วย LLaMA Factory: LoRA, QLoRA และ full

> ส่วนหนึ่งของ Week 25 · DGX Spark: fine-tune, serve และสร้างเอเจนต์ที่รันใน sandbox คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง ทุกแล็บรันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (อ้างอิงคำต่อคำจาก playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- ติดตั้ง LLaMA Factory บน Spark ของคุณตรงตามที่ playbook ของ NVIDIA ทำ: venv, PyTorch สำหรับ CUDA 13, clone repo แล้ว `pip install -e`
- คำนวณด้วยเลขคณิตว่า LoRA, QLoRA และ full fine-tuning เทรนอะไรบ้าง และแบบไหนใส่ใน 128 GB ได้
- สร้างชุดข้อมูลจริงสำหรับงานจริง: **ตัวจัดเส้นทางคำขอของแขกโรงแรม (hotel guest-request router)** ที่ตอบเป็น `{department, priority, reply}` ทั้งภาษาอังกฤษและภาษาไทย
- สร้างไฟล์ YAML สำหรับการเทรน ตรวจทุก key แล้วเปิด `llamafactory-cli train` ภายใต้ `nohup` เพื่อให้งานยังรันต่อแม้แล็ปท็อปของคุณจะหลับไป
- เฝ้าดู loss จากแล็ปท็อป จากนั้นแชตกับ adapter สร้างคำทำนาย (prediction) และ merge เป็นโมเดลเดียวสำหรับ Module 13

**Time** ~55 นาที · **Difficulty** ระดับกลาง · **Hardware** DGX Spark 1 เครื่อง (หรือไม่มีเลยก็ได้: ใช้โหมด DRY + แล็บชุดข้อมูล การคำนวณ และ config ที่รันบนแล็ปท็อป)

**Playbook ทางการที่ครอบคลุม:** [Fine-Tune LLMs with LLaMA Factory](https://build.nvidia.com/spark/llama-factory)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| ทำ Module 01 เสร็จแล้ว | `ssh -o BatchMode=yes spark-a true` | ทุกขั้นตอนบน Spark ในบทนี้รันผ่าน SSH |
| พื้นที่ว่างบน Spark > 50 GB | `df -h ~` บน Spark | playbook ขอพื้นที่ขนาดนี้สำหรับโมเดลและ checkpoint |
| Python ของ repo นี้พร้อม PyYAML | `.venv/bin/python -c "import yaml"` | lab 02–03 และแบบฝึกหัด parse ไฟล์ YAML และ JSON |
| ไม่มีงานใหญ่อื่นรันอยู่บน Spark | `docker ps`, `nvidia-smi` | การเทรนใช้ 128 GB ร่วมกับทุกโปรเซสอื่น |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
.venv/bin/python -c "import yaml; print('PyYAML', yaml.__version__)"
```

**Expected output**

```
PyYAML 6.0.3
```

> 🔐 โมเดลฐานในโมดูลนี้ `Qwen/Qwen3-4B-Instruct-2507` ไม่ใช่แบบ gated คุณไม่ต้องล็อกอิน Hugging Face ถ้าเปลี่ยนไปใช้โมเดลแบบ gated (Llama, Gemma) ให้รัน `hf auth login` ครั้งเดียว **บน Spark** ห้ามรันในแล็บ

✓ Checkpoint: import PyYAML ได้ และ `ssh spark-a true` ใช้ได้ (หรือคุณตัดสินใจแล้วว่าจะเรียนตามในโหมด DRY)

## 1 · LoRA, QLoRA หรือ full: ตัดสินใจด้วยเลขคณิต

fine-tuning คือการเทรนโมเดลต่อด้วยตัวอย่างของคุณเอง มีสามวิธี ความแตกต่างหลักคือแต่ละ **พารามิเตอร์** ใช้หน่วยความจำกี่ไบต์

| วิธี | อะไรถูกเทรน | หน่วยความจำต่อพารามิเตอร์ (AdamW, mixed precision) |
|---|---|---|
| **Full** | ทุก weight | 2 (weight แบบ bf16) + 2 (gradient) + 4 (สำเนา master แบบ fp32) + 8 (Adam moment แบบ fp32 สองตัว) = **16 ไบต์** |
| **LoRA** | เมทริกซ์บาง ๆ สองตัว A (r × in) และ B (out × r) ที่วางข้าง linear layer แต่ละตัวที่ถูกตรึง (frozen) | โมเดลฐานที่ถูกตรึง: **2 ไบต์** มีแค่ adapter ที่จ่าย 16 |
| **QLoRA** | adapter แบบเดียวกัน | linear layer ที่ถูกตรึงเก็บเป็น **4 บิต** (~0.56 ไบต์รวม scale) ส่วน embeddings ยังเป็น bf16 |

LoRA เทรนตัวเลข `r × (in + out)` ตัวต่อ layer ที่ถูกปรับ เมื่อตั้ง `lora_target: all` LLaMA Factory จะปรับ linear layer ทั้ง 7 ตัวในทุก block (q, k, v, o, gate, up, down) Lab 03 นำหลักนี้ไปใช้กับโมเดลจริงสี่ตัว โดยใช้ shape จาก `config.json` ของแต่ละโมเดล:

```bash
# on: laptop
.venv/bin/python week25/09_llama_factory/labs/lab03_config_and_launch.py
```

**Expected output** (step 1 — เป็นการคำนวณ ได้ผลเหมือนกันทุกเครื่อง; บันทึกจาก Mac เครื่องนี้)

```
▣ STEP 1 · how much does each method train, and does it fit in 128 GB?
│ LoRA rank  trainable params  of Qwen3-4B  adapter file (bf16)
│ ─────────  ────────────────  ───────────  ───────────────────  ────────────────────
│ r = 8        16.5 M           0.41 %        33.0 MB            ██░░░░░░░░░░░░░░░░░░
│ r = 16       33.0 M           0.82 %        66.1 MB            █████░░░░░░░░░░░░░░░
│ r = 32       66.1 M           1.64 %       132.1 MB            ██████████░░░░░░░░░░
│ r = 64      132.1 M           3.28 %       264.2 MB            ████████████████████
◆ Qwen3-4B has 4.02 B parameters. LoRA adds two thin matrices A (r×in) and B (out×r) to each of the 7 linear layers in all 36 blocks and trains only those.

│ model                   params   full (16 B/param)   LoRA r=16 (bf16 base)  QLoRA r=16 (4-bit base)
│ ──────────────────────  ───────  ──────────────────  ─────────────────────  ───────────────────────
│ Qwen3-4B-Instruct-2507    4.0 B      74 GB  ✓ fits       19 GB  ✓ fits          13 GB  ✓ fits
│ Qwen3-8B                  8.2 B     141 GB  ✕ > 128      27 GB  ✓ fits          17 GB  ✓ fits
│ Qwen3-32B                32.8 B     534 GB  ✕ > 128      78 GB  ✓ fits          33 GB  ✓ fits
│ Llama-3.3-70B            70.6 B    1139 GB  ✕ > 128     154 GB  ✕ > 128         56 GB  ✓ fits
◆ Rule of thumb, not a measurement: AdamW mixed precision, + 10 GB overhead, activations not counted (they grow with batch × cutoff_len; gradient checkpointing keeps them small).
```

ตารางบอกอะไรคุณ:

1. **full fine-tuning กินหน่วยความจำมหาศาล** มีแค่โมเดล 4B ที่เทรนแบบ full บน Spark เครื่องเดียวได้ โมเดล 8B ต้องใช้ 141 GB แล้วตั้งแต่ยังไม่นับ activations
2. **LoRA คือค่าตั้งต้น** มันเทรน weights ไม่ถึง 1 % และ adapter ที่บันทึกไว้มีขนาดหลักสิบ MB ไม่ใช่ GB คุณเก็บ adapter แยกหนึ่งตัวต่อโรงแรม ต่อภาษา หรือต่อลูกค้าได้
3. **QLoRA คือวิธีที่โมเดล 70B เทรนบน Spark เครื่องเดียวได้** แลกกับความแม่นยำและความเร็วที่ลดลงเล็กน้อย (โมเดลฐาน 4 บิตถูก de-quantize ระหว่างทำงาน) และต้องใช้ `bitsandbytes`

ภาพรวมคู่กันของ NVIDIA ([Fine-Tune Specialized LLMs with Unsloth](https://build.nvidia.com/spark/fine-tuning)) ให้หลักคิดเรื่องขนาดข้อมูลไว้ว่า: คู่ prompt–response ราว **100–1,000** คู่สำหรับ LoRA/QLoRA และ **1,000+** สำหรับ full fine-tuning ชุดข้อมูลของโมดูลนี้มี 504 คู่ จึงใช้ LoRA

✓ Checkpoint: คุณอธิบายได้ว่าทำไม Qwen3-8B จึงใส่ไม่ได้สำหรับ full fine-tuning บน Spark เครื่องเดียว แต่ Llama 3.3 70B ใส่ได้เมื่อใช้ QLoRA

## 2 · ติดตั้ง LLaMA Factory ตามวิธีของ playbook

[playbook](https://build.nvidia.com/spark/llama-factory) ใช้ Python venv ธรรมดา ไม่มี Docker นี่คือคำสั่งของมันตามลำดับ คอร์สรันคำสั่งเหล่านี้ใน home directory ของคุณ path จึงเป็น `~/factoryEnv` และ `~/LLaMA-Factory`:

```bash
# on: spark
nvcc --version            # Step 1: CUDA 12.9 or newer
nvidia-smi
python3 --version
git --version
python3 -m venv factoryEnv                                                             # Step 2
source ./factoryEnv/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130   # Step 3
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"   # Step 4
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git                       # Step 5
cd LLaMA-Factory
pip install -e ".[metrics]"                                                            # Step 6
```

การดาวน์โหลด PyTorch มีขนาดหลาย GB และอาจใช้เวลานานกว่าที่ Lab Runner ยอมให้แล็บรัน (900 s) lab 01 จึงรวม step 2–6 ไว้ในสคริปต์เดียวคือ `spark/setup_factory.sh` แล้วรันภายใต้ `nohup` สคริปต์นี้คือคำสั่งของ playbook บวก `set -e` และการตรวจเพื่อข้ามขั้นที่ทำไปแล้ว รันแล็บครั้งหนึ่งเพื่อดูแผน แล้วรันอีกครั้งพร้อม `SPARK_APPLY=1` เพื่อติดตั้ง:

```bash
# on: laptop
.venv/bin/python week25/09_llama_factory/labs/lab01_factory_setup.py
SPARK_APPLY=1 .venv/bin/python week25/09_llama_factory/labs/lab01_factory_setup.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ในโหมด DRY; ถ้ามี Spark แถว ◈ จะเปลี่ยนเป็น ✓ หรือ ✕)

```
▣ STEP 1 · the playbook's prerequisite checks (Step 1) — read-only
…
│ prerequisite                           result     last line
│ ─────────────────────────────────────  ─────────  ─────────────────────────────────────────
│ CUDA toolkit ≥ 12.9                    ◈ example  Build cuda_13.0.r13.0/compiler.xxxxxxxx_0
│ GPU visible                            ◈ example  NVIDIA GB10, 580.95.05
│ Python 3 + venv                        ◈ example  venv ok
│ Git                                    ◈ example  git version 2.43.0
│ > 50 GB free for models + checkpoints  ◈ example  /dev/nvme0n1p2  3.7T  412G  3.1T  12% /
…
▣ STEP 3 · upload the playbook's steps 2–6 as one script, run it under nohup (opt-in)
…
$ mkdir -p ~/w25/logs && { nohup bash ~/w25/m09/setup_factory.sh > ~/w25/logs/m09_setup.log 2>&1 < /dev/null & }; sleep 3; tail -3 ~/w25/logs/m09_setup.log   [not run]
◆ Installing downloads PyTorch (several GB) and writes ~/factoryEnv and ~/LLaMA-Factory on the Spark. Re-run with SPARK_APPLY=1 to do it.
```

รันแล็บซ้ำได้ทุกเมื่อ มันแค่รายงานความคืบหน้า และขั้นสุดท้ายจะเป็น ✓ เมื่อมีบรรทัด `PyTorch … CUDA: True` ปรากฏขึ้น `< /dev/null` มีความสำคัญ: ถ้าไม่มี ssh จะรอ input ของงานเบื้องหลังไปเรื่อย ๆ และแล็บจะค้าง

> 💡 **LLaMA Board** คือเว็บ UI ของ LLaMA Factory เป็น trainer ตัวเดียวกัน แต่มีฟอร์มที่เขียน YAML ให้คุณ โดยค่าตั้งต้นมันจะ listen บนทุก interface จึงควร bind ไว้กับ localhost แล้วเข้าผ่าน tunnel:
>
> ```bash
> # on: spark
> cd ~/LLaMA-Factory && source ~/factoryEnv/bin/activate
> GRADIO_SERVER_NAME=127.0.0.1 llamafactory-cli webui
> ```
>
> ```bash
> # on: laptop
> ssh -N -L 7860:localhost:7860 spark-a      # then open http://localhost:7860
> ```

ไม่บังคับ: smoke test ของ playbook เองเทรน Qwen3-4B ด้วยชุดข้อมูลเดโมสองชุด ผลลัพธ์ที่ playbook รายงานไว้คือ:

```bash
# on: spark
cd ~/LLaMA-Factory && source ~/factoryEnv/bin/activate
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

**Expected output** (REFERENCE — ยกมาจาก playbook)

```
***** train metrics *****
  epoch                    =        3.0
  total_flos               = 11076559GF
  train_loss               =     0.9993
  train_runtime            = 0:14:32.12
  train_samples_per_second =      3.749
  train_steps_per_second   =      0.471
Figure saved at: saves/qwen3-4b/lora/sft/training_loss.png
```

✓ Checkpoint: lab 01 พิมพ์ `✓ PyTorch in ~/factoryEnv sees the GPU (CUDA: True)` ในโหมด DRY คุณบอกได้ว่าขั้นตอนไหนของ playbook ที่รันภายใต้ `nohup` และเพราะอะไร

## 3 · งานจริง: ตัวจัดเส้นทางคำขอของแขกโรงแรม

playbook เทรนด้วยข้อมูลเดโม ส่วนคุณจะเทรนกับงานที่ **ให้คะแนนได้** แขกเขียนข้อความมา แล้วโมเดลตอบเป็น JSON หนึ่งบรรทัด:

```json
{"department": "engineering", "priority": "urgent", "reply": "Thank you for telling us. An engineer is on the way to room 1412 now. …"}
```

มีหกแผนก (`housekeeping`, `engineering`, `front_desk`, `food_beverage`, `concierge`, `security`) ระดับความเร่งด่วนสองระดับ และคำตอบเป็นภาษาเดียวกับที่แขกใช้ คืออังกฤษหรือไทย โค้ดตรวจผลลัพธ์ที่เป็น JSON ได้ Module 13 จึงวัดความแม่นยำได้จริงแทนที่จะเดา

LLaMA Factory อ่านเรคคอร์ดแบบ **Alpaca** (`instruction`, `input`, `output` และ `system` ที่ไม่บังคับ) และหาชุดข้อมูลจาก **ชื่อ** ใน `dataset_info.json` Lab 02 เขียนทั้งสองอย่างจากแม่แบบ (template) คำขอ 45 แบบ มีแปดแบบที่ถูก **กันไว้ (held out)**: ปรากฏเฉพาะในชุด eval การประเมินจึงทดสอบถ้อยคำใหม่ ไม่ใช่ประโยคที่โมเดลจำได้

```bash
# on: laptop
.venv/bin/python week25/09_llama_factory/labs/lab02_hotel_dataset.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ — ผลคงที่ทุกครั้ง, seed 25)

```
▣ STEP 2 · write the dataset files and the dataset_info.json registry
→ wrote 09_llama_factory/data/hotel_ops.json  (341.4 KB)
→ wrote 09_llama_factory/data/hotel_ops_eval.json  (39.2 KB)
→ wrote 09_llama_factory/data/dataset_info.json  (0.4 KB)

▣ STEP 3 · validate — the checks a failed training run would otherwise teach you
✓ hotel_ops: file_name 'hotel_ops.json' exists next to dataset_info.json
✓ hotel_ops: 504 records, every mapped column present (instruction, input, output, system)
✓ hotel_ops_eval: file_name 'hotel_ops_eval.json' exists next to dataset_info.json
✓ hotel_ops_eval: 60 records, every mapped column present (instruction, input, output, system)
✓ no empty instruction or output
✓ every output is one line of valid JSON
✓ department ∈ 6 allowed values · priority ∈ normal|urgent · reply non-empty
✓ every reply repeats the guest's room number (no invented rooms)
✓ no duplicate requests, and no request appears in both train and eval
✓ longest record ≈ 177 tokens, well under cutoff_len 1024 (nothing gets truncated)

▣ STEP 4 · balance — a router trained on 90% housekeeping learns to say 'housekeeping'
│ department     train  eval  train · Thai  train · urgent
│ ─────────────  ─────  ────  ────────────  ──────────────
│ housekeeping   84     10    24            0
│ engineering    84     10    24            36
│ front_desk     84     10    14            0
│ food_beverage  84     10    28            0
│ concierge      84     10    28            0
│ security       84     10    16            67
◆ priority in train: normal 401 · urgent 103 — urgent is rarer, as in a real hotel. Watch its recall in Module 13, not just overall accuracy.
```

registry จับคู่บทบาทของ LLaMA Factory เข้ากับชื่อคอลัมน์ของคุณ:

```json
"hotel_ops": {
  "file_name": "hotel_ops.json",
  "columns": {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}
}
```

> 💡 จำนวน token เป็นค่าประมาณ (ราว 4 ตัวอักษรต่อ token สำหรับภาษาอังกฤษ และ 2 สำหรับอักษรไทย) ไม่ได้ใช้ tokenizer จริง มันแค่ต้องแสดงให้เห็นว่าไม่มีอะไรเข้าใกล้ `cutoff_len`

✓ Checkpoint: `week25/09_llama_factory/data/` มีไฟล์สามไฟล์ และทุกบรรทัดการ validate เป็น ✓ Module 10, 13 และ capstone นำชุดข้อมูลนี้กลับมาใช้

## 4 · YAML สำหรับการเทรน ทีละ key

การรัน LLaMA Factory หนึ่งครั้งคือไฟล์ YAML หนึ่งไฟล์ Lab 03 (step 3) เริ่มจาก `examples/train_lora/qwen3_lora_sft.yaml` ของ playbook แล้วเขียนไฟล์หกไฟล์ลงใน `week25/09_llama_factory/configs/`:

| ไฟล์ | คำสั่ง | ผลลัพธ์บน Spark (ภายใต้ `~/w25/m09/`) |
|---|---|---|
| `hotel_lora_sft.yaml` | `llamafactory-cli train` | `saves/qwen3-4b-hotel/lora/sft/` (adapter) |
| `hotel_qlora_sft.yaml` | `llamafactory-cli train` | `saves/qwen3-4b-hotel/qlora/sft/` |
| `hotel_full_sft.yaml` | `llamafactory-cli train` | `saves/qwen3-4b-hotel/full/sft/` |
| `hotel_chat.yaml` | `llamafactory-cli chat` | — (แบบโต้ตอบ) |
| `hotel_predict.yaml` | `llamafactory-cli train` (`do_predict`) | `saves/qwen3-4b-hotel/lora/predict/generated_predictions.jsonl` |
| `hotel_merge.yaml` | `llamafactory-cli export` | `saves/qwen3-4b-hotel/merged/` |

อะไรเปลี่ยนไปจากตัวอย่างต้นทาง (upstream) และเพราะอะไร:

| Key | Upstream | ในที่นี้ | ทำไม |
|---|---|---|---|
| `dataset_dir` / `dataset` | `data` / `identity,alpaca_en_demo` | `data` / `hotel_ops` | ชุดข้อมูลของคุณ หาจากชื่อใน `data/dataset_info.json` |
| `eval_dataset` | (comment ไว้) | `hotel_ops_eval` | ได้กราฟ `eval_loss` จากคำขอที่ถูกกันไว้ |
| `lora_rank` / `lora_alpha` | 8 / (ค่าตั้งต้น 2 × rank) | 16 / 32 | เพิ่มความจุอีกนิดสำหรับการตัดสินใจหกทางในสองภาษา |
| `cutoff_len` | 2048 | 1024 | เรคคอร์ดยาวสุดราว ~180 token |
| `per_device_train_batch_size` × `gradient_accumulation_steps` | 1 × 8 | 2 × 4 | effective batch เท่าเดิมคือ 8 แต่ forward pass น้อยครั้งลงและใหญ่ขึ้น |
| `logging_steps` | 10 | 5 | ได้จุด loss มากขึ้นสำหรับการรัน 189 step |
| `output_dir` | `saves/qwen3-4b/lora/sft` | `saves/qwen3-4b-hotel/lora/sft` | แยกออกจาก smoke test ของ playbook |

ที่เหลือทั้งหมด (`template: qwen3_nothink`, `learning_rate: 1.0e-4`, cosine schedule, `bf16: true`, 3 epoch) เป็นไปตามตัวอย่างของ playbook มีสองจุดที่คอร์สเปลี่ยน: ไฟล์ full-FT ตัด `deepspeed: examples/deepspeed/ds_z3_config.json` ของ upstream ออก (ZeRO-3 แบ่ง shard ข้าม GPU หลายตัว แต่ Spark หนึ่งเครื่องมี GPU ตัวเดียว) และไฟล์ QLoRA คัดลอก `quantization_bit: 4` / `quantization_method: bnb` มาจาก `examples/train_qlora/qwen3_lora_sft_otfq.yaml` ของ upstream

แล็บอ่านไฟล์ของมันกลับมาและตรวจ:

**Expected output** (step 3 — บันทึกจาก Mac เครื่องนี้)

```
✓ all 6 files parse as YAML; every key is a known LLaMA Factory argument
✓ datasets ['hotel_ops', 'hotel_ops_eval'] are registered in data/dataset_info.json
✓ one base model and one template (qwen3_nothink) across train, chat, predict and merge
✓ learning_rate is a number (1.0e-4), not the string '1e-4'
✓ chat, predict and merge all load the adapter from saves/qwen3-4b-hotel/lora/sft
✓ merge config has no quantization_bit (upstream warns against it)
```

> ⚠ `learning_rate: 1e-4` ดูเหมือนไม่มีปัญหา แต่ parser แบบ YAML 1.1 ธรรมดาอย่าง PyYAML จะอ่านเป็น **สตริง** `'1e-4'` ตัวอย่างของ upstream เขียนเป็น `1.0e-4` ให้ทำแบบเดียวกัน เพื่อให้ทุกเครื่องมือเห็นค่าตรงกัน

Step 2 ของแล็บตรวจการคำนวณจำนวน step เทียบกับตัวเลขของ playbook เอง การรันของ playbook จบที่ `checkpoint-411`: ชุดข้อมูลเดโมสองชุดของมันมี 91 + 999 = 1,090 เรคคอร์ด ซึ่งได้ ⌈1090 ÷ 8⌉ = 137 step ต่อ epoch × 3 epoch = 411:

**Expected output** (step 2 — บันทึกจาก Mac เครื่องนี้)

```
│ playbook example: 1090 records ÷ (1 × 8) per step = 137 steps/epoch × 3 epochs = 411 steps  → the playbook shows 'checkpoint-411'  ✓
│ and 1090 × 3 records ÷ 872.12 s (train_runtime 0:14:32.12) = 3.749 samples/s → the playbook shows 3.749  ✓
│ your hotel run:   504 records ÷ (2 × 4) per step = 63 steps/epoch × 3 epochs = 189 steps
◆ Rough time estimate: 1512 samples ÷ 3.749 samples/s (the playbook's REFERENCE throughput) ≈ 7 min. Your records are shorter than Alpaca's, so expect it to be quicker — lab 04 shows the real time.
```

✓ Checkpoint: คุณบอกได้ว่า `dataset:` ต้องตรงกับอะไร ทำไม template ต้องเหมือนกันในทั้งสี่ไฟล์ และการรันของคุณจะใช้ optimizer step กี่ step

## 5 · เปิดงานภายใต้ nohup

Step 4 ของ lab 03 คัดลอกข้อมูลและ config ไปที่ `~/w25/m09/` บน Spark ตรวจว่ามี `~/factoryEnv/bin/llamafactory-cli` อยู่ และไม่มีงาน LLaMA Factory อื่นรันอยู่ มันจะเริ่มเทรน **เฉพาะ** เมื่อคุณยินยอม (opt in):

```bash
# on: laptop
SPARK_APPLY=1 .venv/bin/python week25/09_llama_factory/labs/lab03_config_and_launch.py
SPARK_APPLY=1 .venv/bin/python week25/09_llama_factory/labs/lab03_config_and_launch.py --method qlora   # or full
```

คำสั่งที่มันรันบน Spark:

```bash
# on: spark
mkdir -p ~/w25/logs && cd ~/w25/m09 && source ~/factoryEnv/bin/activate && { RECORD_VRAM=1 nohup llamafactory-cli train configs/hotel_lora_sft.yaml > ~/w25/logs/m09_hotel_lora.log 2>&1 < /dev/null & echo $! > ~/w25/logs/m09_hotel_lora.pid; }
```

- `cd ~/w25/m09` มีความสำคัญ: `dataset_dir: data` และ `output_dir: saves/…` เป็น path สัมพัทธ์กับไดเรกทอรีที่คุณรัน
- `nohup … &` ทำให้งานรันต่อหลัง ssh ตัดการเชื่อมต่อ `< /dev/null` ทำให้ ssh กลับมาได้ทันที
- `RECORD_VRAM=1` ทำให้ LLaMA Factory เพิ่ม `vram_allocated` / `vram_reserved` ลงใน `trainer_log.jsonl` Lab 04 พิมพ์ค่าสูงสุดออกมา ให้เทียบกับการคำนวณในส่วนที่ 1

✓ Checkpoint: ในโหมด LIVE แล็บพิมพ์ `started pid …` และ `~/w25/logs/m09_hotel_lora.log` เริ่มโตขึ้น ในโหมด DRY คุณอ่านคำสั่งเปิดงานและอธิบายแต่ละส่วนได้

## 6 · เฝ้าดู loss จากแล็ปท็อป

LLaMA Factory เขียน log สองชุด **console log** (ไฟล์ nohup ของคุณ) มี progress bar และบรรทัด `{'loss': …}` ส่วน **`trainer_log.jsonl`** ในโฟลเดอร์ผลลัพธ์มี JSON หนึ่งบรรทัดต่อ logging step: `current_steps`, `total_steps`, `loss`, `lr`, `epoch`, `percentage`, `elapsed_time`, `remaining_time` และ `eval_loss` ใน step ที่มีการ eval Lab 04 parse ทั้งสองแบบ

ก่อนที่การรันของคุณเองจะมีขึ้น แล็บจะทดสอบ parser ของมันกับ log แบบ EXAMPLE log นั้นสร้างจากสูตร และแล็บติดป้ายไว้ชัดเจน:

```bash
# on: laptop
.venv/bin/python week25/09_llama_factory/labs/lab04_monitor_and_export.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้; ตัวเลข loss เป็น EXAMPLE — ข้อมูลสังเคราะห์ที่สร้างจากสูตรเพื่อทดสอบ parser ไม่ใช่การเทรนจริง)

```
▣ STEP 1 · test the parser on an EXAMPLE log first
◈ EXAMPLE — synthetic trainer_log.jsonl generated by a formula to test the parser. Not a training run.
✓ parsed 40 JSON lines (37 loss + 3 eval), skipped the half-written last line
│ step     epoch  loss    lr        loss (0 … 1.74)
│ ───────  ─────  ──────  ────────  ────────────────────────
│ 5/189    0.08   1.7356  9.98e-05  ████████████████████████
│ 30/189   0.48   0.7523  9.39e-05  ██████████░░░░░░░░░░░░░░
│ 55/189   0.87   0.3905  8.05e-05  █████░░░░░░░░░░░░░░░░░░░
│ 80/189   1.27   0.2574  6.19e-05  ████░░░░░░░░░░░░░░░░░░░░
│ 110/189  1.75   0.2033  3.73e-05  ███░░░░░░░░░░░░░░░░░░░░░
│ 135/189  2.14   0.1886  1.88e-05  ███░░░░░░░░░░░░░░░░░░░░░
│ 160/189  2.54   0.1832  5.70e-06  ███░░░░░░░░░░░░░░░░░░░░░
│ 185/189  2.94   0.1812  1.10e-07  ███░░░░░░░░░░░░░░░░░░░░░
◆ EXAMPLE: 97.88% done · elapsed 0:06:10 · left 0:00:00 · loss 1.736 → 0.181 (90% lower)
◆ eval_loss: 0.487@50 → 0.265@100 → 0.235@150
✓ console parser: 2 loss dicts · 1 eval dict · train metrics block · tqdm 180/189
✓ failure detector recognises OOM and an unregistered dataset name
```

เมื่อเชื่อมต่อ Spark แล้ว step 2–4 จะอ่านการรัน **ของคุณ**: โปรเซสยังทำงานอยู่ไหม loss ล่าสุดเป็นเท่าไร log แสดงความผิดพลาดที่รู้จักหรือไม่ (หน่วยความจำไม่พอ, โมเดลแบบ gated, ชุดข้อมูลที่ไม่ได้ลงทะเบียน, template ที่ไม่รู้จัก, ไม่มี bitsandbytes) และโฟลเดอร์ผลลัพธ์มีสิ่งที่ Step 9 ของ playbook ระบุไว้หรือไม่: `adapter_config.json`, ไดเรกทอรี checkpoint และ `training_loss.png` Lab 04 เป็นแบบอ่านอย่างเดียว จึงรันซ้ำได้บ่อยเท่าที่ต้องการ ถ้าจะ parse log ที่คุณคัดลอกมาเอง ให้รันด้วย `--log path/to/trainer_log.jsonl`

วิธีอ่านกราฟ loss:

| สิ่งที่เห็น | มักหมายความว่า | ลองทำ |
|---|---|---|
| loss ลดลงเร็ว แล้วแบนราบ | โมเดลเรียนรู้รูปแบบแล้ว | เสร็จแล้ว ตรวจความแม่นยำใน Module 13 |
| `eval_loss` เพิ่มขึ้นขณะที่ `loss` ยังลดลงต่อ | over-fitting (จำข้อมูล) | ลดจำนวน epoch หรือใช้ข้อมูลที่หลากหลายกว่า |
| loss แบนราบตั้งแต่ต้น | learning rate ต่ำเกินไป หรือ label ถูก mask | ตรวจ `learning_rate` และการจับคู่คอลัมน์ |
| loss กระโดดเป็น `nan` | learning rate สูงเกินไป | ลดลง (เช่น `5.0e-5`) |

✓ Checkpoint: สามบรรทัดของ step 1 เป็น ✓ ถ้ามี Spark คุณได้เห็นตาราง loss ของการรันของคุณเองและค่าสูงสุดของ `vram_allocated` แล้ว

## 7 · แชต ทำนาย และ merge

เมื่อการเทรนเสร็จ lab 04 รันงานต่อเนื่องสามอย่าง แต่ละอย่างรันเฉพาะเมื่อคุณสั่งด้วย flag ของมัน

```bash
# on: laptop
.venv/bin/python week25/09_llama_factory/labs/lab04_monitor_and_export.py --chat      # 3 guest requests, ~2 min
.venv/bin/python week25/09_llama_factory/labs/lab04_monitor_and_export.py --predict   # 60 held-out answers (nohup)
.venv/bin/python week25/09_llama_factory/labs/lab04_monitor_and_export.py --export    # merge (nohup)
```

- **`llamafactory-cli chat configs/hotel_chat.yaml`** คือ Step 10 ของ playbook ที่ใช้ adapter ของคุณ เป็นแบบโต้ตอบ: พิมพ์คำขอ แล้วพิมพ์ `exit` เพื่อออก แล็บส่งคำขอสามข้อเข้าไปผ่าน pipe คือภาษาอังกฤษหนึ่ง ภาษาไทยหนึ่ง และแบบเร่งด่วนหนึ่ง โดยมี `clear` คั่นระหว่างกันเพื่อไม่ให้ใช้ประวัติร่วมกัน
- **`llamafactory-cli train configs/hotel_predict.yaml`** (`do_predict: true`, `predict_with_generate: true`) เขียน `generated_predictions.jsonl` เป็นบรรทัด `{"prompt", "predict", "label"}` หนึ่งบรรทัดต่อคำขอที่ถูกกันไว้ config ตั้ง `do_sample: false` และ `max_new_tokens: 160`: LLaMA Factory สุ่ม (sample) เป็นค่าตั้งต้น (temperature 0.95, top-p 0.7) ซึ่งจะทำให้คะแนนเปลี่ยนไปในแต่ละครั้งที่รัน Module 13 ให้คะแนนไฟล์นี้
- **`llamafactory-cli export configs/hotel_merge.yaml`** คือ Step 11 ของ playbook มันรวม adapter เข้ากับ weights ของโมเดลฐาน แล้วเขียนเป็นโมเดล Hugging Face ธรรมดาที่ `saves/qwen3-4b-hotel/merged/` ซึ่ง vLLM serve ได้ (Module 13) config ของ upstream เตือนไว้ว่า: *do not use a quantized model or `quantization_bit` when merging* (อย่าใช้โมเดลที่ถูก quantize หรือ `quantization_bit` ตอน merge) ดังนั้นให้ merge จากโมเดลฐานแบบ bf16 แม้จะเทรนด้วย QLoRA ก็ตาม

ทุกอย่างไปอยู่ที่ไหนบน Spark:

| อะไร | Path |
|---|---|
| ชุดข้อมูล + registry | `~/w25/m09/data/{hotel_ops.json, hotel_ops_eval.json, dataset_info.json}` |
| config | `~/w25/m09/configs/*.yaml` |
| LoRA adapter | `~/w25/m09/saves/qwen3-4b-hotel/lora/sft/` |
| คำทำนาย | `~/w25/m09/saves/qwen3-4b-hotel/lora/predict/generated_predictions.jsonl` |
| โมเดลที่ merge แล้ว | `~/w25/m09/saves/qwen3-4b-hotel/merged/` |
| log | `~/w25/logs/m09_*.log` |

> ⚠ การเก็บกวาด (Step 12 ของ playbook) ลบ `~/LLaMA-Factory` และ `~/factoryEnv` ด้วย `rm -rf` ไม่มีแล็บไหนทำสิ่งนี้ให้คุณ เก็บไว้จนกว่า Module 13 จะ serve โมเดลที่ merge แล้วของคุณ

✓ Checkpoint: ในโหมด LIVE `--chat` ตอบในรูปแบบ JSON ที่คุณเทรน และ `saves/qwen3-4b-hotel/merged/` มี `config.json` พร้อม shard แบบ `.safetensors` ในโหมด DRY คุณบอกชื่อคำสั่งต่อเนื่องทั้งสามและสิ่งที่แต่ละคำสั่งเขียนออกมาได้

## Labs — รันแล็บได้ที่นี่

**labs/lab01_factory_setup.py** — ติดตั้ง LLaMA Factory ตามวิธีของ playbook (venv, PyTorch cu130, clone, pip install) ภายใต้ nohup และยืนยันว่ามองเห็น CUDA

**labs/lab02_hotel_dataset.py** — สร้างและ validate ชุดข้อมูลคำขอของแขกโรงแรม (train 504 + eval ที่กันไว้ 60 ทั้งภาษาอังกฤษและไทย) และ dataset_info.json ของมัน

**labs/lab03_config_and_launch.py** — การคำนวณ LoRA vs QLoRA vs full, ไฟล์ YAML หกไฟล์ที่ผ่านการ validate, การอัปโหลด และการเปิด llamafactory-cli train ภายใต้ nohup แบบ opt-in

**labs/lab04_monitor_and_export.py** — parse log การเทรน (ทดสอบกับ EXAMPLE ก่อน) วินิจฉัยความผิดพลาด แล้วแชต ทำนาย และ merge เมื่อสั่ง

Lab 02 และ 03 (step 1–3) รันบนแล็ปท็อปได้ทั้งหมด ส่วน Lab 01, 03 (step 4) และ 04 สั่งงาน Spark ผ่าน SSH หรือแสดงแผนในโหมด DRY

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 09 — แก้ config** เปิด `week25/09_llama_factory/exercises/ex09_fix_the_config.py` ไฟล์ `dataset_info.json` และ YAML สำหรับเทรนของเพื่อนร่วมทีมมีข้อผิดพลาดสี่จุด แต่ละจุดจะทำให้การรันหยุด หรือเทรน adapter ที่ภายหลังถูกใช้กับรูปแบบแชตที่ผิด:

1. `file_name` ชี้ไปที่ไฟล์ที่ไม่มีอยู่
2. การจับคู่คอลัมน์ระบุชื่อคอลัมน์ที่ไม่มีในเรคคอร์ด
3. `dataset:` ไม่ใช่ชื่อที่ลงทะเบียนไว้ใน `dataset_info.json`
4. `template` ไม่ตรงกับโมเดลฐานหรือ config ของแชต

ตัวตรวจทำงานแบบออฟไลน์ มัน parse ข้อความด้วย `json` และ PyYAML แล้วเทียบกับไฟล์จริงใน `data/`

```bash
# on: laptop
.venv/bin/python week25/09_llama_factory/exercises/ex09_fix_the_config.py
```

**Expected output** (เมื่อแก้ TODO ครบทั้งสี่จุด; บันทึกจาก Mac เครื่องนี้)

```
✓ TODO 1: file_name 'hotel_ops.json' exists in data/
✓ TODO 2: every mapped column exists in the records (response → output)
✓ TODO 3: dataset 'hotel_ops' is registered in dataset_info.json
✓ TODO 4: template qwen3_nothink in both train and chat configs

▣ your config, applied: 504 records · effective batch 8 · 189 optimizer steps
│ chat loads the adapter from saves/qwen3-4b-hotel/lora/sft ✓ = train output_dir
```

<details><summary>คำใบ้ — จะหาชื่อคอลัมน์ที่ถูกต้องได้อย่างไร?</summary>

เปิด `week25/09_llama_factory/data/hotel_ops.json` แล้วดู key ของเรคคอร์ดแรก `columns` จับคู่บทบาทของ LLaMA Factory (`prompt`, `query`, `response`, `system`) เข้ากับชื่อ key **ของคุณ**

</details>

<details><summary>คำใบ้ — Qwen3-4B-Instruct-2507 ใช้ template ไหน?</summary>

`examples/train_lora/qwen3_lora_sft.yaml` ของ playbook ใช้ `template: qwen3_nothink` สำหรับโมเดลนี้พอดี config ของ chat, predict และ merge ต้องใช้ตัวเดียวกัน

</details>

<details><summary>ท้าทายเพิ่ม — เพิ่มแผนกที่เจ็ด</summary>

เพิ่มแม่แบบ `spa` ลงใน `T` ของ lab 02 (ภาษาอังกฤษและไทย และกันไว้หนึ่งแบบ) เพิ่ม `spa` ใน `DEPTS` แล้วรัน lab 02 และ 03 ใหม่ ตอนนี้การรันใช้กี่ step? การตรวจข้อไหนใน lab 02 จะจับแม่แบบที่ลืมใส่ `{room}` ในคำตอบได้?

</details>

✓ Checkpoint: บรรทัดตรวจทั้งสี่เป็น ✓

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `CUDA: False` หลังติดตั้ง | wheel แบบ cu130 ไม่ได้ถูกติดตั้ง รัน `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130` ใหม่ภายใน `~/factoryEnv` แล้วอ่าน log ของ lab 01 |
| `Cannot open data/dataset_info.json` | คุณรัน `llamafactory-cli` นอก `~/w25/m09` `dataset_dir: data` เป็น path สัมพัทธ์กับไดเรกทอรีที่คุณรัน |
| `Undefined dataset hotel-ops in dataset_info.json.` | `dataset:` ต้องเป็น key ที่อยู่ใน `dataset_info.json` (แบบฝึกหัด 09) |
| `CUDA out of memory` ระหว่างเทรน | วิธีแก้ของ playbook: ลด `per_device_train_batch_size` หรือเพิ่ม `gradient_accumulation_steps` |
| หน่วยความจำตึงทั้งที่ `free -g` ยังแสดงพื้นที่ว่าง | page cache ของ UMA วิธีแก้ของ playbook: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| `Cannot access gated repo` | โมเดลแบบ gated: ขอสิทธิ์ในหน้าของโมเดล แล้ว `hf auth login` บน Spark |
| QLoRA: `No module named 'bitsandbytes'` | การติดตั้งของ playbook ไม่ได้รวมไว้: `pip install bitsandbytes` ใน `~/factoryEnv` (คอร์สนี้ยังไม่ได้ยืนยันบน GB10) |
| แล็บค้างหลังขึ้น "started" | งานเบื้องหลังยังถือ stdin ของ ssh อยู่ ให้ใส่ `< /dev/null` ในทุกคำสั่ง nohup ที่คุณเขียน |
| loss ของการเทรนไม่ลดลง | คำแนะนำของ playbook: ปรับ `learning_rate` หรือตรวจคุณภาพชุดข้อมูล (lab 02) |

## Next — บทถัดไป

ไปต่อที่ [Lab 10 — Unsloth: LoRA fine-tuning ที่เร็วขึ้น](../10_unsloth/TUTORIAL.md): เทรนชุดข้อมูลโรงแรมชุดเดียวกันด้วย Unsloth และเรียนรู้วิธีเปรียบเทียบสอง framework อย่างยุติธรรม
