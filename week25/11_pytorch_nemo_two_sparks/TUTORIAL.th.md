# ▶ Spark Lab 11 — fine-tune ด้วย PyTorch และ NeMo AutoModel บน Spark หนึ่งและสองเครื่อง

> ส่วนหนึ่งของ Week 25 · DGX Spark: fine-tune, serve และสร้างเอเจนต์ที่รันใน sandbox คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง ทุกแล็บรันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (อ้างอิงคำต่อคำจาก playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- เปรียบเทียบสองเส้นทาง fine-tuning ของ NVIDIA: สูตร **PyTorch** แบบตรง ๆ (Transformers + PEFT + TRL) และ **NeMo AutoModel** (สูตรแบบ YAML)
- คำนวณด้วยเลขคณิตว่าทำไม full fine-tuning ใช้ ~8 ไบต์ต่อพารามิเตอร์ LoRA ~2 และ QLoRA ~0.5 และ activations ทำให้งบบานปลายตรงไหน
- เปิดงาน LoRA บน Spark เครื่องเดียวในเบื้องหลัง (background) พร้อมไฟล์ log แล้วเฝ้าดูด้วย parser ที่อ่านกราฟ loss
- รันตัวอย่าง LoRA, QLoRA และ full-SFT ของ NeMo AutoModel ด้วยวิธีเดียวกัน
- เตรียมการรันบน **Spark สองเครื่อง** (Docker Swarm + Accelerate + FSDP) สำหรับ LoRA ของ Llama 3.1 70B และดูว่า FSDP ส่งอะไรข้ามสาย

**Time** ~60 นาที (บวกเวลาเทรน) · **Difficulty** ระดับสูง · **Hardware** Spark 1 เครื่อง (2 เครื่องสำหรับส่วนที่ 6; หรือไม่มีเลยก็ได้: ใช้โหมด DRY + แล็บที่เป็นการคำนวณ)

**Playbook ทางการที่ครอบคลุม:** [Fine-tune with PyTorch](https://build.nvidia.com/spark/pytorch-fine-tune) · [Fine-tune with NeMo](https://build.nvidia.com/spark/nemo-fine-tune)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| doctor ของ Module 01 เป็น ✓ ทั้งหมดบน Spark A | lab 01-1 | Docker โดยไม่ต้อง sudo, CUDA 13, ว่าง ~119 GiB |
| ทำ Module 02 เสร็จแล้ว (เฉพาะส่วนที่ 6) | lab 02-1 เป็น ✓ ทั้งหมด, lab 02-3 busbw ≥ 21.875 GB/s | FSDP สื่อสารผ่านลิงก์ QSFP |
| ล็อกอิน Hugging Face **บน Spark** | `test -s ~/.cache/huggingface/token && echo ok` บน Spark | โมเดล Llama เป็นแบบ gated |
| สิทธิ์เข้าถึงโมเดลแบบ gated | ยอมรับ licence ในหน้าของแต่ละโมเดล | ไม่อย่างนั้นจะเจอ "Cannot access gated repo" |
| ดิสก์ว่าง ~100 GB | `df -h /` | container สองตัว + weights ของโมเดล |

ล็อกอิน Hugging Face ครั้งเดียว **บน Spark** โทเค็นจะไปอยู่ที่ `~/.cache/huggingface/` ซึ่งทุก container ในโมดูลนี้ mount ไว้ ห้ามใส่โทเค็นในแล็บ ในสคริปต์ที่คุณ commit หรือในแชต:

```bash
# on: spark
hf auth login
```

(ถ้าบน Spark เองไม่มี `hf` ติดตั้งไว้ ให้รันคำสั่งเดียวกันครั้งเดียวภายใน container PyTorch ของส่วนที่ 3: โฟลเดอร์ cache ถูก mount ไว้ การล็อกอินจึงคงอยู่) โมเดลที่โมดูลนี้ดาวน์โหลด: `meta-llama/Llama-3.2-3B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`, `unsloth/Meta-Llama-3.1-70B-bnb-4bit` (สคริปต์ PyTorch) และ `meta-llama/Llama-3.1-8B`, `meta-llama/Meta-Llama-3-70B`, `Qwen/Qwen3-8B` (ตัวอย่าง NeMo) ขอสิทธิ์เข้าถึงในหน้าของแต่ละโมเดลที่คุณวางแผนจะใช้

จากนั้นตรวจการคำนวณหน่วยความจำบนแล็ปท็อปของคุณ:

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
.venv/bin/python week25/11_pytorch_nemo_two_sparks/labs/lab01_finetune_memory.py
```

**Expected output** (เป็นการคำนวณ บันทึกจาก Mac เครื่องนี้; ตารางแรก)

```
▣ STEP 1 · parameters, counted from each model's shape
│ model          total   linear layers  embeddings  LoRA r=8 params (all 7 projections)
│ ─────────────  ──────  ─────────────  ──────────  ───────────────────────────────────
│ Llama 3.2 3B   3.21B   2.82B          0.39B       12,156,928
│ Llama 3.1 8B   8.03B   6.98B          1.05B       20,971,520
│ Llama 3.1 70B  70.55B  68.45B         2.10B       103,546,880
│ Qwen3 8B       8.19B   6.95B          1.24B       21,823,488
```

✓ Checkpoint: Spark ล็อกอิน Hugging Face แล้ว คุณมีสิทธิ์เข้าถึงอย่างน้อย `meta-llama/Llama-3.1-8B-Instruct` และ lab 01 รันบนแล็ปท็อปของคุณได้

## 1 · สองวิธีในการ fine-tune บน Spark

playbook ทั้งสอง fine-tune โมเดลจาก Hugging Face ภายใน container ของ NGC สิ่งที่ต่างกันคือคุณได้เห็นและต้องเขียนเองมากแค่ไหน:

| | Playbook PyTorch | Playbook NeMo AutoModel |
|---|---|---|
| Container | `nvcr.io/nvidia/pytorch:25.11-py3` | `nvcr.io/nvidia/nemo-automodel:26.02` |
| สิ่งที่คุณติดตั้ง | `pip install transformers peft datasets trl bitsandbytes` | ไม่ต้องติดตั้งอะไร: AutoModel อยู่ใน `/opt/Automodel` |
| สูตร (recipe) หนึ่งสูตรคือ | สคริปต์ Python หนึ่งไฟล์ต่อโมเดล (flag แบบ argparse) | ไฟล์ YAML + การ override แบบ `--section.key value` |
| สูตรที่มี | Llama 3.2 3B full SFT · Llama 3.1 8B LoRA · Llama 3.1 70B LoRA (FSDP) · Llama 3.1 70B QLoRA | LoRA (Llama 3.1 8B), QLoRA (Meta-Llama-3 70B), full SFT (Qwen3 8B) |
| ข้อมูล | ตัวอย่าง Alpaca 512 ตัวอย่างโดยค่าตั้งต้น (`--dataset_size`) | SQuAD (ไฟล์สูตรตั้งชื่อว่า `*_squad_*`) ถูก pack เป็น sequence ยาว 1024 token |
| บันทึกโมเดล | เฉพาะสคริปต์ 70B LoRA (`--output_dir`) | เสมอ: `checkpoints/LATEST/` (model, optim, rng, …) |
| Spark สองเครื่อง | **ได้**: Docker Swarm + Accelerate + FSDP (ส่วนที่ 6) | คอลัมน์ Spark ในตาราง matrix ของ playbook เป็น "—": มีแต่ตัวอย่างแบบ Spark เครื่องเดียว |

ใช้สคริปต์ PyTorch เมื่ออยากเห็นทุกบรรทัดของ training loop (แต่ละไฟล์ยาว ~200 บรรทัด อยู่ใน `assets/`) ใช้ NeMo AutoModel เมื่อต้องการสูตรที่ทดสอบแล้ว มี packed sequence และ checkpoint โดยไม่ต้องเขียนโค้ด

✓ Checkpoint: คุณบอกได้ว่าในสองตัวนี้ ตัวไหนรันข้าม Spark สองเครื่องได้ตาม playbook และตัวไหนบันทึก checkpoint เป็นค่าตั้งต้น

## 2 · เลขคณิตหน่วยความจำ: full vs LoRA vs QLoRA

หน่วยความจำสำหรับการเทรนคือ **state + activations**:

```text
state        = weights + gradients + optimizer states       (bytes per parameter depend on the method)
activations  ≈ tokens in flight × hidden × layers × ~34 B    (a rough rule; checkpointing cuts it to ~2 B + one layer)
```

สคริปต์ PyTorch โหลดโมเดลเป็น `bfloat16` และเทรนด้วย `optim="adamw_torch"` ซึ่ง buffer ของ moment สองตัว (m, v) ใช้ dtype เดียวกับพารามิเตอร์ ดังนั้น full fine-tune จึงใช้ **8 ไบต์ต่อพารามิเตอร์**: weight 2 + gradient 2 + m 2 + v 2 สูตรที่เก็บ weights "master" แบบ fp32 และ moment แบบ fp32 จะต้องใช้ 16 ส่วน LoRA ตรึง (freeze) โมเดลฐานไว้ (2 B/param ไม่มี gradient ไม่มี optimizer) และเทรนพารามิเตอร์เพิ่มเติมเพียง ~0.3% QLoRA ยังเก็บโมเดลฐานที่ถูกตรึงไว้เป็น NF4 แบบ 4 บิตด้วย (~0.52 B/param)

Lab 01 นำหลักนี้ไปใช้กับทุกสูตรในโมดูลนี้:

**Expected output** (เป็นการคำนวณ บันทึกจาก Mac เครื่องนี้; "typ" สมมติว่าตัวอย่าง Alpaca ยาวไม่เกิน 300 token ซึ่งเป็นสมมติฐาน ไม่ใช่การวัด)

```
▣ STEP 3 · the recipes: state + activations (+ 10 GB headroom)
│ recipe                               weights  grads+opt  acts (typ)  acts (cap)  total (typ)  fits on
│ ───────────────────────────────────  ───────  ─────────  ──────────  ──────────  ───────────  ───────────────
│ PyTorch full SFT · Llama 3.2 3B         6.4     19.3        8.3        56.3        44.0 GB    1 Spark
│ PyTorch LoRA · Llama 3.1 8B            16.1      0.1       11.9        81.4        38.2 GB    1 Spark
│ PyTorch QLoRA · Llama 3.1 70B          39.7      0.6       54.7       373.5       105.1 GB    1 Spark
│ PyTorch QLoRA +ckpt · Llama 3.1 70B    39.7      0.6        5.0        34.4        55.4 GB    1 Spark
│ PyTorch LoRA (FSDP) · Llama 3.1 70B   141.3      0.6        2.5        17.2       154.5 GB    2 Sparks (FSDP)
│ NeMo full SFT · Qwen3 8B               16.4     49.1        5.8         5.8        81.3 GB    1 Spark
│ NeMo LoRA · Llama 3.1 8B               16.1      0.1        5.1         5.1        31.3 GB    1 Spark
…
▣ STEP 4 · what checkpointing and micro-batch size do to activations (Llama 3.1 70B QLoRA)
│ batch 8 × 2048 tokens · checkpointing off  ██████████████████████████░░  373.5 GB
│ batch 8 ×  300 tokens · checkpointing off  ████░░░░░░░░░░░░░░░░░░░░░░░░   54.7 GB
│ batch 8 × 2048 tokens · checkpointing on   ██░░░░░░░░░░░░░░░░░░░░░░░░░░   34.4 GB
│ batch 1 × 2048 tokens · checkpointing on   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░    4.3 GB
```

สิ่งที่ได้เรียนรู้จากตารางนี้:

1. **สำหรับสูตรส่วนใหญ่ weights ใส่ได้ ตัวที่บานปลายคือ activations** `--seq_length 2048` เป็นแค่เพดาน: สคริปต์ pad แต่ละ batch ให้ยาวเท่าตัวอย่างที่ยาวที่สุดใน batch นั้นเท่านั้น และตัวอย่าง Alpaca ก็สั้น นี่คือเหตุผลที่คู่มือ benchmark ของ NVIDIA รันค่าตั้งต้นของ 70B QLoRA บน Spark เครื่องเดียวได้ ถ้าชี้สคริปต์เดียวกันไปที่เอกสารยาว ๆ ของคุณเอง คอลัมน์ "cap" คือสิ่งที่คุณจะได้
2. **`--gradient_checkpointing` คือคันโยกตัวใหญ่** มันคำนวณ activations ใหม่ใน backward pass แทนการเก็บไว้: ใช้หน่วยความจำสำหรับ activations น้อยลง ~10 เท่า แลกกับการประมวลผลเพิ่มขึ้น ~30% (เป็นหลักคิดทั่วไป) สคริปต์ 3B full และ 70B QLoRA รับ flag นี้ (ปิดไว้โดยค่าตั้งต้น) สคริปต์ 70B LoRA เปิดไว้โดยค่าตั้งต้น ส่วนสคริปต์ 8B LoRA ไม่มี flag นี้
3. **full fine-tuning โมเดล 8B ใส่ใน Spark เครื่องเดียวได้** ที่ 8 B/param (state 64 GB) แต่ใส่ไม่ได้ถ้าใช้ master weights แบบ fp32 (state อย่างเดียวก็ 128 GB แล้ว)
4. **Llama 3.1 70B LoRA แบบ bf16 ต้องใช้ Spark สองเครื่อง**: weights ที่ถูกตรึงมีขนาด 141 GB ด้วย FSDP แต่ละ Spark ถือครึ่งหนึ่ง นี่คือตัวอย่างแบบหลายเครื่อง (multi-node) ของ playbook (ส่วนที่ 6)

แถวของ NeMo สมมติ 8 B/param เหมือนกัน สูตร YAML เป็นตัวกำหนดการตั้งค่า optimizer จริง จึงให้ถือว่าแถวเหล่านั้นเป็นแค่ภาพร่าง และตรวจ `free -g` ของ lab 03 ในการรันของคุณ

✓ Checkpoint: คุณอธิบายได้ว่าทำไม 70B QLoRA ใส่ใน Spark เครื่องเดียวได้ แต่ 70B LoRA แบบ bf16 ใส่ไม่ได้ และบอกชื่อ flag ที่คุณจะเปิดเป็นอันดับแรกเมื่อข้อมูลของคุณยาวขึ้น

## 3 · PyTorch บน Spark เครื่องเดียว: container, สูตร และ flag

นี่คือขั้นตอนแบบ Spark เครื่องเดียวของ playbook pull และเปิด container โดย mount cache ของ Hugging Face และโฟลเดอร์ปัจจุบันไว้:

```bash
# on: spark
docker pull nvcr.io/nvidia/pytorch:25.11-py3
docker run --gpus all -it --rm --ipc=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "${PWD}:/workspace" -w /workspace \
  nvcr.io/nvidia/pytorch:25.11-py3
```

ภายใน container ให้ติดตั้งชุดเครื่องมือสำหรับเทรนและดาวน์โหลดสูตร:

```bash
# on: spark
pip install transformers peft datasets trl bitsandbytes
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-pytorch-fine-tune/assets
python Llama3_8B_LoRA_finetuning.py --dataset_size 100 --num_epochs 1 --batch_size 2
```

> ⚠ Step 6 ของ playbook บอกให้ `cd client-hardware-playbooks/…` หลัง clone `dgx-spark-playbooks` แต่โฟลเดอร์นั้นไม่มีอยู่จริง ให้ใช้ `dgx-spark-playbooks/nvidia/playbook-pytorch-fine-tune/assets` ตามข้างบน

สคริปต์ทั้งสี่ และ flag ที่แต่ละตัว **กำหนดไว้จริง** (อ่านจากสคริปต์ใน `assets/`; README บอกว่า "all scripts support" (ทุกสคริปต์รองรับ) flag บางตัวที่จริง ๆ มีแค่ในบางสคริปต์):

| สคริปต์ | โมเดล (`--model_name` ค่าตั้งต้น) | วิธีการ | ค่าตั้งต้น | flag เพิ่มเติม |
|---|---|---|---|---|
| `Llama3_3B_full_finetuning.py` | `meta-llama/Llama-3.2-3B-Instruct` | full SFT | batch 8, lr 5e-5 | `--gradient_checkpointing` |
| `Llama3_8B_LoRA_finetuning.py` | `meta-llama/Llama-3.1-8B-Instruct` | LoRA r=8, ทั้ง 7 projection | batch 8, lr 1e-4 | `--lora_rank` |
| `Llama3_70B_LoRA_finetuning.py` | `meta-llama/Llama-3.1-70B-Instruct` | LoRA, พร้อมสำหรับ FSDP | batch 4, checkpointing **เปิด** | `--lora_rank`, `--use_torch_compile`, `--output_dir` |
| `Llama3_70B_qLoRA_finetuning.py` | `unsloth/Meta-Llama-3.1-70B-bnb-4bit` | QLoRA (NF4, double quant) | batch 8 | `--lora_rank`, `--gradient_checkpointing` |

ทั้งสี่ตัวรับ `--dtype`, `--batch_size`, `--seq_length` (2048), `--num_epochs` (1), `--gradient_accumulation_steps`, `--learning_rate`, `--dataset_size` (512; 500 สำหรับ 70B LoRA), `--logging_steps` (1) และ `--log_dir` พฤติกรรมสองอย่างที่ควรรู้:

- สคริปต์ 3B และ 8B จะ `torch.compile` โมเดลเสมอ และรัน **warmup training pass** สั้น ๆ ก่อน แล้วจึงเทรนจริง log ของคุณจะมีบรรทัด loss สองชุด (lab 03 ข้ามชุดแรก)
- มีเพียงสคริปต์ 70B LoRA ที่กำหนด `--output_dir` การรันอื่น ๆ จึงไม่บันทึกอะไรเลย: มันพิสูจน์ว่า pipeline ทำงานได้และให้กราฟ loss กับคุณ Module 13 จะบันทึก merge และ serve โมเดลที่ fine-tune แล้ว

✓ Checkpoint: คุณบอกชื่อสคริปต์ PyTorch ตัวเดียวที่บันทึกโมเดลได้ และ flag ที่คุณจะเพิ่มให้สคริปต์ QLoRA เมื่อข้อมูลยาว

## 4 · NeMo AutoModel บน Spark เครื่องเดียว

playbook ของ NeMo ตรวจ host, pull container ของมัน แล้วเปิดเชลล์ข้างใน:

```bash
# on: spark
nvcc --version && python3 --version && nvidia-smi && free -h && docker ps
docker pull nvcr.io/nvidia/nemo-automodel:26.02
docker run \
  --gpus all \
  --ulimit memlock=-1 \
  -it --ulimit stack=67108864 \
  --entrypoint /usr/bin/bash \
  --rm nvcr.io/nvidia/nemo-automodel:26.02
```

ข้างใน ทุกการรันคือ `examples/llm_finetune/finetune.py` + สูตร YAML + การ override ตัวอย่างสามตัวของ playbook (แต่ละตัวหยุดหลัง 20 step):

```bash
# on: spark
cd /opt/Automodel
ls examples/llm_finetune/
export HF_TOKEN=<your_huggingface_token>      # the playbook's way; lab 02 mounts your cached login instead

# LoRA on Llama 3.1 8B
python3 examples/llm_finetune/finetune.py \
-c examples/llm_finetune/llama3_2/llama3_2_1b_squad_peft.yaml \
--model.pretrained_model_name_or_path meta-llama/Llama-3.1-8B \
--packed_sequence.packed_sequence_size 1024 \
--step_scheduler.max_steps 20

# QLoRA on Meta-Llama-3 70B
python3 examples/llm_finetune/finetune.py \
-c examples/llm_finetune/llama3_1/llama3_1_8b_squad_qlora.yaml \
--model.pretrained_model_name_or_path meta-llama/Meta-Llama-3-70B \
--loss_fn._target_ nemo_automodel.components.loss.te_parallel_ce.TEParallelCrossEntropy \
--step_scheduler.local_batch_size 1 \
--packed_sequence.packed_sequence_size 1024 \
--step_scheduler.max_steps 20

# full SFT on Qwen3 8B
python3 examples/llm_finetune/finetune.py \
-c examples/llm_finetune/qwen/qwen3_8b_squad_spark.yaml \
--model.pretrained_model_name_or_path Qwen/Qwen3-8B \
--step_scheduler.local_batch_size 1 \
--step_scheduler.max_steps 20 \
--packed_sequence.packed_sequence_size 1024
```

อ่านการ override ให้เป็นประโยค: *ใช้สูตรนี้ แต่โหลดโมเดลนั้น pack sequence เป็น 1024 token ใช้ micro-batch 1 และหยุดที่ step 20* ไฟล์สูตรกำหนดส่วนที่เหลือ (LoRA rank, learning rate, …) ตัวอย่าง LoRA ตั้งใจใช้สูตรของโมเดล 1B ซ้ำ เพราะ `--model.pretrained_model_name_or_path` เป็นตัวตัดสินว่าจะโหลด weights ตัวไหน **Packing** ต่อตัวอย่างสั้น ๆ เข้าด้วยกันเป็นแถวเต็ม 1024 token จึงไม่เสียการประมวลผลไปกับ padding

เมื่อการรันจบ playbook ตรวจ checkpoint:

```bash
# on: spark
ls -lah checkpoints/LATEST/
```

**Expected output** (REFERENCE — ยกมาจาก playbook; user และ group เป็นค่า placeholder ของ playbook)

```
total 32K
drwxr-xr-x 6 username domain-users 4.0K Oct 16 22:33 .
drwxr-xr-x 4 username domain-users 4.0K Oct 16 22:33 ..
-rw-r--r-- 1 username domain-users 1.6K Oct 16 22:33 config.yaml
drwxr-xr-x 2 username domain-users 4.0K Oct 16 22:33 dataloader
drwxr-xr-x 2 username domain-users 4.0K Oct 16 22:33 model
drwxr-xr-x 2 username domain-users 4.0K Oct 16 22:33 optim
drwxr-xr-x 2 username domain-users 4.0K Oct 16 22:33 rng
-rw-r--r-- 1 username domain-users 1.3K Oct 16 22:33 step_scheduler.pt
```

> ⚠ container ของ playbook รันด้วย `--rm` และเขียน `checkpoints/` ไว้ข้างใน: พอออกจากเชลล์ checkpoint ก็หายไป Lab 02 mount `~/w25/nemo-checkpoints` ไว้ที่ `/opt/Automodel/checkpoints` เพื่อให้มันยังอยู่ `checkpoints/LATEST/model` คือสิ่งที่ขั้นตอนสุดท้าย (ไม่บังคับ) ของ playbook อัปโหลด: `hf upload my-cool-model checkpoints/LATEST/model`

✓ Checkpoint: คุณบอกได้ว่า `--packed_sequence.packed_sequence_size 1024` เปลี่ยนอะไร และ checkpoint ของ NeMo ไปอยู่ที่ไหนเมื่อคุณใช้ lab 02

## 5 · เปิดงานในเบื้องหลัง แล้วเฝ้าดู log

Lab Runner หยุดแล็บแบบ foreground หลัง 900 วินาที แต่การ fine-tune ใช้เวลานานกว่านั้น lab 02 จึงเปิดงานด้วย `nohup … > ~/w25/logs/<job>.log 2>&1 &` แล้วกลับมาทันที มันรัน `docker run` ของ playbook ในแบบที่ไม่ต้องโต้ตอบ (ไม่มี `-it`) พร้อม `--name` เพื่อให้คุณหยุดมันได้ ก่อนอื่นมันตรวจ Docker, ตรวจว่าไม่มีงานอื่นของคอร์สรันอยู่ (ทุกงานใช้ 128 GB ร่วมกัน) และตรวจว่ามีการล็อกอิน Hugging Face อยู่ (มันไม่เคยพิมพ์โทเค็นออกมา)

```bash
# on: laptop
.venv/bin/python week25/11_pytorch_nemo_two_sparks/labs/lab02_launch_job.py
.venv/bin/python week25/11_pytorch_nemo_two_sparks/labs/lab02_launch_job.py --recipe nemo-lora-8b
```

สูตรที่มี: `pytorch-lora-8b` (ค่าตั้งต้น: ตัวอย่างการใช้งานของ playbook), `pytorch-full-3b`, `pytorch-qlora-70b`, `nemo-lora-8b`, `nemo-qlora-70b`, `nemo-sft-qwen3-8b`

**Expected output** (โหมด DRY บันทึกจาก Mac เครื่องนี้: คำสั่งที่มันจะรัน)

```
▣ STEP 2 · start it in the background
◆ Course deviations from the playbooks, on purpose: no `-it` (nothing is attached), a --name so you can stop it, nohup + a log file. The playbook clones the recipes inside the container; this clones them once to ~/w25 on the Spark and mounts them.
$ mkdir -p ~/w25/logs   [DRY]
  [ -d ~/w25/dgx-spark-playbooks/.git ] || { rm -rf ~/w25/dgx-spark-playbooks; git clone --depth 1 https://github.com/NVIDIA/dgx-spark-playbooks ~/w25/dgx-spark-playbooks; }
  cd ~/w25/dgx-spark-playbooks/nvidia/playbook-pytorch-fine-tune/assets
  nohup docker run --gpus all --rm --ipc=host --name w25-m11-pytorch-lora-8b \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "${PWD}:/workspace" -w /workspace \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    bash -c 'pip install "transformers>=4.57.1,<5" "trl>=0.25.1,<0.26" "peft<0.18" datasets "bitsandbytes>=0.48.2" && pip uninstall -y torchao && python Llama3_8B_LoRA_finetuning.py --dataset_size 100 --num_epochs 1 --batch_size 2' \
    > ~/w25/logs/m11_pytorch-lora-8b.log 2>&1 &
  echo "started w25-m11-pytorch-lora-8b → ~/w25/logs/m11_pytorch-lora-8b.log"
```

เฝ้าดูด้วยมือใน ⌨ terminal (Spark A) หรือด้วย lab 03:

```bash
# on: spark
docker ps --filter name=w25-m11
tail -f ~/w25/logs/m11_pytorch-lora-8b.log
```

Lab 03 อ่าน 400 บรรทัดสุดท้ายของ log ล่าสุด และ parse รูปแบบของ trainer ทั้งสองแบบ: dict ต่อ step ของ Hugging Face / TRL (`{'loss': …, 'grad_norm': …, 'learning_rate': …, 'epoch': …}`) บวกบล็อก `TRAINING COMPLETED` ของสคริปต์ และบรรทัดแบบ NeMo `step N … loss X` มันวาดกราฟ loss แจ้งเตือนเมื่อเจอ NaN หรือ loss ที่ไม่ลดลง ตรวจจับ error `out of memory` หรือ error ของโมเดลแบบ gated และแสดง `free -g` กับการใช้ GPU

```bash
# on: laptop
.venv/bin/python week25/11_pytorch_nemo_two_sparks/labs/lab03_watch_log.py
```

**Expected output** (โหมด DRY บันทึกจาก Mac เครื่องนี้ log ที่มัน parse เป็น EXAMPLE: รูปแบบบรรทัดเป็นไปตามคำสั่ง `print()` ของสคริปต์ ทุกตัวเลขเป็นตัวอย่างประกอบ ไม่ใช่การรันบน Spark)

```
▣ STEP 3 · parse it
◆ trainable parameters: 20,971,520  (= lab 01's count for Llama 3.1 8B at r=8)
◆ 1 warmup pass(es) for torch.compile() skipped (3 steps) — the scripts train once to compile, then again for real
│ epoch 0.02  loss  1.6801  ████████████████████████████
│ epoch 0.04  loss  1.6135  █████████████████████████░░░
…
│ epoch 1.00  loss  1.1876  █████░░░░░░░░░░░░░░░░░░░░░░░
✓ loss is finite and falling: first 2 avg 1.647 → last 2 avg 1.194 (27% lower)
│ runtime  samples/s  steps/s  train loss
│ ───────  ─────────  ───────  ──────────
│ 180.0 s  0.556      0.278    1.2764
◆ finished: yes — TRAINING COMPLETED
…
▣ STEP 5 · parser self-test — runs on every machine
✓ HF/TRL format: 3 warmup + 13 training steps, trainable count, summary and TRAINING COMPLETED all found
✓ NeMo-style `step N | loss X` lines parsed (3 steps)
```

บรรทัดจำนวนพารามิเตอร์ที่เทรนได้ (trainable parameters) คือตัวเลขเดียวที่คุณตรวจได้ก่อนเริ่มเทรน: สำหรับ Llama 3.1 8B ที่ rank 8 บนทั้งเจ็ด projection ต้องได้ 20,971,520 พอดี (32 layer × 8 × ผลรวมของขนาด in + out ของแต่ละ projection) ถ้าของคุณต่างออกไป แปลว่า LoRA ไปจับ module อื่นที่ไม่ใช่ตัวที่คุณคิด

> 💡 รูปแบบ log ของ NeMo AutoModel เปลี่ยนไปในแต่ละ release และ playbook ไม่ได้แสดงไว้เลย parser ของ NeMo จึงมองหาแค่ token `step` และ `loss` ถ้า lab 03 หา step ใน log ของ NeMo ไม่เจอ ให้เปิด log แล้วปรับ regex ใน `parse_log()`

✓ Checkpoint: ในโหมด LIVE lab 03 แสดงจำนวนพารามิเตอร์ที่เทรนได้ของการรันของคุณ loss ที่ลดลง และ `TRAINING COMPLETED` (PyTorch) หรือ 20 step (NeMo)

## 6 · Spark สองเครื่อง: Docker Swarm + Accelerate + FSDP

แท็บ **Multi-node fine-tuning** ของ playbook PyTorch ขยายสคริปต์ชุดเดียวกันไปบน Spark สองเครื่อง ชิ้นส่วนต่าง ๆ:

| ชิ้นส่วน | หน้าที่ |
|---|---|
| **Docker Swarm** | รัน container PyTorch หนึ่งชุดบน Spark แต่ละเครื่อง (`docker-compose.yml`, `replicas: 2`, `NVIDIA_GPU` เครื่องละหนึ่ง, host network) |
| **pytorch-ft-entrypoint.sh** | เปิด sshd ในแต่ละ container (พอร์ต 2233) เพื่อให้ container ติดต่อกันได้ |
| **Accelerate** | เปิด training process หนึ่งตัวต่อ Spark จากไฟล์ config (`machine_rank`, `main_process_ip`, `main_process_port`) |
| **FSDP** | แบ่ง (shard) weights, gradients และ optimizer states ไปยังสอง process |
| **NCCL** (Module 02) | ย้าย shard ผ่านลิงก์ QSFP: `NCCL_SOCKET_IFNAME=enp1s0f1np1` ในไฟล์ compose |

Lab 04 ตรวจ Spark ทั้งสองเครื่อง (อ่านอย่างเดียว) เขียนไฟล์ Accelerate สองไฟล์จาก config ของ playbook เอง แล้วพิมพ์ส่วนที่เหลือ:

```bash
# on: laptop
.venv/bin/python week25/11_pytorch_nemo_two_sparks/labs/lab04_two_spark_fsdp.py              # 70B / 8B LoRA
.venv/bin/python week25/11_pytorch_nemo_two_sparks/labs/lab04_two_spark_fsdp.py --config full # 3B full SFT
```

**Expected output** (โหมด DRY บันทึกจาก Mac เครื่องนี้: แถว preflight เป็นรูปแบบ EXAMPLE ส่วนไฟล์เป็นของจริง)

```
│ node     enp1s0f1np1     swarm     GPU UUID  daemon.json NVIDIA_GPU  swarm-resource  recipes
│ ───────  ──────────────  ────────  ────────  ──────────────────────  ──────────────  ───────
│ Spark A  192.168.100.10  inactive  ✓         ✕ step 3                ✕ step 3        ✓
│ Spark B  192.168.100.11  inactive  ✓         ✕ step 3                ✕ step 3        ✓

▣ STEP 2 · write config_fsdp_lora.yaml for each Spark (rank 0 = Spark A, the primary)
── config_fsdp_lora.spark-a.yaml: machine_rank: 0 · main_process_ip: 192.168.100.10 · main_process_port: 29500
── config_fsdp_lora.spark-b.yaml: machine_rank: 1 · main_process_ip: 192.168.100.10 · main_process_port: 29500
✓ both files parse, num_machines 2, ranks 0 and 1, the same main_process_ip on both
```

พอร์ต 29500 เป็นตัวเลือกของคอร์ส พอร์ตว่างใดก็ได้บน Spark A ใช้ได้ จากนั้นทำตาม playbook ทีละขั้นใน ⌨ terminal

**Step 1 — IP ของ interconnect** (บน Spark แต่ละเครื่อง):

```bash
# on: spark
ip -br -4 address
export MN_IF_NAME="enp1s0f1np1"
export MN_IP_ADDRESS="$(ip -4 addr show "$MN_IF_NAME" | awk '/inet / {print $2}' | cut -d/ -f1)"
echo "$MN_IF_NAME $MN_IP_ADDRESS"
```

**Steps 2–3 — ให้ Swarm แจกจ่าย GPU** (บน Spark **ทั้งสอง** เครื่อง; ต้องใช้ sudo) หา UUID แล้วแก้ `/etc/docker/daemon.json` ให้มีบล็อกของ playbook พร้อม UUID **ของคุณ**:

```bash
# on: spark
nvidia-smi -a | grep UUID
sudoedit /etc/docker/daemon.json
```

```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia",
  "node-generic-resources": [
    "NVIDIA_GPU=GPU-45cbf7b3-f919-7228-7a26-b06628ebefa1"
  ]
}
```

```bash
# on: spark
sudo sed -i 's/^#\s*\(swarm-resource\s*=\s*".*"\)/\1/' /etc/nvidia-container-runtime/config.toml
sudo systemctl restart docker
```

**Steps 4–5 — สร้าง swarm** บน Spark A: `docker swarm init --advertise-addr "$MN_IP_ADDRESS"` มันจะพิมพ์บรรทัด `docker swarm join --token …` ออกมา ให้นำบรรทัดนั้นไปรันบน Spark B

**Step 6 — deploy stack** (Spark A) Spark ทั้งสองเครื่องต้องมีสูตรอยู่ที่ path เดียวกัน lab 02 clone ไว้ที่ `~/w25/dgx-spark-playbooks` (รัน lab 02 บน Spark B หนึ่งครั้งด้วย หรือ `git clone` ที่นั่น):

```bash
# on: spark
cd ~/w25/dgx-spark-playbooks/nvidia/playbook-pytorch-fine-tune/assets
chmod +x pytorch-ft-entrypoint.sh
docker stack deploy -c "$PWD/docker-compose.yml" finetuning-multinode
docker stack ps finetuning-multinode
```

**Expected output** (REFERENCE — "healthy output" ของ playbook; ชื่อ service สะกดว่า `finetunine` จริง ๆ ใน `docker-compose.yml`)

```
ID             NAME                                IMAGE                              NODE         DESIRED STATE   CURRENT STATE
vlun7z9cacf9   finetuning-multinode_finetunine.1   nvcr.io/nvidia/pytorch:25.11-py3   <node-a>     Running         Running
tjl49zicvxoi   finetuning-multinode_finetunine.2   nvcr.io/nvidia/pytorch:25.11-py3   <node-b>     Running         Running
```

**Steps 7–9 — เริ่มเทรน** Lab 04 วางไฟล์ Accelerate ของ Spark แต่ละเครื่องไว้ใน `assets/configs/` ให้แล้ว (โหมด LIVE) บน Spark **แต่ละ** เครื่อง ให้ export โทเค็นในเชลล์ แล้วเปิด Accelerate ใน container ของ Spark เครื่องนั้น:

```bash
# on: spark
export FINETUNING_CONTAINER=$(docker ps -q -f name=finetuning-multinode)
export HF_TOKEN=<your-huggingface-token>
docker exec \
  -e HF_TOKEN="$HF_TOKEN" \
  -it "$FINETUNING_CONTAINER" bash -c '
  bash /workspace/install-requirements;
  accelerate launch --config_file=/workspace/configs/config_fsdp_lora.yaml /workspace/Llama3_70B_LoRA_finetuning.py'
```

> 💡 สคริปต์ 70B ใช้ `meta-llama/Llama-3.1-70B-Instruct` เป็นค่าเริ่มต้น ถ้า Spark ของคุณมี Llama 3.3 70B อยู่ใน cache แล้ว (Spark ในห้องเรียนมี) ให้เติม `--model_name meta-llama/Llama-3.3-70B-Instruct` ต่อท้ายชื่อสคริปต์ สถาปัตยกรรมเดียวกัน และไม่ต้องดาวน์โหลดอีก ~140 GB บน Spark แต่ละเครื่อง Lab 04 พิมพ์คำสั่งในรูปแบบนี้ให้แล้ว

helper `run-multi-llama_3b`, `_8b` และ `_70b` ใน `assets/` คือคำสั่งนี้พอดีสำหรับแต่ละสคริปต์ Accelerate เปิดหนึ่ง process ต่อเครื่อง และแต่ละเครื่องอ่าน `machine_rank` ของตัวเอง นี่คือเหตุผลที่ต้องรันคำสั่งบน Spark ทั้งสองเครื่อง (Step 9 ของ playbook ไม่ได้บอกไว้ชัดเจน) ความคืบหน้าจะพิมพ์บน Spark A เท่านั้น ส่วน Spark B ให้ดูด้วย `nvidia-smi`

**Step 10 — เก็บกวาด** (Spark A): `docker stack rm finetuning-multinode`

> ⚠ `pytorch-ft-entrypoint.sh` ตั้งรหัสผ่าน root ของ container เป็น `root` เปิดให้ล็อกอิน SSH ด้วย root ที่พอร์ต 2233 และ mount `~/.ssh` ของคุณแบบอ่านอย่างเดียว บน host network ให้ลบ stack ทิ้งเมื่อเทรนเสร็จ และอย่าเปิดทิ้งไว้บนเครือข่ายที่ใช้ร่วมกับผู้อื่น

✓ Checkpoint: `docker stack ps finetuning-multinode` แสดง task สองตัวที่ Running อยู่บนสอง node ที่ต่างกัน และ lab 04 พิมพ์ไฟล์ Accelerate ทั้งสองพร้อม rank 0 และ 1

## 7 · Spark สองเครื่องช่วยการเทรนเมื่อไร?

FSDP จ่ายค่าหน่วยความจำที่ลดลงครึ่งหนึ่งด้วยปริมาณการรับส่งข้อมูล (traffic) ในโหมด `FULL_SHARD` Spark แต่ละเครื่องจะรวบรวม weights อีกครึ่งหนึ่งของทุก layer ใน forward pass **และอีกครั้ง** ใน backward pass ขั้นตอนสุดท้ายของ Lab 01 นับปริมาณนี้:

**Expected output** (เป็นการคำนวณ บันทึกจาก Mac เครื่องนี้; link = เกณฑ์ผ่าน 21.875 GB/s ของ Module 02)

```
▣ STEP 5 · two Sparks with FSDP: memory per Spark, and what crosses the cable per forward + backward pass
│ job                                     Accelerate config       state, 1 Spark  state per Spark  link per pass  at 21.875 GB/s
│ ──────────────────────────────────────  ──────────────────────  ──────────────  ───────────────  ─────────────  ──────────────
│ Llama 3.1 70B · LoRA (FSDP FULL_SHARD)  config_fsdp_lora.yaml    141.9 GB         71.0 GB         141.3 GB       6.46 s
│ Llama 3.1 8B · LoRA (FSDP FULL_SHARD)   config_fsdp_lora.yaml     16.2 GB          8.1 GB          16.1 GB       0.74 s
│ Llama 3.2 3B · full SFT (FSDP2)         config_finetuning.yaml    25.7 GB         12.9 GB           6.4 GB       0.29 s
```

| สถานการณ์ | ใช้ | ทำไม |
|---|---|---|
| งานใส่ใน Spark เครื่องเดียวไม่ได้ (70B LoRA แบบ bf16, 8B full SFT ที่มี state แบบ fp32) | **Spark สองเครื่อง, FSDP** | เป็นทางเดียวที่จะรันที่ precision นั้นได้ |
| ใส่ได้หลังเปลี่ยน precision (70B → QLoRA) | **Spark เครื่องเดียว** | weights แบบ 4 บิต ~40 GB ไม่มี traffic บนลิงก์ ราคาที่ต้องจ่ายคือคุณภาพที่ห่างไปเล็กน้อยของ QLoRA |
| ใส่ได้สบาย ๆ (8B LoRA, 3B full) | **Spark เครื่องเดียว** หรือสองเครื่องสำหรับ data parallelism | สองเครื่องจะชนะก็ต่อเมื่อแต่ละ pass ใช้เวลาประมวลผลนานกว่าเวลาบนลิงก์มาก (0.29–0.74 s ในที่นี้) |
| ต้องการ throughput สำหรับการรันเล็ก ๆ จำนวนมาก | **หนึ่งงานต่อ Spark** | ไม่มีลิงก์เลย: ได้การทดลองสองเท่า |

กฎสามข้อที่ได้จากการคำนวณ:

1. **ต้นทุนของลิงก์คิดต่อ pass ไม่ใช่ต่อ optimizer step** gradient accumulation ไม่ได้ลดมัน แต่จำนวน token ต่อ pass ที่มากขึ้น (micro-batch ที่ใหญ่ขึ้น) ลดได้
2. **LoRA ไม่ได้ลด traffic ของ weights ใน FSDP** weights ที่ถูกตรึงยังต้องเดินทางอยู่ดี มีแค่ gradient ของ adapter ขนาดจิ๋วที่ถูก
3. **วัดก่อน แล้วค่อยตัดสินใจ** `steps/s` ของ lab 03 บน Spark เครื่องเดียว เทียบกับงานเดียวกันบนสองเครื่อง จะบอกได้ว่า Spark เครื่องที่สองช่วยให้เร็วขึ้นหรือไม่ ตัวเลข 6.46 s สำหรับ 70B เป็นค่าต่ำสุดที่คิดจาก bandwidth อย่างเดียว

✓ Checkpoint: คุณอธิบายได้ว่าทำไม gradient accumulation จึงไม่ลด traffic บนลิงก์ของ FSDP และเลือกได้ว่าแต่ละแถวของตารางควรใช้ Spark หนึ่งหรือสองเครื่อง

## Labs — รันแล็บได้ที่นี่

**labs/lab01_finetune_memory.py** — การคำนวณ: จำนวนพารามิเตอร์ ขนาด LoRA, training state และ activations ของทุกสูตร จัดวางบน Spark หนึ่งหรือสองเครื่อง พร้อม traffic บนลิงก์ของ FSDP ต่อ pass

**labs/lab02_launch_job.py** — เปิดสูตรของ PyTorch หรือ NeMo AutoModel บน Spark A ในเบื้องหลังด้วย nohup และไฟล์ log หลังจากตรวจ Docker, GPU ที่ว่าง และการล็อกอิน Hugging Face

**labs/lab03_watch_log.py** — tail และ parse log การเทรนบน Spark A: กราฟ loss, ตรวจ NaN และ error, สรุปผล, หน่วยความจำ และการใช้ GPU โดย parser ทดสอบตัวเองได้ทุกเครื่อง

**labs/lab04_two_spark_fsdp.py** — ตรวจความพร้อม (preflight) ของ Spark ทั้งสองเครื่องสำหรับ Docker Swarm + FSDP เขียน config ของ Accelerate สองไฟล์จากไฟล์ของ playbook และพิมพ์คำสั่งสำหรับเริ่มเทรน

Lab 01 รันได้ทุกที่ Lab 02 และ 03 รันแบบ LIVE บน Spark A หรือแบบ DRY Lab 04 ตรวจ Spark ทั้งสองเครื่องและสร้างไฟล์ config จริงได้ทั้งสองโหมด ไม่มีแล็บไหนรัน `sudo` หรือเปลี่ยนการตั้งค่าของ Docker

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 11 — งบประมาณการ fine-tune** เปิด `week25/11_pytorch_nemo_two_sparks/exercises/ex11_finetune_budget.py` ในไฟล์มี `TODO` สามจุด:

1. `lora_params(layers, shapes, r)`: LoRA เพิ่มพารามิเตอร์กี่ตัว
2. `state_gb(frozen_b, trainable_b, frozen_bits)`: training state เป็น GB
3. `per_spark_gb(state, activations, n)`: หน่วยความจำต่อ Spark เมื่อ FSDP shard ข้าม Spark *n* เครื่อง

ตัวตรวจทำงานแบบออฟไลน์และฟรี

```bash
# on: laptop
.venv/bin/python week25/11_pytorch_nemo_two_sparks/exercises/ex11_finetune_budget.py
```

**Expected output** (เมื่อทำ TODO ครบทั้งสามจุด บันทึกจาก Mac เครื่องนี้)

```
✓ lora_params: Llama 3.1 8B r=8 = 20,971,520 · Llama 3.1 70B r=8 = 103,546,880
✓ state_gb: 3B full = 25.7 GB · 8B LoRA = 16.2 GB · 70B QLoRA (linear layers in NF4) = 36.1 GB
✓ per_spark_gb: 70B LoRA bf16 → 154.4 GB on one Spark · 83.5 GB each on two

▣ your budget, applied (activations from lab 01's 'typ' column)
│ Llama 3.1 8B LoRA          state   16.2 GB · 1 Spark   38.1 GB · 2 Sparks   30.0 GB each → 1 Spark
│ Llama 3.1 70B LoRA, bf16   state  141.9 GB · 1 Spark  154.4 GB · 2 Sparks   83.5 GB each → 2 Sparks (FSDP)
│ Llama 3.1 8B full SFT      state   64.2 GB · 1 Spark   86.1 GB · 2 Sparks   54.0 GB each → 1 Spark
```

<details><summary>คำใบ้ — การนับ LoRA สำหรับ projection หนึ่งตัว</summary>

projection หนึ่งตัว map จาก `d_in` ไป `d_out` LoRA เพิ่ม `A` ที่มี shape `r × d_in` และ `B` ที่มี shape `d_out × r` จึงได้ `r × (d_in + d_out)` พารามิเตอร์ สำหรับ `q_proj` ของโมเดล 8B (4096 → 4096) ที่ r=8 จะได้ 65,536 รวมทั้งเจ็ด projection แล้วคูณด้วยจำนวน layer

</details>

<details><summary>ท้าทายเพิ่ม — rank 64 บน Spark สองเครื่อง</summary>

เปลี่ยน `r` เป็น 64 สำหรับงาน 70B ตอนนี้มีพารามิเตอร์ที่เทรนได้กี่ตัว เพิ่ม state เท่าไร และแผนเปลี่ยนไปหรือไม่? จากนั้นเปลี่ยน activations เป็นค่า "cap" ของ lab 01 (17.2 GB) แล้วตรวจอีกครั้ง

</details>

✓ Checkpoint: บรรทัดตรวจทั้งสามเป็น ✓

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `Cannot access gated repo for URL` | ยอมรับ licence ของโมเดลบน huggingface.co แล้ว `hf auth login` บน Spark อีกครั้ง |
| error เรื่อง gated repo เฉพาะใน container ของ NeMo | image อาจใช้ `HF_HOME` ที่ต่างออกไป: ตรวจด้วย `docker run --rm --entrypoint env nvcr.io/nvidia/nemo-automodel:26.02 \| grep HF_` แล้ว mount cache ไว้ที่นั่น |
| `docker: permission denied` | `sudo usermod -aG docker "$USER" && newgrp docker` |
| container มองไม่เห็น GPU | ตั้งค่า NVIDIA Container Toolkit รีสตาร์ต Docker แล้วทดสอบ `docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.11-py3 nvidia-smi` |
| `CUDA out of memory` | ลด `--batch_size` เพิ่ม `--gradient_checkpointing` (3B full, 70B QLoRA) ลด `--seq_length` ให้สั้นลง; lab 01 แสดงว่าพจน์ไหนใหญ่เกินไป |
| หน่วยความจำไม่พอทั้งที่งานควรใส่ได้ | unified memory: หยุดงานอื่น แล้ว `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| `unrecognized arguments: --use_torch_compile` | มีแค่สคริปต์ 70B LoRA ที่มี flag นี้ (คู่มือ benchmark ส่งให้สคริปต์อื่นด้วย) สคริปต์ 3B และ 8B compile เสมออยู่แล้ว |
| การรัน QLoRA จบด้วย `AttributeError: … 'output_dir'` หลัง `TRAINING COMPLETED` | ในสำเนาที่เราอ่าน สคริปต์ QLoRA ตรวจ `args.output_dir` โดยไม่ได้กำหนด flag นี้ไว้ การเทรนเสร็จแล้ว แต่ไม่มีอะไรถูกบันทึก |
| bitsandbytes CUDA error ใน QLoRA | คู่มือ benchmark ของ NVIDIA ตั้ง `export BNB_CUDA_VERSION=130` ภายใน container |
| Multi-node: error หรือ timeout | `ACCELERATE_DEBUG_MODE=1 ACCELERATE_LOG_LEVEL=DEBUG TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL` |
| `task: non-zero exit (255)` | `docker ps -a --filter "name=finetuning-multinode"` แล้ว `docker logs <container_id>` |
| `Cannot connect to the Docker daemon` หลังเปลี่ยน swarm | `sudo systemctl stop docker && sudo rm -rf /var/lib/docker/swarm && sudo systemctl start docker` แล้ว init swarm ใหม่ด้วย IP ของ interconnect |
| task ใน stack ค้างอยู่ที่ Pending | ตรวจ GPU UUID ใน `daemon.json`, บรรทัด `swarm-resource` และชื่อ interface ใน `docker-compose.yml` (lab 04) |
| ลบ `~/w25/nemo-checkpoints` ไม่ได้ | container เขียนไฟล์ในฐานะ root: `sudo rm -rf ~/w25/nemo-checkpoints` |

## Next — บทถัดไป

ไปต่อที่ [Lab 12 — fine-tune vision-language model และ FLUX.1](../12_vlm_flux_finetune/TUTORIAL.md): แนวคิด fine-tuning แบบเดียวกัน นำมาใช้กับภาพ
