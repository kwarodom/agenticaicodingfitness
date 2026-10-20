# ▶ Spark Lab 10 — Unsloth: LoRA fine-tuning ที่เร็วขึ้น

> ส่วนหนึ่งของ Week 25 · DGX Spark: fine-tune, serve และสร้างเอเจนต์ที่รันใน sandbox คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง ทุกแล็บรันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (อ้างอิงคำต่อคำจาก playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง ข้อความที่ยกมาจาก playbook คงไว้เป็นภาษาอังกฤษตามต้นฉบับ พร้อมคำแปลกำกับ

**สิ่งที่คุณจะได้ลงมือทำ**
- รัน playbook Unsloth ของ NVIDIA บน Spark ของคุณ: container PyTorch ของมัน การ pip install สองชุด และสคริปต์ validate 60 step
- เทรนตัวจัดเส้นทางคำขอของแขกโรงแรมจาก Module 09 อีกครั้ง คราวนี้ด้วย Unsloth โดยใช้ **ข้อมูลเดียวกันและ hyperparameter เดียวกัน** Lab 02 คัดลอกค่าเหล่านี้มาจาก YAML ของ Module 09
- เรียนรู้ว่าอะไรทำให้การเปรียบเทียบ framework สำหรับ fine-tuning สองตัวยุติธรรม แล้วตรวจสอบจริง: การตั้งค่า วิธีจับเวลา และตัวให้คะแนนตัวเดียวสำหรับไฟล์ prediction ทั้งสองไฟล์

**Time** ~45 นาที · **Difficulty** ระดับกลาง · **Hardware** DGX Spark 1 เครื่อง (หรือไม่มีเลยก็ได้: ใช้โหมด DRY + การตรวจ config, การแปลงข้อมูล และการเปรียบเทียบที่รันบนแล็ปท็อป)

**Playbook ทางการที่ครอบคลุม:** [Fine-Tune Faster with Unsloth](https://build.nvidia.com/spark/unsloth) · [Fine-Tune Specialized LLMs with Unsloth](https://build.nvidia.com/spark/fine-tuning)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| รัน lab 02–03 ของ Module 09 บนแล็ปท็อปแล้ว | `ls week25/09_llama_factory/data week25/09_llama_factory/configs` | โมดูลนี้นำชุดข้อมูลและ YAML นั้นกลับมาใช้ |
| Docker ที่รองรับ GPU บน Spark | `docker info --format '{{json .Runtimes}}'` แสดง `nvidia` | playbook รันใน container |
| พื้นที่ว่างบน Spark ~20 GB | `df -h ~` | container image บวกโมเดล |
| ไม่มีงานเทรนอื่นรันอยู่ | `pgrep -af llamafactory-cli`, `docker ps` | เวลาที่วัดได้จะเทียบกันได้ก็ต่อเมื่อรันทีละงาน |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
ls week25/09_llama_factory/data week25/09_llama_factory/configs
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
week25/09_llama_factory/configs:
hotel_chat.yaml
hotel_full_sft.yaml
hotel_lora_sft.yaml
hotel_merge.yaml
hotel_predict.yaml
hotel_qlora_sft.yaml

week25/09_llama_factory/data:
dataset_info.json
hotel_ops.json
hotel_ops_eval.json
```

✓ Checkpoint: มีทั้งสองโฟลเดอร์ ถ้าไม่มี ให้รัน lab 02 และ 03 ของ Module 09 ก่อน ทั้งสองแล็บไม่ต้องใช้ Spark

## 1 · Unsloth คืออะไร และไม่ใช่อะไร

Unsloth เป็นไลบรารี Python สำหรับ fine-tuning แบบ LoRA และ QLoRA มันแทนที่บางส่วนของเส้นทางการเทรนของ Hugging Face ด้วย GPU kernel ของตัวเอง playbook อธิบายว่ามันลด "training time and memory use compared with standard parameter-efficient fine-tuning workflows" (เวลาเทรนและการใช้หน่วยความจำ เมื่อเทียบกับ workflow การ fine-tune แบบ parameter-efficient มาตรฐาน) playbook เวอร์ชันเก่าใส่ตัวเลขไว้ด้วย ("2× faster on single GPU" — เร็วขึ้น 2 เท่าบน GPU ตัวเดียว) ในฐานะคำกล่าวอ้าง โมดูลนี้ให้คุณ **วัด** คำกล่าวอ้างนั้นบน Spark ของคุณเอง กับงานของคุณเอง

| | LLaMA Factory (Module 09) | Unsloth (โมดูลนี้) |
|---|---|---|
| สิ่งที่คุณเขียน | ไฟล์ YAML | สคริปต์ Python สั้น ๆ |
| รันแบบ | `llamafactory-cli train x.yaml` ใน venv | `python script.py` ใน container PyTorch ของ NVIDIA |
| วิธีการ | SFT, DPO, KTO, reward modelling, pre-training; LoRA, QLoRA, full | LoRA, QLoRA, full (`full_finetuning=True`), RL ผ่าน TRL |
| การเร่งความเร็ว | Hugging Face + PEFT มาตรฐาน | kernel ที่เขียนเอง และ `use_gradient_checkpointing="unsloth"` |
| Web UI | LLaMA Board | ไม่มี (ใช้ notebook) |
| การ export | `llamafactory-cli export` → โมเดล Hugging Face ที่ merge แล้ว | `save_pretrained`, merge แบบ 16 บิต, GGUF (ดู Unsloth wiki) |

playbook ภาพรวมคู่กัน [Fine-Tune Specialized LLMs with Unsloth](https://build.nvidia.com/spark/fine-tuning) เป็นแผนที่ของวิธีการต่าง ๆ มากกว่าชุดคำสั่ง ตัวมันเองระบุว่า "Concrete Unsloth install, launch, and training commands are **not included in this playbook**" (คำสั่งติดตั้ง เปิด และเทรน Unsloth แบบเจาะจง **ไม่ได้รวมอยู่ใน playbook นี้**) คำแนะนำของมัน ยกมาตรง ๆ: LoRA/QLoRA เหมาะกับ "Small- to medium-sized dataset (about 100–1,000 prompt–sample pairs)" (ชุดข้อมูลขนาดเล็กถึงกลาง ราว 100–1,000 คู่) full fine-tuning เหมาะกับ "Large dataset (1,000+ prompt–sample pairs)" (ชุดข้อมูลขนาดใหญ่ 1,000+ คู่) และ reinforcement learning ต้องมี "An action model, a reward model, and an environment" (action model, reward model และ environment) ชุดข้อมูลโรงแรมมี 504 คู่ จึงใช้ LoRA อีกครั้ง

✓ Checkpoint: คุณบอกได้หนึ่งอย่างที่ Unsloth เปลี่ยน (kernel) และหนึ่งอย่างที่ต้อง **ไม่** เปลี่ยนถ้าจะเทียบกับ Module 09 (ข้อมูล โมเดล หรือ hyperparameter)

## 2 · container และการรัน validate ของ playbook

[playbook](https://build.nvidia.com/spark/unsloth) pull container PyTorch ของ NVIDIA เปิดแบบโต้ตอบ ติดตั้งแพ็กเกจสองชุดข้างใน ดาวน์โหลด `test_unsloth.py` แล้วรัน:

```bash
# on: spark
nvcc --version                                  # Step 1: expect CUDA 13.0
nvidia-smi
docker pull nvcr.io/nvidia/pytorch:25.11-py3    # Step 2
docker run --gpus all --ulimit memlock=-1 -it --ulimit stack=67108864 --entrypoint /usr/bin/bash --rm nvcr.io/nvidia/pytorch:25.11-py3   # Step 3
```

```bash
# on: spark
# Steps 4–6, inside the container
pip install transformers peft hf_transfer "datasets==4.3.0" "trl==0.26.1"
pip install --no-deps unsloth unsloth_zoo bitsandbytes
curl -O https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-unsloth/assets/test_unsloth.py
python test_unsloth.py
```

`test_unsloth.py` โหลด `unsloth/Phi-3.5-mini-instruct` แบบ 4 บิต เพิ่ม LoRA adapter (`r = 16` บน linear layer ทั้ง 7 ตัว) แล้วรัน 60 step (`max_steps = 60`, batch 2 × accumulation 4, `optim = "adamw_8bit"`) บน `unified_chip2.jsonl` ของ OIG จาก LAION

container แบบโต้ตอบ `-it` ใช้ภายใต้ `nohup` ไม่ได้ เพราะไม่มีเทอร์มินัล `spark/run_unsloth.sh` จึงรัน image, ulimit และคำสั่ง pip ชุดเดียวกันแบบ **headless** (ไม่มีหน้าจอโต้ตอบ) โดยเปลี่ยนสี่อย่าง ซึ่งระบุไว้ทั้งหมดในส่วนหัวของสคริปต์: ไม่มี `-it`, มี `--name` เพื่อให้ `docker ps` แสดงงาน, mount `~/w25/m10` สำหรับสคริปต์และผลลัพธ์ และ mount `~/.cache/huggingface` (แบบเดียวกับ `launch.sh` ของ playbook VLM) เพื่อให้ดาวน์โหลดโมเดลแค่ครั้งเดียว Lab 01 ตรวจสิ่งที่ต้องมีก่อน (prerequisite) และ image แล้วรัน validate เมื่อคุณยินยอม:

```bash
# on: laptop
.venv/bin/python week25/10_unsloth/labs/lab01_unsloth_validate.py
SPARK_APPLY=1 .venv/bin/python week25/10_unsloth/labs/lab01_unsloth_validate.py   # pull, then run (re-run to follow)
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ ในโหมด DRY)

```
│ prerequisite              result     last line
│ ────────────────────────  ─────────  ────────────────────────────────────────────
│ CUDA 13.0 toolkit         ◈ example  Cuda compilation tools, release 13.0, V13.0.
│ GPU visible               ◈ example  NVIDIA GB10, 580.95.05
│ Docker without sudo       ◈ example  28.3.3
│ NVIDIA container runtime  ◈ example  "nvidia"

▣ STEP 2 · the container image (playbook Step 2: docker pull nvcr.io/nvidia/pytorch:25.11-py3)
…
$ mkdir -p ~/w25/logs && { nohup docker pull nvcr.io/nvidia/pytorch:25.11-py3 > ~/w25/logs/m10_pull.log 2>&1 < /dev/null & }; sleep 2; tail -2 ~/w25/logs/m10_pull.log   [not run]
…
▣ STEP 4 · read the validation log (playbook Step 6: expected output)
$ tail -c 30000 ~/w25/logs/m10_validate.log 2>/dev/null || echo 'no log yet'   [DRY]
◈ REFERENCE — what the playbook says to expect (not your machine):
Expected output in the terminal window:
- "Unsloth: Will patch your computer to enable 2x faster free finetuning"
- Training progress bars showing loss decreasing over 60 steps
- Final training metrics showing completion
```

(ข้อความ patch ที่ยกมานั้นมาจาก playbook เวอร์ชันเก่า `nvidia/unsloth/README.md` เวอร์ชันปัจจุบันอธิบายว่าเป็น "A message that Unsloth will patch the environment for faster fine-tuning" (ข้อความว่า Unsloth จะ patch environment เพื่อให้ fine-tune ได้เร็วขึ้น)) เมื่อเชื่อมต่อ Spark แล้ว step 4 จะแสดงตาราง loss และ `train_runtime` ของการรันของคุณเอง และพิมพ์ `validation finished` เมื่องานจบ

✓ Checkpoint: ในโหมด LIVE lab 01 พิมพ์ `validation finished — Unsloth works in this container` ในโหมด DRY คุณบอกได้สี่ข้อว่า `run_unsloth.sh` ต่างจาก `docker run` แบบโต้ตอบของ playbook อย่างไร และทำไมแต่ละข้อจึงจำเป็น

## 3 · งานโรงแรมเดิม บน Unsloth

การเปรียบเทียบจะมีค่าก็ต่อเมื่อการรันทั้งสองต่างกันแค่ **หนึ่ง** อย่าง คือ framework ดังนั้น lab 02 จึงไม่ให้คุณพิมพ์ hyperparameter เอง มันอ่าน `hotel_lora_sft.yaml` ของ Module 09 แล้วจับคู่แต่ละ key เข้ากับชื่อใน Unsloth/TRL:

```bash
# on: laptop
.venv/bin/python week25/10_unsloth/labs/lab02_hotel_unsloth.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
▣ STEP 1 · derive the Unsloth config from Module 09's YAML
│ LLaMA Factory key           value                        Unsloth / TRL key                           value
│ ──────────────────────────  ───────────────────────────  ──────────────────────────────────────────  ───────────────────────────
│ model_name_or_path          Qwen/Qwen3-4B-Instruct-2507  model_name                                  Qwen/Qwen3-4B-Instruct-2507
│ cutoff_len                  1024                         max_seq_length                              1024
│ quantization_bit            — (bf16)                     load_in_4bit                                False
│ lora_rank / lora_alpha      16 / 32                      r / lora_alpha                              16 / 32
│ lora_dropout                0.05                         lora_dropout                                0.05
│ lora_target                 all                          target_modules                              q k v o gate up down
│ batch × accumulation        2 × 4                        per_device … × gradient_accumulation_steps  2 × 4
│ learning_rate · schedule    0.0001 · cosine              learning_rate · lr_scheduler_type           0.0001 · cosine
│ num_train_epochs · warmup   3.0 · 0.1                    num_train_epochs · warmup_ratio             3.0 · 0.1
│ (prompt masked by default)  answer-only loss             completion_only_loss=True                   answer-only loss
→ wrote 10_unsloth/configs/hotel_unsloth_lora.json

▣ STEP 2 · convert Alpaca records → TRL conversational prompt–completion (answer-only loss)
✓ hotel_ops_chat.jsonl: 504 records, identical text to Module 09's hotel_ops.json (content sha256 d358d3c70ad8)
✓ hotel_ops_eval_chat.jsonl: 60 records, identical text to Module 09's hotel_ops_eval.json (content sha256 0da9728df351)
…
▣ STEP 3 · the training script (adapted from the playbook's test_unsloth.py)
✓ 10_unsloth/spark/train_hotel_unsloth.py compiles (syntax only — it imports unsloth, so it runs on the Spark, not here)
```

รายละเอียดสามข้อที่สำคัญกว่าที่เห็น:

1. **loss เฉพาะคำตอบ (answer-only loss)** โดยค่าตั้งต้น LLaMA Factory คำนวณ loss เฉพาะส่วน response และ mask ส่วน prompt ไว้ Lab 02 เขียนแต่ละเรคคอร์ดเป็น `{"prompt": [system, user], "completion": [assistant]}` ซึ่งเป็นรูปแบบที่ TRL คำนวณ loss เฉพาะ completion สคริปต์ยังตั้ง `completion_only_loss=True` ไว้ด้วย ถ้าคุณเทรนรวม prompt ด้วย loss จะลดลงเร็วจาก system prompt ที่ยาวและซ้ำกัน การรันจะดู "ดีกว่า" ทั้งที่ไม่ได้ดีกว่าจริง (แบบฝึกหัด 10)
2. **โมเดลฐานตัวเดียวกัน** สคริปต์โหลด `Qwen/Qwen3-4B-Instruct-2507` ซึ่งเป็น repo เดียวกับที่ Module 09 ใช้พอดี สคริปต์ของ playbook ใช้โมเดล `unsloth/…-bnb-4bit` ที่ quantize ไว้ล่วงหน้าแทน ซึ่งจะเป็นการรันแบบ QLoRA คือเป็นการทดลองคนละแบบ เพิ่ม `--qlora` ให้ lab 02 ถ้าต้องการการทดลองนั้น
3. **optimizer ตัวเดียวกัน** สคริปต์ของ playbook ใช้ `adamw_8bit` ส่วนสคริปต์โรงแรมใช้ `adamw_torch` คือ AdamW ของ Hugging Face ให้ตรงกับ Hugging Face Trainer ที่ LLaMA Factory ใช้อยู่ สำหรับ LoRA เรื่องนี้แทบไม่มีผลต่อหน่วยความจำ (ดูกราฟใน 📊): optimizer state ของ adapter ขนาด 33 M พารามิเตอร์มีขนาดราว 0.5 GB ไม่ว่าจะแบบไหน

`spark/train_hotel_unsloth.py` คงการเรียกของสคริปต์ playbook ไว้: `FastModel.from_pretrained`, `FastLanguageModel.get_peft_model(… use_gradient_checkpointing="unsloth" …)`, `SFTTrainer(… SFTConfig(…))` ตอนจบมันพิมพ์บรรทัด `W25_RESULT {…}` หนึ่งบรรทัด (runtime, samples/s, train และ eval loss, หน่วยความจำสูงสุด) บันทึก adapter ไว้ที่ `~/w25/m10/saves/qwen3-4b-hotel-unsloth/lora/` และเขียน `generated_predictions.jsonl` สำหรับคำขอ 60 ข้อที่ถูกกันไว้ แบบ greedy ในรูปแบบ `{prompt, predict, label}` ของ LLaMA Factory

เปิดงานเมื่อคุณพร้อม (ทีละงาน: แล็บจะปฏิเสธถ้ามีงานเทรนอื่นรันอยู่):

```bash
# on: laptop
SPARK_APPLY=1 .venv/bin/python week25/10_unsloth/labs/lab02_hotel_unsloth.py
```

คำสั่งที่มันรันบน Spark:

```bash
# on: spark
mkdir -p ~/w25/logs && { nohup bash ~/w25/m10/run_unsloth.sh hotel-lora train_hotel_unsloth.py > ~/w25/logs/m10_hotel_lora.log 2>&1 < /dev/null & }
tail -f ~/w25/logs/m10_hotel_lora.log
```

✓ Checkpoint: บรรทัดการแปลงทั้งสองเป็น ✓ และมีจำนวนเรคคอร์ดเท่ากับ Module 09 ในโหมด LIVE `docker ps` แสดง `w25-unsloth-hotel-lora` และ log เต็มไปด้วยบรรทัด `{'loss': …}`

## 4 · เปรียบเทียบสอง framework อย่างยุติธรรม

"Unsloth เร็วกว่า 2 เท่า" และ "LLaMA Factory ได้ loss ต่ำกว่า" เป็นแค่คำกล่าวอ้าง จนกว่าคุณจะควบคุมปัจจัยอื่นทั้งหมด Lab 03 ทำในสี่ขั้นตอน:

```bash
# on: laptop
.venv/bin/python week25/10_unsloth/labs/lab03_compare_frameworks.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ในโหมด DRY: step 1 เทียบไฟล์ config จริง ส่วนตัวให้คะแนนใน step 4 รันกับ EXAMPLE — prediction สังเคราะห์ที่แก้ด้วยมือ ไม่ใช่ผลลัพธ์ของโมเดล)

```
▣ STEP 1 · is it a fair fight? (from the real config files)
│ what                                      LLaMA Factory (Module 09)             Unsloth (lab 02)                           verdict
│ ────────────────────────────────────────  ────────────────────────────────────  ─────────────────────────────────────────  ──────────────────────────────────
│ base model                                Qwen/Qwen3-4B-Instruct-2507           Qwen/Qwen3-4B-Instruct-2507                ✓ same
│ training data                             hotel_ops (504)                       hotel_ops_chat.jsonl (504)                 ✓ same text (lab 02 checked)
│ base precision                            bf16                                  bf16                                       ✓ same
│ LoRA r / alpha / dropout                  16 / 32 / 0.05                        16 / 32 / 0.05                             ✓ same
│ LoRA targets                              all                                   7 linear layers                            ✓ same
│ effective batch                           2 × 4                                 2 × 4                                      ✓ same
│ learning rate · schedule · warmup         0.0001 · cosine · 0.1                 0.0001 · cosine · 0.1                      ✓ same
│ epochs · max length                       3.0 · 1024                            3.0 · 1024                                 ✓ same
│ loss on                                   answer only (default)                 answer only (completion_only_loss)         ✓ same
│ optimizer                                 HF Trainer default (see log: optim=)  adamw_torch                                ◆ check the LLaMA Factory log line
│ chat format                               template: qwen3_nothink               the tokenizer's own chat template          ◆ compare one rendered prompt
│ kernels, packing, gradient checkpointing  LLaMA Factory defaults                Unsloth kernels · 'unsloth' checkpointing  ◆ the difference you are measuring
✓ every setting that must match does match

▣ STEP 2 · the measurement protocol
│ 1. Same Spark, one job at a time: lab 02 refuses to launch while another training job runs.
│ 2. Time the training loop, not the download: compare train_runtime, which starts after the model is loaded.
│ 3. Run each framework twice. Report both numbers; if they differ by more than a few %, find out why.
│ 4. Peak memory from the framework itself (torch max_memory_allocated): LLaMA Factory with RECORD_VRAM=1,
│    Unsloth in the W25_RESULT line. On unified memory `free -g` also counts the page cache.
│ 5. Quality on the SAME 60 held-out requests, scored by the SAME code (step 4). Loss values are not
│    comparable across frameworks: chat templates and padding can shift them without changing answers.
…
▣ STEP 4 · one scorer for both: department, priority, urgent recall, reply language
◈ EXAMPLE — synthetic predictions written to test the scorer (5 held-out requests, answers edited by hand).
│ EXAMPLE metric  value
│ ──────────────  ─────
│ n               5
│ valid JSON      80 %
│ department      60 %
│ priority        60 %
│ urgent recall   0 %
│ reply language  80 %
✓ scorer: tolerates ```json fences, fails non-JSON answers, catches a wrong department and a missed urgent
```

แถว ◆ ทั้งสามแถวบังคับให้เท่ากันไม่ได้ คุณจึงต้องรายงานไว้:

- **Optimizer** LLaMA Factory พิมพ์ `TrainingArguments` ของมัน รวมถึง `optim=` ไว้ช่วงต้นของ log ถ้าไม่ใช่ `adamw_torch` ให้จดไว้ข้างผลลัพธ์ของคุณ หรือตั้ง `optim:` ใน YAML แล้วเทรนใหม่
- **รูปแบบแชต** LLaMA Factory สร้าง prompt จาก template `qwen3_nothink` ของตัวเอง ส่วน Unsloth/TRL ใช้ chat template ที่มากับ tokenizer สำหรับโมเดลแบบ instruct ทั้งสองควรได้ข้อความเดียวกัน ให้ตรวจ prompt จากแต่ละฝั่งหนึ่งตัวก่อนจะเชื่อความต่างด้านคุณภาพ
- **Kernel** นี่คือสิ่งที่คุณกำลังวัด

ถ้ามี Spark step 3 จะเติมตารางจากการรันสองครั้ง **ของคุณ** (`train_runtime`, samples/s, หน่วยความจำสูงสุด, loss สุดท้าย) Step 4 ให้คะแนนไฟล์ `generated_predictions.jsonl` ทั้งสองไฟล์: ของ Module 09 มาจาก `lab04_monitor_and_export.py --predict` ส่วนการรันของ Unsloth เขียนไฟล์ของตัวเอง ทั้งสองใช้ greedy decoding (`hotel_predict.yaml` ของ Module 09 ตั้ง `do_sample: false` เพราะ LLaMA Factory สุ่มเป็นค่าตั้งต้น) คะแนนจึงไม่ใช่สัญญาณรบกวน (noise)

✓ Checkpoint: ทุกแถวที่ต้องตรงกันเป็น ✓ และคุณอธิบายได้ว่าทำไม loss สุดท้ายของการเทรนจึงถูกระบุว่า "เทียบกันไม่ได้"

## 5 · จะทำอะไรกับ adapter ของ Unsloth

adapter ใน `~/w25/m10/saves/qwen3-4b-hotel-unsloth/lora/` เป็น PEFT adapter มาตรฐาน (`adapter_config.json` + `adapter_model.safetensors`) ชนิดเดียวกับที่ LLaMA Factory เขียน Module 13 จึง serve ได้ด้วยการรองรับ LoRA ของ vLLM หรือ merge ด้วยวิธีเดียวกัน playbook ชี้ไปที่ [Unsloth wiki](https://github.com/unslothai/unsloth/wiki) สำหรับการบันทึกเป็น GGUF (สำหรับ llama.cpp และ Ollama, Module 04) การเทรนต่อจาก checkpoint, chat template แบบกำหนดเอง และ evaluation loop

ถ้าจะลบสิ่งที่ playbook เพิ่มเข้ามา ให้ออกจาก container (flag `--rm` จะลบมันให้) image จะยังอยู่จนกว่าคุณจะรัน `docker rmi nvcr.io/nvidia/pytorch:25.11-py3` ไม่มีแล็บไหนรันคำสั่งนั้นให้คุณ

✓ Checkpoint: คุณรู้ว่า adapter และไฟล์ prediction ของทั้งสอง framework อยู่ที่ไหนบน Spark และรู้ว่า Module 13 ใช้ตัวไหนก็ได้

## Labs — รันแล็บได้ที่นี่

**labs/lab01_unsloth_validate.py** — prerequisite ของ playbook, container image และการรัน test_unsloth.py 60 step ของ playbook เองแบบ headless ภายใต้ nohup

**labs/lab02_hotel_unsloth.py** — สร้าง config ของ Unsloth จาก YAML ของ Module 09 แปลง 504 เรคคอร์ดชุดเดียวกันเป็นรูปแบบ answer-only แล้วเปิดการรัน (opt-in)

**labs/lab03_compare_frameworks.py** — ตรวจว่าการเปรียบเทียบยุติธรรม พิมพ์วิธีการวัด อ่านตัวเลขของทั้งสองการรัน และให้คะแนนไฟล์ prediction ทั้งสองด้วยตัวให้คะแนนตัวเดียว

Lab 02 step 1–3 และ lab 03 step 1, 2 และ 4 รันบนแล็ปท็อปได้จริง ส่วนที่เหลือสั่งงาน Spark หรือแสดงแผนในโหมด DRY

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 10 — ทำให้การเปรียบเทียบที่ไม่ยุติธรรมกลายเป็นยุติธรรม** เปิด `week25/10_unsloth/exercises/ex10_fair_comparison.py` เพื่อนร่วมทีมรายงานว่า "Unsloth เร็วกว่า 2 เท่าและได้ loss ต่ำกว่า" จากการรันที่ใช้การตั้งค่าจาก notebook มีการตั้งค่าสี่ข้อที่ต่างจาก Module 09 ตัวตรวจอ่าน YAML **จริง** ของ Module 09 แล้วบอกคุณว่าข้อไหนบ้าง:

1. LoRA `r` / `lora_alpha`
2. effective batch (`per_device_train_batch_size × gradient_accumulation_steps`)
3. learning rate
4. loss ถูกคำนวณจากส่วนไหน (`completion_only_loss`)

```bash
# on: laptop
.venv/bin/python week25/10_unsloth/exercises/ex10_fair_comparison.py
```

**Expected output** (เมื่อแก้ TODO ครบทั้งสี่จุด; บันทึกจาก Mac เครื่องนี้)

```
✓ TODO 1: LoRA r / alpha = 16 / 32, same as Module 09
✓ TODO 2: effective batch 8, same as Module 09 → the same 189 optimizer steps
✓ TODO 3: learning rate 0.0001 · cosine · warmup 0.1
✓ TODO 4: loss on the answer only, like LLaMA Factory

▣ now a speed or accuracy difference says something about the framework, not the settings.
│ still compare on the same Spark, one job at a time, two runs each, with one scorer (lab 03).
```

<details><summary>คำใบ้ — ทำไมขนาด batch จึงเปลี่ยน "ความเร็ว"?</summary>

จำนวน optimizer step คือ ⌈จำนวนเรคคอร์ด ÷ effective batch⌉ × จำนวน epoch batch ขนาด 16 ใช้ 96 step แทน 189 และ batch ที่ใหญ่ขึ้นทำให้ GPU ทำงานเต็มกว่า การรันจึงจบเร็วขึ้นด้วยเหตุผลที่ไม่เกี่ยวกับ framework เลย

</details>

<details><summary>ท้าทายเพิ่ม — การทดลอง QLoRA</summary>

รัน `lab02_hotel_unsloth.py --qlora` และ `lab03_config_and_launch.py --method qlora` ของ Module 09 ตอนนี้ทั้งสองโหลดโมเดลฐานแบบ 4 บิต เทียบหน่วยความจำสูงสุดกับการคำนวณของ Module 09 (13 GB สำหรับ QLoRA เทียบกับ 19 GB สำหรับ LoRA บวก activations) และเทียบความแม่นยำบนชุดที่กันไว้กับการรันแบบ bf16 การใช้ 4 บิตทำให้งานนี้เสียอะไรไหม?

</details>

✓ Checkpoint: บรรทัดตรวจทั้งสี่เป็น ✓

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `docker: permission denied` | วิธีแก้ของ playbook: `sudo usermod -aG docker $USER && newgrp docker` |
| container เปิดไม่ขึ้นพร้อม error เรื่อง GPU | วิธีแก้ของ playbook: `nvidia-ctk runtime configure --runtime=docker` แล้วรีสตาร์ต Docker |
| `the input device is not a TTY` | คุณรัน `docker run -it …` ภายใต้ nohup ให้ใช้ `run_unsloth.sh` ซึ่งไม่มี `-it` |
| error ตอน pip หรือ import ของ Unsloth / Triton | คำแนะนำของ playbook: ใช้ container image และคำสั่งติดตั้งของมันให้ตรงทุกตัว แล้วลองใหม่ใน container ใหม่ |
| `TypeError: … unexpected keyword argument` จาก `SFTConfig` หรือ `SFTTrainer` | เวอร์ชันแพ็กเกจต่างจากที่ playbook ปักไว้ ให้คง `"datasets==4.3.0" "trl==0.26.1"`: สคริปต์ของ playbook เองส่ง `max_seq_length` และ `tokenizer` กับเวอร์ชันเหล่านี้พอดี |
| CUDA out of memory | วิธีแก้ของ playbook: ลด `per_device_train_batch_size` หรือ `max_seq_length` หรือใช้โมเดลแบบ 4 บิต ให้ลด batch ใน framework **ทั้งสอง** ไม่อย่างนั้นการเปรียบเทียบจะเสีย |
| หน่วยความจำตึงทั้งที่ `free -g` ยังแสดงพื้นที่ว่าง | page cache ของ UMA: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| Lab 03 แสดง `✕ differs` | คุณเปลี่ยนการตั้งค่าของ framework หนึ่งแต่ไม่ได้เปลี่ยนอีกตัว รัน lab 02 ใหม่ ซึ่งจะสร้างค่าจาก Module 09 ใหม่ |

## Next — บทถัดไป

ไปต่อที่ [Lab 11 — fine-tuning ด้วย PyTorch และ NeMo AutoModel](../11_pytorch_nemo_two_sparks/TUTORIAL.md): fine-tune ด้วย PyTorch ธรรมดาและ NeMo AutoModel บน Spark เครื่องเดียว แล้วขยายไปสองเครื่อง
