# ▶ Spark Lab 12 — fine-tune vision-language model และ FLUX.1

> ส่วนหนึ่งของ Week 25 · DGX Spark: fine-tune, serve และสร้างเอเจนต์ที่รันใน sandbox คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง ทุกแล็บรันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (อ้างอิงคำต่อคำจาก playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง ข้อความที่ยกมาจาก playbook คงไว้เป็นภาษาอังกฤษตามต้นฉบับ พร้อมคำแปลกำกับ

**สิ่งที่คุณจะได้ลงมือทำ**
- จัดชุดข้อมูลภาพและวิดีโอให้อยู่ในโครงสร้างที่ trainer ของ playbook ต้องการพอดี และ validate ก่อนที่ GPU จะได้เห็น
- fine-tune **Qwen2.5-VL-7B** ด้วย **GRPO** (การเทรนแบบใช้รางวัล หรือ reward) บน Spark playbook เริ่มงานจากปุ่มใน Streamlit ส่วนคุณจะรันคำสั่งเดียวกันแบบ headless ภายใต้ `nohup`
- เตรียมชุดข้อมูล Dreambooth LoRA ของ **FLUX.1-dev** (concept, trigger word, caption, `data.toml`) รวมถึง concept โรงแรมของคุณเอง
- เทรน FLUX LoRA บน Spark ด้วย `launch_train.sh` ของ playbook เฝ้าดูจากแล็ปท็อป แล้วสร้างภาพด้วย LoRA นั้นใน ComfyUI

**Time** ลงมือทำ ~60 นาที (บวกการเทรนที่ปล่อยทิ้งไว้หลายชั่วโมง) · **Difficulty** ระดับสูง · **Hardware** DGX Spark 1 เครื่อง (หรือไม่มีเลยก็ได้: ใช้โหมด DRY + แล็บชุดข้อมูลที่รันบนแล็ปท็อป)

**Playbook ทางการที่ครอบคลุม:** [Fine-Tune Vision Language Models](https://build.nvidia.com/spark/vlm-finetuning) · [Fine-Tune FLUX.1 for Custom Image Generation](https://build.nvidia.com/spark/flux-finetuning)

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| clone playbook ไว้ที่ root ของ repo | `ls dgx-spark-playbooks/nvidia/playbook-flux-finetuning` | lab 01 และ 03 อ่านไฟล์ตัวอย่างของ playbook เอง |
| PIL ใน Python ของ repo นี้ | `.venv/bin/python -c "import PIL"` | แล็บชุดข้อมูลวาดและตรวจภาพ |
| `ffprobe` (ไม่บังคับ) | `ffprobe -version` | lab 01 วัดวิดีโอตัวอย่าง |
| Hugging Face token ที่มีสิทธิ์เข้าถึง FLUX.1-dev | ยอมรับเงื่อนไขใน [model card](https://huggingface.co/black-forest-labs/FLUX.1-dev) | playbook ทั้งสองดาวน์โหลดโมเดลที่เป็นแบบ gated หรือต้องใช้โทเค็น |
| บัญชี Kaggle | สำหรับชุดข้อมูลไฟป่า (wildfire) | ข้อมูลสำหรับเทรนของสูตรภาพ |
| พื้นที่ว่างบน Spark ~100 GB | `df -h ~` | container สองตัว, FLUX (~34 GB), Qwen2.5-VL (สองครั้ง ดูส่วนที่ 4) และชุดข้อมูล |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
.venv/bin/python -c "import PIL; print('Pillow', PIL.__version__)"
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
Pillow 12.1.1
```

> 🔐 playbook ทั้งสองส่ง Hugging Face token ของคุณผ่านบรรทัดคำสั่ง: `docker build --build-arg HF_TOKEN=$HF_TOKEN` (VLM) และ `download.sh` ซึ่งอ่าน `$HF_TOKEN` (FLUX) แล็บไม่เคยรันสองคำสั่งนี้ มันพิมพ์ออกมาให้คุณรันเอง **บน Spark** ในเชลล์ของคุณ

✓ Checkpoint: import Pillow ได้ และมี `dgx-spark-playbooks/` อยู่ที่ root ของ repo

## 1 · การ fine-tune ด้านภาพสองแบบ

playbook ทั้งสองเทรนโมเดลที่ทำงานในทิศทางตรงข้ามกัน:

| | Vision-language model (VLM) | FLUX.1 (diffusion) |
|---|---|---|
| Input → output | ภาพหรือวิดีโอ → **ข้อความ** | ข้อความ → **ภาพ** |
| โมเดล | Qwen2.5-VL-7B (ภาพ), InternVL3-8B (วิดีโอ) | FLUX.1-dev ขนาด 12B บวก CLIP-L, T5-XXL และ autoencoder |
| วิธีใน playbook | LoRA + **GRPO** (ภาพ), LoRA SFT (วิดีโอ) | **Dreambooth LoRA** แบบหลาย concept (`network_dim` 256) |
| ข้อมูล | ภาพ / คลิปที่ติด label + `metadata.jsonl` | 5–10 ภาพต่อ concept, trigger word หนึ่งคำต่อ concept |
| Trainer | Unsloth + TRL ใน container `vlm_demo` | `sd-scripts` ของ kohya-ss ใน container `flux-train` |
| ทดลองใช้ใน | Streamlit เทียบข้างกันกับโมเดลฐาน (:8501) | workflow ของ ComfyUI (:8188) |
| เวลา (playbook) | GRPO ~100 step: สูงสุด ~2 ชม.; วิดีโอที่ใช้งานได้: หนึ่งวันหรือมากกว่า | 100 epoch: ~4 ชม.; ใช้งานได้หลัง ~90 นาที |

unified memory คือเหตุผลที่ Spark เครื่องเดียวทำได้ทั้งสองแบบ playbook ของ FLUX เก็บ "the Diffusion Transformer, CLIP text encoder, T5 text encoder, and autoencoder resident while you train" (Diffusion Transformer, CLIP text encoder, T5 text encoder และ autoencoder ไว้ในหน่วยความจำตลอดการเทรน) การล้าง UMA ที่ playbook ทั้งสองแนะนำ (`sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`) มีไว้สำหรับตอนที่ไฟล์ใหญ่เหล่านั้นทิ้ง page cache ไว้เต็ม ให้รันสูตร **เดียว** จากสองสูตรในแต่ละครั้ง

✓ Checkpoint: คุณบอกได้ว่าจะ fine-tune โมเดลไหนในสองตัวนี้เพื่อ *ตรวจ* ห้องพักโรงแรมจากภาพถ่าย และตัวไหนเพื่อ *สร้าง* ภาพล็อบบี้ของโรงแรมคุณ

## 2 · ชุดข้อมูลภาพ: ชื่อโฟลเดอร์คือ label

trainer ของสูตรภาพ `ui_image/src/train_image_vlm.py` โหลดข้อมูลด้วยบรรทัดเดียว `load_dataset(config["data"]["dataset_id"])["train"]` โดย `dataset_id` คือ `data` ตัวโหลด image-folder ของ Hugging Face เปลี่ยน **ชื่อโฟลเดอร์ให้เป็น label** จากนั้น `format_instruction()` จับคู่ label กับคำตอบ: `nowildfire` → "No" อย่างอื่นทั้งหมด → "Yes" ดังนั้นโครงสร้างโฟลเดอร์ *ก็คือ* การติด label:

```text
ui_image/data/
├── train/
│   ├── nowildfire/   *.jpg
│   └── wildfire/     *.jpg
├── valid/ …
└── test/ …
```

Lab 01 ตรวจภาพตัวอย่างของ playbook และสร้างเวอร์ชันโรงแรม: *ห้องนี้พร้อมสำหรับแขกคนถัดไปหรือยัง?* โดยใช้ class `ready` และ `needs_attention` ภาพเป็นรูปวาด PIL ขนาดเล็กแบบ **สังเคราะห์ (synthetic)** มันไม่ได้สอนอะไรโมเดลเลย แต่ช่วยให้คุณฝึกจัดโครงสร้างและการตรวจสอบ

```bash
# on: laptop
.venv/bin/python week25/12_vlm_flux_finetune/labs/lab01_vlm_datasets.py
```

**Expected output** (step 1–3 บันทึกจาก Mac เครื่องนี้)

```
▣ STEP 1 · the playbook's own sample images (ui_image/assets/image_vlm/images/)
│ file              size     mode  bytes
│ ────────────────  ───────  ────  ─────
│ nowildfire/1.jpg  350×350  RGB   18 KB
│ wildfire/2.jpg    350×350  RGB   36 KB
│ wildfire/3.jpg    350×350  RGB   26 KB
│ wildfire/4.jpg    350×350  RGB   15 KB
…
▣ STEP 2 · build the hotel version: room_check/{train,test}/{ready,needs_attention}/ (SYNTHETIC stand-ins)
→ wrote 20 PNGs under week25/12_vlm_flux_finetune/data/room_check
…
▣ STEP 3 · validate with imagefolder's rules — and what the trainer does with the folder names
✓ images: splits ['train'] · the same classes in every split ['nowildfire', 'wildfire']
✓ images: every image opens and converts to RGB
◆ playbook samples: no split folders, so everything is 'train': {'nowildfire': 1, 'wildfire': 3}
✓ room_check: splits ['test', 'train'] · the same classes in every split ['needs_attention', 'ready']
✓ room_check: every image opens and converts to RGB
…
  --- ui_image/src/train_image_vlm.py (format_instruction)
  -    if label == "nowildfire":
  +    if label == "ready":
  -    prompt = "Identify if this region has been affected by a wildfire"
  +    prompt = "Does this hotel room need attention before the next guest arrives"
```

diff สุดท้ายนั้นคือบทเรียน: ถ้าใช้โฟลเดอร์โรงแรมโดยไม่แก้โค้ด `ready` จะกลายเป็น "Yes" สำหรับคำถามเรื่องไฟป่าแบบเงียบ ๆ label จะผิด และไม่มีอะไร crash ให้เห็น

✓ Checkpoint: ทั้งสองโฟลเดอร์ผ่านการ validate และคุณบอกได้ว่าต้องแก้สองบรรทัดไหนใน `train_image_vlm.py` เมื่อใช้ class คู่ใหม่

## 3 · ชุดข้อมูลวิดีโอ: JSON หนึ่งเรคคอร์ดต่อหนึ่งคลิป

สูตรวิดีโอ fine-tune InternVL3-8B ให้แปลงคลิปจากกล้องหน้ารถ (dashcam) เป็น JSON ที่มีโครงสร้าง playbook ให้โครงสร้างไว้ดังนี้:

```text
dataset/
├── videos/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── metadata.jsonl
```

ข้อความใน playbook ไม่ได้ระบุฟิลด์ของ `metadata.jsonl` ไว้ แต่มีอยู่ใน notebook สำหรับเทรน `ui_video/train/video_vlm.ipynb`: `video`, `caption`, `event_type`, `rule_violations`, `intended_action`, `traffic_density`, `scene`, `visibility` และค่าที่อนุญาตอยู่ใน prompt ของมัน Lab 01 (step 4–5) วัดคลิปตัวอย่างของ playbook ด้วย `ffprobe` แล้ว validate ไฟล์ metadata เทียบกับ enum เหล่านั้น:

**Expected output** (step 4–5 บันทึกจาก Mac เครื่องนี้; metadata เป็น EXAMPLE — ข้อมูลสังเคราะห์ที่เขียนขึ้นเพื่อทดสอบ validator)

```
▣ STEP 4 · the playbook's sample videos (ui_video/assets/video_vlm/videos/), probed with ffprobe
│ video  size      fps   duration  frames  12 frames =
│ ─────  ────────  ────  ────────  ──────  ────────────
│ 1.mp4  1280×720  30.0  5.1 s     153     every 12th
│ 2.mp4  1280×720  30.5  5.1 s     155     every 12th
│ 3.mp4  640×368   29.9  5.1 s     152     every 12th

▣ STEP 5 · validate a video metadata.jsonl against the notebook's enums (on an EXAMPLE file)
◈ EXAMPLE — a synthetic metadata.jsonl written to test the validator. It does not describe any real video.
✕ line 3: event_type='crash' not in ['collision', 'near_miss', 'no_incident']
✕ line 3: scene='urban' not in ['Highway', 'Rural', 'Sub-urban', 'Urban']
✕ line 3: rule_violations ['tailgating'] not ⊆ ['failure_to_yield', 'ignoring_traffic_signs', 'speeding']
✕ line 3: empty caption
✕ line 3: video path 'clips/video3.mp4' is not under videos/
✓ validator: lines 1–2 pass; line 3's five mistakes (enum, case, violation, caption, path) are all caught
```

มีรายละเอียดหนึ่งที่รู้ได้จากการอ่านโค้ดของ notebook เท่านั้น `get_video_path()` นำ **โฟลเดอร์แม่ (parent)** ของ `dataset_path` มาต่อกับฟิลด์ `video` ของแต่ละเรคคอร์ด ดังนั้นให้ตั้ง `dataset_path = "/path/to/dataset/metadata.jsonl"` (ตัวไฟล์เอง) และเขียน `"video": "videos/video1.mp4"` playbook บอกว่าคุณภาพวิดีโอที่ใช้งานได้ "often needs a long training window (on the order of a day or more)" (มักต้องเทรนเป็นเวลานาน ระดับหนึ่งวันหรือมากกว่า) ตัวปรับของมันคือ 12–16 เฟรมต่อวิดีโอ การสุ่มตัวอย่างตามเวลาแบบสม่ำเสมอ (uniform temporal sampling) และ LoRA ให้เริ่มจากสูตรภาพก่อน

✓ Checkpoint: คุณบอกชื่อฟิลด์ของ `metadata.jsonl` ได้สามฟิลด์พร้อมค่าที่อนุญาต และรู้ว่า `dataset_path` ต้องชี้ไปที่อะไร

## 4 · รันสูตร VLM บน Spark แบบ headless

เส้นทางของ playbook คือ: clone repo, build image `vlm_demo`, เปิดด้วย `launch.sh`, ดาวน์โหลดโมเดลและชุดข้อมูลจาก Kaggle, รัน `streamlit run Image_VLM.py` แล้วกด **Start Finetuning** ปุ่มนั้นเขียน `ui_image/src/train.yaml` และรัน `python -u src/train_image_vlm.py` Lab 02 ทำสองอย่างเดียวกันผ่าน ssh ภายใต้ `nohup`:

```bash
# on: laptop
.venv/bin/python week25/12_vlm_flux_finetune/labs/lab02_vlm_train_on_spark.py
SPARK_APPLY=1 .venv/bin/python week25/12_vlm_flux_finetune/labs/lab02_vlm_train_on_spark.py --steps 5
```

สิ่งที่คุณรันเองครั้งเดียวบน Spark (นี่คือ Step 2–3 และ 5.1–5.2 ของ playbook โดยแก้ path ของ clone ให้ถูกต้องแล้ว):

```bash
# on: spark
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-vlm-finetuning/assets
export HF_TOKEN=<YOUR_HF_TOKEN>
docker build --build-arg HF_TOKEN=$HF_TOKEN -t vlm_demo .
mkdir -p ui_image/data && cd ui_image/data
# paste the cURL command from the Kaggle "Wildfire Prediction Dataset" page (Download → cURL), then:
unzip -qq wildfire-prediction-dataset.zip
rm wildfire-prediction-dataset.zip
```

สามเรื่องที่ข้อความใน playbook ไม่ได้บอก แต่ lab 02 บอก:

1. **path ของ clone** playbook บอกให้ `cd client-hardware-playbooks/nvidia/playbook-vlm-finetuning/assets` แต่โฟลเดอร์ที่ `git clone` สร้างขึ้นคือ `dgx-spark-playbooks`
2. **โทเค็นของคุณอยู่ใน image** Dockerfile รัน `hf auth login --token $HF_TOKEN` ระหว่าง build ห้าม push `vlm_demo` ขึ้น registry และให้ `docker rmi vlm_demo` เมื่อใช้เสร็จ
3. **ดาวน์โหลดโมเดลสองครั้ง** playbook รัน `hf download Qwen/Qwen2.5-VL-7B-Instruct` แต่ `src/train.yaml` เทรน `unsloth/Qwen2.5-VL-7B-Instruct` ซึ่งเป็นอีก repo หนึ่งที่ Unsloth ดาวน์โหลดตอนรันครั้งแรก

**Expected output** (step 4 บันทึกจาก Mac เครื่องนี้ — อ่านจาก `src/train.yaml` ของ playbook เอง)

```
│ train.yaml key       value
│ ───────────────────  ───────────────────────────────────────────
│ model                unsloth/Qwen2.5-VL-7B-Instruct
│ LoRA rank / alpha    32 / 64
│ steps                5 (the repo ships 5; the UI default is 100)
│ batch · generations  4 · 2
│ rewards              format 2.0 · correctness 5.0
│ max_seq_length       16384
│ output_dir           saved_model
```

**GRPO** ต่างจากการเทรนแบบ supervised ของ Module 09 ไม่มีข้อความคำตอบให้เลียนแบบ สำหรับแต่ละภาพ โมเดลจะเขียนคำตอบ `num_generations` คำตอบ `format_reward_func` ให้ +1 เมื่อมีบล็อก `<REASONING>…</REASONING>` หนึ่งบล็อกพอดี และ +1 เมื่อมีบล็อก `<SOLUTION>…</SOLUTION>` หนึ่งบล็อกพอดี ส่วน `correctness_reward_func` ให้ 5.0 เมื่อคำตอบตรงกับ label จากชื่อโฟลเดอร์ การอัปเดตจะให้น้ำหนักกับคำตอบที่ได้คะแนนดีกว่า repo มากับ `steps: 5` ซึ่งเป็น smoke test ส่วนค่าตั้งต้นใน UI ของ playbook คือ 100 step ซึ่ง "can take up to about 2 hours" (อาจใช้เวลาถึงราว 2 ชั่วโมง)

คำสั่งแบบ headless (การเปลี่ยนแปลงของคอร์ส: ไม่มี `-it` ไม่ mount Docker socket และตั้ง `-w` เป็น `ui_image`):

```bash
# on: spark
cd ~/dgx-spark-playbooks/nvidia/playbook-vlm-finetuning/assets
nohup docker run --gpus=all --net=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm --name w25-vlm-grpo -v $(pwd):/vlm_finetuning -v $HOME/.cache/huggingface:/root/.cache/huggingface -w /vlm_finetuning/ui_image vlm_demo python -u src/train_image_vlm.py > ~/w25/logs/m12_vlm_grpo.log 2>&1 < /dev/null &
```

เมื่อเทรนจบ สคริปต์จะ "merges LoRA weights into the base model" (รวม LoRA weights เข้ากับโมเดลฐาน — playbook) ไว้ที่ `ui_image/saved_model/checkpoint-N` จากนั้นเปรียบเทียบโมเดลฐานกับโมเดลที่ fine-tune แล้วแบบเคียงข้างกันในแอป Streamlit ของ playbook: เปิด container ด้วย `sh launch.sh`, `cd /vlm_finetuning/ui_image`, `streamlit run Image_VLM.py` แล้วเปิด `:8501` ผ่าน SSH tunnel การโหลดครั้งแรกจะเปิดเซิร์ฟเวอร์ vLLM สองตัว และอาจใช้เวลาราว 15 นาที

✓ Checkpoint: ในโหมด LIVE lab 02 แสดงบรรทัด `loss` และ `reward` แล้วตามด้วย `saved_model/checkpoint-N` ในโหมด DRY self-test ของ parser เป็น ✓ และคุณอธิบายฟังก์ชันรางวัล (reward function) ทั้งสองได้

## 5 · ชุดข้อมูล Dreambooth ของ FLUX.1: trigger, caption และ data.toml

Dreambooth LoRA สอน **คำใหม่** ให้ FLUX แต่ละ concept ได้ trigger word ที่หายากหนึ่งคำบวกคำบอกประเภท (class word) (`tjtoy toy`, `sparkgpu gpu`) หลังเทรน การใส่ trigger ใน prompt จะเรียก concept นั้นออกมา playbook แนะนำ "about 5–10 images per concept" (ราว 5–10 ภาพต่อ concept) หนึ่งโฟลเดอร์ต่อ concept ภายใต้ `flux_data/` และรายการ `[[datasets.subsets]]` หนึ่งรายการต่อ concept ใน `flux_data/data.toml`

Lab 03 อ่าน `data.toml` และภาพจริงของ playbook คำนวณจำนวน step ของการเทรน และเพิ่ม concept ที่สามคือ `sparkhotel lobby` ภาพแทนแบบ **สังเคราะห์** ขนาด 1024×1024 หกภาพของมันแต่ละภาพมาพร้อมไฟล์ `.caption` sd-scripts จะอ่าน `<image>.caption` ถ้ามี และใช้ `class_tokens` แทนถ้าไม่มี

```bash
# on: laptop
.venv/bin/python week25/12_vlm_flux_finetune/labs/lab03_flux_dataset.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
▣ STEP 1 · the playbook's dataset: flux_data/data.toml and its images
│ [general]  shuffle_caption = False · keep_tokens = 2
│ [[datasets]] resolution = 1024 · batch_size = 1
│ image_dir           class_tokens    images  num_repeats  flip_aug  sizes             .caption files
│ ──────────────────  ──────────────  ──────  ───────────  ────────  ────────────────  ──────────────
│ flux_data/tjtoy     'tjtoy toy'     6       1            True      742–2034 px wide  0
│ flux_data/sparkgpu  'sparkgpu gpu'  7       2            True      650–3018 px wide  0

▣ STEP 2 · the step arithmetic for the playbook's run (launch_train.sh: accumulation 4, 100 epochs, save every 25)
│ images per epoch = 6 × 1 (tjtoy) + 7 × 2 (sparkgpu) = 20 · batch 1 → 20 forward passes
│ optimizer steps  = ⌈20 ÷ 4⌉ = 5 per epoch × 100 epochs = 500 steps
│ time per step    ≈ 4 h ÷ 500 = 29 s   (derived from the playbook's 'about four hours', not a measurement)
│ LoRA files       = flux_dreambooth-000025.safetensors, flux_dreambooth-000050.safetensors, flux_dreambooth-000075.safetensors, flux_dreambooth.safetensors (final)
…
▣ STEP 4 · validate both data.toml files
✓ playbook data.toml: 2 concepts, every folder has 5–10 images, two-word class_tokens, unique triggers · 20 images/epoch → 500 steps for 100 epochs
✓ course data.toml: 3 concepts, every folder has 5–10 images, two-word class_tokens, unique triggers · 26 images/epoch → 700 steps for 100 epochs
◆ with the hotel concept: ≈ 5.6 h for 100 epochs at the playbook-derived pace, or ≈ 1.4 h with --max_train_epochs=25 (lab 04 --epochs 25).
```

ตัวเลือกสี่อย่างที่ควรทำความเข้าใจ:

- **`num_repeats`** sparkgpu ถูกแสดงสองครั้งต่อ epoch จึงได้รับการดูบ่อยเท่ากับ tjtoy แม้จะเป็น concept ที่ยากกว่า repeat มากขึ้นหมายถึง step มากขึ้น
- **`flip_aug`** การกลับภาพแบบกระจกทำให้ของเล่นมีความหลากหลายเพิ่มเป็นสองเท่า แต่สำหรับอะไรก็ตามที่มีข้อความหรือโลโก้ ให้ตั้งเป็น `false` ซึ่ง subset `sparkhotel` ของคอร์สตั้งไว้แล้ว
- **`keep_tokens = 2`** คงคำสองคำแรกของ caption (trigger + class) ไว้ที่เดิม ถ้าคุณเปิด `shuffle_caption`
- **ชื่อไฟล์** เป็นรูปแบบของ sd-scripts เองคือ `{output_name}-{epoch:06d}` อ่านจากโค้ดของ sd-scripts ไม่ใช่จาก playbook ซึ่งบอกแค่ว่าไฟล์มี prefix `flux_dreambooth`

✓ Checkpoint: ไฟล์ `data.toml` ทั้งสองผ่านการ validate และคุณอธิบายได้ว่าทำไม concept โรงแรมจึงไม่ควรใช้ `flip_aug`

## 6 · เทรน FLUX LoRA บน Spark แล้วสร้างภาพ

ก่อนอื่นคือ Step 3 และ 6 ของ playbook ซึ่งคุณรันบน Spark:

```bash
# on: spark
cd ~/dgx-spark-playbooks/nvidia/playbook-flux-finetuning/assets
export HF_TOKEN=<YOUR_HF_TOKEN>
sh download.sh                                   # ~23.8 GB + ~9.8 GB + ~335 MB + ~246 MB
docker build -f Dockerfile.train -t flux-train .
```

**Expected output** (REFERENCE — ยกมาจาก playbook)

```text
models/
├── checkpoints/
│   └── flux1-dev.safetensors
├── loras/
├── text_encoders/
│   ├── clip_l.safetensors
│   └── t5xxl_fp16.safetensors
└── vae/
    └── ae.safetensors
```

`launch_train.sh` เปิด container ด้วย `docker run -it` ซึ่งล้มเหลวภายใต้ `nohup` เพราะไม่มีเทอร์มินัล Lab 04 จึงเขียนสำเนาชื่อ `launch_train_w25.sh` ที่ตัด `-it` ออก และถ้าคุณขอ จะลด `--max_train_epochs` ลง (คำแนะนำของ playbook เองสำหรับการรันที่สั้นลงคือ `--max_train_epochs=25`) จากนั้นรันสำเนานั้นภายใต้ `nohup` ถ้าใช้ `--hotel` มันจะอัปโหลดโฟลเดอร์ `sparkhotel/` และ `data.toml` จาก lab 03 ก่อน โดยเก็บต้นฉบับไว้เป็น `data.toml.orig`

```bash
# on: laptop
.venv/bin/python week25/12_vlm_flux_finetune/labs/lab04_flux_train_on_spark.py
SPARK_APPLY=1 .venv/bin/python week25/12_vlm_flux_finetune/labs/lab04_flux_train_on_spark.py --epochs 25 --hotel
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ ในโหมด DRY)

```
▣ STEP 4 · launch_train.sh, headless, --max_train_epochs=100
…
$ cd ~/dgx-spark-playbooks/nvidia/playbook-flux-finetuning/assets && sed -e 's/docker run -it/docker run/' -e 's/--max_train_epochs=100/--max_train_epochs=100/' launch_train.sh > launch_train_w25.sh && grep -nE 'docker run|max_train_epochs|save_every' launch_train_w25.sh   [not run]
$ mkdir -p ~/w25/logs && cd ~/dgx-spark-playbooks/nvidia/playbook-flux-finetuning/assets && { nohup sh launch_train_w25.sh > ~/w25/logs/m12_flux_train.log 2>&1 < /dev/null & }; sleep 3; tail -2 ~/w25/logs/m12_flux_train.log   [not run]

▣ STEP 5 · monitor: progress, avr_loss, and the LoRA files
…
◈ DRY — testing the progress parser on an EXAMPLE (synthetic sd-scripts lines, not a run):
✓ progress parser: two bars read (step, total, avr_loss)
```

รัน lab 04 ซ้ำเพื่อติดตามการรัน มันพิมพ์ `step s/t`, `avr_loss` ของ sd-scripts (ค่าเฉลี่ยเคลื่อนที่; loss ของ diffusion มี noise มาก จึงควรตัดสิน LoRA จากภาพที่ได้) และไฟล์ใน `models/loras/`

จากนั้นสร้างภาพ (Step 7 ของ playbook) หยุดการเทรนก่อน เพราะ ComfyUI ต้องใช้หน่วยความจำ:

```bash
# on: spark
cd ~/dgx-spark-playbooks/nvidia/playbook-flux-finetuning/assets
docker build -f Dockerfile.inference -t flux-comfyui .
sh launch_comfyui.sh
```

```bash
# on: laptop
ssh -N -L 8188:localhost:8188 spark-a            # then open http://localhost:8188
```

ใน ComfyUI ให้กด `w` โหลด `finetuned_flux.json` แล้วใส่ prompt พร้อม trigger ของคุณ เช่น `tjtoy toy holding sparkgpu gpu in a datacenter` playbook เผื่อเวลาไว้ราวสามนาทีต่อภาพขนาด 1024 px โหลด `base_flux.json` ด้วย prompt เดียวกันเพื่อดูว่า LoRA เพิ่มอะไรเข้ามา

✓ Checkpoint: ในโหมด LIVE `models/loras/` มีอย่างน้อย `flux_dreambooth-000025.safetensors` และ prompt ที่มี trigger สร้าง concept ของคุณออกมาได้ ในโหมด DRY คุณอธิบายได้ว่าทำไม `launch_train.sh` ต้องตัด `-it` ออกจึงจะรันภายใต้ `nohup` ได้

## Labs — รันแล็บได้ที่นี่

**labs/lab01_vlm_datasets.py** — ชุดข้อมูลแบบ image-folder และ video-metadata ในโครงสร้างที่ trainer ของ playbook ต้องการ สร้างและ validate บนแล็ปท็อป

**labs/lab02_vlm_train_on_spark.py** — การเทรน Qwen2.5-VL ด้วย GRPO บน Spark: ตรวจ asset, image, โมเดล และชุดข้อมูล แล้วรันคำสั่ง Start Finetuning แบบ headless

**labs/lab03_flux_dataset.py** — ชุดข้อมูล FLUX และการคำนวณ step ของ playbook พร้อม concept โรงแรมที่มี caption และ data.toml ที่ผ่านการ validate

**labs/lab04_flux_train_on_spark.py** — การเทรน FLUX.1 Dreambooth LoRA บน Spark: โมเดล, image `flux-train`, launch_train.sh แบบ headless, ความคืบหน้า และไฟล์ LoRA

Lab 01 และ 03 รันบนแล็ปท็อปได้ทั้งหมด Lab 02 และ 04 สั่งงาน Spark ผ่าน SSH หรือแสดงแผนในโหมด DRY (parser ของมันทดสอบกับบรรทัด EXAMPLE)

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 12 — แก้ `data.toml` ที่พัง** เปิด `week25/12_vlm_flux_finetune/exercises/ex12_fix_data_toml.py` เพื่อนร่วมทีมเพิ่ม concept โรงแรมและทำผิดไว้สี่จุด:

1. `resolution` เป็นข้อความ ไม่ใช่ตัวเลข
2. สอง concept ใช้ trigger word เดียวกัน
3. `image_dir` ตัวหนึ่งชี้ไปที่โฟลเดอร์ที่ไม่มีอยู่
4. ค่า `class_tokens` ตัวหนึ่งมีแค่คำบอกประเภท ไม่มี trigger

ตัวตรวจทำงานแบบออฟไลน์ มัน parse ข้อความด้วย `tomllib` และตรวจทุกโฟลเดอร์เทียบกับ `flux_data/` ของ lab 03 และของ playbook

```bash
# on: laptop
.venv/bin/python week25/12_vlm_flux_finetune/exercises/ex12_fix_data_toml.py
```

**Expected output** (เมื่อแก้ TODO ครบทั้งสี่จุด; บันทึกจาก Mac เครื่องนี้)

```
✓ TODO 1: resolution = 1024 (a number)
✓ TODO 2: every concept has its own trigger word ['tjtoy', 'sparkgpu', 'sparkhotel']
✓ TODO 3: every image_dir exists
✓ TODO 4: every class_tokens has 2 words: trigger + class

▣ your data.toml, applied (launch_train.sh: gradient accumulation 4, 100 epochs)
│ tjtoy toy          6 images × 1 repeat(s)
│ sparkgpu gpu       7 images × 2 repeat(s)
│ sparkhotel lobby   6 images × 1 repeat(s)
│ 26 images per epoch → 700 optimizer steps · prompt with: 'tjtoy toy' · 'sparkgpu gpu' · 'sparkhotel lobby'
```

<details><summary>คำใบ้ — trigger word ที่ดีเป็นแบบไหน?</summary>

เป็นคำที่โมเดลฐานยังไม่รู้จัก (`tjtoy`, `sparkgpu`) ถ้าใช้คำทั่วไปอย่าง `lobby` เป็น trigger concept ใหม่จะต้องต่อสู้กับทุกอย่างที่ FLUX รู้อยู่แล้วเกี่ยวกับล็อบบี้

</details>

<details><summary>ท้าทายเพิ่ม — concept โรงแรมของจริง</summary>

แทนที่ภาพสังเคราะห์หกภาพใน `flux_data/sparkhotel/` ด้วยภาพถ่ายจริงของล็อบบี้หนึ่งแห่ง 5–10 ภาพ จากหลายมุมและหลายสภาพแสง อัปเดตไฟล์ `.caption` รัน lab 03 ใหม่ แล้วรัน lab 04 ด้วย `--epochs 25 --hotel` เปรียบเทียบ `base_flux.json` กับ `finetuned_flux.json` ด้วย prompt `sparkhotel lobby at night`

</details>

✓ Checkpoint: บรรทัดตรวจทั้งสี่เป็น ✓

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `cd: client-hardware-playbooks/…: No such file or directory` | พิมพ์ผิดใน playbook: โฟลเดอร์ที่ clone มาคือ `dgx-spark-playbooks/nvidia/…` |
| `permission denied` ตอนรัน Docker | วิธีแก้ของ playbook: `sudo usermod -aG docker $USER && newgrp docker` |
| error ตอนดาวน์โหลดหรือยืนยันตัวตนกับ Hugging Face | export `HF_TOKEN` ที่ใช้ได้ และส่ง `--build-arg HF_TOKEN=$HF_TOKEN` (VLM) สำหรับ FLUX ให้ยอมรับเงื่อนไขใน model card ของ FLUX.1-dev ก่อน |
| cURL ของ Kaggle ล้มเหลว | ล็อกอิน Kaggle ยอมรับเงื่อนไขของชุดข้อมูล แล้วคัดลอกคำสั่ง cURL ใหม่ |
| `the input device is not a TTY` | มี `docker run -it` อยู่ภายใต้ nohup ให้ใช้คำสั่งของ lab 02 หรือ `launch_train_w25.sh` ของ lab 04 |
| Streamlit หมุนรออยู่หลายนาที | playbook: การโหลดครั้งแรกจะเปิด vLLM (ภาพ, สูงสุด ~15 นาที) หรือโหลดโมเดล (วิดีโอ, ~10 นาที) |
| OOM หรือหน่วยความจำตึงระหว่างเทรนหรือ inference | หยุด Streamlit, Jupyter หรือ ComfyUI ก่อนเทรน แล้วล้าง cache: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| ภาพ `ready` ถูกตอบว่าไฟป่า "Yes" | คุณเปลี่ยนโฟลเดอร์แต่ไม่ได้แก้ `format_instruction()` (ส่วนที่ 2) |
| ไฟล์ใต้ `saved_model/` หรือ `models/loras/` เป็นของ root | container เขียนไฟล์ในฐานะ root ให้ใช้ `sudo chown -R $USER` กับโฟลเดอร์นั้น หรือคัดลอกไฟล์ออกมาด้วย `docker cp` |

## Next — บทถัดไป

ไปต่อที่ [Lab 13 — ปิดวงจร: ประเมิน merge และ serve](../13_finetune_to_serve/TUTORIAL.md): ให้คะแนนโมเดลที่ fine-tune แล้วบนข้อมูลที่กันไว้ merge แล้ว serve ไว้หลัง gateway ของคอร์ส
