# ▶ Spark Lab 21 — Playbook atlas: แผนที่ Spark playbook ที่เหลือทั้งหมด

> ส่วนหนึ่งของ Week 25 · DGX Spark: fine-tune, serve และสร้างเอเจนต์ที่รันใน sandbox คุณพิมพ์คำสั่งเอง และเห็นผลลัพธ์จริง ทุกแล็บรันในโหมด **DRY** ได้ด้วย (ไม่ต้องมี Spark, $0): จะแสดงคำสั่งให้ดู ส่วนผลลัพธ์เป็นแบบใดแบบหนึ่งคือ RECORDED (บันทึกจาก Spark จริง), REFERENCE (อ้างอิงคำต่อคำจาก playbook ของ NVIDIA) หรือ EXAMPLE (ตัวอย่างที่ติดป้ายไว้ชัดเจน)

> 💬 หมายเหตุภาษา: เนื้อหาบทเรียนเป็นภาษาไทย แต่ผลลัพธ์ที่โปรแกรมพิมพ์ออกเทอร์มินัล (และโค้ดทั้งหมด) เป็นภาษาอังกฤษ ตัวอย่างผลลัพธ์ในกล่องโค้ดจึงเป็นภาษาอังกฤษตรงกับที่คุณจะเห็นจริง

**สิ่งที่คุณจะได้ลงมือทำ**
- สแกน playbook ทางการทั้ง 64 ตัวใน clone บนเครื่อง แล้วดูในตารางเดียวว่าแต่ละตัวใช้เวลานานเท่าไร รันบน DGX Spark ได้ไหม และ Week 25 โมดูลไหนสอนมัน
- ทัวร์ playbook 26 ตัวที่ Module 01–20 ไม่ได้ครอบคลุม แบ่งเป็นเจ็ดธีม: ภาพและวิดีโอ, vision agent, หุ่นยนต์, data science และ HPC, kernel และ pretraining, แพลตฟอร์มและ multi-GPU, และ serving กับเอเจนต์ที่ใช้ได้เฉพาะ Station
- สำหรับแต่ละตัว เรียนรู้ว่ามันมีไว้ทำอะไร ต้องใช้อะไรบ้าง คำสั่งที่ทำให้มันรันได้ และมันใช้กับ Spark ของคุณได้หรือไม่ (ได้ 13 ตัว ไม่ได้ 13 ตัว)
- ถามตัวแนะนำ (recommender) แบบ TF-IDF ขนาดเล็กว่า "playbook ไหนเหมาะกับเป้าหมายของฉัน?" แล้วดูว่าการค้นด้วยคีย์เวิร์ดพลาดตรงไหน
- วางแผนเวลา ดิสก์ และอุปกรณ์เพิ่มเติมสำหรับ playbook ที่คุณเลือก แล้วเขียนชุดทดสอบ (test set) ที่ตัวแนะนำต้องผ่าน

**Time** ~50 นาที · **Difficulty** ระดับเริ่มต้น · **Hardware** ไม่ต้องใช้สำหรับแล็บ (ใช้ Spark 1 เครื่องถ้าจะรัน playbook จริง ส่วน cuPyNumeric ต้องใช้ 2 เครื่อง)

**Playbook ทางการที่ครอบคลุม:** รันบน DGX Spark: [ComfyUI](https://build.nvidia.com/spark/comfyui) · [Multi-modal inference](https://build.nvidia.com/spark/multi-modal-inference) · [Live VLM WebUI](https://build.nvidia.com/spark/live-vlm-webui) · [VSS](https://build.nvidia.com/spark/vss) · [Isaac Sim และ Isaac Lab](https://build.nvidia.com/spark/isaac) · [Reachy photo booth](https://build.nvidia.com/spark/spark-reachy-photo-booth) · [CUDA-X data science](https://build.nvidia.com/spark/cuda-x-data-science) · [Portfolio optimization](https://build.nvidia.com/spark/portfolio-optimization) · [Single-cell RNA](https://build.nvidia.com/spark/single-cell) · [JAX](https://build.nvidia.com/spark/jax) · [cuPyNumeric](https://build.nvidia.com/spark/cupynumeric) · [cuTile kernels](https://build.nvidia.com/spark/cutile-kernels) · [Brev](https://build.nvidia.com/spark/brev) เฉพาะ DGX Station: [GR00T](https://build.nvidia.com/station/gr00t) · [Topic modeling](https://build.nvidia.com/station/topic-modeling) · [Triton kernels](https://build.nvidia.com/station/kernel-dev-ft) · [NanoChat](https://build.nvidia.com/station/nanochat) · [NVFP4 pretraining](https://build.nvidia.com/station/nvfp4-pretraining) · [MIG](https://build.nvidia.com/station/mig) · [Connect two stations](https://build.nvidia.com/station/connect-two-stations) · [SGLang inference](https://build.nvidia.com/station/sglang-inference) · [Agent skills](https://build.nvidia.com/station/dgx-station-ai-skills) · [Healthcare agent](https://build.nvidia.com/station/healthcare-agent) เฉพาะ RTX PC: [Visual gen AI](https://build.nvidia.com/playbooks/visual-gen-ai) · [Controlled video](https://build.nvidia.com/playbooks/video-gen-guide) · [Multi-GPU AI PC](https://build.nvidia.com/playbooks/multi-gpu)

> 🔎 **เกี่ยวกับลิงก์เหล่านี้** แต่ละ URL สร้างจากชื่อไดเรกทอรีของ playbook (`playbook-<slug>` → `<slug>`) **ลิงก์ Spark 12 จาก 13 ลิงก์ยืนยันแล้ว**: slug ของมันปรากฏในหน้า index ของ build.nvidia.com/spark (ดึงข้อมูลเมื่อ 2026-09-29) **cuPyNumeric ไม่อยู่ใน index นั้น** แม้ README ของมันจะระบุ DGX Spark ลิงก์ของมันจึงอาจเจอ 404 นอกจากนี้ index ยังมี Spark playbook หนึ่งตัวชื่อ `pair` ที่ไม่มีไดเรกทอรีใน clone นี้ ส่วนลิงก์ Station และ RTX 13 ลิงก์ **ยังไม่ได้ยืนยัน**: index ของ /spark แสดงเฉพาะ Spark playbook มีสองลิงก์ในกลุ่มนี้ (`/station/connect-two-stations` และ `/station/topic-modeling`) ที่ปรากฏเป็นลิงก์อยู่ใน clone ด้วย Lab 21-1 step 4 พิมพ์หลักฐานทั้งหมดนี้ออกมา ถ้าลิงก์ไหนเจอ 404 ให้ค้นชื่อ playbook บน [build.nvidia.com/spark](https://build.nvidia.com/spark) README ใน clone คือแหล่งความจริงเสมอ

## 0 · ก่อนเริ่ม

| สิ่งที่ต้องมี | วิธีตรวจ | ทำไม |
|---|---|---|
| Python ของ repo นี้ | `.venv/bin/python --version` → 3.13 | ใช้รันแล็บทั้งสามและแบบฝึกหัด |
| clone ของ playbook | `ls dgx-spark-playbooks/nvidia \| head` | ตัวเลขทุกตัวในโมดูลนี้อ่านมาจากมัน |
| Spark (ไม่บังคับ) | `ssh -o BatchMode=yes spark-a true` | ใช้เฉพาะตอนรัน playbook จริง ไม่ได้ใช้กับแล็บ |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
.venv/bin/python --version
ls -d dgx-spark-playbooks/nvidia/playbook-* | wc -l
git -C dgx-spark-playbooks log -1 --format='%h %cd'
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
Python 3.13.13
      65
3410c65 Thu Sep 10 21:01:40 2026 +0000
```

clone มีการเปลี่ยนแปลงเรื่อย ๆ ถ้าจำนวนหรือ commit ของคุณต่างไป แล็บก็ยังใช้ได้: มันอ่านสิ่งที่มีอยู่ และ lab 21-1 จะเตือนเมื่อมี playbook ที่ยังไม่มีธีม

✓ Checkpoint: มี clone อยู่และคุณรู้ commit ของมัน คุณจึงบอกได้ว่า atlas ของคุณอธิบาย playbook เวอร์ชันไหน

## 1 · แผนที่: playbook 64 ตัว ครอบคลุมไปแล้ว 38 ตัว อีก 26 ตัวอยู่ที่นี่

repository ของ NVIDIA มีมากกว่าแค่ DGX Spark playbook README แต่ละตัวมีตาราง **Supported hardware platforms** และ playbook บางตัวระบุแค่ DGX Station (GB300) หรือ RTX PC ก่อนจะเสียเวลาทั้งเย็นไปกับ playbook ตัวไหน ให้ตรวจตารางนั้นก่อน Lab 21-1 อ่านตารางนี้จาก playbook ทั้ง 64 ตัว แล้ว join กับตารางโมดูลใน `week25/AUTHORING.md`:

```bash
# on: laptop
.venv/bin/python week25/21_playbook_atlas/labs/lab21_1_playbook_atlas.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ ตาราง 64 แถวถูกย่อไว้ในที่นี้)

```
▣ STEP 1 · scan the playbook clone
$ ls dgx-spark-playbooks/nvidia/playbook-*/README.md   → 65 files (playbook-test skipped: it is the catalogue template)
◆ 64 playbooks · 48 list DGX Spark in 'Supported hardware platforms'

▣ STEP 2 · every playbook: time, platforms, and the Week 25 module that covers it
│ playbook                     title                                     time    Spark?  platforms                  covered by
│ ───────────────────────────  ────────────────────────────────────────  ──────  ──────  ─────────────────────────  ──────────
│ connect-to-your-spark        Connect Remotely to Your AI Compute       10 min  ✓       Spark                      Module 01
│ …
│ llms                         Run Local LLMs                            6 min   ✕       RTX                        Module 03
│ …
│ vllm                         Serve LLMs with vLLM                      30 min  ✓       Spark + Station + RTX PRO  Module 05
│ …
│ local-coding-agent           Set Up Claude Code with Local Inference   30 min  ✕       Station                    Module 18
│ …
│ brev                         Register AI Compute with Brev             10 min  ✓       Spark + Station            Module 21
│ comfyui                      Generate Images and Videos with ComfyUI   45 min  ✓       Spark + Station            Module 21
│ connect-two-stations         Connect Two Nodes for Distributed Worklo  60 min  ✕       Station                    Module 21
│ …
│ sglang-inference             LLM Inference with SGLang                 30 min  ✕       Station (inferred)         Module 21
│ …
│ vss                          Deploy a Video Search and Summarization   45 min  ✓       Spark                      Module 21

▣ STEP 3 · the 26 playbooks Modules 01–20 do not cover, by theme (this module)
│ theme (tutorial section)   all  Spark  Station only  RTX only  playbooks
│ ─────────────────────────  ───  ─────  ────────────  ────────  ────────────────────────────────────────────────────
│ §2 images & video          4    2      0             2         comfyui, multi-modal-inference, video-gen-guide, vi…
│ §3 vision & video agents   2    2      0             0         live-vlm-webui, vss
│ §4 robotics                3    2      1             0         gr00t, isaac, spark-reachy-photo-booth
│ §5 data science & HPC      6    5      1             0         cuda-x-data-science, cupynumeric, jax, portfolio-op…
│ §6 kernels & pretraining   4    1      3             0         cutile-kernels, kernel-dev-ft, nanochat, nvfp4-pret…
│ §7 platform & multi-GPU    4    1      2             1         brev, connect-two-stations, mig, multi-gpu
│ §8 Station serving/agents  3    0      3             0         dgx-station-ai-skills, healthcare-agent, sglang-inf…
…
▣ STEP 4 · build.nvidia.com slugs — the /spark index (fetched 2026-09-29) and links inside the clone
│ dir slug                  build.nvidia.com/spark index                 linked in a README as
│ ────────────────────────  ───────────────────────────────────────────  ──────────────────────────────────
│ brev                      ✓ confirmed /spark/brev                      —
│ comfyui                   ✓ confirmed /spark/comfyui                   /playbooks/comfyui
│ connect-two-stations      unverified (Station only)                    /station/connect-two-stations
│ cuda-x-data-science       ✓ confirmed /spark/cuda-x-data-science       —
│ cupynumeric               ✕ Spark playbook, not in /spark index        —
│ …
◆ 12 of 26 atlas slugs are confirmed on the /spark index. Station/RTX slugs stay unverified: the index lists Spark playbooks only.
◆ on the /spark index but no playbook-* dir in this clone: pair
◆ list DGX Spark in their README but not on the /spark index: cupynumeric

▣ STEP 5 · consistency checks
✓ every uncovered playbook has a theme (26 of 26)
✓ every playbook named in AUTHORING.md exists in the clone
═ 64 playbooks: 38 covered in Modules 01–20, 26 in this atlas (13 of them run on a DGX Spark).
```

สามสิ่งที่ตารางบอกคุณ:

1. **มีแค่ 48 จาก 64 ตัวที่ระบุ DGX Spark** playbook บางตัวที่คอร์สนี้ใช้อยู่ดี (`llms` และ `fine-tuning` ระบุแค่ RTX ใน clone ปัจจุบัน ส่วน `local-coding-agent` ระบุแค่ Station) ใกล้เคียงพอที่ Module 03, 10 และ 18 จะนำมาดัดแปลงได้ แต่ atlas ไม่ยืดไปไกลขนาดนั้น: playbook ที่ใช้ได้เฉพาะ Station จะถูกทำเครื่องหมายว่า Station only
2. **คอลัมน์เวลาคือค่าประมาณของ NVIDIA** เป็นตัวเลขแรกในบรรทัด "Estimated time" ของ README แต่ละตัว (ค่าบนสุดของช่วง) ไม่ใช่ค่าที่วัดจริง และการรันครั้งแรกที่ต้องดาวน์โหลดไฟล์ใหญ่จะใช้เวลานานกว่านั้น
3. **`sglang-inference` ไม่มีตารางแพลตฟอร์ม** prerequisites ของมันเขียนว่า "NVIDIA DGX Station with GB300" แล็บจึงทำเครื่องหมายว่า `(inferred)` (อนุมานเอา) มันคือ playbook SGLang ฉบับ Station ของตัวที่คุณรันใน Module 06

วิธีอ่านแต่ละรายการด้านล่าง:

| ป้าย | ความหมาย |
|---|---|
| 🟩 **Spark** | ตารางแพลตฟอร์มใน README ระบุ DGX Spark: คุณรันได้ในสัปดาห์นี้ |
| 🟪 **Station only** | ตารางระบุแค่ DGX Station (GB300) อ่านเพื่อเก็บแนวคิด คำสั่งเขียนไว้สำหรับฮาร์ดแวร์อื่น |
| 🟧 **RTX only** | PC ที่ใช้ Windows หรือ Linux กับการ์ด GeForce/RTX PRO มีระบุ playbook คู่เทียบบน Spark ไว้ให้ |

คำสั่งสำหรับ playbook 🟩 อยู่ในบล็อก `bash` พร้อม `# on: spark` ส่วนคำสั่งของ playbook 🟪 และ 🟧 ตั้งใจใส่ไว้ในบล็อก `text`: มีไว้ใช้อ้างอิง และ ⌨ terminal จะไม่เสนอให้รันบน Spark ของคุณ

✓ Checkpoint: lab 21-1 จบด้วย `26 in this atlas (13 of them run on a DGX Spark)` และคุณบอกได้ว่าทำไม `nanochat` อยู่ใน atlas แต่ไม่อยู่ในแผนของคุณ

## 2 · ภาพและวิดีโอ

### ComfyUI 🟩 Spark · 45 นาที · ดิสก์ ≥30 GB (quick start), 70–230 GB (video tier)

ComfyUI คือเว็บแอปแบบ node สำหรับ diffusion model แทนที่จะมีช่อง prompt ช่องเดียว คุณต่อ loader, text encoder, sampler และ decoder เข้าด้วยกันเป็นกราฟที่บันทึกเป็น JSON ได้ ใช้มันเมื่ออยากสร้างภาพหรือวิดีโอในเครื่องโดยควบคุมได้ทุกขั้นตอน และเมื่ออยากแชร์ workflow ที่ทำซ้ำได้ playbook มีสองเส้นทาง: **Image Gen Quick Start** แบบเบาที่ใช้ Python บน host กับ Z-Image-Turbo (หน่วยความจำ GPU ~24 GB) และ **Video Gen Workflow** แบบ container กับ FLUX, Wan 2.1, HunyuanVideo และ Cosmos ในสาม tier (peak ~80 / ~100 / ~120 GB) สำหรับ unified memory 128 GB README แนะนำให้เริ่มจาก Tier 1 Module 12 fine-tune FLUX.1 ไปแล้ว ComfyUI คือที่ที่คุณนำมันมาใช้

```bash
# on: spark
python3 -m venv comfyui-env
source comfyui-env/bin/activate
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
git clone --branch v0.33.2 https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI/
pip install -r requirements.txt
wget -P models/diffusion_models/ https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors
wget -P models/text_encoders/ https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors
wget -P models/vae/ https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors
python main.py --listen 0.0.0.0
```

ในเทอร์มินัลที่สอง `curl -I http://localhost:8188` ควรคืน HTTP 200 จากแล็ปท็อป ให้ทำ tunnel พอร์ต (`ssh -N -L 8188:localhost:8188 spark-a`) แล้วเปิด `http://localhost:8188` → **Templates → Image → Z-Image-Turbo: Text to Image → Run** README บอกว่าภาพควรเสร็จภายใน 30 วินาที เส้นทางวิดีโอเริ่มด้วย `docker build -t comfyui -f assets/Dockerfile .` และ `bash assets/scripts/download-models.sh 1` แล้วทำตามแท็บ **Video Gen Workflow** ของ playbook

### Multi-modal inference ด้วย TensorRT 🟩 Spark · 60 นาที · หน่วยความจำว่าง ≥48 GB สำหรับ FP16 Schnell

playbook นี้รัน FLUX.1 Dev และ Schnell แบบ text-to-image ผ่าน TensorRT diffusion demo ภายใน PyTorch container ของ NVIDIA ที่ BF16, FP8 และ FP4 ใช้มันเพื่อดูว่าคันโยกด้าน precision จาก Module 01 และ 07 ส่งผลอย่างไรกับโมเดลภาพ ไม่ใช่กับ LLM ต้องมีบัญชี Hugging Face ที่ได้รับสิทธิ์เข้าถึง FLUX.1-dev และ FLUX.1-dev-onnx

```bash
# on: spark
docker run --gpus all --ipc=host --ulimit memlock=-1 \
  --ulimit stack=67108864 -it --rm \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/pytorch:25.11-py3
# inside the container:
git clone https://github.com/NVIDIA/TensorRT.git -b main --single-branch && cd TensorRT
export TRT_OSSPATH=/workspace/TensorRT/
cd $TRT_OSSPATH/demo/Diffusion
apt update
apt install -y libgl1 libglu1-mesa libglib2.0-0t64 libxrender1 libxext6 libx11-6 libxrandr2 libxss1 libxcomposite1 libxdamage1 libxfixes3 libxcb1
pip install nvidia-modelopt[torch,onnx]
sed -i '/^nvidia-modelopt\[.*\]=.*/d' requirements.txt
pip3 install -r requirements.txt
pip install onnxconverter_common
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --hf-token=$HF_TOKEN --fp4 --download-onnx-models
```

> 🔐 playbook ส่งโทเค็นเป็น `--hf-token=$HF_TOKEN` ดังนั้นให้ตั้งค่าด้วย `export HF_TOKEN=<YOUR_HUGGING_FACE_TOKEN>` **ภายใน container บน Spark** ในเชลล์นั้นเท่านั้น ห้ามพิมพ์โทเค็นลงในแล็บหรือ Lab Runner เด็ดขาด

เปลี่ยน `--fp4` เป็น `--bf16` หรือ `--quantization-level 4 --fp8` เพื่อเปรียบเทียบ README เตือนว่า FP16 Schnell ต้องใช้หน่วยความจำว่างมากกว่า 48 GB

### Visual gen AI (FLUX.2 + LTX-2) 🟧 RTX only · 13 นาที

เส้นทาง ComfyUI บนเดสก์ท็อป Windows: template เริ่มต้นแบบ text-to-image, ภาพจาก FLUX.2-Dev และ image-to-video ด้วย LTX-2 ทั้งหมดเปิดจาก template browser ของ ComfyUI README ตั้งใจไม่ใส่คำสั่งเชลล์ใด ๆ ("It does **not** invent host Python, Docker, or shell install commands") และชี้ผู้ใช้ Linux ไปที่ ComfyUI playbook **บน Spark ของคุณ ให้ใช้ ComfyUI ด้านบน** แล้วโหลด template เดียวกันจาก template browser ของมัน

### Controlled video ด้วย ComfyUI 🟧 RTX only · 18 นาที · VRAM 16 GB+, RAM 64 GB

workflow จาก storyboard ไปถึง 4K บน Windows 11: สร้าง asset 3 มิติ จัดฉากใน Blender ทำ keyframe แรกและสุดท้ายด้วย FLUX.1 Depth เติมการเคลื่อนไหวด้วย LTX-2.3 แล้ว upscale ด้วย node RTX Video Super Resolution การติดตั้งอยู่ใน repo NVIDIA AI Blueprints สองตัว ไม่ได้อยู่ใน playbook ควรอ่านเพื่อเก็บแนวคิด: **ล็อกองค์ประกอบภาพก่อน แล้วค่อยสร้างการเคลื่อนไหวระหว่าง keyframe** คำสั่งเปิดโปรแกรมคำสั่งเดียวที่มันแสดงไว้เป็นของ Windows:

```text
# on: Windows RTX PC (not a Spark) — reference only
cd C:\3d-object-generation
conda activate 3dwithtrellis311
python app.py
```

✓ Checkpoint: คุณบอกได้ว่าในสี่ playbook ด้านภาพ/วิดีโอ ตัวไหนที่คุณจะรันบน Spark (ComfyUI, TensorRT multi-modal) และ Tier ไหนของ video workflow ใน ComfyUI ที่ใส่ใน 128 GB ได้

## 3 · Vision agent และ video agent

### Live VLM WebUI 🟩 Spark · 30 นาที · เว็บแคม

เบราว์เซอร์แอปที่สตรีมเว็บแคมของคุณไปยัง vision-language model ตัวใดก็ได้ที่เข้ากันได้กับ OpenAI แล้วแสดงคำตอบ latency และโหลดของ GPU แบบสด เป็นเครื่องมือทดสอบสำหรับเปรียบเทียบ VLM ข้าม Ollama, vLLM, SGLang หรือ NIM playbook ใช้ Ollama กับ `llama3.2-vision:11b` ถ้าคุณทำ Module 03 แล้ว Ollama อยู่บน Spark ของคุณแล้ว ข้ามบรรทัดติดตั้งได้

```bash
# on: spark
curl -fsSL https://ollama.com/install.sh | sh        # skip if Module 03 already installed Ollama
ollama pull llama3.2-vision:11b
sudo apt update
sudo apt install -y pipx
pipx ensurepath
source ~/.bashrc
pipx install live-vlm-webui
live-vlm-webui
```

มัน serve HTTPS ที่พอร์ต 8090 ด้วย certificate แบบ self-signed เปิด `https://<SPARK_IP>:8090` (หา IP ด้วย `hostname -I | awk '{print $1}'`) ตั้ง **API Base URL** เป็น `http://localhost:11434/v1` แล้วเลือกโมเดล

> ⚠ อย่าเข้าตัวนี้ผ่าน tunnel `ssh -L` ส่วน troubleshooting ของ README บอกว่า WebRTC จะล้มเหลวด้วย `InvalidStateError` เมื่อผ่าน SSH TCP tunnel และต้องเชื่อมต่อตรงระหว่างเบราว์เซอร์กับ Spark

### Video Search and Summarization (VSS) 🟩 Spark · 45 นาที · >10 GB ใน /tmp · คีย์ NGC + LLM ระยะไกล

VSS AI Blueprint ของ NVIDIA เปลี่ยนวิดีโอให้เป็นเหตุการณ์ที่ค้นหาได้: VLM ในเครื่อง (Cosmos Reason 2) บรรยายฟุตเทจ ส่วน LLM ระยะไกลตอบคำถามและยืนยันการแจ้งเตือน ใช้มันสำหรับวิเคราะห์ภาพจากกล้อง ถาม-ตอบวิดีโอยาว และแจ้งเตือนแบบเรียลไทม์ บน Spark นี่เป็นการ deploy แบบ **hybrid**: VLM รันในเครื่อง ส่วน LLM เป็น endpoint ระยะไกล (เช่น NIM บน build.nvidia.com) VSS 3.2.0 ต้องใช้ driver ≥ 580.95.05 และ CUDA 13.0 ซึ่ง doctor ใน Module 01 ตรวจให้แล้ว

```bash
# on: spark
sudo apt-get install -y git-lfs && git lfs install
git clone https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization.git
cd video-search-and-summarization
git checkout tags/v3.2.0
git lfs install
git lfs pull
docker login nvcr.io
# the playbook exports NGC_CLI_API_KEY, HF_TOKEN and LLM_ENDPOINT_URL in this shell, then:
deploy/docker/scripts/dev-profile.sh up -p base -H DGX-SPARK --use-remote-llm --llm <REMOTE LLM MODEL NAME>
curl -I http://localhost:7777
```

README ยังติดตั้ง **สคริปต์ล้าง cache** ที่รัน `sync && echo 3 > /proc/sys/vm/drop_caches` ทุก 3 วินาทีด้วยสิทธิ์ root (Step 4 ของมัน) อ่านขั้นนั้นก่อนคัดลอกไปใช้: มันเปลี่ยนพฤติกรรมของระบบตลอดเวลาที่รันอยู่ และหยุดได้ด้วย `sudo pkill -f sys-cache-cleaner.sh` สำหรับ workflow การแจ้งเตือน README แนะนำอย่างยิ่งให้ใช้ `--llm nvidia/nvidia-nemotron-nano-9b-v2` รื้อถอนด้วย `deploy/docker/scripts/dev-profile.sh down`

✓ Checkpoint: คุณบอกได้ว่าทำไม Live VLM WebUI ห้ามผ่าน SSH tunnel และครึ่งไหนของ VSS ที่รันบน Spark

## 4 · หุ่นยนต์และ physical AI

### Isaac Sim และ Isaac Lab 🟩 Spark · 30 นาที (build 10–15 นาที) · ดิสก์ ≥50 GB

Isaac Sim คือตัวจำลองหุ่นยนต์ของ NVIDIA ที่สร้างบน Omniverse ส่วน Isaac Lab เพิ่ม environment สำหรับ reinforcement learning ไว้ด้านบน บน Spark คุณจะ **build Isaac Sim จาก source** สำหรับ `linux-aarch64` แล้วเทรนตัวอย่างการเดินของหุ่นมนุษย์ `Isaac-Velocity-Rough-H1-v0` ใช้มันเพื่อเทรนและทดสอบ policy ของหุ่นยนต์ในการจำลองก่อนใช้ฮาร์ดแวร์จริง ต้องตั้ง GCC 11 เป็นคอมไพเลอร์ตั้งต้น

```bash
# on: spark
sudo apt update && sudo apt install -y gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 200
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 200
sudo apt install git-lfs
git clone --depth=1 --recursive https://github.com/isaac-sim/IsaacSim
cd IsaacSim
git lfs install
git lfs pull
./build.sh
export ISAACSIM_PATH="${PWD}/_build/linux-aarch64/release"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
```

จากนั้น clone Isaac Lab เชื่อม (link) เข้ากับ build แล้วเทรนแบบ headless (Step 6–9 ของ playbook):

```bash
# on: spark
git clone --recursive https://github.com/isaac-sim/IsaacLab
cd IsaacLab
ln -sfn "${ISAACSIM_PATH}" "${PWD}/_isaac_sim"
sudo apt install -y libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev
./isaaclab.sh --install
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-H1-v0 --headless
```

> ⚠ `update-alternatives` เปลี่ยนคอมไพเลอร์ตั้งต้นของระบบ ถ้ามีโปรเจกต์อื่นบน Spark ที่ขึ้นกับมัน ให้จด `gcc --version` ปัจจุบันไว้ก่อน

**Expected output** (REFERENCE — อ้างอิงจาก playbook หลังรัน `./build.sh`)

```
BUILD (RELEASE) SUCCEEDED (Took 674.39 seconds)
```

### Isaac GR00T N1.6 fine-tuning 🟪 Station only · 45 นาที · ดิสก์ ~30 GB

GR00T N1.6 เป็นโมเดล vision-language-action ขนาด 3 พันล้านพารามิเตอร์สำหรับทักษะของหุ่นยนต์ humanoid playbook fine-tune ส่วน action head บน benchmark LIBERO Spatial ที่ global batch size 128 บน GB300 หนึ่งตัว แล้วประเมินผลแบบ open-loop ตารางแพลตฟอร์มระบุแค่ DGX Station (HBM3e ~284 GB) และชื่อ GPU ที่คาดไว้คือ `NVIDIA GB300` สิ่งที่นำมาใช้กับ Spark ของคุณได้คือแนวคิดจาก Module 09–12: fine-tune ส่วน adapter เล็ก ๆ บนโมเดลที่ pretrain แล้ว จากนั้นประเมินผลก่อนจะเชื่อใจมัน

```text
# on: DGX Station (not a Spark) — reference only
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
git checkout n1.6-release
I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 bash scripts/deployment/dgpu/install_deps.sh
source .venv/bin/activate
huggingface-cli download nvidia/GR00T-N1.6-3B
```

### Reachy photo booth 🟩 Spark · 2 ชม. · หุ่นยนต์ Reachy Mini จอภาพ และคีย์บอร์ด

stack แบบ multimodal ที่รันในเครื่องครบชุดบน Spark เครื่องเดียว: ReAct agent ของ NeMo Agent Toolkit บน `openai/gpt-oss-20b` (TensorRT-LLM), Parakeet สำหรับ speech-to-text, Kokoro สำหรับ text-to-speech, FLUX.1-Kontext สำหรับปรับสไตล์ภาพ, การติดตามตัวบุคคล และ MinIO สำหรับแชร์รูปผ่าน QR code เป็นตัวอย่างที่ดีที่สุดของ NAT agent จาก Module 14 ที่ควบคุมฮาร์ดแวร์จริง ต้องมีหุ่นยนต์ Reachy Mini Lite, NGC personal API key, โทเค็น Hugging Face ที่ยอมรับ license ของ FLUX.1-Kontext แล้ว และจอที่ต่อกับ Spark โดยตรง

```bash
# on: spark
git clone https://github.com/NVIDIA/spark-reachy-photo-booth.git
cd spark-reachy-photo-booth
cp .env.example .env            # then edit NVIDIA_API_KEY and HF_TOKEN in .env on the Spark
lsusb | grep Reachy
./robot-controller-service/scripts/speaker_setup.sh
docker login nvcr.io -u "\$oauthtoken"
docker compose up --build -d
```

เปิด `http://127.0.0.1:3001` ในเบราว์เซอร์ **บน Spark** README บอกว่า `docker compose up` ครั้งแรกใช้เวลา 30 นาทีถึง 2 ชั่วโมง ครั้งต่อ ๆ ไปประมาณ 5 นาที ตรวจ container ด้วย `docker compose ps` และหยุดด้วย `docker compose down`

✓ Checkpoint: คุณบอกชื่อ playbook ด้านหุ่นยนต์สองตัวที่รันบน Spark ได้ และบอกได้ว่าแต่ละตัวต้องใช้ฮาร์ดแวร์อะไรเพิ่ม

## 5 · Data science และ HPC

### CUDA-X data science (cuDF, cuML) 🟩 Spark · 30 นาที · Kaggle API key

RAPIDS ซึ่งตอนนี้เรียกว่า CUDA-X Data Science รันโค้ด pandas และ scikit-learn บน GPU **โดยไม่ต้องแก้โค้ด**: โหลด `cudf.pandas` หรือ `cuml.accel` แล้วโค้ดเดิมของคุณจะถูกเร่งความเร็ว playbook มีโน้ตบุ๊กสองตัว: workflow ของ pandas ที่มีสตริงขนาดใหญ่ และ LinearSVC, UMAP และ HDBSCAN แบบ scikit-learn ใช้มันเมื่อส่วนที่ช้าในโปรเจกต์ของคุณคือ dataframe หรือ ML แบบดั้งเดิม ไม่ใช่ LLM

```bash
# on: spark
conda create -n rapids-test -c rapidsai -c conda-forge -c nvidia  \
  rapids=26.06 python=3.12 'cuda-version=13.0' \
  jupyter hdbscan umap-learn
conda activate rapids-test
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-cuda-x-data-science/assets
jupyter notebook cudf_pandas_demo.ipynb
```

> ✎ จุดที่คอร์สปรับจากต้นฉบับ: บรรทัด `cd` ใน README เขียนว่า `client-hardware-playbooks/nvidia/…` แต่ `git clone` ด้านบนสร้าง `dgx-spark-playbooks/` คอร์สจึงใช้ path นั้น การแก้แบบเดียวกันใช้กับ JAX, portfolio optimization และ single-cell ด้านล่างด้วย

วาง `kaggle.json` ไว้ในโฟลเดอร์ `assets` ข้างโน้ตบุ๊ก และทำ tunnel ให้ Jupyter ด้วย `ssh -N -L 8888:localhost:8888 spark-a`

### Portfolio optimization ด้วย cuOpt 🟩 Spark · 20 นาที · ดิสก์ ≥30 GB, หน่วยความจำว่าง 40 GB

pipeline พอร์ตการลงทุนแบบ Mean-CVaR: kernel density estimation ของ cuML สร้าง scenario ผลตอบแทน cuOpt แก้ linear program ที่ได้ และโน้ตบุ๊ก backtest กลยุทธ์การปรับสมดุลพอร์ต ใช้เป็นแม่แบบสำหรับ optimisation แบบอิง scenario ใด ๆ ที่ช้าเกินไปบน CPU ทั้งหมดรันภายใน container โน้ตบุ๊ก RAPIDS 25.10

```bash
# on: spark
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-portfolio-optimization/assets
bash ./setup/start_playbook.sh
```

เปิด JupyterLab (tunnel `-L 8888:localhost:8888`) เปิด `cvar_basic.ipynb` แล้ว **เปลี่ยน kernel เป็น "Portfolio Optimization" ก่อนรันเซลล์แรก** README บอกว่าถ้าใช้ kernel ผิด จะพังภายในเซลล์โค้ดที่สอง

### Single-cell RNA analysis 🟩 Spark · 15 นาที · ดิสก์ ≥30 GB, หน่วยความจำว่าง 40 GB

RAPIDS-singlecell ทำตาม API ของ Scanpy ดังนั้น preprocessing, QC, clustering, UMAP, Harmony batch correction และ differential expression ของ single-cell RNA-seq จึงรันบน GPU ได้ ใช้มันถ้าคุณทำงานกับข้อมูล genomics container และตัวเปิดใช้รูปแบบเดียวกับ portfolio optimization:

```bash
# on: spark
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-single-cell/assets
bash ./setup/start_playbook.sh
```

จากนั้นรัน `scRNA_analysis_preprocessing.ipynb` README ระบุว่า GPU คำนวณ neighbour graph แบบตรงเป๊ะ (exact) ขณะที่ Scanpy บน CPU ใช้แบบประมาณ (approximate) จึงคาดได้ว่าผลจะต่างกันเล็กน้อย

### Topic modeling ด้วย BERTopic 🟪 Station only · 45 นาที · ดิสก์ ≥50 GB, ชุดข้อมูล ~14 GB

BERTopic บนรีวิว Amazon หลายล้านรายการ: embedding ด้วย SentenceTransformers แล้วใช้ UMAP และ HDBSCAN บน GPU ของ cuML ที่ใช้แทนตัวเดิมได้ทันที (drop-in) จากนั้นได้ topic map แบบโต้ตอบได้ และแดชบอร์ด Streamlit (ไม่บังคับ) ตารางแพลตฟอร์มระบุแค่ DGX Station และ README ขอหน่วยความจำ GPU 64 GB สำหรับรีวิวหลายสิบล้านรายการ เทคนิคเร่งความเร็ว (`%load_ext cuml.accel`) เป็นตัวเดียวกับที่ CUDA-X data science รันบน Spark ของคุณ

```text
# on: DGX Station (not a Spark) — reference only
conda create -n rapids-25.10 \
  -c rapidsai -c conda-forge \
  cudf=25.10 cuml=25.10 python=3.11 'cuda-version=13.0'
conda activate rapids-25.10
python -m pip install \
  transformers datasets sentence-transformers \
  umap-learn hdbscan==0.8.40 bertopic matplotlib \
  scikit-learn==1.4.2 datamapplot streamlit
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz
jupyter lab
```

### JAX 🟩 Spark · 2 ชม. · พอร์ต 8080

JAX คือ NumPy บน GPU บวกกับ function transform: `jit` คอมไพล์, `grad` หาอนุพันธ์, `vmap` ทำ vectorise playbook สร้าง container ที่มีโน้ตบุ๊ก marimo แล้วให้คุณพอร์ต self-organising map ที่เขียนด้วย NumPy ไปเป็น JAX ทีละขั้น พร้อมวัดผลแต่ละเวอร์ชัน ใช้มันเพื่อเรียนรู้ว่าการคอมไพล์ด้วย XLA เปลี่ยนประสิทธิภาพของโค้ดที่ทำงานกับ array อย่างไร

```bash
# on: spark
docker run --gpus all --rm nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04 nvidia-smi
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-jax/assets
docker build -t jax-playbook .
docker run --gpus all --rm -it \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8080:8080 \
    jax-playbook
```

ทำ tunnel `-L 8080:localhost:8080` แล้วเปิด `http://localhost:8080`

### cuPyNumeric ข้าม Spark สองเครื่อง 🟩 Spark ×2 · 60 นาที · ต่อสายตาม Module 02 ก่อน

> README ระบุ DGX Spark แต่ `cupynumeric` ไม่อยู่ใน index ของ build.nvidia.com/spark ที่ดึงเมื่อ 2026-09-29 ลิงก์เว็บด้านบนจึงอาจเปิดไม่ได้ ให้ทำตาม README ใน clone

cuPyNumeric เป็นไลบรารีที่เข้ากันได้กับ NumPy ซึ่งกระจาย array และ linear algebra ไปยังหลาย GPU และหลายโหนดผ่าน Legate และ MPI playbook แชร์ conda environment ผ่าน NFS ตรวจ MPI ข้าม Spark ทั้งสองเครื่อง แล้วรันการคูณเมทริกซ์ 20k × 20k บนโหนดเดียวและบนสองโหนด เป็น playbook เดียวใน atlas นี้ที่ **ต้องใช้ Spark ทั้งสองเครื่อง** และถือว่าลิงก์ QSFP จาก Module 02 (หรือ NVIDIA Sync Cluster Assistant) ใช้งานได้แล้ว

```bash
# on: spark
export SERVER_IP="192.168.100.11"    # Server node interconnect IP — use your Module 02 addresses
export CLIENT_IP="192.168.100.10"    # Client node interconnect IP
export IB_SUBNET="192.168.100.0/24"  # Interconnect subnet
conda create -n cupynumeric -c legate cupynumeric
mpirun -n 2 -npernode 1 --hostfile $HOME/shared/hostfile hostname
legate --launcher mpirun --launcher-extra="--hostfile $HOME/shared/hostfile --mca oob_tcp_if_include $IB_SUBNET --mca pml ucx" --nodes 2 --gpus 1 --fbmem 40960 --cpus 4 ./examples/gemm.py -n 20000 -i 2
```

นั่นคือหลักหมุดสำคัญ ระหว่างนั้น Step 3–5 ของ README จะเปิด firewall บน interface ของ QSFP (`sudo ufw allow in on …`) ติดตั้ง NFS server และ client เพิ่มบรรทัดลงใน `/etc/exports` และ (ถ้าต้องการ) `/etc/fstab` และติดตั้ง Miniforge ลงในพื้นที่แชร์ ทั้งหมดนี้เป็นการเปลี่ยนแปลงระบบจริงบน Spark ทั้งสองเครื่อง: ทำตาม README สำหรับขั้นเหล่านี้ และใช้ Step 10 ของมันเพื่อย้อนกลับ

✓ Checkpoint: คุณบอกชื่อ playbook ด้าน data science ห้าตัวที่รันบน Spark ได้ และบอกได้ว่าตัวไหนต้องใช้ Spark เครื่องที่สอง

## 6 · Kernel และการเทรนตั้งแต่ศูนย์

### cuTile kernels (TileGym) 🟩 Spark · 60 นาที · ดิสก์ ≥50 GB

cuTile เป็น Python DSL สำหรับเขียน GPU kernel ที่คอมไพล์เป็น Tile IR คุณจึงเขียนเป็น tile ไม่ใช่ thread TileGym คือชุด benchmark ของ NVIDIA สำหรับมัน playbook มีสามแทร็ก: benchmark kernel FMHA, MatMul, RMSNorm, RoPE และ SwiGLU; รัน Qwen2-7B หรือ DeepSeek-V2-Lite โดย monkey-patch kernel ของ cuTile เข้าไป; และสร้าง flash-attention kernel จาก pseudocode ใช้มันเมื่อโมเดลช้าและคุณอยากเห็นว่า custom kernel จะช่วยได้ตรงไหน

```bash
# on: spark
docker pull nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu22.04
docker run --gpus all -it --rm \
  -v ~/TileGym:/workspace/TileGym \
  nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu22.04 \
  /bin/bash
# inside the container: run the playbook's Step 1 apt-get / pip block (Python, nsys, torch 2.9.1 cu130), then:
git clone https://github.com/NVIDIA/TileGym
cd TileGym
git checkout v1.3.0
pip install .
cd tests/benchmark/
python bench_fused_attention.py
```

**Expected output** (REFERENCE — อ้างอิงจาก playbook ซึ่งไม่ได้ระบุว่าตัวเลขมาจากแพลตฟอร์มใด อย่าถือว่าเป็นตัวเลขของ Spark)

```text
fused-attention-batch4-head32-d128-fwd-causal=True-float16-TFLOPS:
     N_CTX     CuTile
0   1024.0  58.188262
1   2048.0  80.906892
2   4096.0  86.189532
3   8192.0  88.891086
4  16384.0  89.491869
✓ PASSED: bench_fused_attention.py
```

### Triton fine-tuning kernels 🟪 Station only · ~2 ชม. · ดิสก์ ≥150 GB

profile ขั้นตอน fine-tuning ของ Llama 3.1 8B ด้วย `torch.profiler` แล้วเขียน Triton kernel สองตัว: fused RMSNorm และ fused cross-entropy ที่ใช้ online softmax ซึ่งไม่ต้องเก็บ tensor logits ทั้งก้อน จากนั้น patch ทั้งสองเข้าไปใน training loop แล้ววัดผล เป็นพี่น้องฝั่งการเทรนของ cuTile เขียนไว้สำหรับ GB300 (`--gpus '"device=N"'` ตรึง GB300 บน Station ที่มีหลาย GPU) บน Spark ของคุณ cuTile playbook คือเส้นทางด้าน kernel ที่ตารางแพลตฟอร์มรองรับ

```text
# on: DGX Station (not a Spark) — reference only
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-kernel-dev-ft/assets      # README says client-hardware-playbooks/…
docker build -t kernel-dev-ft .
python profile_baseline.py            # inside the kernel-dev-ft container
python rmsnorm_test.py
python finetune_optimized.py
```

### NanoChat 🟪 Station only · ติดตั้ง ~30 นาที, รัน d24 เต็ม 12+ ชม.

nanochat ของ Andrej Karpathy เทรนโมเดลสไตล์ ChatGPT ขนาดเล็กตั้งแต่ต้นจนจบ: BPE tokenizer, pretraining, chat fine-tuning, รายงาน แล้วแชตผ่านเว็บหรือ CLI change log ของ README เองเขียนว่า: **"Removed DGX Spark support (not ready yet); playbook is Station / single-node only."** ดังนั้นอย่าวางแผนรันบน Spark ของคุณในสัปดาห์นี้ ต้องใช้คีย์ Weights & Biases และโทเค็น Hugging Face

```text
# on: DGX Station (not a Spark) — reference only
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-nanochat/assets
chmod +x setup.sh launch.sh
./setup.sh
./launch.sh
```

### NVFP4 pretraining ด้วย Megatron Bridge 🟪 Station only · 30 นาที

Module 07 ใช้ NVFP4 เพื่อย่อขนาดโมเดลที่เทรนเสร็จแล้ว playbook นี้ใช้มัน **ระหว่าง pretraining**: recipe `bf16_with_nvfp4_mixed` ของ Megatron Bridge บน Llama 3.1 8B กับข้อมูลจำลอง (mock data) โดยเก็บสี่ layer สุดท้ายไว้เป็น BF16 เพื่อความเสถียร แล้วเปรียบเทียบกับการรัน BF16 ล้วนผ่าน `--disable-fp4`

**Expected output** (REFERENCE — ตาราง "Measured results" ของ playbook, GB300 ตัวเดียว, `nvcr.io/nvidia/nemo:26.04`, global batch 64, sequence 4096)

```text
| Precision | Recipe | Avg step time | Throughput (Model TFLOP/s/GPU) | Peak VRAM |
|---|---|---|---|---|
| BF16 baseline | `bf16_mixed()` | 9.05 s | ~1399 | 221.6 GB |
| NVFP4 (last-4 BF16) | `bf16_with_nvfp4_mixed()` + `first_last_layers_bf16=True`, `num_layers_at_end_in_bf16=4` | **5.39 s** | **~2347** | **207.8 GB** |
```

ตัวเลข peak memory ทั้งสองสูงกว่า 128 GB ของ Spark หนึ่งเครื่อง ซึ่งเป็นเหตุผลหนึ่งที่ตารางแพลตฟอร์มระบุแค่ DGX Station การเปิดรันคือ `docker run … nvcr.io/nvidia/nemo:${TAG}` แล้วภายใน container:

```text
# on: DGX Station (not a Spark) — reference only
torchrun --nproc_per_node=1 pretrain_llama.py > nvfp4.log 2>&1
```

✓ Checkpoint: คุณบอกได้ว่า playbook ด้าน kernel ตัวไหนรันบน Spark (cuTile) และยกบรรทัดใน README ที่ตัด NanoChat ออกไปในตอนนี้ได้

## 7 · แพลตฟอร์มและ multi-GPU

### Brev 🟩 Spark · 10 นาที · บัญชี Brev

NVIDIA Brev ลงทะเบียน Spark ของคุณเป็น GPU node ที่มีการจัดการ (managed) หลังลงทะเบียน เพื่อนร่วมทีมที่คุณเพิ่มใน Brev UI จะเข้าถึงมันผ่าน SSH ได้จากทุกที่ และคุณสร้าง environment มาตรฐานได้ด้วย "Launchables" ใช้มันเมื่อมีมากกว่าหนึ่งคนที่ต้องใช้ Spark ของคุณ และคุณไม่อยากแจกสิทธิ์เข้า tailnet (Module 01) การลงทะเบียนทำผ่านเว็บ UI ของ Brev: **Registered Compute → Register Compute** จะแสดงวิธีติดตั้ง CLI และคำสั่งลงทะเบียนที่ต้องรันบน Spark ด้วย sudo playbook ไม่ได้พิมพ์คำสั่งตายตัวสำหรับขั้นนั้น ให้คัดลอกจาก pop-up คำสั่งที่มันระบุชื่อไว้คือ:

```bash
# on: spark
brev set <my-org>        # if the node registered into the wrong org, then redo registration
brev refresh             # if `brev shell <name>` fails with stale CLI state
brev deregister          # cleanup: remove the Spark from Brev
```

### Multi-Instance GPU (MIG) 🟪 Station only · 15 นาที

MIG แบ่ง GPU ระดับ data centre หนึ่งตัวออกเป็น instance ที่แยกขาดจากกัน แต่ละตัวมีหน่วยความจำและพลังประมวลผลของตัวเอง ผู้ใช้หรืองานหลายตัวจึงใช้ GPU ร่วมกันได้โดยไม่รบกวนกัน playbook เขียนไว้สำหรับ GPU B300 ใน DGX Station พร้อม profile เช่น `1g.35gb` (ID 19) และเตือนว่าการเปิดหรือปิด MIG มีผลกับทุก workload บน GPU นั้น

```text
# on: DGX Station (not a Spark) — reference only
sudo nvidia-smi -mig 1
nvidia-smi mig -lgip -i 0
sudo nvidia-smi mig -cgi 19,19,19,19,19,19,19 -C -i 0
nvidia-smi -L
sudo nvidia-smi -mig 0
```

### สร้าง multi-GPU AI PC 🟧 RTX only · 13 นาที

การ์ด GeForce/RTX PRO สองใบที่เหมือนกันใน PC เครื่องเดียว: tensor parallel ของ llama.cpp (`-sm tensor`) ข้ามการ์ดทั้งสอง การแบ่ง tensor-parallel ของ LM Studio และ node MultiGPU CFG Split ของ ComfyUI พร้อมคู่มือเลือกชิ้นส่วน (PCIe lane, PSU, ระบบระบายความร้อน) สิ่งที่เทียบเท่าบน Spark ของคุณคือ Module 02: Spark สองเครื่องต่อกันด้วยสาย 200 Gb/s พร้อม tensor parallel ของ vLLM ใน Module 05

```text
# on: dual-GPU RTX PC (not a Spark) — reference only
nvidia-smi -L
<app> -m <model_path> -sm tensor -fa 1
```

### Connect two stations 🟪 Station only · 60 นาที

Module 02 ฉบับ DGX Station: QSFP rail แบบ ConnectX-8 สองเส้น เส้นละ 400 Gb/s, RoCEv2, MTU 9000 และการตรวจ GPUDirect ควบคุมจาก control host แยกต่างหากด้วยสคริปต์ที่มีเลขลำดับ โครงสร้างเหมือนสิ่งที่คุณทำกับ Spark สองเครื่อง: ตรวจการเข้าถึง ต่อสาย ตั้งค่า rail ตรวจสอบความถูกต้อง แล้วรันทดสอบ bandwidth

```text
# on: control host for two DGX Stations — reference only
git clone https://github.com/NVIDIA/dgx-spark-playbooks client-hardware-playbooks
cd client-hardware-playbooks/nvidia/playbook-connect-two-stations/assets
cp 00_env.local.example 00_env.local
./01_probe_access.sh
./02_push_assets.sh
./07_validate_setup.sh
```

✓ Checkpoint: คุณรู้ว่าจะใช้ playbook ด้านแพลตฟอร์มตัวไหนเพื่อแชร์ Spark กับเพื่อนร่วมทีม (Brev) และโมดูลไหนของ Spark ที่ใช้แทน playbook multi-GPU PC และ two-station ได้ (Module 02)

## 8 · Serving และเอเจนต์ที่ใช้ได้เฉพาะ Station

สามตัวนี้สำหรับ DGX Station แต่แต่ละตัวเทียบได้กับสิ่งที่คุณสร้างไปในสัปดาห์นี้

### SGLang inference (ฉบับ Station) 🟪 Station only · 20–30 นาที

engine เดียวกับ Module 06 ปรับแต่งสำหรับ GB300 (SM103): `lmsysorg/sglang:latest-cu130`, `--attention-backend flashinfer` (README บอกว่า backend ตั้งต้นทำ CUDA-graph capture บน SM103 ไม่สำเร็จ) ตรวจ prefix caching ผ่าน `#cached-token` ใน log, output ตาม JSON schema และสคริปต์ benchmark แบบหลายเทิร์น บน Spark ของคุณ ให้ใช้ playbook `sglang` จาก Module 06 ส่วนที่ควรลอกไปใช้คือ **วิธีการใน Step 7**: ยืนยัน prefix caching จาก log ของเซิร์ฟเวอร์ ไม่ใช่จาก latency ตามเวลานาฬิกา

```text
# on: DGX Station (not a Spark) — reference only
docker pull lmsysorg/sglang:latest-cu130
docker logs sglang-server 2>&1 | grep "cached-token" | tail -10
```

### Agent skills สำหรับ DGX Station 🟪 Station only · 15 นาที

Agent Skill สี่ตัวพร้อม CLI `dgx-assist` ที่ทำให้ coding agent (Claude Code, Codex, Gemini CLI, Cursor) ตรวจฮาร์ดแวร์จริง ค้นคำแนะนำของ NVIDIA ที่ตรึงเวอร์ชันไว้ จับคู่โมเดลกับ recipe ที่ผ่านการรับรอง ตรวจความพร้อมก่อนเริ่ม (preflight) **ขอให้คุณอนุมัติ** แล้วจึงลงมือทำและตรวจสอบผล นี่คือแนวคิดด้าน governance ของ Module 15 ที่นำมาใช้กับ coding agent บรรทัด clone ใน README ใช้ URL เว็บแบบ `…/blob/main/…` ซึ่ง `git clone` ดึงไม่ได้ บรรทัดด้านล่างคือการแก้ของคอร์ส

```text
# on: DGX Station (not a Spark) — reference only
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-dgx-station-ai-skills      # course fix: README clones a /blob/ URL
assets/install.sh install \
  --harness codex \
  --target /path/to/project \
  --dry-run
```

### Healthcare agents 🟪 Station only · 60 นาที · หน่วยความจำ GPU ว่าง ≥150 GB, ดิสก์ ≥200 GB

OpenClaw agent หกตัวภายใน sandbox ของ OpenShell query ระเบียนผู้ป่วย FHIR สังเคราะห์ หาช่องว่างในการดูแลรักษา (care gap) และทำนายโครงสร้างโปรตีนด้วย OpenFold3 โดยมี Nemotron 3 Super serve ในเครื่องผ่าน Ollama sandbox อนุญาตแค่ endpoint ในรายการสั้น ๆ มันรวม Module 15 (OpenShell) และ 17 (OpenClaw) เข้าด้วยกัน ตัวเลขใน README เอง (Nemotron 3 Super ใช้ ~94 GB บวก OpenFold3 ~40–80 GB) รวมแล้วเกิน 128 GB ของ Spark หนึ่งเครื่อง นี่เป็นการสาธิตบนข้อมูลสังเคราะห์ ไม่ใช่อุปกรณ์ทางการแพทย์ ตามที่ข้อจำกัดความรับผิดชอบ (disclaimer) ของมันระบุ

```text
# on: DGX Station (not a Spark) — reference only
docker compose up -d ollama openfold3
docker compose up model-pull
make status
bash scripts/ensure_openshell_gateway.sh
make setup
make check
```

✓ Checkpoint: สำหรับ playbook ที่ใช้ได้เฉพาะ Station แต่ละตัวในส่วนนี้ คุณบอกได้ว่าโมดูลไหนของ Week 25 ที่ให้ทักษะเดียวกันบน Spark (06, 15, 15 + 17)

## 9 · เลือก playbook ถัดไป: แนะนำ แล้ววางงบ

ตอนนี้คุณรู้จักครบทั้ง 26 ตัวแล้ว มีแล็บเล็ก ๆ สองตัวช่วยคุณเลือก

**Lab 21-2** รับเป้าหมายเป็นภาษาธรรมดา แล้วจัดอันดับ playbook ทั้ง 64 ตัวด้วย TF-IDF cosine similarity บนส่วน overview ของ README แต่ละตัว ไม่มีโมเดลและไม่ใช้เครือข่าย และมันพิมพ์คำที่ตรงกันออกมาให้ คุณจึงเห็นได้ว่า *ทำไม* มันถึงเลือกแบบนั้น:

```bash
# on: laptop
.venv/bin/python week25/21_playbook_atlas/labs/lab21_2_goal_recommender.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ แสดง 2 จาก 10 เป้าหมายตัวอย่าง)

```
▣ STEP 1 · index every playbook overview
◆ 64 playbooks · vocabulary 3401 words after stopwords · all platforms (add --spark to filter)

▣ STEP 2 · recommend for 10 goals

→ goal: "describe what my webcam sees in real time"
│ score  playbook               Spark?  taught in  matched words
│ ─────  ─────────────────────  ──────  ─────────  ──────────────
│ 0.147  live-vlm-webui         ✓       Module 21  webcam, real
│ …
· cite: Stream Real-Time Video to a Vision Language Model — https://build.nvidia.com/spark/live-vlm-webui
        source: dgx-spark-playbooks/nvidia/playbook-live-vlm-webui/README.md

→ goal: "serve an LLM with an OpenAI-compatible API and tool calling"
│ score  playbook  Spark?  taught in  matched words
│ ─────  ────────  ──────  ─────────  ─────────────────────────────────
│ 0.187  nemotron  ✓       Module 06  tool, serve, openai, compatible
│ 0.157  openclaw  ✓       Module 17  calling, openai, compatible, tool
│ 0.117  vllm      ✓       Module 05  serve, openai, compatible, api
…
▣ STEP 3 · where keyword search breaks — two goals it gets wrong (all platforms)

→ goal: "split one GPU between several users"
│ score  playbook     Spark?  taught in  matched words
│ ─────  ───────────  ──────  ─────────  ───────────────────
│ 0.105  multi-gpu    ✕       Module 21  split, gpu, between
│ 0.072  nccl         ✓       Module 02  users, between, gpu
│ 0.050  single-cell  ✓       Module 21  users, gpu, between
◆ a person would pick 'mig': it is not in the top three
…
◆ a person would pick 'nanochat': it is #2 here
```

ส่งเป้าหมายของคุณเองเป็น argument ได้ `--spark` เก็บเฉพาะ playbook ที่ตารางระบุ DGX Spark และความต่างนี้สำคัญ:

```bash
# on: laptop
.venv/bin/python week25/21_playbook_atlas/labs/lab21_2_goal_recommender.py "fine-tune a robot policy"
.venv/bin/python week25/21_playbook_atlas/labs/lab21_2_goal_recommender.py "fine-tune a robot policy" --spark
```

**Expected output** (บันทึกจาก Mac เครื่องนี้ แสดงสองแถวบนสุดของแต่ละการรัน โดยเพิ่มเส้นคั่นระหว่างสองการรัน)

```
│ 0.207  gr00t           ✕       Module 21  robot, tune, fine, policy
│ 0.157  nemo-fine-tune  ✓       Module 11  fine, tune
─── with --spark ───
│ 0.157  nemo-fine-tune  ✓       Module 11  fine, tune
│ 0.141  llama-factory   ✓       Module 09  fine, tune
```

ตัวที่ตรงที่สุดคือ GR00T ซึ่งใช้ได้เฉพาะ Station เมื่อใส่ `--spark` คุณจะได้เครื่องมือ fine-tuning ทั่วไปแทน: เป็นคำตอบที่ตรงไปตรงมาว่าวันนี้ยังไม่มี Spark playbook ตัวไหนที่ fine-tune policy ของหุ่นยนต์

**Lab 21-3** รวมเวลา ดิสก์ และ "สิ่งที่ต้องเตรียม" สำหรับ playbook ชุดใดก็ได้ โดยใช้แค่ตัวเลขใน README เอง ค่าเริ่มต้นจะวางแผนให้ Spark playbook ทุกตัวใน atlas นี้:

```bash
# on: laptop
.venv/bin/python week25/21_playbook_atlas/labs/lab21_3_budget_planner.py
```

**Expected output** (บันทึกจาก Mac เครื่องนี้)

```
▣ STEP 1 · 13 playbooks — every atlas playbook that runs on a DGX Spark
│ playbook                  README time  README disk     bring
│ ────────────────────────  ───────────  ──────────────  ────────────────────────────────────────────────────
│ brev                      10 min       — (none given)  Brev account
│ comfyui                   45 min       30 GB           HF token
│ cuda-x-data-science       30 min       — (none given)  Kaggle key
│ cupynumeric               60 min       — (none given)  2nd Spark
│ cutile-kernels            60 min       50 GB           —
│ isaac                     30 min       50 GB           —
│ jax                       2 h          — (none given)  —
│ live-vlm-webui            30 min       — (none given)  webcam
│ multi-modal-inference     60 min       — (none given)  HF token
│ portfolio-optimization    20 min       30 GB           —
│ single-cell               15 min       30 GB           —
│ spark-reachy-photo-booth  2 h          — (none given)  HF token, NGC key, NVIDIA API key, Reachy Mini, mon…
│ vss                       45 min       10 GB           NGC key, NVIDIA API key

▣ STEP 2 · time: add it up and pack it into sessions
◆ total from the READMEs: 645 min ≈ 10.8 h (first figure of each 'Estimated time' line, upper end of a range)
│ session 1  ████████████████████ 180 min  jax + cupynumeric
│ session 2  ████████████████████ 180 min  spark-reachy-photo-booth + cutile-kernels
│ session 3  ████████████████████ 180 min  multi-modal-inference + comfyui + vss + cuda-x-data-science
│ session 4  ████████████░░░░░░░░ 105 min  isaac + live-vlm-webui + portfolio-optimization + single-cell + brev

▣ STEP 3 · disk: compare with the 500 GB you set aside (--free-gb)
│ cutile-kernels             50 GB  ████████████████████████
│ isaac                      50 GB  ████████████████████████
│ comfyui                    30 GB  ██████████████░░░░░░░░░░
│ portfolio-optimization     30 GB  ██████████████░░░░░░░░░░
│ single-cell                30 GB  ██████████████░░░░░░░░░░
│ vss                        10 GB  █████░░░░░░░░░░░░░░░░░░░
◆ keep everything: 200 GB · clean up after each: peak 50 GB
◆ 7 README(s) give no disk figure: brev, cuda-x-data-science, cupynumeric, jax, live-vlm-webui, multi-modal-inference, spark-reachy-photo-booth — budget those yourself
◆ against 500 GB: fits
…
═ 13 playbooks · 10.8 h of README time · 200 GB if you keep everything (+7 without a figure) · fits in 500 GB.
```

ให้อ่าน "200 GB" เป็นขอบล่าง: README เจ็ดตัวไม่ได้ให้ตัวเลขดิสก์ และ 30 GB ของ ComfyUI เป็นแค่ quick start (video tier ต้องใช้ 70–230 GB) วางแผนชุดของคุณเองด้วย `lab21_3_budget_planner.py comfyui vss isaac --free-gb 300 --session-min 120`

✓ Checkpoint: คุณรันแล็บทั้งสองด้วยเป้าหมายของคุณเองแล้ว และมีรายชื่อ playbook สั้น ๆ พร้อมงบเวลาและดิสก์

## Labs — รันแล็บได้ที่นี่

**labs/lab21_1_playbook_atlas.py** — ตารางเดียวของ playbook ทั้ง 64 ตัว: เวลา แพลตฟอร์ม โมดูลที่ครอบคลุม ธีม และหลักฐานของ slug

**labs/lab21_2_goal_recommender.py** — playbook ไหนเหมาะกับเป้าหมายของฉัน? TF-IDF บน README พร้อมการอ้างอิง และกรณีที่รู้ว่าพลาดสองกรณี

**labs/lab21_3_budget_planner.py** — เวลา ดิสก์ และบัญชีหรือฮาร์ดแวร์ที่ต้องใช้สำหรับ playbook ชุดหนึ่ง จัดเป็นรอบ (session)

ทั้งสามรันบนแล็ปท็อปแบบออฟไลน์ในไม่กี่วินาที ไม่แตะ Spark เลย: อ่านแค่ clone ของ playbook และ `week25/AUTHORING.md`

## Try it yourself — ลองทำเอง

**แบบฝึกหัด 21 — ตัว route เป้าหมายและชุดทดสอบของมัน** เปิด `week25/21_playbook_atlas/exercises/ex21_goal_router.py` ในไฟล์มี `TODO` สามจุด:

1. `tokenize(text)`: แปลงเป็นตัวพิมพ์เล็ก แยกคำตรงอักขระที่ไม่ใช่ตัวอักษรหรือตัวเลข ตัดคำยาว 1 ตัวอักษรและ stopword ออก
2. `idf(docs)`: inverse document frequency แบบ smooth `log((1+N)/(1+df)) + 1`
3. `TEST_SET`: คู่ `(goal, expected playbook)` อย่างน้อย 8 คู่ ครอบคลุม playbook 5 ตัวขึ้นไป โดย 3 ตัวขึ้นไปมาจาก atlas นี้ กฎคือ **อธิบายเป้าหมาย อย่าเอ่ยชื่อเครื่องมือ**: เป้าหมายต้องไม่มีคำที่ยาว 4 ตัวอักษรขึ้นไปจาก slug ของ playbook ตัวนั้น

ตัวตรวจจะทำ index README จริงด้วยฟังก์ชันของคุณ รันเป้าหมายที่มีมาให้สามข้อ แล้วกำหนดว่า playbook ที่คุณคาดไว้ทุกตัวต้องติดสามอันดับแรก

```bash
# on: laptop
.venv/bin/python week25/21_playbook_atlas/exercises/ex21_goal_router.py
```

**Expected output** (คำตอบอ้างอิง บันทึกจาก Mac เครื่องนี้)

```
✓ tokenize: 'Fine-Tune FLUX.1 on the GPU!' → fine, tune, flux, gpu
✓ idf: word in every doc = 1.0 · word in 1 of 2 docs ≈ 1.405
✓ built-in goals: 3/3 land in the top three
✓ test set: 11 goals · 11 playbooks · 10 from the atlas · no goal names its tool

▣ your test set against your router (top three)
│ ✓ #1   live-vlm-webui           ← "stream my laptop camera to a vision model and read what it says"   (got live-vlm-webui, lm-studio, connect-to-your-spark)
│ ✓ #3   comfyui                  ← "turn a text prompt into a picture with a node graph"   (got txt2kg, sglang-inference, comfyui)
│ ✓ #1   vss                      ← "ask questions about hours of recorded video footage"   (got vss, visual-gen-ai, vlm-finetuning)
│ ✓ #2   isaac                    ← "simulate a humanoid robot and train a locomotion policy"   (got gr00t, isaac, spark-reachy-photo-booth)
│ ✓ #1   single-cell              ← "RNA sequencing clustering with scanpy on GPU"   (got single-cell, topic-modeling, connect-multiple-sparks)
│ ✓ #1   portfolio-optimization   ← "minimize conditional value at risk for thousands of assets"   (got portfolio-optimization, flux-finetuning, connect-two-sparks)
│ ✓ #1   cuda-x-data-science      ← "pandas and scikit-learn on the GPU with zero code changes"   (got cuda-x-data-science, topic-modeling, vibe-coding)
│ ✓ #1   cupynumeric              ← "numpy style arrays spread across two machines"   (got cupynumeric, jax, nanochat)
│ ✓ #1   cutile-kernels           ← "benchmark flash attention written in a python tile DSL"   (got cutile-kernels, sglang, sglang-inference)
│ ✓ #2   spark-reachy-photo-booth ← "a small desk robot that takes pictures, talks and restyles them"   (got isaac, spark-reachy-photo-booth, gr00t)
│ ✓ #1   vllm                     ← "serve a model with an OpenAI-compatible endpoint and high throughput"   (got vllm, sglang, nvfp4-quantization)
✓ all 11 expected playbooks in the top three · top-1 8/11
```

ไฟล์ตั้งต้นที่ยังไม่ได้แก้จะหยุดที่สองบรรทัดแรกด้วย `✕ TODO 1` และ `✕ TODO 2` แล้วจบด้วย exit code 1

<details><summary>คำใบ้ — ทำไม IDF ถึงบวก 1 ทั้งในและนอก log?</summary>

`1 +` ด้านในทำให้อัตราส่วนยังนิยามได้เมื่อคำนั้นปรากฏในทุกเอกสาร (`df = N`) และให้น้ำหนักที่เป็นค่าจำกัดกับคำที่ index ไม่เคยเห็น `+ 1` ด้านนอกทำให้คำที่พบบ่อยเหล่านั้นมีน้ำหนัก 1.0 แทนที่จะเป็น 0 จึงยังนับอยู่บ้าง นี่คือการ smooth ที่ scikit-learn ใช้เป็นค่าเริ่มต้น

</details>

<details><summary>คำใบ้ — มีเป้าหมายที่พลาดอยู่เรื่อย ๆ</summary>

พิมพ์ token ของเป้าหมายด้วย `tokenize()` ของคุณ แล้วค้นหาคำเหล่านั้นใน README (`grep -i -c webcam dgx-spark-playbooks/nvidia/playbook-live-vlm-webui/README.md`) ถ้า README ไม่เคยใช้คำของคุณเลย router ก็หามันไม่เจอ ให้เขียนเป้าหมายใหม่ด้วยคำศัพท์ของ README หรือเก็บกรณีที่พลาดไว้เป็นข้อจำกัดที่รู้อยู่แล้ว แล้วเลือกเป้าหมายอื่น ทั้งสองทางเป็นผลลัพธ์ที่ซื่อตรง แต่การซ่อนกรณีที่พลาดไม่ใช่

</details>

<details><summary>ท้าทายเพิ่ม — เอาชนะการค้นด้วยคีย์เวิร์ด</summary>

เพิ่มตารางคำพ้องความหมายเล็ก ๆ (`split → partition`, `users → instances`) ที่ใช้กับเป้าหมายก่อน `tokenize()` แล้วทำให้เป้าหมายเรื่อง MIG ใน step 3 ของ lab 21-2 ขึ้นเป็นอันดับ 1 จากนั้นตรวจว่าชุดทดสอบทั้งหมดของคุณยังผ่าน: การแก้เป้าหมายหนึ่งอาจทำให้อีกเป้าหมายพัง ซึ่งเป็นเหตุผลที่ต้องมีชุดทดสอบ

</details>

✓ Checkpoint: ตัวตรวจพิมพ์ `all … expected playbooks in the top three`

## Troubleshooting — แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `✕ …/dgx-spark-playbooks/nvidia not found` จากแล็บใดก็ได้ | clone playbook ไว้ที่ root ของ repo: `git clone https://github.com/NVIDIA/dgx-spark-playbooks` |
| Lab 21-1 พิมพ์ `✕ no theme yet: <slug>` | NVIDIA เพิ่ม playbook หลังจากเขียนโมดูลนี้ อ่าน README ของมันแล้วเพิ่มลงใน `THEMES` ใน `lab21_1_playbook_atlas.py` |
| `cd client-hardware-playbooks/…` ของ playbook ล้มเหลวทันทีหลัง `git clone …/dgx-spark-playbooks` | clone ชื่อ `dgx-spark-playbooks/` ให้ใช้ path นั้น (ส่วนที่ 5 และ 6 แสดงไว้) |
| `git clone https://github.com/…/blob/main/…` ใน README ของ Agent Skills ล้มเหลว | URL แบบ `/blob/` เป็นหน้าเว็บ ไม่ใช่ repository ให้ clone ที่ root ของ repository แล้ว `cd dgx-spark-playbooks/nvidia/playbook-dgx-station-ai-skills` |
| ลิงก์ build.nvidia.com จากหน้านี้คืน 404 | Spark slug 12 ตัวยืนยันแล้วว่าอยู่ใน index ของ /spark (lab 21-1 step 4) cuPyNumeric ไม่อยู่ในนั้น และลิงก์ Station กับ RTX ยังไม่ได้ยืนยัน ค้นชื่อ playbook บน build.nvidia.com README ใน clone คือแหล่งอ้างอิงหลัก |
| Live VLM WebUI: กล้องใช้ได้ แต่การวิเคราะห์ล้มเหลวด้วย `InvalidStateError` | WebRTC ผ่าน tunnel `ssh -L` ไม่ได้ เปิด `https://<SPARK_IP>:8090` โดยตรงจากเครื่องที่อยู่ในเครือข่ายเดียวกัน |
| หน่วยความจำไม่พอใน ComfyUI, VSS หรือโน้ตบุ๊ก RAPIDS ทั้งที่ `free -g` แสดงว่ายังมีที่ว่าง | ตามหมายเหตุเรื่อง unified memory ของ playbook: ล้าง page cache ด้วย `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| ตัวแนะนำจัด playbook ที่ใช้ได้เฉพาะ Station ไว้อันดับแรก | นั่นคือตัวที่ตรงที่สุดอย่างซื่อตรง ใส่ `--spark` เพื่อดูเฉพาะ playbook ที่ตารางระบุ DGX Spark |

## Next — บทถัดไป

นี่คือโมดูลสุดท้าย [กลับไปที่ capstone](../20_capstone_sovereign_agent/TUTORIAL.md) เพื่อต่อ fine-tune → serve → gateway → NAT agent ใน sandbox แบบครบวงจร แล้วเพิ่ม playbook หนึ่งตัวจากแผนงบของคุณเข้าไป: ComfyUI เป็น tool สร้างภาพให้เอเจนต์ หรือ VSS เป็น tool ด้านวิดีโอ หากต้องการดูภาพรวมทั้งสัปดาห์ กลับไปที่ [ภาพรวม Week 25](../README.md)
