# ▶ Spark Lab 21 — Playbook atlas: every other Spark playbook

> Part of Week 25 · DGX Spark: fine-tune, serve, and build sandboxed agents. You type the commands, you see the real output. Every lab also runs in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's playbook, or a clearly marked EXAMPLE.

**What you'll actually do**
- Scan all 64 official playbooks in the local clone and see, in one table, how long each takes, whether it runs on a DGX Spark, and which Week 25 module teaches it.
- Tour the 26 playbooks that Modules 01–20 do not cover, in seven themes: images and video, vision agents, robotics, data science and HPC, kernels and pretraining, platform and multi-GPU, and Station-only serving and agents.
- For each one, learn what it is for, what it needs, the commands that get it running, and whether it applies to your Spark at all (13 do, 13 do not).
- Ask a small TF-IDF recommender "which playbook for my goal?", and see where keyword search fails.
- Plan the time, disk and extra kit for the playbooks you pick, then write a test set the recommender must pass.

**Time** ~50 min · **Difficulty** beginner · **Hardware** none for the labs (1 Spark to run the playbooks themselves; cuPyNumeric needs 2)

**Official playbooks covered:** Runs on DGX Spark: [ComfyUI](https://build.nvidia.com/spark/comfyui) · [Multi-modal inference](https://build.nvidia.com/spark/multi-modal-inference) · [Live VLM WebUI](https://build.nvidia.com/spark/live-vlm-webui) · [VSS](https://build.nvidia.com/spark/vss) · [Isaac Sim and Isaac Lab](https://build.nvidia.com/spark/isaac) · [Reachy photo booth](https://build.nvidia.com/spark/spark-reachy-photo-booth) · [CUDA-X data science](https://build.nvidia.com/spark/cuda-x-data-science) · [Portfolio optimization](https://build.nvidia.com/spark/portfolio-optimization) · [Single-cell RNA](https://build.nvidia.com/spark/single-cell) · [JAX](https://build.nvidia.com/spark/jax) · [cuPyNumeric](https://build.nvidia.com/spark/cupynumeric) · [cuTile kernels](https://build.nvidia.com/spark/cutile-kernels) · [Brev](https://build.nvidia.com/spark/brev). DGX Station only: [GR00T](https://build.nvidia.com/station/gr00t) · [Topic modeling](https://build.nvidia.com/station/topic-modeling) · [Triton kernels](https://build.nvidia.com/station/kernel-dev-ft) · [NanoChat](https://build.nvidia.com/station/nanochat) · [NVFP4 pretraining](https://build.nvidia.com/station/nvfp4-pretraining) · [MIG](https://build.nvidia.com/station/mig) · [Connect two stations](https://build.nvidia.com/station/connect-two-stations) · [SGLang inference](https://build.nvidia.com/station/sglang-inference) · [Agent skills](https://build.nvidia.com/station/dgx-station-ai-skills) · [Healthcare agent](https://build.nvidia.com/station/healthcare-agent). RTX PC only: [Visual gen AI](https://build.nvidia.com/playbooks/visual-gen-ai) · [Controlled video](https://build.nvidia.com/playbooks/video-gen-guide) · [Multi-GPU AI PC](https://build.nvidia.com/playbooks/multi-gpu)

> 🔎 **About these links.** Each URL is built from the playbook's directory name (`playbook-<slug>` → `<slug>`). **12 of the 13 Spark links are confirmed**: their slugs appear on the build.nvidia.com/spark index (fetched 2026-09-29). **cuPyNumeric is not on that index**, even though its README lists DGX Spark, so its link may 404. The index also lists one Spark playbook, `pair`, that has no directory in this clone. The 13 Station and RTX links are **unverified**: the /spark index lists Spark playbooks only. Two of them (`/station/connect-two-stations` and `/station/topic-modeling`) do appear as links inside the clone. Lab 21-1 step 4 prints all of this evidence. If a link 404s, search the title on [build.nvidia.com/spark](https://build.nvidia.com/spark); the README in the clone is always the source of truth.

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| This repo's Python | `.venv/bin/python --version` → 3.13 | runs the three labs and the exercise |
| The playbook clone | `ls dgx-spark-playbooks/nvidia \| head` | every number in this module is read from it |
| A Spark (optional) | `ssh -o BatchMode=yes spark-a true` | only to run the playbooks themselves, not the labs |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
.venv/bin/python --version
ls -d dgx-spark-playbooks/nvidia/playbook-* | wc -l
git -C dgx-spark-playbooks log -1 --format='%h %cd'
```

**Expected output** (captured on this Mac)

```
Python 3.13.13
      65
3410c65 Thu Sep 10 21:01:40 2026 +0000
```

The clone moves. If your count or commit differs, the labs still work: they read whatever is there, and lab 21-1 warns when a playbook has no theme yet.

✓ Checkpoint: the clone exists and you know its commit, so you can say which version of the playbooks your atlas describes.

## 1 · The map: 64 playbooks, 38 already covered, 26 here

NVIDIA's repository holds more than DGX Spark playbooks. Each README has a **Supported hardware platforms** table, and some playbooks list only DGX Station (GB300) or an RTX PC. Before you spend an evening on a playbook, check that table. Lab 21-1 reads it for all 64 playbooks and joins it with the module table in `week25/AUTHORING.md`:

```bash
# on: laptop
.venv/bin/python week25/21_playbook_atlas/labs/lab21_1_playbook_atlas.py
```

**Expected output** (captured on this Mac; the 64-row table is shortened here)

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

Three things the table tells you:

1. **Only 48 of 64 list DGX Spark.** A few playbooks this course used anyway (`llms` and `fine-tuning` are RTX-only in the current clone, `local-coding-agent` is Station-only) are close enough that Modules 03, 10 and 18 adapted them. The atlas does not stretch that far: a Station-only playbook is marked Station-only.
2. **The time column is NVIDIA's estimate**, the first figure of each README's "Estimated time" line (upper end of a range). It is not a measurement, and first runs with big downloads take longer.
3. **`sglang-inference` has no platform table.** Its prerequisites say "NVIDIA DGX Station with GB300", so the lab marks it `(inferred)`. It is the Station edition of the SGLang playbook you ran in Module 06.

How to read each entry below:

| Badge | Meaning |
|---|---|
| 🟩 **Spark** | the README's platform table lists DGX Spark: you can run it this week |
| 🟪 **Station only** | the table lists only DGX Station (GB300). Read it for the ideas; the commands target other hardware |
| 🟧 **RTX only** | a Windows or Linux PC with GeForce/RTX PRO cards. The Spark counterpart is named |

Commands for 🟩 playbooks are in `bash` blocks with `# on: spark`. Commands for 🟪 and 🟧 playbooks are in `text` blocks on purpose: they are for reference and the ⌨ terminal will not offer to run them on your Spark.

✓ Checkpoint: lab 21-1 ends with `26 in this atlas (13 of them run on a DGX Spark)`, and you can say why `nanochat` is in the atlas but not in your plan.

## 2 · Images and video

### ComfyUI 🟩 Spark · 45 min · ≥30 GB disk (quick start), 70–230 GB (video tiers)

ComfyUI is a node-based web app for diffusion models. Instead of one prompt box, you wire loaders, text encoders, samplers and decoders into a graph that saves as JSON. Use it when you want local image or video generation with full control of every step, and when you want to share a repeatable workflow. The playbook has two paths: a light host-Python **Image Gen Quick Start** with Z-Image-Turbo (~24 GB GPU memory), and a container **Video Gen Workflow** with FLUX, Wan 2.1, HunyuanVideo and Cosmos in three tiers (peak ~80 / ~100 / ~120 GB). On 128 GB of unified memory, the README says to start with Tier 1. Module 12 fine-tuned FLUX.1; ComfyUI is where you use it.

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

In a second terminal, `curl -I http://localhost:8188` should return HTTP 200. From the laptop, tunnel the port (`ssh -N -L 8188:localhost:8188 spark-a`) and open `http://localhost:8188` → **Templates → Image → Z-Image-Turbo: Text to Image → Run**. The README says the image should complete within 30 seconds. The video path starts with `docker build -t comfyui -f assets/Dockerfile .` and `bash assets/scripts/download-models.sh 1`; follow the playbook's **Video Gen Workflow** tab.

### Multi-modal inference with TensorRT 🟩 Spark · 60 min · ≥48 GB free memory for FP16 Schnell

This playbook runs FLUX.1 Dev and Schnell text-to-image through the TensorRT diffusion demo inside NVIDIA's PyTorch container, at BF16, FP8 and FP4. Use it to see what the precision levers from Modules 01 and 07 do to an image model, not an LLM. It needs a Hugging Face account with access to FLUX.1-dev and FLUX.1-dev-onnx.

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

> 🔐 The playbook passes the token as `--hf-token=$HF_TOKEN`, so set it with `export HF_TOKEN=<YOUR_HUGGING_FACE_TOKEN>` **inside the container on the Spark**, in that shell only. Never type a token into a lab or the Lab Runner.

Swap `--fp4` for `--bf16` or `--quantization-level 4 --fp8` to compare. The README warns that FP16 Schnell needs more than 48 GB of available memory.

### Visual gen AI (FLUX.2 + LTX-2) 🟧 RTX only · 13 min

A Windows ComfyUI desktop path: the starter text-to-image template, FLUX.2-Dev images and LTX-2 image-to-video, all from ComfyUI's template browser. The README deliberately ships no shell commands ("It does **not** invent host Python, Docker, or shell install commands") and points Linux users to the ComfyUI playbook. **On your Spark, use ComfyUI above** and load the same templates from its template browser.

### Controlled video with ComfyUI 🟧 RTX only · 18 min · 16 GB+ VRAM, 64 GB RAM

A storyboard-to-4K workflow on Windows 11: generate 3D assets, lay out a Blender scene, make first and last keyframes with FLUX.1 Depth, fill the motion with LTX-2.3, then upscale with the RTX Video Super Resolution node. The installs live in two NVIDIA AI Blueprints repos, not the playbook. It is worth reading for the idea: **lock the composition first, then generate motion between keyframes**. The one launch command it shows is for Windows:

```text
# on: Windows RTX PC (not a Spark) — reference only
cd C:\3d-object-generation
conda activate 3dwithtrellis311
python app.py
```

✓ Checkpoint: you can say which of the four image/video playbooks you would run on a Spark (ComfyUI, TensorRT multi-modal), and which Tier of the ComfyUI video workflow fits 128 GB.

## 3 · Vision and video agents

### Live VLM WebUI 🟩 Spark · 30 min · a webcam

A browser app that streams your webcam to any OpenAI-compatible vision-language model and shows the answer, the latency and the GPU load live. It is a testbed for comparing VLMs across Ollama, vLLM, SGLang or NIM. The playbook uses Ollama with `llama3.2-vision:11b`. If you did Module 03, Ollama is already on your Spark and you can skip its install line.

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

It serves HTTPS on port 8090 with a self-signed certificate. Open `https://<SPARK_IP>:8090` (find it with `hostname -I | awk '{print $1}'`), set **API Base URL** to `http://localhost:11434/v1`, and pick the model.

> ⚠ Do not reach this one through an `ssh -L` tunnel. The README's troubleshooting says WebRTC fails with `InvalidStateError` through an SSH TCP tunnel and needs a direct connection between browser and Spark.

### Video Search and Summarization (VSS) 🟩 Spark · 45 min · >10 GB in /tmp · NGC key + a remote LLM

NVIDIA's VSS AI Blueprint turns video into searchable events: a local VLM (Cosmos Reason 2) describes the footage, a remote LLM answers questions and verifies alerts. Use it for camera analytics, long-video Q&A and real-time alerts. On the Spark this is a **hybrid** deployment: the VLM runs locally, the LLM is a remote endpoint (for example a build.nvidia.com NIM). VSS 3.2.0 needs driver ≥ 580.95.05 and CUDA 13.0, which the Module 01 doctor checks.

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

The README also installs a **cache-cleaner script** that runs `sync && echo 3 > /proc/sys/vm/drop_caches` every 3 seconds as root (its Step 4). Read that step before you copy it: it changes system behaviour for as long as it runs, and you stop it with `sudo pkill -f sys-cache-cleaner.sh`. For the alert workflows the README strongly recommends `--llm nvidia/nvidia-nemotron-nano-9b-v2`. Tear down with `deploy/docker/scripts/dev-profile.sh down`.

✓ Checkpoint: you can say why Live VLM WebUI must not go through an SSH tunnel, and which half of VSS runs on the Spark.

## 4 · Robotics and physical AI

### Isaac Sim and Isaac Lab 🟩 Spark · 30 min (10–15 min build) · ≥50 GB disk

Isaac Sim is NVIDIA's Omniverse-based robot simulator; Isaac Lab adds reinforcement-learning environments on top. On the Spark you **build Isaac Sim from source** for `linux-aarch64`, then train the humanoid locomotion sample `Isaac-Velocity-Rough-H1-v0`. Use it to train and test robot policies in simulation before real hardware. It needs GCC 11 as the default compiler.

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

Then clone Isaac Lab, link it to the build and train headless (the playbook's Steps 6–9):

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

> ⚠ `update-alternatives` changes the system default compiler. Note your current `gcc --version` first if other projects on the Spark depend on it.

**Expected output** (REFERENCE — quoted from the playbook, after `./build.sh`)

```
BUILD (RELEASE) SUCCEEDED (Took 674.39 seconds)
```

### Isaac GR00T N1.6 fine-tuning 🟪 Station only · 45 min · ~30 GB disk

GR00T N1.6 is a 3-billion-parameter vision-language-action model for humanoid robot skills. The playbook fine-tunes its action head on the LIBERO Spatial benchmark at a global batch size of 128 on one GB300, then runs open-loop evaluation. The platform table lists only DGX Station (~284 GB HBM3e), and the expected GPU name is `NVIDIA GB300`. What transfers to your Spark is the idea from Modules 09–12: fine-tune a small adapter path on a pretrained model, then evaluate before you trust it.

```text
# on: DGX Station (not a Spark) — reference only
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
git checkout n1.6-release
I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 bash scripts/deployment/dgpu/install_deps.sh
source .venv/bin/activate
huggingface-cli download nvidia/GR00T-N1.6-3B
```

### Reachy photo booth 🟩 Spark · 2 h · a Reachy Mini robot, monitor and keyboard

A complete local multimodal stack on one Spark: a NeMo Agent Toolkit ReAct agent on `openai/gpt-oss-20b` (TensorRT-LLM), Parakeet speech-to-text, Kokoro text-to-speech, FLUX.1-Kontext image restyling, person tracking, and MinIO for sharing photos by QR code. It is the best worked example of Module 14's NAT agents driving real hardware. It needs a Reachy Mini Lite robot, an NGC personal API key, a Hugging Face token with the FLUX.1-Kontext licences accepted, and a display on the Spark itself.

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

Open `http://127.0.0.1:3001` in a browser **on the Spark**. The README says the first `docker compose up` takes 30 minutes to 2 hours; later starts about 5 minutes. Check containers with `docker compose ps` and stop with `docker compose down`.

✓ Checkpoint: you can name the two robotics playbooks that run on a Spark and the extra hardware each needs.

## 5 · Data science and HPC

### CUDA-X data science (cuDF, cuML) 🟩 Spark · 30 min · Kaggle API key

RAPIDS, now called CUDA-X Data Science, runs pandas and scikit-learn code on the GPU **with no code changes**: load `cudf.pandas` or `cuml.accel` and your existing code is accelerated. The playbook ships two notebooks: a large-strings pandas workflow, and LinearSVC, UMAP and HDBSCAN in scikit-learn style. Use it when the slow part of your project is dataframes or classical ML, not an LLM.

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

> ✎ Course deviation: the README's `cd` line says `client-hardware-playbooks/nvidia/…`, but the `git clone` above creates `dgx-spark-playbooks/`, so the course uses that path. The same fix applies to JAX, portfolio optimization and single-cell below.

Put your `kaggle.json` in the `assets` folder next to the notebooks, and tunnel Jupyter with `ssh -N -L 8888:localhost:8888 spark-a`.

### Portfolio optimization with cuOpt 🟩 Spark · 20 min · ≥30 GB disk, 40 GB free memory

A Mean-CVaR portfolio pipeline: cuML kernel density estimation generates return scenarios, cuOpt solves the resulting linear program, and the notebook backtests rebalancing strategies. Use it as a template for any scenario-based optimisation that is too slow on CPU. It all runs inside the RAPIDS 25.10 notebooks container.

```bash
# on: spark
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-portfolio-optimization/assets
bash ./setup/start_playbook.sh
```

Open JupyterLab (tunnel `-L 8888:localhost:8888`), open `cvar_basic.ipynb`, and **switch the kernel to "Portfolio Optimization" before the first cell**. The README says the wrong kernel fails by the second code cell.

### Single-cell RNA analysis 🟩 Spark · 15 min · ≥30 GB disk, 40 GB free memory

RAPIDS-singlecell follows the Scanpy API, so single-cell RNA-seq preprocessing, QC, clustering, UMAP, Harmony batch correction and differential expression run on the GPU. Use it if you work with genomics data. Same container and launcher pattern as portfolio optimization:

```bash
# on: spark
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-single-cell/assets
bash ./setup/start_playbook.sh
```

Then run `scRNA_analysis_preprocessing.ipynb`. The README notes that the GPU computes the exact neighbour graph, while Scanpy on CPU uses an approximate one, so small differences are expected.

### Topic modeling with BERTopic 🟪 Station only · 45 min · ≥50 GB disk, ~14 GB dataset

BERTopic on millions of Amazon reviews: SentenceTransformers embeddings, then cuML's drop-in GPU UMAP and HDBSCAN, then interactive topic maps and an optional Streamlit dashboard. The platform table lists only DGX Station, and the README asks for 64 GB of GPU memory for tens of millions of reviews. The acceleration trick (`%load_ext cuml.accel`) is the same one CUDA-X data science runs on your Spark.

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

### JAX 🟩 Spark · 2 h · port 8080

JAX is NumPy on the GPU plus function transforms: `jit` compiles, `grad` differentiates, `vmap` vectorises. The playbook builds a container with marimo notebooks and has you port a NumPy self-organising map to JAX step by step, measuring each version. Use it to learn how XLA compilation changes performance for array code.

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

Tunnel `-L 8080:localhost:8080` and open `http://localhost:8080`.

### cuPyNumeric across two Sparks 🟩 Spark ×2 · 60 min · Module 02 cable first

> The README lists DGX Spark, but `cupynumeric` was not on the build.nvidia.com/spark index fetched 2026-09-29, so its web link above may not resolve. Work from the README in the clone.

cuPyNumeric is a NumPy-compatible library that spreads arrays and linear algebra across GPUs and nodes through Legate and MPI. The playbook shares a conda environment over NFS, checks MPI across both Sparks, then runs a 20k × 20k matrix multiplication on one node and on two. It is the only playbook in this atlas that **needs both Sparks**, and it assumes Module 02's QSFP link (or NVIDIA Sync Cluster Assistant) is already working.

```bash
# on: spark
export SERVER_IP="192.168.100.11"    # Server node interconnect IP — use your Module 02 addresses
export CLIENT_IP="192.168.100.10"    # Client node interconnect IP
export IB_SUBNET="192.168.100.0/24"  # Interconnect subnet
conda create -n cupynumeric -c legate cupynumeric
mpirun -n 2 -npernode 1 --hostfile $HOME/shared/hostfile hostname
legate --launcher mpirun --launcher-extra="--hostfile $HOME/shared/hostfile --mca oob_tcp_if_include $IB_SUBNET --mca pml ucx" --nodes 2 --gpus 1 --fbmem 40960 --cpus 4 ./examples/gemm.py -n 20000 -i 2
```

Those are the landmarks. In between, the README's Steps 3–5 open the firewall on the QSFP interfaces (`sudo ufw allow in on …`), install an NFS server and client, append to `/etc/exports` and optionally `/etc/fstab`, and install Miniforge into the share. Those are real system changes on both Sparks: follow the README for them, and use its Step 10 to undo them.

✓ Checkpoint: you can name the five data-science playbooks that run on a Spark, and say which one needs a second Spark.

## 6 · Kernels and training from scratch

### cuTile kernels (TileGym) 🟩 Spark · 60 min · ≥50 GB disk

cuTile is a Python DSL for GPU kernels that compiles to Tile IR, so you write tiles, not threads. TileGym is NVIDIA's benchmark suite for it. The playbook has three tracks: benchmark FMHA, MatMul, RMSNorm, RoPE and SwiGLU kernels; run Qwen2-7B or DeepSeek-V2-Lite with cuTile kernels monkey-patched in; and build a flash-attention kernel from pseudocode. Use it when a model is slow and you want to see where a custom kernel could help.

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

**Expected output** (REFERENCE — quoted from the playbook; it does not say which platform produced these numbers, so do not treat them as Spark figures)

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

### Triton fine-tuning kernels 🟪 Station only · ~2 h · ≥150 GB disk

Profile a Llama 3.1 8B fine-tuning step with `torch.profiler`, then write two Triton kernels: a fused RMSNorm and a fused cross-entropy using online softmax, which avoids storing the full logits tensor. Then patch both into the training loop and measure. It is the training-side sibling of cuTile, written for a GB300 (`--gpus '"device=N"'` pins the GB300 on a multi-GPU Station). On your Spark, the cuTile playbook is the kernel path the platform table supports.

```text
# on: DGX Station (not a Spark) — reference only
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-kernel-dev-ft/assets      # README says client-hardware-playbooks/…
docker build -t kernel-dev-ft .
python profile_baseline.py            # inside the kernel-dev-ft container
python rmsnorm_test.py
python finetune_optimized.py
```

### NanoChat 🟪 Station only · setup ~30 min, full d24 run 12+ h

Andrej Karpathy's nanochat trains a small ChatGPT-style model end to end: a BPE tokenizer, pretraining, chat fine-tuning, a report, then a web or CLI chat. The README's own change log says: **"Removed DGX Spark support (not ready yet); playbook is Station / single-node only."** So do not plan it for your Spark this week. It needs a Weights & Biases key and a Hugging Face token.

```text
# on: DGX Station (not a Spark) — reference only
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-nanochat/assets
chmod +x setup.sh launch.sh
./setup.sh
./launch.sh
```

### NVFP4 pretraining with Megatron Bridge 🟪 Station only · 30 min

Module 07 used NVFP4 to shrink a finished model. This playbook uses it **during pretraining**: Megatron Bridge's `bf16_with_nvfp4_mixed` recipe on Llama 3.1 8B with mock data, keeping the last four layers in BF16 for stability, compared with a plain BF16 run via `--disable-fp4`.

**Expected output** (REFERENCE — the playbook's "Measured results" table, single GB300, `nvcr.io/nvidia/nemo:26.04`, global batch 64, sequence 4096)

```text
| Precision | Recipe | Avg step time | Throughput (Model TFLOP/s/GPU) | Peak VRAM |
|---|---|---|---|---|
| BF16 baseline | `bf16_mixed()` | 9.05 s | ~1399 | 221.6 GB |
| NVFP4 (last-4 BF16) | `bf16_with_nvfp4_mixed()` + `first_last_layers_bf16=True`, `num_layers_at_end_in_bf16=4` | **5.39 s** | **~2347** | **207.8 GB** |
```

Both peak-memory figures are above one Spark's 128 GB, which is one reason the platform table lists only DGX Station. The launch is `docker run … nvcr.io/nvidia/nemo:${TAG}` and then, inside it:

```text
# on: DGX Station (not a Spark) — reference only
torchrun --nproc_per_node=1 pretrain_llama.py > nvfp4.log 2>&1
```

✓ Checkpoint: you can say which kernels playbook runs on a Spark (cuTile), and quote the README line that rules out NanoChat for now.

## 7 · Platform and multi-GPU

### Brev 🟩 Spark · 10 min · a Brev account

NVIDIA Brev registers your Spark as a managed GPU node. After registration, teammates you add in the Brev UI can reach it over SSH from anywhere, and you can standardise environments with "Launchables". Use it when more than one person needs your Spark and you do not want to hand out tailnet access (Module 01). Registration is driven from the Brev web UI: **Registered Compute → Register Compute** shows the CLI install and a registration command to run on the Spark with sudo. The playbook prints no fixed command for that step, so copy it from the pop-up. The commands it does name:

```bash
# on: spark
brev set <my-org>        # if the node registered into the wrong org, then redo registration
brev refresh             # if `brev shell <name>` fails with stale CLI state
brev deregister          # cleanup: remove the Spark from Brev
```

### Multi-Instance GPU (MIG) 🟪 Station only · 15 min

MIG partitions one data-centre GPU into isolated instances with their own memory and compute, so several users or jobs share a GPU without interfering. The playbook is written for B300 GPUs in a DGX Station, with profiles such as `1g.35gb` (ID 19), and warns that enabling or disabling MIG affects every workload on that GPU.

```text
# on: DGX Station (not a Spark) — reference only
sudo nvidia-smi -mig 1
nvidia-smi mig -lgip -i 0
sudo nvidia-smi mig -cgi 19,19,19,19,19,19,19 -C -i 0
nvidia-smi -L
sudo nvidia-smi -mig 0
```

### Build a multi-GPU AI PC 🟧 RTX only · 13 min

Two identical GeForce/RTX PRO cards in one PC: llama.cpp tensor parallel (`-sm tensor`) across both, LM Studio's tensor-parallel split, and ComfyUI's MultiGPU CFG Split node, plus a parts guide (PCIe lanes, PSU, cooling). Your Spark's equivalent is Module 02: two Sparks joined by a 200 Gb/s cable, with vLLM tensor parallel in Module 05.

```text
# on: dual-GPU RTX PC (not a Spark) — reference only
nvidia-smi -L
<app> -m <model_path> -sm tensor -fa 1
```

### Connect two stations 🟪 Station only · 60 min

The DGX Station version of Module 02: two ConnectX-8 QSFP rails at 400 Gb/s each, RoCEv2, MTU 9000 and GPUDirect checks, driven from a separate control host by numbered scripts. The structure mirrors what you did with two Sparks: probe access, cable, configure rails, validate, then run a bandwidth test.

```text
# on: control host for two DGX Stations — reference only
git clone https://github.com/NVIDIA/dgx-spark-playbooks client-hardware-playbooks
cd client-hardware-playbooks/nvidia/playbook-connect-two-stations/assets
cp 00_env.local.example 00_env.local
./01_probe_access.sh
./02_push_assets.sh
./07_validate_setup.sh
```

✓ Checkpoint: you know which platform playbook you would use to share your Spark with a teammate (Brev), and which Spark module replaces the multi-GPU PC and two-station playbooks (Module 02).

## 8 · Station-only serving and agents

These three are for DGX Station, but each one maps onto something you built this week.

### SGLang inference (Station edition) 🟪 Station only · 20–30 min

Same engine as Module 06, tuned for GB300 (SM103): `lmsysorg/sglang:latest-cu130`, `--attention-backend flashinfer` (the README says the default backend fails CUDA-graph capture on SM103), prefix caching checked through `#cached-token` in the logs, JSON-schema output and a multi-turn benchmark script. On your Spark, use the `sglang` playbook from Module 06. The part worth copying is **Step 7's method**: verify prefix caching from the server log, not from wall-clock latency.

```text
# on: DGX Station (not a Spark) — reference only
docker pull lmsysorg/sglang:latest-cu130
docker logs sglang-server 2>&1 | grep "cached-token" | tail -10
```

### Agent skills for DGX Station 🟪 Station only · 15 min

Four Agent Skills plus a `dgx-assist` CLI that make a coding agent (Claude Code, Codex, Gemini CLI, Cursor) inspect the real hardware, search pinned NVIDIA guidance, resolve a model to a qualified recipe, preflight, **ask you to approve**, then act and verify. It is Module 15's governance idea applied to a coding agent. The README's clone line uses a `…/blob/main/…` web URL, which `git clone` cannot fetch; the lines below are the course's fix.

```text
# on: DGX Station (not a Spark) — reference only
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-dgx-station-ai-skills      # course fix: README clones a /blob/ URL
assets/install.sh install \
  --harness codex \
  --target /path/to/project \
  --dry-run
```

### Healthcare agents 🟪 Station only · 60 min · ≥150 GB free GPU memory, ≥200 GB disk

Six OpenClaw agents inside an OpenShell sandbox query synthetic FHIR patient records, find care gaps and predict protein structures with OpenFold3, with Nemotron 3 Super served locally by Ollama. The sandbox allows only a short list of endpoints. It combines Modules 15 (OpenShell) and 17 (OpenClaw). The README's own figures (Nemotron 3 Super ~94 GB resident plus OpenFold3 ~40–80 GB) add up to more than one Spark's 128 GB. It is a demonstration on synthetic data, not a medical device, as its disclaimer says.

```text
# on: DGX Station (not a Spark) — reference only
docker compose up -d ollama openfold3
docker compose up model-pull
make status
bash scripts/ensure_openshell_gateway.sh
make setup
make check
```

✓ Checkpoint: for each Station-only playbook in this section you can name the Week 25 module that gives you the same skill on a Spark (06, 15, 15 + 17).

## 9 · Pick your next playbook: recommend, then budget

You now know all 26. Two small labs help you choose.

**Lab 21-2** takes a goal in plain words and ranks all 64 playbooks by TF-IDF cosine similarity over each README's overview. No model and no network are involved, and it prints the words that matched so you can see *why* it chose:

```bash
# on: laptop
.venv/bin/python week25/21_playbook_atlas/labs/lab21_2_goal_recommender.py
```

**Expected output** (captured on this Mac; 2 of the 10 demo goals shown)

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

Pass your own goal as an argument. `--spark` keeps only playbooks whose table lists DGX Spark, and the difference matters:

```bash
# on: laptop
.venv/bin/python week25/21_playbook_atlas/labs/lab21_2_goal_recommender.py "fine-tune a robot policy"
.venv/bin/python week25/21_playbook_atlas/labs/lab21_2_goal_recommender.py "fine-tune a robot policy" --spark
```

**Expected output** (captured on this Mac; the top two rows of each run, with a separator line added between the two runs)

```
│ 0.207  gr00t           ✕       Module 21  robot, tune, fine, policy
│ 0.157  nemo-fine-tune  ✓       Module 11  fine, tune
─── with --spark ───
│ 0.157  nemo-fine-tune  ✓       Module 11  fine, tune
│ 0.141  llama-factory   ✓       Module 09  fine, tune
```

The best match, GR00T, is Station-only. With `--spark` you get general fine-tuning tools instead: an honest answer that no Spark playbook fine-tunes robot policies today.

**Lab 21-3** adds up time, disk and "what to bring" for any set of playbooks, using only the READMEs' own figures. By default it plans every Spark playbook in this atlas:

```bash
# on: laptop
.venv/bin/python week25/21_playbook_atlas/labs/lab21_3_budget_planner.py
```

**Expected output** (captured on this Mac)

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

Read "200 GB" as a lower bound: seven READMEs give no disk figure, and ComfyUI's 30 GB is the quick start only (its video tiers need 70–230 GB). Plan your own set with `lab21_3_budget_planner.py comfyui vss isaac --free-gb 300 --session-min 120`.

✓ Checkpoint: you ran both labs with a goal of your own, and you have a short list of playbooks with a time and disk budget.

## Labs — run them here

**labs/lab21_1_playbook_atlas.py** — One table of all 64 playbooks: time, platforms, covering module, themes and slug evidence.

**labs/lab21_2_goal_recommender.py** — Which playbook for my goal? TF-IDF over the READMEs, with citations and two known misses.

**labs/lab21_3_budget_planner.py** — Time, disk and accounts or hardware for a set of playbooks, packed into sessions.

All three run on the laptop, offline, in a few seconds. They never touch a Spark: they only read the playbook clone and `week25/AUTHORING.md`.

## Try it yourself

**Exercise 21 — the goal router and its test set.** Open `week25/21_playbook_atlas/exercises/ex21_goal_router.py`. It has three `TODO`s:

1. `tokenize(text)`: lower-case, split on non-alphanumerics, drop 1-letter words and stopwords.
2. `idf(docs)`: the smoothed inverse document frequency `log((1+N)/(1+df)) + 1`.
3. `TEST_SET`: at least 8 `(goal, expected playbook)` pairs covering 5+ playbooks, 3+ from this atlas. The rule is **describe the goal, don't name the tool**: a goal may not contain any 4+ letter word of its playbook's slug.

The checker indexes the real READMEs with your functions, runs three built-in goals, then requires every one of your expected playbooks to land in the top three.

```bash
# on: laptop
.venv/bin/python week25/21_playbook_atlas/exercises/ex21_goal_router.py
```

**Expected output** (the reference solution, captured on this Mac)

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

The unedited starter stops at the first two lines with `✕ TODO 1` and `✕ TODO 2` and exits 1.

<details><summary>Hint — why does IDF add 1 inside and outside the log?</summary>

The `1 +` inside keeps the ratio defined when a word appears in every document (`df = N`) and gives words the index has never seen a finite weight. The `+ 1` outside keeps such common words at weight 1.0 instead of 0, so they still count a little. It is the smoothing scikit-learn uses by default.

</details>

<details><summary>Hint — a goal keeps missing</summary>

Print the goal's tokens with your `tokenize()` and search the README for them (`grep -i -c webcam dgx-spark-playbooks/nvidia/playbook-live-vlm-webui/README.md`). If the README never uses your word, the router cannot find it. Either reword the goal with the README's vocabulary, or keep the miss as a known limitation and choose another goal. Both are honest outcomes; hiding the miss is not.

</details>

<details><summary>Stretch — beat keyword search</summary>

Add a tiny synonym table (`split → partition`, `users → instances`) applied to goals before `tokenize()`, and make lab 21-2's step 3 goal for MIG land at #1. Then check that your whole test set still passes: a fix for one goal can break another, which is why the test set exists.

</details>

✓ Checkpoint: the checker prints `all … expected playbooks in the top three`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `✕ …/dgx-spark-playbooks/nvidia not found` from any lab | Clone the playbooks at the repo root: `git clone https://github.com/NVIDIA/dgx-spark-playbooks` |
| Lab 21-1 prints `✕ no theme yet: <slug>` | NVIDIA added a playbook after this module was written. Read its README and add it to `THEMES` in `lab21_1_playbook_atlas.py` |
| A playbook's `cd client-hardware-playbooks/…` fails right after `git clone …/dgx-spark-playbooks` | The clone is named `dgx-spark-playbooks/`. Use that path (Sections 5 and 6 show it) |
| The Agent Skills README's `git clone https://github.com/…/blob/main/…` fails | A `/blob/` URL is a web page, not a repository. Clone the repository root, then `cd dgx-spark-playbooks/nvidia/playbook-dgx-station-ai-skills` |
| A build.nvidia.com link from this page returns 404 | 12 Spark slugs are confirmed on the /spark index (lab 21-1 step 4). cuPyNumeric is not on it, and the Station and RTX links are unverified. Search the playbook title on build.nvidia.com; the README in the clone is authoritative |
| Live VLM WebUI: camera works, analysis fails with `InvalidStateError` | WebRTC does not pass through an `ssh -L` tunnel. Open `https://<SPARK_IP>:8090` directly from a machine on the same network |
| Memory errors in ComfyUI, VSS, or the RAPIDS notebooks even though `free -g` shows room | The playbooks' unified-memory note: flush the page cache with `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| Recommender ranks a Station-only playbook first | That is the honest best match. Add `--spark` to see only playbooks whose table lists DGX Spark |

## Next

This was the last module. Go [Back to the capstone](../20_capstone_sovereign_agent/TUTORIAL.md) to wire fine-tune → serve → gateway → NAT agent in a sandbox end to end, then add one playbook from your budget plan to it: ComfyUI as an image tool for the agent, or VSS as a video tool. For the whole week at a glance, return to the [Week 25 overview](../README.md).
