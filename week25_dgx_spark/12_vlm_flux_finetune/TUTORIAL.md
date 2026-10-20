# ▶ Spark Lab 12 — Fine-tune a vision-language model and FLUX.1

> Part of Week 25 · DGX Spark: fine-tune, serve, and build sandboxed agents. You type the commands, you see the real output. Every lab also runs in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's playbook, or a clearly marked EXAMPLE.

**What you'll actually do**
- Get image and video datasets into exactly the layout the playbook's trainers expect, and validate them before a GPU sees them.
- Fine-tune **Qwen2.5-VL-7B** with **GRPO** (reward-based training) on the Spark. The playbook starts it from a Streamlit button; you run the same command headless, under `nohup`.
- Prepare a **FLUX.1-dev** Dreambooth LoRA dataset (concepts, trigger words, captions, `data.toml`), including a hotel concept of your own.
- Train the FLUX LoRA on the Spark with the playbook's `launch_train.sh`, watch it from your laptop, and then generate with it in ComfyUI.

**Time** ~60 min of work (plus hours of unattended training) · **Difficulty** advanced · **Hardware** 1 DGX Spark (or none: DRY mode + the dataset labs run on the laptop)

**Official playbooks covered:** [Fine-Tune Vision Language Models](https://build.nvidia.com/spark/vlm-finetuning) · [Fine-Tune FLUX.1 for Custom Image Generation](https://build.nvidia.com/spark/flux-finetuning)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| The playbooks cloned at the repo root | `ls dgx-spark-playbooks/nvidia/playbook-flux-finetuning` | labs 01 and 03 read the playbooks' own sample files |
| PIL in this repo's Python | `.venv/bin/python -c "import PIL"` | the dataset labs draw and check images |
| `ffprobe` (optional) | `ffprobe -version` | lab 01 measures the sample videos |
| A Hugging Face token with FLUX.1-dev access | accept the terms on the [model card](https://huggingface.co/black-forest-labs/FLUX.1-dev) | both playbooks download gated or token-backed models |
| A Kaggle account | for the wildfire dataset | the image recipe's training data |
| ~100 GB free on the Spark | `df -h ~` | two containers, FLUX (~34 GB), Qwen2.5-VL (twice, see Section 4), datasets |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
.venv/bin/python -c "import PIL; print('Pillow', PIL.__version__)"
```

**Expected output** (captured on this Mac)

```
Pillow 12.1.1
```

> 🔐 Both playbooks pass your Hugging Face token on the command line: `docker build --build-arg HF_TOKEN=$HF_TOKEN` (VLM) and `download.sh`, which reads `$HF_TOKEN` (FLUX). The labs never run those two commands. They print them for you to run **on the Spark**, in your own shell.

✓ Checkpoint: Pillow imports, and `dgx-spark-playbooks/` exists at the repo root.

## 1 · Two kinds of image fine-tuning

The two playbooks train models that work in opposite directions:

| | Vision-language model (VLM) | FLUX.1 (diffusion) |
|---|---|---|
| Input → output | image or video → **text** | text → **image** |
| Model | Qwen2.5-VL-7B (image), InternVL3-8B (video) | FLUX.1-dev, 12B, plus CLIP-L, T5-XXL and an autoencoder |
| Method in the playbook | LoRA + **GRPO** (image), LoRA SFT (video) | multi-concept **Dreambooth LoRA** (`network_dim` 256) |
| Data | labelled images / clips + `metadata.jsonl` | 5–10 images per concept, one trigger word each |
| Trainer | Unsloth + TRL in the `vlm_demo` container | kohya-ss `sd-scripts` in the `flux-train` container |
| Try it in | Streamlit, side by side with the base model (:8501) | ComfyUI workflows (:8188) |
| Time (playbook) | ~100 GRPO steps: up to ~2 h; usable video: a day or more | 100 epochs: ~4 h; usable after ~90 min |

Unified memory is why one Spark can do either. The FLUX playbook keeps "the Diffusion Transformer, CLIP text encoder, T5 text encoder, and autoencoder resident while you train". The UMA flush that both playbooks recommend (`sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`) is for moments when those big files leave the page cache full. Run **one** of the two recipes at a time.

✓ Checkpoint: you can say which of the two models you would fine-tune to *check* hotel rooms from photos, and which to *generate* images of your hotel's lobby.

## 2 · The image dataset: folder names are labels

The image recipe's trainer, `ui_image/src/train_image_vlm.py`, loads its data with a single line, `load_dataset(config["data"]["dataset_id"])["train"]`, where `dataset_id` is `data`. Hugging Face's image-folder loader turns **folder names into labels**. Then `format_instruction()` maps the label to an answer: `nowildfire` → "No", anything else → "Yes". So the layout *is* the labelling:

```text
ui_image/data/
├── train/
│   ├── nowildfire/   *.jpg
│   └── wildfire/     *.jpg
├── valid/ …
└── test/ …
```

Lab 01 inspects the playbook's sample images and builds a hotel version: *is this room ready for the next guest?* It uses the classes `ready` and `needs_attention`. The images are small **synthetic** PIL drawings. They teach a model nothing; they let you practise the layout and the checks.

```bash
# on: laptop
.venv/bin/python week25/12_vlm_flux_finetune/labs/lab01_vlm_datasets.py
```

**Expected output** (steps 1–3, captured on this Mac)

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

That last diff is the lesson: with the hotel folders and no code change, `ready` would silently become "Yes" to the wildfire question. The labels would be wrong, and nothing would crash.

✓ Checkpoint: both folders validate, and you can say which two lines of `train_image_vlm.py` must change for a new pair of classes.

## 3 · The video dataset: one JSON record per clip

The video recipe fine-tunes InternVL3-8B to turn a dashcam clip into structured JSON. The playbook gives the layout:

```text
dataset/
├── videos/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── metadata.jsonl
```

The playbook text does not spell out the fields of `metadata.jsonl`. They are in the training notebook, `ui_video/train/video_vlm.ipynb`: `video`, `caption`, `event_type`, `rule_violations`, `intended_action`, `traffic_density`, `scene`, `visibility`, and the allowed values are in its prompt. Lab 01 (steps 4–5) measures the playbook's sample clips with `ffprobe`, then validates a metadata file against those enums:

**Expected output** (steps 4–5, captured on this Mac; the metadata is an EXAMPLE — synthetic, written to test the validator)

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

One detail comes only from reading the notebook's code. `get_video_path()` joins the **parent** of `dataset_path` with each record's `video` field. So set `dataset_path = "/path/to/dataset/metadata.jsonl"` (the file itself) and write `"video": "videos/video1.mp4"`. The playbook says usable video quality "often needs a long training window (on the order of a day or more)". Its knobs are 12–16 frames per video, uniform temporal sampling, and LoRA. Start with the image recipe.

✓ Checkpoint: you can name three fields of `metadata.jsonl` and their allowed values, and you know what `dataset_path` must point at.

## 4 · Run the VLM recipe on the Spark, headless

The playbook's path is: clone the repo, build the `vlm_demo` image, start it with `launch.sh`, download the model and the Kaggle dataset, run `streamlit run Image_VLM.py`, and press **Start Finetuning**. That button writes `ui_image/src/train.yaml` and runs `python -u src/train_image_vlm.py`. Lab 02 does the same two things over ssh, under `nohup`:

```bash
# on: laptop
.venv/bin/python week25/12_vlm_flux_finetune/labs/lab02_vlm_train_on_spark.py
SPARK_APPLY=1 .venv/bin/python week25/12_vlm_flux_finetune/labs/lab02_vlm_train_on_spark.py --steps 5
```

What you run yourself, once, on the Spark (these are the playbook's Steps 2–3 and 5.1–5.2, with the clone path corrected):

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

Three things the playbook text does not tell you, and lab 02 does:

1. **The clone path.** The playbook says `cd client-hardware-playbooks/nvidia/playbook-vlm-finetuning/assets`. The folder that `git clone` creates is `dgx-spark-playbooks`.
2. **Your token is inside the image.** The Dockerfile runs `hf auth login --token $HF_TOKEN` during the build. Never push `vlm_demo` to a registry, and `docker rmi vlm_demo` when you are done.
3. **Two model downloads.** The playbook runs `hf download Qwen/Qwen2.5-VL-7B-Instruct`, but `src/train.yaml` trains `unsloth/Qwen2.5-VL-7B-Instruct`, a separate repo that Unsloth downloads on the first run.

**Expected output** (step 4, captured on this Mac — read from the playbook's own `src/train.yaml`)

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

**GRPO** differs from Module 09's supervised training. There is no answer text to imitate. For each image the model writes `num_generations` answers. `format_reward_func` gives +1 for exactly one `<REASONING>…</REASONING>` block and +1 for exactly one `<SOLUTION>…</SOLUTION>` block. `correctness_reward_func` gives 5.0 when the solution matches the folder label. The update favours the better-scoring answers. The repo ships `steps: 5`, a smoke test; the playbook's UI default is 100 steps, which "can take up to about 2 hours".

The headless command (course deviation: no `-it`, no Docker socket mount, `-w` set to `ui_image`):

```bash
# on: spark
cd ~/dgx-spark-playbooks/nvidia/playbook-vlm-finetuning/assets
nohup docker run --gpus=all --net=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm --name w25-vlm-grpo -v $(pwd):/vlm_finetuning -v $HOME/.cache/huggingface:/root/.cache/huggingface -w /vlm_finetuning/ui_image vlm_demo python -u src/train_image_vlm.py > ~/w25/logs/m12_vlm_grpo.log 2>&1 < /dev/null &
```

When training ends, the script "merges LoRA weights into the base model" (playbook) into `ui_image/saved_model/checkpoint-N`. Then compare base and fine-tuned side by side in the playbook's Streamlit app: start the container with `sh launch.sh`, `cd /vlm_finetuning/ui_image`, `streamlit run Image_VLM.py`, and open `:8501` through an SSH tunnel. The first load starts two vLLM servers and can take about 15 minutes.

✓ Checkpoint: in LIVE mode, lab 02 shows `loss` and `reward` lines and then `saved_model/checkpoint-N`. In DRY mode, its parser self-test is ✓ and you can explain the two reward functions.

## 5 · A FLUX.1 Dreambooth dataset: triggers, captions, data.toml

Dreambooth LoRA teaches FLUX a **new word**. Each concept gets a rare trigger word plus a class word (`tjtoy toy`, `sparkgpu gpu`). After training, the trigger in a prompt calls up the concept. The playbook suggests "about 5–10 images per concept", one folder per concept under `flux_data/`, and a `[[datasets.subsets]]` entry in `flux_data/data.toml` for each.

Lab 03 reads the playbook's real `data.toml` and images, works out the training arithmetic, and adds a third concept, `sparkhotel lobby`. Its six **synthetic** 1024×1024 stand-ins each come with a `.caption` file. sd-scripts reads `<image>.caption` when it exists and falls back to `class_tokens` otherwise.

```bash
# on: laptop
.venv/bin/python week25/12_vlm_flux_finetune/labs/lab03_flux_dataset.py
```

**Expected output** (captured on this Mac)

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

Four choices worth understanding:

- **`num_repeats`.** sparkgpu is shown twice per epoch, so it gets as many looks as tjtoy despite being a harder concept. More repeats means more steps.
- **`flip_aug`.** Mirroring doubles the variety of a toy. For anything with text or a logo, set it to `false`; the course's `sparkhotel` subset does.
- **`keep_tokens = 2`** keeps the first two caption words (trigger + class) in place if you turn on `shuffle_caption`.
- **The file names.** They are sd-scripts' own pattern `{output_name}-{epoch:06d}`, read from sd-scripts' code, not from the playbook, which says only that the files have the `flux_dreambooth` prefix.

✓ Checkpoint: both `data.toml` files validate, and you can explain why the hotel concept should not use `flip_aug`.

## 6 · Train the FLUX LoRA on the Spark, then generate

First the playbook's Steps 3 and 6, which you run on the Spark:

```bash
# on: spark
cd ~/dgx-spark-playbooks/nvidia/playbook-flux-finetuning/assets
export HF_TOKEN=<YOUR_HF_TOKEN>
sh download.sh                                   # ~23.8 GB + ~9.8 GB + ~335 MB + ~246 MB
docker build -f Dockerfile.train -t flux-train .
```

**Expected output** (REFERENCE — quoted from the playbook)

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

`launch_train.sh` starts the container with `docker run -it`, which fails under `nohup` because there is no terminal. Lab 04 writes a copy, `launch_train_w25.sh`, with `-it` removed and, if you ask, a lower `--max_train_epochs` (the playbook's own suggestion for a shorter run is `--max_train_epochs=25`). Then it runs that copy under `nohup`. With `--hotel` it first uploads lab 03's `sparkhotel/` folder and `data.toml`, keeping the original as `data.toml.orig`.

```bash
# on: laptop
.venv/bin/python week25/12_vlm_flux_finetune/labs/lab04_flux_train_on_spark.py
SPARK_APPLY=1 .venv/bin/python week25/12_vlm_flux_finetune/labs/lab04_flux_train_on_spark.py --epochs 25 --hotel
```

**Expected output** (captured on this Mac, DRY mode)

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

Re-run lab 04 to follow the run. It prints `step s/t`, sd-scripts' `avr_loss` (a moving average; diffusion loss is noisy, so judge the LoRA by its images), and the files in `models/loras/`.

Then generate (the playbook's Step 7). Stop training first, because ComfyUI needs the memory:

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

In ComfyUI, press `w`, load `finetuned_flux.json`, and prompt with your triggers, for example `tjtoy toy holding sparkgpu gpu in a datacenter`. The playbook allows about three minutes per 1024 px image. Load `base_flux.json` with the same prompt to see what the LoRA added.

✓ Checkpoint: in LIVE mode, `models/loras/` holds at least `flux_dreambooth-000025.safetensors`, and a trigger prompt produces your concept. In DRY mode, you can explain why `launch_train.sh` needs the `-it` removed to run under `nohup`.

## Labs — run them here

**labs/lab01_vlm_datasets.py** — Image-folder and video-metadata datasets in the layout the playbook's trainers expect, built and validated on the laptop.

**labs/lab02_vlm_train_on_spark.py** — Qwen2.5-VL GRPO training on the Spark: assets, image, model and dataset checks, then the Start Finetuning command run headless.

**labs/lab03_flux_dataset.py** — The playbook's FLUX dataset and step arithmetic, plus a hotel concept with captions and a validated data.toml.

**labs/lab04_flux_train_on_spark.py** — FLUX.1 Dreambooth LoRA training on the Spark: models, flux-train image, headless launch_train.sh, progress and LoRA files.

Labs 01 and 03 run fully on the laptop. Labs 02 and 04 drive the Spark over SSH, or show the plan in DRY mode (their parsers are tested on EXAMPLE lines).

## Try it yourself

**Exercise 12 — fix a broken `data.toml`.** Open `week25/12_vlm_flux_finetune/exercises/ex12_fix_data_toml.py`. A teammate added the hotel concept and made four mistakes:

1. `resolution` is text, not a number.
2. Two concepts share a trigger word.
3. An `image_dir` points at a folder that does not exist.
4. A `class_tokens` value has only the class word, not a trigger.

The checker is offline. It parses the text with `tomllib` and checks every folder against lab 03's `flux_data/` and the playbook's.

```bash
# on: laptop
.venv/bin/python week25/12_vlm_flux_finetune/exercises/ex12_fix_data_toml.py
```

**Expected output** (once all four TODOs are fixed; captured on this Mac)

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

<details><summary>Hint — what makes a good trigger word?</summary>

A word the base model does not already know (`tjtoy`, `sparkgpu`). With a common word such as `lobby` as the trigger, the new concept fights everything FLUX already knows about lobbies.

</details>

<details><summary>Stretch — a real hotel concept</summary>

Replace the six synthetic images in `flux_data/sparkhotel/` with 5–10 real photos of one lobby, from different angles and in different light. Update the `.caption` files, re-run lab 03, then run lab 04 with `--epochs 25 --hotel`. Compare `base_flux.json` and `finetuned_flux.json` on the prompt `sparkhotel lobby at night`.

</details>

✓ Checkpoint: all four checker lines are ✓.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cd: client-hardware-playbooks/…: No such file or directory` | the playbook's typo: the clone is `dgx-spark-playbooks/nvidia/…` |
| `permission denied` running Docker | the playbooks' fix: `sudo usermod -aG docker $USER && newgrp docker` |
| Hugging Face download or auth error | export a valid `HF_TOKEN` and pass `--build-arg HF_TOKEN=$HF_TOKEN` (VLM); for FLUX, accept the terms on the FLUX.1-dev model card first |
| Kaggle cURL fails | sign in to Kaggle, accept the dataset terms, and copy a fresh cURL command |
| `the input device is not a TTY` | a `docker run -it` under nohup. Use lab 02's command or lab 04's `launch_train_w25.sh` |
| Streamlit spinner for many minutes | the playbook: the first load starts vLLM (image, up to ~15 min) or loads the model (video, ~10 min) |
| OOM or memory pressure during train or inference | stop Streamlit, Jupyter or ComfyUI before training, then flush: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| `ready` images answered as wildfire "Yes" | you changed the folders but not `format_instruction()` (Section 2) |
| Files under `saved_model/` or `models/loras/` owned by root | containers write as root. Use `sudo chown -R $USER` on that folder, or copy the files out with `docker cp` |

## Next

Continue to [Lab 13 — close the loop: evaluate, merge, serve](../13_finetune_to_serve/TUTORIAL.md): score your fine-tunes on held-out data, merge them, and serve them behind the course gateway.
