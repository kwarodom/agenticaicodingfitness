#!/usr/bin/env python3
"""Lab 12-4 · Train a FLUX.1-dev Dreambooth LoRA on the Spark — the playbook's recipe, run under nohup.

  1. the playbook assets on the Spark, and the four model files download.sh fetches (you run
     download.sh yourself, because it needs your HF token — printed, never run here)
  2. the flux-train image: docker build -f Dockerfile.train (opt-in, under nohup)
  3. optional --hotel: upload lab 03's sparkhotel concept and data.toml (the original is kept as data.toml.orig)
  4. the playbook's launch_train.sh, made headless: `docker run -it` → `docker run` (no terminal under
     nohup), and --epochs N to lower --max_train_epochs=100 as the playbook suggests (e.g. 25)
  5. a monitor: sd-scripts' progress bar (step, total, avr_loss) and the LoRA files in models/loras/

Opt in with  --yes  or  SPARK_APPLY=1.

Run: .venv/bin/python week25/12_vlm_flux_finetune/labs/lab04_flux_train_on_spark.py
     SPARK_APPLY=1 .venv/bin/python week25/12_vlm_flux_finetune/labs/lab04_flux_train_on_spark.py --epochs 25 --hotel
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check, note, put, result, sh, step, table, warn, where  # noqa: E402

MOD = Path(__file__).resolve().parents[1]
APPLY = "--yes" in sys.argv or os.environ.get("SPARK_APPLY") == "1"
EPOCHS = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 100
HOTEL = "--hotel" in sys.argv
ASSETS = "~/dgx-spark-playbooks/nvidia/playbook-flux-finetuning/assets"
LOG = "~/w25/logs/m12_flux_train.log"
MODELS = ["checkpoints/flux1-dev.safetensors", "vae/ae.safetensors", "text_encoders/clip_l.safetensors",
          "text_encoders/t5xxl_fp16.safetensors"]


def progress(text: str) -> list[tuple[int, int, float]]:
    """sd-scripts' tqdm bar 'steps:  12%|█▏ | 60/500 [..., avr_loss=0.312]' → [(60, 500, 0.312), …]."""
    return [(int(a), int(b), float(c)) for a, b, c in re.findall(r"steps:.*?(\d+)/(\d+) \[[^\]]*avr_loss=([\d.]+)", text)]


banner("Lab 12-4 · FLUX.1-dev Dreambooth LoRA on the Spark",
       f"epochs={EPOCHS} · dataset={'playbook + sparkhotel' if HOTEL else 'playbook sample concepts'} · "
       f"run={'yes (opted in)' if APPLY else 'no (add --yes or SPARK_APPLY=1)'}")

# ── STEP 1 ────────────────────────────────────────────────────────────────────
step(1, "assets and the four model files (playbook Steps 2–3)")
repo = sh(f"test -d {ASSETS} && echo CLONED || echo NO_CLONE", example="CLONED")
if repo.live and "NO_CLONE" in repo.out:
    clone = "cd ~ && git clone https://github.com/NVIDIA/dgx-spark-playbooks"
    sh(clone, timeout=600) if APPLY else print(f"$ {clone}   [not run]")
files = sh(f"cd {ASSETS}/models 2>/dev/null && ls -la {' '.join(MODELS)} 2>&1 | awk '{{print $5, $NF}}'",
           reference="models/\n├── checkpoints/\n│   └── flux1-dev.safetensors\n├── loras/\n├── text_encoders/\n"
                     "│   ├── clip_l.safetensors\n│   └── t5xxl_fp16.safetensors\n└── vae/\n    └── ae.safetensors")
have_models = files.live and all(m in files.out and "No such file" not in files.out for m in MODELS)
if files.live:
    check(have_models, "all four model files are present", "model files missing — run download.sh (below)")
print("  FLUX.1-dev is gated: accept the terms on https://huggingface.co/black-forest-labs/FLUX.1-dev, then ON THE SPARK:")
print(f"  $ cd {ASSETS}")
print("  $ export HF_TOKEN=<YOUR_HF_TOKEN>")
print("  $ nohup sh download.sh > ~/w25/logs/m12_flux_download.log 2>&1 &     # ~34 GB: 23.8 + 9.8 + 0.3 + 0.2")
note("download.sh sends the token in a curl header and skips files that already exist, so re-running it resumes.")

# ── STEP 2 ────────────────────────────────────────────────────────────────────
step(2, "the flux-train image (playbook Step 6: docker build -f Dockerfile.train -t flux-train .)")
img = sh("docker image inspect flux-train --format '{{.Id}}' 2>/dev/null || echo IMAGE_MISSING", example="IMAGE_MISSING")
have_image = img.live and "IMAGE_MISSING" not in img.out
build = (f"mkdir -p ~/w25/logs && cd {ASSETS} && {{ nohup docker build -f Dockerfile.train -t flux-train . "
         "> ~/w25/logs/m12_flux_build.log 2>&1 < /dev/null & }; sleep 2; tail -2 ~/w25/logs/m12_flux_build.log")
if not have_image:
    sh(build, example="#1 [internal] load build definition from Dockerfile.train") if APPLY else print(f"$ {build}   [not run]")
note("Dockerfile.train: nvcr.io/nvidia/pytorch:25.09-py3 + kohya-ss/sd-scripts (branch sd3) + its requirements.")

# ── STEP 3 ────────────────────────────────────────────────────────────────────
step(3, "the dataset: the playbook's two concepts" + (" + sparkhotel (lab 03)" if HOTEL else ""))
if HOTEL:
    ours = MOD / "flux_data"
    if not (ours / "data.toml").is_file():
        sys.exit("✕ run lab03_flux_dataset.py first — it writes flux_data/data.toml and flux_data/sparkhotel/.")
    files_up = [(p, f"{ASSETS}/flux_data/sparkhotel/{p.name}") for p in sorted((ours / "sparkhotel").iterdir())]
    if APPLY:
        sh(f"cp -n {ASSETS}/flux_data/data.toml {ASSETS}/flux_data/data.toml.orig", example="")
        ok_up = all(put(p, r) for p, r in files_up) and put(ours / "data.toml", f"{ASSETS}/flux_data/data.toml")
        if where() != "dry" and not ok_up:
            warn("upload failed")
            sys.exit(1)
    else:
        print(f"$ cp -n {ASSETS}/flux_data/data.toml {ASSETS}/flux_data/data.toml.orig   [not run]")
        print(f"$ scp {len(files_up)} files → {ASSETS}/flux_data/sparkhotel/ + data.toml   [not run]")
    note("Prompt the result with 'sparkhotel lobby' — and remember lab 03's images are synthetic stand-ins.")
else:
    note("Training the playbook's sample concepts, 'tjtoy toy' and 'sparkgpu gpu', exactly as shipped.")

# ── STEP 4 ────────────────────────────────────────────────────────────────────
step(4, f"launch_train.sh, headless, --max_train_epochs={EPOCHS}")
make = (f"cd {ASSETS} && sed -e 's/docker run -it/docker run/' -e 's/--max_train_epochs=100/--max_train_epochs={EPOCHS}/' "
        "launch_train.sh > launch_train_w25.sh && grep -nE 'docker run|max_train_epochs|save_every' launch_train_w25.sh")
launch = (f"mkdir -p ~/w25/logs && cd {ASSETS} && {{ nohup sh launch_train_w25.sh > {LOG} 2>&1 < /dev/null & }}; "
          f"sleep 3; tail -2 {LOG}")
busy = sh("docker ps --filter ancestor=flux-train --format '{{.ID}} {{.Status}}'", example="", quiet=True)
if busy.live and busy.out.strip():
    note(f"flux-train is already running ({busy.out.strip()}) — monitoring only.")
elif not APPLY:
    print(f"$ {make}   [not run]")
    print(f"$ {launch}   [not run]")
elif where() != "dry" and not (have_models and have_image):
    warn("models or the flux-train image are missing (steps 1–2). Stop any running ComfyUI first, too.")
else:
    sh(make, example="54:docker run \\\n45:    --max_train_epochs=25 \\")
    sh(launch, example="")

# ── STEP 5 ────────────────────────────────────────────────────────────────────
step(5, "monitor: progress, avr_loss, and the LoRA files")
log = sh(f"tail -c 20000 {LOG} 2>/dev/null || echo 'no log yet'", quiet=True, example="")
if log.live:
    bars = progress(log.out)
    if bars:
        s, t, loss = bars[-1]
        note(f"step {s}/{t} ({100 * s / t:.0f} %) · avr_loss {loss:.4f} (a moving average; diffusion loss is noisy — "
             "judge the LoRA by its images, not this number)")
    if re.search(r"Traceback|Error|OutOfMemory", log.out):
        warn(f"the log shows an error — ssh <spark> 'tail -60 {LOG}'")
else:
    print("◈ DRY — testing the progress parser on an EXAMPLE (synthetic sd-scripts lines, not a run):")
    ex = ("steps:   1%|▏         | 5/500 [02:31<4:09:00, 30.24s/it, avr_loss=0.412]\r"
          "steps:  25%|██▌       | 125/500 [1:02:11<3:06:33, 29.85s/it, avr_loss=0.3571]")
    check(progress(ex) == [(5, 500, 0.412), (125, 500, 0.3571)],
          "progress parser: two bars read (step, total, avr_loss)", f"parser returned {progress(ex)}")
sh(f"ls -la {ASSETS}/models/loras/ 2>/dev/null | grep safetensors || echo 'no LoRA saved yet'",
   example="flux_dreambooth-000025.safetensors")

print("\n  when you have a LoRA — playbook Step 7, ON THE SPARK (stop training first; ComfyUI needs the memory):")
print(f"  $ cd {ASSETS} && docker build -f Dockerfile.inference -t flux-comfyui .   # once")
print("  $ sh launch_comfyui.sh                                                    # interactive, Ctrl+C to stop")
print("  $ ssh -N -L 8188:localhost:8188 spark-a                                 # on the laptop → http://localhost:8188")
note("In ComfyUI press w, load finetuned_flux.json, prompt with the trigger phrases, e.g. 'tjtoy toy holding sparkgpu "
     "gpu in a datacenter'. The playbook: about three minutes per 1024 px image.")
result("FLUX LoRA path set. Next module: evaluate, merge and serve your fine-tunes.")
