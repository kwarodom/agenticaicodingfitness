#!/usr/bin/env python3
"""Lab 12-2 · Fine-tune Qwen2.5-VL with GRPO on the Spark — the playbook's image recipe, run headless.

The playbook starts training from a button in its Streamlit UI. That button does two things: it
writes ui_image/src/train.yaml and runs `python -u src/train_image_vlm.py`. This lab does the same two
things over ssh, under nohup, so you can watch it from the laptop:

  1. the playbook assets on the Spark (git clone, opt-in)
  2. the `vlm_demo` image (you build it yourself: it needs your HF token — printed, never run here)
  3. the model in the Hugging Face cache, and the Kaggle wildfire dataset's folder layout, checked
  4. train.yaml written from the playbook's own file (steps 5 = its smoke test; --steps N to change)
  5. the headless docker run, opt-in, then a monitor for loss, rewards and the final LoRA merge

Opt in with  --yes  or  SPARK_APPLY=1.  The original train.yaml is kept as train.yaml.orig.

Run: .venv/bin/python week25/12_vlm_flux_finetune/labs/lab02_vlm_train_on_spark.py
     SPARK_APPLY=1 .venv/bin/python week25/12_vlm_flux_finetune/labs/lab02_vlm_train_on_spark.py --steps 5
"""
import ast
import os
import re
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import ROOT, banner, check, note, put, result, sh, step, table, warn  # noqa: E402

APPLY = "--yes" in sys.argv or os.environ.get("SPARK_APPLY") == "1"
STEPS = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 5
ASSETS = "~/dgx-spark-playbooks/nvidia/playbook-vlm-finetuning/assets"
LOCAL_YAML = ROOT / "dgx-spark-playbooks/nvidia/playbook-vlm-finetuning/assets/ui_image/src/train.yaml"
LOG = "~/w25/logs/m12_vlm_grpo.log"


def trainer_dicts(text: str) -> list[dict]:
    """Hugging Face / TRL console dicts that carry a loss, e.g. {'loss': 0.01, 'reward': 3.5, …}."""
    out = []
    for m in re.finditer(r"\{'[^{}]*\}", text):
        try:
            d = ast.literal_eval(m.group(0))
        except (ValueError, SyntaxError):
            continue
        if isinstance(d, dict) and "loss" in d:
            out.append(d)
    return out


def dataset_layout(listing: str) -> dict:
    """'train/wildfire 123' lines → {split: {class: n}}."""
    counts: dict = {}
    for ln in listing.splitlines():
        m = re.match(r"\s*(\w+)/(\w+)\s+(\d+)", ln)
        if m:
            counts.setdefault(m.group(1), {})[m.group(2)] = int(m.group(3))
    return counts


banner("Lab 12-2 · VLM fine-tuning on the Spark — Qwen2.5-VL-7B + GRPO, headless",
       f"steps={STEPS} · run={'yes (opted in)' if APPLY else 'no (add --yes or SPARK_APPLY=1)'}")

# ── STEP 1 ────────────────────────────────────────────────────────────────────
step(1, "the playbook's assets on the Spark (playbook Step 2)")
repo = sh(f"test -d {ASSETS} && ls {ASSETS} || echo 'NO_CLONE'", example="Dockerfile\nREADME.md\nlaunch.sh\nui_image\nui_video")
note("The playbook says `cd client-hardware-playbooks/nvidia/…` after cloning; the folder git creates is "
     "dgx-spark-playbooks, so the course uses ~/dgx-spark-playbooks/nvidia/playbook-vlm-finetuning/assets.")
if repo.live and "NO_CLONE" in repo.out:
    clone = "cd ~ && git clone https://github.com/NVIDIA/dgx-spark-playbooks"
    if APPLY:
        sh(clone, timeout=600)
    else:
        print(f"$ {clone}   [not run]")

# ── STEP 2 ────────────────────────────────────────────────────────────────────
step(2, "the vlm_demo image (playbook Step 3) — you build it, because it needs your token")
img = sh("docker image inspect vlm_demo --format '{{.Id}}' 2>/dev/null || echo 'IMAGE_MISSING'", example="IMAGE_MISSING")
print("  run this yourself ON THE SPARK (the token stays in your shell, never in a lab):")
print(f"  $ cd {ASSETS}")
print("  $ export HF_TOKEN=<YOUR_HF_TOKEN>")
print("  $ nohup docker build --build-arg HF_TOKEN=$HF_TOKEN -t vlm_demo . > ~/w25/logs/m12_vlm_build.log 2>&1 &")
warn("the Dockerfile runs `hf auth login --token $HF_TOKEN` in a layer, so the image contains your token. "
     "Never push vlm_demo to a registry, and delete it (docker rmi vlm_demo) when you are done.")
have_image = img.live and "IMAGE_MISSING" not in img.out

# ── STEP 3 ────────────────────────────────────────────────────────────────────
step(3, "the model and the dataset (playbook Steps 5.1–5.2)")
sh("ls ~/.cache/huggingface/hub 2>/dev/null | grep -i 'qwen2.5-vl' || echo 'no Qwen2.5-VL in the HF cache yet'",
   example="models--Qwen--Qwen2.5-VL-7B-Instruct")
note("The playbook downloads Qwen/Qwen2.5-VL-7B-Instruct, but src/train.yaml trains model_id "
     "unsloth/Qwen2.5-VL-7B-Instruct — a separate repo that Unsloth downloads on the first run (~16 GB more).")
lay = sh(f"cd {ASSETS}/ui_image/data 2>/dev/null && for d in */*/; do echo \"${{d%/}} $(ls \"$d\" | wc -l)\"; done "
         "|| echo 'NO_DATA: download the Kaggle zip into ui_image/data (playbook 5.2)'",
         example="train/nowildfire <count>\ntrain/wildfire <count>\ntest/nowildfire <count>\n…")
if lay.live:
    counts = dataset_layout(lay.out)
    if counts:
        table([[s, c, n] for s, cs in counts.items() for c, n in cs.items()], ["split", "class", "images"])
        check(set(counts) <= {"train", "valid", "validation", "test"} and "train" in counts
              and all(set(cs) == {"nowildfire", "wildfire"} for cs in counts.values()),
              "layout matches what train_image_vlm.py expects: split folders, classes nowildfire / wildfire",
              "unexpected layout — train_image_vlm.py reads load_dataset('data')['train'] and the folder names")
    else:
        warn("no dataset yet: sign in to Kaggle, copy the cURL command for the Wildfire Prediction Dataset, "
             f"run it in {ASSETS}/ui_image/data, then `unzip -qq wildfire-prediction-dataset.zip`.")

# ── STEP 4 ────────────────────────────────────────────────────────────────────
step(4, f"train.yaml — the playbook's own file, steps={STEPS}")
cfg = yaml.safe_load(LOCAL_YAML.read_text(encoding="utf-8")) if LOCAL_YAML.is_file() else None
if cfg is None:
    sys.exit(f"✕ {LOCAL_YAML} not found — clone dgx-spark-playbooks at the repo root.")
shipped = cfg["hyperparameters"]["steps"]
cfg["hyperparameters"]["steps"] = STEPS
table([["model", cfg["model"]["model_id"]], ["LoRA rank / alpha", f"{cfg['model']['lora_config']['rank']} / "
       f"{cfg['model']['lora_config']['alpha']}"], ["steps", f"{STEPS} (the repo ships {shipped}; the UI default is 100)"],
       ["batch · generations", f"{cfg['hyperparameters']['batch_size']} · {cfg['hyperparameters']['num_generations']}"],
       ["rewards", f"format {cfg['hyperparameters']['format_reward']} · correctness {cfg['hyperparameters']['correctness_reward']}"],
       ["max_seq_length", cfg["model"]["max_seq_length"]], ["output_dir", cfg["hyperparameters"]["output_dir"]]],
      ["train.yaml key", "value"])
note("GRPO: for each image the model writes num_generations answers; the reward functions score each one "
     "(format: <REASONING>…</REASONING><SOLUTION>…</SOLUTION>, correctness: the Yes/No against the folder label). There is "
     "no answer text to imitate, only a score. The playbook: ~100 steps can take up to about 2 hours.")
tmp = Path(tempfile.mkdtemp()) / "train.yaml"
tmp.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")

# ── STEP 5 ────────────────────────────────────────────────────────────────────
step(5, "launch headless (what the UI's Start Finetuning button runs), then monitor")
run = (f"mkdir -p ~/w25/logs && cd {ASSETS} && {{ nohup docker run --gpus=all --net=host --ipc=host "
       "--ulimit memlock=-1 --ulimit stack=67108864 --rm --name w25-vlm-grpo "
       "-v $(pwd):/vlm_finetuning -v $HOME/.cache/huggingface:/root/.cache/huggingface "
       f"-w /vlm_finetuning/ui_image vlm_demo python -u src/train_image_vlm.py > {LOG} 2>&1 < /dev/null & }}; "
       f"sleep 3; tail -2 {LOG}")
busy = sh("docker ps --format '{{.Names}}' | grep -E '^w25-' || true", example="", quiet=True)
if busy.live and busy.out.strip():
    note(f"already running: {busy.out.strip()} — monitoring only.")
elif not APPLY:
    print(f"$ cp -n {ASSETS}/ui_image/src/train.yaml {ASSETS}/ui_image/src/train.yaml.orig   [not run]")
    print(f"$ scp train.yaml <spark>:{ASSETS}/ui_image/src/train.yaml   [not run]")
    print(f"$ {run}   [not run]")
elif not have_image and repo.live:
    warn("build vlm_demo first (step 2), then re-run with SPARK_APPLY=1.")
else:
    sh(f"cp -n {ASSETS}/ui_image/src/train.yaml {ASSETS}/ui_image/src/train.yaml.orig", example="")
    put(tmp, f"{ASSETS}/ui_image/src/train.yaml")
    sh(run, example="==========\n== CUDA ==\n==========")

log = sh(f"tail -c 30000 {LOG} 2>/dev/null || echo 'no log yet'", quiet=True, example="")
if log.live:
    rows = trainer_dicts(log.out)
    if rows:
        keys = ["loss"] + [k for k in rows[-1] if "reward" in k][:3]
        table([[i + 1] + [f"{r.get(k, 0):.4g}" if isinstance(r.get(k), (int, float)) else "—" for k in keys]
               for i, r in enumerate(rows[-8:])], ["log line"] + keys)
    merged = "Merging LoRA weights and saving the model" in log.out
    if re.search(r"Traceback|Error", log.out) and not merged:
        warn(f"the log shows an error — ssh <spark> 'tail -60 {LOG}'")
    sh(f"ls {ASSETS}/ui_image/saved_model 2>/dev/null || echo 'no saved_model yet'", example="")
    if merged:
        result("training finished and the LoRA was merged into saved_model/checkpoint-N. Compare base vs fine-tuned "
               "in the Streamlit demo: cd ui_image && streamlit run Image_VLM.py (inside the container, playbook 5.5).")
        sys.exit(0)
else:
    print("◈ DRY — testing the two parsers on an EXAMPLE (synthetic lines written for this course, not a run):")
    ex_log = ("{'loss': 0.0123, 'grad_norm': 1.1, 'learning_rate': 5e-06, 'reward': 3.5, "
              "'rewards/format_reward_func/mean': 1.5, 'epoch': 0.01}\n 40%|████      | 2/5 [01:10<01:45]\n"
              "{'train_runtime': 180.0, 'train_loss': 0.02, 'epoch': 0.02}")
    ex_lay = "train/nowildfire 12\ntrain/wildfire 15\ntest/nowildfire 3\ntest/wildfire 3"
    d, lay_c = trainer_dicts(ex_log), dataset_layout(ex_lay)
    check(len(d) == 1 and d[0]["reward"] == 3.5 and lay_c == {"train": {"nowildfire": 12, "wildfire": 15},
                                                               "test": {"nowildfire": 3, "wildfire": 3}},
          "parsers: 1 loss/reward dict (the train_runtime summary skipped) · a 2-split, 2-class layout",
          f"parser self-test failed: {d} {lay_c}")
result("the image recipe is ready to run headless. Lab 03 prepares a FLUX.1 dataset meanwhile.")
