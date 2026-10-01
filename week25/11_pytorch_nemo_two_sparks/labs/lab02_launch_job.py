#!/usr/bin/env python3
"""Lab 11-2 · Launch a fine-tuning job on Spark A, in the background, with a log file.

Starts one playbook recipe inside its NGC container with `nohup docker run … > ~/w25/logs/<job>.log`
and returns at once — training keeps running after this lab ends. Lab 03 reads the log.

  --recipe pytorch-lora-8b     (default) the PyTorch playbook's usage example: Llama 3.1 8B LoRA, 100 samples
  --recipe pytorch-full-3b     Llama 3.2 3B full SFT, script defaults
  --recipe pytorch-qlora-70b   Llama 3.1 70B QLoRA, script defaults
  --recipe nemo-lora-8b        NeMo AutoModel LoRA on Llama 3.1 8B, 20 steps
  --recipe nemo-qlora-70b      NeMo AutoModel QLoRA on Meta-Llama-3 70B, 20 steps
  --recipe nemo-sft-qwen3-8b   NeMo AutoModel full SFT on Qwen3 8B, 20 steps

Before it starts anything it checks: Docker works, no other w25-m11 job is running, and a Hugging
Face login exists in ~/.cache/huggingface (it never prints the token). In DRY mode it prints the
commands only. Stop a job with: docker stop w25-m11-<recipe>

Run: .venv/bin/python week25/11_pytorch_nemo_two_sparks/labs/lab02_launch_job.py [--recipe …]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, note, result, sh, step, warn  # noqa: E402

PT_IMAGE = "nvcr.io/nvidia/pytorch:25.11-py3"          # PyTorch fine-tune playbook
NEMO_IMAGE = "nvcr.io/nvidia/nemo-automodel:26.02"      # NeMo fine-tune playbook
ASSETS = "~/w25/dgx-spark-playbooks/nvidia/playbook-pytorch-fine-tune/assets"

# recipe: (image, command run inside the container) — copied from the two playbooks
RECIPES = {
    "pytorch-lora-8b": (PT_IMAGE, "python Llama3_8B_LoRA_finetuning.py --dataset_size 100 --num_epochs 1 --batch_size 2"),
    "pytorch-full-3b": (PT_IMAGE, "python Llama3_3B_full_finetuning.py"),
    "pytorch-qlora-70b": (PT_IMAGE, "python Llama3_70B_qLoRA_finetuning.py"),
    "nemo-lora-8b": (NEMO_IMAGE, "python3 examples/llm_finetune/finetune.py "
                                 "-c examples/llm_finetune/llama3_2/llama3_2_1b_squad_peft.yaml "
                                 "--model.pretrained_model_name_or_path meta-llama/Llama-3.1-8B "
                                 "--packed_sequence.packed_sequence_size 1024 --step_scheduler.max_steps 20"),
    "nemo-qlora-70b": (NEMO_IMAGE, "python3 examples/llm_finetune/finetune.py "
                                   "-c examples/llm_finetune/llama3_1/llama3_1_8b_squad_qlora.yaml "
                                   "--model.pretrained_model_name_or_path meta-llama/Meta-Llama-3-70B "
                                   "--loss_fn._target_ nemo_automodel.components.loss.te_parallel_ce.TEParallelCrossEntropy "
                                   "--step_scheduler.local_batch_size 1 --packed_sequence.packed_sequence_size 1024 "
                                   "--step_scheduler.max_steps 20"),
    "nemo-sft-qwen3-8b": (NEMO_IMAGE, "python3 examples/llm_finetune/finetune.py "
                                      "-c examples/llm_finetune/qwen/qwen3_8b_squad_spark.yaml "
                                      "--model.pretrained_model_name_or_path Qwen/Qwen3-8B "
                                      "--step_scheduler.local_batch_size 1 --step_scheduler.max_steps 20 "
                                      "--packed_sequence.packed_sequence_size 1024"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--recipe", choices=sorted(RECIPES), default="pytorch-lora-8b")
args = ap.parse_args()
image, inner = RECIPES[args.recipe]
name = f"w25-m11-{args.recipe}"
log = f"~/w25/logs/m11_{args.recipe}.log"


def launch_cmd() -> str:
    """The playbook's `docker run`, made non-interactive: no -it, a --name, nohup and a log file.
    Course change: pinned versions + `pip uninstall -y torchao`. The playbook's unpinned install now pulls
    transformers 5 / trl 1.x, whose SFTConfig has no `logging_dir` (the playbook scripts pass it), and a peft that
    rejects the container's torchao 0.14 when it attaches LoRA. Tested on pytorch:25.11: transformers 4.57.6,
    trl 0.25.1, peft 0.17.1."""
    if image == PT_IMAGE:
        return f"""mkdir -p ~/w25/logs
[ -d ~/w25/dgx-spark-playbooks/.git ] || { rm -rf ~/w25/dgx-spark-playbooks; git clone --depth 1 https://github.com/NVIDIA/dgx-spark-playbooks ~/w25/dgx-spark-playbooks; }
cd {ASSETS}
nohup docker run --gpus all --rm --ipc=host --name {name} \\
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \\
  -v "${{PWD}}:/workspace" -w /workspace \\
  {image} \\
  bash -c 'pip install "transformers>=4.57.1,<5" "trl>=0.25.1,<0.26" "peft<0.18" datasets "bitsandbytes>=0.48.2" && pip uninstall -y torchao && {inner}' \\
  > {log} 2>&1 &
echo "started {name} → {log}\""""
    return f"""mkdir -p ~/w25/logs ~/w25/nemo-checkpoints
nohup docker run --gpus all --ulimit memlock=-1 --ulimit stack=67108864 --rm --name {name} \\
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \\
  -v "$HOME/w25/nemo-checkpoints:/opt/Automodel/checkpoints" \\
  --entrypoint /usr/bin/bash {image} \\
  -c 'cd /opt/Automodel && {inner}' \\
  > {log} 2>&1 &
echo "started {name} → {log}\""""


banner(f"Lab 11-2 · launch {args.recipe} on Spark A", f"{image} · nohup + log file · lab 03 watches it")

step(1, "can this Spark run it? Docker, a free GPU, a Hugging Face login")
r = sh("docker version --format '{{.Server.Version}}'", timeout=30, example="28.3.3")
docker_ok = r.ok and "permission denied" not in r.out.lower()
if r.live and not docker_ok:
    warn("Docker is not usable without sudo: sudo usermod -aG docker $USER, then log out and in (playbook Step 1).")
busy = sh("docker ps --filter name=w25-m11 --format '{{.Names}} {{.Status}}'", timeout=30, example="").out.strip()
if busy:
    warn(f"a course job is already running: {busy}. One fine-tune at a time — they share 128 GB. "
         f"Stop it with: docker stop {busy.split()[0]}")
tok = sh("test -s ~/.cache/huggingface/token && echo 'hf login: present' || echo 'hf login: missing'",
         timeout=30, example="hf login: present").out
if "missing" in tok:
    warn("No Hugging Face login on this Spark. The Llama models are gated: accept each licence on huggingface.co, "
         "then run `hf auth login` once in the ⌨ terminal on the Spark (Spark A). The container reads it from the "
         "mounted ~/.cache/huggingface.")
img = sh(f"docker image inspect {image} --format '{{{{.Id}}}}' >/dev/null 2>&1 && echo present || echo 'not pulled yet'",
         timeout=30, example="not pulled yet").out
if "not pulled" in img:
    note(f"{image} is not on this Spark yet: `docker run` pulls it first (several GB), so the log starts late.")

step(2, "start it in the background")
note("Course deviations from the playbooks, on purpose: no `-it` (nothing is attached), a --name so you can stop it, "
     "nohup + a log file." + (" The playbook clones the recipes inside the container; this clones them once "
                              "to ~/w25 on the Spark and mounts them." if image == PT_IMAGE else
                              " The HF cache and a checkpoints folder are mounted — with --rm, anything written "
                              "only inside the container disappears when training ends."))
ready = not r.live or (docker_ok and not busy and "missing" not in tok)
if not ready:
    result("not started — fix the ⚠ lines above, then run this lab again.")
    sys.exit(0)
out = sh(launch_cmd(), timeout=180, example=f"started {name} → {log}")

step(3, "what now")
print(f"$ docker ps --filter name={name}")
print(f"$ tail -f {log}")
print(f"$ docker stop {name}                      # to cancel")
if image == NEMO_IMAGE:
    note("NeMo writes checkpoints/epoch_<e>_step_<n>/ — here ~/w25/nemo-checkpoints/ on the Spark. The playbook "
         "also promises a LATEST symlink; the 26.02 container did not create one, so ls the folder.")
else:
    note("The PyTorch scripts save nothing unless the script has --output_dir (only the 70B LoRA script defines it). "
         "This run proves the pipeline and gives you a loss curve.")
if out.live:
    result(f"{args.recipe} is running in the background. Run lab 03 in a minute to read the log.")
else:
    result("DRY: nothing started. Connect Spark A to launch the job for real.")
