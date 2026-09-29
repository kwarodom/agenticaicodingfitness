#!/usr/bin/env python3
"""Lab 09-1 · Set up LLaMA Factory on the Spark exactly as NVIDIA's playbook does, over ssh.

Step 1 runs the playbook's four prerequisite checks (read-only). Step 2 looks for an existing install.
Step 3 uploads spark/setup_factory.sh — the playbook's steps 2–6 (venv, PyTorch cu130, clone, pip
install) — and, only if you opt in, runs it under nohup, because the PyTorch download takes longer
than a lab may run. Step 4 reads the setup log and verifies the install the playbook's way.
Run the lab again at any time: it only reports progress.

Opt in with  --yes  or  SPARK_APPLY=1.  Nothing is ever deleted.

Run: .venv/bin/python week25/09_llama_factory/labs/lab01_factory_setup.py
     SPARK_APPLY=1 .venv/bin/python week25/09_llama_factory/labs/lab01_factory_setup.py
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check, note, put, result, sh, step, table, warn, where  # noqa: E402

MOD = Path(__file__).resolve().parents[1]
APPLY = "--yes" in sys.argv or os.environ.get("SPARK_APPLY") == "1"
LOG = "~/w25/logs/m09_setup.log"

# Upstream examples/train_lora/qwen3_lora_sft.yaml as cloned on 2026-09-29 (LLaMA-Factory commit ce9dc9e).
UPSTREAM_YAML = """### model
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: identity,alpaca_en_demo
template: qwen3_nothink
cutoff_len: 2048
max_samples: 1000
…
### output
output_dir: saves/qwen3-4b/lora/sft
…"""

banner("Lab 09-1 · LLaMA Factory setup on the Spark (the playbook, over ssh)",
       f"venv + PyTorch cu130 + clone + pip install -e · install={'yes (opted in)' if APPLY else 'no (add --yes or SPARK_APPLY=1)'}")

# ── STEP 1 ────────────────────────────────────────────────────────────────────
step(1, "the playbook's prerequisite checks (Step 1) — read-only")
CHECKS = [
    ("CUDA toolkit ≥ 12.9", "nvcc --version | tail -2",
     "Cuda compilation tools, release 13.0, V13.0.xx\nBuild cuda_13.0.r13.0/compiler.xxxxxxxx_0",
     lambda o: bool(re.search(r"release (1[3-9]|12\.9)", o))),
    ("GPU visible", "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader",
     "NVIDIA GB10, 580.95.05", lambda o: "GB10" in o),
    ("Python 3 + venv", "python3 --version && python3 -c 'import venv; print(\"venv ok\")'",
     "Python 3.12.3\nvenv ok", lambda o: "Python 3" in o and "venv ok" in o),
    ("Git", "git --version", "git version 2.43.0", lambda o: "git version" in o),
    ("> 50 GB free for models + checkpoints", "df -h ~ | tail -1",
     "/dev/nvme0n1p2  3.7T  412G  3.1T  12% /", lambda o: bool(re.search(r"\s(\d+(\.\d+)?)T\s|\s([5-9]\d|\d{3,})G\s", o))),
]
rows = []
for title, cmd, ex, test in CHECKS:
    r = sh(cmd, example=ex, timeout=60)
    status = ("✓" if test(r.out) else "✕") if r.live else "◈ example"
    rows.append([title, status, (r.out.strip().splitlines() or ["—"])[-1][:48]])
print()
table(rows, ["prerequisite", "result", "last line"])

# ── STEP 2 ────────────────────────────────────────────────────────────────────
step(2, "is LLaMA Factory already installed?")
state = sh("ls -d ~/factoryEnv ~/LLaMA-Factory 2>/dev/null; "
           "test -x ~/factoryEnv/bin/llamafactory-cli && echo 'CLI_READY' || echo 'CLI_MISSING'; "
           f"test -f {LOG} && tail -1 {LOG} || true",
           example="CLI_MISSING")
installed = state.live and "CLI_READY" in state.out
running = sh("pgrep -af '[s]etup_factory.sh' || echo 'setup not running'", example="setup not running", quiet=True)
in_progress = running.live and "setup not running" not in running.out
if installed:
    note("~/factoryEnv/bin/llamafactory-cli exists — skipping the install, going straight to verification.")
elif in_progress:
    note("the setup script is running right now — step 4 shows its log. Re-run this lab to follow it.")

# ── STEP 3 ────────────────────────────────────────────────────────────────────
step(3, "upload the playbook's steps 2–6 as one script, run it under nohup (opt-in)")
script = MOD / "spark" / "setup_factory.sh"
print("\n".join("  " + ln for ln in script.read_text(encoding="utf-8").splitlines()[7:] if ln.strip()))
launch = (f"mkdir -p ~/w25/logs && {{ nohup bash ~/w25/m09/setup_factory.sh > {LOG} 2>&1 < /dev/null & }}; "
          f"sleep 3; tail -3 {LOG}")
if installed or in_progress:
    pass
elif not APPLY:
    print(f"$ {launch}   [not run]")
    note("Installing downloads PyTorch (several GB) and writes ~/factoryEnv and ~/LLaMA-Factory on the Spark. "
         "Re-run with SPARK_APPLY=1 to do it.")
else:
    if put(script, "~/w25/m09/setup_factory.sh") or where() == "dry":
        sh(launch, example="== Step 2 · Python virtual environment (~/factoryEnv)\n"
                           "== Step 3 · PyTorch with CUDA 13 support\nLooking in indexes: https://download.pytorch.org/whl/cu130")
        note("Running in the background. Re-run this lab in a few minutes to see step 4 turn ✓.")

# ── STEP 4 ────────────────────────────────────────────────────────────────────
step(4, "verify — the playbook's Step 4 check, plus the CLI and the example config (Step 7)")
tail = sh(f"tail -n 4 {LOG} 2>/dev/null || echo 'no setup log yet'",
          example="  | Welcome to LLaMA Factory, version 0.9.x                    |\nSETUP_DONE")
done = tail.live and "SETUP_DONE" in tail.out
if tail.live and re.search(r"(?i)error|Traceback", tail.out) and not done:
    warn(f"the setup log shows an error — read it: ssh <spark> 'tail -50 {LOG}'. Fix, then re-run with SPARK_APPLY=1 "
         "(the script skips steps that are already done).")
v = sh("~/factoryEnv/bin/python -c \"import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')\"",
       example="PyTorch: 2.x.x+cu130, CUDA: True")
if v.live:
    cuda_ok = check("CUDA: True" in v.out, "PyTorch in ~/factoryEnv sees the GPU (CUDA: True)",
                    "PyTorch is missing or reports CUDA: False — still installing? else see Troubleshooting")
else:
    cuda_ok = False
    print("◈ not verified — DRY mode, the line above is an EXAMPLE")
sh("cat ~/LLaMA-Factory/examples/train_lora/qwen3_lora_sft.yaml | head -24", example=UPSTREAM_YAML)
if not tail.live:
    note("The DRY text above is the upstream file as cloned on 2026-09-29 (commit ce9dc9e) — yours may be newer.")

# ── STEP 5 ────────────────────────────────────────────────────────────────────
step(5, "the playbook's own smoke test, and LLaMA Board (both optional, printed not run)")
print("$ cd ~/LLaMA-Factory && source ~/factoryEnv/bin/activate")
print("$ llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml      # playbook Step 8: ~15 min, 411 steps")
print("$ GRADIO_SERVER_NAME=127.0.0.1 llamafactory-cli webui                 # LLaMA Board on :7860, localhost only")
print("$ ssh -N -L 7860:localhost:7860 spark-a                              # on the laptop, then open http://localhost:7860")
note("LLaMA Board is the same trainer with a web form: pick model, dataset, method and it writes the YAML for you. "
     "It listens on 0.0.0.0 by default; GRADIO_SERVER_NAME=127.0.0.1 keeps it off your tailnet.")

if cuda_ok:
    result("LLaMA Factory is installed and PyTorch sees the GB10. Next: lab 02 builds the hotel dataset.")
else:
    result("not verified yet — in DRY mode the rows above are EXAMPLE shapes. With a Spark: SPARK_APPLY=1, "
           "wait for SETUP_DONE, re-run.")
