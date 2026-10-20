#!/usr/bin/env python3
"""Lab 10-1 · Run NVIDIA's Unsloth playbook on the Spark: container, installs, and its 60-step validation.

Step 1 runs the playbook's prerequisite checks (read-only). Step 2 checks for the playbook's container
image, nvcr.io/nvidia/pytorch:25.11-py3, and pulls it if you opt in (under nohup: it is large). Step 3
uploads spark/run_unsloth.sh — the playbook's docker run + pip installs, made headless — and, if you opt
in, runs the playbook's own test_unsloth.py with it under nohup. Step 4 reads that log: the Unsloth
patch message, the loss lines, and the final metrics.

Opt in with  --yes  or  SPARK_APPLY=1.  Re-run the lab at any time to follow progress.

Run: .venv/bin/python week25/10_unsloth/labs/lab01_unsloth_validate.py
     SPARK_APPLY=1 .venv/bin/python week25/10_unsloth/labs/lab01_unsloth_validate.py
"""
import ast
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, bar, note, put, result, sh, step, table, warn, where  # noqa: E402

MOD = Path(__file__).resolve().parents[1]
APPLY = "--yes" in sys.argv or os.environ.get("SPARK_APPLY") == "1"
IMAGE = "nvcr.io/nvidia/pytorch:25.11-py3"
LOG = "~/w25/logs/m10_validate.log"
PULL_LOG = "~/w25/logs/m10_pull.log"


def loss_lines(text: str) -> list[dict]:
    """Hugging Face Trainer console dicts: {'loss': 1.23, 'grad_norm': …, 'learning_rate': …, 'epoch': …}."""
    out = []
    for m in re.finditer(r"\{'loss'[^{}]*\}", text):
        try:
            out.append(ast.literal_eval(m.group(0)))
        except (ValueError, SyntaxError):
            pass
    return out


banner("Lab 10-1 · Unsloth on the Spark — the playbook's container and validation run",
       f"image {IMAGE} · run={'yes (opted in)' if APPLY else 'no (add --yes or SPARK_APPLY=1)'}")

# ── STEP 1 ────────────────────────────────────────────────────────────────────
step(1, "prerequisites (playbook Step 1) — read-only")
rows = []
for title, cmd, ex, test in [
    ("CUDA 13.0 toolkit", "nvcc --version | tail -2",
     "Cuda compilation tools, release 13.0, V13.0.xx", lambda o: "release 13" in o),
    ("GPU visible", "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader",
     "NVIDIA GB10, 580.95.05", lambda o: "GB10" in o),
    ("Docker without sudo", "docker version --format '{{.Server.Version}}'", "28.3.3",
     lambda o: bool(re.match(r"\d+\.\d+", o.strip()))),
    ("NVIDIA container runtime", "(docker info --format '{{json .Runtimes}}' | grep -o '\"nvidia\"'; docker info 2>/dev/null | grep -o 'nvidia.com/gpu=all') "
     "| head -1 | grep . || echo none",   # a registered nvidia runtime, or CDI devices (how this DGX OS exposes the GPU)
     '"nvidia"', lambda o: "nvidia" in o),
]:
    r = sh(cmd, example=ex, timeout=60)
    rows.append([title, ("✓" if test(r.out) else "✕") if r.live else "◈ example",
                 (r.out.strip().splitlines() or ["—"])[-1][:44]])
print()
table(rows, ["prerequisite", "result", "last line"])

# ── STEP 2 ────────────────────────────────────────────────────────────────────
step(2, f"the container image (playbook Step 2: docker pull {IMAGE})")
img = sh(f"docker image inspect {IMAGE} --format '{{{{.Id}}}} {{{{.Size}}}}' 2>/dev/null || echo 'IMAGE_MISSING'",
         example="IMAGE_MISSING")
have_image = img.live and "IMAGE_MISSING" not in img.out
pulling = sh(f"pgrep -af '[d]ocker pull {IMAGE}' || echo 'no pull running'", example="no pull running", quiet=True)
pull_running = pulling.live and "no pull running" not in pulling.out
pull = f"mkdir -p ~/w25/logs && {{ nohup docker pull {IMAGE} > {PULL_LOG} 2>&1 < /dev/null & }}; sleep 2; tail -2 {PULL_LOG}"
if have_image:
    note("image present — no pull needed.")
elif pull_running:
    sh(f"tail -c 400 {PULL_LOG}", example="")
    note("pull in progress — re-run this lab in a few minutes.")
elif APPLY:
    sh(pull, example=f"25.11-py3: Pulling from nvidia/pytorch\n… Pulling fs layer")
    note("pulling in the background. Re-run the lab once it finishes (docker image inspect succeeds).")
else:
    print(f"$ {pull}   [not run]")

# ── STEP 3 ────────────────────────────────────────────────────────────────────
step(3, "the playbook's validation run (Steps 3–6), headless under nohup")
script = MOD / "spark" / "run_unsloth.sh"
print("\n".join("  " + ln for ln in script.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")))
running = sh("docker ps --filter name=w25-unsloth --format '{{.Names}} {{.Status}}' | head -3 || true",
             example="", quiet=True)
job_running = running.live and "w25-unsloth" in running.out
launch = (f"mkdir -p ~/w25/logs && {{ nohup bash ~/w25/m10/run_unsloth.sh validate test_unsloth.py > {LOG} 2>&1 "
          f"< /dev/null & }}; sleep 3; tail -2 {LOG}")
if job_running:
    note(f"an Unsloth job is already running: {running.out.strip()}")
elif not APPLY:
    print(f"$ {launch}   [not run]")
elif not have_image and where() != "dry":
    warn("pull the image first (step 2), then re-run with SPARK_APPLY=1.")
elif put(script, "~/w25/m10/run_unsloth.sh") or where() == "dry":
    sh(launch, example="Collecting transformers\n…")
    note("the pip installs take a few minutes, then the first run downloads Phi-3.5-mini (4-bit) and the LAION OIG data.")

# ── STEP 4 ────────────────────────────────────────────────────────────────────
step(4, "read the validation log (playbook Step 6: expected output)")
log = sh(f"tail -c 30000 {LOG} 2>/dev/null || echo 'no log yet'", quiet=True,
         reference="Expected output in the terminal window:\n"
                   "- \"Unsloth: Will patch your computer to enable 2x faster free finetuning\"\n"
                   "- Training progress bars showing loss decreasing over 60 steps\n"
                   "- Final training metrics showing completion")
if not log.live:
    print("◈ REFERENCE — what the playbook says to expect (not your machine):")
    print(log.out)
else:
    patched = bool(re.search(r"Unsloth: Will patch|Unsloth.*patch", log.out))
    steps = loss_lines(log.out)
    done = "W25_JOB_DONE" in log.out
    print(f"{'✓' if patched else '○'} Unsloth patch message {'seen' if patched else 'not seen yet'}")
    if steps:
        hi = max(s["loss"] for s in steps)
        pick = [steps[i] for i in sorted({round(i * (len(steps) - 1) / 5) for i in range(6)})]
        table([[f"{s.get('epoch', 0):.3f}", f"{s['loss']:.4f}", bar(s["loss"], hi, 24)] for s in pick],
              ["epoch", "loss", f"loss (0 … {hi:.2f})"])
        note(f"{len(steps)} loss lines · {steps[0]['loss']:.3f} → {steps[-1]['loss']:.3f} (your run, LIVE)")
    m = re.search(r"'train_runtime': ([\d.]+)", log.out)
    if m:
        note(f"train_runtime {float(m.group(1)):.0f} s for the playbook's 60 steps")
    if re.search(r"Traceback|Error", log.out) and not done:
        warn(f"the log shows an error — read it: ssh <spark> 'tail -60 {LOG}'")
    if done:
        result("validation finished — Unsloth works in this container. Next: lab 02 trains the hotel dataset.")
        sys.exit(0)
result("not finished yet (or DRY). Re-run this lab to follow the log; lab 02 prepares the hotel run meanwhile.")
