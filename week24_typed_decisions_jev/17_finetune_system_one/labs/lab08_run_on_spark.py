#!/usr/bin/env python3
"""Lab 17-8 · The same fine-tune on a DGX Spark — same code, CUDA instead of MPS/CPU.

  1. find your Spark: SPARK_HOST (env) or week25_dgx_spark/.env.local, and test it with ssh BatchMode
  2. the plan: copy week24_typed_decisions_jev code + data, run Labs 17-4 and 17-5 inside NVIDIA's PyTorch container
  3. --run (⌨ terminal, ≈ 10–20 min incl. first image pull + model download): execute the plan and
     bring back the run files, so Lab 17-2 can show measured Spark numbers next to the Mac's
  4. compare: Mac vs Spark, measured — or say plainly that the Spark side has not been measured yet

The ▶ Run button never starts a remote job: it checks reachability and prints the plan (NOT RUN).
Run:  .venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab08_run_on_spark.py            # plan + status
      SPARK_HOST=me@spark-abcd .venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab08_run_on_spark.py --run
"""
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import ftkit as F  # noqa: E402  (first: may re-run this lab inside week24_typed_decisions_jev/.venv-local)
from jevkit import banner, step, table  # noqa: E402

RUN = "--run" in sys.argv
IMAGE = "nvcr.io/nvidia/pytorch:25.11-py3"           # the container week25_dgx_spark's NVIDIA playbooks use on DGX Spark
REMOTE = "~/agentic-week24"
banner("Lab 17-8 · fine-tune on a DGX Spark", "same labs, CUDA + bf16 — measured, or honestly NOT RUN")


# ── 1 · find the Spark ────────────────────────────────────────────────────────
def spark_host() -> tuple[str, str]:
    if os.environ.get("SPARK_HOST", "").strip():
        return os.environ["SPARK_HOST"].strip(), "SPARK_HOST env"
    env = F.WEEK24.parent / "week25_dgx_spark" / ".env.local"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("SPARK_HOST="):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                if v:
                    return v, "week25_dgx_spark/.env.local"
    return "", ""


def ssh(cmd: str, timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", HOST, cmd],
                          capture_output=True, text=True, timeout=timeout)


step(1, "find the Spark")
HOST, where = spark_host()
reach, info = False, ""
if not HOST:
    print("» SPARK_HOST is not set (env or week25_dgx_spark/.env.local). Set it to your ssh target, e.g.")
    print("    export SPARK_HOST=you@spark-abcd        # a Tailscale name works; the key must be in ssh-agent")
else:
    try:
        r = ssh("hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1; "
                "docker --version 2>/dev/null | head -1", timeout=20)
        reach, info = r.returncode == 0, (r.stdout or r.stderr).strip().replace("\n", " · ")
    except (subprocess.TimeoutExpired, OSError) as e:
        info = type(e).__name__
    print(f"» {HOST} (from {where}): {'✓ reachable · ' + info if reach else '✗ not reachable over ssh BatchMode — ' + info[:120]}")

# ── 2 · the plan ──────────────────────────────────────────────────────────────
step(2, "the plan — exactly what --run does")
SYNC = ["week24_typed_decisions_jev/common", "week24_typed_decisions_jev/16_local_system_one_models/localkit.py", "week24_typed_decisions_jev/16_local_system_one_models/tickets.json",
        "week24_typed_decisions_jev/17_finetune_system_one"]
inner = " && ".join([
    "python -c \"import torch; print('torch==' + torch.__version__)\" > /tmp/keep-torch.txt",   # keep NVIDIA's torch
    "pip install -q -c /tmp/keep-torch.txt 'laya==0.4.8' 'gliner2[train]==2.1.0' 'peft>=0.10' tiktoken sentencepiece",
    "export LOCALKIT_ALLOW_DOWNLOAD=1 LOCALKIT_REEXEC=1 USE_TF=0",          # first run downloads the two base models
    "python week24_typed_decisions_jev/17_finetune_system_one/labs/lab04_finetune_laya.py --train",
    "python week24_typed_decisions_jev/17_finetune_system_one/labs/lab05_finetune_gliner.py --train lora",
])
docker = (f"docker run --rm --gpus all --ipc=host -v {REMOTE}:/w -w /w -v ~/.cache/huggingface:/root/.cache/huggingface "
          f"{IMAGE} bash -lc {shlex.quote(inner)}")
plan = [
    ("copy code + data", f"rsync -az --exclude .runs --exclude __pycache__ {' '.join(SYNC)} {HOST or '$SPARK_HOST'}:{REMOTE}/ (keeps paths)"),
    ("train inside NGC PyTorch", f"ssh $SPARK_HOST {docker[:90]}…"),
    ("bring the run files back", f"scp $SPARK_HOST:{REMOTE}/week24_typed_decisions_jev/17_finetune_system_one/.runs/*/lab0[45]_run.json → recorded/lab08_spark.json"),
]
table([[i + 1, a, b] for i, (a, b) in enumerate(plan)], ["", "step", "command"])
print("» Why a container: the Spark is aarch64 + CUDA; NVIDIA's image ships a matching PyTorch, so nothing")
print("  in our code changes. Inside it Laya picks cuda and turns on bf16 autocast + gradient checkpointing,")
print("  and GLiNER2's trainer gets the GPU it never gets on a Mac.")
print("» Checkpoints stay on the Spark (~1.3 GB): serve from there, or copy back only what passes Lab 17-7.")

# ── 3 · run ───────────────────────────────────────────────────────────────────
spark_rec = F.RECORDED / "lab08_spark.json"
if RUN:
    step(3, "running on the Spark")
    if not reach:
        raise SystemExit("✗ not reachable — fix ssh first (step 1). Nothing was started.")
    t0 = time.time()
    root = F.WEEK24.parent
    subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, f"mkdir -p {REMOTE}"], check=True)
    subprocess.run(["rsync", "-azR", "--exclude", ".runs", "--exclude", "__pycache__", *SYNC, f"{HOST}:{REMOTE}/"],
                   cwd=root, check=True)
    print(f"   ✓ copied in {time.time() - t0:.0f} s — training (live output follows)")
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, docker], timeout=3600)
    if r.returncode != 0:
        raise SystemExit(f"✗ remote run exited {r.returncode} — see the output above")
    got = {}
    for name, rel in (("laya", "laya-ft/lab04_run.json"), ("gliner_lora", "gliner-lora/lab05_run.json")):
        out = ssh(f"cat {REMOTE}/week24_typed_decisions_jev/17_finetune_system_one/.runs/{rel}", timeout=30)
        if out.returncode == 0:
            got[name] = json.loads(out.stdout)
    if "laya" in got:
        lr = got["laya"]
        got["laya_items_per_s"] = round(lr["train_items"] * 3 / lr["seconds"], 2)
    got["host_info"], got["measured"] = info, time.strftime("%Y-%m-%d %H:%M")
    spark_rec.write_text(json.dumps(got, indent=1), encoding="utf-8")
    print(f"   ✓ saved {spark_rec.relative_to(HERE)} — Lab 17-2 now shows measured Spark numbers")

# ── 4 · compare ───────────────────────────────────────────────────────────────
step(4, "Mac vs Spark — measured")
mac = json.loads((F.RECORDED / "lab04_laya.json").read_text()) if (F.RECORDED / "lab04_laya.json").is_file() else {}
mac_gl = json.loads((F.RECORDED / "lab05_gliner.json").read_text()).get("lora", {}) if (F.RECORDED / "lab05_gliner.json").is_file() else {}
sp = json.loads(spark_rec.read_text()) if spark_rec.is_file() else {}
rows = [["Laya full fine-tune, 456 tickets × 3 epochs",
         f"{mac['seconds'] / 60:.1f} min ({mac.get('device', '?')})" if mac else "—",
         f"{sp['laya']['seconds'] / 60:.1f} min (cuda)" if sp.get("laya") else "NOT RUN"],
        [f"GLiNER LoRA, 456 tickets × {len(mac_gl.get('eval_history') or []) or 8} epochs (early stopping)",
         f"{mac_gl['seconds'] / 60:.1f} min ({mac_gl.get('device', '?')})" if mac_gl else "—",
         f"{sp['gliner_lora']['seconds'] / 60:.1f} min ({sp['gliner_lora'].get('device')})" if sp.get("gliner_lora") else "NOT RUN"]]
table(rows, ["job", f"Mac ({mac.get('chip', 'this course')})", "DGX Spark"])
if not sp:
    print("» The Spark column is empty on purpose: no Spark run has been recorded for this module yet, and")
    print("  we do not print guesses as measurements. Run --run once with SPARK_HOST set to fill it in.")
