#!/usr/bin/env python3
"""Lab 01-1 · Spark doctor: is this box ready for a week of fine-tuning and serving?

Eight read-only checks, run on your Spark (locally, or over `ssh $SPARK_HOST`):
architecture, OS, GPU + driver, CUDA, unified memory, free disk, Docker, and
the Hugging Face cache. Each prints the real command first. In DRY mode you
see what NVIDIA's playbooks document (REFERENCE) or an illustrative shape (EXAMPLE).

Nothing here changes the machine. The fixes are printed, never run.

Run: .venv/bin/python week25/01_meet_your_spark/labs/lab01_spark_doctor.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check, result, sh, step, table  # noqa: E402

# (title, command, (kind, dry-run output), pass test, fix if it fails)
#   kind "ref" = text quoted verbatim from an NVIDIA playbook; "ex" = illustrative shape written for this
#   course (the GPU and CUDA rows are shaped from the playbooks' stated minimums: driver 580.95.05+, CUDA 13.0).
CHECKS = [
    ("CPU architecture", "uname -m",
     ("ref", "aarch64"),
     lambda o: "aarch64" in o,
     "DGX Spark is Arm (aarch64). If this says x86_64 you are not on the Spark — check SPARK_HOST."),
    ("Operating system", "cat /etc/dgx-release 2>/dev/null | head -2; lsb_release -ds 2>/dev/null",
     ("ex", "Ubuntu 24.04.3 LTS"),
     lambda o: "24.04" in o or "DGX" in o,
     "Playbooks assume DGX OS (Ubuntu 24.04). Update from DGX Dashboard → Settings → Updates."),
    ("GPU and driver", "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader",
     ("ex", "NVIDIA GB10, 580.95.05"),
     lambda o: "GB10" in o,
     "No GB10 visible. Reboot, then run `nvidia-smi`. Playbooks want driver 580.95.05 or newer."),
    ("CUDA toolkit", "nvcc --version 2>/dev/null | tail -2 || ls -d /usr/local/cuda*",
     ("ex", "Cuda compilation tools, release 13.0"),
     lambda o: bool(re.search(r"release 1[3-9]\.|cuda-1[3-9]", o)),
     "Fine-tuning playbooks use CUDA 13 (cu130 wheels). If nvcc is missing: export PATH=/usr/local/cuda/bin:$PATH"),
    ("Unified memory", "free -g | head -2",
     ("ex", "               total        used        free      shared  buff/cache   available\n"
            "Mem:             119           6         105           0           8         112"),
     lambda o: bool(re.search(r"Mem:\s+1[01]\d|Mem:\s+12\d", o)),
     "Expect ~119 GiB visible (128 GB unified, shared by CPU and GPU). Much less → close other jobs."),
    ("Free disk for models", "df -h / | tail -1",
     ("ex", "/dev/nvme0n1p2  3.7T  412G  3.1T  12% /"),
     lambda o: _free_gb(o) >= 200,
     "This week downloads ~200 GB of models and containers. Free space or add a disk before Module 05."),
    ("Docker without sudo", "docker version --format '{{.Server.Version}}' && docker ps --format '{{.Names}}' | head -3",
     ("ex", "28.3.3"),
     lambda o: bool(re.search(r"^\d+\.\d+", o.strip())) and "permission denied" not in o.lower(),
     "Add yourself to the docker group, then log out and in: sudo usermod -aG docker $USER && newgrp docker"),
    ("Hugging Face cache", "du -sh ~/.cache/huggingface 2>/dev/null || echo 'empty (no models downloaded yet)'",
     ("ex", "empty (no models downloaded yet)"),
     lambda o: True,
     ""),
]


def _free_gb(df_line: str) -> float:
    """Parse the 'Avail' column of `df -h` (e.g. 3.1T, 850G) into GB."""
    parts = df_line.split()
    if len(parts) < 4:
        return 0.0
    m = re.match(r"([\d.]+)([KMGTP]?)", parts[3])
    if not m:
        return 0.0
    scale = {"": 1e-9, "K": 1e-6, "M": 1e-3, "G": 1, "T": 1e3, "P": 1e6}[m.group(2)]
    return float(m.group(1)) * scale


banner("Lab 01-1 · Spark doctor", "eight read-only checks before a week of fine-tuning and serving")

rows, failures = [], []
for i, (title, cmd, (kind, dry_out), test, fix) in enumerate(CHECKS, 1):
    step(i, title)
    r = sh(cmd, timeout=60, **({"reference": dry_out} if kind == "ref" else {"example": dry_out}))
    passed = bool(test(r.out or ""))
    status = "✓" if passed else "✕"
    if r.source != "live":
        status = "◈ " + r.source                       # a reference is not your machine, so no ✓
    rows.append([title, status, (r.out or "").strip().splitlines()[-1][:46] if r.out.strip() else "—"])
    if r.live and not passed and fix:
        failures.append((title, fix))

print()
table(rows, ["check", "result", "last line of output"])

if failures:
    print()
    for title, fix in failures:
        check(False, "", f"{title}: {fix}")
    result(f"{len(failures)} check(s) need attention — fix them before Module 03.")
elif all(r[1] == "✓" for r in rows):
    result("all eight checks passed — this Spark is ready for the week.")
else:
    result("DRY run: REFERENCE rows are what the playbooks state, EXAMPLE rows are illustrative. "
           "Connect a Spark (🖥 Spark setup) to check yours.")
