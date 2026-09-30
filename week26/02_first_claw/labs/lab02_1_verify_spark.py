#!/usr/bin/env python3
"""Lab 02-1 · Verify the Spark: OS, GB10, Docker, kernel for Landlock, and free disk.

Read-only. Step 1 runs the DGX Spark NemoClaw playbook's own three verification commands. Step 2 adds the
checks the rest of the course depends on: can you run `docker ps` without sudo, is the NVIDIA container runtime
registered, is the kernel ≥ 6.2 (Landlock ABI 3 — Module 03's filesystem layer), how much memory and disk are
free, and is Node.js already there. Step 3 turns it into a pass/fail table.

DRY mode (no Spark): every command is shown, the playbook's expected result is quoted as REFERENCE, the rest are
EXAMPLE shapes — and the table says "◈ DRY — not checked", never ✓. Nothing here needs sudo; if Docker needs a
fix, the lab prints the commands for you to run yourself in the ⌨ terminal.

Run: .venv/bin/python week26/02_first_claw/labs/lab02_1_verify_spark.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from clawkit import banner, laptop, note, ok, result, sh, step, table, warn  # noqa: E402

PLAYBOOK_VERIFY = """head -n 2 /etc/os-release
nvidia-smi
docker info --format '{{.ServerVersion}}'"""
VERIFY_REF = "Expected: Ubuntu 24.04 (or your platform's supported OS), a detected NVIDIA GPU, Docker 28.x+."

MIN_FREE_GB = 200      # course rule of thumb for "large Express models can require hundreds of GB"

banner("Lab 02-1 · verify the Spark", "read-only · the playbook's checks + the ones Modules 03–08 need")

step(1, "the playbook's three checks — OS, GPU, Docker server version")
r1 = sh(PLAYBOOK_VERIFY, reference=VERIFY_REF, timeout=60)
live = r1.live

step(2, "the course's extra checks (one read-only command each)")
checks = {}
checks["docker_ps"] = sh("docker ps", timeout=30, example="CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES")
checks["runtimes"] = sh("docker info --format '{{range $k, $v := .Runtimes}}{{$k}} {{end}}'", timeout=30,
                        example="io.containerd.runc.v2 nvidia runc")
checks["kernel"] = sh("uname -sr", timeout=20, example="Linux 6.X.Y-NNNN-nvidia        ← EXAMPLE: your kernel release")
checks["mem"] = sh("free -g", timeout=20, example="""               total        used        free      shared  buff/cache   available
Mem:            <T>         <U>         <F>         <S>         <B>         <A>
Swap:           <T>         <U>         <F>""")
checks["disk"] = sh('df -BG --output=avail,target "$HOME" | tail -n 1', timeout=20, example="   <N>G /")
checks["node"] = sh('node --version 2>/dev/null || echo "node: not installed"', timeout=20, example="v22.X.Y")


def parse_all() -> list[list[str]]:
    """One row per check: [check, value seen, verdict, why it matters]."""
    out = r1.out
    rows = []
    osv = re.search(r'VERSION="?([^"\n]+)', out) or re.search(r'PRETTY_NAME="?([^"\n]+)', out)
    os_ok = "24.04" in out
    rows.append(["OS", osv.group(1) if osv else "?", "✓" if os_ok else "⚠ not 24.04",
                 "the playbook expects Ubuntu 24.04 / DGX OS"])
    gb10 = "GB10" in out
    rows.append(["GPU", "NVIDIA GB10" if gb10 else "no GB10 in nvidia-smi", "✓" if gb10 else "✕",
                 "a Spark reports a GB10 Grace Blackwell GPU"])
    dv = next((ln.strip() for ln in out.splitlines() if re.fullmatch(r"\d+\.\d+(\.\d+)?\S*", ln.strip())), "")
    d_ok = bool(dv) and int(dv.split(".")[0]) >= 28
    rows.append(["Docker server", dv or "no server version", "✓" if d_ok else "✕", "Docker 28.x+ runs the gateway"])
    ps = checks["docker_ps"]
    denied = "permission denied" in ps.out.lower()
    rows.append(["docker ps (no sudo)", "permission denied" if denied else f"exit {ps.code}",
                 "✓" if ps.ok and not denied else "✕", "NemoClaw drives Docker as your user"])
    rt = checks["runtimes"].out
    rows.append(["NVIDIA runtime", "nvidia registered" if "nvidia" in rt else "no nvidia runtime",
                 "✓" if "nvidia" in rt else "⚠", "vLLM containers need --runtime=nvidia"])
    m = re.match(r"\s*Linux\s+(\d+)\.(\d+)", checks["kernel"].out)
    k_ok = bool(m) and (int(m.group(1)), int(m.group(2))) >= (6, 2)          # Linux only: Landlock is a Linux LSM
    rows.append(["kernel ≥ 6.2", checks["kernel"].out.strip() or "?", "✓" if k_ok else "✕",
                 "Landlock ABI 3 — the filesystem layer (Module 03)"])
    mm = re.search(r"^Mem:\s+(\d+)", checks["mem"].out, re.M)
    rows.append(["memory (free -g)", f"{mm.group(1)} GiB total" if mm else "?", "◆ info",
                 "128 GB unified, shared by CPU and GPU"])
    dm = re.search(r"(\d+)G", checks["disk"].out)
    free_gb = int(dm.group(1)) if dm else 0
    rows.append(["free disk in $HOME", f"{free_gb} GB" if dm else "?",
                 "✓" if free_gb >= MIN_FREE_GB else f"⚠ < {MIN_FREE_GB} GB",
                 "Express models + the vLLM image can need hundreds of GB"])
    nv = checks["node"].out.strip()
    rows.append(["Node.js", nv or "?", "◆ info", "the installer adds Node.js 22.16+ if it is missing"])
    return rows


step(3, "the verdict")
if live:
    rows = parse_all()
    ps_ok = checks["docker_ps"].ok
    table(rows, ["check", "seen", "verdict", "why it matters"])
    must = {"GPU", "Docker server", "kernel ≥ 6.2"}          # the Reef runner's L1.1 pass criterion (spec §6)
    failed = [r[0] for r in rows if r[0] in must and not r[2].startswith("✓")]
    if failed:
        print(f"✕ L1.1 not passed: {', '.join(failed)}")
    else:
        ok("L1.1 passed: GB10 detected, Docker present, kernel ≥ 6.2")
    if not ps_ok and "permission denied" in checks["docker_ps"].out.lower():
        warn("Docker refuses your user — fix it yourself in the ⌨ terminal (the lab never runs sudo):")
        for ln in ("sudo usermod -aG docker $USER && newgrp docker",
                   "sudo nvidia-ctk runtime configure --runtime=docker",
                   "sudo systemctl restart docker",
                   "docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi"):
            print(f"→ {ln}")
    elif not ps_ok:
        warn("`docker ps` failed without 'permission denied' — is the Docker daemon running on the Spark?")
else:
    table([[c, "◈ DRY — not checked", why] for c, why in (
        ("OS", "Ubuntu 24.04 / DGX OS"),
        ("GPU", "NVIDIA GB10 in nvidia-smi"),
        ("Docker server", "28.x+"),
        ("docker ps (no sudo)", "no 'permission denied'"),
        ("NVIDIA runtime", "`nvidia` in docker info's runtimes"),
        ("kernel ≥ 6.2", "Landlock ABI 3 (Module 03)"),
        ("memory (free -g)", "info only"),
        (f"free disk ≥ {MIN_FREE_GB} GB", "course rule of thumb"),
        ("Node.js", "info only — the installer adds it"),
    )], ["check", "verdict", "what LIVE mode looks for"])
    warn("DRY: nothing above ran on a Spark. The EXAMPLE lines are shapes, not your machine.")

step(4, "the same kernel question, asked of THIS laptop (for real)")
r = laptop(["uname", "-sr"], quiet=True, show="uname -sr")
print(f"│ this laptop: {r.out.strip()}")
if not r.out.startswith("Linux"):
    note("not Linux → no Landlock, no seccomp, no network namespaces here. That is why the sandbox runs on the "
         "Spark and this laptop only builds, parses and checks things.")
result("Green on GB10, Docker and kernel ≥ 6.2 means the Spark is ready for the one-command installer (Lab 02-2).")
