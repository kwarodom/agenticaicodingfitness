# Week 25: add a second DGX Spark (Spark B)

This is the instructor checklist for turning one classroom Spark into a **two-Spark cluster**. It is needed for
Module 02 (QSFP + NCCL), Module 05 §7 (vLLM tensor parallel, TP=2), Module 11 lab 04 (FSDP across both Sparks),
and the Module 20 capstone (router fine-tune on Spark B). Everything else in the course runs on Spark A alone.

```text
            tailnet / office LAN (ssh from laptops, HTTP to engines)
   laptops ───────────────┬──────────────────────────────┐
                          ▼                              ▼
                ┌──────────────────┐   QSFP cable   ┌──────────────────┐
                │ Spark A (node 1) │◄══════════════►│ Spark B (node 2) │
                │ spark-3b82       │   200 Gb/s     │ spark-b3b6       │
                │ 192.168.100.10   │   RoCE         │ 192.168.100.11   │
                │ 192.168.101.10   │                │ 192.168.101.11   │
                └──────────────────┘                └──────────────────┘
     SPARK_HOST = sparklab@<A>          SPARK_HOST2 = sparklab@<B>
```

Most steps below come from Module 02's `TUTORIAL.md`, which quotes NVIDIA's Connect Two Sparks and NCCL
playbooks. Read §4–§6 there for the reasons behind each step.

**This classroom, as verified on 2026-10-01** (the generic steps below use the playbook's example values):

| | Spark A | Spark B |
|---|---|---|
| Hostname · tailnet | `spark-3b82` · `spark-3b82.<tailnet>.ts.net` | `spark-b3b6` · `spark-b3b6.<tailnet>.ts.net` |
| QSFP cable | **port 0**: `enp1s0f0np0` + `enP2p1s0f0np0` | port 0 |
| Link IPs (set by NVIDIA Sync, MTU 9000) | 192.168.100.57 · 192.168.101.57 | 192.168.100.96 · 192.168.101.96 |
| Day job (`lab_mode.sh` stops it) | nemotron-lightning, supabase-kong, alto-backend | the `altoace` stack + litellm + ollama-bridge (~94 GB) |
| Measured | raw RDMA 196 Gb/s · NCCL 16 GB all_gather busbw ~20.7 GB/s · vLLM TP=2 Llama 3.3 70B 2.5 tok/s | |

The labs find the cabled port by themselves. The playbook's commands and `docker-compose.yml` assume port 1
(`enp1s0f1np1`): with port 0, use the `…f0np0` names wherever they appear.

---

## What to have ready

| Item | Why |
|---|---|
| A **QSFP112 DAC cable** (Ethernet mode), one is enough | 200 Gb/s link between the Sparks. Use the **same physical port on both** (port 0 is next to the RJ45 Ethernet port) |
| **Ethernet (RJ45) for both Sparks**, not Wi-Fi | NCCL/mpirun setup traffic and every student's SSH go over the management network. The office Wi-Fi DNS drops lookups |
| Both Sparks on **DGX OS April 2026 or later** | NVIDIA Sync's Cluster Assistant requires it (DGX Dashboard → Settings → Updates) |
| **The same usernames on both Sparks** | `mpirun` and the playbook scripts assume one username. We use `altoaidev` (admin, instructor tests) and `sparklab` (class) on **both** |
| HF account with **Meta Llama 3.3 license accepted** | Module 11 lab 04 fine-tunes `meta-llama/Llama-3.3-70B-Instruct` (gated, ~140 GB) on both Sparks |
| A monitor + keyboard on Spark B the first time | a fallback way in while networking changes |

---

## Step 1: Spark B basics (on Spark B's console)

1. Finish the DGX OS first-boot wizard and create the admin user **`altoaidev`**, the same name as on Spark A.
   Apply all DGX OS updates, then reboot.
2. Plug in **Ethernet** and note the management IP: `ip -4 addr show enP7s7`.
3. Join the tailnet and tag it like Spark A:
   ```bash
   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up --advertise-tags=tag:dgx-spark
   tailscale status --json | python3 -c 'import sys,json;print(json.load(sys.stdin)["Self"]["DNSName"])'
   ```
   Write down Spark B's tailnet name (e.g. `spark-xxxx.<tailnet>.ts.net`). The access-control grant for
   `tag:dgx-spark` covers it automatically.
4. Check the GPU works in Docker: `docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L` shows `NVIDIA GB10`.

## Step 2: the classroom account and downloads on Spark B

Run the same scripts as on Spark A:

```bash
git clone https://github.com/kwarodom/agenticaicodingfitness.git ~/Documents/agenticaicodingfitness
cd ~/Documents/agenticaicodingfitness
sudo week25/spark_host/setup_sparklab_user.sh
sudo -u sparklab -H bash -lc '~/.local/bin/hf auth login'     # token with Llama 3.3 access
week25/spark_host/ngc_login.sh
week25/spark_host/prefetch.sh pull
week25/spark_host/prefetch.sh hf
sudo -u sparklab -H bash -lc '~/.local/bin/hf download meta-llama/Llama-3.3-70B-Instruct'   # Module 11 lab 04, ~140 GB
```

Spark B has its own day job (the `altoace` stack, ~94 GB). `lab_mode.sh` knows it: set `SPARK_B_ADMIN` on
Spark A (Step 7) and `lab_mode.sh on|off|status` there covers both Sparks.

**Copying a model to the other Spark instead of downloading it twice:** current `hf` versions keep the files in
a shared `~/.cache/huggingface/hub/blobs/` store, and each `models--…/snapshots/` entry only links into it.
Copy both, or the copy has dangling links. As `sparklab` on the receiving Spark, over the cable:

```bash
rsync -a <other-spark-link-ip>:.cache/huggingface/hub/blobs/ ~/.cache/huggingface/hub/blobs/
rsync -a <other-spark-link-ip>:.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/ \
  ~/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/
```

132 GB took under 5 minutes Spark B → A.

**Student keys:** every laptop key that is on Spark A must also be on Spark B:

```bash
sudo week25/spark_host/add_key.sh alice.pub bob.pub …        # same files you used on Spark A
```

## Step 3: cable and link addresses

Plug the QSFP cable into the **same port number** on both Sparks. On each, check that both logical interfaces
of that port show `(Up)`:

```bash
ibdev2netdev
# rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
# roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)     ← cable in port 1
```

Then give the link its addresses, using **one** of these paths:

**Easy path: NVIDIA Sync Cluster Assistant** (on your laptop). Add both Sparks → Settings → Cluster
Assistant → Add New Cluster → pick both. Confirm the network plan. Both links should turn green (≥ 184 Gbit/s).
It also sets up SSH between the Sparks for the user you give it. Copy the network details it shows and keep them.

**Manual path: netplan** (Module 02 §4). On Spark A (node 1, `.10`):

```bash
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      addresses: [192.168.100.10/24]
      dhcp4: no
    enP2p1s0f1np1:
      addresses: [192.168.101.10/24]
      dhcp4: no
EOF
sudo chmod 600 /etc/netplan/40-cx7.yaml && sudo netplan apply
```

On Spark B (node 2), run the same with `.11`. If the cable is in **port 0**, use the `…f0np0` names that
`ibdev2netdev` printed. `.venv/bin/python week25/02_two_sparks_nccl/labs/lab02_netplan_plan.py` generates both
files for you. To undo: `sudo rm /etc/netplan/40-cx7.yaml && sudo netplan apply`.

> ⚠ Keep the console or a second SSH session open the first time. `netplan apply` can blip the network for a moment.

Check it from Spark A:

```bash
ping -c 3 -M do -s 1472 -I enp1s0f1np1 192.168.100.11    # leave MTU at the default 1500
```

## Step 4: passwordless SSH from Spark A to Spark B, for both users

`mpirun`, FSDP and the capstone start processes on Spark B **from Spark A**. Run this once for each user on
Spark A:

```bash
# on Spark A, as altoaidev
ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
ssh-copy-id altoaidev@192.168.100.11                # asks for B's altoaidev password once
ssh -o BatchMode=yes 192.168.100.11 hostname        # → Spark B's hostname, no prompt

# on Spark A, for sparklab (key-only account, so add the key with add_key.sh on B)
sudo -u sparklab -H bash -lc 'ls ~/.ssh/id_ed25519 || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519'
sudo cat /home/sparklab/.ssh/id_ed25519.pub         # copy this line …
# … then on Spark B:  sudo week25/spark_host/add_key.sh "<that line>"
sudo -u sparklab -H ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new 192.168.100.11 hostname
```

The Connect Two Sparks playbook also has an automatic script (`discover-sparks`, Module 02 §5) that
does the same over mDNS.

## Step 5: build NCCL on both Sparks (Module 02 §6)

This needs sudo for one package. Build it **as the user who will run Module 02 live**: `sparklab` for the
class, `altoaidev` for instructor testing. Run on **both** Sparks:

```bash
sudo apt-get update && sudo apt-get install -y libopenmpi-dev perftest
sudo -u sparklab -H bash -lc '
  git clone -b v2.30.7-1 https://github.com/NVIDIA/nccl.git ~/nccl/ && cd ~/nccl/ &&
  make -j src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121" &&
  export CUDA_HOME=/usr/local/cuda MPI_HOME=/usr/lib/aarch64-linux-gnu/openmpi NCCL_HOME=$HOME/nccl/build/ &&
  export LD_LIBRARY_PATH=$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH &&
  git clone https://github.com/NVIDIA/nccl-tests.git ~/nccl-tests/ && cd ~/nccl-tests/ && git checkout -q b4d5bee &&
  make MPI=1'
ls ~sparklab/nccl-tests/build/all_gather_perf ~sparklab/nccl-tests/build/all_reduce_perf
```

`b4d5bee` (nccl-tests 2.20.0) pins both Sparks to the same build: `mpirun` runs the binary on both. The `make`
ends with an error about `ginGetLatency_ping_perf` (`undefined reference to MPI::Win::Free()`). It is harmless:
that is an extra device-API test, and the `*_perf` binaries the labs use are already built. The `ls` is the check.

## Step 6: point the runners at both Sparks

In **🖥 Spark setup** (instructor and students):

| Field | Value |
|---|---|
| Spark A (ssh target) | `sparklab@<spark-a tailnet name or office IP>` (unchanged) |
| **Spark B (ssh target)** | `sparklab@<spark-b tailnet name or office IP>` |
| HTTP hosts | leave empty |

The status bar now shows **Spark A** and **Spark B** chips, both green. Hand out Spark B's address together with
`STUDENT_SETUP.md`: it's the same steps, plus one more field.

## Step 7: verify, in this order

```bash
.venv/bin/python week25/common/sparkkit.py                              # ▣ LIVE · Spark A … · Spark B …
.venv/bin/python week25/02_two_sparks_nccl/labs/lab01_link_check.py     # every line ✓ (port, subnets, MTU, 200 Gb/s, same user, ping + ssh)
.venv/bin/python week25/02_two_sparks_nccl/labs/lab03_nccl_bench.py     # Avg bus bandwidth ≥ 21.875 GB/s
```

Then the two-Spark labs, with lab mode on **on both Sparks** first. On Spark A, once:
`echo 'SPARK_B_ADMIN=altoaidev@<spark-b>' >> week25/.env.local`. After that, `week25/spark_host/lab_mode.sh on`
on Spark A also stops B's day job, and `off` restarts both. Spark B runs its own copy of the script, so keep its
repo current (`git pull` on B).

| Module | What it proves |
|---|---|
| 05 §7: vLLM TP=2 | one model across 256 GB |
| 11 lab 04: FSDP | Llama 3.3 70B LoRA sharded over both Sparks |
| 20 lab 02 `--launch` | router on Spark B, brain + gateway on Spark A |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ibdev2netdev` shows nothing `(Up)` | reseat the cable, check it's the same port number on both Sparks, reboot both |
| lab 01: "same user" ✕ | the ssh users differ. Use `sparklab@` for both Spark A and Spark B in 🖥 Spark setup |
| lab 01: ssh A→B ✕ | Step 4 for the user the runner uses (`sparklab`) |
| NCCL busbw far below 21.875 GB/s | both logical interfaces need an IP on **different** subnets. Cable in the same port on both. `-x NCCL_DEBUG=INFO` should show `NET/IB` |
| mpirun hangs | the management interface must match on both nodes (`enP7s7` on Ethernet). Mixing Wi-Fi and Ethernet breaks it |
