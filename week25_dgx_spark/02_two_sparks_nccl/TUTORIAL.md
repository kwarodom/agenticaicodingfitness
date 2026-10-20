# ▶ Spark Lab 02 — Two Sparks, one cluster: QSFP, 200 Gb/s, NCCL

> Part of Week 25 · DGX Spark: fine-tune, serve, and build sandboxed agents. You type the commands, you see the real output. Every lab also runs in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's playbook, or a clearly marked EXAMPLE.

**What you'll actually do**
- Learn what a second Spark buys you (256 GB for big models and sharded training) and what it does not (a 2× faster single conversation).
- Cable two Sparks with one QSFP cable and name the four ConnectX-7 interfaces each Spark shows.
- Give the link IP addresses with netplan: the lab writes and checks the files, you decide when to apply them.
- Set up passwordless SSH between the Sparks and check the link from both ends.
- Build and run NVIDIA's NCCL test, and learn to read **algbw** vs **busbw** against the playbook's pass mark.
- Work out, with arithmetic, what the 200 Gb/s link costs a tensor-parallel model on every token.

**Time** ~50 min · **Difficulty** intermediate · **Hardware** 2 Sparks (or none: DRY mode + arithmetic labs)

**Official playbooks covered:** [Connect Two Sparks](https://build.nvidia.com/spark/connect-two-sparks) · [NCCL](https://build.nvidia.com/spark/nccl) · [Connect Multiple Sparks](https://build.nvidia.com/spark/connect-multiple-sparks) · [Connect Three Sparks](https://build.nvidia.com/spark/connect-three-sparks) · [Multiple Sparks Through a Switch](https://build.nvidia.com/spark/multi-sparks-through-switch)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Module 01 done on **both** Sparks | lab 01-1 all ✓ on each | driver, CUDA and Docker must match |
| Two DGX Sparks on the April 2026 DGX OS release or later | DGX Dashboard → Settings → Updates | NVIDIA Sync's Cluster Assistant requires it |
| One QSFP cable | a supported QSFP112 DAC cable, Ethernet mode | one cable gives the full 200 Gb/s |
| The **same username** on both Sparks | `whoami` on each | `mpirun` and the playbook scripts assume it |
| SSH aliases `spark-a` and `spark-b` on your laptop | `ssh -o BatchMode=yes spark-b true` | the labs drive both Sparks |

In this module **Spark A** is `SPARK_HOST` (the launcher, "node 1" in the playbooks) and **Spark B** is `SPARK_HOST2` ("node 2"). Add Spark B in **🖥 Spark setup**, then ask `sparkkit` where commands would run:

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
.venv/bin/python week25/common/sparkkit.py
```

**Expected output** (captured on this Mac with no Spark configured)

```
━━ sparkkit self-check
   where would commands run, and which endpoints answer?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
◈ DRY · SPARK_HOST not set · commands are shown, not run; outputs are RECORDED, REFERENCE or EXAMPLE (labelled)
◆ on a Spark: False · SPARK_HOST=— · SPARK_HOST2=—
```

With both set and reachable, the second line becomes `▣ LIVE · Spark A · spark-a · Spark B · spark-b`, and two lines `◆ Spark A: ssh` and `◆ Spark B: ssh` appear.

✓ Checkpoint: `ssh -o BatchMode=yes spark-a whoami` and `ssh -o BatchMode=yes spark-b whoami` both answer with the same username, or you have decided to follow along in DRY mode.

## 1 · What two Sparks buy you, and what they don't

One Spark has 128 GB of unified memory and reads it at 273 GB/s. A second Spark, joined by a 200 Gb/s cable, gives you two things:

| You get | Because | Used in |
|---|---|---|
| **256 GB for one model** | tensor parallelism (TP=2) puts half of every layer on each Spark | Module 05: vLLM serves models that do not fit on one Spark |
| **Sharded training** | FSDP splits weights, gradients and optimizer states across both | Module 11: Llama 3.1 70B LoRA in bf16 |
| **Two independent boxes** | run a different model or job on each | Module 08: one gateway in front of both |

What you do **not** get is a conversation that runs twice as fast. Each token still reads its weights from memory at 273 GB/s per Spark. TP=2 halves the bytes each Spark reads, but every layer then waits for two small exchanges over the cable. The cable is about 12× slower than memory:

```text
memory (per Spark)   ████████████████████████████  273 GB/s
QSFP link            ██░░░░░░░░░░░░░░░░░░░░░░░░░░   25 GB/s  (200 Gb/s ÷ 8)
```

So weights never travel over the cable. Only activations do. Lab 04 (Section 7) turns this into numbers per token.

✓ Checkpoint: you can name one model that needs two Sparks (Module 01, lab 02 shows Llama 3.1 405B at NVFP4), and say why a second Spark does not double the speed of one chat.

## 2 · The cable: QSFP, ConnectX-7, and four interface names

Each Spark has a **ConnectX-7** network chip with **two QSFP ports** on the back. Plug one cable between the two Sparks. The playbook's rule: **use the same physical port on both Sparks** to avoid problems in the NCCL test later. "Port 0" is the QSFP port next to the Ethernet port, "port 1" the one further away (Connect Three Sparks, Step 2).

Linux shows each physical port as **two logical interfaces**, so one Spark has four. `ibdev2netdev` maps each RDMA device to its network interface and says which are Up:

```bash
# on: spark
ibdev2netdev
```

**Expected output** (REFERENCE — quoted from the Connect Two Sparks playbook: the cable is in port 1)

```
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
```

How to read the names:

| Name | What it is |
|---|---|
| `enp1s0f1np1`, `enP2p1s0f1np1` | the **two logical interfaces of physical port 1** (`f1`). Port 0 is `enp1s0f0np0` + `enP2p1s0f0np0` |
| `rocep1s0f1`, `roceP2p1s0f1` | the matching **RoCE** (RDMA over Converged Ethernet) devices. NCCL and `ib_write_bw` use these |
| `enP7s7` | the regular Ethernet port: the **management** network you SSH over (Wi-Fi: `wlP9s9`) |

Two facts from the playbooks that trip people up:

- **One cable is enough for full bandwidth.** With two cables, all four interfaces need an IP address to reach full bandwidth.
- **Both logical interfaces of your port need an address, on different subnets.** NVIDIA's RDMA benchmark runs one test per logical interface and adds them: 92.57 + 97.28 = 189.85 Gb/s (lab 03 re-computes this from the playbook's sample).

Lab 01 runs these checks on both Sparks at once. Run it now to see what is Up; run it again after Section 5.

```bash
# on: laptop
.venv/bin/python week25/02_two_sparks_nccl/labs/lab01_link_check.py
```

**Expected output** (DRY mode, captured on this Mac. The lab fed itself the playbook's interface list, address and speed, plus illustrative text for the rest; the run above prints which is which)

```
▣ STEP 3 · compare the two ends
│ node     RDMA device   netdev         QSFP    IPv4               MTU   speed     user    iface list
│ ───────  ────────────  ─────────────  ──────  ─────────────────  ────  ────────  ──────  ──────────
│ Spark A  rocep1s0f1    enp1s0f1np1    port 1  192.168.100.10/24  1500  200 Gb/s  nvidia  reference
│ Spark A  roceP2p1s0f1  enP2p1s0f1np1  port 1  192.168.101.10/24  1500  200 Gb/s  nvidia  reference
│ Spark B  rocep1s0f1    enp1s0f1np1    port 1  192.168.100.11/24  1500  200 Gb/s  nvidia  reference
│ Spark B  roceP2p1s0f1  enP2p1s0f1np1  port 1  192.168.101.11/24  1500  200 Gb/s  nvidia  reference
◈ on this text: same physical port Up on both ends (port 1: enp1s0f1np1 + enP2p1s0f1np1)
◈ on this text: enp1s0f1np1: both ends on one subnet (192.168.100.10/24 ↔ 192.168.100.11/24)
◈ on this text: enP2p1s0f1np1: both ends on one subnet (192.168.101.10/24 ↔ 192.168.101.11/24)
◈ on this text: each logical interface has its own subnet (192.168.100.x, 192.168.101.x)
◈ on this text: MTU is the same on every link interface (1500)
◈ on this text: every link interface negotiated 200000 Mb/s
◈ on this text: same username on both Sparks (nvidia) — mpirun and the helper scripts need it
```

In DRY mode the checks print `◈ on this text:` instead of ✓, because they test playbook text, not your cable. LIVE, you get ✓ or ✕ with the fix on each ✕ line.

✓ Checkpoint: on both Sparks `ibdev2netdev` shows the **same** two interfaces as `(Up)`. If none are Up, reseat the cable, reboot, and check again.

## 3 · The easy path: NVIDIA Sync Cluster Assistant

Since August 2026 every connection playbook starts with the same advice: **use NVIDIA Sync's Cluster Assistant**, and if it reports success, **skip the manual network and SSH steps**. On your laptop, in NVIDIA Sync (Connect Multiple Sparks playbook, "Configure with NVIDIA Sync"):

1. Add each Spark (**Add New** → name or IP, user name, password).
2. **Settings → Cluster Assistant → Add New Cluster**, name it, pick both Sparks.
3. Sync checks SSH, hardware, system software and `sudo`, then detects the cables and shows a **network plan**. Review it and select **Confirm Network Configuration**.
4. It runs a speed test on each link. A link turns green at the **184 Gbit/s lower bound**.
5. It sets up key-based SSH between the Sparks and adds an SSH alias for each.
6. On the success page, **Copy** the network details and save them to a file.

Cluster Assistant configures the network only. It does not install or run NCCL, vLLM or training. Those are the rest of this module and Modules 05 and 11.

> 💡 Even if Cluster Assistant did everything, run lab 01 and lab 03. They show you *what* it configured (interface names, subnets) and measure the link with NCCL.

✓ Checkpoint: either Cluster Assistant shows both links green (skip to Section 6), or you continue with the manual path in Sections 4 and 5.

## 4 · The manual path: IP addresses with netplan

Without Cluster Assistant you give the link interfaces static IPs yourself. The Connect Two Sparks playbook (Step 3, Option 1) writes one netplan file per Spark: port 1's two logical interfaces on two **different** /24 subnets, node 1 ending in `.10`, node 2 in `.11`. (Two interfaces on the same subnet cause routing ambiguity and NCCL failures, the switch playbook warns.)

Lab 02 reads which interfaces are Up on each Spark, writes both files to `week25/02_two_sparks_nccl/.runs/` on your laptop (gitignored), validates them, and in LIVE mode copies each to `~/w25/40-cx7.yaml` on its Spark. It **changes nothing in `/etc`** unless you run it from a shell with `SPARK_APPLY=1`, and even then only with passwordless `sudo -n` and only if the link has no address yet. The Lab Runner's ▶ button never sets `SPARK_APPLY`.

```bash
# on: laptop
.venv/bin/python week25/02_two_sparks_nccl/labs/lab02_netplan_plan.py
```

**Expected output** (DRY mode, captured on this Mac: generated from the playbook's interface list)

```
▣ STEP 3 · the two files
── week25/02_two_sparks_nccl/.runs/40-cx7.spark-a.yaml
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      addresses:
        - 192.168.100.10/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.101.10/24
      dhcp4: no
── week25/02_two_sparks_nccl/.runs/40-cx7.spark-b.yaml
…
✓ both files parse as YAML with network.version 2 and one entry per Up interface
✓ generated from the playbook's interface list, both files match the playbook's Step 3 files byte for byte
```

Apply it yourself, one Spark at a time, in the ⌨ terminal. These are the playbook's commands for **node 1** (Spark A):

```bash
# on: spark
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      addresses:
        - 192.168.100.10/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.101.10/24
      dhcp4: no
EOF
sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```

On **node 2** (Spark B) run the same with `.11` instead of `.10`:

```bash
# on: spark-b
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      addresses:
        - 192.168.100.11/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.101.11/24
      dhcp4: no
EOF
sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```

If your cable is in port 0, use the interface names lab 02 printed instead. Other options from the playbook:

| Option | Command | Survives a reboot? |
|---|---|---|
| netplan file (above) | `sudo netplan apply` | yes |
| temporary IPs | `sudo ip addr add 192.168.100.10/24 dev enp1s0f1np1 && sudo ip link set enp1s0f1np1 up` (and the same for `enP2p1s0f1np1` with `192.168.101.10/24`) | no |
| **rollback** | `sudo rm /etc/netplan/40-cx7.yaml && sudo netplan apply` | — |

> ⚠ The file names only the ConnectX-7 interfaces, not `enP7s7`. `netplan apply` can still interrupt the network for a moment, so keep a second way in (a local console or a second SSH session) the first time.

✓ Checkpoint: `ip addr show enp1s0f1np1` shows `inet 192.168.100.10/24` on Spark A and `inet 192.168.100.11/24` on Spark B.

## 5 · Passwordless SSH between the Sparks, and a clean link

`mpirun` (Section 6) and multi-node training (Module 11) start processes on Spark B **from Spark A**, so Spark A must reach Spark B over SSH without a password. The playbook's automatic option discovers the other Spark with mDNS (`avahi-browse`, from `avahi-utils`) and shares one key:

```bash
# on: spark
mkdir -p ~/.ssh && chmod 700 ~/.ssh
curl -fsSL -o discover-sparks https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-connect-two-sparks/assets/discover-sparks
chmod +x discover-sparks
bash ./discover-sparks
```

**Expected output** (REFERENCE — quoted from the playbook; it asks for your password once per node)

```
Found: 192.168.100.10 (node-1.local)
Found: 192.168.100.11 (node-2.local)

Setting up shared SSH access across all nodes...
You may be prompted for your password on each node.

Shared SSH setup complete!
All nodes can now SSH to each other using the shared key (id_ed25519_shared).
```

Run `mkdir -p ~/.ssh && chmod 700 ~/.ssh` on Spark B too; the script fails if `~/.ssh` is missing. The manual option, on both Sparks: `ssh-copy-id -i ~/.ssh/id_ed25519.pub <username>@192.168.100.10` and `…@192.168.100.11`.

**MTU: leave it alone, but check both ends match.** The Spark playbooks keep the default MTU (their `ip addr` sample shows `mtu 1500`, and the multi-Spark playbook says to keep the default for switch links). The MTU 9000 you may have seen belongs to the DGX **Station** playbook (ConnectX-8), not the Spark. RDMA traffic negotiates its own path MTU: the RDMA sample prints `Mtu : 1024[B]`. A course extra that proves 1500 works end to end (1472 bytes of data + 28 bytes of headers, "do not fragment"):

```bash
# on: spark
ping -c 3 -M do -s 1472 -I enp1s0f1np1 192.168.100.11
```

Now re-run lab 01. Its step 4 pings Spark B across the cable and runs `ssh -o BatchMode=yes 192.168.100.11 hostname` from Spark A.

```bash
# on: laptop
.venv/bin/python week25/02_two_sparks_nccl/labs/lab01_link_check.py
```

**Expected output** (DRY mode, captured on this Mac; ping and ssh answers are EXAMPLE shapes)

```
▣ STEP 4 · across the cable: ping, then passwordless SSH from Spark A to Spark B
$ ping -c 3 -W 2 -I enp1s0f1np1 192.168.100.11   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
…
3 packets transmitted, 3 received, 0% packet loss
◈ on this text: Spark A reaches 192.168.100.11 over the QSFP link
$ ssh -o BatchMode=yes -o ConnectTimeout=5 192.168.100.11 hostname   [DRY]
◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):
spark-b
◈ on this text: passwordless SSH A → B over the link works (answered: spark-b)
```

✓ Checkpoint: LIVE, every line of lab 01 is ✓: same port, one subnet per interface, same MTU, 200000 Mb/s, same user, ping and SSH across the cable.

## 6 · NCCL: build it, run it, read algbw vs busbw

**NCCL** (NVIDIA Collective Communication Library) is what PyTorch, vLLM and TensorRT-LLM call to move tensors between GPUs, including across your cable. The NCCL playbook builds NCCL **v2.30.7-1** for Blackwell (`sm_121`) and NVIDIA's `nccl-tests` benchmark on **both** Sparks. It needs `sudo` for one package, so run it yourself in the ⌨ terminal, first on Spark A, then on Spark B:

```bash
# on: spark
sudo apt-get update && sudo apt-get install -y libopenmpi-dev
git clone -b v2.30.7-1 https://github.com/NVIDIA/nccl.git ~/nccl/
cd ~/nccl/
make -j src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"
export CUDA_HOME="/usr/local/cuda"
export MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
export NCCL_HOME="$HOME/nccl/build/"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH"
git clone https://github.com/NVIDIA/nccl-tests.git ~/nccl-tests/
cd ~/nccl-tests/
make MPI=1
```

(The playbook's quick start does both Sparks from Spark A: `bash setup.sh <NODE_2_IP>`, then `bash launch.sh --topology direct <NODE_1_IP> <NODE_2_IP>`.)

The test is started **once, on Spark A**. `mpirun` SSHes into Spark B and starts the second rank there. Note what each flag points at:

| Flag | Value | Why |
|---|---|---|
| `-H <A>:1,<B>:1` | the **management** IPs (`ip addr show enP7s7`) | where mpirun SSHes to; one process per Spark |
| `NCCL_SOCKET_IFNAME`, `UCX_NET_DEVICES`, `OMPI_MCA_btl_tcp_if_include` | `enP7s7` | the interface for setup traffic. On Wi-Fi use `wlP9s9` on **every** node |
| `-b 16G -e 16G -f 2` | one 16 GB message | big enough to fill the link |

The bulk data goes over the RoCE devices, which NCCL finds by itself. To see which ones it picked, add `-x NCCL_DEBUG=INFO` and look for lines with `NET/IB` (course tip).

Lab 03 checks the build on both Sparks, finds both management IPs, runs the playbook's 16 GB `all_gather_perf` command, and reads the result:

```bash
# on: laptop
.venv/bin/python week25/02_two_sparks_nccl/labs/lab03_nccl_bench.py
```

**Expected output** (DRY mode, captured on this Mac. The playbook prints no NCCL output, so the table is an EXAMPLE with numbers chosen to make the arithmetic visible, not a measurement)

```
▣ STEP 4 · read the table
│ size    time      algbw       busbw       busbw × 8
│ ──────  ────────  ──────────  ──────────  ─────────
│ 16 GiB  390.5 ms  44.00 GB/s  22.00 GB/s  176 Gb/s
◆ algbw = size ÷ time = 17.18 GB ÷ 0.390 s = 44.00 GB/s
◆ busbw = algbw × (n−1)/n = 44.00 × 0.5 = 22.00 GB/s  (nccl-tests printed 22.00)
│ busbw     █████████████████████████░░░  22.00 GB/s
│ pass mark ████████████████████████░░░░  21.875 GB/s  (playbook script, direct link)
│ line rate ████████████████████████████  25.00 GB/s  (200 Gb/s ÷ 8)
◈ on this EXAMPLE text the verdict would be pass — not your link

▣ STEP 5 · parser self-test on the playbook's published RDMA sample (REFERENCE, runs everywhere)
│ RDMA write test          BW average
│ ───────────────────────  ────────────────────────
│ client 1 (rocep1s0f0)    92.57 Gb/s
│ client 2 (roceP2p1s0f0)  97.28 Gb/s
│ total                    189.85 Gb/s = 23.73 GB/s
✓ parser total 189.85 Gb/s matches the playbook's stated 189.85 Gbps
```

**algbw vs busbw.** `algbw` is simply *message size ÷ time*. But collectives move different amounts of data per GPU: in an all-gather with *n* ranks each GPU receives only (n−1)/n of the result from the others, in an all-reduce it sends and receives 2(n−1)/n of it. nccl-tests multiplies algbw by that factor to get **busbw**, the rate the *wire* actually carried. That makes busbw comparable across operations and GPU counts, and comparable with the cable's 25 GB/s. For two Sparks:

| Operation | busbw factor (n = 2) | Used by |
|---|---|---|
| `all_gather` (the playbook's test) | (n−1)/n = 0.5 | FSDP gathering weight shards (Module 11) |
| `all_reduce` (`--op all_reduce`) | 2(n−1)/n = 1 | tensor parallel (Module 05), gradient sync |

**Did it pass?** NVIDIA's own cluster script (`spark_cluster_setup.py` in the Connect Multiple Sparks assets) reads the `# Avg bus bandwidth` line and warns below **21.875 GB/s (175 Gbps)** for a direct or switch link, and below **10 GB/s (80 Gbps)** for a three-Spark ring. Lab 03 uses the same line and the same numbers.

For the raw fabric without NCCL, the benchmarking guide runs perftest's `ib_write_bw`, one server per logical interface on Spark A and one client each on Spark B (`sudo apt install perftest` on both first). Lab 03 runs a finite version with `--rdma`; the playbook's commands, with its example IPs, are:

```bash
# on: spark
ib_write_bw -d rocep1s0f0 -i 1 -p 12000 -F --report_gbits --run_infinitely
```

```bash
# on: spark-b
ib_write_bw -d rocep1s0f0 -i 1 -p 12000 -F --report_gbits 192.168.200.12 --run_infinitely
```

Repeat with `roceP2p1s0f0`, port `12001` and the second subnet's address in two more terminals, then add the two `BW average` columns. Use your own Up devices and Spark A's addresses.

✓ Checkpoint: LIVE, lab 03 prints `Avg bus bandwidth … ≥ 21.875 GB/s`. Write your busbw down; lab 04 and Module 11 use it.

## 7 · What the link costs a tensor-parallel model

With TP=2, each layer ends with **two all-reduces** of one hidden-state vector per token (bf16: hidden size × 2 bytes). That is tiny: 16 KiB per all-reduce for a 70B model. The weight read shrinks to half per Spark. What adds up is the fixed **latency** α of each all-reduce, paid 2 × layers times per token. The playbooks publish no latency number, so lab 04 treats α as an assumption and sweeps it:

```bash
# on: laptop
.venv/bin/python week25/02_two_sparks_nccl/labs/lab04_tp_link_cost.py
```

**Expected output** (arithmetic, captured on this Mac; busbw = the playbook's 21.875 GB/s pass mark, weights sized for NVFP4)

```
▣ STEP 2 · bytes that cross the link per decoded token (batch 1, bf16 activations)
│ model                  layers  hidden  message   all-reduces  per token  wire time
│ ─────────────────────  ──────  ──────  ────────  ───────────  ─────────  ─────────
│ Llama 3.1 8B           32      4096    8.0 KiB   64           0.52 MB    24 µs
│ Llama 3.3 70B          80      8192    16.0 KiB  160          2.62 MB    120 µs
│ gpt-oss-120b (MoE)     36      2880    5.6 KiB   72           0.41 MB    19 µs
│ Qwen3 235B-A22B (MoE)  94      4096    8.0 KiB   188          1.54 MB    70 µs
│ Llama 3.1 405B         126     16384   32.0 KiB  252          8.26 MB    377 µs

◆ α = 100 µs per all-reduce (assumed)
│ model                  1 Spark         TP=2       of which link  speed-up        TP=2 tok/s ceiling
│ ─────────────────────  ──────────────  ─────────  ─────────────  ──────────────  ──────────────────
│ Llama 3.1 8B             16.5 ms         14.7 ms    6.4 ms       1.12×             68.2
│ Llama 3.3 70B           145.5 ms         88.9 ms   16.1 ms       1.64×             11.3
│ gpt-oss-120b (MoE)       10.5 ms         12.5 ms    7.2 ms       0.84×             80.2
│ Qwen3 235B-A22B (MoE)     — (too big)    41.5 ms   18.9 ms       fits only on 2    24.1
│ Llama 3.1 405B            — (too big)   442.8 ms   25.6 ms       fits only on 2     2.3

▣ STEP 4 · prefill is different: a 4,096-token prompt sends 4,096× the bytes
│ Llama 3.3 70B           10.74 GB          0.49 s
═ Two Sparks buy CAPACITY first: 256 GB holds Llama 405B or Qwen3 235B at 4-bit. …
```

(The full output also shows α = 10 µs and 30 µs. At 10 µs the 70B model reaches 1.95×; the 8B only 1.85×.)

Three lessons:

1. **Decode bytes are small, latency is not.** A token of Llama 3.3 70B sends 2.6 MB across the cable, about 120 µs of wire time. 160 round trips of latency can cost more than that.
2. **Big dense models gain, small or sparse ones may lose.** gpt-oss-120b already reads only ~5B weights per token, so with a slow α, TP=2 is *slower* than one Spark (0.84×). Serve it on one Spark and use the second for something else.
3. **Long prompts load the cable.** Prefilling 4,096 tokens of a 70B model sends ~10.7 GB, about half a second at busbw.

These are upper bounds from arithmetic, not benchmarks: they ignore compute, kernel launches and overlap. Measure α with the playbook's latency test (`ib_write_lat -d rocep1s0f0 -i 1 -p 13000 -F` on Spark A, the same plus Spark A's IP on Spark B), then rerun with `--alpha-us <yours> --busbw <lab 03's value>`. Module 05 measures real vLLM tok/s on one and two Sparks.

✓ Checkpoint: you can explain why TP=2 helps Llama 3.3 70B more than gpt-oss-120b, and name the one number lab 04 has to assume.

## 8 · Three Sparks, four Sparks: ring and switch

The same ideas scale to more Sparks. The Connect Multiple Sparks playbook allows three layouts, and says not to mix direct and switch links:

| Sparks | Layout | Cables | Notes from the playbooks |
|---|---|---|---|
| 2 | **Direct** | 1 | this module |
| 3 | **Direct ring** | 3: A→B, B→C, C→A | each Spark uses **both** QSFP ports (Port0 → next node's Port1), so all four interfaces get an IP on their own subnet |
| 2–4 (more manually) | **Switch** | 1 per Spark | every Spark-facing port at 200 Gbit/s, all on one Layer 2 bridge, default MTU; four Sparks need a switch |

What changes in practice:

- **Ring NCCL test:** the playbook adds `-x NCCL_IB_SUBNET_AWARE_ROUTING=1 -x NCCL_NET_PLUGIN=none` to `mpirun`, and NVIDIA's script expects only **10 GB/s** busbw (each hop is a separate direct link).
- **Switch:** check `ethtool enp1s0f1np1 | grep Speed` shows `Speed: 200000Mb/s` on every node. Auto-negotiation may settle at `100000Mb/s`; then set 200G on the switch port and disable auto-negotiation. The switch can hand out IPs by DHCP (`dhcp4: true` in netplan), or you assign `192.168.100.1–4` / `192.168.101.1–4` by hand.
- **Cluster Assistant** handles two to four Sparks in any of these layouts. Beyond four, use the switch playbook's manual steps.

✓ Checkpoint: you can say how many cables a three-Spark ring needs, and which extra NCCL flags it uses.

## Labs — run them here

**labs/lab01_link_check.py** — Check the QSFP link on both Sparks: Up interfaces, addresses, MTU, speed, user, then ping and passwordless SSH across the cable.

**labs/lab02_netplan_plan.py** — Write and validate the netplan file for each Spark from its Up interfaces; apply only with SPARK_APPLY=1 from a shell.

**labs/lab03_nccl_bench.py** — Run the playbook's two-Spark NCCL test, parse algbw and busbw, and compare with NVIDIA's pass mark.

**labs/lab04_tp_link_cost.py** — Arithmetic: bytes, wire time and latency the link adds to every tensor-parallel token, for five models.

Labs 01–03 run LIVE on two Sparks or DRY. Lab 04 is arithmetic and runs anywhere. Lab 02 never touches `/etc` from the ▶ button, and lab 03's parser self-test runs on NVIDIA's published sample in every mode.

## Try it yourself

**Exercise 02 — read the link.** Open `week25/02_two_sparks_nccl/exercises/ex02_read_the_link.py`. It has three `TODO`s:

1. `up_interfaces(text)`: the netdev names that `ibdev2netdev` reports `(Up)`.
2. `bus_factor(op, n)`: the factor that turns algbw into busbw.
3. `verdict(busbw, topology)`: `"pass"` or `"low"` against NVIDIA's pass marks.

The checker is offline and free. It uses the real `ibdev2netdev` samples from three playbooks.

```bash
# on: laptop
.venv/bin/python week25/02_two_sparks_nccl/exercises/ex02_read_the_link.py
```

**Expected output** (once all three TODOs are done, captured on this Mac)

```
✓ up_interfaces: two-Spark sample → port 1 (2 netdevs) · NCCL sample → port 0 · ring sample → all 4
✓ bus_factor: all_reduce n=2 → 1 · all_gather n=2 → 0.5 · all_reduce n=4 → 1.5 · broadcast → 1
✓ verdict: 22 direct → pass · 21 direct → low · 12 ring → pass · 9.5 ring → low · 21.875 switch → pass

▣ your functions, applied to three practice runs (EXAMPLE numbers, not measurements)
│ all_gather n=2 direct algbw  44.6 GB/s → busbw 22.30 GB/s (178.4 Gb/s) → pass
│ all_reduce n=2 direct algbw  19.0 GB/s → busbw 19.00 GB/s (152.0 Gb/s) → low
│ all_gather n=3 ring   algbw  16.2 GB/s → busbw 10.80 GB/s ( 86.4 Gb/s) → pass
```

<details><summary>Hint — why compare busbw, not algbw, with the cable?</summary>

In a two-rank all-gather each Spark already holds half of the result, so only the other half crosses the cable. algbw counts the whole message and looks twice as fast as the wire. busbw = algbw × (n−1)/n counts only what travelled, which is what the 25 GB/s cable limits.

</details>

<details><summary>Stretch — plan a three-Spark ring</summary>

Take the Connect Three Sparks playbook's netplan: node 1 has `192.168.0.1`, `192.168.1.1`, `192.168.2.1`, `192.168.3.1`. Write down which subnet each cable carries, and which interface on node 2 and node 3 shares a subnet with each of node 1's. Why does the ring need six subnets for three cables?

</details>

✓ Checkpoint: all three checker lines are ✓.

## Troubleshooting

| Symptom | Fix |
|---|---|
| No interface shows `(Up)` in `ibdev2netdev` | Reseat the QSFP cable in the **same** port on both Sparks, reboot both, check again |
| "Network unreachable" | The netplan file is missing or not applied. Check `ip addr show enp1s0f1np1` and run `sudo netplan apply` |
| Both ends Up but ping across the link fails | The two ends are on different subnets, or only one logical interface has an address. Re-run lab 01 |
| `discover-sparks` fails writing keys | `mkdir -p ~/.ssh && chmod 700 ~/.ssh` on both Sparks, then retry |
| `discover-sparks`: `avahi-browse not found` | install `avahi-utils` on both Sparks |
| `mpirun` hangs or times out | SSH between the Sparks asks for a password. Test `ssh <node-2 management IP>` from Spark A, then `mpirun -np 2 -H <A>:1,<B>:1 hostname` |
| NCCL test: `libnccl.so.2: cannot open shared object file` | NCCL is not built on every node, or `LD_LIBRARY_PATH` was not exported before `mpirun` |
| busbw far below 21.875 GB/s | Stop other GPU jobs and rerun. Check both logical interfaces have addresses, and the speed is `200000Mb/s` |
| Switch link at `100000Mb/s` | Set the switch port to 200G and disable auto-negotiation (switch maker's guide) |
| Cluster Assistant: software check fails | Update both Sparks to the April 2026 release or later |

## Next

Continue to [Lab 03 — Ollama + Open WebUI](../03_ollama_open_webui/TUTORIAL.md): your first model server on the Spark, with a chat UI in the browser.
