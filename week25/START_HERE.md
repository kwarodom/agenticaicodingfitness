# Week 25 · Instructor start here: two DGX Sparks, ready and running

Everything you need to get both Sparks ready and teach the course, in order. Details live in three other files.
Open them when a step points there:

| File | What it's for |
|---|---|
| [`spark_host/README.md`](spark_host/README.md) | one-time setup of a Spark for class: the `sparklab` account, keys, tailnet access, downloads |
| [`spark_host/TWO_SPARKS.md`](spark_host/TWO_SPARKS.md) | adding Spark B: cable, link IPs, SSH between the Sparks, NCCL. Has this classroom's verified facts |
| `spark_host/STUDENT_SETUP.md` | the handout for students (gitignored: it has the tailnet name and office IP) |

```text
 students' laptops ─ Spark Lab Runner ─ ssh sparklab@ ─┬─► Spark A  spark-3b82  (SPARK_HOST)
 (Tailscale or office LAN)                              └─► Spark B  spark-b3b6  (SPARK_HOST2)
                                    Spark A ◄══ QSFP 200 Gb/s, port 0 ══► Spark B
```

All commands run **on Spark A as `altoaidev`**, from the repo root (`cd ~/Documents/agenticaicodingfitness`),
unless a step says otherwise.

---

## 1 · Is everything ready? (any time, 5 seconds, read-only)

```bash
week25/spark_host/ready_check.sh
```

It checks both Sparks as `sparklab`, the account students use: SSH, student keys, Hugging Face and nvcr.io
logins, tailnet tags, free memory, container images, the cached models, the capstone's `hotel-ft` adapter,
NCCL on both, and SSH from Spark A to Spark B over the cable. Fix every **✕**. A **⚠** means a single lab
will be affected, and the line says how to fix it.

The one-time setup behind those checks is already done for this classroom (2026-10-01). To rebuild a Spark
from scratch, follow `spark_host/README.md`, then `spark_host/TWO_SPARKS.md`.

## 2 · Before the first class (once per cohort)

1. **Tailnet.** In the Tailscale admin console, invite each student and add them to `group:w25-students`. Tag
   **both** Sparks `tag:dgx-spark` (Machines → ⋯ → Edit ACL tags). The policy grant in `spark_host/README.md`
   only reaches tagged machines. Also keep a grant for `autogroup:admin` to the tag, so your own access survives
   the tagging.
2. **Student keys, on both Sparks.** Collect each laptop's `~/.ssh/id_ed25519.pub`, then:
   ```bash
   sudo week25/spark_host/add_key.sh alice.pub bob.pub …                                  # Spark A
   ssh -t altoaidev@<spark-b> 'cd ~/Documents/agenticaicodingfitness && sudo week25/spark_host/add_key.sh "<one key line>"'   # Spark B, per key
   ```
   (Or copy the `.pub` files to Spark B and run the same `add_key.sh` there.)
3. **Send `spark_host/STUDENT_SETUP.md`** to the students. First compare its office IPs with the
   `Office LAN` lines of `week25/spark_host/lab_mode.sh status`: they are DHCP leases and can change. It covers installing Tailscale, making a key, the
   runner, and filling in **Spark A** and **Spark B** in 🖥 Spark setup.
4. Run `week25/spark_host/ready_check.sh` again. The keys line should now count your students.

## 3 · Class morning (10 minutes)

```bash
week25/spark_host/lab_mode.sh on        # BOTH Sparks: stops A's day job and B's altoace stack
```

Check its two status blocks: about **110 GB available** on each Spark, ports 8000 / 8001 / 4000 free on A, and
8000 free on B. (`SPARK_B_ADMIN` in `week25/.env.local` is what makes it reach Spark B.)

Start your runner. Pick one:

```bash
# (a) on Spark A: open it from your laptop over the tailnet (it asks for SPARK_GUIDE_PASSWORD)
.venv/bin/python week25/00_spark_lab_runner/tutorial_server.py        # → http://spark-3b82.<tailnet>.ts.net:8125
# (b) on your laptop: the same command, after the student setup steps
```

On Spark A, `week25/.env.local` already sets `SPARK_HOST=sparklab@…` and `SPARK_HOST2=sparklab@…`. The runner
therefore drives both Sparks as `sparklab`, the way students' runners do. Both the **Spark A** and **Spark B**
chips should be green.

Smoke test, 1 minute: run **Module 01 lab 01** (Spark doctor) and **Module 02 lab 01** (link check, every line ✓).

## 4 · Teaching the course: who runs what

The Sparks are shared. Anything that takes a fixed port or most of the memory is run **by you, live**; students
watch, then use it (⚡ Ask the Spark, the client labs) or study it in 📄 Dry run. The rest everyone runs.

| Module | Everyone runs | You run live (one at a time) |
|---|---|---|
| 01 Meet your Spark | all labs | — |
| 02 Two Sparks, NCCL | 02-1 link check, 02-2 netplan plan, 02-4 | **02-3** NCCL benchmark, 16 GB on each Spark (step 5a) |
| 03–04 Ollama, llama.cpp, LM Studio | client labs, Ask the Spark | the server launches |
| 05 vLLM | 05-1 launch builder, client labs | 05-2 first server · **§6 two-Spark tensor parallel** (step 5b) |
| 06–07 engine bake-off, NVFP4 | client labs | each engine launch, one at a time |
| 08 LiteLLM gateway | all labs (gateway on each laptop) | — |
| 09–10 LLaMA Factory, Unsloth | dataset and config labs | the training launches |
| 11 PyTorch / NeMo fine-tune | 11-1 memory math, 11-4 preflight + configs | 11-2 training · 11 §6 FSDP is **optional** (step 5d) |
| 12–13 VLM/FLUX, evaluate and serve | scoring, client labs | training and serving launches |
| 14–19 agents, sandbox, RAG | most labs (laptop side) | anything with `--yes` / `SPARK_APPLY=1` on the Spark |
| 20 Capstone | 20-1 preflight, 20-3 agent, 20-4 steps 1, 2, 4 | **20-2 `--launch`** (step 5c), 20-4 `--yes` |

**Between two live launches, free the ports.** Each lab prints its own `docker rm -f …` line. The two-Spark ones
are in step 5.

## 5 · The two-Spark demos, step by step

Verified live on 2026-10-01. They need `lab_mode.sh on` (step 3).

### 5a · NCCL benchmark (Module 02 lab 03), about 2 minutes
In the runner, ▶ Run lab 02-3, or:
```bash
.venv/bin/python week25/02_two_sparks_nccl/labs/lab03_nccl_bench.py --rdma
```
Expected: busbw **≈ 22.5 GB/s** (pass mark 21.875) and RDMA **≈ 196 Gb/s** (≥ 184). If busbw is low, check that
lab mode is on: both Sparks need their memory free.

### 5b · One 70B model across both Sparks (Module 05 §6), about 10 minutes
The tutorial's commands, with this classroom's values: the cable is in **port 0** (`enp1s0f0np0`), and
Llama 3.3 70B is already cached on both Sparks, so skip the `hf download`. The tutorial's Step 1 (the pinned,
patched `run_cluster.sh`) is already done in `sparklab`'s home on both Sparks. Run as `sparklab`:

```bash
# Spark A: the Ray head (tmux keeps it alive)
ssh sparklab@spark-3b82.<tailnet>.ts.net
tmux new -s ray
export MN_IF_NAME=enp1s0f0np0
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
bash ~/run_cluster.sh nvcr.io/nvidia/vllm:26.05-py3 $VLLM_HOST_IP --head ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP -e UCX_NET_DEVICES=$MN_IF_NAME -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME -e GLOO_SOCKET_IFNAME=$MN_IF_NAME -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 -e MASTER_ADDR=$VLLM_HOST_IP
```
```bash
# Spark B: the Ray worker. HEAD_NODE_IP is Spark A's link IP, which Spark A printed (192.168.100.57)
ssh sparklab@spark-b3b6.<tailnet>.ts.net
tmux new -s ray
export MN_IF_NAME=enp1s0f0np0 HEAD_NODE_IP=192.168.100.57
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
bash ~/run_cluster.sh nvcr.io/nvidia/vllm:26.05-py3 $HEAD_NODE_IP --worker ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP -e UCX_NET_DEVICES=$MN_IF_NAME -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME -e GLOO_SOCKET_IFNAME=$MN_IF_NAME -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 -e MASTER_ADDR=$HEAD_NODE_IP
```
```bash
# Spark A, a second terminal: 2 nodes, then serve (weights load in ~5 min)
C=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec $C ray status                              # Active: 2 nodes · 0.0/2.0 GPU
docker exec -d $C bash -c 'HF_HUB_OFFLINE=1 vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 2 \
  --max-model-len 2048 --gpu-memory-utilization 0.8 --distributed-executor-backend ray > /root/.cache/huggingface/vllm_tp2.log 2>&1'
tail -f ~/.cache/huggingface/vllm_tp2.log              # wait for "Application startup complete."
```
Then the tutorial's haiku request, or ⚡ Ask the Spark. Expect about **2.5 tok/s** for one request: two Sparks
add memory, not per-stream speed. That's the §6 lesson.

**Tear down** (as `sparklab`): on A, `tmux kill-session -t ray; docker rm -f $(docker ps -aq --filter name=^node-)`.
Run the same on B.

### 5c · The capstone (Module 20 lab 02), about 10 minutes
The fine-tuned router serves on Spark B, the agent brain and the LiteLLM gateway on Spark A:
```bash
.venv/bin/python week25/20_capstone_sovereign_agent/labs/lab01_plan_and_preflight.py     # both ✓
.venv/bin/python week25/20_capstone_sovereign_agent/labs/lab02_bring_up.py --launch      # 3 × ✓ "is up"
```
It needs the `hotel-ft` adapter on **Spark B** (`ready_check.sh` checks it; one is trained already). To retrain
it, run Module 09 against Spark B: set `SPARK_HOST=sparklab@spark-b3b6.<tailnet>.ts.net` in that shell, then
labs 09-1 (`SPARK_APPLY=1`), 09-2, 09-3 `--launch`, and 09-4. Training takes about 6 minutes.

Students then run lab 20-3 from their laptops. The router on B answers them directly. The brain on A only
listens on 127.0.0.1, so on laptops it appears as the labelled LAPTOP STAND-IN.

**Tear down**: on A, `docker rm -f w25-brain; pkill -u sparklab -x litellm`; on B, `docker rm -f w25-router`
(both as `sparklab`).

### 5d · Optional: FSDP training across both Sparks (Module 11 §6)
Skipped by default. The lesson (a 70B LoRA in bf16 needs two Sparks; FSDP's link cost) comes from labs 11-1 and
11-4, which everyone runs. A live run needs sudo changes to Docker on both Sparks (`/etc/docker/daemon.json`:
the NVIDIA runtime, the GPU UUID, `swarm-resource`), and opens root SSH with password `root` on port 2233 while
it runs. If you want it, prepare it the day before. Lab 11-4 prints the commands; with the cable in port 0,
change `enp1s0f1np1` to `enp1s0f0np0` in `docker-compose.yml`. Remove the stack and undo the Docker changes
straight after.

## 6 · After class

```bash
week25/spark_host/lab_mode.sh off       # BOTH Sparks: day jobs back, vLLM servers last
```
Wait about 5 minutes, then confirm with `week25/spark_host/lab_mode.sh status`. On Spark B, `altoace-llm` should
show `(healthy)` in `docker ps`.

## 7 · When something goes wrong

| Symptom | Cause · fix |
|---|---|
| A runner chip is red | `ssh -o BatchMode=yes sparklab@<spark> true` from that machine. Tailnet off, key missing (`add_key.sh` on that Spark), or not tagged |
| Processes on Spark B killed during a GPU lab (`journalctl -k` shows `Out of memory`) | lab mode was off on B. `lab_mode.sh off` then `on` from Spark A |
| `altoace-llm` restarts after class (`KV cache … larger than the available`) | it started before the rest took their memory. `docker restart altoace-llm` on B (`lab_mode.sh off` already starts it last) |
| A port is busy (8000 / 4000) | a previous launch is still up: `docker ps`, then the tear-down line of that demo |
| NCCL busbw ≈ 20.7 instead of ≈ 22.5 GB/s | memory not free on both Sparks: `lab_mode.sh status` |
| Want the labs in DRY mode (no Spark, or projector only) | the runner's 📄 Dry run switch. Labs 02-1 and 02-3 replay real recordings of these two Sparks |
