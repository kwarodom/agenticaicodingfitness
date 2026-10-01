# Week 25 classroom: one DGX Spark, many laptops

Everyone (instructor and students) runs the **Spark Lab Runner on their own laptop**. The runner drives
this Spark over the tailnet in two ways:

```text
 laptop: tutorial_server.py :8125 ──ssh sparklab@<spark>.<tailnet>.ts.net──►  Spark: docker / nvidia-smi / hf
                                  └─http <spark>.<tailnet>.ts.net:11434, :8000, :4000 …─► engines on the Spark
```

`./lab_mode.sh status` prints the exact `SPARK_HOST` to hand out.

## Instructor: one-time setup on the Spark

```bash
cd ~/Documents/agenticaicodingfitness
sudo week25/spark_host/setup_sparklab_user.sh             # sparklab account, key-only ssh, docker+GPU, hf CLI
sudo -u sparklab -H bash -lc '~/.local/bin/hf auth login' # a READ-only HF token (gated Llama models)
sudo -u sparklab -H docker login nvcr.io -u '$oauthtoken' # password = NGC API key
week25/spark_host/prefetch.sh pull                        # Ollama models + images (~150 GB, shared)
week25/spark_host/prefetch.sh hf                          # HF weights into sparklab's cache
```

**Tailnet (your org tailnet).** Invite each student to the tailnet. The tailnet also holds the Alto IoT nodes,
so give students access to this Spark only. In the admin console's access controls:

```jsonc
"groups":    { "group:w25-students": ["student1@example.com"] },
"tagOwners": { "tag:dgx-spark": ["autogroup:admin"] },
"grants": [
  { "src": ["group:w25-students"], "dst": ["tag:dgx-spark"],
    "ip":  ["22", "11434", "8000", "8001", "30000", "8355", "30080", "1234", "4000", "12000", "8080"] }
]
```

Then tag the Spark (`sudo tailscale up --advertise-tags=tag:dgx-spark`, or do it in the console). A tagged
node also stops expiring its key, so it won't drop off in the middle of a class. With two Sparks, tag **both**:
the grant above only reaches tagged nodes.

**Each laptop's key.** Collect `~/.ssh/id_ed25519.pub` from every laptop:

```bash
sudo week25/spark_host/add_key.sh alice.pub bob.pub     # --list / --remove alice
```

With two Sparks, run the same `add_key.sh` on **both**: every two-Spark lab ssh-es into Spark B as `sparklab` too.

## Instructor: every class

```bash
week25/spark_host/lab_mode.sh on       # stops nemotron-lightning (~100 GB), supabase-kong (:8000), alto-backend (:8001)
week25/spark_host/lab_mode.sh status   # all lab ports free, ~110 GB available
# … class …
week25/spark_host/lab_mode.sh off      # restarts exactly what `on` stopped (vLLM servers last)
```

**Two Sparks.** Put `SPARK_B_ADMIN=altoaidev@<spark-b>` in `week25/.env.local` on Spark A. The same three commands
then also run on Spark B over ssh, where they stop and restart B's own day job (the `altoace` stack, ~94 GB).
Spark B needs this repo at the same path, kept up to date with `git pull`. Run `on` before **every** class that
uses Spark B: the two-Spark labs need most of both Sparks' memory, and a 16 GB NCCL test against B's day job
gets processes on B killed by the kernel's out-of-memory killer.

**One Spark, many learners.** The launch labs (`docker run -p 8000:8000 vllm …`) each bind a fixed port
and use most of the 128 GB. If two people launch them at the same moment, they collide. So the
**instructor runs the launch/fine-tune labs live** and students follow along. Once an engine is up,
everyone can use **⚡ Ask the Spark**, the client labs, and **📄 Dry run** for the rest.

The two-Spark labs follow the same rule, more strictly, since there is only one cluster:

| Lab | Who runs it live | Why |
|---|---|---|
| 02-1 link check, 02-2 netplan plan | everyone | read-only |
| 02-3 NCCL benchmark (16 GB per Spark) | instructor | two at once run both Sparks out of memory |
| 05 §6 vLLM tensor parallel (Ray, Llama 3.3 70B) | instructor | one Ray cluster, :8000, ~100 GB on each Spark |
| 11-4 FSDP preflight + configs | everyone | read-only, writes only `.runs/` and each Spark's `configs/` |
| 11 §6 the FSDP training itself | instructor, optional | needs sudo setup on both Sparks (TWO_SPARKS.md) and opens root ssh on :2233 while it runs |
| 20-1 plan + preflight | everyone | read-only |
| 20-2 `--launch` | instructor | one router on B (:8000), one brain + gateway on A |
| 20-3 agent end to end | everyone | each laptop runs its own LiteLLM proxy; the router on B answers it directly. The brain on A is bound to 127.0.0.1 on purpose, so on laptops `agent-brain` runs as the labelled LAPTOP STAND-IN; the instructor's run on Spark A uses the real brain |
| 20-4 policy + scorecard | everyone; `--yes` (the OpenShell sandbox on A) instructor | steps 1, 2 and 4 run on the laptop |

## Student: before the first class

1. **Tailscale:** accept the tailnet invite, install Tailscale, sign in, then run `tailscale ping <spark>`.
2. **SSH key:** if you don't have one, run `ssh-keygen -t ed25519`. Send `~/.ssh/id_ed25519.pub` to the
   instructor (the public key only, never the private one).
3. **Runner:**
   ```bash
   git clone https://github.com/kwarodom/agenticaicodingfitness.git && cd agenticaicodingfitness
   uv venv -p 3.13 .venv && uv pip install -p .venv fastapi uvicorn pyyaml pillow
   .venv/bin/python week25/00_spark_lab_runner/tutorial_server.py      # → http://127.0.0.1:8125
   ```
4. **Connect:** open **🖥 Spark setup** and set `SPARK_HOST = sparklab@<spark>.<tailnet>.ts.net`. Leave
   the HTTP host empty. The **Spark A** chip should turn green.
   To check from a terminal: `ssh -o BatchMode=yes sparklab@<spark>.<tailnet>.ts.net nvidia-smi -L`.
