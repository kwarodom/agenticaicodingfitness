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
node also stops expiring its key, so it won't drop off in the middle of a class.

**Each laptop's key.** Collect `~/.ssh/id_ed25519.pub` from every laptop:

```bash
sudo week25/spark_host/add_key.sh alice.pub bob.pub     # --list / --remove alice
```

## Instructor: every class

```bash
week25/spark_host/lab_mode.sh on       # stops nemotron-lightning (~100 GB), supabase-kong (:8000), alto-backend (:8001)
week25/spark_host/lab_mode.sh status   # all lab ports free, ~110 GB available
# … class …
week25/spark_host/lab_mode.sh off      # restarts exactly what `on` stopped
```

**One Spark, many learners.** The launch labs (`docker run -p 8000:8000 vllm …`) each bind a fixed port
and use most of the 128 GB. If two people launch them at the same moment, they collide. So the
**instructor runs the launch/fine-tune labs live** and students follow along. Once an engine is up,
everyone can use **⚡ Ask the Spark**, the client labs, and **📄 Dry run** for the rest.

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
