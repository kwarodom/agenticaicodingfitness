# Connect a laptop to the DGX Spark (remote, authenticated) for Week 19

Run every Week 19 app in **REAL** mode against the shared **AltoTech DGX Spark**
(`spark-3b82`, NVIDIA GB10) from any laptop — over the encrypted **Tailscale** mesh,
authenticated by your Tailscale device identity. No public internet exposure, no API
key to leak, cloud cost **$0** (inference runs on the Spark).

> Week 19 reaches a DGX three ways via `DGX_CONN` (`local | tunnel | cloud`).
> A DGX on another network, exposed over Tailscale, is the **`tunnel`** path — and
> `config.py` auto-detects `.ts.net` hosts as a tunnel.
>
> Remember the golden rule from the README: the **tutorial web apps run on your
> laptop**; only the **model** runs on the Spark. This file wires your laptop's apps
> to the Spark's model. The big-iron chapter commands (`docker run nvcr.io/...`,
> `trtllm-serve`, `nemo`, `litellm --config`, `mpirun`) are still homework you run
> *on* a DGX — they are not driven from here.

## What the Spark already provides (server side — nothing to do there)

- **Ollama** on `:11434`, bound to all interfaces (persists across reboot), serving
  `nemotron-3-super:120b`, `qwen3.6:35b-a3b-q8_0`, `llama3.3:70b`, `gemma4:latest`,
  `gemma3:4b`, `qwen3-vl:32b`, `qwen3:32b`, … (9 models).
- Reachable on the tailnet as `spark-3b82.tail461566.ts.net` (`100.111.125.74`).

## One-time setup on the laptop

```bash
# 1. Install Tailscale and join the tailnet (browser login as an authorized user).
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up            # you must be admitted to tailnet tail461566.ts.net

# 2. Confirm the laptop can see the Spark.
tailscale status | grep spark-3b82
curl -s http://spark-3b82.tail461566.ts.net:11434/v1/models | head
```

If step 2 shows the models list, you're connected.

## Every time you run the tutorials

```bash
# From the repo root:
source week19/connect-remote.env       # sets DGX_CONN + DGX_TUNNEL_URL

uv venv && source .venv/bin/activate
uv pip install -r week19/sovereign_dgx/requirements.txt
.venv/bin/python week19/sovereign_dgx/tutorial_server.py   # → http://127.0.0.1:8092
```

Open the URL. The 🔌 Connection panel should read **`tunnel` · REAL · on your DGX · $0**.
Any other Week 19 app works the same way — swap the folder and use its port:

| App | Port |
|---|---|
| `sovereign_dgx` | 8092 |
| `dgx_finetune` | 8093 |
| `dgx_observability` | 8094 |
| `self_evolving_agent_v2` | 8095 |
| `dgx_litellm` | 8096 |

## Model note

The apps auto-pick `nemotron-3-super:120b`. Its Ollama build streams a **reasoning**
channel and leaves the OpenAI `content` field empty — the tutorial UI renders the
reasoning stream, so demos still show output. For snappier, plain-answer output (and
for non-streaming steps like tool-calling), pin a non-reasoning model:

```bash
export DGX_MODEL=llama3.3:70b     # or gemma4:latest
```

(Already available as a commented line in `connect-remote.env`.)

## Adding a new laptop to the lab (tailnet admin)

A new PC only reaches the Spark after it's **admitted to the tailnet**:

1. On the new PC: `sudo tailscale up` and log in with an authorized identity.
2. A tailnet admin approves the device at `login.tailscale.com/admin/machines`
   (only needed if device approval is on for the tailnet).
3. Verify from the new PC: `curl -s http://spark-3b82.tail461566.ts.net:11434/v1/models`.

### Authorization scoping (optional, admin only)

By default any device on `tail461566.ts.net` can reach `:11434`. To restrict access to
specific laptops, add a grant in the Tailscale admin console
(`login.tailscale.com/admin/acls`):

```jsonc
{ "action": "accept",
  "src": ["laptop-owner@your-domain"],   // or a device tag you assign
  "dst": ["spark-3b82:11434"] }
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| App shows **SIM**, not REAL | `source week19/connect-remote.env` in the *same* shell; `curl …:11434/v1/models` must succeed. |
| `curl` to the Spark hangs / no route | `tailscale status` — is the laptop up and is `spark-3b82` listed? Re-run `sudo tailscale up`. |
| DNS name won't resolve | Enable MagicDNS, or use the IP: `export DGX_TUNNEL_URL=http://100.111.125.74:11434/v1`. |
| Demo output looks empty | Reasoning model — see **Model note**; pin `DGX_MODEL=llama3.3:70b`. |
