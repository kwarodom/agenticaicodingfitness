# ▣ Week 19 · App 5 — Serving models on a DGX with LiteLLM

An **interactive, explainable web app + runnable demos** for the **serving /
gateway layer** of the sovereign stack: put one OpenAI-compatible URL + key in
front of every model you run on the DGX, with routing, fallbacks, hot-swap,
virtual keys, and observability — all on-prem, $0.

```
apps / agents → ▣ LiteLLM gateway (one URL + key) → Ollama · vLLM · llama.cpp · TRT-LLM
                                                            (all on the DGX)
```

## Three situations — auto-detected

| Situation | When | What runs |
|-----------|------|-----------|
| **PROXY**  | a real LiteLLM proxy answers at `LITELLM_BASE_URL` (default `:4000`) | calls go through the gateway; routing is real |
| **DIRECT** | no proxy, but a backend (Ollama/vLLM) is reachable | **real** backend generation; routing/keys shown via the simulator |
| **SIM**    | nothing reachable | the gateway is fully simulated (no GPU, $0) |

The router, fallback, virtual-key, and budget behaviours are illustrated with a
faithful simulator (`litesim.py`) — those are *gateway* features — while the
generation itself is real whenever a proxy or backend is up. Real `litellm`
commands and config are always shown.

---

## Quick start

```bash
uv pip install -r week19/dgx_litellm/requirements.txt

# (optional) run a REAL gateway — step 1 writes the config for you
pip install 'litellm[proxy]'
litellm --config .sandbox/litellm_config.yaml --port 4000
export LITELLM_BASE_URL=http://localhost:4000
# and a backend behind it:
ollama run qwen3.6:35b-a3b-q8_0      # gemma4:12b on a Mac

# the interactive web app
.venv/bin/python week19/dgx_litellm/tutorial_server.py
# → open http://127.0.0.1:8096
```

Env overrides: `LITELLM_BASE_URL`, `LITELLM_KEY`, `DGX_BASE_URL` (the backend),
`DGX_MODE=sim|real`.

---

## Layout

```
week19/dgx_litellm/
├── README.md · requirements.txt · config.py
├── litesim.py          → the LiteLLM router simulator (model_list, routing, keys)
├── liteview.py         → the gateway-call engine (proxy / direct / sim)
├── tutorial_server.py  → FastAPI control plane
├── static/guide.html   → clickable, streaming UI
└── demos/
    ├── step01_install_config.py    Ch 2 · install + write a real config.yaml
    ├── step02_unified_endpoint.py  Ch 3 · one URL/key, many models
    ├── step03_routing.py           Ch 4 · routing & load-balancing across Sparks
    ├── step04_fallbacks.py         Ch 5 · fallbacks, retries, cooldowns
    ├── step05_hotswap.py           Ch 6 · model management & VRAM hot-swap
    ├── step06_keys_budgets.py      Ch 7 · virtual keys, budgets, rate limits
    └── step07_observability.py     Ch 8 · logging callbacks → Phoenix on the DGX
```

---

## The 8 chapters

| Ch | Demo | The one thing to notice |
|----|------|--------------------------|
| 1  | *(concept)* | A real deployment runs *several* runtimes — LiteLLM unifies them. |
| 2  | `step01` | A `model_list` maps **friendly aliases → real backends**; `litellm --config`. |
| 3  | `step02` | Every model is **one base_url + key** away; only the model name changes. |
| 4  | `step03` | One alias, **many Sparks** — shuffle / usage-based / latency routing. |
| 5  | `step04` | **Fallback chains** keep the gateway answering when a Spark fails. |
| 6  | `step05` | **Hot-swap** offers more models than fit in 128 GB of VRAM at once. |
| 7  | `step06` | **Virtual keys** = per-team allow-lists, rate limits, on-prem **quotas**. |
| 8  | `step07` | The gateway is the **one funnel** to trace → Phoenix on the DGX. |

---

## Where this sits in Week 19

```
App 1 sovereign_dgx → the runtimes (Ollama/vLLM/llama.cpp) this gateway fronts
App 2 dgx_finetune  → models tuned to your domain, served behind an alias
App 3 dgx_observability → LiteLLM's callbacks stream every call here (Phoenix)
App 4 self_evolving_agent_v2 → an app that calls the gateway's one URL
App 5 (THIS) dgx_litellm → the SERVING / gateway layer that ties them together
```

App 1 showed *how* to serve a model; App 5 is how you serve **many** of them in
production behind a single, governable, observable endpoint — the control point
of a sovereign DGX deployment, with nothing leaving the box.
