#!/usr/bin/env python3
"""Lab 08-1 · Build the gateway config: every engine on both Sparks, one alias each, laptop fallbacks.

Reads your Spark settings (SPARK_HOST / SPARK_API_HOST, as in Module 01), asks each engine that
answers which model it serves, and writes ONE LiteLLM config.yaml: aliases for Ollama :11434,
vLLM :8000, SGLang :30000 and TensorRT-LLM :8355 on Spark A and Spark B, a load-balanced `chat`
alias across both Sparks' vLLM, fallbacks to the laptop's Ollama, and a master key read from the
environment. Then it validates the file twice: with PyYAML, and with LiteLLM's own Router.

Runs on the laptop, offline. Without a Spark the hosts are the placeholders spark-a / spark-b.

Run: .venv/bin/python week25/08_litellm_gateway/labs/lab01_build_config.py
     add --on-spark [--laptop-url http://<laptop>:11434/v1] to also write the variant for a gateway
     that runs ON Spark A (Spark A's engines on localhost), and copy it to ~/w25/litellm/ when LIVE.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gateway import RUNS, WEEK, build_config, for_spark, write_config  # noqa: E402
from sparkkit import api_host, banner, check, note, put, result, step, table, where  # noqa: E402

banner("Lab 08-1 · build the LiteLLM gateway config", "all engines × both Sparks → one config.yaml · runs offline")

step(1, "where each alias points")
print(f"│ Spark A HTTP host: {api_host('a') or '(not set → placeholder spark-a)'} · "
      f"Spark B: {api_host('b') or '(not set → placeholder spark-b)'}")
cfg, rows = build_config()
table(rows, ["alias", "LiteLLM model (provider/id)", "api_base", "model id from"])

step(2, "write it")
path = write_config(cfg)
text = path.read_text()
print(f"│ {path.relative_to(WEEK.parent)} · {len(text.splitlines())} lines · {len(cfg['model_list'])} deployments")
print("│ ─── excerpt ───")
lines = text.splitlines()
rs, fb, ls = (lines.index(k) for k in ("router_settings:", "  fallbacks:", "litellm_settings:"))
for ln in lines[:14] + ["  …"] + lines[rs:fb + 5] + ["  …"] + lines[ls:]:
    print("│ " + ln)

step(3, "validate — PyYAML, then LiteLLM's own Router")
import yaml  # noqa: E402

back = yaml.safe_load(text)
ok = check(back == cfg, "PyYAML: the file parses back to exactly the generated config",
           "PyYAML: the file does not round-trip")
aliases = {m["model_name"] for m in back["model_list"]}
ok &= check(all(src in aliases and all(d in aliases for d in dst) for fb in back["router_settings"]["fallbacks"]
                for src, dst in fb.items()),
            "every fallback names aliases that exist", "a fallback points at an alias that is not in model_list")
ok &= check(back["general_settings"]["master_key"].startswith("os.environ/"),
            "master_key is read from the environment (os.environ/LITELLM_MASTER_KEY), not stored in the file",
            "master_key is a literal secret in the file")
try:
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")      # no download of the price list
    from importlib.metadata import version  # noqa: E402

    import litellm  # noqa: E402

    router = litellm.Router(model_list=back["model_list"], **back["router_settings"])
    groups = sorted(set(router.get_model_names()))
    ok &= check(len(router.model_list) == len(back["model_list"]),
                f"litellm {version('litellm')} Router accepted {len(router.model_list)} deployments in "
                f"{len(groups)} aliases · strategy {router.routing_strategy}",
                "the Router dropped deployments")
except Exception as e:  # noqa: BLE001
    ok &= check(False, "", f"LiteLLM Router rejected the config: {type(e).__name__}: {str(e)[:200]}")

note("Engines on the Spark have NO authentication. The master key protects the gateway only — keep the engine "
     "ports off the network (bind to localhost, or firewall them) so clients must come through :4000.")
if not ok:
    sys.exit(1)

if "--on-spark" in sys.argv:
    step(4, "the variant for a gateway ON Spark A")
    i = sys.argv.index("--laptop-url") + 1 if "--laptop-url" in sys.argv else 0
    laptop_url = sys.argv[i] if i and i < len(sys.argv) else ""
    spark_cfg = for_spark(cfg, laptop_url)
    sp = write_config(spark_cfg, RUNS / "litellm.spark.yaml")
    table([[m["model_name"], m["litellm_params"]["api_base"]] for m in spark_cfg["model_list"]], ["alias", "api_base"])
    print(f"│ fallbacks: {spark_cfg['router_settings']['fallbacks'][:2]} …")
    if not laptop_url:
        note("no --laptop-url: the laptop fallbacks are dropped (localhost on a Spark is the Spark), and `chat` "
             "falls back to Spark A's Ollama instead.")
    put(sp, "~/w25/litellm/config.yaml")
    if where("a") == "dry":
        note("DRY: not copied. With a Spark configured, the file lands in ~/w25/litellm/config.yaml on Spark A.")

result(f"config ready: {path.relative_to(WEEK.parent)}. Lab 02 starts LiteLLM with it on this laptop.")
