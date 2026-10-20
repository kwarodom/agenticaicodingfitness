#!/usr/bin/env python3
"""Lab 13-4 · Route and gate: put the fine-tune behind one gateway alias, then decide whether it ships.

Part 1 starts Module 08's LiteLLM proxy on this laptop with a config for ONE alias clients use,
`hotel-router`, which tries in order:
    1. hotel-ft on Spark A's vLLM   (the LoRA adapter from lab 13-3)
    2. hotel-base on Spark A's vLLM (the same base model, prompt-only)
    3. gemma4:12b on this laptop     (last resort, LAPTOP STAND-IN)
It sends five held-out guest messages and prints which backend answered each one, from LiteLLM's
response headers. Clients never change: they always ask for "hotel-router".

Part 2 puts the saved prediction files side by side (baseline from lab 13-1, fine-tune from lab 13-2)
and applies the ship gate. The proxy is always stopped on exit.

Run: .venv/bin/python week25/13_finetune_to_serve/labs/lab04_route_and_gate.py [--n 5]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "08_litellm_gateway" / "labs"))
import _gateway as G  # noqa: E402  (Module 08's config writer + proxy runner)
import _hotel_eval as H  # noqa: E402
from sparkkit import banner, note, result, step, table, url, warn  # noqa: E402

N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 5
BASE = "Qwen/Qwen3-4B-Instruct-2507"
vllm_a = url("vllm", "a") or "http://spark-a:8000/v1"

CFG = {
    "model_list": [
        {"model_name": "hotel-router",
         "litellm_params": {"model": "hosted_vllm/hotel-ft", "api_base": vllm_a, "api_key": "none"}},
        {"model_name": "hotel-base",
         "litellm_params": {"model": f"hosted_vllm/{BASE}", "api_base": vllm_a, "api_key": "none"}},
        {"model_name": "hotel-laptop",
         "litellm_params": {"model": "openai/gemma4:12b", "api_base": G.LAPTOP, "api_key": "none",
                            "extra_body": {"reasoning_effort": "none"}}},
    ],
    "router_settings": {"num_retries": 0, "timeout": 120,
                        "fallbacks": [{"hotel-router": ["hotel-base", "hotel-laptop"]},
                                      {"hotel-base": ["hotel-laptop"]}]},
    "litellm_settings": {"drop_params": True},
    "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
}

banner("Lab 13-4 · route and gate", "one alias for clients · a side-by-side ship decision")

step(1, "write the gateway config: hotel-router → hotel-ft → hotel-base → laptop")
cfg_path = G.write_config(CFG, G.RUNS.parent.parent / "13_finetune_to_serve" / ".runs" / "hotel-gateway.yaml")
table([[d["model_name"], d["litellm_params"]["model"], d["litellm_params"]["api_base"]] for d in CFG["model_list"]],
      ["alias", "backend model", "api_base"])
print("│ fallbacks: hotel-router → hotel-base → hotel-laptop")
note(f"config → {cfg_path.relative_to(G.WEEK)}")

step(2, f"start the gateway and send {N} held-out messages to `hotel-router`")
answered = []
with G.Proxy(cfg_path) as px:
    for it in H.load_eval()[:: max(1, 60 // N)][:N]:
        body = {"model": "hotel-router", "max_tokens": 160, "temperature": 0,
                "messages": [{"role": "system", "content": it["system"]}, {"role": "user", "content": it["instruction"]}]}
        code, hdr, js = G.raw("POST", px.base + "/v1/chat/completions", px.key, body)
        hl = {k.lower(): v for k, v in hdr.items()}
        text = ((js.get("choices") or [{}])[0].get("message") or {}).get("content", "") if code == 200 else ""
        s = H.score_item(text, it["output"], it["instruction"])
        answered.append([it["instruction"][:34] + "…", code, js.get("model", "—"),
                         hl.get("x-litellm-attempted-fallbacks", "0"),
                         "✓" if s["department_ok"] and s["json_valid"] else "✕"])
table(answered, ["guest message", "HTTP", "answered by", "fallbacks", "right dept + JSON"])
if all(r[3] != "0" for r in answered):
    note("Every answer needed a fallback: the fine-tune is not being served right now. Clients noticed nothing — "
         "which is exactly why you must LOG `x-litellm-attempted-fallbacks`, or a dead fine-tune goes unnoticed.")

step(3, "side by side: baseline vs fine-tune, same 60 messages, same gate")
files = {"baseline": sorted(H.RUNS.glob("predictions_baseline_*.jsonl")),
         "fine-tune": sorted(H.RUNS.glob("predictions_finetune_*.jsonl"))}
metrics = {}
for name, paths in files.items():
    if paths:
        rows = H.read_predictions(paths[-1])
        metrics[name] = (paths[-1].name, H.summarize([H.score_item(r["predict"], r["label"], r.get("prompt", ""))
                                                      for r in rows]))
if not metrics:
    warn("no prediction files yet — run lab 13-1 (baseline) and lab 13-2 (fine-tune) first")
else:
    names = list(metrics)
    table([[k] + [f"{metrics[n][1][k]:.0%}" for n in names] + [f"≥ {v:.0%}"] for k, v in H.GATE.items()],
          ["gate metric"] + [f"{n} ({metrics[n][0]})" for n in names] + ["needed"])
    if "fine-tune" not in metrics:
        warn("no fine-tune predictions: the ship decision needs Module 09's adapter scored on a Spark (lab 13-2).")
    else:
        ok = all(metrics["fine-tune"][1][k] >= v for k, v in H.GATE.items())
        better = metrics["fine-tune"][1]["department_acc"] >= metrics.get("baseline", ("", {"department_acc": 0}))[1]["department_acc"]
        result(f"SHIP the fine-tune" if ok and better else "DO NOT SHIP yet: fix the ✕ metrics, retrain, re-score")

result("The gate, not a feeling, decides. Keep the baseline file: every retrain is scored against it.")
