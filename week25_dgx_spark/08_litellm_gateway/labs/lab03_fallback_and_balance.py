#!/usr/bin/env python3
"""Lab 08-3 · Fallbacks, retries, cooldown and load balancing — watch the router decide.

Starts the gateway on this laptop with lab 01's config plus two demo aliases, then:
  1. calls `spark-a/vllm` twice. With no Spark serving vLLM, the only deployment fails and the
     alias's fallback (laptop-fast, Ollama on this Mac) answers. The x-litellm-* headers prove it.
  2. calls `chat`, the alias load-balanced across BOTH Sparks' vLLM: both down → fallback.
  3. calls `one-spark-down` six times: ONE alias, TWO deployments (Spark A's vLLM + the laptop).
     A call that lands on the dead Spark is retried on the other deployment; once the Spark has
     failed more than allowed_fails times it is put in cooldown and skipped.
  4. calls `laptop-pair` eight times: two deployments (the same laptop Ollama under two names,
     standing in for Spark A and Spark B), and counts which one the router picked.
If your Sparks DO serve these models, the Spark answers instead, and the lab says so.

Run: .venv/bin/python week25/08_litellm_gateway/labs/lab03_fallback_and_balance.py
"""
import collections
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gateway import LAPTOP, RUNS, Proxy, build_config, raw, write_config  # noqa: E402
from sparkkit import banner, check, note, result, step, table  # noqa: E402

banner("Lab 08-3 · fallbacks, retries, cooldown, load balancing",
       "a primary that is down, a fallback that is up · x-litellm-* headers show every decision")

cfg, _ = build_config(demo=True)
demo = write_config(cfg, RUNS / "litellm.demo.yaml")
fb = {k: v for d in cfg["router_settings"]["fallbacks"] for k, v in d.items()}
rs = cfg["router_settings"]
print(f"│ router_settings: num_retries={rs['num_retries']} · timeout={rs['timeout']} s · "
      f"allowed_fails={rs['allowed_fails']} · cooldown_time={rs['cooldown_time']} s · "
      f"fallbacks: spark-a/vllm → {fb['spark-a/vllm']} · chat → {fb['chat']}")


def ask(gw, alias: str, prompt: str = "Reply with one word: ready.") -> list:
    t0 = time.perf_counter()
    code, h, body = raw("POST", gw.base + "/v1/chat/completions", gw.key,
                        {"model": alias, "messages": [{"role": "user", "content": prompt}], "max_tokens": 20,
                         "temperature": 0})
    secs = time.perf_counter() - t0
    h = {k.lower(): v for k, v in h.items()}
    text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or \
        (body.get("error") or {}).get("message", "")[:60]
    return [alias, code, h.get("x-litellm-model-group", "—"), h.get("x-litellm-model-api-base", "—"),
            h.get("x-litellm-attempted-retries", "—"), h.get("x-litellm-attempted-fallbacks", "—"),
            f"{secs:.1f} s", text.strip().replace("\n", " ")[:24]]


HEAD = ["alias asked", "HTTP", "answered by (group)", "api_base", "retries", "fallbacks", "time", "reply"]

with Proxy(demo) as gw:
    step(1, "spark-a/vllm, twice — one deployment on Spark A, fallback on the laptop")
    calls = [ask(gw, "spark-a/vllm"), ask(gw, "spark-a/vllm")]
    table(calls, HEAD)
    if calls[0][3].startswith(LAPTOP):
        check(all(c[5] not in ("0", "—") for c in calls), "Spark A's vLLM did not answer → both calls fell back to "
              "laptop-fast (Ollama on this Mac, LAPTOP STAND-IN)", "fallback header missing")
        note("An alias with ONE deployment has nowhere else to send traffic, so every call tries Spark A, fails and "
             "falls back. Here the failure is instant (the placeholder host does not resolve); a Spark that is "
             "off but resolvable costs a connect timeout per call. The time column is mostly the laptop's reply.")
    else:
        note(f"Spark A's vLLM answered ({calls[0][3]}) — no fallback needed. Stop vLLM on the Spark to see one.")

    step(2, "chat — load-balanced across Spark A and Spark B vLLM")
    table([ask(gw, "chat")], HEAD)
    note("Neither of the `chat` group's deployments (Spark A, Spark B) answered, so the group's fallback ran. "
         "With one Spark up, `chat` keeps answering from that Spark — the point of two deployments.")

    step(3, "one-spark-down × 6 — retries inside the group, then cooldown")
    rows = [ask(gw, "one-spark-down", f"Reply with the number {i + 1} only.") for i in range(6)]
    table([[i + 1] + r[1:2] + r[3:5] + r[6:7] for i, r in enumerate(rows)],
          ["call", "HTTP", "api_base that answered", "retries", "time"])
    hit = [i + 1 for i, r in enumerate(rows) if r[4] not in ("0", "—")]
    if rows[0][3].startswith(LAPTOP) or hit:
        print(f"│ calls that first landed on the dead Spark and were retried on the laptop: {hit or 'none'}")
        note(f"num_retries={rs['num_retries']} retried those calls on the group's other deployment — the client saw "
             f"HTTP 200 every time. allowed_fails={rs['allowed_fails']}: once the Spark failed more than that, it "
             f"went into cooldown for {rs['cooldown_time']} s and later calls stopped trying it (retries 0). "
             "simple-shuffle is random, so which calls hit the Spark changes from run to run.")
    else:
        note("every call went to the live deployment — shuffle never picked the dead Spark this run. Run again.")

    step(4, "laptop-pair × 8 — the balancer at work (two names for the SAME laptop Ollama)")
    picks = collections.Counter()
    for i in range(8):
        row = ask(gw, "laptop-pair", f"Reply with the number {i + 1} only.")
        picks[row[3]] += 1
    for base, n in sorted(picks.items()):
        print(f"│ {base:30s} {'█' * n * 3} {n}")
    check(len(picks) == 2, "simple-shuffle spread the calls over both deployments",
          "every call went to one deployment (possible with 8 random picks — run again)")

    step(5, "the gateway's own log")
    log = [re.sub(r"\x1b\[[0-9;]*m", "", ln).strip() for ln in gw.log.read_text(errors="replace").splitlines()]
    access = [ln for ln in log if '"POST /v1/chat/completions' in ln]
    for ln in access[:3]:
        print("│ " + ln[:150])
    ok200 = sum(" 200 " in ln for ln in access)
    print(f"│ … {len(access)} requests logged, {ok200} of them HTTP 200 — the failed Spark attempts are not in this log")
    note("By default the proxy logs one access line per request. Which deployment failed and which answered is "
         "in the x-litellm-* headers above; add --detailed_debug to the litellm command to log every routing step.")

result("fallbacks keep answers coming when a Spark is down; retries and cooldown decide how long you wait; "
       "two deployments under one alias spread the load. All of it is config — no client changed.")
