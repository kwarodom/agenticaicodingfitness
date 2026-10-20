#!/usr/bin/env python3
"""Exercise 08 · Finish a LiteLLM gateway config for two Sparks.

The YAML below is a gateway with one alias, `chat`, served by vLLM on Spark A. Fill in the four
TODOs in the YAML, save, then run:
    .venv/bin/python week25/08_litellm_gateway/exercises/ex08_complete_the_gateway.py

The checker is free and offline: it parses your YAML with PyYAML, checks the four TODOs, then
asks LiteLLM's own Router to load it. Stuck? Compare with exercises/solutions/.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check  # noqa: E402

CONFIG_YAML = """
model_list:
  - model_name: chat
    litellm_params:
      model: hosted_vllm/nvidia/Llama-3.1-8B-Instruct-FP8
      api_base: http://spark-a:8000/v1
      api_key: none

  # ── TODO 1 ── add a SECOND deployment of `chat`: the same model on Spark B (host spark-b, port 8000).
  #    Same model_name → LiteLLM load-balances between the two Sparks.

  - model_name: laptop-fast
    litellm_params:
      model: openai/gemma3:4b
      api_base: http://localhost:11434/v1
      api_key: none

router_settings:
  routing_strategy: simple-shuffle
  # ── TODO 2 ── when both Sparks fail, send `chat` to `laptop-fast`:  fallbacks: [{"chat": ["laptop-fast"]}]
  # ── TODO 3 ── retry once (num_retries), give up after 120 s (timeout), and bench a deployment for 30 s
  #    after one failure (allowed_fails, cooldown_time).

general_settings:
  # ── TODO 4 ── require a master key, read from the LITELLM_MASTER_KEY environment variable.
  #    (Never write the key itself into this file. LiteLLM reads "os.environ/NAME" as "the env var NAME".)
  {}
"""


# ─────────────────────────── checker — no need to edit below ────────────────
def main() -> None:
    import yaml

    banner("Exercise 08 · complete the gateway", "offline checker · PyYAML + LiteLLM's Router · no Spark needed",
           status=False)
    try:
        cfg = yaml.safe_load(CONFIG_YAML) or {}
    except yaml.YAMLError as e:
        check(False, "", f"the YAML does not parse: {str(e).splitlines()[0]} — check the indentation")
        sys.exit(1)
    ml = cfg.get("model_list") or []
    rs = cfg.get("router_settings") or {}
    gs = cfg.get("general_settings") or {}
    chat = [m.get("litellm_params", {}) for m in ml if m.get("model_name") == "chat"]
    bases = {p.get("api_base", "") for p in chat}
    ok = check(len(chat) == 2 and any("spark-b:8000" in b for b in bases) and any("spark-a:8000" in b for b in bases)
               and all(p.get("model") == chat[0].get("model") for p in chat),
               "TODO 1: `chat` has two deployments — vLLM on spark-a:8000 and spark-b:8000, same model",
               f"TODO 1: want two `chat` deployments (spark-a:8000 and spark-b:8000, same model); found {sorted(bases)}")
    fbs = {k: v for d in (rs.get("fallbacks") or []) if isinstance(d, dict) for k, v in d.items()}
    ok &= check(fbs.get("chat") == ["laptop-fast"],
                "TODO 2: chat falls back to laptop-fast",
                f"TODO 2: router_settings.fallbacks should map chat → ['laptop-fast'] (got {rs.get('fallbacks')!r})")
    ok &= check(rs.get("num_retries") == 1 and rs.get("timeout") == 120 and rs.get("allowed_fails") == 1
                and rs.get("cooldown_time") == 30,
                "TODO 3: num_retries 1 · timeout 120 · allowed_fails 1 · cooldown_time 30",
                "TODO 3: set num_retries: 1, timeout: 120, allowed_fails: 1, cooldown_time: 30 under router_settings "
                f"(got {({k: rs.get(k) for k in ('num_retries', 'timeout', 'allowed_fails', 'cooldown_time')})})")
    mk = gs.get("master_key") if isinstance(gs, dict) else None
    ok &= check(mk == "os.environ/LITELLM_MASTER_KEY",
                "TODO 4: master_key comes from the environment, not from the file",
                "TODO 4: general_settings.master_key should be os.environ/LITELLM_MASTER_KEY"
                + (" — never a literal key" if isinstance(mk, str) and mk.startswith("sk-") else f" (got {mk!r})"))
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)

    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    try:
        import litellm

        router = litellm.Router(model_list=ml, **rs)
        check(len(router.model_list) == len(ml),
              f"LiteLLM Router loaded it: {len(router.model_list)} deployments, aliases "
              f"{sorted(set(router.get_model_names()))}", "the Router dropped a deployment")
    except Exception as e:  # noqa: BLE001
        check(False, "", f"LiteLLM Router rejected it: {type(e).__name__}: {str(e)[:200]}")
        sys.exit(1)
    out = Path(__file__).resolve().parents[1] / ".runs" / "ex08.config.yaml"
    out.parent.mkdir(exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"\n═ saved to week25/08_litellm_gateway/.runs/{out.name}. Try it live (it will fall back to your laptop):")
    print("  export LITELLM_MASTER_KEY=sk-$(openssl rand -hex 12)")
    print(f"  week25/.venv-litellm/bin/litellm --config week25/08_litellm_gateway/.runs/{out.name} --port 4000")


if __name__ == "__main__":
    main()
