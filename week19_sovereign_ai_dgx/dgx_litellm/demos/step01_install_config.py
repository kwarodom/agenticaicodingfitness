#!/usr/bin/env python3
"""PART 1 · Install LiteLLM & write the config  [BEGINNER]

LiteLLM gives you ONE OpenAI-compatible gateway in front of every model you serve
on the DGX. You describe your backends in a config.yaml `model_list` (friendly
alias → real backend), then `litellm --config` starts the proxy on :4000. This
demo writes a REAL config into .sandbox/ that maps aliases to your DGX runtimes,
prints the install + launch commands, and makes one gateway call.

Run:  python demos/step01_install_config.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import liteview  # noqa: E402

CONFIG_YAML = """\
# litellm_config.yaml — one gateway in front of all DGX backends (sovereign)
model_list:
  - model_name: dgx-fast                 # friendly alias your apps call
    litellm_params:
      model: ollama/qwen3.6:35b-a3b-q8_0 # 35B MoE on Ollama (Spark)
      api_base: {backend}
  - model_name: dgx-smart
    litellm_params:
      model: openai/llama3.3:70b-nvfp4    # a vLLM endpoint (OpenAI-compatible)
      api_base: http://localhost:8000/v1
  - model_name: dgx-tiny
    litellm_params:
      model: openai/phi3.5:3.8b           # llama.cpp server for cheap/fast work
      api_base: http://localhost:8080/v1

litellm_settings:
  drop_params: true                       # tolerate provider param differences
  num_retries: 2
"""

COMMANDS = """\
# install + run the LiteLLM proxy on the DGX
pip install 'litellm[proxy]'
litellm --config litellm_config.yaml --port 4000
# → an OpenAI-compatible gateway is now on http://localhost:4000

# point ANY app at it (one URL, one key, every backend):
export LITELLM_BASE_URL=http://localhost:4000
"""


def main() -> None:
    liteview.banner("PART 1", "Install LiteLLM & write the config", "BEGINNER")
    liteview.mode_line()

    sb = config.ensure_sandbox()
    yaml = CONFIG_YAML.format(backend=config.BACKEND_URL)
    (sb / "litellm_config.yaml").write_text(yaml)
    print("Wrote a real gateway config → .sandbox/litellm_config.yaml\n")
    print(yaml)
    print("Install and launch it:\n")
    print(COMMANDS)

    print("Reading the config:")
    print("  • model_list maps a FRIENDLY alias (dgx-fast) → a real backend.")
    print("  • The same file can point at Ollama, vLLM, llama.cpp, TRT-LLM, NIM —")
    print("    any OpenAI-compatible local endpoint. One gateway, many runtimes.")
    print("  • Your apps only ever learn the aliases; you can swap backends underneath.\n")

    print("One call through the gateway:")
    liteview.generate("In one sentence, what does a LiteLLM gateway give a DGX deployment?",
                      alias="dgx-fast", max_tokens=200)

    print("\nTakeaway: LiteLLM turns a pile of DGX runtimes into ONE stable, OpenAI-")
    print("compatible URL. Next: call several models through that single endpoint.")


if __name__ == "__main__":
    main()
