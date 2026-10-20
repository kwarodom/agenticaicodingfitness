#!/usr/bin/env python3
"""PART 3 · Call a NIM — same OpenAI API, many ways  [INTERMEDIATE]

A NIM is a drop-in OpenAI endpoint. curl, the OpenAI SDK, streaming — identical to
the cloud, just pointed at your DGX. This demo shows the calls and runs one live
(or simulated) generation against the connected endpoint.

Run:  python demos/step03_call_nim.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import view  # noqa: E402

CALLS = """\
# curl
curl http://<dgx>:8000/v1/chat/completions -H "Content-Type: application/json" \\
  -d '{"model":"nemotron-3-super","messages":[{"role":"user","content":"Hello"}]}'

# Python — OpenAI SDK, only base_url changes
from openai import OpenAI
client = OpenAI(base_url="http://<dgx>:8000/v1", api_key="not-needed")
client.chat.completions.create(model="nemotron-3-super", messages=[...], stream=True)

# list what the NIM serves
curl http://<dgx>:8000/v1/models
"""


def main() -> None:
    view.banner("PART 3", "Call a NIM — same OpenAI API", "INTERMEDIATE")
    view.mode_line()
    print("Every NIM speaks the OpenAI API — your existing code works unchanged:\n")
    print(CALLS)
    print(f"Live call against the connected endpoint ({config.MODEL}):\n")
    view.generate("In two sentences, why does an OpenAI-compatible NIM make it easy to "
                  "swap a cloud model for a sovereign one?", max_tokens=300,
                  title="calling the endpoint")
    print("\nTakeaway: 'sovereign' isn't a different API — it's the SAME API, served from")
    print("your DGX. Apps don't know the difference; your data does. Next: the catalog.")


if __name__ == "__main__":
    main()
