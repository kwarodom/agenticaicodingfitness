#!/usr/bin/env python3
"""Lab 16-0 · One-time setup: download the four local models (≈ 3 GB) into the Hugging Face cache.

Run this ONCE from the ⌨ terminal, not from a ▶ Run button (it can take longer than the 150 s lab limit):
    week24_typed_decisions_jev/.venv-local/bin/python week24_typed_decisions_jev/16_local_system_one_models/labs/lab00_download_models.py

  convaiinnovations/laya            English (root, ~808 MB) + multilingual subfolder (~647 MB)
  fastino/GLiNER2.5-Decide          340M, English decisions
  fastino/GLiNER2.5-multi-Decide    287M, multilingual decisions (Thai)
  urchade/gliner_multi-v2.1         classic GLiNER span extractor, multilingual
After this, every Module 16 lab loads offline in a few seconds.
"""
import os
import sys
import time
from pathlib import Path

os.environ["LOCALKIT_ALLOW_DOWNLOAD"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import localkit as L  # noqa: E402

print("━" * 72 + "\n━━ Lab 16-0 · download local System One models\n" + "━" * 72)
L.banner_local()
if not L.where():
    print("✕ laya / gliner2 are not installed. Create week24_typed_decisions_jev/.venv-local first — see the top of localkit.py.")
    sys.exit(1)

t0 = time.time()
import laya  # noqa: E402

for sub in (None, "multilingual"):
    name = f"{L.LAYA_REPO}{'/' + sub if sub else ''}"
    t = time.time()
    agent = laya.load(L.LAYA_REPO, subfolder=sub) if sub else laya.load(L.LAYA_REPO)
    print(f"✓ {name:<40} {time.time() - t:6.1f} s")
    del agent
from gliner2 import AutoExtractor  # noqa: E402

for repo in (L.DECIDE_EN, L.DECIDE_MULTI):
    t = time.time()
    AutoExtractor.from_pretrained(repo)
    print(f"✓ {repo:<40} {time.time() - t:6.1f} s")
from gliner import GLiNER  # noqa: E402

t = time.time()
GLiNER.from_pretrained(L.GLINER_NER)
print(f"✓ {L.GLINER_NER:<40} {time.time() - t:6.1f} s")
print(f"\n◆ done in {time.time() - t0:.0f} s — the Module 16 labs now run offline.")
