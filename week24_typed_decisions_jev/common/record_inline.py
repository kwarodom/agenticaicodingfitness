#!/usr/bin/env python3
"""Record every inline block in week24/*/TUTORIAL.md and TUTORIAL.th.md so DRY mode can replay it.

  ```jev  {"state":…, "questions":…}              → one live Jev call (jev-1.13.0)
  ```llm  {"system":…, "user":…, "max_tokens":…}  → one call per LLM provider that has a key
                                                    (each provider's default model)

Maintainer tool; needs keys. Recordings never contain keys.
    .venv/bin/python week24/common/record_inline.py            # all modules
    .venv/bin/python week24/common/record_inline.py 11_jev_plus_llm
"""
import json
import os
import sys
from pathlib import Path

os.environ["JEV_RECORD"] = "1"
os.environ.setdefault("JEV_MODE", "live")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jevkit  # noqa: E402
import llmkit  # noqa: E402


def blocks(md: str):
    out, buf, lang = [], [], None
    for line in md.splitlines():
        t = line.strip()
        if lang is None and (t.startswith("```jev") or t.startswith("```llm")):
            lang, buf = t[3:].strip(), []
        elif lang is not None and t.startswith("```"):
            out.append((lang, "\n".join(buf)))
            lang = None
        elif lang is not None:
            buf.append(line)
    return out


def main():
    only = set(sys.argv[1:])
    n = bad = 0
    files = sorted(jevkit.WEEK24.glob("[0-9][0-9]_*/TUTORIAL.md")) + sorted(jevkit.WEEK24.glob("[0-9][0-9]_*/TUTORIAL.th.md"))
    seen = set()
    for tut in files:
        if only and tut.parent.name not in only:
            continue
        for i, (lang, raw) in enumerate(blocks(tut.read_text(encoding="utf-8")), 1):
            if (lang, raw) in seen:
                continue
            seen.add((lang, raw))
            tag = f"{tut.parent.name}/{tut.name} {lang} block {i}"
            try:
                spec = json.loads(raw)
                if lang == "jev":
                    resp = jevkit.ask(spec["state"], spec["questions"], model_id=spec.get("model"), quiet=True)
                    jevkit.validate(resp, spec["questions"])
                    n += 1
                    print(f"✓ {tag} · {resp['usage']['input_tokens']} tok")
                else:
                    for p in llmkit.providers():
                        if not p["has_key"] and not p["key_optional"]:
                            continue
                        try:
                            r = llmkit.generate(p["id"], system=spec.get("system", ""), user=spec["user"],
                                                max_tokens=int(spec.get("max_tokens", 400)))
                            n += 1
                            print(f"✓ {tag} · {p['id']} {r['usage']['output_tokens']} out tok")
                        except llmkit.LLMError as e:
                            print(f"⚠ {tag} · {p['id']}: {str(e)[:100]}")
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"✕ {tag}: {e}")
    print(f"═ recorded {n} call(s), {bad} failure(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
