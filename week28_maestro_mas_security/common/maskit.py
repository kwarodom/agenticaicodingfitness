#!/usr/bin/env python3
"""maskit — the Week 28 lab kit: output helpers + the two brains every agent can run on.

Week 28 teaches the security of multi-agent systems (MAS) with the CSA MAESTRO framework. Every lab
drives the same system, the **Hotel Ops Ensemble** (`ensemble.py`), and every agent in it thinks with a
"brain":

  • SIM brain (default) — a deterministic, offline, instant STAND-IN for a gullible LLM. It reads every
    piece of text in its context and obeys every instruction it finds there, whoever wrote it. That is the
    exact weakness prompt injection exploits, made reproducible so a lab gives the same answer every run.
    It is a TEACHING MODEL, not a measurement of any real model.
  • LIVE brain — a real local model on Ollama (http://localhost:11434), chosen with MAS_BRAIN=ollama and
    MAS_MODEL=<model>. Real models are less predictable: sometimes they resist an injection the SIM brain
    falls for, sometimes the reverse. The runner's 🧠 SIM / 🦙 LIVE switch sets these two variables.

Output glyphs the runner colours: ━ banner · ▣ step · ✓ pass · ✕ fail · ⚠ warn · ◈ note/dry · ◆ metric ·
═ result · → action · │ table · ⚔ attack · 🛡 defense.

    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
    from maskit import banner, step, ok, attack, defend, table
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

COMMON = Path(__file__).resolve().parent            # …/week28_maestro_mas_security/common
WEEK = COMMON.parent                                 # …/week28_maestro_mas_security
ROOT = WEEK.parent                                   # …/agenticaicodingfitness
SOURCES = COMMON / "sources"
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODELS = ("gemma4:12b", "gemma4:latest", "nemotron-3-nano:latest")   # gemma4:12b ≈ 11 s per tool call on an M4 Max


# ── output helpers (same look as Weeks 25–27) ─────────────────────────────────
def banner(title: str, sub: str = "", *, brain: bool = True) -> None:
    """Lab header. brain=False for labs and exercises that never run an agent."""
    print("━" * 72)
    print(f"━━ {title}")
    if sub:
        print(f"   {sub}")
    print("━" * 72)
    if brain:
        if brain_mode() == "ollama":
            print(f"▣ 🦙 LIVE brain · Ollama {brain_model()} · real model: results can differ run to run")
        else:
            print("◈ 🧠 SIM brain · deterministic stand-in for a gullible LLM (switch to 🦙 LIVE in the runner)")


def step(n, text: str) -> None:
    print(f"\n▣ STEP {n} · {text}", flush=True)


def ok(msg: str) -> None:
    print(f"✓ {msg}")


def fail(msg: str) -> None:
    print(f"✕ {msg}")


def warn(msg: str) -> None:
    print(f"⚠ {msg}")


def note(msg: str) -> None:
    print(f"◆ {msg}")


def dim(msg: str) -> None:
    print(f"◈ {msg}")


def result(msg: str) -> None:
    print(f"═ {msg}")


def attack(msg: str) -> None:
    print(f"⚔ {msg}", flush=True)


def defend(msg: str) -> None:
    print(f"🛡 {msg}", flush=True)


def table(rows: list[list], headers: list[str], maxw: int = 52) -> None:
    cols = [headers] + [[str(c) for c in r] for r in rows]
    widths = [min(max(len(r[i]) for r in cols), maxw) for i in range(len(headers))]
    clip = lambda s, w: s if len(s) <= w else s[:w - 1] + "…"          # noqa: E731
    fmt = lambda r: "  ".join(clip(c, widths[i]).ljust(widths[i]) for i, c in enumerate(r))  # noqa: E731
    print("│ " + fmt(headers))
    print("│ " + "  ".join("─" * w for w in widths))
    for r in cols[1:]:
        print("│ " + fmt(r))


def bar(value: float, vmax: float, width: int = 28) -> str:
    n = 0 if vmax <= 0 else max(0, min(width, round(width * value / vmax)))
    return "█" * n + "░" * (width - n)


def check(cond: bool, good: str, bad: str) -> bool:
    """Exercise checker line: ✓ or ✕."""
    print(("✓ " + good) if cond else ("✕ " + bad))
    return bool(cond)


def finish(good: bool, done_msg: str) -> None:
    """End an exercise checker: exit 1 with a hint when any check failed."""
    if not good:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)
    print(f"\n═ {done_msg}")


def runs_dir(lab_file: str | Path) -> Path:
    """<module>/.runs/ for artefacts a lab writes (gitignored)."""
    p = Path(lab_file).resolve()
    module = next(q for q in p.parents if re.match(r"^\d\d_", q.name))
    d = module / ".runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def source_text(name: str) -> str:
    """A snapshot of a cited web page in week28_maestro_mas_security/common/sources/ (what audit_references.py checks against)."""
    return (SOURCES / name).read_text(encoding="utf-8")


# ── brains ─────────────────────────────────────────────────────────────────────
def brain_mode() -> str:
    return "ollama" if os.environ.get("MAS_BRAIN", "sim").lower() in ("ollama", "live") else "sim"


def ollama_models() -> list[str]:
    try:
        with urlopen(OLLAMA + "/api/tags", timeout=2) as r:            # noqa: S310 — localhost
            return [m["name"] for m in json.loads(r.read()).get("models", []) if "cloud" not in m["name"]]
    except Exception:  # noqa: BLE001
        return []


def brain_model() -> str:
    m = os.environ.get("MAS_MODEL", "").strip()
    if m:
        return m
    have = ollama_models()
    return next((x for x in DEFAULT_MODELS if x in have), have[0] if have else DEFAULT_MODELS[0])


def ollama_chat(messages: list[dict], *, tools: list[dict] | None = None, model: str | None = None,
                max_tokens: int = 1024, temperature: float = 0.0, timeout: float = 240) -> dict:
    """One OpenAI-compatible chat call to local Ollama. Returns {"content", "tool_calls", "seconds"}."""
    body = {"model": model or brain_model(), "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "stream": False,
            "reasoning_effort": "none"}      # thinking models otherwise spend the whole budget thinking
    if tools:
        body["tools"] = tools
    t0 = time.time()
    req = Request(OLLAMA + "/v1/chat/completions", data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as r:                            # noqa: S310 — localhost
        data = json.loads(r.read())
    msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
    calls = []
    for c in msg.get("tool_calls") or []:
        fn = c.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        calls.append({"name": fn.get("name", ""), "args": args if isinstance(args, dict) else {}})
    return {"content": msg.get("content") or "", "tool_calls": calls, "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":                          # python maskit.py → which brain would a lab use?
    banner("maskit self-check", "which brain, which local models")
    print(f"◆ MAS_BRAIN={os.environ.get('MAS_BRAIN', '(unset → sim)')} · model={brain_model()}")
    print(f"◆ Ollama local models: {', '.join(ollama_models()) or 'none (Ollama not running?)'}")
