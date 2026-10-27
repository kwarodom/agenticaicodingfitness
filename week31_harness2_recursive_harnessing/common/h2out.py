#!/usr/bin/env python3
"""h2out — the Week 31 output helpers and the optional LIVE model.

Week 31 teaches Harness²: improving the harness around a frozen model (prompts, guardrails, memory, tool
guidance, subagents, skills, scripts, plugins) from execution evidence. Every lab drives a SIMULATED
operations agent (`ops_sim.py`) and the course's own Propose–Probe–Compose loop (`h2kit.py`).

  • SIM (default) — deterministic, offline, instant. The executor and the editors are TEACHING MODELS: plain
    Python rules that behave like an agent with known weaknesses. Same seed, same output.
  • LIVE — a real local model on Ollama (http://localhost:11434) plays ONE role in a lab (usually the editor),
    chosen with H2_BRAIN=ollama and H2_MODEL=<model>. The runner's 🧠 SIM / 🦙 LIVE switch sets both.

Output glyphs the runner colours: ━ banner · ▣ step · ✓ pass · ✕ fail · ⚠ warn · ◈ note/dry · ◆ metric ·
═ result · → action · │ table · ✎ edit · ⚖ verdict.

    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
    from h2out import banner, step, ok, table
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

COMMON = Path(__file__).resolve().parent            # …/week31_harness2_recursive_harnessing/common
WEEK = COMMON.parent                                 # …/week31_harness2_recursive_harnessing
ROOT = WEEK.parent                                   # …/agenticaicodingfitness
SOURCES = COMMON / "sources"
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODELS = ("gemma4:12b", "gemma4:latest", "gemma3:4b")   # gemma4:12b ≈ 10 s per call on an M4 Max


# ── output helpers (same look as Weeks 25–30) ─────────────────────────────────
def banner(title: str, sub: str = "", *, brain: bool = True) -> None:
    """Lab header. brain=False for labs and exercises that never call a model."""
    print("━" * 72)
    print(f"━━ {title}")
    if sub:
        print(f"   {sub}")
    print("━" * 72)
    if brain:
        if brain_mode() == "ollama":
            print(f"▣ 🦙 LIVE · Ollama {brain_model()} plays the model role in this lab · results can differ run to run")
        else:
            print("◈ 🧠 SIM · deterministic teaching model of an agent and its editors (switch to 🦙 LIVE in the runner)")


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


def edit(msg: str) -> None:
    print(f"✎ {msg}", flush=True)


def verdict(msg: str) -> None:
    print(f"⚖ {msg}", flush=True)


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


def show_text(title: str, text: str, max_lines: int = 40) -> None:
    """Print a file's text indented, with a header line (used to show harness files)."""
    print(f"┌─ {title}")
    lines = text.rstrip("\n").splitlines() or ["(empty)"]
    for ln in lines[:max_lines]:
        print(f"│ {ln}")
    if len(lines) > max_lines:
        print(f"│ … {len(lines) - max_lines} more lines")
    print("└─")


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
    """A saved source in common/sources/ (what audit_references.py checks quotes against)."""
    return (SOURCES / name).read_text(encoding="utf-8")


# ── the optional LIVE model ───────────────────────────────────────────────────
def brain_mode() -> str:
    return "ollama" if os.environ.get("H2_BRAIN", "sim").lower() in ("ollama", "live") else "sim"


def ollama_models() -> list[str]:
    try:
        with urlopen(OLLAMA + "/api/tags", timeout=2) as r:            # noqa: S310 — localhost
            return [m["name"] for m in json.loads(r.read()).get("models", []) if "cloud" not in m["name"]]
    except Exception:  # noqa: BLE001
        return []


def brain_model() -> str:
    m = os.environ.get("H2_MODEL", "").strip()
    if m:
        return m
    have = ollama_models()
    return next((x for x in DEFAULT_MODELS if x in have), have[0] if have else DEFAULT_MODELS[0])


_LIVE_CALLS = 0
LIVE_CALL_CAP = 6                                    # a lab never makes more than this many model calls


def ollama_chat(messages: list[dict], *, model: str | None = None, max_tokens: int = 1200,
                temperature: float = 0.0, timeout: float = 300) -> dict:
    """One OpenAI-compatible chat call to local Ollama. Returns {"content", "seconds", "model"}."""
    global _LIVE_CALLS
    if _LIVE_CALLS >= LIVE_CALL_CAP:
        raise RuntimeError(f"LIVE call cap reached ({LIVE_CALL_CAP} per lab)")
    _LIVE_CALLS += 1
    body = {"model": model or brain_model(), "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "stream": False,
            "reasoning_effort": "none"}      # thinking models otherwise spend the whole budget thinking
    t0 = time.time()
    req = Request(OLLAMA + "/v1/chat/completions", data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as r:                            # noqa: S310 — localhost
        data = json.loads(r.read())
    msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
    return {"content": msg.get("content") or "", "seconds": round(time.time() - t0, 1), "model": body["model"]}


if __name__ == "__main__":                          # python h2out.py → which mode would a lab use?
    banner("h2out self-check", "which mode, which local models")
    print(f"◆ H2_BRAIN={os.environ.get('H2_BRAIN', '(unset → sim)')} · model={brain_model()}")
    print(f"◆ Ollama local models: {', '.join(ollama_models()) or 'none (Ollama not running?)'}")
