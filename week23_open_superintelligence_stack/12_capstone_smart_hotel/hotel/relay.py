#!/usr/bin/env python3
"""NeMo Relay — observe, learn & optimize  (Week 23 · App 8 + App 9 economics).

Two jobs, exactly like the deck:
  • OBSERVE   — every agent turn / llm call / tool call is captured as a span.
                render_trace() prints a Phoenix-style trace tree with latency + cost.
  • OPTIMIZE  — the Router right-sizes the model per request (Nano for cheap/simple,
                Super for hard reasoning) → cost & latency win. Telemetry would export
                to Phoenix / Datadog / LangSmith over OTel.

Also accounts inference economics (App 9): tokens, cost/M-token, and a rough
energy figure — so "usefulness" is measured, not just token count.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# illustrative price/perf for the two Nemotron tiers served by the NIM (per 1M tokens, tok/s)
MODELS = {
    "nemotron-3-nano:30b-a3b":  {"mtok_usd": 0.10, "tok_s": 54, "watts": 240},
    "nemotron-3-super:120b-a12b": {"mtok_usd": 0.60, "tok_s": 20, "watts": 700},
}
NANO = "nemotron-3-nano:30b-a3b"
SUPER = "nemotron-3-super:120b-a12b"


@dataclass
class Span:
    kind: str        # agent | llm | tool
    name: str
    model: str = ""
    tokens: int = 0
    ms: int = 0
    ok: bool = True
    depth: int = 0
    detail: str = ""


class Relay:
    def __init__(self):
        self.spans: list[Span] = []
        self.exporters = ["Phoenix", "Datadog", "LangSmith"]

    # ── OPTIMIZE: right-size the model for the request ────────────────────────
    def route(self, difficulty: str) -> str:
        """difficulty ∈ 'simple'|'hard' → the cheapest model that can do it."""
        return SUPER if difficulty == "hard" else NANO

    # ── OBSERVE: record spans ─────────────────────────────────────────────────
    def span(self, kind, name, *, model="", tokens=0, ms=0, ok=True, depth=0, detail="") -> Span:
        sp = Span(kind, name, model, tokens, ms, ok, depth, detail)
        self.spans.append(sp)
        return sp

    # ── economics (App 9) ────────────────────────────────────────────────────
    def totals(self) -> dict:
        toks = sum(s.tokens for s in self.spans if s.kind == "llm")
        cost = sum(s.tokens / 1_000_000 * MODELS.get(s.model, {}).get("mtok_usd", 0)
                   for s in self.spans if s.kind == "llm")
        ms = sum(s.ms for s in self.spans)
        # rough energy: tokens / tok_s * watts, summed per llm span
        wh = 0.0
        for s in self.spans:
            if s.kind == "llm" and s.model in MODELS:
                m = MODELS[s.model]
                wh += (s.tokens / max(m["tok_s"], 1)) * m["watts"] / 3600.0
        nano = sum(1 for s in self.spans if s.model == NANO)
        supr = sum(1 for s in self.spans if s.model == SUPER)
        return {"llm_tokens": toks, "cost_usd": round(cost, 5), "latency_ms": ms,
                "energy_wh": round(wh, 2), "nano_calls": nano, "super_calls": supr}

    # ── Phoenix-style trace tree ──────────────────────────────────────────────
    def render_trace(self, title="hotel-ops turn") -> list[str]:
        t = self.totals()
        out = [f"▤ PHOENIX TRACE · {title}",
               f"  status ✓  · total cost ${t['cost_usd']:.5f} · latency {t['latency_ms']}ms "
               f"· {t['llm_tokens']} tok · ~{t['energy_wh']} Wh",
               "  " + "─" * 58]
        for s in self.spans:
            pad = "   " * s.depth
            tag = {"agent": "▣", "llm": "◆", "tool": "→"}.get(s.kind, "·")
            mark = "✓" if s.ok else "✗"
            extra = f"  [{s.model.split(':')[0]}·{s.tokens}tok]" if s.kind == "llm" else ""
            det = f"  {s.detail}" if s.detail else ""
            out.append(f"  {pad}{tag} {mark} {s.kind}:{s.name}{extra}{det}  ({s.ms}ms)")
        out.append(f"  → telemetry exported (OTel) to: {', '.join(self.exporters)}")
        return out
