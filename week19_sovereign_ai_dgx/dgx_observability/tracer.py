#!/usr/bin/env python3
"""A tiny **OpenTelemetry-shaped tracer** for sovereign agents.

This is a teaching-sized version of what `openinference-instrumentation` +
OpenTelemetry emit and what Arize Phoenix renders. Each span carries the
OTel GenAI semantic-convention attributes (gen_ai.*), nests under a parent, and
records latency + token usage + status. `render_tree()` prints the waterfall the
way Phoenix shows it; `to_phoenix()` shows how you'd export for real.

Keeping it dependency-free means the whole app runs in SIM with no GPU and no
Phoenix install — yet the span SHAPE is exactly what you'd send to real Phoenix.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

# OpenInference span kinds (what Phoenix groups by).
LLM, TOOL, CHAIN, AGENT, RETRIEVER = "LLM", "TOOL", "CHAIN", "AGENT", "RETRIEVER"
_SYM = {LLM: "◇", TOOL: "▸", CHAIN: "▤", AGENT: "◆", RETRIEVER: "▦"}


@dataclass
class Span:
    name: str
    kind: str
    span_id: int
    parent_id: int | None
    start: float
    end: float | None = None
    attributes: dict = field(default_factory=dict)
    status: str = "OK"            # OK | ERROR
    depth: int = 0

    @property
    def latency_ms(self) -> float:
        return ((self.end or self.start) - self.start) * 1000


class Tracer:
    """Collects spans during one agent run; render or export afterward."""

    def __init__(self, project: str = "sovereign-agent"):
        self.project = project
        self.spans: list[Span] = []
        self._stack: list[Span] = []
        self._next = 1

    def span(self, name: str, kind: str, **attributes):
        return _SpanCtx(self, name, kind, attributes)

    # internal -----------------------------------------------------------------
    def _open(self, name, kind, attributes) -> Span:
        s = Span(name=name, kind=kind, span_id=self._next,
                 parent_id=self._stack[-1].span_id if self._stack else None,
                 start=time.time(), attributes=dict(attributes),
                 depth=len(self._stack))
        self._next += 1
        self.spans.append(s)
        self._stack.append(s)
        return s

    def _close(self, s: Span, status: str = "OK"):
        s.end = time.time()
        s.status = status
        if self._stack and self._stack[-1] is s:
            self._stack.pop()

    # reporting ----------------------------------------------------------------
    def totals(self) -> dict:
        inp = sum(s.attributes.get("gen_ai.usage.input_tokens", 0) for s in self.spans)
        out = sum(s.attributes.get("gen_ai.usage.output_tokens", 0) for s in self.spans)
        llm = [s for s in self.spans if s.kind == LLM]
        tools = [s for s in self.spans if s.kind == TOOL]
        root = self.spans[0].latency_ms if self.spans else 0.0
        return {"spans": len(self.spans), "llm_calls": len(llm), "tool_calls": len(tools),
                "input_tokens": inp, "output_tokens": out,
                "total_tokens": inp + out, "latency_ms": root,
                "errors": sum(1 for s in self.spans if s.status == "ERROR")}

    def render_tree(self) -> str:
        out = [f"  Phoenix project: {self.project}   (spans render top-down, like the UI)"]
        for s in self.spans:
            pad = "    " + "  " * s.depth
            sym = _SYM.get(s.kind, "•")
            line = f"{pad}{sym} [{s.kind}] {s.name}  ·  {s.latency_ms:.0f} ms"
            tk = ""
            if s.attributes.get("gen_ai.usage.output_tokens"):
                tk = (f"  ·  {s.attributes.get('gen_ai.usage.input_tokens',0)}→"
                      f"{s.attributes['gen_ai.usage.output_tokens']} tok")
            st = "" if s.status == "OK" else "  ·  ERROR ✗"
            out.append(line + tk + st)
            # show a couple of key attributes the way Phoenix surfaces them
            for k in ("gen_ai.request.model", "tool.name", "input.value", "output.value"):
                if k in s.attributes:
                    v = str(s.attributes[k])
                    if len(v) > 70:
                        v = v[:69] + "…"
                    out.append(f"{pad}    {k} = {v}")
        return "\n".join(out)

    def to_phoenix_hint(self) -> str:
        return (
            "# Export these spans to a REAL Phoenix instance (one-time setup):\n"
            "pip install arize-phoenix openinference-instrumentation-openai\n"
            "python -m phoenix.server.main serve        # UI at http://localhost:6006\n\n"
            "from phoenix.otel import register\n"
            "from openinference.instrumentation.openai import OpenAIInstrumentor\n"
            "tracer_provider = register(project_name='sovereign-agent')  # → :6006\n"
            "OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)\n"
            "# now EVERY OpenAI-SDK call to your DGX is auto-traced into Phoenix."
        )


class _SpanCtx:
    def __init__(self, tracer, name, kind, attributes):
        self.t, self.name, self.kind, self.attrs = tracer, name, kind, attributes
        self.span: Span | None = None

    def __enter__(self) -> Span:
        self.span = self.t._open(self.name, self.kind, self.attrs)
        return self.span

    def __exit__(self, exc_type, exc, tb):
        self.t._close(self.span, status="ERROR" if exc_type else "OK")
        return False
