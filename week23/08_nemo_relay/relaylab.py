#!/usr/bin/env python3
"""REAL NeMo Relay → Phoenix instrumentation for App 08.

When `nemo-relay` is installed AND a local Phoenix (`arize-phoenix`) is reachable,
this turns the SIM "observe" story into **real OpenInference spans** you can inspect
in the Phoenix trace tree at http://localhost:6006. Otherwise the demos fall back to
SIM, so a laptop with no GPU still learns every concept — $0, nothing required.

One-time setup on the DGX (already done on spark-3b82)::

    uv pip install nemo-relay arize-phoenix
    python -m phoenix.server.main serve        # UI + OTLP collector on :6006

Then any App 08 demo auto-detects Phoenix and emits real spans. The real Relay API
(v0.4) is a scope-stack + subscriber model, not the `Relay(...).instrument()` sketch
the SIM once showed:

    import nemo_relay as R
    cfg = R.OpenInferenceConfig(); cfg.endpoint = "http://localhost:6006/v1/traces"
    cfg.set_resource_attribute("openinference.project.name", "hermes-agent")
    R.OpenInferenceSubscriber(cfg).register("openinference")   # export every scope
    with R.scope.scope("hermes-turn", R.ScopeType.Agent): ...   # each scope = a span
"""
from __future__ import annotations

import os

PHOENIX_URL = os.environ.get("PHOENIX_URL", "http://localhost:6006").rstrip("/")
OTLP_ENDPOINT = PHOENIX_URL + "/v1/traces"
PROJECT = os.environ.get("RELAY_PROJECT", "hermes-agent")


def phoenix_up(timeout: float = 2.0) -> bool:
    """True if a local Phoenix collector answers (UI + OTLP live on the same port)."""
    import urllib.request
    try:
        urllib.request.urlopen(PHOENIX_URL, timeout=timeout)
        return True
    except Exception:
        return False


def relay_installed() -> bool:
    try:
        import nemo_relay  # noqa: F401
        return True
    except Exception:
        return False


def ready() -> bool:
    """REAL Relay is possible only with the package installed AND Phoenix running."""
    return relay_installed() and phoenix_up()


def status_line() -> str:
    if not relay_installed():
        return "Relay: SIM (nemo-relay not installed — `uv pip install nemo-relay`)"
    if not phoenix_up():
        return (f"Relay: SIM (Phoenix not reachable at {PHOENIX_URL} — "
                "`python -m phoenix.server.main serve`)")
    return f"Relay: REAL → OpenInference spans exported to Phoenix ({PHOENIX_URL}, project '{PROJECT}')"


_SUB = None  # one registered subscriber per process, shared by all demos


def _subscriber(project: str):
    """Configure + register an OpenInference subscriber that exports to Phoenix."""
    import nemo_relay as R
    cfg = R.OpenInferenceConfig()
    cfg.endpoint = OTLP_ENDPOINT
    cfg.service_name = project
    cfg.set_resource_attribute("openinference.project.name", project)   # routes to a named project
    sub = R.OpenInferenceSubscriber(cfg)
    sub.register("openinference")
    return sub


def instrument(project: str = PROJECT):
    """Register the Phoenix subscriber once; safe to call from every demo."""
    global _SUB
    if _SUB is None:
        _SUB = _subscriber(project)
    return _SUB


def flush() -> None:
    if _SUB is not None:
        _SUB.force_flush()


def record_llm(name: str, model: str, prompt: str, call_fn) -> tuple[str, float]:
    """Wrap a REAL model call in an LLM span and return (answer, latency_ms).

    `call_fn(prompt, model) -> str` performs the actual inference; the span (with
    the chosen model) is exported to Phoenix. Used by the router demo (step03).
    """
    import time
    import nemo_relay as R
    instrument()
    h = R.scope.push(name, R.ScopeType.Llm, input={"prompt": prompt}, metadata={"model": model})
    t0 = time.time()
    answer = (call_fn(prompt, model) or "").strip()
    dt_ms = (time.time() - t0) * 1000.0
    R.scope.pop(h, output={"text": answer[:400], "model": model, "latency_ms": round(dt_ms, 1)})
    flush()
    return answer, dt_ms


def fetch_spans(project: str = PROJECT) -> list[dict]:
    """Read the real spans Phoenix has collected for `project`, as a latency-sorted
    tree-ish list: [{name, kind, status, latency_ms, span_id, parent_id, trace_id}]."""
    from phoenix.client import Client
    df = Client(base_url=PHOENIX_URL).spans.get_spans_dataframe(project_name=project)
    if not len(df):
        return []
    out = []
    for _, r in df.iterrows():
        lat = (r["end_time"] - r["start_time"]).total_seconds() * 1000.0
        out.append({
            "name": r["name"],
            "kind": str(r.get("span_kind", "")).lower(),
            "status": "✗" if str(r.get("status_code")) == "ERROR" else "✓",
            "latency_ms": round(lat, 1),
            "span_id": r.get("context.span_id"),
            "parent_id": r.get("parent_id"),
            "trace_id": r.get("context.trace_id"),
        })
    return sorted(out, key=lambda s: s["latency_ms"], reverse=True)


def trace_stats(project: str = PROJECT) -> dict:
    """Real counts for the export/loop demo: spans, traces, total measured latency."""
    from phoenix.client import Client
    df = Client(base_url=PHOENIX_URL).spans.get_spans_dataframe(project_name=project)
    if not len(df):
        return {"spans": 0, "traces": 0, "total_latency_ms": 0.0}
    lat = ((df["end_time"] - df["start_time"]).dt.total_seconds() * 1000.0).sum()
    return {
        "spans": int(len(df)),
        "traces": int(df["context.trace_id"].nunique()),
        "total_latency_ms": round(float(lat), 1),
    }


def observed_turn(llm_call, *, project: str = PROJECT) -> list[dict]:
    """Run ONE Hermes agent turn as a real Relay span tree, exported to Phoenix.

    `llm_call(prompt) -> str` performs the real DGX inference recorded by the LLM
    span. Returns the captured spans (for console printing). Call only when ready().
    """
    import nemo_relay as R
    sub = instrument(project)
    spans: list[dict] = []

    ag = R.scope.push("hermes-turn", R.ScopeType.Agent, input={"task": "run the repo tests"})
    spans.append({"kind": "agent", "name": "hermes-turn", "in": "run the repo tests", "out": ""})

    t1 = R.scope.push("tool:terminal", R.ScopeType.Tool, input={"cmd": "ls ./repo && cat TODO.md"})
    R.scope.pop(t1, output={"stdout": "3 files · TODO.md (412 B)"})
    spans.append({"kind": "tool", "name": "tool:terminal",
                  "in": "ls ./repo && cat TODO.md", "out": "3 files · TODO.md (412 B)"})

    prompt = ("In two sentences, why must a long-running agent record every tool and "
              "LLM call as telemetry before you can improve it?")
    llm = R.scope.push("llm:plan", R.ScopeType.Llm, input={"prompt": prompt})
    answer = (llm_call(prompt) or "").strip()
    R.scope.pop(llm, output={"text": answer[:400]})
    spans.append({"kind": "llm", "name": "llm:plan", "in": prompt, "out": answer})

    t2 = R.scope.push("tool:execute_code", R.ScopeType.Tool, input={"cmd": "pytest -q"})
    R.scope.pop(t2, output={"result": "12 passed in 1.8s"})
    spans.append({"kind": "tool", "name": "tool:execute_code",
                  "in": "pytest -q", "out": "12 passed in 1.8s"})

    R.scope.pop(ag, output={"status": "done"})
    sub.force_flush()
    return spans


if __name__ == "__main__":
    print(status_line())
