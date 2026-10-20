#!/usr/bin/env python3
"""Exercise 05 · Telemetry: two exporters, one collector rule, one decorator.

Fill in the three TODOs, save, then run:
    .venv/bin/python week26/05_tracing/exercises/ex05_telemetry.py

The checker is free and offline (no model calls, no network). It uses the course's policykit model and, if
week26/.venv-nat exists, the real `nat validate` and the installed NAT source. Stuck? Compare with
exercises/solutions/.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import policykit as pk  # noqa: E402
import yaml  # noqa: E402
from clawkit import NAT, ROOT, WEEK, banner, check, laptop  # noqa: E402

# ── TODO 1 ── (Part 4 exercise 1) a `general.telemetry.tracing` block with TWO exporters at once:
#   a `phoenix` exporter (endpoint http://localhost:6006/v1/traces, project alto-ops-claw) AND a `file_backup`
#   file exporter. Use the keys NAT 1.9.0 really requires (lab 05-1 printed them) — not the research tutorial's
#   `# path etc.`
TELEMETRY_YAML = """
general:
  telemetry:
    tracing:
      phoenix:
        _type: phoenix
        endpoint: http://localhost:6006/v1/traces
        project: alto-ops-claw
      file_backup:
        _type: file
        # path etc.
"""

# ── TODO 2 ── (Part 4 exercise 3) a sandboxed NAT exports to otel.alto.local:4318 and the collector logs 403.
#   Fix this endpoint entry so an OTLP/HTTP export (POST /v1/traces from /usr/bin/python3.12) is allowed —
#   and keep `enforcement: enforce` (switching back to audit only hides the problem).
COLLECTOR_ENDPOINT = {
    "host": "otel.alto.local",
    "port": 4318,
    "protocol": "rest",
    "enforcement": "enforce",
    "access": "read-only",
}

# ── TODO 3 ── (Part 4 exercise 5) the NAT decorator that traces an arbitrary Python function that is not a
#   registered NAT function: its full import path ("package.module.name"), and the three event types it emits.
DECORATOR = ""          # e.g. "nat.some.module.decorator_name"
EVENTS = set()          # e.g. {"…", "…", "…"}


# ─────────────────────────── checker — no need to edit below ────────────────
PY = "/usr/bin/python3.12"
TRACKING_SRC = WEEK / ".venv-nat" / "lib" / "python3.12" / "site-packages" / "nat" / "plugins" / "profiler" / \
    "decorators" / "function_tracking.py"
BASE = ROOT / "week26" / "common" / "alto_ops" / "src" / "alto_ops" / "configs" / "workflow.laptop.yml"


def check_telemetry() -> bool:
    try:
        doc = yaml.safe_load(TELEMETRY_YAML) or {}
        tracing = doc["general"]["telemetry"]["tracing"] or {}
    except Exception as e:  # noqa: BLE001
        return check(False, "", f"TODO 1: TELEMETRY_YAML does not parse to general.telemetry.tracing ({e})")
    by_type = {}
    for name, ex in tracing.items():
        by_type.setdefault((ex or {}).get("_type"), []).append((name, ex or {}))
    ph = by_type.get("phoenix", [(None, {})])[0][1]
    fi = by_type.get("file", [(None, {})])[0][1]
    shape = (len(tracing) >= 2 and "/v1/traces" in str(ph.get("endpoint", "")) and ph.get("project")
             and fi.get("output_path") and fi.get("project"))
    good = check(bool(shape), f"telemetry: {len(tracing)} exporters — phoenix + file (output_path, project)",
                 "TODO 1: need a phoenix exporter (endpoint …/v1/traces + project) AND a file exporter with the two "
                 "keys NAT 1.9 requires: output_path and project")
    if good and NAT.exists() and BASE.is_file():
        cfg = yaml.safe_load(BASE.read_text(encoding="utf-8"))
        cfg["general"] = doc["general"]
        with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False, encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        r = laptop([NAT, "validate", "--config_file", f.name], quiet=True, timeout=120,
                   show="nat validate --config_file <your block + workflow.laptop.yml>")
        Path(f.name).unlink(missing_ok=True)
        good &= check(r.ok, "the real `nat validate` (NAT 1.9.0) accepts your block",
                      "TODO 1: `nat validate` rejects it: " + next((ln.strip() for ln in r.out.splitlines()
                                                                  if "Invalid configuration" in ln), r.out[-200:]))
    return good


def check_collector() -> bool:
    ep = dict(COLLECTOR_ENDPOINT)
    policy = pk.build_policy(read_only=["/usr", "/lib", "/etc"], read_write=["/tmp"],
                             groups={"otel_collector": pk.group("otel_collector", [ep], [PY])})
    errs, _ = pk.validate(policy)
    post = {"op": "http", "host": "otel.alto.local", "port": 4318, "binary": PY, "method": "POST", "path": "/v1/traces"}
    d, why = pk.decide(policy, post)
    enforced = ep.get("enforcement") == "enforce"
    tight = ep.get("access") != "full" and ep.get("protocol") == "rest" and ep.get("host") == "otel.alto.local"
    return check(not errs and d == "allow" and enforced and tight and "audit" not in why,
                 "collector: POST /v1/traces allowed under enforce (rest, otel.alto.local, not `full`)",
                 f"TODO 2: policykit says {d} — {why}" + (f" · validate: {errs[0]}" if errs else "") +
                 ("" if enforced else " · keep enforcement: enforce") +
                 ("" if tight else " · keep protocol rest + host otel.alto.local, and do not use access: full"))


def check_decorator() -> bool:
    want = "nat.plugins.profiler.decorators.function_tracking.track_function"
    src = TRACKING_SRC.read_text(encoding="utf-8") if TRACKING_SRC.is_file() else ""
    in_src = (not src) or ("def track_function" in src and all(f"IntermediateStepType.{e}" in src for e in EVENTS))
    ok_path = DECORATOR.strip() == want
    ok_events = {e.upper() for e in EVENTS} == {"SPAN_START", "SPAN_CHUNK", "SPAN_END"}
    return check(ok_path and ok_events and in_src,
                 f"decorator: @track_function from nat.plugins.profiler.decorators.function_tracking · "
                 f"SPAN_START / SPAN_CHUNK (generators) / SPAN_END" + (" — found in the installed NAT 1.9.0" if src else ""),
                 f"TODO 3: DECORATOR={DECORATOR!r} EVENTS={sorted(EVENTS) or '{}'} — look in "
                 "nat/plugins/profiler/decorators/ in week26/.venv-nat (grep for 'def track_function' and "
                 "'IntermediateStepType.')")


def main() -> None:
    banner("Exercise 05 · telemetry", "offline checker · free · no Spark, no model calls", status=False)
    good = check_telemetry()
    good &= check_collector()
    good &= check_decorator()
    if not good:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)
    print("\n═ Done. Two exporters, one collector rule that survives enforce, one decorator for your own code.")


if __name__ == "__main__":
    main()
