#!/usr/bin/env python3
"""Lab 08-3 · Evidence and grade: mark Alto Ops Claw v1 against the capstone scheme. It never awards points it cannot prove.

The research tutorial's marking scheme: six deliverables × 15 points + 10 bonus. This grader splits each
deliverable into what THIS LAPTOP can verify (the bundle, the configs, the policy model, the laptop stand-in
runs from labs 08-1/08-2/08-4) and what only the Spark can prove (a sandbox that exists, a 403 in the
logs, Phoenix traces, eval and sizing files, the sandbox tax). The rules are fixed:

  • Spark points need Spark evidence: a LIVE read-only command, or a RECORDED transcript of the lab's exact
    command (week26/common/recorded/, written with SPARK_RECORD=1). An EXAMPLE never counts. In DRY every
    Spark-only item is "not yet evidenced", and it says why.
  • A laptop point is awarded only when the file it needs exists, and (for the bundle) still matches its
    MANIFEST.json hashes.
  • The bonus ("a policy OpenShell's prover accepts with no new credentialed hosts") is never auto-awarded:
    the grader shows the laptop pre-check, and a human awards it from the prover's own output.

No LLM calls, no servers. Run: .venv/bin/python week26/08_capstone_alto_ops_claw/labs/lab08_3_evidence_and_grade.py
"""
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capkit as ck  # noqa: E402
import clawkit  # noqa: E402
import policykit as pk  # noqa: E402
from clawkit import ROOT, banner, note, ok, result, sh, step, table, warn, where  # noqa: E402

banner("Lab 08-3 · evidence and grade", "laptop evidence from .runs/ · Spark evidence LIVE or RECORDED only")
SCORE = []                       # (deliverable, item, points_awarded, points_possible, plane, evidence)


def award(d: str, item: str, pts: int, got: bool, plane: str, ev: str) -> None:
    SCORE.append((d, item, pts if got else 0, pts, plane, ev))
    glyph = "◈" if ev.startswith("◈") else ("·" if pts == 0 else "✓" if got else "✕")
    print(f"{glyph} [{d}] {pts if got else 0}/{pts} · {item}\n    {ev.lstrip('◈ ')}")


# ── 0 · what evidence exists? ──────────────────────────────────────────────────────────────────────────────
step(0, "inventory — what is on disk, what is recorded, where Spark commands would run")
assemble = ck.load_json(ck.RUNS / "assemble.json")
boundary = ck.load_json(ck.LAPTOP_RUNS / "boundary.json")
runbook = ck.load_json(ck.RUNS / "runbook.json")
recorded = sorted(clawkit.RECORDED.glob("*.json")) if clawkit.RECORDED.is_dir() else []
rec_ours = []
for p in recorded:
    d = ck.load_json(p) or {}
    if ck.SANDBOX in d.get("cmd", "") or ck.REMOTE in d.get("cmd", ""):
        rec_ours.append(d)
table([
    ["lab 08-1 · assemble.json", "✓ " + assemble["date"] if assemble else "✕ missing — run lab 08-1"],
    ["lab 08-2 · boundary.json", "✓ " + boundary["date"] if boundary else "✕ missing — run lab 08-2"],
    ["lab 08-4 · runbook.json + RUNBOOK.md", "✓ " + runbook["date"] if runbook else "✕ missing — run lab 08-4"],
    ["recorded Spark transcripts (common/recorded/)", f"{len(recorded)} files · {len(rec_ours)} about {ck.SANDBOX}"],
    ["Spark commands run", {"dry": "nowhere (DRY)", "ssh": f"over ssh ({clawkit.host()})",
                            "local": "on this Spark"}[where()]],
], ["evidence source", "status"])


def spark(key: str, live_cmd: str | None = None) -> tuple[str, str]:
    """(output, source) for a lab's read-only command: LIVE (running `live_cmd` or the command itself), else its
    RECORDED transcript. ('', 'none') when neither exists. An EXAMPLE is never returned."""
    cmd = ck.CMD[key]
    if where() != "dry":
        r = sh(live_cmd or cmd, quiet=True, timeout=120)
        return r.out, "live"
    rec = clawkit._replay(clawkit._rec_key("a", cmd.strip()))          # the lab's own command, recorded
    if rec:
        return rec.get("out", ""), f"recorded {rec.get('date', '?')}"
    return "", "none"


def spark_item(d: str, item: str, pts: int, key: str, test, *, live_cmd: str | None = None) -> None:
    out, src = spark(key, live_cmd)
    if src == "none":
        award(d, item, pts, False, "spark", "◈ not yet evidenced (DRY: no live Spark, no recorded transcript)")
        return
    got = bool(test(out))
    award(d, item, pts, got, "spark", f"{src}: " + ("✓ matches" if got else "✕ output does not show it"))


# ── 1 · laptop evidence ────────────────────────────────────────────────────────────────────────────────────
step(1, "laptop evidence — verified from files on this Mac")
man = ck.load_json(ck.BUNDLE / "MANIFEST.json") or {"files": []}
stale = [f["path"] for f in man["files"] if not (ck.BUNDLE / f["path"]).is_file()
         or ck.sha256(ck.BUNDLE / f["path"]) != f["sha256"]]
bundle_ok = bool(man["files"]) and not stale and bool(assemble and assemble.get("ok"))
if stale:
    warn(f"bundle files changed since lab 08-1 wrote MANIFEST.json: {stale[:4]} — re-run lab 08-1")
dchk = (assemble or {}).get("dockerfile_checks", {})
award("1 image", "bundle assembled, MANIFEST hashes match, Dockerfile: non-root + NAT + chiller tool", 5,
      bundle_ok and all(dchk.values()) and bool(dchk), "laptop",
      f"{len(man['files'])} files · {sum(dchk.values())}/{len(dchk)} Dockerfile checks" if dchk else "no assemble.json")

prod = pk.load(str(ck.POLICIES / "prod-policy.yaml"))
d_w, _ = pk.decide(prod, {"op": "mcp", "host": "bms.alto.local", "port": 8443, "binary": "/usr/bin/python3.12",
                          "tool": "write_setpoint"})
hard = (prod.get("landlock") or {}).get("compatibility") == "hard_requirement"
award("2 policy", "prod policy: hard_requirement · write_setpoint denied (policykit) · parsed by the 0.0.111 CLI", 5,
      hard and d_w == "deny" and bool((assemble or {}).get("parser_prod_policy"))
      and bool((boundary or {}).get("policykit_ok")), "laptop",
      f"landlock={'hard_requirement' if hard else '?'} · write_setpoint → {d_w} · "
      f"agent asked for a ticket only: {(boundary or {}).get('agent_ok', '—')}")

sandbox_wf = yaml.safe_load((ck.CONFIGS / "workflow.sandbox.yml").read_text(encoding="utf-8"))
exporters = {v["_type"] for v in sandbox_wf["general"]["telemetry"]["tracing"].values()}
spans = (boundary or {}).get("trace_spans", {})
cross = (assemble or {}).get("cross_check", [])
award("3 telemetry", "phoenix + otelcollector exporters, both allowed by the policy · a laptop trace with LLM spans", 5,
      {"phoenix", "otelcollector"} <= exporters and bool(cross) and all(c[1].startswith("✓") for c in cross)
      and spans.get("LLM_END", 0) > 0, "laptop",
      f"exporters {sorted(exporters)} · laptop trace: {spans.get('WORKFLOW_START', 0)} workflows, "
      f"{spans.get('LLM_END', 0)} LLM spans (LAPTOP STAND-IN)")

rows = [json.loads(ln) for ln in (ck.BUNDLE / "data" / "alto_ops_eval.jsonl").read_text().splitlines()] \
    if (ck.BUNDLE / "data" / "alto_ops_eval.jsonl").is_file() else []
agent = (boundary or {}).get("agent", [])
award("4 eval", "20-question dataset with CSV-computed answers · eval_config validates · laptop stand-in eval ran", 5,
      len(rows) == 20 and all(r.get("answer") for r in rows) and bool((assemble or {}).get("nat_validate_ok"))
      and len(agent) == ck.LAPTOP_ROWS, "laptop",
      f"{len(rows)} rows · laptop eval: {len(agent)} questions · "
      + " · ".join(ln.split(":")[0] + ":" + ln.split(":")[1] for ln in (boundary or {}).get("eval_summary", [])[1:2]))

award("5 sandbox tax", "(nothing to measure on a laptop: there is no sandbox here)", 0, False, "laptop",
      "the tax is a Spark-only number by definition")

sections = (runbook or {}).get("sections", {})
award("6 runbook", "RUNBOOK.md: versions · snapshot · rebuild · rotate · upgrade 0.0.116 vs 0.1.2 · one page · "
      "commands parse", 10,
      bool(sections) and all(sections.values()) and (runbook or {}).get("lines", 999) <= 90
      and bool((runbook or {}).get("parsed_all")) and (ck.RUNS / "RUNBOOK.md").is_file(), "laptop",
      f"{sum(sections.values())}/5 sections · {(runbook or {}).get('lines', '?')} lines" if sections else "no runbook.json")

# ── 2 · Spark evidence ─────────────────────────────────────────────────────────────────────────────────────
step(2, "Spark evidence — LIVE read-only commands or RECORDED transcripts; EXAMPLE never counts")
spark_item("1 image", f"sandbox {ck.SANDBOX} exists on the Spark (openshell sandbox list)", 10, "sandbox_list",
           lambda o: ck.SANDBOX in o)
spark_item("2 policy", "the enforced policy has hard_requirement + the write_setpoint deny (policy get --full)", 5,
           "policy_full", lambda o: "hard_requirement" in o and "write_setpoint" in o)
spark_item("2 policy", "openshell logs show the write_setpoint call denied", 5, "logs",
           lambda o: any(("deny" in ln.lower() or "403" in ln) and ("write_setpoint" in ln or "bms.alto.local" in ln)
                         for ln in o.splitlines()))
spark_item("3 telemetry", "Phoenix answers on the Spark (:6006)", 3, "phoenix_up", lambda o: o.strip().endswith("200"))
spark_item("3 telemetry", "the blunt write request was sent and saved (/v1/workflow/full)", 3, "deny_request",
           lambda o: "write_setpoint" in o,
           live_cmd=f"tail -c 1500 {ck.REMOTE}/evidence/workflow_full_deny.jsonl")
spark_item("3 telemetry", "the policy plane shows inspect_for_inference for the agent's LLM calls", 4, "logs",
           lambda o: "inspect_for_inference" in o)
spark_item("4 eval", "nat eval outputs for Nano AND Super (inference_optimization.json in both)", 6, "eval_outputs",
           lambda o: o.count("inference_optimization.json") >= 2)
spark_item("4 eval", "nat sizing calc output for the 40-user estimate", 4, "eval_outputs",
           lambda o: "sizing" in o and "No such file" not in o.split("sizing", 1)[-1][:200])
spark_item("5 sandbox tax", "p95 files for BOTH legs: host (localhost:8000) and sandbox (inference.local)", 15,
           "tax_outputs", lambda o: o.count("inference_optimization.json") >= 2)
spark_item("6 runbook", "Spark versions checked (openshell --version on the Spark)", 5, "versions",
           lambda o: "openshell" in o.lower())

# ── 3 · bonus ──────────────────────────────────────────────────────────────────────────────────────────────
step(3, "bonus — a policy OpenShell's prover accepts with no new credentialed hosts")
cred_fields = ("credential_binding", "credential_signing", "allow_uninspected_credentials",
               "request_body_credential_rewrite", "websocket_credential_rewrite")
cred_hosts = [e["host"] for g in prod["network_policies"].values() for e in g["endpoints"]
              if any(e.get(f) for f in cred_fields)]
provider_hosts = [e["host"] for g in prod["network_policies"].values() for e in g["endpoints"]
                  if e["host"] in pk.INFERENCE_PROVIDER_HOSTS]
findings = [f for f in pk.harden(prod, tier="restricted") if f[0] != "OK"]
pre = not cred_hosts and not provider_hosts and not findings
print(("✓ " if pre else "✕ ") + f"laptop pre-check: {len(cred_hosts)} credentialed hosts · {len(provider_hosts)} "
      f"inference-provider hosts · {len(findings)} harden findings (policykit, a teaching model)")
note("The research tutorial names OpenShell's prover but gives no command for it. Its verdict comes from your "
     "OpenShell version on the Spark, so a reviewer awards these 10 points by reading that output. This grader "
     "never does.")

# ── 4 · the marking table ──────────────────────────────────────────────────────────────────────────────────
step(4, "the marking table")
table([[d, item[:60], plane, f"{got}/{pts}", ev[:58]] for d, item, got, pts, plane, ev in SCORE],
      ["deliverable", "item", "plane", "points", "evidence"])
per = {}
for d, _, got, pts, _, _ in SCORE:
    a, b = per.get(d, (0, 0))
    per[d] = (a + got, b + pts)
table([[d, f"{a}/{b}", "█" * a + "░" * (b - a)] for d, (a, b) in per.items()], ["deliverable", "score", ""])
lap = sum(g for _, _, g, _, p, _ in SCORE if p == "laptop")
lap_max = sum(t for _, _, _, t, p, _ in SCORE if p == "laptop")
spk = sum(g for _, _, g, _, p, _ in SCORE if p == "spark")
spk_max = sum(t for _, _, _, t, p, _ in SCORE if p == "spark")
total, total_max = lap + spk, lap_max + spk_max
print(f"◆ laptop-verifiable: {lap}/{lap_max} · Spark-proven: {spk}/{spk_max} · bonus: 0/10 (human review)")
(ck.RUNS / "grade.json").write_text(json.dumps({
    "date": time.strftime("%Y-%m-%d %H:%M"), "mode": where(), "total": total, "max": total_max, "bonus": 0,
    "laptop": [lap, lap_max], "spark": [spk, spk_max],
    "items": [{"deliverable": d, "item": i, "points": g, "of": t, "plane": p, "evidence": e}
              for d, i, g, t, p, e in SCORE]}, indent=1), encoding="utf-8")
ok(f"→ {(ck.RUNS / 'grade.json').relative_to(ROOT)}")
if spk == 0:
    warn(f"{spk_max} of the {total_max} points need the Spark. DRY or not, this grade is honest: "
         f"{total}/{total_max} is what the evidence on disk proves today.")
result(f"Alto Ops Claw v1: {total}/{total_max} (+ bonus by review). Evidence first, points second.")
