#!/usr/bin/env python3
"""Lab 06-4 · The sandbox tax: the same eval on two paths, and the p95 difference.

Layer 4 of 4 (research tutorial Part 5, Lab 5.4). The method: run the identical `nat eval` (a) on the Spark host
against http://localhost:8000/v1 and (b) inside the OpenShell sandbox against https://inference.local/v1, at
max_concurrency 1 and 4, and compare p95 workflow runtime from inference_optimization.json. Record
`openshell --version` with the number. No official overhead figure exists, so your measurement is the reference.
  1. The Spark legs. Uploading into the sandbox goes through change(); DRY → EXAMPLE. The laptop OpenShell CLI
     parse-checks the openshell commands offline.
  2. LAPTOP STAND-IN: the same eval twice, (A) straight to Ollama and (B) through a local reverse proxy with an
     allow-list (benchkit.HopProxy). It is NOT OpenShell: no TLS interception, no Landlock, no network namespace.
     It gives the method a second path to measure. 2 questions × 2 legs ≈ 8 LLM calls.
  3. The delta calculator over the two result directories, with a "within noise?" verdict.

Run: .venv/bin/python week26/06_benchmarking/labs/lab06_4_sandbox_tax.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clawkit import (LAPTOP_OLLAMA, NAT, OPENSHELL_PINNED, ROOT, banner, change, free_port, laptop, note, ok,  # noqa: E402
                     openshell_offline, parsed_ok, result, sandbox, sh, step, table, up, warn)
from benchkit import CONFIGS, DATA, RUNS, HopProxy, read_profile, rel, save_summary  # noqa: E402

banner("Lab 06-4 · the sandbox tax", "same eval · two paths · p95 delta · no official figure exists")
SB = "alto-ops"                      # the NAT sandbox from research tutorial Lab 3.8 (Module 04)
print(f"◆ NAT sandbox: {SB} (research tutorial Lab 3.8) · NemoClaw sandbox for the harness labs: {sandbox()}")


def tax(a_dir: Path, b_dir: Path) -> dict:
    """The delta calculator: p50/p95/mean of workflow runtime for two result dirs, and whether B−A beats the noise."""
    a, b = read_profile(a_dir), read_profile(b_dir)
    wa, wb = a.get("wf") or {}, b.get("wf") or {}
    d95 = (wb.get("p95") or 0) - (wa.get("p95") or 0)
    lo_a, hi_a = wa.get("ninety_fifth_interval") or [0, 0]
    lo_b, hi_b = wb.get("ninety_fifth_interval") or [0, 0]
    overlap = not (hi_a < lo_b or hi_b < lo_a)
    n = min(wa.get("n") or 0, wb.get("n") or 0)
    return {"n": n, "a": a, "b": b, "p95_a": wa.get("p95"), "p95_b": wb.get("p95"), "delta_p95_s": d95,
            "delta_pct": 100 * d95 / wa["p95"] if wa.get("p95") else None, "ci_a": [lo_a, hi_a], "ci_b": [lo_b, hi_b],
            "within_noise": overlap}


def show_tax(t: dict, la: str, lb: str) -> None:
    table([[la, f"{t['a'].get('p50_runtime') or 0:.2f}", f"{t['p95_a']:.2f}", f"{t['a']['wf']['mean']:.2f}",
            f"{t['ci_a'][0]:.2f}–{t['ci_a'][1]:.2f}", t["a"]["wf"]["n"]],
           [lb, f"{t['b'].get('p50_runtime') or 0:.2f}", f"{t['p95_b']:.2f}", f"{t['b']['wf']['mean']:.2f}",
            f"{t['ci_b'][0]:.2f}–{t['ci_b'][1]:.2f}", t["b"]["wf"]["n"]]],
          ["path", "p50 s", "p95 s", "mean s", "95% CI of mean", "n"])
    print(f"◆ Δ p95 = {t['delta_p95_s']:+.2f} s ({t['delta_pct']:+.1f} %)")
    if t["n"] < 5:
        warn(f"n = {t['n']} per leg: too few rows for a confidence interval to mean much. Treat the verdict below "
             "as a demonstration of the check, not as evidence.")
    if t["delta_p95_s"] < 0:
        warn("path B came out FASTER, and an extra hop cannot do that. The difference is drift between the two "
             "runs (other labs on this Ollama, a warm cache), not a tax. Interleave the legs and repeat them.")
    elif t["within_noise"]:
        warn("the two confidence intervals overlap: this delta is WITHIN NOISE. Do not publish it as a tax. Add rows "
             "and repetitions (`nat eval --reps`) until the intervals separate, or report 'not measurable at n = …'.")
    else:
        ok("the intervals do not overlap: the difference is larger than the run-to-run noise")


# ── 1 · the Spark: host leg vs sandbox leg ───────────────────────────────────────
step(1, "the Spark — record the OpenShell version, then run the identical eval on both paths at c=1 and c=4")
sh("openshell --version", example=f"openshell {OPENSHELL_PINNED}", timeout=30)
sh("cd ~/alto_ops && for c in 1 4; do nat eval --config_file eval_config.yml "
   "--override eval.general.max_concurrency $c --override eval.general.output_dir ./.tmp/tax/host_c$c/; done",
   example="… two evals: ./.tmp/tax/host_c1/ and ./.tmp/tax/host_c4/, each with inference_optimization.json",
   timeout=3600)
UPLOAD = f"cd ~/alto_ops && openshell sandbox upload {SB} ./eval_config.sandbox.yml /sandbox/eval_config.yml"
change(UPLOAD, preview=f"openshell sandbox get {SB}",
       example=f"Name: {SB}\nPhase: Ready\n… (read-only preview: the sandbox exists and is Ready before anything is uploaded)")
EXEC = (f"openshell sandbox exec -n {SB} --workdir /sandbox -- nat eval --config_file /sandbox/eval_config.yml "
        "--override eval.general.max_concurrency 1 --override eval.general.output_dir /sandbox/.tmp/tax/sandbox_c1/")
sh(EXEC, example="=== EVALUATION SUMMARY ===\nWorkflow Status: COMPLETED (workflow_output.json)\n"
                 "Workflow Runtime (p95): …s\nLLM Latency (p95): …s", timeout=3600)
sh(f"openshell sandbox download {SB} /sandbox/.tmp/tax/sandbox_c1 ./.tmp/tax/sandbox_c1",
   example="… (downloads inference_optimization.json and the rest; repeat both lines with c=4)", timeout=300)
note(f"{rel(CONFIGS / 'eval_config.sandbox.yml')} = the Spark eval config with `base_url: https://inference.local/v1` "
     "and /sandbox paths. Only the path to the model changes, so the delta is the sandbox's cost. Upload data/ "
     "(CSV + jsonl) the same way. LLM judges run after the workflow, so they do not change workflow p95.")

print("◆ offline parse-check of those openshell commands with the laptop CLI (it reaches no gateway):")
for args in (["sandbox", "upload", SB, str(CONFIGS / "eval_config.sandbox.yml"), "/sandbox/eval_config.yml"],
             ["sandbox", "exec", "-n", SB, "--workdir", "/sandbox", "--", "nat", "eval", "--config_file",
              "/sandbox/eval_config.yml"],
             ["sandbox", "download", SB, "/sandbox/.tmp/tax/sandbox_c1", "./.tmp/tax/sandbox_c1"]):
    r = openshell_offline(args, home=RUNS / "openshell-home")
    print(("✓ parsed    " if parsed_ok(r) else "✕ rejected  ") + "openshell " + " ".join(args[:3]) + " …")

# ── 2 · LAPTOP STAND-IN: two paths to the same model ─────────────────────────────
step(2, "LAPTOP STAND-IN — (A) NAT → Ollama directly · (B) NAT → hop proxy with an allow-list → Ollama")
if not up(LAPTOP_OLLAMA):
    warn("laptop Ollama is not answering on :11434 — start it, then re-run")
    sys.exit(0)
src = DATA / "alto_ops_eval.jsonl"
if not src.is_file():
    warn("data/alto_ops_eval.jsonl is missing — run lab 06-2 first (it builds the dataset from the CSV)")
    sys.exit(0)
rows = RUNS / "tax_rows.jsonl"
rows.write_text("".join(src.read_text(encoding="utf-8").splitlines(keepends=True)[:2]), encoding="utf-8")
CFG = CONFIGS / "eval_config.yml"
A, B = RUNS / "tax" / "direct_c1", RUNS / "tax" / "hop_c1"


def leg(out: Path, base_url: str | None) -> bool:
    argv = [NAT, "eval", "--config_file", rel(CFG), "--override", "eval.general.max_concurrency", "1",
            "--override", "eval.general.output_dir", rel(out) + "/", "--override", "eval.general.dataset.file_path", rel(rows)]
    if base_url:
        argv += ["--override", "llms.local_llm.base_url", base_url]
    r = laptop(argv, cwd=ROOT, quiet=True, timeout=600, show="nat " + " ".join(str(a) for a in argv[1:]))
    line = next((ln for ln in r.out.splitlines() if ln.startswith("Workflow Runtime (p95)")), "")
    print(f"◆ {line or 'no summary line'}")
    return r.ok and (out / "inference_optimization.json").is_file()


okA = leg(A, None)
port = free_port(8090)
print(f"◆ hop proxy on 127.0.0.1:{port} → {LAPTOP_OLLAMA} (allow: POST /v1/chat/completions, GET /v1/models; deny the rest)")
with HopProxy(LAPTOP_OLLAMA, port) as hop:
    okB = leg(B, f"http://127.0.0.1:{port}/v1")
    from urllib.request import urlopen
    try:
        urlopen(f"http://127.0.0.1:{port}/api/pull", timeout=5)          # noqa: S310 — not on the allow-list
    except Exception as e:  # noqa: BLE001
        print(f"✓ GET /api/pull through the hop → {getattr(e, 'code', type(e).__name__)} (denied, like an unlisted endpoint)")
if not (okA and okB):
    warn("one leg did not finish — see the lines above")
    sys.exit(0)
ov = sorted(hop.overhead_ms)
print(f"◆ the hop itself: {hop.allowed} allowed · {hop.denied} denied · its own time per LLM call "
      f"≈ {sum(ov) / len(ov):.2f} ms (max {ov[-1]:.2f} ms), next to ≈ {sum(hop.upstream_ms) / len(hop.upstream_ms) / 1000:.1f} s upstream")

# ── 3 · the delta calculator ─────────────────────────────────────────────────────
step(3, "the delta calculator over the two result directories")
t = tax(A, B)
show_tax(t, "A · direct (host stand-in)", "B · via hop proxy (sandbox stand-in)")
note(f"The one trustworthy laptop number is the hop's own time: ≈ {sum(ov) / len(ov):.1f} ms per LLM call, measured "
     "inside the proxy. Run-to-run noise on this shared Ollama is seconds. How big the tax is on a Spark, no source "
     "says, so measure it with enough rows and repetitions to see it above the noise.")
save_summary("sandbox", {"p95_direct_s": t["p95_a"], "p95_hop_s": t["p95_b"], "delta_p95_s": t["delta_p95_s"],
                         "within_noise": t["within_noise"], "hop_overhead_ms": sum(ov) / len(ov), "rows": 2,
                         "concurrency": 1})
(RUNS / "tax" / "tax_report.json").write_text(json.dumps({k: v for k, v in t.items() if k not in ("a", "b")}, indent=1),
                                               encoding="utf-8")

step(4, "what to publish with a sandbox-tax number")
table([["openshell --version", "on the Spark (NemoClaw pins " + OPENSHELL_PINNED + ")"],
       ["nat --version · model · engine", "the same on both legs"],
       ["rows · reps · max_concurrency", "both legs, c=1 and c=4"],
       ["p95 host · p95 sandbox · Δ", "from inference_optimization.json"],
       ["95% CI of the mean, both legs", "is Δ outside the noise?"]], ["record", "why"])
warn("LAPTOP STAND-IN: leg B is a Python proxy on this Mac, not OpenShell. Its delta says nothing about the sandbox on "
     "a Spark. No official overhead figure exists either: your Spark measurement is the reference.")
result("Same eval, two paths, same settings. Publish Δ p95 only with its noise and the OpenShell version.")
