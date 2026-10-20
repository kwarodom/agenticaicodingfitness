#!/usr/bin/env python3
"""Shared Jev/Laya benchmark, Python 3.10+. Requires adjacent jev_lab.py.

No inference by default: init and plan use only the standard library.
run requires --live-jev and/or --live-laya for explicitly selected providers.
No actions, cloud fallback, automatic threshold tuning, or silent truncation.
Laya preflight is version-locked to the inspected 0.3.20 runtime.
"""
import argparse
import collections
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jev_lab as lab

VERSION = "alto-shared-benchmark-1.0"
LAYA_VERSION = "0.3.20"
NON_ROUTING_LABELS = {"unknown", "other"}


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def dump(path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def seed_rows():
    """Hand-authored toy labels. Partial labels are explicit, never inferred."""
    rows = []
    targets = {
        "email": [{"category": x} for x in ("support", "sales", "finance", "hr")],
        "hr": [
            {"python_evidence": "evidenced", "integration_evidence": "evidenced", "building_evidence": "evidenced"},
            {"python_evidence": "unclear", "integration_evidence": "not_stated", "building_evidence": "not_stated"},
        ],
        "rag": [{"relevant": True, "injection": False}, {"relevant": False, "injection": True}],
        "afdd": [{"safety_concern": False, "severity": 1}, {"safety_concern": False}],
        "lead": [{"solution": "air_side", "buying_stage": 2}, {"solution": "portfolio", "buying_stage": 1}],
        "point": [{"kind": "supply_air_temperature_sensor", "ambiguous": False},
                  {"kind": "zone_air_temperature_setpoint", "ambiguous": False},
                  {"kind": "unknown", "ambiguous": True}],
    }
    for task in lab.SAMPLES:
        for i, source in enumerate(lab.SAMPLES[task]):
            row = json.loads(json.dumps(source))
            row.update(lab=task, language="th" if (task == "intent" and i == 1) or (task == "email" and i == 3) else "en")
            if task in targets:
                row["expected"] = targets[task][i]
            row["split"] = "smoke"
            rows.append(row)
    rows += [
        {"id": "b-th-hvac", "lab": "intent", "language": "th", "split": "smoke",
         "state": {"message": "ห้องพักร้อนมาก ช่วยวิเคราะห์สาเหตุที่แอร์ไม่เย็น"},
         "expected": {"intent": "hvac"}},
        {"id": "b-mixed-code", "lab": "intent", "language": "mixed", "split": "smoke",
         "state": {"message": "ช่วยเขียน Python function ตรวจสอบ JSON payload"},
         "expected": {"intent": "coding", "complexity": 1}},
        {"id": "b-en-negation", "lab": "email", "language": "en", "split": "smoke",
         "state": {"subject": "Next year pricing", "body": "There is no current outage. Please quote next year's maintenance plan."},
         "expected": {"category": "sales", "urgent": False}},
        {"id": "b-th-finance", "lab": "email", "language": "th", "split": "smoke",
         "state": {"subject": "สอบถามใบแจ้งหนี้", "body": "ได้รับใบแจ้งหนี้ซ้ำสองฉบับ ช่วยตรวจสอบยอดเรียกเก็บด้วย"},
         "expected": {"category": "finance", "urgent": False}},
    ]
    return rows


def load_rows(path, limit):
    if limit < 1:
        raise ValueError("limit must be positive")
    raw = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not raw:
        raise ValueError("Empty corpus")
    ids, rows = set(), []
    for r in raw:  # Validate whole corpus even when selecting only the first N.
        if not isinstance(r.get("id"), str) or not r["id"] or r["id"] in ids:
            raise ValueError("Missing or duplicate string id")
        ids.add(r["id"])
        if r.get("lab") not in lab.QUESTIONS or r.get("language") not in ("en", "th", "mixed", "other"):
            raise ValueError("Unknown lab or language tag")
        if r.get("split") not in ("smoke", "train", "calibration", "test"):
            raise ValueError("Explicit split required")
        questions = lab.QUESTIONS[r["lab"]]
        expected = r.get("expected")
        if not isinstance(expected, dict) or not expected or not set(expected) <= set(questions):
            raise ValueError("Each row needs a nonempty expected subset of question IDs")
        for key, value in expected.items():
            q = questions[key]
            if q["type"] == "choice" and (not isinstance(value, str) or value not in q["criteria"]):
                raise ValueError("Invalid Choice gold label")
            if q["type"] == "noul" and type(value) is not bool:
                raise ValueError("Noul gold must be true/false, not a probability")
            if q["type"] == "score" and (type(value) is not int or not 0 <= value < len(q["criteria"])):
                raise ValueError("Score gold must be an integer ordinal level")
        # This shared string is the ONLY data passed to either model as state.
        state = canonical(lab.prepare(r["lab"], r))
        rows.append({**r, "shared_state": state, "state_sha256": hashlib.sha256(state.encode()).hexdigest(),
                     "schema_sha256": digest(questions)})
    chosen = rows[:limit]
    if len({r["split"] for r in chosen}) != 1:
        raise ValueError("Do not mix corpus splits in a run")
    return chosen


def laya_preflight(tok, state, questions, max_len, head_max_len):
    """Conservative full-text budget check for laya==0.3.20 build_sequence.

    Refuses option 48-token caps, head shrinking, state truncation, and mask
    replacement. Runtime-specific formatting must be re-audited on upgrade.
    No tensor/model execution is performed here.
    """
    def tokens(s):
        if tok.mask_token and tok.mask_token in s:
            raise ValueError("laya_mask_token_rewrite")
        return tok(s, add_special_tokens=False)["input_ids"]

    state_n = len(tokens(state))
    budgets = {}
    for qid, q in questions.items():
        typ, crit = q["type"], q.get("criteria")
        if typ == "choice":
            opts = [f"{k}: {v}" if v else k for k, v in crit.items()]
        elif typ == "score":
            opts = [f"level {i}: {v}" for i, v in enumerate(crit)]
        else:
            crit = crit or {}
            opts = ["false: " + crit.get("false", "no, the statement does not hold"),
                    "true: " + crit.get("true", "yes, the statement holds")]
        counts = [len(tokens(" " + text)) for text in opts]
        if max(counts) > 48:
            raise ValueError("laya_option_would_truncate")
        option_n = sum(n + 1 for n in counts)  # one MASK per option
        instruction_n = len(tokens(f"{typ} question: {q['instructions']}"))
        if head_max_len - option_n < 16 or instruction_n > head_max_len - option_n:
            raise ValueError("laya_head_would_truncate")
        full_n = instruction_n + option_n + state_n + 4  # CLS + 3 SEP
        if full_n > max_len:
            raise ValueError("laya_state_would_truncate")
        budgets[qid] = full_n
    return budgets


class LayaBackend:
    def __init__(self, args):
        if importlib.metadata.version("laya") != LAYA_VERSION:
            raise ValueError("laya_runtime_version_mismatch_required_0.3.20")
        if args.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from huggingface_hub import snapshot_download
        import laya
        import torch
        torch.manual_seed(args.seed)
        prefix = "" if args.checkpoint == "english" else args.checkpoint + "/"
        snapshot = Path(snapshot_download(
            "convaiinnovations/laya", revision=args.laya_revision,
            allow_patterns=[prefix + x for x in ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")],
            local_files_only=args.offline))
        model_dir = snapshot / prefix
        self.agent = laya.load(str(model_dir), device=args.device)
        self.requested_device = args.device
        if self.agent.device.type != args.device:
            raise ValueError("laya_device_fallback_during_setup")
        self.max_len = args.max_len or int(self.agent.cfg["max_len"])
        self.head_len = args.head_max_len or int(self.agent.cfg.get("head_max_len", 192))
        cap = 8192 if args.checkpoint == "multilingual" else int(self.agent.cfg["max_len"])
        if not 1 <= self.head_len < self.max_len <= cap:
            raise ValueError("Invalid token budget for selected checkpoint")
        self.meta = {"runtime": LAYA_VERSION, "checkpoint": args.checkpoint,
                     "requested_revision": args.laya_revision, "resolved_revision": snapshot.name,
                     "device": str(self.agent.device), "max_len": self.max_len,
                     "head_max_len": self.head_len,
                     "weight_dtype": str(next(self.agent.model.parameters()).dtype),
                     "runtime_autocast_dtype": str(self.agent.dtype),
                     "torch_version": torch.__version__,
                     "transformers_version": importlib.metadata.version("transformers"),
                     "huggingface_hub_version": importlib.metadata.version("huggingface_hub"),
                     "config_sha256": digest(self.agent.cfg)}

    def __call__(self, state, questions):
        budget = laya_preflight(self.agent.tok, state, questions, self.max_len, self.head_len)
        response = self.agent.predict(state, questions, max_len=self.max_len, head_max_len=self.head_len)
        if self.agent.device.type != self.requested_device:
            raise ValueError("laya_device_changed_during_inference")
        return response, {"untruncated_tokens_per_question": budget,
                          "actual_device": str(self.agent.device)}


def percentile(values, q):
    values = sorted(values)
    return values[max(0, math.ceil(q * len(values)) - 1)] if values else None


def classification_view(q, gold, a):
    typ = q["type"]
    if typ == "noul":
        p = a["noul"]
        probs = {False: 1 - p, True: p}
        prediction = p >= .5
    elif typ == "choice":
        probs, prediction = a["probabilities"], a["choice"]
    else:
        probs = {int(k): v for k, v in a["probabilities"].items()}
        prediction = max(probs, key=probs.get)
    return prediction, probs, probs[prediction], float(prediction == gold)


def decision_metrics(entries, threshold):
    """One task/question/language group. Failures count in denominator."""
    valid = [e for e in entries if e["answer"] is not None]
    details = []
    for e in valid:
        pred, probs, confidence, correct = classification_view(e["question"], e["gold"], e["answer"])
        accepted = confidence >= threshold and pred not in NON_ROUTING_LABELS
        details.append((e, pred, probs, confidence, correct, accepted))
    total, n = len(entries), len(valid)
    correct = sum(d[4] for d in details)
    accepted = [d for d in details if d[5]]
    confusion = collections.Counter((str(d[0]["gold"]), str(d[1])) for d in details)
    labels = sorted({s for pair in confusion for s in pair})
    f1s = []
    for label in labels:
        tp = confusion[label, label]
        fp = sum(v for (g, p), v in confusion.items() if p == label and g != label)
        fn = sum(v for (g, p), v in confusion.items() if g == label and p != label)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    bins = []
    for b in range(10):
        ds = [d for d in details if min(9, int(d[3] * 10)) == b]
        if ds:
            bins.append({"lower": b / 10, "n": len(ds), "confidence": statistics.mean(d[3] for d in ds),
                         "accuracy": statistics.mean(d[4] for d in ds)})
    ece = sum(b["n"] / n * abs(b["confidence"] - b["accuracy"]) for b in bins) if n else None
    typ = entries[0]["question"]["type"]
    # Multiclass sum-of-squares Brier. Binary Brier uses the conventional positive-class square.
    brier = statistics.mean(
        (d[0]["answer"]["noul"] - int(d[0]["gold"])) ** 2 if typ == "noul"
        else sum((p - int(k == d[0]["gold"])) ** 2 for k, p in d[2].items())
        for d in details) if n else None
    return {
        "type": typ, "labeled_decisions": total, "valid_decisions": n, "failures": total - n,
        "accuracy_successes_only": correct / n if n else None,
        "accuracy_failures_count_wrong": correct / total,
        "macro_f1_observed_labels_successes_only": statistics.mean(f1s) if f1s else None,
        "threshold_top_probability": threshold, "accepted_decisions": len(accepted),
        "coverage_all_labeled": len(accepted) / total,
        "accepted_accuracy": statistics.mean(d[4] for d in accepted) if accepted else None,
        "wrong_accepted": sum(not d[4] for d in accepted),
        "brier_successes_only": brier, "ece10_successes_only": ece, "calibration_bins": bins,
        "score_mae_successes_only": statistics.mean(abs(e["answer"]["score"] - e["gold"]) for e in valid) if n and typ == "score" else None,
        "confusion": [{"gold": g, "predicted": p, "n": count} for (g, p), count in sorted(confusion.items())],
    }


def summarize(manifest, results):
    measured = [r for r in results if r["phase"] == "measured"]
    summaries = {}
    for provider in manifest["providers"]:
        runs = [r for r in measured if r["provider"] == provider]
        successful = [r for r in runs if r["status"] == "ok"]
        all_costs = [r.get("reported_cost_usd") for r in results if r["provider"] == provider]
        known = [v for v in all_costs if lab.number(v) and v >= 0]
        groups = collections.defaultdict(list)
        for r in runs:
            for qid, gold in r["expected"].items():
                for language in (r["language"], "all"):
                    groups[f"{r['lab']}/{qid}/{language}"].append(
                        {"gold": gold, "question": lab.QUESTIONS[r["lab"]][qid],
                         "answer": r.get("answers", {}).get(qid)})
        threshold = manifest["thresholds"][provider]
        summaries[provider] = {
            "planned_measured_calls": manifest["rows"] * manifest["repeats"],
            "recorded_measured_calls": len(runs),
            "run_complete": len(runs) == manifest["rows"] * manifest["repeats"],
            "calls_measured": len(runs), "successful_calls": len(successful),
            "failed_calls": len(runs) - len(successful),
            "error_counts": dict(collections.Counter(r["error"] for r in runs if r["status"] != "ok")),
            "latency_ms_all_attempts": {"p50": percentile([r["elapsed_ms"] for r in runs], .5),
                                       "p95": percentile([r["elapsed_ms"] for r in runs], .95)},
            "latency_ms_successes": {"p50": percentile([r["elapsed_ms"] for r in successful], .5),
                                    "p95": percentile([r["elapsed_ms"] for r in successful], .95)},
            "reported_cost_usd_including_warmups": sum(known) if known else None,
            "cost_known_call_records": len(known), "cost_total_call_records": len(all_costs),
            "groups": {k: decision_metrics(v, threshold) for k, v in sorted(groups.items())},
            "threshold_sweep_exploratory": {
                str(t): {k: {metric: value for metric, value in decision_metrics(v, t).items()
                             if metric in ("coverage_all_labeled", "accepted_accuracy", "wrong_accepted")}
                         for k, v in sorted(groups.items()) if k.endswith("/all")}
                for t in (0.5, 0.7, 0.8, 0.9, 0.95)},
        }
    # Matched intersection, not a replacement for failure-inclusive statistics.
    pairs = collections.defaultdict(dict)
    for r in measured:
        pairs[(r["id"], r["repeat"])][r["provider"]] = r
    comparisons = collections.defaultdict(list)
    for pair in pairs.values():
        if len(pair) != 2 or any(r["status"] != "ok" for r in pair.values()):
            continue
        j, l = pair["jev"], pair["laya"]
        for qid, gold in j["expected"].items():
            q = lab.QUESTIONS[j["lab"]][qid]
            jp = classification_view(q, gold, j["answers"][qid])[0]
            lp = classification_view(q, gold, l["answers"][qid])[0]
            comparisons[f"{j['lab']}/{qid}"].append((jp == gold, lp == gold, jp == lp))
    paired = {k: {"both_valid_decisions": len(v),
                  "jev_accuracy": statistics.mean(x[0] for x in v),
                  "laya_accuracy": statistics.mean(x[1] for x in v),
                  "prediction_agreement": statistics.mean(x[2] for x in v)}
              for k, v in sorted(comparisons.items())}
    return {"mode": "measured_not_mock", "manifest": manifest, "providers": summaries, "paired_success_intersection": paired,
            "cautions": ["Toy labels are not a benchmark of production quality.",
                         "Top probability is not provider-specific confidence. Acceptance is a metric, not permission.",
                         "Repeated trials are correlated; no independent-sample confidence interval is claimed.",
                         "Failures reduce coverage. Cost may omit retry/failed-call billing.",
                         "Threshold sweep is exploratory; do not choose a production threshold on test data."]}


def markdown_report(summary):
    lines = ["# Jev versus Laya benchmark", "",
             "Measured results from this run only. Acceptance means passing an experimental metric gate, never authorization to act.", "",
             "## Runtime and failures", "",
             "| Provider | Calls | Failed | p50 all ms | p95 all ms | Reported cost USD |",
             "|---|---:|---:|---:|---:|---:|"]
    fmt = lambda x: "n/a" if x is None else f"{x:.4f}"
    for name, s in summary["providers"].items():
        lines.append(f"| {name} | {s['calls_measured']} | {s['failed_calls']} | {fmt(s['latency_ms_all_attempts']['p50'])} | {fmt(s['latency_ms_all_attempts']['p95'])} | {fmt(s['reported_cost_usd_including_warmups'])} |")
    lines += ["", "## Per-question quality", "",
              "| Provider / task / question / language | Labeled | Valid | Accuracy valid | Coverage | Accepted accuracy | Wrong accepted |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, s in summary["providers"].items():
        for key, g in s["groups"].items():
            lines.append(f"| {name}/{key} | {g['labeled_decisions']} | {g['valid_decisions']} | {fmt(g['accuracy_successes_only'])} | {fmt(g['coverage_all_labeled'])} | {fmt(g['accepted_accuracy'])} | {g['wrong_accepted']} |")
    if any(not s["run_complete"] for s in summary["providers"].values()):
        lines += ["", "WARNING: Incomplete run. Denominators below include recorded attempts only; do not compare as a completed benchmark."]
    lines += ["", "## Interpretation", ""] + ["- " + x for x in summary["cautions"]]
    lines += ["", "See summary.json for confusion matrices, Brier, ECE, Score MAE, threshold sweeps and matched-pair results.",
              "See manifest.json for dataset/schema hashes, actual runtime configuration, checkpoint revision and setup time.", ""]
    return "\n".join(lines)


def evaluate_call(provider, row, repeat, phase, fn):
    record = {k: row[k] for k in ("id", "lab", "language", "expected", "state_sha256", "schema_sha256")}
    record.update(provider=provider, repeat=repeat, phase=phase, execute=False)
    start = time.perf_counter()
    try:
        result, extra = fn(row["shared_state"], lab.QUESTIONS[row["lab"]])
        usage = result.get("usage", {})
        cost = usage.get("cost") if isinstance(usage, dict) else None
        record["reported_cost_usd"] = cost if lab.number(cost) and cost >= 0 else None
        questions = lab.QUESTIONS[row["lab"]]
        validated = lab.validate(result, questions)
        safe_answers = {}
        for qid, a in validated.items():
            q = questions[qid]
            if q["type"] == "score" and a["legend"] != {str(i): c for i, c in enumerate(q["criteria"])}:
                raise ValueError("Score legend mismatch")
            safe_answers[qid] = {k: a[k] for k in ("type", "choice", "score", "noul", "probabilities", "confidence", "legend") if k in a}
        record["answers"] = safe_answers
        safe_usage = {k: v for k, v in usage.items() if k in ("cost", "input_tokens", "output_tokens") and lab.number(v) and v >= 0} if isinstance(usage, dict) else {}
        record.update(status="ok", model=result["model"], usage=safe_usage, diagnostics=extra)
    except Exception as exc:
        # No upstream body, source state or secrets in exception output.
        text = str(exc)
        safe_error = text if isinstance(exc, ValueError) and text.startswith("laya_") else type(exc).__name__
        record.update(status="error", error=safe_error)
        record.pop("answers", None)
    record["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 3)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("init", "plan", "run", "report"))
    p.add_argument("--input", type=Path, default=Path("benchmark_data/smoke.jsonl"))
    p.add_argument("--out", type=Path, default=Path("benchmark_run"))
    p.add_argument("--dir", type=Path, default=Path("benchmark_data"))
    p.add_argument("--providers", nargs="+", choices=("jev", "laya"), default=["jev", "laya"])
    p.add_argument("--live-jev", action="store_true")
    p.add_argument("--live-laya", action="store_true")
    p.add_argument("--jev-provider", choices=tuple(lab.PROVIDERS), default="typesafe")
    p.add_argument("--checkpoint", choices=("english", "multilingual", "typed-decisions"), default="multilingual")
    p.add_argument("--laya-revision", default="main", help="Prefer an audited Hugging Face commit SHA")
    p.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    p.add_argument("--offline", action="store_true", help="Laya download cache only; prohibits Jev in same run")
    p.add_argument("--max-len", type=int)
    p.add_argument("--head-max-len", type=int)
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--warmup", type=int, default=0, help="Extra calls per provider, possibly billed; excluded from quality")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--jev-threshold", type=float, default=.8)
    p.add_argument("--laya-threshold", type=float, default=.8)
    args = p.parse_args()
    if args.command == "init":
        args.dir.mkdir(parents=True, exist_ok=False)
        with (args.dir / "smoke.jsonl").open("x", encoding="utf-8") as f:
            for row in seed_rows():
                f.write(canonical(row) + "\n")
        print(f"Created {len(seed_rows())} fictional smoke examples; not a held-out production benchmark.")
        return
    if args.command == "report":
        manifest = json.loads((args.out / "manifest.json").read_text())
        results = [json.loads(s) for s in (args.out / "results.jsonl").read_text().splitlines() if s.strip()]
        summary = summarize(manifest, results)
        # Rebuild to new files only, never overwrite a prior report.
        dump(args.out / "summary-rebuilt.json", summary)
        with (args.out / "report-rebuilt.md").open("x", encoding="utf-8") as f:
            f.write(markdown_report(summary))
        return
    if len(set(args.providers)) != len(args.providers) or args.repeats < 1 or args.warmup < 0:
        p.error("Providers must be unique; repeats >=1; warmup >=0")
    if any(not math.isfinite(t) or not .5 <= t <= 1 for t in (args.jev_threshold, args.laya_threshold)):
        p.error("Thresholds must be finite in [0.5, 1]")
    if args.offline and "jev" in args.providers:
        p.error("--offline requires --providers laya; do not mix with a cloud call")
    rows = load_rows(args.input, args.limit)
    manifest = {
        "runner_version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "providers": args.providers,
        "input_file_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "selected_corpus_sha256": digest([{k: r[k] for k in ("id", "lab", "shared_state", "expected")} for r in rows]),
        "schema_hashes": {r["lab"]: r["schema_sha256"] for r in rows},
        "lab_code_sha256": hashlib.sha256(Path(lab.__file__).read_bytes()).hexdigest(),
        "rows": len(rows), "split": rows[0]["split"], "repeats": args.repeats, "warmup": args.warmup,
        "seed": args.seed, "thresholds": {"jev": args.jev_threshold, "laya": args.laya_threshold},
        "configuration": vars(args) | {"input": str(args.input), "out": str(args.out), "dir": str(args.dir)},
        "max_logical_calls_per_provider": len(rows) * args.repeats + args.warmup,
        "jev_http_attempt_cap_per_logical_call": 4,
        "gold_policy": "Partial expected labels only; unlabeled answers are never treated as gold",
    }
    if args.command == "plan":
        print(json.dumps({"mode": "dry_run_no_inference", "manifest": manifest,
                          "example_shared_request": {"state": rows[0]["shared_state"],
                                                     "questions": lab.QUESTIONS[rows[0]["lab"]]}},
                         ensure_ascii=False, indent=2))
        return
    if "jev" in args.providers and not args.live_jev or "laya" in args.providers and not args.live_laya:
        p.error("Explicit --live-jev / --live-laya required for each selected provider")
    if args.out.exists():
        p.error("Output directory exists; choose a new name before inference")
    if "jev" in args.providers and not os.getenv(lab.PROVIDERS[args.jev_provider][1]):
        p.error("Missing selected Jev provider key")
    args.out.mkdir(parents=True)
    funcs, setup_errors = {}, {}
    for provider in args.providers:
        start = time.perf_counter()
        try:
            if provider == "jev":
                def jev(state, questions):
                    payload = {"model": lab.PROVIDERS[args.jev_provider][2], "state": state, "questions": questions}
                    response, _ = lab.request_live(payload, args.jev_provider)
                    return response, {}
                funcs[provider] = jev
                manifest["jev"] = {"provider": args.jev_provider, "endpoint": lab.PROVIDERS[args.jev_provider][0],
                                   "model_requested": lab.PROVIDERS[args.jev_provider][2]}
            else:
                backend = LayaBackend(args)
                funcs[provider] = backend
                manifest["laya"] = backend.meta
        except Exception as exc:
            setup_errors[provider] = str(exc) if isinstance(exc, ValueError) and str(exc).startswith("laya_") else type(exc).__name__
        manifest.setdefault("setup_ms", {})[provider] = round((time.perf_counter() - start) * 1000, 3)
    manifest["setup_errors"] = setup_errors
    dump(args.out / "manifest.json", manifest)
    if setup_errors:
        # Do not bill Jev if its comparison partner cannot initialize.
        dump(args.out / "setup-error.json", {"error_types": setup_errors, "inference_started": False})
        raise SystemExit("Backend setup failed; no inference started. See setup-error.json.")
    results, rng = [], random.Random(args.seed)
    schedule = [(rows[0], i, "warmup") for i in range(args.warmup)]
    for repeat in range(args.repeats):
        order = list(rows)
        rng.shuffle(order)
        schedule += [(r, repeat, "measured") for r in order]
    with (args.out / "results.jsonl").open("x", encoding="utf-8") as f:
        for index, (row, repeat, phase) in enumerate(schedule):
            order = args.providers if index % 2 == 0 else list(reversed(args.providers))
            for provider in order:
                record = evaluate_call(provider, row, repeat, phase, funcs[provider])
                results.append(record)
                f.write(canonical(record) + "\n")
                f.flush()
                print(f"{phase} {row['id']} {provider}: {record['status']}", flush=True)
    summary = summarize(manifest, results)
    dump(args.out / "summary.json", summary)
    with (args.out / "report.md").open("x", encoding="utf-8") as f:
        f.write(markdown_report(summary))
    print(f"Completed. Open {args.out / 'report.md'}; full metrics in summary.json.")
    if any(r["status"] != "ok" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
