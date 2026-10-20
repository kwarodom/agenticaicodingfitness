"""Offline tests only. All predictions are explicitly synthetic fixtures."""
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jev_laya_benchmark as b


def fixture(questions):
    answers = {}
    for key, q in questions.items():
        typ = q["type"]
        if typ == "noul":
            answers[key] = {"type": typ, "noul": .1}
        else:
            labels = list(q["criteria"]) if typ == "choice" else [str(i) for i in range(len(q["criteria"]))]
            a = {"type": typ, "confidence": 1, "probabilities": {label: float(i == 0) for i, label in enumerate(labels)}}
            if typ == "choice":
                a["choice"] = labels[0]
            else:
                a.update(score=0.0, legend=dict(zip(labels, q["criteria"])))
            answers[key] = a
    return {"model": "OFFLINE_FIXTURE_NOT_MODEL_INFERENCE", "answers": answers}


class Tokens:
    mask_token = "[MASK]"
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split())))}


class SharedBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / "smoke.jsonl"
        self.write(b.seed_rows())
        self.row = b.load_rows(self.input, 1)[0]

    def write(self, rows):
        self.input.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    def cli(self, *args):
        with patch.object(sys, "argv", ["benchmark"] + list(args)), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return b.main()

    def entry(self, gold="yes", prediction="yes", probability=.9):
        q = {"type": "choice", "criteria": {"yes": "a", "no": "b"}}
        return {"gold": gold, "question": q,
                "answer": {"choice": prediction, "probabilities": {prediction: probability, ("no" if prediction == "yes" else "yes"): 1-probability}}}

    def test_all_seven_labs_in_seed(self):
        self.assertEqual(len(b.seed_rows()), 28)
        self.assertEqual({r["lab"] for r in b.load_rows(self.input, 100)}, set(b.lab.QUESTIONS))

    def test_gold_never_in_state(self):
        self.assertNotIn("expected", self.row["shared_state"])
        self.assertNotIn("language", self.row["shared_state"])
        self.assertEqual(json.loads(self.row["shared_state"]), b.lab.prepare("intent", self.row))

    def test_duplicate_ids(self):
        self.write([b.seed_rows()[0]] * 2)
        with self.assertRaises(ValueError): b.load_rows(self.input, 1)

    def test_bad_gold(self):
        row = b.seed_rows()[0]
        row["expected"] = {"intent": "invented"}
        self.write([row])
        with self.assertRaises(ValueError): b.load_rows(self.input, 1)

    def test_boolean_not_float_gold(self):
        row = b.seed_rows()[-1]
        row["expected"]["urgent"] = .1
        self.write([row])
        with self.assertRaises(ValueError): b.load_rows(self.input, 1)

    def test_bad_score_gold(self):
        row = b.seed_rows()[0]
        row["expected"] = {"complexity": 3}
        self.write([row])
        with self.assertRaises(ValueError): b.load_rows(self.input, 1)

    def test_mixed_splits_rejected(self):
        rows = b.seed_rows()[:2]
        rows[1]["split"] = "test"
        self.write(rows)
        with self.assertRaises(ValueError): b.load_rows(self.input, 2)

    def test_invalid_language(self):
        row = b.seed_rows()[0]
        row["language"] = "undocumented"
        self.write([row])
        with self.assertRaises(ValueError): b.load_rows(self.input, 1)

    def test_all_response_contracts(self):
        for questions in b.lab.QUESTIONS.values():
            b.lab.validate(fixture(questions), questions)

    def test_probability_gate_and_failure_denominator(self):
        entries = [self.entry(), self.entry(gold="no"), self.entry()]
        entries[-1]["answer"] = None
        m = b.decision_metrics(entries, .8)
        self.assertEqual(m["coverage_all_labeled"], 2/3)
        self.assertEqual(m["accepted_accuracy"], .5)
        self.assertEqual(m["wrong_accepted"], 1)
        self.assertEqual(m["accuracy_failures_count_wrong"], 1/3)
        self.assertAlmostEqual(m["ece10_successes_only"], .4)
        self.assertAlmostEqual(m["brier_successes_only"], .82)

    def test_no_success_null_not_zero(self):
        e = self.entry()
        e["answer"] = None
        m = b.decision_metrics([e], .8)
        self.assertIsNone(m["accuracy_successes_only"])
        self.assertIsNone(m["accepted_accuracy"])
        self.assertEqual(m["coverage_all_labeled"], 0)

    def test_noul_brier_convention(self):
        e = {"gold": True, "question": {"type": "noul"}, "answer": {"noul": .8}}
        m = b.decision_metrics([e], .8)
        self.assertAlmostEqual(m["brier_successes_only"], .04)

    def test_score_mae_and_argmax(self):
        e = {"gold": 2, "question": {"type": "score"},
             "answer": {"score": 1.7, "probabilities": {"0": 0, "1": .3, "2": .7}}}
        m = b.decision_metrics([e], .8)
        self.assertAlmostEqual(m["score_mae_successes_only"], .3)
        self.assertEqual(m["accuracy_successes_only"], 1)
        self.assertEqual(m["coverage_all_labeled"], 0)

    def test_unknown_always_review_metric(self):
        e = {"gold": "unknown", "question": {"type": "choice"},
             "answer": {"choice": "unknown", "probabilities": {"unknown": 1, "hvac": 0}}}
        self.assertEqual(b.decision_metrics([e], .8)["coverage_all_labeled"], 0)

    def test_percentiles(self):
        self.assertEqual(b.percentile([2, 1, 4, 3], .5), 2)
        self.assertEqual(b.percentile([2, 1, 4, 3], .95), 4)
        self.assertIsNone(b.percentile([], .5))

    def test_preflight_valid(self):
        self.assertTrue(b.laya_preflight(Tokens(), "short state", b.lab.QUESTIONS["email"], 1024, 256))

    def test_preflight_state_overflow(self):
        with self.assertRaisesRegex(ValueError, "state_would_truncate"):
            b.laya_preflight(Tokens(), "word " * 1000, b.lab.QUESTIONS["email"], 512, 256)

    def test_preflight_head_overflow(self):
        with self.assertRaisesRegex(ValueError, "head_would_truncate"):
            b.laya_preflight(Tokens(), "short", b.lab.QUESTIONS["email"], 1024, 16)

    def test_preflight_option_cap(self):
        q = {"q": {"type": "choice", "instructions": "choose", "criteria": {"a": "word " * 50, "b": "short"}}}
        with self.assertRaisesRegex(ValueError, "option_would_truncate"):
            b.laya_preflight(Tokens(), "short", q, 1024, 512)

    def test_preflight_mask_rewrite(self):
        with self.assertRaisesRegex(ValueError, "mask_token_rewrite"):
            b.laya_preflight(Tokens(), "[MASK]", b.lab.QUESTIONS["email"], 1024, 256)

    def test_record_failure_no_secret(self):
        def bad(*_): raise RuntimeError("secret_source_text")
        r = b.evaluate_call("jev", self.row, 0, "measured", bad)
        self.assertEqual(r["status"], "error")
        self.assertNotIn("secret_source_text", json.dumps(r))
        self.assertFalse(r["execute"])

    def test_malformed_answer_fails_to_review(self):
        r = b.evaluate_call("laya", self.row, 0, "measured", lambda *_: ({"answers": {}}, {}))
        self.assertEqual(r["status"], "error")
        self.assertNotIn("answers", r)

    def test_plan_no_inference(self):
        with patch.object(b, "LayaBackend", side_effect=AssertionError("must not load")), patch.object(b.lab, "request_live", side_effect=AssertionError("must not call")):
            self.cli("plan", "--input", str(self.input), "--limit", "28")

    def test_run_requires_both_permissions(self):
        with self.assertRaises(SystemExit):
            self.cli("run", "--input", str(self.input), "--live-jev")

    def test_offline_disallows_cloud(self):
        with self.assertRaises(SystemExit):
            self.cli("plan", "--input", str(self.input), "--offline")

    def test_output_preflight(self):
        with self.assertRaises(SystemExit):
            self.cli("run", "--input", str(self.input), "--out", str(self.root), "--live-jev", "--live-laya")

    def test_mocked_end_to_end_equal_inputs_and_outputs(self):
        seen = {"jev": [], "laya": []}
        class FakeLaya:
            meta = {"checkpoint": "OFFLINE_FIXTURE"}
            def __init__(self, args): pass
            def __call__(self, state, questions):
                seen["laya"].append((state, copy.deepcopy(questions)))
                return fixture(questions), {}
        def fake_jev(payload, provider):
            seen["jev"].append((payload["state"], copy.deepcopy(payload["questions"])))
            out = fixture(payload["questions"])
            out["usage"] = {"cost": .001}
            return out, 0
        output = self.root / "run"
        with patch.object(b, "LayaBackend", FakeLaya), patch.object(b.lab, "request_live", fake_jev), patch.dict("os.environ", {"OPENROUTER_API_KEY": "OFFLINE_TEST_VALUE"}):
            self.cli("run", "--input", str(self.input), "--out", str(output), "--limit", "28",
                     "--live-jev", "--live-laya", "--warmup", "1")
        self.assertEqual(seen["jev"], seen["laya"])
        self.assertEqual(len(seen["jev"]), 29)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["providers"]["jev"]["calls_measured"], 28)
        self.assertAlmostEqual(summary["providers"]["jev"]["reported_cost_usd_including_warmups"], .029)
        self.assertIsNone(summary["providers"]["laya"]["reported_cost_usd_including_warmups"])
        self.assertTrue(summary["paired_success_intersection"])
        for r in (output / "results.jsonl").read_text().splitlines():
            self.assertNotIn("shared_state", json.loads(r))
        self.cli("report", "--out", str(output))
        self.assertTrue((output / "report-rebuilt.md").exists())

    def test_setup_failure_prevents_jev_billing(self):
        output = self.root / "broken"
        with patch.object(b, "LayaBackend", side_effect=RuntimeError("load failed")), patch.object(b.lab, "request_live") as request, patch.dict("os.environ", {"OPENROUTER_API_KEY": "OFFLINE_TEST_VALUE"}):
            with self.assertRaises(SystemExit):
                self.cli("run", "--input", str(self.input), "--out", str(output), "--live-jev", "--live-laya")
            request.assert_not_called()
        self.assertFalse(json.loads((output / "setup-error.json").read_text())["inference_started"])

    def test_incomplete_report_flag(self):
        manifest = {"providers": ["jev"], "thresholds": {"jev": .8}, "rows": 2, "repeats": 1}
        r = b.evaluate_call("jev", self.row, 0, "measured", lambda s, q: (fixture(q), {}))
        summary = b.summarize(manifest, [r])
        self.assertFalse(summary["providers"]["jev"]["run_complete"])
        self.assertIn("Incomplete run", b.markdown_report(summary))

    def test_cost_preserved_on_response_failure(self):
        def invalid(s, q):
            return {"usage": {"cost": .01}, "answers": {}}, {}
        r = b.evaluate_call("jev", self.row, 0, "measured", invalid)
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["reported_cost_usd"], .01)

    def test_nonfinite_threshold_rejected(self):
        with self.assertRaises(SystemExit):
            self.cli("plan", "--input", str(self.input), "--jev-threshold", "nan")

    def test_missing_key_stops_before_laya_load(self):
        with patch.dict("os.environ", {}, clear=True), patch.object(b, "LayaBackend") as laya:
            with self.assertRaises(SystemExit):
                self.cli("run", "--input", str(self.input), "--out", str(self.root / "no-key"),
                         "--live-jev", "--live-laya")
            laya.assert_not_called()


if __name__ == "__main__":
    unittest.main()
