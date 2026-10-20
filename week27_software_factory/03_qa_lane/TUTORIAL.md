# ▶ Factory Lab 03 — The QA lane

> Part of Week 27. The Tester is a different agent from the Builder, forms its own hypotheses, and treats "tests pass" as the start of the conversation, not the end.

**What you'll actually do**
- Run `fitness-qa` on the Lab 02 PRs; one of them has an AC with no test.
- Build the test-gap ledger (AC → asserting `file:line` at unit / component / e2e, or `none`).
- Write a High gap red-first and prove it: break the guarded line, see red, restore, see green.
- Capture forensics (console, page errors, 5xx, HAR) with a Playwright fixture and bounce a PR to Build.

**Time** ~3 h · **Difficulty** intermediate · **Needs** Lab 02 PRs open; optional Node for Playwright

**Sources:** super-board [`super-qa/SKILL.md`](https://github.com/EricTechPro/super-board/blob/main/skills/super-qa/SKILL.md) ("What counts as green", "Test-gap check") · O'Reilly, [Inside a Software Factory](https://www.oreilly.com/radar/inside-a-software-factory/) (separate SE and QA agents) · brief §8

## 1 · What counts as green (15 min)

Read the rule out loud: **a 200 response with a blank body is a bug, not a green test.** Every e2e run in the QA lane asserts five things: no console errors, no uncaught page errors, no 5xx, no 401/403 on auth pages, and a page-type non-blank guard. For Alto Mini the guards are: rooms table ≥ 1 row; alerts table ≥ 1 row or an empty-state element; history `<pre>` not containing `loading…` or `error:`.

## 2 · The ledger (45 min)

Run `fitness-qa` on T1's PR (headless: `bash scripts/factory-run.sh qa` after moving the card to QA, or interactively: `Use the fitness-qa skill on PR #<P>`). Expected ledger shape in the `[tester]` comment:

```
| AC | unit | component | e2e | gap |
| AC1 kwh_rollup([]) == 0.0 | tests/test_rollup.py:4 | n/a | n/a | — |
| AC2 None ignored          | tests/test_rollup.py:5 | n/a | n/a | — |
| AC3 2-dp rounding         | tests/test_rollup.py:3 | n/a | n/a | — |
Edge: 0/1/many ✔ · None-only list → gap (witness: [None] expected 0.0) · very large list → n/a
Test the tests: test_ignores_none recomputes nothing from the code under test ✔ · surviving mutant: `if r is not None` → `if r` drops 0.0 readings — no test catches it → Medium
```

Now run it on T2's PR. The seeded trap: the Builder usually adds a default timezone but writes no test for "tz omitted" — an AC with no test at any rung is **High**.

## 3 · Red first (45 min)

For the High gap the Tester writes the test itself, then proves it can fail:

```bash
# on: laptop, in the QA worktree
../../.venv/bin/python -m pytest -q tests/test_history.py::test_default_tz   # green on first run? not yet evidence
sed -i 's/cfg.get("tz", "Asia\/Bangkok")/cfg["tz"]/' app/energy/daily.py      # break the guarded line
../../.venv/bin/python -m pytest -q tests/test_history.py::test_default_tz   # must be RED
git checkout app/energy/daily.py && ../../.venv/bin/python -m pytest -q      # GREEN again
```

**Expected output**

```
F                                                                        [100%]
KeyError: 'tz'
...
8 passed
```

If the gap needs **app code** to change before it is testable, that is a **Fail**: the Tester bounces to Build (card QA → Ready) with witness, expected, rung and target file — it does not fix the app.

## 4 · Forensics with Playwright (optional, 45 min)

```bash
# on: laptop, in week27/00_alto_mini
npm init -y >/dev/null && npm i -D @playwright/test && npx playwright install chromium
cp ../factory/e2e/report-fixture.ts e2e/   # fixture: console/pageerror/5xx/auth collectors + screenshot per step
make dev &  &&  npx playwright test e2e --reporter=line
```

The fixture fails the test when any collector fired and writes screenshots per step to the test output dir; the QA lane copies them to `docs/qa/report/pr-<P>/`. Open the HAR and find the `/api/rooms/1410/history` 500 that the Lab 02 T1 fix should have removed.

## 5 · Exercise

`ex03_ledger.py`: given a PR diff file list and a pytest `--collect-only -q` listing, print the ledger skeleton with `none` for every AC that no test name mentions. Then extend it to flag "a test that asserts on its own mock" by grepping for `assert mock` patterns. Solution provided.
