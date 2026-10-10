# Evals — behavioural checks for the Week 27 lanes (run with `claude plugin eval`)

Each case seeds a throwaway repo (`seed.sh`, via `--scaffold`), runs a real agent with the plugin loaded, and grades
what it did. `gh` is stubbed (offline) and logs every call to `<repo>/.gh-log` so graders can tell whether the
agent tried to merge. Default: 3 runs with the plugin, 3 without; the report shows WITH / W/OUT and the delta.

    PATH=$PWD/evals/_stubs/bin:$PATH claude plugin eval . --scaffold --allow-tools Bash Write Edit --runs 3 --max-cost-usd 5
    # one case: --case test-gap   · debug: --runs 1 --keep-temp

| Case | Proves | Graders |
|---|---|---|
| test-gap | fitness-qa flags an AC with no test as High and writes it red-first or bounces | ran-tests, no-merge, verdict (llm ×2) |
| review-remembers | fitness-review loads the prior report, marks R1 fixed / R2 not fixed with file:line, bounces, no merge | loaded-prior, no-merge, verdict (llm ×2) |
| ponytail-overengineering | fitness-review reports Over-engineering (Should fix) on a correct tested PR and still calls it merge-ready | reran-tests, no-merge, verdict (llm ×2) |

Field names follow https://code.claude.com/docs/en/plugin-evals as read on 2026-10-10. Run `claude plugin eval init`
and compare before relying on grader `type` names; this suite has not yet been executed against a live CLI.
