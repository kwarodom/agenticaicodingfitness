---
type: llm
weight: 2
---
PASS only if the agent produced a test-gap ledger mapping each acceptance criterion to a test or 'none', ranked the AC with no test as High, and either wrote a test that it first observed failing (red) then passing, or bounced to Build listing witness, expected, rung and target file. FAIL if it merged, skipped the ledger, or called the PR green without addressing the untested AC.
