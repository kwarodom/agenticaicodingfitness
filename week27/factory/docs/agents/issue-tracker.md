# Issue tracker conventions

## Ticket format
```markdown
## Problem
<1–3 sentences from the user's side. No implementation plan.>

## Context
- **Where:** <page>, `<route>`
  - a. <step>
  - b. <step>
- **Who:** <who sees it>

## Fix
<intended change in 1–3 lines>

## Acceptance Criteria
- [ ] <checkable outcome a test can assert>
- [ ] <checkable outcome>

## Risk
🟢 Low · <one line>

## Blocked by
- None.
```

## Lint rules
| Rule | Value |
|---|---|
| Title | `<emoji> [<kind>] <scope>: <what>` · ≤ 70 chars |
| Acceptance Criteria | 2–5 `- [ ]` lines, each checkable; never "works well", "is fast" |
| Risk | 🟢 Low · 🟡 Medium · 🔴 High + one line |
| Blocked by | `- #N — why` or exactly `- None.` |
| Labels routing merge to a human | `money`, `auth`, `schema` (destructive only), `control`, `tenant`, `pdpa` |
| Size | one card = one PR ≤ 400 changed lines; bigger → split |
| Board columns | Backlog → Ready → Building → QA → Review → Done, plus Blocked |
