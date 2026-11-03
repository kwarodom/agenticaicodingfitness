import { expect, test } from 'claude-code/testing'

import { assess, beltOf } from '../hooks/score'
import { isPathScoped, goldenTasks } from '../hooks/scan'
import { EMPTY_TALLY, added, isBroadApproval, isCommit, isCorrection, isPrCreate, isTestCommand, missingSections, namesVerifier, prBodyOf, windowOf } from '../hooks/signals'
import { bareScan } from './kit'

test('corrections are recognised in English and Thai, and ordinary prompts are not', () => {
  // Arrange
  const fixes = ['No, that is the wrong file', "that's not what I asked for", 'still failing on the same test', 'it didn’t work', 'ไม่ใช่ครับ แก้ไฟล์อื่น', 'ยังไม่ผ่านเลย', 'ผิดครับ']
  const plain = ['Add a seat meter to Settings', 'Notice how the band wraps', 'Now run the tests', '/clear', 'สรุปการจัดซื้อสัปดาห์นี้']

  // Act
  const caught = fixes.map(isCorrection)
  const passed = plain.map(isCorrection)

  // Assert
  expect(caught.filter(Boolean)).toHaveLength(fixes.length - 1)
  expect(passed.some(Boolean)).toBe(false)
})

test('a prompt that names its check is told apart from one that does not', () => {
  expect(namesVerifier('Done when pytest core/router passes')).toBe(true)
  expect(namesVerifier('Prove to me this works')).toBe(true)
  expect(namesVerifier('เสร็จเมื่อเทสผ่านทั้งหมด')).toBe(true)
  expect(namesVerifier('Add a button')).toBe(false)
})

test('test runs, commits and PR creation are recognised in shell commands', () => {
  expect(isTestCommand('cd backend && python -m pytest core/router -q')).toBe(true)
  expect(isTestCommand('npx vitest run src/test')).toBe(true)
  expect(isTestCommand('bash .claude/hooks/tests/run_hook_tests.sh')).toBe(true)
  expect(isTestCommand('git status')).toBe(false)
  expect(isCommit('git -C /w commit -q -F - <<EOF')).toBe(true)
  expect(isCommit('git commit-tree abc')).toBe(false)
  expect(isPrCreate('gh pr create --draft --title x')).toBe(true)
})

test('a PR body in the command is read for its three sections; one from a file is unseen, never missing', () => {
  // Arrange
  const full = "gh pr create --body-file - <<'EOF'\n## Testing\nok\n## Risk\nLow\n## How to revert\ngit revert\nEOF"
  const thin = 'gh pr create --body "## Summary\nfixes it"'
  const fromFile = 'gh pr create --body-file /tmp/body.md'

  // Act
  const bodies = [full, thin, fromFile].map(prBodyOf)

  // Assert
  expect(missingSections(bodies[0]!)).toEqual([])
  expect(missingSections(bodies[1]!)).toEqual(['Testing', 'Risk', 'How to revert'])
  expect(bodies[2]).toEqual({ isSeen: false })
  expect(missingSections(bodies[2]!)).toEqual([])
})

test('only approvals that can run anything count as broad', () => {
  expect(['Bash(python3 *)', 'Bash(python3 -)', 'Bash(gh api *)', 'Bash', "Bash(backend/.venv/bin/python -c ' *)"].every(isBroadApproval)).toBe(true)
  expect(['Bash(git status)', 'Bash(npm test)', 'Bash(gh pr *)', 'Read(//tmp/**)'].some(isBroadApproval)).toBe(false)
})

test('front matter with paths: scopes a rule; none, or front matter that never closes, does not', () => {
  expect(isPathScoped('---\npaths:\n  - "**/*.py"\n---\n# x')).toBe(true)
  expect(isPathScoped('# no front matter')).toBe(false)
  expect(isPathScoped('---\npaths: ["*.ts"]\n# never closed')).toBe(false)
})

test('a golden set counts list items, numbered items and headings as tasks', () => {
  expect(goldenTasks('router.md', '# Router\n- one\n- two\n1. three\n## four\nprose')).toBe(4)
  expect(goldenTasks('router.jsonl', '{"q":1}\n{"q":2}\n')).toBe(2)
})

test('the window sums the last fourteen days only', () => {
  // Arrange
  const days = { '2026-10-29': added(EMPTY_TALLY, { prompts: 3 }), '2026-10-16': added(EMPTY_TALLY, { prompts: 5 }), '2026-10-15': added(EMPTY_TALLY, { prompts: 100 }) }

  // Act
  const window = windowOf(days, '2026-10-29', 14)

  // Assert
  expect(window.prompts).toBe(8)
})

test('with too little behaviour, verification and context are not measured, and the belt cannot pass Green', () => {
  // Arrange
  const scan = bareScan({ claudeMdLines: 120, alwaysOnBytes: 9_000, hooks: 3, preToolUseHooks: 2, guardHooks: 1, hasHookTests: true, skills: 1, agents: 1, goldenSets: 1, hasReviewMd: true, hasPrTemplate: true, hasCodeowners: true, worktrees: 2, drafts: 1 })

  // Act
  const result = assess(scan, EMPTY_TALLY, {}, null)

  // Assert
  expect(result.scores.find(one => one.id === 'verification')?.score).toBeNull()
  expect(result.scores.find(one => one.id === 'context')?.score).toBeNull()
  expect(result.measured).toBe(6)
  expect(['white', 'green']).toContain(result.belt)
  expect(result.gate).toContain('measured')
})

test('the lowest score sets the belt, and Black needs a declared field deployment', () => {
  expect(beltOf([4, 4, 4, 4, 4, 4, 4, 4], {})).toBe('brown')
  expect(beltOf([4, 4, 4, 4, 4, 4, 4, 4], { field: { at: 1, note: '' } })).toBe('black')
  expect(beltOf([3, 3, 3, 3, 3, 3, 3, 3], {})).toBe('blue')
  expect(beltOf([3, 3, 3, 3, 3, 3, 2, 3], {})).toBe('green')
  expect(beltOf([4, 4, 4, 4, 4, 4, 4, null], {})).toBe('green')
  expect(beltOf([null, null, null, null, null, null, null, null], {})).toBe('none')
})

test('the next changes start from the lowest scores', () => {
  // Arrange
  const scan = bareScan({ claudeMdLines: 222, alwaysOnBytes: 35_812, rulesTotal: 10, rulesUnscoped: 10, hooks: 12, preToolUseHooks: 3, guardHooks: 2, hasHookTests: true, skills: 2, broadApprovals: 0 })

  // Act
  const result = assess(scan, EMPTY_TALLY, {}, null)

  // Assert
  expect(result.next).toHaveLength(3)
  expect(result.next.some(line => line.startsWith('Automation'))).toBe(false)
  expect(result.scores.find(one => one.id === 'automation')?.score).toBe(4)
  expect(result.scores.find(one => one.id === 'instructions')?.score).toBe(2)
})

test('a commit habit with tested commits and evidenced PRs scores verification at 4', () => {
  // Arrange
  const tally = added(EMPTY_TALLY, { commits: 10, testedCommits: 10, prsSeen: 3, prsEvidenced: 3 })

  // Act
  const result = assess(bareScan({ claudeMdLines: 50 }), tally, {}, null)

  // Assert
  expect(result.scores.find(one => one.id === 'verification')?.score).toBe(4)
})

test('a defined reviewer agent lifts review design to 2 even before any review files exist', () => {
  // Act
  const result = assess(bareScan({ claudeMdLines: 50, agents: 3, reviewerAgents: 1 }), EMPTY_TALLY, {}, null)

  // Assert
  expect(result.scores.find(one => one.id === 'review')?.score).toBe(2)
})

test('three corrections in a row and no /clear hold context hygiene at 0', () => {
  // Arrange
  const tally = added(EMPTY_TALLY, { prompts: 40, thirdStrikes: 4, turns: 30 })

  // Act
  const result = assess(bareScan({ claudeMdLines: 50 }), tally, {}, null)

  // Assert
  expect(result.scores.find(one => one.id === 'context')?.score).toBe(0)
})
