import { expect, test } from 'claude-code/testing'

import { prOfReview, reviewerOf, roundsLine, roundsMet, roundsOf, withRead } from '../hooks/rounds'
import type { RoundsLog } from '../hooks/rounds'
import { assess } from '../hooks/score'
import { EMPTY_TALLY, added } from '../hooks/signals'
import { bareScan } from './kit'

const TODAY = '2026-11-02'

/** A log with `n` PRs read once each by one reviewer today, starting at #100. */
const oncePerPr = (n: number): RoundsLog => {
  let log: RoundsLog = {}
  for (let i = 0; i < n; i++) log = withRead(log, String(100 + i), 'trust-security', TODAY)
  return log
}

test('a reviewer run names its PR in the description only; a test replay names none', () => {
  // Arrange
  const runs = [
    { description: 'Review PR 869 safety plane', prompt: '' },
    { description: 'routing-referee round 3 #865', prompt: '' },
    { description: 'Security review of the fix commits', prompt: 'Review #831 at abc123' },
    { description: 'Golden run 1 task 4', prompt: 'Replay #683 at 1234abcd' },
    { description: 'trust-security T10 run 1', prompt: 'the diff of #720' },
  ]

  // Act
  const prs = runs.map(prOfReview)

  // Assert
  expect(prs).toEqual(['869', '865', '?', null, null])
})

test('the PR word wins over other numbers, task names do not hide a named PR, and other repositories do not count', () => {
  // Arrange
  const descriptions = [
    'Round #2 review of PR 869',
    'Review fix for issue #860 in PR #869',
    'trust-security re-review of #869 after T2 fix',
    'Review task 2 fixes on #869',
    'Re-review PR#869 fixes',
    'Review PR-872',
    'Re-review pull request 872',
    'Review other-plugins#22',
    'routing-referee round 3 on 865',
  ]

  // Act
  const prs = descriptions.map(description => prOfReview({ description }))

  // Assert
  expect(prs).toEqual(['869', '869', '869', '869', '869', '872', '872', '?', '?'])
})

test('only reviewer agents start a round', () => {
  expect(reviewerOf('Agent', { subagent_type: 'trust-security' })).toBe('trust-security')
  expect(reviewerOf('Agent', { subagent_type: 'routing-referee' })).toBe('routing-referee')
  expect(reviewerOf('Agent', { subagent_type: 'frontend' })).toBeNull()
  expect(reviewerOf('Skill', { skill: 'code-review' })).toBeNull()
})

test("a PR's rounds are its busiest reviewer's reads, so two reviewers on one head make one round", () => {
  // Arrange
  let log: RoundsLog = {}
  log = withRead(log, '865', 'routing-referee', TODAY)
  log = withRead(log, '865', 'routing-referee', TODAY)
  log = withRead(log, '865', 'routing-referee', TODAY)
  log = withRead(log, '683', 'routing-referee', TODAY)
  log = withRead(log, '683', 'trust-security', TODAY)
  log = withRead(log, '?', 'trust-security', TODAY)

  // Act
  const rounds = roundsOf(log, TODAY)

  // Assert
  expect(rounds.prs).toBe(2)
  expect(rounds.mean).toBe(2)
  expect(rounds.worst[0]).toEqual({ pr: '865', rounds: 3 })
  expect(rounds.unnamed).toBe(1)
  expect(rounds.named).toBe(5)
  expect(roundsLine(rounds)).toBe('rounds 2 over 2 PRs · worst #865 (3) · 1 review named no PR')
})

test('reads older than the window do not count, and reads older than sixty days are dropped', () => {
  // Arrange
  let log: RoundsLog = {}
  log = withRead(log, '500', 'trust-security', '2026-09-20')
  log = withRead(log, '501', 'trust-security', '2026-08-01', '2026-09-20')

  // Act
  const later = withRead(log, '502', 'trust-security', TODAY)
  const rounds = roundsOf(later, TODAY)

  // Assert
  expect(Object.keys(later).sort()).toEqual(['500', '502'])
  expect(rounds.prs).toBe(1)
})

test('rounds meet the bar at ten PRs under 1.5, with at most one review in ten unnamed', () => {
  // Arrange
  const nine = roundsOf(oncePerPr(9), TODAY)
  const ten = roundsOf(oncePerPr(10), TODAY)
  let slow = oncePerPr(10)
  for (let i = 0; i < 5; i++) slow = withRead(slow, String(100 + i), 'trust-security', TODAY)
  const oneUnnamed = roundsOf(withRead(oncePerPr(10), '?', 'trust-security', TODAY), TODAY)
  const twoUnnamed = roundsOf(withRead(withRead(oncePerPr(10), '?', 'a', TODAY), '?', 'a', TODAY), TODAY)

  // Act
  const met = [nine, ten, roundsOf(slow, TODAY), oneUnnamed, twoUnnamed].map(roundsMet)

  // Assert
  expect(roundsOf(slow, TODAY).mean).toBe(1.5)
  expect(met).toEqual([false, true, false, true, false])
})

test('review design reaches 4 on measured rounds only; a declaration no longer lifts it', () => {
  // Arrange
  const scan = bareScan({ claudeMdLines: 50, hasReviewMd: true, hasPrTemplate: true, hasCodeowners: true, reviewerAgents: 2 })
  const window = added(EMPTY_TALLY, { reviews: 3 })
  const declared = { rounds: { at: 1, note: 'tracked by hand' } }

  // Act
  const byDeclaring = assess(scan, window, declared, null)
  const byMeasuring = assess(scan, window, {}, null, roundsOf(oncePerPr(10), TODAY))
  const review = (one: typeof byDeclaring) => one.scores.find(score => score.id === 'review')

  // Assert
  expect(review(byDeclaring)?.score).toBe(3)
  expect(review(byDeclaring)?.evidence).toContain('rounds not measured yet')
  expect(review(byMeasuring)?.score).toBe(4)
  expect(review(byMeasuring)?.evidence).toContain('rounds 1 over 10 PRs')
})
