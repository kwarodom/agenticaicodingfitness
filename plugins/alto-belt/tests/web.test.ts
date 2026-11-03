import { expect, test } from 'claude-code/testing'

import { assess } from '../hooks/score'
import { EMPTY_TALLY } from '../hooks/signals'
import { BELT_IDS, assetsScript, dataScript, webDataOf, webDirOf } from '../hooks/web'
import { bareScan } from './kit'

const LIVE = { prompts: 3, streak: 0, contextPercent: 20, testsSinceCommit: 0, commits: 1 }

test('the page data carries the belt, all eight skills and the next moves', () => {
  // Arrange
  const assessment = assess(bareScan({ claudeMdLines: 50 }), EMPTY_TALLY, {}, null)

  // Act
  const data = webDataOf(assessment, bareScan({ name: 'demo' }), LIVE, null, [], 1)

  // Assert
  expect(data.repo).toBe('demo')
  expect(data.scores).toHaveLength(8)
  expect(data.next.length).toBeGreaterThan(0)
  expect(data.scores[0]).toHaveProperty('name', 'Verification')
})

test('text that could close the script tag is escaped in data.js', () => {
  // Arrange
  const assessment = assess(bareScan({ claudeMdLines: 50 }), EMPTY_TALLY, {}, null)
  const data = webDataOf(assessment, bareScan({ name: '</script><b>x' }), LIVE, null, [], 1)

  // Act
  const script = dataScript(data)

  // Assert
  expect(script).not.toContain('</script>')
  expect(script).toStartWith('window.__BELT_DATA__ = ')
})

test('each repository gets its own page folder under ~/.claude/alto-belt', () => {
  expect(webDirOf('/home/dev/', 'app-1a2b')).toBe('/home/dev/.claude/alto-belt/app-1a2b')
})

test('the art script names one badge for each of the five belts', () => {
  // Arrange
  const art = Object.fromEntries(BELT_IDS.map(id => [id, `data:image/webp;base64,${id}`]))

  // Act
  const script = assetsScript(art)

  // Assert
  expect(script).toStartWith('window.__BELT_ART__ = ')
  for (const id of ['white', 'green', 'blue', 'brown', 'black']) expect(script).toContain(`"${id}":"data:image/webp;base64,`)
})
