const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../web/display-layout.js'), 'utf8');
const api = vm.runInNewContext(source.replaceAll('export function', 'function') + '\n({resolveDisplayLayout, applyScoreLayout})');
for (const side of ['LEFT', 'RIGHT']) {
  test(`${side} display moves complete cards without mutating physical mapping`, () => {
    const manifest = { participants: { player: 'LEFT', agent: 'RIGHT' }, display: {
      player_side: side, agent_side: side === 'LEFT' ? 'RIGHT' : 'LEFT', video_rotation_deg: 0,
    }};
    const original = JSON.stringify(manifest);
    const display = api.resolveDisplayLayout(manifest);
    const player = { style: {}, score: 10, gesture: '布' };
    const agent = { style: {}, score: 19, gesture: '石头' };
    api.applyScoreLayout(player, agent, display);
    assert.equal(player.style.gridColumn, side === 'LEFT' ? '1' : '3');
    assert.equal(agent.style.gridColumn, side === 'LEFT' ? '3' : '1');
    assert.equal(player.style.gridRow, agent.style.gridRow);
    assert.equal(player.score, 10);
    assert.equal(agent.gesture, '石头');
    assert.equal(JSON.stringify(manifest), original);
  });
}
test('no display config preserves legacy physical placement', () => {
  const d = api.resolveDisplayLayout({ participants: { player: 'RIGHT', agent: 'LEFT' } });
  assert.equal(d.player_side, 'RIGHT');
  assert.equal(d.video_rotation_deg, 0);
});
