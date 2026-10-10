const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[2], 'utf8');

const cards = ['dice', 'rps', 'other'].map((id) => ({
  dataset: { game: id }, selected: false, attributes: {},
  classList: { toggle(_key, value) { cards.find((card) => card.dataset.game === id).selected = value; } },
  setAttribute(key, value) { this.attributes[key] = value; },
}));
const log = [];
const context = vm.createContext({
  state: { phase: 'select', selectedGame: 'rps' },
  games: [{ id: 'rps', enabled: true }, { id: 'dice', enabled: true }],
  activeGame: null,
  gameModules: {
    dice: { enter(manifest) { log.push(`enter:${manifest.id}`); context.state.phase = 'rules'; } },
    rps: { enter(manifest) { log.push(`enter:${manifest.id}`); context.state.phase = 'rules'; } },
  },
  window: { ArenaTheme: { renderPhase() {} } },
  document: { querySelectorAll: () => cards, addEventListener(type, handler) { if (type === 'keydown') context.keydown = handler; } },
  setIdleReturn(value) { log.push(`idle:${value}`); },
  stopStandbyListening() { log.push('stop-standby'); },
  startSelectListening() { log.push('listen-select'); },
  stopSelectListening() { log.push('stop-select'); },
  stopSpeech() { log.push('stop-speech'); },
  setPhase(phase) { context.state.phase = phase; log.push(phase); },
  toast(message) { throw Error(message); },
});
vm.runInContext(source.slice(source.indexOf('function selectDefaultGame()'), source.indexOf('// ---- 启动 ----')), context);
vm.runInContext(source.slice(source.indexOf('function returnToSelect()'), source.indexOf('// ---- 游戏列表 ----')), context);
vm.runInContext(source.slice(source.indexOf('function wakeFromStandby()'), source.indexOf('// --- 待机语音监听')), context);
const keyStart = source.indexOf("document.addEventListener('keydown', (event) => {");
vm.runInContext(source.slice(keyStart, source.indexOf('async function loadGames()', keyStart)), context);

// Prefer dice even when the backend lists RPS first. A selection is not a round.
context.selectDefaultGame();
assert.equal(context.state.selectedGame, 'dice');
assert.equal(cards[0].attributes['aria-selected'], 'true');
assert.equal(cards[1].attributes['aria-selected'], 'false');
assert.deepEqual(log, []);

function press(key, repeat = false) { context.keydown({ key, repeat, preventDefault() {} }); }
press('ArrowDown'); assert.equal(context.state.selectedGame, 'rps');
press('ArrowUp'); assert.equal(context.state.selectedGame, 'dice');
press('ArrowDown', true); assert.equal(context.state.selectedGame, 'dice');
press('ArrowDown'); press('Enter');
assert.equal(context.state.phase, 'rules');
assert.deepEqual(log, ['stop-select', 'enter:rps']);

// Returning from the auxiliary game keeps teardown/listening and resets selection.
log.length = 0;
context.activeGame = { teardown() { log.push('teardown'); } };
context.returnToSelect();
assert.equal(context.activeGame, null);
assert.equal(context.state.phase, 'select');
assert.equal(context.state.selectedGame, 'dice');
assert.deepEqual(log, ['idle:false', 'teardown', 'stop-standby', 'select', 'listen-select']);
press('Enter');
assert.equal(log.at(-1), 'enter:dice');

// An idle visitor may have highlighted RPS. A fresh wake restores the main
// challenge without entering it; a server-selected game can still override it.
context.state.phase = 'standby'; context.state.selectedGame = 'rps';
log.length = 0;
assert.equal(context.wakeFromStandby(), true);
assert.equal(context.state.selectedGame, 'dice');
assert.deepEqual(log, ['select', 'stop-standby', 'listen-select']);
assert.equal(context.wakeFromStandby(), false);

context.games = [{ id: 'dice', enabled: false }, { id: 'rps', enabled: true }];
context.selectDefaultGame(); assert.equal(context.state.selectedGame, 'rps');
context.games = [{ id: 'other', enabled: true }];
context.selectDefaultGame(); assert.equal(context.state.selectedGame, 'other');
context.games = [{ id: 'dice', enabled: false }, { id: 'rps', enabled: false }];
context.selectDefaultGame(); assert.equal(context.state.selectedGame, null);
assert.ok(cards.every((card) => card.attributes['aria-selected'] === 'false'));
console.log('Hall defaults, return/teardown, standby wake, key selection, confirmation and disabled-game fallback passed.');
