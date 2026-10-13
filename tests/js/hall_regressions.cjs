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
const requests = [];
let requestHandler = async () => ({});
const context = vm.createContext({
  state: { phase: 'select', selectedGame: 'rps' },
  games: [{ id: 'rps', enabled: true }, { id: 'dice', enabled: true }],
  activeGame: null,
  selectSource: null,
  boardInputSource: { local_client: false, enabled: false },
  requestJson(url, options) {
    requests.push({ url, payload: JSON.parse(options.body) });
    return requestHandler(url, options);
  },
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
vm.runInContext(source.slice(source.indexOf('// ---- 板端大厅选中项同步 ----') >= 0
  ? source.indexOf('// ---- 板端大厅选中项同步 ----') : source.indexOf('function selectDefaultGame()'),
  source.indexOf('// ---- 启动 ----')), context);
vm.runInContext(source.slice(source.indexOf('function returnToSelect()'), source.indexOf('// ---- 游戏列表 ----')), context);
vm.runInContext(source.slice(source.indexOf('function wakeFromStandby()'), source.indexOf('// --- 待机语音监听')), context);
const keyStart = source.indexOf("document.addEventListener('keydown', (event) => {");
vm.runInContext(source.slice(keyStart, source.indexOf('async function loadGames()', keyStart)), context);

async function main() {
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
// Local highlight and physical confirm share the same cursor after returning.
context.games = [{ id: 'dice', enabled: true }, { id: 'rps', enabled: true }];
context.boardInputSource = { local_client: true, enabled: true };
let boardCursor = 'rps';
requestHandler = async (_url, options) => { boardCursor = JSON.parse(options.body).game; return { game: boardCursor }; };
context.state.phase = 'result';
context.returnToSelect();
await new Promise(setImmediate);
assert.equal(boardCursor, 'dice', 'returning to the hall must sync the physical-confirm cursor');
assert.equal(requests.at(-1).url, '/api/input/board/selection');
assert.equal(context.state.selectedGame, 'dice');

// Delayed selection requests must remain in order; immediate confirmation waits.
let releaseFirst;
let requestCount = 0;
requestHandler = async (_url, options) => {
  const game = JSON.parse(options.body).game;
  requestCount++;
  if (requestCount === 1) await new Promise((resolve) => { releaseFirst = resolve; });
  boardCursor = game;
  return { game };
};
requests.length = 0; log.length = 0;
context.selectGame('rps');
const entering = context.enterSelectedGame();
context.selectGame('dice');
context.enterSelectedGame(); // Same physical Enter also arrives by the SSE path.
await new Promise(setImmediate);
assert.equal(requestCount, 1, 'only one selection request may be in flight');
assert.equal(context.state.phase, 'select', 'a delayed sync must not start a round');
releaseFirst();
await entering;
assert.deepEqual(requests.map((r) => r.payload.game), ['rps', 'dice']);
assert.equal(boardCursor, 'dice');
assert.equal(log.filter((entry) => entry.startsWith('enter:')).length, 1);
assert.equal(log.at(-1), 'enter:dice');

// Local navigation uses the existing board SSE only while its reader is usable.
context.state.phase = 'select';
context.boardInputSource.available = true;
context.selectSource = { readyState: 1 };
const beforeNativeArrow = requests.length;
press('ArrowDown');
assert.equal(context.state.selectedGame, 'dice', 'DOM echo must not navigate twice');
assert.equal(requests.length, beforeNativeArrow);
context.selectGame('rps'); // Existing board_navigate event supplies the target.
await context.enterSelectedGame();
assert.equal(boardCursor, 'rps');
context.boardInputSource.available = false;
context.state.phase = 'select';
press('ArrowUp');
assert.equal(context.state.selectedGame, 'dice', 'unavailable reader must retain web keyboard');
await new Promise(setImmediate);
context.selectSource = null;

// A mouse change to RPS is respected by the next confirm.
context.state.phase = 'select';
context.selectGame('rps');
await context.enterSelectedGame();
assert.equal(boardCursor, 'rps');
assert.equal(log.at(-1), 'enter:rps');

// Cancelled entry must not resume after returning to the hall.
context.state.phase = 'select';
let releaseCancelled;
requestHandler = () => new Promise((resolve) => { releaseCancelled = resolve; });
context.selectGame('dice');
const cancelledEntry = context.enterSelectedGame();
await new Promise(setImmediate);
context.returnToSelect();
requestHandler = async (_url, options) => ({ game: JSON.parse(options.body).game });
releaseCancelled({ game: 'dice' });
await cancelledEntry;
assert.equal(context.state.phase, 'select');

// Failed synchronization does not silently enter with an old board cursor.
let feedback = [];
context.toast = (message) => feedback.push(message);
requestHandler = async () => { throw new Error('offline'); };
context.selectGame('rps');
await context.enterSelectedGame();
assert.equal(context.state.phase, 'select');
assert.equal(feedback.length, 1);
requestHandler = async (_url, options) => { boardCursor = JSON.parse(options.body).game; return { game: boardCursor }; };
await context.enterSelectedGame();
assert.equal(boardCursor, 'rps');
assert.equal(context.state.phase, 'rules');

// A remote page chooses independently and never posts a board-cursor change.
context.state.phase = 'select';
context.boardInputSource = { local_client: false, enabled: true };
requests.length = 0;
context.selectGame('dice');
context.enterSelectedGame();
assert.equal(context.state.phase, 'rules');
assert.deepEqual(requests, []);

// No enabled game -> no sync request and no entry.
context.state.phase = 'select';
context.boardInputSource = { local_client: true, enabled: true };
context.games = [{ id: 'dice', enabled: false }];
context.selectDefaultGame();
await context.enterSelectedGame();
assert.equal(context.state.selectedGame, null);
assert.equal(context.state.phase, 'select');
assert.deepEqual(requests, []);
console.log('Hall defaults, local cursor sync, ordered requests, immediate/duplicate confirm, cancellation, retry and remote selection passed.');
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
