import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const root = process.argv[2] || process.cwd();
const transport = fs.readFileSync(`${root}/web/shared-session.js`, 'utf8');
const { createSessionClient } = await import(`data:text/javascript;base64,${Buffer.from(transport).toString('base64')}`);
class Source {
  static all = [];
  constructor(url) { this.url = url; this.readyState = 1; this.events = {}; Source.all.push(this); }
  addEventListener(name, fn) { this.events[name] = fn; }
  close() { this.readyState = 2; }
  emit(name, data) { this.events[name]?.({ data: JSON.stringify(data) }); }
}
globalThis.EventSource = Source;
let current = { session_id: 'session-1', revision: 1, shared: true, phase: 'select', selected_game: 'dice', round_id: null };
const history = [];
const requests = [];
let created = 0;
async function request(url, options) {
  if (url === '/api/session') return structuredClone(current);
  const cmd = JSON.parse(options.body); requests.push(cmd);
  if (cmd.action === 'select') current.selected_game = cmd.game;
  if (cmd.action === 'enter' && !current.round_id) {
    created++; current.phase = 'game'; current.round_id = 'round-1'; current.game_id = current.selected_game;
    current.round = { round_id: 'round-1', state: 'rules', state_version: 1, status: 'running' };
  }
  current.revision++;
  for (const source of Source.all) source.emit('update', structuredClone(current));
  return structuredClone(current);
}
const a = createSessionClient({ requestJson: request, onSnapshot: (s) => history.push(['A', s]) });
const b = createSessionClient({ requestJson: request, onSnapshot: (s) => history.push(['B', s]) });
await a.connect(); await b.connect();
await a.action('select', { game: 'rps' });
assert.equal(b.snapshot.selected_game, 'rps');
await Promise.all([a.action('enter'), b.action('enter')]);
assert.equal(created, 1);
assert.equal(a.snapshot.round_id, b.snapshot.round_id);
const reloaded = createSessionClient({ requestJson: request, onSnapshot() {} });
await reloaded.connect();
assert.equal(reloaded.snapshot.game_id, 'rps');
assert.equal(created, 1);
a.close();
assert.equal(b.snapshot.round_id, 'round-1');
const previous = b.snapshot.revision;
Source.all[1].emit('update', { ...current, revision: 0, round_id: 'old' });
assert.equal(b.snapshot.revision, previous);
await b.action('intent', { intent: 'confirm' });
assert.equal(requests.at(-1).state, 'rules');
assert.equal(requests.at(-1).state_version, 1);
assert.ok(requests.at(-1).request_id);

// Exercise the real RoundClient: attach restores projection without creating a
// round, replaying old announcements, acknowledging speech or cancelling peers.
const app = fs.readFileSync(`${root}/web/app.js`, 'utf8');
const log = [];
const snap = { round_id: 'round-1', game_id: 'rps', state: 'result', state_version: 8,
  status: 'running', event_sequence: 31, result: { winner_role: 'PLAYER' },
  observations: [{ event: 'video', url: 'http://video' }, { event: 'robot', phase: 'completed' }],
  events: [{ event: 'speech', sequence: 2, text: 'old rules' }, { event: 'state_changed', sequence: 31, state: 'result' }] };
const ctx = vm.createContext({
  sharedSession: { shared: true, subscribeRound() { return () => {}; }, action: async (...args) => { log.push(['action', ...args]); return { round_id: 'round-1', round: snap }; } },
  EventSource: Source, showAsrFeedback() {},
  requestJson: async (url, options) => { log.push(['request', url, options]); return snap; },
});
vm.runInContext(app.slice(app.indexOf('function createRoundClient('), app.indexOf('// ---- 网络 ----')), ctx);
const client = ctx.createRoundClient('rps', {
  onSpeech() { log.push(['speech']); }, onSyncState(s) { log.push(['sync', s.state]); },
  onEvent(e) { log.push([e.event]); },
});
await client.attach('round-1');
assert.ok(log.some((entry) => entry[0] === 'sync' && entry[1] === 'result'));
assert.ok(log.some((entry) => entry[0] === 'video'));
assert.ok(log.some((entry) => entry[0] === 'robot'));
assert.equal(log.filter((entry) => entry[0] === 'speech').length, 0);
await client.submitIntent('speech_done', { directive_id: 'old' });
assert.equal(log.filter((entry) => entry[0] === 'action').length, 0);
await client.cancel();
assert.equal(log.filter((entry) => entry[0] === 'request' && entry[2]?.method === 'POST').length, 0);
console.log('PASS: shared two-client selection, one round, reconnect, stale snapshot, state guard and detach without cancellation');
