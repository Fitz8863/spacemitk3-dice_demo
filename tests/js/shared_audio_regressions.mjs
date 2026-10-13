import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const root = process.argv[2] || process.cwd();
const app = fs.readFileSync(`${root}/web/app.js`, 'utf8');
let shown = false, fetches = 0, acks = 0;
const ctx = vm.createContext({
  sharedSession: { shared: true }, boardInputSource: { local_client: true },
  state: { sound: true, ttsRequestId: 0 }, activeRound: null,
  $(id) { return { classList: { remove() { shown = true; }, add() {} }, addEventListener() {} }; },
  stopSpeech() {}, getSpeechAudioContext() { return { state: 'suspended' }; },
  fetch: async () => { fetches++; throw new Error('should not fetch'); },
});
const start = app.indexOf('// All remote displays consume');
const end = app.indexOf('// ---- 页面离开保险 ----', start);
vm.runInContext(app.slice(start, end), ctx);
const round = { roundId: 'round-1', submitIntent() { acks++; } };
await ctx.playSharedDirective(round, { directive_id: 'x' });
assert.equal(fetches, 0, 'local display must not echo the backend loudspeaker');
ctx.boardInputSource.local_client = false;
ctx.state.sound = false;
await ctx.playSharedDirective(round, { directive_id: 'x' });
assert.equal(fetches, 0, 'muted display must not create or advance speech');
ctx.state.sound = true;
await ctx.playSharedDirective(round, { directive_id: 'x' });
assert.equal(shown, true, 'autoplay-blocked remote gets a sound unlock action');
assert.equal(acks, 0, 'web playback and mute must not acknowledge shared speech');
console.log('PASS: local echo suppression, remote mute, autoplay unlock and no shared web speech acknowledgements');

// Playback/speaker failures are deliberately silent in the UI. The backend
// continues the shared round and keeps the diagnostic in its own event/log.
const clientSource = app.slice(app.indexOf('function createRoundClient('), app.indexOf('// ---- 网络 ----'));
let toastCount = 0;
let stream;
const clientContext = vm.createContext({
  sharedSession: undefined,
  requestJson: async () => ({ round_id: 'round-2' }),
  showAsrFeedback() {},
  toast() { toastCount += 1; },
  EventSource: class {
    constructor() { stream = this; this.handlers = {}; }
    addEventListener(name, fn) { this.handlers[name] = fn; }
    close() {}
    emit(name, data) { this.handlers[name]?.({ data: JSON.stringify(data) }); }
  },
});
vm.runInContext(clientSource, clientContext);
const client = clientContext.createRoundClient('dice', {});
await client.start();
stream.emit('update', { status: 'running', state: 'rules', event_sequence: 1, revision: 1,
  events: [{ event: 'audio_error', kind: 'board_playback', sequence: 1 }] });
assert.equal(toastCount, 0, 'audio failures must not show a frontend toast');
console.log('PASS: audio errors stay silent in the frontend');
