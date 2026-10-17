// Exercise the real entry paths without connecting to devices or a game server.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import path from 'node:path';
const root = path.resolve(process.argv[2] || '.');
const app = fs.readFileSync(path.join(root, 'web/app.js'), 'utf8');
const cached = { id: 'dice', enabled: true, participants: { player: 'LEFT', agent: 'RIGHT' },
  display: { player_side: 'LEFT', agent_side: 'RIGHT', video_rotation_deg: 0 } };
const fresh = { ...cached,
  display: { player_side: 'RIGHT', agent_side: 'LEFT', video_rotation_deg: 180 } };

function context() {
  const entries = [];
  const ctx = vm.createContext({
    console, clearTimeout, idleTimer: null, clearIdleReturn() {},
    standbySource: null, selectSource: null, sessionApplying: false,
    sessionRoundId: null, lastSharedMode: false, sessionRender: Promise.resolve(),
    activeGame: null, games: [cached], boardInputSource: { local_client: false },
    sharedSession: { shared: true, snapshot: null },
    gameModules: { dice: { async enter(manifest) { entries.push(manifest); ctx.state.phase = 'rules'; }, teardown() {} } },
    refreshDisplayManifest: async () => fresh, requestJson: async () => ({}),
    setPhase() {}, enterStandby() {}, selectGame() {}, returnToSelect() {},
    state: { phase: 'select', selectedGame: 'dice' },
    boardSelectionEntry: null, boardSelectionVersion: 0, boardSelectionSync: Promise.resolve(true),
    syncBoardSelection() {}, stopSelectListening() {}, toast() {},
  });
  vm.runInContext(app.slice(app.indexOf('function applySessionSnapshot('),
    app.indexOf('const CONTROLLER_COPY =')), ctx);
  vm.runInContext(app.slice(app.indexOf('function enterSelectedGame('),
    app.indexOf('// ---- 启动 ----')), ctx);
  return { ctx, entries };
}
const snapshot = { shared: true, round_id: 'round-1', game_id: 'dice' };
{
  const { ctx, entries } = context();
  ctx.sharedSession.snapshot = snapshot;
  ctx.applySessionSnapshot(snapshot);
  await ctx.sessionRender;
  assert.equal(entries.length, 1);
  assert.equal(entries[0].display.player_side, 'RIGHT', 'shared entry must refresh cached display');
  assert.equal(entries[0].display.video_rotation_deg, 180);
  assert.equal(entries[0].participants.player, 'LEFT');
}
{
  const { ctx, entries } = context();
  let complete;
  ctx.refreshDisplayManifest = () => new Promise(resolve => { complete = resolve; });
  ctx.sharedSession.snapshot = snapshot;
  ctx.applySessionSnapshot(snapshot);
  await Promise.resolve(); await Promise.resolve();
  ctx.sharedSession.snapshot = { shared: true, round_id: null, phase: 'select' };
  complete(fresh);
  await ctx.sessionRender;
  assert.equal(entries.length, 0, 'a delayed configuration response must not reopen an exited round');
}
for (const local of [false, true]) {
  const { ctx, entries } = context();
  ctx.sharedSession = null;
  ctx.boardInputSource.local_client = local;
  await Promise.all([ctx.enterSelectedGame(), ctx.enterSelectedGame()]);
  assert.equal(entries.length, 1, 'parallel confirmation must create only one game entry');
  assert.equal(entries[0].display.player_side, 'RIGHT');
  assert.equal(entries[0].display.video_rotation_deg, 180);
  assert.equal(ctx.boardSelectionEntry, null);
}
{
  const { ctx, entries } = context();
  ctx.sharedSession = null;
  let complete;
  ctx.refreshDisplayManifest = () => new Promise(resolve => { complete = resolve; });
  const pending = ctx.enterSelectedGame();
  ctx.state.selectedGame = 'rps';
  complete(fresh);
  await pending;
  assert.equal(entries.length, 0, 'changing selection must cancel an old delayed entry');
}
console.log('PASS: fresh shared/local display, consistent video rotation, duplicate confirmation and stale entry protection');
