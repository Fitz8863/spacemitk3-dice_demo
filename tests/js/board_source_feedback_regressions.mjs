// 板端键盘合成事件 vs 语音事件的反馈分流回归（驱动真实 web/app.js）。
//
// 「听到「board」，已生效」误弹的回归钉：服务端把板端键盘的按键包装成
// 待机/列表语音总线事件（text 恒为 "board"），前端曾把所有总线事件都当
// 语音渲染 showAsrFeedback。2026-10-10 修法=合成事件带 source:'board'，
// 前端对 board 来源只做动作（唤醒/进局/高亮），不弹语音浮条。
// 自查法：把 harness 指向 `git show HEAD:` 的旧 app.js 应当 FAIL。
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import path from 'node:path';

const root = path.resolve(process.argv[2] || '.');
const app = fs.readFileSync(path.join(root, 'web/app.js'), 'utf8');

const sources = [];
class EventSource {
  constructor(url) { this.url = url; this.listeners = {}; sources.push(this); }
  addEventListener(name, fn) { this.listeners[name] = fn; }
  close() { this.closed = true; }
}

const feedback = [];
const actions = { wake: 0, enterGame: [], enterSelected: 0, selectGame: [] };
const context = vm.createContext({
  EventSource,
  requestJson: async () => ({}),
  $: () => null,
  showAsrFeedback: (event) => feedback.push(event),
  wakeFromStandby: () => { actions.wake++; },
  enterGameById: (id) => { actions.enterGame.push(id); },
  enterSelectedGame: () => { actions.enterSelected++; },
  selectGame: (id) => { actions.selectGame.push(id); },
});

// 待机层：从监听注释起、到列表层注释止（含 renderStandbyVoiceHint 与
// stopStandbyListening）。standbySource 的原声明在切片外（app.js 顶部），
// 这里预声明补位；selectSource 的声明在列表层切片内自带。
vm.runInContext('let standbySource = null;', context);
const standbySlice = app.slice(
  app.indexOf('// --- 待机语音监听'),
  app.indexOf('// --- 列表页语音选游戏'),
);
vm.runInContext(standbySlice, context);
// 列表层：到 enterGameById 定义前止（本 harness 自带同名桩）。
const selectSlice = app.slice(
  app.indexOf('// --- 列表页语音选游戏'),
  app.indexOf('function enterGameById('),
);
vm.runInContext(selectSlice, context);

const bus = (payload) => ({ data: JSON.stringify({ timestamp_ms: Date.now(), ...payload }) });
const standbyBus = () => sources[0].listeners.bus;
const selectBus = () => sources[1].listeners.bus;

// --- 待机层 ---
context.startStandbyListening();
assert.equal(sources[0].url, '/api/asr/standby/stream');

// 板端按键唤醒：动作照做，不弹语音浮条
standbyBus()(bus({ status: 'wake', text: 'board', source: 'board' }));
assert.equal(actions.wake, 1);
assert.equal(feedback.length, 0, 'board-sourced wake must not show voice feedback');

// 真语音唤醒：浮条照弹（VM 跨 realm 对象原型不同，逐字段断言）
standbyBus()(bus({ status: 'wake', text: '开始游戏' }));
assert.equal(actions.wake, 2);
assert.equal(feedback.length, 1);
assert.equal(feedback[0].status, 'submitted');
assert.equal(feedback[0].text, '开始游戏');

// 板端按键直达对局：进局但不弹浮条
standbyBus()(bus({ status: 'selected', game_id: 'dice', text: 'board', source: 'board' }));
assert.deepEqual(actions.enterGame, ['dice']);
assert.equal(feedback.length, 1, 'board-sourced selected must not show voice feedback');

// 板端来源的未知状态：静默（连 unmatched 浮条也不弹）
standbyBus()(bus({ status: 'other', text: 'board', source: 'board' }));
assert.equal(feedback.length, 1);
// 真语音未匹配：unmatched 浮条照弹
standbyBus()(bus({ status: 'other', text: '随便说说' }));
assert.equal(feedback.length, 2);
assert.equal(feedback[1].status, 'unmatched');

// --- 列表层 ---
context.startSelectListening();
assert.equal(sources[1].url, '/api/asr/select/stream');

// 板端光标同步：只挪高亮（原有行为不变）
selectBus()(bus({ status: 'board_navigate', game_id: 'rps', text: 'board', source: 'board' }));
assert.deepEqual(actions.selectGame, ['rps']);
assert.equal(feedback.length, 2);

// 板端绿键进局：进局但不弹浮条
selectBus()(bus({ status: 'selected', game_id: 'dice', text: 'board', source: 'board' }));
assert.deepEqual(actions.enterGame, ['dice', 'dice']);
assert.equal(feedback.length, 2, 'board-sourced select must not show voice feedback');

// 真语音确认：浮条 + 进局
selectBus()(bus({ status: 'confirm', text: '进入游戏' }));
assert.equal(actions.enterSelected, 1);
assert.equal(feedback.length, 3);
assert.equal(feedback[2].status, 'submitted');
assert.equal(feedback[2].text, '进入游戏');

console.log('PASS: board-sourced bus events act without voice toasts; voice events still toast');
