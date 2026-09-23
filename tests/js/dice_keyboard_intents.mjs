// 用 Node 驱动真实的前端游戏模块，验证"按键"与"屏幕按钮"落在同一个动作上。
//
// 这是唯一能真正验证键盘映射的方式：web/games/dice.js 是纯 ES 模块，
// 所有 DOM 与引擎依赖都从 register(engine) 注入，所以可以在没有浏览器的
// 情况下把 onKey 跑起来，并比较它与按钮 click 监听器产生的意图是否一致。
//
// 用法: node tests/js/dice_keyboard_intents.mjs <web/games/dice.js 绝对路径>
// 退出码 0 = 全部符合预期，1 = 有断言失败。

const dicePath = process.argv[2];
if (!dicePath) {
  console.error('用法: node dice_keyboard_intents.mjs <web/games/dice.js 绝对路径>');
  process.exit(2);
}

function makeElement() {
  return {
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    style: {},
    dataset: {},
    listeners: {},
    addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); },
    removeEventListener() {},
    textContent: '',
    innerHTML: '',
    src: '',
    offsetWidth: 1,
    querySelector: () => makeElement(),
  };
}

globalThis.document = { querySelector: () => makeElement(), addEventListener() {} };
globalThis.window = { location: { href: 'http://127.0.0.1:8080/' } };

const { register } = await import(dicePath);

const elements = new Map();
const $ = (id) => {
  if (!elements.has(id)) elements.set(id, makeElement());
  return elements.get(id);
};

const calls = [];
const state = { phase: '', sound: true };
let onStateChange = null;

const roundStub = {
  roundId: 'probe-round',
  async start() {},
  submitIntent(intent, payload) {
    if (roundStub.rejectWith) {
      const error = roundStub.rejectWith;
      roundStub.rejectWith = null;
      return Promise.reject(error);
    }
    calls.push([intent, payload]);
    return Promise.resolve();
  },
  cancel() {},
};

let returnToSelectCalls = 0;

const engine = {
  state,
  $,
  setPhase(phase) { state.phase = phase; },
  toast() {},
  stopSpeech() {},
  async requestJson() { return {}; },
  returnToSelect() { returnToSelectCalls += 1; },
  createRoundClient(_gameId, callbacks) {
    onStateChange = callbacks.onStateChange;
    return roundStub;
  },
  playDirective() {},
  setActiveRound() {},
  setIdleReturn() {},
};

const game = register(engine);
await game.enter({ id: 'dice', participants: { player: 'LEFT', agent: 'RIGHT' } });

// 后端把状态机推进到某个状态时，前端看到的就是这条 state_changed。
function enterState(name, view) {
  onStateChange(name, { view }, {});
}

function press(key) {
  calls.length = 0;
  game.onKey({ key });
  return JSON.stringify(calls);
}

function clickButton(id) {
  calls.length = 0;
  const listeners = elements.get(id).listeners.click || [];
  listeners.forEach((fn) => fn());
  return JSON.stringify(calls);
}

const failures = [];

function expect(label, actual, expected) {
  if (actual !== expected) {
    failures.push(`✗ ${label}\n    实际: ${actual}\n    期望: ${expected}`);
  } else {
    console.log(`✓ ${label} → ${actual}`);
  }
}

// 机械臂剧本（2026-09-23）：机械臂摇骰中无"停止"动作（等臂完成）；人工模式
// （manual_shaking，共用 shaking 视图）的 Esc 才是停止摇骰；arm_failed 的
// ↓/↑/Esc 分别对应机械臂重试 / 人工模式 / 退出本局。
enterState('shaking', 'shaking');
expect('机械臂摇骰中按 Esc（不应有动作）', press('Escape'), '[]');
expect('机械臂摇骰中按 Enter（不应有动作）', press('Enter'), '[]');
enterState('manual_shaking', 'shaking');
expect('人工摇骰中点击「停止摇骰」按钮', clickButton('stopShake'), '[["stop_shake",{}]]');
expect('人工摇骰中按 Esc', press('Escape'), clickButton('stopShake'));
enterState('arm_failed', 'arm_failed');
expect('机械臂失败页按 ↓（机械臂重试）', press('ArrowDown'), clickButton('armRetry'));
expect('机械臂失败页按 ↑（人工模式）', press('ArrowUp'), clickButton('armManual'));
expect('机械臂失败页按 Esc（退出本局）', press('Escape'), clickButton('armBack'));

// 绿键=Enter / 红键=Esc 通则（2026-09-23 晚拍板）：结果页与识别失败页的
// 绿色「再来一局」也必须吃 Enter；红键各态仍为返回/退出。
enterState('result', 'result');
expect('结果页按 Enter（再来一局）', press('Enter'), clickButton('newRound'));
expect('结果页按 Esc 仍是返回', press('Escape'), '[["back",{}]]');
enterState('analysis', 'analysis');
expect('识别失败页按 Enter（再来一局）', press('Enter'), clickButton('analysisNewRound'));
expect('识别失败页按 ↓（重新识别）', press('ArrowDown'), clickButton('analysisRetry'));
expect('识别失败页按 Esc（退出）', press('Escape'), clickButton('analysisBackToGames'));

// ---- 回合终结后的假死修复（2026-09-23 rps→dice 卡死 bug 第三层）----
// 回合 error/cancelled 后一切意图都被 409(ROUND_CLOSED) 静默拒绝；旧实现
// 静默吞掉导致错误页"按钮能按却毫无反应"。现在任何按键都必须导航回列表。
enterState('analysis', 'analysis');
roundStub.rejectWith = Object.assign(new Error('round is no longer running'), {
  code: 'ROUND_CLOSED', silent: true,
});
const navBefore = returnToSelectCalls;
press('Enter');
await new Promise((resolve) => setTimeout(resolve, 0));
if (returnToSelectCalls !== navBefore + 1) {
  failures.push('✗ 回合终结后按键应导航回列表\\n    实际 returnToSelect 调用: ' + (returnToSelectCalls - navBefore));
} else {
  console.log('✓ 回合终结后按键导航回列表 → returnToSelect 调用 +1');
}
// 正常对局中的时机不合（ROUND_INTENT_REJECTED）仍应静默、不导航。
roundStub.rejectWith = Object.assign(new Error('intent not accepted'), {
  code: 'ROUND_INTENT_REJECTED', silent: true,
});
const navBefore2 = returnToSelectCalls;
press('Enter');
await new Promise((resolve) => setTimeout(resolve, 0));
if (returnToSelectCalls !== navBefore2) {
  failures.push('✗ 时机不合的拒绝不应导航回列表');
}

// ---- rps：同一套「绿=Enter / 红=Esc」通则 ----
// 两游戏共用同一份 DOM（元素桩共享），先拆掉 dice 的监听器再注册 rps，
// 否则 clickButton 会同时触发两个游戏的 handler。
game.teardown();
const rpsPath = dicePath.replace(/dice\.js$/, 'rps.js');
const { register: registerRps } = await import(rpsPath);
onStateChange = null;
const rpsGame = registerRps(engine);
await rpsGame.enter({ id: 'rps', participants: { player: 'LEFT', agent: 'RIGHT' } });
const rpsState = (name, view) => onStateChange(name, { view }, {});
const rpsPress = (key) => { calls.length = 0; rpsGame.onKey({ key }); return JSON.stringify(calls); };
const rpsClick = (id) => {
  calls.length = 0;
  (elements.get(id).listeners.click || []).forEach((fn) => fn());
  return JSON.stringify(calls);
};
rpsState('rules', 'rules');
expect('rps 规则页按 Enter（确认）', rpsPress('Enter'), rpsClick('confirmRules'));
expect('rps 规则页按 Esc（返回）', rpsPress('Escape'), rpsClick('backFromRules'));
expect('rps 规则页按 ↓（重复规则）', rpsPress('ArrowDown'), rpsClick('repeatRules'));
rpsState('result', 'result');
expect('rps 结果页按 Enter（再来一局）', rpsPress('Enter'), rpsClick('newRound'));
expect('rps 结果页按 Esc（返回列表）', rpsPress('Escape'), rpsClick('backToGames'));
rpsState('analysis', 'analysis');
expect('rps 识别失败页按 Enter（再来一局）', rpsPress('Enter'), rpsClick('analysisNewRound'));
expect('rps 识别失败页按 ↓（重新识别）', rpsPress('ArrowDown'), rpsClick('analysisRetry'));
expect('rps 识别失败页按 Esc（退出）', rpsPress('Escape'), rpsClick('analysisBackToGames'));
rpsState('play', 'play');
expect('rps 出拳中按 Enter（不应有动作）', rpsPress('Enter'), '[]');
expect('rps 出拳中按 Esc（不应有动作）', rpsPress('Escape'), '[]');

if (failures.length) {
  console.error(`\n${failures.length} 项不符合预期:\n${failures.join('\n')}`);
  process.exit(1);
}
console.log('\n全部符合预期');
