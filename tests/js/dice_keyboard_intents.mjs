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
    calls.push([intent, payload]);
    return Promise.resolve();
  },
  cancel() {},
};

const engine = {
  state,
  $,
  setPhase(phase) { state.phase = phase; },
  toast() {},
  stopSpeech() {},
  async requestJson() { return {}; },
  returnToSelect() {},
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

// 回归护栏：其余既有映射不能被这次改动带偏。
enterState('rules', 'rules');
expect('规则页按 Esc 仍是返回', press('Escape'), '[["back",{}]]');
expect('规则页按 ↓ 仍是重复规则', press('ArrowDown'), clickButton('repeatRules'));
enterState('ready', 'ready');
expect('准备页按 Enter 仍是开始摇骰', press('Enter'), '[["start_shake",{}]]');
enterState('result', 'result');
expect('结果页按 Esc 仍是返回', press('Escape'), '[["back",{}]]');

if (failures.length) {
  console.error(`\n${failures.length} 项不符合预期:\n${failures.join('\n')}`);
  process.exit(1);
}
console.log('\n全部符合预期');
