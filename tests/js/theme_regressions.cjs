const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[2], 'utf8');

function boot(saved, blocked = false) {
  const values = new Map(saved === undefined ? [] : [['spacemit.arena.theme.v1', saved]]);
  const nodes = new Map();
  const listeners = new Map();
  const calls = [];
  function node(text = '') {
    const attributes = {};
    return {
      textContent: text, value: '', dataset: {}, attributes,
      classList: { toggle: (key, value) => { attributes[key] = value; } },
      setAttribute: (key, value) => { attributes[key] = value; },
      removeAttribute: (key) => { delete attributes[key]; },
      addEventListener: (type, handler) => listeners.set(type, handler),
      querySelectorAll: () => [],
    };
  }
  for (const id of ['startGame', 'confirmRules', 'repeatRules', 'startShake', 'gameStartText', 'themeSelect', 'arenaBreadcrumb', 'arenaSelected']) nodes.set(id, node('旧文案'));
  const ready = node('旧准备标题');
  const diceDescription = node('原骰子介绍');
  const rpsDescription = node('原猜拳介绍');
  const steps = [0, 1, 2, 3].map((i) => { const element = node(); element.dataset.step = String(i); return element; });
  const stepContainer = node(); stepContainer.querySelectorAll = () => steps;
  nodes.set('arenaSteps', stepContainer);
  const root = { dataset: {} };
  const meta = { content: '' };
  const document = {
    documentElement: root,
    getElementById: (id) => nodes.get(id),
    querySelector: (selector) => selector.includes('theme-color') ? meta
      : selector.includes('ready') ? ready
      : selector.includes('.game-copy small') ? (selector.includes('"dice"') ? diceDescription : rpsDescription) : null,
    addEventListener: (type, handler) => listeners.set(type, handler),
  };
  const context = {
    document, window: { dispatchEvent: (event) => calls.push(event.type), addEventListener: (type, handler) => listeners.set(type, handler) },
    Event: class { constructor(type) { this.type = type; } },
    localStorage: {
      getItem: (key) => { if (blocked) throw Error('storage blocked'); return values.get(key); },
      setItem: (key, value) => { if (blocked) throw Error('storage blocked'); values.set(key, value); },
    },
  };
  vm.runInNewContext(source, context);
  listeners.get('DOMContentLoaded')();
  return { api: context.window.ArenaTheme, root, meta, nodes, steps, ready, diceDescription, rpsDescription, values, calls, listeners };
}

for (const value of [undefined, 'invalid', 'light', '']) {
  const app = boot(value);
  assert.equal(app.root.dataset.theme, 'dark-arena');
  assert.equal(app.meta.content, '#2a3b34');
  assert.equal(app.nodes.get('themeSelect').value, 'dark-arena');
}
const app = boot();
const copy = ['后端标题', '后端动作与提示必须保留'];
assert.equal(app.api.resolveMeta('select', copy, 'dice')[0], '骰王挑战赛');
assert.equal(app.api.resolveMeta('select', copy, 'dice')[1], copy[1]);
assert.equal(app.diceDescription.textContent, '与机械臂同台摇骰，让 AI 识别点数、揭晓胜负。');
assert.equal(app.api.resolveMeta('rules', copy, 'dice')[0], '三步，开始挑战。');
assert.equal(app.api.resolveMeta('rules', copy, 'rps')[0], '听口令，一起出拳。');
assert.equal(app.api.resolveMeta('rules', copy, 'dice')[1], copy[1]);
for (const phase of ['analysis', 'result', 'shaking', 'play']) assert.equal(app.api.resolveMeta(phase, copy, 'dice'), copy);
app.api.renderPhase('analysis', 'dice', 'dice');
assert.equal(app.steps[2].attributes['aria-current'], 'step');
assert.equal(app.steps[1].attributes['aria-current'], undefined);
app.api.set('storybook');
assert.equal(app.api.resolveMeta('select', copy, 'dice'), copy);
assert.equal(app.diceDescription.textContent, '原骰子介绍');
assert.equal(app.rpsDescription.textContent, '原猜拳介绍');
assert.equal(app.api.resolveMeta('rules', copy, 'dice'), copy);
assert.equal(app.nodes.get('startGame').textContent, '旧文案');
assert.equal(app.ready.textContent, '旧准备标题');
assert.equal(app.meta.content, '#f7f9fc');
assert.equal(boot(app.values.get('spacemit.arena.theme.v1')).api.current, 'storybook');
app.api.set('dark-arena');
assert.equal(app.nodes.get('startGame').textContent, '开始挑战');
assert.equal(app.ready.textContent, '握好骰盅，准备开始。');
app.api.set('unregistered-theme');
assert.equal(app.api.current, 'dark-arena');
const blocked = boot(undefined, true);
blocked.api.set('storybook');
assert.equal(blocked.api.current, 'storybook');
let stopped = false;
app.listeners.get('keydown')({ stopPropagation() { stopped = true; } });
assert.equal(stopped, true);
assert.equal(app.calls.length, 2);
app.listeners.get('storage')({ key: 'spacemit.arena.theme.v1', newValue: 'storybook' });
assert.equal(app.api.current, 'storybook');
assert.equal(app.meta.content, '#f7f9fc');
app.listeners.get('storage')({ key: 'other-setting', newValue: 'dark-arena' });
assert.equal(app.api.current, 'storybook');
console.log('Theme defaults, persistence, blocked storage, phase copy, legacy restoration and key isolation passed.');
