const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[2], 'utf8');

function boot({ reduced = false, theme = 'dark-arena', missingPanel = false } = {}) {
  function classes(initial = []) {
    const values = new Set(initial);
    return {
      contains: (value) => values.has(value),
      toggle: (value, on) => on ? values.add(value) : values.delete(value),
    };
  }
  const body = { dataset: { phase: 'rules' }, classList: classes() };
  const root = { dataset: { theme } };
  const motions = [];
  let reads = 0;
  let sourceWrites = 0;
  const frame = {};
  Object.defineProperty(frame, 'src', {
    get: () => '/stream-player.html?stream=/dice/det',
    set: () => { sourceWrites++; throw Error('A layout change must not reconnect video'); },
  });
  const panel = {
    classList: classes(['dice-video']), style: { willChange: '' },
    getBoundingClientRect() {
      reads++;
      return body.classList.contains('arena-video-focus')
        ? { left: 76, top: 180, width: 1220, height: 730 }
        : { left: 76, top: 270, width: 994, height: 600 };
    },
    animate(frames, options) {
      const motion = { frames, options, cancelled: false, cancel() { this.cancelled = true; } };
      motions.push(motion);
      return motion;
    },
  };
  const failure = { classList: classes(['hidden']) };
  const judge = { classList: classes() };
  const nodes = { analysisStreamPanel: missingPanel ? null : panel,
    analysisFailureActions: failure, stepJudge: judge, analysisStream: frame };
  const listeners = new Map();
  const mediaListeners = new Map();
  const media = { matches: reduced, addEventListener: (type, callback) => mediaListeners.set(type, callback) };
  let callback;
  const observed = [];
  vm.runInNewContext(source, {
    document: { body, documentElement: root, readyState: 'complete', getElementById: (id) => nodes[id] },
    window: {
      matchMedia: () => media, addEventListener: (type, handler) => listeners.set(type, handler),
      fetch() { throw Error('Presentation must never submit a game action'); },
    },
    MutationObserver: class {
      constructor(handler) { callback = handler; }
      observe(node, options) { observed.push({ node, options }); }
    },
  });
  return { body, root, panel, frame, failure, judge, motions, media, observed,
    flush: () => callback?.(), listeners, mediaListeners,
    get reads() { return reads; }, get sourceWrites() { return sourceWrites; },
  };
}

const app = boot();
const focus = () => app.body.classList.contains('arena-video-focus');
const phase = (value) => { app.body.dataset.phase = value; app.flush(); };
assert.equal(focus(), false);
assert.equal(app.motions.length, 0);
for (const value of ['game_start', 'countdown', 'shaking', 'play', 'rps-play', 'analysis']) {
  phase(value);
  assert.equal(focus(), true, value);
}
assert.equal(app.motions.length, 1, 'Active phase changes must not replay the same zoom');
assert.equal(app.reads, 2, 'Measure twice for one size change, never per animation frame');
assert.equal(app.motions[0].options.duration, 180);
assert.equal(app.motions[0].options.easing, 'ease-out');
assert.equal(app.panel.style.willChange, 'transform');
assert.ok(app.motions[0].frames[0].transform.includes('scale('));
assert.equal(app.motions[0].frames[1].transform, 'none');

// The result locks while the phase is still analysis. Shrink at that point.
app.judge.classList.toggle('active', true); app.flush();
assert.equal(focus(), false);
assert.equal(app.motions[0].cancelled, true, 'Rapid changes cancel the old transition');
app.motions[1].onfinish();
assert.equal(app.panel.style.willChange, '');
phase('result');
assert.equal(app.motions.length, 2);
for (const value of ['rules', 'ready', 'rehome', 'arm_failed', 'select', 'standby']) {
  phase(value); assert.equal(focus(), false, value);
}

// A retry can detect again, and a diagnosis returns to the normal layout.
app.judge.classList.toggle('active', false); phase('analysis');
assert.equal(focus(), true);
app.failure.classList.toggle('hidden', false); app.flush();
assert.equal(focus(), false);
app.failure.classList.toggle('hidden', true); app.flush();
assert.equal(focus(), true);
app.root.dataset.theme = 'storybook'; app.flush();
assert.equal(focus(), false, 'Legacy theme keeps its established layout');
app.root.dataset.theme = 'dark-arena'; app.flush();
assert.equal(focus(), true);
app.panel.classList.toggle('hidden', true); app.flush();
assert.equal(focus(), false, 'A hidden stream never stays enlarged');
assert.equal(app.sourceWrites, 0);
assert.equal(app.frame.src, '/stream-player.html?stream=/dice/det');
assert.ok(app.observed.some(({ node, options }) => node === app.body && options.attributeFilter[0] === 'data-phase'));

const calm = boot({ reduced: true });
calm.body.dataset.phase = 'shaking'; calm.flush();
assert.equal(calm.body.classList.contains('arena-video-focus'), true);
assert.equal(calm.motions.length, 0);
assert.equal(calm.reads, 0, 'Reduced motion needs no animation measurements');
const leaving = boot();
leaving.body.dataset.phase = 'play'; leaving.flush();
leaving.mediaListeners.get('change')({ matches: true });
assert.equal(leaving.motions[0].cancelled, true);
assert.equal(leaving.panel.style.willChange, '');
leaving.body.dataset.phase = 'result'; leaving.flush();
leaving.listeners.get('pagehide')();
assert.equal(leaving.motions[1].cancelled, true);
assert.equal(leaving.panel.style.willChange, '');
boot({ missingPanel: true });
console.log('Video focus phases, result locking, failure/retry, legacy, interruption and reduced motion passed.');
