// Run with: node --test tests/test_stream_player.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

function player(rotation = '') {
  const listeners = {}, windowEvents = {}, scripts = [], timers = [];
  const loading = { hidden: false };
  const video = {
    style: {}, readyState: 0, videoWidth: 0, srcObject: null,
    addEventListener: (name, fn) => { listeners[name] = fn; },
    play: () => Promise.resolve(),
    requestVideoFrameCallback: (fn) => { video.frame = fn; return 1; },
    cancelVideoFrameCallback: () => { video.frame = null; },
  };
  let callbacks, closed = false;
  const readers = [], delays = [], cancelled = [];
  const context = {
    URL, URLSearchParams, console,
    location: { href: 'http://arena/stream-player.html', search: '?stream=http%3A%2F%2Fcamera%3A8889%2Fdice%2Fdet&rotation=' + rotation },
    document: {
      getElementById: (id) => id === 'video' ? video : loading,
      addEventListener() {},
      createElement: () => ({ remove() {} }),
      head: { append: (script) => scripts.push(script) },
    },
    window: { addEventListener: (name, fn) => { windowEvents[name] = fn; } },
    setTimeout: (fn, delay) => { timers.push(fn); delays.push(delay); return timers.length; },
    clearTimeout: (id) => { cancelled.push(id); },
    MediaMTXWebRTCReader: class {
      constructor(options) { callbacks = options; this.closed = false; readers.push(this); }
      close() { closed = true; this.closed = true; }
    },
  };
  vm.runInNewContext(readFileSync(join(__dirname, '../web/stream-player.js'), 'utf8'), context);
  return { video, loading, listeners, scripts, timers, windowEvents, readers, delays, cancelled,
    callbacks: () => callbacks, closed: () => closed };
}

test('wait for a decoded video frame, not player load or track arrival', () => {
  const p = player();
  assert.equal(p.scripts[0].src, 'http://camera:8889/dice/det/reader.js');
  p.scripts[0].onload();
  assert.equal(p.callbacks().url, 'http://camera:8889/dice/det/whep');
  assert.equal(p.loading.hidden, false);
  p.callbacks().onTrack({ streams: [{}] });
  assert.equal(p.loading.hidden, false);
  p.listeners.playing();
  assert.equal(p.loading.hidden, false);
  p.video.readyState = 4;
  p.video.videoWidth = 1920;
  p.video.frame();
  assert.equal(p.loading.hidden, true);
  p.callbacks().onError('stream not found');
  assert.equal(p.loading.hidden, false);
  p.timers.at(-1)();
  p.callbacks().onTrack({ streams: [{}] });
  p.listeners.playing();
  p.video.frame();
  assert.equal(p.loading.hidden, true);
});

test('leaving the player closes WebRTC and ignores late callbacks', () => {
  const p = player();
  p.scripts[0].onload();
  p.windowEvents.pagehide();
  assert.equal(p.closed(), true);
  p.callbacks().onTrack({ streams: [{}] });
  assert.equal(p.video.srcObject, null);
});

test('reader script failures retry without removing the loading scene', () => {
  const p = player();
  p.scripts[0].onerror();
  assert.equal(p.loading.hidden, false);
  assert.equal(p.timers.length, 1);
  p.timers[0]();
  assert.equal(p.scripts.length, 2);
  p.windowEvents.pagehide();
  p.timers[0]();
  assert.equal(p.scripts.length, 2);
});

for (const angle of [0, 90, 180, 270]) {
  test(`video ${angle} degree rotation fits the box without rotating the loader`, () => {
    const p = player(String(angle));
    assert.equal(p.video.style.transform, `translate(-50%, -50%) rotate(${angle}deg)`);
    assert.equal(p.video.style.width, angle % 180 ? '100vh' : '100vw');
    assert.equal(p.video.style.height, angle % 180 ? '100vw' : '100vh');
    assert.equal(p.loading.hidden, false);
    p.scripts[0].onload();
    assert.equal(p.callbacks().url, 'http://camera:8889/dice/det/whep');
  });
}
test('invalid display rotation falls back to unrotated video', () => {
  assert.equal(player('junk').video.style.transform, 'translate(-50%, -50%) rotate(0deg)');
});


test('cold startup retries in 250ms and retires the old reader', () => {
  const p = player();
  p.scripts[0].onload();
  const old = p.callbacks();
  old.onError('stream not found');
  assert.equal(p.readers[0].closed, true);
  assert.equal(p.delays.at(-1), 250);
  p.timers.at(-1)();
  const active = {};
  p.callbacks().onTrack({ streams: [active] });
  old.onTrack({ streams: [{}] });
  old.onError('late error');
  assert.equal(p.video.srcObject, active);
  assert.equal(p.timers.length, 1);
  assert.equal(p.readers[1].closed, false);
});

test('persistent failures back off, a decoded frame restores fast reconnect', () => {
  const p = player();
  p.scripts[0].onload();
  for (let i = 0; i < 21; i++) {
    p.callbacks().onError('offline');
    assert.equal(p.delays.at(-1), i < 20 ? 250 : 2000);
    p.timers.at(-1)();
  }
  p.video.readyState = 4;
  p.video.videoWidth = 1280;
  p.listeners.playing();
  p.video.frame();
  p.callbacks().onError('temporary disconnect');
  assert.equal(p.delays.at(-1), 250);
});

test('pagehide cancels pending reconnect without reviving the player', () => {
  const p = player();
  p.scripts[0].onload();
  const old = p.callbacks();
  old.onError('not ready');
  p.windowEvents.pagehide();
  assert.ok(p.cancelled.includes(1));
  p.timers[0]();
  old.onError('late');
  assert.equal(p.readers.length, 1);
  assert.equal(p.timers.length, 1);
});
