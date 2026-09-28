// Run with: node --test tests/test_stream_player.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

function player() {
  const listeners = {}, windowEvents = {}, scripts = [], timers = [];
  const loading = { hidden: false };
  const video = {
    readyState: 0, videoWidth: 0, srcObject: null,
    addEventListener: (name, fn) => { listeners[name] = fn; },
    play: () => Promise.resolve(),
    requestVideoFrameCallback: (fn) => { video.frame = fn; return 1; },
    cancelVideoFrameCallback: () => { video.frame = null; },
  };
  let callbacks, closed = false;
  const context = {
    URL, URLSearchParams, console,
    location: { href: 'http://arena/stream-player.html', search: '?stream=http%3A%2F%2Fcamera%3A8889%2Fdice%2Fdet' },
    document: {
      getElementById: (id) => id === 'video' ? video : loading,
      addEventListener() {},
      createElement: () => ({ remove() {} }),
      head: { append: (script) => scripts.push(script) },
    },
    window: { addEventListener: (name, fn) => { windowEvents[name] = fn; } },
    setTimeout: (fn) => { timers.push(fn); return timers.length; }, clearTimeout() {},
    MediaMTXWebRTCReader: class {
      constructor(options) { callbacks = options; }
      close() { closed = true; }
    },
  };
  vm.runInNewContext(readFileSync(join(__dirname, '../web/stream-player.js'), 'utf8'), context);
  return { video, loading, listeners, scripts, timers, windowEvents,
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
