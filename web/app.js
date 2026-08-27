// Dice Arena 前端引擎：跨游戏通用的视图切换、TTS、网络与游戏选择。
// 具体游戏（摇骰子 / 猜拳）通过 registerGame 挂载各自的 state machine 与文案。
import { register as registerDice } from './games/dice.js';
import { register as registerRps } from './games/rps.js';

const state = {
  phase: 'select',
  sound: true,
  selectedGame: 'dice',
  ttsAudio: null,
  ttsObjectUrl: null,
  ttsAbortController: null,
  ttsPlaybackCancel: null,
  ttsRequestId: 0,
  ttsFallbackNotified: false,
  ttsConfigErrorNotified: false,
};

const $ = (id) => document.getElementById(id);
const views = [...document.querySelectorAll('[data-view]')];

const SELECT_META = ['GAME SELECT', '选择一场游戏', '欢迎来到 Dice Arena，选择游戏后按 OK 开始。', '选择游戏开始体验'];

const gameModules = {};
let activeGame = null; // 当前挂载的游戏模块（有 enter/teardown/onKey/phaseMeta）
let games = []; // GET /api/games 返回的列表

// ---- 视图切换 ----
function setPhase(phase) {
  state.phase = phase;
  views.forEach((view) => view.classList.toggle('hidden', view.dataset.view !== phase));
  const meta = (activeGame && activeGame.phaseMeta && activeGame.phaseMeta[phase]) || SELECT_META;
  $('phaseKicker').textContent = meta[0];
  $('phaseTitle').textContent = meta[1];
  $('phaseCopy').textContent = meta[2];
  $('stageFooterText').textContent = meta[3];

  const phases = activeGame ? activeGame.phases : ['select'];
  const count = activeGame ? activeGame.progressCount : 1;
  const progressEl = $('progressDots');
  if (progressEl.children.length !== count) {
    progressEl.innerHTML = Array.from({ length: count }, () => '<span></span>').join('');
  }
  const index = phases.indexOf(phase);
  const activeUpTo = Math.max(0, Math.min(count - 1, index));
  Array.from(progressEl.children).forEach((dot, i) => {
    dot.classList.toggle('active', i <= activeUpTo);
  });
}

// ---- 提示 ----
function toast(message) {
  const node = $('toast');
  node.textContent = message;
  node.classList.add('show');
  setTimeout(() => node.classList.remove('show'), 2800);
}

// ---- TTS 文案（来自当前游戏的 manifest.texts） ----
function getTtsConfig() {
  const manifest = games.find((game) => game.id === state.selectedGame);
  if (!manifest || !manifest.texts || typeof manifest.texts !== 'object') {
    throw new Error('配置中缺少 texts 对象');
  }
  const speed = Number(manifest.speed ?? 1.0);
  if (!Number.isFinite(speed) || speed < 0.25 || speed > 4.0) {
    throw new Error('speed 必须在 0.25 到 4.0 之间');
  }
  return {
    voice: typeof manifest.voice === 'string' && manifest.voice.trim() ? manifest.voice.trim() : 'default',
    speed,
    texts: manifest.texts,
  };
}

function renderTtsText(template, values = {}) {
  return template.replace(/\{([a-zA-Z0-9_]+)\}/g, (match, name) => (
    Object.prototype.hasOwnProperty.call(values, name) ? String(values[name]) : match
  ));
}

function stopSpeech() {
  state.ttsRequestId += 1;
  state.ttsAbortController?.abort();
  state.ttsAbortController = null;
  state.ttsPlaybackCancel?.();
  state.ttsPlaybackCancel = null;
  if (state.ttsAudio) {
    state.ttsAudio.pause();
    state.ttsAudio.src = '';
    state.ttsAudio = null;
  }
  if (state.ttsObjectUrl) {
    URL.revokeObjectURL(state.ttsObjectUrl);
    state.ttsObjectUrl = null;
  }
  if ('speechSynthesis' in window) window.speechSynthesis.cancel();
}

// Deliberately do not call browser speech synthesis. Dice Arena must use the
// Qwen3-TTS model running on the K3 board; browser speech would hide a broken
// backend and make the voice sound unrelated to the configured speaker.

function createTtsFrameQueue() {
  const items = [];
  const waiters = [];
  let finished = false;
  let failure = null;

  return {
    push(item) {
      if (finished) return;
      const waiter = waiters.shift();
      if (waiter) waiter.resolve(item);
      else items.push(item);
    },
    finish(error = null) {
      if (finished) return;
      finished = true;
      failure = error;
      while (waiters.length) {
        const waiter = waiters.shift();
        if (failure) waiter.reject(failure);
        else waiter.resolve(null);
      }
    },
    next() {
      if (items.length) return Promise.resolve(items.shift());
      if (finished) return failure ? Promise.reject(failure) : Promise.resolve(null);
      return new Promise((resolve, reject) => waiters.push({ resolve, reject }));
    },
  };
}

async function* readTtsFrames(reader) {
  // /api/tts/stream uses a small length-prefixed protocol so a WAV can be
  // played as soon as it is complete, without waiting for the whole response.
  let buffer = new Uint8Array(0);
  let streamDone = false;

  const append = (chunk) => {
    const merged = new Uint8Array(buffer.byteLength + chunk.byteLength);
    merged.set(buffer);
    merged.set(chunk, buffer.byteLength);
    buffer = merged;
  };
  const readMore = async () => {
    if (streamDone) return false;
    const { value, done } = await reader.read();
    if (done) {
      streamDone = true;
      return false;
    }
    if (value?.byteLength) append(value);
    return true;
  };
  const ensure = async (size) => {
    while (buffer.byteLength < size && await readMore()) {}
    return buffer.byteLength >= size;
  };
  const take = (size) => {
    const value = buffer.slice(0, size);
    buffer = buffer.slice(size);
    return value;
  };
  const uint32 = (bytes) => new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(0, false);

  while (await ensure(4)) {
    const length = uint32(buffer.slice(0, 4));
    if (length === 0) {
      return;
    }
    if (length === 0xffffffff) {
      if (!await ensure(8)) throw new Error('TTS 流在错误帧中提前结束');
      take(4);
      const messageLength = uint32(buffer.slice(0, 4));
      if (messageLength > 64 * 1024 || !await ensure(4 + messageLength)) {
        throw new Error('TTS 错误帧无效');
      }
      take(4);
      const message = new TextDecoder().decode(take(messageLength));
      throw new Error(message || 'TTS 流生成失败');
    }
    if (length < 44 || length > 32 * 1024 * 1024) {
      throw new Error('TTS 音频帧长度无效');
    }
    if (!await ensure(4 + length)) throw new Error('TTS 音频流提前结束');
    take(4);
    yield new Blob([take(length)], { type: 'audio/wav' });
  }
  throw new Error('TTS 流没有结束帧');
}

async function requestSpeechStream(message, requestId, options, queue) {
  const controller = new AbortController();
  state.ttsAbortController = controller;
  let reader = null;
  try {
    const response = await fetch('/api/tts/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: message, voice: options.voice, speed: options.speed }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error || `HTTP ${response.status}`);
    }
    if (!response.body) throw new Error('浏览器不支持 TTS 流式响应');
    reader = response.body.getReader();
    for await (const blob of readTtsFrames(reader)) {
      if (!state.sound || requestId !== state.ttsRequestId) {
        throw new DOMException('Speech request was cancelled', 'AbortError');
      }
      queue.push(blob);
    }
    queue.finish();
  } catch (error) {
    if (error.name === 'AbortError' || controller.signal.aborted || requestId !== state.ttsRequestId) {
      queue.finish(new DOMException('Speech request was cancelled', 'AbortError'));
    } else {
      queue.finish(error);
    }
  } finally {
    try { await reader?.cancel(); } catch (_) { /* response already ended */ }
    reader?.releaseLock();
    if (state.ttsAbortController === controller) state.ttsAbortController = null;
  }
}

async function playSpeechBlob(blob, requestId) {
  if (!state.sound || requestId !== state.ttsRequestId) {
    throw new DOMException('Speech playback was cancelled', 'AbortError');
  }

  const objectUrl = URL.createObjectURL(blob);
  const audio = new Audio(objectUrl);
  let released = false;
  let cancelled = false;
  let playbackError = null;
  let finishPlayback;
  const playbackDone = new Promise((resolve) => { finishPlayback = resolve; });
  const release = () => {
    if (released) return;
    released = true;
    if (state.ttsAudio === audio) state.ttsAudio = null;
    if (state.ttsObjectUrl === objectUrl) state.ttsObjectUrl = null;
    URL.revokeObjectURL(objectUrl);
    finishPlayback();
  };

  state.ttsAudio = audio;
  state.ttsObjectUrl = objectUrl;
  const cancelPlayback = () => {
    cancelled = true;
    audio.pause();
    release();
  };
  state.ttsPlaybackCancel = cancelPlayback;
  audio.addEventListener('ended', release, { once: true });
  audio.addEventListener('error', () => {
    playbackError = new Error('浏览器无法播放 K3 Qwen3-TTS 返回的 WAV');
    release();
  }, { once: true });

  try {
    await audio.play();
    await playbackDone;
    if (playbackError) throw playbackError;
  } catch (error) {
    release();
    throw error;
  } finally {
    if (state.ttsPlaybackCancel === cancelPlayback) state.ttsPlaybackCancel = null;
    release();
  }
  if (cancelled || !state.sound || requestId !== state.ttsRequestId) {
    throw new DOMException('Speech playback was cancelled', 'AbortError');
  }
}

async function speak(message, options = { voice: 'default', speed: 1.0 }) {
  if (!state.sound || !message) return;
  stopSpeech();
  const requestId = state.ttsRequestId;
  const queue = createTtsFrameQueue();
  const producer = requestSpeechStream(message, requestId, options, queue);
  let playedFrames = 0;
  try {
    // One HTTP request is made for the complete announcement. The producer
    // keeps reading later WAV frames while the consumer plays the first one.
    while (true) {
      const blob = await queue.next();
      if (blob === null) break;
      await playSpeechBlob(blob, requestId);
      playedFrames += 1;
    }
    await producer;
  } catch (error) {
    await producer.catch(() => {});
    if (error.name === 'AbortError' || requestId !== state.ttsRequestId || !state.sound) return;
    console.error(`K3 Qwen3-TTS stream failed after ${playedFrames} frame(s):`, error);
    toast('K3 Qwen3-TTS 播放失败，未使用浏览器替代语音');
  }
}

function speakState(key, values = {}) {
  if (!state.sound) return;
  try {
    const config = getTtsConfig();
    const template = config.texts[key];
    if (typeof template !== 'string' || !template.trim()) {
      throw new Error(`未配置 TTS 状态文案：${key}`);
    }
    speak(renderTtsText(template, values), config);
  } catch (error) {
    console.error(`Failed to load TTS state ${key}:`, error);
    if (!state.ttsConfigErrorNotified) {
      state.ttsConfigErrorNotified = true;
      toast('TTS 文案配置加载失败，请检查后端 /api/games');
    }
  }
}

// ---- 网络 ----
async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

// ---- 游戏注册表 ----
function registerGame(module) {
  gameModules[module.id] = module;
}

function returnToSelect() {
  if (activeGame && activeGame.teardown) activeGame.teardown();
  activeGame = null;
  setPhase('select');
}

// ---- 游戏列表 ----
function renderGameList() {
  const list = $('gameList');
  list.innerHTML = '';
  games.forEach((game) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.game = game.id;
    button.className = 'game-option' + (game.enabled ? '' : ' disabled');
    button.setAttribute('role', 'option');
    button.setAttribute('aria-selected', 'false');
    button.disabled = !game.enabled;

    const icon = document.createElement('span');
    icon.className = 'game-icon';
    icon.setAttribute('aria-hidden', 'true');
    icon.textContent = game.icon || '🎮';

    const copy = document.createElement('span');
    copy.className = 'game-copy';
    const name = document.createElement('strong');
    name.textContent = game.name;
    const desc = document.createElement('small');
    desc.textContent = game.description || '';
    copy.append(name, desc);

    button.append(icon, copy);

    if (game.enabled) {
      const arrow = document.createElement('span');
      arrow.className = 'game-arrow';
      arrow.setAttribute('aria-hidden', 'true');
      arrow.textContent = '→';
      button.append(arrow);
      button.addEventListener('click', () => selectGame(game.id));
    } else {
      const lock = document.createElement('span');
      lock.className = 'game-lock';
      lock.textContent = '即将开放';
      button.append(lock);
    }

    list.append(button);
  });
}

function selectGame(id) {
  state.selectedGame = id;
  document.querySelectorAll('.game-option').forEach((item) => {
    const selected = item.dataset.game === id;
    item.classList.toggle('selected', selected);
    item.setAttribute('aria-selected', selected ? 'true' : 'false');
  });
}

function enterSelectedGame() {
  const module = gameModules[state.selectedGame];
  if (!module) {
    toast(`未注册游戏：${state.selectedGame}`);
    return;
  }
  activeGame = module;
  module.enter();
}

// ---- 启动 ----
const engine = {
  state,
  $,
  setPhase,
  toast,
  speak,
  speakState,
  stopSpeech,
  requestJson,
  renderTtsText,
  returnToSelect,
};

registerDice(engine);
registerRps(engine);

$('startGame').addEventListener('click', enterSelectedGame);
$('gameList').addEventListener('dblclick', enterSelectedGame);
$('soundToggle').addEventListener('click', () => {
  state.sound = !state.sound;
  if (!state.sound) stopSpeech();
  $('soundToggle').textContent = state.sound ? '🔊' : '🔇';
  toast(state.sound ? 'K3 Qwen3-TTS 播报已开启' : '语音播报已关闭');
});

document.addEventListener('keydown', (event) => {
  if (state.phase === 'select') {
    if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
      const enabled = games.filter((game) => game.enabled);
      if (!enabled.length) return;
      const ids = enabled.map((game) => game.id);
      const index = ids.indexOf(state.selectedGame);
      const next = ids[(index + (event.key === 'ArrowDown' ? 1 : -1) + ids.length) % ids.length];
      selectGame(next);
    }
    if (event.key === 'Enter') enterSelectedGame();
    return;
  }
  if (activeGame && activeGame.onKey) activeGame.onKey(event);
});

async function loadGames() {
  try {
    const payload = await requestJson('/api/games');
    games = Array.isArray(payload.games) ? payload.games : [];
    renderGameList();
    const firstEnabled = games.find((game) => game.enabled);
    if (firstEnabled) selectGame(firstEnabled.id);
  } catch (error) {
    console.error('Failed to load game list:', error);
    toast('游戏列表加载失败，请检查后端 /api/games');
  }
}

setPhase('select');
loadGames();
