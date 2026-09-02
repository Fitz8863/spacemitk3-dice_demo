// Dice Arena 前端引擎：跨游戏通用的视图切换、语音指令播放、网络与游戏选择。
// 游戏流程由后端权威状态机驱动：前端通过 RoundClient 创建对局、提交意图、
// 订阅事件流并渲染；台词以后端下发的 speech 指令播放，await 指令播完回执。
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
  speechAudioContext: null,
};

const $ = (id) => document.getElementById(id);
const views = [...document.querySelectorAll('[data-view]')];

const SELECT_META = ['选择一场游戏', '欢迎来到 Dice Arena，选择游戏后按 OK 开始。'];

const gameModules = {};
let activeGame = null; // 当前挂载的游戏模块（有 enter/teardown/onKey）
let games = []; // GET /api/games 返回的列表

const CONTROLLER_COPY = {
  select: `使用 <span class="controller-hint"><span class="controller-key controller-key-yellow" aria-label="黄色按钮"></span><span>黄色 · 向上选择</span></span> 和 <span class="controller-hint"><span class="controller-key controller-key-blue" aria-label="蓝色按钮"></span><span>蓝色 · 向下选择</span></span> 选择游戏，选中后按 <span class="controller-hint"><span class="controller-key controller-key-green" aria-label="绿色按钮"></span><span>绿色 · 确认</span></span>。`,
  rules: `听完规则后按 <span class="controller-hint"><span class="controller-key controller-key-green" aria-label="绿色按钮"></span><span>绿色 · 确认</span></span>，按 <span class="controller-hint"><span class="controller-key controller-key-blue" aria-label="蓝色按钮"></span><span>蓝色 · 重听</span></span>，按 <span class="controller-hint"><span class="controller-key controller-key-red" aria-label="红色按钮"></span><span>红色 · 返回</span></span>。`,
  ready: `按 <span class="controller-hint"><span class="controller-key controller-key-green" aria-label="绿色按钮"></span><span>绿色 · 开始摇骰</span></span>，按 <span class="controller-hint"><span class="controller-key controller-key-red" aria-label="红色按钮"></span><span>红色 · 返回</span></span>。`,
};

function renderPhaseCopy(phase, fallback) {
  const node = $('phaseCopy');
  const copy = CONTROLLER_COPY[phase];
  if (copy) {
    node.innerHTML = copy;
    node.classList.remove('hidden');
  } else if (fallback) {
    node.textContent = fallback;
    node.classList.remove('hidden');
  } else {
    node.textContent = '';
    node.classList.add('hidden');
  }
}

// ---- 视图切换 ----
function setPhase(phase, meta) {
  state.phase = phase;
  // Phases style themselves via body[data-phase] (e.g. the open phase lowers
  // the stage header into mid-screen).
  document.body.dataset.phase = phase;
  views.forEach((view) => view.classList.toggle('hidden', view.dataset.view !== phase));
  const resolved = meta || (activeGame && activeGame.phaseMeta && activeGame.phaseMeta[phase]) || SELECT_META;
  $('phaseTitle').textContent = resolved[0];
  renderPhaseCopy(phase, resolved[1]);
}

// ---- 提示 ----
function toast(message) {
  const node = $('toast');
  node.textContent = message;
  node.classList.add('show');
  setTimeout(() => node.classList.remove('show'), 2800);
}

// ---- 语音播放（后端 speech 指令驱动） ----
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
  if ('speechSynthesis' in window) window.speechSynthesis.cancel(); // clean up any external/browser utterance
}

// Deliberately do not call browser speech synthesis. Dice Arena must use the
// TTS provider selected by the backend; browser speech would hide a broken
// provider and make the voice unrelated to the configured model/speaker.

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
  // Speech frame endpoints use a small length-prefixed protocol so a WAV can
  // be played as soon as it is complete, without waiting for the whole response.
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
    playbackError = new Error('浏览器无法播放当前 TTS provider 返回的 WAV');
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

// 无缝播报播放器：流式 TTS 的每个 WAV 帧解码成 AudioBuffer 后，在同一个
// WebAudio 时间线上背靠背排片，消除逐帧新建 Audio 元素带来的切换间隙。
// 生成速度远快于实时播放，排片进度本身构成抗抖动缓冲。
function getSpeechAudioContext() {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return null;
  if (!state.speechAudioContext) {
    try {
      state.speechAudioContext = new Ctx();
    } catch (_) {
      return null;
    }
  }
  if (state.speechAudioContext.state === 'suspended') {
    state.speechAudioContext.resume().catch(() => {});
  }
  return state.speechAudioContext;
}

function createSpeechScheduler(requestId) {
  const context = getSpeechAudioContext();
  if (!context) return null;
  const player = {
    context,
    nextAt: 0,
    sources: [],
    cancelled: false,
    resolveDrained: null,
  };
  const cancelSource = (source) => {
    try { source.stop(); } catch (_) { /* already ended */ }
    source.disconnect();
  };
  player.schedule = async (blob) => {
    if (player.cancelled || requestId !== state.ttsRequestId || !state.sound) {
      throw new DOMException('Speech playback was cancelled', 'AbortError');
    }
    const buffer = await context.decodeAudioData(await blob.arrayBuffer());
    if (player.cancelled || requestId !== state.ttsRequestId || !state.sound) {
      throw new DOMException('Speech playback was cancelled', 'AbortError');
    }
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    // 第一帧留一点起播水位，后续帧紧贴前一帧结尾，形成连续时间线。
    const startAt = Math.max(player.nextAt, context.currentTime + 0.06);
    source.start(startAt);
    player.nextAt = startAt + buffer.duration;
    player.sources.push(source);
    source.onended = () => {
      const index = player.sources.indexOf(source);
      if (index >= 0) player.sources.splice(index, 1);
      if (!player.sources.length && player.resolveDrained) {
        player.resolveDrained();
        player.resolveDrained = null;
      }
    };
  };
  player.cancel = () => {
    player.cancelled = true;
    for (const source of player.sources.splice(0)) cancelSource(source);
    player.nextAt = 0;
    if (player.resolveDrained) {
      player.resolveDrained();
      player.resolveDrained = null;
    }
  };
  player.waitDrained = () => {
    if (player.cancelled || !player.sources.length) return Promise.resolve();
    return new Promise((resolve) => { player.resolveDrained = resolve; });
  };
  return player;
}

// 播放一条后端 speech 指令：拉帧 → 排片播放 → await 指令播完回执。
// 播放失败也会回执，让状态机立即继续而不等 30 秒兜底。
async function playDirective(round, directive) {
  const acknowledge = () => {
    if (directive.await && round && round.roundId) {
      round.submitIntent('speech_done', { directive_id: directive.directive_id })
        .catch(() => { /* round may already be closed; the engine fallback covers it */ });
    }
  };
  if (!state.sound) {
    acknowledge();
    return;
  }
  stopSpeech();
  const requestId = state.ttsRequestId;
  const queue = createTtsFrameQueue();
  const controller = new AbortController();
  state.ttsAbortController = controller;
  let reader = null;
  const producer = (async () => {
    try {
      const response = await fetch(`/api/game/rounds/${round.roundId}/speech`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ directive_id: directive.directive_id }),
        signal: controller.signal,
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || `语音请求失败：HTTP ${response.status}`);
      }
      if (!response.body) throw new Error('浏览器不支持语音流式响应');
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
  })();
  const scheduler = createSpeechScheduler(requestId);
  if (scheduler) state.ttsPlaybackCancel = scheduler.cancel;
  let playedFrames = 0;
  try {
    // One HTTP request per directive. The producer keeps reading later WAV
    // frames while the consumer plays the first one; the scheduler lines
    // frames up back-to-back so streaming TTS never gaps between frames.
    while (true) {
      const blob = await queue.next();
      if (blob === null) break;
      if (scheduler) await scheduler.schedule(blob);
      else await playSpeechBlob(blob, requestId);
      playedFrames += 1;
    }
    if (scheduler) await scheduler.waitDrained();
    await producer;
    acknowledge();
  } catch (error) {
    await producer.catch(() => {});
    if (error.name === 'AbortError' || requestId !== state.ttsRequestId || !state.sound) return;
    console.error(`Speech directive ${directive.directive_id} failed after ${playedFrames} frame(s):`, error);
    toast('语音播放失败，请检查语音组件');
    acknowledge();
  } finally {
    if (scheduler && state.ttsPlaybackCancel === scheduler.cancel) {
      state.ttsPlaybackCancel = null;
    }
  }
}

// ---- 权威对局客户端 ----
// 创建 round、订阅 SSE 事件流、提交意图。事件按 sequence 去重，SSE 断线由
// EventSource 自动重连，重连快照会重放全部事件，靠 sequence 过滤保持幂等。
function createRoundClient(gameId, handlers) {
  let roundId = null;
  let source = null;
  let lastSequence = 0;
  // Set once the round reaches a terminal snapshot or the client cancels:
  // afterwards every event is stale game state and must not reach the UI,
  // otherwise a finished view resurrects itself after returnToSelect().
  let closed = false;

  function dispatch(event, snapshot) {
    if (event.event === 'state_changed') {
      handlers.onStateChange?.(event.state, event.ui || {}, snapshot || null);
    } else if (event.event === 'speech') {
      handlers.onSpeech?.(event);
    } else if (event.event === 'tick') {
      handlers.onTick?.(event);
    } else if (event.event === 'round_complete') {
      handlers.onComplete?.(event, snapshot || null);
    } else {
      handlers.onEvent?.(event, snapshot || null);
    }
  }

  function ingest(snapshot) {
    if (closed) return;
    for (const item of (snapshot.events || [])) {
      const sequence = Number(item.sequence || 0);
      if (sequence > lastSequence) {
        lastSequence = sequence;
        dispatch(item, snapshot);
      }
    }
    // A terminal snapshot ends the client: round_complete has just been
    // dispatched (onComplete navigates or shows the failure), and the
    // snapshot's top-level state is stale game state — syncing it would
    // resurrect the finished view on top of the game list.
    if (snapshot.status && snapshot.status !== 'running') {
      closed = true;
      teardownStream();
      return;
    }
    // Top-level state keeps the view aligned even if a state_changed event
    // slid between two SSE deliveries (or after a reconnect snapshot).
    if (typeof snapshot.state === 'string' && snapshot.state) {
      handlers.onSyncState?.(snapshot);
    }
  }

  function subscribe() {
    if (source) source.close();
    source = new EventSource(`/api/game/rounds/${roundId}/stream`);
    const handle = (event) => {
      try {
        ingest(JSON.parse(event.data));
      } catch (_) { /* malformed frame: the next snapshot re-aligns state */ }
    };
    source.addEventListener('snapshot', handle);
    source.addEventListener('update', handle);
    source.addEventListener('complete', (event) => {
      handle(event);
      source?.close();
    });
    source.onerror = () => { /* EventSource retries; server resends a snapshot */ };
  }

  async function start() {
    const payload = await requestJson('/api/game/rounds', {
      method: 'POST',
      body: JSON.stringify({ game: gameId }),
    });
    roundId = payload.round_id;
    lastSequence = 0;
    subscribe();
    return payload;
  }

  async function submitIntent(intent, payload = {}) {
    if (!roundId) throw new Error('对局尚未创建');
    const snapshot = await requestJson(`/api/game/rounds/${roundId}/intents`, {
      method: 'POST',
      body: JSON.stringify({ intent, ...payload }),
    });
    // Ingest the response so terminal transitions (e.g. an exit intent)
    // navigate even when the SSE stream is broken; sequence dedup keeps
    // this idempotent with the stream.
    if (snapshot && typeof snapshot === 'object' && 'status' in snapshot) {
      ingest(snapshot);
    }
    return snapshot;
  }

  async function cancel() {
    if (!roundId) return;
    const target = roundId;
    roundId = null;
    closed = true;
    teardownStream();
    try {
      await requestJson(`/api/game/rounds/${target}/cancel`, { method: 'POST', body: '{}' });
    } catch (_) { /* already finished */ }
  }

  function teardownStream() {
    if (source) {
      source.close();
      source = null;
    }
  }

  return {
    start,
    submitIntent,
    cancel,
    teardownStream,
    get roundId() { return roundId; },
  };
}

// ---- 网络 ----
async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const errorCode = payload.code || 'UNKNOWN_ERROR';
    const errorMessage = payload.error || `HTTP ${response.status}`;

    // An intent pressed at the wrong moment is normal gameplay, not an error.
    if (errorCode === 'ROUND_INTENT_REJECTED' || errorCode === 'ROUND_CLOSED') {
      const error = new Error(errorMessage);
      error.code = errorCode;
      error.silent = true;
      throw error;
    }
    if (errorCode === 'GAME_DISABLED') {
      toast('该游戏即将开放，敬请期待');
    } else if (errorCode === 'JOB_ALREADY_EXISTS') {
      toast('已有分析任务正在运行，请稍后再试');
    } else if (errorCode === 'COMPONENT_NOT_READY') {
      toast('系统组件未就绪，请检查后端服务');
    } else if (errorCode === 'TTS_SERVICE_ERROR') {
      toast('TTS 服务异常，语音播报暂时不可用');
    } else {
      toast(`错误：${errorMessage}`);
    }

    const error = new Error(errorMessage);
    error.code = errorCode;
    throw error;
  }
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
  const manifest = games.find((game) => game.id === state.selectedGame);
  if (!manifest) {
    toast(`缺少游戏配置：${state.selectedGame}`);
    return;
  }
  activeGame = module;
  module.enter(manifest);
}

// ---- 启动 ----
const engine = {
  state,
  $,
  setPhase,
  toast,
  stopSpeech,
  requestJson,
  returnToSelect,
  createRoundClient,
  playDirective,
};

registerGame(registerDice(engine));
registerGame(registerRps(engine));

$('startGame').addEventListener('click', enterSelectedGame);
$('gameList').addEventListener('dblclick', enterSelectedGame);
$('soundToggle').addEventListener('click', () => {
  state.sound = !state.sound;
  if (!state.sound) stopSpeech();
  $('soundToggle').textContent = state.sound ? '🔊' : '🔇';
  toast(state.sound ? 'TTS 播报已开启' : '语音播报已关闭');
});

document.addEventListener('keydown', (event) => {
  const controllerKey = ['Enter', 'Escape', 'ArrowDown', 'ArrowUp'].includes(event.key);
  if (controllerKey) event.preventDefault();
  if (event.repeat) return;

  if (state.phase === 'select') {
    if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
      const enabled = games.filter((game) => game.enabled);
      if (!enabled.length) return;
      const ids = enabled.map((game) => game.id);
      const index = ids.indexOf(state.selectedGame);
      const next = ids[(index + (event.key === 'ArrowDown' ? 1 : -1) + ids.length) % ids.length];
      selectGame(next);
    } else if (event.key === 'Enter') {
      enterSelectedGame();
    } else if (event.key === 'Escape') {
      stopSpeech();
    }
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
