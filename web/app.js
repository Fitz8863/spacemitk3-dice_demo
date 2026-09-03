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
  resetIdleTimer();
}

// ---- 待机页（全体游戏之前） ----
// 游戏列表页空闲 idle_seconds（全局配置 standby.idle_seconds，默认 120s）后
// 进入待机动画；任意输入或语音唤醒词唤醒回列表。boot_standby=true 时页面
// 加载直接进待机（配置来自 /api/games 的 standby 投影，热加载生效）。
// 对局页面不待机（对局有自身节奏，玩家离场由服务端 SSE 看门收回合 →
// round_complete → 回到列表 → 计时自然恢复）。待机中的第一次输入只唤醒
// 不穿透，防止睡着时误开一局。唤醒词监听在服务端（无回合也开麦），轮询
// /api/asr/standby/events 拿唤醒事件；唤醒同样不穿透——只回列表不进游戏。
const IDLE_ENTER_SECONDS = 120; // /api/games 不可用或未配置时的兜底
let idleTimer = null;
let standbySettings = { enabled: true, idle_seconds: IDLE_ENTER_SECONDS, boot_standby: false, wake_phrases: [] };
let standbyEventCursor = 0;
let standbyPollTimer = null;

function applyStandbySettings(settings) {
  if (!settings || typeof settings !== 'object') return;
  standbySettings = {
    enabled: settings.enabled !== false,
    idle_seconds: Number(settings.idle_seconds) > 0 ? Number(settings.idle_seconds) : IDLE_ENTER_SECONDS,
    boot_standby: settings.boot_standby === true,
    wake_phrases: Array.isArray(settings.wake_phrases) ? settings.wake_phrases : [],
  };
  resetIdleTimer();
}

function resetIdleTimer() {
  clearTimeout(idleTimer);
  if (state.phase !== 'select' || !standbySettings.enabled) return;
  idleTimer = setTimeout(enterStandby, standbySettings.idle_seconds * 1000);
}

function enterStandby() {
  clearTimeout(idleTimer);
  idleTimer = null;
  if (state.phase !== 'select' || !standbySettings.enabled) return;
  setPhase('standby');
  const wakeNode = $('standbyWakeWords');
  if (wakeNode) {
    if (standbySettings.wake_phrases.length) {
      wakeNode.textContent = `或对我说：${standbySettings.wake_phrases.join(' / ')}`;
      wakeNode.classList.remove('hidden');
    } else {
      wakeNode.classList.add('hidden');
    }
  }
  startStandbyListening();
}

function wakeFromStandby() {
  if (state.phase !== 'standby') return false;
  setPhase('select');
  stopStandbyListening();
  return true;
}

// --- 待机语音监听（服务端会话 + 轮询唤醒事件） ---
function startStandbyListening() {
  stopStandbyListening();
  if (!standbySettings.wake_phrases.length) return; // 没配唤醒词就纯按键/触摸唤醒
  requestJson('/api/asr/standby', {
    method: 'POST',
    body: JSON.stringify({ listen: true }),
  }).then((payload) => {
    // 事件纪元从本次监听开始：cursor 之前的历史事件（上一段待机期间环境
    // 人声/幻觉产生的旧唤醒）绝不重放——刷新页面瞬间被"旧唤醒"唤醒过的 bug。
    standbyEventCursor = Number(payload?.cursor || 0);
  }).catch(() => { /* 后端不可用不阻塞待机画面 */ });
  standbyPollTimer = setInterval(async () => {
    try {
      const payload = await requestJson('/api/asr/standby/events');
      for (const event of (payload.events || [])) {
        if (event.sequence > standbyEventCursor) {
          standbyEventCursor = event.sequence;
          if (event.status === 'wake') {
            // 保鲜窗：超过 10 秒的唤醒事件视为过期（防御旧事件重放）。
            if (Date.now() - Number(event.timestamp_ms || 0) <= 10000) {
              showAsrFeedback({ status: 'submitted', text: event.text });
              wakeFromStandby();
              return;
            }
          } else {
            showAsrFeedback({ status: 'unmatched', text: event.text });
          }
        }
      }
    } catch (_) { /* 轮询失败下次再试 */ }
  }, 1200);
}

function stopStandbyListening() {
  if (standbyPollTimer) {
    clearInterval(standbyPollTimer);
    standbyPollTimer = null;
  }
  requestJson('/api/asr/standby', {
    method: 'POST',
    body: JSON.stringify({ listen: false }),
  }).catch(() => { /* already stopped */ });
}

// 捕获阶段拦截：待机中的输入只用于唤醒，绝不继续派发给列表/按钮。
// keydown/click 都是一次性完整事件（click 由点击或触摸 tap 合成）——
// 唤醒用的这一次事件被吞掉，不会落到列表项上误开一局。keyup 兜底那些
// 只上报释放事件的按键设备。
['keydown', 'keyup', 'click'].forEach((type) => {
  window.addEventListener(type, (event) => {
    if (state.phase === 'standby' && type !== 'click') {
      // 待机页可见诊断：确认设备按键是否到达页面（按键后屏幕左上角闪现）。
      const probe = $('standbyProbe');
      if (probe) {
        probe.textContent = `收到按键: ${event.key || '?'}`;
        probe.classList.add('show');
        clearTimeout(probe.__timer);
        probe.__timer = setTimeout(() => probe.classList.remove('show'), 1500);
      }
    }
    if (wakeFromStandby()) {
      event.stopPropagation();
      event.preventDefault();
    }
    resetIdleTimer();
  }, { capture: true });
});

// ---- 提示 ----
function toast(message) {
  const node = $('toast');
  node.textContent = message;
  node.classList.add('show');
  setTimeout(() => node.classList.remove('show'), 2800);
}

// ---- ASR 语音反馈 ----
// 后端把每句识别结果以 asr 观测事件广播（生效/被播报闸暂缓/当前状态
// 不支持/未匹配触发词）。任何游戏的语音输入都在引擎层统一反馈，游戏
// 模块不感知；播报中与未匹配也要提示，否则玩家不知道语音通道在工作。
let asrFeedbackTimer = null;
function showAsrFeedback(event) {
  const node = $('asrFeedback');
  if (!node) return;
  const heard = `听到「${event.text || ''}」`;
  let message;
  let level;
  if (event.status === 'submitted') {
    message = `🎤 ${heard}，已生效`;
    level = 'ok';
  } else if (event.status === 'suppressed') {
    message = `🎤 ${heard}——正在播报，暂不生效；不想等可按绿色按钮`;
    level = 'wait';
  } else if (event.status === 'rejected') {
    message = `🎤 ${heard}（当前步骤不支持语音操作，请使用按键）`;
    level = 'info';
  } else {
    message = `🎤 ${heard}（未匹配语音指令）`;
    level = 'info';
  }
  node.textContent = message;
  node.className = `asr-feedback show level-${level}`;
  clearTimeout(asrFeedbackTimer);
  asrFeedbackTimer = setTimeout(() => node.classList.remove('show'), level === 'wait' ? 3600 : 2400);
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

// 播放一条后端 speech 指令：拉帧 → 排片播放 → 回执 speech_done。
// 回执不再仅限 await 指令：引擎用它清除"播报进行中"登记（ASR 播报闸
// 依赖该登记），所以每条指令播完/失败/被顶替都要回执，finally 统一出口。
async function playDirective(round, directive) {
  const acknowledge = () => {
    if (round && round.roundId) {
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
  } catch (error) {
    await producer.catch(() => {});
    if (error.name === 'AbortError' || requestId !== state.ttsRequestId || !state.sound) return;
    console.error(`Speech directive ${directive.directive_id} failed after ${playedFrames} frame(s):`, error);
    toast('语音播放失败，请检查语音组件');
  } finally {
    if (scheduler && state.ttsPlaybackCancel === scheduler.cancel) {
      state.ttsPlaybackCancel = null;
    }
    // 所有退出路径（成功/失败/被新指令顶替）都恰好回执一次：await 的
    // 唤醒引擎等待者，非 await 的释放播报闸登记。
    acknowledge();
  }
}

// ---- 页面离开保险 ----
// 关标签/刷新/跳转时尽力取消当前对局（sendBeacon 在页面卸载后仍会送达，
// /cancel 无请求体，不会毒化 keep-alive 连接）。断网/休眠等 sendBeacon
// 覆盖不到的断开由服务端 SSE 活性看门兜底：最后一个消费者断开并超过
// 宽限期后自动取消回合、释放麦克风。
let activeRound = null;
window.addEventListener('pagehide', () => {
  const roundId = activeRound?.roundId;
  if (!roundId) return;
  try {
    navigator.sendBeacon?.(`/api/game/rounds/${roundId}/cancel`);
  } catch (_) { /* best effort: the server watchdog covers the rest */ }
});

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
    } else if (event.event === 'asr') {
      showAsrFeedback(event);
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
      // No body: a cancel carries no payload, and an unread body would
      // corrupt keep-alive connections on the server.
      await requestJson(`/api/game/rounds/${target}/cancel`, { method: 'POST' });
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
  stopStandbyListening(); // 对局结束回列表：确保没有残留的待机监听/轮询
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
  // 游戏模块在 enter/teardown 时登记当前对局客户端，pagehide 保险据此取 roundId
  setActiveRound(client) { activeRound = client || null; },
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
    applyStandbySettings(payload.standby);
    renderGameList();
    const firstEnabled = games.find((game) => game.enabled);
    if (firstEnabled) selectGame(firstEnabled.id);
  } catch (error) {
    console.error('Failed to load game list:', error);
    toast('游戏列表加载失败，请检查后端 /api/games');
  }
}

setPhase('select');
loadGames().then(() => {
  // boot_standby=true：页面加载即待机（设备就绪的舞台状态）；false 停在列表。
  if (standbySettings.boot_standby) enterStandby();
});
