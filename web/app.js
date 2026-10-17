// 时空游乐舱前端引擎：跨游戏通用的视图切换、语音指令播放、网络与游戏选择。
// 游戏流程由后端权威状态机驱动：前端通过 RoundClient 创建对局、提交意图、
// 订阅事件流并渲染；台词以后端下发的 speech 指令播放，await 指令播完回执。
import { register as registerDice } from './games/dice.js';
import { register as registerRps } from './games/rps.js';
import { createSessionClient } from './shared-session.js';
import { refreshDisplayManifest } from './display-layout.js';

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

// The companion is ambient decoration on the menu, but becomes part of the
// camera composition during a game. Reparenting the existing SVG keeps one
// accessible mascot in the DOM and makes its position follow the video frame
// instead of the viewport. Game modules toggle .dice-video when entering and
// leaving, so a small observer is enough for both dice and RPS.
function bindBuddyToLiveVideo() {
  const panel = $('analysisStreamPanel');
  const home = document.querySelector('.buddy-panel');
  const buddy = home?.querySelector('.buddy-art');
  const homeAnchor = home?.querySelector('.buddy-label');
  if (!panel || !home || !buddy || !homeAnchor) return;

  const sync = () => {
    if (panel.classList.contains('dice-video')) {
      if (buddy.parentElement !== panel) panel.append(buddy);
      return;
    }
    if (buddy.parentElement !== home) home.insertBefore(buddy, homeAnchor);
  };

  new MutationObserver(sync).observe(panel, {
    attributes: true,
    attributeFilter: ['class'],
  });
  sync();
}

const SELECT_META = ['挑个游戏，开玩吧！', '选好游戏，再按下绿色按钮，小搭子陪你一起出发。'];

let currentPhaseMeta = SELECT_META;
function renderThemeHeading() {
  const phase = state.phase;
  const resolved = window.ArenaTheme.resolveMeta(phase, currentPhaseMeta, activeGame?.id);
  $('phaseTitle').textContent = resolved[0];
  renderPhaseCopy(phase, resolved[1]);
  window.ArenaTheme.renderPhase(state.phase, activeGame?.id, state.selectedGame);
}
window.addEventListener('arena-theme-change', renderThemeHeading);

const gameModules = {};
let activeGame = null; // 当前挂载的游戏模块（有 enter/teardown/onKey）
let games = []; // GET /api/games 返回的列表
let boardInputSource = { local_client: false, enabled: false, available: false };
let sharedSession = null;
let sessionRender = Promise.resolve();
let sessionApplying = false;
let sessionRoundId = null;
let lastSharedMode = false;
let lastActivitySent = 0;

function sessionAction(action, data = {}) {
  return sharedSession.action(action, data).catch((error) => {
    if (!error.silent && error.code !== 'SESSION_STALE') console.error('Shared action failed:', error);
  });
}

function sessionActivity() {
  if (!sharedSession?.shared || Date.now() - lastActivitySent < 1000) return;
  lastActivitySent = Date.now();
  sessionAction('activity');
}

function applySessionSnapshot(snapshot) {
  boardInputSource.available = snapshot.board_input?.available === true;
  boardInputSource.enabled = snapshot.board_input?.enabled === true;
  sessionRender = sessionRender.catch(console.error).then(async () => {
    if (sharedSession.snapshot !== snapshot) return; // A newer snapshot already arrived.
    if (!snapshot.shared) {
      if (lastSharedMode) {
        lastSharedMode = false;
        sessionApplying = true;
        try { returnToSelect(); } finally { sessionApplying = false; }
        if (snapshot.phase === 'standby') enterStandby();
      }
      return;
    }
    lastSharedMode = true;
    clearTimeout(idleTimer);
    clearIdleReturn();
    if (standbySource) { standbySource.close(); standbySource = null; }
    if (selectSource) { selectSource.close(); selectSource = null; }
    sessionApplying = true;
    try {
      if (!snapshot.round_id) {
        if (activeGame) activeGame.teardown();
        activeGame = null;
        sessionRoundId = null;
        selectGame(snapshot.selected_game, true);
        setPhase(snapshot.phase === 'standby' ? 'standby' : 'select');
        return;
      }
      if (sessionRoundId !== snapshot.round_id) {
        const cached = games.find((game) => game.id === snapshot.game_id);
        const module = gameModules[snapshot.game_id];
        if (!cached || !module) throw new Error(`Unknown shared game: ${snapshot.game_id}`);
        const manifest = await refreshDisplayManifest(cached, requestJson);
        // An exit or a newer round may arrive while the configuration is loading.
        if (!sharedSession.shared || sharedSession.snapshot?.round_id !== snapshot.round_id) return;
        activeGame?.teardown();
        activeGame = module;
        sessionRoundId = snapshot.round_id;
        await module.enter(manifest, { roundId: snapshot.round_id });
      }
    } finally { sessionApplying = false; }
  });
}


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
// 视图在 .stage-body 的 grid 叠层（所有 .view 同一格）上交叉淡化：旧视图加
// .view-leave 淡出 130ms 后再隐藏，新视图照常 fadeIn——消除「旧视图一帧
// 消失、新视图从零亮起」的闪断。计时器只在引擎层；同相位重入与快速连切
// （如 shaking→stop_call 同视图）都先清挂起计时器并回收 leave 类。
// 收敛必须全量：130ms 动画窗内可能叠着不止一个旧视图（快速连切/回跳），
// 后台标签页还会节流计时器——只按 DOM 顺序收回第一个可见视图会让中间
// 切过的视图永久叠在屏上（2026-10-09 实测 rules 三卡片与 ready 标题叠显）。
// 因此每个离场视图独立计时收敛，进入新视图时把其余视图全部送入离场。
let phaseLeaveTimers = new Map();
const PHASE_LEAVE_MS = 130;
function setPhase(phase, meta) {
  state.phase = phase;
  // Phases style themselves via body[data-phase] (e.g. the open phase lowers
  // the stage header into mid-screen).
  document.body.dataset.phase = phase;
  const incoming = views.find((view) => view.dataset.view === phase) || null;
  for (const view of views) {
    const pending = phaseLeaveTimers.get(view);
    if (view === incoming) {
      // 回跳进一个尚在离场动画的视图：立即恢复显示并撤销其隐藏计时。
      if (pending) {
        clearTimeout(pending);
        phaseLeaveTimers.delete(view);
      }
      view.classList.remove('view-leave');
      view.classList.remove('hidden');
      continue;
    }
    if (view.classList.contains('hidden') && !pending) {
      continue; // 本就不可见，也无待收敛动画
    }
    if (pending) {
      continue; // 已在离场动画中，计时器到期会隐藏，无需重复
    }
    view.classList.add('view-leave');
    phaseLeaveTimers.set(view, setTimeout(() => {
      phaseLeaveTimers.delete(view);
      view.classList.remove('view-leave');
      view.classList.add('hidden');
    }, PHASE_LEAVE_MS));
  }
  const resolved = meta || (activeGame && activeGame.phaseMeta && activeGame.phaseMeta[phase]) || SELECT_META;
  currentPhaseMeta = resolved;
  renderThemeHeading();
  resetIdleTimer();
}

// ---- 待机页（全体游戏之前） ----
// 游戏列表页空闲 idle_seconds（全局配置 standby.idle_seconds，默认 120s）后
// 进入待机动画；任意输入或语音唤醒词唤醒回列表。boot_standby=true 时页面
// 加载直接进待机（配置来自 /api/games 的 standby 投影，热加载生效）。
// 对局页面不待机（对局有自身节奏，玩家离场由服务端 SSE 看门收回合 →
// round_complete → 回到列表 → 计时自然恢复）。待机中的第一次输入只唤醒
// 不穿透，防止睡着时误开一局。待机语音监听在服务端（无回合也开麦），
// 轮询 /api/asr/standby/events：唤醒词回列表；点名游戏（词表 = 游戏名，
// 优先级高于唤醒词）直达对局。
const IDLE_ENTER_SECONDS = 120; // /api/games 不可用或未配置时的兜底
let idleTimer = null;
let standbySettings = { enabled: true, idle_seconds: IDLE_ENTER_SECONDS, boot_standby: false, wake_phrases: [] };
let standbySource = null;

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
  if (typeof sharedSession !== 'undefined' && sharedSession?.shared) return;
  clearTimeout(idleTimer);
  if (state.phase !== 'select' || !standbySettings.enabled) return;
  idleTimer = setTimeout(enterStandby, standbySettings.idle_seconds * 1000);
}

// --- 对局空闲返回（引擎层计时，游戏模块声明） ---
// 对局的等待态（听规则/准备/结果/裁决失败）没有 duration 也没有 on_expire，
// 玩家走开后回合会永远停在那里。服务端的 SSE 看门只管"浏览器断开"，不管
// "人在不在"，所以由引擎在这类状态计时：超时走与关页同一条取消路径回列表
// （teardown → round.cancel），列表的待机计时随后自然接管。阈值复用
// standby.idle_seconds，不新增配置；待机功能关闭时一并关闭。
// 游戏模块只声明"当前在等玩家操作"（setIdleReturn），计时器留在引擎层。
let idleReturnTimer = null;
let idleReturnArmed = false;

function clearIdleReturn() {
  if (idleReturnTimer) {
    clearTimeout(idleReturnTimer);
    idleReturnTimer = null;
  }
}

function armIdleReturn() {
  if (typeof sharedSession !== 'undefined' && sharedSession?.shared) return;
  clearIdleReturn();
  if (!idleReturnArmed || !standbySettings.enabled) return;
  idleReturnTimer = setTimeout(() => {
    idleReturnTimer = null;
    toast('长时间无操作，已结束本局');
    returnToSelect();
  }, standbySettings.idle_seconds * 1000);
}

function setIdleReturn(armed) {
  if (typeof sharedSession !== 'undefined' && sharedSession?.shared) return;
  idleReturnArmed = armed === true;
  if (idleReturnArmed) armIdleReturn();
  else clearIdleReturn();
}

function enterStandby() {
  if (typeof sharedSession !== 'undefined' && sharedSession?.shared) return;
  boardSelectionEntry = null;
  clearTimeout(idleTimer);
  idleTimer = null;
  if (state.phase !== 'select' || !standbySettings.enabled) return;
  stopSelectListening(); // 列表语音选游戏让位给待机唤醒/点名
  setPhase('standby');
  startStandbyListening();
}

function wakeFromStandby() {
  if (typeof sharedSession !== 'undefined' && sharedSession?.shared) {
    if (state.phase !== 'standby') return false;
    sessionAction('wake');
    return true;
  }
  if (state.phase !== 'standby') return false;
  setPhase('select');
  selectDefaultGame();
  stopStandbyListening();
  startSelectListening();
  return true;
}

// --- 待机语音监听（服务端会话 + 轮询事件） ---
// 词表在服务端拼：游戏名在前、唤醒词在后（"我想玩摇骰子游戏"选游戏，
// 即使唤醒词"游戏"也在句中）；所以这里不再前端预判唤醒词是否为空。
function startStandbyListening() {
  if (typeof sharedSession !== 'undefined' && sharedSession?.shared) return;
  stopStandbyListening();
  requestJson('/api/asr/standby', {
    method: 'POST',
    body: JSON.stringify({ listen: true }),
  }).then(() => {
    // 事件纪元从本次监听开始：POST listen 清掉总线里的历史事件（上一段
    // 待机期间环境人声/幻觉产生的旧唤醒）绝不重放——刷新页面瞬间被"旧
    // 唤醒"唤醒过的 bug。
  }).catch(() => { /* 后端不可用不阻塞待机画面 */ });
  // SSE 推送（EventSource 不吃后台标签页定时器节流——轮询版被 Chromium
  // 钳到 1s/1min，是"板端按键好一会才反应"的根因）；断线由 EventSource
  // 自动重连，id 序号续传不重放旧事件。
  standbySource = new EventSource('/api/asr/standby/stream');
  standbySource.addEventListener('bus', (message) => {
    let event = null;
    try { event = JSON.parse(message.data); } catch (_) { return; }
    // 保鲜窗：超过 10 秒的事件视为过期（防御旧事件重放）。
    const fresh = Date.now() - Number(event.timestamp_ms || 0) <= 10000;
    // 板端键盘的合成事件带 source:'board'：动作照做（唤醒/进局/高亮），
    // 但不弹「听到…」语音浮条——按键不是语音，弹了就是误导（2026-10-10）。
    const heard = event.source !== 'board';
    if (event.status === 'board_input' && event.source === 'board') {
      boardInputSource.enabled = event.enabled === true;
      boardInputSource.available = event.available === true;
      return;
    }
    if (event.status === 'wake') {
      if (fresh) {
        if (heard) showAsrFeedback({ status: 'submitted', text: event.text });
        wakeFromStandby();
      }
    } else if (event.status === 'selected' && event.game_id) {
      // 待机页直接点名游戏：跳过列表直达对局（板端键盘单次按压同款）。
      if (fresh) {
        if (heard) showAsrFeedback({ status: 'submitted', text: event.text });
        wakeFromStandby();
        if (!heard && boardInputSource.local_client) enterSelectedGame();
        else enterGameById(event.game_id);
      }
    } else if (event.status !== 'board_navigate' && heard) {
      showAsrFeedback({ status: 'unmatched', text: event.text });
    }
  });
}

function stopStandbyListening() {
  if (typeof sharedSession !== 'undefined' && sharedSession?.shared) return;
  if (standbySource) {
    standbySource.close();
    standbySource = null;
  }
  requestJson('/api/asr/standby', {
    method: 'POST',
    body: JSON.stringify({ listen: false }),
  }).catch(() => { /* already stopped */ });
}

// --- 列表页语音选游戏（与待机层同构：会话 + 纪元 + 轮询 + 保鲜窗） ---
// 玩家在游戏列表说"我想玩摇骰子游戏"（点名）或"进入游戏"（确认当前
// 高亮项），服务端在常驻引擎上换绑选路路由（词表 = 各启用游戏名/别名
// + 全局配置里的确认词），命中推 selected/confirm 事件；前端走与绿色
// 按钮同一条 enterSelectedGame 路径进对局。进对局/进待机时轮询停止
// （服务端路由由回合/待机会话自动顶掉）。
let selectSource = null;

function startSelectListening() {
  if (typeof sharedSession !== 'undefined' && sharedSession?.shared) return;
  stopSelectListening();
  requestJson('/api/asr/select', {
    method: 'POST',
    body: JSON.stringify({ listen: true }),
  }).catch(() => { /* 后端不可用不阻塞列表页 */ });
  // SSE 推送（同待机层：EventSource 不吃后台标签页定时器节流）。
  selectSource = new EventSource('/api/asr/select/stream');
  selectSource.addEventListener('bus', (message) => {
    let event = null;
    try { event = JSON.parse(message.data); } catch (_) { return; }
    // 保鲜窗：超过 10 秒的选择事件视为过期（防御旧事件重放）。
    const fresh = Date.now() - Number(event.timestamp_ms || 0) <= 10000;
    // 板端键盘合成事件（source:'board'）只做动作，不弹语音浮条（2026-10-10）。
    const heard = event.source !== 'board';
    if (event.status === 'board_input' && event.source === 'board') {
      boardInputSource.enabled = event.enabled === true;
      boardInputSource.available = event.available === true;
      return;
    }
    if (event.status === 'selected' && event.game_id) {
      if (fresh) {
        if (heard) showAsrFeedback({ status: 'submitted', text: event.text });
        // A physical Enter may have raced the page's pending default/mouse sync.
        // The local page owns its highlight; remote pages keep the existing relay.
        if (!heard && boardInputSource.local_client) enterSelectedGame();
        else enterGameById(event.game_id);
      }
    } else if (event.status === 'confirm') {
      // 语音确认：进入当前高亮的游戏（上下键/点选已改写选中态）。
      if (fresh) {
        if (heard) showAsrFeedback({ status: 'submitted', text: event.text });
        enterSelectedGame();
      }
    } else if (event.status === 'board_navigate' && event.game_id) {
      // 板端键盘光标同步（input_board_enable）：只挪高亮、不进局——
      // 进局走 selected 事件或本地 Enter，enterSelectedGame 的相位
      // 守卫保证同一记按键不会建两个回合。
      if (fresh) selectGame(event.game_id);
    } else if (event.status !== 'wake' && heard) {
      showAsrFeedback({ status: 'unmatched', text: event.text });
    }
  });
}

function stopSelectListening() {
  if (typeof sharedSession !== 'undefined' && sharedSession?.shared) return;
  if (selectSource) {
    selectSource.close();
    selectSource = null;
  }
  requestJson('/api/asr/select', {
    method: 'POST',
    body: JSON.stringify({ listen: false }),
  }).catch(() => { /* already stopped */ });
}

function enterGameById(gameId) {
  const manifest = games.find((game) => game.id === gameId && game.enabled);
  if (!manifest) return; // 事件指向禁用/未知的游戏：当作没听见
  selectGame(gameId);
  enterSelectedGame();
}

// 捕获阶段拦截：待机中的输入只用于唤醒，绝不继续派发给列表/按钮。
// 唤醒通道只有两种：语音唤醒词（服务端监听）+ 按键（keydown/keyup，
// keyup 兜底只上报释放事件的按键设备）。刻意不含 click——鼠标误点不
// 应唤醒待机。
['keydown', 'keyup'].forEach((type) => {
  window.addEventListener(type, (event) => {
    if (sharedSession?.shared && ['Enter', 'Escape', 'ArrowDown', 'ArrowUp'].includes(event.key)
        && boardInputSource.local_client && boardInputSource.available && sharedSession.ready) {
      event.stopPropagation(); event.preventDefault(); return;
    }
    if (sharedSession?.shared && type === 'keydown') sessionActivity();
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
    armIdleReturn();
  }, { capture: true });
});

// 对局空闲返回把"人在场"的证据放宽到指针输入：与待机唤醒不同（那里刻意排除
// click 防误唤醒），按钮点击本身就是要重置空闲计时的信号。
window.addEventListener('pointerdown', () => { armIdleReturn(); sessionActivity(); }, { capture: true });

// ---- 提示 ----
let toastTimer = null;
function toast(message) {
  const node = $('toast');
  node.textContent = message;
  node.classList.add('show');
  // 连续 toast 时先清旧计时器，否则上一条的隐藏定时器会把新 toast 提前掐掉。
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove('show'), 2800);
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

// Deliberately do not call browser speech synthesis. The game UI must use the
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

async function playSpeechBlob(blob, requestId, onStarted = () => {}) {
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
    onStarted();
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

function createSpeechScheduler(requestId, onStarted = null) {
  const context = getSpeechAudioContext();
  if (!context) return null;
  const player = {
    context,
    nextAt: 0,
    sources: [],
    cancelled: false,
    resolveDrained: null,
    startTimer: null,
    notifiedStart: false,
  };
  const cancelSource = (source) => {
    try { source.stop(); } catch (_) { /* already ended */ }
    source.disconnect();
  };
  player.schedule = async (blob) => {
    if (player.cancelled || requestId !== state.ttsRequestId || !state.sound) {
      throw new DOMException('Speech playback was cancelled', 'AbortError');
    }
    if (onStarted) await context.resume();
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
    if (onStarted && !player.notifiedStart) {
      player.notifiedStart = true;
      const checkStarted = () => {
        if (player.cancelled || requestId !== state.ttsRequestId) return;
        if (context.state === 'running' && context.currentTime >= startAt) onStarted();
        else player.startTimer = setTimeout(checkStarted, 10);
      };
      player.startTimer = setTimeout(checkStarted, Math.max(0, (startAt - context.currentTime) * 1000));
    }
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
    clearTimeout(player.startTimer);
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

// 「回执压后」用的最小定时原语。
function waitSeconds(seconds) {
  return new Promise((resolve) => setTimeout(resolve, seconds * 1000));
}

// 播放一条后端 speech 指令：拉帧 → 排片播放 → 回执 speech_done。
// 回执不再仅限 await 指令：引擎用它清除"播报进行中"登记（ASR 播报闸
// 依赖该登记），所以每条指令播完/失败/被顶替都要回执，finally 统一出口。
//
// options.ackHoldSeconds = 把回执压后 N 秒再发。后端对 await 的台词是
// 「收到 speech_done 才发下一句、才启动该状态的 duration 计时」，所以压后
// 回执＝在这句与下一句之间插入 N 秒静音，且下游整条序列（含状态机计时器）
// 一起后移——游戏节奏与「语音↔屏幕倒计时」的相对对齐都不变。用于需要停顿的
// 转场（骰子：停！之后静 2 秒再念开盖词）。
async function playDirective(round, directive, options = {}) {
  if (typeof sharedSession !== 'undefined' && sharedSession?.shared) return playSharedDirective(round, directive);
  const ackHoldSeconds = Math.max(0, Number(options.ackHoldSeconds) || 0);
  const targetRoundId = round?.roundId;
  let startNotified = false;
  let startAck = Promise.resolve();
  const notifyStarted = () => {
    if (startNotified || !directive.notify_start || !targetRoundId || round.roundId !== targetRoundId) return;
    startNotified = true;
    startAck = round.submitIntent('speech_started', { directive_id: directive.directive_id })
      .catch((error) => { console.error('Speech start acknowledgement failed:', error); });
  };
  const acknowledge = () => {
    if (round && round.roundId === targetRoundId) {
      round.submitIntent('speech_done', { directive_id: directive.directive_id })
        .catch(() => { /* round may already be closed; the engine fallback covers it */ });
    }
    // 播报结束即重新起算空闲退出：规则/结果这类长台词播到一半被计时器掐断，
    // 在台上是可见事故，所以空闲窗口从"台词播完"而不是"台词开始"算起。
    armIdleReturn();
  };
  if (!state.sound) {
    notifyStarted();
    await startAck;
    if (directive.notify_start) await waitSeconds(Number(directive.playback_seconds) || 0);
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
  const scheduler = createSpeechScheduler(requestId, directive.notify_start ? notifyStarted : null);
  if (scheduler) state.ttsPlaybackCancel = scheduler.cancel;
  let playedFrames = 0;
  // 只有"正常播完"才做回执压后；被取消/被新指令顶替时立即回执，不去为一条
  // 已经作废的等待多压 2 秒。
  let playbackComplete = false;
  try {
    // One HTTP request per directive. The producer keeps reading later WAV
    // frames while the consumer plays the first one; the scheduler lines
    // frames up back-to-back so streaming TTS never gaps between frames.
    while (true) {
      const blob = await queue.next();
      if (blob === null) break;
      if (scheduler) await scheduler.schedule(blob);
      else await playSpeechBlob(blob, requestId, notifyStarted);
      playedFrames += 1;
    }
    if (scheduler) await scheduler.waitDrained();
    await producer;
    playbackComplete = true;
  } catch (error) {
    await producer.catch(() => {});
    if (error.name === 'AbortError' || requestId !== state.ttsRequestId || !state.sound) return;
    console.error(`Speech directive ${directive.directive_id} failed after ${playedFrames} frame(s):`, error);
  } finally {
    if (scheduler && state.ttsPlaybackCancel === scheduler.cancel) {
      state.ttsPlaybackCancel = null;
    }
    await startAck;
    if (playbackComplete && ackHoldSeconds > 0) await waitSeconds(ackHoldSeconds);
    // 所有退出路径（成功/失败/被新指令顶替）都恰好回执一次：await 的
    // 唤醒引擎等待者，非 await 的释放播报闸登记。
    acknowledge();
  }
}

// All remote displays consume the same cached audio. Only the backend player
// acknowledges it; muting or an autoplay block cannot advance the round.
let audioUnlockNeeded = false;
function showAudioUnlock() {
  audioUnlockNeeded = true;
  $('audioUnlock')?.classList.remove('hidden');
}

async function playSharedDirective(round, directive) {
  if (boardInputSource.local_client || !state.sound || !round?.roundId) return;
  stopSpeech();
  const context = getSpeechAudioContext();
  if (context && context.state !== 'running') {
    showAudioUnlock();
    return;
  }
  const requestId = state.ttsRequestId;
  const controller = new AbortController();
  state.ttsAbortController = controller;
  const queue = createTtsFrameQueue();
  let reader;
  const producer = (async () => {
    try {
      const response = await fetch(`/api/game/rounds/${round.roundId}/speech`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ directive_id: directive.directive_id, live: true }), signal: controller.signal,
      });
      if (!response.ok || !response.body) throw new Error(`Audio HTTP ${response.status}`);
      reader = response.body.getReader();
      for await (const blob of readTtsFrames(reader)) queue.push(blob);
      queue.finish();
    } catch (error) { queue.finish(error); }
    finally { try { await reader?.cancel(); } catch (_) {} reader?.releaseLock(); }
  })();
  const scheduler = createSpeechScheduler(requestId);
  if (scheduler) state.ttsPlaybackCancel = scheduler.cancel;
  try {
    while (true) {
      const blob = await queue.next();
      if (blob === null || requestId !== state.ttsRequestId) break;
      if (scheduler) await scheduler.schedule(blob);
      else await playSpeechBlob(blob, requestId);
    }
    if (scheduler) await scheduler.waitDrained();
    await producer;
  } catch (error) {
    controller.abort();
    if (error.name === 'NotAllowedError') showAudioUnlock();
    else if (error.name !== 'AbortError') console.error('Shared speech failed:', error);
  } finally {
    await producer.catch(() => {});
    if (state.ttsPlaybackCancel === scheduler?.cancel) state.ttsPlaybackCancel = null;
    if (state.ttsAbortController === controller) state.ttsAbortController = null;
  }
}

async function resumeSharedAudio() {
  const roundId = activeRound?.roundId;
  if (!sharedSession?.shared || !roundId || boardInputSource.local_client || !state.sound) return;
  const audio = await requestJson(`/api/game/rounds/${roundId}/speech/current`);
  if (audio.directive && activeRound?.roundId === roundId) playSharedDirective(activeRound, audio.directive);
}

$('audioUnlock')?.addEventListener('click', async () => {
  const context = getSpeechAudioContext();
  try {
    await context?.resume();
    audioUnlockNeeded = false;
    $('audioUnlock')?.classList.add('hidden');
    await resumeSharedAudio();
  } catch (_) { showAudioUnlock(); }
});

// ---- 页面离开保险 ----
// 关标签/刷新/跳转时尽力取消当前对局（sendBeacon 在页面卸载后仍会送达，
// /cancel 无请求体，不会毒化 keep-alive 连接）。断网/休眠等 sendBeacon
// 覆盖不到的断开由服务端 SSE 活性看门兜底：最后一个消费者断开并超过
// 宽限期后自动取消回合、释放麦克风。
let activeRound = null;
window.addEventListener('pagehide', () => {
  if (typeof sharedSession !== 'undefined' && sharedSession?.shared) return;
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
  let currentSnapshot = null;
  let firstSharedSnapshot = true;
  let lastAudioDirective = null;
  let detachSessionRound = null;
  // Set once the round reaches a terminal snapshot or the client cancels:
  // afterwards every event is stale game state and must not reach the UI,
  // otherwise a finished view resurrects itself after returnToSelect().
  let closed = false;

  function dispatch(event, snapshot) {
    if (event.event === 'state_changed') {
      handlers.onStateChange?.(event.state, event.ui || {}, snapshot || null);
    } else if (event.event === 'speech') {
      lastAudioDirective = event.directive_id;
      handlers.onSpeech?.(event);
    } else if (event.event === 'audio_error') {
      // Audio is best effort in shared mode. The backend clock and board
      // player keep the round moving; do not disturb the display with a toast.
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
    currentSnapshot = snapshot;
    if (typeof sharedSession !== 'undefined' && sharedSession?.shared) {
      if (firstSharedSnapshot) {
        firstSharedSnapshot = false;
        lastSequence = Number(snapshot.event_sequence || 0);
        handlers.onSyncState?.(snapshot);
        for (const item of (snapshot.observations || [])) {
          if (item.event === 'tick') handlers.onTick?.(item);
          else handlers.onEvent?.(item, snapshot);
        }
        // Finished error rounds keep their failure view; do not replay old speech.
        if (snapshot.status === 'error') handlers.onComplete?.({ status: 'error' }, snapshot);
        return;
      }
    }
    for (const item of (snapshot.events || [])) {
      const sequence = Number(item.sequence || 0);
      if (sequence > lastSequence) {
        lastSequence = sequence;
        if (typeof sharedSession !== 'undefined' && sharedSession?.shared && item.event === 'speech'
            && (item.state_version !== snapshot.state_version || snapshot.speech_active === false)) continue;
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
    if (typeof sharedSession !== 'undefined' && sharedSession?.shared) {
      detachSessionRound?.();
      detachSessionRound = sharedSession.subscribeRound(roundId, ingest);
      return;
    }
    if (source) source.close();
    source = new EventSource(`/api/game/rounds/${roundId}/stream`);
    const subscribedRoundId = roundId;
    const subscribedSource = source;
    const handle = (event) => {
      if (subscribedRoundId !== roundId) return;
      try {
        ingest(JSON.parse(event.data));
      } catch (_) { /* malformed frame: the next snapshot re-aligns state */ }
    };
    source.addEventListener('snapshot', (event) => {
      if (typeof sharedSession !== 'undefined' && sharedSession?.shared) firstSharedSnapshot = true;
      handle(event);
      if (typeof sharedSession !== 'undefined' && sharedSession?.shared) {
        const id = roundId;
        requestJson(`/api/game/rounds/${id}/speech/current`).then((audio) => {
          if (roundId === id && !closed && audio.directive) {
            lastAudioDirective = audio.directive.directive_id;
            handlers.onSpeech?.(audio.directive);
          }
        }).catch(console.error);
      }
    });
    source.addEventListener('update', handle);
    source.addEventListener('complete', (event) => {
      handle(event);
      subscribedSource.close();
    });
    source.onerror = () => { /* EventSource retries; server resends a snapshot */ };
  }

  async function start() {
    if (typeof sharedSession !== 'undefined' && sharedSession?.shared) return attach(sharedSession.snapshot.round_id);
    const payload = await requestJson('/api/game/rounds', {
      method: 'POST',
      body: JSON.stringify({ game: gameId }),
    });
    teardownStream();
    roundId = payload.round_id;
    closed = false;
    lastSequence = 0;
    subscribe();
    return payload;
  }

  async function attach(id) {
    teardownStream();
    roundId = id;
    closed = false;
    lastSequence = 0;
    firstSharedSnapshot = true;
    currentSnapshot = await requestJson(`/api/game/rounds/${id}`);
    ingest(currentSnapshot);
    if (closed) return currentSnapshot;
    subscribe();
    if (typeof sharedSession !== 'undefined' && sharedSession?.shared) {
      requestJson(`/api/game/rounds/${id}/speech/current`).then((audio) => {
        if (roundId === id && !closed && audio.directive && audio.directive.directive_id !== lastAudioDirective) {
          lastAudioDirective = audio.directive.directive_id;
          handlers.onSpeech?.(audio.directive);
        }
      }).catch(console.error);
    }
    return currentSnapshot;
  }

  async function submitIntent(intent, payload = {}) {
    if (typeof sharedSession !== 'undefined' && sharedSession?.shared) {
      if (intent === 'speech_started' || intent === 'speech_done') return currentSnapshot;
      const result = await sharedSession.action('intent', { round_id: roundId,
        state: currentSnapshot?.state, state_version: currentSnapshot?.state_version,
        intent, payload });
      if (result.round && result.round_id === roundId) ingest(result.round);
      return result.round;
    }
    if (!roundId) throw new Error('对局尚未创建');
    const targetRoundId = roundId;
    const snapshot = await requestJson(`/api/game/rounds/${targetRoundId}/intents`, {
      method: 'POST',
      body: JSON.stringify({ intent, ...payload }),
    });
    // Ingest the response so terminal transitions (e.g. an exit intent)
    // navigate even when the SSE stream is broken; sequence dedup keeps
    // this idempotent with the stream.
    if (targetRoundId === roundId && snapshot && typeof snapshot === 'object' && 'status' in snapshot) {
      ingest(snapshot);
    }
    return snapshot;
  }

  async function cancel() {
    if (typeof sharedSession !== 'undefined' && sharedSession?.shared) { roundId = null; closed = true; teardownStream(); return; }
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
    detachSessionRound?.();
    detachSessionRound = null;
    if (source) {
      source.close();
      source = null;
    }
  }

  return {
    start,
    attach,
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
    if (errorCode === 'ROUND_INTENT_REJECTED' || errorCode === 'ROUND_CLOSED' || errorCode === 'SESSION_STALE' || (url === '/api/session' && response.status === 404)) {
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
  if (typeof sharedSession !== 'undefined' && sharedSession?.shared) {
    if (!sessionApplying && sharedSession.snapshot?.round_id) sessionAction('exit');
    return;
  }
  boardSelectionEntry = null; // A pending confirmation must not revive a departed screen.
  setIdleReturn(false); // 离开对局即解除空闲退出（回列表后由待机计时接管）
  if (activeGame && activeGame.teardown) activeGame.teardown();
  activeGame = null;
  stopStandbyListening(); // 对局结束回列表：确保没有残留的待机监听/轮询
  setPhase('select');
  selectDefaultGame();
  startSelectListening(); // 列表页恢复语音选游戏
}

// ---- 游戏列表 ----
function renderGameList() {
  const list = $('gameList');
  list.innerHTML = '';
  // 游戏启停宏控（backend/config.json 的 games_enabled）：关闭的游戏不渲染
  // 卡片——不灰显、彻底不显示（用户 2026-09-29 拍板）。
  games.filter((game) => game.enabled).forEach((game) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.game = game.id;
    button.className = 'game-option';
    button.setAttribute('role', 'option');
    button.setAttribute('aria-selected', 'false');

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
    if (game.id === 'rps') {
      const category = document.createElement('span');
      category.className = 'arena-only game-category';
      category.textContent = '加赛体验';
      copy.append(category);
    }
    copy.append(name, desc);

    button.append(icon, copy);

    const arrow = document.createElement('span');
    arrow.className = 'game-arrow';
    arrow.setAttribute('aria-hidden', 'true');
    arrow.textContent = '→';
    button.append(arrow);
    button.addEventListener('click', () => selectGame(game.id));

    list.append(button);
  });
}

// ---- 板端大厅选中项同步 ----
let boardSelectionSync = Promise.resolve(true);
let boardSelectionVersion = 0;
let boardSelectionGame = null;
let boardSelectionPending = false;
let boardSelectionEntry = null;

function syncBoardSelection(id) {
  if (typeof sharedSession !== 'undefined' && sharedSession?.shared) return Promise.resolve(true);
  if (!boardInputSource.local_client || !games.some((game) => game.id === id && game.enabled)) {
    return Promise.resolve(true);
  }
  if (boardSelectionGame === id && boardSelectionPending) return boardSelectionSync;
  boardSelectionGame = id;
  boardSelectionPending = true;
  const version = ++boardSelectionVersion;
  // One in-flight request prevents an older reply from overwriting a newer choice.
  // Keep failures in-band so a subsequent selection/confirmation can retry.
  boardSelectionSync = boardSelectionSync.then(async () => {
    try {
      const result = await requestJson('/api/input/board/selection', {
        method: 'POST', body: JSON.stringify({ game: id }),
      });
      if (result.game !== id) throw new Error('board selection was not acknowledged');
      return true;
    } catch (error) {
      console.warn('Board selection sync failed:', error);
      return false;
    }
  }).then((ok) => {
    if (version === boardSelectionVersion) boardSelectionPending = false;
    return ok;
  });
  return boardSelectionSync;
}

function selectDefaultGame() {
  if (typeof sharedSession !== 'undefined' && sharedSession?.shared) return;
  // The exhibition's main challenge is dice. Respect game enablement and
  // fall back to the first available game; selecting never starts a round.
  const preferred = games.find((game) => game.id === 'dice' && game.enabled)
    || games.find((game) => game.enabled);
  selectGame(preferred?.id || null);
}

function selectGame(id, fromSession = false) {
  if (typeof sharedSession !== 'undefined' && sharedSession?.shared && !fromSession) return sessionAction('select', { game: id });
  state.selectedGame = id;
  document.querySelectorAll('.game-option').forEach((item) => {
    const selected = item.dataset.game === id;
    item.classList.toggle('selected', selected);
    item.setAttribute('aria-selected', selected ? 'true' : 'false');
  });
  window.ArenaTheme.renderPhase(state.phase, activeGame?.id, id);
  syncBoardSelection(id);
}

function enterSelectedGame() {
  if (typeof sharedSession !== 'undefined' && sharedSession?.shared) return sessionAction('enter');
  // Reserve the entry before waiting: DOM Enter and the board SSE callback may
  // arrive together while selection synchronization is still in flight.
  if (state.phase !== 'select') return;
  if (boardSelectionEntry !== null) return;
  if (!games.some((game) => game.id === state.selectedGame && game.enabled)) return;
  const attempt = {};
  boardSelectionEntry = attempt;

  async function enter() {
    const cached = games.find((game) => game.id === state.selectedGame && game.enabled);
    if (!cached) return;
    const module = gameModules[cached.id];
    if (!module) {
      toast(`未注册游戏：${cached.id}`);
      return;
    }
    const manifest = await refreshDisplayManifest(cached, requestJson);
    if (boardSelectionEntry !== attempt || state.phase !== 'select'
        || state.selectedGame !== manifest.id) return;
    stopSelectListening();
    activeGame = module;
    await module.enter(manifest);
  }

  // Remote pages never write the board cursor or wait for its synchronization.
  if (!boardInputSource.local_client) {
    return enter().finally(() => {
      if (boardSelectionEntry === attempt) boardSelectionEntry = null;
    });
  }
  syncBoardSelection(state.selectedGame); // Retries a previously failed sync.
  return (async () => {
    try {
      while (boardSelectionEntry === attempt && state.phase === 'select') {
        const version = boardSelectionVersion;
        const ok = await boardSelectionSync;
        if (boardSelectionEntry !== attempt || state.phase !== 'select') return;
        if (version !== boardSelectionVersion) continue; // User changed the highlight.
        if (!ok) {
          toast('选中游戏同步失败，请再次确认');
          return;
        }
        await enter();
        return;
      }
    } finally {
      if (boardSelectionEntry === attempt) boardSelectionEntry = null;
    }
  })();
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
  // 游戏模块声明"当前状态在等玩家操作"；计时与超时动作留在引擎层
  setIdleReturn,
};

registerGame(registerDice(engine));
registerGame(registerRps(engine));
bindBuddyToLiveVideo();

$('startGame').addEventListener('click', enterSelectedGame);
$('gameList').addEventListener('dblclick', enterSelectedGame);
$('soundToggle').addEventListener('click', () => {
  if (sharedSession?.shared && boardInputSource.local_client) { toast('板端语音由默认扬声器播放'); return; }
  state.sound = !state.sound;
  if (!state.sound) stopSpeech();
  $('soundToggle').textContent = state.sound ? '🔊' : '🔇';
  toast(state.sound ? 'TTS 播报已开启' : '语音播报已关闭');
  if (state.sound && sharedSession?.shared) { getSpeechAudioContext(); resumeSharedAudio().catch(console.error); }
});

document.addEventListener('keydown', (event) => {
  const controllerKey = ['Enter', 'Escape', 'ArrowDown', 'ArrowUp'].includes(event.key);
  if (controllerKey) event.preventDefault();
  if (event.repeat) return;
  if (typeof sharedSession !== 'undefined' && sharedSession?.shared && state.phase === 'select' && ['ArrowUp', 'ArrowDown'].includes(event.key)) {
    sessionAction('navigate', { direction: event.key === 'ArrowDown' ? 'down' : 'up' });
    return;
  }

  if (state.phase === 'select') {
    if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
      // The same physical arrow also arrives through board_navigate. Let that
      // existing channel handle local navigation once; retain web-key fallback.
      if (boardInputSource.local_client && boardInputSource.available && selectSource?.readyState === 1) return;
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
    boardInputSource = {
      local_client: payload.board_input?.local_client === true,
      enabled: payload.board_input?.enabled === true,
      available: payload.board_input?.available === true,
    };
    applyStandbySettings(payload.standby);
    renderGameList();
    sharedSession = createSessionClient({ requestJson, onSnapshot: applySessionSnapshot });
    try {
      await sharedSession.connect();
    } catch (error) {
      // An older running backend can serve these static files during rollout.
      // Keep its established controls until the service is deliberately switched.
      if (error.code !== 'UNKNOWN_ERROR') throw error;
      sharedSession = null;
    }
    if (!sharedSession?.shared) selectDefaultGame();
  } catch (error) {
    console.error('Failed to load game list:', error);
    toast('游戏列表加载失败，请检查后端 /api/games');
  }
}

setPhase('select');
loadGames().then(() => {
  if (sharedSession?.shared) return;
  // boot_standby=true：页面加载即待机（设备就绪的舞台状态）；false 停在列表
  // 并直接开语音选游戏监听。
  if (standbySettings.boot_standby) enterStandby();
  else startSelectListening();
});

// 测试钩子：视图切换的自动化验证入口（CDP/无头浏览器直调）。
// 仅用于真实页面环境里驱动 setPhase 做叠层回归测试，业务逻辑勿绕过它。
window.__setPhase = setPhase;
