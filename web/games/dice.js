import { resolveDisplayLayout, applyScoreLayout } from '../display-layout.js';

// 摇骰子游戏模块：后端权威状态机的声明式前端。
// 后端通过 round 事件流驱动一切：state_changed 切视图、speech 播台词、
// tick 渲染倒计时、adjudication 透传渲染分析进度；本模块只提交意图
// （实体按键/页面按钮）并渲染，不自行推进游戏状态。
export function register(engine) {
  const {
    state, $, setPhase, toast, stopSpeech, requestJson,
    returnToSelect, createRoundClient, playDirective, setActiveRound, setIdleReturn,
  } = engine;

  const dicePips = {
    1: [4], 2: [0, 8], 3: [0, 4, 8], 4: [0, 2, 6, 8],
    5: [0, 2, 4, 6, 8], 6: [0, 2, 3, 5, 6, 8],
  };

  let playerDice = [];
  let agentDice = [];
  let countdownAudioContext = null;
  let participantSides = null;
  let displayLayout = null;
  let activeVisionUrl = '';
  let round = null;
  let lastRenderedState = '';
  let lastCountdownValue = '';
  let lastRobotFailureReason = '';

  // 后端 ui 文案缺省时的前端兜底（正常路径都来自 manifest state_machine.ui）。
  const phaseMeta = {
    rules: ['先听三步小秘诀', '听完后按绿色按钮出发；蓝色按钮可以再听一次，红色按钮返回。'],
    ready: ['握好骰盅，好运要出发啦！', '按下绿色按钮，你和机械臂就会一起开始摇骰。'],
    rehome: ['机械臂回起点啦', '稍等一下，小搭子正在摆好姿势，下一局马上就绪。'],
    game_start: ['抓稳骰盅，挑战开始！', '机械臂正在拿好骰盅，准备和你同步开摇。'],
    countdown: ['一起倒数，准备摇！', '机械臂已经就位，倒计时结束后双方同时开始。'],
    shaking: ['摇起来，把好运叫醒！', '机械臂正在摇骰，你也要持续摇动骰盅哦。'],
    stop_call: ['摇好了！', '停！请保持骰子和骰盅位置不动，马上开始判定。'],
    arm_failed: ['机械臂刚刚卡了一下', '蓝色按钮让它再试一次；红色按钮退出本局。'],
    analysis: ['点数侦探正在认真数', '请让骰子保持不动，数完点数马上揭晓胜负。'],
    result: ['本局结果', ''],
  };

  function sum(dice) { return dice.reduce((a, b) => a + b, 0); }

  function configureParticipants(manifest) {
    const participants = manifest && manifest.participants;
    const player = participants && participants.player;
    const agent = participants && participants.agent;
    if (!['LEFT', 'RIGHT'].includes(player)
        || !['LEFT', 'RIGHT'].includes(agent)
        || player === agent) {
      throw new Error('游戏参与者左右位置配置无效');
    }
    participantSides = { player, agent };
    displayLayout = resolveDisplayLayout(manifest);
    applyScoreLayout($('playerScoreSide'), $('agentScoreSide'), displayLayout);
  }

  // ---- 实时画面（MediaMTX WebRTC iframe，保持原有边界） ----
  function stopVisionStream() {
    const panel = $('analysisStreamPanel');
    const frame = $('analysisStream');
    if (!panel || !frame) return;
    activeVisionUrl = '';
    frame.src = 'about:blank';
    panel.classList.add('hidden');
  }

  function startVisionStream(event) {
    const panel = $('analysisStreamPanel');
    const frame = $('analysisStream');
    if (!panel || !frame) return;
    const configuredUrl = event && typeof event.url === 'string' ? event.url.trim() : '';
    if (!configuredUrl) return;
    let streamUrl;
    try {
      streamUrl = new URL(configuredUrl, window.location.href);
      if (!['http:', 'https:'].includes(streamUrl.protocol)) return;
    } catch (_) {
      return;
    }
    const normalizedUrl = streamUrl.toString();
    panel.classList.remove('hidden');
    // Reuse the connection across repeated video events and adjudication.
    if (activeVisionUrl === normalizedUrl && frame.src !== 'about:blank') return;
    activeVisionUrl = normalizedUrl;
    const playerUrl = new URL('./stream-player.html', window.location.href);
    playerUrl.searchParams.set('stream', normalizedUrl);
    playerUrl.searchParams.set('rotation', String(displayLayout?.video_rotation_deg || 0));
    frame.src = playerUrl.toString();
  }

  // ---- 结果与比分渲染 ----
  // 一面的九宫格点数：比分板与摇骰页六面共用这一份布局
  function faceSpans(value) {
    return Array.from({ length: 9 }, (_, i) => `<span class="${dicePips[value].includes(i) ? 'on' : ''}"></span>`).join('');
  }

  function diceMarkup(values, className = '') {
    return values.map((value) => `<div class="die ${className}" aria-label="${value}点">${faceSpans(value)}</div>`).join('');
  }

  // 摇骰页的主角骰：真六面立方体，翻滚与换面全靠 CSS（.die-cube 的 tumble），
  // 这里只负责把六面按"对面之和为 7"摆好。游戏模块不许用计时器——测试用子串
  // 匹配断言本文件里不出现那两个计时器 API 名，所以连注释里也别写它们。
  function diceCubeMarkup() {
    return `<div class="die-cube">${[1, 6, 2, 5, 3, 4]
      .map((value) => `<div class="die die-face die-face-${value}">${faceSpans(value)}</div>`).join('')}</div>`;
  }

  function updateScores(player = sum(playerDice), agent = sum(agentDice)) {
    $('playerScore').textContent = playerDice.length ? player : '—';
    $('agentScore').textContent = agentDice.length ? agent : '—';
    $('playerDice').innerHTML = diceMarkup(playerDice);
    $('agentDice').innerHTML = diceMarkup(agentDice, 'agent-die');
  }

  // ---- 倒计时音效（纯前端 UI 反馈） ----
  function prepareCountdownAudio() {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    if (!countdownAudioContext) {
      try {
        countdownAudioContext = new AudioContext();
      } catch (_) {
        return;
      }
    }
    if (countdownAudioContext.state === 'suspended') {
      countdownAudioContext.resume().catch(() => {});
    }
  }

  function playCountdownCue(seconds) {
    if (!state.sound || !countdownAudioContext || countdownAudioContext.state === 'closed') return;
    const context = countdownAudioContext;
    const now = context.currentTime;
    try {
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      const secondaryOscillator = context.createOscillator();
      const secondaryGain = context.createGain();
      const frequency = seconds === 1 ? 880 : seconds === 2 ? 740 : 620;
      const secondaryFrequency = seconds === 1 ? 1320 : seconds === 2 ? 1110 : 930;
      const volume = seconds === 1 ? 0.28 : 0.22;
      const secondaryStart = now + 0.12;
      oscillator.type = 'triangle';
      oscillator.frequency.setValueAtTime(frequency, now);
      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.exponentialRampToValueAtTime(volume, now + 0.015);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.24);
      oscillator.connect(gain);
      gain.connect(context.destination);
      oscillator.start(now);
      oscillator.stop(now + 0.28);

      secondaryOscillator.type = 'triangle';
      secondaryOscillator.frequency.setValueAtTime(secondaryFrequency, secondaryStart);
      secondaryGain.gain.setValueAtTime(0.0001, secondaryStart);
      secondaryGain.gain.exponentialRampToValueAtTime(volume * 0.85, secondaryStart + 0.015);
      secondaryGain.gain.exponentialRampToValueAtTime(0.0001, secondaryStart + 0.24);
      secondaryOscillator.connect(secondaryGain);
      secondaryGain.connect(context.destination);
      secondaryOscillator.start(secondaryStart);
      secondaryOscillator.stop(secondaryStart + 0.28);
    } catch (_) {
      // Audio feedback is optional; a browser audio limitation must not stop the game.
    }
  }

  // ---- 后端事件渲染 ----
  function renderState(stateName, ui) {
    if (stateName === lastRenderedState) return;
    lastRenderedState = stateName;
    // 进入新状态时清空倒计时去重值，让新一段倒计时的第一个数字也播放弹出动画。
    lastCountdownValue = '';
    const view = ui.view || stateName;
    const meta = [ui.title || '', ui.copy || ''];
    setPhase(view, meta[0] || meta[1] ? meta : undefined);
    if (view === 'ready') $('readyButtonHint').innerHTML = $('phaseCopy').innerHTML;
    // 等玩家操作的状态没有 duration/on_expire，玩家走开后会永远停住；交给引擎
    // 在这些状态挂空闲退出计时器（超时取消回合回列表），离开即解除。
    setIdleReturn(['rules', 'ready', 'result', 'analysis_failed', 'arm_failed'].includes(stateName));
    if (stateName === 'analysis') {
      resetAnalysisSteps();
    } else if (stateName === 'ready' || stateName === 'rules') {
      playerDice = [];
      agentDice = [];
      updateScores();
    }
    if (stateName === 'rehome') {
      enterRehomeView();
    } else if (stateName === 'game_start') {
      enterGameStartView();
    } else if (stateName === 'shake_countdown') {
      enterCountdownView();
    } else if (stateName === 'shaking' || stateName === 'stop_call') {
      // stop_call（摇完喊停）复用 shaking 视图，文案由后端 ui 驱动切换。
      enterShakingView();
    } else if (stateName === 'arm_failed') {
      enterArmFailedView();
    }
  }

  // ---- 机械臂进度 / 视图布局 ----
  function updateArmProgress(event) {
    // 抓取发生在过场页/倒计时页、摇骰发生在摇骰页：三行同款进度条各自更新。
    const rows = [
      { row: $('armRehomeRow'), text: $('armRehomeText'), step: $('armRehomeStep') },
      { row: $('armGameStartRow'), text: $('armGameStartText'), step: $('armGameStartStep') },
      { row: $('armCountdownRow'), text: $('armCountdownText'), step: $('armCountdownStep') },
      { row: $('armShakingRow'), text: $('armShakingText'), step: $('armShakingStep') },
    ];
    for (const { row, text, step } of rows) {
      if (!row || row.classList.contains('hidden')) continue;
      if (event.zh) text.textContent = event.zh;
      if (event.phase === 'AUTO_RETRY') {
        step.textContent = '';
      } else if (event.progress) {
        step.textContent = event.progress;
      }
    }
  }

  function resetArmProgressRows() {
    const rows = [
      { row: $('armRehomeRow'), text: $('armRehomeText'), step: $('armRehomeStep') },
      { row: $('armGameStartRow'), text: $('armGameStartText'), step: $('armGameStartStep') },
      { row: $('armCountdownRow'), text: $('armCountdownText'), step: $('armCountdownStep') },
      { row: $('armShakingRow'), text: $('armShakingText'), step: $('armShakingStep') },
    ];
    for (const { row, text, step } of rows) {
      if (!row) continue;
      text.textContent = '机械臂准备中';
      step.textContent = '';
    }
  }

  function enterRehomeView() {
    // 离开结果页的归位过场：臂进度行可见（显示"手势 home"）。
    const row = $('armRehomeRow');
    if (row) row.classList.remove('hidden');
  }

  function enterGameStartView() {
    // 开局过场：大字动画 + 臂进度行（抓取在此时进行）。
    const row = $('armGameStartRow');
    if (row) row.classList.remove('hidden');
  }

  function enterCountdownView() {
    // 三二一倒计时 + 抓取进度行（抓取偶发偏慢时仍可见）。
    const row = $('armCountdownRow');
    if (row) row.classList.remove('hidden');
  }

  function enterShakingView() {
    const title = $('shakingTitle');
    if (title) title.textContent = '摇骰进行中';
    const armRow = $('armShakingRow');
    if (armRow) armRow.classList.remove('hidden');
  }

  function friendlyRobotFailure(reason) {
    const text = String(reason || '');
    if (text.includes('Hand start')) {
      return '机械臂手指位置校验未通过（摇骰后的正常扰动），按蓝色按钮重试即可恢复。';
    }
    if (text.includes('interrupted') || text.includes('timed out')) {
      return '机械臂动作被中断或超时，可按蓝色按钮重试。';
    }
    return text ? `机械臂未能完成动作：${text}` : '机械臂未能完成动作，可按蓝色按钮重试。';
  }

  function enterArmFailedView() {
    // robot_result 观察事件可能已滑出事件窗，用记录的原因兜底。
    $('armFailedStatus').textContent = friendlyRobotFailure(lastRobotFailureReason);
  }


  // 弹出动画重放：纯 CSS 动画只在元素首次显示时播放，数字变化时摘掉 .pop、
  // 读一次 offsetWidth 触发重排、再加回来，动画即从头播放。
  function renderCountdownNumber(value) {
    const text = String(value);
    if (text === lastCountdownValue) return;
    lastCountdownValue = text;
    const node = $('countdownNumber');
    node.textContent = text;
    node.classList.remove('pop');
    void node.offsetWidth;
    node.classList.add('pop');
  }

  function renderTick(event) {
    const remaining = Number(event.remaining_ms);
    if (!Number.isFinite(remaining) || remaining < 0) return;
    if (lastRenderedState === 'shake_countdown') {
      // 每个数字显示 tick_seconds 秒（manifest 状态机的节奏字段），
      // 与预录语音"三二一"的语速对齐；缺省回退按整秒取整。
      const perNumber = Number(event.tick_seconds || 1) * 1000;
      const total = Math.max(1, Math.round(Number(event.duration_seconds || 3) / (event.tick_seconds || 1)));
      renderCountdownNumber(Math.min(total, Math.max(1, Math.ceil(remaining / perNumber))));
    }
  }

  function resetAnalysisSteps() {
    $('stepCapture').classList.add('active');
    $('stepCapture').classList.remove('failed');
    $('stepCapture').querySelector('span').textContent = '✓';
    $('stepDetect').classList.remove('active', 'failed');
    $('stepDetect').querySelector('span').textContent = '2';
    $('stepJudge').classList.remove('active', 'failed');
    $('stepJudge').querySelector('span').textContent = '3';
    $('analysisTitle').textContent = '正在识别骰子';
    $('analysisFailureActions').classList.add('hidden');
    document.querySelector('.analysis-spinner')?.classList.remove('hidden');
    $('analysisStatus').textContent = '正在请求 K3 YOLOv8 推理进程…';
  }

  function markAnalysisFailure() {
    $('stepDetect').classList.add('active', 'failed');
    $('stepDetect').querySelector('span').textContent = '✕';
    $('stepJudge').classList.remove('active');
    $('stepJudge').classList.remove('failed');
    $('stepJudge').querySelector('span').textContent = '3';
  }

  function diagnosisDetails(diagnosis) {
    const reasonCode = typeof diagnosis.reason_code === 'string'
      ? diagnosis.reason_code.trim() : '';
    const reasonLabels = {
      INCOMPLETE_OBJECTS: '检测数量不完整',
      OVERLAPPING_OBJECTS: '疑似骰子叠放',
      LOW_LIGHT: '光线可能不足',
      OCCLUDED: '目标可能被遮挡',
      NO_OBJECTS_DETECTED: '未检测到目标',
      UNSTABLE_DETECTION: '检测结果不稳定',
      SCENE_GEOMETRY_UNCLEAR: '左右区域不清晰',
      UNKNOWN: '无法确定具体原因',
    };
    const reason = reasonCode
      ? `原因：${reasonLabels[reasonCode] || reasonCode}（${reasonCode}）`
      : '';
    const counts = diagnosis.detected_counts;
    const countText = counts && typeof counts === 'object'
      ? Object.entries(counts)
        .filter(([, value]) => Number.isFinite(Number(value)))
        .map(([name, value]) => `${name}=${Number(value)}`)
        .join('、')
      : '';
    return [reason, countText ? `检测数量：${countText}` : ''].filter(Boolean);
  }

  function updateAnalysisProgress(event) {
    if (event.phase === 'detecting') {
      $('stepDetect').classList.add('active');
      $('stepDetect').querySelector('span').textContent = '…';
      const count = Number(event.stable_count || 0);
      const required = Number(event.stable_frames || 0);
      // Display guard only: the runtime resets the streak on frames it cannot
      // adjudicate, so count must never read above the threshold. Clamping
      // keeps an older runtime binary from showing "48/30".
      const shownCount = required > 0 ? Math.min(count, required) : count;
      $('analysisStatus').textContent = shownCount > 0 && required > 0
        ? `YOLOv8 正在检测双方各 5 颗骰子，并等待稳定帧（${shownCount}/${required}）…`
        : 'YOLOv8 正在检测双方各 5 颗骰子，并等待稳定帧…';
    } else if (event.phase === 'holding') {
      const remaining = Number(event.remaining_ms);
      $('stepDetect').classList.add('active');
      $('stepDetect').querySelector('span').textContent = '✓';
      $('stepJudge').classList.add('active');
      $('stepJudge').querySelector('span').textContent = '✓';
      $('analysisStatus').textContent = Number.isFinite(remaining) && remaining > 0
        ? `结果已锁定，实时画面将继续播放 ${Math.ceil(remaining / 1000)} 秒…`
        : '结果已锁定，实时画面仍在播放…';
    }
  }

  function assertResultParticipants(result) {
    if (!participantSides
        || result.player_side !== participantSides.player
        || result.agent_side !== participantSides.agent) {
      throw new Error('裁决结果与游戏参与者位置配置不一致');
    }
  }

  function showResult(result) {
    assertResultParticipants(result);
    playerDice = Array.isArray(result.player_values) ? result.player_values : [];
    agentDice = Array.isArray(result.agent_values) ? result.agent_values : [];
    const player = Number(result.player_score);
    const agent = Number(result.agent_score);
    if (!Number.isFinite(player) || !Number.isFinite(agent)) {
      throw new Error('裁决结果缺少有效的玩家或 Agent 分数');
    }
    updateScores(player, agent);
    const banner = $('resultBanner');
    const winnerRole = result.winner_role;
    const tie = winnerRole === 'TIE';
    const playerWins = winnerRole === 'PLAYER';
    if (!tie && !playerWins && winnerRole !== 'AGENT') {
      throw new Error('裁决结果缺少有效 winner_role');
    }
    $('resultEmoji').textContent = tie ? '🤝' : playerWins ? '🏆' : '✨';
    $('resultTitle').textContent = tie ? '平局！' : playerWins ? '玩家获胜' : 'Agent 获胜';
    $('resultSubtitle').textContent = `YOLOv8：玩家 ${player} : Agent ${agent}（纯视觉判定）`;
    banner.classList.toggle('loss', !playerWins && !tie);
  }

  function showDiagnosis(result) {
    const diagnosis = result && result.diagnosis && typeof result.diagnosis === 'object'
      ? result.diagnosis : {};
    markAnalysisFailure();
    document.querySelector('.analysis-spinner')?.classList.add('hidden');
    $('analysisTitle').textContent = '本次裁决未完成';
    const details = diagnosisDetails(diagnosis);
    $('analysisStatus').textContent = [
      diagnosis.message
        || '当前画面无法形成稳定检测结果，请检查摆放和光线后重新开始。',
      ...details,
    ].join(' ');
    $('analysisFailureActions').classList.remove('hidden');
    $('analysisRetry').classList.remove('hidden');
  }

  function handleRoundEvent(event, snapshot) {
    if (event.event === 'video') {
      startVisionStream(event);
    } else if (event.event === 'robot') {
      // 机械臂阶段进度（provider 的 phase_started 直译：zh + n/10）。
      updateArmProgress(event);
    } else if (event.event === 'robot_result') {
      // 完成结果只用于失败页文案；路由本身由后端状态机完成。
      if (typeof event.route === 'string' && event.route.endsWith('.failed')) {
        lastRobotFailureReason = event.reason || '';
      }
    } else if (event.event === 'phase' || event.event === 'progress') {
      updateAnalysisProgress(event);
    } else if (event.event === 'result' && snapshot && snapshot.result) {
      // Physical result relayed by the provider; role-projected rendering
      // happens on the result state below with the authoritative snapshot.
      updateAnalysisProgress({ phase: 'holding' });
    } else if (event.event === 'diagnosis' && snapshot && snapshot.result) {
      showDiagnosis(snapshot.result);
    }
  }

  // ---- 意图提交 ----
  function submitIntent(intent, payload = {}) {
    if (!round) return Promise.resolve();
    return round.submitIntent(intent, payload).catch((error) => {
      if (error.silent) {
        // 回合已终结（ROUND_CLOSED，如视觉异常结束）后一切意图都被静默
        // 拒绝——错误页"按钮能按却毫无反应"假死的根因。此时任何按键都
        // 导航回列表，让玩家能继续。
        if (error.code === 'ROUND_CLOSED') returnToSelect();
        return; // 按键时机不合状态属正常对局
      }
      console.error(`Intent ${intent} failed:`, error);
    });
  }

  function analysisFailureVisible() {
    const actions = $('analysisFailureActions');
    return actions && !actions.classList.contains('hidden');
  }

  const handlers = {
    startShake: () => {
      prepareCountdownAudio();
      submitIntent('start_shake');
    },
    readyBack: () => submitIntent('back'),
    confirmRules: () => submitIntent('confirm'),
    repeatRules: () => {
      toast('正在重复播报游戏规则');
      submitIntent('repeat');
    },
    backFromRules: () => submitIntent('back'),
    analysisRetry: () => submitIntent('retry'),
    analysisNewRound: () => submitIntent('new_round'),
    analysisBackToGames: () => submitIntent('back'),
    newRound: () => submitIntent('new_round'),
    backToGames: () => submitIntent('back'),
    armRetry: () => submitIntent('retry'),
    armBack: () => submitIntent('back'),
  };

  function onKey(event) {
    if (event.key === 'Escape') {
      // 红键/Esc 与屏幕上的红色按钮同义。摇骰中（shaking/stop_call）无动作
      // ——机械臂摇物理上不应被打断，等臂完成事件自然推进（2026-09-24
      // 人工摇骰模式移除后，摇骰环节只剩机械臂一条路）。
      if (['rules', 'ready', 'result', 'arm_failed'].includes(state.phase)) submitIntent('back');
      else if (state.phase === 'analysis' && analysisFailureVisible()) submitIntent('back');
      return;
    }
    if (event.key === 'Enter') {
      // 绿键/Enter 与屏幕上的绿色按钮同义：规则确认、开始摇骰、结果页再来
      // 一局、识别失败页再来一局。
      if (state.phase === 'rules') submitIntent('confirm');
      else if (state.phase === 'ready') handlers.startShake();
      else if (state.phase === 'result') handlers.newRound();
      else if (state.phase === 'analysis' && analysisFailureVisible()) handlers.analysisNewRound();
      return;
    }
    if (event.key === 'ArrowDown') {
      if (state.phase === 'rules') handlers.repeatRules();
      else if (state.phase === 'analysis' && analysisFailureVisible()) submitIntent('retry');
      else if (state.phase === 'arm_failed') handlers.armRetry();
      return;
    }
  }

  // ---- 对局生命周期 ----
  async function enter(manifest) {
    $('analysisStreamPanel').classList.add('dice-video');
    configureParticipants(manifest);
    stopVisionStream();
    // 摇骰页放一颗会翻滚的六面骰（diceCubeMarkup），观众看到的是在翻的骰子，
    // 不是固定的某一面；翻滚过程不参与判定。
    $('shakeCup').innerHTML = `<div class="shake-bounce">${diceCubeMarkup()}</div>`;
    playerDice = [];
    agentDice = [];
    lastRobotFailureReason = '';
    resetArmProgressRows();
    $('analysisFailureActions').classList.add('hidden');
    document.querySelector('.analysis-spinner')?.classList.remove('hidden');
    updateScores();
    Object.entries(handlers).forEach(([id, fn]) => $(id).addEventListener('click', fn));
    lastRenderedState = '';
    setPhase('rules');

    round = createRoundClient(manifest.id, {
      onStateChange: (stateName, ui, snapshot) => {
        renderState(stateName, ui);
        if (stateName === 'result' && snapshot && snapshot.result) {
          showResult(snapshot.result);
        } else if (stateName === 'analysis_failed' && snapshot && snapshot.result) {
          showDiagnosis(snapshot.result);
          toast('视觉裁决未完成，可按蓝色按钮重新识别');
        }
      },
      onSpeech: (directive) => {
        playDirective(round, directive);
      },
      onTick: renderTick,
      onEvent: handleRoundEvent,
      onComplete: (event) => {
        if (event.status === 'error') {
          stopVisionStream();
          markAnalysisFailure();
          document.querySelector('.analysis-spinner')?.classList.add('hidden');
          $('analysisTitle').textContent = '识别未完成';
          $('analysisStatus').textContent = '视觉裁决异常结束，请重新开始一局';
          $('analysisFailureActions').classList.remove('hidden');
          $('analysisRetry').classList.remove('hidden');
          // 回合已终结但界面停在失败页等玩家操作，同样需要空闲退出兜底。
          setIdleReturn(true);
          toast('K3 视觉裁决失败');
          return;
        }
        // exited / cancelled: the player left, follow them back to the list.
        returnToSelect();
      },
      onSyncState: (snapshot) => {
        // The round is gone after teardown (returnToSelect/cancel); a stale
        // snapshot must never re-render a finished game view.
        if (!round || !round.roundId) return;
        // 注意此处传空的 ui：renderState 会退回用状态名当视图名。状态名与
        // 视图名不同的（shake_countdown → countdown、stop_call → shaking、
        // analysis_failed → analysis）需要显式映射，否则会隐藏全部视图。
        const viewOverrides = {
          stop_call: { view: 'shaking' },
          arm_failed: { view: 'arm_failed' },
        };
        if (snapshot.state && snapshot.state !== lastRenderedState) {
          renderState(snapshot.state, viewOverrides[snapshot.state] || {});
          if (snapshot.state === 'result' && snapshot.result) showResult(snapshot.result);
          else if (snapshot.state === 'analysis_failed' && snapshot.result) showDiagnosis(snapshot.result);
        }
      },
    });

    try {
      await round.start();
      setActiveRound(round); // 登记给 pagehide 保险：离开页面即取消对局
    } catch (error) {
      console.error('Failed to start round:', error);
      toast('对局创建失败，请检查后端服务');
      returnToSelect();
    }
  }

  function teardown() {
    $('analysisStreamPanel').classList.remove('dice-video');
    stopVisionStream();
    participantSides = null;
    Object.entries(handlers).forEach(([id, fn]) => $(id).removeEventListener('click', fn));
    stopSpeech();
    setActiveRound(null);
    if (round) {
      round.cancel();
      round = null;
    }
    lastRenderedState = '';
  }

  return {
    id: 'dice',
    phases: ['select', 'rules', 'ready', 'rehome', 'game_start', 'countdown', 'shaking', 'arm_failed', 'analysis', 'result'],
    progressCount: 6,
    phaseMeta,
    enter,
    teardown,
    onKey,
  };
}
