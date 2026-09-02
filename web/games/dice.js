// 摇骰子游戏模块：后端权威状态机的声明式前端。
// 后端通过 round 事件流驱动一切：state_changed 切视图、speech 播台词、
// tick 渲染倒计时、adjudication 透传渲染分析进度；本模块只提交意图
// （实体按键/页面按钮）并渲染，不自行推进游戏状态。
export function register(engine) {
  const {
    state, $, setPhase, toast, stopSpeech, requestJson,
    returnToSelect, createRoundClient, playDirective,
  } = engine;

  const dicePips = {
    1: [4], 2: [0, 8], 3: [0, 4, 8], 4: [0, 2, 6, 8],
    5: [0, 2, 4, 6, 8], 6: [0, 2, 3, 5, 6, 8],
  };

  let playerDice = [];
  let agentDice = [];
  let countdownAudioContext = null;
  let visionStreamToken = 0;
  let participantSides = null;
  let round = null;
  let lastRenderedState = '';

  // 后端 ui 文案缺省时的前端兜底（正常路径都来自 manifest state_machine.ui）。
  const phaseMeta = {
    rules: ['游戏规则', '听完规则后按 Enter 确认，按 ↓ 可以再听一次。'],
    ready: ['准备好了吗？', '人手操作模式已开启，拿起骰盅后点击开始。'],
    countdown: ['同步倒计时', ''],
    shaking: ['摇骰进行中', '双方同时摇骰，准备好后可提前停止。'],
    open: ['你准备好了吗？听语音倒计时同时开盖', ''],
    analysis: ['正在判定胜负', '视觉裁决器正在识别骰子点数，随后由大模型复核。'],
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
    $('playerScoreSide').style.gridColumn = player === 'LEFT' ? '1' : '3';
    $('agentScoreSide').style.gridColumn = agent === 'LEFT' ? '1' : '3';
  }

  // ---- 实时画面（MediaMTX WebRTC iframe，保持原有边界） ----
  function stopVisionStream() {
    visionStreamToken += 1;
    const panel = $('analysisStreamPanel');
    const frame = $('analysisStream');
    if (!panel || !frame) return;
    frame.onload = null;
    frame.onerror = null;
    frame.src = 'about:blank';
    panel.classList.add('hidden');
    const status = $('analysisStreamState');
    if (status) status.textContent = '实时画面已关闭';
  }

  function startVisionStream(event) {
    const panel = $('analysisStreamPanel');
    const frame = $('analysisStream');
    if (!panel || !frame) return;
    const configuredUrl = event && typeof event.url === 'string' ? event.url.trim() : '';
    if (!configuredUrl) return;

    const token = ++visionStreamToken;
    let streamUrl;
    try {
      streamUrl = new URL(configuredUrl, window.location.href);
      if (!['http:', 'https:'].includes(streamUrl.protocol)) return;
    } catch (_) {
      return;
    }
    // The MediaMTX WebRTC page reads these options and starts muted playback,
    // which is allowed when the analysis page opens without a user gesture.
    streamUrl.searchParams.set('autoplay', '1');
    streamUrl.searchParams.set('muted', '1');
    streamUrl.searchParams.set('controls', '0');
    streamUrl.searchParams.set('playsinline', '1');

    panel.classList.remove('hidden');
    const status = $('analysisStreamState');
    if (status) status.textContent = '正在连接实时画面…';
    frame.onload = () => {
      if (token !== visionStreamToken || state.phase !== 'analysis') return;
      if (status) status.textContent = '播放页面已加载，等待 YOLO 画面…';
    };
    frame.onerror = () => {
      if (token !== visionStreamToken || state.phase !== 'analysis') return;
      if (status) status.textContent = '实时画面连接失败，识别仍会继续';
    };
    frame.src = streamUrl.toString();
  }

  // ---- 结果与比分渲染 ----
  function diceMarkup(values, className = '') {
    return values.map((value) => `<div class="die ${className}" aria-label="${value}点">${Array.from({ length: 9 }, (_, i) => `<span class="${dicePips[value].includes(i) ? 'on' : ''}"></span>`).join('')}</div>`).join('');
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
    const view = ui.view || stateName;
    const meta = [ui.title || '', ui.copy || ''];
    setPhase(view, meta[0] || meta[1] ? meta : undefined);
    if (stateName === 'analysis') {
      resetAnalysisSteps();
    } else if (stateName === 'ready' || stateName === 'rules') {
      playerDice = [];
      agentDice = [];
      updateScores();
    }
  }

  function renderTick(event) {
    const remaining = Number(event.remaining_ms);
    if (!Number.isFinite(remaining) || remaining < 0) return;
    // Ceil, not floor: the first tick fires when the full budget is still
    // on the clock (remaining_ms just under 3000), and the player must see
    // 3 → 2 → 1 exactly like the previous synchronous countdown did.
    if (lastRenderedState === 'shake_countdown' || lastRenderedState === 'vision_countdown') {
      $('countdownNumber').textContent = Math.max(1, Math.ceil(remaining / 1000));
    } else if (lastRenderedState === 'shaking') {
      const seconds = Math.max(1, Math.ceil(remaining / 1000));
      const shakeSeconds = $('shakeSeconds');
      const urgent = seconds <= 3;
      shakeSeconds.textContent = String(seconds).padStart(2, '0');
      shakeSeconds.classList.toggle('is-urgent', urgent);
      if (urgent) playCountdownCue(seconds);
    }
  }

  function resetAnalysisSteps() {
    stopVisionStream();
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
      $('analysisStatus').textContent = count > 0 && required > 0
        ? `YOLOv8 正在检测双方各 5 颗骰子，并等待稳定帧（${count}/${required}）…`
        : 'YOLOv8 正在检测双方各 5 颗骰子，并等待稳定帧…';
    } else if (event.phase === 'verifying') {
      $('stepDetect').classList.add('active');
      $('stepDetect').querySelector('span').textContent = '✓';
      $('stepJudge').classList.add('active');
      $('stepJudge').querySelector('span').textContent = '…';
      $('analysisStatus').textContent = 'YOLOv8 结果已稳定，正在调用大模型复核…';
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
    stopVisionStream();
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
    const verificationText = result.source === 'yolo_timeout_fallback'
      ? '大模型超时，采用 YOLOv8'
      : result.source === 'yolo_failure_fallback'
        ? '大模型请求失败，采用 YOLOv8'
        : result.source === 'llm_override'
          ? '大模型结果不一致，以大模型结果为准'
          : result.source === 'yolo_only'
            ? '当前未启用大模型'
            : '大模型复核一致';
    $('resultSubtitle').textContent = `YOLOv8：玩家 ${player} : Agent ${agent}；${verificationText}`;
    banner.classList.toggle('loss', !playerWins && !tie);
  }

  function showDiagnosis(result) {
    stopVisionStream();
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
  }

  function handleRoundEvent(event, snapshot) {
    if (event.event === 'video') {
      startVisionStream(event);
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
      if (error.silent) return; // 按键时机不合状态属正常对局
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
    stopShake: () => submitIntent('stop_shake'),
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
  };

  function onKey(event) {
    if (event.key === 'Escape') {
      if (['rules', 'ready', 'result'].includes(state.phase)) submitIntent('back');
      else if (state.phase === 'analysis' && analysisFailureVisible()) submitIntent('back');
      return;
    }
    if (event.key === 'Enter') {
      if (state.phase === 'rules') submitIntent('confirm');
      else if (state.phase === 'ready') handlers.startShake();
      return;
    }
    if (state.phase === 'rules' && event.key === 'ArrowDown') {
      handlers.repeatRules();
      return;
    }
    if (state.phase === 'analysis' && event.key === 'ArrowDown') {
      if (analysisFailureVisible()) submitIntent('retry');
      return;
    }
    if (event.key === 'ArrowUp') {
      return;
    }
  }

  // ---- 对局生命周期 ----
  async function enter(manifest) {
    configureParticipants(manifest);
    stopVisionStream();
    playerDice = [];
    agentDice = [];
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
        if (snapshot.state && snapshot.state !== lastRenderedState) {
          renderState(snapshot.state, {});
          if (snapshot.state === 'result' && snapshot.result) showResult(snapshot.result);
          else if (snapshot.state === 'analysis_failed' && snapshot.result) showDiagnosis(snapshot.result);
        }
      },
    });

    try {
      await round.start();
    } catch (error) {
      console.error('Failed to start round:', error);
      toast('对局创建失败，请检查后端服务');
      returnToSelect();
    }
  }

  function teardown() {
    stopVisionStream();
    participantSides = null;
    Object.entries(handlers).forEach(([id, fn]) => $(id).removeEventListener('click', fn));
    stopSpeech();
    if (round) {
      round.cancel();
      round = null;
    }
    lastRenderedState = '';
  }

  return {
    id: 'dice',
    phases: ['select', 'rules', 'ready', 'countdown', 'shaking', 'open', 'analysis', 'result'],
    progressCount: 6,
    phaseMeta,
    enter,
    teardown,
    onKey,
  };
}
