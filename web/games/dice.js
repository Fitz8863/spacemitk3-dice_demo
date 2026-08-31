// 摇骰子游戏模块：状态机、阶段文案、骰子渲染与胜负结果。
// 通过 engine 注入视图切换 / TTS / 网络能力，不含游戏列表与通用壳逻辑。
export function register(engine) {
  const { state, $, setPhase, speakState, requestJson, toast, stopSpeech, returnToSelect } = engine;

  const dicePips = {
    1: [4], 2: [0, 8], 3: [0, 4, 8], 4: [0, 2, 6, 8],
    5: [0, 2, 4, 6, 8], 6: [0, 2, 3, 5, 6, 8],
  };

  let round = 1;
  let playerDice = [];
  let agentDice = [];
  let shakeTimer = null;
  let countdownTimer = null;
  let revealTransitionTimer = null;
  let countdownAudioContext = null;
  let visionStreamToken = 0;
  let pendingAnalysisResult = null;
  let participantSides = null;

  const phases = ['select', 'rules', 'ready', 'countdown', 'shaking', 'open', 'analysis', 'result'];
  const SHAKE_DURATION_SECONDS = 10;
  const shakeCountdownMeta = ['同步倒计时', '与 Agent 保持同步，倒计时结束后开始摇骰。'];
  const visionCountdownMeta = ['准备视觉裁决', '请保持骰子和骰盅位置不动，倒计时结束后开始视觉裁决。'];
  const phaseMeta = {
    select: ['选择一场游戏', '欢迎来到 Dice Arena，选择游戏后按 OK 开始。'],
    rules: ['游戏规则', '听完规则后按 Enter 确认，按 ↓ 可以再听一次。'],
    ready: ['准备好了吗？', '人手操作模式已开启，拿起骰盅后点击开始。'],
    countdown: shakeCountdownMeta,
    shaking: ['摇骰进行中', '双方同时摇骰，准备好后可提前停止。'],
    open: ['同时开盖', '请同时打开骰盅，开盖过场结束后自动进入倒计时。'],
    analysis: ['正在判定胜负', '视觉裁决器正在识别骰子点数，随后由大模型复核。'],
    result: ['本局结果', '点数已经锁定，看看谁赢下了这一局。'],
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

  function assertResultParticipants(result) {
    if (!participantSides
        || result.player_side !== participantSides.player
        || result.agent_side !== participantSides.agent) {
      throw new Error('裁决结果与游戏参与者位置配置不一致');
    }
  }

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
      // iframe load only confirms that the MediaMTX player page loaded.
      // The embedded page still needs to negotiate WebRTC and receive a track.
      if (status) status.textContent = '播放页面已加载，等待 YOLO 画面…';
    };
    frame.onerror = () => {
      if (token !== visionStreamToken || state.phase !== 'analysis') return;
      if (status) status.textContent = '实时画面连接失败，识别仍会继续';
    };
    frame.src = streamUrl.toString();
  }

  function diceMarkup(values, className = '') {
    return values.map((value) => `<div class="die ${className}" aria-label="${value}点">${Array.from({ length: 9 }, (_, i) => `<span class="${dicePips[value].includes(i) ? 'on' : ''}"></span>`).join('')}</div>`).join('');
  }

  function updateScores(player = sum(playerDice), agent = sum(agentDice)) {
    $('playerScore').textContent = playerDice.length ? player : '—';
    $('agentScore').textContent = agentDice.length ? agent : '—';
    $('playerDice').innerHTML = diceMarkup(playerDice);
    $('agentDice').innerHTML = diceMarkup(agentDice, 'agent-die');
  }

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

  function updateShakeCountdown(seconds) {
    const shakeSeconds = $('shakeSeconds');
    const urgent = seconds <= 3 && seconds > 0;
    shakeSeconds.textContent = String(seconds).padStart(2, '0');
    shakeSeconds.classList.toggle('is-urgent', urgent);
    if (urgent) playCountdownCue(seconds);
  }

  function countdown(next, label, hint, meta = shakeCountdownMeta) {
    clearInterval(countdownTimer);
    let seconds = 3;
    $('countdownHint').textContent = hint;
    $('countdownNumber').textContent = seconds;
    phaseMeta.countdown = meta;
    setPhase('countdown');
    countdownTimer = setInterval(() => {
      seconds -= 1;
      $('countdownNumber').textContent = Math.max(0, seconds);
      if (seconds <= 0) {
        clearInterval(countdownTimer);
        next();
      }
    }, 900);
  }

  function beginShake() {
    setPhase('shaking');
    let seconds = SHAKE_DURATION_SECONDS;
    updateShakeCountdown(seconds);
    clearInterval(shakeTimer);
    shakeTimer = setInterval(() => {
      seconds -= 1;
      updateShakeCountdown(Math.max(0, seconds));
      if (seconds <= 0) stopShake();
    }, 1000);
  }

  function stopShake() {
    clearInterval(shakeTimer);
    clearTimeout(revealTransitionTimer);
    setPhase('open');
    speakState('reveal_ready');
    revealTransitionTimer = setTimeout(() => {
      revealTransitionTimer = null;
      beginRevealCountdown();
    }, 2000);
  }

  function beginRevealCountdown() {
    countdown(
      reveal,
      'VISION COUNTDOWN',
      '请保持骰子和骰盅位置不动。',
      visionCountdownMeta,
    );
  }

  function resetAnalysisSteps() {
    stopVisionStream();
    pendingAnalysisResult = null;
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

  function updateAnalysisProgress(job) {
    if (job.phase === 'detecting') {
      $('stepDetect').classList.add('active');
      $('stepDetect').querySelector('span').textContent = '…';
      const count = Number(job.stable_count || 0);
      const required = Number(job.stable_frames || 0);
      $('analysisStatus').textContent = count > 0 && required > 0
        ? `YOLOv8 正在检测双方各 5 颗骰子，并等待稳定帧（${count}/${required}）…`
        : 'YOLOv8 正在检测双方各 5 颗骰子，并等待稳定帧…';
    } else if (job.phase === 'verifying') {
      $('stepDetect').classList.add('active');
      $('stepDetect').querySelector('span').textContent = '✓';
      $('stepJudge').classList.add('active');
      $('stepJudge').querySelector('span').textContent = '…';
      $('analysisStatus').textContent = 'YOLOv8 结果已稳定，正在调用大模型复核…';
    } else if (job.phase === 'holding') {
      const remaining = Number(job.remaining_ms);
      $('stepDetect').classList.add('active');
      $('stepDetect').querySelector('span').textContent = '✓';
      $('stepJudge').classList.add('active');
      $('stepJudge').querySelector('span').textContent = '✓';
      $('analysisStatus').textContent = Number.isFinite(remaining) && remaining > 0
        ? `结果已锁定，实时画面将继续播放 ${Math.ceil(remaining / 1000)} 秒…`
        : '结果已锁定，实时画面仍在播放…';
    } else {
      $('analysisStatus').textContent = 'K3 推理进程已启动，等待识别结果…';
    }
  }

  function updateStructuredEvent(event) {
    if (!event || typeof event !== 'object') return;
    if (event.event === 'video') {
      startVisionStream(event);
    } else if (event.event === 'result') {
      pendingAnalysisResult = event.result && typeof event.result === 'object'
        ? event.result
        : event;
    } else if (event.event === 'diagnosis' || event.diagnosed) {
      pendingAnalysisResult = event.result && typeof event.result === 'object'
        ? event.result
        : event;
    } else if (event.event === 'phase' || event.event === 'progress') {
      updateAnalysisProgress(event);
    }
  }

  function applyAnalysisSnapshot(snapshot, eventSequence) {
    let latestHoldingEvent = null;
    for (const event of (snapshot.events || [])) {
      const sequence = Number(event.sequence || 0);
      if (sequence > eventSequence.value) {
        eventSequence.value = sequence;
        if (event.phase === 'holding') latestHoldingEvent = event;
        updateStructuredEvent(event);
      }
    }
    // A provider emits ``complete`` before its worker returns to ComponentJob;
    // during that small window the snapshot can still be ``running`` with
    // only the earlier result event available. Wait for the terminal success
    // snapshot (which carries the canonical result) before closing SSE or
    // switching to the result view.
    const terminal = snapshot.status === 'success'
      || (snapshot.phase === 'complete' && snapshot.result);
    if (snapshot.status === 'error' && snapshot.result && snapshot.result.diagnosed) {
      pendingAnalysisResult = snapshot.result;
      return true;
    }
    if (terminal && (snapshot.result || pendingAnalysisResult)) {
      $('stepDetect').classList.add('active');
      $('stepDetect').querySelector('span').textContent = '✓';
      $('stepJudge').classList.add('active');
      $('stepJudge').querySelector('span').textContent = '✓';
      const result = snapshot.result || pendingAnalysisResult;
      pendingAnalysisResult = result;
      $('analysisStatus').textContent = result.source === 'yolo_timeout_fallback'
        ? '大模型请求超时，已使用稳定的 YOLOv8 结果，结果已锁定。'
        : result.source === 'llm_override'
          ? 'YOLOv8 与大模型结果不一致，已以大模型结果为准。'
          : result.source === 'yolo_only'
            ? '未启用大模型复核，已使用稳定的 YOLOv8 结果。'
            : 'YOLOv8 与大模型复核一致，结果已锁定。';
      return true;
    }
    if (snapshot.status === 'error' || snapshot.cancelled) {
      throw new Error(snapshot.error || 'K3 视觉裁决已取消');
    }
    // SSE snapshots expose the lifecycle phase at the top level, while the
    // countdown value belongs to the structured holding event. Preserve that
    // richer event instead of overwriting it with a generic message.
    updateAnalysisProgress(latestHoldingEvent || snapshot);
    return false;
  }

  function streamAnalysis(jobId) {
    if (!('EventSource' in window)) return Promise.reject(new Error('浏览器不支持 SSE'));
    return new Promise((resolve, reject) => {
      const source = new EventSource(`/api/adjudicate/${jobId}/stream`);
      const eventSequence = { value: 0 };
      let settled = false;

      const finish = (error = null) => {
        if (settled) return;
        settled = true;
        source.close();
        if (error) reject(error); else resolve();
      };
      const handle = (event) => {
        try {
          const snapshot = JSON.parse(event.data);
          if (applyAnalysisSnapshot(snapshot, eventSequence)) finish();
          else if (snapshot.status === 'error') finish(new Error(snapshot.error || 'K3 视觉裁决失败'));
        } catch (error) {
          finish(error);
        }
      };
      source.addEventListener('snapshot', handle);
      source.addEventListener('update', handle);
      source.addEventListener('complete', handle);
      source.onerror = () => {
        if (!settled) finish(new Error('SSE 连接中断'));
      };
    });
  }

  async function pollAnalysisFallback(jobId) {
    for (;;) {
      const job = await requestJson(`/api/adjudicate/${jobId}`);
      if (applyAnalysisSnapshot(job, { value: 0 })) return;
      await new Promise((resolve) => setTimeout(resolve, 700));
    }
  }

  async function pollAnalysis(jobId) {
    try {
      await streamAnalysis(jobId);
    } catch (error) {
      // SSE is the primary push path. Keep the existing snapshot polling as a
      // compatibility fallback for older browsers or an upgraded backend that
      // is temporarily unavailable.
      if (error.message !== 'SSE 连接中断' && error.message !== '浏览器不支持 SSE') {
        throw error;
      }
      await pollAnalysisFallback(jobId);
    }
    const job = await requestJson(`/api/adjudicate/${jobId}`);
    if (job.status === 'error' && job.result && job.result.diagnosed) {
      showDiagnosis(job.result);
    } else if ((job.status === 'success' || job.phase === 'complete') && (job.result || pendingAnalysisResult)) {
      showResult(job.result || pendingAnalysisResult);
    }
  }

  async function reveal() {
    setPhase('analysis');
    resetAnalysisSteps();
    speakState('analysis_started');
    try {
      const job = await requestJson('/api/adjudicate', { method: 'POST', body: JSON.stringify({ game: 'dice' }) });
      await pollAnalysis(job.job_id);
    } catch (error) {
      stopVisionStream();
      markAnalysisFailure();
      document.querySelector('.analysis-spinner')?.classList.add('hidden');
      $('analysisTitle').textContent = '识别未完成';
      $('analysisStatus').textContent = error.message;
      $('analysisFailureActions').classList.remove('hidden');
      toast(`K3 视觉裁决失败：${error.message}`);
    }
  }

  function showDiagnosis(result) {
    stopVisionStream();
    if (!result || result.retry_required !== true) {
      throw new Error('诊断结果缺少 retry_required 标记');
    }
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
    toast('视觉裁决未完成，请重新开始一局');
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
    setPhase('result');
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
      ? '超时，采用 YOLOv8'
      : result.source === 'yolo_failure_fallback'
        ? '请求失败，采用 YOLOv8'
      : result.source === 'llm_override'
        ? '结果不一致，以大模型结果为准'
        : result.source === 'yolo_only'
          ? '未启用大模型，采用 YOLOv8'
          : '复核一致';
    $('resultSubtitle').textContent = `YOLOv8：玩家 ${player} : Agent ${agent}；大模型${verificationText}`;
    banner.classList.toggle('loss', !playerWins && !tie);
    const resultTtsKey = tie ? 'result_tie' : playerWins ? 'result_player_win' : 'result_agent_win';
    speakState(resultTtsKey, { player_score: player, agent_score: agent });
  }

  function resetRound() {
    stopVisionStream();
    clearTimeout(revealTransitionTimer);
    revealTransitionTimer = null;
    round += 1;
    $('roundNumber').textContent = String(round).padStart(2, '0');
    playerDice = [];
    agentDice = [];
    $('analysisFailureActions').classList.add('hidden');
    $('stepDetect').classList.remove('failed');
    $('stepJudge').classList.remove('failed');
    updateScores();
    setPhase('ready');
  }

  function backFromRules() {
    stopSpeech();
    returnToSelect();
  }

  function repeatRules() {
    toast('正在重复播报游戏规则');
    speakState('rules_intro');
  }

  function confirmRules() {
    setPhase('ready');
    speakState('rules_confirmed');
  }

  const handlers = {
    startShake: () => {
      prepareCountdownAudio();
      speakState('shake_started');
      countdown(beginShake, 'GET READY', '和 Agent 同步');
    },
    stopShake: () => stopShake(),
    analysisNewRound: () => resetRound(),
    analysisBackToGames: () => returnToSelect(),
    newRound: () => resetRound(),
    backToGames: () => returnToSelect(),
    repeatRules,
    confirmRules,
    backFromRules: () => backFromRules(),
  };

  function enter(manifest) {
    configureParticipants(manifest);
    stopVisionStream();
    round = 1;
    playerDice = [];
    agentDice = [];
    $('roundNumber').textContent = '01';
    $('analysisFailureActions').classList.add('hidden');
    document.querySelector('.analysis-spinner')?.classList.remove('hidden');
    updateScores();
    Object.entries(handlers).forEach(([id, fn]) => $(id).addEventListener('click', fn));
    setPhase('rules');
    speakState('rules_intro');
  }

  function teardown() {
    stopVisionStream();
    clearInterval(shakeTimer);
    clearInterval(countdownTimer);
    clearTimeout(revealTransitionTimer);
    shakeTimer = null;
    countdownTimer = null;
    revealTransitionTimer = null;
    participantSides = null;
    Object.entries(handlers).forEach(([id, fn]) => $(id).removeEventListener('click', fn));
    stopSpeech();
  }

  function onKey(event) {
    if (event.key === 'Escape') {
      if (state.phase === 'rules') backFromRules();
      else if (state.phase === 'ready' || state.phase === 'result') returnToSelect();
      else if (state.phase === 'analysis' && !$('analysisFailureActions').classList.contains('hidden')) {
        returnToSelect();
      }
      return;
    }
    if (event.key === 'Enter') {
      if (state.phase === 'rules') confirmRules();
      else if (state.phase === 'ready') handlers.startShake();
      return;
    }
    if (state.phase === 'rules' && event.key === 'ArrowDown') {
      repeatRules();
      return;
    }
    if (event.key === 'ArrowUp') {
      return;
    }
  }

  return {
    id: 'dice',
    phases,
    progressCount: 6,
    phaseMeta,
    enter,
    teardown,
    onKey,
  };
}
