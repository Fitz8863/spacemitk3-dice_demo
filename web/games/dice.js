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
  let visionStreamToken = 0;
  let pendingAnalysisResult = null;

  const phases = ['select', 'rules', 'ready', 'countdown', 'shaking', 'open', 'analysis', 'result'];
  const phaseMeta = {
    select: ['GAME SELECT', '选择一场游戏', '欢迎来到 Dice Arena，选择游戏后按 OK 开始。', '选择游戏开始体验'],
    rules: ['GAME RULES', '游戏规则', '听完规则后按 Enter 确认，按 ↓ 可以再听一次。', 'Enter 确认 · ↓ 重听规则'],
    ready: ['READY CHECK', '准备好了吗？', '人手操作模式已开启，拿起骰盅后点击开始。', '等待玩家开始'],
    countdown: ['SYNC COUNTDOWN', '同步倒计时', '与 Agent 保持同步，倒计时结束后开始摇骰。', '倒计时进行中'],
    shaking: ['SHAKE PHASE', '摇骰进行中', '双方同时摇骰，准备好后可提前停止。', '双方摇骰中'],
    open: ['REVEAL', '同时开盖', '把骰盅放回区域，确认双方都已开盖。', '等待双方开盖'],
    analysis: ['VISION ADJUDICATION', '正在判定胜负', '视觉裁决器正在识别骰子点数，随后由大模型复核。', '视觉裁决中'],
    result: ['ROUND RESULT', '本局结果', '点数已经锁定，看看谁赢下了这一局。', '结果已播报'],
  };

  function sum(dice) { return dice.reduce((a, b) => a + b, 0); }

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

  function updateScores() {
    const player = sum(playerDice);
    const agent = sum(agentDice);
    $('playerScore').textContent = playerDice.length ? player : '—';
    $('agentScore').textContent = agentDice.length ? agent : '—';
    $('playerDice').innerHTML = diceMarkup(playerDice);
    $('agentDice').innerHTML = diceMarkup(agentDice, 'agent-die');
  }

  function countdown(next, label, hint) {
    clearInterval(countdownTimer);
    let seconds = 3;
    $('countdownLabel').textContent = label;
    $('countdownHint').textContent = hint;
    $('countdownNumber').textContent = seconds;
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
    let seconds = 8;
    $('shakeSeconds').textContent = String(seconds).padStart(2, '0');
    speakState('shake_started');
    clearInterval(shakeTimer);
    shakeTimer = setInterval(() => {
      seconds -= 1;
      $('shakeSeconds').textContent = String(Math.max(0, seconds)).padStart(2, '0');
      if (seconds <= 0) stopShake();
    }, 1000);
  }

  function stopShake() {
    clearInterval(shakeTimer);
    countdown(() => setPhase('open'), 'STOP COUNTDOWN', '倒计时结束后，请同时开盖。');
    speakState('shake_stopped');
  }

  function resetAnalysisSteps() {
    stopVisionStream();
    pendingAnalysisResult = null;
    $('stepCapture').classList.add('active');
    $('stepCapture').querySelector('span').textContent = '✓';
    $('stepDetect').classList.remove('active');
    $('stepDetect').querySelector('span').textContent = '2';
    $('stepJudge').classList.remove('active');
    $('stepJudge').querySelector('span').textContent = '3';
    $('analysisTitle').textContent = '正在识别骰子';
    $('analysisRetry').classList.add('hidden');
    $('analysisStatus').textContent = '正在请求 K3 YOLOv8 推理进程…';
  }

  function updateAnalysisProgress(job) {
    if (job.phase === 'detecting') {
      $('stepDetect').classList.add('active');
      $('stepDetect').querySelector('span').textContent = '✓';
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
    } else if (event.event === 'phase' || event.event === 'progress') {
      updateAnalysisProgress(event);
    }
  }

  function applyAnalysisSnapshot(snapshot, eventSequence) {
    for (const event of (snapshot.events || [])) {
      const sequence = Number(event.sequence || 0);
      if (sequence > eventSequence.value) {
        eventSequence.value = sequence;
        updateStructuredEvent(event);
      }
    }
    if (snapshot.phase === 'holding') {
      updateAnalysisProgress(snapshot);
    }
    // A provider emits ``complete`` before its worker returns to ComponentJob;
    // during that small window the snapshot can still be ``running`` with
    // only the earlier result event available. Wait for the terminal success
    // snapshot (which carries the canonical result) before closing SSE or
    // switching to the result view.
    const terminal = snapshot.status === 'success'
      || (snapshot.phase === 'complete' && snapshot.result);
    if (terminal && (snapshot.result || pendingAnalysisResult)) {
      $('stepDetect').classList.add('active');
      $('stepDetect').querySelector('span').textContent = '✓';
      $('stepJudge').classList.add('active');
      $('stepJudge').querySelector('span').textContent = '✓';
      const result = snapshot.result || pendingAnalysisResult;
      pendingAnalysisResult = result;
      $('analysisStatus').textContent = result.source === 'yolo_timeout_fallback'
        ? '大模型请求超时，已使用稳定的 YOLOv8 结果，结果已锁定。'
        : 'YOLOv8 与大模型复核一致，结果已锁定。';
      return true;
    }
    if (snapshot.status === 'error' || snapshot.cancelled) {
      throw new Error(snapshot.error || 'K3 视觉裁决已取消');
    }
    updateAnalysisProgress(snapshot);
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
    if ((job.status === 'success' || job.phase === 'complete') && (job.result || pendingAnalysisResult)) {
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
      $('analysisTitle').textContent = '识别未完成';
      $('analysisStatus').textContent = error.message;
      $('analysisRetry').classList.remove('hidden');
      toast(`K3 视觉裁决失败：${error.message}`);
    }
  }

  function showResult(result) {
    stopVisionStream();
    playerDice = Array.isArray(result.first_dice) ? result.first_dice : [];
    agentDice = Array.isArray(result.second_dice) ? result.second_dice : [];
    updateScores();
    setPhase('result');
    const player = Number(result.first_sum);
    const agent = Number(result.second_sum);
    const banner = $('resultBanner');
    const winner = result.winner;
    const tie = winner === 'TIE';
    const playerWins = winner === 'LEFT';
    $('resultEmoji').textContent = tie ? '🤝' : playerWins ? '🏆' : '✨';
    $('resultTitle').textContent = tie ? '平局！' : playerWins ? '玩家获胜' : 'Agent 获胜';
    const verificationText = result.source === 'yolo_timeout_fallback'
      ? '超时，采用 YOLOv8'
      : `复核：${result.llm_winner || winner}`;
    $('resultSubtitle').textContent = `YOLOv8：${player} : ${agent}；大模型${verificationText}`;
    banner.classList.toggle('loss', !playerWins && !tie);
    const resultTtsKey = tie ? 'result_tie' : playerWins ? 'result_player_win' : 'result_agent_win';
    speakState(resultTtsKey, { player_score: player, agent_score: agent });
  }

  function resetRound() {
    stopVisionStream();
    round += 1;
    $('roundNumber').textContent = String(round).padStart(2, '0');
    playerDice = [];
    agentDice = [];
    $('analysisRetry').classList.add('hidden');
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
    startShake: () => countdown(beginShake, 'GET READY', '和 Agent 同步'),
    stopShake: () => stopShake(),
    revealDice: () => reveal(),
    analysisRetry: () => reveal(),
    newRound: () => resetRound(),
    backToGames: () => returnToSelect(),
    repeatRules,
    confirmRules,
    backFromRules: () => backFromRules(),
  };

  function enter() {
    stopVisionStream();
    round = 1;
    playerDice = [];
    agentDice = [];
    $('roundNumber').textContent = '01';
    $('analysisRetry').classList.add('hidden');
    updateScores();
    Object.entries(handlers).forEach(([id, fn]) => $(id).addEventListener('click', fn));
    setPhase('rules');
    speakState('rules_intro');
  }

  function teardown() {
    stopVisionStream();
    clearInterval(shakeTimer);
    clearInterval(countdownTimer);
    shakeTimer = null;
    countdownTimer = null;
    Object.entries(handlers).forEach(([id, fn]) => $(id).removeEventListener('click', fn));
    stopSpeech();
  }

  function onKey(event) {
    if (state.phase === 'rules') {
      if (event.key === 'Escape') {
        backFromRules();
        return;
      }
      if (event.key === 'Enter') {
        event.preventDefault();
        if (!event.repeat) confirmRules();
        return;
      }
      if (event.key === 'ArrowDown') {
        event.preventDefault();
        if (!event.repeat) repeatRules();
        return;
      }
    }
    if (event.key.toLowerCase() === 'q' && state.phase === 'shaking') stopShake();
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
