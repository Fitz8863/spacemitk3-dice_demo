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

  const phases = ['select', 'rules', 'ready', 'countdown', 'shaking', 'open', 'analysis', 'result'];
  const phaseMeta = {
    select: ['GAME SELECT', '选择一场游戏', '欢迎来到 Dice Arena，选择游戏后按 OK 开始。', '选择游戏开始体验'],
    rules: ['GAME RULES', '游戏规则', '听完规则后确认，Agent 会带你完成整局游戏。', '确认规则后开始'],
    ready: ['READY CHECK', '准备好了吗？', '人手操作模式已开启，拿起骰盅后点击开始。', '等待玩家开始'],
    countdown: ['SYNC COUNTDOWN', '同步倒计时', '与 Agent 保持同步，倒计时结束后开始摇骰。', '倒计时进行中'],
    shaking: ['SHAKE PHASE', '摇骰进行中', '双方同时摇骰，准备好后可提前停止。', '双方摇骰中'],
    open: ['REVEAL', '同时开盖', '把骰盅放回区域，确认双方都已开盖。', '等待双方开盖'],
    analysis: ['VISION ANALYSIS', '正在判定胜负', 'K3 上的 YOLOv8 正在识别，随后由大模型复核。', '视觉识别中'],
    result: ['ROUND RESULT', '本局结果', '点数已经锁定，看看谁赢下了这一局。', '结果已播报'],
  };

  function sum(dice) { return dice.reduce((a, b) => a + b, 0); }

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
      $('analysisStatus').textContent = 'YOLOv8 正在检测双方各 5 颗骰子，并等待稳定帧…';
    } else if (job.phase === 'verifying') {
      $('stepDetect').classList.add('active');
      $('stepDetect').querySelector('span').textContent = '✓';
      $('stepJudge').classList.add('active');
      $('stepJudge').querySelector('span').textContent = '…';
      $('analysisStatus').textContent = 'YOLOv8 结果已稳定，正在调用大模型复核…';
    } else {
      $('analysisStatus').textContent = 'K3 推理进程已启动，等待识别结果…';
    }
  }

  async function pollAnalysis(jobId) {
    for (;;) {
      const job = await requestJson(`/api/analyze/${jobId}`);
      if (job.status === 'success' && job.result) {
        $('stepDetect').classList.add('active');
        $('stepDetect').querySelector('span').textContent = '✓';
        $('stepJudge').classList.add('active');
        $('stepJudge').querySelector('span').textContent = '✓';
        $('analysisStatus').textContent = 'YOLOv8 与大模型复核一致，结果已锁定。';
        setTimeout(() => showResult(job.result), 450);
        return;
      }
      if (job.status === 'error') throw new Error(job.error || 'K3 视觉分析失败');
      updateAnalysisProgress(job);
      await new Promise((resolve) => setTimeout(resolve, 700));
    }
  }

  async function reveal() {
    setPhase('analysis');
    resetAnalysisSteps();
    speakState('analysis_started');
    try {
      const job = await requestJson('/api/analyze', { method: 'POST', body: JSON.stringify({ game: 'dice' }) });
      await pollAnalysis(job.job_id);
    } catch (error) {
      $('analysisTitle').textContent = '识别未完成';
      $('analysisStatus').textContent = error.message;
      $('analysisRetry').classList.remove('hidden');
      toast(`K3 视觉分析失败：${error.message}`);
    }
  }

  function showResult(result) {
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
    $('resultSubtitle').textContent = `YOLOv8：${player} : ${agent}；大模型复核：${result.llm_winner || winner}`;
    banner.classList.toggle('loss', !playerWins && !tie);
    const resultTtsKey = tie ? 'result_tie' : playerWins ? 'result_player_win' : 'result_agent_win';
    speakState(resultTtsKey, { player_score: player, agent_score: agent });
  }

  function resetRound() {
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

  const handlers = {
    startShake: () => countdown(beginShake, 'GET READY', '和 Agent 同步'),
    stopShake: () => stopShake(),
    revealDice: () => reveal(),
    analysisRetry: () => reveal(),
    newRound: () => resetRound(),
    backToGames: () => returnToSelect(),
    repeatRules: () => { toast('正在重复播报游戏规则'); speakState('rules_intro'); },
    confirmRules: () => { setPhase('ready'); speakState('rules_confirmed'); },
    backFromRules: () => backFromRules(),
  };

  function enter() {
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
    clearInterval(shakeTimer);
    clearInterval(countdownTimer);
    shakeTimer = null;
    countdownTimer = null;
    Object.entries(handlers).forEach(([id, fn]) => $(id).removeEventListener('click', fn));
    stopSpeech();
  }

  function onKey(event) {
    if (event.key === 'Escape' && state.phase === 'rules') backFromRules();
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
