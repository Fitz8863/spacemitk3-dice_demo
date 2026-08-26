(() => {
  const state = {
    phase: 'select',
    round: 1,
    selectedGame: 'dice',
    playerDice: [],
    agentDice: [],
    sound: true,
    stream: null,
    shakeTimer: null,
    countdownTimer: null,
    analysisJobId: null,
    ttsAudio: null,
    ttsObjectUrl: null,
    ttsAbortController: null,
    ttsRequestId: 0,
    ttsFallbackNotified: false,
  };

  const $ = (id) => document.getElementById(id);
  const views = [...document.querySelectorAll('[data-view]')];
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
  const dicePips = {
    1: [4], 2: [0, 8], 3: [0, 4, 8], 4: [0, 2, 6, 8],
    5: [0, 2, 4, 6, 8], 6: [0, 2, 3, 5, 6, 8],
  };

  function setPhase(phase) {
    state.phase = phase;
    views.forEach((view) => view.classList.toggle('hidden', view.dataset.view !== phase));
    const [kicker, title, copy, footer] = phaseMeta[phase];
    $('phaseKicker').textContent = kicker;
    $('phaseTitle').textContent = title;
    $('phaseCopy').textContent = copy;
    $('stageFooterText').textContent = footer;
    const index = ['select', 'rules', 'ready', 'countdown', 'shaking', 'open', 'analysis', 'result'].indexOf(phase);
    document.querySelectorAll('#progressDots span').forEach((dot, i) => {
      dot.classList.toggle('active', i <= Math.max(0, Math.min(5, index)));
    });
    updateAgent(phase);
    updateScoreState(phase);
  }

  function updateAgent(phase) {
    const tasks = {
      taskWelcome: ['select', 'rules', 'ready'],
      taskCountdown: ['countdown', 'shaking', 'open'],
      taskVision: ['analysis'],
      taskResult: ['result'],
    };
    Object.entries(tasks).forEach(([id, phases]) => {
      const node = $(id);
      node.classList.toggle('active', phases.includes(phase));
      if (phase === 'result' && id === 'taskResult') node.classList.add('done');
      if (phase === 'analysis' && id === 'taskCountdown') node.classList.add('done');
    });
    const quotes = {
      select: '“欢迎来到 Dice Arena，准备好和我一起摇骰子了吗？”',
      rules: '“双方各 5 颗骰子，点数总和更大的一方获胜。”',
      ready: '“拿好骰盅，点击开始，我们马上同步摇骰。”',
      countdown: '“3、2、1，开始！”',
      shaking: '“摇起来！我会和你保持同步。”',
      open: '“3、2、1，停。请同时开盖。”',
      analysis: '“YOLOv8 先看骰子，大模型再确认结果。”',
      result: '“结果出来了，恭喜获胜者！”',
    };
    $('agentQuote').textContent = quotes[phase];
  }

  function updateScoreState(phase) {
    const copy = {
      select: '等待游戏开始', rules: '等待确认规则', ready: '等待玩家开始',
      countdown: '同步倒计时中', shaking: '双方摇骰中', open: '等待开盖确认',
      analysis: 'YOLOv8 + 大模型复核中', result: '本局结果已锁定',
    };
    $('scoreState').innerHTML = `<span class="state-dot"></span> ${copy[phase]}`;
  }

  function sum(dice) { return dice.reduce((a, b) => a + b, 0); }

  function diceMarkup(values, className = '') {
    return values.map((value) => `<div class="die ${className}" aria-label="${value}点">${Array.from({ length: 9 }, (_, i) => `<span class="${dicePips[value].includes(i) ? 'on' : ''}"></span>`).join('')}</div>`).join('');
  }

  function updateScores() {
    const player = sum(state.playerDice);
    const agent = sum(state.agentDice);
    $('livePlayerScore').textContent = state.playerDice.length ? player : '—';
    $('liveAgentScore').textContent = state.agentDice.length ? agent : '—';
    $('playerScore').textContent = state.playerDice.length ? player : '—';
    $('agentScore').textContent = state.agentDice.length ? agent : '—';
    $('playerDice').innerHTML = diceMarkup(state.playerDice);
    $('agentDice').innerHTML = diceMarkup(state.agentDice, 'agent-die');
  }

  function toast(message) {
    const node = $('toast');
    node.textContent = message;
    node.classList.add('show');
    setTimeout(() => node.classList.remove('show'), 2800);
  }

  function stopSpeech() {
    state.ttsRequestId += 1;
    state.ttsAbortController?.abort();
    state.ttsAbortController = null;
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

  function browserSpeechFallback(message) {
    if (!state.sound || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.lang = 'zh-CN';
    utterance.rate = 0.96;
    window.speechSynthesis.speak(utterance);
  }

  async function speak(message) {
    if (!state.sound || !message) return;
    stopSpeech();
    const requestId = state.ttsRequestId;
    const controller = new AbortController();
    state.ttsAbortController = controller;
    try {
      const response = await fetch('/api/tts/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: message, voice: 'default', speed: 1.0 }),
        signal: controller.signal,
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || `HTTP ${response.status}`);
      }
      const blob = await response.blob();
      if (!state.sound || requestId !== state.ttsRequestId) return;
      state.ttsAbortController = null;
      const objectUrl = URL.createObjectURL(blob);
      state.ttsObjectUrl = objectUrl;
      const audio = new Audio(objectUrl);
      state.ttsAudio = audio;
      const release = () => {
        if (state.ttsAudio === audio) state.ttsAudio = null;
        URL.revokeObjectURL(objectUrl);
        if (state.ttsObjectUrl === objectUrl) state.ttsObjectUrl = null;
      };
      audio.addEventListener('ended', release, { once: true });
      audio.addEventListener('error', release, { once: true });
      await audio.play();
    } catch (error) {
      if (error.name === 'AbortError' || requestId !== state.ttsRequestId || !state.sound) return;
      console.warn('K3 Qwen3-TTS failed; using browser speech fallback:', error);
      if (!state.ttsFallbackNotified) {
        state.ttsFallbackNotified = true;
        toast('K3 TTS 暂不可用，已临时使用浏览器语音');
      }
      browserSpeechFallback(message);
    }
  }

  function countdown(next, label, hint) {
    clearInterval(state.countdownTimer);
    let seconds = 3;
    $('countdownLabel').textContent = label;
    $('countdownHint').textContent = hint;
    $('countdownNumber').textContent = seconds;
    setPhase('countdown');
    state.countdownTimer = setInterval(() => {
      seconds -= 1;
      $('countdownNumber').textContent = Math.max(0, seconds);
      if (seconds <= 0) {
        clearInterval(state.countdownTimer);
        next();
      }
    }, 900);
  }

  function beginShake() {
    setPhase('shaking');
    let seconds = 8;
    $('shakeSeconds').textContent = String(seconds).padStart(2, '0');
    speak('开始摇骰');
    clearInterval(state.shakeTimer);
    state.shakeTimer = setInterval(() => {
      seconds -= 1;
      $('shakeSeconds').textContent = String(Math.max(0, seconds)).padStart(2, '0');
      if (seconds <= 0) stopShake();
    }, 1000);
  }

  function stopShake() {
    clearInterval(state.shakeTimer);
    countdown(() => setPhase('open'), 'STOP COUNTDOWN', '倒计时结束后，请同时开盖。');
    speak('停止摇骰');
  }

  async function requestJson(url, options = {}) {
    const response = await fetch(url, {
      ...options,
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    return payload;
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
    state.analysisJobId = jobId;
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
    speak('正在调用 K3 YOLOv8 和大模型复核');
    try {
      const job = await requestJson('/api/analyze', { method: 'POST', body: '{}' });
      await pollAnalysis(job.job_id);
    } catch (error) {
      $('analysisTitle').textContent = '识别未完成';
      $('analysisStatus').textContent = error.message;
      $('analysisRetry').classList.remove('hidden');
      toast(`K3 视觉分析失败：${error.message}`);
    }
  }

  function showResult(result) {
    state.playerDice = Array.isArray(result.first_dice) ? result.first_dice : [];
    state.agentDice = Array.isArray(result.second_dice) ? result.second_dice : [];
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
    speak(tie ? '本局平局，再来一局吧' : playerWins ? '恭喜你，玩家获胜' : '本局 Agent 获胜');
  }

  function resetRound() {
    state.round += 1;
    $('roundNumber').textContent = String(state.round).padStart(2, '0');
    $('roundMini').textContent = String(state.round).padStart(2, '0');
    state.playerDice = [];
    state.agentDice = [];
    $('analysisRetry').classList.add('hidden');
    updateScores();
    setPhase('ready');
  }

  document.querySelectorAll('.game-option:not(.disabled)').forEach((button) => button.addEventListener('click', () => {
    document.querySelectorAll('.game-option').forEach((item) => {
      item.classList.toggle('selected', item === button);
      item.setAttribute('aria-selected', item === button ? 'true' : 'false');
    });
    state.selectedGame = button.dataset.game;
  }));

  function enterSelectedGame() {
    if (state.selectedGame === 'dice') {
      setPhase('rules');
      speak('摇骰子规则');
    }
  }

  $('startGame').addEventListener('click', enterSelectedGame);
  $('gameList').addEventListener('dblclick', enterSelectedGame);
  $('repeatRules').addEventListener('click', () => {
    toast('正在重复播报游戏规则');
    speak('双方各摇五颗骰子，停止后同时开盖。K3 上的视觉模型会识别每颗骰子的点数，再由大模型确认胜负。');
  });
  $('confirmRules').addEventListener('click', () => { setPhase('ready'); speak('规则确认完成，请准备'); });
  $('startShake').addEventListener('click', () => countdown(beginShake, 'GET READY', '和 Agent 同步'));
  $('stopShake').addEventListener('click', stopShake);
  $('revealDice').addEventListener('click', reveal);
  $('analysisRetry').addEventListener('click', reveal);
  $('newRound').addEventListener('click', resetRound);
  $('backToGames').addEventListener('click', () => {
    state.round = 1;
    $('roundNumber').textContent = '01';
    $('roundMini').textContent = '01';
    setPhase('select');
  });
  $('soundToggle').addEventListener('click', () => {
    state.sound = !state.sound;
    if (!state.sound) stopSpeech();
    $('soundToggle').textContent = state.sound ? '🔊' : '🔇';
    toast(state.sound ? 'K3 Qwen3-TTS 播报已开启' : '语音播报已关闭');
  });
  $('cameraButton').addEventListener('click', async () => {
    if (state.stream) {
      state.stream.getTracks().forEach((track) => track.stop());
      state.stream = null;
      $('cameraFrame').classList.remove('camera-active');
      $('cameraStatus').textContent = '未连接浏览器预览';
      $('cameraButton').textContent = '开启本地预览 ↗';
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      toast('当前浏览器不支持预览；K3 后端仍会直接使用板端摄像头');
      return;
    }
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
      $('cameraVideo').srcObject = state.stream;
      $('cameraFrame').classList.add('camera-active');
      $('cameraStatus').textContent = '浏览器预览已连接';
      $('cameraButton').textContent = '关闭本地预览';
      toast('预览已连接，实际识别由 K3 板端摄像头完成');
    } catch {
      toast('浏览器摄像头权限未开启；K3 后端仍会直接使用板端摄像头');
    }
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
      const options = [...document.querySelectorAll('.game-option:not(.disabled)')];
      const current = options.findIndex((option) => option.classList.contains('selected'));
      options[(current + (event.key === 'ArrowDown' ? 1 : -1) + options.length) % options.length].click();
    }
    if (event.key === 'Enter' && state.phase === 'select') enterSelectedGame();
    if (event.key.toLowerCase() === 'q' && state.phase === 'shaking') stopShake();
  });

  updateScores();
  setPhase('select');
})();
