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
    ttsPlaybackCancel: null,
    ttsRequestId: 0,
    ttsFallbackNotified: false,
    ttsConfigPromise: null,
    ttsConfigErrorNotified: false,
  };

  const TTS_TEXTS_URL = './tts-texts.json';

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

  // Deliberately do not call browser speech synthesis.  Dice Arena must use
  // the Qwen3-TTS model running on the K3 board; browser speech would hide a
  // broken backend and make the voice sound unrelated to the configured
  // speaker embedding.

  async function loadTtsConfig() {
    if (!state.ttsConfigPromise) {
      state.ttsConfigPromise = fetch(TTS_TEXTS_URL, { cache: 'no-store' })
        .then(async (response) => {
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const config = await response.json();
          if (!config || typeof config !== 'object' || !config.texts || typeof config.texts !== 'object') {
            throw new Error('配置中缺少 texts 对象');
          }
          const speed = Number(config.speed ?? 1.0);
          if (!Number.isFinite(speed) || speed < 0.25 || speed > 4.0) {
            throw new Error('speed 必须在 0.25 到 4.0 之间');
          }
          return {
            voice: typeof config.voice === 'string' && config.voice.trim() ? config.voice.trim() : 'default',
            speed,
            texts: config.texts,
          };
        });
    }
    return state.ttsConfigPromise;
  }

  function renderTtsText(template, values = {}) {
    return template.replace(/\{([a-zA-Z0-9_]+)\}/g, (match, name) => (
      Object.prototype.hasOwnProperty.call(values, name) ? String(values[name]) : match
    ));
  }

  async function speakState(key, values = {}) {
    if (!state.sound) return;
    try {
      const config = await loadTtsConfig();
      const template = config.texts[key];
      if (typeof template !== 'string' || !template.trim()) {
        throw new Error(`未配置 TTS 状态文案：${key}`);
      }
      await speak(renderTtsText(template, values), config);
    } catch (error) {
      console.error(`Failed to load TTS state ${key}:`, error);
      if (!state.ttsConfigErrorNotified) {
        state.ttsConfigErrorNotified = true;
        toast('TTS 文案配置加载失败，请检查 web/tts-texts.json');
      }
    }
  }

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
    speakState('shake_started');
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
    speakState('shake_stopped');
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
    speakState('analysis_started');
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
    const resultTtsKey = tie ? 'result_tie' : playerWins ? 'result_player_win' : 'result_agent_win';
    speakState(resultTtsKey, { player_score: player, agent_score: agent });
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
      speakState('rules_intro');
    }
  }

  $('startGame').addEventListener('click', enterSelectedGame);
  $('gameList').addEventListener('dblclick', enterSelectedGame);
  $('repeatRules').addEventListener('click', () => {
    toast('正在重复播报游戏规则');
    speakState('rules_intro');
  });
  $('confirmRules').addEventListener('click', () => { setPhase('ready'); speakState('rules_confirmed'); });
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
