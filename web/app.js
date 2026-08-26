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
    analysis: ['VISION ANALYSIS', '正在判定胜负', 'YOLOv8 正在识别双方骰子并计算点数总和。', '视觉识别中'],
    result: ['ROUND RESULT', '本局结果', '点数已经锁定，看看谁赢下了这一局。', '结果已播报'],
  };
  const dicePips = { 1:[4], 2:[0,8], 3:[0,4,8], 4:[0,2,6,8], 5:[0,2,4,6,8], 6:[0,2,3,5,6,8] };

  function setPhase(phase) {
    state.phase = phase;
    views.forEach((view) => view.classList.toggle('hidden', view.dataset.view !== phase));
    const [kicker, title, copy, footer] = phaseMeta[phase];
    $('phaseKicker').textContent = kicker;
    $('phaseTitle').textContent = title;
    $('phaseCopy').textContent = copy;
    $('stageFooterText').textContent = footer;
    const index = ['select','rules','ready','countdown','shaking','open','analysis','result'].indexOf(phase);
    document.querySelectorAll('#progressDots span').forEach((dot, i) => dot.classList.toggle('active', i <= Math.max(0, Math.min(5, index))));
    updateAgent(phase);
    updateScoreState(phase);
  }

  function updateAgent(phase) {
    const tasks = { taskWelcome: ['select','rules','ready'], taskCountdown: ['countdown','shaking','open'], taskVision: ['analysis'], taskResult: ['result'] };
    Object.entries(tasks).forEach(([id, phases]) => {
      const node = $(id); node.classList.toggle('active', phases.includes(phase));
      if (phase === 'result' && id === 'taskResult') node.classList.add('done');
      if (phase === 'analysis' && id === 'taskCountdown') node.classList.add('done');
    });
    const quotes = {
      select:'“欢迎来到 Dice Arena，准备好和我一起摇骰子了吗？”',
      rules:'“双方各 5 颗骰子，点数总和更大的一方获胜。”',
      ready:'“拿好骰盅，点击开始，我们马上同步摇骰。”',
      countdown:'“3、2、1，开始！”', shaking:'“摇起来！我会和你保持同步。”',
      open:'“3、2、1，停。请同时开盖。”', analysis:'“我正在看清每一颗骰子，请保持不动。”', result:'“结果出来了，恭喜获胜者！”',
    };
    $('agentQuote').textContent = quotes[phase];
  }

  function updateScoreState(phase) {
    const copy = { select:'等待游戏开始',rules:'等待确认规则',ready:'等待玩家开始',countdown:'同步倒计时中',shaking:'双方摇骰中',open:'等待开盖确认',analysis:'YOLOv8 视觉识别中',result:'本局结果已锁定' };
    $('scoreState').innerHTML = `<span class="state-dot"></span> ${copy[phase]}`;
  }

  function randomDice() { return Array.from({ length:5 }, () => Math.floor(Math.random() * 6) + 1); }
  function sum(dice) { return dice.reduce((a,b) => a + b, 0); }
  function diceMarkup(values, className='') {
    return values.map((value) => `<div class="die ${className}" aria-label="${value}点">${Array.from({length:9}, (_, i) => `<span class="${dicePips[value].includes(i) ? 'on' : ''}"></span>`).join('')}</div>`).join('');
  }
  function updateScores() {
    const player = sum(state.playerDice), agent = sum(state.agentDice);
    $('livePlayerScore').textContent = state.playerDice.length ? player : '—';
    $('liveAgentScore').textContent = state.agentDice.length ? agent : '—';
    $('playerScore').textContent = player;
    $('agentScore').textContent = agent;
    $('playerDice').innerHTML = diceMarkup(state.playerDice);
    $('agentDice').innerHTML = diceMarkup(state.agentDice, 'agent-die');
  }

  function toast(message) { const node = $('toast'); node.textContent = message; node.classList.add('show'); setTimeout(() => node.classList.remove('show'), 2200); }
  function speak(message) { if (!state.sound || !('speechSynthesis' in window)) return; window.speechSynthesis.cancel(); const utterance = new SpeechSynthesisUtterance(message); utterance.lang = 'zh-CN'; utterance.rate = .96; window.speechSynthesis.speak(utterance); }
  function countdown(next, label, hint, seconds = 3) {
    clearInterval(state.countdownTimer); setPhase('countdown'); $('countdownLabel').textContent = label; $('countdownHint').textContent = hint;
    let value = seconds; $('countdownNumber').textContent = value; $('countdownNumber').style.animation='none'; void $('countdownNumber').offsetWidth; $('countdownNumber').style.animation=''; speak(value === 3 ? `${label}，三` : String(value));
    state.countdownTimer = setInterval(() => { value -= 1; if (value > 0) { $('countdownNumber').textContent = value; $('countdownNumber').style.animation='none'; void $('countdownNumber').offsetWidth; $('countdownNumber').style.animation=''; speak(String(value)); } else { clearInterval(state.countdownTimer); next(); } }, 900);
  }
  function beginShake() { setPhase('shaking'); let seconds = 8; $('shakeSeconds').textContent = String(seconds).padStart(2, '0'); speak('开始摇骰'); clearInterval(state.shakeTimer); state.shakeTimer = setInterval(() => { seconds -= 1; $('shakeSeconds').textContent = String(Math.max(0, seconds)).padStart(2, '0'); if (seconds <= 0) stopShake(); }, 1000); }
  function stopShake() { clearInterval(state.shakeTimer); countdown(() => setPhase('open'), 'STOP COUNTDOWN', '倒计时结束后，请同时开盖。'); speak('停止摇骰'); }
  function reveal() { state.playerDice = randomDice(); state.agentDice = randomDice(); updateScores(); setPhase('analysis'); speak('正在识别骰子'); const steps = [['stepCapture', 450], ['stepDetect', 1150], ['stepJudge', 1950]]; steps.forEach(([id, delay]) => setTimeout(() => { $(id).classList.add('active'); $(id).querySelector('span').textContent = '✓'; }, delay)); setTimeout(showResult, 2750); }
  function showResult() { setPhase('result'); const player = sum(state.playerDice), agent = sum(state.agentDice); const banner = $('resultBanner'); const playerWins = player > agent, tie = player === agent; $('resultEmoji').textContent = tie ? '🤝' : playerWins ? '🏆' : '✨'; $('resultTitle').textContent = tie ? '平局！' : playerWins ? '玩家获胜' : 'Agent 获胜'; $('resultSubtitle').textContent = tie ? '双方点数相同，再来一局分出胜负。' : playerWins ? '恭喜你，点数总和更高！' : '这次差一点，和 Agent 再来一局吧。'; banner.classList.toggle('loss', !playerWins && !tie); speak(tie ? '本局平局，再来一局吧' : playerWins ? '恭喜你，玩家获胜' : '本局 Agent 获胜'); }
  function resetRound() { state.round += 1; $('roundNumber').textContent = String(state.round).padStart(2,'0'); $('roundMini').textContent = String(state.round).padStart(2,'0'); state.playerDice=[];state.agentDice=[];updateScores(); setPhase('ready'); }

  document.querySelectorAll('.game-option:not(.disabled)').forEach((button) => button.addEventListener('click', () => { document.querySelectorAll('.game-option').forEach((item) => { item.classList.toggle('selected', item === button); item.setAttribute('aria-selected', item === button ? 'true' : 'false'); }); state.selectedGame = button.dataset.game; }));
  function enterSelectedGame() { if (state.selectedGame === 'dice') { setPhase('rules'); speak('摇骰子规则'); } }
  $('startGame').addEventListener('click', enterSelectedGame);
  $('gameList').addEventListener('dblclick', enterSelectedGame);
  $('repeatRules').addEventListener('click', () => { toast('正在重复播报游戏规则'); speak('双方各摇五颗骰子，停止后同时开盖。视觉模型会识别每颗骰子的点数，点数总和更大的一方获胜。'); });
  $('confirmRules').addEventListener('click', () => { setPhase('ready'); speak('规则确认完成，请准备'); });
  $('startShake').addEventListener('click', () => countdown(beginShake, 'GET READY', '和 Agent 同步')); $('stopShake').addEventListener('click', stopShake); $('revealDice').addEventListener('click', reveal); $('newRound').addEventListener('click', resetRound); $('backToGames').addEventListener('click', () => { state.round=1; $('roundNumber').textContent='01'; $('roundMini').textContent='01'; setPhase('select'); });
  $('soundToggle').addEventListener('click', () => { state.sound=!state.sound; $('soundToggle').textContent=state.sound?'🔊':'🔇'; toast(state.sound?'语音播报已开启':'语音播报已关闭'); });
  $('cameraButton').addEventListener('click', async () => { if (state.stream) { state.stream.getTracks().forEach(t=>t.stop()); state.stream=null; $('cameraFrame').classList.remove('camera-active'); $('cameraStatus').textContent='未连接摄像头'; $('cameraButton').textContent='开启摄像头 ↗'; return; } if (!navigator.mediaDevices?.getUserMedia) { toast('当前浏览器不支持摄像头，继续使用演示模式'); return; } try { state.stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'},audio:false}); $('cameraVideo').srcObject=state.stream; $('cameraFrame').classList.add('camera-active'); $('cameraStatus').textContent='摄像头已连接'; $('cameraButton').textContent='关闭摄像头'; toast('摄像头已连接，YOLOv8 接口待接入'); } catch { toast('摄像头权限未开启，继续使用演示模式'); } });
  document.addEventListener('keydown', (event) => { if (event.key === 'ArrowUp' || event.key === 'ArrowDown') { const options=[...document.querySelectorAll('.game-option:not(.disabled)')]; const current=options.findIndex(o=>o.classList.contains('selected')); options[(current+(event.key==='ArrowDown'?1:-1)+options.length)%options.length].click(); } if (event.key === 'Enter' && state.phase === 'select') enterSelectedGame(); if (event.key.toLowerCase() === 'q' && state.phase === 'shaking') stopShake(); });
  updateScores(); setPhase('select');
})();
