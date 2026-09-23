// 猜拳游戏模块：后端权威状态机的声明式前端（视觉接入前的骨架）。
// 后端 pipeline 当前用判定桩（source:"stub"）出结果，但事件与结果字段的
// 形状与真实视觉流一致；视觉模型接入后本模块无需结构改动。
// rules/analysis/result 三个视图与 dice 共用同一份 DOM：进入本游戏时把
// dice 专属文案换成猜拳措辞，teardown 必须还原。
export function register(engine) {
  const {
    state, $, setPhase, toast, stopSpeech,
    returnToSelect, createRoundClient, playDirective, setActiveRound, setIdleReturn,
  } = engine;

  const GESTURE_GLYPHS = { '石头': '✊', '剪刀': '✌', '布': '🖐' };

  // 状态名 → 视图名。play 的视图与状态同名（ui.view 也写 "play"），重连
  // 快照路径（onSyncState）只有状态名，仅 analysis_failed 需要映射。
  const VIEW_BY_STATE = { 'analysis_failed': 'analysis' };

  // 后端 ui 文案缺省时的前端兜底。
  const phaseMeta = {
    rules: ['游戏规则', '听完规则后按 Enter 开始，按 ↓ 可以再听一次。'],
    play: ['石头 · 剪刀 · 布', '听口令出拳，念到「布」时亮出手势。'],
    analysis: ['正在判定胜负', '视觉裁决器正在识别双方手势并判定胜负。'],
    result: ['本局结果', ''],
  };

  const RULES_MARKUP = [
    '<div class="rules-item"><span>01</span><div><strong>听口令出拳</strong><p>听语音念「石头，剪刀，布」，随时可以按绿色按钮开始。</p></div></div>',
    '<div class="rules-item"><span>02</span><div><strong>同时亮手势</strong><p>念到「布」时双方同时亮出手势，不要提前或延后。</p></div></div>',
    '<div class="rules-item"><span>03</span><div><strong>视觉判定</strong><p>石头赢剪刀，剪刀赢布，布赢石头；相同的手势是平局。</p></div></div>',
  ].join('');

  let round = null;
  let lastRenderedState = '';
  let visionStreamToken = 0;
  let participantSides = null;
  let savedRulesMarkup = '';
  let savedDetectLabel = '';

  function submitIntent(intent, payload = {}) {
    if (!round) return Promise.resolve();
    return round.submitIntent(intent, payload).catch((error) => {
      if (error.silent) {
        // 回合已终结（ROUND_CLOSED）后一切意图都被静默拒绝——错误页假死
        // 的根因；此时任何按键都导航回列表（与 dice 同款修复）。
        if (error.code === 'ROUND_CLOSED') returnToSelect();
        return; // 按键时机不合状态属正常对局
      }
      console.error(`Intent ${intent} failed:`, error);
    });
  }

  const handlers = {
    confirmRules: () => submitIntent('confirm'),
    repeatRules: () => {
      toast('正在重复播报游戏规则');
      submitIntent('repeat');
    },
    backFromRules: () => submitIntent('back'),
    newRound: () => submitIntent('new_round'),
    backToGames: () => submitIntent('back'),
    analysisRetry: () => submitIntent('retry'),
    analysisNewRound: () => submitIntent('new_round'),
    analysisBackToGames: () => submitIntent('back'),
  };

  // ---- 实时画面（MediaMTX WebRTC iframe，与 dice.js 同款共享面板） ----
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
      if (status) status.textContent = '播放页面已加载，等待手势画面…';
    };
    frame.onerror = () => {
      if (token !== visionStreamToken || state.phase !== 'analysis') return;
      if (status) status.textContent = '实时画面连接失败，识别仍会继续';
    };
    frame.src = streamUrl.toString();
  }

  function analysisFailureVisible() {
    const actions = $('analysisFailureActions');
    return actions && !actions.classList.contains('hidden');
  }

  function onKey(event) {
    if (event.key === 'Escape') {
      if (['rules', 'result'].includes(state.phase)) submitIntent('back');
      else if (state.phase === 'analysis' && analysisFailureVisible()) submitIntent('back');
      return;
    }
    if (event.key === 'Enter') {
      // 绿键/Enter 与绿色按钮同义：规则确认、结果页/识别失败页再来一局。
      if (state.phase === 'rules') handlers.confirmRules();
      else if (state.phase === 'result') handlers.newRound();
      else if (state.phase === 'analysis' && analysisFailureVisible()) handlers.analysisNewRound();
      return;
    }
    if (event.key === 'ArrowDown') {
      if (state.phase === 'rules') handlers.repeatRules();
      else if (state.phase === 'analysis' && analysisFailureVisible()) handlers.analysisRetry();
    }
  }

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

  function renderState(stateName, ui) {
    if (stateName === lastRenderedState) return;
    lastRenderedState = stateName;
    if (stateName !== 'analysis' && stateName !== 'analysis_failed') stopVisionStream();
    const view = ui.view || VIEW_BY_STATE[stateName] || stateName;
    const meta = [ui.title || '', ui.copy || ''];
    setPhase(view, meta[0] || meta[1] ? meta : undefined);
    // 等玩家操作的状态没有 duration/on_expire，玩家走开后会永远停住；交给
    // 引擎挂空闲退出计时器（超时取消回合回列表），离开即解除。
    setIdleReturn(['rules', 'result', 'analysis_failed'].includes(stateName));
    if (stateName === 'analysis') resetAnalysisSteps();
  }

  function resetAnalysisSteps() {
    $('stepCapture').classList.add('active');
    $('stepCapture').classList.remove('failed');
    $('stepCapture').querySelector('span').textContent = '✓';
    $('stepDetect').classList.remove('active', 'failed');
    $('stepDetect').querySelector('span').textContent = '2';
    $('stepJudge').classList.remove('active', 'failed');
    $('stepJudge').querySelector('span').textContent = '3';
    $('analysisTitle').textContent = '正在识别手势';
    $('analysisFailureActions').classList.add('hidden');
    document.querySelector('.analysis-spinner')?.classList.remove('hidden');
    $('analysisStatus').textContent = '正在识别双方手势…';
  }

  function updateAnalysisProgress(event) {
    if (event.phase === 'detecting') {
      $('stepDetect').classList.add('active');
      $('stepDetect').querySelector('span').textContent = '…';
      $('analysisStatus').textContent = '正在识别双方手势…';
    } else if (event.phase === 'verifying') {
      $('stepDetect').classList.add('active');
      $('stepDetect').querySelector('span').textContent = '✓';
      $('stepJudge').classList.add('active');
      $('stepJudge').querySelector('span').textContent = '…';
      $('analysisStatus').textContent = event.llm === false
        ? '手势识别完成，正在判定胜负…'
        : '手势识别完成，正在调用大模型复核…';
    } else if (event.phase === 'holding') {
      $('stepDetect').classList.add('active');
      $('stepDetect').querySelector('span').textContent = '✓';
      $('stepJudge').classList.add('active');
      $('stepJudge').querySelector('span').textContent = '✓';
      $('analysisStatus').textContent = '结果已锁定…';
    }
  }

  function gestureMarkup(choice) {
    const glyph = GESTURE_GLYPHS[choice] || '❔';
    return `<div class="rps-gesture" aria-label="${choice}">${glyph}</div>`;
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
    $('playerDice').innerHTML = gestureMarkup(result.player_choice);
    $('agentDice').innerHTML = gestureMarkup(result.agent_choice);
    $('playerScore').textContent = result.player_choice || '—';
    $('agentScore').textContent = result.agent_choice || '—';
    const banner = $('resultBanner');
    const winnerRole = result.winner_role;
    const tie = winnerRole === 'TIE';
    const playerWins = winnerRole === 'PLAYER';
    if (!tie && !playerWins && winnerRole !== 'AGENT') {
      throw new Error('裁决结果缺少有效 winner_role');
    }
    $('resultEmoji').textContent = tie ? '🤝' : playerWins ? '🏆' : '✨';
    $('resultTitle').textContent = tie ? '平局！' : playerWins ? '玩家获胜' : 'Agent 获胜';
    const stubNote = result.source === 'stub' ? '（演示判定，视觉模型尚未接入）' : '';
    $('resultSubtitle').textContent = tie
      ? `双方都出了${result.player_choice}${stubNote}，再来一局？`
      : `${result.player_choice} 对 ${result.agent_choice}${stubNote}`;
    banner.classList.toggle('loss', !playerWins && !tie);
  }

  function showDiagnosis(result) {
    const diagnosis = result && result.diagnosis && typeof result.diagnosis === 'object'
      ? result.diagnosis : {};
    $('stepDetect').classList.add('active', 'failed');
    $('stepDetect').querySelector('span').textContent = '✕';
    $('stepJudge').classList.remove('active');
    document.querySelector('.analysis-spinner')?.classList.add('hidden');
    $('analysisTitle').textContent = '本次裁决未完成';
    $('analysisStatus').textContent = diagnosis.message
      || '当前画面无法形成稳定识别结果，请重新开始。';
    $('analysisFailureActions').classList.remove('hidden');
  }

  function handleRoundEvent(event, snapshot) {
    if (event.event === 'video') {
      startVisionStream(event);
    } else if (event.event === 'phase' || event.event === 'progress') {
      updateAnalysisProgress(event);
    }
  }

  // ---- 对局生命周期 ----
  async function enter(manifest) {
    configureParticipants(manifest);
    // 共享视图换上猜拳措辞（HTML 默认是 dice 的），teardown 还原。
    const rulesList = document.querySelector('.rules-list');
    savedRulesMarkup = rulesList.innerHTML;
    rulesList.innerHTML = RULES_MARKUP;
    const detectLabel = $('stepDetect').querySelector('strong');
    savedDetectLabel = detectLabel.textContent;
    detectLabel.textContent = '识别手势';
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
        }
      },
      onSpeech: (directive) => {
        playDirective(round, directive);
      },
      onTick: () => {},
      onEvent: handleRoundEvent,
      onComplete: (event) => {
        if (event.status === 'error') {
          document.querySelector('.analysis-spinner')?.classList.add('hidden');
          $('analysisTitle').textContent = '识别未完成';
          $('analysisStatus').textContent = '裁决异常结束，请重新开始一局';
          $('analysisFailureActions').classList.remove('hidden');
          // 回合已终结但界面停在失败页等玩家操作，同样需要空闲退出兜底。
          setIdleReturn(true);
          toast('裁决失败');
          return;
        }
        // exited / cancelled: the player left, follow them back to the list.
        returnToSelect();
      },
      onSyncState: (snapshot) => {
        if (!round || !round.roundId) return;
        // 重连快照只有状态名，renderState 会用 VIEW_BY_STATE 补视图名。
        if (snapshot.state && snapshot.state !== lastRenderedState) {
          renderState(snapshot.state, {});
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
    // 还原共享视图的 dice 默认文案（guards 覆盖 enter 中途失败的情况）。
    if (savedRulesMarkup) {
      document.querySelector('.rules-list').innerHTML = savedRulesMarkup;
      savedRulesMarkup = '';
    }
    if (savedDetectLabel) {
      $('stepDetect').querySelector('strong').textContent = savedDetectLabel;
      savedDetectLabel = '';
    }
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
    id: 'rps',
    phases: ['select', 'rules', 'rps-play', 'analysis', 'result'],
    progressCount: 4,
    phaseMeta,
    enter,
    teardown,
    onKey,
  };
}
