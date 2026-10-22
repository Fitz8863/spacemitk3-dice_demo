// Presentation only: themes never submit intents or change round state.
(() => {
  'use strict';
  const THEME = 'dark-arena';
  const STORAGE_KEY = 'spacemit.arena.theme.v1';
  const root = document.documentElement;
  const meta = document.querySelector('meta[name="theme-color"]');
  function clearThemePreference() {
    try { localStorage.removeItem(STORAGE_KEY); } catch (_) { /* Kiosk storage may be blocked. */ }
  }
  function paint() {
    root.dataset.theme = THEME;
    if (meta) meta.content = '#2a3b34';
  }
  clearThemePreference();
  paint(); // Apply before stylesheets, including the stream-player frame.
  window.addEventListener('storage', (event) => {
    if (event.key !== STORAGE_KEY) return;
    clearThemePreference();
    paint();
  });

  const titles = {
    select: '骰王挑战赛', rules: '三步，开始挑战。',
    ready: '握好骰盅，准备开始。', rehome: '下一局，马上就绪。',
    game_start: '挑战开始。', countdown: '一起倒数，准备摇。',
    arm_failed: '机械臂动作未完成。',
  };
  const labels = {
    startGame: '开始挑战', confirmRules: '准备好了', repeatRules: '重听规则',
    startShake: '开始摇骰', gameStartText: '挑战开始',
  };
  function updateHallCards() {
    const descriptions = {
      dice: '与机械臂同台摇骰，让 AI 识别点数、揭晓胜负。',
      rps: '石头、剪刀、布，与 AI 轻松过招。',
    };
    for (const [id, copy] of Object.entries(descriptions)) {
      const node = document.querySelector(`[data-game="${id}"] .game-copy small`);
      if (!node) continue;
      node.textContent = copy;
    }
  }
  function updateLabels() {
    updateHallCards();
    for (const [id, copy] of Object.entries(labels)) {
      const node = document.getElementById(id);
      if (!node) continue;
      node.textContent = copy;
    }
    const readyTitle = document.querySelector('[data-view="ready"] .view-title');
    if (readyTitle) {
      readyTitle.textContent = titles.ready;
    }
  }

  function renderPhase(phase, gameId, selectedId) {
    updateHallCards();
    const gameName = (id) => document.querySelector(`[data-game="${id}"] .game-copy strong`)?.textContent || '游戏';
    const breadcrumb = document.getElementById('arenaBreadcrumb');
    if (breadcrumb) breadcrumb.textContent = ['select', 'standby'].includes(phase)
      ? '游戏大厅' : `${gameName(gameId)}`;
    const step = ['rules', 'ready', 'rehome'].includes(phase) ? 0
      : phase === 'analysis' ? 2 : phase === 'result' ? 3 : 1;
    const steps = document.getElementById('arenaSteps');
    if (steps) {
      steps.classList.toggle('hidden', ['select', 'standby'].includes(phase));
      steps.querySelectorAll('[data-step]').forEach((node) => {
        const active = Number(node.dataset.step) === step;
        node.classList.toggle('active', active);
        if (active) node.setAttribute('aria-current', 'step');
        else node.removeAttribute('aria-current');
      });
    }
    const selected = document.getElementById('arenaSelected');
    if (selected) selected.textContent = selectedId ? `已选择 · ${gameName(selectedId)}` : '暂无可用游戏';
  }

  window.ArenaTheme = {
    get current() { return THEME; },
    resolveMeta(phase, fallback, gameId) {
      const title = gameId === 'rps' && phase === 'rules' ? '听口令，一起出拳。' : titles[phase];
      return title ? [title, fallback[1]] : fallback;
    },
    renderPhase,
  };
  document.addEventListener('DOMContentLoaded', updateLabels);
})();
