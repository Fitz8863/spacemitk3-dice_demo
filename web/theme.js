// Presentation only: themes never submit intents or change round state.
(() => {
  'use strict';
  const DEFAULT_THEME = 'dark-arena';
  const STORAGE_KEY = 'spacemit.arena.theme.v1';
  const THEMES = ['dark-arena', 'storybook'];
  const isTheme = (value) => THEMES.includes(value);
  let saved;
  try { saved = localStorage.getItem(STORAGE_KEY); } catch (_) { /* Kiosk storage may be blocked. */ }
  let current = isTheme(saved) ? saved : DEFAULT_THEME;
  const root = document.documentElement;
  const meta = document.querySelector('meta[name="theme-color"]');
  function paint() {
    root.dataset.theme = current;
    if (meta) meta.content = current === 'dark-arena' ? '#2a3b34' : '#f7f9fc';
  }
  paint(); // Run before stylesheets to avoid a light flash on kiosk startup.
  // Same-origin stream-player frames share the choice without reconnecting video.
  window.addEventListener('storage', (event) => {
    if (event.key !== STORAGE_KEY) return;
    current = isTheme(event.newValue) ? event.newValue : DEFAULT_THEME;
    paint();
    const select = document.getElementById('themeSelect');
    if (select) select.value = current;
    updateLabels();
    window.dispatchEvent(new Event('arena-theme-change'));
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
  const originals = new Map();
  const cardOriginals = new WeakMap();
  function updateHallCards() {
    const descriptions = {
      dice: '与机械臂同台摇骰，让 AI 识别点数、揭晓胜负。',
      rps: '石头、剪刀、布，与 AI 轻松过招。',
    };
    for (const [id, copy] of Object.entries(descriptions)) {
      const node = document.querySelector(`[data-game="${id}"] .game-copy small`);
      if (!node) continue;
      if (!cardOriginals.has(node)) cardOriginals.set(node, node.textContent);
      node.textContent = current === 'dark-arena' ? copy : cardOriginals.get(node);
    }
  }
  function updateLabels() {
    updateHallCards();
    for (const [id, copy] of Object.entries(labels)) {
      const node = document.getElementById(id);
      if (!node) continue;
      if (!originals.has(id)) originals.set(id, node.textContent);
      node.textContent = current === 'dark-arena' ? copy : originals.get(id);
    }
    const readyTitle = document.querySelector('[data-view="ready"] .view-title');
    if (readyTitle) {
      if (!originals.has('readyTitle')) originals.set('readyTitle', readyTitle.textContent);
      readyTitle.textContent = current === 'dark-arena' ? titles.ready : originals.get('readyTitle');
    }
  }

  function renderPhase(phase, gameId, selectedId) {
    updateHallCards();
    const gameName = (id) => document.querySelector(`[data-game="${id}"] .game-copy strong`)?.textContent || '游戏';
    const breadcrumb = document.getElementById('arenaBreadcrumb');
    if (breadcrumb) breadcrumb.textContent = ['select', 'standby'].includes(phase)
      ? '游乐舱 / 游戏大厅' : `游乐舱 / ${gameName(gameId)}`;
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
    get current() { return current; },
    set(value) {
      if (!isTheme(value)) return;
      current = value;
      try { localStorage.setItem(STORAGE_KEY, current); } catch (_) { /* Keep the in-memory choice. */ }
      paint();
      const select = document.getElementById('themeSelect');
      if (select) select.value = current;
      updateLabels();
      window.dispatchEvent(new Event('arena-theme-change'));
    },
    resolveMeta(phase, fallback, gameId) {
      if (current !== 'dark-arena') return fallback;
      const title = gameId === 'rps' && phase === 'rules' ? '听口令，一起出拳。' : titles[phase];
      return title ? [title, fallback[1]] : fallback;
    },
    renderPhase,
  };
  document.addEventListener('DOMContentLoaded', () => {
    updateLabels();
    const select = document.getElementById('themeSelect');
    if (select) {
      select.value = current;
      select.addEventListener('change', () => window.ArenaTheme.set(select.value));
      // The selector must not forward navigation keys to physical-button handlers.
      for (const name of ['keydown', 'keyup']) select.addEventListener(name, (event) => event.stopPropagation());
    }
  });
})();
