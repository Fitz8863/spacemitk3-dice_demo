// Presentation only: resize one existing video panel without touching its stream.
(() => {
  'use strict';
  function init() {
    const panel = document.getElementById('analysisStreamPanel');
    if (!panel) return;
    const body = document.body;
    const root = document.documentElement;
    const failure = document.getElementById('analysisFailureActions');
    const judge = document.getElementById('stepJudge');
    const playing = new Set(['game_start', 'countdown', 'shaking', 'play', 'rps-play']);
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    let animation = null;
    const originalWillChange = panel.style.willChange;

    function stopMotion() {
      if (animation) animation.cancel();
      animation = null;
      panel.style.willChange = originalWillChange;
    }

    function sync() {
      const arena = root.dataset.theme === 'dark-arena';
      const visible = !panel.classList.contains('hidden');
      const detecting = body.dataset.phase === 'analysis'
        && (!failure || failure.classList.contains('hidden'))
        && (!judge || !judge.classList.contains('active'));
      const focus = arena && visible && (playing.has(body.dataset.phase) || detecting);
      if (body.classList.contains('arena-video-focus') === focus) return;

      // Read the currently displayed bounds once, including an interrupted
      // transition. Change layout once, then animate only its transform (FLIP).
      const animate = arena && visible && !reducedMotion.matches
        && typeof panel.animate === 'function';
      const first = animate ? panel.getBoundingClientRect() : null;
      stopMotion();
      body.classList.toggle('arena-video-focus', focus);
      if (!first || first.width <= 0 || first.height <= 0) return;
      const last = panel.getBoundingClientRect();
      if (last.width <= 0 || last.height <= 0) return;
      if (first.width === last.width && first.height === last.height
          && first.left === last.left && first.top === last.top) return;
      panel.style.willChange = 'transform';
      const motion = panel.animate([
        { transformOrigin: '0 0', transform: `translate(${first.left - last.left}px, ${first.top - last.top}px) scale(${first.width / last.width}, ${first.height / last.height})` },
        { transformOrigin: '0 0', transform: 'none' },
      ], { duration: 180, easing: 'ease-out' });
      animation = motion;
      motion.onfinish = () => {
        if (animation !== motion) return;
        animation = null;
        panel.style.willChange = originalWillChange;
      };
    }

    // Watch presentation state only. Result locking happens before the result
    // page; failure actions can appear while the phase still reads "analysis".
    const observer = new MutationObserver(sync);
    observer.observe(body, { attributes: true, attributeFilter: ['data-phase'] });
    observer.observe(root, { attributes: true, attributeFilter: ['data-theme'] });
    for (const node of [panel, failure, judge]) {
      if (node) observer.observe(node, { attributes: true, attributeFilter: ['class'] });
    }
    reducedMotion.addEventListener('change', (event) => { if (event.matches) stopMotion(); });
    window.addEventListener('pagehide', stopMotion);
    sync();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
  else init();
})();
