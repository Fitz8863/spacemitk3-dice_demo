// Run against tests/support/shared_browser_server.py, never the board service.
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const base = process.argv[2] || 'http://127.0.0.1:18681';
let browser;
(async () => {
  browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true,
    args: ['--no-sandbox', '--autoplay-policy=user-gesture-required'] });
  const errors = [], actions = [], streams = [];
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  async function page() {
    const p = await context.newPage();
    p.on('pageerror', (error) => errors.push(error.message));
    p.on('request', (req) => {
      if (req.url().endsWith('/api/game/rounds') && req.method() === 'POST') actions.push(req.url());
      if (req.url().includes('/api/game/rounds/') && req.url().endsWith('/stream')) streams.push(req.url());
    });
    await p.goto(base, { waitUntil: 'domcontentloaded' });
    await p.waitForTimeout(500);
    return p;
  }
  const session = async () => (await context.request.get(`${base}/api/session`)).json();
  async function phase(p, expected) {
    await p.waitForFunction((v) => document.body.dataset.phase === v, expected);
  }
  async function selected(p, game) {
    await p.waitForFunction((id) => document.querySelector(`.game-option[data-game="${id}"]`)?.getAttribute('aria-selected') === 'true', game);
  }
  await context.request.post(`${base}/test/reset`, { data: {} });
  const a = await page(), b = await page();
  await phase(a, 'standby'); await phase(b, 'standby');
  // A board direction wakes both pages and navigates both to RPS.
  await context.request.post(`${base}/test/board`, { data: { key: 'down' } });
  await phase(a, 'select'); await phase(b, 'select');
  await selected(a, 'rps'); await selected(b, 'rps');
  await Promise.all([a.keyboard.press('Enter'), b.keyboard.press('Enter')]);
  await phase(a, 'rules'); await phase(b, 'rules');
  const first = await session();
  const beforeRps = await (await context.request.get(`${base}/test/metrics`)).json();
  assert.equal(first.game_id, 'rps');
  assert.equal(streams.length, 0, 'one session SSE carries round updates for every display');
  assert.equal(actions.length, 0, 'shared displays must not create private rounds');
  // A fresh display has no user activation; it must offer an unlock control.
  const c = await page();
  await phase(c, 'rules');
  const unlock = c.locator('#audioUnlock');
  if (await unlock.isVisible().catch(() => false)) {
    await unlock.click();
    await unlock.waitFor({ state: 'hidden' });
  }
  await a.reload({ waitUntil: 'domcontentloaded' });
  await a.waitForTimeout(500);
  await phase(a, 'rules');
  assert.equal((await session()).round_id, first.round_id);
  await b.close();
  assert.equal((await session()).round_id, first.round_id);
  // Remote confirmation and board confirmation both reach the one round.
  await c.keyboard.press('Enter');
  await phase(a, 'result'); await phase(c, 'result');
  let metrics = await (await context.request.get(`${base}/test/metrics`)).json();
  const newRobotCalls = metrics.robot_calls.slice(beforeRps.robot_calls.length);
  const newProductions = metrics.productions.slice(beforeRps.productions.length);
  assert.equal(newRobotCalls.filter((c) => c === 'throw_gesture').length, 1);
  assert.equal(new Set(newProductions.map((v) => v.join(':'))).size, newProductions.length);
  // Joining at result restores the result instead of replaying the rules.
  const d = await page(); await phase(d, 'result');
  await d.keyboard.press('Escape');
  await phase(a, 'select'); await phase(c, 'select'); await phase(d, 'select');
  await selected(a, 'dice'); await selected(c, 'dice');
  await context.request.post(`${base}/test/board`, { data: { key: 'confirm' } });
  await phase(a, 'rules'); await phase(c, 'rules');
  assert.equal((await session()).game_id, 'dice');
  await c.keyboard.press('Enter');
  await phase(a, 'ready'); await phase(c, 'ready');
  await c.keyboard.press('Enter');
  await phase(a, 'result'); await phase(c, 'result');
  assert.equal(await a.locator('#resultTitle').textContent(), await c.locator('#resultTitle').textContent());
  await a.screenshot({ path: '/tmp/k3-shared-session-browser.png', fullPage: true });
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ verdict: 'PASS', clients: 4, private_round_requests: actions.length,
    checks: ['board navigation', 'parallel Enter', 'same round stream', 'autoplay unlock', 'reload', 'close one display',
      'RPS simulated throw once', 'late result join', 'shared exit', 'default dice', 'dice result'], page_errors: errors }));
  await browser.close();
})().catch(async (error) => { console.error(error); await browser?.close(); process.exitCode = 1; });
