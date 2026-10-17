// Screen placement is independent of the physical sides used for adjudication.
export function resolveDisplayLayout(manifest) {
  const display = manifest.display || {
    player_side: manifest.participants.player,
    agent_side: manifest.participants.agent,
    video_rotation_deg: 0,
  };
  if (!['LEFT', 'RIGHT'].includes(display.player_side)
      || !['LEFT', 'RIGHT'].includes(display.agent_side)
      || display.player_side === display.agent_side
      || ![0, 90, 180, 270].includes(display.video_rotation_deg)) {
    throw new Error('游戏显示方向配置无效');
  }
  return { ...display };
}

export async function refreshDisplayManifest(manifest, requestJson) {
  // The game list can stay open for hours after its initial configuration load.
  // Refresh presentation at entry, keeping the round's physical role mapping.
  try {
    const payload = await requestJson('/api/games', {
      cache: 'no-store', signal: AbortSignal.timeout(2000),
    });
    const current = payload.games?.find(game => game.id === manifest.id);
    if (current?.display) {
      return { ...manifest, display: resolveDisplayLayout(current) };
    }
  } catch (error) {
    // A temporary network failure must not prevent an otherwise valid entry.
    console.warn('Display configuration refresh failed; keeping loaded layout', error);
  }
  return manifest;
}

export function applyScoreLayout(playerCard, agentCard, display) {
  // Explicit rows prevent CSS grid auto-placement from moving reversed cards
  // onto separate rows. Names, scores, gestures and winner roles stay intact.
  playerCard.style.gridColumn = display.player_side === 'LEFT' ? '1' : '3';
  agentCard.style.gridColumn = display.agent_side === 'LEFT' ? '1' : '3';
  playerCard.style.gridRow = '1';
  agentCard.style.gridRow = '1';
}
