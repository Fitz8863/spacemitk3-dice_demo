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

export function applyScoreLayout(playerCard, agentCard, display) {
  // Explicit rows prevent CSS grid auto-placement from moving reversed cards
  // onto separate rows. Names, scores, gestures and winner roles stay intact.
  playerCard.style.gridColumn = display.player_side === 'LEFT' ? '1' : '3';
  agentCard.style.gridColumn = display.agent_side === 'LEFT' ? '1' : '3';
  playerCard.style.gridRow = '1';
  agentCard.style.gridRow = '1';
}
