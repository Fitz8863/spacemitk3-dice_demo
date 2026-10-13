// Shared session transport. Browser input never creates a private game round.
export function createSessionClient({ requestJson, onSnapshot, onConnection = () => {} }) {
  let snapshot = null;
  let source = null;
  let pending = Promise.resolve();
  let localClient = false;
  const roundSubscribers = new Set();
  let nonce = 0;
  const clientId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;

  function ingest(value) {
    if (!value || typeof value !== 'object' || !value.session_id) return;
    if (snapshot?.session_id === value.session_id && Number(value.revision) < Number(snapshot.revision)) return;
    if ('local_client' in value) localClient = value.local_client === true;
    snapshot = value;
    onSnapshot(value);
    for (const listener of roundSubscribers) listener(value);
  }

  async function refresh() {
    ingest(await requestJson('/api/session'));
  }

  async function connect() {
    await refresh();
    source = new EventSource('/api/session/stream');
    const receive = (event) => {
      try { ingest(JSON.parse(event.data)); } catch (error) { console.error('Session snapshot failed:', error); }
    };
    source.addEventListener('snapshot', receive);
    source.addEventListener('update', receive);
    source.onopen = () => onConnection(true);
    source.onerror = () => onConnection(false);
  }

  function action(name, data = {}) {
    const command = { action: name, ...data, session_id: snapshot?.session_id,
      request_id: `${clientId}:${++nonce}` };
    // Capture the state the user actually saw, not the state when a delayed
    // request reaches the head of the queue.
    if (name === 'intent' || name === 'exit') {
      command.round_id ??= snapshot?.round_id;
      command.state ??= snapshot?.round?.state;
      command.state_version ??= snapshot?.round?.state_version;
    }
    const send = async () => {
      let response;
      try {
        response = await requestJson('/api/session/actions', { method: 'POST', body: JSON.stringify(command) });
      } catch (error) {
        if (error.code === 'SESSION_STALE') await refresh();
        // One transport retry retains the request id. Business rejections are final.
        if (error.code) throw error;
        response = await requestJson('/api/session/actions', { method: 'POST', body: JSON.stringify(command) });
      }
      ingest(response);
      return response;
    };
    const result = pending.then(send);
    pending = result.catch(() => {});
    return result;
  }

  return { connect, action, refresh,
    subscribeRound(roundId, receive) {
      const listener = (value) => { if (value.round_id === roundId && value.round) receive(value.round); };
      roundSubscribers.add(listener);
      return () => roundSubscribers.delete(listener);
    },
    close() { source?.close(); roundSubscribers.clear(); },
    get snapshot() { return snapshot; }, get shared() { return snapshot?.shared === true; },
    get localClient() { return localClient; }, get ready() { return source?.readyState === 1; } };
}
