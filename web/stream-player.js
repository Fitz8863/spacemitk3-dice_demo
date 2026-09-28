// Presentation-only WebRTC player. Camera inference and game state stay upstream.
(() => {
  const video = document.getElementById('video');
  const loading = document.getElementById('loading');
  const rotation = Number(new URLSearchParams(location.search).get('rotation') || 0);
  const angle = [0, 90, 180, 270].includes(rotation) ? rotation : 0;
  video.style.transform = `translate(-50%, -50%) rotate(${angle}deg)`;
  // The video is portrait before quarter-turns; swap its box, not the loader.
  video.style.width = angle % 180 ? '100vh' : '100vw';
  video.style.height = angle % 180 ? '100vw' : '100vh';
  let reader = null;
  let retryTimer = null;
  let frameCallback = null;
  let disposed = false;
  let connectionGeneration = 0;
  let retryAttempts = 0;
  let streamUrl;

  function showLoading() {
    if (disposed) return;
    loading.hidden = false;
    if (frameCallback !== null) video.cancelVideoFrameCallback(frameCallback);
    frameCallback = null;
  }

  function revealFrame() {
    frameCallback = null;
    if (!disposed && video.readyState >= 2 && video.videoWidth > 0) {
      loading.hidden = true;
      retryAttempts = 0;
    }
  }

  function awaitFrame() {
    if (disposed) return;
    if (typeof video.requestVideoFrameCallback === 'function') {
      if (frameCallback === null) frameCallback = video.requestVideoFrameCallback(revealFrame);
    } else {
      // Older browsers: playing + decoded data, never iframe.onload/onTrack alone.
      revealFrame();
    }
  }

  video.addEventListener('playing', awaitFrame);
  video.addEventListener('waiting', showLoading);
  video.addEventListener('emptied', showLoading);
  video.addEventListener('ended', showLoading);
  video.addEventListener('error', showLoading);
  document.addEventListener('visibilitychange', () => {
    document.body.classList.toggle('offscreen', document.hidden);
  });

  try {
    const configured = new URLSearchParams(location.search).get('stream');
    if (!configured) return;
    streamUrl = new URL(configured, location.href);
    if (!['http:', 'https:'].includes(streamUrl.protocol)) return;
    if (!streamUrl.pathname.endsWith('/')) streamUrl.pathname += '/';
  } catch (_) {
    return;
  }

  function closeReader() {
    // Invalidate callbacks before closing: a retired connection must not
    // replace the new track or schedule a second retry.
    connectionGeneration += 1;
    const previous = reader;
    reader = null;
    if (previous !== null) previous.close();
  }

  function retry(action) {
    if (disposed) return;
    clearTimeout(retryTimer);
    // Cold camera startup normally takes a few seconds. Probe quickly in
    // that window; back off if the camera/server remains unavailable.
    const delay = retryAttempts++ < 20 ? 250 : 2000;
    retryTimer = setTimeout(() => {
      retryTimer = null;
      if (!disposed) action();
    }, delay);
  }

  function connect() {
    if (disposed) return;
    const endpoint = new URL('whep', streamUrl);
    endpoint.search = streamUrl.search;
    const generation = ++connectionGeneration;
    reader = new MediaMTXWebRTCReader({
      url: endpoint.toString(),
      onError: (error) => {
        if (disposed || generation !== connectionGeneration) return;
        console.debug('Video stream reconnecting:', error);
        showLoading();
        // close() also cancels MediaMTX's built-in 2-second retry timer.
        closeReader();
        retry(connect);
      },
      onTrack: (event) => {
        if (disposed || generation !== connectionGeneration) return;
        showLoading();
        video.srcObject = event.streams[0];
        video.play().catch(() => {
          if (generation === connectionGeneration) showLoading();
        });
      },
    });
  }

  function loadReader() {
    if (disposed) return;
    const script = document.createElement('script');
    script.src = new URL('reader.js', streamUrl).toString();
    script.onload = connect;
    script.onerror = () => {
      script.remove();
      retry(loadReader);
    };
    document.head.append(script);
  }

  window.addEventListener('pagehide', () => {
    disposed = true;
    clearTimeout(retryTimer);
    if (frameCallback !== null) video.cancelVideoFrameCallback(frameCallback);
    closeReader();
    video.srcObject = null;
  });
  loadReader();
})();
