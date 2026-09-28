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

  function connect() {
    if (disposed) return;
    const endpoint = new URL('whep', streamUrl);
    endpoint.search = streamUrl.search;
    reader = new MediaMTXWebRTCReader({
      url: endpoint.toString(),
      onError: (error) => {
        // MediaMTX retries the WHEP connection itself. Keep its technical errors
        // in the console, and keep the friendly scene covering the video.
        console.debug('Video stream reconnecting:', error);
        showLoading();
      },
      onTrack: (event) => {
        if (disposed) return;
        showLoading();
        video.srcObject = event.streams[0];
        video.play().catch(showLoading);
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
      if (!disposed) retryTimer = setTimeout(loadReader, 2000);
    };
    document.head.append(script);
  }

  window.addEventListener('pagehide', () => {
    disposed = true;
    clearTimeout(retryTimer);
    if (frameCallback !== null) video.cancelVideoFrameCallback(frameCallback);
    if (reader !== null) reader.close();
    video.srcObject = null;
  });
  loadReader();
})();
