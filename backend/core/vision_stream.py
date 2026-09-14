"""Vision streaming lifecycle: bring a game's camera up when it is entered.

Entering a game should already show the table.  The YOLO runtime can capture
frames and publish RTSP without running inference (it is spawned in its
``--prewarm`` mode, where ``adjudication_active`` stays false), so the camera no
longer has to wait for the first adjudication phase to exist.

This manager owns *when* that stream starts and stops; the provider owns *how*
(``start_streaming``/``stop_streaming``).  Keeping the trigger outside the
provider means the adjudication path is untouched: a round that finds a warm
runtime simply reuses it, and a round in a deployment with no warm stream still
starts one lazily exactly as before.

The lifecycle policy is deployment-wide (``vision_always_on`` in
``backend/config.json``):

* ``true``  -- start on game entry and keep the stream for the whole process,
  across rounds and games.
* ``false`` -- start on game entry, but tear it down once the round reaches a
  terminal status, so the camera follows the game's lifetime.

The policy is read when a round starts, so switching it takes effect from the
next game: ``false`` reaches its teardown on the next round that ends, and
``true`` simply stops scheduling teardowns.

Failures here are deliberately swallowed.  A busy or unplugged camera must
never stop a player from entering a game; adjudication reports the real error
in its own phase.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Mapping

from core.arena_config import arena_vision_always_on
from core.errors import DiceArenaError
from core.games import resolve_provider_id

_TERMINAL_STATUSES = {"exited", "cancelled", "error"}
_WATCHER_POLL_SECONDS = 2.0


class VisionStreamManager:
    """Starts a game's camera/RTSP stream when the game is entered.

    Only one round owns the stream at a time.  ``create_round`` cancels any
    leftover active round while creating the new one, so a teardown scheduled by
    the old round can fire *after* the new round already started its stream;
    every teardown therefore re-checks ownership under the lock and an outdated
    watcher exits without touching the new round's runtime.
    """

    def __init__(
        self,
        *,
        components: Any,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self._components = components
        self._log = log or (lambda line: print(f"[vision] {line}", flush=True))
        self._lock = threading.Lock()
        self._provider: Any = None
        self._provider_id = ""
        self._owner_round_id: str | None = None
        self._watcher: threading.Thread | None = None

    def start_for_round(self, round_: Any, *, arena: Mapping[str, Any] | None = None) -> bool:
        """Bring this round's game camera up; True when a stream is active.

        Called on game entry, before any adjudication.  The runtime is spawned
        in prewarm mode, so the detector session is loaded but no inference runs
        until the adjudication phase asks for it.

        Re-entering a game does not restart the camera: the provider reuses a
        warm runtime whose launch signature still matches, and rebuilds only
        when the profile's model/camera settings changed.

        With ``vision_always_on`` false, a watcher is also armed to release the
        stream when this round ends.  With it true no watcher exists at all, so
        the stream stays for the process lifetime.
        """
        profile = (round_.manifest or {}).get("vision_profile")
        if not isinstance(profile, Mapping):
            # A game without a visual profile (for example rps today) has no
            # camera of its own; never start one on its behalf.
            return False
        provider_id = resolve_provider_id(round_.manifest, "vision_adjudicator")
        if not provider_id:
            self._log("game declares no vision_adjudicator slot; no stream started")
            return False
        try:
            provider = self._components.require(
                provider_id, expected_type="vision", expected_role="adjudicator"
            )
        except DiceArenaError as exc:
            self._log(f"vision provider {provider_id} unavailable: {exc.message}")
            return False
        start = getattr(provider, "start_streaming", None)
        if not callable(start):
            # Cloud/fixture providers have no resident camera.
            return False
        try:
            started = bool(start(profile, on_log=self._log))
        except Exception as exc:  # a provider bug must not break game entry
            self._log(f"stream start raised for round {str(round_.id)[:8]}: {exc!r}")
            return False
        if not started:
            # The provider refused (no resident camera, or nothing to start), so
            # there is no stream to own and no later teardown to hand it.
            return False
        always_on = arena_vision_always_on(arena)
        with self._lock:
            self._provider = provider
            self._provider_id = provider_id
            self._owner_round_id = round_.id
            if always_on:
                self._watcher = None
            else:
                self._watcher = threading.Thread(
                    target=self._watch_round,
                    args=(round_, provider),
                    daemon=True,
                    name="vision-stream",
                )
        self._log(
            f"stream up for round {str(round_.id)[:8]} (provider {provider_id}, "
            f"vision_always_on={always_on})"
        )
        if not always_on:
            watcher = self._watcher
            if watcher is not None:
                watcher.start()
        return True

    def _watch_round(self, round_: Any, provider: Any) -> None:
        """Release the stream once this round is over (game-lifetime policy).

        Exits without touching anything when a newer round has taken ownership,
        which is the normal case when a player leaves and immediately enters
        another game: ``create_round`` cancels the old round and starts the new
        stream before this poll wakes up.
        """
        while True:
            time.sleep(_WATCHER_POLL_SECONDS)
            with self._lock:
                if self._owner_round_id != round_.id or self._provider is not provider:
                    return
            if round_.status in _TERMINAL_STATUSES:
                self._release_owned_stream(round_.id, provider)
                return

    def _release_owned_stream(self, round_id: str, provider: Any) -> bool:
        """Stop the stream only if ``round_id`` is still the owner.

        Ownership is re-checked under the lock so a teardown decided for an
        older round can never release the stream a newer round already owns.
        """
        with self._lock:
            if self._owner_round_id != round_id or self._provider is not provider:
                return False
            self._provider = None
            self._provider_id = ""
            self._owner_round_id = None
            self._watcher = None
        stop = getattr(provider, "stop_streaming", None)
        if callable(stop):
            try:
                stop()
            except Exception as exc:
                self._log(f"stream stop failed: {exc!r}")
        self._log(f"stream down for round {str(round_id)[:8]}")
        return True

    def stop(self) -> None:
        """Detach the current owner and stop its stream.

        Called at process shutdown and whenever a new round replaces the
        previous owner.
        """
        with self._lock:
            provider = self._provider
            round_id = self._owner_round_id
            watcher = self._watcher
            self._provider = None
            self._provider_id = ""
            self._owner_round_id = None
            self._watcher = None
        if provider is not None:
            stop = getattr(provider, "stop_streaming", None)
            if callable(stop):
                try:
                    stop()
                except Exception as exc:
                    self._log(f"stream stop failed: {exc!r}")
        if round_id is not None:
            self._log(f"stream down for round {str(round_id)[:8]}")
        if watcher is not None and watcher is not threading.current_thread():
            watcher.join(timeout=_WATCHER_POLL_SECONDS + 1.0)
