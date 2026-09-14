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

Failures here are deliberately swallowed.  A busy or unplugged camera must
never stop a player from entering a game; adjudication reports the real error in
its own phase.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Mapping

from core.errors import DiceArenaError
from core.games import resolve_provider_id


class VisionStreamManager:
    """Starts a game's camera/RTSP stream when the game is entered.

    Only one round owns the stream at a time.  The manager tracks which round
    that is, so a teardown decided for an older round can never release the
    stream a newer round already owns.
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

    def start_for_round(self, round_: Any) -> bool:
        """Bring this round's game camera up; True when a stream is active.

        Called on game entry, before any adjudication.  The runtime is spawned
        in prewarm mode, so the detector session is loaded but no inference runs
        until the adjudication phase asks for it.

        Re-entering a game does not restart the camera: the provider reuses a
        warm runtime whose launch signature still matches, and rebuilds only
        when the profile's model/camera settings changed.
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
        with self._lock:
            self._provider = provider
            self._provider_id = provider_id
            self._owner_round_id = round_.id
        self._log(f"stream up for round {str(round_.id)[:8]} (provider {provider_id})")
        return True

    def stop(self) -> None:
        """Detach the current owner and stop its stream.

        Called at process shutdown and whenever a new round replaces the
        previous owner.
        """
        with self._lock:
            provider = self._provider
            self._provider = None
            self._provider_id = ""
            self._owner_round_id = None
        if provider is not None:
            stop = getattr(provider, "stop_streaming", None)
            if callable(stop):
                try:
                    stop()
                except Exception as exc:
                    self._log(f"stream stop failed: {exc!r}")
