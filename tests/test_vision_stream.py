"""Vision streaming lifecycle: game entry starts the camera, policy stops it."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.errors import ComponentNotFoundError  # noqa: E402
from core.vision_stream import VisionStreamManager  # noqa: E402


class _Round:
    """Minimal round double: an id and the effective manifest."""

    def __init__(self, round_id: str, manifest: dict | None = None) -> None:
        self.id = round_id
        self.manifest = manifest if manifest is not None else {}


class _Provider:
    id = "vision_yolov8_adjudicator"

    def __init__(self, *, result: bool = True, raises: Exception | None = None) -> None:
        self.result = result
        self.raises = raises
        self.started: list[tuple[dict, object]] = []
        self.stop_calls = 0

    def start_streaming(self, profile, *, on_log=None):
        self.started.append((dict(profile), on_log))
        if self.raises is not None:
            raise self.raises
        return self.result

    def stop_streaming(self) -> None:
        self.stop_calls += 1


class _PlainProvider:
    """A provider without the optional streaming hooks (cloud/fixture)."""

    id = "vision_plain"


class _Registry:
    def __init__(self, providers: dict) -> None:
        self._providers = providers
        self.requires: list[tuple] = []

    def require(self, provider_id, *, expected_type=None, expected_role=None):
        self.requires.append((provider_id, expected_type, expected_role))
        provider = self._providers.get(provider_id)
        if provider is None:
            raise ComponentNotFoundError(provider_id)
        return provider


def _profile() -> dict:
    return {
        "game_id": "dice",
        "vision": {"model": "m.onnx", "stable_frames": 30, "expected_count": 5},
        "multi_view": {"enabled": False},
        "video": {"enabled": False, "path": "/dice/det"},
    }


def _manifest(profile: dict | None = None, slot: str | None = "vision_yolov8_adjudicator") -> dict:
    manifest: dict = {"id": "dice"}
    if profile is not None:
        manifest["vision_profile"] = profile
    if slot is not None:
        manifest["providers"] = {"vision_adjudicator": slot}
    return manifest


def test_game_entry_starts_the_stream_with_the_rounds_profile():
    provider = _Provider()
    manager = VisionStreamManager(components=_Registry({provider.id: provider}))
    assert manager.start_for_round(_Round("r1", _manifest(_profile()))) is True
    assert len(provider.started) == 1
    profile, on_log = provider.started[0]
    # The provider receives the round's own profile, and a log sink so a camera
    # failure surfaces in the server log rather than vanishing.
    assert profile["game_id"] == "dice"
    assert profile["vision"]["expected_count"] == 5
    assert callable(on_log)


def test_game_without_a_vision_profile_never_opens_a_camera():
    provider = _Provider()
    manager = VisionStreamManager(components=_Registry({provider.id: provider}))
    assert manager.start_for_round(_Round("rps", _manifest(None))) is False
    assert provider.started == []


def test_game_without_a_vision_slot_never_opens_a_camera():
    provider = _Provider()
    manager = VisionStreamManager(components=_Registry({provider.id: provider}))
    assert manager.start_for_round(_Round("r1", _manifest(_profile(), slot=None))) is False
    assert provider.started == []


def test_provider_without_streaming_hooks_is_not_an_error():
    plain = _PlainProvider()
    manager = VisionStreamManager(components=_Registry({plain.id: plain}))
    assert manager.start_for_round(_Round("r1", _manifest(_profile(), slot=plain.id))) is False


def test_unavailable_provider_does_not_block_game_entry():
    manager = VisionStreamManager(components=_Registry({}))
    assert manager.start_for_round(_Round("r1", _manifest(_profile()))) is False


def test_camera_failure_does_not_block_game_entry():
    provider = _Provider(raises=RuntimeError("camera busy"))
    manager = VisionStreamManager(components=_Registry({provider.id: provider}))
    # Entering a game must survive a busy or unplugged camera; adjudication
    # still starts the runtime lazily and reports the real error there.
    assert manager.start_for_round(_Round("r1", _manifest(_profile()))) is False


def test_provider_declining_the_stream_does_not_claim_ownership():
    provider = _Provider(result=False)
    manager = VisionStreamManager(components=_Registry({provider.id: provider}))
    manager.start_for_round(_Round("r1", _manifest(_profile())))
    # A provider that refused must not be handed a later teardown.
    manager.stop()
    assert provider.stop_calls == 0


def test_stop_releases_the_owned_stream_once():
    provider = _Provider()
    manager = VisionStreamManager(components=_Registry({provider.id: provider}))
    manager.start_for_round(_Round("r1", _manifest(_profile())))
    manager.stop()
    assert provider.stop_calls == 1
    # Idempotent: a second stop has no owner left to release.
    manager.stop()
    assert provider.stop_calls == 1


def test_reentering_a_game_does_not_stop_the_shared_stream():
    provider = _Provider()
    manager = VisionStreamManager(components=_Registry({provider.id: provider}))
    manager.start_for_round(_Round("r1", _manifest(_profile())))
    manager.start_for_round(_Round("r2", _manifest(_profile())))
    # The provider reuses a warm runtime whose signature still matches, so the
    # manager must not tear it down on the way into the next game.
    assert provider.stop_calls == 0
    assert len(provider.started) == 2


# ---- vision_always_on decides whether the stream outlives the game --------

class _StatusRound(_Round):
    """Round double whose status the test drives."""

    def __init__(self, round_id: str, manifest: dict | None = None, status: str = "running"):
        super().__init__(round_id, manifest)
        self.status = status


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_always_on_keeps_the_stream_after_the_game_ends():
    provider = _Provider()
    manager = VisionStreamManager(components=_Registry({provider.id: provider}))
    round_ = _StatusRound("r1", _manifest(_profile()))
    manager.start_for_round(round_, arena={"vision_always_on": True})
    round_.status = "exited"
    # No watcher is armed in this mode, so the stream stays until the process
    # itself is torn down.
    import time as _time

    _time.sleep(0.3)
    assert provider.stop_calls == 0
    manager.stop()
    assert provider.stop_calls == 1


def test_game_lifetime_releases_the_stream_when_the_round_ends():
    provider = _Provider()
    manager = VisionStreamManager(components=_Registry({provider.id: provider}))
    round_ = _StatusRound("r1", _manifest(_profile()))
    manager.start_for_round(round_, arena={"vision_always_on": False})
    assert provider.stop_calls == 0
    round_.status = "exited"
    assert _wait_for(lambda: provider.stop_calls == 1)


def test_game_lifetime_keeps_the_stream_while_the_round_runs():
    provider = _Provider()
    manager = VisionStreamManager(components=_Registry({provider.id: provider}))
    round_ = _StatusRound("r1", _manifest(_profile()))
    manager.start_for_round(round_, arena={"vision_always_on": False})
    # "再来一局" stays inside the same round, so a running round must never be
    # mistaken for the game being over.
    import time as _time

    _time.sleep(0.3)
    assert provider.stop_calls == 0
    manager.stop()


def test_cancelled_and_error_rounds_also_release_the_stream():
    for status in ("cancelled", "error"):
        provider = _Provider()
        manager = VisionStreamManager(components=_Registry({provider.id: provider}))
        round_ = _StatusRound("r1", _manifest(_profile()))
        manager.start_for_round(round_, arena={"vision_always_on": False})
        round_.status = status
        assert _wait_for(lambda: provider.stop_calls == 1), status
        manager.stop()


def test_a_stale_teardown_cannot_release_the_new_rounds_stream():
    provider = _Provider()
    manager = VisionStreamManager(components=_Registry({provider.id: provider}))
    old_round = _StatusRound("r1", _manifest(_profile()))
    manager.start_for_round(old_round, arena={"vision_always_on": False})

    # The player leaves and immediately enters another game: create_round
    # cancels the old round while the new one starts its own stream.
    old_round.status = "cancelled"
    new_round = _StatusRound("r2", _manifest(_profile()))
    manager.start_for_round(new_round, arena={"vision_always_on": False})

    # The outdated watcher must not tear down what the new round now owns.
    # Wait past its poll interval so it really wakes up and decides.
    import time as _time

    _time.sleep(2.5)
    assert provider.stop_calls == 0
    new_round.status = "exited"
    assert _wait_for(lambda: provider.stop_calls == 1)
