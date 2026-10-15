"""Vision streaming lifecycle: game entry starts the camera, policy stops it."""
from __future__ import annotations

import sys
import threading
import pytest
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
    id = "vision_yolov8_objdetect"

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


def _manifest(profile: dict | None = None, slot: str | None = "vision_yolov8_objdetect") -> dict:
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


@pytest.mark.parametrize("always_on", [False, True])
def test_fast_cross_game_switch_stops_the_previous_providers_stream(always_on):
    # rps(v10) → 一秒内进 dice(v8)：老回合的 teardown watcher（2s 轮询）来
    # 不及醒来就被新回合顶掉所有权——老 provider 的 runtime 若不在这里被
    # 主动停掉，就会继续占着共享相机，新 provider 的摄像头打不开、当局
    # 直接丢局（2026-09-24 rps→dice 真机复现的根因）。
    v10 = _Provider()
    v10.id = "vision_yolov10_objdetect"
    v8 = _Provider()
    manager = VisionStreamManager(components=_Registry({v10.id: v10, v8.id: v8}))
    rps_profile = _profile()
    rps_profile["game_id"] = "rps"
    old_round = _StatusRound("r1", _manifest(rps_profile, slot=v10.id))
    manager.start_for_round(old_round, arena={"vision_always_on": always_on})

    # 玩家退出 rps 的同一瞬间进 dice——不等 watcher 的 2s 轮询。
    old_round.status = "cancelled"
    new_round = _StatusRound("r2", _manifest(_profile(), slot=v8.id))
    assert manager.start_for_round(new_round, arena={"vision_always_on": always_on}) is True

    # 跨 provider 交接：老 v10 的流已被主动停掉，新 v8 的流照常起来。
    assert v10.stop_calls == 1
    assert len(v8.started) == 1
    assert v8.stop_calls == 0

    # 老 watcher 醒来后不得再动任何流（等过一个轮询周期验证）。
    import time as _time

    _time.sleep(2.5)
    assert v10.stop_calls == 1
    assert v8.stop_calls == 0

    # Persistent mode retains the new stream too; game-lifetime mode releases it.
    new_round.status = "exited"
    if always_on:
        _time.sleep(0.3)
        assert v8.stop_calls == 0
        manager.stop()
        assert v8.stop_calls == 1
    else:
        assert _wait_for(lambda: v8.stop_calls == 1)


class _ExclusiveCameraProvider(_Provider):
    """Hold one simulated camera until an explicitly released stop completes."""

    def __init__(self, provider_id, camera):
        super().__init__()
        self.id = provider_id
        self.camera = camera
        self.start_attempted = threading.Event()
        self.stop_entered = threading.Event()
        self.allow_stop = threading.Event()
        self.open = False

    def start_streaming(self, profile, *, on_log=None, on_event=None):
        self.start_attempted.set()
        if self.camera["owner"] not in (None, self):
            raise RuntimeError("old provider still owns the camera")
        self.camera["owner"] = self
        self.open = True
        return super().start_streaming(profile, on_log=on_log)

    def stop_streaming(self):
        self.stop_calls += 1
        self.stop_entered.set()
        if not self.allow_stop.wait(5):
            raise RuntimeError("test did not release provider shutdown")
        if self.camera["owner"] is self:
            self.camera["owner"] = None
        self.open = False


@pytest.mark.parametrize("same_provider", [True, False], ids=["same-game", "cross-game"])
@pytest.mark.parametrize("release_path", ["watcher", "stop"])
@pytest.mark.parametrize("next_always_on", [False, True], ids=["round-lifetime", "persistent"])
def test_new_entry_waits_for_in_progress_camera_release(
    monkeypatch, same_provider, release_path, next_always_on,
):
    """A release already in flight must finish before a new owner may start."""
    monkeypatch.setattr("core.vision_stream._WATCHER_POLL_SECONDS", 0.01)
    camera = {"owner": None}
    old_provider = _ExclusiveCameraProvider("vision_old", camera)
    new_provider = old_provider if same_provider else _ExclusiveCameraProvider("vision_new", camera)
    if new_provider is not old_provider:
        new_provider.allow_stop.set()
    registry = _Registry({old_provider.id: old_provider, new_provider.id: new_provider})
    manager = VisionStreamManager(components=registry)
    old_round = _StatusRound("old", _manifest(_profile(), slot=old_provider.id))
    new_round = _StatusRound("new", _manifest(_profile(), slot=new_provider.id))
    entry_requested = threading.Event()
    entry_done = threading.Event()
    results = []
    errors = []
    stop_thread = None
    start_thread = None

    def enter():
        entry_requested.set()
        try:
            results.append(manager.start_for_round(new_round, arena={"vision_always_on": next_always_on}))
        except BaseException as exc:
            errors.append(exc)
        finally:
            entry_done.set()

    try:
        assert manager.start_for_round(old_round, arena={"vision_always_on": release_path != "watcher"})
        new_provider.start_attempted.clear()
        if release_path == "watcher":
            old_round.status = "exited"
        else:
            stop_thread = threading.Thread(target=manager.stop, daemon=True)
            stop_thread.start()
        assert old_provider.stop_entered.wait(2), "old camera release never started"
        start_thread = threading.Thread(target=enter, daemon=True)
        start_thread.start()
        assert entry_requested.wait(1)
        # In the regression the owner has been cleared, but its actual camera
        # stop has not finished: reuse is killed later or cross-game open fails.
        assert not new_provider.start_attempted.wait(0.1), "new camera started during old release"
        old_provider.allow_stop.set()
        assert entry_done.wait(2), "game entry deadlocked with the release watcher"
        assert not errors
        assert results == [True]
        assert new_provider.open and camera["owner"] is new_provider
        assert old_provider.stop_calls == 1
    finally:
        old_provider.allow_stop.set()
        new_provider.allow_stop.set()
        if start_thread is not None:
            start_thread.join(timeout=2)
        if stop_thread is not None:
            stop_thread.join(timeout=2)
        new_round.status = "exited"
        manager.stop()


def test_concurrent_game_entries_serialize_camera_start_and_ownership():
    """Two starts cannot race before either one has published its owner."""
    camera = {"owner": None}
    first = _ExclusiveCameraProvider("vision_first", camera)
    second = _ExclusiveCameraProvider("vision_second", camera)
    first.allow_stop.set()
    second.allow_stop.set()
    start_entered = threading.Event()
    allow_start = threading.Event()
    second_requested = threading.Event()
    original_start = first.start_streaming
    results = {}
    errors = []

    def paused_start(*args, **kwargs):
        start_entered.set()
        if not allow_start.wait(5):
            raise RuntimeError("test did not release camera startup")
        return original_start(*args, **kwargs)

    first.start_streaming = paused_start
    manager = VisionStreamManager(components=_Registry({first.id: first, second.id: second}))

    def enter(name, provider):
        if name == "second":
            second_requested.set()
        try:
            round_ = _StatusRound(name, _manifest(_profile(), slot=provider.id))
            results[name] = manager.start_for_round(round_, arena={"vision_always_on": True})
        except BaseException as exc:
            errors.append(exc)

    one = threading.Thread(target=enter, args=("first", first), daemon=True)
    two = threading.Thread(target=enter, args=("second", second), daemon=True)
    try:
        one.start()
        assert start_entered.wait(2)
        two.start()
        assert second_requested.wait(1)
        assert not second.start_attempted.wait(0.1), "second start bypassed in-progress camera startup"
        allow_start.set()
        one.join(timeout=2)
        two.join(timeout=2)
        assert not one.is_alive() and not two.is_alive()
        assert not errors
        assert results == {"first": True, "second": True}
        assert first.stop_calls == 1
        assert not first.open and second.open and camera["owner"] is second
    finally:
        allow_start.set()
        one.join(timeout=2)
        if two.ident is not None:
            two.join(timeout=2)
        manager.stop()
