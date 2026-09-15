"""Resident, game-agnostic YOLOv8 adjudication provider."""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, wait
from pathlib import Path
from typing import Any, Callable, Mapping

from core.vision import VisionAdjudicationRequest, VisionAdjudicatorProvider
from components.vision_yolov8_adjudicator.process import (
    YoloRuntimeProcess,
    _snapshot_path,
    region_split,
)
from components.vision_yolov8_adjudicator.rules import (
    diagnose_detection_failure,
    evaluate_rule,
    finalize_outcome,
    project_result,
    fuse_yolo_outcomes,
)
from components.vision_yolov8_adjudicator.profile import (
    ProfileError,
    compose_video_url,
    load_component_config,
    load_runtime_config,
    resolve_project_path,
    resolve_runtime_config_path,
)


COMPONENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = COMPONENT_DIR.parents[2]


def resolve_runtime_binary(
    component_dir: Path | None = None,
    project_root: Path | None = None,
) -> Path:
    """Resolve the runtime binary the way the process launcher resolves it.

    JSON configuration is the single source of truth (environment overrides
    were removed), so the component ``config.json`` owns ``runtime.binary`` and
    it is resolved against the repository root exactly as
    :meth:`YoloRuntimeProcess.start` does.  A game profile may override
    ``runtime.binary`` for one round, but readiness has to be answerable before
    any round exists, so this reports the deployment default.
    """
    directory = Path(component_dir) if component_dir is not None else COMPONENT_DIR
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    component = load_component_config(directory)
    runtime = component.get("runtime")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    configured = runtime.get("binary")
    if not isinstance(configured, str) or not configured.strip():
        raise ProfileError("vision component config must declare runtime.binary")
    return resolve_project_path(configured, root)


def detected_divider_ratio(
    observation: Mapping[str, Any], orientation: str, enabled: bool = True
) -> float | None:
    """Return where the runtime actually found the divider, as a 0..1 ratio.

    The whole point of ``divider_regions`` grouping is to split on the boundary
    the scene shows, not on a number someone typed into a manifest: the demo
    mat's red/blue seam sits near 0.505 while the configured default is 0.5, and
    the two only stay interchangeable until the camera moves.  So the runtime's
    located boundary wins whenever it has one, and the profile's
    ``vision.divider.position`` only covers the frames where it does not.

    ``enabled`` mirrors the runtime's own gate: with divider detection off the
    runtime never looks for a boundary, so a stale ``divider`` key left in an
    observation must not start steering the split.

    ``found`` is checked with ``is True`` rather than ``is not False`` because
    the runtime still emits a placeholder ``point`` of ``[0, 0]`` when it fails
    to locate anything; treating absence as a ratio of 0 would dump every
    detection into the second participant.
    """
    if not enabled:
        return None
    divider = observation.get("divider")
    if not isinstance(divider, Mapping) or divider.get("found") is not True:
        return None
    point = divider.get("point")
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    axis = 1 if orientation == "horizontal" else 0
    extent = observation.get("height" if orientation == "horizontal" else "width")
    if not isinstance(extent, (int, float)) or extent <= 0:
        return None
    coordinate = point[axis]
    if not isinstance(coordinate, (int, float)):
        return None
    ratio = float(coordinate) / float(extent)
    if not 0.0 < ratio < 1.0:
        return None
    return ratio


def normalize_observation(
    profile: Mapping[str, Any], observation: Mapping[str, Any]
) -> dict[str, Any]:
    """Project generic detector boxes into the profile's participants.

    The C++ runtime deliberately emits only model-neutral detections.  A
    profile may opt into ``x_midpoint`` or an explicit ``divider_regions``
    boundary; a
    runtime-provided ``participants`` object remains authoritative when it is
    available (for example, a future detector with its own tracker).

    For ``divider_regions`` the boundary is the divider the runtime located in
    this very frame (:func:`detected_divider_ratio`), falling back to the
    profile's ``vision.divider.position`` only when the scene offered no
    boundary.  The runtime's region gate makes the same substitution, so both
    halves of the pipeline always split on the same pixel.
    """
    result = dict(observation)
    vision = profile.get("vision", {})
    vision = vision if isinstance(vision, Mapping) else {}
    participant_names = vision.get("participants")
    if not isinstance(participant_names, list) or len(participant_names) < 2:
        return result
    existing = result.get("participants")
    if isinstance(existing, Mapping):
        result["participants"] = {
            str(name): existing[name]
            for name in participant_names
            if name in existing
        }
        return result
    grouping = str(vision.get("grouping") or vision.get("participant_assignment") or "x_midpoint")
    if grouping not in {"x_midpoint", "divider_regions"}:
        return result
    class_map = vision.get("class_map", {})
    if not isinstance(class_map, Mapping):
        return result
    detections = result.get("detections")
    if not isinstance(detections, list):
        return result
    width = result.get("width")
    if not isinstance(width, (int, float)) or width <= 0:
        width = max(
            (float(box[2]) for box in detections
             if isinstance(box, Mapping)
             and isinstance(box.get("bbox"), (list, tuple))
             and len(box["bbox"]) >= 3
             and isinstance(box["bbox"][2], (int, float))),
            default=0.0,
        )
    if width <= 0:
        return result
    grouped: dict[str, list[Any]] = {
        str(participant_names[0]): [],
        str(participant_names[1]): [],
    }
    divider = vision.get("divider", {})
    position = 0.5
    orientation = "vertical"
    if grouping == "divider_regions" and isinstance(divider, Mapping):
        try:
            position = float(divider.get("position", 0.5))
        except (TypeError, ValueError):
            position = 0.5
        orientation = str(divider.get("orientation", "vertical"))
    if grouping == "divider_regions":
        detected = detected_divider_ratio(
            result, orientation, enabled=vision.get("divider_detection", True) is not False
        )
        if detected is not None:
            position = detected
    for detection in detections:
        if not isinstance(detection, Mapping):
            continue
        bbox = detection.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        try:
            center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
            center_y = (float(bbox[1]) + float(bbox[3])) / 2.0
        except (TypeError, ValueError):
            continue
        class_id = detection.get("class_id")
        # ``class_N`` is a runtime diagnostic label, never a game value.
        # Unknown class IDs must be ignored until the profile declares them.
        mapped = class_map.get(str(class_id))
        if mapped is None:
            continue
        if isinstance(mapped, str) and isinstance(profile.get("rule"), Mapping):
            if profile["rule"].get("kind") == "numeric_compare":
                try:
                    numeric = float(mapped.strip())
                    mapped = int(numeric) if numeric.is_integer() else numeric
                except (TypeError, ValueError):
                    continue
        if grouping == "divider_regions" and orientation == "horizontal":
            height = result.get("height")
            if not isinstance(height, (int, float)) or height <= 0:
                height = max(
                    (float(box[3]) for box in detections
                     if isinstance(box, Mapping)
                     and isinstance(box.get("bbox"), (list, tuple))
                     and len(box["bbox"]) >= 4
                     and isinstance(box["bbox"][3], (int, float))),
                    default=0.0,
                )
            boundary = float(height) * position
            is_first = center_y < boundary
        else:
            boundary = float(width) * (position if grouping == "divider_regions" else 0.5)
            is_first = center_x < boundary
        participant = grouped[str(participant_names[0])] if is_first else grouped[str(participant_names[1])]
        participant.append(mapped)
    if any(grouped.values()):
        result["participants"] = grouped
    return result


def _has_incomplete_expected_counts(
    profile: Mapping[str, Any], observations: list[Mapping[str, Any]]
) -> bool:
    """Return whether stable detector evidence has an incomplete object count."""
    vision = profile.get("vision", {}) if isinstance(profile, Mapping) else {}
    vision = vision if isinstance(vision, Mapping) else {}
    expected = vision.get("expected_count")
    if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
        return False
    participant_names = vision.get("participants")
    names = [str(name) for name in participant_names] if isinstance(participant_names, list) else []
    for observation in observations:
        participants = observation.get("participants")
        if not isinstance(participants, Mapping):
            continue
        keys = names or [str(key) for key in participants]
        for name in keys:
            values = participants.get(name)
            count = len(values) if isinstance(values, (list, tuple)) else (0 if values is None else 1)
            if count != expected:
                return True
    return False


def _parse_result_line(line: str) -> dict[str, Any] | None:
    raw = line.strip()
    if raw.startswith("[RESULT] "):
        raw = raw[9:]
    if not raw.startswith("{"):
        return None
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _consume_legacy_log_line(
    line: str,
    on_log: Callable[[str], None],
    on_event: Callable[[dict[str, Any]], None],
) -> dict[str, Any] | None:
    """Parse only the explicitly tagged result envelope from old runtimes."""
    if not line.startswith("[RESULT] "):
        on_log(line)
        return None
    parsed = _parse_result_line(line)
    if parsed is None:
        on_log(line)
        return None
    event = dict(parsed)
    event.setdefault("event", "result")
    on_event(event)
    return event


class VisionYolov8Adjudicator(VisionAdjudicatorProvider):
    id = "vision_yolov8_adjudicator"
    type = "vision"
    role = "adjudicator"
    name = "YOLOv8 Vision Adjudicator"
    version = "2.0"

    def __init__(self, manifest: dict[str, Any] | None = None, *, runtime_factory: Callable[..., Any] | None = None, verifier: Any | None = None) -> None:
        super().__init__(manifest)
        self.runtime_factory = runtime_factory or (lambda view_id="default": YoloRuntimeProcess())
        # LLM engines arrive per round on the request (``llm_provider``,
        # resolved by the pipeline from the global ``llm`` slot).  This
        # attribute remains the test-injection seam for verifier doubles and
        # doubles as the fallback when a round carries no provider.
        self.verifier = verifier
        self._runtime_cache: dict[str, Any] = {}
        # A resident process receives its snapshot root at process creation;
        # subsequent rounds reuse the same camera process.  Keep one private
        # root per view alive for that process lifetime and remove each
        # single-use snapshot after verification.
        self._runtime_snapshot_dirs: dict[str, Path] = {}
        self._runtime_signatures: dict[str, str] = {}
        # The cache is now reachable from two entry points -- a game-entry
        # ``start_streaming`` and an adjudication round -- and multi-view
        # profiles spawn their runtimes concurrently, so every read/write of
        # the cache/signature/snapshot-root triple is serialized here.
        self._runtime_lock = threading.RLock()

    def health(self) -> dict[str, Any]:
        """Report deployment readiness of this provider's runtime binary.

        ``ready`` answers the deploy-time question -- is the YOLO executable
        installed and runnable? -- and is what ``server.py`` republishes as the
        top-level ``yolo_ready`` field.  It is deliberately a static check, not
        a liveness probe of a resident process, and ``ok`` mirrors it so this
        component can no longer claim health while its binary is missing.
        LLM readiness belongs to the ``llm`` component, so this payload stays
        silent about it.
        """
        payload: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "role": self.role,
        }
        try:
            binary = resolve_runtime_binary()
        except Exception as exc:
            # /api/health must always answer; report the failure in-band
            # instead of raising and losing the identity fields above.
            return {**payload, "ok": False, "ready": False, "binary": "", "error": str(exc)}
        ready = binary.is_file() and os.access(binary, os.X_OK)
        return {**payload, "ok": ready, "ready": ready, "binary": str(binary)}

    def _round_llm(self, request: VisionAdjudicationRequest) -> Any:
        """The LLM engine for one round: request-injected provider first.

        Production rounds carry the provider resolved from the global
        ``llm`` slot; the constructor ``verifier`` seam (test doubles, or a
        provider injected some other way) is the fallback.
        """
        provider = getattr(request, "llm_provider", None)
        return provider if provider is not None else self.verifier

    def _stop_all_runtimes(self) -> None:
        """Release every cached runtime and its private snapshot root.

        Best-effort by contract: this runs at process shutdown and at a
        game-lifetime round boundary, where a partially dead runtime must not
        turn cleanup into a second failure.
        """
        with self._runtime_lock:
            runtimes = list(self._runtime_cache.items())
            snapshot_dirs = list(self._runtime_snapshot_dirs.values())
            self._runtime_cache.clear()
            self._runtime_snapshot_dirs.clear()
            self._runtime_signatures.clear()
        import shutil

        for _view_id, runtime in runtimes:
            try:
                stop = getattr(runtime, "stop", None)
                if callable(stop):
                    stop()
            except Exception:
                # Shutdown is best-effort; continue cleaning other views and
                # private snapshot roots even if one runtime is already dead.
                pass
        for root in snapshot_dirs:
            shutil.rmtree(root, ignore_errors=True)

    def shutdown(self) -> None:
        """Stop all resident runtimes owned by this provider instance.

        Resident mode intentionally keeps camera/RTSP workers alive between
        rounds, but those workers must not outlive the backend process.  The
        server calls this hook during SIGTERM cleanup so a restart does not
        leave an orphan holding the camera device.
        """
        self._stop_all_runtimes()

    def stop_streaming(self) -> None:
        """Stop the camera/RTSP stream started for the current game.

        The game-lifetime counterpart of :meth:`start_streaming`: the manager
        calls it when the deployment does not keep vision warm across games.
        Identical work to ``shutdown()`` today, but kept as a separate name so
        the two intents can diverge (for example releasing only the detector)
        without touching callers.
        """
        self._stop_all_runtimes()

    def _resolve_views(self, profile: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        """The view list a profile runs with, single-view fallback included."""
        multi = profile.get("multi_view", {}) if isinstance(profile, Mapping) else {}
        views = multi.get("views") if isinstance(multi, Mapping) and multi.get("enabled") else None
        if not isinstance(views, list) or not views:
            return [{"id": "default"}]
        return [view for view in views if isinstance(view, Mapping)]

    def _ensure_runtime(
        self,
        profile: Mapping[str, Any],
        view_id: str,
        on_log: Callable[[str], None],
    ) -> Any:
        """Return this view's runtime, starting or rebuilding it as needed.

        The single place a resident YOLO process is created, shared by the
        game-entry ``start_streaming`` and by ``adjudicate`` so a game that
        preloaded its camera is never restarted when the round actually runs.
        A cached runtime survives across rounds; a changed signature (model,
        stable frames, confidence, camera, runtime block) invalidates it.
        """
        vid = str(view_id)
        keep_warm = self._resident_mode(profile)
        signature = self._runtime_signature(profile, vid)
        with self._runtime_lock:
            rt = self._runtime_cache.get(vid) if keep_warm else None
            if rt is not None and self._runtime_signatures.get(vid) != signature:
                try:
                    stop = getattr(rt, "stop", None)
                    if callable(stop):
                        stop()
                finally:
                    self._runtime_cache.pop(vid, None)
                    self._runtime_signatures.pop(vid, None)
                    old_root = self._runtime_snapshot_dirs.pop(vid, None)
                if old_root is not None:
                    import shutil
                    shutil.rmtree(old_root, ignore_errors=True)
                rt = None
            if rt is not None:
                return rt
            rt = self.runtime_factory(vid)
            snapshot_dir = Path(tempfile.mkdtemp(prefix=f"vision-runtime-{vid}-"))
            try:
                rt.start(
                    profile,
                    vid,
                    prewarm=True,
                    snapshot_dir=snapshot_dir,
                    on_log=on_log,
                )
            except TypeError:
                # Keep injected test/fallback runtimes source-compatible
                # while the production adapter receives the per-job
                # directory above.
                rt.start(profile, vid, prewarm=True)
            if keep_warm:
                self._runtime_cache[vid] = rt
                self._runtime_snapshot_dirs[vid] = snapshot_dir
                self._runtime_signatures[vid] = signature
            return rt

    def _snapshot_root_for(self, view_id: str) -> Path | None:
        """This view's cached snapshot root, read under the cache lock.

        The stream manager may tear the cache down from its watcher thread
        while an adjudication is still winding down, so this read is locked
        like every other access to the cache triple.
        """
        with self._runtime_lock:
            return self._runtime_snapshot_dirs.get(view_id)

    def start_streaming(
        self,
        profile: Mapping[str, Any],
        *,
        on_log: Callable[[str], None] | None = None,
    ) -> bool:
        """Start this game's camera/RTSP stream without running inference.

        Called when a player enters a game, so the table is already visible
        while the round is still in its rules/ready states.  The runtime is
        spawned in ``--prewarm`` mode: the camera and RTSP publisher come up
        and the detector session is loaded, but ``adjudication_active`` stays
        false, so no OpenCL preprocessing and no inference run until the
        adjudication phase sends ``START_ADJUDICATION``.

        Failures are swallowed on purpose.  Entering a game must never break
        because a camera is busy or unplugged; adjudication still starts the
        runtime lazily and reports the real error in its own phase.
        """
        log = on_log or (lambda _line: None)
        if not isinstance(profile, Mapping):
            return False
        started = False
        for view in self._resolve_views(profile):
            vid = str(view.get("id", "default"))
            try:
                self._ensure_runtime(profile, vid, log)
                started = True
            except Exception as exc:
                log(f"[vision] stream start failed for view {vid}: {exc!r}")
        return started

    @staticmethod
    def _runtime_signature(profile: Mapping[str, Any], view_id: str) -> str:
        """Return the profile-owned runtime inputs that require a restart.

        The cache is keyed by ``view_id``, and every game's single-view profile
        uses the id ``"default"`` — so this signature is the *only* thing
        keeping two games apart.  It must therefore cover every profile-owned
        value that changes what the resident process does, not just the ones
        that look like "hardware": ``divider_detection``, ``expected_count``,
        the resolved region split, the ``grouping`` mode and the RTSP path are
        all forwarded to the runtime on its command line.

        Missing any of them let a second game whose model / stable frames /
        camera agreed with the first reuse the first game's process, running
        with the wrong divider gate, the wrong per-side count and the wrong
        RTSP mount.  Keeping this list complete is what makes one shared camera
        safe here: a differing signature tears the old process down and builds
        a new one instead of silently inheriting it.
        """
        vision = profile.get("vision", {})
        vision = vision if isinstance(vision, Mapping) else {}
        multi = profile.get("multi_view", {})
        multi = multi if isinstance(multi, Mapping) else {}
        view_data: Mapping[str, Any] = {}
        for candidate in multi.get("views", []):
            if isinstance(candidate, Mapping) and str(candidate.get("id")) == view_id:
                view_data = candidate
                break
        video = profile.get("video", {})
        video_path = video.get("path") if isinstance(video, Mapping) else None
        # Compare the *resolved* region split (what the runtime is launched
        # with) rather than the raw divider block, which may be absent.
        region_position, region_orientation = region_split(vision)
        payload = {
            # Game identity: two profiles that agree on every numeric knob are
            # still different games (different classes, rules and prompts).
            "game_id": profile.get("game_id"),
            "model": vision.get("model"),
            "stable_frames": vision.get("stable_frames"),
            "confidence": vision.get("confidence", vision.get("conf")),
            "divider_detection": vision.get("divider_detection"),
            "expected_count": vision.get("expected_count"),
            "region_position": region_position,
            "region_orientation": region_orientation,
            "grouping": vision.get("grouping"),
            "video_path": video_path,
            "camera": view_data.get("camera"),
            "runtime": profile.get("runtime"),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)

    @staticmethod
    def _video_base_url(profile: Mapping[str, Any]) -> str:
        """Resolve the deployment WebRTC origin without exposing RTSP details."""
        video = profile.get("video", {})
        profile_base = video.get("webrtc_base_url") if isinstance(video, Mapping) else ""
        component: Mapping[str, Any] = {}
        try:
            component = load_component_config(Path(__file__).parent)
        except Exception:
            component = {}
        runtime_base = ""
        configured_runtime = component.get("runtime", {})
        has_explicit_runtime_config = isinstance(configured_runtime, Mapping) and configured_runtime.get("config")
        if has_explicit_runtime_config:
            try:
                runtime = load_runtime_config(resolve_runtime_config_path(component))
                runtime_video = runtime.get("video", {})
                if isinstance(runtime_video, Mapping):
                    runtime_base = runtime_video.get("webrtc_base_url", "")
            except Exception:
                pass
        component_video = component.get("video", {})
        component_base = component_video.get("webrtc_base_url", "") if isinstance(component_video, Mapping) else ""
        return str(profile_base or runtime_base or component_base or "")

    @staticmethod
    def _video_event(profile: Mapping[str, Any], view_id: str, event: Mapping[str, Any]) -> dict[str, Any] | None:
        """Project runtime video notifications to the public WebRTC endpoint.

        Runtime URLs may be RTSP, loopback, or otherwise deployment-private;
        only the profile-owned WebRTC base URL and path are allowed to cross
        the provider boundary.
        """
        multi = profile.get("multi_view", {})
        path: Any = None
        if isinstance(multi, Mapping) and multi.get("enabled"):
            for view in multi.get("views", []):
                if isinstance(view, Mapping) and str(view.get("id")) == view_id:
                    video = view.get("video")
                    path = video.get("path") if isinstance(video, Mapping) else None
                    break
        if path is None:
            video = profile.get("video")
            path = video.get("path") if isinstance(video, Mapping) else None
        if not isinstance(path, str) or not path.strip():
            return None
        video_enabled = True
        if isinstance(multi, Mapping) and multi.get("enabled"):
            for view in multi.get("views", []):
                if isinstance(view, Mapping) and str(view.get("id")) == view_id:
                    view_video = view.get("video")
                    if isinstance(view_video, Mapping):
                        video_enabled = bool(view_video.get("enabled", True))
                    break
        elif isinstance(profile.get("video"), Mapping):
            video_enabled = bool(profile["video"].get("enabled", True))
        if not video_enabled:
            return None
        base = VisionYolov8Adjudicator._video_base_url(profile)
        if not isinstance(base, str) or not base.strip():
            return None
        return {"event": "video", "url": compose_video_url(base, path), "view_id": view_id}

    @staticmethod
    def _adjudication_timeout(profile: Mapping[str, Any], fallback: float) -> float:
        """Resolve the game-owned total budget, retaining the global fallback."""
        timeouts = profile.get("timeouts", {})
        value = timeouts.get("adjudication_seconds") if isinstance(timeouts, Mapping) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return float(value)
        return float(fallback)

    @staticmethod
    def _yolo_detection_timeout(profile: Mapping[str, Any], fallback: float) -> float:
        """Resolve the shorter wait for a stable YOLO observation."""
        timeouts = profile.get("timeouts", {})
        value = timeouts.get("yolo_detection_seconds") if isinstance(timeouts, Mapping) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return float(value)
        return float(fallback)

    @staticmethod
    def _llm_timeout(profile: Mapping[str, Any], fallback: float) -> float:
        """Resolve the single per-request timeout shared by all LLM calls."""
        llm = profile.get("llm", {})
        value = llm.get("timeout_seconds") if isinstance(llm, Mapping) else None
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
        ):
            return float(value)
        return float(fallback)

    @staticmethod
    def _resident_mode(profile: Mapping[str, Any]) -> bool:
        """Return whether the deployment keeps camera/runtime processes warm."""
        configured: Mapping[str, Any] = {}
        try:
            component = load_component_config(Path(__file__).parent)
            runtime = component.get("runtime", {})
            if isinstance(runtime, Mapping):
                configured = runtime
        except Exception:
            configured = {}
        profile_runtime = profile.get("runtime", {})
        profile_runtime = profile_runtime if isinstance(profile_runtime, Mapping) else {}
        mode = profile_runtime.get("mode", configured.get("mode", "per_request"))
        prewarm = profile_runtime.get(
            "prewarm_camera", configured.get("prewarm_camera", False)
        )
        return mode == "resident" and bool(prewarm)

    def _diagnose_failure(
        self,
        request: VisionAdjudicationRequest,
        profile: Mapping[str, Any],
        observations: list[Mapping[str, Any]],
        on_event: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        """Explain a failed YOLO round from detector evidence, never a winner.

        The reason comes from the local evidence rules only: the runtime and
        this provider already know the per-side object counts, whether the
        divider was located and whether any frame qualified, which is all the
        player-facing hint needs.  The multimodal LLM diagnosis path was
        removed on 2026-09-14 (it cost one image request per failure while the
        local rules already named the cause), so ``source`` is always
        ``local``.
        """
        from components.vision_yolov8_adjudicator.rules import diagnose_detection_failure

        normalized_observations = [normalize_observation(profile, item) for item in observations]
        evidence: dict[str, Any] = {"views": [dict(item) for item in normalized_observations]}
        for item in normalized_observations:
            for key in ("participants", "detections", "divider", "width", "height"):
                if key in item and key not in evidence:
                    evidence[key] = item[key]
        diagnosis = dict(diagnose_detection_failure(profile, evidence))
        diagnosis["source"] = "local"
        diagnosis["retry"] = True
        result = {
            "adjudicated": False,
            "verified": False,
            "diagnosed": True,
            "retry_required": True,
            "diagnosis": diagnosis,
            "profile_id": profile.get("game_id") or request.game_id,
            "provider_id": self.id,
            "evidence": evidence,
        }
        on_event({"event": "diagnosis", **result})
        return result

    def adjudicate(self, request: VisionAdjudicationRequest, *, on_log: Callable[[str], None], on_event: Callable[[dict[str, Any]], None], is_cancelled: Callable[[], bool], timeout_seconds: float | None = None) -> dict[str, Any]:
        profile = request.profile; runtimes = []
        keep_warm = self._resident_mode(profile)
        multi = profile.get("multi_view", {}) if isinstance(profile.get("multi_view"), Mapping) else {}
        views = multi.get("views") if multi.get("enabled") else None
        if not isinstance(views, list) or not views: views = [{"id": "default"}]
        ordered: list[dict[str, Any]] = []
        cleanup_paths: set[Path] = set()
        strict_snapshot_roots: dict[str, Path] = {}
        round_completed = False
        round_started = False
        try:
            def start(v):
                # A game that preloaded its camera on entry already cached the
                # runtime, so this is normally a lookup and the round starts
                # without paying the spawn/model-load latency again.
                vid = str(v.get("id", "default"))
                rt = self._ensure_runtime(profile, vid, on_log)
                if isinstance(rt, YoloRuntimeProcess):
                    root = self._snapshot_root_for(vid)
                    if root is not None:
                        strict_snapshot_roots[vid] = root.resolve()
                return rt
            with ThreadPoolExecutor(max_workers=len(views)) as pool: runtimes.extend(pool.map(start, views))
            emitted_video_views: set[str] = set()
            # A resident runtime's startup event is one-shot and may have
            # already been consumed by an earlier round.  Publish the
            # profile-owned WebRTC URL synchronously for every new round so
            # clients can attach to the live stream before detection runs.
            for view in views:
                vid = str(view.get("id", "default"))
                video_event = self._video_event(profile, vid, {"event": "video"})
                if video_event is not None:
                    on_event(video_event)
                    emitted_video_views.add(vid)
            pre_wait = float(profile.get("lifecycle", {}).get("pre_adjudication_wait_seconds", 0))
            if pre_wait > 0:
                # Game-owned settling window before detection.  Like the
                # post-result hold, it must not consume the adjudication
                # budget, which only starts after START_ADJUDICATION below.
                end = time.monotonic() + pre_wait
                while time.monotonic() < end:
                    if is_cancelled():
                        if not keep_warm:
                            for rt in runtimes:
                                try:
                                    rt.stop()
                                except Exception:
                                    pass
                        raise RuntimeError("cancelled")
                    remaining = max(0, end-time.monotonic()); on_event({"event":"phase", "phase":"pre_wait", "remaining_ms":int(remaining*1000)})
                    time.sleep(min(0.25, remaining))
            for rt in runtimes:
                rt.send({"command":"START_ADJUDICATION", "request_id":request.request_id, "profile_id":profile.get("game_id")})
            round_started = True
            on_event({"event":"phase", "phase":"detecting"}); observations = {}
            latest_by_view: dict[str, dict[str, Any]] = {}
            latest_lock = threading.Lock()
            def collect(rt, view):
                vid = str(view.get("id", "default")); found = None; active_seen = False
                for event in rt.events():
                    if event.get("event") == "video":
                        video_event = self._video_event(profile, vid, event)
                        if video_event is not None and vid not in emitted_video_views:
                            on_event(video_event)
                            emitted_video_views.add(vid)
                    elif event.get("event") == "progress":
                        if event.get("phase") == "detecting":
                            active_seen = True
                        on_event({**event, "view_id": vid})
                    elif event.get("event") == "runtime_exit":
                        returncode = event.get("returncode")
                        raise RuntimeError(
                            f"YOLO runtime exited before stable observation "
                            f"(returncode={returncode})"
                        )
                    elif event.get("event") in {"diagnostic_snapshot", "observation"}:
                        candidate = dict(event, view_id=vid)
                        with latest_lock:
                            latest_by_view[vid] = candidate
                        if event.get("event") == "observation" and event.get("stable"):
                            found = candidate
                            break
                    elif event.get("event") == "phase" and event.get("phase") == "detecting":
                        active_seen = True
                    elif event.get("event") == "cancelled":
                        # A resident runtime can leave the previous round's
                        # CANCEL acknowledgement queued in the event pipe.
                        # It is only a cancellation for this collector after
                        # the current round has visibly entered detecting.
                        if active_seen:
                            break
                        continue
                    elif (
                        event.get("event") == "phase"
                        and event.get("phase") == "idle"
                        and active_seen
                    ):
                        break
                return vid, found
            fallback_timeout = float(timeout_seconds or request.timeout_seconds)
            deadline = time.monotonic() + self._adjudication_timeout(profile, fallback_timeout)
            yolo_deadline = min(
                deadline,
                time.monotonic() + self._yolo_detection_timeout(profile, fallback_timeout),
            )
            pool = ThreadPoolExecutor(max_workers=len(runtimes))
            futures = [pool.submit(collect, rt, view) for rt, view in zip(runtimes, views)]
            pending = set(futures)
            timed_out = False
            try:
                while pending:
                    if is_cancelled():
                        raise RuntimeError("cancelled")
                    remaining = max(0.0, yolo_deadline - time.monotonic())
                    if remaining <= 0:
                        timed_out = True
                        break
                    done, pending = wait(pending, timeout=min(remaining, 0.1))
                    for future in done:
                        vid, found = future.result()
                        if found is not None:
                            observations[vid] = found
            except RuntimeError:
                if not keep_warm:
                    for view, rt in zip(views, runtimes):
                        try:
                            rt.stop()
                        except Exception:
                            pass
                        vid = str(view.get("id", "default"))
                        if self._runtime_cache.get(vid) is rt:
                            self._runtime_cache.pop(vid, None)
                            self._runtime_signatures.pop(vid, None)
                            root = self._runtime_snapshot_dirs.pop(vid, None)
                            if root is not None:
                                import shutil
                                shutil.rmtree(root, ignore_errors=True)
                raise
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
            # Cached resident runtimes may have emitted their video event in a
            # previous round.  Ensure each current view still gets one event,
            # without duplicating a startup event observed above.
            for view in views:
                vid = str(view.get("id", "default"))
                if vid not in emitted_video_views:
                    video_event = self._video_event(profile, vid, {"event": "video"})
                    if video_event is not None:
                        on_event(video_event)
            if is_cancelled():
                raise RuntimeError("cancelled")
            if not observations and not timed_out:
                timed_out = True
            if timed_out:
                # The last diagnostic_snapshot can be queued in the runtime
                # pipe just after the deadline. Ask the runtime to stop
                # inference, then give each collector a bounded drain window
                # before taking its latest evidence.
                for rt in runtimes:
                    try:
                        rt.send({"command": "STOP_ADJUDICATION", "request_id": request.request_id})
                    except Exception:
                        pass
                done_after, pending = wait(pending, timeout=0.25)
                for future in done_after:
                    try:
                        vid, found = future.result()
                    except Exception:
                        continue
                    if found is not None:
                        observations[vid] = found
                with latest_lock:
                    diagnostic_observations = [dict(latest_by_view[key]) for key in sorted(latest_by_view)]
                if not diagnostic_observations:
                    diagnostic_observations = [{"view_id": str(view.get("id", "default"))} for view in views]
                diagnosis_result = self._diagnose_failure(
                    request,
                    profile,
                    diagnostic_observations,
                    on_event,
                )
                on_event({"event": "complete", "phase": "complete"})
                round_completed = True
                if not keep_warm:
                    for view, rt in zip(views, runtimes):
                        try:
                            rt.stop()
                        except Exception:
                            pass
                return diagnosis_result
            min_views = int(multi.get("min_views", 1))
            if len(observations) < min_views: raise RuntimeError("minimum views not reached")
            ordered = [observations[k] for k in sorted(observations)]
            vals = [str(o["yolo_outcome"]) for o in ordered if o.get("yolo_outcome")]
            rule = dict(profile.get("rule", {}) if isinstance(profile.get("rule"), Mapping) else {})
            vision = profile.get("vision", {}) if isinstance(profile.get("vision"), Mapping) else {}
            if "expected_count" not in rule and vision.get("expected_count") is not None:
                rule["expected_count"] = vision["expected_count"]
            participants = vision.get("participants")
            normalized = []
            for observation in ordered:
                normalized.append(normalize_observation(profile, observation))
            # A runtime-provided verdict must not bypass the profile's
            # expected object count.  A stable but incomplete frame is a
            # diagnosable retry condition, not a winner.
            if _has_incomplete_expected_counts(profile, normalized):
                for rt in runtimes:
                    try:
                        rt.send({"command": "STOP_ADJUDICATION", "request_id": request.request_id})
                    except Exception:
                        pass
                diagnosis_result = self._diagnose_failure(
                    request,
                    profile,
                    ordered,
                    on_event,
                )
                on_event({"event": "complete", "phase": "complete"})
                round_completed = True
                if not keep_warm:
                    for view, rt in zip(views, runtimes):
                        try:
                            rt.stop()
                        except Exception:
                            pass
                return diagnosis_result
            # If every view supplies a runtime verdict, fuse those votes.
            # Otherwise evaluate the declared profile rule for each view so a
            # missing/legacy yolo_outcome cannot discard a camera's evidence.
            computed = [
                str(item.get("yolo_outcome")) if item.get("yolo_outcome") else evaluate_rule(rule, [item])
                for item in normalized
            ]
            if len(computed) > 1:
                yolo = fuse_yolo_outcomes(computed)
                if yolo is None: raise RuntimeError("no strict majority across views")
            else:
                yolo = computed[0] if computed else evaluate_rule(rule, normalized)
            on_event({"event":"phase", "phase":"verifying"}); cfg = profile.get("llm", {})
            cfg = cfg if isinstance(cfg, Mapping) else {}
            status, out = ("timeout", None) if cfg.get("enabled", True) else ("disabled", None)
            reask = None
            if cfg.get("enabled", True):
                paths = []
                for observation in ordered:
                    snapshot = observation.get("snapshot")
                    raw = snapshot.get("path") if isinstance(snapshot, Mapping) else None
                    if not isinstance(raw, str) or not raw.strip() or not Path(raw).is_absolute():
                        raise ValueError("snapshot.path must be an absolute path")
                    path = Path(raw).resolve()
                    # Real runtimes are constrained to this round's private
                    # directory.  Custom injected runtimes may use their own
                    # fixture directory for compatibility with older callers;
                    # those paths are still absolute, regular image files.
                    view_id = str(observation.get("view_id", "default"))
                    if view_id in strict_snapshot_roots:
                        try:
                            path = _snapshot_path(observation, strict_snapshot_roots[view_id])
                        except Exception:
                            raise ValueError("snapshot.path must stay inside task directory")
                    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"} or not path.is_file():
                        raise ValueError("snapshot.path must reference an existing JPEG or PNG")
                    paths.append(path)
                    cleanup_paths.add(path)
                verifier = self._round_llm(request)
                model_override = str(cfg.get("model") or "").strip() or None
                effort_override = str(cfg.get("reasoning_effort") or "").strip() or None
                remaining = max(0.0, deadline - time.monotonic())
                if verifier is None:
                    # The profile asks for verification but the deployment
                    # resolved no LLM: same semantics as profile-level
                    # disabled — detector-only result, never a round failure.
                    status, out = "disabled", None
                    on_log("[vision] no LLM provider attached; verification disabled")
                elif remaining > 0:
                    llm_timeout = min(
                        self._llm_timeout(profile, fallback_timeout), remaining
                    )
                    vr = verifier.verify(image_paths=paths, system_prompt=cfg.get("system_prompt", ""), user_prompt=cfg.get("user_prompt_template", ""), allowed_outcomes=cfg.get("allowed_outcomes", []), timeout_seconds=llm_timeout, model=model_override, reasoning_effort=effort_override)
                    status, out = vr.status, vr.outcome
                    if status not in {"success", "disabled"}:
                        # Otherwise the failure reason is unrecoverable after
                        # the fact: the result payload keeps only the outcome,
                        # and the re-ask log below fires on a dissent only.
                        on_log(f"[vision] verification {status}: {getattr(vr, 'error', None) or 'no error detail'}")
                    # A lone dissent from the verifier is re-asked once within
                    # the same budget: an answer corroborated by the re-ask may
                    # override, an answer that flips back confirms the detector,
                    # and anything else leaves the detector's result standing.
                    if (
                        status == "success"
                        and isinstance(out, str)
                        and yolo is not None
                        and out.strip() != str(yolo).strip()
                    ):
                        reask_remaining = max(0.0, deadline - time.monotonic())
                        if reask_remaining > 0:
                            reask_timeout = min(
                                self._llm_timeout(profile, fallback_timeout), reask_remaining
                            )
                            vr2 = verifier.verify(image_paths=paths, system_prompt=cfg.get("system_prompt", ""), user_prompt=cfg.get("user_prompt_template", ""), allowed_outcomes=cfg.get("allowed_outcomes", []), timeout_seconds=reask_timeout, model=model_override, reasoning_effort=effort_override)
                            reask = {"outcome": vr2.outcome, "status": vr2.status}
                        else:
                            reask = {"outcome": None, "status": "timeout"}
                        on_log(f"[vision] LLM dissented ({out}); re-ask -> {reask.get('status')}:{reask.get('outcome')}")
            decision = finalize_outcome(
                yolo_outcome=yolo,
                llm_outcome=out,
                llm_status=status,
                reask=reask,
                tie_value=str(rule.get("tie_value", "TIE")),
            )
            final = project_result(profile, decision, {**normalized[0], "views": normalized})
            final_command = {"command": "FINAL_RESULT", "request_id": request.request_id,
                             "outcome": final["outcome"],
                             # ``source`` is the compact runtime annotation
                             # understood by the C++ overlay; retain the
                             # canonical ``decision_source`` in the public
                             # Python result contract.
                             "source": final.get("decision_source", "provider"),
                             "decision_source": final.get("decision_source", "provider"),
                             "verification": final.get("verification", {})}
            for rt in runtimes:
                try:
                    rt.send(final_command)
                except Exception as exc:
                    on_log(f"[vision] FINAL_RESULT send failed: {exc}")
            on_event({"event":"result", **final})
            # Stop inference immediately after the verdict.  The resident
            # camera/RTSP pipeline remains alive, so the browser can continue
            # displaying the holding frame without spending YOLO cycles.
            for rt in runtimes:
                try:
                    rt.send({"command": "STOP_ADJUDICATION", "request_id": request.request_id})
                except Exception as exc:
                    on_log(f"[vision] STOP_ADJUDICATION send failed: {exc}")
            # The adjudication deadline bounds detection and LLM verification.
            # Holding starts only after a verdict exists, so consuming it from
            # the remaining adjudication budget would silently shorten the
            # game-owned post-result display contract.
            hold = float(profile.get("lifecycle", {}).get("post_result_hold_seconds", 0))
            if hold > 0:
                end = time.monotonic() + hold
                while time.monotonic() < end:
                    if is_cancelled(): raise RuntimeError("cancelled")
                    remaining = max(0, end-time.monotonic()); on_event({"event":"phase", "phase":"holding", "remaining_ms":int(remaining*1000)})
                    time.sleep(min(0.25, remaining))
            on_event({"event":"complete", "phase":"complete"}); round_completed = True
            if not keep_warm:
                for view, rt in zip(views, runtimes):
                    try:
                        rt.stop()
                    except Exception as exc:
                        on_log(f"[vision] runtime stop failed: {exc}")
                    vid = str(view.get("id", "default"))
                    if self._runtime_cache.get(vid) is rt:
                        self._runtime_cache.pop(vid, None)
                        self._runtime_signatures.pop(vid, None)
                        root = self._runtime_snapshot_dirs.pop(vid, None)
                        if root is not None:
                            import shutil
                            shutil.rmtree(root, ignore_errors=True)
            return final
        finally:
            # Only unlink paths that crossed the validation boundary above.
            # In particular, never trust an arbitrary snapshot path supplied
            # by a failed or malicious runtime event.
            for path in cleanup_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            # If a round failed or was cancelled, explicitly return each
            # resident runtime to idle.  Normal rounds already sent STOP
            # before holding; no process is restarted between requests.
            if round_started and not round_completed:
                for rt in runtimes:
                    try:
                        rt.send({"command": "CANCEL", "request_id": request.request_id})
                    except Exception:
                        pass
