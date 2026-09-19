from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vision" / "yolov8_adjudicator" / "src" / "main.cpp"
CMAKE = ROOT / "vision" / "yolov8_adjudicator" / "CMakeLists.txt"
RUNTIME_CONFIG = ROOT / "backend" / "games" / "dice" / "adjudicator_config.json"


def test_shared_runtime_config_stays_deleted():
    """共享部署默认已于 2026-09-20 删除：runtime_config 必填、每游戏一份。

    复活它会同时复活「游戏忘了声明就静默用别人摄像头」的负价值兜底；
    要恢复先改 resolver + validate_profile 并回滚本条。
    """
    assert not (ROOT / "vision" / "yolov8_adjudicator" / "config.json").exists()


def test_runtime_has_adjudicator_directory_and_no_objdetect_directory():
    assert SOURCE.parent.parent.name == "yolov8_adjudicator"
    assert not (ROOT / "vision" / "yolov8_objdetect").exists()


def test_divider_assist_runs_only_during_active_adjudication():
    source = SOURCE.read_text(encoding="utf-8")
    assert "a.divider_detection_enabled && adjudication_active.load()" in source


def test_generic_build_does_not_link_or_include_dice_verifier_directly():
    source = SOURCE.read_text(encoding="utf-8")
    cmake = CMAKE.read_text(encoding="utf-8")
    assert 'llm_dice_verifier' not in cmake
    assert '#include "llm_dice_verifier.h"' not in source
    assert '#ifdef VISION_GENERIC_ONLY' not in source


def test_generic_control_path_is_not_gated_by_dice_judgment():
    source = SOURCE.read_text(encoding="utf-8")
    assert "judge_dice" not in source
    assert "DiceJudgment" not in source
    # The gate is generic detector evidence plus profile data (region count and
    # split), never a game-specific judgment.
    assert "const bool evidence_usable = region_ok && divider_ready;" in source
    assert "const bool region_gate = a.expected_count > 0;" in source


def test_generic_control_path_runs_configured_divider_assist():
    source = SOURCE.read_text(encoding="utf-8")
    assert "divider_detection_enabled" in source
    assert "detect_black_divider" in source
    assert "detect_red_blue_divider" in source
    assert '\\"divider\\"' in source
    assert "draw_scene_assist" in source


def test_scene_divider_prefers_colour_split_and_falls_back_to_the_line():
    """The gate asks "is the left/right split readable", not which signal proved it."""
    source = SOURCE.read_text(encoding="utf-8")
    # The active adjudication phase resolves the divider through the dispatcher.
    assert "divider_assist.valid = detect_scene_divider(bgr, divider_assist);" in source
    start = source.index("static bool detect_scene_divider(")
    end = source.index("static void draw_scene_assist", start)
    dispatcher = source[start:end]
    assert "if (detect_red_blue_divider(bgr, divider)) return true;" in dispatcher
    assert "return detect_black_divider(bgr, divider);" in dispatcher
    # Both signals stay available: the colour split is the glare-proof one, the
    # printed line remains for scenes without a coloured mat.
    assert "static bool detect_red_blue_divider(const cv::Mat& bgr, DividerLine& divider)" in source
    assert "static bool detect_black_divider(const cv::Mat& bgr, DividerLine& divider)" in source


def test_control_fd_can_start_yolo_when_config_default_is_disabled():
    source = SOURCE.read_text(encoding="utf-8")
    assert "const bool yolo_runtime_enabled = a.yolov8_enabled || runtime_has_control" in source
    assert "if (!a.yolov8_enabled && a.control_fd < 0)" in source


def test_generic_stability_signature_uses_only_detected_categories():
    """Confidence and all box geometry must be irrelevant to stability."""
    source = SOURCE.read_text(encoding="utf-8")
    start = source.index("static std::string detection_signature")
    end = source.index("// Ultralytics-style vivid palette", start)
    signature = source[start:end]
    assert "d.class_id" in signature
    assert "d.confidence" not in signature
    for coordinate in ("d.x1", "d.y1", "d.x2", "d.y2"):
        assert coordinate not in signature
    assert "quant" not in signature.lower()


def test_runtime_emits_category_stability_progress():
    source = SOURCE.read_text(encoding="utf-8")
    assert '\\"event\\":\\"progress\\"' in source
    assert '\\"stable_count\\":' in source
    assert '\\"stable_frames\\":' in source


def test_stability_streak_only_advances_on_frames_a_profile_can_adjudicate():
    """A frame missing objects, the divider, or the profile's layout resets it.

    Three conditions gate the streak: a located divider, the profile's object
    count in each region, and an unchanged per-region class multiset.  Counting
    frames that fail them let the stable_frames gate be satisfied with evidence
    the profile must reject -- first the visible "48/30" overrun, then stable
    observations rejected downstream as incomplete while the detection timeout
    never got its budget.
    """
    source = SOURCE.read_text(encoding="utf-8")
    start = source.index("const bool region_gate = a.expected_count > 0;")
    end = source.index("if (evidence_usable &&", start)
    streak = source[start:end]
    assert "const bool evidence_usable = region_ok && divider_ready;" in streak
    assert "if (!evidence_usable) {" in streak
    assert "generic_stable_count.store(0);" in streak
    # The layout check uses the profile's count and the frame's effective split,
    # and feeds a per-region signature so any point-value change on either side
    # restarts the streak.
    assert "region_layout_usable(item->detections, item->width, item->height," in streak
    assert "a.expected_count, region_boundary," in streak
    assert "&region_signature_text)" in streak
    # The observation gate reuses the same predicate.
    assert "stable_count >= a.stable_frames" in source
    assert "!item->detections.empty() && divider_ready &&" not in source


def test_stability_split_uses_the_located_divider_over_the_configured_ratio():
    """The gate must split where the scene splits, not where the manifest says.

    ``vision.divider.position`` is only a fallback: as long as the gate used it
    unconditionally the detected boundary -- the thing that actually keeps the
    split right when the camera moves -- never reached the count check, so
    ``grouping: divider_regions`` changed whether the divider gated a frame but
    never where left ended.  The profile's ``--region-position`` may therefore
    appear only as the fallback assignment.
    """
    source = SOURCE.read_text(encoding="utf-8")
    start = source.index("double region_boundary = a.region_position;")
    end = source.index("if (evidence_usable &&", start)
    split = source[start:end]
    # The fallback is the launch argument, overridden by a located divider.
    assert "if (a.divider_detection_enabled && divider_assist.valid) {" in split
    assert "region_boundary = detected / region_extent;" in split
    assert "a.expected_count, region_boundary," in split
    # The configured ratio can no longer reach the count check directly.
    assert "a.expected_count, a.region_position," not in source
    # And the fallback stays inside the frame, so a degenerate divider cannot
    # silently push every detection into one region.
    assert "detected > 0.0 && detected < region_extent" in split


def test_region_count_gate_is_profile_data_and_off_by_default():
    """The runtime learns the count from the profile, never from a game rule.

    ``--expected-count 0`` (the default) preserves the generic behaviour every
    profile without ``vision.expected_count`` relies on, so adding the gate
    cannot change another game's stability semantics.
    """
    source = SOURCE.read_text(encoding="utf-8")
    assert "int expected_count = 0;" in source
    assert '"--expected-count"' in source
    assert '"--region-position"' in source
    assert '"--region-orientation"' in source
    start = source.index("static bool region_layout_usable(")
    end = source.index("// Ultralytics-style vivid palette", start)
    gate = source[start:end]
    assert "int expected_count" in gate
    assert "regions[0].size() == expected" in gate
    assert "regions[1].size() == expected" in gate


def test_stability_streak_stops_once_the_stable_observation_is_sent():
    """Bounding proof: the counter cannot be reported above stable_frames.

    Counting only happens while no observation has been sent, and every usable
    frame from stable_count == stable_frames onward sends one, so the streak
    never exceeds the threshold in an emitted progress event.
    """
    source = SOURCE.read_text(encoding="utf-8")
    block_start = source.index("if (a.control_fd >= 0 && adjudication_active.load() &&")
    block_end = source.index("if (!a.no_display || rtsp_streamer.running())", block_start)
    block = source[block_start:block_end]
    assert "!generic_observation_sent.load()) {" in block
    assert block.index("generic_observation_sent.exchange(true)") > block.index(
        "generic_last_reported_count.exchange(stable_count)"
    )


def test_runtime_does_not_embed_legacy_dice_llm_verifier():
    source = SOURCE.read_text(encoding="utf-8")
    assert not (SOURCE.parent / "llm_dice_verifier.cpp").exists()
    assert not (SOURCE.parent / "llm_dice_verifier.h").exists()
    assert "llm_dice_verifier" not in source


def test_runtime_has_no_cpp_llm_or_legacy_dice_state_machine():
    source = SOURCE.read_text(encoding="utf-8")
    for symbol in (
        "LlmDiceVerifier",
        "AsyncLlmVerifier",
        "LlmRequest",
        "LlmResponse",
        "LlmVerificationState",
        "DiceResultSnapshot",
        "--llm-url",
        "--llm-model",
        "--llm-timeout",
        "--no-llm",
        "rejudge_on_change",
    ):
        assert symbol not in source


def test_runtime_config_owns_hardware_only_settings():
    """每游戏的硬件配置只放"这台板子+这张桌子"的属性。

    LLM 凭证在 LLM 组件；模型/稳定帧/阈值/分界线门控这些**游戏语义**参数在
    游戏 manifest（并按游戏各自可覆盖）。此前它们混在共享文件里、会被所有
    游戏继承——2026-09-15 起归位，这里把边界钉住（共享文件本身已于
    2026-09-20 删除，现检查的是 dice 的 per-game 硬件文件）。
    """
    config = json.loads(RUNTIME_CONFIG.read_text(encoding="utf-8"))
    assert "llm" not in config
    assert "rejudge_on_change" not in config
    for game_owned in (
        "model",
        "stable_frames",
        "conf",
        "divider_detection",
        "display_enabled",
        "yolov8_enabled",
    ):
        assert game_owned not in config, f"{game_owned} 属于游戏 manifest，不该在硬件配置里"
    # The hardware it does own must stay.
    for hardware in ("camera", "width", "height", "fps", "ep_affinity", "focus", "zoom"):
        assert hardware in config, f"{hardware} 是硬件属性，必须留在硬件配置里"
    # Without rtsp.enabled the provider never adds --rtsp, i.e. no stream at all.
    assert config["rtsp"]["enabled"] is True


def test_runtime_resolves_config_relative_model_path():
    source = SOURCE.read_text(encoding="utf-8")
    assert "config_path.parent_path() / model_path" in source
    assert "std::filesystem::absolute(std::filesystem::path(path))" in source
