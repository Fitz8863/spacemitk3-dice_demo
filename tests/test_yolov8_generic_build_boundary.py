from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vision" / "yolov8_objdetect" / "src" / "main.cpp"
CMAKE = ROOT / "vision" / "yolov8_objdetect" / "CMakeLists.txt"
RUNTIME_CONFIG = ROOT / "backend" / "games" / "dice" / "adjudicator_config.json"


def test_shared_runtime_config_stays_deleted():
    """共享部署默认已于 2026-09-20 删除：runtime_config 必填、每游戏一份。

    复活它会同时复活「游戏忘了声明就静默用别人摄像头」的负价值兜底；
    要恢复先改 resolver + validate_profile 并回滚本条。
    """
    assert not (ROOT / "vision" / "yolov8_objdetect" / "config.json").exists()


def test_runtime_uses_objdetect_directory_and_no_adjudicator_directory():
    assert SOURCE.parent.parent.name == "yolov8_objdetect"
    assert not (ROOT / "vision" / "yolov8_adjudicator").exists()


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
    # The stability gate is pure detection evidence (non-empty frame plus a
    # located divider when assistance is on), never a game-specific judgment.
    assert "const bool evidence_usable = !item->detections.empty() && divider_ready;" in source


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
    # Colour split first, dark line only as the fallback (2026-09-21: the
    # dispatcher also tags which signal won for the rate-limited log line).
    assert 'if (detect_red_blue_divider(bgr, divider)) source = "red/blue";' in dispatcher
    assert 'else if (detect_black_divider(bgr, divider)) source = "black";' in dispatcher
    assert "else return false;" in dispatcher
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


def test_stability_streak_only_advances_on_frames_the_consumer_can_use():
    """A frame missing objects or the divider resets it; game completeness moved out.

    Two conditions gate the streak (jsonl-events-v2 semantics): a located
    divider when assistance is on, and an unchanged whole-frame class multiset
    with at least one detection.  Per-side completeness such as "five dice per
    side" is the Python provider's count check on the stable observation, so
    the runtime must not carry any region gate of its own.
    """
    source = SOURCE.read_text(encoding="utf-8")
    start = source.index("const std::string signature = detection_signature(item->detections);")
    end = source.index("if (evidence_usable &&", start)
    streak = source[start:end]
    assert "const bool evidence_usable = !item->detections.empty() && divider_ready;" in streak
    assert "if (!evidence_usable) {" in streak
    assert "generic_stable_count.store(0);" in streak
    # Stability compares the whole-frame class multiset only.
    assert "const std::string signature = detection_signature(item->detections);" in streak
    # The observation gate reuses the same predicate.
    assert "stable_count >= a.stable_frames" in source
    # No region gate may come back: the count/split checks live in Python now.
    for forbidden in ("region_layout_usable", "expected_count", "region_position",
                      "region_orientation", '"--expected-count"', '"--region-position"',
                      '"--region-orientation"'):
        assert forbidden not in source, forbidden


def test_region_gate_lives_in_python_not_the_runtime():
    """Per-side completeness is the provider's concern since jsonl-events-v2.

    The runtime is a pure object detector: it never learns how many objects a
    side needs or where the split falls.  ``expected_count`` stays in the game
    manifest and is validated by ``_has_incomplete_expected_counts`` on the
    stable observation, so an incomplete layout is a fast diagnosed retry
    instead of a silent streak reset.
    """
    source = SOURCE.read_text(encoding="utf-8")
    for forbidden in (
        "int expected_count",
        '"--expected-count"',
        '"--region-position"',
        '"--region-orientation"',
        "static bool region_layout_usable(",
    ):
        assert forbidden not in source, forbidden
    provider = (ROOT / "backend" / "components" / "vision_yolov8_objdetect" / "provider.py").read_text(encoding="utf-8")
    assert "_has_incomplete_expected_counts(profile, normalized)" in provider
    # And the launcher no longer forwards region arguments to the runtime.
    process = (ROOT / "backend" / "components" / "vision_yolov8_objdetect" / "process.py").read_text(encoding="utf-8")
    assert '"--expected-count"' not in process
    assert '"--region-position"' not in process


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
    """游戏的 runtime 配置文件持有 C++ 消费的全部参数（2026-09-20 起）。

    归属轴是「C++ runtime 消费 vs Python 框架消费」：model/conf/
    stable_frames/divider_detection + 摄像头/EP/焦距等硬件住这里
    （backend/games/<id>/adjudicator_config.json）；class_map/grouping/
    expected_count 这些游戏语义（Python 规则引擎消费）留在 manifest。
    """
    config = json.loads(RUNTIME_CONFIG.read_text(encoding="utf-8"))
    assert "llm" not in config
    assert "rejudge_on_change" not in config
    # Game semantics consumed by the Python rule engine stay in the manifest.
    for python_owned in ("class_map", "grouping", "expected_count"):
        assert python_owned not in config, f"{python_owned} 属于游戏 manifest（Python 消费）"
    # The C++-consumed runtime parameters moved here from the manifest.
    for runtime_param in ("model", "conf", "stable_frames", "divider_detection"):
        assert runtime_param in config, f"{runtime_param} 是 runtime 参数，必须在此文件里"
    assert 0 < config["conf"] < 1
    # The hardware it owns must stay.
    for hardware in ("camera", "width", "height", "fps", "ep_affinity", "focus", "zoom"):
        assert hardware in config, f"{hardware} 是硬件属性，必须留在此文件里"
    # Without rtsp.enabled the provider never adds --rtsp, i.e. no stream at all.
    assert config["rtsp"]["enabled"] is True


def test_runtime_resolves_config_relative_model_path():
    source = SOURCE.read_text(encoding="utf-8")
    assert "config_path.parent_path() / model_path" in source
    assert "std::filesystem::absolute(std::filesystem::path(path))" in source
