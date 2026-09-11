from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vision" / "yolov8_adjudicator" / "src" / "main.cpp"
CMAKE = ROOT / "vision" / "yolov8_adjudicator" / "CMakeLists.txt"
RUNTIME_CONFIG = ROOT / "vision" / "yolov8_adjudicator" / "config.json"


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
    # The gate is generic detector evidence, never a game-specific judgment.
    assert "const bool evidence_usable = !item->detections.empty() && divider_ready;" in source


def test_generic_control_path_runs_configured_divider_assist():
    source = SOURCE.read_text(encoding="utf-8")
    assert "divider_detection_enabled" in source
    assert "detect_black_divider" in source
    assert '\\"divider\\"' in source
    assert "draw_scene_assist" in source


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
    """A frame missing objects or the divider must reset, not grow, the streak.

    Counting unusable frames let the stable_frames gate be satisfied with
    evidence gathered while no divider was visible, and produced the visible
    "48/30" overrun on the analysis page.
    """
    source = SOURCE.read_text(encoding="utf-8")
    start = source.index("const std::string signature = detection_signature(item->detections);")
    end = source.index("if (evidence_usable &&", start)
    streak = source[start:end]
    assert "const bool evidence_usable = !item->detections.empty() && divider_ready;" in streak
    assert "if (!evidence_usable) {" in streak
    assert "generic_stable_count.store(0);" in streak
    # The observation gate reuses the same predicate.
    assert "stable_count >= a.stable_frames" in source
    assert "!item->detections.empty() && divider_ready &&" not in source


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
    config = json.loads(RUNTIME_CONFIG.read_text(encoding="utf-8"))
    assert "llm" not in config
    assert "rejudge_on_change" not in config
    assert config["yolov8_enabled"] is False
    assert config["divider_detection"] is False


def test_runtime_resolves_config_relative_model_path():
    source = SOURCE.read_text(encoding="utf-8")
    assert "config_path.parent_path() / model_path" in source
    assert "std::filesystem::absolute(std::filesystem::path(path))" in source
