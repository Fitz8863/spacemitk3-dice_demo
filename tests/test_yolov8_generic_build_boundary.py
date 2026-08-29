from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vision" / "yolov8_objdetect" / "src" / "main.cpp"
CMAKE = ROOT / "vision" / "yolov8_objdetect" / "CMakeLists.txt"


def test_generic_build_does_not_link_or_include_dice_verifier_directly():
    source = SOURCE.read_text(encoding="utf-8")
    cmake = CMAKE.read_text(encoding="utf-8")
    assert 'option(VISION_GENERIC_ONLY' in cmake
    assert 'src/llm_dice_verifier.cpp' not in cmake.split('if(NOT VISION_GENERIC_ONLY)', 1)[0]
    assert '#include "llm_dice_verifier.h"' in source
    assert '#ifdef VISION_GENERIC_ONLY' in source


def test_generic_control_path_is_not_gated_by_dice_judgment():
    source = SOURCE.read_text(encoding="utf-8")
    marker = 'if (a.control_fd < 0) {\n                    judgment = judge_dice'
    assert marker in source
    assert 'if (!item->detections.empty()' in source


def test_generic_stability_signature_ignores_detector_jitter():
    """Confidence/one-pixel box noise must not reset a stable round forever."""
    source = SOURCE.read_text(encoding="utf-8")
    start = source.index("static std::string detection_signature")
    end = source.index("// Ultralytics-style vivid palette", start)
    signature = source[start:end]
    assert "d.confidence" not in signature
    assert "d.x1" in signature and "d.y1" in signature
    assert "d.x2" in signature and "d.y2" in signature
    assert "quant" in signature.lower()


def test_generic_stub_accepts_legacy_config_shape_without_linking_verifier():
    source = SOURCE.read_text(encoding="utf-8")
    stub = source.split("#ifdef VISION_GENERIC_ONLY", 1)[1].split(
        "#else", 1
    )[0]
    assert "std::string url" in stub
    assert "std::string api_key" in stub
    assert "std::string model" in stub
    assert "int timeout_seconds" in stub
    assert "std::string system_prompt" in stub
    assert "std::string user_prompt_template" in stub
