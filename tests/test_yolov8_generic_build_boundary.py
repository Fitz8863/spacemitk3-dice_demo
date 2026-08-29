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
