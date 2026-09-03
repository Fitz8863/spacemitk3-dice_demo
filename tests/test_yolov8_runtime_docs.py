from pathlib import Path


README = Path(__file__).resolve().parents[1] / "vision/yolov8_adjudicator/README.md"


def test_runtime_readme_describes_private_profile_driven_protocol():
    text = README.read_text(encoding="utf-8")
    for phrase in (
        "vision_yolov8_adjudicator",
        "START_ADJUDICATION",
        "control-fd",
        "event-fd",
        "snapshot-dir",
        "manifest.json",
        "vision_profile",
        "video.webrtc_base_url",
        "MediaMTX",
        "post_result_hold_seconds",
        "diagnostic_snapshot",
        "yolo_detection_seconds",
        "yolo_fallback",
    ):
        assert phrase in text


def test_runtime_readme_has_no_dice_only_cli_or_cpp_llm_contract():
    text = README.read_text(encoding="utf-8")
    stale_sections = (
        "## 双方骰子裁决",
        "--llm-url URL",
        "--llm-model NAME",
        "--llm-timeout N",
        "只有 YOLO 连续得到相同的有效 5+5 结果达到",
        "模板支持 `{left_name}`、`{right_name}`、`{left_sum}`、`{right_sum}`",
    )
    for phrase in stale_sections:
        assert phrase not in text
