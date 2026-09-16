#include "yolov8_postprocess.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

namespace {

OutputView view(const std::vector<float>& data, std::vector<int64_t> shape) {
    return OutputView{data.data(), std::move(shape)};
}

bool near(float value, float expected, float tolerance) {
    return std::fabs(value - expected) <= tolerance;
}

// 合成 COCO17 pose 输出 [1, 56, anchors]：模型侧 box/关键点已是 640 letterbox
// 像素坐标，类别分与关键点置信度已是概率。验证 cls 筛选、xywh→xyxy、letterbox
// 反映射、(x,y,conf)×17 关键点布局、越界 clamp 与同类 NMS 抑制。
void test_decode_pose_output() {
    const int keypoint_count = 17;
    const int anchors = 4;
    const int channels = kPoseBoxChannels + kPoseClassChannels + 3 * keypoint_count;
    assert(channels == 56);
    std::vector<float> det(static_cast<size_t>(channels * anchors), 0.0f);
    auto at = [&det, anchors](int channel, int anchor) -> float& {
        return det[static_cast<size_t>(channel * anchors + anchor)];
    };
    // anchor 0: person 0.9，box 模型坐标 (270,280)-(370,360)
    at(0, 0) = 320.0f; at(1, 0) = 320.0f; at(2, 0) = 100.0f; at(3, 0) = 80.0f;
    at(4, 0) = 0.9f;
    // kpt 0: (300,300) conf 0.9；kpt 1: (340,300) conf 0.1（低置信只透传不过滤）
    at(5 + 0 * 3 + 0, 0) = 300.0f;
    at(5 + 0 * 3 + 1, 0) = 300.0f;
    at(5 + 0 * 3 + 2, 0) = 0.9f;
    at(5 + 1 * 3 + 0, 0) = 340.0f;
    at(5 + 1 * 3 + 1, 0) = 300.0f;
    at(5 + 1 * 3 + 2, 0) = 0.1f;
    // anchor 1: person 0.85，独立框；kpt 0 在负坐标，反映射后应 clamp 到 (0,0)
    at(0, 1) = 100.0f; at(1, 1) = 100.0f; at(2, 1) = 60.0f; at(3, 1) = 60.0f;
    at(4, 1) = 0.85f;
    at(5 + 0 * 3 + 0, 1) = -10.0f;
    at(5 + 0 * 3 + 1, 1) = -10.0f;
    at(5 + 0 * 3 + 2, 1) = 0.9f;
    // anchor 2: person 0.8，与 anchor 0 高度重叠，应被 NMS 抑制
    at(0, 2) = 322.0f; at(1, 2) = 322.0f; at(2, 2) = 100.0f; at(3, 2) = 80.0f;
    at(4, 2) = 0.8f;
    // anchor 3: person 0.2 低于 conf，应被过滤
    at(0, 3) = 160.0f; at(1, 3) = 160.0f; at(2, 3) = 40.0f; at(3, 3) = 40.0f;
    at(4, 3) = 0.2f;

    // letterbox 几何：scale=2（源像素/模型像素），pad_x=4，pad_y=8，1280x1280
    auto candidates = decode_pose_output(view(det, {1, channels, anchors}), keypoint_count,
                                         0.25f, 2.0f, 4, 8, 1280, 1280);
    assert(candidates.size() == 3);

    const std::vector<int> kept = class_aware_nms(candidates, 0.45f, 100);
    assert(kept.size() == 2);

    const PoseCandidate* main = nullptr;
    const PoseCandidate* other = nullptr;
    for (int index : kept) {
        if (candidates[static_cast<size_t>(index)].score > 0.85f) {
            main = &candidates[static_cast<size_t>(index)];
        } else {
            other = &candidates[static_cast<size_t>(index)];
        }
    }
    assert(main && other);
    assert(main->class_id == 0);
    assert(main->keypoint_count == 17);
    assert(near(main->score, 0.9f, 1e-6));
    // 模型坐标 (270,280,370,360) → 图像坐标 ((270-4)*2,(280-8)*2,(370-4)*2,(360-8)*2)
    assert(near(main->box.x, 532.0f, 1e-2));
    assert(near(main->box.y, 544.0f, 1e-2));
    assert(near(main->box.width, 200.0f, 1e-2));
    assert(near(main->box.height, 160.0f, 1e-2));
    assert(near(other->box.x, 132.0f, 1e-2));
    assert(near(other->box.y, 124.0f, 1e-2));
    assert(near(other->score, 0.85f, 1e-6));

    // 关键点反映射与置信度透传
    assert(near(main->keypoints[0].x, 592.0f, 1e-2));
    assert(near(main->keypoints[0].y, 584.0f, 1e-2));
    assert(near(main->keypoints[0].confidence, 0.9f, 1e-6));
    assert(near(main->keypoints[1].x, 672.0f, 1e-2));
    assert(near(main->keypoints[1].y, 584.0f, 1e-2));
    assert(near(main->keypoints[1].confidence, 0.1f, 1e-6));
    // 未写入的关键点全零透传
    assert(near(main->keypoints[16].confidence, 0.0f, 1e-6));
    // 负坐标 clamp 到图像边界
    assert(near(other->keypoints[0].x, 0.0f, 1e-6));
    assert(near(other->keypoints[0].y, 0.0f, 1e-6));
    assert(near(other->keypoints[0].confidence, 0.9f, 1e-6));

    std::cout << "decode_pose_output test passed\n";
}

// 合成 hand21 输出 [1, 68, anchors]：验证 21 点布局透传与 NMS。
void test_decode_hand_output() {
    const int keypoint_count = 21;
    const int anchors = 2;
    const int channels = kPoseBoxChannels + kPoseClassChannels + 3 * keypoint_count;
    assert(channels == 68);
    std::vector<float> det(static_cast<size_t>(channels * anchors), 0.0f);
    auto at = [&det, anchors](int channel, int anchor) -> float& {
        return det[static_cast<size_t>(channel * anchors + anchor)];
    };
    // anchor 0: hand 0.7，box (100,100,80,80)，kpt0 (90,90) conf 0.8、kpt20 (170,170) conf 0.6
    at(0, 0) = 140.0f; at(1, 0) = 140.0f; at(2, 0) = 80.0f; at(3, 0) = 80.0f;
    at(4, 0) = 0.7f;
    at(5 + 0 * 3 + 0, 0) = 90.0f;
    at(5 + 0 * 3 + 1, 0) = 90.0f;
    at(5 + 0 * 3 + 2, 0) = 0.8f;
    at(5 + 20 * 3 + 0, 0) = 170.0f;
    at(5 + 20 * 3 + 1, 0) = 170.0f;
    at(5 + 20 * 3 + 2, 0) = 0.6f;
    // anchor 1: hand 0.3，与 anchor 0 重叠 → NMS 抑制
    at(0, 1) = 142.0f; at(1, 1) = 142.0f; at(2, 1) = 80.0f; at(3, 1) = 80.0f;
    at(4, 1) = 0.3f;

    auto candidates = decode_pose_output(view(det, {1, channels, anchors}), keypoint_count,
                                         0.25f, 1.0f, 0, 0, 640, 640);
    assert(candidates.size() == 1);
    const PoseCandidate& hand = candidates.front();
    assert(hand.keypoint_count == 21);
    assert(near(hand.score, 0.7f, 1e-6));
    assert(near(hand.box.x, 100.0f, 1e-2));
    assert(near(hand.keypoints[0].x, 90.0f, 1e-2));
    assert(near(hand.keypoints[0].confidence, 0.8f, 1e-6));
    assert(near(hand.keypoints[20].x, 170.0f, 1e-2));
    assert(near(hand.keypoints[20].confidence, 0.6f, 1e-6));
    const std::vector<int> kept = class_aware_nms(candidates, 0.45f, 100);
    assert(kept.size() == 1);
    std::cout << "decode_hand_output test passed\n";
}

void test_keypoint_count_from_channels() {
    assert(keypoint_count_from_channels(56) == 17);
    assert(keypoint_count_from_channels(68) == 21);
    bool threw = false;
    try { keypoint_count_from_channels(57); } catch (const std::exception&) { threw = true; }
    assert(threw);
    threw = false;
    try { keypoint_count_from_channels(5); } catch (const std::exception&) { threw = true; }
    assert(threw);
    std::cout << "keypoint_count_from_channels test passed\n";
}

// 全零输入（无任何高置信 anchor）应产出空候选。
void test_decode_empty() {
    const int anchors = 8;
    const int channels = 68;
    std::vector<float> det(static_cast<size_t>(channels * anchors), 0.0f);
    auto candidates = decode_pose_output(view(det, {1, channels, anchors}), 21, 0.25f,
                                         1.0f, 0, 0, 640, 640);
    assert(candidates.empty());
    const std::vector<int> kept = class_aware_nms(candidates, 0.45f, 100);
    assert(kept.empty());
    std::cout << "decode empty test passed\n";
}

void test_label_for_class() {
    assert(label_for_class(0, {"hand"}) == "hand");
    assert(label_for_class(1, {"hand"}) == "class_1");
    std::cout << "label test passed\n";
}

}  // namespace

int main() {
    test_decode_pose_output();
    test_decode_hand_output();
    test_keypoint_count_from_channels();
    test_decode_empty();
    test_label_for_class();
    std::cout << "yolov8 pose postprocess tests passed\n";
    return 0;
}
