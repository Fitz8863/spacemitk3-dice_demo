#include "yolov8_postprocess.h"

#include <cassert>
#include <cstdint>
#include <cmath>
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

// 标准 2 输出布局：合成 4 个 anchor（2 类），验证 xywh 解码、类别分（已激活概率）
// 直接与 conf 比较、conf 过滤、同类 NMS 抑制、mask 系数切片与轮廓组装。
void test_decode_standard2() {
    const int anchors = 4;
    const int classes = 2;
    const int channels = 4 + classes + kMaskChannels;
    std::vector<float> det(static_cast<size_t>(channels * anchors), 0.0f);
    auto at = [&det, anchors](int channel, int anchor) -> float& {
        return det[static_cast<size_t>(channel * anchors + anchor)];
    };
    // anchor 0: class 0，高置信度（类别分是图内已激活的概率，直接使用）
    at(0, 0) = 320.0f; at(1, 0) = 320.0f; at(2, 0) = 100.0f; at(3, 0) = 80.0f;
    at(4 + 0, 0) = 0.9f;
    // anchor 1: class 1，独立框
    at(0, 1) = 100.0f; at(1, 1) = 100.0f; at(2, 1) = 60.0f; at(3, 1) = 60.0f;
    at(4 + 1, 1) = 0.85f;
    // anchor 2: class 0，与 anchor 0 高度重叠，应被 NMS 抑制
    at(0, 2) = 322.0f; at(1, 2) = 322.0f; at(2, 2) = 100.0f; at(3, 2) = 80.0f;
    at(4 + 0, 2) = 0.8f;
    // anchor 3: class 0，0.2 低于 conf，应被过滤
    at(0, 3) = 160.0f; at(1, 3) = 160.0f; at(2, 3) = 40.0f; at(3, 3) = 40.0f;
    at(4 + 0, 3) = 0.2f;
    // anchor 0 的 mask 系数：仅通道 0 有效
    at(4 + classes + 0, 0) = 1.0f;

    // proto：通道 0 在模型坐标 (270,280)-(370,360) 区域为正 logit
    std::vector<float> proto(static_cast<size_t>(kMaskChannels * 160 * 160), 0.0f);
    for (int y = 70; y < 90; ++y) {
        for (int x = 67; x < 93; ++x) {
            proto[static_cast<size_t>(y * 160 + x)] = 1.0f;
        }
    }

    // letterbox 几何：scale=2（源像素/模型像素），pad_x=4，pad_y=8，1280x1280
    auto candidates = decode_standard2(view(det, {1, channels, anchors}), 0.25f,
                                       2.0f, 4, 8, 1280, 1280);
    assert(candidates.size() == 3);

    const std::vector<int> kept = class_aware_nms(candidates, 0.45f, 100);
    assert(kept.size() == 2);

    const Candidate* ball = nullptr;
    const Candidate* other = nullptr;
    for (int index : kept) {
        if (candidates[static_cast<size_t>(index)].class_id == 0 && !ball) {
            ball = &candidates[static_cast<size_t>(index)];
        } else {
            other = &candidates[static_cast<size_t>(index)];
        }
    }
    assert(ball && other);
    assert(other->class_id == 1);
    assert(near(ball->score, 0.9f, 1e-6));
    assert(near(other->score, 0.85f, 1e-6));
    // 模型坐标 (270,280,370,360) → 图像坐标 ((270-4)*2,(280-8)*2,(370-4)*2,(360-8)*2)
    assert(near(ball->box.x, 532.0f, 1e-2));
    assert(near(ball->box.y, 544.0f, 1e-2));
    assert(near(ball->box.width, 200.0f, 1e-2));
    assert(near(ball->box.height, 160.0f, 1e-2));
    assert(near(other->box.x, 132.0f, 1e-2));
    assert(near(other->box.y, 124.0f, 1e-2));

    const auto contours = build_contours(*ball, view(proto, {1, kMaskChannels, 160, 160}),
                                         2.0f, 4, 8, 1280, 1280);
    assert(!contours.empty());
    for (const auto& contour : contours) {
        for (const auto& point : contour) {
            assert(point.x >= 532 - 8 && point.x <= 732 + 8);
            assert(point.y >= 544 - 8 && point.y <= 704 + 8);
        }
    }
    std::cout << "decode_standard2 test passed\n";
}

// SpaceMIT 13 输出布局：DFL 全零分布（softmax 期望 = 7.5）+ 单 anchor 独热类别分，
// 锁定搬移后的解码行为。
void test_decode_spacemit13() {
    std::vector<float> box0(64 * 80 * 80, 0.0f);
    std::vector<float> score0(80 * 80 * 80, 0.0f);
    std::vector<float> sum0(80 * 80, 0.0f);
    std::vector<float> box1(64 * 40 * 40, 0.0f);
    std::vector<float> score1(80 * 40 * 40, 0.0f);
    std::vector<float> sum1(40 * 40, 0.0f);
    std::vector<float> box2(64 * 20 * 20, 0.0f);
    std::vector<float> score2(80 * 20 * 20, 0.0f);
    std::vector<float> sum2(20 * 20, 0.0f);
    std::vector<float> coeff0(32 * 80 * 80, 0.0f);
    std::vector<float> coeff1(32 * 40 * 40, 0.0f);
    std::vector<float> coeff2(32 * 20 * 20, 0.0f);
    std::vector<float> proto(static_cast<size_t>(kMaskChannels * 160 * 160), 0.0f);

    const int anchor = 20 * 80 + 10;  // grid (10, 20)
    score0[static_cast<size_t>(32 * 6400 + anchor)] = 0.9f;
    sum0[static_cast<size_t>(anchor)] = 0.9f;
    coeff0[static_cast<size_t>(0 * 6400 + anchor)] = 1.0f;
    coeff0[static_cast<size_t>(1 * 6400 + anchor)] = -1.0f;
    for (int y = 26; y < 56; ++y) {
        for (int x = 6; x < 36; ++x) {
            proto[static_cast<size_t>(y * 160 + x)] = 1.0f;
        }
    }

    std::vector<OutputView> views;
    views.push_back(view(box0, {1, 64, 80, 80}));
    views.push_back(view(score0, {1, 80, 80, 80}));
    views.push_back(view(sum0, {1, 1, 80, 80}));
    views.push_back(view(box1, {1, 64, 40, 40}));
    views.push_back(view(score1, {1, 80, 40, 40}));
    views.push_back(view(sum1, {1, 1, 40, 40}));
    views.push_back(view(box2, {1, 64, 20, 20}));
    views.push_back(view(score2, {1, 80, 20, 20}));
    views.push_back(view(sum2, {1, 1, 20, 20}));
    views.push_back(view(coeff0, {1, 32, 80, 80}));
    views.push_back(view(coeff1, {1, 32, 40, 40}));
    views.push_back(view(coeff2, {1, 32, 20, 20}));
    views.push_back(view(proto, {1, 32, 160, 160}));

    auto candidates = decode_spacemit13(views, 0.25f, 1.0f, 0, 0, 640, 640);
    assert(candidates.size() == 1);
    const Candidate& only = candidates.front();
    assert(near(only.score, 0.9f, 1e-6));
    assert(only.class_id == 32);
    // stride=8，grid 中心 (84,164)，uniform DFL 距离 7.5*8=60
    assert(near(only.box.x, 24.0f, 1e-3));
    assert(near(only.box.y, 104.0f, 1e-3));
    assert(near(only.box.width, 120.0f, 1e-3));
    assert(near(only.box.height, 120.0f, 1e-3));

    const std::vector<int> kept = class_aware_nms(candidates, 0.45f, 100);
    assert(kept.size() == 1);
    const auto contours = build_contours(only, views[12], 1.0f, 0, 0, 640, 640);
    assert(!contours.empty());
    for (const auto& contour : contours) {
        for (const auto& point : contour) {
            assert(point.x >= 24 - 4 && point.x <= 144 + 4);
            assert(point.y >= 104 - 4 && point.y <= 224 + 4);
        }
    }
    std::cout << "decode_spacemit13 test passed\n";
}

}  // namespace

int main() {
    test_decode_standard2();
    test_decode_spacemit13();
    std::cout << "yolov8 postprocess tests passed\n";
    return 0;
}
