#pragma once

#include <opencv2/core.hpp>
#include <array>
#include <string>

// 关键点数由模型输出通道推导（channels = 5 + 3*kpt）：
// 56 → COCO 人体 17 点；68 → 机械手 21 点（MediaPipe hand 布局）。
// 数组按上限分配，实际点数见 PoseDetection::keypoint_count。
constexpr int kMaxKeypoints = 21;

struct PoseKeypoint {
    float x = 0.0f;
    float y = 0.0f;
    float confidence = 0.0f;
};

struct PoseDetection {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float confidence = 0.0f;
    int class_id = -1;
    std::string label;
    int keypoint_count = 0;
    std::array<PoseKeypoint, kMaxKeypoints> keypoints{};
};
