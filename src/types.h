#pragma once

#include <opencv2/core.hpp>
#include <array>
#include <string>

// COCO 人体骨架关键点数（nose/eyes/ears/shoulders/elbows/wrists/hips/knees/ankles），
// 对应 pose 模型输出 [1, 5+17*3, 8400] 中的 17 组 (x,y,conf)。
constexpr int kPoseKeypointCount = 17;

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
    std::array<PoseKeypoint, kPoseKeypointCount> keypoints{};
};
