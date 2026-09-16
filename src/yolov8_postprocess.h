#pragma once

#include "types.h"

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// YOLOv8-pose 后处理共享模块：模型输出为单张量 [1, 4+1+3*17, anchors]，
// box 与关键点坐标已在图内解码为 640 letterbox 像素坐标，类别分与关键点
// 置信度已在图内 sigmoid（概率域）。这里只做阈值筛选、NMS 与 letterbox 反映射。
constexpr int kModelInputWidth = 640;
constexpr int kModelInputHeight = 640;
constexpr int kPoseBoxChannels = 4;
constexpr int kPoseClassChannels = 1;

struct OutputView {
    const float* data = nullptr;
    std::vector<int64_t> shape;
};

struct PoseCandidate {
    cv::Rect2f box;  // 原图坐标系
    float score = 0.0f;
    int class_id = -1;
    std::array<PoseKeypoint, kPoseKeypointCount> keypoints{};  // 原图坐标系
};

std::string label_for_class(int id, const std::vector<std::string>& names);

std::vector<int> class_aware_nms(const std::vector<PoseCandidate>& candidates,
                                 float iou_threshold, int max_detections);

// 解码 pose 检测输出并映射回原图坐标系，cls 置信度过滤在 NMS 之前；
// 关键点置信度只透传不过滤，绘制时由 kpt_conf_threshold 决定。
std::vector<PoseCandidate> decode_pose_output(
    const OutputView& detection, float conf_threshold, float scale, int pad_x,
    int pad_y, int image_width, int image_height);

// YOLO_POSE_DEBUG=1 时在首次推理后打印输出各段数值统计，
// 用于确认置信度是 logit 还是已激活概率（量化导出工具决定）。
void debug_dump_pose(const OutputView& detection);
