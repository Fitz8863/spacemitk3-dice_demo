#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// YOLOv8-seg 后处理共享模块：Candidate/NMS/mask 组装服务两种模型输出布局，
// 各 decode_ 函数产出的 Candidate.box 已映射回原图坐标系。
constexpr int kModelInputWidth = 640;
constexpr int kModelInputHeight = 640;
constexpr int kDflBins = 16;
constexpr int kMaskChannels = 32;
constexpr int kSpaceMITOutputCount = 13;
constexpr int kStandardOutputCount = 2;

struct OutputView {
    const float* data = nullptr;
    std::vector<int64_t> shape;
};

struct Candidate {
    cv::Rect2f box;
    float score = 0.0f;
    int class_id = -1;
    std::array<float, kMaskChannels> coefficients{};
};

std::string label_for_class(int id, const std::vector<std::string>& names);

std::vector<int> class_aware_nms(const std::vector<Candidate>& candidates,
                                 float iou_threshold, int max_detections);

std::vector<std::vector<cv::Point>> build_contours(
    const Candidate& candidate, const OutputView& prototype, float scale, int pad_x, int pad_y,
    int image_width, int image_height);

// SpaceMIT 13 输出布局：3 x {DFL box, class score, score-sum} + 3 x mask coeff + proto，
// 类别分与 score-sum 是图内已归一化的概率，直接与 conf 比较。
std::vector<Candidate> decode_spacemit13(
    const std::vector<OutputView>& views, float conf_threshold, float scale, int pad_x,
    int pad_y, int image_width, int image_height);

// 官方标准 2 输出布局的检测输出 [1, 4+classes+32, anchors]：
// 通道 0-3 为 cx,cy,w,h（640 letterbox 像素坐标），4..3+classes 为类别分，其后 32 通道为
// mask 系数。与 SpaceMIT 13 输出导出约定一致，类别分是图内已激活的概率（不再 sigmoid）；
// 官方原始 logit 导出不适用此路径。proto 由调用方传给 build_contours。
std::vector<Candidate> decode_standard2(
    const OutputView& detection, float conf_threshold, float scale, int pad_x,
    int pad_y, int image_width, int image_height);

// YOLO_SEG_DEBUG=1 时在首次标准 2 输出推理后打印各段数值统计，
// 用于确认类别分是 logit 还是已激活概率（量化导出工具决定）。
void debug_dump_standard2(const OutputView& detection, const OutputView& prototype);
