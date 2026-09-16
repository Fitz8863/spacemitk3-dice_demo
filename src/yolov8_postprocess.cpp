#include "yolov8_postprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

std::string label_for_class(int id, const std::vector<std::string>& names) {
    if (id >= 0 && id < static_cast<int>(names.size())) {
        return names[static_cast<size_t>(id)];
    }
    return "class_" + std::to_string(id);
}

namespace {

float box_iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float inter = (a & b).area();
    const float denom = a.area() + b.area() - inter;
    return denom > 0.0f ? inter / denom : 0.0f;
}

cv::Point2f map_model_point(float x, float y, float scale, int pad_x, int pad_y,
                            int image_width, int image_height) {
    x = (x - static_cast<float>(pad_x)) * scale;
    y = (y - static_cast<float>(pad_y)) * scale;
    return {std::clamp(x, 0.0f, static_cast<float>(image_width)),
            std::clamp(y, 0.0f, static_cast<float>(image_height))};
}

cv::Rect2f map_model_rect(const cv::Rect2f& box, float scale, int pad_x, int pad_y,
                          int image_width, int image_height) {
    const cv::Point2f top_left = map_model_point(box.x, box.y, scale, pad_x, pad_y,
                                                 image_width, image_height);
    const cv::Point2f bottom_right = map_model_point(box.x + box.width, box.y + box.height,
                                                     scale, pad_x, pad_y, image_width, image_height);
    return cv::Rect2f(top_left.x, top_left.y,
                      bottom_right.x - top_left.x, bottom_right.y - top_left.y);
}

}  // namespace

std::vector<int> class_aware_nms(const std::vector<PoseCandidate>& candidates,
                                 float iou_threshold, int max_detections) {
    std::vector<int> order(candidates.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(), [&candidates](int a, int b) {
        return candidates[static_cast<size_t>(a)].score > candidates[static_cast<size_t>(b)].score;
    });
    std::vector<int> kept;
    for (int index : order) {
        const PoseCandidate& candidate = candidates[static_cast<size_t>(index)];
        bool suppressed = false;
        for (int selected : kept) {
            const PoseCandidate& prior = candidates[static_cast<size_t>(selected)];
            if (candidate.class_id == prior.class_id && box_iou(candidate.box, prior.box) > iou_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(index);
            if (static_cast<int>(kept.size()) >= max_detections) break;
        }
    }
    return kept;
}

int keypoint_count_from_channels(int64_t channels) {
    const int64_t head = kPoseBoxChannels + kPoseClassChannels;
    if (channels < head + 3 || (channels - head) % 3 != 0 ||
        (channels - head) / 3 > kMaxKeypoints) {
        throw std::runtime_error("unexpected YOLOv8-pose channel count: " + std::to_string(channels));
    }
    return static_cast<int>((channels - head) / 3);
}

std::vector<PoseCandidate> decode_pose_output(
    const OutputView& detection, int keypoint_count, float conf_threshold,
    float scale, int pad_x, int pad_y, int image_width, int image_height) {
    if (keypoint_count < 1 || keypoint_count > kMaxKeypoints ||
        detection.shape.size() != 3 || detection.shape[0] != 1 ||
        detection.shape[1] != kPoseBoxChannels + kPoseClassChannels + 3 * keypoint_count ||
        detection.shape[2] <= 0) {
        throw std::runtime_error("unexpected YOLOv8-pose detection output shape");
    }
    const int anchors = static_cast<int>(detection.shape[2]);
    const float* data = detection.data;
    const int kpt_base = kPoseBoxChannels + kPoseClassChannels;
    std::vector<PoseCandidate> candidates;
    for (int anchor = 0; anchor < anchors; ++anchor) {
        const float score = data[static_cast<size_t>(kPoseBoxChannels) * anchors + anchor];
        if (!std::isfinite(score) || score < conf_threshold) continue;
        const float cx = data[static_cast<size_t>(0) * anchors + anchor];
        const float cy = data[static_cast<size_t>(1) * anchors + anchor];
        const float w = data[static_cast<size_t>(2) * anchors + anchor];
        const float h = data[static_cast<size_t>(3) * anchors + anchor];
        PoseCandidate candidate;
        const cv::Rect2f model_box(cx - w * 0.5f, cy - h * 0.5f, w, h);
        candidate.box = map_model_rect(model_box, scale, pad_x, pad_y, image_width, image_height);
        candidate.score = score;
        candidate.class_id = 0;
        candidate.keypoint_count = keypoint_count;
        if (candidate.box.width < 1.0f || candidate.box.height < 1.0f) continue;
        for (int kpt = 0; kpt < keypoint_count; ++kpt) {
            const size_t kpt_index = static_cast<size_t>(kpt_base + kpt * 3) * anchors + anchor;
            PoseKeypoint& point = candidate.keypoints[static_cast<size_t>(kpt)];
            point.confidence = data[kpt_index + 2 * anchors];
            const cv::Point2f mapped = map_model_point(data[kpt_index], data[kpt_index + anchors],
                                                       scale, pad_x, pad_y, image_width, image_height);
            point.x = mapped.x;
            point.y = mapped.y;
        }
        candidates.push_back(std::move(candidate));
    }
    return candidates;
}

void debug_dump_pose(const OutputView& detection, int keypoint_count) {
    static const bool enabled = std::getenv("YOLO_POSE_DEBUG") != nullptr;
    static bool dumped = false;
    if (!enabled || dumped) return;
    dumped = true;
    if (detection.shape.size() != 3 || detection.shape[0] != 1 || keypoint_count < 1) return;
    const int channels = static_cast<int>(detection.shape[1]);
    const int anchors = static_cast<int>(detection.shape[2]);
    if (channels < kPoseBoxChannels + kPoseClassChannels || anchors <= 0) return;
    const int kpt_base = kPoseBoxChannels + kPoseClassChannels;
    const float* data = detection.data;
    const auto stats = [anchors](const float* first_row, int rows) {
        float mn = std::numeric_limits<float>::infinity();
        float mx = -std::numeric_limits<float>::infinity();
        double sum = 0.0;
        size_t count = static_cast<size_t>(rows) * anchors;
        for (int row = 0; row < rows; ++row) {
            const float* r = first_row + static_cast<size_t>(row) * anchors;
            for (int a = 0; a < anchors; ++a) {
                mn = std::min(mn, r[a]);
                mx = std::max(mx, r[a]);
                sum += r[a];
            }
        }
        return std::array<float, 3>{mn, mx, static_cast<float>(sum / static_cast<double>(count))};
    };
    const auto box = stats(data, kPoseBoxChannels);
    const auto score = stats(data + static_cast<size_t>(kPoseBoxChannels) * anchors, kPoseClassChannels);
    const auto kpt_xy = stats(data + static_cast<size_t>(kpt_base) * anchors, keypoint_count * 2);
    const auto kpt_conf = stats(data + static_cast<size_t>(kpt_base + keypoint_count * 2) * anchors,
                                keypoint_count);
    int over_conf = 0;
    const float* score_row = data + static_cast<size_t>(kPoseBoxChannels) * anchors;
    for (int a = 0; a < anchors; ++a) {
        if (score_row[a] >= 0.25f) ++over_conf;
    }
    std::cout << "[debug] pose anchors=" << anchors << " channels=" << channels
              << " kpt=" << keypoint_count << "\n"
              << "[debug] box rows    min=" << box[0] << " max=" << box[1] << " mean=" << box[2] << "\n"
              << "[debug] class row   min=" << score[0] << " max=" << score[1] << " mean=" << score[2]
              << " (values>=0.25: " << over_conf << ")\n"
              << "[debug] kpt xy      min=" << kpt_xy[0] << " max=" << kpt_xy[1] << " mean=" << kpt_xy[2] << "\n"
              << "[debug] kpt conf    min=" << kpt_conf[0] << " max=" << kpt_conf[1] << " mean=" << kpt_conf[2] << "\n";
}
