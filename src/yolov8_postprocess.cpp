#include "yolov8_postprocess.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

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

std::vector<int> class_aware_nms(const std::vector<Candidate>& candidates,
                                 float iou_threshold, int max_detections) {
    std::vector<int> order(candidates.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(), [&candidates](int a, int b) {
        return candidates[static_cast<size_t>(a)].score > candidates[static_cast<size_t>(b)].score;
    });
    std::vector<int> kept;
    for (int index : order) {
        const Candidate& candidate = candidates[static_cast<size_t>(index)];
        bool suppressed = false;
        for (int selected : kept) {
            const Candidate& prior = candidates[static_cast<size_t>(selected)];
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

std::vector<std::vector<cv::Point>> build_contours(
    const Candidate& candidate, const OutputView& prototype, float scale, int pad_x, int pad_y,
    int image_width, int image_height) {
    if (prototype.shape.size() != 4 || prototype.shape[0] != 1 ||
        prototype.shape[1] != kMaskChannels || prototype.shape[2] <= 0 || prototype.shape[3] <= 0) {
        throw std::runtime_error("unexpected YOLOv8-seg prototype shape");
    }
    const int proto_h = static_cast<int>(prototype.shape[2]);
    const int proto_w = static_cast<int>(prototype.shape[3]);
    const float model_x1 = candidate.box.x / scale + pad_x;
    const float model_y1 = candidate.box.y / scale + pad_y;
    const float model_x2 = (candidate.box.x + candidate.box.width) / scale + pad_x;
    const float model_y2 = (candidate.box.y + candidate.box.height) / scale + pad_y;
    int x1 = std::clamp(static_cast<int>(std::floor(model_x1 * proto_w / kModelInputWidth)), 0, proto_w - 1);
    int y1 = std::clamp(static_cast<int>(std::floor(model_y1 * proto_h / kModelInputHeight)), 0, proto_h - 1);
    int x2 = std::clamp(static_cast<int>(std::ceil(model_x2 * proto_w / kModelInputWidth)), x1 + 1, proto_w);
    int y2 = std::clamp(static_cast<int>(std::ceil(model_y2 * proto_h / kModelInputHeight)), y1 + 1, proto_h);

    cv::Mat mask(y2 - y1, x2 - x1, CV_8U, cv::Scalar(0));
    const int plane = proto_h * proto_w;
    for (int y = y1; y < y2; ++y) {
        uint8_t* row = mask.ptr<uint8_t>(y - y1);
        for (int x = x1; x < x2; ++x) {
            float logit = 0.0f;
            const int pixel = y * proto_w + x;
            for (int channel = 0; channel < kMaskChannels; ++channel) {
                logit += candidate.coefficients[static_cast<size_t>(channel)] *
                         prototype.data[channel * plane + pixel];
            }
            row[x - x1] = logit > 0.0f ? 255 : 0;
        }
    }
    std::vector<std::vector<cv::Point>> raw;
    cv::findContours(mask, raw, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<std::vector<cv::Point>> contours;
    for (const auto& contour : raw) {
        if (contour.size() < 3 || cv::contourArea(contour) < 4.0) continue;
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 1.5, true);
        if (approx.size() < 3) continue;
        std::vector<cv::Point> mapped;
        mapped.reserve(approx.size());
        for (const auto& point : approx) {
            const float model_x = (point.x + x1 + 0.5f) * kModelInputWidth / static_cast<float>(proto_w);
            const float model_y = (point.y + y1 + 0.5f) * kModelInputHeight / static_cast<float>(proto_h);
            const cv::Point2f original = map_model_point(model_x, model_y, scale, pad_x, pad_y,
                                                          image_width, image_height);
            mapped.emplace_back(cvRound(original.x), cvRound(original.y));
        }
        contours.push_back(std::move(mapped));
    }
    return contours;
}

std::vector<Candidate> decode_spacemit13(
    const std::vector<OutputView>& views, float conf_threshold, float scale, int pad_x,
    int pad_y, int image_width, int image_height) {
    if (views.size() != static_cast<size_t>(kSpaceMITOutputCount)) {
        throw std::runtime_error("invalid YOLOv8-seg output count");
    }
    std::vector<Candidate> candidates;
    for (int branch = 0; branch < 3; ++branch) {
        const int base = branch * 3;
        const auto& box_shape = views[static_cast<size_t>(base)].shape;
        const auto& score_shape = views[static_cast<size_t>(base + 1)].shape;
        if (box_shape.size() != 4 || score_shape.size() != 4 || box_shape[0] != 1 ||
            box_shape[1] != 4 * kDflBins || score_shape[0] != 1 || score_shape[2] <= 0 || score_shape[3] <= 0) {
            throw std::runtime_error("unexpected YOLOv8-seg DFL output shape");
        }
        const int grid_h = static_cast<int>(box_shape[2]);
        const int grid_w = static_cast<int>(box_shape[3]);
        const int anchors = grid_h * grid_w;
        const int classes = static_cast<int>(score_shape[1]);
        const float* box_data = views[static_cast<size_t>(base)].data;
        const float* score_data = views[static_cast<size_t>(base + 1)].data;
        const float* score_sum = views[static_cast<size_t>(base + 2)].data;
        const float stride_x = static_cast<float>(kModelInputWidth) / grid_w;
        const float stride_y = static_cast<float>(kModelInputHeight) / grid_h;
        for (int anchor = 0; anchor < anchors; ++anchor) {
            if (score_sum[anchor] < conf_threshold) continue;
            float best_score = -1.0f;
            int best_class = -1;
            for (int class_id = 0; class_id < classes; ++class_id) {
                const float score = score_data[class_id * anchors + anchor];
                if (std::isfinite(score) && score > best_score) {
                    best_score = score;
                    best_class = class_id;
                }
            }
            if (best_class < 0 || best_score < conf_threshold) continue;
            std::array<float, 4> distance{};
            for (int coord = 0; coord < 4; ++coord) {
                const size_t base_offset = static_cast<size_t>(coord * kDflBins * anchors + anchor);
                float max_logit = box_data[base_offset];
                for (int bin = 1; bin < kDflBins; ++bin) {
                    max_logit = std::max(max_logit, box_data[base_offset + static_cast<size_t>(bin * anchors)]);
                }
                float exp_sum = 0.0f;
                float weighted = 0.0f;
                for (int bin = 0; bin < kDflBins; ++bin) {
                    const float value = std::exp(box_data[base_offset + static_cast<size_t>(bin * anchors)] - max_logit);
                    exp_sum += value;
                    weighted += value * bin;
                }
                distance[static_cast<size_t>(coord)] = weighted / exp_sum;
            }
            const int grid_y = anchor / grid_w;
            const int grid_x = anchor % grid_w;
            const float cx = (grid_x + 0.5f) * stride_x;
            const float cy = (grid_y + 0.5f) * stride_y;
            Candidate candidate;
            const cv::Rect2f model_box(cx - distance[0] * stride_x, cy - distance[1] * stride_y,
                                       (distance[0] + distance[2]) * stride_x,
                                       (distance[1] + distance[3]) * stride_y);
            candidate.box = map_model_rect(model_box, scale, pad_x, pad_y, image_width, image_height);
            candidate.score = best_score;
            candidate.class_id = best_class;
            if (candidate.box.width < 1.0f || candidate.box.height < 1.0f) continue;
            const float* coeff = views[static_cast<size_t>(9 + branch)].data;
            for (int channel = 0; channel < kMaskChannels; ++channel) {
                candidate.coefficients[static_cast<size_t>(channel)] = coeff[channel * anchors + anchor];
            }
            candidates.push_back(std::move(candidate));
        }
    }
    return candidates;
}

std::vector<Candidate> decode_standard2(
    const OutputView& detection, float conf_threshold, float scale, int pad_x,
    int pad_y, int image_width, int image_height) {
    if (detection.shape.size() != 3 || detection.shape[0] != 1 ||
        detection.shape[1] < 4 + 1 + kMaskChannels || detection.shape[2] <= 0) {
        throw std::runtime_error("unexpected YOLOv8-seg detection output shape");
    }
    const int channels = static_cast<int>(detection.shape[1]);
    const int anchors = static_cast<int>(detection.shape[2]);
    const int classes = channels - 4 - kMaskChannels;
    const float* data = detection.data;
    std::vector<Candidate> candidates;
    for (int anchor = 0; anchor < anchors; ++anchor) {
        float best_logit = -std::numeric_limits<float>::infinity();
        int best_class = -1;
        for (int class_id = 0; class_id < classes; ++class_id) {
            const float logit = data[static_cast<size_t>((4 + class_id) * anchors + anchor)];
            if (std::isfinite(logit) && logit > best_logit) {
                best_logit = logit;
                best_class = class_id;
            }
        }
        if (best_class < 0) continue;
        const float score = 1.0f / (1.0f + std::exp(-best_logit));
        if (score < conf_threshold) continue;
        const float cx = data[static_cast<size_t>(anchor)];
        const float cy = data[static_cast<size_t>(anchors + anchor)];
        const float w = data[static_cast<size_t>(2 * anchors + anchor)];
        const float h = data[static_cast<size_t>(3 * anchors + anchor)];
        Candidate candidate;
        const cv::Rect2f model_box(cx - w * 0.5f, cy - h * 0.5f, w, h);
        candidate.box = map_model_rect(model_box, scale, pad_x, pad_y, image_width, image_height);
        candidate.score = score;
        candidate.class_id = best_class;
        if (candidate.box.width < 1.0f || candidate.box.height < 1.0f) continue;
        const size_t coeff_base = static_cast<size_t>(4 + classes) * anchors + anchor;
        for (int channel = 0; channel < kMaskChannels; ++channel) {
            candidate.coefficients[static_cast<size_t>(channel)] =
                data[coeff_base + static_cast<size_t>(channel) * anchors];
        }
        candidates.push_back(std::move(candidate));
    }
    return candidates;
}
