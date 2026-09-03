#include "yolov8_seg_detector.h"

#include <onnxruntime_cxx_api.h>
#include <spacemit_ort_env.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace {
constexpr int kInputWidth = 640;
constexpr int kInputHeight = 640;
constexpr int kDflBins = 16;
constexpr int kMaskChannels = 32;
constexpr int kExpectedOutputs = 13;

std::string label_for_class(int id, const std::vector<std::string>& names) {
    if (id >= 0 && id < static_cast<int>(names.size())) {
        return names[static_cast<size_t>(id)];
    }
    return "class_" + std::to_string(id);
}

struct Candidate {
    cv::Rect2f box;
    float score = 0.0f;
    int class_id = -1;
    std::array<float, kMaskChannels> coefficients{};
};

float box_iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float inter = (a & b).area();
    const float denom = a.area() + b.area() - inter;
    return denom > 0.0f ? inter / denom : 0.0f;
}

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

struct OutputView {
    const float* data = nullptr;
    std::vector<int64_t> shape;
};

cv::Point2f map_model_point(float x, float y, float scale, int pad_x, int pad_y,
                            int image_width, int image_height) {
    x = (x - static_cast<float>(pad_x)) * scale;
    y = (y - static_cast<float>(pad_y)) * scale;
    return {std::clamp(x, 0.0f, static_cast<float>(image_width)),
            std::clamp(y, 0.0f, static_cast<float>(image_height))};
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
    int x1 = std::clamp(static_cast<int>(std::floor(model_x1 * proto_w / kInputWidth)), 0, proto_w - 1);
    int y1 = std::clamp(static_cast<int>(std::floor(model_y1 * proto_h / kInputHeight)), 0, proto_h - 1);
    int x2 = std::clamp(static_cast<int>(std::ceil(model_x2 * proto_w / kInputWidth)), x1 + 1, proto_w);
    int y2 = std::clamp(static_cast<int>(std::ceil(model_y2 * proto_h / kInputHeight)), y1 + 1, proto_h);

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
            const float model_x = (point.x + x1 + 0.5f) * kInputWidth / static_cast<float>(proto_w);
            const float model_y = (point.y + y1 + 0.5f) * kInputHeight / static_cast<float>(proto_h);
            const cv::Point2f original = map_model_point(model_x, model_y, scale, pad_x, pad_y,
                                                          image_width, image_height);
            mapped.emplace_back(cvRound(original.x), cvRound(original.y));
        }
        contours.push_back(std::move(mapped));
    }
    return contours;
}

void print_shape(const std::string& name, const std::vector<int64_t>& shape) {
    std::cout << "  " << name << " [";
    for (size_t i = 0; i < shape.size(); ++i) std::cout << (i ? "," : "") << shape[i];
    std::cout << "]\n";
}
}  // namespace

struct Yolov8SegDetector::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "yolov8-seg-k3"};
    std::unique_ptr<Ort::Session> session;
    std::string input_name;
    std::vector<std::string> output_names;
    std::vector<int64_t> input_shape;
    std::vector<std::string> class_names;
};

Yolov8SegDetector::Yolov8SegDetector() = default;
Yolov8SegDetector::~Yolov8SegDetector() = default;

bool Yolov8SegDetector::init(const std::string& model_path, int intra_threads,
                             const std::string& ep_affinity,
                             const std::vector<std::string>& class_names) {
    try {
        impl_ = std::make_unique<Impl>();
        impl_->class_names = class_names;
        Ort::SessionOptions options;
        const int threads = std::max(1, intra_threads);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetIntraOpNumThreads(threads);
        options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        std::unordered_map<std::string, std::string> ep_options;
        ep_options["SPACEMIT_EP_INTRA_THREAD_NUM"] = std::to_string(threads);
        ep_options["SPACEMIT_EP_INTER_THREAD_NUM"] = "1";
        if (!ep_affinity.empty()) {
            ep_options["SPACEMIT_EP_INTRA_THREAD_AFFINITY"] = ep_affinity;
        }
        Ort::SessionOptionsSpaceMITEnvInit(options, ep_options);
        impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), options);

        Ort::AllocatorWithDefaultOptions allocator;
        impl_->input_name = impl_->session->GetInputNameAllocated(0, allocator).get();
        impl_->input_shape = impl_->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        const size_t output_count = impl_->session->GetOutputCount();
        if (output_count != kExpectedOutputs) {
            throw std::runtime_error("YOLOv8-seg expected 13 outputs, got " + std::to_string(output_count));
        }
        impl_->output_names.reserve(output_count);
        std::cout << "Model loaded with SpaceMIT EP: " << model_path << "\n";
        print_shape("input " + impl_->input_name, impl_->input_shape);
        for (size_t i = 0; i < output_count; ++i) {
            impl_->output_names.emplace_back(impl_->session->GetOutputNameAllocated(i, allocator).get());
            const auto shape = impl_->session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            print_shape("output" + std::to_string(i) + " " + impl_->output_names.back(), shape);
        }
        if (impl_->input_shape.size() != 4 || impl_->input_shape[1] != 3 ||
            impl_->input_shape[2] != kInputHeight || impl_->input_shape[3] != kInputWidth) {
            throw std::runtime_error("expected input shape [1,3,640,640]");
        }
        std::cout << "SpaceMIT EP affinity: " << (ep_affinity.empty() ? "runtime default" : ep_affinity) << "\n";
        std::cout << "Configured class names: " << impl_->class_names.size() << "\n";
        const auto class_count_from_output = [&]() -> int {
            for (int branch = 0; branch < 3; ++branch) {
                const auto shape = impl_->session->GetOutputTypeInfo(static_cast<size_t>(branch * 3 + 1))
                                        .GetTensorTypeAndShapeInfo().GetShape();
                if (shape.size() == 4 && shape[1] > 0) return static_cast<int>(shape[1]);
            }
            return 0;
        }();
        if (class_count_from_output > 0 &&
            static_cast<int>(impl_->class_names.size()) != class_count_from_output) {
            std::cerr << "Warning: class_names has " << impl_->class_names.size()
                      << " entries, but model outputs " << class_count_from_output
                      << " classes; unmatched IDs will use class_<id>\n";
        }
        return true;
    } catch (const std::exception& exception) {
        std::cerr << "Model init failed: " << exception.what() << "\n";
        impl_.reset();
        return false;
    }
}

bool Yolov8SegDetector::ready() const { return impl_ && impl_->session; }

std::vector<SegmentationDetection> Yolov8SegDetector::infer(
    const float* input, std::size_t input_count, float conf_threshold,
    float iou_threshold, int max_detections, float scale, int pad_x,
    int pad_y, int image_width, int image_height) {
    if (!ready()) throw std::runtime_error("detector is not initialized");
    if (!input || input_count != 3ULL * kInputWidth * kInputHeight) {
        throw std::runtime_error("invalid YOLOv8-seg input tensor size");
    }
    Ort::MemoryInfo memory_info("Cpu", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
    const std::array<int64_t, 4> input_shape{1, 3, kInputHeight, kInputWidth};
    Ort::Value tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input), input_count,
                                                         input_shape.data(), input_shape.size());
    std::vector<const char*> output_name_ptrs;
    output_name_ptrs.reserve(impl_->output_names.size());
    for (const auto& name : impl_->output_names) output_name_ptrs.push_back(name.c_str());
    const char* input_name = impl_->input_name.c_str();
    auto outputs = impl_->session->Run(Ort::RunOptions{nullptr}, &input_name, &tensor, 1,
                                       output_name_ptrs.data(), output_name_ptrs.size());
    if (outputs.size() != kExpectedOutputs) throw std::runtime_error("invalid YOLOv8-seg output count");

    std::vector<OutputView> views;
    views.reserve(outputs.size());
    for (auto& output : outputs) {
        if (!output.IsTensor()) throw std::runtime_error("YOLOv8-seg output is not a tensor");
        auto info = output.GetTensorTypeAndShapeInfo();
        views.push_back({output.GetTensorData<float>(), info.GetShape()});
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
        const float stride_x = static_cast<float>(kInputWidth) / grid_w;
        const float stride_y = static_cast<float>(kInputHeight) / grid_h;
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
            candidate.box = cv::Rect2f(cx - distance[0] * stride_x, cy - distance[1] * stride_y,
                                       (distance[0] + distance[2]) * stride_x,
                                       (distance[1] + distance[3]) * stride_y);
            const cv::Point2f top_left = map_model_point(candidate.box.x, candidate.box.y, scale, pad_x, pad_y,
                                                          image_width, image_height);
            const cv::Point2f bottom_right = map_model_point(candidate.box.x + candidate.box.width,
                                                              candidate.box.y + candidate.box.height, scale, pad_x,
                                                              pad_y, image_width, image_height);
            candidate.box = cv::Rect2f(top_left.x, top_left.y,
                                       bottom_right.x - top_left.x, bottom_right.y - top_left.y);
            candidate.score = best_score;
            candidate.class_id = best_class;
            if (candidate.box.width < 1.0f || candidate.box.height < 1.0f) continue;
            const float* coeff = views[static_cast<size_t>(9 + branch)].data;
            for (int channel = 0; channel < kMaskChannels; ++channel) {
                candidate.coefficients[static_cast<size_t>(channel)] = coeff[channel * anchors + anchor];
            }
            candidates.push_back(candidate);
        }
    }

    const std::vector<int> kept = class_aware_nms(candidates, iou_threshold, std::max(1, max_detections));
    std::vector<SegmentationDetection> detections;
    detections.reserve(kept.size());
    for (int index : kept) {
        const Candidate& candidate = candidates[static_cast<size_t>(index)];
        SegmentationDetection detection;
        detection.x1 = candidate.box.x;
        detection.y1 = candidate.box.y;
        detection.x2 = candidate.box.x + candidate.box.width;
        detection.y2 = candidate.box.y + candidate.box.height;
        detection.confidence = candidate.score;
        detection.class_id = candidate.class_id;
        detection.label = label_for_class(candidate.class_id, impl_->class_names);
        detection.mask_contours = build_contours(candidate, views[12], scale, pad_x, pad_y,
                                                  image_width, image_height);
        detections.push_back(std::move(detection));
    }
    return detections;
}
