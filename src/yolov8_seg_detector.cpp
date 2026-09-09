#include "yolov8_seg_detector.h"

#include <onnxruntime_cxx_api.h>
#include <spacemit_ort_env.h>

#include "yolov8_postprocess.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

enum class OutputLayout { kSpaceMIT13, kStandard2 };

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
    OutputLayout layout = OutputLayout::kSpaceMIT13;
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
        if (output_count == static_cast<size_t>(kSpaceMITOutputCount)) {
            impl_->layout = OutputLayout::kSpaceMIT13;
            std::cout << "Output layout: SpaceMIT 13-output\n";
        } else if (output_count == static_cast<size_t>(kStandardOutputCount)) {
            impl_->layout = OutputLayout::kStandard2;
            std::cout << "Output layout: standard Ultralytics 2-output\n";
        } else {
            throw std::runtime_error(
                "YOLOv8-seg expects 13 outputs (SpaceMIT layout) or 2 outputs (standard layout), got " +
                std::to_string(output_count));
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
            impl_->input_shape[2] != kModelInputHeight || impl_->input_shape[3] != kModelInputWidth) {
            throw std::runtime_error("expected input shape [1,3,640,640]");
        }
        std::cout << "SpaceMIT EP affinity: " << (ep_affinity.empty() ? "runtime default" : ep_affinity) << "\n";
        std::cout << "Configured class names: " << impl_->class_names.size() << "\n";
        int class_count_from_output = 0;
        if (impl_->layout == OutputLayout::kSpaceMIT13) {
            for (int branch = 0; branch < 3; ++branch) {
                const auto shape = impl_->session->GetOutputTypeInfo(static_cast<size_t>(branch * 3 + 1))
                                        .GetTensorTypeAndShapeInfo().GetShape();
                if (shape.size() == 4 && shape[1] > 0) {
                    class_count_from_output = static_cast<int>(shape[1]);
                    break;
                }
            }
        } else {
            const auto shape = impl_->session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() == 3 && shape[1] > 4 + kMaskChannels) {
                class_count_from_output = static_cast<int>(shape[1]) - 4 - kMaskChannels;
            }
        }
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
    if (!input || input_count != 3ULL * kModelInputWidth * kModelInputHeight) {
        throw std::runtime_error("invalid YOLOv8-seg input tensor size");
    }
    Ort::MemoryInfo memory_info("Cpu", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
    const std::array<int64_t, 4> input_shape{1, 3, kModelInputHeight, kModelInputWidth};
    Ort::Value tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input), input_count,
                                                         input_shape.data(), input_shape.size());
    std::vector<const char*> output_name_ptrs;
    output_name_ptrs.reserve(impl_->output_names.size());
    for (const auto& name : impl_->output_names) output_name_ptrs.push_back(name.c_str());
    const char* input_name = impl_->input_name.c_str();
    auto outputs = impl_->session->Run(Ort::RunOptions{nullptr}, &input_name, &tensor, 1,
                                       output_name_ptrs.data(), output_name_ptrs.size());
    const size_t expected_outputs = impl_->layout == OutputLayout::kSpaceMIT13
                                        ? static_cast<size_t>(kSpaceMITOutputCount)
                                        : static_cast<size_t>(kStandardOutputCount);
    if (outputs.size() != expected_outputs) throw std::runtime_error("invalid YOLOv8-seg output count");

    std::vector<OutputView> views;
    views.reserve(outputs.size());
    for (auto& output : outputs) {
        if (!output.IsTensor()) throw std::runtime_error("YOLOv8-seg output is not a tensor");
        auto info = output.GetTensorTypeAndShapeInfo();
        views.push_back({output.GetTensorData<float>(), info.GetShape()});
    }

    std::vector<Candidate> candidates;
    if (impl_->layout == OutputLayout::kSpaceMIT13) {
        candidates = decode_spacemit13(views, conf_threshold, scale, pad_x, pad_y,
                                       image_width, image_height);
    } else {
        debug_dump_standard2(views[0], views[1]);
        candidates = decode_standard2(views[0], conf_threshold, scale, pad_x, pad_y,
                                      image_width, image_height);
    }
    const OutputView& prototype = impl_->layout == OutputLayout::kSpaceMIT13 ? views[12] : views[1];

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
        detection.mask_contours = build_contours(candidate, prototype, scale, pad_x, pad_y,
                                                  image_width, image_height);
        detections.push_back(std::move(detection));
    }
    return detections;
}
