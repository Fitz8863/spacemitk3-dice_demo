#include "yolov8_pose_detector.h"

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

void print_shape(const std::string& name, const std::vector<int64_t>& shape) {
    std::cout << "  " << name << " [";
    for (size_t i = 0; i < shape.size(); ++i) std::cout << (i ? "," : "") << shape[i];
    std::cout << "]\n";
}
}  // namespace

struct Yolov8PoseDetector::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "yolov8-pose-k3"};
    std::unique_ptr<Ort::Session> session;
    std::string input_name;
    std::string output_name;
    std::vector<int64_t> input_shape;
    std::vector<std::string> class_names;
};

Yolov8PoseDetector::Yolov8PoseDetector() = default;
Yolov8PoseDetector::~Yolov8PoseDetector() = default;

bool Yolov8PoseDetector::init(const std::string& model_path, int intra_threads,
                              const std::string& ep_affinity,
                              const std::vector<std::string>& class_names,
                              bool ep_enabled) {
    try {
        impl_ = std::make_unique<Impl>();
        impl_->class_names = class_names;
        Ort::SessionOptions options;
        const int threads = std::max(1, intra_threads);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetIntraOpNumThreads(threads);
        options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (ep_enabled) {
            std::unordered_map<std::string, std::string> ep_options;
            ep_options["SPACEMIT_EP_INTRA_THREAD_NUM"] = std::to_string(threads);
            ep_options["SPACEMIT_EP_INTER_THREAD_NUM"] = "1";
            if (!ep_affinity.empty()) {
                ep_options["SPACEMIT_EP_INTRA_THREAD_AFFINITY"] = ep_affinity;
            }
            Ort::SessionOptionsSpaceMITEnvInit(options, ep_options);
        }
        impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), options);

        Ort::AllocatorWithDefaultOptions allocator;
        impl_->input_name = impl_->session->GetInputNameAllocated(0, allocator).get();
        impl_->input_shape = impl_->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        const size_t output_count = impl_->session->GetOutputCount();
        if (output_count != 1) {
            throw std::runtime_error("YOLOv8-pose expects a single output tensor, got " +
                                     std::to_string(output_count));
        }
        impl_->output_name = impl_->session->GetOutputNameAllocated(0, allocator).get();
        const auto output_shape =
            impl_->session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "Model loaded with " << (ep_enabled ? "SpaceMIT EP" : "CPU EP (SpaceMIT EP disabled)")
                  << ": " << model_path << "\n";
        print_shape("input " + impl_->input_name, impl_->input_shape);
        print_shape("output0 " + impl_->output_name, output_shape);
        if (impl_->input_shape.size() != 4 || impl_->input_shape[1] != 3 ||
            impl_->input_shape[2] != kModelInputHeight || impl_->input_shape[3] != kModelInputWidth) {
            throw std::runtime_error("expected input shape [1,3,640,640]");
        }
        constexpr int expected_channels =
            kPoseBoxChannels + kPoseClassChannels + 3 * kPoseKeypointCount;
        if (output_shape.size() != 3 || output_shape[0] != 1 ||
            output_shape[1] != expected_channels || output_shape[2] <= 0) {
            throw std::runtime_error("expected single pose output [1," +
                                     std::to_string(expected_channels) +
                                     ",anchors] (4 box + 1 class + 17*3 keypoints)");
        }
        std::cout << "SpaceMIT EP affinity: " << (ep_affinity.empty() ? "runtime default" : ep_affinity) << "\n";
        std::cout << "Configured class names: " << impl_->class_names.size() << "\n";
        if (impl_->class_names.size() != 1) {
            std::cerr << "Warning: YOLOv8-pose models a single class, but class_names has "
                      << impl_->class_names.size() << " entries\n";
        }
        return true;
    } catch (const std::exception& exception) {
        std::cerr << "Model init failed: " << exception.what() << "\n";
        impl_.reset();
        return false;
    }
}

bool Yolov8PoseDetector::ready() const { return impl_ && impl_->session; }

std::vector<PoseDetection> Yolov8PoseDetector::infer(
    const float* input, std::size_t input_count, float conf_threshold,
    float iou_threshold, int max_detections, float scale, int pad_x,
    int pad_y, int image_width, int image_height) {
    if (!ready()) throw std::runtime_error("detector is not initialized");
    if (!input || input_count != 3ULL * kModelInputWidth * kModelInputHeight) {
        throw std::runtime_error("invalid YOLOv8-pose input tensor size");
    }
    Ort::MemoryInfo memory_info("Cpu", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
    const std::array<int64_t, 4> input_shape{1, 3, kModelInputHeight, kModelInputWidth};
    Ort::Value tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input), input_count,
                                                        input_shape.data(), input_shape.size());
    const char* input_name = impl_->input_name.c_str();
    const char* output_name = impl_->output_name.c_str();
    auto outputs = impl_->session->Run(Ort::RunOptions{nullptr}, &input_name, &tensor, 1,
                                       &output_name, 1);
    if (outputs.size() != 1 || !outputs[0].IsTensor()) {
        throw std::runtime_error("invalid YOLOv8-pose output");
    }
    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    const OutputView detection{outputs[0].GetTensorData<float>(), info.GetShape()};

    debug_dump_pose(detection);
    std::vector<PoseCandidate> candidates = decode_pose_output(
        detection, conf_threshold, scale, pad_x, pad_y, image_width, image_height);
    const std::vector<int> kept = class_aware_nms(candidates, iou_threshold, std::max(1, max_detections));

    std::vector<PoseDetection> detections;
    detections.reserve(kept.size());
    for (int index : kept) {
        const PoseCandidate& candidate = candidates[static_cast<size_t>(index)];
        PoseDetection pose;
        pose.x1 = candidate.box.x;
        pose.y1 = candidate.box.y;
        pose.x2 = candidate.box.x + candidate.box.width;
        pose.y2 = candidate.box.y + candidate.box.height;
        pose.confidence = candidate.score;
        pose.class_id = candidate.class_id;
        pose.label = label_for_class(candidate.class_id, impl_->class_names);
        pose.keypoints = candidate.keypoints;
        detections.push_back(std::move(pose));
    }
    return detections;
}
