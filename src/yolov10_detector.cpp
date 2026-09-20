#include "yolov10_detector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <spacemit_ort_env.h>
#include <stdexcept>
#include <unordered_map>

namespace {
constexpr int kModelWidth = 640;
constexpr int kModelHeight = 640;
// Ultralytics YOLOv10 end-to-end export: [1, 300, 6] rows of
// (x1, y1, x2, y2, conf, class_id) in 640x640 letterbox pixel coordinates.
// conf is already post-sigmoid and rows are sorted by conf descending, so no
// NMS is needed here.
constexpr int kRowStride = 6;
constexpr int kMaxRows = 300;
}  // namespace

struct Yolov10Detector::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "yolov10-k3"};
    std::unique_ptr<Ort::Session> session;
};

Yolov10Detector::Yolov10Detector() = default;
Yolov10Detector::~Yolov10Detector() = default;

namespace {
// This PPQ-quantized graph mixes INT8 and UINT8 zero points, which trips ORT's
// extended graph optimizations (QuantizeLinear type inference). BASIC is the
// highest level verified to load it; DISABLE_ALL is kept as a last resort.
Ort::SessionOptions make_session_options(int intra_threads, bool ep_enabled,
                                         const std::string& ep_affinity,
                                         GraphOptimizationLevel level) {
    Ort::SessionOptions options;
    const int ep_threads = std::max(1, intra_threads);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetIntraOpNumThreads(ep_threads);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(level);
    if (ep_enabled) {
        std::unordered_map<std::string, std::string> ep_options;
        ep_options["SPACEMIT_EP_INTRA_THREAD_NUM"] = std::to_string(ep_threads);
        ep_options["SPACEMIT_EP_INTER_THREAD_NUM"] = "1";
        if (!ep_affinity.empty()) {
            ep_options["SPACEMIT_EP_INTRA_THREAD_AFFINITY"] = ep_affinity;
            std::cout << "SpaceMIT EP affinity: " << ep_affinity << "\n";
        }
        Ort::SessionOptionsSpaceMITEnvInit(options, ep_options);
    }
    return options;
}
}  // namespace

bool Yolov10Detector::init(const std::string& model_path, int intra_threads,
                           const std::string& ep_affinity, bool ep_enabled) {
    try {
        impl_ = std::make_unique<Impl>();
        // ORT_ENABLE_BASIC is tried first; if even BASIC cannot load this
        // quantized graph, fall back to ORT_DISABLE_ALL once.
        for (const GraphOptimizationLevel level :
             {GraphOptimizationLevel::ORT_ENABLE_BASIC,
              GraphOptimizationLevel::ORT_DISABLE_ALL}) {
            try {
                Ort::SessionOptions options = make_session_options(
                    intra_threads, ep_enabled, ep_affinity, level);
                impl_->session = std::make_unique<Ort::Session>(
                    impl_->env, model_path.c_str(), options);
                std::cout << "Model loaded (graph optimization "
                          << (level == GraphOptimizationLevel::ORT_ENABLE_BASIC
                                  ? "BASIC"
                                  : "DISABLE_ALL")
                          << ", " << (ep_enabled ? "SpaceMIT EP" : "CPU") << "): "
                          << model_path << "\n";
                break;
            } catch (const std::exception& e) {
                if (level == GraphOptimizationLevel::ORT_ENABLE_BASIC) {
                    std::cerr << "Model load with BASIC optimizations failed: "
                              << e.what() << "\nRetrying with ORT_DISABLE_ALL...\n";
                    impl_->session.reset();
                } else {
                    throw;
                }
            }
        }
        if (!impl_->session) throw std::runtime_error("session creation failed");

        Ort::AllocatorWithDefaultOptions allocator;
        input_name_ = impl_->session->GetInputNameAllocated(0, allocator).get();
        output_name_ = impl_->session->GetOutputNameAllocated(0, allocator).get();
        input_shape_ = impl_->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        output_shape_ = impl_->session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "  input " << input_name_ << " [";
        for (std::size_t i = 0; i < input_shape_.size(); ++i)
            std::cout << (i ? "," : "") << input_shape_[i];
        std::cout << "] output " << output_name_ << " [";
        for (std::size_t i = 0; i < output_shape_.size(); ++i)
            std::cout << (i ? "," : "") << output_shape_[i];
        std::cout << "]\n";

        if (!input_shape_.empty() && input_shape_.size() != 4)
            throw std::runtime_error("unexpected input rank; expected rank 4");
        if (!input_shape_.empty()) {
            if (input_shape_[1] > 0 && input_shape_[1] != 3)
                throw std::runtime_error("unexpected input channels; expected 3");
            if (input_shape_[2] > 0 && input_shape_[2] != kModelHeight)
                throw std::runtime_error("unexpected input height; expected 640");
            if (input_shape_[3] > 0 && input_shape_[3] != kModelWidth)
                throw std::runtime_error("unexpected input width; expected 640");
        }
        // SpaceMIT EP may report dynamic dimensions as an empty shape before
        // Run(); only validate when the shape is known.
        if (!output_shape_.empty() &&
            (output_shape_.size() != 3 || output_shape_[2] != kRowStride)) {
            throw std::runtime_error("unexpected output shape; expected [1,300,6]");
        }
        session_opaque_ = impl_->session.get();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Model init failed: " << e.what() << "\n";
        impl_.reset();
        session_opaque_ = nullptr;
        return false;
    }
}

std::vector<Detection> Yolov10Detector::infer(const float* data, std::size_t count,
                                              float conf_threshold, float scale,
                                              int pad_x, int pad_y, int image_width,
                                              int image_height) {
    if (!impl_ || !impl_->session) throw std::runtime_error("detector is not initialized");
    if (!data || count != static_cast<std::size_t>(3 * kModelWidth * kModelHeight))
        throw std::runtime_error("unexpected input tensor size; expected 3*640*640 floats");

    Ort::MemoryInfo memory_info("Cpu", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
    const std::array<int64_t, 4> input_shape{1, 3, kModelHeight, kModelWidth};
    Ort::Value input = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(data), count,
                                                        input_shape.data(), input_shape.size());
    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    auto outputs = impl_->session->Run(Ort::RunOptions{nullptr}, input_names, &input, 1,
                                       output_names, 1);
    if (outputs.empty() || !outputs[0].IsTensor())
        throw std::runtime_error("YOLOv10 output is not a tensor");

    auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> shape = output_info.GetShape();
    if (shape.size() != 3 || shape[2] != kRowStride)
        throw std::runtime_error("YOLOv10 output must be [1,N,6]");
    const float* values = outputs[0].GetTensorData<float>();
    const std::size_t rows = static_cast<std::size_t>(shape[1]);

    // Small output telemetry to distinguish a truly empty frame from an
    // output-layout/preprocessing mismatch.
    static int debug_frames = 0;
    if (debug_frames < 3) {
        std::cerr << "YOLOv10 output debug: rows=" << rows
                  << " best_conf=" << std::fixed << std::setprecision(4)
                  << values[4] << " cls=" << values[5] << "\n";
        ++debug_frames;
    }

    std::vector<Detection> detections;
    detections.reserve(rows);
    for (std::size_t r = 0; r < rows; ++r) {
        const float* p = values + r * kRowStride;
        if (!(p[4] >= conf_threshold)) continue;
        const int class_id = static_cast<int>(p[5]);
        if (class_id < 0 || class_id > 255) continue;
        Detection d;
        d.x1 = (p[0] - static_cast<float>(pad_x)) * scale;
        d.y1 = (p[1] - static_cast<float>(pad_y)) * scale;
        d.x2 = (p[2] - static_cast<float>(pad_x)) * scale;
        d.y2 = (p[3] - static_cast<float>(pad_y)) * scale;
        d.x1 = std::clamp(d.x1, 0.0f, static_cast<float>(image_width));
        d.y1 = std::clamp(d.y1, 0.0f, static_cast<float>(image_height));
        d.x2 = std::clamp(d.x2, 0.0f, static_cast<float>(image_width));
        d.y2 = std::clamp(d.y2, 0.0f, static_cast<float>(image_height));
        d.confidence = p[4];
        d.class_id = class_id;
        if (d.x2 > d.x1 && d.y2 > d.y1) detections.push_back(d);
    }
    return detections;
}
