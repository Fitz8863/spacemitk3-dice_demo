// NPU(SpaceMIT EP) vs 纯 CPU 同输入输出对比诊断工具。
// 用途：量化模型在 NPU 上分类分支疑似输出异常（空场景误检 person）时，
// 用同一份预处理输入分别跑两种 session，逐通道对比输出差异。
// 用法：./build/ep_probe [模型路径]，不碰摄像头。
#include "opencl_preprocess.h"

#include <onnxruntime_cxx_api.h>
#include <spacemit_ort_env.h>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct RunResult {
    std::vector<float> data;
    std::vector<int64_t> shape;
};

RunResult run_session(Ort::Session& session, const float* input, size_t input_count) {
    Ort::MemoryInfo memory_info("Cpu", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
    const std::array<int64_t, 4> input_shape{1, 3, 640, 640};
    Ort::Value tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input),
                                                        input_count, input_shape.data(), 4);
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_owned = session.GetInputNameAllocated(0, allocator);
    auto output_name_owned = session.GetOutputNameAllocated(0, allocator);
    const char* input_name = input_name_owned.get();
    const char* output_name = output_name_owned.get();
    auto outputs = session.Run(Ort::RunOptions{nullptr}, &input_name, &tensor, 1, &output_name, 1);
    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    RunResult result;
    result.shape = info.GetShape();
    const size_t count = info.GetElementCount();
    result.data.assign(outputs[0].GetTensorData<float>(), outputs[0].GetTensorData<float>() + count);
    return result;
}

// 通道主序 [1, C, A]：打印每通道 max，返回通道 -> max 表。
std::vector<float> channel_maxima(const RunResult& r) {
    const int channels = static_cast<int>(r.shape[1]);
    const int anchors = static_cast<int>(r.shape[2]);
    std::vector<float> maxima(static_cast<size_t>(channels), 0.0f);
    for (int c = 0; c < channels; ++c) {
        const float* row = r.data.data() + static_cast<size_t>(c) * anchors;
        maxima[static_cast<size_t>(c)] = *std::max_element(row, row + anchors);
    }
    return maxima;
}

void print_stats(const char* tag, const RunResult& r) {
    const int channels = static_cast<int>(r.shape[1]);
    const int anchors = static_cast<int>(r.shape[2]);
    const float* data = r.data.data();
    std::cout << "[" << tag << "] shape [1," << channels << "," << anchors << "]\n";
    const auto row_stats = [&](const char* name, int c) {
        const float* row = data + static_cast<size_t>(c) * anchors;
        float mn = row[0], mx = row[0];
        double sum = 0.0;
        int over25 = 0;
        for (int a = 0; a < anchors; ++a) {
            mn = std::min(mn, row[a]);
            mx = std::max(mx, row[a]);
            sum += row[a];
            if (row[a] >= 0.25f) ++over25;
        }
        std::cout << "  " << name << " ch" << c << ": min=" << mn << " max=" << mx
                  << " mean=" << sum / anchors << " (>=0.25: " << over25 << ")\n";
    };
    row_stats("box cx", 0);
    row_stats("box cy", 1);
    row_stats("cls  ", 4);
    row_stats("kpt0 x", 5);
    row_stats("kpt0 conf", 7);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string model_path = argc > 1 ? argv[1] : "models/yolov8n-pose.q.onnx";

    // 与 --self-test 相同的合成输入：720p 全灰 NV12，不含任何人形纹理。
    OpenClPreprocessor preprocessor;
    if (!preprocessor.init()) return 4;
    cv::Mat synthetic(720 * 3 / 2, 1280, CV_8UC1, cv::Scalar(128));
    const auto prepared = preprocessor.preprocess(synthetic);
    const size_t input_count = prepared.data->size();
    std::vector<float> input_copy(prepared.data->data(), prepared.data->data() + input_count);
    std::cout << "input prepared: " << input_count << " floats, scale=" << prepared.scale
              << " pad=(" << prepared.pad_x << "," << prepared.pad_y << ")\n";

    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ep-probe"};

    // NPU session（与主程序相同的注册方式）
    Ort::SessionOptions npu_options;
    npu_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    npu_options.SetIntraOpNumThreads(2);
    npu_options.SetInterOpNumThreads(1);
    npu_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    std::unordered_map<std::string, std::string> ep_options;
    ep_options["SPACEMIT_EP_INTRA_THREAD_NUM"] = "2";
    ep_options["SPACEMIT_EP_INTER_THREAD_NUM"] = "1";
    ep_options["SPACEMIT_EP_INTRA_THREAD_AFFINITY"] = "12;13";
    Ort::SessionOptionsSpaceMITEnvInit(npu_options, ep_options);
    Ort::Session npu_session(env, model_path.c_str(), npu_options);

    // 纯 CPU session（不注册 EP）
    Ort::SessionOptions cpu_options;
    cpu_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    cpu_options.SetIntraOpNumThreads(4);
    cpu_options.SetInterOpNumThreads(1);
    cpu_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session cpu_session(env, model_path.c_str(), cpu_options);

    const RunResult npu = run_session(npu_session, input_copy.data(), input_count);
    const RunResult cpu = run_session(cpu_session, input_copy.data(), input_count);
    if (npu.data.size() != cpu.data.size() || npu.shape != cpu.shape) {
        std::cerr << "shape mismatch between NPU and CPU runs\n";
        return 1;
    }

    print_stats("NPU", npu);
    print_stats("CPU", cpu);

    // 全张量差异
    double diff_sum = 0.0;
    float diff_max = 0.0f;
    size_t diff_max_index = 0;
    for (size_t i = 0; i < npu.data.size(); ++i) {
        const float d = std::fabs(npu.data[i] - cpu.data[i]);
        diff_sum += d;
        if (d > diff_max) { diff_max = d; diff_max_index = i; }
    }
    const int anchors = static_cast<int>(npu.shape[2]);
    std::cout << "diff: max=" << diff_max << " (channel " << diff_max_index / anchors
              << ", anchor " << diff_max_index % anchors << ") mean="
              << diff_sum / static_cast<double>(npu.data.size()) << "\n";

    // 逐通道 max 差异 top10：坏分支会在个别通道上拉开数量级
    const auto npu_max = channel_maxima(npu);
    const auto cpu_max = channel_maxima(cpu);
    const int channels = static_cast<int>(npu.shape[1]);
    std::vector<std::pair<float, int>> diffs;
    for (int c = 0; c < channels; ++c) {
        diffs.emplace_back(std::fabs(npu_max[static_cast<size_t>(c)] - cpu_max[static_cast<size_t>(c)]), c);
    }
    std::sort(diffs.begin(), diffs.end(), std::greater<>());
    std::cout << "per-channel |max(NPU)-max(CPU)| top10:\n";
    for (int i = 0; i < 10 && i < static_cast<int>(diffs.size()); ++i) {
        const int c = diffs[static_cast<size_t>(i)].second;
        std::cout << "  ch" << c << ": NPU " << npu_max[static_cast<size_t>(c)]
                  << " vs CPU " << cpu_max[static_cast<size_t>(c)]
                  << "  |diff|=" << diffs[static_cast<size_t>(i)].first << "\n";
    }
    return 0;
}
