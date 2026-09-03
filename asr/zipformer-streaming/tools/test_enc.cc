// 最小 encoder 测试：载入模型，构造零特征 + 零状态，跑一次 chunk
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include <onnxruntime_cxx_api.h>
#ifdef USE_SPACEMIT_EP
#include <spacemit_ort_env.h>
#endif

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_enc <encoder.onnx> [intra_threads] [--ep] "
                     "[--noconv]\n";
        return 1;
    }
    Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "test_enc");
    Ort::SessionOptions so;
    int threads = 1;
    bool use_ep = false, noconv = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--ep") use_ep = true;
        else if (a == "--noconv") noconv = true;
        else threads = std::stoi(a);
    }
#ifdef USE_SPACEMIT_EP
    if (use_ep) {
        std::unordered_map<std::string, std::string> ep_options = {
                {"SPACEMIT_EP_INTRA_THREAD_NUM", "1"},
                {"SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD", "1"},
                {"SPACEMIT_EP_PERFER_CORE_ARCH", "0x5064"}};
        if (noconv) ep_options["SPACEMIT_EP_DISABLE_OP_TYPE_FILTER"] = "Conv";
        auto st = Ort::SessionOptionsSpaceMITEnvInit(so, ep_options);
        if (!st.IsOK()) {
            std::cerr << "EP init failed: " << st.GetErrorMessage() << "\n";
            return 1;
        }
        std::cerr << "SpaceMIT EP enabled (noconv=" << noconv << ")\n";
    } else
#endif
    {
        so.SetIntraOpNumThreads(threads);
        so.SetInterOpNumThreads(1);
    }
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 复刻 stream_asr 的会话创建顺序：先 decoder/joiner（全程存活），再 encoder
    std::vector<std::unique_ptr<Ort::Session>> keepalive;
    if (const char* dir = std::getenv("DEC_FIRST")) {
        std::string d = dir;
        keepalive.push_back(std::make_unique<Ort::Session>(
                env, (d + "/decoder-epoch-99-avg-1.onnx").c_str(), so));
        keepalive.push_back(std::make_unique<Ort::Session>(
                env, (d + "/joiner-epoch-99-avg-1.onnx").c_str(), so));
        std::cerr << "dec/joiner sessions created first (kept alive)\n";
    }

    std::cerr << "loading session...\n";
    Ort::Session enc(env, argv[1], so);
    std::cerr << "session loaded, inputs=" << enc.GetInputCount()
              << " outputs=" << enc.GetOutputCount() << "\n";

    Ort::AllocatorWithDefaultOptions alloc;
    std::vector<std::string> in_names, out_names;
    for (size_t i = 0; i < enc.GetInputCount(); ++i)
        in_names.emplace_back(enc.GetInputNameAllocated(i, alloc).get());
    for (size_t i = 0; i < enc.GetOutputCount(); ++i)
        out_names.emplace_back(enc.GetOutputNameAllocated(i, alloc).get());
    std::vector<const char*> in_p, out_p;
    for (auto& s : in_names) in_p.push_back(s.c_str());
    for (auto& s : out_names) out_p.push_back(s.c_str());

    Ort::MemoryInfo mem =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> inputs;
    int T = 39, D = 80;
    std::vector<float> x(T * D, 0.0f);
    if (std::getenv("NOISY")) {  // 非零输入，模拟真实 fbank 特征
        for (int i = 0; i < T * D; ++i)
            x[i] = 0.1f * std::sin(0.37f * i) + 0.05f * std::cos(0.11f * i);
    }
    if (std::getenv("FBANK_LIKE")) {  // 对数梅尔量级 [-16, 0]
        for (int i = 0; i < T * D; ++i)
            x[i] = -8.0f + 8.0f * std::sin(0.21f * i);
    }
    if (std::getenv("MIXED")) {
        // 复刻 stream_asr 第一个 chunk: 7 帧零填充 + 9 帧真实 + 23 帧零填充
        for (int i = 0; i < T * D; ++i) x[i] = 0.0f;
        for (int i = 7 * D; i < 16 * D; ++i)
            x[i] = -8.0f + 8.0f * std::sin(0.21f * i);
    }
    std::array<int64_t, 3> xs{1, T, D};
    inputs.push_back(
            Ort::Value::CreateTensor<float>(mem, x.data(), x.size(), xs.data(), 3));

    for (size_t i = 1; i < in_names.size(); ++i) {
        auto info = enc.GetInputTypeInfo(i);
        auto ti = info.GetTensorTypeAndShapeInfo();
        auto shape = ti.GetShape();
        for (auto& d : shape)
            if (d < 0) d = 1;
        size_t numel = 1;
        for (auto d : shape) numel *= static_cast<size_t>(d);
        if (ti.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            auto t = Ort::Value::CreateTensor<int64_t>(alloc, shape.data(),
                                                       shape.size());
            std::fill_n(t.GetTensorMutableData<int64_t>(), numel, 0);
            inputs.push_back(std::move(t));
        } else {
            auto t = Ort::Value::CreateTensor<float>(alloc, shape.data(),
                                                     shape.size());
            std::fill_n(t.GetTensorMutableData<float>(), numel, 0.0f);
            inputs.push_back(std::move(t));
        }
    }
    std::cerr << "inputs built: " << inputs.size() << "\n";

    // 多 chunk 压测：状态在 chunk 间传递（模拟 stream_asr 真实用法）
    int n_rounds = 1;
    if (const char* r = std::getenv("ROUNDS")) n_rounds = std::stoi(r);

    try {
        double total = 0;
        std::vector<Ort::Value> cur_states;
        for (int round = 0; round < n_rounds; ++round) {
            auto t0 = std::chrono::steady_clock::now();
            std::vector<Ort::Value> run_inputs;
            run_inputs.push_back(Ort::Value::CreateTensor<float>(
                    mem, x.data(), x.size(), xs.data(), 3));
            for (size_t i = 1; i < inputs.size(); ++i) {
                // 深拷贝上一轮的输出状态（用 allocator 分配，避免悬垂）
                auto& src = cur_states.empty() ? inputs[i] : cur_states[i - 1];
                auto ti = src.GetTensorTypeAndShapeInfo();
                auto shape = ti.GetShape();
                size_t numel = 1;
                for (auto d : shape) numel *= static_cast<size_t>(d);
                if (src.GetTensorTypeAndShapeInfo().GetElementType() ==
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                    auto t = Ort::Value::CreateTensor<int64_t>(alloc, shape.data(),
                                                               shape.size());
                    std::memcpy(t.GetTensorMutableData<int64_t>(),
                                src.GetTensorData<int64_t>(), numel * 8);
                    run_inputs.push_back(std::move(t));
                } else {
                    auto t = Ort::Value::CreateTensor<float>(alloc, shape.data(),
                                                             shape.size());
                    std::memcpy(t.GetTensorMutableData<float>(),
                                src.GetTensorData<float>(), numel * 4);
                    run_inputs.push_back(std::move(t));
                }
            }

            auto outputs = enc.Run(Ort::RunOptions{nullptr}, in_p.data(),
                                   run_inputs.data(), run_inputs.size(),
                                   out_p.data(), out_p.size());
            double ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t0)
                                .count();
            total += ms;
            if (round == 0) {
                for (size_t i = 0; i < outputs.size(); ++i) {
                    auto ti = outputs[i].GetTensorTypeAndShapeInfo();
                    std::string sh;
                    for (auto d : ti.GetShape()) sh += std::to_string(d) + " ";
                    std::cerr << "  out[" << i << "] " << out_names[i] << " [" << sh
                              << "]\n";
                }
            }
            cur_states.clear();
            for (size_t i = 1; i < outputs.size(); ++i)
                cur_states.push_back(std::move(outputs[i]));
            std::cerr << "round " << round + 1 << "/" << n_rounds << " ok, "
                      << ms << "ms\n";
        }
        std::cerr << "ALL ROUNDS OK: n=" << n_rounds << " avg=" << total / n_rounds
                  << "ms\n";
    } catch (const std::exception& e) {
        std::cerr << "CAUGHT: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
