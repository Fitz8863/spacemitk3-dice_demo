// 流式 Zipformer Transducer ASR 实现（sherpa-onnx streaming zipformer 模型）
//
// 模型三件套:
//   encoder: x[1,T,80] + 状态张量 → encoder_out[1,out_T,512] + 新状态（chunk 因果流式）
//   decoder: y[1,context_size]int64 → decoder_out[1,512]
//   joiner : encoder_out[1,512] + decoder_out[1,512] → logit[1,vocab]
// 解码: 每 decode_chunk_len(32) 帧 fbank 跑一次 encoder，逐帧贪心 transducer 解码。
#include "zipformer_streaming.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <onnxruntime_cxx_api.h>
#ifdef USE_SPACEMIT_EP
#include <spacemit_ort_env.h>
#endif

#include "online-feature.h"

namespace zstream {

namespace {

constexpr int kSampleRate = 16000;
constexpr int kFeatDim = 80;
constexpr int kEncOutDim = 512;        // joiner_dim（encoder 输出维度）
constexpr int kMaxTokensPerFrame = 10; // 单帧贪心解码的发射上限（防死循环）

using Clock = std::chrono::steady_clock;

double SinceMs(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ▁ (U+2581, 0xE2 0x96 0x81) → 空格，并去掉首部空格
std::string PieceToText(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (static_cast<unsigned char>(s[i]) == 0xE2 && i + 2 < s.size() &&
            static_cast<unsigned char>(s[i + 1]) == 0x96 &&
            static_cast<unsigned char>(s[i + 2]) == 0x81) {
            out += ' ';
            i += 2;
        } else {
            out += s[i];
        }
    }
    size_t b = out.find_first_not_of(' ');
    return b == std::string::npos ? "" : out.substr(b);
}

}  // namespace

struct StreamingASR::Impl {
    Options opts;
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "zipformer_streaming"};

    std::unique_ptr<Ort::Session> enc;
    std::unique_ptr<Ort::Session> dec;
    std::unique_ptr<Ort::Session> joiner;
    Ort::AllocatorWithDefaultOptions alloc;

    // 模型元数据（从 onnx metadata 读取）
    int T = 39;             // encoder 输入帧数（含左上下文）
    int shift = 32;         // 每次推进的新帧数 = decode_chunk_len
    int context_size = 2;   // decoder 上下文 token 数
    int vocab_size = 6254;
    int blank_id = 0;

    // encoder 输入输出名（含 35 个状态张量，顺序进/出一一对应）
    std::vector<std::string> enc_in_names, enc_out_names;
    std::vector<const char*> enc_in_ptr, enc_out_ptr;
    std::vector<Ort::Value> states;  // 当前 encoder 状态（new_cached_*）

    // decoder / joiner 名字
    std::vector<std::string> dec_in_names, dec_out_names;
    std::vector<std::string> join_in_names, join_out_names;
    std::vector<const char*> dec_in_ptr, dec_out_ptr;
    std::vector<const char*> join_in_ptr, join_out_ptr;

    // 特征
    std::unique_ptr<knf::OnlineFbank> fbank;
    int frames_done = 0;  // encoder 已消费的帧数

    // transducer 假设
    std::vector<int64_t> ctx;           // decoder 上下文，初始 -1 填充
    std::vector<int> hyp;               // 已发射 token id
    std::vector<int64_t> cached_ctx;    // decoder 缓存对应的上下文
    std::vector<float> cached_dec_out;  // 缓存的 decoder 输出

    // 词表
    std::vector<std::string> id2tok;

    // 统计
    int64_t total_samples = 0;
    double infer_sec = 0.0;
    int n_chunks = 0;
    double enc_ms = 0.0, dec_ms = 0.0, join_ms = 0.0;
    bool finished = false;

    mutable std::mutex mu;

    // ------------------------------------------------------------------
    Impl(const Options& o) : opts(o) {
        std::string dir = opts.model_dir;
        while (!dir.empty() && dir.back() == '/') dir.pop_back();

        std::string enc_path = opts.encoder_file.empty()
                                   ? dir + "/encoder-epoch-99-avg-1.q.onnx"
                                   : opts.encoder_file;
        std::string dec_path = dir + "/decoder-epoch-99-avg-1.onnx";
        std::string join_path = dir + "/joiner-epoch-99-avg-1.onnx";
        std::string tok_path = dir + "/tokens.txt";

        LoadTokens(tok_path);

        // decoder / joiner：小模型，走 CPU
        dec = MakeCpuSession(dec_path);
        joiner = MakeCpuSession(join_path);
        CollectNames(*dec, &dec_in_names, &dec_out_names);
        CollectNames(*joiner, &join_in_names, &join_out_names);

        // decoder 元数据: context_size
        {
            auto meta = dec->GetModelMetadata();
            auto v = meta.LookupCustomMetadataMapAllocated("context_size", alloc);
            if (v.get()) context_size = std::stoi(v.get());
        }

        // encoder：SpaceMIT EP 或 CPU
        bool ep_ok = false;
        if (opts.use_spacemit_ep) {
            enc = MakeEpSession(enc_path, &ep_ok);
        } else {
            enc = MakeCpuSession(enc_path);
        }
        CollectNames(*enc, &enc_in_names, &enc_out_names);

        // encoder 元数据: T / decode_chunk_len
        {
            auto meta = enc->GetModelMetadata();
            auto t = meta.LookupCustomMetadataMapAllocated("T", alloc);
            auto dcl = meta.LookupCustomMetadataMapAllocated("decode_chunk_len", alloc);
            if (t.get()) T = std::stoi(t.get());
            if (dcl.get()) shift = std::stoi(dcl.get());
        }

        // joiner 输出最后一维 = vocab
        {
            auto info = joiner->GetOutputTypeInfo(0);
            auto ti = info.GetTensorTypeAndShapeInfo();
            auto shape = ti.GetShape();
            vocab_size = static_cast<int>(shape.back());
        }
        blank_id = FindBlank();

        InitStates();

        // 特征：与 sherpa-onnx 一致 —— knf 默认参数 + dither 0 + 80 维
        knf::FbankOptions fopts;
        fopts.frame_opts.samp_freq = kSampleRate;
        fopts.frame_opts.dither = 0;
        fopts.mel_opts.num_bins = kFeatDim;
        fbank = std::make_unique<knf::OnlineFbank>(fopts);

        ctx.assign(context_size, -1);
        cached_ctx = ctx;

        // 生成 Run() 用的 const char* 名字数组（必须在整个构造完成后进行）
        BuildPtrs(enc_in_names, &enc_in_ptr);
        BuildPtrs(enc_out_names, &enc_out_ptr);
        BuildPtrs(dec_in_names, &dec_in_ptr);
        BuildPtrs(dec_out_names, &dec_out_ptr);
        BuildPtrs(join_in_names, &join_in_ptr);
        BuildPtrs(join_out_names, &join_out_ptr);

        // 加载日志走 stderr：stdout 保留给识别结果（--jsonl 模式下必须纯净）
        std::cerr << "[stream_asr] encoder: " << enc_path << " ("
                  << (opts.use_spacemit_ep ? (ep_ok ? "SpaceMIT EP" : "CPU fallback")
                                           : "CPU")
                  << ")\n[stream_asr] T=" << T << " chunk_shift=" << shift
                  << " context_size=" << context_size << " vocab=" << vocab_size
                  << " blank=" << blank_id << " states=" << states.size()
                  << std::endl;
    }

    // ---------------- 会话与初始化 ----------------

    std::unique_ptr<Ort::Session> MakeCpuSession(const std::string& path) {
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(opts.cpu_threads);
        so.SetInterOpNumThreads(1);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        return std::make_unique<Ort::Session>(env, path.c_str(), so);
    }

    std::unique_ptr<Ort::Session> MakeEpSession(const std::string& path,
                                                bool* ep_ok) {
#ifdef USE_SPACEMIT_EP
        try {
            Ort::SessionOptions so;
            // 数值来自模型包内 zipformer.config（SpaceMIT 官方导出配置）
            std::unordered_map<std::string, std::string> ep_options = {
                    {"SPACEMIT_EP_INTRA_THREAD_NUM", "1"},
                    {"SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD", "1"},
                    {"SPACEMIT_EP_PERFER_CORE_ARCH", "0x5064"}};
            if (opts.ep_disable_conv) {
                ep_options["SPACEMIT_EP_DISABLE_OP_TYPE_FILTER"] = "Conv";
            }
            Ort::Status st = Ort::SessionOptionsSpaceMITEnvInit(so, ep_options);
            if (!st.IsOK()) {
                std::cerr << "[stream_asr] EP init failed: " << st.GetErrorMessage()
                          << ", fallback to CPU\n";
                *ep_ok = false;
                return MakeCpuSession(path);
            }
            so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            *ep_ok = true;
            return std::make_unique<Ort::Session>(env, path.c_str(), so);
        } catch (const std::exception& e) {
            std::cerr << "[stream_asr] EP session error: " << e.what()
                      << ", fallback to CPU\n";
            *ep_ok = false;
            return MakeCpuSession(path);
        }
#else
        (void)path;
        (void)ep_ok;
        throw std::runtime_error("本二进制编译时未启用 SpaceMIT EP，请用 --cpu");
#endif
    }

    void CollectNames(Ort::Session& s, std::vector<std::string>* ins,
                      std::vector<std::string>* outs) {
        for (size_t i = 0; i < s.GetInputCount(); ++i)
            ins->emplace_back(s.GetInputNameAllocated(i, alloc).get());
        for (size_t i = 0; i < s.GetOutputCount(); ++i)
            outs->emplace_back(s.GetOutputNameAllocated(i, alloc).get());
    }

    // 名字向量填完后，生成供 Run() 使用的 const char* 指针数组。
    // 指向 name vector 内部 string 的存储，之后不得再改动 name 向量。
    static void BuildPtrs(const std::vector<std::string>& names,
                          std::vector<const char*>* ptrs) {
        ptrs->clear();
        for (const auto& nm : names) ptrs->push_back(nm.c_str());
    }

    void LoadTokens(const std::string& path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("cannot open " + path);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            auto pos = line.find_last_of(' ');
            id2tok.push_back(line.substr(0, pos));
        }
    }

    int FindBlank() {
        for (int i = 0; i < static_cast<int>(id2tok.size()); ++i)
            if (id2tok[i] == "<blk>" || id2tok[i] == "<pad>") return i;
        return 0;
    }

    // 按声明的输入形状初始化 encoder 状态（动态维→1，全零）
    void InitStates() {
        Ort::MemoryInfo mem =
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        states.clear();
        for (size_t i = 1; i < enc_in_names.size(); ++i) {
            auto info = enc->GetInputTypeInfo(i);
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
                states.push_back(std::move(t));
            } else {
                auto t =
                        Ort::Value::CreateTensor<float>(alloc, shape.data(), shape.size());
                std::fill_n(t.GetTensorMutableData<float>(), numel, 0.0f);
                states.push_back(std::move(t));
            }
        }
    }

    // ---------------- 音频输入 ----------------

    void AcceptWaveform(const float* samples, int n) {
        std::lock_guard<std::mutex> lock(mu);
        fbank->AcceptWaveform(kSampleRate, samples, n);
        total_samples += n;
    }

    // ---------------- 推理 ----------------

    // 组装 [frames_done-left, frames_done+shift) 窗口并跑一次 encoder，
    // 再对每个输出帧做贪心 transducer 解码。
    void DecodeChunk() {
        std::lock_guard<std::mutex> lock(mu);

        int ready = fbank->NumFramesReady();
        int left = T - shift;
        int start = frames_done - left;

        // x[1, T, 80]：首 chunk 头部补零，尾部不足时补零
        std::vector<float> x(static_cast<size_t>(T) * kFeatDim, 0.0f);
        for (int i = 0; i < T; ++i) {
            int src = start + i;
            if (src < 0 || src >= ready) continue;
            const float* fr = fbank->GetFrame(src);
            std::copy(fr, fr + kFeatDim, x.data() + static_cast<size_t>(i) * kFeatDim);
        }

        Ort::MemoryInfo mem =
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 3> x_shape = {1, T, kFeatDim};

        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(mem, x.data(), x.size(),
                                                         x_shape.data(), 3));
        for (auto& s : states) inputs.push_back(std::move(s));
        states.clear();

        auto t0 = Clock::now();
        if (opts.verbose) {
            float mn = x[0], mx = x[0];
            int bad = 0;
            for (float v : x) {
                if (!std::isfinite(v)) ++bad;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            std::cerr << "[dbg] before enc.Run chunk#" << n_chunks + 1
                      << " feat[min=" << mn << " max=" << mx << " nonfinite=" << bad
                      << "]" << std::endl;
        }
        auto outputs = enc->Run(Ort::RunOptions{nullptr}, enc_in_ptr.data(),
                                inputs.data(), inputs.size(), enc_out_ptr.data(),
                                enc_out_ptr.size());
        if (opts.verbose)
            std::cerr << "[dbg] after enc.Run outputs=" << outputs.size()
                      << std::endl;
        double ms = SinceMs(t0);
        enc_ms += ms;
        infer_sec += ms / 1000.0;

        auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t out_T = shape[1];
        std::vector<float> enc_out(outputs[0].GetTensorData<float>(),
                                   outputs[0].GetTensorData<float>() +
                                           out_T * kEncOutDim);

        for (size_t i = 1; i < outputs.size(); ++i)
            states.push_back(std::move(outputs[i]));

        frames_done += shift;
        ++n_chunks;

        // 逐帧贪心 transducer 解码
        if (opts.verbose)
            std::cerr << "[dbg] decode " << out_T << " frames" << std::endl;
        for (int64_t t = 0; t < out_T; ++t) {
            const float* frame = enc_out.data() + t * kEncOutDim;
            for (int rep = 0; rep < kMaxTokensPerFrame; ++rep) {
                EnsureDecoderOut();
                int id = JoinerArgmax(frame);
                if (id == blank_id) break;
                hyp.push_back(id);
                // 上下文滑动: 去掉最老 token，追加新 token
                for (int k = 0; k + 1 < context_size; ++k) ctx[k] = ctx[k + 1];
                ctx[context_size - 1] = id;
            }
        }
    }

    // decoder 输出只依赖最近 context_size 个 token，做简单缓存
    void EnsureDecoderOut() {
        if (!cached_dec_out.empty() && cached_ctx == ctx) return;
        if (opts.verbose) std::cerr << "[dbg] dec.Run" << std::endl;

        Ort::MemoryInfo mem =
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 2> y_shape = {1, static_cast<int64_t>(context_size)};
        Ort::Value y = Ort::Value::CreateTensor<int64_t>(mem, ctx.data(), ctx.size(),
                                                         y_shape.data(), 2);

        auto t0 = Clock::now();
        auto out = dec->Run(Ort::RunOptions{nullptr}, dec_in_ptr.data(), &y, 1,
                            dec_out_ptr.data(), dec_out_ptr.size());
        double ms = SinceMs(t0);
        dec_ms += ms;
        infer_sec += ms / 1000.0;

        auto ti = out[0].GetTensorTypeAndShapeInfo();
        size_t n = 1;
        for (auto d : ti.GetShape()) n *= static_cast<size_t>(d);
        cached_dec_out.assign(out[0].GetTensorData<float>(),
                              out[0].GetTensorData<float>() + n);
        cached_ctx = ctx;
    }

    int JoinerArgmax(const float* frame) {
        Ort::MemoryInfo mem =
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 2> shape = {1, kEncOutDim};
        Ort::Value e = Ort::Value::CreateTensor<float>(
                mem, const_cast<float*>(frame), kEncOutDim, shape.data(), 2);
        Ort::Value d = Ort::Value::CreateTensor<float>(
                mem, cached_dec_out.data(), cached_dec_out.size(), shape.data(), 2);
        Ort::Value ins[2] = {std::move(e), std::move(d)};

        auto t0 = Clock::now();
        if (opts.verbose) std::cerr << "[dbg] joiner.Run" << std::endl;
        auto out = joiner->Run(Ort::RunOptions{nullptr}, join_in_ptr.data(), ins, 2,
                               join_out_ptr.data(), join_out_ptr.size());
        if (opts.verbose) std::cerr << "[dbg] joiner.Run done" << std::endl;
        double ms = SinceMs(t0);
        join_ms += ms;
        infer_sec += ms / 1000.0;

        const float* logits = out[0].GetTensorData<float>();
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < vocab_size; ++i) {
            if (logits[i] > bv) {
                bv = logits[i];
                best = i;
            }
        }
        return best;
    }

    // ---------------- 对外接口 ----------------

    std::string PollPartial(bool* has_new) {
        *has_new = false;
        while (fbank->NumFramesReady() - frames_done >= shift && !finished) {
            DecodeChunk();
            *has_new = true;
        }
        return *has_new ? BuildText() : "";
    }

    std::string InputFinished() {
        {
            std::lock_guard<std::mutex> lock(mu);
            fbank->InputFinished();
            finished = true;
        }
        while (fbank->NumFramesReady() > frames_done) {
            DecodeChunk();  // 不足一个 chunk 的尾巴：尾部补零强制解码
        }
        return BuildText();
    }

    void FlushPartial() {
        // DecodeChunk 内部自取锁，这里不能持锁调用（mutex 不可重入）
        while (true) {
            {
                std::lock_guard<std::mutex> lock(mu);
                if (fbank->NumFramesReady() <= frames_done) break;
            }
            DecodeChunk();  // 尾部不足一个 chunk 的帧：补零强制解码
        }
    }

    void ResetHypothesis() {
        std::lock_guard<std::mutex> lock(mu);
        // 只清 token 层：假设、decoder 上下文与缓存。
        // 特征流和 encoder 状态保持连续（cached_len 等位置编码不跳变），
        // 句首不缺左侧声学上下文，首词不失准。
        hyp.clear();
        ctx.assign(context_size, -1);
        cached_ctx = ctx;
        cached_dec_out.clear();
    }

    std::string CurrentText() const {
        std::lock_guard<std::mutex> lock(mu);
        return BuildText();
    }

    std::string BuildText() const {
        std::string s;
        for (int id : hyp) {
            if (id < 0 || id >= static_cast<int>(id2tok.size())) continue;
            const std::string& t = id2tok[id];
            if (t == "<blk>" || t == "<sos/eos>" || t == "<unk>") continue;
            s += t;
        }
        return PieceToText(s);
    }

    // ---------------- 统计 ----------------

    double audio_seconds() const {
        return static_cast<double>(total_samples) / kSampleRate;
    }
    double infer_seconds() const { return infer_sec; }
    int processed_frames() const { return frames_done; }
    int chunk_count() const { return n_chunks; }
    int emitted_tokens() const { return static_cast<int>(hyp.size()); }
};

// ---------------------------------------------------------------------

StreamingASR::StreamingASR(const Options& opts) : impl_(std::make_unique<Impl>(opts)) {}
StreamingASR::~StreamingASR() = default;

void StreamingASR::AcceptWaveform(const float* samples, int n) {
    impl_->AcceptWaveform(samples, n);
}

std::string StreamingASR::PollPartial(bool* has_new) {
    return impl_->PollPartial(has_new);
}

std::string StreamingASR::InputFinished() { return impl_->InputFinished(); }

void StreamingASR::ResetHypothesis() { impl_->ResetHypothesis(); }

void StreamingASR::FlushPartial() { impl_->FlushPartial(); }

std::string StreamingASR::CurrentText() const { return impl_->CurrentText(); }

double StreamingASR::audio_seconds() const { return impl_->audio_seconds(); }
double StreamingASR::infer_seconds() const { return impl_->infer_seconds(); }
int StreamingASR::processed_frames() const { return impl_->processed_frames(); }
int StreamingASR::chunk_count() const { return impl_->chunk_count(); }
int StreamingASR::emitted_tokens() const { return impl_->emitted_tokens(); }

}  // namespace zstream
