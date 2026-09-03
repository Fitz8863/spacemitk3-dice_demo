// 流式 Zipformer Transducer ASR（sherpa-onnx streaming zipformer 模型，K3 真流式推理）
//
// 模型三件套:
//   encoder: x[1,T,80] + 状态张量 → encoder_out[1,out_T,512] + 新状态（chunk 因果流式）
//   decoder: y[1,context_size]int64 → decoder_out[1,512]
//   joiner : encoder_out[1,512] + decoder_out[1,512] → logit[1,vocab]
// 解码: 每 decode_chunk_len(32) 帧 fbank 跑一次 encoder，逐帧贪心 transducer 解码。
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace zstream {

struct Options {
    std::string model_dir;      // 含 encoder/decoder/joiner .onnx 与 tokens.txt
    std::string encoder_file;   // 空 = 自动选 encoder-epoch-99-avg-1.q.onnx
    bool use_spacemit_ep = true;  // encoder 走 SpaceMIT EP（decoder/joiner 始终 CPU）
    bool ep_disable_conv = false; // 追加 SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=Conv
    int cpu_threads = 2;          // CPU 会话线程数（EP 下 decoder/joiner 用）
    bool verbose = false;         // 调试打印
};

class StreamingASR {
public:
    explicit StreamingASR(const Options& opts);
    ~StreamingASR();

    // 送入 16kHz 单声道 float 样本（可分多次、任意长度）
    void AcceptWaveform(const float* samples, int n);

    // 攒够一个 chunk（32 帧 = 320ms）就推进一次推理。
    // 返回当前部分文本；has_new=false 表示这次没有新 token（文本未变）。
    std::string PollPartial(bool* has_new);

    // 输入结束：解码剩余不足一个 chunk 的帧，返回最终文本
    std::string InputFinished();

    // 只重置 token 假设（已输出文本/token 上下文/decoder 缓存），
    // 保留特征流与 encoder 声学状态的连续性。
    // 用于 VAD 断句：下一句从头显示，但声学上下文连续，句首词不失准。
    void ResetHypothesis();

    // 把特征缓冲里不足一个 chunk 的尾帧立即解码完（补零推进）。
    // VAD 断句取文本前必须先调它，否则最多丢 320ms 尾部音频。
    void FlushPartial();

    // 当前累积文本（不触发推理）
    std::string CurrentText() const;

    // ---- 统计 ----
    double audio_seconds() const;   // 已喂入的音频时长
    double infer_seconds() const;   // 累计推理耗时（encoder+decoder+joiner）
    int processed_frames() const;   // encoder 已消费的 fbank 帧数
    int chunk_count() const;        // 已跑的 encoder chunk 次数
    int emitted_tokens() const;     // 已输出的 token 数

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace zstream
