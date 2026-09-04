/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * SpacemitAudioSDK 管道流式识别示例 (stdin PCM, 无 PortAudio 依赖)
 *
 * 两种断句方式:
 *   1. 定时断句 (--flush N): 每累计 N 秒音频识别一次
 *   2. VAD 断句 (--vad): 检测到语音停顿立刻识别, 适合实时字幕场景
 *      (说完一句 ~0.5 秒内出文字; 纯静音不会产生垃圾输出)
 *
 * 音频采集交给外部工具 (如 arecord), 因此不需要 audio 组件和 PortAudio。
 *
 * Usage:
 *   arecord -D default -f S16_LE -r 16000 -c 1 | ./asr_pipe_demo --vad
 *   arecord -D plughw:2,0 -f S16_LE -r 16000 -c 1 | ./asr_pipe_demo --language zh
 *   cat audio_16k_mono.pcm | ./asr_pipe_demo
 */

#include <csignal>
#include <cerrno>
#include <unistd.h>
#include <cmath>
#include <cstdio>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "asr_service.h"

constexpr int SAMPLE_RATE = 16000;
constexpr size_t READ_CHUNK_BYTES = 6400;  // 200ms @ 16kHz mono s16le

static std::atomic<bool> g_running{true};
static std::atomic<bool> g_waked{false};
static std::vector<std::string> g_wake_words;  // 唤醒词列表 (空 = 不检测)
static std::string g_wake_hit;
// --jsonl: stdout 只输出 JSON Lines 事件 (ready/partial/sentence), 供下游
// 程序解析; banner 与诊断信息全部转到 stderr, 保证 stdout 可直接 pipe。
static bool g_jsonl = false;

static std::ostream& diag() { return g_jsonl ? std::cerr : std::cout; }

static std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

static void checkWakeWord(const std::string& text) {
    if (g_wake_words.empty() || g_waked.load()) return;
    for (const auto& w : g_wake_words) {
        if (!w.empty() && text.find(w) != std::string::npos) {
            g_wake_hit = w;
            g_waked.store(true);
            diag() << "\n*************************************" << std::endl;
            diag() << "  [🔔 唤醒] 识别到唤醒词: \"" << w << "\"" << std::endl;
            diag() << "  原文: " << text << std::endl;
            diag() << "*************************************" << std::endl;
            return;
        }
    }
}

void signalHandler(int) {
    g_running = false;
}

class PipeCallback : public SpacemiT::AsrEngineCallback {
public:
    void OnOpen() override {}

    void OnEvent(std::shared_ptr<SpacemiT::RecognitionResult> result) override {
        if (!result) return;

        std::string text = result->GetText();
        if (text.empty()) return;

        if (result->IsSentenceEnd()) {
            checkWakeWord(text);
            std::lock_guard<std::mutex> lock(mutex_);
            last_text_ = text;
            if (g_jsonl) {
                std::cout << "{\"type\":\"sentence\",\"text\":\"" << jsonEscape(text) << "\"";
                if (result->GetAudioDuration() > 0) {
                    std::cout << ",\"audio_ms\":" << result->GetAudioDuration()
                              << ",\"proc_ms\":" << result->GetProcessingTime()
                              << ",\"rtf\":" << std::fixed << std::setprecision(3)
                              << result->GetRTF();
                }
                std::cout << "}" << std::endl;
            } else {
                std::cout << ">>> " << text;
                if (result->GetAudioDuration() > 0) {
                    std::cout << "    (音频 " << result->GetAudioDuration()
                        << "ms, 处理 " << result->GetProcessingTime()
                        << "ms, RTF " << std::fixed << std::setprecision(3)
                        << result->GetRTF() << ")";
                }
                std::cout << std::endl;
            }
        } else {
            // 流式引擎的中间结果, 内容变化时才打印
            std::lock_guard<std::mutex> lock(mutex_);
            if (text != last_partial_) {
                last_partial_ = text;
                if (g_jsonl) {
                    std::cout << "{\"type\":\"partial\",\"text\":\""
                              << jsonEscape(text) << "\"}" << std::endl;
                } else {
                    std::cout << "    ... " << text << std::endl;
                }
            }
        }
    }

    void OnError(std::shared_ptr<SpacemiT::RecognitionResult> result) override {
        if (result) {
            std::cerr << "[回调] 错误: " << result->GetText() << std::endl;
        }
    }

    std::string getLastText() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_text_;
    }

private:
    mutable std::mutex mutex_;
    std::string last_text_;
    std::string last_partial_;
};

// ---------------------------------------------------------------------------
// 音频块: 原始 PCM + 能量 (RMS), 供 VAD 断句使用
// ---------------------------------------------------------------------------
struct AudioChunkData {
    std::vector<uint8_t> bytes;
    float rms = 0.0f;
    float seconds = 0.0f;
    bool speech = false;
};

static float calcRms(const uint8_t* data, size_t size) {
    const int16_t* samples = reinterpret_cast<const int16_t*>(data);
    size_t n = size / sizeof(int16_t);
    if (n == 0) return 0.0f;
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        acc += double(samples[i]) * samples[i];
    }
    return static_cast<float>(std::sqrt(acc / n));
}

// 后台线程从 stdin 读 PCM, 按块累积 (含 RMS)
class PipeReader {
public:
    void start() {
        thread_ = std::thread([this] { run(); });
    }

    void join() {
        if (thread_.joinable()) thread_.join();
    }

    // 取走所有已到达的音频块
    void drain(std::deque<AudioChunkData>& out) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& c : buffer_) out.push_back(std::move(c));
        buffer_.clear();
    }

    bool eof() const { return eof_.load(); }

private:
    void run() {
        std::vector<char> chunk(READ_CHUNK_BYTES);
        while (g_running) {
            ssize_t n = ::read(STDIN_FILENO, chunk.data(), chunk.size());
            if (n > 0) {
                AudioChunkData c;
                c.bytes.assign(chunk.begin(), chunk.begin() + n);
                c.rms = calcRms(c.bytes.data(), c.bytes.size());
                c.seconds = static_cast<float>(c.bytes.size()) / (sizeof(int16_t) * SAMPLE_RATE);
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    buffer_.push_back(std::move(c));
                }
            } else if (n == 0) {
                eof_.store(true);
                return;
            } else if (errno != EINTR) {
                eof_.store(true);
                return;
            }
        }
    }

    mutable std::mutex mutex_;
    std::deque<AudioChunkData> buffer_;
    std::atomic<bool> eof_{false};
    std::thread thread_;
};

// ---------------------------------------------------------------------------
// VAD 断句器: 检测语音停顿, 决定何时 Flush
// ---------------------------------------------------------------------------
class VadSegmenter {
public:
    VadSegmenter(float thresh, float pause_secs, float max_utt_secs)
        : thresh_(thresh), pause_secs_(pause_secs), max_utt_secs_(max_utt_secs) {}

    // 喂入一个块; 返回 true 表示此刻应 Flush 出字幕
    bool feed(const AudioChunkData& c) {
        if (utterance_.empty() && c.rms <= thresh_) {
            return false;  // 丢弃语音开始前的静音
        }
        AudioChunkData cc = c;
        cc.speech = c.rms > thresh_;
        utterance_.push_back(std::move(cc));
        if (cc.speech) {
            have_speech_ = true;
            silence_secs_ = 0.0f;
        } else {
            silence_secs_ += cc.seconds;
        }
        if (!have_speech_) {
            // 理论上到不了这里 (开头静音已被丢弃), 兜底丢弃
            if (utteranceSeconds() > 2.0f) reset();
            return false;
        }
        return silence_secs_ >= pause_secs_ || utteranceSeconds() >= max_utt_secs_;
    }

    bool eofRemainder() const { return have_speech_; }

    // 取出待识别音频: 保留到最后一个语音块及其后 1 个静音块 (防截断)
    std::vector<uint8_t> takeAudio() {
        std::vector<uint8_t> out;
        size_t last_speech = utterance_.size();
        for (size_t i = utterance_.size(); i > 0; --i) {
            if (utterance_[i - 1].speech) {
                last_speech = i - 1;
                break;
            }
        }
        size_t end = std::min(utterance_.size(), last_speech + 2);
        for (size_t i = 0; i < end; ++i) {
            out.insert(out.end(), utterance_[i].bytes.begin(), utterance_[i].bytes.end());
        }
        reset();
        return out;
    }

private:
    float utteranceSeconds() const {
        float s = 0.0f;
        for (const auto& c : utterance_) s += c.seconds;
        return s;
    }

    void reset() {
        utterance_.clear();
        have_speech_ = false;
        silence_secs_ = 0.0f;
    }

    float thresh_;
    float pause_secs_;
    float max_utt_secs_;
    std::deque<AudioChunkData> utterance_;
    bool have_speech_ = false;
    float silence_secs_ = 0.0f;
};

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [选项]  (stdin: 16kHz mono PCM16)" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --language <L>     zh | en | ja | ko | yue | auto (默认 auto)" << std::endl;
    std::cout << "  --provider <EP>    cpu | spacemit (默认 spacemit)" << std::endl;
    std::cout << "  --core-arch <A>    核架构: x100 | a100 (默认 x100)" << std::endl;
    std::cout << "  --flush <N>        定时断句: 每 N 秒音频识别一次 (默认 3)" << std::endl;
    std::cout << "  --vad              VAD 断句: 说完一句自动出字幕 (实时字幕推荐)" << std::endl;
    std::cout << "  --vad-thresh <N>   语音判定 RMS 阈值 (默认 400, 环境噪杂时调大)" << std::endl;
    std::cout << "  --pause <N>        停顿多少秒判定一句话结束 (默认 0.5)" << std::endl;
    std::cout << "  --max-utt <N>      单句最长秒数, 超过强制断句 (默认 6)" << std::endl;
    std::cout << "  --engine <name>    sensevoice | zipformer (默认 sensevoice)" << std::endl;
    std::cout << "  --jsonl            stdout 输出 JSON Lines 事件 (ready/partial/sentence)," << std::endl;
    std::cout << "                    banner 转 stderr, 供下游程序解析" << std::endl;
    std::cout << "  --model-dir <DIR>  模型目录 (默认 ~/.cache/models/asr/sensevoice/)" << std::endl;
    std::cout << "  --emotion          启用情绪识别" << std::endl;
    std::cout << "  --wake <词,词>     唤醒词检测: 识别文本包含任一词即触发提示" << std::endl;
    std::cout << "  --wake-exit        唤醒后自动退出 (供脚本接入下一级流程)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  arecord -D default -f S16_LE -r 16000 -c 1 | " << program << " --vad" << std::endl;
    std::cout << "  cat audio.pcm | " << program << " --language zh --flush 5" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string language = "auto";
    std::string provider = "spacemit";
    std::string core_arch = "x100";
    std::string engine = "sensevoice";
    float flush_seconds = 3.0f;
    bool enable_emotion = false;
    bool use_vad = false;
    float vad_thresh = 400.0f;
    float pause_secs = 0.5f;
    float max_utt_secs = 6.0f;
    std::string wake_words;
    bool wake_exit = false;
    std::string model_dir;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if ((arg == "--language" || arg == "-l") && i + 1 < argc) {
            language = argv[++i];
        } else if ((arg == "--provider" || arg == "-p") && i + 1 < argc) {
            provider = argv[++i];
        } else if (arg == "--core-arch" && i + 1 < argc) {
            core_arch = argv[++i];
        } else if ((arg == "--engine" || arg == "-e") && i + 1 < argc) {
            engine = argv[++i];
        } else if (arg == "--model-dir" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (arg == "--jsonl") {
            g_jsonl = true;
        } else if (arg == "--flush" && i + 1 < argc) {
            flush_seconds = std::stof(argv[++i]);
        } else if (arg == "--vad") {
            use_vad = true;
        } else if (arg == "--vad-thresh" && i + 1 < argc) {
            vad_thresh = std::stof(argv[++i]);
        } else if (arg == "--pause" && i + 1 < argc) {
            pause_secs = std::stof(argv[++i]);
        } else if (arg == "--max-utt" && i + 1 < argc) {
            max_utt_secs = std::stof(argv[++i]);
        } else if (arg == "--wake" && i + 1 < argc) {
            wake_words = argv[++i];
        } else if (arg == "--wake-exit") {
            wake_exit = true;
        } else if (arg == "--emotion") {
            enable_emotion = true;
        } else {
            std::cerr << "未知参数: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    if (!wake_words.empty()) {
        std::string w;
        std::istringstream iss(wake_words);
        while (std::getline(iss, w, ',')) {
            if (!w.empty()) g_wake_words.push_back(w);
        }
    }

    diag() << "========================================" << std::endl;
    diag() << "  SpacemitAudioSDK 管道流式识别" << std::endl;
    diag() << "========================================" << std::endl;
    diag() << "引擎: " << engine << ", 语言: " << language
        << ", Provider: " << provider << ", 核架构: " << core_arch << std::endl;
    if (!model_dir.empty()) {
        diag() << "模型目录: " << model_dir << std::endl;
    }
    if (use_vad) {
        diag() << "断句: VAD (停顿 " << pause_secs << "s 出字幕, 阈值 "
            << vad_thresh << ", 单句上限 " << max_utt_secs << "s)" << std::endl;
    } else {
        diag() << "断句: 定时 (每 " << flush_seconds << "s)" << std::endl;
    }
    if (!g_wake_words.empty()) {
        diag() << "唤醒词: ";
        for (size_t i = 0; i < g_wake_words.size(); ++i) {
            diag() << (i ? " / " : "") << g_wake_words[i];
        }
        diag() << (wake_exit ? "  (唤醒后退出)" : "") << std::endl;
    }
    diag() << std::endl;

    if (engine == "zipformer") {
        diag() << "提示: zipformer 在 K3 + SpaceMIT EP 下需设置" << std::endl;
        diag() << "      export SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=\"Conv\"" << std::endl;
        diag() << std::endl;
    }

    SpacemiT::AsrConfig config = SpacemiT::AsrConfig::Preset(engine);
    config.language = language;
    config.punctuation = true;
    config.provider = provider;
    config.core_arch = core_arch;
    config.enable_emotion = enable_emotion;
    if (!model_dir.empty()) {
        config.model_dir = model_dir;
    }

    auto asrEngine = std::make_shared<SpacemiT::AsrEngine>(config);
    if (!asrEngine->IsInitialized()) {
        std::cerr << "ASR 引擎初始化失败!" << std::endl;
        return 1;
    }

    diag() << ">>> Warmup..." << std::endl;
    {
        std::vector<float> silence(8000, 0.0f);
        auto t0 = std::chrono::steady_clock::now();
        asrEngine->Recognize(silence, SAMPLE_RATE);
        auto t1 = std::chrono::steady_clock::now();
        diag() << "Warmup done: " << std::fixed << std::setprecision(0)
            << std::chrono::duration<double, std::milli>(t1 - t0).count()
            << " ms" << std::endl;
    }
    // 模型加载与 warmup 全部完成: 下游 (常驻服务的 prewarm) 以此为准。
    if (g_jsonl) {
        std::cout << "{\"type\":\"ready\"}" << std::endl;
    }
    diag() << std::endl;

    auto callback = std::make_shared<PipeCallback>();
    asrEngine->SetCallback(callback);
    asrEngine->Start();

    PipeReader reader;
    reader.start();

    diag() << ">>> 等待 stdin 音频流 (16kHz mono PCM16)，Ctrl+C 退出" << std::endl;
    diag() << "========================================" << std::endl;

    VadSegmenter segmenter(vad_thresh, pause_secs, max_utt_secs);
    std::deque<AudioChunkData> incoming;
    float buffered_since_flush = 0.0f;
    int sentence_count = 0;

    auto doFlush = [&](bool verbose_tag) {
        std::vector<uint8_t> pcm = use_vad
            ? segmenter.takeAudio()
            : [&] {
                std::vector<uint8_t> all;
                for (const auto& c : incoming) {
                    all.insert(all.end(), c.bytes.begin(), c.bytes.end());
                }
                incoming.clear();
                return all;
              }();
        buffered_since_flush = 0.0f;
        if (pcm.empty()) return;
        sentence_count++;
        if (verbose_tag) {
            diag() << "[句子 " << sentence_count << "] Flush..." << std::endl;
        }
        asrEngine->SendAudioFrame(pcm);
        asrEngine->Flush();
    };

    while (g_running) {
        reader.drain(incoming);

        if (use_vad) {
            // 逐块喂给断句器, 一旦某块触发断句就停在该边界上:
            // 触发块属于当前句, 其后的块留给下一句。这样即使上游一次
            // 灌入一大批音频 (如 cat 文件), 句子也不会被并成一句。
            size_t consumed = 0;
            bool flush_now = false;
            for (; consumed < incoming.size(); ++consumed) {
                if (segmenter.feed(incoming[consumed])) {
                    flush_now = true;
                    ++consumed;
                    break;
                }
            }
            if (flush_now) {
                doFlush(false);
            }
            if (reader.eof() && segmenter.eofRemainder()) {
                doFlush(false);  // 收尾: 把最后一句话送出去
            }
            incoming.erase(incoming.begin(), incoming.begin() + consumed);
        } else {
            for (const auto& c : incoming) buffered_since_flush += c.seconds;
            if (buffered_since_flush >= flush_seconds) {
                doFlush(true);
            }
            incoming.clear();
        }

        if (g_waked && wake_exit) {
            diag() << ">>> 唤醒退出, 此处可衔接后续流程 (LLM / TTS)" << std::endl;
            break;
        }
        if (reader.eof()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // 收尾: 处理剩余音频
    reader.drain(incoming);
    if (use_vad) {
        for (const auto& c : incoming) segmenter.feed(c);
        incoming.clear();
        if (segmenter.eofRemainder()) doFlush(false);
    } else if (!incoming.empty()) {
        for (const auto& c : incoming) buffered_since_flush += c.seconds;
        doFlush(true);
    }

    reader.join();
    asrEngine->Stop();

    diag() << std::endl;
    diag() << "========================================" << std::endl;
    diag() << "共识别 " << sentence_count << " 段" << std::endl;
    if (!callback->getLastText().empty()) {
        diag() << "最后结果: " << callback->getLastText() << std::endl;
    }
    diag() << "Done." << std::endl;

    return 0;
}
