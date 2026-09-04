/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * ASR 一体化实时识别示例: audio 采集组件 + model-zoo-asr 识别组件 (单进程)
 *
 * 整合了两个组件:
 *   - SpacemitAudio::AudioCapture (PortAudio, 进程内麦克风采集)
 *   - SpacemiT::AsrEngine (SenseVoice/Zipformer 识别)
 * 支持: VAD 断句 / 定时断句 / 实时字幕 / 唤醒词文本匹配
 *
 * Usage:
 *   ./asr_live_demo                          # 默认设备, 定时断句 3s
 *   ./asr_live_demo --vad                    # VAD 断句 (说完一句立即出字)
 *   ./asr_live_demo --vad --wake 小美        # 唤醒词检测
 *   ./asr_live_demo -l                       # 列出音频设备
 */

#include <csignal>
#include <cmath>
#include <cstring>

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
#include "audio_base.hpp"

constexpr int SAMPLE_RATE = 16000;

static std::atomic<bool> g_running{true};
static std::atomic<bool> g_waked{false};
static std::vector<std::string> g_wake_words;
static std::string g_wake_hit;

static void signalHandler(int) {
    g_running = false;
}

static void checkWakeWord(const std::string& text) {
    if (g_wake_words.empty() || g_waked.load()) return;
    for (const auto& w : g_wake_words) {
        if (!w.empty() && text.find(w) != std::string::npos) {
            g_wake_hit = w;
            g_waked.store(true);
            std::cout << "\n*************************************" << std::endl;
            std::cout << "  [🔔 唤醒] 识别到唤醒词: \"" << w << "\"" << std::endl;
            std::cout << "  原文: " << text << std::endl;
            std::cout << "*************************************" << std::endl;
            return;
        }
    }
}

class LiveCallback : public SpacemiT::AsrEngineCallback {
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
            std::cout << ">>> " << text;
            if (result->GetAudioDuration() > 0) {
                std::cout << "    (音频 " << result->GetAudioDuration()
                    << "ms, 处理 " << result->GetProcessingTime()
                    << "ms, RTF " << std::fixed << std::setprecision(3)
                    << result->GetRTF() << ")";
            }
            std::cout << std::endl;
        } else {
            std::lock_guard<std::mutex> lock(mutex_);
            if (text != last_partial_) {
                last_partial_ = text;
                std::cout << "    ... " << text << std::endl;
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
// 音频块: 单声道 PCM + RMS
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

// 采集回调 -> 线程安全队列 (主循环消费)
class CaptureQueue {
public:
    void push(AudioChunkData&& c) {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_.push_back(std::move(c));
    }

    void drain(std::deque<AudioChunkData>& out) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& c : buffer_) out.push_back(std::move(c));
        buffer_.clear();
    }

private:
    std::mutex mutex_;
    std::deque<AudioChunkData> buffer_;
};

// ---------------------------------------------------------------------------
// VAD 断句器
// ---------------------------------------------------------------------------
class VadSegmenter {
public:
    VadSegmenter(float thresh, float pause_secs, float max_utt_secs)
        : thresh_(thresh), pause_secs_(pause_secs), max_utt_secs_(max_utt_secs) {}

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
            if (utteranceSeconds() > 2.0f) reset();
            return false;
        }
        return silence_secs_ >= pause_secs_ || utteranceSeconds() >= max_utt_secs_;
    }

    bool hasSpeech() const { return have_speech_; }

    // 取出待识别音频: 保留到最后一个语音块及其后 1 个静音块
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
    std::cout << "Usage: " << program << " [选项]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -i, --input <N>    输入设备索引 (-1 为默认, 跟随系统设置)" << std::endl;
    std::cout << "  -c, --channels <N> 采集通道数 (默认 1)" << std::endl;
    std::cout << "  -l, --list         列出音频设备" << std::endl;
    std::cout << "  --language <L>     zh | en | ja | ko | yue | auto (默认 auto)" << std::endl;
    std::cout << "  --provider <EP>    cpu | spacemit (默认 spacemit)" << std::endl;
    std::cout << "  --core-arch <A>    核架构: x100 | a100 (默认 x100)" << std::endl;
    std::cout << "  --flush <N>        定时断句: 每 N 秒识别一次 (默认 3)" << std::endl;
    std::cout << "  --vad              VAD 断句: 说完一句立即出字幕" << std::endl;
    std::cout << "  --vad-thresh <N>   语音判定 RMS 阈值 (默认 400)" << std::endl;
    std::cout << "  --pause <N>        停顿多少秒判定一句结束 (默认 0.5)" << std::endl;
    std::cout << "  --max-utt <N>      单句最长秒数 (默认 6)" << std::endl;
    std::cout << "  --wake <词,词>     唤醒词检测" << std::endl;
    std::cout << "  --wake-exit        唤醒后退出 (衔接后续流程)" << std::endl;
    std::cout << "  --engine <name>    sensevoice | zipformer (默认 sensevoice)" << std::endl;
    std::cout << "  --emotion          启用情绪识别" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program << " -l" << std::endl;
    std::cout << "  " << program << " --vad --language auto" << std::endl;
    std::cout << "  " << program << " --vad --wake 小美 --wake-exit" << std::endl;
}

int main(int argc, char* argv[]) {
    int device_index = -1;
    int channels = 1;
    bool list_devices = false;
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

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-l" || arg == "--list") {
            list_devices = true;
        } else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            device_index = std::stoi(argv[++i]);
        } else if ((arg == "-c" || arg == "--channels") && i + 1 < argc) {
            channels = std::stoi(argv[++i]);
        } else if ((arg == "--language") && i + 1 < argc) {
            language = argv[++i];
        } else if ((arg == "--provider" || arg == "-p") && i + 1 < argc) {
            provider = argv[++i];
        } else if (arg == "--core-arch" && i + 1 < argc) {
            core_arch = argv[++i];
        } else if ((arg == "--engine" || arg == "-e") && i + 1 < argc) {
            engine = argv[++i];
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

    if (list_devices) {
        std::cout << "可用音频输入设备:" << std::endl;
        for (const auto& [index, name] : SpacemitAudio::AudioCapture::ListDevices()) {
            std::cout << "  [" << index << "] " << name << std::endl;
        }
        return 0;
    }

    if (!wake_words.empty()) {
        std::string w;
        std::istringstream iss(wake_words);
        while (std::getline(iss, w, ',')) {
            if (!w.empty()) g_wake_words.push_back(w);
        }
    }

    std::cout << "========================================" << std::endl;
    std::cout << "  ASR 一体化实时识别 (采集+识别)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "引擎: " << engine << ", 语言: " << language
        << ", Provider: " << provider << ", 核架构: " << core_arch << std::endl;
    if (use_vad) {
        std::cout << "断句: VAD (停顿 " << pause_secs << "s 出字幕, 阈值 "
            << vad_thresh << ", 单句上限 " << max_utt_secs << "s)" << std::endl;
    } else {
        std::cout << "断句: 定时 (每 " << flush_seconds << "s)" << std::endl;
    }
    if (!g_wake_words.empty()) {
        std::cout << "唤醒词: ";
        for (size_t i = 0; i < g_wake_words.size(); ++i) {
            std::cout << (i ? " / " : "") << g_wake_words[i];
        }
        std::cout << (wake_exit ? "  (唤醒后退出)" : "") << std::endl;
    }
    std::cout << std::endl;

    if (engine == "zipformer") {
        std::cout << "提示: zipformer 在 K3 + SpaceMIT EP 下需设置" << std::endl;
        std::cout << "      export SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=\"Conv\"" << std::endl;
        std::cout << std::endl;
    }

    SpacemiT::AsrConfig config = SpacemiT::AsrConfig::Preset(engine);
    config.language = language;
    config.punctuation = true;
    config.provider = provider;
    config.core_arch = core_arch;
    config.enable_emotion = enable_emotion;

    auto asrEngine = std::make_shared<SpacemiT::AsrEngine>(config);
    if (!asrEngine->IsInitialized()) {
        std::cerr << "ASR 引擎初始化失败!" << std::endl;
        return 1;
    }

    std::cout << ">>> Warmup..." << std::endl;
    {
        std::vector<float> silence(8000, 0.0f);
        asrEngine->Recognize(silence, SAMPLE_RATE);
        std::cout << "Warmup done" << std::endl;
    }
    std::cout << std::endl;

    auto callback = std::make_shared<LiveCallback>();
    asrEngine->SetCallback(callback);
    asrEngine->Start();

    CaptureQueue queue;
    VadSegmenter segmenter(vad_thresh, pause_secs, max_utt_secs);

    // 采集回调: 多通道混单声道, 计 RMS, 入队
    SpacemitAudio::AudioCapture capture(device_index);
    capture.SetCallback([&](const uint8_t* data, size_t size) {
        if (!g_running.load()) return;
        AudioChunkData c;
        if (channels <= 1) {
            c.bytes.assign(data, data + size);
        } else {
            // 交织多通道 -> 平均混音为单声道
            const int16_t* src = reinterpret_cast<const int16_t*>(data);
            size_t frames = size / sizeof(int16_t) / channels;
            std::vector<uint8_t> mono(frames * sizeof(int16_t));
            int16_t* dst = reinterpret_cast<int16_t*>(mono.data());
            for (size_t f = 0; f < frames; ++f) {
                int32_t sum = 0;
                for (int ch = 0; ch < channels; ++ch) {
                    sum += src[f * channels + ch];
                }
                dst[f] = static_cast<int16_t>(sum / channels);
            }
            c.bytes = std::move(mono);
        }
        c.rms = calcRms(c.bytes.data(), c.bytes.size());
        c.seconds = static_cast<float>(c.bytes.size()) / (sizeof(int16_t) * SAMPLE_RATE);
        queue.push(std::move(c));
    });

    std::cout << ">>> 启动音频采集 (设备 " << device_index << ", "
        << SAMPLE_RATE << "Hz " << channels << "ch)..." << std::endl;
    if (!capture.Start(SAMPLE_RATE, channels, 4096)) {
        std::cerr << "音频采集启动失败! 用 -l 查看设备" << std::endl;
        return 1;
    }

    std::cout << ">>> 开始识别, Ctrl+C 停止" << std::endl;
    std::cout << "========================================" << std::endl;

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
            std::cout << "[句子 " << sentence_count << "] Flush..." << std::endl;
        }
        asrEngine->SendAudioFrame(pcm);
        asrEngine->Flush();
    };

    while (g_running) {
        queue.drain(incoming);

        if (use_vad) {
            bool flush_now = false;
            for (const auto& c : incoming) {
                flush_now = segmenter.feed(c) || flush_now;
            }
            if (flush_now) {
                doFlush(false);
            }
        } else {
            for (const auto& c : incoming) buffered_since_flush += c.seconds;
            if (buffered_since_flush >= flush_seconds) {
                doFlush(true);
            }
        }
        incoming.clear();

        if (g_waked && wake_exit) {
            std::cout << ">>> 唤醒退出, 此处可衔接后续流程 (LLM / TTS)" << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // 收尾剩余音频
    queue.drain(incoming);
    if (use_vad) {
        for (const auto& c : incoming) segmenter.feed(c);
        incoming.clear();
        if (segmenter.hasSpeech()) doFlush(false);
    } else if (!incoming.empty()) {
        for (const auto& c : incoming) buffered_since_flush += c.seconds;
        doFlush(true);
    }

    capture.Stop();
    capture.Close();
    asrEngine->Stop();

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "共识别 " << sentence_count << " 段" << std::endl;
    if (!callback->getLastText().empty()) {
        std::cout << "最后结果: " << callback->getLastText() << std::endl;
    }
    std::cout << "Done." << std::endl;

    return 0;
}
