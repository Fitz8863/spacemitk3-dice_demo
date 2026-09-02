// stream_asr: 流式 zipformer 识别演示
//
// 输入模式:
//   --wav FILE   识别 wav 文件（默认只出最终结果与统计）
//   --realtime   wav 按实时节奏喂入，模拟边说边出字
//   --pcm        从 stdin 读 s16le/16kHz/mono 裸流（配合 arecord 管道做真麦克风）
//
// 断句（VAD）:
//   --pcm 默认开启；说完一句停顿约 0.6s 自动出整句并重置解码状态。
//   --vad / --no-vad 强制开关；--vad-rms/--vad-pause-ms/--vad-max-ms 调参。
#include "zipformer_streaming.h"

#include <signal.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <sndfile.h>

namespace {

volatile sig_atomic_t g_stop = 0;
void OnSigint(int) { g_stop = 1; }

// 抓 terminate 时的真实原因
void TermHandler() {
    if (auto eptr = std::current_exception()) {
        try {
            std::rethrow_exception(eptr);
        } catch (const std::exception& e) {
            std::cerr << "\n[TERM] active exception: " << e.what() << "\n";
        } catch (...) {
            std::cerr << "\n[TERM] active exception (unknown)\n";
        }
    } else {
        std::cerr << "\n[TERM] no active exception (thread/abort path)\n";
    }
    std::abort();
}

constexpr int kSampleRate = 16000;
constexpr int kChunkMs = 100;  // 每次喂入的音频时长
constexpr int kChunkSamples = kSampleRate * kChunkMs / 1000;
constexpr int kTailChars = 44;  // 单行字幕窗显示的最后字符数

std::string DefaultModelDir() {
    return "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20";
}

// UTF-8 安全的尾部截取（避免切在多字节字符中间）
std::string TailUtf8(const std::string& s, size_t max_chars) {
    if (s.size() <= max_chars) return s;  // 字节数都不超，必然安全
    // 数出最后 max_chars 个字符的字节范围
    size_t cnt = 0, byte_end = s.size();
    for (size_t i = s.size(); i-- > 0;) {
        if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80) {  // 字符首字节
            ++cnt;
            if (cnt >= max_chars) {
                return "…" + s.substr(i);
            }
        }
    }
    return s;
}

std::string TimeStr(double sec) {
    char buf[16];
    snprintf(buf, sizeof(buf), "%.1f", sec);
    return buf;
}

// ---------------- 能量 VAD 断句 ----------------
struct EnergyVad {
    bool enabled = false;
    double rms_threshold = 400;  // s16 量级的 RMS 门限，低于视为静音
    int pause_ms = 600;          // 连续静音该时长 → 断句
    int max_ms = 8000;           // 单句最长时长，到点强制断句

    int silence_ms_acc = 0;
    int speech_ms_acc = 0;

    void Feed(const float* s, int n, int ms) {
        double sum = 0;
        for (int i = 0; i < n; ++i) sum += static_cast<double>(s[i]) * s[i];
        double rms = std::sqrt(sum / n) * 32768.0;
        if (rms < rms_threshold) {
            silence_ms_acc += ms;
        } else {
            silence_ms_acc = 0;
            speech_ms_acc += ms;
        }
    }

    // 该断句了吗（静音足够久 或 单句过长）
    bool ShouldBreak(bool has_hypothesis) const {
        if (!enabled || !has_hypothesis) return false;
        return silence_ms_acc >= pause_ms || speech_ms_acc >= max_ms;
    }

    void OnBreak() {
        silence_ms_acc = 0;
        speech_ms_acc = 0;
    }
};

// ---------------- 显示 ----------------

// 单行字幕：只显示最近 kTailChars 个字符，\r 刷新（文本超屏也不会刷屏）
void PrintPartial(double audio_sec, const std::string& text, size_t* last_len) {
    std::string line =
            "[" + TimeStr(audio_sec) + "s] " + TailUtf8(text, kTailChars);
    size_t pad = line.size() < *last_len ? *last_len - line.size() : 0;
    std::cout << "\r" << line << std::string(pad, ' ') << std::flush;
    *last_len = line.size();
}

// 结束当前字幕行，整句单独占一行
void PrintSentence(double audio_sec, const std::string& text) {
    std::cout << "\r" << std::string(78, ' ') << "\r";  // 清掉字幕行
    std::cout << "[" << TimeStr(audio_sec) << "s] " << text << "\n" << std::flush;
}

void PrintStats(const zstream::StreamingASR& asr) {
    double audio = asr.audio_seconds();
    double infer = asr.infer_seconds();
    printf("\n[统计] 音频=%.2fs 推理=%.2fs RTF=%.3f chunks=%d tokens=%d\n",
           audio, infer, audio > 0 ? infer / audio : 0.0, asr.chunk_count(),
           asr.emitted_tokens());
}

void PrintUsage() {
    std::cout
            << "用法: stream_asr [选项]\n"
            << "  --model-dir DIR    模型目录 (默认 " << DefaultModelDir() << ")\n"
            << "  --wav FILE         识别 wav 文件（默认只出最终结果）\n"
            << "  --realtime         wav 按实时节奏喂入，模拟边说边出字\n"
            << "  --pcm              从 stdin 读 s16le/16k/mono 裸流（麦克风管道）\n"
            << "  --cpu              不用 SpaceMIT EP，encoder 纯 CPU\n"
            << "  --encoder FILE     指定 encoder onnx（默认 q.onnx；CPU 配 int8.onnx）\n"
            << "  --ep-disable-conv  SpaceMIT EP 禁用 Conv 算子\n"
            << "  --threads N        CPU 会话线程数 (默认 2)\n"
            << "断句（VAD）:\n"
            << "  --vad / --no-vad   开/关静音断句（--pcm 默认开，--wav 默认关）\n"
            << "  --vad-rms N        静音 RMS 门限，s16 量级 (默认 400)\n"
            << "  --vad-pause-ms N   停顿多久断句 (默认 600)\n"
            << "  --vad-max-ms N     单句最长，到点强断 (默认 8000)\n"
            << "  -v                 调试输出\n";
}

// 读 wav → 单声道 16k float
bool ReadWavMono16k(const std::string& path, std::vector<float>* out) {
    SF_INFO info;
    std::memset(&info, 0, sizeof(info));
    SNDFILE* f = sf_open(path.c_str(), SFM_READ, &info);
    if (!f) {
        std::cerr << "打不开音频文件: " << path << " (" << sf_strerror(nullptr)
                  << ")\n";
        return false;
    }
    std::vector<float> data(info.frames * info.channels);
    sf_count_t n = sf_read_float(f, data.data(), data.size());
    sf_close(f);
    if (n <= 0) return false;
    n /= info.channels;

    std::vector<float> mono(n);
    for (sf_count_t i = 0; i < n; ++i) {
        float sum = 0;
        for (int ch = 0; ch < info.channels; ++ch) sum += data[i * info.channels + ch];
        mono[i] = sum / info.channels;
    }

    if (info.samplerate != kSampleRate) {  // 线性重采样到 16k
        double ratio = static_cast<double>(kSampleRate) / info.samplerate;
        size_t new_n = static_cast<size_t>(n * ratio);
        std::vector<float> rs(new_n);
        for (size_t i = 0; i < new_n; ++i) {
            double src = i / ratio;
            size_t i0 = static_cast<size_t>(src);
            size_t i1 = std::min(i0 + 1, static_cast<size_t>(n) - 1);
            double fr = src - i0;
            rs[i] = static_cast<float>(mono[i0] * (1 - fr) + mono[i1] * fr);
        }
        *out = std::move(rs);
    } else {
        *out = std::move(mono);
    }
    return true;
}

// 每次喂入 100ms 后的公共处理：VAD 判断 + 字幕刷新 + 断句出整句
void StepLoop(zstream::StreamingASR& asr, EnergyVad& vad, size_t* last_len,
              std::vector<std::string>* sentences) {
    bool has_new = false;
    asr.PollPartial(&has_new);
    bool has_hyp = asr.emitted_tokens() > 0;

    if (has_new) {
        PrintPartial(asr.audio_seconds(), asr.CurrentText(), last_len);
    }

    if (vad.ShouldBreak(has_hyp)) {
        asr.FlushPartial();  // 先解码完缓冲尾帧，最多 320ms，否则断句会丢字
        std::string sentence = asr.CurrentText();
        if (!sentence.empty()) {
            PrintSentence(asr.audio_seconds(), sentence);
            if (sentences) sentences->push_back(sentence);
        }
        asr.ResetHypothesis();  // 只重置 token 假设，声学状态连续，下一句从头显示
        vad.OnBreak();
        *last_len = 0;
    }
}

int RunWav(zstream::StreamingASR& asr, const std::string& wav, bool realtime,
           EnergyVad& vad, bool verbose) {
    std::vector<float> audio;
    if (!ReadWavMono16k(wav, &audio)) return 1;

    std::cout << "音频: " << wav << "  " << audio.size() / kSampleRate << "s"
              << (vad.enabled ? "  (VAD 断句开)" : "") << "\n";

    std::vector<std::string> sentences;
    size_t last_len = 0;
    size_t pos = 0;

    while (pos < audio.size() && !g_stop) {
        size_t n = std::min(static_cast<size_t>(kChunkSamples), audio.size() - pos);
        auto tick = std::chrono::steady_clock::now();
        vad.Feed(audio.data() + pos, static_cast<int>(n), kChunkMs);
        asr.AcceptWaveform(audio.data() + pos, static_cast<int>(n));
        pos += n;

        StepLoop(asr, vad, &last_len, &sentences);

        if (verbose) {
            std::cout << "chunk#" << asr.chunk_count() << " t="
                      << static_cast<int>(asr.audio_seconds()) << "s "
                      << asr.CurrentText() << std::endl;
        }
        if (realtime) {  // 按实时节奏喂入
            auto feed_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - tick)
                                   .count();
            int sleep_ms = kChunkMs - static_cast<int>(feed_ms);
            if (sleep_ms > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }
    }

    std::string final_text = asr.InputFinished();
    if (!final_text.empty()) {
        if (vad.enabled && last_len > 0) {
            PrintSentence(asr.audio_seconds(), final_text);
            sentences.push_back(final_text);
        } else if (!vad.enabled) {
            std::cout << "识别结果: " << final_text << "\n";
        }
    } else if (last_len > 0) {
        std::cout << "\n";
    }

    if (vad.enabled && !sentences.empty()) {
        std::cout << "---- 分句结果 (" << sentences.size() << " 句) ----\n";
        for (size_t i = 0; i < sentences.size(); ++i)
            std::cout << i + 1 << ". " << sentences[i] << "\n";
    }
    PrintStats(asr);
    return 0;
}

int RunPcm(zstream::StreamingASR& asr, EnergyVad& vad) {
    std::cout << "stdin 读取 s16le/16kHz/mono 裸流，说话即出字，Ctrl-C 结束\n"
              << "例如: arecord -D default -f S16_LE -r 16000 -c 1 -t raw"
                 " | ./build/stream_asr --pcm\n"
              << (vad.enabled
                          ? "(VAD 断句开: 停顿约 " +
                                    std::to_string(vad.pause_ms / 1000.0).substr(0, 3) +
                                    "s 出整句)\n"
                          : "(VAD 断句关: 文本持续累积)\n");

    std::vector<std::string> sentences;
    size_t last_len = 0;
    std::vector<int16_t> buf(kChunkSamples);

    while (!g_stop) {
        size_t got = fread(buf.data(), sizeof(int16_t), buf.size(), stdin);
        if (got == 0) {
            if (feof(stdin)) break;
            continue;
        }
        std::vector<float> samples(got);
        for (size_t i = 0; i < got; ++i)
            samples[i] = static_cast<float>(buf[i]) / 32768.0f;
        vad.Feed(samples.data(), static_cast<int>(got),
                 static_cast<int>(got * 1000 / kSampleRate));
        asr.AcceptWaveform(samples.data(), static_cast<int>(got));

        StepLoop(asr, vad, &last_len, &sentences);
    }

    // 流结束：把最后一段假设出完
    std::string final_text = asr.InputFinished();
    if (!final_text.empty()) {
        if (last_len > 0) std::cout << "\n";
        if (vad.enabled) {
            std::cout << "[end] " << final_text << "\n";
            sentences.push_back(final_text);
        } else {
            std::cout << "识别结果: " << final_text << "\n";
        }
    }
    if (vad.enabled && !sentences.empty()) {
        std::cout << "---- 分句结果 (" << sentences.size() << " 句) ----\n";
        for (size_t i = 0; i < sentences.size(); ++i)
            std::cout << i + 1 << ". " << sentences[i] << "\n";
    }
    PrintStats(asr);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    signal(SIGINT, OnSigint);
    std::set_terminate(TermHandler);

    zstream::Options opts;
    std::string wav;
    bool realtime = false, pcm = false, verbose = false;
    bool vad_on = false, vad_off = false;
    EnergyVad vad;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "缺少参数值: " << what << "\n";
                exit(1);
            }
            return argv[++i];
        };
        if (a == "--model-dir") opts.model_dir = next("--model-dir");
        else if (a == "--wav") wav = next("--wav");
        else if (a == "--realtime") realtime = true;
        else if (a == "--pcm") pcm = true;
        else if (a == "--cpu") opts.use_spacemit_ep = false;
        else if (a == "--encoder") opts.encoder_file = next("--encoder");
        else if (a == "--ep-disable-conv") opts.ep_disable_conv = true;
        else if (a == "--threads") opts.cpu_threads = std::stoi(next("--threads"));
        else if (a == "--vad") vad_on = true;
        else if (a == "--no-vad") vad_off = true;
        else if (a == "--vad-rms") vad.rms_threshold = std::stod(next("--vad-rms"));
        else if (a == "--vad-pause-ms") vad.pause_ms = std::stoi(next("--vad-pause-ms"));
        else if (a == "--vad-max-ms") vad.max_ms = std::stoi(next("--vad-max-ms"));
        else if (a == "-v" || a == "--verbose") { verbose = true; opts.verbose = true; }
        else if (a == "-h" || a == "--help") { PrintUsage(); return 0; }
        else { std::cerr << "未知参数: " << a << "\n"; PrintUsage(); return 1; }
    }

    if (opts.model_dir.empty()) opts.model_dir = DefaultModelDir();
    if (wav.empty() && !pcm) {
        PrintUsage();
        return 1;
    }
    // 默认：--pcm 开 VAD，--wav 关 VAD；可用 --vad/--no-vad 覆盖
    vad.enabled = vad_off ? false : (pcm ? true : vad_on);

    try {
        zstream::StreamingASR asr(opts);
        if (pcm) return RunPcm(asr, vad);
        return RunWav(asr, wav, realtime, vad, verbose);
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }
}
