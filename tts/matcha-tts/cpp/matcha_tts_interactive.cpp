#include <sherpa-onnx/c-api/c-api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <csignal>
#include <iostream>
#include <mutex>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

std::atomic<bool> g_stop{false};

void OnSignal(int) { g_stop.store(true); }

struct ModelPaths {
    std::string acoustic;
    std::string vocoder;
    std::string tokens;
    std::string lexicon;
    std::string data_dir;
    std::string rule_fsts;
};

struct AudioChunk {
    std::vector<int16_t> samples;
    int sample_rate = 0;
    int sentence_index = 0;
};

struct Options {
    std::string model_dir;
    std::string audio_player = "aplay";
    std::string ep_affinity;
    std::string warmup_text = "你好。";
    int ep_threads = 2;
    float speed = 1.0f;
    int chunk_target = 70;
    int chunk_max = 120;
    bool enable_affinity = true;
    bool enable_warmup = true;
    bool enable_playback = true;
};

class AudioQueue {
public:
    void Push(AudioChunk chunk) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (finished_) {
                return;
            }
            queue_.push(std::move(chunk));
        }
        condition_.notify_one();
    }

    bool Pop(AudioChunk &chunk) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return finished_ || !queue_.empty(); });
        if (queue_.empty()) {
            return false;
        }
        chunk = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void Finish() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            finished_ = true;
        }
        condition_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    std::queue<AudioChunk> queue_;
    bool finished_ = false;
};

SherpaOnnxOfflineTtsConfig MakeConfig(const ModelPaths &paths, int ep_threads) {
    SherpaOnnxOfflineTtsConfig config{};
    config.model.num_threads = ep_threads;
    config.model.debug = 0;
    config.model.provider = "spacemit";
    config.model.matcha.acoustic_model = paths.acoustic.c_str();
    config.model.matcha.vocoder = paths.vocoder.c_str();
    config.model.matcha.tokens = paths.tokens.c_str();
    config.model.matcha.lexicon = paths.lexicon.c_str();
    config.model.matcha.data_dir = paths.data_dir.c_str();
    config.model.matcha.noise_scale = 0.667f;
    config.model.matcha.length_scale = 1.0f;
    config.rule_fsts = paths.rule_fsts.c_str();
    config.max_num_sentences = 1;
    config.silence_scale = 0.2f;
    return config;
}

int ParsePositiveInt(const std::string &value, const char *option) {
    if (value.empty() || std::any_of(value.begin(), value.end(),
                                     [](unsigned char c) { return !std::isdigit(c); })) {
        throw std::invalid_argument(std::string(option) + " requires a positive integer");
    }
    const long parsed = std::stol(value);
    if (parsed < 1 || parsed > 1024) {
        throw std::invalid_argument(std::string(option) + " must be in the range 1..1024");
    }
    return static_cast<int>(parsed);
}

float ParseSpeed(const std::string &value) {
    size_t consumed = 0;
    const float speed = std::stof(value, &consumed);
    if (consumed != value.size() || speed <= 0.0f) {
        throw std::invalid_argument("--speed requires a positive number");
    }
    return speed;
}

std::string ValidateAffinity(const std::string &value, int expected_threads) {
    if (value.empty()) {
        throw std::invalid_argument("--ep-affinity cannot be empty");
    }

    std::set<int> ids;
    size_t start = 0;
    while (start <= value.size()) {
        const size_t end = value.find(';', start);
        const std::string item = value.substr(
            start, end == std::string::npos ? std::string::npos : end - start);
        if (item.empty() || std::any_of(item.begin(), item.end(),
                                        [](unsigned char c) { return !std::isdigit(c); })) {
            throw std::invalid_argument(
                "--ep-affinity must be a semicolon-separated list such as 8;9");
        }
        const long parsed = std::stol(item);
        if (parsed < 0 || parsed > 1023) {
            throw std::invalid_argument("--ep-affinity CPU id must be in the range 0..1023");
        }
        if (!ids.insert(static_cast<int>(parsed)).second) {
            throw std::invalid_argument("--ep-affinity cannot contain duplicate CPU ids");
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }

    if (static_cast<int>(ids.size()) != expected_threads) {
        throw std::invalid_argument(
            "--ep-affinity CPU count must equal --ep-threads");
    }
    return value;
}

std::string DefaultAffinity(int ep_threads) {
    if (ep_threads > 8) {
        throw std::invalid_argument(
            "default A100 affinity supports at most 8 threads; pass --no-affinity or --ep-affinity");
    }
    std::string value;
    for (int i = 0; i < ep_threads; ++i) {
        if (!value.empty()) {
            value += ';';
        }
        value += std::to_string(8 + i);
    }
    return value;
}

void PrintUsage(const char *program) {
    std::cout
        << "用法: " << program << " [选项]\n\n"
        << "交互式句子级流式 Matcha TTS：每句完整生成后立即播放，不保存 WAV。\n\n"
        << "选项:\n"
        << "  --model-dir PATH       Matcha 模型目录\n"
        << "  --ep-threads N         SpacemiT EP 工作线程数（默认: 2）\n"
        << "  --ep-affinity LIST     EP 线程核列表，如 8;9（默认: 8;9）\n"
        << "  --no-affinity          不设置 EP 专用核绑定\n"
        << "  --warmup               启动后预热（默认开启）\n"
        << "  --no-warmup            关闭启动预热\n"
        << "  --warmup-text TEXT     预热文本（默认: 你好。）\n"
        << "  --speed N              语速倍率（默认: 1.0）\n"
        << "  --chunk-target N       优先切分长度（默认: 70 个 UTF-8 字符）\n"
        << "  --chunk-max N          最大片段长度（默认: 120 个 UTF-8 字符）\n"
        << "  --audio-player PATH    原始 PCM 播放器（默认: aplay）\n"
        << "  --no-play              只合成，不播放\n"
        << "  --threads N            --ep-threads 的兼容别名\n"
        << "  -h, --help             显示帮助\n\n"
        << "示例:\n"
        << "  " << program << "\n"
        << "  " << program << " --ep-threads 8 --ep-affinity '8;9;10;11;12;13;14;15'\n"
        << "  " << program << " --no-affinity --no-play\n";
}

Options ParseOptions(int argc, char **argv, const std::string &default_model_dir) {
    Options options;
    options.model_dir = default_model_dir;
    bool affinity_explicit = false;

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto require_value = [&](const char *option) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(option) + " requires a value");
            }
            return argv[++i];
        };

        if (argument == "-h" || argument == "--help") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else if (argument == "--model-dir") {
            options.model_dir = require_value("--model-dir");
        } else if (argument == "--ep-threads" || argument == "--threads") {
            options.ep_threads = ParsePositiveInt(
                require_value(argument.c_str()), argument.c_str());
        } else if (argument == "--ep-affinity") {
            options.ep_affinity = require_value("--ep-affinity");
            affinity_explicit = true;
        } else if (argument == "--no-affinity") {
            options.enable_affinity = false;
        } else if (argument == "--warmup") {
            options.enable_warmup = true;
        } else if (argument == "--no-warmup") {
            options.enable_warmup = false;
        } else if (argument == "--warmup-text") {
            options.warmup_text = require_value("--warmup-text");
        } else if (argument == "--speed") {
            options.speed = ParseSpeed(require_value("--speed"));
        } else if (argument == "--chunk-target") {
            options.chunk_target = ParsePositiveInt(
                require_value("--chunk-target"), "--chunk-target");
        } else if (argument == "--chunk-max") {
            options.chunk_max = ParsePositiveInt(
                require_value("--chunk-max"), "--chunk-max");
        } else if (argument == "--audio-player") {
            options.audio_player = require_value("--audio-player");
        } else if (argument == "--no-play") {
            options.enable_playback = false;
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }

    if (options.enable_affinity) {
        if (!affinity_explicit) {
            options.ep_affinity = DefaultAffinity(options.ep_threads);
        }
        options.ep_affinity = ValidateAffinity(options.ep_affinity, options.ep_threads);
    }
    if (options.chunk_target > options.chunk_max) {
        throw std::invalid_argument("--chunk-target cannot exceed --chunk-max");
    }
    if (options.warmup_text.empty()) {
        throw std::invalid_argument("--warmup-text cannot be empty");
    }
    if (options.audio_player.empty() && options.enable_playback) {
        throw std::invalid_argument("--audio-player cannot be empty when playback is enabled");
    }
    return options;
}

void ConfigureSpacemiTEp(const Options &options) {
    // The user-facing interface is CLI-only. These internal variables are set
    // before Sherpa creates its sessions because this C API exposes no EP
    // options structure for affinity.
    const std::string thread_count = std::to_string(options.ep_threads);
    if (setenv("SPACEMIT_EP_INTRA_THREAD_NUM", thread_count.c_str(), 1) != 0) {
        throw std::runtime_error("failed to set SPACEMIT_EP_INTRA_THREAD_NUM");
    }
    if (options.enable_affinity) {
        if (setenv("SPACEMIT_EP_INTRA_THREAD_AFFINITY",
                   options.ep_affinity.c_str(), 1) != 0) {
            throw std::runtime_error(
                "failed to set SPACEMIT_EP_INTRA_THREAD_AFFINITY");
        }
    } else {
        unsetenv("SPACEMIT_EP_INTRA_THREAD_AFFINITY");
    }
}

bool IsAsciiDigit(const std::string &character) {
    return character.size() == 1 &&
        std::isdigit(static_cast<unsigned char>(character[0]));
}

bool IsAsciiTokenCharacter(const std::string &character) {
    if (character.size() != 1) {
        return false;
    }
    const unsigned char value = static_cast<unsigned char>(character[0]);
    return std::isalnum(value) || std::string("._:/@%+-").find(character[0]) !=
        std::string::npos;
}

std::string LowerAscii(std::string value) {
    for (char &character : value) {
        character = static_cast<char>(std::tolower(
            static_cast<unsigned char>(character)));
    }
    return value;
}

std::string AsciiTokenAround(const std::vector<std::string> &characters,
                             size_t index) {
    size_t begin = index;
    while (begin > 0 && IsAsciiTokenCharacter(characters[begin - 1])) {
        --begin;
    }
    size_t end = index + 1;
    while (end < characters.size() && IsAsciiTokenCharacter(characters[end])) {
        ++end;
    }

    std::string token;
    for (size_t i = begin; i < end; ++i) {
        token += characters[i];
    }
    return token;
}

bool IsProtectedPeriod(const std::vector<std::string> &characters, size_t index) {
    if (index == 0 || index + 1 >= characters.size()) {
        return false;
    }

    const std::string &previous = characters[index - 1];
    const std::string &next = characters[index + 1];
    if (IsAsciiDigit(previous) && IsAsciiDigit(next)) {
        return true;
    }

    const std::string token = AsciiTokenAround(characters, index);
    if (token.find("@") != std::string::npos ||
        token.find("://") != std::string::npos) {
        return true;
    }

    std::string compact;
    for (const char character : token) {
        if (character != '.') {
            compact += character;
        }
    }
    compact = LowerAscii(compact);
    static const std::set<std::string> abbreviations = {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs",
        "etc", "eg", "ie", "us", "uk", "no"
    };
    return abbreviations.count(compact) != 0;
}


bool IsProtectedSoftPunctuation(
    const std::vector<std::string> &characters, size_t index) {
    if (index == 0 || index + 1 >= characters.size()) {
        return false;
    }
    const std::string &character = characters[index];
    const std::string &previous = characters[index - 1];
    const std::string &next = characters[index + 1];
    if ((character == ":" || character == ",") &&
        IsAsciiDigit(previous) && IsAsciiDigit(next)) {
        return true;
    }

    const std::string token = AsciiTokenAround(characters, index);
    return token.find("@") != std::string::npos ||
        token.find("://") != std::string::npos;
}

std::vector<std::string> Utf8Characters(const std::string &text) {
    std::vector<std::string> characters;
    for (size_t index = 0; index < text.size();) {
        const unsigned char first = static_cast<unsigned char>(text[index]);
        size_t length = 1;
        if ((first & 0x80) == 0) {
            length = 1;
        } else if ((first & 0xE0) == 0xC0) {
            length = 2;
        } else if ((first & 0xF0) == 0xE0) {
            length = 3;
        } else if ((first & 0xF8) == 0xF0) {
            length = 4;
        }
        if (index + length > text.size()) {
            length = 1;
        }
        characters.push_back(text.substr(index, length));
        index += length;
    }
    return characters;
}

bool IsOpeningQuote(const std::string &character) {
    return character == "“" || character == "‘" || character == "«" ||
        character == "‹" || character == "「" || character == "『";
}

bool IsClosingQuote(const std::string &character) {
    return character == "”" || character == "’" || character == "»" ||
        character == "›" || character == "」" || character == "』";
}

bool IsOpeningBracket(const std::string &character) {
    return character == "（" || character == "(" || character == "【" ||
        character == "[" || character == "{" || character == "《";
}

bool IsClosingBracket(const std::string &character) {
    return character == "）" || character == ")" || character == "】" ||
        character == "]" || character == "}" || character == "》";
}

bool IsMatchingDelimiter(const std::string &opening,
                         const std::string &closing) {
    return (opening == "（" && closing == "）") ||
        (opening == "(" && closing == ")") ||
        (opening == "【" && closing == "】") ||
        (opening == "[" && closing == "]") ||
        (opening == "{" && closing == "}") ||
        (opening == "《" && closing == "》") ||
        (IsOpeningQuote(opening) && IsClosingQuote(closing));
}

bool IsStrongPunctuation(const std::string &character) {
    return character == "。" || character == "！" || character == "？" ||
        character == "；" || character == "!" || character == "?" ||
        character == ";" || (character == ".");
}

bool IsSoftPunctuation(const std::string &character) {
    return character == "，" || character == "," || character == "、" ||
        character == "：" || character == ":" || character == "—" ||
        character == "–" || character == "…";
}

bool IsClosingDelimiter(const std::string &character) {
    return IsClosingQuote(character) || IsClosingBracket(character) ||
        character == "\"";
}

std::string TrimAsciiWhitespace(std::string value) {
    auto is_space = [](char character) {
        return std::isspace(static_cast<unsigned char>(character)) != 0;
    };
    while (!value.empty() && is_space(value.front())) {
        value.erase(value.begin());
    }
    while (!value.empty() && is_space(value.back())) {
        value.pop_back();
    }
    return value;
}

struct SoftBoundary {
    size_t position;
    int priority;
};

bool IsQuoteOpeningDelimiter(const std::string &character) {
    return IsOpeningQuote(character);
}

bool IsInsideQuote(const std::vector<std::string> &delimiter_stack,
                  bool ascii_double_quote_open) {
    if (ascii_double_quote_open) {
        return true;
    }
    return std::any_of(delimiter_stack.begin(), delimiter_stack.end(),
                       [](const std::string &delimiter) {
                           return IsQuoteOpeningDelimiter(delimiter);
                       });
}

std::vector<std::string> SplitSentences(const std::string &text,
                                        int target_length,
                                        int max_length) {
    const std::vector<std::string> characters = Utf8Characters(text);
    std::vector<std::string> sentences;
    std::vector<std::string> current;
    std::vector<SoftBoundary> soft_boundaries;
    std::vector<size_t> whitespace_boundaries;
    std::vector<std::string> delimiter_stack;
    bool ascii_double_quote_open = false;
    bool pending_quote_sentence = false;

    auto append_current = [&]() {
        std::string sentence;
        for (const auto &character : current) {
            sentence += character;
        }
        sentence = TrimAsciiWhitespace(std::move(sentence));
        if (!sentence.empty()) {
            sentences.push_back(std::move(sentence));
        }
        current.clear();
        soft_boundaries.clear();
        whitespace_boundaries.clear();
        pending_quote_sentence = false;
    };
    auto split_at = [&](size_t boundary) {
        if (boundary == 0 || boundary > current.size()) {
            return false;
        }
        std::string sentence;
        for (size_t i = 0; i < boundary; ++i) {
            sentence += current[i];
        }
        sentence = TrimAsciiWhitespace(std::move(sentence));
        if (!sentence.empty()) {
            sentences.push_back(std::move(sentence));
        }
        std::vector<std::string> remainder(
            current.begin() + static_cast<std::ptrdiff_t>(boundary), current.end());
        current = std::move(remainder);
        soft_boundaries.clear();
        whitespace_boundaries.clear();
        pending_quote_sentence = false;
        return true;
    };
    auto update_delimiters = [&](const std::string &character) {
        if (IsOpeningQuote(character) || IsOpeningBracket(character)) {
            delimiter_stack.push_back(character);
        } else if (IsClosingQuote(character) || IsClosingBracket(character)) {
            if (!delimiter_stack.empty() &&
                IsMatchingDelimiter(delimiter_stack.back(), character)) {
                delimiter_stack.pop_back();
            }
        } else if (character == "\"") {
            ascii_double_quote_open = !ascii_double_quote_open;
        }
    };
    auto append_boundary_tail = [&](size_t &index) {
        while (index + 1 < characters.size()) {
            const size_t next_index = index + 1;
            const std::string &next = characters[next_index];
            const bool next_period_protected = next == "." &&
                IsProtectedPeriod(characters, next_index);
            if (!IsClosingDelimiter(next) &&
                !(IsStrongPunctuation(next) && !next_period_protected)) {
                break;
            }
            ++index;
            current.push_back(next);
            update_delimiters(next);
        }
    };
    auto choose_soft_boundary = [&](size_t minimum_position) -> size_t {
        if (soft_boundaries.empty()) {
            return 0;
        }
        const SoftBoundary *best = nullptr;
        for (const auto &candidate : soft_boundaries) {
            if (candidate.position < minimum_position) {
                continue;
            }
            if (best == nullptr || candidate.position > best->position ||
                (candidate.position == best->position &&
                 candidate.priority > best->priority)) {
                best = &candidate;
            }
        }
        return best == nullptr ? 0 : best->position;
    };

    for (size_t index = 0; index < characters.size(); ++index) {
        const std::string &character = characters[index];
        if (character == "\n" || character == "\r") {
            append_current();
            delimiter_stack.clear();
            ascii_double_quote_open = false;
            continue;
        }

        const bool protected_before = !delimiter_stack.empty() ||
            ascii_double_quote_open;
        const bool quote_before = IsInsideQuote(
            delimiter_stack, ascii_double_quote_open);
        current.push_back(character);
        update_delimiters(character);

        const bool period_protected = character == "." &&
            IsProtectedPeriod(characters, index);
        const bool strong = !period_protected && IsStrongPunctuation(character);

        if (strong) {
            if (!protected_before) {
                // Keep runs such as "？！" together, including closing quote
                // or bracket characters immediately following the terminator.
                while (index + 1 < characters.size() &&
                       IsStrongPunctuation(characters[index + 1])) {
                    ++index;
                    current.push_back(characters[index]);
                }
                append_boundary_tail(index);
                append_current();
                continue;
            }
            if (quote_before) {
                // A terminator inside dialogue becomes a boundary once its
                // closing quote arrives, e.g. “快跑！”然后……
                pending_quote_sentence = true;
            }
        }

        const bool soft_protected = IsProtectedSoftPunctuation(characters, index);
        if (IsSoftPunctuation(character) && !soft_protected &&
            !protected_before) {
            int priority = 2;
            if (character == "：" || character == ":") {
                priority = 1;
            } else if (character == "，" || character == "," ||
                       character == "、") {
                priority = 3;
            }
            soft_boundaries.push_back({current.size(), priority});
        }
        if (character == " " || character == "\t") {
            whitespace_boundaries.push_back(current.size());
        }

        const bool closes_ascii_quote = character == "\"" &&
            !ascii_double_quote_open;
        if (pending_quote_sentence &&
            (IsClosingDelimiter(character) || closes_ascii_quote) &&
            !IsInsideQuote(delimiter_stack, ascii_double_quote_open)) {
            append_boundary_tail(index);
            append_current();
            continue;
        }

        if (!protected_before && static_cast<int>(current.size()) >= target_length) {
            const size_t boundary = choose_soft_boundary(
                std::max<size_t>(1, static_cast<size_t>(target_length * 55 / 100)));
            if (boundary != 0) {
                split_at(boundary);
                continue;
            }
        }

        if (static_cast<int>(current.size()) > max_length) {
            const size_t soft_boundary = choose_soft_boundary(
                current.size() > static_cast<size_t>(target_length / 2)
                    ? current.size() - static_cast<size_t>(target_length / 2)
                    : 1);
            if (soft_boundary != 0) {
                split_at(soft_boundary);
            } else if (!whitespace_boundaries.empty()) {
                split_at(whitespace_boundaries.back());
            } else {
                split_at(static_cast<size_t>(max_length));
            }
        }
    }

    append_current();
    return sentences;
}

std::vector<int16_t> ConvertToInt16(const float *samples, int32_t count) {
    std::vector<int16_t> result(static_cast<size_t>(count));
    for (int32_t i = 0; i < count; ++i) {
        float sample = std::max(-1.0f, std::min(1.0f, samples[i]));
        result[static_cast<size_t>(i)] = static_cast<int16_t>(sample * 32767.0f);
    }
    return result;
}

std::string Basename(const std::string &path) {
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

class RawAudioPlayer {
public:
    explicit RawAudioPlayer(std::string command)
        : command_(std::move(command)) {}

    ~RawAudioPlayer() { Close(); }

    bool Start(int sample_rate) {
        int pipe_fds[2];
        if (pipe(pipe_fds) != 0) {
            std::cerr << "[播放] 创建音频管道失败: " << std::strerror(errno)
                      << "\n";
            return false;
        }

        child_pid_ = fork();
        if (child_pid_ < 0) {
            std::cerr << "[播放] 创建播放器进程失败: " << std::strerror(errno)
                      << "\n";
            close(pipe_fds[0]);
            close(pipe_fds[1]);
            return false;
        }

        if (child_pid_ == 0) {
            close(pipe_fds[1]);
            if (dup2(pipe_fds[0], STDIN_FILENO) < 0) {
                _exit(126);
            }
            close(pipe_fds[0]);

            const std::string name = Basename(command_);
            const std::string rate = std::to_string(sample_rate);
            if (name == "paplay") {
                execlp(command_.c_str(), command_.c_str(), "--raw",
                       "--format=s16le", "--rate", rate.c_str(), "--channels",
                       "1", nullptr);
            } else if (name == "ffplay") {
                execlp(command_.c_str(), command_.c_str(), "-nodisp", "-autoexit",
                       "-loglevel", "error", "-f", "s16le", "-ar", rate.c_str(),
                       "-ac", "1", "-i", "-", nullptr);
            } else {
                // aplay is the preferred board-side ALSA player.
                execlp(command_.c_str(), command_.c_str(), "-q", "-t", "raw", "-f",
                       "S16_LE", "-c", "1", "-r", rate.c_str(), "-", nullptr);
            }
            _exit(127);
        }

        close(pipe_fds[0]);
        write_fd_ = pipe_fds[1];
        sample_rate_ = sample_rate;
        return true;
    }

    bool Write(const std::vector<int16_t> &samples) {
        if (write_fd_ < 0) {
            return false;
        }

        const char *data = reinterpret_cast<const char *>(samples.data());
        size_t remaining = samples.size() * sizeof(int16_t);
        while (remaining > 0) {
            const ssize_t written = write(write_fd_, data, remaining);
            if (written < 0) {
                if (errno == EINTR) {
                    continue;
                }
                std::cerr << "[播放] 写入音频失败: " << std::strerror(errno)
                          << "\n";
                return false;
            }
            data += written;
            remaining -= static_cast<size_t>(written);
        }
        return true;
    }

    void Close() {
        if (write_fd_ >= 0) {
            close(write_fd_);
            write_fd_ = -1;
        }
        if (child_pid_ > 0) {
            int status = 0;
            while (waitpid(child_pid_, &status, 0) < 0 && errno == EINTR) {
            }
            if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
                std::cerr << "[播放] 播放器退出码: " << WEXITSTATUS(status)
                          << "；请检查声卡/默认输出设备\n";
            } else if (WIFSIGNALED(status)) {
                std::cerr << "[播放] 播放器被信号 " << WTERMSIG(status)
                          << " 终止\n";
            }
            child_pid_ = -1;
        }
    }

private:
    std::string command_;
    pid_t child_pid_ = -1;
    int write_fd_ = -1;
    int sample_rate_ = 0;
};

void SynthesisThread(const SherpaOnnxOfflineTts *tts,
                     const std::vector<std::string> &sentences,
                     const SherpaOnnxGenerationConfig &generation,
                     AudioQueue &audio_queue) {
    std::cerr << "[合成] 共 " << sentences.size()
              << " 句；每句完整生成后立即交给播放线程\n";

    for (size_t i = 0; i < sentences.size() && !g_stop.load(); ++i) {
        std::cerr << "[合成] 第 " << (i + 1) << " 段文本: " << sentences[i] << "\n";
        const auto begin = std::chrono::steady_clock::now();
        const SherpaOnnxGeneratedAudio *audio =
            SherpaOnnxOfflineTtsGenerateWithConfig(
                tts, sentences[i].c_str(), &generation, nullptr, nullptr);
        const auto end = std::chrono::steady_clock::now();

        if (!audio || !audio->samples || audio->n <= 0 || audio->sample_rate <= 0) {
            std::cerr << "[合成] 第 " << (i + 1) << " 句失败\n";
            if (audio) {
                SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
            }
            continue;
        }

        const double elapsed = std::chrono::duration<double>(end - begin).count();
        const double audio_seconds =
            static_cast<double>(audio->n) / audio->sample_rate;
        std::cerr << "[合成] 第 " << (i + 1) << " 句完成："
                  << audio_seconds << " 秒音频，耗时 " << elapsed
                  << " 秒，RTF=" << elapsed / audio_seconds << "\n";

        AudioChunk chunk;
        chunk.samples = ConvertToInt16(audio->samples, audio->n);
        chunk.sample_rate = audio->sample_rate;
        chunk.sentence_index = static_cast<int>(i + 1);
        audio_queue.Push(std::move(chunk));
        SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
    }

    audio_queue.Finish();
    std::cerr << "[合成] 当前输入完成\n";
}

void PlaybackThread(AudioQueue &audio_queue, const std::string &player_command,
                    bool enable_playback) {
    RawAudioPlayer player(player_command);
    bool player_started = false;
    bool player_failed = false;
    int played_sentences = 0;

    AudioChunk chunk;
    while (audio_queue.Pop(chunk)) {
        if (!enable_playback) {
            continue;
        }

        if (!player_started && !player_failed) {
            if (!player.Start(chunk.sample_rate)) {
                player_failed = true;
                std::cerr << "[播放] 无法启动播放器 '" << player_command
                          << "'；请使用 --no-play 只合成不播放\n";
                continue;
            }
            player_started = true;
            std::cerr << "[播放] 已开始实时播放（句子 "
                      << chunk.sentence_index << "）\n";
        }

        if (player_started && player.Write(chunk.samples)) {
            ++played_sentences;
        } else if (player_started) {
            player_failed = true;
            player.Close();
        }
    }

    player.Close();
    if (enable_playback && played_sentences > 0) {
        std::cerr << "[播放] 播放完成，共 " << played_sentences << " 句\n";
    }
}

int RunInteractive(const ModelPaths &paths, const Options &options) {
    ConfigureSpacemiTEp(options);
    const auto config = MakeConfig(paths, options.ep_threads);
    std::cerr << "[C++] 创建常驻 Matcha TTS...\n";
    const auto init_begin = std::chrono::steady_clock::now();
    const SherpaOnnxOfflineTts *tts = SherpaOnnxCreateOfflineTts(&config);
    const auto init_end = std::chrono::steady_clock::now();
    if (!tts) {
        std::cerr << "[C++] 创建 TTS 失败\n";
        return 1;
    }

    std::cerr << "[C++] 模型已加载，初始化耗时 "
              << std::chrono::duration<double>(init_end - init_begin).count()
              << " 秒\n";

    SherpaOnnxGenerationConfig generation{};
    generation.sid = 0;
    generation.speed = options.speed;
    generation.silence_scale = 0.2f;

    std::cerr << "[C++] 句子级流式模式：按标点切句，生成一整句后立即播放；不保存 WAV\n";
    std::cerr << "[C++] EP 线程数: " << options.ep_threads << "\n";
    std::cerr << "[C++] 切句长度: target=" << options.chunk_target
              << ", max=" << options.chunk_max << " UTF-8 字符\n";
    std::cerr << "[C++] EP affinity: "
              << (options.enable_affinity ? options.ep_affinity : "disabled") << "\n";
    std::cerr << "[C++] 预热: " << (options.enable_warmup ? "enabled" : "disabled")
              << (options.enable_warmup ? " (\"" + options.warmup_text + "\")" : "")
              << "\n";
    std::cerr << "[C++] 播放器: "
              << (options.enable_playback ? options.audio_player : "disabled") << "\n";

    if (options.enable_warmup) {
        const auto warmup_begin = std::chrono::steady_clock::now();
        const SherpaOnnxGeneratedAudio *warmup_audio =
            SherpaOnnxOfflineTtsGenerateWithConfig(
                tts, options.warmup_text.c_str(), &generation, nullptr, nullptr);
        const auto warmup_end = std::chrono::steady_clock::now();
        if (!warmup_audio || !warmup_audio->samples || warmup_audio->n <= 0) {
            std::cerr << "[C++] 预热失败；继续进入交互模式\n";
        } else {
            const double elapsed =
                std::chrono::duration<double>(warmup_end - warmup_begin).count();
            const double audio_seconds =
                static_cast<double>(warmup_audio->n) / warmup_audio->sample_rate;
            std::cerr << "[C++] 预热完成：" << audio_seconds
                      << " 秒音频，耗时 " << elapsed
                      << " 秒，RTF=" << elapsed / audio_seconds << "\n";
        }
        if (warmup_audio) {
            SherpaOnnxDestroyOfflineTtsGeneratedAudio(warmup_audio);
        }
    }

    std::cerr << "[C++] 输入文字后回车；:q 或 Ctrl-D 退出\n";

    std::string text;
    while (!g_stop.load()) {
        std::cout << "\nTTS> " << std::flush;
        if (!std::getline(std::cin, text)) {
            break;
        }
        if (text.empty()) {
            continue;
        }
        if (text == ":q" || text == ":quit" || text == ":exit") {
            break;
        }

        const std::vector<std::string> sentences = SplitSentences(
            text, options.chunk_target, options.chunk_max);
        if (sentences.empty()) {
            continue;
        }
        std::cerr << "[C++] 当前输入切分为 " << sentences.size()
                  << " 段（target=" << options.chunk_target
                  << ", max=" << options.chunk_max << "）\n";

        AudioQueue audio_queue;
        std::thread playback_thread(PlaybackThread, std::ref(audio_queue),
                                     options.audio_player, options.enable_playback);
        std::thread synthesis_thread(SynthesisThread, tts, std::cref(sentences),
                                     std::cref(generation), std::ref(audio_queue));
        synthesis_thread.join();
        playback_thread.join();
    }

    SherpaOnnxDestroyOfflineTts(tts);
    std::cerr << "\n[C++] TTS 已退出\n";
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    std::signal(SIGINT, OnSignal);
    std::signal(SIGTERM, OnSignal);
    std::signal(SIGPIPE, SIG_IGN);

    // Resolve the default model directory relative to this binary so the
    // migrated tree is self-contained (build-cpp/../matcha-model).
    std::string default_model_dir = "matcha-model";
    {
        char buffer[4096];
        const ssize_t length = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
        if (length > 0) {
            buffer[length] = '\0';
            const std::string exe(buffer);
            const size_t slash = exe.find_last_of('/');
            if (slash != std::string::npos) {
                default_model_dir = exe.substr(0, slash) + "/../matcha-model";
            }
        }
    }

    try {
        const Options options = ParseOptions(argc, argv, default_model_dir);
        ModelPaths paths;
        paths.acoustic = options.model_dir + "/model-steps-3.q.onnx";
        paths.vocoder = options.model_dir + "/vocos-16khz-univ.q.onnx";
        paths.tokens = options.model_dir + "/tokens.txt";
        paths.lexicon = options.model_dir + "/lexicon.txt";
        paths.data_dir = options.model_dir + "/espeak-ng-data";
        paths.rule_fsts = options.model_dir + "/date-zh.fst," +
            options.model_dir + "/number-zh.fst";
        return RunInteractive(paths, options);
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n"
                  << "use --help to show available options\n";
        return 2;
    }
}
