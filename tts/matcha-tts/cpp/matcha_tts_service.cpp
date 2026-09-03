// Resident Matcha-TTS service for the Dice Arena backend (tts_matcha).
//
// Process model: one long-lived child owned by the Python provider. The
// provider writes one request per stdin line and reads JSONL events from
// stdout; stderr stays human-readable diagnostics. Audio never touches the
// board speaker — WAV frames are streamed to the browser by the backend.
//
// Request line (TSV, exactly three fields; the text field may contain tabs):
//   <id>\t<speed>\t<text>
// Event stream (one JSON object per stdout line, flushed per event):
//   {"event":"ready","sample_rate":16000,"voice":"0",...}   after warmup
//   {"event":"sentence","id":"..","seq":1,"text":".."}      before each synth
//   {"event":"audio","id":"..","seq":1,"duration_seconds":..,"wav_b64":".."}
//   {"event":"done","id":"..","sentences":N,"audio_seconds":..,"elapsed_seconds":..}
//   {"event":"error","id":"..","message":".."}              request-level failure
//
// Warmup is mandatory: the "ready" event is only emitted after one synthesis
// completes, and a failed warmup exits non-zero so the backend refuses to
// start with a broken local TTS engine.
//
// Orphan protection: a watchdog thread polls getppid() and exits once the
// backend process is gone (the child is reparented to init). This is
// deliberately thread-level-safe, unlike PR_SET_PDEATHSIG which would fire
// if the spawning parent *thread* exits while the backend still runs.
#include <sherpa-onnx/c-api/c-api.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <sys/types.h>
#include <unistd.h>

namespace {

// ---- options ---------------------------------------------------------------

struct Options {
    std::string model_dir;
    std::string ep_affinity;
    std::string warmup_text = "你好。";
    int ep_threads = 2;
    int chunk_target = 70;
    int chunk_max = 120;
    bool enable_affinity = true;
};

struct ModelPaths {
    std::string acoustic;
    std::string vocoder;
    std::string tokens;
    std::string lexicon;
    std::string data_dir;
    std::string rule_fsts;
};

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
            throw std::invalid_argument(
                "--ep-affinity CPU id must be in the range 0..1023");
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
        << "常驻 Matcha TTS 服务：stdin 收请求行，stdout 出 JSONL 事件。\n"
        << "请求行格式: <id>\\t<speed>\\t<text>（text 为行内剩余内容）\n\n"
        << "选项:\n"
        << "  --model-dir PATH       Matcha 模型目录（默认: 二进制旁 ../matcha-model）\n"
        << "  --ep-threads N         SpacemiT EP 工作线程数（默认: 2）\n"
        << "  --ep-affinity LIST     EP 线程核列表，如 8;9（默认: 8;9…）\n"
        << "  --no-affinity          不设置 EP 专用核绑定\n"
        << "  --warmup-text TEXT     预热文本（默认: 你好。；预热失败进程退出非 0）\n"
        << "  --chunk-target N       优先切分长度（默认: 70 个 UTF-8 字符）\n"
        << "  --chunk-max N          最大片段长度（默认: 120 个 UTF-8 字符）\n"
        << "  -h, --help             显示帮助\n";
}

std::string DefaultModelDir() {
    char buffer[4096];
    const ssize_t length = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (length > 0) {
        buffer[length] = '\0';
        const std::string exe(buffer);
        const size_t slash = exe.find_last_of('/');
        if (slash != std::string::npos) {
            return exe.substr(0, slash) + "/../matcha-model";
        }
    }
    return "matcha-model";
}

Options ParseOptions(int argc, char **argv) {
    Options options;
    options.model_dir = DefaultModelDir();
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
        } else if (argument == "--warmup-text") {
            options.warmup_text = require_value("--warmup-text");
        } else if (argument == "--chunk-target") {
            options.chunk_target = ParsePositiveInt(
                require_value("--chunk-target"), "--chunk-target");
        } else if (argument == "--chunk-max") {
            options.chunk_max = ParsePositiveInt(
                require_value("--chunk-max"), "--chunk-max");
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

// ---- sentence splitting (same layered policy as the interactive tool) ------

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

// ---- audio encoding --------------------------------------------------------

std::vector<int16_t> ConvertToInt16(const float *samples, int32_t count) {
    std::vector<int16_t> result(static_cast<size_t>(count));
    for (int32_t i = 0; i < count; ++i) {
        float sample = std::max(-1.0f, std::min(1.0f, samples[i]));
        result[static_cast<size_t>(i)] = static_cast<int16_t>(sample * 32767.0f);
    }
    return result;
}

void AppendLe32(std::vector<uint8_t> &out, uint32_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xFF));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((value >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((value >> 24) & 0xFF));
}

void AppendLe16(std::vector<uint8_t> &out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xFF));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
}

// Canonical 44-byte RIFF/WAVE header + mono 16-bit PCM payload.
std::vector<uint8_t> BuildWav(const std::vector<int16_t> &samples,
                              int32_t sample_rate) {
    const uint32_t data_size =
        static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    std::vector<uint8_t> wav;
    wav.reserve(44 + data_size);

    const std::string riff = "RIFF";
    wav.insert(wav.end(), riff.begin(), riff.end());
    AppendLe32(wav, 36 + data_size);
    const std::string wave = "WAVE";
    wav.insert(wav.end(), wave.begin(), wave.end());
    const std::string fmt = "fmt ";
    wav.insert(wav.end(), fmt.begin(), fmt.end());
    AppendLe32(wav, 16);          // PCM chunk size
    AppendLe16(wav, 1);           // audio format: PCM
    AppendLe16(wav, 1);           // channels: mono
    AppendLe32(wav, static_cast<uint32_t>(sample_rate));
    AppendLe32(wav, static_cast<uint32_t>(sample_rate) * 2);  // byte rate
    AppendLe16(wav, 2);           // block align
    AppendLe16(wav, 16);          // bits per sample
    const std::string data = "data";
    wav.insert(wav.end(), data.begin(), data.end());
    AppendLe32(wav, data_size);

    const auto *bytes = reinterpret_cast<const uint8_t *>(samples.data());
    wav.insert(wav.end(), bytes, bytes + data_size);
    return wav;
}

const char kBase64Alphabet[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string Base64Encode(const std::vector<uint8_t> &bytes) {
    std::string out;
    out.reserve(((bytes.size() + 2) / 3) * 4);
    size_t i = 0;
    while (i + 3 <= bytes.size()) {
        const uint32_t triple = (static_cast<uint32_t>(bytes[i]) << 16) |
            (static_cast<uint32_t>(bytes[i + 1]) << 8) |
            static_cast<uint32_t>(bytes[i + 2]);
        out += kBase64Alphabet[(triple >> 18) & 0x3F];
        out += kBase64Alphabet[(triple >> 12) & 0x3F];
        out += kBase64Alphabet[(triple >> 6) & 0x3F];
        out += kBase64Alphabet[triple & 0x3F];
        i += 3;
    }
    const size_t remaining = bytes.size() - i;
    if (remaining == 1) {
        const uint32_t triple = static_cast<uint32_t>(bytes[i]) << 16;
        out += kBase64Alphabet[(triple >> 18) & 0x3F];
        out += kBase64Alphabet[(triple >> 12) & 0x3F];
        out += "==";
    } else if (remaining == 2) {
        const uint32_t triple = (static_cast<uint32_t>(bytes[i]) << 16) |
            (static_cast<uint32_t>(bytes[i + 1]) << 8);
        out += kBase64Alphabet[(triple >> 18) & 0x3F];
        out += kBase64Alphabet[(triple >> 12) & 0x3F];
        out += kBase64Alphabet[(triple >> 6) & 0x3F];
        out += '=';
    }
    return out;
}

// ---- event output ----------------------------------------------------------

std::string JsonEscape(const std::string &value) {
    static const char *kHex = "0123456789abcdef";
    std::string out;
    out.reserve(value.size() + 8);
    for (const unsigned char character : value) {
        switch (character) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (character < 0x20) {
                    out += "\\u00";
                    out += kHex[(character >> 4) & 0xF];
                    out += kHex[character & 0xF];
                } else {
                    // Raw UTF-8 bytes pass through; the consumer decodes UTF-8.
                    out += static_cast<char>(character);
                }
        }
    }
    return out;
}

// Emits one JSONL event on stdout. A write failure means the consumer is
// gone; exit quietly instead of spinning on a dead pipe.
void EmitEvent(const std::string &event) {
    std::cout << event << "\n" << std::flush;
    if (!std::cout.good()) {
        std::cerr << "[服务] stdout 写入失败，退出\n";
        std::exit(0);
    }
}

void EmitError(const std::string &id, const std::string &message) {
    EmitEvent("{\"event\":\"error\",\"id\":\"" + JsonEscape(id) +
              "\",\"message\":\"" + JsonEscape(message) + "\"}");
}

// ---- request handling ------------------------------------------------------

struct Request {
    std::string id;
    std::string text;
    float speed = 1.0f;
};

// One request per line: "<id>\t<speed>\t<text>"; the text field keeps any
// further tabs. Returns false for malformed lines (reported by the caller).
bool ParseRequestLine(const std::string &line, Request &request) {
    const size_t first_tab = line.find('\t');
    if (first_tab == std::string::npos) {
        return false;
    }
    const size_t second_tab = line.find('\t', first_tab + 1);
    if (second_tab == std::string::npos) {
        return false;
    }

    std::string id = line.substr(0, first_tab);
    const std::string speed_field =
        line.substr(first_tab + 1, second_tab - first_tab - 1);
    std::string text = line.substr(second_tab + 1);
    if (!text.empty() && text.back() == '\r') {
        text.pop_back();
    }
    text = TrimAsciiWhitespace(std::move(text));

    if (id.empty() || id.size() > 64 ||
        !std::all_of(id.begin(), id.end(), [](unsigned char c) {
            return std::isalnum(c) || c == '_' || c == '-' || c == '.';
        })) {
        return false;
    }
    if (speed_field.empty()) {
        return false;
    }
    size_t consumed = 0;
    float speed = 0.0f;
    try {
        speed = std::stof(speed_field, &consumed);
    } catch (const std::exception &) {
        return false;
    }
    if (consumed != speed_field.size() || speed <= 0.0f || speed > 10.0f) {
        return false;
    }
    if (text.empty()) {
        return false;
    }

    request.id = std::move(id);
    request.text = std::move(text);
    request.speed = speed;
    return true;
}

void HandleRequest(const SherpaOnnxOfflineTts *tts, const Options &options,
                   const Request &request) {
    const auto begin = std::chrono::steady_clock::now();
    std::vector<std::string> sentences = SplitSentences(
        request.text, options.chunk_target, options.chunk_max);
    if (sentences.empty()) {
        EmitError(request.id, "text has no speakable content");
        return;
    }

    double total_audio_seconds = 0.0;
    for (size_t i = 0; i < sentences.size(); ++i) {
        EmitEvent("{\"event\":\"sentence\",\"id\":\"" + JsonEscape(request.id) +
                  "\",\"seq\":" + std::to_string(i + 1) + ",\"text\":\"" +
                  JsonEscape(sentences[i]) + "\"}");

        SherpaOnnxGenerationConfig generation{};
        generation.sid = 0;
        generation.speed = request.speed;
        generation.silence_scale = 0.2f;

        const SherpaOnnxGeneratedAudio *audio =
            SherpaOnnxOfflineTtsGenerateWithConfig(
                tts, sentences[i].c_str(), &generation, nullptr, nullptr);
        if (!audio || !audio->samples || audio->n <= 0 || audio->sample_rate <= 0) {
            if (audio) {
                SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
            }
            EmitError(request.id, "sentence " + std::to_string(i + 1) +
                      " synthesis failed");
            return;
        }

        const double audio_seconds =
            static_cast<double>(audio->n) / audio->sample_rate;
        total_audio_seconds += audio_seconds;
        const std::vector<int16_t> samples =
            ConvertToInt16(audio->samples, audio->n);
        const int32_t sample_rate = audio->sample_rate;
        SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);

        const std::string wav_b64 = Base64Encode(BuildWav(samples, sample_rate));
        std::ostringstream event;
        event << "{\"event\":\"audio\",\"id\":\"" << JsonEscape(request.id)
              << "\",\"seq\":" << (i + 1)
              << ",\"sample_rate\":" << sample_rate
              << ",\"duration_seconds\":" << audio_seconds
              << ",\"wav_b64\":\"" << wav_b64 << "\"}";
        EmitEvent(event.str());
    }

    const auto end = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double>(end - begin).count();
    std::ostringstream done;
    done << "{\"event\":\"done\",\"id\":\"" << JsonEscape(request.id)
         << "\",\"sentences\":" << sentences.size()
         << ",\"audio_seconds\":" << total_audio_seconds
         << ",\"elapsed_seconds\":" << elapsed << "}";
    EmitEvent(done.str());
}

// ---- lifecycle -------------------------------------------------------------

// Exits once the backend process disappears (the child gets reparented to
// init). Polling getppid() is correct at process level regardless of which
// parent thread originally spawned us.
void StartParentWatchdog() {
    std::thread([] {
        for (;;) {
            if (getppid() == 1) {
                std::cerr << "[服务] 父进程已退出，服务自行结束\n";
                std::exit(0);
            }
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }).detach();
}

int RunService(const ModelPaths &paths, const Options &options) {
    ConfigureSpacemiTEp(options);

    std::cerr << "[服务] 创建常驻 Matcha TTS（EP 线程 " << options.ep_threads
              << "，affinity "
              << (options.enable_affinity ? options.ep_affinity : "disabled")
              << "）...\n";
    const auto init_begin = std::chrono::steady_clock::now();
    const SherpaOnnxOfflineTtsConfig config = MakeConfig(paths, options.ep_threads);
    const SherpaOnnxOfflineTts *tts = SherpaOnnxCreateOfflineTts(&config);
    const auto init_end = std::chrono::steady_clock::now();
    if (!tts) {
        std::cerr << "[服务] 创建 TTS 失败\n";
        EmitError("", "failed to create the Matcha TTS engine");
        return 1;
    }
    std::cerr << "[服务] 模型已加载，初始化耗时 "
              << std::chrono::duration<double>(init_end - init_begin).count()
              << " 秒\n";

    // Warmup is part of startup: the ready event (and the backend's boot)
    // only happen after one full synthesis completes.
    int32_t ready_sample_rate = 0;
    {
        SherpaOnnxGenerationConfig generation{};
        generation.sid = 0;
        generation.speed = 1.0f;
        generation.silence_scale = 0.2f;
        const auto warmup_begin = std::chrono::steady_clock::now();
        const SherpaOnnxGeneratedAudio *warmup_audio =
            SherpaOnnxOfflineTtsGenerateWithConfig(
                tts, options.warmup_text.c_str(), &generation, nullptr, nullptr);
        const auto warmup_end = std::chrono::steady_clock::now();
        if (!warmup_audio || !warmup_audio->samples || warmup_audio->n <= 0 ||
            warmup_audio->sample_rate <= 0) {
            if (warmup_audio) {
                SherpaOnnxDestroyOfflineTtsGeneratedAudio(warmup_audio);
            }
            std::cerr << "[服务] 预热失败，退出\n";
            EmitError("", "warmup synthesis failed");
            SherpaOnnxDestroyOfflineTts(tts);
            return 1;
        }
        ready_sample_rate = warmup_audio->sample_rate;
        const double elapsed =
            std::chrono::duration<double>(warmup_end - warmup_begin).count();
        const double audio_seconds =
            static_cast<double>(warmup_audio->n) / warmup_audio->sample_rate;
        std::cerr << "[服务] 预热完成：" << audio_seconds << " 秒音频，耗时 "
                  << elapsed << " 秒，RTF=" << elapsed / audio_seconds << "\n";
        SherpaOnnxDestroyOfflineTtsGeneratedAudio(warmup_audio);
    }

    StartParentWatchdog();
    EmitEvent("{\"event\":\"ready\",\"sample_rate\":" +
              std::to_string(ready_sample_rate) +
              ",\"voice\":\"0\",\"ep_threads\":" +
              std::to_string(options.ep_threads) +
              ",\"ep_affinity\":\"" +
              (options.enable_affinity ? JsonEscape(options.ep_affinity) : "") +
              "\",\"chunk_target\":" + std::to_string(options.chunk_target) +
              ",\"chunk_max\":" + std::to_string(options.chunk_max) + "}");
    std::cerr << "[服务] 就绪，stdin 等待请求行：<id>\\t<speed>\\t<text>\n";

    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const std::string trimmed = TrimAsciiWhitespace(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        Request request;
        if (!ParseRequestLine(trimmed, request)) {
            EmitError("", "malformed request line (expected <id>\\t<speed>\\t<text>)");
            continue;
        }
        HandleRequest(tts, options, request);
    }

    SherpaOnnxDestroyOfflineTts(tts);
    std::cerr << "[服务] stdin 结束，服务退出\n";
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    std::signal(SIGPIPE, SIG_IGN);

    try {
        const Options options = ParseOptions(argc, argv);
        ModelPaths paths;
        paths.acoustic = options.model_dir + "/model-steps-3.q.onnx";
        paths.vocoder = options.model_dir + "/vocos-16khz-univ.q.onnx";
        paths.tokens = options.model_dir + "/tokens.txt";
        paths.lexicon = options.model_dir + "/lexicon.txt";
        paths.data_dir = options.model_dir + "/espeak-ng-data";
        paths.rule_fsts = options.model_dir + "/date-zh.fst," +
            options.model_dir + "/number-zh.fst";
        return RunService(paths, options);
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n"
                  << "use --help to show available options\n";
        return 2;
    }
}
