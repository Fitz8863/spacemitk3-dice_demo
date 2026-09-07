/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llama_audio_client.hpp"

#include <curl/curl.h>
#include <sndfile.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace asr::llama_audio {
namespace {

size_t curlWriteCallback(char* ptr, size_t size, size_t nmemb, void* user_data) {
    auto* response = static_cast<std::string*>(user_data);
    response->append(ptr, size * nmemb);
    return size * nmemb;
}

void appendLe16(std::string& out, uint16_t value) {
    out.push_back(static_cast<char>(value & 0xff));
    out.push_back(static_cast<char>((value >> 8) & 0xff));
}

void appendLe32(std::string& out, uint32_t value) {
    out.push_back(static_cast<char>(value & 0xff));
    out.push_back(static_cast<char>((value >> 8) & 0xff));
    out.push_back(static_cast<char>((value >> 16) & 0xff));
    out.push_back(static_cast<char>((value >> 24) & 0xff));
}

std::string wavEncode(const std::vector<float>& samples, int sample_rate) {
    constexpr uint16_t kChannels = 1;
    constexpr uint16_t kBitsPerSample = 16;
    constexpr uint16_t kBlockAlign = kChannels * kBitsPerSample / 8;

    if (sample_rate <= 0 ||
        samples.size() > (std::numeric_limits<uint32_t>::max() - 36) / kBlockAlign) {
        return {};
    }

    const uint32_t data_size = static_cast<uint32_t>(samples.size() * kBlockAlign);
    std::string wav;
    wav.reserve(44 + data_size);
    wav.append("RIFF", 4);
    appendLe32(wav, 36 + data_size);
    wav.append("WAVEfmt ", 8);
    appendLe32(wav, 16);
    appendLe16(wav, 1);
    appendLe16(wav, kChannels);
    appendLe32(wav, static_cast<uint32_t>(sample_rate));
    appendLe32(wav, static_cast<uint32_t>(sample_rate) * kBlockAlign);
    appendLe16(wav, kBlockAlign);
    appendLe16(wav, kBitsPerSample);
    wav.append("data", 4);
    appendLe32(wav, data_size);

    for (float sample : samples) {
        const float clamped = std::clamp(sample, -1.0f, 1.0f);
        const auto pcm = static_cast<int16_t>(std::lrint(clamped * 32767.0f));
        appendLe16(wav, static_cast<uint16_t>(pcm));
    }
    return wav;
}

void appendUtf8(std::string& out, uint32_t codepoint) {
    if (codepoint <= 0x7f) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
        out.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
        out.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
}

bool parseHex4(const std::string& json, size_t offset, uint32_t& value) {
    if (offset + 4 > json.size()) {
        return false;
    }
    value = 0;
    for (size_t i = 0; i < 4; ++i) {
        const char c = json[offset + i];
        value <<= 4;
        if (c >= '0' && c <= '9') {
            value |= static_cast<uint32_t>(c - '0');
        } else if (c >= 'a' && c <= 'f') {
            value |= static_cast<uint32_t>(c - 'a' + 10);
        } else if (c >= 'A' && c <= 'F') {
            value |= static_cast<uint32_t>(c - 'A' + 10);
        } else {
            return false;
        }
    }
    return true;
}

bool extractJsonString(const std::string& json, const std::string& key, std::string& value) {
    const std::string quoted_key = "\"" + key + "\"";
    size_t pos = json.find(quoted_key);
    if (pos == std::string::npos) {
        return false;
    }
    pos = json.find(':', pos + quoted_key.size());
    if (pos == std::string::npos) {
        return false;
    }
    do {
        ++pos;
    } while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos])));
    if (pos >= json.size() || json[pos] != '"') {
        return false;
    }

    value.clear();
    for (++pos; pos < json.size(); ++pos) {
        const char c = json[pos];
        if (c == '"') {
            return true;
        }
        if (c != '\\') {
            value.push_back(c);
            continue;
        }
        if (++pos >= json.size()) {
            return false;
        }
        switch (json[pos]) {
            case '"': value.push_back('"'); break;
            case '\\': value.push_back('\\'); break;
            case '/': value.push_back('/'); break;
            case 'b': value.push_back('\b'); break;
            case 'f': value.push_back('\f'); break;
            case 'n': value.push_back('\n'); break;
            case 'r': value.push_back('\r'); break;
            case 't': value.push_back('\t'); break;
            case 'u': {
                uint32_t codepoint = 0;
                if (!parseHex4(json, pos + 1, codepoint)) {
                    return false;
                }
                pos += 4;
                if (codepoint >= 0xd800 && codepoint <= 0xdbff &&
                    pos + 6 < json.size() && json[pos + 1] == '\\' && json[pos + 2] == 'u') {
                    uint32_t low = 0;
                    if (parseHex4(json, pos + 3, low) && low >= 0xdc00 && low <= 0xdfff) {
                        codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                        pos += 6;
                    }
                }
                appendUtf8(value, codepoint);
                break;
            }
            default: return false;
        }
    }
    return false;
}

bool addMimeField(curl_mime* mime, const char* name, const std::string& value) {
    curl_mimepart* part = curl_mime_addpart(mime);
    return part &&
        curl_mime_name(part, name) == CURLE_OK &&
        curl_mime_data(part, value.c_str(), CURL_ZERO_TERMINATED) == CURLE_OK;
}

}  // namespace

ErrorInfo postTranscription(
        const std::vector<float>& samples,
        int sample_rate,
        const TranscriptionRequest& request,
        std::string& out_text) {
    const std::string wav = wavEncode(samples, sample_rate);
    if (wav.empty()) {
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR, "WAV encoding failed");
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR, "curl_easy_init failed");
    }

    std::string response;
    curl_mime* mime = curl_mime_init(curl);
    if (!mime) {
        curl_easy_cleanup(curl);
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR, "curl_mime_init failed");
    }

    curl_mimepart* file_part = curl_mime_addpart(mime);
    const bool file_ok = file_part &&
        curl_mime_name(file_part, "file") == CURLE_OK &&
        curl_mime_filename(file_part, "audio.wav") == CURLE_OK &&
        curl_mime_type(file_part, "audio/wav") == CURLE_OK &&
        curl_mime_data(file_part, wav.data(), wav.size()) == CURLE_OK;
    bool mime_ok = file_ok &&
        addMimeField(mime, "model", request.model) &&
        addMimeField(mime, "response_format", "json") &&
        addMimeField(mime, "temperature", "0") &&
        addMimeField(mime, "max_tokens", std::to_string(request.max_tokens));
    if (!request.prompt.empty()) {
        mime_ok = mime_ok && addMimeField(mime, "prompt", request.prompt);
    }
    if (!request.language.empty()) {
        mime_ok = mime_ok && addMimeField(mime, "language", request.language);
    }
    if (!mime_ok) {
        curl_mime_free(mime);
        curl_easy_cleanup(curl);
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR, "Failed to build multipart request");
    }

    curl_easy_setopt(curl, CURLOPT_URL, request.endpoint.c_str());
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, static_cast<long>(request.timeout_sec));
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

    const CURLcode curl_result = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_mime_free(mime);
    curl_easy_cleanup(curl);

    if (curl_result != CURLE_OK) {
        const ErrorCode code = curl_result == CURLE_OPERATION_TIMEDOUT
            ? ErrorCode::TIMEOUT
            : ErrorCode::NETWORK_ERROR;
        return ErrorInfo::error(
            code,
            "llama-server request failed: " + std::string(curl_easy_strerror(curl_result)));
    }
    if (http_code != 200) {
        return ErrorInfo::error(
            ErrorCode::INFERENCE_FAILED,
            "llama-server returned HTTP " + std::to_string(http_code),
            response.substr(0, 300));
    }
    if (!extractJsonString(response, "text", out_text)) {
        return ErrorInfo::error(
            ErrorCode::INFERENCE_FAILED,
            "Invalid transcription response from llama-server",
            response.substr(0, 300));
    }
    return ErrorInfo::ok();
}

std::vector<float> convertToMonoFloat(const AudioChunk& audio) {
    if (!audio.data || audio.size_bytes == 0 || audio.channels <= 0) {
        return {};
    }

    const size_t sample_size = audio.format == AudioFormat::PCM_S16LE
        ? sizeof(int16_t)
        : audio.format == AudioFormat::PCM_F32LE ? sizeof(float) : 0;
    if (sample_size == 0 ||
        audio.size_bytes % (sample_size * static_cast<size_t>(audio.channels)) != 0) {
        return {};
    }

    std::vector<float> interleaved;
    if (audio.format == AudioFormat::PCM_S16LE) {
        const auto* input = static_cast<const int16_t*>(audio.data);
        const size_t count = audio.size_bytes / sizeof(int16_t);
        interleaved.resize(count);
        for (size_t i = 0; i < count; ++i) {
            interleaved[i] = static_cast<float>(input[i]) / 32768.0f;
        }
    } else {
        const auto* input = static_cast<const float*>(audio.data);
        const size_t count = audio.size_bytes / sizeof(float);
        interleaved.assign(input, input + count);
    }

    if (audio.channels == 1) {
        return interleaved;
    }

    const size_t frames = interleaved.size() / static_cast<size_t>(audio.channels);
    std::vector<float> mono(frames);
    for (size_t frame = 0; frame < frames; ++frame) {
        float sum = 0.0f;
        for (int channel = 0; channel < audio.channels; ++channel) {
            sum += interleaved[frame * audio.channels + channel];
        }
        mono[frame] = sum / audio.channels;
    }
    return mono;
}

ErrorInfo readAudioFile(
        const std::string& file_path,
        std::vector<float>& mono,
        int& sample_rate,
        int64_t& audio_duration_ms) {
    SF_INFO info{};
    SNDFILE* file = sf_open(file_path.c_str(), SFM_READ, &info);
    if (!file) {
        return ErrorInfo::error(
            ErrorCode::MODEL_NOT_FOUND,
            "Cannot open audio file: " + file_path,
            sf_strerror(nullptr));
    }
    if (info.frames <= 0 || info.samplerate <= 0 || info.channels <= 0) {
        sf_close(file);
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG, "Invalid or empty audio file");
    }

    std::vector<float> interleaved(
        static_cast<size_t>(info.frames) * static_cast<size_t>(info.channels));
    const sf_count_t frames_read = sf_readf_float(file, interleaved.data(), info.frames);
    sf_close(file);
    if (frames_read <= 0) {
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG, "Invalid or empty audio file");
    }

    mono.resize(static_cast<size_t>(frames_read));
    for (sf_count_t frame = 0; frame < frames_read; ++frame) {
        float sum = 0.0f;
        for (int channel = 0; channel < info.channels; ++channel) {
            sum += interleaved[static_cast<size_t>(frame) * info.channels + channel];
        }
        mono[static_cast<size_t>(frame)] = sum / info.channels;
    }

    sample_rate = info.samplerate;
    audio_duration_ms = frames_read * 1000 / info.samplerate;
    return ErrorInfo::ok();
}

RecognitionResult buildResult(
        const std::string& text,
        int64_t audio_duration_ms,
        int64_t processing_time_ms,
        Language language) {
    RecognitionResult result;
    SentenceResult sentence;
    sentence.text = text;
    sentence.begin_time_ms = 0;
    sentence.end_time_ms = static_cast<int32_t>(audio_duration_ms);
    sentence.confidence = 1.0f;
    sentence.is_final = true;
    sentence.detected_language = language;

    result.sentences.push_back(std::move(sentence));
    result.audio_duration_ms = audio_duration_ms;
    result.processing_time_ms = processing_time_ms;
    result.rtf = audio_duration_ms > 0
        ? static_cast<float>(processing_time_ms) / audio_duration_ms
        : 0.0f;
    result.first_result_latency_ms = processing_time_ms;
    result.final_result_latency_ms = processing_time_ms;
    return result;
}

}  // namespace asr::llama_audio
