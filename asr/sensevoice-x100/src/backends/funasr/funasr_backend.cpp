/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "funasr_backend.hpp"

#include <chrono>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "audio_utils.hpp"
#include "backends/llama_audio/llama_audio_client.hpp"

namespace asr {

FunASRBackend::FunASRBackend() = default;
FunASRBackend::~FunASRBackend() {
    shutdown();
}

ErrorInfo FunASRBackend::initialize(const ASRConfig& config) {
    if (initialized_.load()) {
        return ErrorInfo::error(ErrorCode::ALREADY_STARTED, "Already initialized");
    }

    config_ = config;
    if (!config_.api_endpoint.empty()) {
        const auto endpoint = config_.extra_params.find("endpoint");
        if (endpoint == config_.extra_params.end() || endpoint->second.empty()) {
            return ErrorInfo::error(
                ErrorCode::INVALID_CONFIG,
                "FunASR cloud transport is not implemented; use ASRConfig::funasr()");
        }
    }

    auto get = [&](const std::string& key, const std::string& fallback) {
        const auto it = config_.extra_params.find(key);
        return it != config_.extra_params.end() && !it->second.empty() ? it->second : fallback;
    };

    endpoint_ = get("endpoint", "http://127.0.0.1:8063/v1/audio/transcriptions");
    model_ = get("model", "funasr");
    try {
        const std::string timeout = get("timeout", "60");
        size_t parsed = 0;
        timeout_sec_ = std::stoll(timeout, &parsed);
        if (parsed != timeout.size()) {
            return ErrorInfo::error(ErrorCode::INVALID_CONFIG, "Invalid Fun-ASR timeout");
        }
    } catch (const std::exception&) {
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG, "Invalid Fun-ASR timeout");
    }
    if (timeout_sec_ <= 0 ||
        timeout_sec_ > static_cast<int64_t>(std::numeric_limits<long>::max())) {
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG, "Fun-ASR timeout must be positive");
    }

    std::cout << "[FunASR] endpoint=" << endpoint_
        << " model=" << model_
        << " timeout=" << timeout_sec_ << "s" << std::endl;
    initialized_.store(true);
    return ErrorInfo::ok();
}

void FunASRBackend::shutdown() {
    initialized_.store(false);
}

ErrorInfo FunASRBackend::transcribe(const std::vector<float>& samples, std::string& out_text) {
    llama_audio::TranscriptionRequest request;
    request.endpoint = endpoint_;
    request.model = model_;
    request.language = config_.language == Language::AUTO
        ? std::string()
        : languageToString(config_.language);
    request.timeout_sec = timeout_sec_;
    return llama_audio::postTranscription(samples, config_.sample_rate, request, out_text);
}

ErrorInfo FunASRBackend::recognize(const AudioChunk& audio, RecognitionResult& result) {
    if (!initialized_.load()) {
        return ErrorInfo::error(ErrorCode::NOT_INITIALIZED, "Not initialized");
    }
    if (audio.sample_rate <= 0) {
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG, "Invalid sample rate");
    }

    const auto start = std::chrono::steady_clock::now();
    auto mono = llama_audio::convertToMonoFloat(audio);
    if (mono.empty()) {
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG, "Empty or unsupported audio");
    }
    const int64_t audio_ms = static_cast<int64_t>(mono.size()) * 1000 / audio.sample_rate;
    auto model_audio = audio_utils::normalizeSampleRate(
        std::move(mono), audio.sample_rate, config_.sample_rate);
    if (model_audio.empty()) {
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG, "Empty audio after resampling");
    }

    std::string text;
    const ErrorInfo error = transcribe(model_audio, text);
    if (!error.isOk()) {
        return error;
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    result = buildResult(text, audio_ms, elapsed);
    return ErrorInfo::ok();
}

ErrorInfo FunASRBackend::recognizeFile(
        const std::string& file_path, RecognitionResult& result) {
    if (!initialized_.load()) {
        return ErrorInfo::error(ErrorCode::NOT_INITIALIZED, "Not initialized");
    }

    const auto start = std::chrono::steady_clock::now();
    std::vector<float> mono;
    int source_sample_rate = 0;
    int64_t audio_ms = 0;
    ErrorInfo error = llama_audio::readAudioFile(
        file_path, mono, source_sample_rate, audio_ms);
    if (!error.isOk()) {
        return error;
    }

    auto model_audio = audio_utils::normalizeSampleRate(
        std::move(mono), source_sample_rate, config_.sample_rate);
    if (model_audio.empty()) {
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG, "Empty audio after resampling");
    }

    std::string text;
    error = transcribe(model_audio, text);
    if (!error.isOk()) {
        return error;
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    result = buildResult(text, audio_ms, elapsed);
    return ErrorInfo::ok();
}

RecognitionResult FunASRBackend::buildResult(
        const std::string& text,
        int64_t audio_duration_ms,
        int64_t processing_time_ms) const {
    return llama_audio::buildResult(
        text, audio_duration_ms, processing_time_ms, config_.language);
}

}  // namespace asr
