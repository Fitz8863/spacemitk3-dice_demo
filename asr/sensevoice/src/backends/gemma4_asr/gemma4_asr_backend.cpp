/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gemma4_asr_backend.hpp"

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

ErrorInfo Gemma4ASRBackend::initialize(const ASRConfig& config) {
    if (initialized_.load()) {
        return ErrorInfo::error(ErrorCode::ALREADY_STARTED, "Already initialized");
    }

    config_ = config;
    auto get = [&](const std::string& key, const std::string& fallback) {
        const auto it = config_.extra_params.find(key);
        return it != config_.extra_params.end() && !it->second.empty() ? it->second : fallback;
    };

    endpoint_ = get("endpoint", "http://127.0.0.1:8063/v1/audio/transcriptions");
    model_ = get("model", "gemma4-asr");
    try {
        const std::string timeout = get("timeout", "60");
        size_t parsed = 0;
        timeout_sec_ = std::stoll(timeout, &parsed);
        if (parsed != timeout.size()) {
            return ErrorInfo::error(ErrorCode::INVALID_CONFIG, "Invalid Gemma4 ASR timeout");
        }
    } catch (const std::exception&) {
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG, "Invalid Gemma4 ASR timeout");
    }
    if (timeout_sec_ <= 0 ||
        timeout_sec_ > static_cast<int64_t>(std::numeric_limits<long>::max())) {
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG, "Gemma4 ASR timeout must be positive");
    }

    std::cout << "[Gemma4ASR] endpoint=" << endpoint_
        << " model=" << model_
        << " task=" << recognitionTaskToString(config_.task)
        << " timeout=" << timeout_sec_ << "s" << std::endl;
    initialized_.store(true);
    return ErrorInfo::ok();
}

void Gemma4ASRBackend::shutdown() {
    initialized_.store(false);
}

std::string Gemma4ASRBackend::prompt() const {
    if (config_.task == RecognitionTask::TRANSLATE) {
        return "Translate this audio into English. Return only the English translation.";
    }
    return "Transcribe this audio. Return only the transcription.";
}

ErrorInfo Gemma4ASRBackend::request(
        const std::vector<float>& samples, std::string& out_text) {
    llama_audio::TranscriptionRequest request;
    request.endpoint = endpoint_;
    request.model = model_;
    request.prompt = prompt();
    request.timeout_sec = timeout_sec_;
    return llama_audio::postTranscription(samples, config_.sample_rate, request, out_text);
}

ErrorInfo Gemma4ASRBackend::recognize(
        const AudioChunk& audio, RecognitionResult& result) {
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
    const ErrorInfo error = request(model_audio, text);
    if (!error.isOk()) {
        return error;
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    result = llama_audio::buildResult(text, audio_ms, elapsed, config_.language);
    return ErrorInfo::ok();
}

ErrorInfo Gemma4ASRBackend::recognizeFile(
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
    error = request(model_audio, text);
    if (!error.isOk()) {
        return error;
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    result = llama_audio::buildResult(text, audio_ms, elapsed, config_.language);
    return ErrorInfo::ok();
}

}  // namespace asr
