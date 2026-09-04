/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef LLAMA_AUDIO_CLIENT_HPP
#define LLAMA_AUDIO_CLIENT_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "../../asr_types.hpp"

namespace asr::llama_audio {

struct TranscriptionRequest {
    std::string endpoint;
    std::string model;
    std::string prompt;
    std::string language;
    int64_t timeout_sec = 60;
    int max_tokens = 512;
};

ErrorInfo postTranscription(
    const std::vector<float>& samples,
    int sample_rate,
    const TranscriptionRequest& request,
    std::string& out_text);

std::vector<float> convertToMonoFloat(const AudioChunk& audio);

ErrorInfo readAudioFile(
    const std::string& file_path,
    std::vector<float>& mono,
    int& sample_rate,
    int64_t& audio_duration_ms);

RecognitionResult buildResult(
    const std::string& text,
    int64_t audio_duration_ms,
    int64_t processing_time_ms,
    Language language);

}  // namespace asr::llama_audio

#endif  // LLAMA_AUDIO_CLIENT_HPP
