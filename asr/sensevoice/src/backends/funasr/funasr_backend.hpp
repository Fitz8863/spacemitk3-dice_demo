/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FUNASR_BACKEND_HPP
#define FUNASR_BACKEND_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../asr_backend.hpp"

namespace asr {

/// Fun-ASR backend backed by llama-server's OpenAI transcription endpoint.
class FunASRBackend : public IASRBackend {
public:
    FunASRBackend();
    ~FunASRBackend() override;

    ErrorInfo initialize(const ASRConfig& config) override;
    void shutdown() override;
    bool isInitialized() const override { return initialized_.load(); }

    BackendType getType() const override { return BackendType::FUNASR; }
    std::string getName() const override { return "Fun-ASR"; }
    std::string getVersion() const override { return "1.0.4"; }
    bool supportsStreaming() const override { return false; }

    std::vector<AudioFormat> getSupportedFormats() const override {
        return {AudioFormat::PCM_S16LE, AudioFormat::PCM_F32LE};
    }
    std::vector<int> getSupportedSampleRates() const override {
        return {16000};
    }

    ErrorInfo recognize(const AudioChunk& audio, RecognitionResult& result) override;
    ErrorInfo recognizeFile(const std::string& file_path, RecognitionResult& result) override;

private:
    ErrorInfo transcribe(const std::vector<float>& samples, std::string& out_text);
    RecognitionResult buildResult(const std::string& text,
        int64_t audio_duration_ms,
        int64_t processing_time_ms) const;

    ASRConfig config_;
    std::atomic<bool> initialized_{false};
    std::string endpoint_;
    std::string model_;
    int64_t timeout_sec_ = 60;
};

}  // namespace asr

#endif  // FUNASR_BACKEND_HPP
