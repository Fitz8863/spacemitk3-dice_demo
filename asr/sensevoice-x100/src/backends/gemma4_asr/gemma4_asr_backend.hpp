/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GEMMA4_ASR_BACKEND_HPP
#define GEMMA4_ASR_BACKEND_HPP

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "../asr_backend.hpp"

namespace asr {

class Gemma4ASRBackend : public IASRBackend {
public:
    ErrorInfo initialize(const ASRConfig& config) override;
    void shutdown() override;
    bool isInitialized() const override { return initialized_.load(); }

    BackendType getType() const override { return BackendType::GEMMA4_ASR; }
    std::string getName() const override { return "Gemma4 ASR"; }
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
    ErrorInfo request(const std::vector<float>& samples, std::string& out_text);
    std::string prompt() const;

    ASRConfig config_;
    std::atomic<bool> initialized_{false};
    std::string endpoint_;
    std::string model_;
    int64_t timeout_sec_ = 60;
};

}  // namespace asr

#endif  // GEMMA4_ASR_BACKEND_HPP
