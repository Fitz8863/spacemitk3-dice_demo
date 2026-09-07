/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sndfile.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "asr_config.hpp"
#include "backends/funasr/funasr_backend.hpp"

namespace {

constexpr int kInputSampleRate = 8000;
constexpr int kInputChannels = 2;
constexpr int kInputFrames = 800;

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "ASSERTION FAILED: " << message << std::endl;
        std::exit(1);
    }
}

std::vector<int16_t> makeStereoAudio() {
    std::vector<int16_t> audio(kInputFrames * kInputChannels);
    for (int frame = 0; frame < kInputFrames; ++frame) {
        const int16_t sample = static_cast<int16_t>((frame % 200) * 100 - 10000);
        audio[frame * kInputChannels] = sample;
        audio[frame * kInputChannels + 1] = static_cast<int16_t>(-sample);
    }
    return audio;
}

asr::AudioChunk makeAudioChunk(const std::vector<int16_t>& audio) {
    return {
        audio.data(),
        audio.size() * sizeof(int16_t),
        asr::AudioFormat::PCM_S16LE,
        kInputSampleRate,
        kInputChannels,
        -1,
    };
}

asr::ASRConfig makeConfig(
        const std::string& endpoint,
        const std::string& model,
        int timeout = 3) {
    auto config = asr::ASRConfig::funasr(endpoint, model, timeout);
    config.language = asr::Language::ZH;
    return config;
}

void writeTestWave(const std::string& path, const std::vector<int16_t>& audio) {
    SF_INFO info{};
    info.samplerate = kInputSampleRate;
    info.channels = kInputChannels;
    info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    SNDFILE* file = sf_open(path.c_str(), SFM_WRITE, &info);
    require(file != nullptr, "failed to create test WAV");
    const sf_count_t written = sf_writef_short(file, audio.data(), kInputFrames);
    const int close_result = sf_close(file);
    require(written == kInputFrames && close_result == 0, "failed to write test WAV");
}

void verifySuccess(
        const std::string& endpoint,
        const std::string& audio_path,
        const std::vector<int16_t>& audio) {
    asr::FunASRBackend backend;
    auto error = backend.initialize(makeConfig(endpoint, "contract-model"));
    require(error.isOk(), "valid FunASR config must initialize");

    const std::string expected = "Hello \"FunASR\"\n\xE4\xB8\xAD\xE6\x96\x87";

    asr::RecognitionResult memory_result;
    error = backend.recognize(makeAudioChunk(audio), memory_result);
    require(error.isOk(), "in-memory recognition must succeed");
    require(memory_result.getText() == expected, "JSON text escapes must decode correctly");
    require(memory_result.audio_duration_ms == 100, "stereo input duration must use frame count");
    require(memory_result.sentences.size() == 1, "recognition must return one final sentence");
    require(memory_result.sentences[0].detected_language == asr::Language::ZH,
            "recognition must preserve configured language");

    asr::RecognitionResult file_result;
    error = backend.recognizeFile(audio_path, file_result);
    require(error.isOk(), "file recognition must succeed");
    require(file_result.getText() == expected, "file recognition must parse server text");
    require(file_result.audio_duration_ms == 100, "file duration must use source sample rate");
}

void verifyServerErrors(
        const std::string& endpoint,
        const std::vector<int16_t>& audio) {
    const auto chunk = makeAudioChunk(audio);

    {
        asr::FunASRBackend backend;
        auto error = backend.initialize(makeConfig(endpoint, "http-error"));
        require(error.isOk(), "HTTP error test backend must initialize");
        asr::RecognitionResult result;
        error = backend.recognize(chunk, result);
        require(error.code == asr::ErrorCode::INFERENCE_FAILED,
                "non-200 response must report INFERENCE_FAILED");
        require(error.detail.find("contract failure") != std::string::npos,
                "non-200 response must retain server error detail");
    }

    {
        asr::FunASRBackend backend;
        auto error = backend.initialize(makeConfig(endpoint, "malformed-response"));
        require(error.isOk(), "malformed response test backend must initialize");
        asr::RecognitionResult result;
        error = backend.recognize(chunk, result);
        require(error.code == asr::ErrorCode::INFERENCE_FAILED,
                "missing response text must report INFERENCE_FAILED");
    }

    {
        asr::FunASRBackend backend;
        auto error = backend.initialize(makeConfig(endpoint, "slow-response", 1));
        require(error.isOk(), "timeout test backend must initialize");
        asr::RecognitionResult result;
        error = backend.recognize(chunk, result);
        require(error.code == asr::ErrorCode::TIMEOUT,
                "request timeout must report TIMEOUT");
    }
}

void verifyInvalidConfig(const std::string& endpoint) {
    auto invalid_timeout = makeConfig(endpoint, "contract-model");
    invalid_timeout.extra_params["timeout"] = "3s";
    asr::FunASRBackend backend;
    auto error = backend.initialize(invalid_timeout);
    require(error.code == asr::ErrorCode::INVALID_CONFIG,
            "invalid timeout must fail initialization");

    auto legacy = asr::ASRConfig::funasrCloud("test-token");
    legacy.extra_params["endpoint"] = "";
    error = asr::ConfigValidator::validate(legacy);
    require(error.code == asr::ErrorCode::INVALID_CONFIG,
            "legacy cloud transport must fail config validation");

    asr::FunASRBackend legacy_backend;
    error = legacy_backend.initialize(legacy);
    require(error.code == asr::ErrorCode::INVALID_CONFIG,
            "legacy cloud transport must fail backend initialization");
}

}  // namespace

int main(int argc, char** argv) {
    require(argc == 3, "expected endpoint and temporary audio path");
    const std::string endpoint = argv[1];
    const std::string audio_path = argv[2];
    const auto audio = makeStereoAudio();

    writeTestWave(audio_path, audio);
    verifySuccess(endpoint, audio_path, audio);
    verifyServerErrors(endpoint, audio);
    verifyInvalidConfig(endpoint);
    std::remove(audio_path.c_str());

    std::cout << "PASS --funasr-backend-contract" << std::endl;
    return 0;
}
