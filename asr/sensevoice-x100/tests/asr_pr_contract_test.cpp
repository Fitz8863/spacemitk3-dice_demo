/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "asr_config.hpp"
#include "asr_service.h"
#include "asr_types.hpp"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "ASSERTION FAILED: " << message << std::endl;
        std::exit(1);
    }
}

bool contains(const std::vector<std::string>& values, const std::string& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

void verify_public_presets() {
    const auto presets = SpacemiT::AsrConfig::AvailablePresets();
    require(contains(presets, "sensevoice"), "sensevoice preset must be advertised");
    require(contains(presets, "funasr"), "funasr preset must be advertised");
    require(contains(presets, "qwen3-asr"), "qwen3-asr preset must be advertised");
    require(contains(presets, "gemma4-asr"), "gemma4-asr preset must be advertised");
    require(contains(presets, "zipformer"), "zipformer preset must be advertised");

    const auto sensevoice = SpacemiT::AsrConfig::Preset("sensevoice");
    require(sensevoice.engine == "sensevoice", "sensevoice preset must select sensevoice engine");
    require(sensevoice.sample_rate == 16000, "sensevoice preset must keep 16 kHz sample rate");
    require(sensevoice.provider == "spacemit", "sensevoice preset must keep default provider");
    require(!sensevoice.model_dir.empty(), "sensevoice preset must provide a model directory");

    const auto funasr = SpacemiT::AsrConfig::Preset("funasr");
    require(funasr.engine == "funasr", "funasr preset must select funasr engine");
    require(funasr.model_dir.empty(), "funasr preset must not require a local model directory");
    require(funasr.endpoint.find("/v1/audio/transcriptions") != std::string::npos,
            "funasr preset must use the transcription endpoint");
    require(funasr.model == "funasr", "funasr preset must use the funasr model alias");

    const auto qwen3 = SpacemiT::AsrConfig::Preset("qwen3-asr");
    require(qwen3.engine == "qwen3-asr", "qwen3 preset must select qwen3-asr engine");
    require(qwen3.model_dir.empty(), "qwen3 preset must not require a local model directory");
    require(qwen3.provider == "cpu", "qwen3 preset must select cpu provider by default");
    require(qwen3.endpoint.find("/v1/chat/completions") != std::string::npos,
            "qwen3 preset must keep the chat completions endpoint");
    require(qwen3.model == "qwen3-asr", "qwen3 preset must keep the qwen3-asr model alias");

    const auto gemma4 = SpacemiT::AsrConfig::Preset("gemma4-asr");
    require(gemma4.engine == "gemma4-asr",
            "gemma4 preset must select gemma4-asr engine");
    require(gemma4.model_dir.empty(),
            "gemma4 preset must not require a local model directory");
    require(gemma4.endpoint.find("/v1/audio/transcriptions") != std::string::npos,
            "gemma4 preset must use the transcription endpoint");
    require(gemma4.model == "gemma4-asr",
            "gemma4 preset must use the gemma4-asr model alias");
    require(gemma4.task == "transcribe",
            "gemma4 preset must default to transcription");

    const auto zipformer = SpacemiT::AsrConfig::Preset("zipformer");
    require(zipformer.engine == "zipformer", "zipformer preset must select zipformer engine");
    require(!zipformer.model_dir.empty(), "zipformer preset must provide a model directory");
}

void verify_internal_config_contract() {
    const auto sensevoice = asr::ASRConfig::sensevoice("/tmp/asr-model");
    require(sensevoice.backend == asr::BackendType::SENSEVOICE,
            "internal sensevoice config must select SENSEVOICE backend");
    require(sensevoice.model_path == "/tmp/asr-model/model_quant_optimized.onnx",
            "sensevoice config must derive model path from model dir");
    require(sensevoice.vocab_path == "/tmp/asr-model/tokens.txt",
            "sensevoice config must derive vocab path from model dir");
    require(asr::ConfigValidator::validate(sensevoice).isOk(),
            "valid sensevoice config must pass validation");

    const auto funasr = asr::ASRConfig::funasr();
    require(funasr.backend == asr::BackendType::FUNASR,
            "funasr config must select FUNASR backend");
    require(funasr.extra_params.at("endpoint").find("/v1/audio/transcriptions") !=
                std::string::npos,
            "funasr config must use the transcription endpoint");
    require(asr::ConfigValidator::validate(funasr).isOk(),
            "valid funasr server config must pass validation");

    const auto gemma4 = asr::ASRConfig::gemma4Asr();
    require(gemma4.backend == asr::BackendType::GEMMA4_ASR,
            "gemma4 config must select GEMMA4_ASR backend");
    require(gemma4.task == asr::RecognitionTask::TRANSCRIBE,
            "gemma4 config must default to transcription");
    require(gemma4.extra_params.at("endpoint").find("/v1/audio/transcriptions") !=
                std::string::npos,
            "gemma4 config must use the transcription endpoint");
    require(asr::ConfigValidator::validate(gemma4).isOk(),
            "valid gemma4 server config must pass validation");

    const auto translation = gemma4.withTask(asr::RecognitionTask::TRANSLATE);
    require(gemma4.task == asr::RecognitionTask::TRANSCRIBE,
            "withTask must not mutate the source config");
    require(translation.task == asr::RecognitionTask::TRANSLATE,
            "withTask must set translation on the returned config");
    require(asr::ConfigValidator::validate(translation).isOk(),
            "gemma4 translation config must pass validation");

    const auto unsupported_translation =
        funasr.withTask(asr::RecognitionTask::TRANSLATE);
    require(
        asr::ConfigValidator::validate(unsupported_translation).code ==
            asr::ErrorCode::INVALID_CONFIG,
        "translation must be rejected by backends without translation support");

    const auto legacy_funasr = asr::ASRConfig::funasrCloud("test-token");
    require(legacy_funasr.backend == asr::BackendType::FUNASR,
            "legacy funasr cloud factory must remain source compatible");
    const auto legacy_error = asr::ConfigValidator::validate(legacy_funasr);
    require(legacy_error.code == asr::ErrorCode::INVALID_CONFIG,
            "legacy funasr cloud transport must fail validation");
    require(legacy_error.message.find("not implemented") != std::string::npos,
            "legacy funasr cloud validation must explain the unsupported transport");

    const auto streaming = sensevoice.withStreaming(40);
    require(sensevoice.mode == asr::RecognitionMode::OFFLINE,
            "withStreaming must not mutate the source config");
    require(streaming.mode == asr::RecognitionMode::STREAMING,
            "withStreaming must set streaming mode on returned config");
    require(streaming.chunk_size_ms == 40,
            "withStreaming must preserve requested chunk size");
    require(streaming.return_partial_results,
            "withStreaming must enable partial results");

    const auto no_vad = sensevoice.withoutVAD();
    require(sensevoice.vad_enabled, "withoutVAD must not mutate the source config");
    require(!no_vad.vad_enabled, "withoutVAD must disable VAD on returned config");

    const auto word_ts = sensevoice.withWordTimestamps();
    require(word_ts.return_word_timestamps,
            "withWordTimestamps must enable word timestamps");

    require(!sensevoice.enable_emotion,
            "sensevoice config must keep emotion output disabled by default");
    const auto with_emotion = sensevoice.withEmotion();
    require(!sensevoice.enable_emotion,
            "withEmotion must not mutate the source config");
    require(with_emotion.enable_emotion,
            "withEmotion must enable emotion output on returned config");
}

void verify_audio_chunk_contract() {
    const int16_t pcm16[4] = {0, 1, -2, 3};
    const auto pcm16_chunk = asr::AudioChunk::fromPCM16(pcm16, 4, 16000, 1);
    require(pcm16_chunk.data == pcm16, "PCM16 chunk must keep the original data pointer");
    require(pcm16_chunk.size_bytes == sizeof(pcm16), "PCM16 chunk must report byte size");
    require(pcm16_chunk.format == asr::AudioFormat::PCM_S16LE,
            "PCM16 chunk must report PCM_S16LE format");
    require(pcm16_chunk.sample_rate == 16000, "PCM16 chunk must keep sample rate");
    require(pcm16_chunk.channels == 1, "PCM16 chunk must keep channel count");

    const float pcmf[3] = {0.0f, 0.5f, -0.5f};
    const auto pcmf_chunk = asr::AudioChunk::fromPCMFloat(pcmf, 3, 8000, 1);
    require(pcmf_chunk.data == pcmf, "float chunk must keep the original data pointer");
    require(pcmf_chunk.size_bytes == sizeof(pcmf), "float chunk must report byte size");
    require(pcmf_chunk.format == asr::AudioFormat::PCM_F32LE,
            "float chunk must report PCM_F32LE format");
    require(pcmf_chunk.sample_rate == 8000, "float chunk must keep sample rate");
}

void verify_emotion_metadata_contract() {
    asr::SentenceResult internal_sentence;
    internal_sentence.emotion = "happy";
    require(internal_sentence.emotion == "happy",
            "internal sentence result must expose emotion metadata");

    SpacemiT::Sentence public_sentence;
    public_sentence.emotion = "neutral";
    require(public_sentence.emotion == "neutral",
            "public sentence result must expose emotion metadata");
}

void verify_invalid_config_error_path() {
    bool threw = false;
    try {
        (void)SpacemiT::AsrConfig::Preset("does-not-exist");
    } catch (const std::invalid_argument& exc) {
        threw = std::string(exc.what()).find("Unknown ASR preset") != std::string::npos;
    }
    require(threw, "unknown public preset must throw a useful invalid_argument");

    asr::ASRConfig missing_model;
    missing_model.backend = asr::BackendType::SENSEVOICE;
    auto err = asr::ConfigValidator::validate(missing_model);
    require(err.code == asr::ErrorCode::INVALID_CONFIG,
            "local backend without model path must report INVALID_CONFIG");

    auto invalid_rate = asr::ASRConfig::sensevoice("/tmp/asr-model");
    invalid_rate.sample_rate = 44100;
    err = asr::ConfigValidator::validate(invalid_rate);
    require(err.code == asr::ErrorCode::UNSUPPORTED_SAMPLE_RATE,
            "unsupported sample rate must report UNSUPPORTED_SAMPLE_RATE");

    auto invalid_channels = asr::ASRConfig::sensevoice("/tmp/asr-model");
    invalid_channels.channels = 3;
    err = asr::ConfigValidator::validate(invalid_channels);
    require(err.code == asr::ErrorCode::INVALID_CONFIG,
            "invalid channel count must report INVALID_CONFIG");
}

}  // namespace

int main(int argc, char** argv) {
    require(argc == 2, "expected one test mode argument");
    const std::string mode = argv[1];

    if (mode == "--config-contract") {
        verify_public_presets();
        verify_internal_config_contract();
        verify_audio_chunk_contract();
        verify_emotion_metadata_contract();
    } else if (mode == "--invalid-config-error-path") {
        verify_invalid_config_error_path();
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 2;
    }

    std::cout << "PASS " << mode << std::endl;
    return 0;
}
