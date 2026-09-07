/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "asr_backend.hpp"

#include <iostream>
#include <memory>
#include <vector>

#include "backends/funasr/funasr_backend.hpp"
#include "backends/gemma4_asr/gemma4_asr_backend.hpp"
#include "backends/qwen3_asr/qwen3_asr_backend.hpp"
#include "backends/sensevoice/sensevoice_backend.hpp"
#include "backends/zipformer/zipformer_backend.hpp"

namespace asr {

std::unique_ptr<IASRBackend> ASRBackendFactory::create(BackendType type) {
    switch (type) {
        case BackendType::SENSEVOICE:
            return std::make_unique<SenseVoiceBackend>();

        case BackendType::FUNASR:
            return std::make_unique<FunASRBackend>();

        case BackendType::GEMMA4_ASR:
            return std::make_unique<Gemma4ASRBackend>();

        case BackendType::WHISPER:
            // TODO(spacemit): Implement Whisper backend
            std::cerr << "[ASRBackendFactory] Whisper backend not yet implemented" << std::endl;
            return nullptr;

        case BackendType::PARAFORMER:
            // TODO(spacemit): Implement Paraformer backend
            std::cerr << "[ASRBackendFactory] Paraformer backend not yet implemented" << std::endl;
            return nullptr;

        case BackendType::QWEN3_ASR:
            return std::make_unique<Qwen3ASRBackend>();

        case BackendType::ZIPFORMER:
            return std::make_unique<ZipformerBackend>();

        default:
            std::cerr << "[ASRBackendFactory] Unknown backend type" << std::endl;
            return nullptr;
    }
}

bool ASRBackendFactory::isAvailable(BackendType type) {
    switch (type) {
        case BackendType::SENSEVOICE:
        case BackendType::FUNASR:
        case BackendType::GEMMA4_ASR:
        case BackendType::QWEN3_ASR:
        case BackendType::ZIPFORMER:
            return true;
        default:
            return false;
    }
}

std::vector<BackendType> ASRBackendFactory::getAvailableBackends() {
    std::vector<BackendType> backends;
    backends.push_back(BackendType::SENSEVOICE);
    backends.push_back(BackendType::FUNASR);
    backends.push_back(BackendType::QWEN3_ASR);
    backends.push_back(BackendType::ZIPFORMER);
    backends.push_back(BackendType::GEMMA4_ASR);
    return backends;
}

}  // namespace asr
