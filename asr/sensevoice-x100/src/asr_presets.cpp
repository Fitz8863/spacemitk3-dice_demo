/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "asr_service.h"

#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace SpacemiT {

static const std::map<std::string, std::function<AsrConfig()>>& getPresets() {
    static const std::map<std::string, std::function<AsrConfig()>> presets = {
        {"sensevoice", []() {
            AsrConfig config;
            config.engine = "sensevoice";
            config.model_dir = "~/.cache/models/asr/sensevoice";
            return config;
        }},
        {"funasr", []() {
            AsrConfig config;
            config.engine = "funasr";
            config.model_dir = "";
            config.provider = "cpu";
            config.endpoint = "http://127.0.0.1:8063/v1/audio/transcriptions";
            config.model = "funasr";
            return config;
        }},
        {"qwen3-asr", []() {
            AsrConfig config;
            config.engine = "qwen3-asr";
            config.model_dir = "";
            config.provider = "cpu";
            config.endpoint = "http://127.0.0.1:8063/v1/chat/completions";
            config.model = "qwen3-asr";
            return config;
        }},
        {"gemma4-asr", []() {
            AsrConfig config;
            config.engine = "gemma4-asr";
            config.model_dir = "";
            config.provider = "cpu";
            config.endpoint = "http://127.0.0.1:8063/v1/audio/transcriptions";
            config.model = "gemma4-asr";
            config.task = "transcribe";
            return config;
        }},
        {"zipformer", []() {
            AsrConfig config;
            config.engine = "zipformer";
            config.model_dir = "~/.cache/models/asr/zipformer";
            return config;
        }},
    };
    return presets;
}

AsrConfig AsrConfig::Preset(const std::string& name) {
    const auto& presets = getPresets();
    auto it = presets.find(name);
    if (it == presets.end()) {
        throw std::invalid_argument("Unknown ASR preset: '" + name + "'");
    }
    return it->second();
}

std::vector<std::string> AsrConfig::AvailablePresets() {
    const auto& presets = getPresets();
    std::vector<std::string> names;
    names.reserve(presets.size());
    for (const auto& [name, _] : presets) {
        names.push_back(name);
    }
    return names;
}

}  // namespace SpacemiT
