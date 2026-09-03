#include "config.h"

#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <vector>

namespace {

bool read_string(const cv::FileNode& node, const char* key, std::string& value) {
    const cv::FileNode child = node[key];
    if (child.empty()) return true;
    if (!child.isString()) return false;
    child >> value;
    return true;
}

bool read_int(const cv::FileNode& node, const char* key, int& value) {
    const cv::FileNode child = node[key];
    if (child.empty()) return true;
    if (!child.isInt() && !child.isReal()) return false;
    child >> value;
    return true;
}

bool read_float(const cv::FileNode& node, const char* key, float& value) {
    const cv::FileNode child = node[key];
    if (child.empty()) return true;
    if (!child.isInt() && !child.isReal()) return false;
    child >> value;
    return true;
}

bool read_bool(const cv::FileNode& node, const char* key, bool& value) {
    const cv::FileNode child = node[key];
    if (child.empty()) return true;
    if (child.isInt() || child.isReal()) {
        double number = 0.0;
        child >> number;
        value = number != 0.0;
        return true;
    }
    if (child.isString()) {
        std::string text;
        child >> text;
        std::transform(text.begin(), text.end(), text.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (text == "true" || text == "1") { value = true; return true; }
        if (text == "false" || text == "0") { value = false; return true; }
    }
    return false;
}

}  // namespace

bool load_config(const std::string& path, AppConfig& config, std::string& error) {
    try {
        cv::FileStorage storage(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
        if (!storage.isOpened()) {
            error = "cannot open config file: " + path;
            return false;
        }
        const cv::FileNode root = storage.root();
        if (!root.isMap()) {
            error = "config root must be a JSON object";
            return false;
        }
#define READ_STRING(key, field) \
        if (!read_string(root, key, config.field)) { error = std::string("config ") + key + " must be a string"; return false; }
#define READ_INT(key, field) \
        if (!read_int(root, key, config.field)) { error = std::string("config ") + key + " must be numeric"; return false; }
#define READ_FLOAT(key, field) \
        if (!read_float(root, key, config.field)) { error = std::string("config ") + key + " must be numeric"; return false; }
#define READ_BOOL(key, field) \
        if (!read_bool(root, key, config.field)) { error = std::string("config ") + key + " must be boolean"; return false; }
        READ_STRING("model", model)
        READ_INT("camera", camera)
        READ_STRING("device", device)
        READ_INT("width", width)
        READ_INT("height", height)
        READ_INT("fps", fps)
        READ_INT("intra_threads", intra_threads)
        READ_STRING("ep_affinity", ep_affinity)
        READ_FLOAT("conf", conf)
        READ_FLOAT("iou", iou)
        READ_INT("max_detections", max_detections)
        READ_INT("queue_depth", queue_depth)
        READ_BOOL("display_enabled", display_enabled)
        READ_STRING("decoder", decoder)
        READ_INT("focus", focus)
        READ_INT("zoom", zoom)
#undef READ_STRING
#undef READ_INT
#undef READ_FLOAT
#undef READ_BOOL
        if (!validate_ep_affinity(config.ep_affinity, config.intra_threads, error)) return false;
        if (config.model.empty() || config.width <= 0 || config.height <= 0 || config.fps <= 0 ||
            config.conf < 0.0f || config.conf > 1.0f || config.iou < 0.0f || config.iou > 1.0f ||
            config.max_detections < 1 || config.queue_depth < 1) {
            error = "invalid numeric config value";
            return false;
        }
        if (config.decoder != "auto" && config.decoder != "hw" && config.decoder != "sw") {
            error = "decoder must be auto, hw, or sw";
            return false;
        }
        return true;
    } catch (const cv::Exception& exception) {
        error = "failed to parse config: " + std::string(exception.what());
        return false;
    }
}
