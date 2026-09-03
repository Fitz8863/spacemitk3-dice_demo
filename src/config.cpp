#include "config.h"

#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include <algorithm>
#include <cctype>
#include <sstream>

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

bool read_size(const cv::FileNode& node, const char* key, std::size_t& value) {
    int parsed = 0;
    if (!read_int(node, key, parsed) || parsed < 0) return false;
    value = static_cast<std::size_t>(parsed);
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

bool read_class_names(const cv::FileNode& root, std::vector<std::string>& names,
                     std::string& error) {
    const cv::FileNode node = root["class_names"];
    if (node.empty()) return true;
    if (!node.isSeq()) {
        error = "config class_names must be an array of strings";
        return false;
    }
    std::vector<std::string> parsed;
    for (const auto& item : node) {
        if (!item.isString()) {
            error = "config class_names must contain only strings";
            return false;
        }
        std::string name;
        item >> name;
        if (name.empty()) {
            error = "config class_names cannot contain empty names";
            return false;
        }
        parsed.push_back(std::move(name));
    }
    if (parsed.empty()) {
        error = "config class_names cannot be empty";
        return false;
    }
    names = std::move(parsed);
    return true;
}

bool read_required_string(const cv::FileNode& node, const char* key,
                          std::string& value, std::string& error) {
    if (!read_string(node, key, value)) {
        error = std::string("config ") + key + " must be a string";
        return false;
    }
    return true;
}

bool read_rtsp(const cv::FileNode& root, AppConfig& config, std::string& error) {
    const cv::FileNode node = root["rtsp"];
    if (node.empty()) return true;
    if (!node.isMap()) {
        error = "config rtsp must be an object";
        return false;
    }
    if (!read_bool(node, "enabled", config.rtsp_enabled) ||
        !read_required_string(node, "host", config.rtsp_host, error) ||
        !read_int(node, "port", config.rtsp_port) ||
        !read_required_string(node, "path", config.rtsp_path, error)) {
        if (error.empty()) error = "invalid rtsp configuration";
        return false;
    }
    return true;
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
        config.config_path = path;
        if (!read_required_string(root, "model", config.model, error) ||
            !read_class_names(root, config.class_names, error) ||
            !read_int(root, "width", config.width) ||
            !read_int(root, "height", config.height) ||
            !read_int(root, "fps", config.fps) ||
            !read_int(root, "intra_threads", config.intra_threads) ||
            !read_required_string(root, "ep_affinity", config.ep_affinity, error) ||
            !read_size(root, "queue_depth", config.queue_depth) ||
            !read_float(root, "conf", config.conf) ||
            !read_float(root, "iou", config.iou) ||
            !read_int(root, "max_detections", config.max_detections) ||
            !read_int(root, "focus", config.focus) ||
            !read_int(root, "zoom", config.zoom) ||
            !read_bool(root, "display_enabled", config.display_enabled) ||
            !read_bool(root, "yolov8_enabled", config.yolov8_enabled) ||
            !read_bool(root, "self_test", config.self_test) ||
            !read_bool(root, "no_display", config.no_display) ||
            !read_int(root, "max_frames", config.max_frames) ||
            !read_required_string(root, "dump_input", config.dump_input, error) ||
            !read_required_string(root, "decoder", config.decoder, error)) {
            if (error.empty()) error = "invalid config value";
            return false;
        }

        const cv::FileNode camera = root["camera"];
        if (!camera.empty()) {
            if (camera.isString()) {
                camera >> config.device;
            } else if (camera.isInt() || camera.isReal()) {
                camera >> config.camera;
                config.device.clear();
            } else {
                error = "config camera must be a device path string or numeric index";
                return false;
            }
        }
        std::string configured_device;
        if (!read_required_string(root, "device", configured_device, error)) return false;
        if (!configured_device.empty()) config.device = configured_device;

        if (!read_rtsp(root, config, error)) return false;
        if (!validate_ep_affinity(config.ep_affinity, config.intra_threads, error)) return false;
        config.queue_depth = std::max<std::size_t>(1, config.queue_depth);
        if (config.model.empty() || config.width <= 0 || config.height <= 0 || config.fps <= 0 ||
            config.intra_threads < 1 || config.conf < 0.0f || config.conf > 1.0f ||
            config.iou < 0.0f || config.iou > 1.0f || config.max_detections < 1 ||
            config.max_frames < 0 || config.rtsp_port < 1 || config.rtsp_port > 65535) {
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

std::string normalize_rtsp_host(std::string host) {
    if (host.empty() || host == "0.0.0.0" || host == "*") return "127.0.0.1";
    return host;
}

std::string normalize_rtsp_path(std::string path) {
    if (path.empty()) return "/dice";
    if (path.front() != '/') path.insert(path.begin(), '/');
    while (path.size() > 1 && path.back() == '/') path.pop_back();
    return path;
}
