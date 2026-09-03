#pragma once

#include "affinity.h"
#include <cstddef>
#include <string>
#include <vector>

struct AppConfig {
    std::string config_path = "config.json";
    std::string model = "models/yolov8n-seg.q.onnx";
    std::vector<std::string> class_names;
    int camera = 1;
    std::string device;
    int width = 1280;
    int height = 720;
    int fps = 25;
    int intra_threads = 2;
    std::string ep_affinity = "12;13";
    std::size_t queue_depth = 2;
    float conf = 0.50f;
    float iou = 0.45f;
    int max_detections = 100;
    int focus = -1;
    int zoom = -1;
    bool display_enabled = true;
    bool yolov8_enabled = true;
    bool self_test = false;
    bool no_display = false;
    int max_frames = 0;
    std::string dump_input;
    std::string decoder = "auto";

    bool rtsp_enabled = false;
    std::string rtsp_host = "127.0.0.1";
    int rtsp_port = 8554;
    std::string rtsp_path = "/dice";
};

bool load_config(const std::string& path, AppConfig& config, std::string& error);

std::string normalize_rtsp_host(std::string host);
std::string normalize_rtsp_path(std::string path);
