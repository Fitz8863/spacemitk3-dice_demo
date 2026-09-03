#pragma once

#include <string>
#include "affinity.h"

struct AppConfig {
    std::string model = "models/yolov8n-seg.q.onnx";
    int camera = 1;
    std::string device;
    int width = 1280;
    int height = 720;
    int fps = 24;
    int intra_threads = 2;
    std::string ep_affinity = "12;13";
    float conf = 0.25f;
    float iou = 0.45f;
    int max_detections = 100;
    int queue_depth = 2;
    bool display_enabled = true;
    std::string decoder = "auto";
    int focus = -1;
    int zoom = -1;
};

bool load_config(const std::string& path, AppConfig& config, std::string& error);
