#pragma once

#include "types.h"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

class Yolov8SegDetector {
public:
    Yolov8SegDetector();
    ~Yolov8SegDetector();
    Yolov8SegDetector(const Yolov8SegDetector&) = delete;
    Yolov8SegDetector& operator=(const Yolov8SegDetector&) = delete;

    bool init(const std::string& model_path, int intra_threads,
              const std::string& ep_affinity,
              const std::vector<std::string>& class_names);
    std::vector<SegmentationDetection> infer(
        const float* input, std::size_t input_count, float conf_threshold,
        float iou_threshold, int max_detections, float scale, int pad_x,
        int pad_y, int image_width, int image_height);
    bool ready() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
