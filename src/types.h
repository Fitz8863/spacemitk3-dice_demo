#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct SegmentationDetection {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float confidence = 0.0f;
    int class_id = -1;
    std::string label;
    std::vector<std::vector<cv::Point>> mask_contours;
};
