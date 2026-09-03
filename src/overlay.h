#pragma once
#include "types.h"
#include <opencv2/core.hpp>
void draw_detections(cv::Mat& image, const std::vector<SegmentationDetection>& detections);
