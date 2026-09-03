#pragma once
#include "types.h"
#include <opencv2/core.hpp>
#include <string>
void draw_detections(cv::Mat& image, const std::vector<SegmentationDetection>& detections);

std::string format_pipeline_status(double capture_fps, double infer_fps,
                                   double display_fps, std::size_t detection_count,
                                   double preprocess_ms, double infer_ms,
                                   const std::string& ep_affinity);
