#pragma once
#include "types.h"
#include <opencv2/core.hpp>
#include <string>

// 绘制 pose 检测结果：COCO 骨架连线 + 关键点 + 检测框 + 标签。
// 关键点置信度低于 kpt_conf_threshold 的点与经过它的连线不画。
void draw_detections(cv::Mat& image, const std::vector<PoseDetection>& detections,
                     float kpt_conf_threshold);

std::string format_pipeline_status(double preprocess_fps, double infer_fps,
                                   double display_fps, std::size_t detection_count,
                                   const std::string& ep_affinity);
