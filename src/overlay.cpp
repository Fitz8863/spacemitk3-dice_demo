#include "overlay.h"

#include "types.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <sstream>

namespace {

// COCO 17 点人体骨架边（Ultralytics 可视化约定），共 19 条。
constexpr int kCocoEdges[][2] = {
    {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12},
    {5, 11},  {6, 12},  {5, 6},   {5, 7},   {6, 8},
    {7, 9},   {8, 10},  {1, 2},   {0, 1},   {0, 2},
    {1, 3},   {2, 4},   {3, 5},   {4, 6},
};
constexpr int kCocoEdgeCount = sizeof(kCocoEdges) / sizeof(kCocoEdges[0]);

// MediaPipe hand 21 点骨架边（0 腕；1-4 拇指；5-8 食指；9-12 中指；
// 13-16 无名指；17-20 小指；各指根部连腕），共 21 条。
constexpr int kHandEdges[][2] = {
    {0, 1},  {1, 2},   {2, 3},   {3, 4},
    {0, 5},  {5, 6},   {6, 7},   {7, 8},   {5, 9},
    {9, 10}, {10, 11}, {11, 12}, {9, 13},
    {13, 14}, {14, 15}, {15, 16}, {13, 17},
    {17, 18}, {18, 19}, {19, 20},
};
constexpr int kHandEdgeCount = sizeof(kHandEdges) / sizeof(kHandEdges[0]);

cv::Scalar color_for_class(int class_id) {
    return cv::Scalar(40 + (class_id * 67) % 200,
                      80 + (class_id * 43) % 160,
                      120 + (class_id * 29) % 120);
}

}  // namespace

void draw_detections(cv::Mat& image, const std::vector<PoseDetection>& detections,
                     float kpt_conf_threshold) {
    if (image.empty() || detections.empty()) return;

    for (const auto& detection : detections) {
        const cv::Scalar color = color_for_class(detection.class_id);
        const int* edges = nullptr;
        int edge_count = 0;
        if (detection.keypoint_count == 17) {
            edges = &kCocoEdges[0][0];
            edge_count = kCocoEdgeCount;
        } else if (detection.keypoint_count == 21) {
            edges = &kHandEdges[0][0];
            edge_count = kHandEdgeCount;
        }
        for (int edge = 0; edge < edge_count; ++edge) {
            const PoseKeypoint& a = detection.keypoints[edges[edge * 2]];
            const PoseKeypoint& b = detection.keypoints[edges[edge * 2 + 1]];
            if (a.confidence < kpt_conf_threshold || b.confidence < kpt_conf_threshold) continue;
            cv::line(image, {cvRound(a.x), cvRound(a.y)}, {cvRound(b.x), cvRound(b.y)},
                     color, 2, cv::LINE_AA);
        }
        for (int k = 0; k < detection.keypoint_count && k < kMaxKeypoints; ++k) {
            const PoseKeypoint& keypoint = detection.keypoints[k];
            if (keypoint.confidence < kpt_conf_threshold) continue;
            cv::circle(image, {cvRound(keypoint.x), cvRound(keypoint.y)}, 4, color, -1, cv::LINE_AA);
        }

        cv::rectangle(image, cv::Point(cvRound(detection.x1), cvRound(detection.y1)),
                      cv::Point(cvRound(detection.x2), cvRound(detection.y2)), color, 2);
        std::ostringstream label;
        label.setf(std::ios::fixed);
        label.precision(2);
        label << detection.label << " " << detection.confidence;
        cv::putText(image, label.str(), cv::Point(cvRound(detection.x1), std::max(18, cvRound(detection.y1) - 4)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv::LINE_AA);
    }
}

std::string format_pipeline_status(double preprocess_fps, double infer_fps,
                                   double display_fps, std::size_t detection_count,
                                   const std::string& ep_affinity) {
    std::ostringstream text;
    text.setf(std::ios::fixed);
    text.precision(1);
    text << "PRE " << preprocess_fps << "  INF " << infer_fps
         << "  DISP " << display_fps << "  det " << detection_count
         << "  EP " << ep_affinity;
    return text.str();
}
