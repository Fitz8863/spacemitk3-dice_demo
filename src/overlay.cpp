#include "types.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <sstream>

void draw_detections(cv::Mat& image, const std::vector<SegmentationDetection>& detections) {
    for (const auto& detection : detections) {
        const cv::Scalar color(40 + (detection.class_id * 67) % 200,
                               80 + (detection.class_id * 43) % 160,
                               120 + (detection.class_id * 29) % 120);
        cv::Mat mask = image.clone();
        for (const auto& contour : detection.mask_contours) {
            std::vector<std::vector<cv::Point>> contours{contour};
            cv::fillPoly(mask, contours, color);
        }
        cv::addWeighted(mask, 0.35, image, 0.65, 0.0, image);
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
                                   double preprocess_ms, double infer_ms,
                                   const std::string& ep_affinity) {
    std::ostringstream text;
    text.setf(std::ios::fixed);
    text.precision(1);
    text << "PRE " << preprocess_fps << "  INF " << infer_fps
         << "  DISP " << display_fps << "  det " << detection_count
         << "  pre " << preprocess_ms << "ms"
         << "  inf " << infer_ms << "ms"
         << "  EP " << ep_affinity;
    return text.str();
}
