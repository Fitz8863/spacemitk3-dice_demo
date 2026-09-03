#include "types.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <sstream>

void draw_detections(cv::Mat& image, const std::vector<SegmentationDetection>& detections) {
    if (image.empty() || detections.empty()) return;

    // Blend once per detection, not once per contour. Each operation is
    // limited to the union of that detection's mask contours, so no full
    // 1280x720 clone is needed on the display thread.
    const cv::Rect image_bounds(0, 0, image.cols, image.rows);
    for (const auto& detection : detections) {
        const cv::Scalar color(40 + (detection.class_id * 67) % 200,
                               80 + (detection.class_id * 43) % 160,
                               120 + (detection.class_id * 29) % 120);
        cv::Rect roi_rect;
        bool has_contour = false;
        for (const auto& contour : detection.mask_contours) {
            if (contour.size() < 3) continue;
            const cv::Rect contour_rect = cv::boundingRect(contour) & image_bounds;
            if (contour_rect.empty()) continue;
            roi_rect = has_contour ? (roi_rect | contour_rect) : contour_rect;
            has_contour = true;
        }
        if (has_contour && !roi_rect.empty()) {
            std::vector<std::vector<cv::Point>> local_contours;
            local_contours.reserve(detection.mask_contours.size());
            for (const auto& contour : detection.mask_contours) {
                if (contour.size() < 3) continue;
                std::vector<cv::Point> local_contour;
                local_contour.reserve(contour.size());
                for (const auto& point : contour) {
                    local_contour.emplace_back(point.x - roi_rect.x,
                                                point.y - roi_rect.y);
                }
                local_contours.push_back(std::move(local_contour));
            }
            cv::Mat image_roi = image(roi_rect);
            cv::Mat mask_overlay = image_roi.clone();
            cv::fillPoly(mask_overlay, local_contours, color);
            cv::addWeighted(mask_overlay, 0.35, image_roi, 0.65, 0.0, image_roi);
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
