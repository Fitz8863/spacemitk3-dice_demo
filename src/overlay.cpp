#include "types.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <sstream>

namespace {

// 叠加显示配色：按 class_id 索引。选高饱和、与桌面红布（RGB 约 209,74,86）
// 以及盖子绿色都有强反差的颜色——原来那套递推算出的暗棕 BGR(40,80,120)
// 在红布上几乎看不见。cv::Scalar 是 BGR 顺序。
const cv::Scalar kClassColors[] = {
    cv::Scalar(255, 255, 0),  // class 0（cap）：亮青，红色的补色，红底上对比最强
    cv::Scalar(0, 255, 255),  // class 1（ground）：亮黄
};
const cv::Scalar kOverflowColor(255, 0, 255);  // 亮品红，类别数超出配色表时兜底

cv::Scalar color_for_class(int class_id) {
    const int count = static_cast<int>(sizeof(kClassColors) / sizeof(kClassColors[0]));
    if (class_id >= 0 && class_id < count) return kClassColors[class_id];
    return kOverflowColor;
}

}  // namespace

void draw_detections(cv::Mat& image, const std::vector<SegmentationDetection>& detections) {
    if (image.empty() || detections.empty()) return;

    // Blend once per detection, not once per contour. Each operation is
    // limited to the union of that detection's mask contours, so no full
    // 1280x720 clone is needed on the display thread.
    const cv::Rect image_bounds(0, 0, image.cols, image.rows);
    for (const auto& detection : detections) {
        const cv::Scalar color = color_for_class(detection.class_id);
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
