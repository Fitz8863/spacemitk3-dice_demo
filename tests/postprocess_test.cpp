#include "types.h"
#include <cassert>
#include <iostream>

int main() {
    PoseDetection detection;
    detection.class_id = 0;
    detection.confidence = 0.9f;
    detection.label = "hand";
    detection.keypoint_count = 21;
    detection.keypoints[0] = {10.0f, 20.0f, 0.95f};
    detection.keypoints[20] = {30.0f, 40.0f, 0.05f};
    assert(detection.keypoints.size() == 21);
    assert(detection.keypoint_count == 21);
    assert(detection.keypoints[0].confidence == 0.95f);
    assert(detection.keypoints[20].confidence == 0.05f);
    detection.keypoint_count = 17;
    assert(detection.keypoint_count == 17);
    assert(detection.class_id == 0);
    std::cout << "pose result seam test passed\n";
    return 0;
}
