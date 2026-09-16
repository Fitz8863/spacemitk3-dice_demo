#include "types.h"
#include <cassert>
#include <iostream>

int main() {
    PoseDetection detection;
    detection.class_id = 0;
    detection.confidence = 0.9f;
    detection.label = "person";
    detection.keypoints[0] = {10.0f, 20.0f, 0.95f};
    detection.keypoints[16] = {30.0f, 40.0f, 0.05f};
    assert(detection.keypoints.size() == 17);
    assert(detection.keypoints[0].confidence == 0.95f);
    assert(detection.keypoints[16].confidence == 0.05f);
    assert(detection.class_id == 0);
    std::cout << "pose result seam test passed\n";
    return 0;
}
