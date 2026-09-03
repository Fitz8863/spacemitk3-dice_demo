#include "types.h"
#include <cassert>
#include <iostream>

int main() {
    SegmentationDetection detection;
    detection.class_id = 0;
    detection.confidence = 0.9f;
    detection.mask_contours.push_back({{0, 0}, {10, 0}, {10, 10}});
    assert(detection.class_id == 0);
    assert(detection.mask_contours.size() == 1);
    std::cout << "segmentation result seam test passed\n";
}
