#include "overlay.h"
#include <cassert>
#include <iostream>
#include <string>

int main() {
    const std::string text = format_pipeline_status(24.0, 23.5, 24.0, 1, 7.2, 34.4, "12;13");
    assert(text == "CAP 24.0  INF 23.5  DISP 24.0  det 1  pre 7.2ms  inf 34.4ms  EP 12;13");
    std::cout << "status overlay test passed\n";
}
