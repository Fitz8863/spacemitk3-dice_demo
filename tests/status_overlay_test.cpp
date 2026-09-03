#include "overlay.h"
#include <cassert>
#include <iostream>
#include <string>

int main() {
    const std::string text = format_pipeline_status(24.0, 23.5, 24.0, 1, 7.2, 34.4, "12;13");
    const std::string expected =
        "PRE 24.0  INF 23.5  DISP 24.0  det 1  pre 7.2ms  inf 34.4ms  EP 12;13";
    if (text != expected) {
        std::cerr << "unexpected status text: " << text << "\n";
        return 1;
    }
    std::cout << "status overlay test passed\n";
}
