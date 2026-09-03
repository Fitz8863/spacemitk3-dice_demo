#include "config.h"
#include <cassert>
#include <iostream>

int main() {
    assert(normalize_rtsp_host("") == "127.0.0.1");
    assert(normalize_rtsp_host("0.0.0.0") == "127.0.0.1");
    assert(normalize_rtsp_path("dice") == "/dice");
    assert(normalize_rtsp_path("/dice/") == "/dice");
    std::cout << "rtsp config helper tests passed\n";
}
