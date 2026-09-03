#include "config.h"
#include <cassert>
#include <fstream>
#include <iostream>

int main() {
    const char* path = "/tmp/yolov8_seg_config_contract.json";
    {
        std::ofstream out(path);
        out << R"({
  "model": "models/test.q.onnx",
  "class_names": ["one", "two", "three"],
  "camera": "/dev/video1",
  "device": "",
  "width": 1280,
  "height": 720,
  "fps": 25,
  "intra_threads": 2,
  "ep_affinity": "12;13",
  "queue_depth": 2,
  "conf": 0.5,
  "focus": 0,
  "zoom": 160,
  "display_enabled": false,
  "yolov8_enabled": true,
  "no_display": false,
  "max_frames": 30,
  "dump_input": "/tmp/input.bin",
  "decoder": "auto",
  "rtsp": {"enabled": true, "host": "127.0.0.1", "port": 8554, "path": "/dice"}
})";
    }
    AppConfig config;
    std::string error;
    assert(load_config(path, config, error));
    assert(config.camera == 1);
    assert(config.class_names.size() == 3);
    assert(config.class_names[0] == "one");
    assert(config.class_names[2] == "three");
    assert(config.device.empty());
    assert(config.width == 1280 && config.height == 720 && config.fps == 25);
    assert(config.rtsp_enabled && config.rtsp_host == "127.0.0.1");
    assert(config.rtsp_port == 8554 && config.rtsp_path == "/dice");
    assert(config.max_frames == 30);

    {
        std::ofstream out(path);
        out << R"({"camera": 2, "device": "/dev/video9", "rtsp": {"enabled": false}})";
    }
    AppConfig override_config;
    assert(load_config(path, override_config, error));
    assert(override_config.camera == 2);
    assert(override_config.device == "/dev/video9");
    std::cout << "config contract test passed\n";
}
