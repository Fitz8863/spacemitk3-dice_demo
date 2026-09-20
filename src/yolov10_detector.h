#pragma once

#include <memory>
#include <string>

struct Detection {
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0, confidence = 0;
    int class_id = -1;
};

class Yolov10Detector {
public:
    Yolov10Detector();
    ~Yolov10Detector();
    // ep_enabled=false builds a plain CPU session; used to isolate SpaceMIT EP
    // issues from model/graph issues on this quantized graph.
    bool init(const std::string& model_path, int intra_threads = 1,
              const std::string& ep_affinity = {}, bool ep_enabled = true);
    std::vector<Detection> infer(const float* data, size_t count, float conf_threshold,
                                 float scale, int pad_x, int pad_y,
                                 int image_width, int image_height);
    bool ready() const { return session_opaque_ != nullptr; }
    const std::string& input_name() const { return input_name_; }
    const std::string& output_name() const { return output_name_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    void* session_opaque_ = nullptr;
    std::string input_name_, output_name_;
};
