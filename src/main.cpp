#include "config.h"
#include "gstreamer_camera.h"
#include "opencl_preprocess.h"
#include "overlay.h"
#include "yolov8_seg_detector.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <iomanip>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace {
volatile std::sig_atomic_t stop_requested = 0;
void handle_signal(int) { stop_requested = 1; }

void usage(const char* executable) {
    std::cout << "Usage: " << executable << " [--config PATH] [options]\n"
              << "  --config PATH       JSON config, default config.json\n"
              << "  --model PATH        override model path\n"
              << "  --camera N          override camera index\n"
              << "  --device PATH       override V4L2 device\n"
              << "  --width N --height N --fps N\n"
              << "  --intra-threads N   SpaceMIT EP intra threads\n"
              << "  --ep-affinity LIST  SpaceMIT EP cores, e.g. 12;13\n"
              << "  --conf FLOAT --iou FLOAT\n"
              << "  --no-display        run inference without a window\n"
              << "  --max-frames N      stop after N frames (0=unlimited)\n"
              << "  --self-test         initialize OpenCL and run one model inference\n"
              << "  --help              show this help\n";
}

bool need_value(int& index, int argc, char** argv, std::string& value) {
    if (index + 1 >= argc) {
        std::cerr << argv[index] << " requires a value\n";
        return false;
    }
    value = argv[++index];
    return true;
}

bool parse_args(int argc, char** argv, AppConfig& config,
                bool& self_test, bool& no_display, int& max_frames) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        std::string value;
        if (key == "--help" || key == "-h") { usage(argv[0]); return false; }
        if (key == "--config") {
            if (!need_value(i, argc, argv, value)) return false;
        } else if (key == "--model") {
            if (!need_value(i, argc, argv, config.model)) return false;
        } else if (key == "--device") {
            if (!need_value(i, argc, argv, config.device)) return false;
        } else if (key == "--ep-affinity") {
            if (!need_value(i, argc, argv, config.ep_affinity)) return false;
        } else if (key == "--decoder") {
            if (!need_value(i, argc, argv, config.decoder)) return false;
        } else if (key == "--camera" || key == "--width" || key == "--height" || key == "--fps" ||
                   key == "--intra-threads" || key == "--max-detections" || key == "--queue-depth" ||
                   key == "--focus" || key == "--zoom" || key == "--max-frames") {
            if (!need_value(i, argc, argv, value)) return false;
            try {
                int parsed = std::stoi(value);
                if (key == "--camera") config.camera = parsed;
                else if (key == "--width") config.width = parsed;
                else if (key == "--height") config.height = parsed;
                else if (key == "--fps") config.fps = parsed;
                else if (key == "--intra-threads") config.intra_threads = parsed;
                else if (key == "--max-detections") config.max_detections = parsed;
                else if (key == "--queue-depth") config.queue_depth = parsed;
                else if (key == "--focus") config.focus = parsed;
                else if (key == "--zoom") config.zoom = parsed;
                else max_frames = parsed;
            } catch (const std::exception&) {
                std::cerr << key << " requires an integer\n";
                return false;
            }
        } else if (key == "--conf" || key == "--iou") {
            if (!need_value(i, argc, argv, value)) return false;
            try {
                if (key == "--conf") config.conf = std::stof(value);
                else config.iou = std::stof(value);
            } catch (const std::exception&) {
                std::cerr << key << " requires a number\n";
                return false;
            }
        } else if (key == "--self-test") {
            self_test = true;
        } else if (key == "--no-display") {
            no_display = true;
        } else {
            std::cerr << "Unknown option: " << key << "\n";
            usage(argv[0]);
            return false;
        }
    }
    std::string error;
    if (!validate_ep_affinity(config.ep_affinity, config.intra_threads, error)) {
        std::cerr << "Invalid EP affinity: " << error << "\n";
        return false;
    }
    if (config.width <= 0 || config.height <= 0 || config.fps <= 0 || config.intra_threads < 1 ||
        config.conf < 0.0f || config.conf > 1.0f || config.iou < 0.0f || config.iou > 1.0f ||
        config.max_detections < 1 || max_frames < 0) {
        std::cerr << "Invalid numeric argument\n";
        return false;
    }
    return true;
}

cv::Mat nv12_to_bgr(const cv::Mat& nv12) {
    if (nv12.empty() || nv12.rows * 2 % 3 != 0) throw std::runtime_error("invalid NV12 frame");
    cv::Mat bgr;
    cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
    return bgr;
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    AppConfig config;
    std::string config_path = "config.json";
    bool self_test = false;
    bool no_display = false;
    int max_frames = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) != "--config") continue;
        if (i + 1 >= argc) {
            std::cerr << "--config requires a JSON file path\n";
            return 2;
        }
        config_path = argv[++i];
    }
    std::string error;
    if (!load_config(config_path, config, error)) {
        std::cerr << error << "\n";
        return 2;
    }
    if (!parse_args(argc, argv, config, self_test, no_display, max_frames)) return 2;

    std::cout << "Configuration: model=" << config.model
              << " camera=" << config.camera
              << " device=" << (config.device.empty() ? "<from camera>" : config.device)
              << " size=" << config.width << "x" << config.height << "@" << config.fps
              << " intra_threads=" << config.intra_threads
              << " ep_affinity=" << (config.ep_affinity.empty() ? "<runtime default>" : config.ep_affinity)
              << "\n";
    if (!std::filesystem::exists(config.model)) {
        std::cerr << "Model not found: " << config.model << "\n";
        return 3;
    }

    OpenClPreprocessor preprocessor;
    if (!preprocessor.init()) return 4;
    std::cout << "OpenCL preprocess device: " << preprocessor.device_name() << "\n";

    Yolov8SegDetector detector;
    if (!detector.init(config.model, config.intra_threads, config.ep_affinity)) return 5;

    if (self_test) {
        try {
            cv::Mat synthetic(720 * 3 / 2, 1280, CV_8UC1, cv::Scalar(128));
            const auto prepared = preprocessor.preprocess(synthetic);
            const auto detections = detector.infer(prepared.data->data(), prepared.data->size(), config.conf,
                                                   config.iou, config.max_detections, prepared.scale,
                                                   prepared.pad_x, prepared.pad_y, 1280, 720);
            std::cout << "Self-test passed: preprocess_ms=" << prepared.ms
                      << " detections=" << detections.size() << "\n";
            return 0;
        } catch (const std::exception& exception) {
            std::cerr << "Self-test failed: " << exception.what() << "\n";
            return 6;
        }
    }

    GstreamerMjpegCamera camera;
    if (!camera.open(config.camera, config.device, config.width, config.height, config.fps,
                     config.focus, config.zoom, max_frames > 0 ? max_frames : 0,
                     config.decoder)) {
        return 7;
    }
    std::cout << "Camera opened: device=" << camera.device() << " decoder=" << camera.decoder()
              << " negotiated_fps=" << camera.negotiated_fps() << "\n";

    uint64_t frames = 0;
    auto last_stats = std::chrono::steady_clock::now();
    uint64_t stats_frames = 0;
    while (!stop_requested && (max_frames == 0 || static_cast<int>(frames) < max_frames)) {
        GstreamerFrame frame;
        if (!camera.read(frame, 1000)) {
            if (!camera.isOpen()) break;
            continue;
        }
        try {
            const auto prepared = preprocessor.preprocess(frame.nv12);
            const auto detections = detector.infer(prepared.data->data(), prepared.data->size(), config.conf,
                                                   config.iou, config.max_detections, prepared.scale,
                                                   prepared.pad_x, prepared.pad_y, config.width, config.height);
            ++frames;
            ++stats_frames;
            std::cout << "Frame " << frames << ": detections=" << detections.size()
                      << " preprocess_ms=" << prepared.ms << std::endl;
            if (!no_display && config.display_enabled) {
                cv::Mat bgr = nv12_to_bgr(frame.nv12);
                draw_detections(bgr, detections);
                const auto now = std::chrono::steady_clock::now();
                const double elapsed = std::chrono::duration<double>(now - last_stats).count();
                if (elapsed >= 1.0) {
                    const double fps = stats_frames / elapsed;
                    std::ostringstream text;
                    text.setf(std::ios::fixed);
                    text.precision(1);
                    text << "FPS " << fps << "  pre " << prepared.ms << " ms  det " << detections.size()
                         << "  EP " << config.ep_affinity;
                    cv::putText(bgr, text.str(), {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                                {0, 255, 0}, 2, cv::LINE_AA);
                    stats_frames = 0;
                    last_stats = now;
                }
                cv::imshow("YOLOv8-seg Camera", bgr);
                const int key = cv::waitKey(1);
                if (key == 27 || key == 'q' || key == 'Q') break;
            }
        } catch (const std::exception& exception) {
            std::cerr << "Frame " << frames << " failed: " << exception.what() << "\n";
            camera.close();
            return 8;
        }
    }
    camera.close();
    if (!no_display && config.display_enabled) cv::destroyAllWindows();
    std::cout << "Stopped cleanly after " << frames << " frames\n";
    return 0;
}
