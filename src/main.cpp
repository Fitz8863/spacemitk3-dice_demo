#include "config.h"
#include "gstreamer_camera.h"
#include "latest_queue.h"
#include "opencl_preprocess.h"
#include "rtsp_streamer.h"
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
#include <mutex>
#include <thread>
#include <vector>
#include <sstream>
#include <string>

namespace {
volatile std::sig_atomic_t stop_requested = 0;
void handle_signal(int) { stop_requested = 1; }

void usage(const char* executable) {
    std::cout << "Usage: " << executable << " [--config PATH] [options]\n"
              << "  --config PATH       JSON config, default config.json\n"
              << "  --model PATH        override model path\n"
              << "  --camera VALUE      V4L2 index or device path, e.g. /dev/video1\n"
              << "  --device PATH       explicit V4L2 node, overrides --camera\n"
              << "  --width N --height N --fps N\n"
              << "  --intra-threads N   SpaceMIT EP intra threads\n"
              << "  --ep-affinity LIST  SpaceMIT EP cores, e.g. 12;13\n"
              << "  --conf FLOAT --iou FLOAT\n"
              << "  --queue-depth N     compatibility queue depth setting\n"
              << "  --focus N --zoom N  optional V4L2 controls (-1=unchanged)\n"
              << "  --no-display        run inference without a window\n"
              << "  --max-frames N      stop after N frames (0=unlimited)\n"
              << "  --rtsp              enable RTSP publishing\n"
              << "  --no-rtsp           disable RTSP publishing\n"
              << "  --rtsp-host HOST    RTSP server host\n"
              << "  --rtsp-port N       RTSP server port\n"
              << "  --rtsp-path PATH    RTSP path\n"
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
        } else if (key == "--camera") {
            if (!need_value(i, argc, argv, value)) return false;
            if (!value.empty() && value.find_first_not_of("0123456789") == std::string::npos) {
                config.camera = std::stoi(value);
                config.device.clear();
            } else {
                config.device = value;
            }
        } else if (key == "--width" || key == "--height" || key == "--fps" ||
                   key == "--intra-threads" || key == "--max-detections" || key == "--queue-depth" ||
                   key == "--focus" || key == "--zoom" || key == "--max-frames") {
            if (!need_value(i, argc, argv, value)) return false;
            try {
                int parsed = std::stoi(value);
                if (key == "--width") config.width = parsed;
                else if (key == "--height") config.height = parsed;
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
        } else if (key == "--rtsp") {
            config.rtsp_enabled = true;
        } else if (key == "--no-rtsp") {
            config.rtsp_enabled = false;
        } else if (key == "--rtsp-host") {
            if (!need_value(i, argc, argv, config.rtsp_host)) return false;
        } else if (key == "--rtsp-path") {
            if (!need_value(i, argc, argv, config.rtsp_path)) return false;
        } else if (key == "--rtsp-port") {
            if (!need_value(i, argc, argv, value)) return false;
            try {
                config.rtsp_port = std::stoi(value);
            } catch (const std::exception&) {
                std::cerr << "--rtsp-port requires an integer\n";
                return false;
            }
        } else if (key == "--self-test") {
            self_test = true;
        } else if (key == "--no-display") {
            no_display = true;
        } else if (key == "--no-yolov8") {
            config.yolov8_enabled = false;
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
        config.max_detections < 1 || max_frames < 0 ||
        config.rtsp_port < 1 || config.rtsp_port > 65535) {
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
    self_test = config.self_test;
    no_display = config.no_display || !config.display_enabled;
    max_frames = config.max_frames;
    if (!parse_args(argc, argv, config, self_test, no_display, max_frames)) return 2;

    std::cout << "Configuration: model=" << config.model
              << " camera=" << config.camera
              << " device=" << (config.device.empty() ? "<from camera>" : config.device)
              << " size=" << config.width << "x" << config.height << "@" << config.fps
              << " intra_threads=" << config.intra_threads
              << " ep_affinity=" << (config.ep_affinity.empty() ? "<runtime default>" : config.ep_affinity)
              << "\n";
    if (!config.yolov8_enabled) {
        std::cerr << "yolov8_enabled=false is not supported by this segmentation executable\n";
        return 2;
    }
    if (!std::filesystem::exists(config.model)) {
        std::cerr << "Model not found: " << config.model << "\n";
        return 3;
    }

    OpenClPreprocessor preprocessor;
    if (!preprocessor.init()) return 4;
    std::cout << "OpenCL preprocess device: " << preprocessor.device_name() << "\n";

    Yolov8SegDetector detector;
    if (!detector.init(config.model, config.intra_threads, config.ep_affinity,
                       config.class_names)) return 5;

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

    struct CapturedFrame {
        uint64_t id = 0;
        int width = 0;
        int height = 0;
        cv::Mat nv12;
        std::shared_ptr<void> owner;
    };
    struct PreparedFrame {
        uint64_t id = 0;
        int width = 0;
        int height = 0;
        OpenClPreprocessor::Result prepared;
        std::shared_ptr<CapturedFrame> captured;
    };
    struct InferenceResult {
        uint64_t id = 0;
        int width = 0;
        int height = 0;
        std::vector<SegmentationDetection> detections;
        std::shared_ptr<CapturedFrame> captured;
        double preprocess_ms = 0.0;
        double infer_ms = 0.0;
        double postprocess_ms = 0.0;
    };

    LatestQueue<CapturedFrame> display_queue;
    LatestQueue<CapturedFrame> preprocess_input_queue;
    std::unique_ptr<GstreamerMjpegCamera> camera;
    RtspStreamer rtsp_streamer;
    LatestQueue<PreparedFrame> prepared_queue;
    LatestQueue<InferenceResult> result_queue;
    std::atomic<bool> stop{false};
    std::atomic<bool> capture_done{false};
    std::atomic<bool> inference_done{false};
    std::atomic<uint64_t> captured_count{0};
    std::atomic<uint64_t> dropped_capture{0};
    std::atomic<uint64_t> dropped_prepared{0};
    std::atomic<uint64_t> dropped_result{0};
    std::atomic<uint64_t> inferred_count{0};
    std::atomic<uint64_t> displayed_count{0};
    std::string stage_error;
    std::mutex stage_error_mutex;
    auto report_stage_error = [&](const char* stage, const std::exception& exception) {
        {
            std::lock_guard<std::mutex> lock(stage_error_mutex);
            stage_error = std::string(stage) + ": " + exception.what();
        }
        std::cerr << stage_error << "\n";
        stop.store(true, std::memory_order_release);
    };

    const int requested_frames = max_frames;
    std::thread capture_thread([&] {
        uint64_t id = 0;
        try {
            camera = std::make_unique<GstreamerMjpegCamera>();
            if (!camera->open(config.camera, config.device, config.width, config.height, config.fps,
                              config.focus, config.zoom, config.decoder)) {
                throw std::runtime_error("camera open failed");
            }
            std::cout << "Camera opened: device=" << camera->device()
                      << " decoder=" << camera->decoder()
                      << " negotiated_fps=" << camera->negotiated_fps() << "\n";
            if (config.rtsp_enabled && !rtsp_streamer.start(
                    config.rtsp_host, config.rtsp_port, config.rtsp_path,
                    config.width, config.height, camera->negotiated_fps())) {
                throw std::runtime_error("RTSP publisher start failed");
            }
            while (!stop.load(std::memory_order_acquire) && !stop_requested &&
                   (requested_frames == 0 || static_cast<int>(id) < requested_frames)) {
                GstreamerFrame frame;
                if (!camera->read(frame, 1000)) {
                    if (!camera->isOpen()) break;
                    continue;
                }
                auto captured = std::make_shared<CapturedFrame>();
                captured->id = id++;
                captured->width = frame.nv12.cols;
                captured->height = frame.nv12.rows * 2 / 3;
                captured->nv12 = std::move(frame.nv12);
                captured->owner = std::move(frame.owner);
                const bool display_replaced = display_queue.push(captured);
                const bool preprocess_replaced = preprocess_input_queue.push(std::move(captured));
                if (display_replaced || preprocess_replaced) {
                    dropped_capture.fetch_add(1, std::memory_order_relaxed);
                }
                captured_count.fetch_add(1, std::memory_order_relaxed);
            }
        } catch (const std::exception& exception) {
            report_stage_error("Capture stage failed", exception);
        }
        capture_done.store(true, std::memory_order_release);
        display_queue.close();
        preprocess_input_queue.close();
    });

    std::thread preprocess_thread([&] {
        try {
            while (!stop.load(std::memory_order_acquire) && !stop_requested) {
                auto captured = preprocess_input_queue.wait_pop_latest(std::chrono::milliseconds(100));
                if (!captured) {
                    if (preprocess_input_queue.closed_and_empty()) break;
                    continue;
                }
                auto prepared = std::make_shared<PreparedFrame>();
                prepared->id = captured->id;
                prepared->width = captured->width;
                prepared->height = captured->height;
                prepared->captured = std::move(captured);
                prepared->prepared = preprocessor.preprocess(prepared->captured->nv12);
                if (prepared_queue.push(std::move(prepared))) {
                    dropped_prepared.fetch_add(1, std::memory_order_relaxed);
                }
            }
        } catch (const std::exception& exception) {
            report_stage_error("Preprocess stage failed", exception);
        }
        prepared_queue.close();
    });

    std::thread inference_thread([&] {
        try {
            while (!stop.load(std::memory_order_acquire) && !stop_requested) {
                auto prepared = prepared_queue.wait_pop_latest(std::chrono::milliseconds(100));
                if (!prepared) {
                    if (prepared_queue.closed_and_empty()) break;
                    continue;
                }
                auto result = std::make_shared<InferenceResult>();
                result->id = prepared->id;
                result->width = prepared->width;
                result->height = prepared->height;
                result->captured = std::move(prepared->captured);
                result->preprocess_ms = prepared->prepared.ms;
                const auto infer_start = std::chrono::steady_clock::now();
                result->detections = detector.infer(
                    prepared->prepared.data->data(), prepared->prepared.data->size(),
                    config.conf, config.iou, config.max_detections,
                    prepared->prepared.scale, prepared->prepared.pad_x,
                    prepared->prepared.pad_y, prepared->width, prepared->height);
                result->infer_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - infer_start).count();
                if (result_queue.push(std::move(result))) {
                    dropped_result.fetch_add(1, std::memory_order_relaxed);
                }
                inferred_count.fetch_add(1, std::memory_order_relaxed);
            }
        } catch (const std::exception& exception) {
            report_stage_error("Inference stage failed", exception);
        }
        inference_done.store(true, std::memory_order_release);
        result_queue.close();
    });

    auto last_result = std::make_shared<InferenceResult>();
    auto last_stats = std::chrono::steady_clock::now();
    uint64_t last_captured = 0;
    uint64_t last_inferred = 0;
    uint64_t last_displayed = 0;
    double display_fps = 0.0;
    double capture_fps = 0.0;
    double infer_fps = 0.0;
    std::string status_text = format_pipeline_status(
        0.0, 0.0, 0.0, 0, 0.0, 0.0, config.ep_affinity);
    bool display_initialized = false;
    if (!no_display && config.display_enabled) {
        cv::namedWindow("YOLOv8-seg Camera", cv::WINDOW_NORMAL);
        cv::resizeWindow("YOLOv8-seg Camera", config.width, config.height);
        display_initialized = true;
    }

    while (!stop.load(std::memory_order_acquire) && !stop_requested) {
        if (const auto result = result_queue.try_pop_latest()) {
            last_result = std::move(result);
        }
        if (const auto captured = display_queue.try_pop_latest()) {
            cv::Mat bgr = nv12_to_bgr(captured->nv12);
            if (last_result && last_result->captured &&
                last_result->captured->id <= captured->id) {
                draw_detections(bgr, last_result->detections);
            }
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - last_stats).count();
            if (elapsed >= 1.0) {
                const uint64_t current_captured = captured_count.load(std::memory_order_relaxed);
                const uint64_t current_inferred = inferred_count.load(std::memory_order_relaxed);
                const uint64_t current_displayed = displayed_count.load(std::memory_order_relaxed);
                capture_fps = (current_captured - last_captured) / elapsed;
                infer_fps = (current_inferred - last_inferred) / elapsed;
                display_fps = (current_displayed - last_displayed) / elapsed;
                status_text = format_pipeline_status(
                    capture_fps, infer_fps, display_fps, last_result->detections.size(),
                    last_result->preprocess_ms, last_result->infer_ms, config.ep_affinity);
                std::cout << status_text
                          << " drop(cap/pre/res)=" << dropped_capture.load()
                          << "/" << dropped_prepared.load()
                          << "/" << dropped_result.load() << std::endl;
                last_captured = current_captured;
                last_inferred = current_inferred;
                last_displayed = current_displayed;
                last_stats = now;
            }
            // Draw the cached status on every display frame. Previously this
            // putText call lived only in the one-second stats branch, so the
            // text appeared for one frame and disappeared for the next frames.
            if (!bgr.empty()) cv::putText(bgr, status_text, {12, 28}, cv::FONT_HERSHEY_SIMPLEX,
                                          0.62, {0, 255, 0}, 2, cv::LINE_AA);
            if (rtsp_streamer.running()) rtsp_streamer.publish(bgr);
            if (display_initialized) {
                cv::imshow("YOLOv8-seg Camera", bgr);
                displayed_count.fetch_add(1, std::memory_order_relaxed);
                const int key = cv::waitKey(1) & 0xff;
                if (key == 27 || key == 'q' || key == 'Q') {
                    stop.store(true, std::memory_order_release);
                    break;
                }
            } else {
                displayed_count.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            if (inference_done.load(std::memory_order_acquire) &&
                capture_done.load(std::memory_order_acquire) &&
                display_queue.closed_and_empty() && result_queue.closed_and_empty()) {
                break;
            }
            if (display_initialized) {
                const int key = cv::waitKey(1) & 0xff;
                if (key == 27 || key == 'q' || key == 'Q') {
                    stop.store(true, std::memory_order_release);
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    stop.store(true, std::memory_order_release);
    display_queue.close();
    preprocess_input_queue.close();
    prepared_queue.close();
    result_queue.close();
    if (capture_thread.joinable()) capture_thread.join();
    rtsp_streamer.stop();
    camera.reset();
    if (preprocess_thread.joinable()) preprocess_thread.join();
    if (inference_thread.joinable()) inference_thread.join();
    if (display_initialized) cv::destroyAllWindows();

    std::cout << "Stopped cleanly: captured=" << captured_count.load()
              << " inferred=" << inferred_count.load()
              << " displayed=" << displayed_count.load()
              << " dropped(cap/pre/res)=" << dropped_capture.load()
              << "/" << dropped_prepared.load()
              << "/" << dropped_result.load() << "\n";
    {
        std::lock_guard<std::mutex> lock(stage_error_mutex);
        if (!stage_error.empty()) return 8;
    }
    return 0;

}
