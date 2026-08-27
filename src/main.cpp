#include "gstreamer_camera.h"
#include "llm_dice_verifier.h"
#include "mjpeg_streamer.h"
#include "opencl_preprocess.h"
#include "yolov8_detector.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <csignal>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <deque>



namespace {
using Clock = std::chrono::steady_clock;
volatile sig_atomic_t g_signal_stop = 0;
void on_signal(int) { g_signal_stop = 1; }

// Bounded queue: it can keep a few frames, but never grows without bound.
// When full, the oldest frame is discarded so latency cannot accumulate.
template <typename T>
class FrameQueue {
public:
    explicit FrameQueue(size_t capacity) : capacity_(std::max<size_t>(1, capacity)) {}

    bool push(std::shared_ptr<T> value) {
        bool dropped = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) return false;
            if (queue_.size() >= capacity_) {
                queue_.pop_front();
                dropped = true;
            }
            queue_.push_back(std::move(value));
        }
        cv_.notify_one();
        return dropped;
    }

    // Return the newest pending item and discard any older pending items.
    // This is the low-latency policy for an inference/display consumer.
    bool popLatest(std::shared_ptr<T>& value, const std::atomic<bool>& abort,
                   bool prebuffer = false, size_t prebuffer_count = 1) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto ready = [&] {
            return closed_ || abort.load() || g_signal_stop != 0 ||
                   (!queue_.empty() && (!prebuffer || queue_.size() >= prebuffer_count));
        };
        // Poll periodically so Ctrl-C can be observed even while a stage is
        // waiting for a frame. A signal handler cannot safely notify this CV;
        // polling also checks g_signal_stop so Ctrl-C wakes every stage.
        // Keep waiting after a spurious/periodic timeout; returning false here
        // would make the consumer exit whenever a frame takes longer than the
        // polling interval to arrive.
        while (!ready()) {
            cv_.wait_for(lock, std::chrono::milliseconds(100));
        }
        // If the producer closed normally, drain any frames already queued.
        // On abort, also consume an already available newest frame, then stop
        // once the queue is empty.
        if ((abort.load() || g_signal_stop != 0) && queue_.empty()) return false;
        if (queue_.empty()) return false;
        value = std::move(queue_.back());
        if (queue_.size() > 1) dropped_pending_ += queue_.size() - 1;
        queue_.clear();
        return static_cast<bool>(value);
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    size_t takeDroppedPending() {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t n = dropped_pending_;
        dropped_pending_ = 0;
        return n;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.clear();
        dropped_pending_ = 0;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<T>> queue_;
    size_t capacity_;
    size_t dropped_pending_ = 0;
    bool closed_ = false;
};

struct PreparedFrame {
    uint64_t id = 0;
    int width = 0;
    int height = 0;
    OpenClPreprocessor::Result prep;
    std::shared_ptr<cv::Mat> nv12;
    std::shared_ptr<void> gst_owner;
};

struct InferenceResult {
    uint64_t id = 0;
    int width = 0;
    int height = 0;
    std::shared_ptr<cv::Mat> nv12;
    std::shared_ptr<void> gst_owner;
    std::vector<Detection> detections;
};

struct DividerLine {
    bool valid = false;
    bool horizontal = false;
    cv::Point2f point;
    cv::Point2f direction;
    cv::Point2f normal;
};

struct DiceJudgment {
    bool divider_found = false;
    bool valid = false;
    bool horizontal_divider = false;
    int first_count = 0;
    int second_count = 0;
    int first_sum = 0;
    int second_sum = 0;
    std::string first_name = "LEFT";
    std::string second_name = "RIGHT";
    std::vector<int> first_values;
    std::vector<int> second_values;
    std::string message;
    std::string overlay;
    DividerLine divider;
    std::vector<int> sides;
};

struct Args {
    std::string config_path = "config.json";
    std::string model = "models/best.q.onnx";
    int camera = 1;
    std::string device;
    int width = 1280, height = 720, fps = 25;
    int intra_threads = 2;
    std::string ep_affinity = "14;15";
    int focus = 0, zoom = 150;
    float conf = 0.50f;
    size_t queue_depth = 2;
    int stable_frames = 20;
    bool rejudge_on_change = false;
    bool no_display = false;
    int max_frames = 0;
    bool self_test = false;
    std::string dump_input;
    bool yolov8_enabled = true;
    bool llm_enabled = true;
    std::string llm_url = "https://api.rvcompute.com:60000/v1";
    std::string llm_model = "gpt-5.4-mini";
    int llm_timeout_seconds = 20;
    std::string llm_system_prompt;
    std::string llm_user_prompt_template;
    std::string llm_api_key;
    bool no_llm = false;
    bool stream_enabled = false;
    std::string stream_host = "0.0.0.0";
    int stream_port = 8080;
    int stream_jpeg_quality = 80;
};

struct Stats {
    std::atomic<uint64_t> prepared{0};
    std::atomic<uint64_t> inferred{0};
    std::atomic<uint64_t> presented{0};
    std::atomic<uint64_t> dropped_pre{0};
    std::atomic<uint64_t> dropped_result{0};
    std::atomic<uint64_t> detected_frames{0};
    std::atomic<uint64_t> detections{0};
    std::mutex timing_mutex;
    double pre_ms = 0.0;
    double infer_ms = 0.0;
    double display_ms = 0.0;

    void addPre(double v) { std::lock_guard<std::mutex> l(timing_mutex); pre_ms += v; }
    void addInfer(double v) { std::lock_guard<std::mutex> l(timing_mutex); infer_ms += v; }
    void addDisplay(double v) { std::lock_guard<std::mutex> l(timing_mutex); display_ms += v; }
    void averages(double& p, double& i, double& d) {
        std::lock_guard<std::mutex> l(timing_mutex);
        p = prepared ? pre_ms / static_cast<double>(prepared.load()) : 0.0;
        i = inferred ? infer_ms / static_cast<double>(inferred.load()) : 0.0;
        d = presented ? display_ms / static_cast<double>(presented.load()) : 0.0;
    }
};

static bool has_help_option(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--help" || option == "-h") return true;
    }
    return false;
}

static bool find_config_path(int argc, char** argv, std::string& config_path) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) != "--config") continue;
        if (i + 1 >= argc) {
            std::cerr << "--config requires a JSON file path\n";
            return false;
        }
        config_path = argv[++i];
    }
    return true;
}

template <typename T>
static void read_config_value(const cv::FileNode& root, const char* key, T& value) {
    const cv::FileNode node = root[key];
    if (!node.empty()) node >> value;
}

static bool read_config_bool(const cv::FileNode& root, const char* key, bool& value) {
    const cv::FileNode node = root[key];
    if (node.empty()) return true;

    if (node.isInt() || node.isReal()) {
        double numeric = 0.0;
        node >> numeric;
        value = numeric != 0.0;
        return true;
    }
    if (node.isString()) {
        std::string text;
        node >> text;
        std::transform(text.begin(), text.end(), text.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (text == "true" || text == "1") {
            value = true;
            return true;
        }
        if (text == "false" || text == "0") {
            value = false;
            return true;
        }
    }

    std::cerr << "config " << key << " must be true or false\n";
    return false;
}

static bool load_config(const std::string& path, Args& a) {
    try {
        cv::FileStorage file(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
        if (!file.isOpened()) {
            std::cerr << "Cannot open JSON config: " << path << "\n";
            return false;
        }

        const cv::FileNode root = file.root();
        read_config_value(root, "model", a.model);
        const cv::FileNode camera = root["camera"];
        if (!camera.empty()) {
            if (camera.isString()) {
                camera >> a.device;
            } else if (camera.isInt() || camera.isReal()) {
                camera >> a.camera;
                a.device.clear();
            } else {
                std::cerr << "config camera must be a device path string such as /dev/video1 "
                             "or a numeric camera index\n";
                return false;
            }
        }
        read_config_value(root, "width", a.width);
        read_config_value(root, "height", a.height);
        read_config_value(root, "fps", a.fps);
        read_config_value(root, "intra_threads", a.intra_threads);
        read_config_value(root, "ep_affinity", a.ep_affinity);
        read_config_value(root, "conf", a.conf);
        read_config_value(root, "stable_frames", a.stable_frames);
        if (!read_config_bool(root, "rejudge_on_change", a.rejudge_on_change)) return false;
        read_config_value(root, "focus", a.focus);
        read_config_value(root, "zoom", a.zoom);
        if (!read_config_bool(root, "yolov8_enabled", a.yolov8_enabled)) return false;

        int queue_depth = static_cast<int>(a.queue_depth);
        read_config_value(root, "queue_depth", queue_depth);
        if (queue_depth < 0) {
            std::cerr << "config queue_depth must be >= 0\n";
            return false;
        }
        a.queue_depth = static_cast<std::size_t>(queue_depth);

        const cv::FileNode llm = root["llm"];
        if (!llm.empty()) {
            if (!llm.isMap()) {
                std::cerr << "config llm must be a JSON object\n";
                return false;
            }
            if (!read_config_bool(llm, "enabled", a.llm_enabled)) return false;
            read_config_value(llm, "url", a.llm_url);
            read_config_value(llm, "model", a.llm_model);
            read_config_value(llm, "timeout_seconds", a.llm_timeout_seconds);
            read_config_value(llm, "api_key", a.llm_api_key);
            read_config_value(llm, "system_prompt", a.llm_system_prompt);
            read_config_value(llm, "user_prompt_template", a.llm_user_prompt_template);
        }

        const cv::FileNode stream = root["stream"];
        if (!stream.empty()) {
            if (!stream.isMap()) {
                std::cerr << "config stream must be a JSON object\n";
                return false;
            }
            if (!read_config_bool(stream, "enabled", a.stream_enabled)) return false;
            read_config_value(stream, "host", a.stream_host);
            read_config_value(stream, "port", a.stream_port);
            read_config_value(stream, "jpeg_quality", a.stream_jpeg_quality);
        }
        a.config_path = path;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Failed to parse JSON config " << path << ": " << e.what() << "\n";
        return false;
    }
}

static void usage(const char* exe) {
    std::cout << "Usage: " << exe << " [options]\n"
              << "  --config PATH      JSON config file (default config.json)\n"
              << "  --model PATH       ONNX model; overrides config.json\n"
              << "  --camera VALUE     V4L2 index or device path, e.g. /dev/video1\n"
              << "  --device PATH      explicit V4L2 node, overrides --camera\n"
              << "  --width N --height N --fps N\n"
              << "  --conf FLOAT       confidence threshold\n"
              << "  --queue-depth N    keep up to N frames per pipeline queue\n"
              << "  --stable-frames N  valid YOLO frames required before LLM\n"
              << "  --rejudge-on-change restart stable check and LLM after dice change\n"
              << "  --no-rejudge-on-change keep one-shot-per-process behavior\n"
              << "  --focus N          fixed manual focus (-1 unchanged)\n"
              << "  --zoom N           zoom absolute value (-1 unchanged)\n"
              << "  --intra-threads N  SpaceMIT EP threads\n"
              << "  --ep-affinity LIST bind EP threads to cores, e.g. 14;15\n"
              << "  --no-display       run pipeline without window\n"
              << "  --max-frames N     stop after N frames enter preprocess (0=unlimited)\n"
              << "  --dump-input PATH  dump first preprocessed tensor as float32\n"
              << "  --no-yolov8        bypass preprocessing/inference and display camera frames only\n"
              << "  --self-test        initialize OpenCL GPU and model, run one inference\n"
              << "  --llm-url URL      override the config LLM base URL\n"
              << "  --llm-model NAME   override the config LLM model\n"
              << "  --llm-timeout N    LLM request timeout in seconds\n"
              << "  --no-llm           disable LLM verification and use stable YOLO result\n"
              << "  --stream           enable MJPEG HTTP streaming\n"
              << "  --stream-host HOST bind MJPEG server, default 0.0.0.0\n"
              << "  --stream-port N    MJPEG HTTP port, default 8080\n"
              << "  --stream-quality N JPEG quality 1-100, default 80\n"
              << "  --no-stream        disable MJPEG HTTP streaming\n";
}

static bool validate_args(Args& a) {
    a.queue_depth = std::clamp<std::size_t>(a.queue_depth, 1, 8);
    if (a.model.empty()) {
        std::cerr << "model path must not be empty\n";
        return false;
    }
    if (a.width <= 0 || a.height <= 0 || a.fps <= 0) {
        std::cerr << "width, height, and fps must all be > 0\n";
        return false;
    }
    if (a.conf < 0.0f || a.conf > 1.0f) {
        std::cerr << "confidence threshold must be between 0 and 1\n";
        return false;
    }
    if (a.stable_frames < 1) {
        std::cerr << "--stable-frames must be >= 1\n";
        return false;
    }
    if (a.llm_timeout_seconds < 1) {
        std::cerr << "--llm-timeout must be >= 1\n";
        return false;
    }
    if (a.stream_port < 1 || a.stream_port > 65535) {
        std::cerr << "stream port must be between 1 and 65535\n";
        return false;
    }
    if (a.stream_jpeg_quality < 1 || a.stream_jpeg_quality > 100) {
        std::cerr << "stream JPEG quality must be between 1 and 100\n";
        return false;
    }
    if (a.intra_threads < 1) {
        std::cerr << "--intra-threads must be >= 1\n";
        return false;
    }
    if (!a.ep_affinity.empty()) {
        std::size_t count = 1;
        for (const char c : a.ep_affinity) {
            if (c == ';') ++count;
            else if (c < '0' || c > '9') {
                std::cerr << "--ep-affinity must be a semicolon-separated list of core IDs, "
                             "for example 14;15\n";
                return false;
            }
        }
        if (a.ep_affinity.front() == ';' || a.ep_affinity.back() == ';' ||
            a.ep_affinity.find(";;") != std::string::npos) {
            std::cerr << "--ep-affinity contains an empty core ID\n";
            return false;
        }
        if (count != static_cast<std::size_t>(a.intra_threads)) {
            std::cerr << "--ep-affinity contains " << count
                      << " core IDs, but --intra-threads is " << a.intra_threads
                      << "; the counts must match\n";
            return false;
        }
    }
    if (a.llm_enabled && !a.no_llm) {
        if (a.llm_system_prompt.empty() || a.llm_user_prompt_template.empty()) {
            std::cerr << "config llm.system_prompt and llm.user_prompt_template "
                         "must not be empty\n";
            return false;
        }
        for (const char* token : {"{left_name}", "{right_name}",
                                  "{left_sum}", "{right_sum}"}) {
            if (a.llm_user_prompt_template.find(token) == std::string::npos) {
                std::cerr << "config llm.user_prompt_template is missing placeholder "
                          << token << "\n";
                return false;
            }
        }
    }
    return true;
}

static bool parse(int argc, char** argv, Args& a) {
    auto need = [&](int& i) -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
    bool device_override = false;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string k = argv[i];
            const char* v = nullptr;
            if (k == "--config" && (v = need(i))) a.config_path = v;
            else if (k == "--model" && (v = need(i))) a.model = v;
            else if (k == "--camera" && (v = need(i))) {
                const std::string camera_value = v;
                if (!device_override) {
                    if (!camera_value.empty() &&
                        camera_value.find_first_not_of("0123456789") == std::string::npos) {
                        a.camera = std::stoi(camera_value);
                        a.device.clear();
                    } else {
                        a.device = camera_value;
                    }
                }
            }
            else if (k == "--device" && (v = need(i))) {
                a.device = v;
                device_override = true;
            }
            else if (k == "--width" && (v = need(i))) a.width = std::stoi(v);
            else if (k == "--height" && (v = need(i))) a.height = std::stoi(v);
            else if (k == "--fps" && (v = need(i))) a.fps = std::stoi(v);
            else if (k == "--conf" && (v = need(i))) a.conf = std::stof(v);
            else if (k == "--queue-depth" && (v = need(i))) {
                a.queue_depth = static_cast<std::size_t>(std::stoul(v));
            } else if (k == "--stable-frames" && (v = need(i))) {
                a.stable_frames = std::stoi(v);
            } else if (k == "--rejudge-on-change") a.rejudge_on_change = true;
            else if (k == "--no-rejudge-on-change") a.rejudge_on_change = false;
            else if (k == "--focus" && (v = need(i))) a.focus = std::stoi(v);
            else if (k == "--zoom" && (v = need(i))) a.zoom = std::stoi(v);
            else if (k == "--intra-threads" && (v = need(i))) a.intra_threads = std::stoi(v);
            else if (k == "--ep-affinity" && (v = need(i))) a.ep_affinity = v;
            else if (k == "--max-frames" && (v = need(i))) a.max_frames = std::stoi(v);
            else if (k == "--no-display") a.no_display = true;
            else if (k == "--dump-input" && (v = need(i))) a.dump_input = v;
            else if (k == "--no-yolov8") a.yolov8_enabled = false;
            else if (k == "--self-test") a.self_test = true;
            else if (k == "--llm-url" && (v = need(i))) a.llm_url = v;
            else if (k == "--llm-model" && (v = need(i))) a.llm_model = v;
            else if (k == "--llm-timeout" && (v = need(i))) {
                a.llm_timeout_seconds = std::stoi(v);
            } else if (k == "--no-llm") a.no_llm = true;
            else if (k == "--stream") a.stream_enabled = true;
            else if (k == "--stream-host" && (v = need(i))) a.stream_host = v;
            else if (k == "--stream-port" && (v = need(i))) a.stream_port = std::stoi(v);
            else if (k == "--stream-quality" && (v = need(i))) a.stream_jpeg_quality = std::stoi(v);
            else if (k == "--no-stream") a.stream_enabled = false;
            else {
                std::cerr << "Unknown or incomplete option: " << k << "\n";
                usage(argv[0]);
                return false;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Invalid command-line value: " << e.what() << "\n";
        return false;
    }
    return validate_args(a);
}

// Ultralytics-style vivid palette. OpenCV colors are BGR.
static const std::array<cv::Scalar, 6> kClassColors = {
    cv::Scalar(56, 56, 255),    // red       #FF3838
    cv::Scalar(151, 157, 255),  // light red #FF9D97
    cv::Scalar(31, 112, 255),   // orange    #FF701F
    cv::Scalar(29, 178, 255),   // amber     #FFB21D
    cv::Scalar(49, 210, 207),   // yellow    #CFD231
    cv::Scalar(10, 249, 72),    // green     #48F90A
};

static bool is_tcm_resource_error(const std::string& message) {
    return message.find("tcm buffer acquire failed") != std::string::npos ||
           message.find("tcm buffer release failed") != std::string::npos ||
           message.find("wait tcm buffer failed") != std::string::npos;
}

static void print_tcm_resource_hint(const std::string& message,
                                    const std::string& ep_affinity) {
    std::cerr << "SpaceMIT EP TCM resource error: " << message << "\n"
              << "The requested EP affinity ("
              << (ep_affinity.empty() ? "runtime default" : ep_affinity)
              << ") does not directly identify the internal TCM block. "
              << "Another EP process or stale runtime state may still own the TCM. "
              << "Check owners with `spacemit-tcm-smi -i`; only when no EP process "
              << "is running, clear stale blocks with `spacemit-tcm-smi -c`.\n";
}

static void draw_detections(cv::Mat& bgr, const std::vector<Detection>& ds) {
    for (const auto& d : ds) {
        cv::Rect r(static_cast<int>(d.x1), static_cast<int>(d.y1),
                   std::max(1, static_cast<int>(d.x2 - d.x1)),
                   std::max(1, static_cast<int>(d.y2 - d.y1)));
        const size_t class_index =
            static_cast<size_t>(std::max(0, d.class_id)) % kClassColors.size();
        const cv::Scalar& color = kClassColors[class_index];

        cv::rectangle(bgr, r, color, 2, cv::LINE_AA);
        std::ostringstream label;
        label << (d.class_id + 1) << " " << std::fixed << std::setprecision(2)
              << d.confidence;
        int baseline = 0;
        const auto size = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX,
                                          .6, 1, &baseline);
        const int label_top = std::max(0, r.y - size.height - baseline - 8);
        const int label_bottom = std::max(size.height + baseline + 2, r.y);
        cv::rectangle(bgr, cv::Point(r.x, label_top),
                      cv::Point(r.x + size.width + 6, label_bottom),
                      color, cv::FILLED);

        // Pick a contrasting label color based on the palette luminance.
        const int brightness = static_cast<int>(0.114 * color[0] +
                                                0.587 * color[1] +
                                                0.299 * color[2]);
        const cv::Scalar text_color = brightness > 160
            ? cv::Scalar(0, 0, 0)
            : cv::Scalar(255, 255, 255);
        cv::putText(bgr, label.str(), cv::Point(r.x + 3, label_bottom - 4),
                    cv::FONT_HERSHEY_SIMPLEX, .6, text_color, 1, cv::LINE_AA);
    }
}

static bool detect_black_divider(const cv::Mat& bgr, DividerLine& divider) {
    if (bgr.empty()) return false;

    // The divider is a known piece of scene geometry: it is a long, nearly
    // vertical line close to the horizontal center of the camera view. Work
    // on a reduced image and search only the central corridor so unrelated
    // dark objects at the bottom/edges cannot win by having a large area.
    constexpr double kDividerScale = 0.25;
    cv::Mat small;
    cv::resize(bgr, small, cv::Size(), kDividerScale, kDividerScale, cv::INTER_AREA);
    cv::Mat gray;
    cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);

    constexpr int kDividerGrayThreshold = 45;
    cv::Mat dark;
    cv::inRange(gray, cv::Scalar(0), cv::Scalar(kDividerGrayThreshold), dark);

    const int roi_x = static_cast<int>(0.30 * small.cols);
    const int roi_width = static_cast<int>(0.40 * small.cols);
    const int roi_y = static_cast<int>(0.05 * small.rows);
    const int roi_height = static_cast<int>(0.90 * small.rows);
    const cv::Rect roi(roi_x, roi_y, roi_width, roi_height);
    cv::Mat central = cv::Mat::zeros(dark.size(), dark.type());
    dark(roi).copyTo(central(roi));

    // A vertical kernel repairs small breaks along the expected line without
    // joining a long horizontal mark below the dice area.
    const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(3, 9));
    cv::morphologyEx(central, central, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(central, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const int min_height = std::max(20, static_cast<int>(0.55 * roi.height));
    const float center_x = 0.5f * static_cast<float>(small.cols);
    const std::vector<cv::Point>* best = nullptr;
    double best_score = 0.0;
    for (const auto& contour : contours) {
        if (contour.size() < 5) continue;
        const cv::Rect bounds = cv::boundingRect(contour);
        const double area = cv::contourArea(contour);
        if (bounds.height < min_height || area < 80.0) continue;

        // Reject horizontal marks and diagonals. The fitted direction is more
        // stable than comparing bounding-box width/height for broken lines.
        cv::Vec4f fitted;
        cv::fitLine(contour, fitted, cv::DIST_L2, 0.0, 0.01, 0.01);
        const float direction_norm = std::sqrt(fitted[0] * fitted[0] +
                                                fitted[1] * fitted[1]);
        if (direction_norm < 1e-4f) continue;
        const float verticality = std::abs(fitted[1]) / direction_norm;
        if (verticality < 0.90f) continue;

        const float contour_center_x = bounds.x + 0.5f * bounds.width;
        const float center_error = std::abs(contour_center_x - center_x) /
                                   std::max(1.0f, 0.5f * small.cols);
        // Keep the line near the middle even inside the central corridor.
        if (center_error > 0.20f) continue;

        const double narrowness = static_cast<double>(bounds.height) /
                                  std::max(1, bounds.width);
        const double center_score = 1.0 - center_error;
        const double score = area * std::min(narrowness, 20.0) * center_score;
        if (score > best_score) {
            best_score = score;
            best = &contour;
        }
    }
    if (!best) return false;

    cv::Vec4f fitted;
    cv::fitLine(*best, fitted, cv::DIST_L2, 0.0, 0.01, 0.01);
    cv::Point2f direction(fitted[0], fitted[1]);
    const float norm = std::sqrt(direction.dot(direction));
    if (norm < 1e-4f) return false;
    direction *= (1.0f / norm);
    if (direction.y < 0.0f) direction *= -1.0f;

    cv::Point2f normal(-direction.y, direction.x);
    if (normal.x < 0.0f) normal *= -1.0f;

    divider.valid = true;
    divider.horizontal = false;
    divider.point = cv::Point2f(
        fitted[2] / static_cast<float>(kDividerScale),
        fitted[3] / static_cast<float>(kDividerScale));
    divider.direction = direction;
    divider.normal = normal;
    return true;
}

static DiceJudgment judge_dice(const cv::Mat& bgr, const std::vector<Detection>& detections) {
    DiceJudgment result;
    result.sides.assign(detections.size(), -1);
    result.divider_found = detect_black_divider(bgr, result.divider);
    if (!result.divider_found) {
        result.message = "未检测到中间黑色分界线，未执行骰子点数求和判断";
        result.overlay = "INVALID: black divider not found";
        return result;
    }

    result.horizontal_divider = result.divider.horizontal;
    if (result.horizontal_divider) {
        result.first_name = "UPPER";
        result.second_name = "LOWER";
    }
    const float tolerance = std::max(4.0f, 0.005f * std::min(bgr.cols, bgr.rows));
    for (size_t i = 0; i < detections.size(); ++i) {
        const Detection& d = detections[i];
        if (d.class_id < 0 || d.class_id >= 6) continue;
        const cv::Point2f center((d.x1 + d.x2) * 0.5f, (d.y1 + d.y2) * 0.5f);
        const cv::Point2f delta = center - result.divider.point;
        const float signed_distance = delta.dot(result.divider.normal);
        if (std::abs(signed_distance) <= tolerance) continue;
        const int side = signed_distance < 0.0f ? 0 : 1;
        result.sides[i] = side;
        const int value = d.class_id + 1;
        if (side == 0) {
            ++result.first_count;
            result.first_sum += value;
            result.first_values.push_back(value);
        } else {
            ++result.second_count;
            result.second_sum += value;
            result.second_values.push_back(value);
        }
    }
    std::sort(result.first_values.begin(), result.first_values.end());
    std::sort(result.second_values.begin(), result.second_values.end());

    if (result.first_count != 5 || result.second_count != 5) {
        std::ostringstream message;
        message << "两侧骰子数量必须都达到5个，当前 " << result.first_name << "="
                << result.first_count << ", " << result.second_name << "="
                << result.second_count << "，未执行求和判断";
        result.message = message.str();
        result.overlay = "INVALID: " + result.first_name + "/" + result.second_name +
                         " must both contain 5 dice (" + std::to_string(result.first_count) +
                         "/" + std::to_string(result.second_count) + ")";
        return result;
    }

    result.valid = true;
    std::string winner = "TIE";
    if (result.first_sum > result.second_sum) winner = result.first_name + " WINS";
    if (result.second_sum > result.first_sum) winner = result.second_name + " WINS";
    std::ostringstream message;
    message << result.first_name << "总和=" << result.first_sum << "，"
            << result.second_name << "总和=" << result.second_sum << "，结果：";
    if (winner == "TIE") message << "平局";
    else message << (result.first_sum > result.second_sum ? result.first_name : result.second_name)
                 << "获胜";
    result.message = message.str();
    result.overlay = result.first_name + "=" + std::to_string(result.first_sum) +
                     "  " + result.second_name + "=" + std::to_string(result.second_sum) +
                     "  " + winner;
    return result;
}

static void draw_divider_and_judgment(cv::Mat& bgr, const DiceJudgment& judgment) {
    if (judgment.divider_found) {
        const cv::Point2f p = judgment.divider.point;
        const cv::Point2f v = judgment.divider.direction * 2000.0f;
        const cv::Point p1(static_cast<int>(p.x - v.x), static_cast<int>(p.y - v.y));
        const cv::Point p2(static_cast<int>(p.x + v.x), static_cast<int>(p.y + v.y));
        cv::line(bgr, p1, p2, cv::Scalar(0, 165, 255), 3, cv::LINE_AA);
    }
    const cv::Scalar color = judgment.valid ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 0, 255);
    // Keep the judgment text directly on the image instead of drawing a solid
    // black panel in the top-left corner. A thin shadow preserves readability
    // without leaving the unwanted black background/bottom edge.
    const cv::Point text_origin(18, 48);
    cv::putText(bgr, judgment.overlay, text_origin, cv::FONT_HERSHEY_SIMPLEX,
                .7, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
    cv::putText(bgr, judgment.overlay, text_origin, cv::FONT_HERSHEY_SIMPLEX,
                .7, color, 2, cv::LINE_AA);
}

static LlmWinner yolo_winner(const DiceJudgment& judgment) {
    if (judgment.first_sum > judgment.second_sum) return LlmWinner::Left;
    if (judgment.second_sum > judgment.first_sum) return LlmWinner::Right;
    return LlmWinner::Tie;
}

struct DiceResultSnapshot {
    int first_count = 0;
    int second_count = 0;
    int first_sum = 0;
    int second_sum = 0;
    std::vector<int> first_values;
    std::vector<int> second_values;
    std::string first_name;
    std::string second_name;
    bool horizontal_divider = false;
    LlmWinner winner = LlmWinner::Unknown;
};

static DiceResultSnapshot make_dice_snapshot(const DiceJudgment& judgment) {
    DiceResultSnapshot snapshot;
    snapshot.first_count = judgment.first_count;
    snapshot.second_count = judgment.second_count;
    snapshot.first_sum = judgment.first_sum;
    snapshot.second_sum = judgment.second_sum;
    snapshot.first_values = judgment.first_values;
    snapshot.second_values = judgment.second_values;
    snapshot.first_name = judgment.first_name;
    snapshot.second_name = judgment.second_name;
    snapshot.horizontal_divider = judgment.horizontal_divider;
    snapshot.winner = yolo_winner(judgment);
    return snapshot;
}

static bool same_dice_snapshot(const DiceResultSnapshot& lhs,
                               const DiceResultSnapshot& rhs) {
    return lhs.first_count == rhs.first_count &&
           lhs.second_count == rhs.second_count &&
           lhs.first_sum == rhs.first_sum &&
           lhs.second_sum == rhs.second_sum &&
           lhs.first_values == rhs.first_values &&
           lhs.second_values == rhs.second_values &&
           lhs.first_name == rhs.first_name &&
           lhs.second_name == rhs.second_name &&
           lhs.horizontal_divider == rhs.horizontal_divider &&
           lhs.winner == rhs.winner;
}

static const char* winner_label(LlmWinner winner) {
    switch (winner) {
    case LlmWinner::Left: return "LEFT";
    case LlmWinner::Right: return "RIGHT";
    case LlmWinner::Tie: return "TIE";
    default: return "UNKNOWN";
    }
}

struct LlmRequest {
    uint64_t generation = 0;
    DiceResultSnapshot snapshot;
    std::string first_name;
    std::string second_name;
    int first_sum = 0;
    int second_sum = 0;
};

struct LlmResponse {
    uint64_t generation = 0;
    LlmVerificationResult result = LlmVerificationResult::Failure;
    LlmWinner winner = LlmWinner::Unknown;
    std::string error;
};

// Runs the potentially slow network request away from the HighGUI/display
// thread. There is one active request and at most one queued request; when a
// new stable scene is found while a request is running, only the latest scene
// is retained for the next request.
class AsyncLlmVerifier {
public:
    explicit AsyncLlmVerifier(const LlmDiceVerifier& verifier) : verifier_(verifier) {}
    ~AsyncLlmVerifier() { stop(); }

    void start() {
        worker_ = std::thread(&AsyncLlmVerifier::run, this);
    }

    bool submit(LlmRequest request) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) return false;
            queued_request_ = std::move(request);
        }
        cv_.notify_one();
        return true;
    }

    bool tryPop(LlmResponse& response) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (responses_.empty()) return false;
        response = std::move(responses_.front());
        responses_.pop_front();
        return true;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            queued_request_.reset();
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

private:
    void run() {
        while (true) {
            LlmRequest request;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] { return stopping_ || queued_request_.has_value(); });
                if (stopping_ && !queued_request_) return;
                request = std::move(*queued_request_);
                queued_request_.reset();
            }

            LlmResponse response;
            response.generation = request.generation;
            response.result = verifier_.verify_once(
                request.first_name, request.second_name,
                request.first_sum, request.second_sum,
                response.winner, response.error);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                responses_.push_back(std::move(response));
            }
        }
    }

    const LlmDiceVerifier& verifier_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::optional<LlmRequest> queued_request_;
    std::deque<LlmResponse> responses_;
    std::thread worker_;
    bool stopping_ = false;
};


} // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    Args a;
    if (has_help_option(argc, argv)) {
        usage(argv[0]);
        return 0;
    }
    if (!find_config_path(argc, argv, a.config_path)) return 2;
    if (!load_config(a.config_path, a)) return 2;
    if (!parse(argc, argv, a)) return argc > 1 ? 2 : 0;
    if (const char* env_url = std::getenv("DICE_LLM_URL")) {
        if (a.llm_url == "https://api.rvcompute.com:60000/v1") a.llm_url = env_url;
    }
    if (const char* env_model = std::getenv("DICE_LLM_MODEL")) {
        if (a.llm_model == "gpt-5.4-mini") a.llm_model = env_model;
    }
    if (const char* env_key = std::getenv("DICE_LLM_API_KEY")) {
        a.llm_api_key = env_key;
        // The verifier passes the secret to curl through a private pipe. Do
        // not leave it in the environment inherited by later child processes.
        ::unsetenv("DICE_LLM_API_KEY");
    }
    if (a.yolov8_enabled && !std::filesystem::exists(a.model)) {
        std::cerr << "Model not found: " << a.model << "\n";
        return 2;
    }

    std::unique_ptr<OpenClPreprocessor> pre;
    std::unique_ptr<Yolov8Detector> detector;
    if (a.yolov8_enabled) {
        pre = std::make_unique<OpenClPreprocessor>();
        if (!pre->init()) return 4;
        detector = std::make_unique<Yolov8Detector>();
        if (!detector->init(a.model, a.intra_threads, a.ep_affinity)) return 5;
    } else {
        // Camera-only mode intentionally avoids all YOLOv8/OpenCL/ORT setup.
        std::cerr << "[YOLOv8] disabled; displaying camera frames without preprocessing or inference.\n";
    }
    LlmDiceVerifier llm_verifier({a.llm_url, a.llm_api_key, a.llm_model,
                                  a.llm_timeout_seconds,
                                  a.llm_system_prompt, a.llm_user_prompt_template});
    const bool llm_enabled = a.yolov8_enabled && a.llm_enabled && !a.no_llm;
    if (!a.yolov8_enabled && a.llm_enabled) {
        std::cerr << "[LLM] disabled because YOLOv8 is disabled.\n";
    } else if (!llm_enabled) {
        std::cerr << "[LLM] disabled; a stable YOLO result will directly determine the winner.\n";
    } else if (!llm_verifier.configured()) {
        std::cerr << "[LLM] verification disabled: set llm.api_key in config.json "
                     "or DICE_LLM_API_KEY.\n";
    }
    if (a.self_test) {
        if (!a.yolov8_enabled) {
            std::cerr << "--self-test requires yolov8_enabled=true\n";
            return 2;
        }
        try {
            // Exercise the actual OpenCL NV12 -> tensor path as well as the
            // SpaceMIT EP model path. This catches kernel/image/queue errors
            // that a model-only zero-tensor test cannot detect.
            constexpr int synthetic_width = 1280;
            constexpr int synthetic_height = 720;
            cv::Mat synthetic_nv12(synthetic_height * 3 / 2, synthetic_width,
                                   CV_8UC1, cv::Scalar(128));
            const auto prep_result = pre->preprocess(synthetic_nv12);
            if (!prep_result.data || prep_result.data->size() != 3 * 640 * 640) {
                throw std::runtime_error("OpenCL self-test returned an invalid tensor");
            }
            for (float value : *prep_result.data) {
                if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
                    throw std::runtime_error("OpenCL self-test returned invalid tensor values");
                }
            }
            const auto ds = detector->infer(prep_result.data->data(), prep_result.data->size(),
                                           a.conf, prep_result.scale, prep_result.pad_x,
                                           prep_result.pad_y, synthetic_width, synthetic_height);
            std::cout << "Self-test passed: OpenCL preprocess " << prep_result.ms
                      << " ms, " << ds.size() << " detections.\n";
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "Self-test failed: " << e.what() << "\n";
            return 6;
        }
    }

    GstreamerMjpegCamera camera;
    if (!camera.open(a.camera, a.device, a.width, a.height, a.fps,
                     a.focus, a.zoom, a.max_frames)) {
        std::cerr << "Camera open failed.\n";
        return 3;
    }
    MjpegStreamer streamer;
    if (a.stream_enabled && !streamer.start(a.stream_host, a.stream_port,
                                             a.stream_jpeg_quality)) {
        camera.close();
        return 7;
    }

    if (!a.no_display) {
        try {
            cv::namedWindow("yolov8-k3", cv::WINDOW_NORMAL);
            cv::resizeWindow("yolov8-k3", a.width, a.height);
        } catch (const cv::Exception& e) {
            std::cerr << "Display initialization failed: " << e.what() << "\n";
            streamer.stop();
            camera.close();
            return 7;
        }
    }

    if (!a.yolov8_enabled) {
        const auto direct_start = Clock::now();
        int frame_count = 0;
        while (!g_signal_stop && (a.max_frames <= 0 || frame_count < a.max_frames)) {
            GstreamerFrame frame;
            if (!camera.read(frame, 1000)) continue;
            ++frame_count;
            if (a.no_display && !streamer.running()) continue;
            cv::Mat bgr;
            if (frame.nv12.empty()) continue;
            cv::cvtColor(frame.nv12, bgr, cv::COLOR_YUV2BGR_NV12);
            streamer.publish(bgr);
            if (a.no_display) continue;
            cv::imshow("yolov8-k3", bgr);
            const int key = cv::waitKey(1) & 0xff;
            if (key == 'q' || key == 27) break;
        }
        streamer.stop();
        camera.close();
        if (!a.no_display) cv::destroyAllWindows();
        const double elapsed = std::chrono::duration<double>(Clock::now() - direct_start).count();
        std::cout << "Done. camera-only frames=" << frame_count
                  << " elapsed_s=" << elapsed
                  << " fps=" << frame_count / std::max(.001, elapsed) << "\n";
        return 0;
    }

    std::atomic<bool> abort{false};
    std::atomic<bool> preprocess_done{false};
    std::atomic<bool> inference_done{false};
    std::atomic<bool> inference_failed{false};
    std::atomic<bool> camera_stop{false};
    FrameQueue<PreparedFrame> prepared_queue(a.queue_depth);
    FrameQueue<InferenceResult> result_queue(a.queue_depth);
    Stats stats;
    const auto start = Clock::now();

    // Thread 1: OpenCV VideoCapture -> GStreamer -> spacemitdec/jpegdec -> NV12,
    // followed by OpenCL GPU preprocessing. appsink keeps only the newest frame.
    std::thread preprocess_thread([&] {
        uint64_t id = 0;
        int timeout_count = 0;
        while (!abort.load() && !camera_stop.load() &&
               (a.max_frames <= 0 || static_cast<int>(id) < a.max_frames)) {
            GstreamerFrame frame;
            if (!camera.read(frame, 1000)) {
                if (abort.load()) break;
                if (++timeout_count >= 10) {
                    std::cerr << "Camera read timeout/error in GStreamer stage\n";
                    abort.store(true);
                    break;
                }
                continue;
            }
            timeout_count = 0;
            auto packet = std::make_shared<PreparedFrame>();
            packet->id = id++;
            packet->width = frame.nv12.cols;
            packet->height = (frame.nv12.rows * 2) / 3;
            packet->nv12 = std::make_shared<cv::Mat>(std::move(frame.nv12));
            packet->gst_owner = std::move(frame.owner);
            try {
                const auto t0 = Clock::now();
                packet->prep = pre->preprocess(*packet->nv12);
                if (!a.dump_input.empty() && packet->id == 0) {
                    std::ofstream dump(a.dump_input, std::ios::binary);
                    if (!dump) throw std::runtime_error("cannot open --dump-input path");
                    dump.write(reinterpret_cast<const char*>(packet->prep.data->data()),
                               static_cast<std::streamsize>(packet->prep.data->size() * sizeof(float)));
                }
                stats.addPre(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                if (prepared_queue.push(std::move(packet))) stats.dropped_pre.fetch_add(1);
                stats.dropped_pre.fetch_add(prepared_queue.takeDroppedPending());
                stats.prepared.fetch_add(1);
            } catch (const std::exception& e) {
                std::cerr << "Preprocess stage failed: " << e.what() << "\n";
                abort.store(true);
                break;
            }
        }
        preprocess_done.store(true);
        prepared_queue.close();
    });

    // Thread 2: one SpaceMIT EP session; only this thread touches detector.
    std::thread inference_thread([&] {
        uint64_t id = 0;
        std::shared_ptr<PreparedFrame> packet;
        while (prepared_queue.popLatest(packet, abort)) {
            if (!packet) continue;
            try {
                const auto t0 = Clock::now();
                auto result = std::make_shared<InferenceResult>();
                result->id = packet->id;
                result->width = packet->width;
                result->height = packet->height;
                result->nv12 = packet->nv12;
                result->gst_owner = packet->gst_owner;
                try {
                    result->detections = detector->infer(packet->prep.data->data(), packet->prep.data->size(),
                                                        a.conf, packet->prep.scale, packet->prep.pad_x,
                                                        packet->prep.pad_y, packet->width, packet->height);
                } catch (const std::exception& e) {
                    if (is_tcm_resource_error(e.what())) {
                        print_tcm_resource_hint(e.what(), a.ep_affinity);
                    }
                    throw;
                }
                stats.addInfer(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                stats.inferred.fetch_add(1);
                if (!result->detections.empty()) stats.detected_frames.fetch_add(1);
                stats.detections.fetch_add(result->detections.size());
                if (result_queue.push(std::move(result))) stats.dropped_result.fetch_add(1);
                stats.dropped_result.fetch_add(result_queue.takeDroppedPending());
                ++id;
            } catch (const std::exception& e) {
                std::cerr << "Inference stage failed: " << e.what() << "\n";
                inference_failed.store(true);
                abort.store(true);
                break;
            }
        }
        inference_done.store(true);
        result_queue.close();
    });

    // Thread 3 is the display stage logically; it runs on the main/UI thread
    // because OpenCV HighGUI on this board must own the X11 event loop here.
    struct LlmVerificationState {
        // Each initial or changed result must remain identical for the
        // configured number of consecutive valid 5+5 frames.
        int stable_count = 0;
        bool have_stable_candidate = false;
        DiceResultSnapshot stable_candidate;
        std::string last_stability_message;

        bool attempted = false;
        bool succeeded = false;
        bool agreement = false;
        bool timeout_fallback = false;
        bool printed = false;
        DiceResultSnapshot attempted_snapshot;
        LlmWinner llm_winner = LlmWinner::Unknown;
        uint64_t next_generation = 0;
        uint64_t attempted_generation = 0;
        uint64_t active_generation = 0;
        bool request_in_flight = false;
        bool request_completed = false;
        bool fallback_verdict_available = false;
        std::optional<LlmRequest> deferred_request;
    } llm_state;

    std::unique_ptr<AsyncLlmVerifier> async_llm;
    if (llm_enabled) {
        async_llm = std::make_unique<AsyncLlmVerifier>(llm_verifier);
        async_llm->start();
    }

    bool first_display = true;
    uint64_t shown = 0;
    auto last_report = start;
    auto drain_llm_responses = [&] {
        if (!async_llm) return;
        LlmResponse response;
        while (async_llm->tryPop(response)) {
            if (response.generation == llm_state.active_generation) {
                llm_state.request_in_flight = false;
            }
            if (response.generation == llm_state.attempted_generation) {
                llm_state.request_completed = true;
                if (response.result == LlmVerificationResult::Success) {
                    llm_state.succeeded = true;
                    llm_state.agreement =
                        response.winner == llm_state.attempted_snapshot.winner;
                    llm_state.llm_winner = response.winner;
                    std::cout << "[LLM] one-shot result="
                              << winner_label(response.winner) << "\n";
                } else if (response.result == LlmVerificationResult::Timeout) {
                    llm_state.timeout_fallback = true;
                    llm_state.succeeded = true;
                    llm_state.agreement = true;
                    llm_state.llm_winner = llm_state.attempted_snapshot.winner;
                    llm_state.fallback_verdict_available = true;
                    std::cerr << "[LLM] one-shot verification timed out: "
                              << response.error << "; using the stable YOLO result\n";
                } else {
                    std::cerr << "[LLM] one-shot verification failed: "
                              << response.error << "\n";
                }
            }
        }
        if (!llm_state.request_in_flight && llm_state.deferred_request) {
            const uint64_t generation = llm_state.deferred_request->generation;
            if (async_llm->submit(std::move(*llm_state.deferred_request))) {
                llm_state.deferred_request.reset();
                llm_state.request_in_flight = true;
                llm_state.active_generation = generation;
                std::cout << "[LLM] queued latest stable result for verification\n";
            }
        }
    };

    while (true) {
        if (g_signal_stop) abort.store(true);
        std::shared_ptr<InferenceResult> item;
        if (!result_queue.popLatest(item, abort, first_display, a.queue_depth)) {
            drain_llm_responses();
            if (g_signal_stop) abort.store(true);
            if (inference_done.load()) break;
            if (abort.load()) break;
            continue;
        }
        first_display = false;
        drain_llm_responses();
        if (!item) continue;

        const auto t0 = Clock::now();
        cv::Mat bgr;
        DiceJudgment judgment;
        if (item->nv12 && !item->nv12->empty()) {
            cv::cvtColor(*item->nv12, bgr, cv::COLOR_YUV2BGR_NV12);
            judgment = judge_dice(bgr, item->detections);
            DiceJudgment display_judgment = judgment;
            static std::string last_judgment;
            if (!judgment.valid) {
                llm_state.stable_count = 0;
                llm_state.have_stable_candidate = false;
                if (judgment.message != last_judgment) {
                    std::cout << "Dice judgment: " << judgment.message << "\n";
                    last_judgment = judgment.message;
                }
            } else {
                const DiceResultSnapshot current_snapshot = make_dice_snapshot(judgment);
                const bool matches_attempted =
                    llm_state.attempted &&
                    same_dice_snapshot(current_snapshot, llm_state.attempted_snapshot);
                const bool needs_stability =
                    !llm_state.attempted ||
                    (a.rejudge_on_change && !matches_attempted);
                bool waiting_for_stability = false;

                if (llm_state.attempted && a.rejudge_on_change && matches_attempted) {
                    // A transient changed frame must not trigger another LLM
                    // call if the scene returns to the already judged result.
                    llm_state.stable_count = 0;
                    llm_state.have_stable_candidate = false;
                    llm_state.last_stability_message.clear();
                } else if (needs_stability) {
                    const bool same_candidate =
                        llm_state.have_stable_candidate &&
                        same_dice_snapshot(current_snapshot, llm_state.stable_candidate);
                    if (same_candidate) {
                        ++llm_state.stable_count;
                    } else {
                        llm_state.have_stable_candidate = true;
                        llm_state.stable_count = 1;
                        llm_state.stable_candidate = current_snapshot;
                        llm_state.last_stability_message.clear();
                        if (llm_state.attempted && a.rejudge_on_change) {
                            std::cout << "[YOLO] dice result changed; restarting stable check\n";
                        }
                    }

                    if (llm_state.stable_count < a.stable_frames) {
                        waiting_for_stability = true;
                        display_judgment.valid = false;
                        display_judgment.overlay =
                            std::string(llm_state.attempted ? "Changed result stable check "
                                                            : "YOLO stable check ") +
                            std::to_string(llm_state.stable_count) + "/" +
                            std::to_string(a.stable_frames);
                        const std::string stability_message =
                            std::string(llm_state.attempted ? "[YOLO] changed 5+5 result "
                                                            : "[YOLO] stable 5+5 result ") +
                            std::to_string(llm_state.stable_count) + "/" +
                            std::to_string(a.stable_frames) + ": " + judgment.message;
                        if (stability_message != llm_state.last_stability_message) {
                            std::cout << stability_message << "\n";
                            llm_state.last_stability_message = stability_message;
                        }
                    } else {
                        // A stable changed snapshot starts a new one-shot LLM
                        // cycle. The previous response is never reused.
                        const bool is_rejudgment = llm_state.attempted;
                        llm_state.attempted = true;
                        llm_state.succeeded = false;
                        llm_state.agreement = false;
                        llm_state.timeout_fallback = false;
                        llm_state.request_completed = false;
                        llm_state.fallback_verdict_available = false;
                        llm_state.printed = false;
                        llm_state.attempted_snapshot = current_snapshot;
                        llm_state.llm_winner = LlmWinner::Unknown;

                        if (!llm_enabled) {
                            llm_state.succeeded = true;
                            llm_state.agreement = true;
                            llm_state.llm_winner = llm_state.attempted_snapshot.winner;
                            std::cout << "[YOLO] "
                                      << (is_rejudgment ? "changed result" : "stable result")
                                      << " reached " << a.stable_frames
                                      << " consecutive frames; using YOLO result directly: "
                                      << winner_label(llm_state.llm_winner) << "\n";
                        } else {
                            LlmRequest request;
                            request.generation = ++llm_state.next_generation;
                            request.snapshot = current_snapshot;
                            request.first_name = judgment.first_name;
                            request.second_name = judgment.second_name;
                            request.first_sum = judgment.first_sum;
                            request.second_sum = judgment.second_sum;
                            llm_state.attempted_generation = request.generation;
                            std::cout << "[YOLO] "
                                      << (is_rejudgment ? "changed result" : "stable result")
                                      << " reached " << a.stable_frames
                                      << " consecutive frames; scheduling LLM verification\n";
                            if (!llm_state.request_in_flight &&
                                async_llm->submit(std::move(request))) {
                                llm_state.request_in_flight = true;
                                llm_state.active_generation = llm_state.attempted_generation;
                            } else {
                                llm_state.deferred_request = std::move(request);
                                std::cout << "[LLM] verification already running; keeping latest result\n";
                            }
                        }

                        llm_state.stable_count = 0;
                        llm_state.have_stable_candidate = false;
                        llm_state.last_stability_message.clear();
                    }
                }

                if (!waiting_for_stability) {
                    const bool same_snapshot =
                        llm_state.attempted &&
                        same_dice_snapshot(current_snapshot, llm_state.attempted_snapshot);
                    if (llm_enabled && !llm_state.request_completed &&
                        llm_state.request_in_flight && same_snapshot) {
                        display_judgment.valid = false;
                        display_judgment.overlay = "LLM verification pending";
                    } else if (llm_enabled && llm_state.fallback_verdict_available &&
                               llm_state.timeout_fallback && same_snapshot) {
                        // OpenCV Hershey fonts cannot render UTF-8 Chinese text.
                        // Keep the image overlay ASCII-only; the terminal log below
                        // retains the Chinese judgment.message for operators.
                        display_judgment.overlay =
                            "YOLO fallback (LLM timeout): " + judgment.overlay;
                        if (!llm_state.printed) {
                            std::cout << "Dice judgment by YOLO fallback (LLM timeout): "
                                      << judgment.message << "\n";
                            llm_state.printed = true;
                        }
                    } else if (llm_state.succeeded && llm_state.agreement && same_snapshot) {
                        if (!llm_state.printed) {
                            if (llm_enabled) {
                                std::cout << "Dice judgment verified by YOLO + LLM: "
                                          << judgment.message << "\n";
                            } else {
                                std::cout << "Dice judgment by YOLO: "
                                          << judgment.message << "\n";
                            }
                            llm_state.printed = true;
                        }
                    } else {
                        display_judgment.valid = false;
                        if (!llm_state.attempted ||
                            (!llm_state.succeeded && !llm_state.timeout_fallback)) {
                            display_judgment.overlay =
                                "LLM verification unavailable; winner suppressed";
                        } else if (llm_state.succeeded && !llm_state.agreement) {
                            display_judgment.overlay =
                                "LLM/YOLO mismatch; winner suppressed";
                        } else {
                            display_judgment.overlay =
                                "Frame differs from verified snapshot; winner suppressed";
                        }
                    }
                }
            }
            if (!a.no_display || streamer.running()) {
                draw_detections(bgr, item->detections);
                draw_divider_and_judgment(bgr, display_judgment);
                const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
                const double fps = elapsed > 0.0 ? shown / elapsed : 0.0;
                cv::putText(bgr, "DISPLAY " + std::to_string(static_cast<int>(fps)) +
                                      " FPS  DET " + std::to_string(item->detections.size()),
                            {10, 96}, cv::FONT_HERSHEY_SIMPLEX, .75,
                            {0, 255, 255}, 2, cv::LINE_AA);
                streamer.publish(bgr);
                if (!a.no_display) {
                    cv::imshow("yolov8-k3", bgr);
                    const int key = cv::waitKey(1) & 0xff;
                    if (key == 'q' || key == 27) abort.store(true);
                }
            }
        } else {
            std::cerr << "Display frame transfer/YUV conversion failed\n";
        }
        stats.addDisplay(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        stats.presented.fetch_add(1);
        ++shown;

        const auto now = Clock::now();
        if (now - last_report >= std::chrono::seconds(2)) {
            const double elapsed = std::chrono::duration<double>(now - start).count();
            double p = 0.0, i = 0.0, d = 0.0;
            stats.averages(p, i, d);
            std::cout << "pipeline prepared=" << stats.prepared.load()
                      << " infer=" << stats.inferred.load()
                      << " display=" << stats.presented.load()
                      << " fps_pre=" << stats.prepared.load() / std::max(.001, elapsed)
                      << " fps_infer=" << stats.inferred.load() / std::max(.001, elapsed)
                      << " fps_display=" << stats.presented.load() / std::max(.001, elapsed)
                      << " dropped_pre=" << stats.dropped_pre.load()
                      << " dropped_result=" << stats.dropped_result.load()
                      << " detected_frames=" << stats.detected_frames.load()
                      << " detections=" << stats.detections.load()
                      << " pre_ms=" << p << " infer_ms=" << i << " display_ms=" << d << "\n";
            last_report = now;
        }
        if (abort.load()) break;
    }

    // Ctrl-C/q/exception: stop capture first, then join workers and release
    // every frame owner before draining and destroying the GStreamer pipeline.
    camera_stop.store(true);
    abort.store(true);
    prepared_queue.close();
    result_queue.close();
    if (preprocess_thread.joinable()) preprocess_thread.join();
    if (inference_thread.joinable()) inference_thread.join();
    prepared_queue.clear();
    result_queue.clear();
    if (async_llm) async_llm->stop();
    streamer.stop();
    camera.close();
    if (!a.no_display) cv::destroyAllWindows();

    const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    double p = 0.0, i = 0.0, d = 0.0;
    stats.averages(p, i, d);
    std::cout << "Done. prepared=" << stats.prepared.load()
              << " infer=" << stats.inferred.load()
              << " display=" << stats.presented.load()
              << " elapsed_s=" << elapsed
              << " fps_pre=" << stats.prepared.load() / std::max(.001, elapsed)
              << " fps_infer=" << stats.inferred.load() / std::max(.001, elapsed)
              << " fps_display=" << stats.presented.load() / std::max(.001, elapsed)
              << " dropped_pre=" << stats.dropped_pre.load()
              << " dropped_result=" << stats.dropped_result.load()
              << " detected_frames=" << stats.detected_frames.load()
              << " detections=" << stats.detections.load()
              << " pre_ms=" << p << " infer_ms=" << i << " display_ms=" << d << "\n";
    if (inference_failed.load()) {
        std::cerr << "Pipeline stopped because the inference stage failed; no further frames were processed.\n";
        return 8;
    }
    return 0;
}
