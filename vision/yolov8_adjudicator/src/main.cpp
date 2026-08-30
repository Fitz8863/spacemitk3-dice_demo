#include "gstreamer_camera.h"
#include "rtsp_streamer.h"
#include "opencl_preprocess.h"
#include "yolov8_detector.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
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
#include <cerrno>
#include <unistd.h>
#include <poll.h>



namespace {
using Clock = std::chrono::steady_clock;
volatile sig_atomic_t g_signal_stop = 0;
int g_event_fd = -1;
std::mutex g_event_mutex;
void on_signal(int) { g_signal_stop = 1; }

void emit_event(const std::string& json) {
    std::lock_guard<std::mutex> lock(g_event_mutex);
    const std::string line = json + "\n";
    // When launched by the Python package without an inherited event pipe,
    // stdout is the event transport.  Keep this fallback JSONL-only so the
    // resident adapter can consume events directly from Popen.stdout.
    if (g_event_fd < 0) {
        std::cout << line << std::flush;
        return;
    }
    std::size_t offset = 0;
    while (offset < line.size()) {
        const ssize_t written = ::write(g_event_fd, line.data() + offset, line.size() - offset);
        if (written > 0) {
            offset += static_cast<std::size_t>(written);
        } else if (written < 0 && errno == EINTR) {
            continue;
        } else {
            break;
        }
    }
}

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
    bool no_display = false;
    int max_frames = 0;
    bool self_test = false;
    std::string dump_input;
    bool yolov8_enabled = false;
    bool divider_detection_enabled = false;
    bool rtsp_enabled = false;
    int event_fd = -1;
    int control_fd = -1;
    std::string snapshot_dir;
    bool prewarm = false;
    std::string view_id = "default";
    std::string rtsp_host = "0.0.0.0";
    int rtsp_port = 8554;
    std::string rtsp_path = "/dice";
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
        // An explicitly configured device path overrides camera, while an
        // empty device keeps the camera path/index selected above.
        std::string configured_device;
        read_config_value(root, "device", configured_device);
        if (!configured_device.empty()) a.device = configured_device;
        read_config_value(root, "width", a.width);
        read_config_value(root, "height", a.height);
        read_config_value(root, "fps", a.fps);
        read_config_value(root, "intra_threads", a.intra_threads);
        read_config_value(root, "ep_affinity", a.ep_affinity);
        read_config_value(root, "conf", a.conf);
        read_config_value(root, "stable_frames", a.stable_frames);
        read_config_value(root, "focus", a.focus);
        read_config_value(root, "zoom", a.zoom);
        read_config_value(root, "max_frames", a.max_frames);
        read_config_value(root, "dump_input", a.dump_input);
        if (!read_config_bool(root, "self_test", a.self_test)) return false;
        bool display_enabled = !a.no_display;
        if (!read_config_bool(root, "display_enabled", display_enabled)) return false;
        // Accept the old negative spelling too, but prefer display_enabled in
        // new configurations because it reads naturally in JSON.
        if (root["display_enabled"].empty()) {
            if (!read_config_bool(root, "no_display", a.no_display)) return false;
        } else {
            a.no_display = !display_enabled;
        }
        if (!read_config_bool(root, "yolov8_enabled", a.yolov8_enabled)) return false;
        if (!read_config_bool(root, "divider_detection", a.divider_detection_enabled)) return false;

        int queue_depth = static_cast<int>(a.queue_depth);
        read_config_value(root, "queue_depth", queue_depth);
        if (queue_depth < 0) {
            std::cerr << "config queue_depth must be >= 0\n";
            return false;
        }
        a.queue_depth = static_cast<std::size_t>(queue_depth);

        const cv::FileNode rtsp = root["rtsp"];
        if (!rtsp.empty()) {
            if (!rtsp.isMap()) {
                std::cerr << "config rtsp must be a JSON object\n";
                return false;
            }
            if (!read_config_bool(rtsp, "enabled", a.rtsp_enabled)) return false;
            read_config_value(rtsp, "host", a.rtsp_host);
            read_config_value(rtsp, "port", a.rtsp_port);
            read_config_value(rtsp, "path", a.rtsp_path);
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
              << "  --stable-frames N  matching YOLO observations required for stability\n"
              << "  --divider-detection enable reusable scene divider assistance\n"
              << "  --no-divider-detection disable scene divider assistance\n"
              << "  --focus N          fixed manual focus (-1 unchanged)\n"
              << "  --zoom N           zoom absolute value (-1 unchanged)\n"
              << "  --intra-threads N  SpaceMIT EP threads\n"
              << "  --ep-affinity LIST bind EP threads to cores, e.g. 14;15\n"
              << "  --no-display       run pipeline without window (config: display_enabled=false)\n"
              << "  --max-frames N     stop after N frames enter preprocess (0=unlimited)\n"
              << "  --dump-input PATH  dump first preprocessed tensor as float32\n"
              << "  --yolov8           enable YOLO preprocessing/inference\n"
              << "  --no-yolov8        bypass preprocessing/inference and display camera frames only\n"
              << "  --self-test        initialize OpenCL GPU and model, run one inference\n"
              << "  --rtsp             publish H.264 to RTSP server with SpaceMIT VPU\n"
              << "  --event-fd FD      write structured JSONL events to an inherited FD\n"
              << "  --control-fd FD    read vision-control-v1 JSONL commands from an inherited FD\n"
              << "  --snapshot-dir PATH save stable-frame JPEG snapshots under PATH\n"
              << "  --prewarm          keep camera/RTSP resident and wait for START_ADJUDICATION\n"
              << "  --view-id ID       identify this camera view in observation events\n"
              << "  --rtsp-host HOST   RTSP server host, default 127.0.0.1\n"
              << "  --rtsp-port N      RTSP server port, default 8554\n"
              << "  --rtsp-path PATH   RTSP mount path, default /dice\n"
              << "  --no-rtsp          disable RTSP publishing\n";
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
    if (a.rtsp_port < 1 || a.rtsp_port > 65535) {
        std::cerr << "RTSP port must be between 1 and 65535\n";
        return false;
    }
    if (a.rtsp_path.empty()) {
        std::cerr << "RTSP path must not be empty\n";
        return false;
    }
    if (a.event_fd != -1 && a.event_fd < 3) {
        std::cerr << "--event-fd must be -1 or an inherited file descriptor >= 3\n";
        return false;
    }
    if (a.control_fd < -1) {
        std::cerr << "--control-fd must be -1 or a valid inherited file descriptor\n";
        return false;
    }
    if (a.snapshot_dir.empty()) a.snapshot_dir = "/tmp/vision-snapshots";
    if (a.view_id.empty()) a.view_id = "default";
    if (a.rtsp_path.front() != '/') a.rtsp_path.insert(a.rtsp_path.begin(), '/');
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
            } else if (k == "--divider-detection") a.divider_detection_enabled = true;
            else if (k == "--no-divider-detection") a.divider_detection_enabled = false;
            else if (k == "--focus" && (v = need(i))) a.focus = std::stoi(v);
            else if (k == "--zoom" && (v = need(i))) a.zoom = std::stoi(v);
            else if (k == "--intra-threads" && (v = need(i))) a.intra_threads = std::stoi(v);
            else if (k == "--ep-affinity" && (v = need(i))) a.ep_affinity = v;
            else if (k == "--max-frames" && (v = need(i))) a.max_frames = std::stoi(v);
            else if (k == "--no-display") a.no_display = true;
            else if (k == "--dump-input" && (v = need(i))) a.dump_input = v;
            else if (k == "--yolov8") a.yolov8_enabled = true;
            else if (k == "--no-yolov8") a.yolov8_enabled = false;
            else if (k == "--self-test") a.self_test = true;
            else if (k == "--rtsp") a.rtsp_enabled = true;
            else if (k == "--event-fd" && (v = need(i))) a.event_fd = std::stoi(v);
            else if (k == "--control-fd" && (v = need(i))) a.control_fd = std::stoi(v);
            else if (k == "--snapshot-dir" && (v = need(i))) a.snapshot_dir = v;
            else if (k == "--prewarm") a.prewarm = true;
            else if (k == "--view-id" && (v = need(i))) a.view_id = v;
            else if (k == "--rtsp-host" && (v = need(i))) a.rtsp_host = v;
            else if (k == "--rtsp-port" && (v = need(i))) a.rtsp_port = std::stoi(v);
            else if (k == "--rtsp-path" && (v = need(i))) a.rtsp_path = v;
            else if (k == "--no-rtsp") a.rtsp_enabled = false;
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

// Read one newline-delimited control command without blocking the video loop.
// The protocol is intentionally tiny and forwards JSON untouched to higher
// layers; command matching is limited to the well-known vision-control-v1
// verbs so malformed input cannot alter runtime options.
static std::optional<std::string> read_control_command(int fd, std::string& buffer) {
    if (fd < 0) return std::nullopt;
    const size_t buffered_newline = buffer.find('\n');
    if (buffered_newline != std::string::npos) {
        std::string line = buffer.substr(0, buffered_newline);
        buffer.erase(0, buffered_newline + 1);
        return line;
    }
    char chunk[1024];
    const ssize_t n = ::read(fd, chunk, sizeof(chunk));
    if (n <= 0) return std::nullopt;
    buffer.append(chunk, static_cast<size_t>(n));
    const size_t newline = buffer.find('\n');
    if (newline == std::string::npos) return std::nullopt;
    std::string line = buffer.substr(0, newline);
    buffer.erase(0, newline + 1);
    return line;
}

static std::string control_command_name(const std::string& json) {
    for (const char* command : {"START_ADJUDICATION", "STOP_ADJUDICATION",
                                "FINAL_RESULT", "CANCEL"}) {
        if (json.find(std::string("\"command\":\"") + command + "\"") != std::string::npos ||
            json.find(std::string("\"command\": \"") + command + "\"") != std::string::npos) {
            return command;
        }
    }
    return {};
}

static std::string json_string_field(const std::string& json, const char* key) {
    const std::string marker = std::string("\"") + key + "\"";
    const size_t start = json.find(marker);
    if (start == std::string::npos) return {};
    const size_t colon = json.find(':', start + marker.size());
    if (colon == std::string::npos) return {};
    const size_t quote = json.find('"', colon + 1);
    if (quote == std::string::npos) return {};
    const size_t end = json.find('"', quote + 1);
    if (end == std::string::npos) return {};
    return json.substr(quote + 1, end - quote - 1);
}

// FINAL_RESULT carries the provider's generic outcome envelope as
// {"outcome":{"kind":"winner","value":"LEFT"}}.  Keep parsing
// deliberately small and dependency-free here: the runtime only needs the
// scalar value to annotate its own event, while Python remains the owner of
// the full result contract.
static std::string json_outcome_value(const std::string& json) {
    const std::string marker = "\"outcome\"";
    const size_t start = json.find(marker);
    if (start == std::string::npos) return {};
    const size_t colon = json.find(':', start + marker.size());
    if (colon == std::string::npos) return {};
    const size_t first = json.find_first_not_of(" \t\r\n", colon + 1);
    if (first == std::string::npos) return {};
    if (json[first] == '"') {
        const size_t end = json.find('"', first + 1);
        return end == std::string::npos ? std::string{} :
               json.substr(first + 1, end - first - 1);
    }
    if (json[first] != '{') return {};
    const size_t object_end = json.find('}', first + 1);
    if (object_end == std::string::npos) return {};
    return json_string_field(json.substr(first, object_end - first + 1), "value");
}

static std::string json_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (const char c : text) {
        if (c == '\\' || c == '"') out.push_back('\\');
        if (c == '\n') { out += "\\n"; continue; }
        if (c == '\r') { out += "\\r"; continue; }
        out.push_back(c);
    }
    return out;
}

static std::string save_snapshot(const Args& args, const cv::Mat& bgr,
                                 uint64_t frame_id) {
    std::error_code error;
    std::filesystem::create_directories(args.snapshot_dir, error);
    if (error) return {};
    const std::filesystem::path path = std::filesystem::path(args.snapshot_dir) /
        ("stable-" + args.view_id + "-" + std::to_string(frame_id) + ".jpg");
    try {
        if (!cv::imwrite(path.string(), bgr, {cv::IMWRITE_JPEG_QUALITY, 90})) return {};
    } catch (const cv::Exception&) {
        return {};
    }
    return path.string();
}

static void emit_observation(const Args& args, const InferenceResult& item,
                             const std::string& snapshot_path,
                             const DividerLine* divider_assist = nullptr) {
    std::ostringstream event;
    event << "{\"event\":\"observation\",\"view_id\":\""
          << json_escape(args.view_id) << "\",\"frame_id\":" << item.id
          << ",\"stable\":true,\"width\":" << item.width
          << ",\"height\":" << item.height;
    if (!snapshot_path.empty()) {
        event << ",\"snapshot\":{\"format\":\"image/jpeg\",\"path\":\""
              << json_escape(snapshot_path) << "\"}";
    }
    if (divider_assist) {
        event << ",\"divider\":{\"found\":" << (divider_assist->valid ? "true" : "false")
              << ",\"horizontal\":" << (divider_assist->horizontal ? "true" : "false")
              << ",\"point\":[" << divider_assist->point.x << "," << divider_assist->point.y << "]"
              << ",\"direction\":[" << divider_assist->direction.x << "," << divider_assist->direction.y << "]"
              << ",\"normal\":[" << divider_assist->normal.x << "," << divider_assist->normal.y << "]}";
    }
    event << ",\"detections\":[";
    for (size_t i = 0; i < item.detections.size(); ++i) {
        if (i) event << ',';
        const auto& d = item.detections[i];
        event << "{\"class_id\":" << d.class_id
              << ",\"label\":\"class_" << d.class_id
              << "\",\"confidence\":" << std::fixed << std::setprecision(4) << d.confidence
              << ",\"bbox\":[" << d.x1 << ',' << d.y1 << ','
              << d.x2 << ',' << d.y2 << "]}";
    }
    event << "]}";
    emit_event(event.str());
}

static std::string detection_signature(const std::vector<Detection>& detections) {
    // Detector confidence and box coordinates naturally move by a few pixels
    // between adjacent frames.  Stability is about the observed object
    // layout, not bit-identical floating point output, so quantize geometry
    // and compare detections independent of the NMS output order.
    const auto quantize = [](float value) {
        constexpr float kGridPixels = 8.0f;
        return std::llround(value / kGridPixels);
    };
    std::vector<std::string> tokens;
    tokens.reserve(detections.size());
    for (const auto& d : detections) {
        std::ostringstream token;
        token << d.class_id << ':' << quantize(d.x1) << ',' << quantize(d.y1)
              << ',' << quantize(d.x2) << ',' << quantize(d.y2);
        tokens.push_back(token.str());
    }
    std::sort(tokens.begin(), tokens.end());
    std::ostringstream out;
    out << tokens.size();
    for (const auto& token : tokens) {
        out << '|' << token;
    }
    return out.str();
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

static void draw_scene_assist(cv::Mat& bgr, const DividerLine& divider,
                              const std::string& status) {
    if (divider.valid) {
        const cv::Point2f p = divider.point;
        const cv::Point2f v = divider.direction * 2000.0f;
        const cv::Point p1(static_cast<int>(p.x - v.x), static_cast<int>(p.y - v.y));
        const cv::Point p2(static_cast<int>(p.x + v.x), static_cast<int>(p.y + v.y));
        cv::line(bgr, p1, p2, cv::Scalar(0, 165, 255), 3, cv::LINE_AA);
    }
    const cv::Scalar color = divider.valid ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 0, 255);
    // Keep the judgment text directly on the image instead of drawing a solid
    // black panel in the top-left corner. A thin shadow preserves readability
    // without leaving the unwanted black background/bottom edge.
    const cv::Point text_origin(18, 48);
    cv::putText(bgr, status, text_origin, cv::FONT_HERSHEY_SIMPLEX,
                .7, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
    cv::putText(bgr, status, text_origin, cv::FONT_HERSHEY_SIMPLEX,
                .7, color, 2, cv::LINE_AA);
}


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
    // A prewarmed runtime is controlled through the process stdin by
    // default.  The Python adapter writes vision-control-v1 JSONL commands
    // to that stream; an explicit descriptor still takes precedence for
    // supervisors that pass one.  This keeps resident mode usable without
    // requiring every caller to know Unix fd conventions.
    if (a.prewarm && a.control_fd < 0) a.control_fd = STDIN_FILENO;
    g_event_fd = a.event_fd;
    emit_event("{\"event\":\"started\",\"component\":\"vision_yolov8_adjudicator\",\"protocol\":\"jsonl-events-v1\"}");
    const bool runtime_has_control = a.control_fd >= 0;
    const bool yolo_runtime_enabled = a.yolov8_enabled || runtime_has_control;
    if (yolo_runtime_enabled && !std::filesystem::exists(a.model)) {
        std::cerr << "Model not found: " << a.model << "\n";
        return 2;
    }

    std::unique_ptr<OpenClPreprocessor> pre;
    std::unique_ptr<Yolov8Detector> detector;
    if (yolo_runtime_enabled) {
        pre = std::make_unique<OpenClPreprocessor>();
        if (!pre->init()) return 4;
        detector = std::make_unique<Yolov8Detector>();
        if (!detector->init(a.model, a.intra_threads, a.ep_affinity)) return 5;
    } else {
        // Camera-only mode intentionally avoids all YOLOv8/OpenCL/ORT setup.
        std::cerr << "[YOLOv8] disabled; displaying camera frames without preprocessing or inference.\n";
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
    emit_event(std::string("{\"event\":\"phase\",\"phase\":\"") +
               (a.prewarm ? "idle" : "detecting") + "\"}");
    RtspStreamer rtsp_streamer;
    if (a.rtsp_enabled && !rtsp_streamer.start(a.rtsp_host, a.rtsp_port, a.rtsp_path,
                                                a.width, a.height, camera.negotiated_fps())) {
        rtsp_streamer.stop();
        camera.close();
        return 8;
    }
    emit_event("{\"event\":\"ready\",\"view_id\":\"" + json_escape(a.view_id) + "\"}");
    if (rtsp_streamer.running()) {
        emit_event("{\"event\":\"video\",\"view_id\":\"" + json_escape(a.view_id) +
                   "\",\"url\":\"" + json_escape(rtsp_streamer.url()) + "\"}");
    }

    if (!a.no_display) {
        try {
            cv::namedWindow("yolov8-k3", cv::WINDOW_NORMAL);
            cv::resizeWindow("yolov8-k3", a.width, a.height);
        } catch (const cv::Exception& e) {
            std::cerr << "Display initialization failed: " << e.what() << "\n";
            rtsp_streamer.stop();
            camera.close();
            return 7;
        }
    }

    if (!a.yolov8_enabled && a.control_fd < 0) {
        const auto direct_start = Clock::now();
        int frame_count = 0;
        while (!g_signal_stop && (a.max_frames <= 0 || frame_count < a.max_frames)) {
            GstreamerFrame frame;
            if (!camera.read(frame, 1000)) continue;
            ++frame_count;
            if (a.no_display && !rtsp_streamer.running()) continue;
            cv::Mat bgr;
            if (frame.nv12.empty()) continue;
            cv::cvtColor(frame.nv12, bgr, cv::COLOR_YUV2BGR_NV12);
            rtsp_streamer.publish(bgr);
            if (a.no_display) continue;
            cv::imshow("yolov8-k3", bgr);
            const int key = cv::waitKey(1) & 0xff;
            if (key == 'q' || key == 27) break;
        }
        rtsp_streamer.stop();
        camera.close();
        if (!a.no_display) cv::destroyAllWindows();
        const double elapsed = std::chrono::duration<double>(Clock::now() - direct_start).count();
        std::cout << "Done. camera-only frames=" << frame_count
                  << " elapsed_s=" << elapsed
                  << " fps=" << frame_count / std::max(.001, elapsed) << "\n";
        return 0;
    }

    std::atomic<bool> abort{false};
    // Any explicit control channel owns round activation, even if a caller
    // omitted --prewarm; this keeps command-driven providers deterministic.
    std::atomic<bool> adjudication_active{!a.prewarm && a.control_fd < 0};
    std::atomic<bool> generic_observation_sent{false};
    std::atomic<int> generic_stable_count{0};
    std::string generic_last_signature;
    std::mutex generic_mutex;
    std::thread control_thread;
    if (a.control_fd >= 0) {
        control_thread = std::thread([&] {
            std::string buffer;
            while (!g_signal_stop && !abort.load()) {
                struct pollfd pfd{a.control_fd, POLLIN | POLLHUP | POLLERR, 0};
                const int ready = ::poll(&pfd, 1, 250);
                if (ready <= 0) continue;
                if (pfd.revents & (POLLHUP | POLLERR)) break;
                auto line = read_control_command(a.control_fd, buffer);
                if (!line) continue;
                const std::string command = control_command_name(*line);
                if (command == "START_ADJUDICATION") {
                    adjudication_active.store(true);
                    generic_observation_sent.store(false);
                    generic_stable_count.store(0);
                    std::lock_guard<std::mutex> lock(generic_mutex);
                    generic_last_signature.clear();
                    emit_event("{\"event\":\"phase\",\"phase\":\"detecting\"}");
                } else if (command == "STOP_ADJUDICATION") {
                    adjudication_active.store(false);
                    emit_event("{\"event\":\"phase\",\"phase\":\"idle\"}");
                } else if (command == "CANCEL") {
                    adjudication_active.store(false);
                    emit_event("{\"event\":\"cancelled\"}");
                } else if (command == "FINAL_RESULT") {
                    // Provider owns the generic verdict payload. Preserve it
                    // as an opaque result event and let the job state machine
                    // interpret its fields.
                    const std::string outcome = json_outcome_value(*line);
                    const std::string source = json_string_field(*line, "source");
                    emit_event("{\"event\":\"result\",\"outcome\":{\"kind\":\"winner\",\"value\":\"" +
                               json_escape(outcome) + "\"},\"source\":\"" +
                               json_escape(source.empty() ? "provider" : source) + "\"}");
                    emit_event("{\"event\":\"complete\",\"phase\":\"complete\"}");
                }
            }
        });
    }
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
                if (!adjudication_active.load()) {
                    // Resident prewarm keeps capture and RTSP alive while
                    // avoiding OpenCL preprocessing and detector work.
                    auto idle_result = std::make_shared<InferenceResult>();
                    idle_result->id = packet->id;
                    idle_result->width = packet->width;
                    idle_result->height = packet->height;
                    idle_result->nv12 = packet->nv12;
                    idle_result->gst_owner = packet->gst_owner;
                    if (result_queue.push(std::move(idle_result))) stats.dropped_result.fetch_add(1);
                    stats.dropped_result.fetch_add(result_queue.takeDroppedPending());
                    stats.prepared.fetch_add(1);
                    continue;
                }
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
                if (adjudication_active.load()) {
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

    // Thread 3 is the display/stability stage. It runs on the main/UI thread
    // because OpenCV HighGUI on this board must own the X11 event loop here.
    bool first_display = true;
    uint64_t shown = 0;
    auto last_report = start;
    while (true) {
        if (g_signal_stop) abort.store(true);
        std::shared_ptr<InferenceResult> item;
        if (!result_queue.popLatest(item, abort, first_display, a.queue_depth)) {
            if (g_signal_stop) abort.store(true);
            if (inference_done.load()) break;
            if (abort.load()) break;
            continue;
        }
        first_display = false;
        if (!item) continue;

        const auto t0 = Clock::now();
        cv::Mat bgr;
        DividerLine divider_assist;
        if (item->nv12 && !item->nv12->empty()) {
            cv::cvtColor(*item->nv12, bgr, cv::COLOR_YUV2BGR_NV12);
            if (a.divider_detection_enabled) {
                divider_assist.valid = detect_black_divider(bgr, divider_assist);
            }
            std::string scene_status = !adjudication_active.load()
                ? "Ready"
                : (a.divider_detection_enabled
                    ? (divider_assist.valid ? "Divider detected" : "Searching divider")
                    : "YOLO observation");
                if (a.control_fd >= 0 && adjudication_active.load() &&
                    !generic_observation_sent.load()) {
                    const std::string signature = detection_signature(item->detections);
                    int stable_count = 0;
                    {
                        std::lock_guard<std::mutex> lock(generic_mutex);
                        if (signature == generic_last_signature) stable_count = generic_stable_count.fetch_add(1) + 1;
                        else { generic_last_signature = signature; generic_stable_count.store(1); stable_count = 1; }
                    }
                    // Generic provider mode uses the detector's stable
                    // observation contract. An empty detection frame is not
                    // useful evidence, but every non-empty object layout is a
                    // valid candidate for a game profile to interpret.
                    const bool divider_ready = !a.divider_detection_enabled || divider_assist.valid;
                    if (!item->detections.empty() && divider_ready &&
                        stable_count >= a.stable_frames &&
                        !generic_observation_sent.exchange(true)) {
                        const std::string snapshot_path = save_snapshot(a, bgr, item->id);
                        emit_observation(a, *item, snapshot_path,
                                         a.divider_detection_enabled ? &divider_assist : nullptr);
                    }
                }
            if (!a.no_display || rtsp_streamer.running()) {
                draw_detections(bgr, item->detections);
                draw_scene_assist(bgr, divider_assist, scene_status);
                const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
                const double fps = elapsed > 0.0 ? shown / elapsed : 0.0;
                cv::putText(bgr, "DISPLAY " + std::to_string(static_cast<int>(fps)) +
                                      " FPS  DET " + std::to_string(item->detections.size()),
                            {10, 96}, cv::FONT_HERSHEY_SIMPLEX, .75,
                            {0, 255, 255}, 2, cv::LINE_AA);
                rtsp_streamer.publish(bgr);
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
    if (control_thread.joinable()) control_thread.join();
    prepared_queue.clear();
    result_queue.clear();
    rtsp_streamer.stop();
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
