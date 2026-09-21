#include "gstreamer_camera.h"
#include "rps_mapper.h"
#include "rtsp_streamer.h"
#include "opencl_preprocess.h"
#include "yolov10_detector.h"
#include "control_protocol.h"

#include <opencv2/core.hpp>
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
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <deque>
#include <cerrno>
#include <unistd.h>

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
        // waiting for a frame; returning false here would make the consumer
        // exit whenever a frame takes longer than the polling interval.
        while (!ready()) {
            cv_.wait_for(lock, std::chrono::milliseconds(100));
        }
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
    // Inference crop inside the streamed (rotated) frame: the model only
    // sees this region; detections come back in crop coords and are shifted
    // by the origin into streamed-frame coordinates.
    int crop_w = 0;
    int crop_h = 0;
    float crop_ox = 0.0f;
    float crop_oy = 0.0f;
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
    // Detections in streamed-frame coordinates with the class_id already
    // collapsed onto the game labels when rps_mode is on (see RpsMapper).
    std::vector<Detection> detections;
};

struct Args {
    std::string config_path = "config.json";
    std::string model = "model/yolov10n_gestures.q.onnx";
    int camera = 1;
    std::string device;
    int width = 1280, height = 720, fps = 25;
    int intra_threads = 2;
    std::string ep_affinity = "14;15";
    bool ep_enabled = true;
    int focus = 0, zoom = 150;
    float conf = 0.25f;
    size_t queue_depth = 2;
    // Rotate every captured frame before preprocessing/streaming (mounting
    // the camera sideways or upside down). Width/height describe the
    // *captured* frame; the inference/stream frame is rotated accordingly.
    bool rotate_enabled = false;
    std::string rotate_direction = "cw";  // "cw" | "ccw" (90-degree modes only)
    int rotate_angle = 90;                // 90 | 180
    int max_frames = 0;
    bool self_test = false;
    std::string dump_input;
    // Model vocabulary, declared by the game's runtime config (or --classes).
    // The package has no built-in class table: rps_mode and
    // filter_no_gesture both require it to be declared.
    std::vector<std::string> class_names;
    bool filter_no_gesture = false;
    // Stability gate: the folded class multiset must repeat for this many
    // consecutive frames before a stable observation is emitted (1 = the
    // first usable frame is already stable).
    int stable_frames = 20;
    // Region of interest (fractions of the streamed frame): the model only
    // sees this region and only detections whose box center falls inside are
    // kept. Used to exclude the robot arm's half of the view; the ROI border
    // is drawn on the stream for alignment.
    bool roi_enabled = false;
    float roi_x = 0.0f, roi_y = 0.0f, roi_w = 1.0f, roi_h = 1.0f;
    // Label collapsing: fold the model vocabulary onto game labels via
    // rps_map; off keeps raw declared classes. Both the table and the
    // vocabulary come from config — no defaults live in the binary.
    bool rps_mode = false;
    RpsMapper::LabelMap rps_map;
    bool yolov10_enabled = true;
    bool rtsp_enabled = false;
    int event_fd = -1;
    int control_fd = -1;
    std::string snapshot_dir;
    bool prewarm = false;
    std::string view_id = "default";
    std::string rtsp_host = "127.0.0.1";
    int rtsp_port = 8554;
    std::string rtsp_path = "/rps/det";
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
        // Hardware config files are launched from the repository root by the
        // provider or a diagnostic script. Resolve the model relative to the
        // JSON file, while still allowing --model to override it later during
        // command-line parsing.
        if (!a.model.empty()) {
            const std::filesystem::path model_path(a.model);
            if (!model_path.is_absolute()) {
                const std::filesystem::path config_path =
                    std::filesystem::absolute(std::filesystem::path(path));
                a.model = (config_path.parent_path() / model_path).lexically_normal().string();
            }
        }
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
        const cv::FileNode classes = root["classes"];
        if (!classes.empty()) {
            if (!classes.isSeq()) {
                std::cerr << "config classes must be an array of non-empty strings\n";
                return false;
            }
            std::vector<std::string> names;
            for (const cv::FileNode& item : classes) {
                if (!item.isString()) {
                    std::cerr << "config classes must be an array of non-empty strings\n";
                    return false;
                }
                std::string name;
                item >> name;
                if (name.empty()) {
                    std::cerr << "config classes must be an array of non-empty strings\n";
                    return false;
                }
                names.push_back(std::move(name));
            }
            a.class_names = std::move(names);
        }
        if (!read_config_bool(root, "filter_no_gesture", a.filter_no_gesture)) return false;
        read_config_value(root, "stable_frames", a.stable_frames);
        const cv::FileNode roi = root["roi"];
        if (!roi.empty()) {
            if (!roi.isMap()) {
                std::cerr << "config roi must be a JSON object {enabled,x,y,w,h}\n";
                return false;
            }
            if (!read_config_bool(roi, "enabled", a.roi_enabled)) return false;
            read_config_value(roi, "x", a.roi_x);
            read_config_value(roi, "y", a.roi_y);
            read_config_value(roi, "w", a.roi_w);
            read_config_value(roi, "h", a.roi_h);
        }
        if (!read_config_bool(root, "rotate_90ccw", a.rotate_enabled)) {
            return false;
        }
        if (a.rotate_enabled) {
            // Legacy key: the original implementation was 90 CCW.
            a.rotate_direction = "ccw";
            a.rotate_angle = 90;
        }
        const cv::FileNode rotate = root["rotate"];
        if (!rotate.empty()) {
            if (!rotate.isMap()) {
                std::cerr << "config rotate must be a JSON object "
                             "{enabled, direction, angle}\n";
                return false;
            }
            if (!read_config_bool(rotate, "enabled", a.rotate_enabled)) return false;
            std::string direction = a.rotate_direction;
            read_config_value(rotate, "direction", direction);
            std::transform(direction.begin(), direction.end(), direction.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (direction != "cw" && direction != "ccw") {
                std::cerr << "config rotate.direction must be \"cw\" or \"ccw\"\n";
                return false;
            }
            a.rotate_direction = direction;
            int angle = a.rotate_angle;
            read_config_value(rotate, "angle", angle);
            if (angle != 90 && angle != 180) {
                std::cerr << "config rotate.angle must be 90 or 180\n";
                return false;
            }
            a.rotate_angle = angle;
        }
        if (!read_config_bool(root, "rps_mode", a.rps_mode)) return false;
        const cv::FileNode rps_map = root["rps_map"];
        if (!rps_map.empty()) {
            if (!rps_map.isMap()) {
                std::cerr << "config rps_map must be a JSON object of label -> class-name array\n";
                return false;
            }
            RpsMapper::LabelMap map;
            for (auto it = rps_map.begin(); it != rps_map.end(); ++it) {
                const std::string label = (*it).name();
                const cv::FileNode sources = *it;
                if (!sources.isSeq()) {
                    std::cerr << "config rps_map." << label << " must be an array of class names\n";
                    return false;
                }
                std::vector<std::string> names;
                for (const cv::FileNode& item : sources) {
                    if (!item.isString()) {
                        std::cerr << "config rps_map." << label << " must contain class-name strings\n";
                        return false;
                    }
                    std::string name;
                    item >> name;
                    if (name.empty()) {
                        std::cerr << "config rps_map." << label << " contains an empty class name\n";
                        return false;
                    }
                    names.push_back(std::move(name));
                }
                map[label] = std::move(names);
            }
            if (!map.empty()) a.rps_map = std::move(map);
        }
        read_config_value(root, "focus", a.focus);
        read_config_value(root, "zoom", a.zoom);
        if (!read_config_bool(root, "yolov10_enabled", a.yolov10_enabled)) return false;
        read_config_value(root, "max_frames", a.max_frames);
        read_config_value(root, "dump_input", a.dump_input);
        if (!read_config_bool(root, "self_test", a.self_test)) return false;

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
              << "  --classes LIST     comma-separated class names, e.g. like,ok,peace\n"
              << "  --show-no-gesture  draw the no_gesture class too (config: filter_no_gesture=false)\n"
              << "  --no-rps           keep raw class labels instead of the rps_map folding\n"
              << "  --rotate MODE      rotate frames before inference/streaming: none, 90cw, 90ccw, 180\n"
              << "                     (config: rotate{enabled,direction,angle})\n"
              << "  --stable-frames N  consecutive frames the class multiset must repeat (1=off)\n"
              << "  --queue-depth N    keep up to N frames per pipeline queue\n"
              << "  --focus N          fixed manual focus (-1 unchanged)\n"
              << "  --zoom N           zoom absolute value (-1 unchanged)\n"
              << "  --intra-threads N  SpaceMIT EP threads\n"
              << "  --ep-affinity LIST bind EP threads to cores, e.g. 14;15\n"
              << "  --no-ep            disable SpaceMIT EP, run the model on CPU\n"
              << "  --max-frames N     stop after N frames enter preprocess (0=unlimited)\n"
              << "  --dump-input PATH  dump first preprocessed tensor as float32\n"
              << "  --no-yolov10       bypass preprocessing/inference and stream camera frames only\n"
              << "  --self-test        initialize OpenCL GPU and model, run one inference\n"
              << "  --no-display       accepted for framework compatibility (this runtime is headless)\n"
              << "  --rtsp             publish H.264 to RTSP server with SpaceMIT VPU\n"
              << "  --rtsp-host HOST   RTSP server host, default 127.0.0.1\n"
              << "  --rtsp-port N      RTSP server port, default 8554\n"
              << "  --rtsp-path PATH   RTSP mount path, default /rps/det\n"
              << "  --no-rtsp          disable RTSP publishing\n"
              << "  --event-fd FD      write structured JSONL events to an inherited FD\n"
              << "  --control-fd FD    read vision-control-v1 JSONL commands from an inherited FD\n"
              << "  --snapshot-dir PATH save stable-frame JPEG snapshots under PATH\n"
              << "  --prewarm          keep camera/RTSP resident and wait for START_ADJUDICATION\n"
              << "  --view-id ID       identify this camera view in observation events\n";
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
    if (a.rtsp_port < 1 || a.rtsp_port > 65535) {
        std::cerr << "RTSP port must be between 1 and 65535\n";
        return false;
    }
    if (a.rtsp_path.empty()) {
        std::cerr << "RTSP path must not be empty\n";
        return false;
    }
    if (a.rtsp_path.front() != '/') a.rtsp_path.insert(a.rtsp_path.begin(), '/');
    if (a.rotate_angle != 90 && a.rotate_angle != 180) {
        std::cerr << "rotate angle must be 90 or 180\n";
        return false;
    }
    if (a.rotate_direction != "cw" && a.rotate_direction != "ccw") {
        std::cerr << "rotate direction must be \"cw\" or \"ccw\"\n";
        return false;
    }
    if (a.stable_frames < 1) {
        std::cerr << "stable_frames must be >= 1 (1 disables the stability gate)\n";
        return false;
    }
    if (a.roi_x < 0.0f || a.roi_y < 0.0f || a.roi_w <= 0.0f || a.roi_h <= 0.0f ||
        a.roi_x + a.roi_w > 1.0f || a.roi_y + a.roi_h > 1.0f) {
        std::cerr << "config roi must satisfy 0<=x, 0<=y, w>0, h>0, x+w<=1, y+h<=1 "
                     "(fractions of the streamed frame)\n";
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
    // The package carries no built-in vocabulary: everything that needs class
    // names must have them declared in config (or --classes).
    if (a.rps_mode && a.rps_map.empty()) {
        std::cerr << "rps_mode requires a non-empty rps_map in config\n";
        return false;
    }
    if (a.rps_mode && a.class_names.empty()) {
        std::cerr << "rps_mode requires the classes array in config\n";
        return false;
    }
    if (a.filter_no_gesture && a.class_names.empty()) {
        std::cerr << "filter_no_gesture requires the classes array in config\n";
        return false;
    }
    return true;
}

// Split a comma-separated class list into names. An empty token is rejected
// instead of skipped: dropping it would shift every following class_id
// mapping, so a typo must fail loudly rather than silently relabel classes.
static std::vector<std::string> split_class_names(const std::string& text) {
    std::vector<std::string> names;
    std::string token;
    auto flush = [&]() {
        const auto begin = token.find_first_not_of(" \t");
        if (begin == std::string::npos) {
            throw std::invalid_argument("--classes contains an empty class name");
        }
        const auto end = token.find_last_not_of(" \t");
        names.push_back(token.substr(begin, end - begin + 1));
        token.clear();
    };
    for (const char c : text) {
        if (c == ',') flush();
        else token.push_back(c);
    }
    flush();
    return names;
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
            else if (k == "--classes" && (v = need(i))) a.class_names = split_class_names(v);
            else if (k == "--show-no-gesture") a.filter_no_gesture = false;
            else if (k == "--no-rps") a.rps_mode = false;
            else if (k == "--rotate-90ccw") {  // legacy alias
                a.rotate_enabled = true;
                a.rotate_direction = "ccw";
                a.rotate_angle = 90;
            }
            else if (k == "--stable-frames" && (v = need(i))) {
                a.stable_frames = std::stoi(v);
            } else if (k == "--rotate" && (v = need(i))) {
                const std::string mode = v;
                if (mode == "none") {
                    a.rotate_enabled = false;
                } else if (mode == "90ccw") {
                    a.rotate_enabled = true; a.rotate_direction = "ccw"; a.rotate_angle = 90;
                } else if (mode == "90cw") {
                    a.rotate_enabled = true; a.rotate_direction = "cw"; a.rotate_angle = 90;
                } else if (mode == "180") {
                    a.rotate_enabled = true; a.rotate_angle = 180;
                } else {
                    std::cerr << "--rotate must be none, 90cw, 90ccw, or 180\n";
                    return false;
                }
            }
            else if (k == "--queue-depth" && (v = need(i))) {
                a.queue_depth = static_cast<std::size_t>(std::stoul(v));
            } else if (k == "--focus" && (v = need(i))) a.focus = std::stoi(v);
            else if (k == "--zoom" && (v = need(i))) a.zoom = std::stoi(v);
            else if (k == "--intra-threads" && (v = need(i))) a.intra_threads = std::stoi(v);
            else if (k == "--ep-affinity" && (v = need(i))) a.ep_affinity = v;
            else if (k == "--no-ep") a.ep_enabled = false;
            else if (k == "--max-frames" && (v = need(i))) a.max_frames = std::stoi(v);
            else if (k == "--dump-input" && (v = need(i))) a.dump_input = v;
            else if (k == "--no-yolov10") a.yolov10_enabled = false;
            else if (k == "--self-test") a.self_test = true;
            else if (k == "--no-display") { /* headless runtime: nothing to disable */ }
            else if (k == "--rtsp") a.rtsp_enabled = true;
            else if (k == "--rtsp-host" && (v = need(i))) a.rtsp_host = v;
            else if (k == "--rtsp-port" && (v = need(i))) a.rtsp_port = std::stoi(v);
            else if (k == "--rtsp-path" && (v = need(i))) a.rtsp_path = v;
            else if (k == "--no-rtsp") a.rtsp_enabled = false;
            else if (k == "--event-fd" && (v = need(i))) a.event_fd = std::stoi(v);
            else if (k == "--control-fd" && (v = need(i))) a.control_fd = std::stoi(v);
            else if (k == "--snapshot-dir" && (v = need(i))) a.snapshot_dir = v;
            else if (k == "--prewarm") a.prewarm = true;
            else if (k == "--view-id" && (v = need(i))) a.view_id = v;
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

static std::string save_diagnostic_snapshot(const Args& args, const cv::Mat& bgr) {
    std::error_code error;
    std::filesystem::create_directories(args.snapshot_dir, error);
    if (error) return {};
    const std::filesystem::path path = std::filesystem::path(args.snapshot_dir) /
        ("latest-" + args.view_id + ".jpg");
    try {
        if (!cv::imwrite(path.string(), bgr, {cv::IMWRITE_JPEG_QUALITY, 85})) return {};
    } catch (const cv::Exception&) {
        return {};
    }
    return path.string();
}

// Label text for one detection in the space it was emitted in: the folded
// game label when the mapper is on, otherwise the declared class name (or a
// neutral class_N diagnostic tag when no vocabulary was declared).
static std::string detection_label(const Detection& d, const RpsMapper& rps_mapper,
                                   const std::vector<std::string>& class_names) {
    if (rps_mapper.enabled()) return rps_mapper.folded_label(d.class_id);
    if (d.class_id >= 0 && static_cast<std::size_t>(d.class_id) < class_names.size()) {
        return class_names[static_cast<std::size_t>(d.class_id)];
    }
    return "class_" + std::to_string(d.class_id);
}

static void emit_observation(const Args& args, const InferenceResult& item,
                             const std::string& snapshot_path,
                             const RpsMapper& rps_mapper,
                             const std::vector<std::string>& class_names,
                             const char* event_name = "observation",
                             bool stable = true) {
    std::ostringstream event;
    event << "{\"event\":\"" << event_name << "\",\"view_id\":\""
          << json_escape(args.view_id) << "\",\"frame_id\":" << item.id
          << ",\"stable\":" << (stable ? "true" : "false") << ",\"width\":" << item.width
          << ",\"height\":" << item.height;
    if (!snapshot_path.empty()) {
        event << ",\"snapshot\":{\"format\":\"image/jpeg\",\"path\":\""
              << json_escape(snapshot_path) << "\"}";
    }
    event << ",\"detections\":[";
    for (size_t i = 0; i < item.detections.size(); ++i) {
        if (i) event << ',';
        const auto& d = item.detections[i];
        event << "{\"class_id\":" << d.class_id
              << ",\"label\":\"" << json_escape(detection_label(d, rps_mapper, class_names))
              << "\",\"confidence\":" << std::fixed << std::setprecision(4) << d.confidence
              << ",\"bbox\":[" << d.x1 << ',' << d.y1 << ','
              << d.x2 << ',' << d.y2 << "]}";
    }
    event << "]}";
    emit_event(event.str());
}

static std::string detection_signature(const std::vector<Detection>& detections) {
    // Adjudication stability is semantic: the same multiset of detected
    // categories must persist. Confidence and box geometry are deliberately
    // excluded because normal detector jitter must never reset a stable
    // result. The class ids here are already collapsed onto game labels, so
    // source-class flicker within one label (palm -> stop, both "Paper")
    // counts as the same signature. The current boxes remain available in
    // the final observation for overlay and grouping only.
    std::vector<int> categories;
    categories.reserve(detections.size());
    for (const auto& d : detections) {
        categories.push_back(d.class_id);
    }
    std::sort(categories.begin(), categories.end());
    std::ostringstream out;
    out << categories.size();
    for (const int class_id : categories) {
        out << '|' << class_id;
    }
    return out.str();
}

// Label colors chosen to stand out on the red/blue game mats: the default
// Ultralytics palette starts with pure red/orange, which vanish on the red
// mat. BGR: cyan / magenta / yellow.
static const std::array<cv::Scalar, 3> kRpsColors = {
    cv::Scalar(255, 255, 0),    // cyan    -> label 0
    cv::Scalar(255, 0, 255),    // magenta -> label 1
    cv::Scalar(0, 255, 255),    // yellow  -> label 2
};
// Ultralytics-style vivid palette for the unfolded (raw class) mode.
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

static void draw_detections(cv::Mat& bgr, const std::vector<Detection>& ds,
                            const RpsMapper& rps_mapper,
                            const std::vector<std::string>& class_names) {
    for (const auto& d : ds) {
        cv::Rect r(static_cast<int>(d.x1), static_cast<int>(d.y1),
                   std::max(1, static_cast<int>(d.x2 - d.x1)),
                   std::max(1, static_cast<int>(d.y2 - d.y1)));
        // Folded mode colors by game label (one color per label regardless of
        // which source class triggered it); raw mode uses the class palette.
        cv::Scalar color;
        if (rps_mapper.enabled()) {
            color = kRpsColors[static_cast<std::size_t>(std::max(0, d.class_id)) %
                               kRpsColors.size()];
        } else {
            const size_t class_index =
                static_cast<size_t>(std::max(0, d.class_id)) % kClassColors.size();
            color = kClassColors[class_index];
        }
        std::ostringstream label;
        label << detection_label(d, rps_mapper, class_names) << " " << std::fixed
              << std::setprecision(2) << d.confidence;
        // Box outline first. The dark under-stroke keeps the outline visible
        // even when the label color is close to the scene background.
        cv::rectangle(bgr, r, cv::Scalar(0, 0, 0), 5, cv::LINE_AA);
        cv::rectangle(bgr, r, color, 2, cv::LINE_AA);
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
    emit_event("{\"event\":\"started\",\"component\":\"vision_yolov10_objdetect\",\"protocol\":\"jsonl-events-v2\"}");
    const bool runtime_has_control = a.control_fd >= 0;
    const bool yolo_runtime_enabled = a.yolov10_enabled || runtime_has_control;
    if (yolo_runtime_enabled && !std::filesystem::exists(a.model)) {
        std::cerr << "Model not found: " << a.model << "\n";
        return 2;
    }

    // Vocabulary from config only. filter_no_gesture drops the HaGRID
    // fallback class by id; rps_mode folds source classes onto game labels.
    int no_gesture_id = -1;
    if (a.filter_no_gesture) {
        const auto it = std::find(a.class_names.begin(), a.class_names.end(), "no_gesture");
        if (it == a.class_names.end()) {
            std::cerr << "filter_no_gesture=true but 'no_gesture' is not in the class list; "
                         "set filter_no_gesture=false or fix config classes\n";
            return 2;
        }
        no_gesture_id = static_cast<int>(it - a.class_names.begin());
        std::cout << "Filtering class_id " << no_gesture_id << " (no_gesture)\n";
    }
    RpsMapper rps_mapper;
    if (a.rps_mode) {
        try {
            rps_mapper.build(a.rps_map, a.class_names);
        } catch (const std::invalid_argument& e) {
            std::cerr << "rps_map error: " << e.what() << "\n";
            return 2;
        }
        std::cout << "RPS mode: ";
        for (const auto& [label, sources] : a.rps_map) {
            std::cout << label << "<={";
            for (size_t i = 0; i < sources.size(); ++i)
                std::cout << (i ? "," : "") << sources[i];
            std::cout << "} ";
        }
        std::cout << "\n";
    }

    // Derived rotation mode: 0 = none, 1 = 90 CCW, 2 = 90 CW, 3 = 180.
    // 180 keeps the frame dimensions; the 90-degree modes swap them.
    const int rotate_mode = !a.rotate_enabled ? 0
        : (a.rotate_angle == 180 ? 3 : (a.rotate_direction == "ccw" ? 1 : 2));
    const bool swap_dims = (rotate_mode == 1 || rotate_mode == 2);
    if (rotate_mode != 0) {
        std::cout << "Rotation: "
                  << (rotate_mode == 3 ? "180" :
                      (rotate_mode == 1 ? "90 ccw" : "90 cw")) << "\n";
    }

    // Inference crop: with roi enabled the model only sees the ROI region of
    // the streamed frame, so hands inside it appear ~1/roi fraction larger in
    // model space. The rect is even-aligned so the letterbox pads stay even.
    const int stream_w = swap_dims ? a.height : a.width;
    const int stream_h = swap_dims ? a.width : a.height;
    const auto even_floor = [](float v) { return std::max(0, static_cast<int>(v) & ~1); };
    int crop_w = a.roi_enabled ? even_floor(a.roi_w * stream_w) : stream_w;
    int crop_h = a.roi_enabled ? even_floor(a.roi_h * stream_h) : stream_h;
    int crop_ox = a.roi_enabled ? even_floor(a.roi_x * stream_w) : 0;
    int crop_oy = a.roi_enabled ? even_floor(a.roi_y * stream_h) : 0;
    crop_w = std::min(crop_w, even_floor(stream_w));
    crop_h = std::min(crop_h, even_floor(stream_h));
    crop_ox = std::min(crop_ox, stream_w - crop_w);
    crop_oy = std::min(crop_oy, stream_h - crop_h);
    std::cout << "Inference crop: " << crop_w << "x" << crop_h << " at +" << crop_ox
              << ",+" << crop_oy << " (stream " << stream_w << "x" << stream_h
              << ", hand scale x" << std::fixed << std::setprecision(2)
              << (static_cast<float>(std::max(stream_w, stream_h)) /
                  std::max(1, std::max(crop_w, crop_h)))
              << ")\n";

    std::unique_ptr<OpenClPreprocessor> pre;
    std::unique_ptr<Yolov10Detector> detector;
    if (yolo_runtime_enabled) {
        pre = std::make_unique<OpenClPreprocessor>();
        if (!pre->init()) return 4;
        detector = std::make_unique<Yolov10Detector>();
        if (!detector->init(a.model, a.intra_threads, a.ep_affinity, a.ep_enabled)) return 5;
    } else {
        // Camera-only mode intentionally avoids all YOLOv10/OpenCL/ORT setup.
        std::cerr << "[YOLOv10] disabled; streaming camera frames without preprocessing or inference.\n";
    }
    if (a.self_test) {
        if (!yolo_runtime_enabled) {
            std::cerr << "--self-test requires yolov10_enabled=true\n";
            return 2;
        }
        try {
            // Exercise the actual OpenCL NV12 -> tensor path as well as the
            // model path. This catches kernel/image/queue errors that a
            // model-only zero-tensor test cannot detect.
            constexpr int synthetic_width = 1280;
            constexpr int synthetic_height = 720;
            cv::Mat synthetic_nv12(synthetic_height * 3 / 2, synthetic_width,
                                   CV_8UC1, cv::Scalar(128));
            const auto prep_result = pre->preprocess(synthetic_nv12, rotate_mode);
            if (!prep_result.data || prep_result.data->size() != 3 * 640 * 640) {
                throw std::runtime_error("OpenCL self-test returned an invalid tensor");
            }
            for (float value : *prep_result.data) {
                if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
                    throw std::runtime_error("OpenCL self-test returned invalid tensor values");
                }
            }
            auto ds = detector->infer(prep_result.data->data(), prep_result.data->size(),
                                      a.conf, prep_result.scale, prep_result.pad_x,
                                      prep_result.pad_y, synthetic_width, synthetic_height);
            // Cross-check the model's raw class ids against the declared
            // vocabulary: a classes/model mismatch is loud here instead of a
            // silent label shift in production.
            if (!a.class_names.empty()) {
                for (const auto& d : ds) {
                    if (d.class_id < 0 ||
                        static_cast<std::size_t>(d.class_id) >= a.class_names.size()) {
                        throw std::runtime_error(
                            "model returned class_id " + std::to_string(d.class_id) +
                            " outside the declared classes (" +
                            std::to_string(a.class_names.size()) + ")");
                    }
                }
            }
            if (no_gesture_id >= 0) {
                ds.erase(std::remove_if(ds.begin(), ds.end(),
                                        [&](const Detection& d) {
                                            return d.class_id == no_gesture_id;
                                        }),
                         ds.end());
            }
            rps_mapper.fold(ds);
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
    // The published frame is the rotated one, so its size differs from the
    // captured size when rotation is enabled.
    const int out_width = swap_dims ? a.height : a.width;
    const int out_height = swap_dims ? a.width : a.height;
    RtspStreamer rtsp_streamer;
    if (a.rtsp_enabled && !rtsp_streamer.start(a.rtsp_host, a.rtsp_port, a.rtsp_path,
                                                out_width, out_height, camera.negotiated_fps())) {
        rtsp_streamer.stop();
        camera.close();
        return 8;
    }
    emit_event("{\"event\":\"ready\",\"view_id\":\"" + json_escape(a.view_id) + "\"}");
    if (rtsp_streamer.running()) {
        emit_event("{\"event\":\"video\",\"view_id\":\"" + json_escape(a.view_id) +
                   "\",\"url\":\"" + json_escape(rtsp_streamer.url()) + "\"}");
    }

    if (!yolo_runtime_enabled) {
        // Camera-only passthrough: raw (rotated) frames to RTSP, no inference.
        const auto direct_start = Clock::now();
        int frame_count = 0;
        while (!g_signal_stop && (a.max_frames <= 0 || frame_count < a.max_frames)) {
            GstreamerFrame frame;
            if (!camera.read(frame, 1000)) continue;
            ++frame_count;
            if (!rtsp_streamer.running()) continue;
            cv::Mat bgr;
            if (frame.nv12.empty()) continue;
            cv::cvtColor(frame.nv12, bgr, cv::COLOR_YUV2BGR_NV12);
            if (rotate_mode == 1) cv::rotate(bgr, bgr, cv::ROTATE_90_COUNTERCLOCKWISE);
            else if (rotate_mode == 2) cv::rotate(bgr, bgr, cv::ROTATE_90_CLOCKWISE);
            else if (rotate_mode == 3) cv::rotate(bgr, bgr, cv::ROTATE_180);
            rtsp_streamer.publish(bgr);
        }
        rtsp_streamer.stop();
        camera.close();
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
    std::atomic<int> generic_last_reported_count{0};
    std::string generic_last_signature;
    std::mutex generic_mutex;
    std::thread control_thread;
    if (a.control_fd >= 0) {
        control_thread = std::thread([&] {
            vision_control::CommandReader reader;
            while (!g_signal_stop && !abort.load()) {
                const auto lines = reader.read_ready(a.control_fd, 250);
                if (lines.empty() && reader.closed()) break;
                for (const std::string& line : lines) {
                    const std::string command = control_command_name(line);
                    if (command == "START_ADJUDICATION") {
                        adjudication_active.store(true);
                        generic_observation_sent.store(false);
                        generic_stable_count.store(0);
                        generic_last_reported_count.store(0);
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
                        // A final verdict closes the inference phase by
                        // itself. STOP_ADJUDICATION remains an idempotent
                        // lifecycle command, but is no longer a single point
                        // of failure for releasing YOLO compute.
                        adjudication_active.store(false);
                        const std::string outcome = json_outcome_value(line);
                        const std::string source = json_string_field(line, "source");
                        emit_event("{\"event\":\"result\",\"outcome\":{\"kind\":\"winner\",\"value\":\"" +
                                   json_escape(outcome) + "\"},\"source\":\"" +
                                   json_escape(source.empty() ? "provider" : source) + "\"}");
                        // The Python provider owns post-result holding and is
                        // the only layer allowed to publish lifecycle complete.
                    }
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
            // The NV12 buffer stays unrotated; with rotation the OpenCL
            // kernel maps sample coordinates so preprocessing sees the frame
            // rotated, and the logical frame dims are swapped.
            const int src_w = frame.nv12.cols;
            const int src_h = (frame.nv12.rows * 2) / 3;
            packet->width = swap_dims ? src_h : src_w;
            packet->height = swap_dims ? src_w : src_h;
            packet->crop_w = crop_w;
            packet->crop_h = crop_h;
            packet->crop_ox = static_cast<float>(crop_ox);
            packet->crop_oy = static_cast<float>(crop_oy);
            packet->nv12 = std::make_shared<cv::Mat>(std::move(frame.nv12));
            packet->gst_owner = std::move(frame.owner);
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
            try {
                const auto t0 = Clock::now();
                packet->prep = pre->preprocess(*packet->nv12, rotate_mode,
                                               packet->crop_w, packet->crop_h,
                                               packet->crop_ox, packet->crop_oy);
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

    // Thread 2: one ORT session; only this thread touches detector.
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
                                                         packet->prep.pad_y, packet->crop_w, packet->crop_h);
                } catch (const std::exception& e) {
                    if (is_tcm_resource_error(e.what())) {
                        print_tcm_resource_hint(e.what(), a.ep_affinity);
                    }
                    throw;
                }
                // Boxes come back in crop coordinates; shift them into
                // streamed-frame coordinates so every downstream consumer
                // (ROI gate, drawing, observation) works in one space.
                if (packet->crop_ox != 0.0f || packet->crop_oy != 0.0f) {
                    for (auto& d : result->detections) {
                        d.x1 += packet->crop_ox;
                        d.x2 += packet->crop_ox;
                        d.y1 += packet->crop_oy;
                        d.y2 += packet->crop_oy;
                    }
                }
                if (no_gesture_id >= 0) {
                    result->detections.erase(
                        std::remove_if(result->detections.begin(), result->detections.end(),
                                       [&](const Detection& d) {
                                           return d.class_id == no_gesture_id;
                                       }),
                        result->detections.end());
                }
                // Fold source classes onto game labels BEFORE the stability
                // signature: within-label flicker (palm -> stop, both
                // "Paper") must never reset the streak.
                rps_mapper.fold(result->detections);
                // ROI gate: keep only detections whose center falls inside
                // the region of interest (fractions of the streamed frame).
                // Excludes the robot arm's half of the view deterministically.
                if (a.roi_enabled) {
                    const float fw = static_cast<float>(result->width);
                    const float fh = static_cast<float>(result->height);
                    const float x0 = a.roi_x * fw, y0 = a.roi_y * fh;
                    const float x1 = (a.roi_x + a.roi_w) * fw;
                    const float y1 = (a.roi_y + a.roi_h) * fh;
                    result->detections.erase(
                        std::remove_if(result->detections.begin(),
                                       result->detections.end(),
                                       [&](const Detection& d) {
                                           const float cx = (d.x1 + d.x2) * 0.5f;
                                           const float cy = (d.y1 + d.y2) * 0.5f;
                                           return cx < x0 || cx > x1 || cy < y0 || cy > y1;
                                       }),
                        result->detections.end());
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

    // Thread 3 is the display/stability stage on the main thread: stability
    // state, snapshot writing and RTSP drawing all live here.
    bool first_display = true;
    uint64_t shown = 0;
    auto last_diagnostic_snapshot = start;
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
        if (item->nv12 && !item->nv12->empty()) {
            cv::cvtColor(*item->nv12, bgr, cv::COLOR_YUV2BGR_NV12);
            // Detections/geometry are in rotated-frame coordinates, so the
            // drawn-on picture and every snapshot must be rotated the same way.
            if (rotate_mode == 1) cv::rotate(bgr, bgr, cv::ROTATE_90_COUNTERCLOCKWISE);
            else if (rotate_mode == 2) cv::rotate(bgr, bgr, cv::ROTATE_90_CLOCKWISE);
            else if (rotate_mode == 3) cv::rotate(bgr, bgr, cv::ROTATE_180);
            if (a.control_fd >= 0 && adjudication_active.load() &&
                    !generic_observation_sent.load()) {
                const auto diagnostic_now = Clock::now();
                if (diagnostic_now - last_diagnostic_snapshot >= std::chrono::milliseconds(250)) {
                    const std::string diagnostic_path = save_diagnostic_snapshot(a, bgr);
                    emit_observation(a, *item, diagnostic_path, rps_mapper,
                                     a.class_names, "diagnostic_snapshot", false);
                    last_diagnostic_snapshot = diagnostic_now;
                }
                // Only a frame the consumer can actually use may advance the
                // stability streak: at least one detection must be present
                // and the folded class multiset must repeat the previous
                // frame's, so any value change restarts it. Game-level
                // completeness is the consumer's concern: the Python
                // provider validates its own expectations on the stable
                // observation and routes failures to its diagnosis path.
                // That is also why the streak can never be reported above
                // stable_frames.
                const std::string signature = detection_signature(item->detections);
                const bool evidence_usable = !item->detections.empty();
                int stable_count = 0;
                {
                    std::lock_guard<std::mutex> lock(generic_mutex);
                    if (!evidence_usable) {
                        generic_last_signature.clear();
                        generic_stable_count.store(0);
                    } else if (signature == generic_last_signature) {
                        stable_count = generic_stable_count.fetch_add(1) + 1;
                    } else {
                        generic_last_signature = signature;
                        generic_stable_count.store(1);
                        stable_count = 1;
                    }
                }
                if (generic_last_reported_count.exchange(stable_count) != stable_count) {
                    emit_event("{\"event\":\"progress\",\"phase\":\"detecting\",\"stable_count\":" +
                               std::to_string(stable_count) +
                               ",\"stable_frames\":" + std::to_string(a.stable_frames) + "}");
                }
                // Only a layout that passed the evidence gate may be
                // published as stable once the streak reaches
                // ``stable_frames``; exactly one observation per round.
                if (evidence_usable &&
                    stable_count >= a.stable_frames &&
                    !generic_observation_sent.exchange(true)) {
                    const std::string snapshot_path = save_snapshot(a, bgr, item->id);
                    emit_observation(a, *item, snapshot_path, rps_mapper, a.class_names);
                }
            }
            if (rtsp_streamer.running()) {
                // ROI visualization: labelled border around the recognized
                // region (kept cheap — a full-region overlay costs several
                // ms/frame on this board and drags the stream down).
                if (a.roi_enabled) {
                    const cv::Rect roi_r(
                        static_cast<int>(a.roi_x * bgr.cols),
                        static_cast<int>(a.roi_y * bgr.rows),
                        std::max(1, static_cast<int>(a.roi_w * bgr.cols)),
                        std::max(1, static_cast<int>(a.roi_h * bgr.rows)));
                    cv::rectangle(bgr, roi_r, cv::Scalar(0, 0, 0), 7, cv::LINE_AA);
                    cv::rectangle(bgr, roi_r, cv::Scalar(80, 255, 80), 3, cv::LINE_AA);
                    cv::putText(bgr, "ROI", cv::Point(roi_r.x + 12, roi_r.y + 34),
                                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(80, 255, 80),
                                2, cv::LINE_AA);
                }
                draw_detections(bgr, item->detections, rps_mapper, a.class_names);
                const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
                const double fps = elapsed > 0.0 ? shown / elapsed : 0.0;
                cv::putText(bgr, "DISPLAY " + std::to_string(static_cast<int>(fps)) +
                                      " FPS  DET " + std::to_string(item->detections.size()),
                            {10, 28}, cv::FONT_HERSHEY_SIMPLEX, .75,
                            {0, 255, 255}, 2, cv::LINE_AA);
                rtsp_streamer.publish(bgr);
            }
        } else {
            std::cerr << "Frame transfer/YUV conversion failed\n";
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

    // Ctrl-C/exception: stop capture first, then join workers and release
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
