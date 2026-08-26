#include "gstreamer_camera.h"
#include "llm_dice_verifier.h"
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
    bool no_display = false;
    int max_frames = 0;
    bool self_test = false;
    std::string dump_input;
    std::string llm_url = "https://api.rvcompute.com:60000/v1";
    std::string llm_model = "gpt-5.4-mini";
    std::string llm_system_prompt;
    std::string llm_user_prompt_template;
    std::string llm_api_key;
    bool no_llm = false;
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

static bool load_config(const std::string& path, Args& a) {
    try {
        cv::FileStorage file(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
        if (!file.isOpened()) {
            std::cerr << "Cannot open JSON config: " << path << "\n";
            return false;
        }

        const cv::FileNode root = file.root();
        read_config_value(root, "model", a.model);
        read_config_value(root, "camera", a.camera);
        read_config_value(root, "width", a.width);
        read_config_value(root, "height", a.height);
        read_config_value(root, "fps", a.fps);
        read_config_value(root, "intra_threads", a.intra_threads);
        read_config_value(root, "ep_affinity", a.ep_affinity);
        read_config_value(root, "conf", a.conf);
        read_config_value(root, "focus", a.focus);
        read_config_value(root, "zoom", a.zoom);

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
            read_config_value(llm, "url", a.llm_url);
            read_config_value(llm, "model", a.llm_model);
            read_config_value(llm, "system_prompt", a.llm_system_prompt);
            read_config_value(llm, "user_prompt_template", a.llm_user_prompt_template);
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
              << "  --camera N         V4L2 camera index\n"
              << "  --device PATH      explicit V4L2 node, overrides --camera\n"
              << "  --width N --height N --fps N\n"
              << "  --conf FLOAT       confidence threshold\n"
              << "  --queue-depth N    keep up to N frames per pipeline queue\n"
              << "  --focus N          fixed manual focus (-1 unchanged)\n"
              << "  --zoom N           zoom absolute value (-1 unchanged)\n"
              << "  --intra-threads N  SpaceMIT EP threads\n"
              << "  --ep-affinity LIST bind EP threads to cores, e.g. 14;15\n"
              << "  --no-display       run pipeline without window\n"
              << "  --max-frames N     stop after N frames enter preprocess (0=unlimited)\n"
              << "  --dump-input PATH  dump first preprocessed tensor as float32\n"
              << "  --self-test        initialize OpenCL GPU and model, run one inference\n"
              << "  --llm-url URL      override the config LLM base URL\n"
              << "  --llm-model NAME   override the config LLM model\n"
              << "  --no-llm           disable LLM verification\n";
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
    if (!a.no_llm) {
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
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string k = argv[i];
            const char* v = nullptr;
            if (k == "--config" && (v = need(i))) a.config_path = v;
            else if (k == "--model" && (v = need(i))) a.model = v;
            else if (k == "--camera" && (v = need(i))) a.camera = std::stoi(v);
            else if (k == "--device" && (v = need(i))) a.device = v;
            else if (k == "--width" && (v = need(i))) a.width = std::stoi(v);
            else if (k == "--height" && (v = need(i))) a.height = std::stoi(v);
            else if (k == "--fps" && (v = need(i))) a.fps = std::stoi(v);
            else if (k == "--conf" && (v = need(i))) a.conf = std::stof(v);
            else if (k == "--queue-depth" && (v = need(i))) {
                a.queue_depth = static_cast<std::size_t>(std::stoul(v));
            } else if (k == "--focus" && (v = need(i))) a.focus = std::stoi(v);
            else if (k == "--zoom" && (v = need(i))) a.zoom = std::stoi(v);
            else if (k == "--intra-threads" && (v = need(i))) a.intra_threads = std::stoi(v);
            else if (k == "--ep-affinity" && (v = need(i))) a.ep_affinity = v;
            else if (k == "--max-frames" && (v = need(i))) a.max_frames = std::stoi(v);
            else if (k == "--no-display") a.no_display = true;
            else if (k == "--dump-input" && (v = need(i))) a.dump_input = v;
            else if (k == "--self-test") a.self_test = true;
            else if (k == "--llm-url" && (v = need(i))) a.llm_url = v;
            else if (k == "--llm-model" && (v = need(i))) a.llm_model = v;
            else if (k == "--no-llm") a.no_llm = true;
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
        if (side == 0) {
            ++result.first_count;
            result.first_sum += d.class_id + 1;
        } else {
            ++result.second_count;
            result.second_sum += d.class_id + 1;
        }
    }

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
    cv::rectangle(bgr, cv::Rect(8, 8, std::min(bgr.cols - 16, 760), 62),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(bgr, judgment.overlay, cv::Point(18, 48), cv::FONT_HERSHEY_SIMPLEX,
                .7, color, 2, cv::LINE_AA);
}

static LlmWinner yolo_winner(const DiceJudgment& judgment) {
    if (judgment.first_sum > judgment.second_sum) return LlmWinner::Left;
    if (judgment.second_sum > judgment.first_sum) return LlmWinner::Right;
    return LlmWinner::Tie;
}

static const char* winner_label(LlmWinner winner) {
    switch (winner) {
    case LlmWinner::Left: return "LEFT";
    case LlmWinner::Right: return "RIGHT";
    case LlmWinner::Tie: return "TIE";
    default: return "UNKNOWN";
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
    if (!std::filesystem::exists(a.model)) {
        std::cerr << "Model not found: " << a.model << "\n";
        return 2;
    }

    OpenClPreprocessor pre;
    if (!pre.init()) return 4;
    Yolov8Detector detector;
    if (!detector.init(a.model, a.intra_threads, a.ep_affinity)) return 5;
    LlmDiceVerifier llm_verifier({a.llm_url, a.llm_api_key, a.llm_model,
                                  a.llm_system_prompt, a.llm_user_prompt_template});
    if (!a.no_llm && !llm_verifier.configured()) {
        std::cerr << "[LLM] verification disabled: set DICE_LLM_API_KEY (the key is never stored in the repository).\n";
    }
    if (a.self_test) {
        try {
            // Exercise the actual OpenCL NV12 -> tensor path as well as the
            // SpaceMIT EP model path. This catches kernel/image/queue errors
            // that a model-only zero-tensor test cannot detect.
            constexpr int synthetic_width = 1280;
            constexpr int synthetic_height = 720;
            cv::Mat synthetic_nv12(synthetic_height * 3 / 2, synthetic_width,
                                   CV_8UC1, cv::Scalar(128));
            const auto prep_result = pre.preprocess(synthetic_nv12);
            if (!prep_result.data || prep_result.data->size() != 3 * 640 * 640) {
                throw std::runtime_error("OpenCL self-test returned an invalid tensor");
            }
            for (float value : *prep_result.data) {
                if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
                    throw std::runtime_error("OpenCL self-test returned invalid tensor values");
                }
            }
            const auto ds = detector.infer(prep_result.data->data(), prep_result.data->size(),
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
    if (!a.no_display) {
        try {
            cv::namedWindow("yolov8-k3", cv::WINDOW_NORMAL);
            cv::resizeWindow("yolov8-k3", a.width, a.height);
        } catch (const cv::Exception& e) {
            std::cerr << "Display initialization failed: " << e.what() << "\n";
            camera.close();
            return 7;
        }
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
                packet->prep = pre.preprocess(*packet->nv12);
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
                    result->detections = detector.infer(packet->prep.data->data(), packet->prep.data->size(),
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
        bool attempted = false;
        bool succeeded = false;
        bool agreement = false;
        bool printed = false;
        int left_sum = 0;
        int right_sum = 0;
        LlmWinner yolo_winner = LlmWinner::Unknown;
        LlmWinner llm_winner = LlmWinner::Unknown;
    } llm_state;

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
        DiceJudgment judgment;
        if (item->nv12 && !item->nv12->empty()) {
            cv::cvtColor(*item->nv12, bgr, cv::COLOR_YUV2BGR_NV12);
            judgment = judge_dice(bgr, item->detections);
            DiceJudgment display_judgment = judgment;
            static std::string last_judgment;
            if (!judgment.valid) {
                if (judgment.message != last_judgment) {
                    std::cout << "Dice judgment: " << judgment.message << "\n";
                    last_judgment = judgment.message;
                }
            } else if (a.no_llm) {
                display_judgment.valid = false;
                display_judgment.overlay = "LLM verification disabled; winner suppressed";
            } else {
                if (!llm_state.attempted) {
                    // Freeze the exact first valid 5+5 snapshot. The single LLM
                    // answer must never be reused to approve a later, different frame.
                    llm_state.attempted = true;
                    llm_state.left_sum = judgment.first_sum;
                    llm_state.right_sum = judgment.second_sum;
                    llm_state.yolo_winner = yolo_winner(judgment);

                    std::string llm_error;
                    llm_state.succeeded = llm_verifier.verify_once(
                        judgment.first_name, judgment.second_name,
                        judgment.first_sum, judgment.second_sum,
                        llm_state.llm_winner, llm_error);
                    if (llm_state.succeeded) {
                        llm_state.agreement =
                            llm_state.llm_winner == llm_state.yolo_winner;
                        std::cout << "[LLM] one-shot result="
                                  << winner_label(llm_state.llm_winner) << "\n";
                    } else {
                        std::cerr << "[LLM] one-shot verification failed: "
                                  << llm_error << "\n";
                    }
                }

                const bool same_snapshot =
                    judgment.first_sum == llm_state.left_sum &&
                    judgment.second_sum == llm_state.right_sum &&
                    yolo_winner(judgment) == llm_state.yolo_winner;
                if (llm_state.succeeded && llm_state.agreement && same_snapshot) {
                    if (!llm_state.printed) {
                        std::cout << "Dice judgment verified by YOLO + LLM: "
                                  << judgment.message << "\n";
                        llm_state.printed = true;
                    }
                } else {
                    display_judgment.valid = false;
                    if (!llm_state.succeeded) {
                        display_judgment.overlay =
                            "LLM verification unavailable; winner suppressed";
                    } else if (!llm_state.agreement) {
                        display_judgment.overlay =
                            "LLM/YOLO mismatch; winner suppressed";
                    } else {
                        display_judgment.overlay =
                            "Frame differs from verified snapshot; winner suppressed";
                    }
                }
            }
            if (!a.no_display) {
                draw_detections(bgr, item->detections);
                draw_divider_and_judgment(bgr, display_judgment);
                const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
                const double fps = elapsed > 0.0 ? shown / elapsed : 0.0;
                cv::putText(bgr, "DISPLAY " + std::to_string(static_cast<int>(fps)) +
                                      " FPS  DET " + std::to_string(item->detections.size()),
                            {10, 96}, cv::FONT_HERSHEY_SIMPLEX, .75,
                            {0, 255, 255}, 2, cv::LINE_AA);
                cv::imshow("yolov8-k3", bgr);
                const int key = cv::waitKey(1) & 0xff;
                if (key == 'q' || key == 27) abort.store(true);
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
