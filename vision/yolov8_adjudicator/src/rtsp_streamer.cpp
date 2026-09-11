#include "rtsp_streamer.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <gst/app/gstappsrc.h>
#include <iostream>
#include <linux/videodev2.h>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <sys/ioctl.h>
#include <unistd.h>

namespace {

void ensure_gstreamer_initialized() {
    static const bool initialized = [] {
        gst_init(nullptr, nullptr);
        return true;
    }();
    (void)initialized;
}

// The MPP stack behind spacemith264enc segfaults on boards that expose no
// V4L2 M2M device at all (the MPP module probe returns NULL and the pipeline
// dereferences it), so hardware encoding must be gated on this probe instead
// of on a graceful runtime failure. Mirrors the decoder-side
// findV4l2M2mDecoder gate in gstreamer_camera.cpp.
bool vpu_m2m_device_available() {
    namespace fs = std::filesystem;
    std::error_code error;
    const fs::path video4linux("/sys/class/video4linux");
    if (!fs::exists(video4linux, error)) return false;
    for (const auto& entry : fs::directory_iterator(video4linux, error)) {
        if (error) break;
        const std::string node = "/dev/" + entry.path().filename().string();
        const int fd = ::open(node.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
        if (fd < 0) continue;
        v4l2_capability capability{};
        const bool queried = ::ioctl(fd, VIDIOC_QUERYCAP, &capability) == 0;
        ::close(fd);
        if (!queried) continue;
        const std::uint32_t caps =
            (capability.capabilities & V4L2_CAP_DEVICE_CAPS)
                ? capability.device_caps : capability.capabilities;
        if (caps & (V4L2_CAP_VIDEO_M2M | V4L2_CAP_VIDEO_M2M_MPLANE)) return true;
    }
    return false;
}

std::string normalize_host(std::string host) {
    if (host.empty() || host == "0.0.0.0" || host == "*") return "127.0.0.1";
    return host;
}

std::string normalize_path(std::string path) {
    if (path.empty()) return "/dice";
    if (path.front() != '/') path.insert(path.begin(), '/');
    while (path.size() > 1 && path.back() == '/') path.pop_back();
    return path;
}

std::string gst_error_text(GError* error) {
    const std::string text = error && error->message ? error->message : "unknown GStreamer error";
    if (error) g_error_free(error);
    return text;
}

}  // namespace

RtspStreamer::~RtspStreamer() { stop(); }

bool RtspStreamer::start(const std::string& host, int port, const std::string& path,
                         int width, int height, int fps) {
    stop();
    if (port < 1 || port > 65535 || width <= 0 || height <= 0 || fps <= 0) {
        std::cerr << "[RTSP] invalid destination port, dimensions, or FPS\n";
        return false;
    }

    host_ = normalize_host(host);
    port_ = port;
    path_ = normalize_path(path);
    width_ = width;
    height_ = height;
    fps_ = fps;
    stopping_.store(false);

    // Encoder selection mirrors the decoder-side fallback pattern: try the
    // VPU hardware encoder first, fall back to x264enc when the board has no
    // VPU (or the hardware pipeline fails to start). SPACEMIT_FORCE_SOFTWARE_
    // ENCODER forces the software path (testing / VPU-diagnostics escape
    // hatch, symmetric to SPACEMIT_FORCE_SOFTWARE_DECODER).
    bool hardware = vpu_m2m_device_available();
    if (std::getenv("SPACEMIT_FORCE_SOFTWARE_ENCODER") != nullptr) {
        std::cerr << "[Encoder] Software encoder forced by SPACEMIT_FORCE_SOFTWARE_ENCODER\n";
        hardware = false;
    } else if (!hardware) {
        std::cerr << "[Encoder] Hardware encoder unavailable (no V4L2 M2M device); "
                     "using software encoder x264enc\n";
    }
    if (hardware && !initialize_pipeline(true)) {
        std::cerr << "[Encoder] Hardware encoder spacemith264enc failed; "
                     "falling back to software encoder x264enc\n";
        hardware = false;
    }
    if (!hardware && !initialize_pipeline(false)) return false;
    hardware_encoder_ = hardware;

    running_.store(true);
    start_time_ = std::chrono::steady_clock::now();
    encoder_thread_ = std::thread(&RtspStreamer::encoder_loop, this);
    std::cerr << "[RTSP] publishing H.264 (encoder="
              << (hardware_encoder_ ? "spacemith264enc VPU" : "x264enc software")
              << ") to " << url()
              << " (RTSP client sink; MediaMTX/server must be listening)\n";
    return true;
}

bool RtspStreamer::initialize_pipeline(bool use_hardware_encoder) {
    ensure_gstreamer_initialized();

    std::ostringstream description;
    description << "appsrc name=source is-live=true do-timestamp=true format=time "
                << "block=false max-bytes=" << (width_ * height_ * 3 * 2)
                << " caps=video/x-raw,format=BGR,width=" << width_
                << ",height=" << height_ << ",framerate=" << fps_ << "/1 "
                << "! queue max-size-buffers=2 leaky=downstream "
                << "! videoconvert n-threads=2 "
                // The VPU encoder takes NV12; x264enc is fed I420 (libx264's
                // native format, always supported).
                << "! video/x-raw,format=" << (use_hardware_encoder ? "NV12" : "I420") << " ";
    if (use_hardware_encoder) {
        description << "! spacemith264enc coding-width=" << width_
                    << " code-hight=" << height_ << " ";
    } else {
        // ultrafast keeps 720p software encoding comfortably above 30 fps on
        // the X100 cores (measured ~71 fps); zerolatency keeps the demo snappy.
        description << "! x264enc tune=zerolatency speed-preset=ultrafast bitrate=2000 ";
    }
    description << "! h264parse config-interval=-1 "
                << "! video/x-h264,stream-format=byte-stream,alignment=au "
                // rtspclientsink creates the RTP payloader itself. Feeding it
                // rtph264pay output would make the sink reject the link.
                << "! rtspclientsink location=rtsp://" << host_ << ':' << port_ << path_
                << " protocols=tcp latency=0";

    std::cerr << "[RTSP] GStreamer publish pipeline: " << description.str() << "\n";
    GError* parse_error = nullptr;
    pipeline_ = gst_parse_launch(description.str().c_str(), &parse_error);
    if (!pipeline_) {
        std::cerr << "[RTSP] pipeline creation failed: " << gst_error_text(parse_error)
                  << "\n"
                  << "[RTSP] make sure the GStreamer rtspclientsink plugin is installed\n";
        return false;
    }

    GstElement* source = gst_bin_get_by_name(GST_BIN(pipeline_), "source");
    if (!source || !GST_IS_APP_SRC(source)) {
        std::cerr << "[RTSP] appsrc was not created\n";
        if (source) gst_object_unref(source);
        destroy_pipeline();
        return false;
    }
    appsrc_ = GST_APP_SRC(source);
    gst_app_src_set_stream_type(appsrc_, GST_APP_STREAM_TYPE_STREAM);
    gst_app_src_set_max_bytes(appsrc_, static_cast<guint64>(width_ * height_ * 3 * 2));

    const GstStateChangeReturn state = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (state == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "[RTSP] publish pipeline could not enter PLAYING state\n";
        destroy_pipeline();
        return false;
    }
    return true;
}

void RtspStreamer::publish(const cv::Mat& bgr) {
    if (!running_.load() || bgr.empty()) return;
    cv::Mat frame;
    if (bgr.cols != width_ || bgr.rows != height_) {
        cv::resize(bgr, frame, cv::Size(width_, height_), 0.0, 0.0, cv::INTER_LINEAR);
    } else {
        frame = bgr.clone();
    }
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_frame_ = std::move(frame);
        ++frame_sequence_;
    }
    frame_cv_.notify_one();
}

void RtspStreamer::encoder_loop() {
    std::uint64_t consumed_sequence = 0;
    while (!stopping_.load()) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(frame_mutex_);
            frame_cv_.wait_for(lock, std::chrono::milliseconds(50), [&] {
                return stopping_.load() || frame_sequence_ != consumed_sequence;
            });
            if (stopping_.load()) break;
            if (frame_sequence_ != consumed_sequence && !latest_frame_.empty()) {
                frame = latest_frame_.clone();
                consumed_sequence = frame_sequence_;
            }
        }
        if (frame.empty() || !appsrc_) {
            check_bus();
            continue;
        }

        const std::size_t bytes = frame.total() * frame.elemSize();
        GstBuffer* buffer = gst_buffer_new_allocate(nullptr, bytes, nullptr);
        GstMapInfo mapping{};
        if (!buffer || !gst_buffer_map(buffer, &mapping, GST_MAP_WRITE)) {
            if (buffer) gst_buffer_unref(buffer);
            std::cerr << "[RTSP] could not allocate/map an input frame buffer\n";
            check_bus();
            continue;
        }
        std::memcpy(mapping.data, frame.data, bytes);
        gst_buffer_unmap(buffer, &mapping);

        const auto elapsed = std::chrono::steady_clock::now() - start_time_;
        GST_BUFFER_PTS(buffer) = static_cast<GstClockTime>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
        GST_BUFFER_DTS(buffer) = GST_BUFFER_PTS(buffer);
        GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(1, GST_SECOND, fps_);

        const GstFlowReturn flow = gst_app_src_push_buffer(appsrc_, buffer);
        if (flow != GST_FLOW_OK && flow != GST_FLOW_FLUSHING && flow != GST_FLOW_EOS) {
            std::cerr << "[RTSP] appsrc push failed: " << gst_flow_get_name(flow) << "\n";
        }
        check_bus();
    }
}

void RtspStreamer::check_bus() {
    if (!pipeline_) return;
    GstBus* bus = gst_element_get_bus(pipeline_);
    GstMessage* message = nullptr;
    while ((message = gst_bus_pop_filtered(
                bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING)))) {
        GError* error = nullptr;
        gchar* debug = nullptr;
        if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
            gst_message_parse_error(message, &error, &debug);
            std::cerr << "[RTSP] GStreamer error: "
                      << (error && error->message ? error->message : "unknown") << "\n";
        } else {
            gst_message_parse_warning(message, &error, &debug);
            std::cerr << "[RTSP] GStreamer warning: "
                      << (error && error->message ? error->message : "unknown") << "\n";
        }
        if (error) g_error_free(error);
        if (debug) g_free(debug);
        gst_message_unref(message);
    }
    gst_object_unref(bus);
}

void RtspStreamer::destroy_pipeline() {
    if (pipeline_) gst_element_set_state(pipeline_, GST_STATE_NULL);
    if (appsrc_) {
        gst_object_unref(appsrc_);
        appsrc_ = nullptr;
    }
    if (pipeline_) {
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
}

void RtspStreamer::stop() {
    const bool was_running = running_.exchange(false);
    stopping_.store(true);
    frame_cv_.notify_all();
    if (encoder_thread_.joinable()) encoder_thread_.join();
    destroy_pipeline();
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_frame_.release();
        frame_sequence_ = 0;
    }
    if (was_running) std::cerr << "[RTSP] publisher stopped\n";
}

std::string RtspStreamer::url() const {
    return "rtsp://" + host_ + ':' + std::to_string(port_) + path_;
}
