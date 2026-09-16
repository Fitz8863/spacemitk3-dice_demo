#include "rtsp_streamer.h"
#include "config.h"

#include <algorithm>
#include <cctype>
#include <new>
#include <gst/app/gstappsrc.h>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <sstream>

namespace {

void ensure_gstreamer_initialized() {
    static const bool initialized = [] {
        gst_init(nullptr, nullptr);
        return true;
    }();
    (void)initialized;
}

std::string gst_error_text(GError* error) {
    const std::string text = error && error->message ? error->message : "unknown GStreamer error";
    if (error) g_error_free(error);
    return text;
}

struct GstFrameOwner {
    std::shared_ptr<const cv::Mat> frame;
};

void release_gst_frame_owner(gpointer data) {
    delete static_cast<GstFrameOwner*>(data);
}

GstBuffer* wrap_frame_without_copy(const std::shared_ptr<const cv::Mat>& frame) {
    if (!frame || frame->empty() || !frame->isContinuous()) return nullptr;
    const gsize bytes = static_cast<gsize>(frame->total() * frame->elemSize());
    auto* owner = new (std::nothrow) GstFrameOwner{frame};
    if (!owner) return nullptr;
    GstBuffer* buffer = gst_buffer_new_wrapped_full(
        GST_MEMORY_FLAG_READONLY, const_cast<guint8*>(frame->data), bytes,
        0, bytes, owner, release_gst_frame_owner);
    if (!buffer) delete owner;
    return buffer;
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

    host_ = normalize_rtsp_host(host);
    port_ = port;
    path_ = normalize_rtsp_path(path);
    width_ = width;
    height_ = height;
    fps_ = fps;
    stopping_.store(false);

    if (!initialize_pipeline()) return false;
    running_.store(true);
    start_time_ = std::chrono::steady_clock::now();
    encoder_thread_ = std::thread(&RtspStreamer::encoder_loop, this);
    std::cerr << "[RTSP] publishing SpaceMIT VPU H.264 to " << url()
              << " (RTSP client sink; MediaMTX/server must be listening)\n";
    return true;
}

bool RtspStreamer::initialize_pipeline() {
    ensure_gstreamer_initialized();

    std::ostringstream description;
    description << "appsrc name=source is-live=true do-timestamp=true format=time "
                << "block=false max-bytes=" << (width_ * height_ * 3 * 2)
                << " caps=video/x-raw,format=BGR,width=" << width_
                << ",height=" << height_ << ",framerate=" << fps_ << "/1 "
                << "! queue max-size-buffers=2 leaky=downstream "
                << "! videoconvert n-threads=2 "
                << "! video/x-raw,format=NV12 "
                << "! spacemith264enc coding-width=" << width_
                << " code-hight=" << height_ << " "
                << "! h264parse config-interval=-1 "
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
    publish(std::move(frame));
}

void RtspStreamer::publish(cv::Mat&& bgr) {
    if (!running_.load() || bgr.empty()) return;
    auto frame = std::make_shared<cv::Mat>(std::move(bgr));
    if (frame->cols != width_ || frame->rows != height_) {
        auto resized = std::make_shared<cv::Mat>();
        cv::resize(*frame, *resized, cv::Size(width_, height_), 0.0, 0.0, cv::INTER_LINEAR);
        frame = std::move(resized);
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
        std::shared_ptr<const cv::Mat> frame;
        {
            std::unique_lock<std::mutex> lock(frame_mutex_);
            frame_cv_.wait_for(lock, std::chrono::milliseconds(50), [&] {
                return stopping_.load() || frame_sequence_ != consumed_sequence;
            });
            if (stopping_.load()) break;
            if (frame_sequence_ != consumed_sequence && latest_frame_) {
                frame = std::move(latest_frame_);
                consumed_sequence = frame_sequence_;
            }
        }
        if (!frame || frame->empty() || !appsrc_) {
            check_bus();
            continue;
        }

        GstBuffer* buffer = wrap_frame_without_copy(frame);
        if (!buffer) {
            std::cerr << "[RTSP] frame is not continuous or buffer wrapping failed\n";
            check_bus();
            continue;
        }

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
        latest_frame_.reset();
        frame_sequence_ = 0;
    }
    if (was_running) std::cerr << "[RTSP] publisher stopped\n";
}

std::string RtspStreamer::url() const {
    return "rtsp://" + host_ + ':' + std::to_string(port_) + path_;
}
