#include "rtsp_streamer.h"

#include "config.h"

#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <new>
#include <sstream>

namespace yuan {
namespace {

void ensureGstInit() {
    static const bool once = [] { gst_init(nullptr, nullptr); return true; }();
    (void)once;
}

std::string gstErrorText(GError* e) {
    const std::string t = (e && e->message) ? e->message : "unknown GStreamer error";
    if (e) g_error_free(e);
    return t;
}

// 帧持有者：把 shared_ptr<Mat> 挂到 GstBuffer 上，缓冲释放时自动回收
struct GstFrameOwner {
    std::shared_ptr<const cv::Mat> frame;
};

void releaseGstFrameOwner(gpointer data) {
    delete static_cast<GstFrameOwner*>(data);
}

// ★ 零拷贝：直接包住 Mat 的内存，不 memcpy。
//   前提是 Mat 连续（isContinuous）；不连续就返回 nullptr 让上层去 clone。
GstBuffer* wrapFrame(const std::shared_ptr<const cv::Mat>& frame) {
    if (!frame || frame->empty() || !frame->isContinuous()) return nullptr;
    const gsize bytes = static_cast<gsize>(frame->total() * frame->elemSize());
    auto* owner = new (std::nothrow) GstFrameOwner{frame};
    if (!owner) return nullptr;
    GstBuffer* buf = gst_buffer_new_wrapped_full(
        GST_MEMORY_FLAG_READONLY, const_cast<guint8*>(frame->data), bytes,
        0, bytes, owner, releaseGstFrameOwner);
    if (!buf) delete owner;
    return buf;
}

}  // namespace

RtspStreamer::~RtspStreamer() { stop(); }

std::string RtspStreamer::url() const {
    return "rtsp://" + host_ + ':' + std::to_string(port_) + path_;
}

bool RtspStreamer::start(const std::string& host, int port, const std::string& path,
                         int width, int height, int fps) {
    stop();
    if (port < 1 || port > 65535 || width <= 0 || height <= 0 || fps <= 0) {
        std::fprintf(stderr, "[rtsp] 目标端口/分辨率/FPS 非法\n");
        return false;
    }
    host_ = normalize_rtsp_host(host);
    port_ = port;
    path_ = normalize_rtsp_path(path);
    width_ = width;
    height_ = height;
    fps_ = fps;
    stopping_.store(false);
    pushed_.store(0);

    if (!initializePipeline()) return false;

    running_.store(true);
    start_time_ = std::chrono::steady_clock::now();
    encoder_thread_ = std::thread(&RtspStreamer::encoderLoop, this);
    std::fprintf(stderr, "[rtsp] 已开始推流: %s  (H.264 / spacemith264enc → MediaMTX)\n",
                 url().c_str());
    return true;
}

bool RtspStreamer::initializePipeline() {
    ensureGstInit();

    // 目标分辨率取"偶数化"后的值：H.264 要求宽高为偶数
    const int w = width_ & ~1;
    const int h = height_ & ~1;
    if (w != width_ || h != height_) {
        std::fprintf(stderr, "[rtsp] 宽高取偶数: %dx%d -> %dx%d\n", width_, height_, w, h);
        width_ = w; height_ = h;
    }

    std::ostringstream d;
    d << "appsrc name=source is-live=true do-timestamp=true format=time block=false"
      << " max-bytes=" << (width_ * height_ * 3 * 2)
      << " caps=video/x-raw,format=BGR,width=" << width_
      << ",height=" << height_ << ",framerate=" << fps_ << "/1 "
      // leaky=downstream：编码跟不上就丢旧帧，不回压识别主循环
      << "! queue max-size-buffers=2 leaky=downstream "
      << "! videoconvert n-threads=2 "
      << "! video/x-raw,format=NV12 "
      << "! spacemith264enc coding-width=" << width_
      << " code-hight=" << height_ << " "
      << "! h264parse config-interval=-1 "
      << "! video/x-h264,stream-format=byte-stream,alignment=au "
      << "! rtspclientsink location=rtsp://" << host_ << ':' << port_ << path_
      << " protocols=tcp latency=0";

    std::fprintf(stderr, "[rtsp] 管线: %s\n", d.str().c_str());

    GError* perr = nullptr;
    pipeline_ = gst_parse_launch(d.str().c_str(), &perr);
    if (!pipeline_) {
        std::fprintf(stderr, "[rtsp] 建管线失败: %s\n", gstErrorText(perr).c_str());
        std::fprintf(stderr, "[rtsp] 确认 gstreamer1.0-rtsp 与 spacemith264enc 插件已安装，"
                             "且 MediaMTX 已在 %s:%d 监听\n", host_.c_str(), port_);
        return false;
    }

    GstElement* src = gst_bin_get_by_name(GST_BIN(pipeline_), "source");
    if (!src || !GST_IS_APP_SRC(src)) {
        std::fprintf(stderr, "[rtsp] 管线里没有 appsrc\n");
        if (src) gst_object_unref(src);
        destroyPipeline();
        return false;
    }
    appsrc_ = GST_APP_SRC(src);
    gst_app_src_set_stream_type(appsrc_, GST_APP_STREAM_TYPE_STREAM);
    gst_app_src_set_max_bytes(appsrc_, static_cast<guint64>(width_ * height_ * 3 * 2));

    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        std::fprintf(stderr, "[rtsp] 管线无法进入 PLAYING（MediaMTX 没起？端口不通？）\n");
        destroyPipeline();
        return false;
    }
    return true;
}

void RtspStreamer::publish(const cv::Mat& bgr) {
    if (!running_.load() || bgr.empty()) return;
    cv::Mat copy;
    if (bgr.cols != width_ || bgr.rows != height_)
        cv::resize(bgr, copy, cv::Size(width_, height_), 0, 0, cv::INTER_LINEAR);
    else
        copy = bgr.clone();     // const 版本必须克隆：调用方还要继续用
    publish(std::move(copy));
}

void RtspStreamer::publish(cv::Mat&& bgr) {
    if (!running_.load() || bgr.empty()) return;
    auto frame = std::make_shared<cv::Mat>(std::move(bgr));
    if (frame->cols != width_ || frame->rows != height_) {
        auto rs = std::make_shared<cv::Mat>();
        cv::resize(*frame, *rs, cv::Size(width_, height_), 0, 0, cv::INTER_LINEAR);
        frame = std::move(rs);
    }
    {
        std::lock_guard<std::mutex> lk(frame_mutex_);
        latest_frame_ = std::move(frame);
        ++frame_sequence_;
    }
    frame_cv_.notify_one();
}

void RtspStreamer::encoderLoop() {
    std::uint64_t consumed = 0;
    while (!stopping_.load()) {
        std::shared_ptr<const cv::Mat> frame;
        {
            std::unique_lock<std::mutex> lk(frame_mutex_);
            frame_cv_.wait_for(lk, std::chrono::milliseconds(50), [&] {
                return stopping_.load() || frame_sequence_ != consumed;
            });
            if (stopping_.load()) break;
            if (frame_sequence_ != consumed && latest_frame_) {
                frame = std::move(latest_frame_);
                consumed = frame_sequence_;
            }
        }
        if (!frame || frame->empty() || !appsrc_) { checkBus(); continue; }

        GstBuffer* buf = wrapFrame(frame);
        if (!buf) {
            std::fprintf(stderr, "[rtsp] 帧不连续，无法零拷贝包装\n");
            checkBus();
            continue;
        }
        const auto elapsed = std::chrono::steady_clock::now() - start_time_;
        GST_BUFFER_PTS(buf) = static_cast<GstClockTime>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
        GST_BUFFER_DTS(buf) = GST_BUFFER_PTS(buf);
        GST_BUFFER_DURATION(buf) = gst_util_uint64_scale_int(1, GST_SECOND, fps_);

        const GstFlowReturn flow = gst_app_src_push_buffer(appsrc_, buf);
        if (flow == GST_FLOW_OK) {
            pushed_.fetch_add(1);
        } else if (flow != GST_FLOW_FLUSHING && flow != GST_FLOW_EOS) {
            std::fprintf(stderr, "[rtsp] appsrc push 失败: %s\n", gst_flow_get_name(flow));
        }
        checkBus();
    }
}

void RtspStreamer::checkBus() {
    if (!pipeline_) return;
    GstBus* bus = gst_element_get_bus(pipeline_);
    GstMessage* msg = nullptr;
    while ((msg = gst_bus_pop_filtered(
                bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING)))) {
        GError* err = nullptr;
        gchar* dbg = nullptr;
        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
            gst_message_parse_error(msg, &err, &dbg);
            std::fprintf(stderr, "[rtsp] GStreamer 错误: %s\n",
                         (err && err->message) ? err->message : "unknown");
        } else {
            gst_message_parse_warning(msg, &err, &dbg);
            std::fprintf(stderr, "[rtsp] GStreamer 警告: %s\n",
                         (err && err->message) ? err->message : "unknown");
        }
        if (err) g_error_free(err);
        if (dbg) g_free(dbg);
        gst_message_unref(msg);
    }
    gst_object_unref(bus);
}

void RtspStreamer::destroyPipeline() {
    if (pipeline_) gst_element_set_state(pipeline_, GST_STATE_NULL);
    if (appsrc_) { gst_object_unref(appsrc_); appsrc_ = nullptr; }
    if (pipeline_) { gst_object_unref(pipeline_); pipeline_ = nullptr; }
}

void RtspStreamer::stop() {
    const bool was = running_.exchange(false);
    stopping_.store(true);
    frame_cv_.notify_all();
    if (encoder_thread_.joinable()) encoder_thread_.join();
    destroyPipeline();
    {
        std::lock_guard<std::mutex> lk(frame_mutex_);
        latest_frame_.reset();
        frame_sequence_ = 0;
    }
    if (was) std::fprintf(stderr, "[rtsp] 推流已停止（共推送 %lld 帧）\n", pushed_.load());
}

}  // namespace yuan
