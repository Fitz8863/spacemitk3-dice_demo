#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <opencv2/core.hpp>

// Publishes the application's rendered frames to an RTSP server. The frames
// are encoded by the SpaceMIT VPU through spacemith264enc and sent as H.264
// RTP over RTSP (typically to MediaMTX running on the same board).
class RtspStreamer {
public:
    RtspStreamer() = default;
    ~RtspStreamer();
    RtspStreamer(const RtspStreamer&) = delete;
    RtspStreamer& operator=(const RtspStreamer&) = delete;

    bool start(const std::string& host, int port, const std::string& path,
               int width, int height, int fps);
    void publish(const cv::Mat& bgr);
    void stop();
    bool running() const { return running_.load(); }
    std::string url() const;

private:
    bool initialize_pipeline();
    void destroy_pipeline();
    void encoder_loop();
    void check_bus();

    std::string host_ = "127.0.0.1";
    int port_ = 8554;
    std::string path_ = "/dice";
    int width_ = 0;
    int height_ = 0;
    int fps_ = 25;

    std::atomic<bool> running_{false};
    std::atomic<bool> stopping_{false};

    GstElement* pipeline_ = nullptr;
    GstAppSrc* appsrc_ = nullptr;
    std::thread encoder_thread_;

    std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    cv::Mat latest_frame_;
    std::uint64_t frame_sequence_ = 0;
    std::chrono::steady_clock::time_point start_time_{};
};
