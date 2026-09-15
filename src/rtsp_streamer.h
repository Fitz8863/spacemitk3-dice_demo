// RTSP 推流（H.264，经 SpaceMIT VPU 硬编）→ MediaMTX
//
// 管线（与 dice-game 的 yolov8_segdetect 保持一致，已在该工程验证）：
//
//   appsrc → queue(leaky) → videoconvert → NV12
//          → spacemith264enc → h264parse → rtspclientsink → MediaMTX
//
// 三个要点：
//   1) 编码线程异步 + latest-only：网络或编码变慢时只丢旧帧，
//      绝不阻塞摄像头采集 / 识别主循环。
//   2) 帧零拷贝包装：用 gst_buffer_new_wrapped_full + shared_ptr 持有者，
//      不 clone（BGR 720p 一帧就是 2.7MB，clone 会很贵）。
//   3) rtspclientsink 自己会建 RTP payloader，喂它 rtph264pay 的输出反而链接失败。

#pragma once

#include "circle_detector.h"

#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

// 前置声明，避免把 GStreamer 头暴露给所有包含者
typedef struct _GstElement GstElement;
typedef struct _GstAppSrc  GstAppSrc;

namespace yuan {

class RtspStreamer {
public:
    RtspStreamer() = default;
    ~RtspStreamer();
    RtspStreamer(const RtspStreamer&) = delete;
    RtspStreamer& operator=(const RtspStreamer&) = delete;

    bool start(const std::string& host, int port, const std::string& path,
               int width, int height, int fps);
    void stop();

    // 把一帧 BGR 交给推流（内部只保留最新一帧）。
    // rvalue 版本会把 Mat 移走，调用方之后不得再使用该 Mat。
    void publish(const cv::Mat& bgr);
    void publish(cv::Mat&& bgr);

    // 把"未绘制的原始帧 + 检测结果"交给推流，由编码线程负责画标注。
    // 这样绘制（~10ms/帧）就离开了识别主循环的关键路径，
    // 检测结果已是定型数据，换线程绘制不影响精度。
    void publish(OverlayJob&& job);

    bool running() const { return running_.load(); }
    std::string url() const;
    // 异步线程有没有真的在出帧（用于判断 VPU 编码是否活着）
    long long pushed() const { return pushed_.load(); }
    // 编码线程里画标注的平均耗时（ms），用于确认绘制确实被挪走了
    double drawMsAvg() const { return draw_ms_avg_.load(); }

private:
    bool initializePipeline();
    void destroyPipeline();
    void encoderLoop();
    void checkBus();

    std::string host_ = "127.0.0.1";
    int port_ = 8554;
    std::string path_ = "/dice/circles";
    int width_ = 0;
    int height_ = 0;
    int fps_ = 30;

    std::atomic<bool> running_{false};
    std::atomic<bool> stopping_{false};
    std::atomic<long long> pushed_{0};

    GstElement* pipeline_ = nullptr;
    GstAppSrc*  appsrc_ = nullptr;
    std::thread encoder_thread_;

    std::mutex              frame_mutex_;
    std::condition_variable frame_cv_;
    std::shared_ptr<const cv::Mat> latest_frame_;
    // 待绘制的任务（绘制在编码线程里做）。与 latest_frame_ 互斥使用。
    std::shared_ptr<OverlayJob>    latest_job_;
    std::uint64_t           frame_sequence_ = 0;
    std::chrono::steady_clock::time_point start_time_{};
    // 诊断：编码线程里画一帧平均花多少 ms
    std::atomic<double>     draw_ms_avg_{0.0};
};

}  // namespace yuan
