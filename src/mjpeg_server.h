// 极简 MJPEG over HTTP 预览服务（零外部依赖，只用 POSIX socket + OpenCV imencode）
//
// 为什么不用 cv::imshow：板子通常从 SSH 连过去，没有 DISPLAY/WAYLAND_DISPLAY，
// imshow 开不了窗口。这里把带标注的画面编成 JPEG，用标准
// multipart/x-mixed-replace 推给浏览器 —— 手机/笔记本打开网址就能看，
// 不需要在板子上装任何东西，也不需要 X 转发。
//
// 路由：
//   GET /              交互式预览页（深色主题 + 实时统计）
//   GET /stream.mjpg   MJPEG 流（可直接丢给 <img>、VLC、ffplay）
//   GET /status        最近一帧的统计 + 检测结果 JSON（给上层程序消费）
//   GET /snapshot.jpg  最近一帧单张 JPEG（抓图用）
//
// 设计要点：采集循环只 publish 最新一帧，客户端各自按自己的速度取。
// **没有客户端连接时完全不编码 JPEG**（imencode 在 riscv64 上不便宜），
// 所以挂着预览服务对检测耗时几乎无影响。

#pragma once

#include <opencv2/core.hpp>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace yuan {

class MjpegServer {
public:
    MjpegServer() = default;
    ~MjpegServer();

    MjpegServer(const MjpegServer&) = delete;
    MjpegServer& operator=(const MjpegServer&) = delete;

    // 启动监听。bind_addr 默认 "0.0.0.0"（同网段/tailnet 可访问）
    bool start(const std::string& bind_addr, int port, std::string& err);
    void stop();

    // 发布一帧（内部做 JPEG 编码；无客户端时直接返回）
    void publish(const cv::Mat& bgr, int quality = 80);

    // 同时发布"未标注的原始帧"：只有 /raw.jpg 被请求时才编码，所以平时零成本。
    // 用途：调算法时要拿干净的原始帧，不能用带 overlay 的调试图。
    void publish(const cv::Mat& vis, const cv::Mat& raw, int quality = 80);

    // 更新 /status 的 JSON 文本
    void setStatusJson(std::string json);

    bool running() const { return running_.load(); }
    int  port() const { return port_; }
    int  clients() const { return clients_.load(); }

    // 现在真的需要画面吗？（有人连流，或有人正在等 /snapshot、/raw）
    // 调用方用它来决定要不要花 CPU 去画叠加图 —— 没人看就别画。
    bool wantsFrame() const { return clients_.load() > 0 || want_frame_.load(); }

private:
    void acceptLoop();
    void clientLoop(int fd);
    void wakeClients();

    int         listen_fd_ = -1;
    int         port_ = 0;
    std::string bind_addr_ = "0.0.0.0";
    std::atomic<bool> running_{false};
    std::thread accept_thread_;

    std::mutex               clients_mu_;
    std::vector<std::thread> client_threads_;
    std::atomic<int>         clients_{0};

    std::mutex              mu_;
    std::condition_variable cv_;
    std::vector<unsigned char> jpeg_;
    std::vector<unsigned char> raw_jpeg_;
    unsigned long long      seq_ = 0;
    std::string             status_json_ = "{}";
    // /snapshot.jpg 是"一次性拉一帧"，需要临时强制编码（平时无客户端不编码省 CPU）
    std::atomic<bool>       want_frame_{false};
    std::atomic<bool>       want_raw_{false};
};

}  // namespace yuan
