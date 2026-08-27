#pragma once

#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

// Small HTTP MJPEG server for LAN preview. Browsers can display the stream
// directly with an <img> tag, without requiring WebRTC/RTSP support.
class MjpegStreamer {
public:
    MjpegStreamer() = default;
    ~MjpegStreamer();
    MjpegStreamer(const MjpegStreamer&) = delete;
    MjpegStreamer& operator=(const MjpegStreamer&) = delete;

    bool start(const std::string& host, int port, int jpeg_quality);
    void publish(const cv::Mat& bgr);
    void stop();
    bool running() const { return running_.load(); }
    std::string url() const;

private:
    void accept_loop();
    void handle_client(int client_fd);
    void serve_stream(int client_fd);
    void serve_snapshot(int client_fd);
    void serve_index(int client_fd);
    bool send_all(int client_fd, const void* data, std::size_t size) const;
    bool send_text(int client_fd, const std::string& text) const;
    bool wait_for_frame(std::vector<std::uint8_t>& jpeg, std::uint64_t& sequence,
                        std::uint64_t previous_sequence, int timeout_ms);
    void remove_client(int client_fd);

    std::string host_ = "0.0.0.0";
    int port_ = 8080;
    int jpeg_quality_ = 80;
    std::atomic<bool> stopping_{false};
    std::atomic<bool> running_{false};
    std::atomic<int> listen_fd_{-1};
    std::thread accept_thread_;

    mutable std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    std::vector<std::uint8_t> latest_jpeg_;
    std::uint64_t frame_sequence_ = 0;

    std::mutex client_mutex_;
    std::vector<int> active_clients_;
    std::vector<std::thread> client_threads_;
};
