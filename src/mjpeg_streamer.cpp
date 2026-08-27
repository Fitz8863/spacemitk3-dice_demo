#include "mjpeg_streamer.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

namespace {

bool send_all_impl(int fd, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::size_t offset = 0;
    while (offset < size) {
        const ssize_t written = ::send(fd, bytes + offset, size - offset, MSG_NOSIGNAL);
        if (written > 0) {
            offset += static_cast<std::size_t>(written);
        } else if (written < 0 && errno == EINTR) {
            continue;
        } else {
            return false;
        }
    }
    return true;
}

bool is_path(const std::string& request_path, const char* expected) {
    return request_path == expected || request_path.rfind(std::string(expected) + "?", 0) == 0;
}

}  // namespace

MjpegStreamer::~MjpegStreamer() { stop(); }

bool MjpegStreamer::start(const std::string& host, int port, int jpeg_quality) {
    stop();
    if (port < 1 || port > 65535 || jpeg_quality < 1 || jpeg_quality > 100) {
        std::cerr << "[Stream] invalid port or JPEG quality\n";
        return false;
    }

    host_ = host.empty() ? "0.0.0.0" : host;
    port_ = port;
    jpeg_quality_ = jpeg_quality;
    stopping_.store(false);

    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        std::cerr << "[Stream] socket failed: " << std::strerror(errno) << "\n";
        return false;
    }
    int reuse = 1;
    (void)::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<std::uint16_t>(port_));
    if (host_ == "0.0.0.0" || host_ == "*" || host_ == "") {
        address.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (::inet_pton(AF_INET, host_.c_str(), &address.sin_addr) != 1) {
        std::cerr << "[Stream] host must be an IPv4 address or 0.0.0.0: " << host_ << "\n";
        ::close(fd);
        return false;
    }

    if (::bind(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        std::cerr << "[Stream] bind " << host_ << ":" << port_ << " failed: "
                  << std::strerror(errno) << "\n";
        ::close(fd);
        return false;
    }
    if (::listen(fd, 8) != 0) {
        std::cerr << "[Stream] listen failed: " << std::strerror(errno) << "\n";
        ::close(fd);
        return false;
    }
    // Keep accept() interruptible during shutdown, including on platforms
    // where closing a descriptor from another thread does not wake it.
    timeval timeout{1, 0};
    (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    listen_fd_.store(fd);
    running_.store(true);
    accept_thread_ = std::thread(&MjpegStreamer::accept_loop, this);
    std::cerr << "[Stream] MJPEG server listening on " << host_ << ":" << port_
              << " (open http://<board-ip>:" << port_ << "/)\n";
    return true;
}

void MjpegStreamer::publish(const cv::Mat& bgr) {
    if (!running_.load() || bgr.empty()) return;
    std::vector<std::uint8_t> jpeg;
    const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};
    if (!cv::imencode(".jpg", bgr, jpeg, params)) {
        std::cerr << "[Stream] JPEG encoding failed\n";
        return;
    }
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_jpeg_ = std::move(jpeg);
        ++frame_sequence_;
    }
    frame_cv_.notify_all();
}

void MjpegStreamer::stop() {
    const bool was_running = running_.exchange(false);
    stopping_.store(true);
    frame_cv_.notify_all();

    const int listen_fd = listen_fd_.exchange(-1);
    if (listen_fd >= 0) {
        (void)::shutdown(listen_fd, SHUT_RDWR);
        ::close(listen_fd);
    }

    {
        std::lock_guard<std::mutex> lock(client_mutex_);
        for (const int client_fd : active_clients_) {
            (void)::shutdown(client_fd, SHUT_RDWR);
        }
    }

    if (accept_thread_.joinable()) accept_thread_.join();

    std::vector<std::thread> clients;
    {
        std::lock_guard<std::mutex> lock(client_mutex_);
        clients.swap(client_threads_);
    }
    for (auto& client : clients) {
        if (client.joinable()) client.join();
    }

    {
        std::lock_guard<std::mutex> lock(client_mutex_);
        active_clients_.clear();
    }
    if (was_running) std::cerr << "[Stream] MJPEG server stopped\n";
}

std::string MjpegStreamer::url() const {
    std::ostringstream stream;
    stream << "http://<board-ip>:" << port_ << "/";
    return stream.str();
}

void MjpegStreamer::accept_loop() {
    while (!stopping_.load()) {
        const int server_fd = listen_fd_.load();
        if (server_fd < 0) break;
        const int client_fd = ::accept(server_fd, nullptr, nullptr);
        if (client_fd < 0) {
            if (stopping_.load() || errno == EBADF || errno == EINVAL) break;
            if (errno == EINTR) continue;
            std::cerr << "[Stream] accept failed: " << std::strerror(errno) << "\n";
            continue;
        }

        std::lock_guard<std::mutex> lock(client_mutex_);
        if (stopping_.load()) {
            (void)::shutdown(client_fd, SHUT_RDWR);
            ::close(client_fd);
            break;
        }
        active_clients_.push_back(client_fd);
        client_threads_.emplace_back(&MjpegStreamer::handle_client, this, client_fd);
    }
}

void MjpegStreamer::handle_client(int client_fd) {
    char request_buffer[8192];
    std::size_t used = 0;
    while (used + 1 < sizeof(request_buffer)) {
        const ssize_t received = ::recv(client_fd, request_buffer + used,
                                        sizeof(request_buffer) - used - 1, 0);
        if (received > 0) {
            used += static_cast<std::size_t>(received);
            request_buffer[used] = '\0';
            if (std::strstr(request_buffer, "\r\n\r\n") != nullptr) break;
        } else if (received < 0 && errno == EINTR) {
            continue;
        } else {
            break;
        }
    }

    std::istringstream request{std::string(request_buffer, used)};
    std::string method;
    std::string target;
    std::string version;
    request >> method >> target >> version;
    if (method == "GET" && is_path(target, "/stream.mjpg")) {
        serve_stream(client_fd);
    } else if (method == "GET" && is_path(target, "/snapshot.jpg")) {
        serve_snapshot(client_fd);
    } else if (method == "GET" && (target == "/" || is_path(target, "/index.html"))) {
        serve_index(client_fd);
    } else {
        const std::string response =
            "HTTP/1.1 404 Not Found\r\nConnection: close\r\n"
            "Content-Type: text/plain\r\nContent-Length: 9\r\n\r\nNot found\n";
        (void)send_text(client_fd, response);
    }

    (void)::shutdown(client_fd, SHUT_RDWR);
    ::close(client_fd);
    remove_client(client_fd);
}

void MjpegStreamer::serve_stream(int client_fd) {
    const std::string headers =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "Cache-Control: no-cache, no-store, must-revalidate\r\n"
        "Pragma: no-cache\r\n"
        "Connection: close\r\n\r\n";
    if (!send_text(client_fd, headers)) return;

    std::uint64_t last_sequence = 0;
    while (!stopping_.load()) {
        std::vector<std::uint8_t> jpeg;
        std::uint64_t sequence = last_sequence;
        if (!wait_for_frame(jpeg, sequence, last_sequence, 1000)) continue;
        last_sequence = sequence;
        const std::string part_header =
            "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
            std::to_string(jpeg.size()) + "\r\n\r\n";
        if (!send_text(client_fd, part_header) ||
            !send_all(client_fd, jpeg.data(), jpeg.size()) ||
            !send_text(client_fd, "\r\n")) {
            return;
        }
    }
}

void MjpegStreamer::serve_snapshot(int client_fd) {
    std::vector<std::uint8_t> jpeg;
    std::uint64_t sequence = 0;
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        if (!latest_jpeg_.empty()) {
            jpeg = latest_jpeg_;
            sequence = frame_sequence_;
        }
    }
    if (jpeg.empty() && !wait_for_frame(jpeg, sequence, 0, 2000)) {
        const std::string response =
            "HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n"
            "Content-Type: text/plain\r\nContent-Length: 12\r\n\r\nNo frame yet\n";
        (void)send_text(client_fd, response);
        return;
    }
    const std::string headers =
        "HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\n"
        "Cache-Control: no-cache\r\nContent-Length: " +
        std::to_string(jpeg.size()) + "\r\nConnection: close\r\n\r\n";
    (void)send_text(client_fd, headers);
    (void)send_all(client_fd, jpeg.data(), jpeg.size());
}

void MjpegStreamer::serve_index(int client_fd) {
    const std::string body =
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>SpaceMIT K3 camera</title>"
        "<style>body{background:#111;color:#eee;font-family:sans-serif;text-align:center}"
        "img{max-width:96vw;height:auto}</style></head><body>"
        "<h2>SpaceMIT K3 camera stream</h2><img src=\"/stream.mjpg\" alt=\"MJPEG stream\">"
        "<p>Snapshot: <a href=\"/snapshot.jpg\">/snapshot.jpg</a></p></body></html>";
    const std::string headers =
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
        "Cache-Control: no-cache\r\nContent-Length: " + std::to_string(body.size()) +
        "\r\nConnection: close\r\n\r\n";
    (void)send_text(client_fd, headers);
    (void)send_text(client_fd, body);
}

bool MjpegStreamer::send_all(int client_fd, const void* data, std::size_t size) const {
    return send_all_impl(client_fd, data, size);
}

bool MjpegStreamer::send_text(int client_fd, const std::string& text) const {
    return send_all(client_fd, text.data(), text.size());
}

void MjpegStreamer::remove_client(int client_fd) {
    std::lock_guard<std::mutex> lock(client_mutex_);
    const auto it = std::find(active_clients_.begin(), active_clients_.end(), client_fd);
    if (it != active_clients_.end()) active_clients_.erase(it);
}

bool MjpegStreamer::wait_for_frame(std::vector<std::uint8_t>& jpeg,
                                   std::uint64_t& sequence,
                                   std::uint64_t previous_sequence,
                                   int timeout_ms) {
    std::unique_lock<std::mutex> lock(frame_mutex_);
    frame_cv_.wait_for(lock, std::chrono::milliseconds(std::max(1, timeout_ms)), [&] {
        return stopping_.load() ||
               (frame_sequence_ != previous_sequence && !latest_jpeg_.empty());
    });
    if (stopping_.load() || latest_jpeg_.empty()) return false;
    jpeg = latest_jpeg_;
    sequence = frame_sequence_;
    return true;
}
