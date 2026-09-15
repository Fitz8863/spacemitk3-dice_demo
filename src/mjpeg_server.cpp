#include "mjpeg_server.h"

#include <opencv2/imgcodecs.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>

namespace yuan {
namespace {

const char* kIndexHtml = R"HTML(<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>circle_detect 预览</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#0d1117; color:#e6edf3;
         font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
         display:flex; flex-direction:column; height:100vh; }
  header { padding:10px 16px; background:#161b22; border-bottom:1px solid #30363d;
           display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
  h1 { font-size:14px; margin:0; font-weight:600; letter-spacing:.02em; }
  .dot { width:8px; height:8px; border-radius:50%; background:#3fb950;
         box-shadow:0 0 8px #3fb950; display:inline-block; margin-right:6px; }
  #stat { color:#8b949e; }
  #stat b { color:#e6edf3; font-weight:600; }
  main { flex:1; display:flex; align-items:center; justify-content:center;
         padding:12px; min-height:0; }
  img { max-width:100%; max-height:100%; border-radius:8px;
        border:1px solid #30363d; background:#000; }
  footer { padding:8px 16px; color:#6e7681; font-size:12px;
           border-top:1px solid #30363d; }
  code { color:#79c0ff; }
</style></head>
<body>
<header>
  <h1><span class="dot"></span>circle_detect 实时预览</h1>
  <div id="stat">连接中…</div>
</header>
<main><img id="v" src="/stream.mjpg" alt="MJPEG 预览流"></main>
<footer>MJPEG 流：<code>/stream.mjpg</code> ・ 单帧抓图：<code>/snapshot.jpg</code>
  ・ 检测 JSON：<code>/status</code></footer>
<script>
const el = document.getElementById('stat'), img = document.getElementById('v');
img.onerror = () => { el.textContent = '流断开，正在重连…'; setTimeout(() => img.src = '/stream.mjpg?t=' + Date.now(), 1500); };
async function tick() {
  try {
    const s = await (await fetch('/status', {cache:'no-store'})).json();
    const cs = (s.circles || []).map(c =>
      `${c.side}(${c.cx.toFixed(0)},${c.cy.toFixed(0)}) r=${c.r.toFixed(0)} s=${c.score.toFixed(2)}`).join(' ・ ');
    el.innerHTML = `<b>${s.fps.toFixed(1)}</b> fps ・ `
      + `<b>${s.detect_ms.toFixed(1)}</b> ms ・ `
      + `检出 <b>${s.count}</b> ・ ${s.method} ・ 帧 <b>${s.frames}</b>`
      + (cs ? `<br>${cs}` : '');
  } catch (e) { el.textContent = '服务未响应…'; }
}
setInterval(tick, 400); tick();
</script>
</body></html>
)HTML";

bool sendAll(int fd, const char* p, size_t len) {
    while (len > 0) {
        const ssize_t w = ::send(fd, p, len, MSG_NOSIGNAL);
        if (w <= 0) {
            if (w < 0 && errno == EINTR) continue;
            return false;
        }
        p += w;
        len -= static_cast<size_t>(w);
    }
    return true;
}

bool sendStr(int fd, const std::string& s) { return sendAll(fd, s.data(), s.size()); }

// 读到请求头结束（\r\n\r\n）或缓冲区满为止
std::string readRequest(int fd) {
    std::string req;
    char buf[1024];
    while (req.size() < 8192) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        req.append(buf, static_cast<size_t>(n));
        if (req.find("\r\n\r\n") != std::string::npos) break;
    }
    return req;
}

std::string httpHeader(const char* status, const char* ctype, size_t len) {
    char h[320];
    std::snprintf(h, sizeof(h),
                  "HTTP/1.1 %s\r\n"
                  "Access-Control-Allow-Origin: *\r\n"
                  "Cache-Control: no-store, no-cache, must-revalidate\r\n"
                  "Pragma: no-cache\r\n"
                  "Connection: close\r\n"
                  "Content-Type: %s\r\n"
                  "Content-Length: %zu\r\n"
                  "\r\n",
                  status, ctype, len);
    return h;
}

}  // namespace

MjpegServer::~MjpegServer() { stop(); }

void MjpegServer::wakeClients() {
    cv_.notify_all();
}

bool MjpegServer::start(const std::string& bind_addr, int port, std::string& err) {
    if (running_.load()) { err = "预览服务已在运行"; return false; }

    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) { err = std::string("socket() 失败: ") + std::strerror(errno); return false; }

    int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::inet_pton(AF_INET, bind_addr.c_str(), &addr.sin_addr) != 1) {
        err = "非法的 bind 地址: " + bind_addr;
        ::close(listen_fd_); listen_fd_ = -1; return false;
    }
    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        err = "bind " + bind_addr + ":" + std::to_string(port) + " 失败: " + std::strerror(errno);
        ::close(listen_fd_); listen_fd_ = -1; return false;
    }
    if (::listen(listen_fd_, 16) < 0) {
        err = std::string("listen() 失败: ") + std::strerror(errno);
        ::close(listen_fd_); listen_fd_ = -1; return false;
    }

    bind_addr_ = bind_addr;
    port_ = port;
    running_.store(true);
    accept_thread_ = std::thread([this] { acceptLoop(); });
    return true;
}

void MjpegServer::stop() {
    if (!running_.exchange(false)) return;
    wakeClients();
    if (listen_fd_ >= 0) {
        ::shutdown(listen_fd_, SHUT_RDWR);
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
    if (accept_thread_.joinable()) accept_thread_.join();

    std::vector<std::thread> threads;
    {
        std::lock_guard<std::mutex> lk(clients_mu_);
        threads.swap(client_threads_);
    }
    for (auto& t : threads) if (t.joinable()) t.join();
}

void MjpegServer::acceptLoop() {
    while (running_.load()) {
        sockaddr_in peer{};
        socklen_t plen = sizeof(peer);
        const int fd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&peer), &plen);
        if (fd < 0) {
            if (errno == EINTR) continue;
            break;   // listen socket 已关闭
        }
        int one = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        {
            std::lock_guard<std::mutex> lk(clients_mu_);
            client_threads_.emplace_back([this, fd] { clientLoop(fd); });
        }
    }
}

void MjpegServer::clientLoop(int fd) {
    clients_.fetch_add(1);
    struct Dec {
        std::atomic<int>* c;
        ~Dec() { c->fetch_sub(1); }
    } dec{&clients_};

    const std::string req = readRequest(fd);
    const bool want_stream = req.find("GET /stream") != std::string::npos;
    const bool want_status = req.find("GET /status") != std::string::npos;
    const bool want_raw    = req.find("GET /raw") != std::string::npos;
    const bool want_snap   = !want_raw && req.find("GET /snapshot") != std::string::npos;

    if (want_status) {
        std::string js;
        { std::lock_guard<std::mutex> lk(mu_); js = status_json_; }
        if (!js.empty() && js.back() != '\n') js += '\n';
        sendStr(fd, httpHeader("200 OK", "application/json; charset=utf-8", js.size()));
        sendStr(fd, js);
        ::close(fd);
        return;
    }

    if (want_snap || want_raw) {
        std::vector<unsigned char> snap;
        {
            std::unique_lock<std::mutex> lk(mu_);
            const unsigned long long before = seq_;
            want_frame_.store(true);      // 让 publish() 临时恢复编码
            if (want_raw) want_raw_.store(true);
            cv_.wait_for(lk, std::chrono::milliseconds(1500),
                         [this, before] { return seq_ != before; });
            snap = want_raw ? raw_jpeg_ : jpeg_;
            want_frame_.store(false);
            want_raw_.store(false);
        }
        if (snap.empty()) {
            const std::string msg = "no frame yet";
            sendStr(fd, httpHeader("503 Service Unavailable", "text/plain", msg.size()));
            sendStr(fd, msg);
        } else {
            sendStr(fd, httpHeader("200 OK", "image/jpeg", snap.size()));
            sendAll(fd, reinterpret_cast<const char*>(snap.data()), snap.size());
        }
        ::close(fd);
        return;
    }

    // ---- 首页 / 404 ----
    if (!want_stream) {
        const bool want_index = req.find("GET / ") != std::string::npos ||
                                req.find("GET /index.html") != std::string::npos ||
                                req.find("GET /?") != std::string::npos;
        if (!want_index) {
            const std::string msg =
                "404 not found\nroutes: /  /stream.mjpg  /snapshot.jpg  /raw.jpg  /status\n";
            sendStr(fd, httpHeader("404 Not Found", "text/plain; charset=utf-8", msg.size()));
            sendStr(fd, msg);
        } else {
            const std::string html(kIndexHtml);
            sendStr(fd, httpHeader("200 OK", "text/html; charset=utf-8", html.size()));
            sendStr(fd, html);
        }
        ::close(fd);
        return;
    }

    // ---- MJPEG 流 ----
    static const char kStreamHdr[] =
        "HTTP/1.1 200 OK\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Cache-Control: no-store, no-cache, must-revalidate\r\n"
        "Pragma: no-cache\r\n"
        "Connection: close\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "\r\n";
    if (!sendAll(fd, kStreamHdr, sizeof(kStreamHdr) - 1)) { ::close(fd); return; }

    unsigned long long last = 0;
    std::vector<unsigned char> frame;
    while (running_.load()) {
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait_for(lk, std::chrono::milliseconds(500),
                         [this, last] { return seq_ != last || !running_.load(); });
            if (!running_.load()) break;
            if (seq_ == last) continue;      // 超时：回头检查 running_
            last = seq_;
            frame = jpeg_;
        }
        if (frame.empty()) continue;

        char hdr[160];
        const int hn = std::snprintf(hdr, sizeof(hdr),
                                     "--frame\r\nContent-Type: image/jpeg\r\n"
                                     "Content-Length: %zu\r\n\r\n", frame.size());
        if (hn <= 0) break;
        if (!sendAll(fd, hdr, static_cast<size_t>(hn))) break;
        if (!sendAll(fd, reinterpret_cast<const char*>(frame.data()), frame.size())) break;
        if (!sendAll(fd, "\r\n", 2)) break;
    }
    ::close(fd);
}

void MjpegServer::publish(const cv::Mat& bgr, int quality) {
    if (!running_.load() || bgr.empty()) return;
    // ★ 没人看就不编码（imencode 在 riscv64 上不便宜）；
    //   /snapshot.jpg 会临时把 want_frame_ 立起来，保证点一下就能抓到图。
    if (clients_.load() <= 0 && !want_frame_.load()) return;

    std::vector<unsigned char> buf;
    const std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, quality};
    if (!cv::imencode(".jpg", bgr, buf, params)) return;

    {
        std::lock_guard<std::mutex> lk(mu_);
        jpeg_.swap(buf);
        ++seq_;
    }
    cv_.notify_all();
}

void MjpegServer::publish(const cv::Mat& vis, const cv::Mat& raw, int quality) {
    // ★ 顺序很重要：raw 必须在 ++seq_ 之前编好。
    //   反过来的话，等 raw 的客户端会被"可见帧"的 seq_ 提前唤醒，
    //   读到的还是上一帧甚至空的 raw_jpeg_（实测首帧必 503）。
    if (running_.load() && !raw.empty() && want_raw_.load()) {
        std::vector<unsigned char> buf;
        const std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, quality};
        if (cv::imencode(".jpg", raw, buf, params)) {
            std::lock_guard<std::mutex> lk(mu_);
            raw_jpeg_.swap(buf);
        }
    }
    publish(vis, quality);   // 这里负责 ++seq_ 和 notify_all
}

void MjpegServer::setStatusJson(std::string json) {
    std::lock_guard<std::mutex> lk(mu_);
    status_json_ = std::move(json);
}

}  // namespace yuan
