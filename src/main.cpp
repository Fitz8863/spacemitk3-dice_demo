// circle_detect —— K3 摄像头圆圈识别（纯传统图像算法，无模型）
//
// 用法示例：
//   ./build/circle_detect --image frame.jpg --debug-dir out/      # 单图
//   ./build/circle_detect --image data/ --summary                 # 整个目录批量
//   ./build/circle_detect --device /dev/video1 --preview          # 浏览器实时预览
//   ./build/circle_detect --device /dev/video1 --show             # 板子桌面窗口
//   ./build/circle_detect --device 1 --zoom 160 --save-video out.avi
//   ./build/circle_detect --self-test                             # 合成图自检
//
// 输出：stdout 每帧一行 JSON（JSON Lines），便于被上层脚本/服务消费。
//
// 可视化有四种出口，可以任意组合：
//   --preview [PORT]    MJPEG over HTTP，浏览器直接看（板子走 SSH 时用这个）
//   --show              OpenCV 窗口（需要板子本地有 DISPLAY/Wayland）
//   --debug-dir DIR     overlay_XXXX.jpg + mask_XXXX.png 落盘
//   --save-video FILE   带标注的视频文件

#include "circle_detector.h"
#include "frame_source.h"
#include "mjpeg_server.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>

namespace fs = std::filesystem;
using namespace yuan;

namespace {

// Ctrl-C / SIGTERM 时优雅退出：否则 --save-video 的 AVI 索引写不完，文件放不出来
std::atomic<bool> g_stop{false};
void onSignal(int) { g_stop.store(true); }

struct Args {
    std::string image;
    std::string video;
    std::string device = "/dev/video1";
    std::string backend = "auto";
    std::string debug_dir;
    std::string out_json;
    std::string save_video;
    std::string preview_bind = "0.0.0.0";

    int    width = 1280, height = 720, fps = 30;
    int    frames = 0;       // 0 = 图片一张 / 视频全部 / 摄像头无限
    int    zoom = -1, focus = -1;
    int    expected = 0;
    int    max_circles = 4;
    int    work_width = 640;
    int    sat_max = 70, val_min = 110;
    int    close_ksize = 9, open_ksize = 5;
    int    preview_port = 0;      // 0 = 关
    int    preview_width = 0;     // 0 = 原尺寸推流；>0 先缩到这个宽度再编码（省 CPU）
    int    jpeg_quality = 80;
    int    threads = 0;           // 0 = OpenCV 默认；1 = 强制单线程（更省 CPU）
    double min_radius_frac = 0.05, max_radius_frac = 0.35;
    double min_circularity = 0.55, min_fill_ratio = 0.75, min_inlier_ratio = 0.55;
    double ring_dark_margin = 22.0, ring_min_ratio = 0.55;
    double max_axis_ratio = 2.5;
    double hough_dp = 1.2, hough_param1 = 120, hough_param2 = 38, hough_min_dist_frac = 0.20;
    double smooth_alpha = 0.0;   // >0 开启时序平滑
    std::string method = "auto";
    bool   no_ring_check = false;
    bool   no_ellipse = false;    // 关掉椭圆拟合（强制按圆拟合，对比用）
    bool   no_fill_holes = false; // 关掉盘内空洞填充（对比用）
    bool   loop = false;
    bool   show = false, quiet = false, verbose = false, summary = false, self_test = false;
    bool   no_hud = false, no_mask_inset = false;
};

void usage() {
    std::puts(
        "circle_detect —— K3 摄像头圆圈识别（传统 CV，无模型）\n"
        "\n"
        "输入:\n"
        "  --image PATH        单张图片，或图片目录（批量）\n"
        "  --video PATH        视频文件\n"
        "  --device DEV        摄像头，/dev/video1 或索引 1（默认 /dev/video1）\n"
        "  --width/--height/--fps    采集参数（默认 1280x720@30）\n"
        "  --backend auto|v4l2|gst   采集后端（默认 auto：先 V4L2，失败退 GStreamer）\n"
        "  --zoom N            设 zoom_absolute（<0 = 不动，默认 -1）\n"
        "  --focus N           关自动对焦并设 focus_absolute（<0 = 不动，默认 -1）\n"
        "  --frames N          处理多少帧后停止（0 = 不限/全部）\n"
        "  --loop              图片/视频循环播放（配合 --preview 做常驻预览）\n"
        "\n"
        "可视化（可组合）:\n"
        "  --preview [PORT]    开 MJPEG HTTP 预览，浏览器看带圈的实时画面（默认端口 8099）\n"
        "  --preview-bind ADDR 预览监听地址（默认 0.0.0.0）\n"
        "  --preview-width N   预览推流宽度（0=原尺寸；设 640 可省 ~4 倍编码开销）\n"
        "  --jpeg-quality N    预览 JPEG 质量 1..100（默认 80）\n"
        "  --show              开 OpenCV 窗口（需板子本地有图形会话）\n"
        "  --debug-dir DIR     保存 overlay_XXXX.jpg / mask_XXXX.png\n"
        "  --save-video FILE   保存带标注的视频（.avi / .mp4）\n"
        "  --no-hud            预览上不画顶部状态条\n"
        "  --no-mask-inset     预览上不画右下角掩码缩略图\n"
        "\n"
        "算法:\n"
        "  --method auto|mask|hough   检测路径（默认 auto）\n"
        "  --work-width N      工作分辨率宽度（0=原图，默认 640，越小越快）\n"
        "  --sat-max N         白垫饱和度上限（默认 70）\n"
        "  --val-min N         白垫亮度下限（默认 110）\n"
        "  --close-ksize N     闭运算核（填骰子空洞，默认 9）\n"
        "  --open-ksize N      开运算核（去噪，默认 5）\n"
        "  --min-radius-frac F 最小半径 = F*min(w,h)（默认 0.05）\n"
        "  --max-radius-frac F 最大半径 = F*min(w,h)（默认 0.35）\n"
        "  --min-circularity F 圆度下限（默认 0.55）\n"
        "  --min-fill-ratio F  填充率下限（默认 0.75）\n"
        "  --min-inlier-ratio F RANSAC 内点比例下限（默认 0.55）\n"
        "  --ring-margin F     环带需比内盘暗多少（默认 22）\n"
        "  --ring-ratio F      环带角度覆盖下限（默认 0.55）\n"
        "  --no-ring-check     关闭环带验证（调试用）\n"
        "  --no-ellipse        关掉椭圆拟合，强制按正圆拟合（对比用）\n"
        "  --no-fill-holes     关掉\"填盘内空洞\"（对比用；关掉后盘上放东西会掉检）\n"
        "  --max-axis-ratio F  长短轴比上限，超过就认为不是盘子（默认 2.5）\n"
        "  --expected N        期望圆个数，不够时触发 Hough 兜底；只保留最高的 N 个\n"
        "  --max-circles N     最多输出几个（默认 4）\n"
        "  --smooth F          时序平滑系数 0..1（默认 0=关）\n"
        "  --threads N         识别用几个 CPU 线程（0=OpenCV 默认）\n"
        "                      实测多核只快 ~4ms 却多烧 16ms CPU，建议 1\n"
        "\n"
        "输出:\n"
        "  --out-json FILE     把逐帧 JSON 同时写入文件\n"
        "  --summary           结束打印汇总 JSON\n"
        "  --quiet             不打印逐帧 JSON\n"
        "  --verbose           打印候选接受/拒绝的详细原因\n"
        "  --self-test         合成图自检（无需摄像头）\n"
        "  -h, --help\n");
}

double nowSec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// 滑动窗口帧率
struct FpsMeter {
    std::deque<double> ts;
    void tick() {
        ts.push_back(nowSec());
        while (ts.size() > 31) ts.pop_front();
    }
    double value() const {
        if (ts.size() < 2) return 0.0;
        const double d = ts.back() - ts.front();
        return d > 0 ? double(ts.size() - 1) / d : 0.0;
    }
};

// 非回环 IPv4 地址（用于打印可访问的预览网址）
std::vector<std::string> localIPv4() {
    std::vector<std::string> out;
    ifaddrs* ifa = nullptr;
    if (::getifaddrs(&ifa) != 0) return out;
    for (ifaddrs* p = ifa; p; p = p->ifa_next) {
        if (!p->ifa_addr || p->ifa_addr->sa_family != AF_INET) continue;
        const auto* sin = reinterpret_cast<const sockaddr_in*>(p->ifa_addr);
        char buf[INET_ADDRSTRLEN] = {0};
        if (!::inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf))) continue;
        const std::string ip(buf);
        if (ip == "127.0.0.1") continue;
        out.push_back(ip);
    }
    ::freeifaddrs(ifa);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::string jsonEscape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof(b), "\\u%04x", c);
                    o += b;
                } else {
                    o += c;
                }
        }
    }
    return o;
}

std::string frameJson(const CircleResult& c, size_t idx) {
    char b[640];
    std::snprintf(b, sizeof(b),
                  "{\"id\":%zu,\"cx\":%.2f,\"cy\":%.2f,\"r\":%.2f,"
                  "\"a\":%.2f,\"b\":%.2f,\"angle\":%.1f,\"axis_ratio\":%.3f,\"ellipse\":%s,"
                  "\"side\":\"%s\",\"score\":%.3f,\"circularity\":%.3f,"
                  "\"fill\":%.3f,\"inlier\":%.3f,\"ring\":%.3f,\"contrast\":%.1f,"
                  "\"method\":\"%s\"}",
                  idx, c.cx, c.cy, c.r,
                  c.a, c.b, c.angle, c.axis_ratio, c.is_ellipse ? "true" : "false",
                  c.side == 0 ? "LEFT" : (c.side == 1 ? "RIGHT" : "UNKNOWN"),
                  c.score, c.circularity, c.fill_ratio, c.inlier_ratio,
                  c.ring_ratio, c.contrast, c.method.c_str());
    return b;
}

// 时序平滑：按"从左到右"的序号做指数滑动平均，数量变化时重置
struct Smoother {
    double alpha = 0.0;
    std::vector<cv::Point3d> state;
    void apply(std::vector<CircleResult>& cs) {
        if (alpha <= 0) return;
        if (state.size() != cs.size()) {
            state.clear();
            for (const auto& c : cs) state.emplace_back(c.cx, c.cy, c.r);
            return;
        }
        for (size_t i = 0; i < cs.size(); ++i) {
            state[i].x = alpha * cs[i].cx + (1 - alpha) * state[i].x;
            state[i].y = alpha * cs[i].cy + (1 - alpha) * state[i].y;
            state[i].z = alpha * cs[i].r  + (1 - alpha) * state[i].z;
            cs[i].cx = state[i].x; cs[i].cy = state[i].y; cs[i].r = state[i].z;
        }
    }
};

std::vector<std::string> listImages(const std::string& dir) {
    std::vector<std::string> out;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp")
            out.push_back(e.path().string());
    }
    std::sort(out.begin(), out.end());
    return out;
}

int runSelfTest(const CircleParams& base) {
    int fails = 0;
    bool dump_next_trace = false;
    auto check = [&](bool ok, const char* what) {
        std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
        if (!ok) { ++fails; dump_next_trace = true; }
    };
    auto trace = [&](const DetectResult& r) {
        for (const auto& t : r.trace) std::printf("          | %s\n", t.c_str());
    };

    CircleParams p = base;
    p.expected = 2;
    p.method = "auto";
    p.trace = true;          // 失败时能看到候选被拒的确切原因
    CircleDetector det(p);

    std::puts("[self-test] 1) 合成红蓝地垫 + 两个白垫深环 -> 应检出 2 个");
    {
        const int W = 1280, H = 720;
        const double cxL = 0.20 * W, cxR = 0.72 * W, cy = 0.62 * H, r = 0.115 * H;
        cv::Mat scene = makeSyntheticScene(W, H, 0.20, 0.72, 0.62, 0.115);
        DetectResult res = det.detect(scene);
        std::printf("        检出 %zu 个 (method=%s, %.1fms)\n",
                    res.circles.size(), res.method_used.c_str(), res.latency_ms);
        for (const auto& c : res.circles)
            std::printf("          cx=%.1f cy=%.1f r=%.1f score=%.3f side=%s\n",
                        c.cx, c.cy, c.r, c.score, c.side == 0 ? "LEFT" : "RIGHT");
        check(res.circles.size() == 2, "数量 == 2");
        if (res.circles.size() == 2) {
            const double tol = 0.15 * r;
            check(std::hypot(res.circles[0].cx - cxL, res.circles[0].cy - cy) < tol &&
                  std::hypot(res.circles[1].cx - cxR, res.circles[1].cy - cy) < tol,
                  "圆心误差 < 15% r");
            check(std::fabs(res.circles[0].r - r) < tol && std::fabs(res.circles[1].r - r) < tol,
                  "半径误差 < 15% r");
            check(res.circles[0].side == 0 && res.circles[1].side == 1, "左右标注正确");
        }
    }

    std::puts("[self-test] 2) 纯地垫（无白垫）-> 应检出 0 个");
    {
        cv::Mat mat(720, 1280, CV_8UC3);
        mat(cv::Rect(0, 0, 640, 720)).setTo(cv::Scalar(40, 40, 200));
        mat(cv::Rect(640, 0, 640, 720)).setTo(cv::Scalar(200, 90, 30));
        DetectResult res = det.detect(mat);
        std::printf("        检出 %zu 个\n", res.circles.size());
        check(res.circles.empty(), "无误检");
    }

    std::puts("[self-test] 3) 地垫上放一个白色方块 -> 应被几何判据拒绝");
    {
        cv::Mat mat(720, 1280, CV_8UC3);
        mat(cv::Rect(0, 0, 640, 720)).setTo(cv::Scalar(40, 40, 200));
        mat(cv::Rect(640, 0, 640, 720)).setTo(cv::Scalar(200, 90, 30));
        cv::rectangle(mat, cv::Rect(200, 300, 180, 180), cv::Scalar(240, 240, 240), -1);
        DetectResult res = det.detect(mat);
        std::printf("        检出 %zu 个\n", res.circles.size());
        check(res.circles.empty(), "方块未误判为圆");
    }

    std::puts("[self-test] 4) 白垫边缘搭一颗骰子 -> 圆心/半径应仍然准确（RANSAC 抗凸起）");
    {
        const int W = 1280, H = 720;
        const double cxL = 0.20 * W, cy = 0.62 * H, r = 0.115 * H;
        cv::Mat scene = makeSyntheticScene(W, H, 0.20, 0.72, 0.62, 0.115);
        cv::rectangle(scene, cv::Rect((int)(cxL - r - 34), (int)(cy - r - 34), 88, 88),
                      cv::Scalar(240, 240, 240), -1);
        DetectResult res = det.detect(scene);
        std::printf("        检出 %zu 个\n", res.circles.size());
        check(res.circles.size() == 2, "数量 == 2");
        if (!res.circles.empty()) {
            const double tol = 0.18 * r;
            const double e = std::hypot(res.circles[0].cx - cxL, res.circles[0].cy - cy);
            std::printf("        左盘圆心误差 = %.2f px (容差 %.1f), r=%.1f (真值 %.1f)\n",
                        e, tol, res.circles[0].r, r);
            check(e < tol, "左盘圆心仍在容差内");
        }
    }

    std::puts("[self-test] 5) 斜视：白垫画成 1.6:1 的椭圆 + 一侧倾斜 25 度 -> 应检出椭圆");
    {
        const int W = 1280, H = 720;
        const double cxL = 0.22 * W, cy = 0.55 * H, r = 0.13 * H;
        const double ar = 1.6, ang = 25.0;
        cv::Mat scene = makeSyntheticScene(W, H, 0.22, 0.74, 0.55, 0.13, ar, ang);
        DetectResult res = det.detect(scene);
        std::printf("        检出 %zu 个\n", res.circles.size());
        check(res.circles.size() == 2, "数量 == 2");
        if (res.circles.size() == 2) {
            const auto& c = res.circles[0];
            std::printf("        a=%.1f b=%.1f a/b=%.2f angle=%.1f r_eq=%.1f "
                        "(真值 a/b=%.2f angle=%.0f r_eq=%.1f)\n",
                        c.a, c.b, c.axis_ratio, c.angle, c.r, ar, ang, r);
            check(std::fabs(c.axis_ratio - ar) < 0.22, "轴比误差 < 0.22");
            check(std::fabs(c.r - r) < 0.15 * r, "等效半径准确（面积没跑偏）");
            check(std::hypot(c.cx - cxL, c.cy - cy) < 0.12 * r, "圆心准确");
            check(c.is_ellipse, "正确标记为椭圆");
        }
        if (dump_next_trace) { trace(res); dump_next_trace = false; }
    }

    std::puts("[self-test] 6) 盘中央压一个深色圆盖（约占盘面 45%）-> 填内部空洞后仍应检出");
    {
        const int W = 1280, H = 720;
        const double cxL = 0.20 * W, cy = 0.62 * H, r = 0.115 * H;
        cv::Mat scene = makeSyntheticScene(W, H, 0.20, 0.72, 0.62, 0.115);
        // 完全落在左盘内部的深色圆盖（模拟骰盅/深色物件压住盘心）
        cv::circle(scene, cv::Point((int)cxL, (int)cy), (int)(0.67 * r),
                   cv::Scalar(20, 20, 20), -1);
        DetectResult res = det.detect(scene);
        std::printf("        检出 %zu 个\n", res.circles.size());
        check(res.circles.size() == 2, "数量 == 2（圆盖被当成盘内空洞填掉）");
        if (!res.circles.empty()) {
            const double e = std::hypot(res.circles[0].cx - cxL, res.circles[0].cy - cy);
            std::printf("        左盘圆心误差 = %.2f px, r=%.1f（真值 %.1f）\n", e, res.circles[0].r, r);
            check(e < 0.12 * r && std::fabs(res.circles[0].r - r) < 0.12 * r, "圆心/半径准确");
            if (dump_next_trace) { trace(res); dump_next_trace = false; }
        }
    }

    std::printf("[self-test] %s（失败 %d 项）\n", fails == 0 ? "全部通过" : "存在失败", fails);
    return fails == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    ::signal(SIGPIPE, SIG_IGN);   // 预览客户端断开时别把进程干掉
    ::signal(SIGINT, onSignal);   // Ctrl-C  -> 优雅收尾（写完视频索引）
    ::signal(SIGTERM, onSignal);  // pkill   -> 同上
    if (argc <= 1) { usage(); return 2; }

    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&](const char*& out) -> bool {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "[err] 参数 %s 缺少取值\n", argv[i]);
                return false;
            }
            out = argv[++i];
            return true;
        };
        // --preview 的端口可选：后面跟的是纯数字才吃掉
        auto optionalInt = [&](int dflt) -> int {
            if (i + 1 < argc) {
                const char* s = argv[i + 1];
                bool digits = (*s != '\0');
                for (const char* p = s; *p; ++p) if (*p < '0' || *p > '9') { digits = false; break; }
                if (digits) { ++i; return std::atoi(s); }
            }
            return dflt;
        };
        const char* v = nullptr;
        if (k == "-h" || k == "--help") { usage(); return 0; }
        else if (k == "--self-test") a.self_test = true;
        else if (k == "--show") a.show = true;
        else if (k == "--quiet") a.quiet = true;
        else if (k == "--verbose") a.verbose = true;
        else if (k == "--summary") a.summary = true;
        else if (k == "--no-ring-check") a.no_ring_check = true;
        else if (k == "--no-ellipse") a.no_ellipse = true;
        else if (k == "--no-fill-holes") a.no_fill_holes = true;
        else if (k == "--max-axis-ratio") { if (!val(v)) return 2; a.max_axis_ratio = std::atof(v); }
        else if (k == "--no-hud") a.no_hud = true;
        else if (k == "--no-mask-inset") a.no_mask_inset = true;
        else if (k == "--loop") a.loop = true;
        else if (k == "--preview") a.preview_port = optionalInt(8099);
        else if (k == "--preview-bind") { if (!val(v)) return 2; a.preview_bind = v; }
        else if (k == "--preview-width") { if (!val(v)) return 2; a.preview_width = std::atoi(v); }
        else if (k == "--jpeg-quality") { if (!val(v)) return 2; a.jpeg_quality = std::atoi(v); }
        else if (k == "--threads") { if (!val(v)) return 2; a.threads = std::atoi(v); }
        else if (k == "--image")    { if (!val(v)) return 2; a.image = v; }
        else if (k == "--video")    { if (!val(v)) return 2; a.video = v; }
        else if (k == "--device")   { if (!val(v)) return 2; a.device = v; }
        else if (k == "--backend")  { if (!val(v)) return 2; a.backend = v; }
        else if (k == "--debug-dir"){ if (!val(v)) return 2; a.debug_dir = v; }
        else if (k == "--out-json") { if (!val(v)) return 2; a.out_json = v; }
        else if (k == "--save-video") { if (!val(v)) return 2; a.save_video = v; }
        else if (k == "--method")   { if (!val(v)) return 2; a.method = v; }
        else if (k == "--width")    { if (!val(v)) return 2; a.width = std::atoi(v); }
        else if (k == "--height")   { if (!val(v)) return 2; a.height = std::atoi(v); }
        else if (k == "--fps")      { if (!val(v)) return 2; a.fps = std::atoi(v); }
        else if (k == "--frames")   { if (!val(v)) return 2; a.frames = std::atoi(v); }
        else if (k == "--zoom")     { if (!val(v)) return 2; a.zoom = std::atoi(v); }
        else if (k == "--focus")    { if (!val(v)) return 2; a.focus = std::atoi(v); }
        else if (k == "--expected") { if (!val(v)) return 2; a.expected = std::atoi(v); }
        else if (k == "--max-circles") { if (!val(v)) return 2; a.max_circles = std::atoi(v); }
        else if (k == "--work-width")  { if (!val(v)) return 2; a.work_width = std::atoi(v); }
        else if (k == "--sat-max")  { if (!val(v)) return 2; a.sat_max = std::atoi(v); }
        else if (k == "--val-min")  { if (!val(v)) return 2; a.val_min = std::atoi(v); }
        else if (k == "--close-ksize") { if (!val(v)) return 2; a.close_ksize = std::atoi(v); }
        else if (k == "--open-ksize")  { if (!val(v)) return 2; a.open_ksize = std::atoi(v); }
        else if (k == "--min-radius-frac") { if (!val(v)) return 2; a.min_radius_frac = std::atof(v); }
        else if (k == "--max-radius-frac") { if (!val(v)) return 2; a.max_radius_frac = std::atof(v); }
        else if (k == "--min-circularity") { if (!val(v)) return 2; a.min_circularity = std::atof(v); }
        else if (k == "--min-fill-ratio")  { if (!val(v)) return 2; a.min_fill_ratio = std::atof(v); }
        else if (k == "--min-inlier-ratio"){ if (!val(v)) return 2; a.min_inlier_ratio = std::atof(v); }
        else if (k == "--ring-margin") { if (!val(v)) return 2; a.ring_dark_margin = std::atof(v); }
        else if (k == "--ring-ratio")  { if (!val(v)) return 2; a.ring_min_ratio = std::atof(v); }
        else if (k == "--hough-dp")    { if (!val(v)) return 2; a.hough_dp = std::atof(v); }
        else if (k == "--hough-param2"){ if (!val(v)) return 2; a.hough_param2 = std::atof(v); }
        else if (k == "--smooth")      { if (!val(v)) return 2; a.smooth_alpha = std::atof(v); }
        else {
            std::fprintf(stderr, "[err] 未知参数: %s（-h 看帮助）\n", k.c_str());
            return 2;
        }
    }

    CircleParams p;
    p.work_width        = a.work_width;
    p.sat_max           = a.sat_max;
    p.val_min           = a.val_min;
    p.close_ksize       = a.close_ksize;
    p.open_ksize        = a.open_ksize;
    p.min_radius_frac   = a.min_radius_frac;
    p.max_radius_frac   = a.max_radius_frac;
    p.min_circularity   = a.min_circularity;
    p.min_fill_ratio    = a.min_fill_ratio;
    p.min_inlier_ratio  = a.min_inlier_ratio;
    p.ring_dark_margin  = a.ring_dark_margin;
    p.ring_min_ratio    = a.ring_min_ratio;
    p.require_ring      = !a.no_ring_check;
    p.ellipse_fit       = !a.no_ellipse;
    p.fill_holes        = !a.no_fill_holes;
    p.max_axis_ratio    = a.max_axis_ratio;
    p.hough_dp          = a.hough_dp;
    p.hough_param1      = a.hough_param1;
    p.hough_param2      = a.hough_param2;
    p.hough_min_dist_frac = a.hough_min_dist_frac;
    p.method            = a.method;
    p.expected          = a.expected;
    p.max_circles       = a.max_circles;
    p.trace             = a.verbose;

    if (a.self_test) return runSelfTest(p);

    // 线程数：实测把识别钉在 1 个核上，CPU/帧从 62.8ms 降到 47.9ms 而帧率不变
    // （多出来的那部分是跨核同步的 sys 时间），所以给个显式开关。
    if (a.threads > 0) {
        cv::setNumThreads(a.threads);
        std::fprintf(stderr, "[cpu] 识别线程数 = %d (cv::setNumThreads)\n", a.threads);
    }

    if (a.image.empty() && a.video.empty() && a.device.empty()) {
        std::fprintf(stderr, "[err] 必须指定 --image / --video / --device 之一（-h 看帮助）\n");
        return 2;
    }

    // 需要画叠加图吗？—— 预览/窗口/存视频/存调试图，任一开启就要
    const bool need_overlay = a.show || a.preview_port > 0 ||
                              !a.debug_dir.empty() || !a.save_video.empty();
    // 其中这些用途每帧都必须画；纯预览可以按"有没有人看"动态跳过
    const bool always_overlay = a.show || !a.debug_dir.empty() || !a.save_video.empty();
    // 掩码缩略图要用到 res.mask，所以只要会画预览就保留它
    p.keep_debug = need_overlay;

    if (!a.debug_dir.empty()) {
        std::error_code ec;
        fs::create_directories(a.debug_dir, ec);
    }

    std::ofstream json_out;
    if (!a.out_json.empty()) {
        json_out.open(a.out_json, std::ios::out | std::ios::trunc);
        if (!json_out) std::fprintf(stderr, "[warn] 无法写 %s\n", a.out_json.c_str());
    }
    auto emit = [&](const std::string& line) {
        if (!a.quiet) { std::fwrite(line.data(), 1, line.size(), stdout); std::fputc('\n', stdout); }
        if (json_out) { json_out << line << '\n'; json_out.flush(); }
    };

    // ---- MJPEG 预览服务 ----
    MjpegServer preview;
    if (a.preview_port > 0) {
        std::string err;
        if (!preview.start(a.preview_bind, a.preview_port, err)) {
            std::fprintf(stderr, "[err] 预览服务启动失败: %s\n", err.c_str());
            return 5;
        }
        std::fprintf(stderr, "[preview] MJPEG 预览已启动，浏览器打开：\n");
        std::fprintf(stderr, "          http://127.0.0.1:%d/\n", a.preview_port);
        for (const auto& ip : localIPv4())
            std::fprintf(stderr, "          http://%s:%d/\n", ip.c_str(), a.preview_port);
        std::fprintf(stderr, "          纯流地址（VLC/ffplay）：http://<ip>:%d/stream.mjpg\n",
                     a.preview_port);
    }

    // ---- 组装任务列表：图片目录 -> 多张图；否则单一来源 ----
    std::vector<std::string> images;
    bool dir_mode = false;
    {
        std::error_code ec;
        dir_mode = !a.image.empty() && fs::is_directory(a.image, ec);
    }
    if (dir_mode) {
        images = listImages(a.image);
        if (images.empty()) {
            std::fprintf(stderr, "[err] 目录里没有图片: %s\n", a.image.c_str());
            preview.stop();
            return 2;
        }
    } else if (!a.image.empty()) {
        images.push_back(a.image);
    }

    CircleDetector det(p);
    Smoother sm; sm.alpha = a.smooth_alpha;
    FpsMeter fps_meter;

    cv::VideoWriter writer;
    std::string src_desc = dir_mode ? ("dir:" + a.image) : a.image;

    long   total_frames = 0, frames_with = 0, total_circles = 0;
    double total_ms = 0;

    auto process = [&](const cv::Mat& bgr, const std::string& name, long idx) -> bool {
        if (g_stop.load()) return false;   // Ctrl-C：交回主循环做收尾
        DetectResult res = det.detect(bgr);
        sm.apply(res.circles);
        fps_meter.tick();

        ++total_frames;
        total_ms += res.latency_ms;
        total_circles += (long)res.circles.size();
        if (!res.circles.empty()) ++frames_with;

        if (a.verbose) {
            for (const auto& t : res.trace) std::fprintf(stderr, "  [trace] %s\n", t.c_str());
            for (const auto& c : res.circles)
                std::fprintf(stderr,
                             "  [cand] %s cx=%.1f cy=%.1f r=%.1f score=%.3f circ=%.3f "
                             "fill=%.3f inlier=%.3f ring=%.3f contrast=%.1f\n",
                             c.method.c_str(), c.cx, c.cy, c.r, c.score, c.circularity,
                             c.fill_ratio, c.inlier_ratio, c.ring_ratio, c.contrast);
        }

        // ---- 叠加图（预览/窗口/存视频/调试图共用）----
        // ★ 只在真需要时才画：存盘类需求每帧都要；纯预览且没人看时直接跳过，
        //   否则白白花 ~10ms/帧去 clone 720p 再画圈（实测就是这个数）。
        cv::Mat overlay;
        const bool need_now = always_overlay || (preview.running() && preview.wantsFrame());
        if (need_now) {
            OverlayOptions oo;
            oo.hud        = !a.no_hud;
            oo.mask_inset = !a.no_mask_inset;
            oo.fps        = fps_meter.value();
            oo.frame_index = idx;
            oo.source     = name;
            drawOverlay(bgr, res, overlay, oo);
        }

        // ---- stdout / 文件 JSON ----
        std::string line = "{\"type\":\"frame\",\"index\":" + std::to_string(idx) +
                           ",\"name\":\"" + jsonEscape(name) + "\"" +
                           ",\"width\":" + std::to_string(bgr.cols) +
                           ",\"height\":" + std::to_string(bgr.rows) +
                           ",\"method\":\"" + res.method_used + "\"" +
                           ",\"latency_ms\":" + std::to_string(res.latency_ms) +
                           ",\"count\":" + std::to_string(res.circles.size()) +
                           ",\"circles\":[";
        for (size_t i = 0; i < res.circles.size(); ++i) {
            if (i) line += ",";
            line += frameJson(res.circles[i], i);
        }
        line += "]}";
        emit(line);

        // ---- MJPEG 预览 ----
        if (preview.running()) {
            if (!overlay.empty()) {
                if (a.preview_width > 0 && overlay.cols > a.preview_width) {
                    cv::Mat small;
                    cv::resize(overlay, small, cv::Size(a.preview_width,
                               cvRound(overlay.rows * (double)a.preview_width / overlay.cols)),
                               0, 0, cv::INTER_AREA);
                    preview.publish(small, bgr, a.jpeg_quality);
                } else {
                    preview.publish(overlay, bgr, a.jpeg_quality);
                }
            }
            // /status：给上层程序 + 预览页上的实时统计用
            std::string st = "{\"fps\":" + std::to_string(fps_meter.value()) +
                             ",\"detect_ms\":" + std::to_string(res.latency_ms) +
                             ",\"count\":" + std::to_string(res.circles.size()) +
                             ",\"method\":\"" + res.method_used + "\"" +
                             ",\"frames\":" + std::to_string(total_frames) +
                             ",\"wall_ms\":" + std::to_string(total_ms) +
                             ",\"clients\":" + std::to_string(preview.clients()) +
                             ",\"circles\":[";
            for (size_t i = 0; i < res.circles.size(); ++i) {
                if (i) st += ",";
                const auto& c = res.circles[i];
                char cb[192];
                std::snprintf(cb, sizeof(cb),
                              "{\"side\":\"%s\",\"cx\":%.2f,\"cy\":%.2f,\"r\":%.2f,\"score\":%.3f}",
                              c.side == 0 ? "LEFT" : (c.side == 1 ? "RIGHT" : "UNKNOWN"),
                              c.cx, c.cy, c.r, c.score);
                st += cb;
            }
            st += "]}";
            preview.setStatusJson(std::move(st));
        }

        // ---- 存调试图 ----
        if (!a.debug_dir.empty() && !overlay.empty()) {
            char fn[512];
            std::snprintf(fn, sizeof(fn), "%s/overlay_%04ld.jpg", a.debug_dir.c_str(), idx);
            cv::imwrite(fn, overlay);
            if (!res.mask.empty()) {
                std::snprintf(fn, sizeof(fn), "%s/mask_%04ld.png", a.debug_dir.c_str(), idx);
                cv::imwrite(fn, res.mask);
            }
        }

        // ---- 存视频 ----
        if (!a.save_video.empty() && !overlay.empty()) {
            if (!writer.isOpened()) {
                const std::string ext = fs::path(a.save_video).extension().string();
                const double wfps = (a.fps > 0) ? a.fps : 25.0;
                const int fourcc = (ext == ".mp4")
                                       ? cv::VideoWriter::fourcc('m', 'p', '4', 'v')
                                       : cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
                writer.open(a.save_video, fourcc, wfps, overlay.size());
                if (!writer.isOpened() && ext == ".mp4") {   // mp4v 不可用就退 MJPEG/avi
                    std::fprintf(stderr, "[warn] mp4v 编码器不可用，改用 MJPG\n");
                    writer.open(a.save_video, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                wfps, overlay.size());
                }
                if (!writer.isOpened()) {
                    std::fprintf(stderr, "[err] 无法打开视频写入: %s\n", a.save_video.c_str());
                } else {
                    std::fprintf(stderr, "[video] 录制到 %s (%.0f fps)\n",
                                 a.save_video.c_str(), wfps);
                }
            }
            if (writer.isOpened()) writer.write(overlay);
        }

        // ---- OpenCV 窗口 ----
        if (a.show) {
            cv::Mat view = overlay.empty() ? bgr : overlay;
            try {
                cv::imshow("circle_detect", view);
                const int key = cv::waitKey(1) & 0xFF;
                if (key == 27 || key == 'q') return false;
            } catch (const cv::Exception& e) {
                std::fprintf(stderr, "[warn] 无法开窗口（%s）\n"
                                     "        板子走 SSH 时没有 DISPLAY，请改用 --preview\n",
                             e.what());
                a.show = false;
            }
        }
        return true;
    };

    int rc = 0;
    const double t_start = nowSec();

    if (dir_mode) {
        // --loop 时常驻预览，把解码后的图缓起来，别每圈重解码 JPEG
        std::vector<cv::Mat> cache;
        if (a.loop) {
            if (images.size() > 300)
                std::fprintf(stderr, "[warn] --loop 下图片较多(%zu)，内存占用会偏高\n", images.size());
            cache.reserve(images.size());
            for (const auto& f : images) {
                cv::Mat img = cv::imread(f, cv::IMREAD_COLOR);
                if (img.empty()) { std::fprintf(stderr, "[warn] 跳过无法读取的图片: %s\n", f.c_str()); continue; }
                cache.push_back(img);
            }
            if (cache.empty()) { std::fprintf(stderr, "[err] 没有可用图片\n"); preview.stop(); return 2; }
        }

        long idx = 0;
        bool stop = false;
        while (!stop) {
            for (size_t i = 0; i < images.size(); ++i) {
                if (g_stop.load() || (a.frames > 0 && idx >= a.frames)) { stop = true; break; }
                cv::Mat img = a.loop ? cache[i < cache.size() ? i : 0] : cv::imread(images[i], cv::IMREAD_COLOR);
                if (img.empty()) { std::fprintf(stderr, "[warn] 跳过无法读取的图片: %s\n", images[i].c_str()); continue; }
                if (!process(img, fs::path(images[i]).filename().string(), idx)) { stop = true; break; }
                ++idx;
            }
            if (stop || !a.loop) break;
        }
    } else {
        SourceOptions so;
        so.image = a.image;
        so.video = a.video;
        so.device = a.device;
        so.width = a.width; so.height = a.height; so.fps = a.fps;
        so.backend = a.backend;
        so.zoom = a.zoom; so.focus = a.focus;
        so.verbose = a.verbose;

        std::string err;
        FrameSource src;
        std::fprintf(stderr, "[src] 打开 %s ...\n",
                     !so.image.empty() ? so.image.c_str()
                                       : (!so.video.empty() ? so.video.c_str() : so.device.c_str()));
        if (!src.open(so, err)) {
            std::fprintf(stderr, "[err] %s\n", err.c_str());
            preview.stop();
            return 3;
        }
        std::fprintf(stderr, "[src] %s\n", src.describe().c_str());
        src_desc = src.describe();

        const bool unlimited = (a.frames <= 0);
        long idx = 0;
        cv::Mat frame;
        bool stop = false;
        while (!stop) {
            while (true) {
                if (g_stop.load() || (!unlimited && idx >= a.frames)) { stop = true; break; }
                if (!src.read(frame)) break;
                if (!process(frame, src.describe(), idx)) { stop = true; break; }
                ++idx;
                if (!src.isStream()) break;   // 单张图片
            }
            if (stop || !a.loop) break;
            if (!src.rewind()) break;         // 摄像头不能回卷，直接结束
            if (a.verbose) std::fprintf(stderr, "[src] --loop：回到开头\n");
        }
        rc = (idx > 0) ? 0 : 4;
    }

    if (writer.isOpened()) writer.release();
    preview.stop();

    const double wall_ms = (nowSec() - t_start) * 1000.0;

    if (a.summary) {
        char b[512];
        std::snprintf(b, sizeof(b),
                      "{\"type\":\"summary\",\"frames\":%ld,\"frames_with_circles\":%ld,"
                      "\"total_circles\":%ld,\"avg_circles\":%.3f,\"avg_latency_ms\":%.2f,"
                      "\"detect_hit_rate\":%.3f,\"wall_ms\":%.1f}",
                      total_frames, frames_with, total_circles,
                      total_frames ? double(total_circles) / total_frames : 0.0,
                      total_frames ? total_ms / total_frames : 0.0,
                      total_frames ? double(frames_with) / total_frames : 0.0, wall_ms);
        std::puts(b);   // --quiet 只压逐帧输出，汇总照常打印
        if (json_out) json_out << b << '\n';
    }
    return rc;
}
