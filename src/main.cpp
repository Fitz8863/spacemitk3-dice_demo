// circle_detect —— K3 摄像头圆圈识别（纯传统图像算法，无模型）
//
// 用法示例：
//   ./build/circle_detect --image frame.jpg --debug-dir out/      # 单图
//   ./build/circle_detect --image data/ --summary                 # 整个目录批量
//   ./build/circle_detect --device /dev/video1 --show             # 板子桌面窗口
//   ./build/circle_detect --device 1 --zoom 160 --save-video out.avi
//   ./build/circle_detect --self-test                             # 合成图自检
//
// 输出：stdout 每帧一行 JSON（JSON Lines），便于被上层脚本/服务消费。
//
// 可视化有四种出口，可以任意组合：
//   --show              OpenCV 窗口（需要板子本地有 DISPLAY/Wayland）
//   --debug-dir DIR     overlay_XXXX.jpg + mask_XXXX.png 落盘
//   --save-video FILE   带标注的视频文件

#include "circle_detector.h"
#include "config.h"
#include "frame_source.h"
#include "latest_queue.h"
#include "rtsp_streamer.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
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
#include <cerrno>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace yuan;

namespace {

// Ctrl-C / SIGTERM 时优雅退出：否则 --save-video 的 AVI 索引写不完，文件放不出来
std::atomic<bool> g_stop{false};
void onSignal(int) { g_stop.store(true); }

// ---------------------------------------------------------------------------
// 保护 stdout
//
// spacemith264enc 背后的 SpaceMIT VPU 库会直接往 **stdout** 打 `[MPP-DEBUG] ...`
// （实测一次编码器初始化 34 行），足以把本程序的 JSON Lines 冲烂。库是闭源的、
// 也没找到日志级别开关，所以只能在 fd 层面隔离：
//   1) dup(STDOUT_FILENO) 留一份"真正的 stdout"专门给 JSON 用
//   2) dup2(STDERR_FILENO, STDOUT_FILENO) 把 fd 1 指到 stderr
// 之后第三方库往 fd 1 写的东西全落到 stderr，JSON 仍写进原来的管道/文件。
// 对调用方透明：`circle_detect --quiet > out.jsonl` 照常工作。
// ---------------------------------------------------------------------------
int g_json_fd = -1;

void protectStdout() {
    g_json_fd = ::dup(STDOUT_FILENO);
    if (g_json_fd >= 0) ::dup2(STDERR_FILENO, STDOUT_FILENO);
}

void writeJsonLine(const std::string& line) {
    std::string s = line;
    s += '\n';
    if (g_json_fd < 0) {           // 没保护过（如 --self-test、--dump-config）
        std::fwrite(s.data(), 1, s.size(), stdout);
        return;
    }
    size_t off = 0;
    while (off < s.size()) {
        const ssize_t w = ::write(g_json_fd, s.data() + off, s.size() - off);
        if (w < 0) { if (errno == EINTR) continue; break; }
        if (w == 0) break;
        off += static_cast<size_t>(w);
    }
}

// 只在命令行出现、不写进 config.json 的开关
struct CliOnly {
    bool self_test = false;
    bool dump_config = false;
    bool no_ring_check = false;
    bool no_ellipse = false;
    bool no_fill_holes = false;
    bool no_hud = false;
    bool no_mask_inset = false;
    bool no_rtsp = false;
    std::string config_path = "config.json";
};

void usage() {
    std::puts(
        "circle_detect —— K3 摄像头圆圈识别（传统 CV，无模型）\n"
        "\n"
        "参数来源：默认读当前目录的 config.json，命令行参数可临时覆盖它。\n"
        "  优先级：命令行  >  config.json  >  代码内默认值\n"
        "\n"
        "配置:\n"
        "  --config PATH       指定配置文件（默认 config.json）\n"
        "  --dump-config       打印一份默认配置文本，可重定向成 config.json\n"
        "\n"
        "输入（config.json 同名键）：\n"
        "  --image PATH        单张图片，或图片目录（批量）\n"
        "  --video PATH        视频文件\n"
        "  --device DEV        摄像头，/dev/video1 或索引 1\n"
        "  --width/--height/--fps    采集参数\n"
        "  --backend auto|v4l2|gst   采集后端\n"
        "  --zoom N / --focus N      设 v4l2 控制（<0 = 不动）\n"
        "  --frames N          处理多少帧后停止（0 = 不限/全部）\n"
        "  --loop              图片/视频循环播放\n"
        "\n"
        "推流:\n"
        "  --rtsp              开 RTSP 推流（H.264 → MediaMTX；config 里 rtsp 段配地址）\n"
        "  --no-rtsp           临时关掉 RTSP\n"
        "  --show              开 OpenCV 窗口（需板子本地有图形会话）\n"
        "  --debug-dir DIR     保存 overlay_XXXX.jpg / mask_XXXX.png / raw_XXXX.jpg\n"
        "                      （raw_ 是没画任何东西的原图，调算法时用这张）\n"
        "  --save-video FILE   保存带标注的视频（.avi / .mp4）\n"
        "  --no-hud / --no-mask-inset   画面上不画状态条 / 掩码缩略图\n"
        "\n"
        "算法（config.json 同名键）:\n"
        "  --method auto|mask|hough   检测路径\n"
        "  --work-width N      工作分辨率宽度（0=原图，越小越快）\n"
        "  --sat-max N / --val-min N  白垫 HSV 阈值\n"
        "  --mask-space hsv|min       白垫判别方式（min=用三通道最小值，免疫白平衡漂移）\n"
        "  --min-channel-thr N        mask-space=min 时的下限（默认 140）\n"
        "  --close-ksize N / --open-ksize N   形态学核\n"
        "  --min-radius-frac F / --max-radius-frac F\n"
        "  --min-circularity F / --min-fill-ratio F / --min-inlier-ratio F\n"
        "  --ring-val-max N / --ring-ratio F   环带暗判据 V 阈值（默认 130）/ 覆盖比例下限\n"
        "  --hough-cooldown N  Hough 兜底跑一次后冷却 N 帧（默认 5，0=不降频）\n"
        "  --max-axis-ratio F  长短轴比上限（斜视）\n"
        "  --expected N        期望圆个数，不够时触发 Hough 兜底\n"
        "  --max-circles N     最多输出几个\n"
        "  --smooth F          时序平滑系数 0..1\n"
        "  --threads N         识别线程数（实测 1 最省 CPU，帧率不变）\n"
        "  --no-ring-check / --no-ellipse / --no-fill-holes   对比实验用\n"
        "\n"
        "输出:\n"
        "  --out-json FILE     逐帧 JSON 同时写入文件\n"
        "  --summary / --quiet / --verbose\n"
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
    // ★ 每个用例用一个全新的检测器：6 个用例是互不相关的独立场景，
    //   共享实例会让 Hough 降频的跨帧状态串味（实测曾导致用例 4 少检出一个盘）。
    //   "跨帧降频行为"由用例 7 单独验证。

    std::puts("[self-test] 1) 合成红蓝地垫 + 两个白垫深环 -> 应检出 2 个");
    {
        const int W = 1280, H = 720;
        const double cxL = 0.20 * W, cxR = 0.72 * W, cy = 0.62 * H, r = 0.115 * H;
        cv::Mat scene = makeSyntheticScene(W, H, 0.20, 0.72, 0.62, 0.115);
        CircleDetector det(p);
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
        CircleDetector det(p);
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
        CircleDetector det(p);
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
        CircleDetector det(p);
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
        if (dump_next_trace) { trace(res); dump_next_trace = false; }
    }

    std::puts("[self-test] 5) 斜视：白垫画成 1.6:1 的椭圆 + 一侧倾斜 25 度 -> 应检出椭圆");
    {
        const int W = 1280, H = 720;
        const double cxL = 0.22 * W, cy = 0.55 * H, r = 0.13 * H;
        const double ar = 1.6, ang = 25.0;
        cv::Mat scene = makeSyntheticScene(W, H, 0.22, 0.74, 0.55, 0.13, ar, ang);
        CircleDetector det(p);
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
        CircleDetector det(p);
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

    std::puts("[self-test] 7) Hough 降频：有用时不压召回，无用时才退让");
    {
        // 场景：两个正常盘 -> mask 直接达标，永不触发 Hough
        //       再人为把 expected 抬到 3 -> mask 永远不够 -> 每帧都试 Hough
        //       Hough 补不上 -> 连续无用后应进入冷却，耗时下降
        const int W = 1280, H = 720;
        cv::Mat scene = makeSyntheticScene(W, H, 0.20, 0.72, 0.62, 0.115);

        CircleParams p2 = p;
        p2.expected = 3;              // 逼出 Hough 兜底
        p2.hough_cooldown_frames = 5;
        p2.hough_useless_limit = 2;
        p2.trace = false;
        CircleDetector det2(p2);

        std::vector<double> lat;
        int used = 0;
        for (int i = 0; i < 10; ++i) {
            DetectResult r = det2.detect(scene);
            lat.push_back(r.latency_ms);
            if (det2.lastFrameUsedHough()) ++used;
        }
        double first2 = (lat[0] + lat[1]) / 2.0, last5 = 0;
        for (int i = 5; i < 10; ++i) last5 += lat[i];
        last5 /= 5.0;
        std::printf("        10 帧里跑了 %d 次 Hough；前 2 帧均 %.1fms，后 5 帧均 %.1fms\n",
                    used, first2, last5);
        check(used < 10, "无用后确实降频（不是每帧都跑）");
        check(used >= 3, "仍在周期性重试（没有彻底放弃兜底）");
        check(last5 < first2, "降频后耗时下降");
    }

    std::puts("[self-test] 8) 白盘直接放在地垫上（无深色环）-> 环带暗度判据应拒绝");
    {
        // 回归用例（2026-09-18）：环带判据收紧前，白盘 + 一圈亮色地垫
        // （地垫 V~208、白盘 V~240，contrast 甚至 > 0）会被误认成盘。
        // 现在"环带必须大半是暗（V < ring_val_max）"应把这种候选拒掉。
        const int W = 1280, H = 720;
        const double r = 0.115 * H;
        cv::Mat mat(720, 1280, CV_8UC3);
        mat(cv::Rect(0, 0, 640, 720)).setTo(cv::Scalar(40, 40, 200));
        mat(cv::Rect(640, 0, 640, 720)).setTo(cv::Scalar(200, 90, 30));
        cv::circle(mat, cv::Point((int)(0.20 * W), (int)(0.62 * H)), (int)r,
                   cv::Scalar(240, 240, 240), -1);   // 没有外环，直接贴地垫
        CircleDetector det(p);
        DetectResult res = det.detect(mat);
        std::printf("        检出 %zu 个\n", res.circles.size());
        check(res.circles.empty(), "无深色环的白盘未被误认（ring not dark 拦截）");
        if (dump_next_trace) { trace(res); dump_next_trace = false; }
    }

    std::printf("[self-test] %s（失败 %d 项）\n", fails == 0 ? "全部通过" : "存在失败", fails);
    return fails == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    ::signal(SIGPIPE, SIG_IGN);   // 预览客户端断开时别把进程干掉
    ::signal(SIGINT, onSignal);   // Ctrl-C  -> 优雅收尾（写完视频索引）
    ::signal(SIGTERM, onSignal);  // pkill   -> 同上
    // ---- 第一遍：先把 --config / -h 挑出来，其余参数留到配置加载后再覆盖 ----
    std::string config_path;
    bool config_explicit = false;
    bool want_help = false;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--config" && i + 1 < argc) { config_path = argv[++i]; config_explicit = true; }
        else if (k == "-h" || k == "--help") want_help = true;
    }
    if (want_help) { usage(); return 0; }
    if (config_path.empty()) config_path = "config.json";

    // ---- 读 config.json。参数优先级：命令行 > config.json > 代码默认值 ----
    AppConfig a;
    {
        std::error_code ec;
        const bool exists = fs::exists(config_path, ec);
        if (!exists && !config_explicit) {
            std::fprintf(stderr,
                         "[cfg] 没找到 %s，本次用内置默认值运行。\n"
                         "      想要配置文件：./build/circle_detect --dump-config > %s\n",
                         config_path.c_str(), config_path.c_str());
        } else {
            std::string err;
            if (!load_config(config_path, a, err)) {
                std::fprintf(stderr, "[err] 读取配置失败: %s\n", err.c_str());
                return 6;
            }
            std::fprintf(stderr, "[cfg] 已加载 %s\n", config_path.c_str());
        }
    }

    CliOnly cli;
    cli.config_path = config_path;

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
        const char* v = nullptr;
        if (k == "-h" || k == "--help" || k == "--config") { if (k == "--config") ++i; }
        else if (k == "--dump-config") cli.dump_config = true;
        else if (k == "--self-test") cli.self_test = true;
        else if (k == "--show") a.show = true;
        else if (k == "--quiet") a.quiet = true;
        else if (k == "--verbose") a.verbose = true;
        else if (k == "--summary") a.summary = true;
        else if (k == "--no-ring-check") cli.no_ring_check = true;
        else if (k == "--no-ellipse") cli.no_ellipse = true;
        else if (k == "--no-fill-holes") cli.no_fill_holes = true;
        else if (k == "--max-axis-ratio") { if (!val(v)) return 2; a.max_axis_ratio = std::atof(v); }
        else if (k == "--no-hud") cli.no_hud = true;
        else if (k == "--no-mask-inset") cli.no_mask_inset = true;
        else if (k == "--loop") a.loop = true;
        else if (k == "--rtsp") a.rtsp_enabled = true;
        else if (k == "--no-rtsp") cli.no_rtsp = true;
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
        else if (k == "--frames")   { if (!val(v)) return 2; a.max_frames = std::atoi(v); }
        else if (k == "--zoom")     { if (!val(v)) return 2; a.zoom = std::atoi(v); }
        else if (k == "--focus")    { if (!val(v)) return 2; a.focus = std::atoi(v); }
        else if (k == "--expected") { if (!val(v)) return 2; a.expected = std::atoi(v); }
        else if (k == "--max-circles") { if (!val(v)) return 2; a.max_circles = std::atoi(v); }
        else if (k == "--work-width")  { if (!val(v)) return 2; a.work_width = std::atoi(v); }
        else if (k == "--sat-max")  { if (!val(v)) return 2; a.sat_max = std::atoi(v); }
        else if (k == "--mask-space"){ if (!val(v)) return 2; a.mask_space = v; }
        else if (k == "--min-channel-thr"){ if (!val(v)) return 2; a.min_channel_thr = std::atoi(v); }
        else if (k == "--val-min")  { if (!val(v)) return 2; a.val_min = std::atoi(v); }
        else if (k == "--close-ksize") { if (!val(v)) return 2; a.close_ksize = std::atoi(v); }
        else if (k == "--open-ksize")  { if (!val(v)) return 2; a.open_ksize = std::atoi(v); }
        else if (k == "--min-radius-frac") { if (!val(v)) return 2; a.min_radius_frac = std::atof(v); }
        else if (k == "--max-radius-frac") { if (!val(v)) return 2; a.max_radius_frac = std::atof(v); }
        else if (k == "--min-circularity") { if (!val(v)) return 2; a.min_circularity = std::atof(v); }
        else if (k == "--min-fill-ratio")  { if (!val(v)) return 2; a.min_fill_ratio = std::atof(v); }
        else if (k == "--min-inlier-ratio"){ if (!val(v)) return 2; a.min_inlier_ratio = std::atof(v); }
        else if (k == "--ring-val-max") { if (!val(v)) return 2; a.ring_val_max = std::atoi(v); }
        else if (k == "--ring-ratio")  { if (!val(v)) return 2; a.ring_min_ratio = std::atof(v); }
        else if (k == "--hough-dp")    { if (!val(v)) return 2; a.hough_dp = std::atof(v); }
        else if (k == "--hough-param2"){ if (!val(v)) return 2; a.hough_param2 = std::atof(v); }
        else if (k == "--hough-cooldown"){ if (!val(v)) return 2; a.hough_cooldown_frames = std::atoi(v); }
        else if (k == "--smooth")      { if (!val(v)) return 2; a.smooth_alpha = std::atof(v); }
        else {
            std::fprintf(stderr, "[err] 未知参数: %s（-h 看帮助）\n", k.c_str());
            return 2;
        }
    }

    // --no-xxx 是"临时关掉配置里打开的开关"，优先级最高
    if (cli.no_rtsp)      a.rtsp_enabled = false;
    if (cli.no_hud)       a.overlay_hud = false;
    if (cli.no_mask_inset) a.overlay_mask_inset = false;
    if (cli.no_ring_check) a.require_ring = false;
    if (cli.no_ellipse)    a.ellipse_fit = false;
    if (cli.no_fill_holes) a.fill_holes = false;

    if (cli.dump_config) {
        std::fputs(default_config_json().c_str(), stdout);
        return 0;
    }

    CircleParams p;
    p.work_width        = a.work_width;
    p.mask_space        = a.mask_space;
    p.min_channel_thr   = a.min_channel_thr;
    p.sat_max           = a.sat_max;
    p.val_min           = a.val_min;
    p.close_ksize       = a.close_ksize;
    p.open_ksize        = a.open_ksize;
    p.min_radius_frac   = a.min_radius_frac;
    p.max_radius_frac   = a.max_radius_frac;
    p.min_circularity   = a.min_circularity;
    p.min_fill_ratio    = a.min_fill_ratio;
    p.min_inlier_ratio  = a.min_inlier_ratio;
    p.ring_val_max      = a.ring_val_max;
    p.ring_min_ratio    = a.ring_min_ratio;
    p.require_ring      = a.require_ring;
    p.ellipse_fit       = a.ellipse_fit;
    p.fill_holes        = a.fill_holes;
    p.max_axis_ratio    = a.max_axis_ratio;
    p.report_ellipse_at = a.report_ellipse_at;
    p.rect_kernel       = a.rect_kernel;
    p.hough_dp          = a.hough_dp;
    p.hough_param1      = a.hough_param1;
    p.hough_param2      = a.hough_param2;
    p.hough_min_dist_frac = a.hough_min_dist_frac;
    p.hough_cooldown_frames = a.hough_cooldown_frames;
    p.hough_useless_limit   = a.hough_useless_limit;
    p.method            = a.method;
    p.expected          = a.expected;
    p.max_circles       = a.max_circles;
    p.trace             = a.verbose;

    if (cli.self_test) return runSelfTest(p);

    // 从这里往后第三方库（VPU）可能往 stdout 打日志，先隔离好
    protectStdout();

    // 线程数：实测把识别钉在 1 个核上，CPU/帧从 62.8ms 降到 47.9ms 而帧率不变
    // （多出来的那部分是跨核同步的 sys 时间），所以给个显式开关。
    if (a.threads > 0) {
        cv::setNumThreads(a.threads);
        std::fprintf(stderr, "[cpu] 识别线程数 = %d (cv::setNumThreads)\n", a.threads);
    }

    if (a.image.empty() && a.video.empty() && a.device.empty()) {
        std::fprintf(stderr, "[err] 必须配置 image / video / device 之一（见 %s）\n",
                     config_path.c_str());
        return 2;
    }

    // 谁需要叠加图？
    //   need_overlay_now —— 必须**在主循环里立刻画**的消费者（窗口/调试图/存视频）
    //   rtsp_gets_job    —— RTSP 改由编码线程绘制，主循环只交出原始帧+结果
    // 只开 RTSP 时（生产常态）主循环完全不画，省下 ~10ms/帧。
    const bool need_overlay_now = a.show || !a.debug_dir.empty() || !a.save_video.empty();
    const bool rtsp_gets_job    = a.rtsp_enabled;
    // 掩码缩略图要用到 res.mask，所以只要有消费者就保留
    p.keep_debug = need_overlay_now || rtsp_gets_job;

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
        if (!a.quiet) writeJsonLine(line);
        if (json_out) { json_out << line << '\n'; json_out.flush(); }
    };

    // ---- RTSP 推流（生产路径：H.264 → MediaMTX）----
    RtspStreamer rtsp;
    if (a.rtsp_enabled) {
        // 推流分辨率 = 采集分辨率（overlay 就是原图尺寸）
        if (!rtsp.start(a.rtsp_host, a.rtsp_port, a.rtsp_path, a.width, a.height, a.fps)) {
            std::fprintf(stderr, "[warn] RTSP 推流启动失败，继续跑识别（不影响检测）\n");
        } else {
            const std::string host = normalize_rtsp_host(a.rtsp_host);
            std::fprintf(stderr, "[rtsp] 拉流地址：\n");
            std::fprintf(stderr, "          RTSP   : %s\n", rtsp.url().c_str());
            std::fprintf(stderr, "          WebRTC : http://%s:8889%s/\n",
                         host.c_str(), a.rtsp_path.c_str());
            // 同一条流换个网卡地址也能拉，顺手列出来
            for (const auto& ip : localIPv4()) {
                if (ip == host) continue;
                std::fprintf(stderr,
                             "          其他网卡: rtsp://%s:%d%s   http://%s:8889%s/\n",
                             ip.c_str(), a.rtsp_port, a.rtsp_path.c_str(),
                             ip.c_str(), a.rtsp_path.c_str());
            }
        }
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
            rtsp.stop();
            return 2;
        }
    } else if (!a.image.empty()) {
        images.push_back(a.image);
    }

    Smoother sm; sm.alpha = a.smooth_alpha;
    FpsMeter fps_meter;

    cv::VideoWriter writer;
    std::string src_desc = dir_mode ? ("dir:" + a.image) : a.image;

    long   total_frames = 0, frames_with = 0, total_circles = 0;
    double total_ms = 0;
    double total_a_ms = 0, total_b_ms = 0;   // A/B 分段累计（看流水线均衡度）
    long   submitted = 0;
    // 采集线程耗时（跨线程，用原子累加）
    std::atomic<long long> cap_us_sum{0};
    std::atomic<long>      cap_n{0};

    // =====================================================================
    // 流水线（3 个线程，三个单槽 latest-only 队列）
    //
    //   采集线程：read            -> q_cap
    //   主线程  ：q_cap -> A(像素->候选) -> q_ab -> 消费结果并收尾
    //   B 线程  ：q_ab -> B(候选->结果) -> q_bm
    //
    // latest-only 的含义：生产者**替换**未被消费的旧任务而不是排队，
    // 所以任何一级变慢都只丢旧帧，不会把延迟传导到采集。
    // 详见 docs/pipeline-threading.md
    // =====================================================================
    CircleDetector a_det(p);          // A 无跨帧状态
    CircleDetector b_det(p);          // ★ B 独占：跨帧状态（Hough 冷却）只在这里读写
    LatestQueue<CandidateSet> q_ab;   // A -> B
    LatestQueue<DetectJob>    q_bm;   // B -> 主线程
    long dropped_by_b = 0;            // q_ab 推入时替换掉旧任务的次数

    std::thread b_thread([&] {
        while (true) {
            auto cs = q_ab.wait_pop_latest(std::chrono::milliseconds(100));
            if (!cs) {
                if (q_ab.closed_and_empty()) break;
                continue;
            }
            auto job = std::make_shared<DetectJob>();
            job->cs   = cs;
            job->a_ms = cs->a_ms;
            const double t0 = nowSec();
            job->res = b_det.fitCandidates(*cs);
            job->b_ms = (nowSec() - t0) * 1000.0;
            job->res.latency_ms = job->a_ms + job->b_ms;   // 总耗时在此定下，HUD/JSON 直接用
            q_bm.push(job);
        }
        q_bm.close();
    });
    std::fprintf(stderr, "[pipeline] 已启用：采集 / A+B / RTSP 三线程\n");

    // 提交一帧：A 阶段，然后交给 B 线程
    auto submit = [&](const cv::Mat& bgr, const std::string& name, long idx) {
        const double tA0 = nowSec();
        CandidateSet cs = a_det.extractCandidates(bgr, idx);
        cs.a_ms = (nowSec() - tA0) * 1000.0;
        cs.name = name;
        if (q_ab.push(std::make_shared<CandidateSet>(std::move(cs)))) ++dropped_by_b;
    };

    bool user_quit = false;   // 窗口里按了 q/ESC

    // 一帧的收尾：平滑、统计、JSON、推流、落盘、显示。
    // ★ 只在主线程调用（imshow / 信号 / 文件写入都有线程亲和性要求）。
    auto finishFrame = [&](DetectJob& job) {
        DetectResult& res = job.res;
        const cv::Mat& bgr = *job.cs->frame;
        const std::string& name = job.cs->name;
        const long idx = job.cs->index;

        sm.apply(res.circles);
        fps_meter.tick();

        ++total_frames;
        total_ms += res.latency_ms;
        total_a_ms += job.a_ms;
        total_b_ms += job.b_ms;
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

        // ---- 叠加图 ----
        // 主循环只在"窗口/调试图/存视频"需要时画；RTSP 那份由编码线程画。
        OverlayOptions oo;
        oo.hud        = a.overlay_hud;
        oo.mask_inset = a.overlay_mask_inset;
        oo.crosshair  = a.overlay_crosshair;
        oo.axes       = a.overlay_axes;
        oo.fps        = fps_meter.value();
        oo.frame_index = idx;
        oo.source     = name;

        cv::Mat overlay;
        if (need_overlay_now) drawOverlay(bgr, res, overlay, oo);

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

        // ---- 存调试图 ----
        if (!a.debug_dir.empty()) {
            char fn[512];
            // raw_：没画任何东西的原图。调算法（改阈值/看掩码对不对）时必须用这张，
            //       不能用画了圈和文字的 overlay_ —— 那会把边界信息污染掉。
            std::snprintf(fn, sizeof(fn), "%s/raw_%04ld.jpg", a.debug_dir.c_str(), idx);
            cv::imwrite(fn, bgr);
            if (!overlay.empty()) {
                std::snprintf(fn, sizeof(fn), "%s/overlay_%04ld.jpg", a.debug_dir.c_str(), idx);
                cv::imwrite(fn, overlay);
            }
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
                if (key == 27 || key == 'q') user_quit = true;
            } catch (const cv::Exception& e) {
                std::fprintf(stderr, "[warn] 无法开窗口（%s）：板子走 SSH 时没有 DISPLAY\n",
                             e.what());
                a.show = false;
            }
        }

        // ---- RTSP 推流 ----
        // 把未绘制的原始帧 + 检测结果交给编码线程，由它画标注再编 H.264。
        // 绘制因此离开识别主循环的关键路径（~10ms/帧），且检测结果已定型，
        // 换线程绘制对精度零影响。
        if (rtsp.running()) {
            OverlayJob job;
            job.frame = bgr;          // 浅拷贝头，深拷贝交给编码线程那次绘制
            job.det   = res;
            job.opt   = oo;
            rtsp.publish(std::move(job));
        }
    };

    // 非阻塞消费一个已完成的结果（供主循环调用）
    auto pump = [&]() -> bool {
        if (auto job = q_bm.try_pop_latest()) { finishFrame(*job); return true; }
        return false;
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
            if (cache.empty()) { std::fprintf(stderr, "[err] 没有可用图片\n"); rtsp.stop(); return 2; }
        }

        long idx = 0;
        bool stop = false;
        while (!stop) {
            for (size_t i = 0; i < images.size(); ++i) {
                if (g_stop.load() || user_quit || (a.max_frames > 0 && idx >= a.max_frames)) { stop = true; break; }
                pump();   // 先消化已完成的结果，避免 q_bm 里堆压
                cv::Mat img = a.loop ? cache[i < cache.size() ? i : 0] : cv::imread(images[i], cv::IMREAD_COLOR);
                if (img.empty()) { std::fprintf(stderr, "[warn] 跳过无法读取的图片: %s\n", images[i].c_str()); continue; }
                submit(img, fs::path(images[i]).filename().string(), idx);
                ++submitted; ++idx;
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
            rtsp.stop();
            return 3;
        }
        std::fprintf(stderr, "[src] %s\n", src.describe().c_str());
        src_desc = src.describe();

        const bool unlimited = (a.max_frames <= 0);
        long idx = 0;
        bool stop = false;

        // ---- 采集线程 -----------------------------------------------------
        // 为什么必须独立：实测把采集留在主循环时，MJPEG 解码会与 B 线程争抢，
        // 取帧从 18.5ms 涨到 35.3ms —— 主循环反而被采集拖住，流水线比单线程还慢。
        // 独立成线程后取帧不再受 A/B 干扰，且能与它们重叠。
        // （参考工程 yolo_segdetect 也是 capture/preprocess/inference 三线程。）
        LatestQueue<CaptureFrame> q_cap;
        std::atomic<bool> cap_stop{false};
        std::thread cap_thread([&] {
            long i = 0;
            while (!cap_stop.load()) {
                auto cf = std::make_shared<CaptureFrame>();
                // 每帧新建 Mat：read() 自己分配缓冲，这样交给下游的帧不会被
                // 下一帧覆盖（否则就是数据竞争）。零拷贝，只是不复用缓冲。
                const double tc0 = nowSec();
                if (!src.read(cf->frame)) break;
                cap_us_sum.fetch_add((long long)((nowSec() - tc0) * 1e6));
                cap_n.fetch_add(1);
                cf->index = i++;
                cf->name  = src.describe();
                q_cap.push(cf);
            }
            q_cap.close();
        });

        while (!stop) {
            while (true) {
                if (g_stop.load() || user_quit || (!unlimited && idx >= a.max_frames)) { stop = true; break; }
                auto cf = q_cap.wait_pop_latest(std::chrono::milliseconds(100));
                if (!cf) {
                    if (q_cap.closed_and_empty()) { stop = true; break; }
                    continue;
                }
                pump();   // 先消化已完成的结果
                submit(cf->frame, cf->name, cf->index);
                ++submitted; ++idx;
            }
            if (stop || !a.loop) break;
            if (!src.rewind()) break;         // 摄像头不能回卷，直接结束
            if (a.verbose) std::fprintf(stderr, "[src] --loop：回到开头\n");
        }

        cap_stop.store(true);
        q_cap.close();
        if (cap_thread.joinable()) cap_thread.join();
        rc = (idx > 0) ? 0 : 4;
    }

    // ---- 收尾顺序：主循环 -> B 线程 -> RTSP ----
    q_ab.close();                     // 告诉 B 没有新任务了
    if (b_thread.joinable()) b_thread.join();   // B 退出时会 q_bm.close()
    while (pump()) {}                 // 排空最后的结果（latest-only，最多 1 个）
    if (dropped_by_b > 0)
        std::fprintf(stderr, "[pipeline] B 端被替换掉的旧任务: %ld 个\n", dropped_by_b);
    if (rtsp.running())
        std::fprintf(stderr, "[rtsp] 编码线程绘制平均 %.1f ms\n", rtsp.drawMsAvg());

    if (writer.isOpened()) writer.release();
    rtsp.stop();

    const double wall_ms = (nowSec() - t_start) * 1000.0;

    if (a.summary) {
        char b[512];
        std::snprintf(b, sizeof(b),
                      "{\"type\":\"summary\",\"frames\":%ld,\"frames_with_circles\":%ld,"
                      "\"total_circles\":%ld,\"avg_circles\":%.3f,\"avg_latency_ms\":%.2f,"
                      "\"detect_hit_rate\":%.3f,\"wall_ms\":%.1f,"
                      "\"processed_fps\":%.2f,\"result_fps\":%.2f,"
                      "\"submitted\":%ld,\"dropped\":%ld,"
                      "\"avg_a_ms\":%.2f,\"avg_b_ms\":%.2f,"
                      "\"cap_ms\":%.2f}",
                      total_frames, frames_with, total_circles,
                      total_frames ? double(total_circles) / total_frames : 0.0,
                      total_frames ? total_ms / total_frames : 0.0,
                      total_frames ? double(frames_with) / total_frames : 0.0, wall_ms,
                      wall_ms > 0 ? submitted / wall_ms * 1000.0 : 0.0,
                      wall_ms > 0 ? total_frames / wall_ms * 1000.0 : 0.0,
                      submitted, dropped_by_b,
                      total_frames ? total_a_ms / total_frames : 0.0,
                      total_frames ? total_b_ms / total_frames : 0.0,
                      cap_n.load() ? (double)cap_us_sum.load() / cap_n.load() / 1000.0 : 0.0);
        writeJsonLine(b);   // --quiet 只压逐帧输出，汇总照常打印
        if (json_out) json_out << b << '\n';
    }
    return rc;
}
