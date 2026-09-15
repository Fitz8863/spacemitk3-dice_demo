#include "config.h"

#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

namespace yuan {
namespace {

// --- 类型化读取：字段缺失就用结构体里的默认值，类型不对就报错 -------------
// 刻意"严格"：写错类型会被拒绝并明确指出是哪个键，而不是静默用默认值 ——
// 静默回退是配置类 bug 最难查的一类。

bool get_str(const cv::FileNode& n, const char* k, std::string& v, std::string& err) {
    const cv::FileNode c = n[k];
    if (c.empty()) return true;
    if (!c.isString()) { err = std::string("config ") + k + " must be a string"; return false; }
    c >> v;
    return true;
}

bool get_int(const cv::FileNode& n, const char* k, int& v, std::string& err) {
    const cv::FileNode c = n[k];
    if (c.empty()) return true;
    if (!c.isInt() && !c.isReal()) { err = std::string("config ") + k + " must be a number"; return false; }
    c >> v;
    return true;
}

bool get_dbl(const cv::FileNode& n, const char* k, double& v, std::string& err) {
    const cv::FileNode c = n[k];
    if (c.empty()) return true;
    if (!c.isInt() && !c.isReal()) { err = std::string("config ") + k + " must be a number"; return false; }
    c >> v;
    return true;
}

bool get_bool(const cv::FileNode& n, const char* k, bool& v, std::string& err) {
    const cv::FileNode c = n[k];
    if (c.empty()) return true;
    if (c.isInt() || c.isReal()) { double d = 0; c >> d; v = (d != 0.0); return true; }
    if (c.isString()) {
        std::string s; c >> s;
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (s == "true" || s == "1" || s == "yes" || s == "on")  { v = true;  return true; }
        if (s == "false" || s == "0" || s == "no" || s == "off") { v = false; return true; }
    }
    err = std::string("config ") + k + " must be a boolean";
    return false;
}

bool get_map(const cv::FileNode& root, const char* key, cv::FileNode& out, std::string& err) {
    const cv::FileNode c = root[key];
    if (c.empty()) return true;
    if (!c.isMap()) { err = std::string("config ") + key + " must be an object"; return false; }
    out = c;
    return true;
}

}  // namespace

std::string normalize_rtsp_host(std::string host) {
    if (host.empty() || host == "0.0.0.0" || host == "*") return "127.0.0.1";
    return host;
}

std::string normalize_rtsp_path(std::string path) {
    if (path.empty()) return "/dice/circles";
    if (path.front() != '/') path.insert(path.begin(), '/');
    while (path.size() > 1 && path.back() == '/') path.pop_back();
    return path;
}

bool load_config(const std::string& path, AppConfig& c, std::string& error) {
    try {
        cv::FileStorage fs(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
        if (!fs.isOpened()) {
            error = "打不开配置文件: " + path;
            return false;
        }
        const cv::FileNode root = fs.root();
        if (!root.isMap()) {
            error = "config 根节点必须是一个 JSON 对象";
            return false;
        }
        c.config_path = path;

        // ---- 输入源 ----
        if (!get_str(root, "image", c.image, error) ||
            !get_str(root, "video", c.video, error) ||
            !get_str(root, "backend", c.backend, error) ||
            !get_int(root, "width", c.width, error) ||
            !get_int(root, "height", c.height, error) ||
            !get_int(root, "fps", c.fps, error) ||
            !get_int(root, "focus", c.focus, error) ||
            !get_int(root, "zoom", c.zoom, error) ||
            !get_int(root, "max_frames", c.max_frames, error) ||
            !get_bool(root, "loop", c.loop, error)) return false;

        // camera 既可以是设备路径字符串，也可以是数字索引（沿用 dice-game 惯例）
        const cv::FileNode cam = root["camera"];
        if (!cam.empty()) {
            if (cam.isString()) {
                cam >> c.device;
            } else if (cam.isInt() || cam.isReal()) {
                int idx = 0; cam >> idx;
                c.device = std::to_string(idx);
            } else {
                error = "config camera 必须是设备路径字符串或数字索引";
                return false;
            }
        }
        std::string dev;
        if (!get_str(root, "device", dev, error)) return false;
        if (!dev.empty()) c.device = dev;   // device 显式给了就优先

        // ---- 算法 ----
        if (!get_int(root, "work_width", c.work_width, error) ||
            !get_int(root, "threads", c.threads, error) ||
            !get_str(root, "method", c.method, error) ||
            !get_int(root, "expected", c.expected, error) ||
            !get_int(root, "max_circles", c.max_circles, error) ||
            !get_int(root, "sat_max", c.sat_max, error) ||
            !get_int(root, "val_min", c.val_min, error) ||
            !get_int(root, "close_ksize", c.close_ksize, error) ||
            !get_int(root, "open_ksize", c.open_ksize, error) ||
            !get_bool(root, "rect_kernel", c.rect_kernel, error) ||
            !get_bool(root, "fill_holes", c.fill_holes, error) ||
            !get_dbl(root, "min_radius_frac", c.min_radius_frac, error) ||
            !get_dbl(root, "max_radius_frac", c.max_radius_frac, error) ||
            !get_dbl(root, "min_circularity", c.min_circularity, error) ||
            !get_dbl(root, "min_fill_ratio", c.min_fill_ratio, error) ||
            !get_dbl(root, "min_inlier_ratio", c.min_inlier_ratio, error) ||
            !get_bool(root, "ellipse_fit", c.ellipse_fit, error) ||
            !get_dbl(root, "max_axis_ratio", c.max_axis_ratio, error) ||
            !get_dbl(root, "report_ellipse_at", c.report_ellipse_at, error) ||
            !get_bool(root, "require_ring", c.require_ring, error) ||
            !get_dbl(root, "ring_dark_margin", c.ring_dark_margin, error) ||
            !get_dbl(root, "ring_min_ratio", c.ring_min_ratio, error) ||
            !get_dbl(root, "hough_dp", c.hough_dp, error) ||
            !get_dbl(root, "hough_param1", c.hough_param1, error) ||
            !get_dbl(root, "hough_param2", c.hough_param2, error) ||
            !get_dbl(root, "hough_min_dist_frac", c.hough_min_dist_frac, error) ||
            !get_dbl(root, "smooth_alpha", c.smooth_alpha, error)) return false;

        // ---- rtsp ----
        cv::FileNode rtsp;
        if (!get_map(root, "rtsp", rtsp, error)) return false;
        if (!rtsp.empty()) {
            if (!get_bool(rtsp, "enabled", c.rtsp_enabled, error) ||
                !get_str(rtsp, "host", c.rtsp_host, error) ||
                !get_int(rtsp, "port", c.rtsp_port, error) ||
                !get_str(rtsp, "path", c.rtsp_path, error)) return false;
        }

        // ---- preview ----
        cv::FileNode pv;
        if (!get_map(root, "preview", pv, error)) return false;
        if (!pv.empty()) {
            if (!get_bool(pv, "enabled", c.preview_enabled, error) ||
                !get_int(pv, "port", c.preview_port, error) ||
                !get_str(pv, "bind", c.preview_bind, error) ||
                !get_int(pv, "width", c.preview_width, error) ||
                !get_int(pv, "jpeg_quality", c.jpeg_quality, error)) return false;
        }

        // ---- overlay ----
        cv::FileNode ov;
        if (!get_map(root, "overlay", ov, error)) return false;
        if (!ov.empty()) {
            if (!get_bool(ov, "hud", c.overlay_hud, error) ||
                !get_bool(ov, "mask_inset", c.overlay_mask_inset, error) ||
                !get_bool(ov, "crosshair", c.overlay_crosshair, error) ||
                !get_bool(ov, "axes", c.overlay_axes, error)) return false;
        }

        // ---- 输出 ----
        if (!get_str(root, "debug_dir", c.debug_dir, error) ||
            !get_str(root, "save_video", c.save_video, error) ||
            !get_str(root, "out_json", c.out_json, error) ||
            !get_bool(root, "show", c.show, error) ||
            !get_bool(root, "summary", c.summary, error) ||
            !get_bool(root, "quiet", c.quiet, error) ||
            !get_bool(root, "verbose", c.verbose, error)) return false;

        // ---- 取值校验 ----
        auto bad = [&](const std::string& why) { error = "配置值非法: " + why; return false; };
        if (c.width <= 0 || c.height <= 0) return bad("width/height 必须为正");
        if (c.fps <= 0) return bad("fps 必须为正");
        if (c.max_frames < 0) return bad("max_frames 不能为负");
        if (c.work_width < 0) return bad("work_width 不能为负");
        if (c.threads < 0) return bad("threads 不能为负");
        if (c.method != "auto" && c.method != "mask" && c.method != "hough")
            return bad("method 只能是 auto / mask / hough");
        if (c.backend != "auto" && c.backend != "v4l2" && c.backend != "gst")
            return bad("backend 只能是 auto / v4l2 / gst");
        if (c.sat_max < 0 || c.sat_max > 255) return bad("sat_max 需在 0..255");
        if (c.val_min < 0 || c.val_min > 255) return bad("val_min 需在 0..255");
        if (c.min_radius_frac <= 0 || c.max_radius_frac <= c.min_radius_frac)
            return bad("半径比例区间非法");
        if (c.max_axis_ratio < 1.0) return bad("max_axis_ratio 必须 >= 1");
        if (c.ellipse_fit && c.max_axis_ratio < c.report_ellipse_at)
            return bad("max_axis_ratio 不能小于 report_ellipse_at");
        if (c.smooth_alpha < 0.0 || c.smooth_alpha > 1.0) return bad("smooth_alpha 需在 0..1");
        if (c.jpeg_quality < 1 || c.jpeg_quality > 100) return bad("jpeg_quality 需在 1..100");
        if (c.preview_enabled && (c.preview_port < 1 || c.preview_port > 65535))
            return bad("preview.port 非法");
        if (c.rtsp_enabled && (c.rtsp_port < 1 || c.rtsp_port > 65535))
            return bad("rtsp.port 非法");

        // RTSP 地址规范化（空 / 0.0.0.0 / * 一律回环，避免推到自己监听不到的地方）
        c.rtsp_host = normalize_rtsp_host(c.rtsp_host);
        c.rtsp_path = normalize_rtsp_path(c.rtsp_path);
        return true;
    } catch (const cv::Exception& e) {
        error = "解析配置失败: " + std::string(e.what());
        return false;
    }
}

std::string default_config_json() {
    AppConfig d;
    std::ostringstream o;
    o << "{\n"
      << "  \"camera\": \"" << d.device << "\",\n"
      << "  \"device\": \"\",\n"
      << "  \"width\": " << d.width << ",\n"
      << "  \"height\": " << d.height << ",\n"
      << "  \"fps\": " << d.fps << ",\n"
      << "  \"backend\": \"" << d.backend << "\",\n"
      << "  \"focus\": " << d.focus << ",\n"
      << "  \"zoom\": " << d.zoom << ",\n"
      << "  \"max_frames\": " << d.max_frames << ",\n"
      << "  \"loop\": " << (d.loop ? "true" : "false") << ",\n"
      << "  \"work_width\": " << d.work_width << ",\n"
      << "  \"threads\": " << d.threads << ",\n"
      << "  \"method\": \"" << d.method << "\",\n"
      << "  \"expected\": " << d.expected << ",\n"
      << "  \"max_circles\": " << d.max_circles << ",\n"
      << "  \"rtsp\": {\"enabled\": " << (d.rtsp_enabled ? "true" : "false")
      << ", \"host\": \"" << d.rtsp_host << "\", \"port\": " << d.rtsp_port
      << ", \"path\": \"" << d.rtsp_path << "\"},\n"
      << "  \"preview\": {\"enabled\": " << (d.preview_enabled ? "true" : "false")
      << ", \"port\": " << d.preview_port << ", \"bind\": \"" << d.preview_bind
      << "\", \"width\": " << d.preview_width << ", \"jpeg_quality\": " << d.jpeg_quality << "}\n"
      << "}\n";
    return o.str();
}

}  // namespace yuan
