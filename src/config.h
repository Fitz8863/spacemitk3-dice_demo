// config.json 读取（用 cv::FileStorage 的 JSON 后端，不引第三方 JSON 库）
//
// 参数优先级：命令行参数 > config.json > 代码内默认值。
// 也就是说 config.json 里的值会先填进 AppConfig，之后命令行再覆盖，
// 所以"改配置"和"临时试一下"两种用法都成立。

#pragma once

#include <string>
#include <vector>

namespace yuan {

struct AppConfig {
    std::string config_path = "config.json";

    // ---- 输入源 ----------------------------------------------------------
    std::string image;                  // 单张图片或目录（批量），空=不用
    std::string video;                  // 视频文件
    std::string device = "/dev/video1"; // 摄像头：/dev/videoN 或索引 "1"
    std::string backend = "auto";       // auto | v4l2 | gst
    int  width = 1280;
    int  height = 720;
    int  fps = 30;
    int  focus = -1;                    // <0 = 不设置
    int  zoom = -1;                     // <0 = 不设置
    int  max_frames = 0;                // 0 = 不限
    bool loop = false;                  // 图片/视频循环

    // ---- 算法 ------------------------------------------------------------
    int    work_width = 640;
    int    threads = 1;                 // 0 = OpenCV 默认；1 = 单线程（实测更省 CPU）
    std::string method = "auto";        // auto | mask | hough
    int    expected = 0;                // >0：不足则触发 Hough 兜底，并只保留前 N 个
    int    max_circles = 4;

    int  sat_max = 70;
    int  val_min = 110;
    int  close_ksize = 9;
    int  open_ksize = 5;
    bool rect_kernel = true;
    bool fill_holes = true;

    double min_radius_frac = 0.05;
    double max_radius_frac = 0.35;
    double min_circularity = 0.55;
    double min_fill_ratio = 0.75;
    double min_inlier_ratio = 0.55;
    bool   ellipse_fit = true;
    double max_axis_ratio = 2.5;
    double report_ellipse_at = 1.08;

    bool   require_ring = true;
    double ring_dark_margin = 22.0;
    double ring_min_ratio = 0.55;

    double hough_dp = 1.2;
    double hough_param1 = 120;
    double hough_param2 = 38;
    double hough_min_dist_frac = 0.20;

    double smooth_alpha = 0.0;

    // ---- RTSP 推流 -------------------------------------------------------
    // H.264 硬编 → MediaMTX；用 RTSP 或 WebRTC 观看
    bool        rtsp_enabled = false;
    std::string rtsp_host = "127.0.0.1";
    int         rtsp_port = 8554;
    std::string rtsp_path = "/dice/circles";

    // ---- 叠加图开关 ------------------------------------------------------
    bool overlay_hud = true;
    bool overlay_mask_inset = true;
    bool overlay_crosshair = true;
    bool overlay_axes = true;

    // ---- 输出 ------------------------------------------------------------
    std::string debug_dir;
    std::string save_video;
    std::string out_json;
    bool show = false;
    bool summary = false;
    bool quiet = false;
    bool verbose = false;
};

// 读取 config.json。失败时填 error 并返回 false。
bool load_config(const std::string& path, AppConfig& config, std::string& error);

// RTSP 目标地址规范化：空/0.0.0.0/* -> 127.0.0.1；path 补前导 '/'、去尾 '/'
std::string normalize_rtsp_host(std::string host);
std::string normalize_rtsp_path(std::string path);

// 生成一份带注释说明的默认配置文本（--dump-config 用）
std::string default_config_json();

}  // namespace yuan
