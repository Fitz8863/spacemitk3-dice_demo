// 取帧：单张图片 / 视频文件 / V4L2 摄像头（OpenCV V4L2 或 GStreamer 后端）

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <string>

namespace yuan {

struct SourceOptions {
    std::string image;              // 单张图片或图片目录
    std::string video;              // 视频文件
    std::string device = "/dev/video1";  // /dev/videoN 或数字索引
    int         width  = 1280;
    int         height = 720;
    int         fps    = 30;
    std::string backend = "auto";   // auto | v4l2 | gst
    int         zoom   = -1;        // >=0 时用 v4l2-ctl 设 zoom_absolute（-1=不动）
    int         focus  = -1;        // >=0 时关自动对焦并设 focus_absolute（-1=不动）
    int         buffersize = 1;
    bool        verbose = false;
};

class FrameSource {
public:
    // 打开数据源；失败返回 false 并填充 err
    bool open(const SourceOptions& opt, std::string& err);

    // 读一帧到 BGR。图片源只成功一次。
    bool read(cv::Mat& bgr);

    // 回到开头（供 --loop 循环播放用）。摄像头无法回卷，返回 false。
    bool rewind();

    bool isStream() const { return stream_; }
    const std::string& describe() const { return desc_; }
    double fps() const { return fps_; }

private:
    cv::VideoCapture cap_;
    cv::Mat          still_;
    bool             still_done_ = false;
    bool             stream_ = false;
    std::string      desc_;
    double           fps_ = 0;
    SourceOptions    opt_;
};

// 用 v4l2-ctl 设置摄像头控制（沿用 dice-game 生产链路的语义：
// zoom/focus < 0 表示"跳过不设置"）。返回是否成功。
bool applyV4l2Controls(const std::string& device, int zoom, int focus, bool verbose);

}  // namespace yuan
