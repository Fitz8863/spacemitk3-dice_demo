#include "frame_source.h"

#include <opencv2/imgcodecs.hpp>

#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace yuan {
namespace {

// "1" -> "/dev/video1"（沿用 dice-game 的 camera 索引语义）
std::string normalizeDevice(const std::string& dev) {
    if (dev.empty()) return "/dev/video1";
    if (dev[0] == '/') return dev;
    bool digits = true;
    for (char c : dev) if (c < '0' || c > '9') { digits = false; break; }
    return digits ? ("/dev/video" + dev) : dev;
}

std::string gstPipeline(const std::string& dev, int w, int h, int fps) {
    std::ostringstream os;
    os << "v4l2src device=" << dev << " io-mode=2 ! "
       << "image/jpeg,width=" << w << ",height=" << h << ",framerate=" << fps << "/1 ! "
       << "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
       << "appsink max-buffers=1 drop=true sync=false";
    return os.str();
}

}  // namespace

bool applyV4l2Controls(const std::string& device, int zoom, int focus, bool verbose) {
    if (zoom < 0 && focus < 0) return true;   // 语义同生产链路：负数 = 跳过

    std::string ctl;
    if (focus >= 0) {
        ctl += "focus_automatic_continuous=0,focus_absolute=" + std::to_string(focus);
    }
    if (zoom >= 0) {
        if (!ctl.empty()) ctl += ",";
        ctl += "zoom_absolute=" + std::to_string(zoom);
    }
    const std::string cmd = "v4l2-ctl -d " + device + " -c " + ctl + " >/dev/null 2>&1";
    if (verbose) std::fprintf(stderr, "[src] %s\n", cmd.c_str());
    return std::system(cmd.c_str()) == 0;
}

bool FrameSource::open(const SourceOptions& opt, std::string& err) {
    opt_ = opt;
    cap_.release();
    still_ = cv::Mat();
    still_done_ = false;
    stream_ = false;

    // --- 单张图片 ---------------------------------------------------------
    if (!opt.image.empty()) {
        still_ = cv::imread(opt.image, cv::IMREAD_COLOR);
        if (still_.empty()) { err = "无法读取图片: " + opt.image; return false; }
        desc_ = "image:" + opt.image;
        stream_ = false;
        return true;
    }

    // --- 视频文件 ---------------------------------------------------------
    if (!opt.video.empty()) {
        if (!cap_.open(opt.video, cv::CAP_FFMPEG)) { err = "无法打开视频: " + opt.video; return false; }
        desc_ = "video:" + opt.video;
        stream_ = true;
        fps_ = cap_.get(cv::CAP_PROP_FPS);
        if (!(fps_ > 0)) fps_ = 25.0;
        return true;
    }

    // --- 摄像头 -----------------------------------------------------------
    const std::string dev = normalizeDevice(opt.device);
    applyV4l2Controls(dev, opt.zoom, opt.focus, opt.verbose);

    auto openV4L2 = [&]() -> bool {
        if (!cap_.open(dev, cv::CAP_V4L2)) return false;
        cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap_.set(cv::CAP_PROP_FRAME_WIDTH,  opt.width);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, opt.height);
        cap_.set(cv::CAP_PROP_FPS,          opt.fps);
        cap_.set(cv::CAP_PROP_BUFFERSIZE,   opt.buffersize);
        cv::Mat probe;
        if (!cap_.read(probe) || probe.empty()) { cap_.release(); return false; }
        return true;
    };
    auto openGst = [&]() -> bool {
        const std::string pipe = gstPipeline(dev, opt.width, opt.height, opt.fps);
        if (opt.verbose) std::fprintf(stderr, "[src] gst: %s\n", pipe.c_str());
        if (!cap_.open(pipe, cv::CAP_GSTREAMER)) return false;
        cv::Mat probe;
        if (!cap_.read(probe) || probe.empty()) { cap_.release(); return false; }
        return true;
    };

    bool ok = false;
    if (opt.backend == "v4l2")      ok = openV4L2();
    else if (opt.backend == "gst")  ok = openGst();
    else                            ok = openV4L2() || openGst();

    if (!ok) {
        err = "无法打开摄像头 " + dev +
              "（可能被其它进程占用，用 fuser/lsof 查；或换 --backend gst）";
        return false;
    }

    const int real_w = (int)cap_.get(cv::CAP_PROP_FRAME_WIDTH);
    const int real_h = (int)cap_.get(cv::CAP_PROP_FRAME_HEIGHT);
    fps_ = cap_.get(cv::CAP_PROP_FPS);
    if (!(fps_ > 0)) fps_ = opt.fps;

    char buf[256];
    std::snprintf(buf, sizeof(buf), "camera:%s %dx%d @%.0ffps (%s)", dev.c_str(),
                  real_w, real_h, fps_, opt.backend.c_str());
    desc_ = buf;
    stream_ = true;
    return true;
}

bool FrameSource::read(cv::Mat& bgr) {
    if (!still_.empty()) {
        if (still_done_) return false;
        bgr = still_;
        still_done_ = true;
        return true;
    }
    return cap_.read(bgr) && !bgr.empty();
}

bool FrameSource::rewind() {
    if (!still_.empty()) {                 // 单张图片：允许反复取
        still_done_ = false;
        return true;
    }
    if (!cap_.isOpened()) return false;
    if (stream_ && opt_.video.empty()) return false;   // 摄像头不能回卷
    return cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
}

}  // namespace yuan
