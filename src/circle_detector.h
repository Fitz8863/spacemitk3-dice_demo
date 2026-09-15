// 传统图像算法圆/椭圆盘检测（无模型 / no ONNX, no NN）
//
// 场景：摄像头俯拍或斜拍红蓝地垫上的"白色圆垫 + 深色圆环"（骰子盘）。
// 目标：纯传统 CV 定位这两个盘，不依赖任何训练模型。
//
// 三个必须解决的现实问题，以及对应设计：
//
//  【1】盘上放着东西（骰子、骰盅、手）——会啃掉白垫掩码
//       → 形态学闭运算只能补小洞；改用 flood-fill 填"内部空洞"：
//         从图像四边泛洪背景，凡是没被泛洪到、又不是白垫的区域就是被白垫
//         包住的洞，无论多大一律填掉。于是掩码恒等于整个盘子的轮廓。
//
//  【2】斜视时圆盘在画面里是椭圆，硬用圆拟合会"长短轴各错一半"
//       → 各向异性归一化（whitening）+ 圆 RANSAC = 椭圆 RANSAC：
//         用轮廓点的二阶矩求白化矩阵 W，把椭圆"搓"成圆，在归一化空间里
//         跑原来的圆 RANSAC，再映射回原图就是椭圆。
//         好处：圆度/填充率/RANSAC 全部在归一化空间判定，天然与离心率无关，
//         不需要为椭圆另写一套几何判据。
//
//  【3】盘边搭着东西会撑出凸包，最小二乘会被带偏
//       → RANSAC 三点定圆 + 只对"内点"迭代重估白化矩阵，凸起天然变外点。
//
// 另有一条不依赖地垫颜色的兜底路径 HoughCircles，候选同样过 validate()。

#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace yuan {

// ---------------------------------------------------------------------------
// 参数（全部可通过命令行覆盖，见 main.cpp）
// ---------------------------------------------------------------------------
struct CircleParams {
    // 工作分辨率：0 = 用原始尺寸。缩小可显著提速，坐标会自动映射回原图。
    int    work_width        = 640;

    // --- mask 路径 ---------------------------------------------------------
    int    sat_max           = 70;    // 白垫 HSV 饱和度上限 (0..255)
    int    val_min           = 110;   // 白垫 HSV 亮度下限 (0..255)
    int    close_ksize       = 9;     // 闭运算核：填掉骰子/骰点造成的空洞
    int    open_ksize        = 5;     // 开运算核：去掉细碎噪点
    bool   rect_kernel       = true;  // true=矩形核(可分离，快 3~4 倍)；false=椭圆核
    bool   fill_holes        = true;  // ★ flood-fill 填盘内空洞（抗盘上物体遮挡）

    // --- 几何约束（半径按 min(w,h) 的比例给，与分辨率无关）------------------
    double min_radius_frac   = 0.05;
    double max_radius_frac   = 0.35;
    double min_circularity   = 0.55;  // 归一化空间里的 4*pi*A / P^2
    double min_fill_ratio    = 0.75;  // 归一化空间里的 A / (pi*r^2)
    double min_inlier_ratio  = 0.55;  // RANSAC 内点 / 轮廓点
    bool   ellipse_fit       = true;  // ★ 允许椭圆拟合（斜视必备）
    double max_axis_ratio    = 2.5;   // 长短轴比上限，超过就认为不是盘子
    double report_ellipse_at = 1.08;  // 轴比超过多少就在结果里标记 is_ellipse

    // --- 环带验证（核心判据）----------------------------------------------
    bool   require_ring      = true;
    double ring_dark_margin  = 22.0;  // 环带亮度需比内盘低多少 (0..255)
    double ring_min_ratio    = 0.55;  // 环带角度覆盖比例下限

    // --- hough 路径 --------------------------------------------------------
    double hough_dp            = 1.2;
    double hough_param1        = 120;  // Canny 高阈值
    double hough_param2        = 38;   // 累加器阈值（越大越严）
    double hough_min_dist_frac = 0.20; // minDist = frac * min(w,h)

    // --- 汇总 --------------------------------------------------------------
    std::string method       = "auto"; // mask | hough | auto
    int    max_circles       = 4;      // 最多输出几个（按 score 排序）
    int    expected          = 0;      // >0 时只保留 score 最高的 N 个
    bool   keep_debug        = true;   // 是否生成 mask
    bool   trace             = false;  // 记录每个候选被接受/拒绝的原因
};

// ---------------------------------------------------------------------------
// 检测结果
//
// 圆是椭圆的特例：圆时 a == b == r、angle 无意义。
// r 始终给"等效半径" sqrt(a*b)，方便只关心大小的调用方沿用旧字段。
// ---------------------------------------------------------------------------
struct CircleResult {
    double cx = 0, cy = 0;          // 中心（原图坐标）
    double r  = 0;                  // 等效半径 sqrt(a*b)
    double a  = 0, b = 0;           // 半长轴 / 半短轴
    double angle = 0;               // 长轴方向（度）
    bool   is_ellipse = false;      // 轴比 >= report_ellipse_at
    double axis_ratio = 1.0;        // a / b

    double circularity = 0;         // 归一化空间圆度（与离心率无关）
    double fill_ratio  = 0;         // 归一化空间填充率
    double inlier_ratio = 0;        // RANSAC 内点比例
    double ring_ratio  = 0;         // 环带"非白"角度覆盖率
    double contrast    = 0;         // 内盘亮度 - 环带亮度
    double score       = 0;         // 综合置信度 0..1
    int    side        = -1;        // 0=LEFT 1=RIGHT -1=未知
    std::string method;             // mask | hough
};

struct DetectResult {
    std::vector<CircleResult> circles;
    std::string method_used;        // 实际生效的路径 mask|hough|none
    double latency_ms = 0;
    cv::Mat mask;                   // 白垫二值图（keep_debug 时有效）
    std::vector<std::string> trace; // trace=true 时的候选诊断
};

// ---------------------------------------------------------------------------
class CircleDetector {
public:
    explicit CircleDetector(const CircleParams& p = CircleParams()) : p_(p) {}

    DetectResult detect(const cv::Mat& bgr);

    const CircleParams& params() const { return p_; }
    CircleParams&       params()       { return p_; }

    // 半径范围（给定图像尺寸，原图坐标）
    void radiusRange(const cv::Size& sz, double& rmin, double& rmax) const;

private:
    std::vector<CircleResult> detectByMask(const cv::Mat& bgr, const cv::Mat& hsv,
                                           double scale, cv::Mat& mask_out,
                                           std::vector<std::string>& trace);
    std::vector<CircleResult> detectByHough(const cv::Mat& bgr, const cv::Mat& hsv,
                                            std::vector<std::string>& trace);

    // 内盘/环带采样 + 判据。shape 用参数方程 p(t) = c + M*(cos t, sin t) 表示，
    // 圆和椭圆共用同一套代码。
    bool validateShape(const cv::Mat& hsv, const cv::Point2f& c, const cv::Matx22d& M,
                       CircleResult& out, std::string* why = nullptr) const;

    CircleParams p_;
};

// 调试图 / 预览画面的绘制选项
struct OverlayOptions {
    bool        hud        = true;   // 顶部状态条（帧号/耗时/方法）
    bool        mask_inset = true;   // 右下角"算法眼里的白垫掩码"缩略图
    bool        crosshair  = true;   // 圆心十字
    bool        axes       = true;   // 椭圆时画出长短轴
    double      fps        = 0;      // 有值就显示在状态条上
    long        frame_index = -1;    // >=0 时显示帧号
    std::string source;              // 附加信息（数据源描述）
};

void drawOverlay(const cv::Mat& bgr, const DetectResult& res, cv::Mat& out,
                 const OverlayOptions& opt = OverlayOptions());

// 合成一张"红蓝地垫 + 两个白圆垫"的测试图，用于 --self-test
// axis_ratio > 1 时把白垫画成椭圆，用于测试斜视场景
cv::Mat makeSyntheticScene(int width = 1280, int height = 720,
                           double left_cx = 0.20, double right_cx = 0.72,
                           double cy = 0.62, double r_frac = 0.115,
                           double axis_ratio = 1.0, double angle_deg = 0.0);

} // namespace yuan
