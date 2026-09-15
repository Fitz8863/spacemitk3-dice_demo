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

#include <memory>
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
    // 白垫判别方式：
    //   "hsv" —— S<=sat_max 且 V>=val_min（原始方案）
    //   "min" —— min(B,G,R) >= min_channel_thr（★ 对白平衡漂移免疫）
    //
    // 为什么需要 "min"：白垫是画面里唯一"三个通道都亮"的区域。红垫 B≈40、
    // 蓝垫 R≈12、深色环 V 低，所以地垫/环总有一个通道很暗。自动白平衡漂移只
    // 改通道间的**比例**，不改"三个通道都亮"这个事实 —— 实测白平衡偏暖时
    // 白垫 S 从 9 漂到 73（阈值 70 之外），掩码碎掉、一个盘都检不出；
    // 而 min(B,G,R) 中位仍有 174~183，稳稳高于地垫的 12~51。
    //
    // ★★ 实测结论（2026-09-15，不可作为默认值）：
    //   在正常光照帧上，min 模式会把白垫边界"吃掉"一圈 ——
    //     hsv          : r = 119.0 / 121.1（绿圈压在盘边界上，正确）
    //     min thr=130  : r = 108.0 / 109.3（偏小 11px）
    //     min thr=140  : r = 107.4 / 107.9（偏小 11px）
    //     min thr<=120 : 左盘 132.0（反而偏大 13px，掩码碎裂后拟合失真）
    //   扫遍阈值都到不了 hsv 的精度。根因：白垫与深色环的交界像素是"白+深灰"的
    //   混合，其 B 通道被拉低，min(B,G,R) 因此提前跌破阈值，边界被削。
    //
    //   它确实能救"白平衡漂移"场景（偏暖帧 hsv 只能检出 1 个且要 90ms 跑 Hough，
    //   min thr>=130 能检出 2 个且只要 35ms），但代价是半径小 10px ——
    //   在"精度优先"的前提下不能作为默认值。
    //
    //   定位：**应急可选模式**，默认仍为 "hsv"。
    std::string mask_space     = "hsv";   // hsv(默认，精度高) | min(抗白平衡漂移，但边界偏小)
    int    min_channel_thr     = 140;     // mask_space=min 时，三通道最小值下限
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
    // Hough 兜底降频。设计要点：**只在 Hough 被证明无用之后才降频**。
    //   - Hough 一旦补上了盘（helpful）-> 冷却立刻清零，下一帧照跑
    //   - Hough 连续 hough_useless_limit 次一个盘都没补上 -> 才开始冷却
    //   - mask 检出数量发生变化（场景变了）-> 立刻清零，马上允许重试
    // 这样"能补就继续补"（不降召回），只在它确实白烧 CPU 时才退让。
    int    hough_cooldown_frames = 5;   // 进入冷却后跳过几帧才开始再试
    int    hough_useless_limit   = 2;   // 连续几次无用后才进入冷却
    // 0 = 不降频（每帧都允许兜底，即旧行为）

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
// 检测分成两个可独立执行的阶段，供多线程流水线使用：
//
//   A  extractCandidates()  像素 -> 候选轮廓   （~18ms：resize/色彩/掩码/形态学/泛洪/轮廓/预筛）
//   B  fitCandidates()      候选 -> 最终结果   （~24ms：椭圆 RANSAC/环带验证/打分/去重/排序）
//
// 顺序模式下 detect() = A + B，**与流水线共用同一份实现**，不会分叉。
//
// 线程归属：B 独占一个 CircleDetector 实例（见 CircleDetector 私有成员的说明）。
// ---------------------------------------------------------------------------

// A 阶段的一个候选轮廓。area/r_area 一起带上，B 阶段打印诊断时要用。
struct CandidateContour {
    std::vector<cv::Point> pts;
    int    orig_index = 0;   // 在原 contours 里的序号（保持 "mask#N" 诊断编号不变）
    double area = 0;
    double r_area = 0;
};

// A 阶段的完整产物：B 阶段所需的一切都在这里，不需要回头碰 A 的对象。
struct CandidateSet {
    std::shared_ptr<const cv::Mat> frame;   // 原始帧（未缩放）
    cv::Mat small;                          // 缩放后 BGR（B 的 Hough + 半径尺度基准）
    cv::Mat hsv;                            // small 的 HSV（★ 预计算，避免逐候选重算）
    cv::Mat mask;                           // 白垫掩码（overlay 的 mask_inset + 诊断）
    std::vector<CandidateContour> candidates;  // 通过预筛的候选
    std::vector<std::string> trace;         // A 阶段的诊断行
    double scale = 1.0;
    long   index = 0;                       // 真实帧号（丢帧时不能靠计数推断）
    std::string name;                       // 帧名（JSON 输出用）
    double a_ms = 0;                        // A 阶段耗时（分段统计）
    int    width = 0, height = 0;           // 原始帧尺寸
};

// 采集线程的产物：一帧原始画面 + 帧号 + 帧名
struct CaptureFrame {
    cv::Mat     frame;
    long        index = 0;
    std::string name;
};

// 一次完整检测的结果包（B 阶段的产物 + 计时）
struct DetectJob {
    std::shared_ptr<const CandidateSet> cs;
    DetectResult res;
    double a_ms = 0, b_ms = 0;
};

// ---------------------------------------------------------------------------
class CircleDetector {
public:
    explicit CircleDetector(const CircleParams& p = CircleParams()) : p_(p) {}

    DetectResult detect(const cv::Mat& bgr);

    // A 阶段：像素 -> 候选。无跨帧状态，可安全并发/复用于不同帧。
    CandidateSet extractCandidates(const cv::Mat& bgr_in, long index = 0);

    // B 阶段：候选 -> 结果。**持有跨帧状态**（Hough 冷却等），不可并发调用同一实例。
    DetectResult fitCandidates(const CandidateSet& cs);

    const CircleParams& params() const { return p_; }
    CircleParams&       params()       { return p_; }

    // 半径范围（给定图像尺寸，原图坐标）
    void radiusRange(const cv::Size& sz, double& rmin, double& rmax) const;

    // 本帧实际是否跑了 Hough 兜底（诊断用）
    bool lastFrameUsedHough() const { return last_used_hough_; }

private:
    // B 阶段用：把一组候选轮廓拟合 + 验证成圆/椭圆
    std::vector<CircleResult> fitMaskCandidates(const CandidateSet& cs,
                                                std::vector<std::string>& trace);
    std::vector<CircleResult> detectByHough(const cv::Mat& bgr, const cv::Mat& hsv,
                                            std::vector<std::string>& trace);

    // 内盘/环带采样 + 判据。shape 用参数方程 p(t) = c + M*(cos t, sin t) 表示，
    // 圆和椭圆共用同一套代码。
    // ★ hsv 由调用方预算好传进来：本函数是**逐候选**调用的，之前在这里现算
    //   整图 cvtColor，2 个候选就白烧约 2.3ms（纯重复计算，判据不变）。
    bool validateShape(const cv::Mat& bgr, const cv::Mat& hsv,
                       const cv::Point2f& c, const cv::Matx22d& M,
                       CircleResult& out, std::string* why = nullptr) const;

    CircleParams p_;

    // ---- 跨帧状态：全部只在 B 阶段（fitCandidates）里读写 ------------------
    // 因此**一个 CircleDetector 实例只能被 B 线程独占使用**；
    // A 阶段（extractCandidates）不碰这些成员，可以安全地并行/复用。
    // 如果以后要加状态，先想清楚它属于 A 还是 B —— 放错就是数据竞争。
    // Hough 降频的跨帧状态
    int  hough_cooldown_left_  = 0;   // 还剩几帧不许跑 Hough  [B only]
    int  hough_useless_streak_ = 0;   // 连续几次 Hough 没补上盘
    int  prev_mask_count_      = -1;  // 上一帧 mask 检出数（变化=场景变了）
    bool last_used_hough_      = false;
    bool last_hough_helpful_   = false;
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

// ---------------------------------------------------------------------------
// 绘制任务：把"画标注"这件事推迟到别的线程做
//
// 为什么需要：生产链路上 detect() -> drawOverlay() -> rtsp.publish() 全在主循环
// 串行，而绘制（clone 720p + 画圈写字）实测约 10ms/帧 —— 占关键路径近三成。
// 而 RTSP 编码线程大部分时间在等 VPU，把绘制挪过去就基本白赚。
//
// 检测结果在这里是**已定型的纯数据**，换个线程画不改变任何判据，因此对精度
// 零影响。
// ---------------------------------------------------------------------------
struct OverlayJob {
    cv::Mat        frame;   // 待绘制的原始帧（BGR，捕获分辨率）
    DetectResult   det;     // 检测结果
    OverlayOptions opt;
};

// 画到 job.frame 上（原地）。供推流线程调用。
void renderOverlay(OverlayJob& job);

// 合成一张"红蓝地垫 + 两个白圆垫"的测试图，用于 --self-test
// axis_ratio > 1 时把白垫画成椭圆，用于测试斜视场景
cv::Mat makeSyntheticScene(int width = 1280, int height = 720,
                           double left_cx = 0.20, double right_cx = 0.72,
                           double cy = 0.62, double r_frac = 0.115,
                           double axis_ratio = 1.0, double angle_deg = 0.0);

} // namespace yuan
