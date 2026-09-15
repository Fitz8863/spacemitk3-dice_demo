// 传统图像算法圆/椭圆盘检测实现。设计说明见 circle_detector.h 顶部。

#include "circle_detector.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

namespace yuan {
namespace {

inline double nowMs() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

inline bool sampleHSV(const cv::Mat& hsv, double x, double y, cv::Vec3b& out) {
    const int xi = cvRound(x), yi = cvRound(y);
    if (xi < 0 || yi < 0 || xi >= hsv.cols || yi >= hsv.rows) return false;
    out = hsv.at<cv::Vec3b>(yi, xi);
    return true;
}

inline bool sampleBGR(const cv::Mat& bgr, double x, double y, cv::Vec3b& out) {
    const int xi = cvRound(x), yi = cvRound(y);
    if (xi < 0 || yi < 0 || xi >= bgr.cols || yi >= bgr.rows) return false;
    out = bgr.at<cv::Vec3b>(yi, xi);
    return true;
}

// HSV 像素是否"不是白垫"：要么够暗（深色环/阴影），要么够饱和（红蓝地垫）
// 像素是否"不是白垫"。
//   hsv 模式：靠饱和度/亮度（S 高=彩色地垫，V 低=深色环）
//   min 模式：靠"三通道最小值"（白垫三通道都亮；红垫 B 低、蓝垫 R 低）
// 两种模式都只在采样点上调用，成本可忽略。
inline bool nonWhite(const cv::Vec3b& p, int sat_thr, int val_thr) {
    return p[1] > sat_thr || p[2] < val_thr;
}

// BGR 版本的"不是白垫"：三通道最小值低于阈值 => 至少有一个通道不亮
inline bool nonWhiteMin(const cv::Vec3b& p, int min_thr) {
    return std::min({p[0], p[1], p[2]}) < min_thr;
}

inline double clamp01(double v) { return v < 0 ? 0 : (v > 1 ? 1 : v); }

// ---------------------------------------------------------------------------
// 填"内部空洞"：从图像四边泛洪背景，凡是泛洪不到、又不是白垫的像素，
// 就是被白垫整圈包住的洞 —— 无论多大一律补上。
// 这是"盘上放着骰子/骰盅/手"时掩码仍然完整的关键。
// ---------------------------------------------------------------------------
void fillInteriorHoles(cv::Mat& mask) {
    if (mask.empty()) return;
    cv::Mat m;
    cv::copyMakeBorder(mask, m, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::floodFill(m, cv::Point(0, 0), cv::Scalar(255));   // 只填与边界连通的背景
    cv::Mat holes;
    cv::bitwise_not(m, holes);                            // 剩下的 0 就是内部空洞
    holes = holes(cv::Rect(1, 1, mask.cols, mask.rows));
    cv::bitwise_or(mask, holes, mask);
}

// ---------------------------------------------------------------------------
// 各向异性归一化（whitening）：把椭圆"搓"成圆
//
//   p' = W · (p - mu)，  W = diag(1/sqrt(l1), 1/sqrt(l2)) · V^T
//
// 其中 l1>=l2 是轮廓点协方差矩阵的特征值，V 是特征向量。
// 实心椭圆的协方差特征值正好是 a²/4、b²/4，所以白化后半长轴都变成 2，
// 于是"椭圆 RANSAC"就退化成了原来的"圆 RANSAC"，几何判据也全部
// 在归一化空间里做 —— 圆度/填充率天然与离心率无关。
// ---------------------------------------------------------------------------
struct Whitening {
    cv::Point2f mu{0.f, 0.f};
    cv::Matx22d W{1, 0, 0, 1};     // 正向：原图 -> 归一化
    cv::Matx22d Winv{1, 0, 0, 1};  // 反向
    bool identity = true;          // true = 只做了平移（圆模式）
    double ratio = 1.0;            // sqrt(l1/l2)：点集自身的轴比估计

    cv::Point2f fwd(const cv::Point2f& p) const {
        const double dx = p.x - mu.x, dy = p.y - mu.y;
        return cv::Point2f((float)(W(0, 0) * dx + W(0, 1) * dy),
                           (float)(W(1, 0) * dx + W(1, 1) * dy));
    }
    cv::Point2f inv(const cv::Point2f& q) const {
        return cv::Point2f((float)(mu.x + Winv(0, 0) * q.x + Winv(0, 1) * q.y),
                           (float)(mu.y + Winv(1, 0) * q.x + Winv(1, 1) * q.y));
    }
};

// cond_limit：协方差特征值比的上限（防止退化成一条线时数值爆掉）。
// 真正的"太扁就不认"由最终轴比检查负责，这里给宽松值。
//
// ★ 必须用"填充区域"的二阶矩，不能用轮廓点的二阶矩：
//   实心椭圆 E[x²]=a²/4、E[y²]=b²/4 → sqrt 比恰好是 a/b，白化一次到位；
//   而轮廓点（弧长均匀）的比只有 a/b 的 0.89 倍左右（实测 a/b=1.6 → 1.428），
//   白化不足会让归一化后的形状仍是个椭圆，RANSAC 只圈住中间一条带，
//   用这条带再估白化会越估越扁 —— 迭代直接发散（实测轴比从 1.6 一路跑到 3.5）。
//   cv::moments(contour) 算的正是多边形填充区域的矩，所以直接用它。
bool computeWhitening(const std::vector<cv::Point>& pts, bool use_hull,
                      double cond_limit, Whitening& out) {
    if (pts.size() < 8) return false;

    std::vector<cv::Point> poly;
    if (use_hull) {
        cv::convexHull(pts, poly);          // 凸包对"被啃掉的缺口"天然免疫
        if (poly.size() < 3) return false;
    } else {
        poly = pts;
    }

    const cv::Moments m = cv::moments(poly, false);
    if (!(m.m00 > 1e-6)) return false;

    const double sxx = m.mu20 / m.m00, sxy = m.mu11 / m.m00, syy = m.mu02 / m.m00;
    const cv::Mat C = (cv::Mat_<double>(2, 2) << sxx, sxy, sxy, syy);
    cv::Mat evals, evecs;
    if (!cv::eigen(C, evals, evecs)) return false;

    double l1 = evals.at<double>(0), l2 = evals.at<double>(1);
    cv::Matx22d V(evecs.at<double>(0, 0), evecs.at<double>(0, 1),
                  evecs.at<double>(1, 0), evecs.at<double>(1, 1));
    // cv::eigen 的特征值/eigenvector 顺序在文档里没写死，这里按大小自行排序，
    // 免得依赖实现细节（顺序反了会把长短轴调换，结果完全错）。
    if (l1 < l2) { std::swap(l1, l2); V = cv::Matx22d(V(1, 0), V(1, 1), V(0, 0), V(0, 1)); }
    if (!(l2 > 1e-9)) return false;
    if (l1 / l2 > cond_limit * cond_limit) return false;

    const double s1 = std::sqrt(l1), s2 = std::sqrt(l2);
    const cv::Matx22d D(1.0 / s1, 0, 0, 1.0 / s2);

    out.mu = cv::Point2f((float)(m.m10 / m.m00), (float)(m.m01 / m.m00));
    out.W = D * V;
    out.Winv = V.t() * cv::Matx22d(s1, 0, 0, s2);
    out.identity = false;
    out.ratio = std::sqrt(l1 / l2);
    return true;
}

// 三点外接圆。共线返回 false。
bool circleFrom3(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c,
                 cv::Point2f& ctr, double& r) {
    const double d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
    if (std::fabs(d) < 1e-9) return false;
    const double a2 = a.x * a.x + a.y * a.y;
    const double b2 = b.x * b.x + b.y * b.y;
    const double c2 = c.x * c.x + c.y * c.y;
    const double ux = (a2 * (b.y - c.y) + b2 * (c.y - a.y) + c2 * (a.y - b.y)) / d;
    const double uy = (a2 * (c.x - b.x) + b2 * (a.x - c.x) + c2 * (b.x - a.x)) / d;
    ctr = cv::Point2f((float)ux, (float)uy);
    r = std::hypot(a.x - ux, a.y - uy);
    return std::isfinite(r) && r > 1e-6;
}

// Kasa 代数圆拟合，对一组内点做精修（迭代一次剔除离群点）
bool kasaFit(const std::vector<cv::Point2f>& pts, cv::Point2f& center, double& radius) {
    std::vector<cv::Point2f> cur = pts;
    for (int iter = 0; iter < 2; ++iter) {
        const size_t n = cur.size();
        if (n < 8) return false;
        double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, sxz = 0, syz = 0, sz = 0;
        for (const auto& p : cur) {
            const double x = p.x, y = p.y, z = x * x + y * y;
            sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
            sxz += x * z; syz += y * z; sz += z;
        }
        cv::Mat A = (cv::Mat_<double>(3, 3) << sxx, sxy, sx,
                                               sxy, syy, sy,
                                               sx,  sy,  (double)n);
        cv::Mat b = (cv::Mat_<double>(3, 1) << -sxz, -syz, -sz);
        cv::Mat sol;
        if (!cv::solve(A, b, sol, cv::DECOMP_SVD)) return false;
        const double D = sol.at<double>(0), E = sol.at<double>(1), F = sol.at<double>(2);
        const double cx = -D / 2.0, cy = -E / 2.0;
        const double rr = cx * cx + cy * cy - F;
        if (!(rr > 1e-6)) return false;
        center = cv::Point2f((float)cx, (float)cy);
        radius = std::sqrt(rr);

        if (iter == 0) {
            std::vector<double> res;
            res.reserve(n);
            for (const auto& p : cur) res.push_back(std::fabs(std::hypot(p.x - cx, p.y - cy) - radius));
            std::vector<double> sorted = res;
            std::sort(sorted.begin(), sorted.end());
            const double thr = std::max(sorted[sorted.size() / 2] * 3.0, 1.5);
            std::vector<cv::Point2f> keep;
            keep.reserve(n);
            for (size_t i = 0; i < n; ++i)
                if (res[i] <= thr) keep.push_back(cur[i]);
            if (keep.size() < 8) break;
            cur.swap(keep);
        }
    }
    return true;
}

// 归一化空间里的圆 RANSAC。凸起/遮挡造成的离群点会被内点计数自然淘汰。
bool ransacCircleF(const std::vector<cv::Point2f>& P, double rmin, double rmax,
                   cv::Point2f& center, double& radius, double& inlier_ratio,
                   unsigned seed = 20260915u) {
    const size_t n = P.size();
    if (n < 12) return false;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> pick(0, n - 1);

    size_t best_cnt = 0;
    cv::Point2f best_c;
    double best_r = 0;

    // 300 次足够：白化后内点比例通常 >0.7，三次全中的概率 ~0.34，
    // 300 次里几乎必然命中好几次最优解。
    for (int it = 0; it < 300; ++it) {
        const size_t i0 = pick(rng);
        size_t i1 = pick(rng), i2 = pick(rng);
        if (i1 == i0) i1 = (i1 + 1) % n;
        if (i2 == i0 || i2 == i1) i2 = (i2 + 2) % n;
        if (i2 == i0 || i2 == i1) continue;

        cv::Point2f c;
        double r = 0;
        if (!circleFrom3(P[i0], P[i1], P[i2], c, r)) continue;
        if (r < rmin || r > rmax) continue;

        const double tol = std::max(0.04 * r, 1e-3);
        size_t cnt = 0;
        for (const auto& p : P)
            if (std::fabs(std::hypot(p.x - c.x, p.y - c.y) - r) <= tol) ++cnt;
        if (cnt > best_cnt) { best_cnt = cnt; best_c = c; best_r = r; }
    }
    if (best_cnt < std::max<size_t>(10, n / 4)) return false;

    // 内点精修（归一化空间里半径量级是 O(1)，容差按比例给）
    const double tol = std::max(0.04 * best_r, 1e-3);
    std::vector<cv::Point2f> inliers;
    inliers.reserve(best_cnt);
    for (const auto& p : P)
        if (std::fabs(std::hypot(p.x - best_c.x, p.y - best_c.y) - best_r) <= tol)
            inliers.push_back(p);

    cv::Point2f rc;
    double rr = 0;
    if (kasaFit(inliers, rc, rr) && rr >= rmin && rr <= rmax) { best_c = rc; best_r = rr; }

    size_t cnt = 0;
    const double tol2 = std::max(0.04 * best_r, 1e-3);
    for (const auto& p : P)
        if (std::fabs(std::hypot(p.x - best_c.x, p.y - best_c.y) - best_r) <= tol2) ++cnt;

    center = best_c;
    radius = best_r;
    inlier_ratio = double(cnt) / n;
    return true;
}

// ---------------------------------------------------------------------------
// 拟合结果：形状用参数方程 p(t) = c + M·(cos t, sin t) 表示
// 圆是 M = r·I 的特例，所以绘制/采样/验证三者共用一套代码。
// ---------------------------------------------------------------------------
struct ShapeFit {
    cv::Point2f c;
    cv::Matx22d M{1, 0, 0, 1};
    double a = 0, b = 0, angle = 0;   // 由 M 的 SVD 得到
    double r_equiv = 0;               // sqrt(a*b)
    double inlier_ratio = 0;
    double circ_white = 0;            // 归一化空间圆度
    double fill_white = 0;            // 归一化空间填充率
    double moment_ratio = 0;          // 诊断用：归一化前的点集轴比估计
    double wr = 0;                    // 诊断用：归一化空间拟合半径
};

inline cv::Point2f shapePoint(const cv::Point2f& c, const cv::Matx22d& M,
                              double t, double s) {
    const double ct = std::cos(t) * s, st = std::sin(t) * s;
    return cv::Point2f((float)(c.x + M(0, 0) * ct + M(0, 1) * st),
                       (float)(c.y + M(1, 0) * ct + M(1, 1) * st));
}

// 主拟合：白化 + 圆 RANSAC，迭代让白化矩阵收敛到"只用内点"估计
bool fitShapeRobust(const std::vector<cv::Point>& contour, bool allow_ellipse,
                    double max_axis_ratio, double rmin, double rmax, ShapeFit& out,
                    std::string* why = nullptr) {
    const size_t n = contour.size();
    if (n < 12) { if (why) *why = "轮廓点太少"; return false; }

    std::vector<cv::Point2f> P;
    P.reserve(n);
    for (const auto& p : contour) P.emplace_back((float)p.x, (float)p.y);

    // ★ 抽稀：拟合不需要几百个边界点，~220 个足够，RANSAC 单轮成本直接降 3 倍。
    //   矩估计仍然用完整轮廓（面积才准），所以抽稀只影响拟合采样。
    const size_t stride = std::max<size_t>(1, P.size() / 220);
    std::vector<cv::Point2f> Pf;
    Pf.reserve(P.size() / stride + 1);
    for (size_t i = 0; i < P.size(); i += stride) Pf.push_back(P[i]);

    std::vector<cv::Point> work = contour;      // 当前用于估白化矩阵的点（完整顺序，算矩用）
    Whitening W;
    cv::Point2f wc;
    double wr = 0, winl = 0;

    const int kIters = allow_ellipse ? 3 : 1;   // 圆模式白化恒定，迭代没有意义
    double prev_ratio = -1.0;

    for (int iter = 0; iter < kIters; ++iter) {
        double lo = rmin, hi = rmax;
        if (allow_ellipse) {
            // 宽松条件数限制：真正的"太扁不认"留给下面的轴比检查。
            // 第 0 轮用整条轮廓的填充矩；之后用内点的凸包矩（凸包填平缺口，
            // 所以"盘边搭着东西"不会把轴向估计带偏）。
            if (!computeWhitening(work, iter > 0, 8.0, W)) {
                if (why) *why = "白化失败（协方差退化或过于细长）";
                return false;
            }
            out.moment_ratio = W.ratio;
            // 收敛就早停：轴比几乎不变说明白化矩阵已经稳定
            if (prev_ratio > 0 && std::fabs(W.ratio - prev_ratio) < 0.005 * prev_ratio) break;
            prev_ratio = W.ratio;
        } else {
            cv::Point2f mu(0.f, 0.f);
            for (const auto& p : work) mu += cv::Point2f((float)p.x, (float)p.y);
            mu *= 1.0f / float(work.size());
            W.mu = mu; W.W = cv::Matx22d(1, 0, 0, 1); W.Winv = W.W; W.identity = true;
        }

        // 白化后的点，以及（椭圆模式下）自适应的半径搜索区间
        std::vector<cv::Point2f> wp;
        wp.reserve(Pf.size());
        for (const auto& p : Pf) wp.push_back(W.fwd(p));

        if (allow_ellipse) {
            // 白化把半长轴都压到 2 左右，留足遮挡造成的偏差余量
            lo = 0.8; hi = 4.5;
        }

        if (!ransacCircleF(wp, lo, hi, wc, wr, winl, 20260915u + 7919u * (unsigned)iter)) {
            if (why) {
                char b[160];
                std::snprintf(b, sizeof(b), "归一化空间圆 RANSAC 失败（点数=%zu, r 区间=[%.2f,%.2f]）",
                              Pf.size(), lo, hi);
                *why = b;
            }
            return false;
        }
        out.wr = wr;

        // 用内点重估白化矩阵：被骰子啃掉/凸出的部分不再影响轴向估计
        const double tol = std::max(0.04 * wr, 1e-3);
        std::vector<cv::Point> next;
        next.reserve(work.size());
        size_t k = 0;
        for (size_t i = 0; i < P.size(); ++i) {
            if (i % stride != 0) continue;
            const cv::Point2f& q = wp[k++];
            if (std::fabs(std::hypot(q.x - wc.x, q.y - wc.y) - wr) <= tol)
                next.push_back(contour[i]);
        }
        if (next.size() < 12) break;
        work.swap(next);
    }

    // 映射回原图：中心 c = mu + Winv·wc，形状矩阵 M = wr·Winv
    out.c = W.inv(wc);
    out.M = W.Winv * wr;

    cv::Mat wsv, usv, vtsv;
    cv::SVD::compute(cv::Mat(out.M), wsv, usv, vtsv);
    out.a = wsv.at<double>(0);
    out.b = wsv.at<double>(1);
    out.angle = std::atan2(usv.at<double>(1, 0), usv.at<double>(0, 0)) * 180.0 / CV_PI;
    out.r_equiv = std::sqrt(std::max(0.0, out.a * out.b));
    out.inlier_ratio = winl;

    char b[256];
    if (!(out.b > 1e-6)) { if (why) *why = "退化形状（短轴为 0）"; return false; }
    if (out.a / out.b > max_axis_ratio) {
        if (why) { std::snprintf(b, sizeof(b), "太扁：轴比 %.2f > %.2f (点集估计 rho=%.2f, wr=%.2f)",
                                 out.a / out.b, max_axis_ratio, out.moment_ratio, out.wr); *why = b; }
        return false;
    }
    if (out.r_equiv < rmin * 0.7 || out.r_equiv > rmax * 1.4) {
        if (why) { std::snprintf(b, sizeof(b), "等效半径 %.1f 超出 [%.1f, %.1f]",
                                 out.r_equiv, rmin * 0.7, rmax * 1.4); *why = b; }
        return false;
    }

    // 归一化空间里的圆度 / 填充率（与离心率无关）
    std::vector<cv::Point2f> wp;
    wp.reserve(P.size());
    for (const auto& p : P) wp.push_back(W.fwd(p));
    const double area = std::fabs(cv::contourArea(wp));
    const double peri = cv::arcLength(wp, true);
    out.circ_white = peri > 0 ? 4.0 * CV_PI * area / (peri * peri) : 0.0;
    out.fill_white = area / (CV_PI * wr * wr);
    return true;
}

}  // namespace

void CircleDetector::radiusRange(const cv::Size& sz, double& rmin, double& rmax) const {
    const double base = std::min(sz.width, sz.height);
    rmin = p_.min_radius_frac * base;
    rmax = p_.max_radius_frac * base;
}

// ---------------------------------------------------------------------------
// 内盘 / 环带采样 + 判据（mask 与 hough 共用，圆与椭圆共用）
//
// 物理模型：一个"白色盘面"被一圈"非白环带"包住。
//   - 内盘：p(t, 0 / 0.32 / 0.50 / 0.68)，要求 >=60% 是白的
//   - 环带：p(t, 1.08 / 1.16 / 1.24)，要求多数角度上整条射线被非白占据
// 参数 s 是相对形状边界的缩放，所以椭圆盘上环带会自动跟着椭圆走。
// ---------------------------------------------------------------------------
bool CircleDetector::validateShape(const cv::Mat& bgr, const cv::Point2f& c,
                                   const cv::Matx22d& M, CircleResult& out,
                                   std::string* why) const {
    // 判别所需的数据：hsv 模式要 HSV 图，min 模式直接看 BGR
    const bool use_min = (p_.mask_space == "min");
    cv::Mat hsv;
    if (!use_min) cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    const cv::Mat& src = use_min ? bgr : hsv;
    auto isWhite = [&](const cv::Vec3b& px) {
        return use_min ? !nonWhiteMin(px, p_.min_channel_thr)
                       : !nonWhite(px, p_.sat_max, p_.val_min);
    };
    auto sample = [&](double x, double y, cv::Vec3b& px) {
        return sampleBGR(src, x, y, px);
    };
    const int kAngles = 72;
    // 盘面证据取"近边缘带"：盘中央被骰盅/骰子压住时，靠边的这一圈仍然露着白
    const double kRimFrac[] = {0.62, 0.78, 0.92};
    // 整体弱要求：只是排除"整个盘都是黑/彩"的情况，不做强约束
    const double kCoreFrac[] = {0.0, 0.30, 0.50, 0.72, 0.90};
    const double kRingFrac[] = {1.08, 1.16, 1.24};
    int rim_total = 0, rim_white_n = 0, core_total = 0, core_white_n = 0;
    double in_v_sum = 0;
    for (int a = 0; a < kAngles; ++a) {
        const double t = 2.0 * CV_PI * a / kAngles;
        for (double f : kCoreFrac) {
            const cv::Point2f q = shapePoint(c, M, t, f);
            cv::Vec3b px;
            if (!sample(q.x, q.y, px)) continue;
            ++core_total;
            in_v_sum += px[2];
            if (isWhite(px)) ++core_white_n;
        }
        for (double f : kRimFrac) {
            const cv::Point2f q = shapePoint(c, M, t, f);
            cv::Vec3b px;
            if (!sample(q.x, q.y, px)) continue;
            ++rim_total;
            if (isWhite(px)) ++rim_white_n;
        }
    }
    if (core_total < 16 || rim_total < 16) {
        if (why) *why = "circle too close to image border";
        return false;
    }
    const double rim_white  = double(rim_white_n) / rim_total;
    const double core_white = double(core_white_n) / core_total;
    const double inner_v = in_v_sum / core_total;

    // --- 环带 -------------------------------------------------------------
    int angle_hit = 0, ring_total = 0, ring_nonwhite = 0;
    double ring_v_sum = 0;
    for (int a = 0; a < kAngles; ++a) {
        const double t = 2.0 * CV_PI * a / kAngles;
        int hits = 0, valid = 0;
        for (double f : kRingFrac) {
            const cv::Point2f q = shapePoint(c, M, t, f);
            cv::Vec3b px;
            if (!sample(q.x, q.y, px)) continue;
            ++valid; ++ring_total;
            ring_v_sum += px[2];
            if (!isWhite(px)) { ++hits; ++ring_nonwhite; }
        }
        if (valid > 0 && hits * 2 >= valid) ++angle_hit;
    }
    if (ring_total < 16) { if (why) *why = "ring band out of image"; return false; }
    const double ring_ratio = double(angle_hit) / kAngles;
    const double ring_nonwhite_ratio = double(ring_nonwhite) / ring_total;
    const double contrast = inner_v - ring_v_sum / ring_total;

    char b[288];
    if (rim_white < 0.55) {
        if (why) { std::snprintf(b, sizeof(b), "rim not white: %.2f < 0.55 (盘面近边缘不白，可能不是盘子)",
                                 rim_white); *why = b; }
        return false;
    }
    if (core_white < 0.20) {
        if (why) { std::snprintf(b, sizeof(b), "core almost non-white: %.2f < 0.20 (inner_v=%.0f)",
                                 core_white, inner_v); *why = b; }
        return false;
    }
    if (p_.require_ring) {
        if (ring_ratio < p_.ring_min_ratio) {
            if (why) { std::snprintf(b, sizeof(b), "ring coverage low: %.2f < %.2f",
                                     ring_ratio, p_.ring_min_ratio); *why = b; }
            return false;
        }
        if (contrast < p_.ring_dark_margin && ring_nonwhite_ratio < 0.75) {
            if (why) { std::snprintf(b, sizeof(b), "ring contrast weak: %.1f < %.1f and nonwhite %.2f < 0.75",
                                     contrast, p_.ring_dark_margin, ring_nonwhite_ratio); *why = b; }
            return false;
        }
    }

    out.ring_ratio = ring_ratio;
    out.contrast   = contrast;
    if (why) {
        std::snprintf(b, sizeof(b), "ok rim_white=%.2f core_white=%.2f ring=%.2f contrast=%.1f",
                      rim_white, core_white, ring_ratio, contrast);
        *why = b;
    }
    return true;
}

// ---------------------------------------------------------------------------
// 路径 1：白垫掩码 + 填内部空洞 + 白化椭圆 RANSAC
// ---------------------------------------------------------------------------
std::vector<CircleResult> CircleDetector::detectByMask(const cv::Mat& bgr, const cv::Mat& hsv,
                                                       double scale, cv::Mat& mask_out,
                                                       std::vector<std::string>& trace) {
    std::vector<CircleResult> found;

    // 白垫掩码。两种判别，见 CircleParams::mask_space 的说明。
    cv::Mat mask;
    if (p_.mask_space == "min") {
        // ★ 三通道最小值：白垫三通道都亮，红垫 B 低 / 蓝垫 R 低 / 深色环 V 低。
        //   对白平衡漂移免疫，且只需一次 inRange（比 cvtColor+inRange 还快 ~0.7ms）。
        const int t = p_.min_channel_thr;
        cv::inRange(bgr, cv::Scalar(t, t, t), cv::Scalar(255, 255, 255), mask);
    } else {
        // S 低 + V 高：红/蓝地垫饱和度高、深色环亮度低，都被排除
        cv::inRange(hsv, cv::Scalar(0, 0, p_.val_min), cv::Scalar(179, p_.sat_max, 255), mask);
    }

    // 矩形核是可分离的（OpenCV 走行列两趟），比椭圆核快 3~4 倍
    const int ktype = p_.rect_kernel ? cv::MORPH_RECT : cv::MORPH_ELLIPSE;
    if (p_.close_ksize >= 3) {
        cv::Mat k = cv::getStructuringElement(ktype, cv::Size(p_.close_ksize, p_.close_ksize));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k);
    }
    if (p_.open_ksize >= 3) {
        cv::Mat k = cv::getStructuringElement(ktype, cv::Size(p_.open_ksize, p_.open_ksize));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k);
    }
    // ★ 盘上压着骰子/骰盅/手时，闭运算补不了那么大的缺口，靠泛洪填内部空洞
    if (p_.fill_holes) fillInteriorHoles(mask);

    if (p_.keep_debug) mask_out = mask.clone();

    double rmin = 0, rmax = 0;
    radiusRange(bgr.size(), rmin, rmax);
    (void)scale;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    char b[288];
    for (size_t ci = 0; ci < contours.size(); ++ci) {
        const auto& c = contours[ci];
        const double area = cv::contourArea(c);
        if (area < CV_PI * rmin * rmin * 0.5) {
            if (p_.trace) { std::snprintf(b, sizeof(b), "mask#%zu area=%.0f -> reject: too small", ci, area); trace.push_back(b); }
            continue;
        }

        // 用"面积等效半径"做预筛：r = sqrt(A/pi) 对离心率不变（椭圆时正好等于
        // sqrt(a*b)），所以斜视下不会像 minEnclosingCircle 那样被长轴撑大而误杀。
        // 顺带省掉一次 minEnclosingCircle（O(n·hull)）。
        const double r_area = std::sqrt(area / CV_PI);
        if (r_area < rmin * 0.7 || r_area > rmax * 1.4) {
            if (p_.trace) { std::snprintf(b, sizeof(b), "mask#%zu area=%.0f r_area=%.1f -> reject: 等效半径超出 [%.0f,%.0f]",
                                          ci, area, r_area, rmin * 0.7, rmax * 1.4); trace.push_back(b); }
            continue;
        }

        ShapeFit fit;
        std::string fit_why;
        if (!fitShapeRobust(c, p_.ellipse_fit, p_.max_axis_ratio, rmin, rmax, fit,
                            p_.trace ? &fit_why : nullptr)) {
            if (p_.trace) { std::snprintf(b, sizeof(b), "mask#%zu area=%.0f r_area=%.1f -> reject: %s",
                                          ci, area, r_area, fit_why.c_str()); trace.push_back(b); }
            continue;
        }
        if (fit.inlier_ratio < p_.min_inlier_ratio) {
            if (p_.trace) { std::snprintf(b, sizeof(b), "mask#%zu area=%.0f r=%.1f ratio=%.2f -> reject: inlier %.3f < %.2f",
                                          ci, area, fit.r_equiv, fit.a / fit.b, fit.inlier_ratio, p_.min_inlier_ratio); trace.push_back(b); }
            continue;
        }
        if (fit.circ_white < p_.min_circularity || fit.fill_white < p_.min_fill_ratio) {
            if (p_.trace) { std::snprintf(b, sizeof(b), "mask#%zu area=%.0f r=%.1f inlier=%.3f -> reject: 归一化 circ=%.3f(>=%.2f) fill=%.3f(>=%.2f)",
                                          ci, area, fit.r_equiv, fit.inlier_ratio,
                                          fit.circ_white, p_.min_circularity,
                                          fit.fill_white, p_.min_fill_ratio); trace.push_back(b); }
            continue;
        }

        CircleResult cand;
        std::string why;
        const bool ok = validateShape(bgr, fit.c, fit.M, cand, p_.trace ? &why : nullptr);
        if (p_.trace) {
            std::snprintf(b, sizeof(b),
                          "mask#%zu area=%.0f -> %s r=%.1f a/b=%.2f ang=%.1f circ=%.3f fill=%.3f inlier=%.3f (rho=%.2f wr=%.2f) | %s",
                          ci, area, ok ? "ACCEPT" : "reject:", fit.r_equiv, fit.a / fit.b,
                          fit.angle, fit.circ_white, fit.fill_white, fit.inlier_ratio,
                          fit.moment_ratio, fit.wr, why.c_str());
            trace.push_back(b);
        }
        if (!ok) continue;

        cand.cx = fit.c.x; cand.cy = fit.c.y;
        cand.a = fit.a; cand.b = fit.b; cand.angle = fit.angle;
        cand.r = fit.r_equiv;
        cand.axis_ratio = fit.a / fit.b;
        cand.is_ellipse = cand.axis_ratio >= p_.report_ellipse_at;
        cand.circularity  = fit.circ_white;
        cand.fill_ratio   = fit.fill_white;
        cand.inlier_ratio = fit.inlier_ratio;
        cand.method = "mask";
        found.push_back(cand);
    }
    return found;
}

// ---------------------------------------------------------------------------
// 路径 2：HoughCircles + 同样的物理验证（不依赖地垫颜色的兜底路径）
// 注意 HoughCircles 只能找圆，斜视下精度不如 mask 路径，仅作兜底。
// ---------------------------------------------------------------------------
std::vector<CircleResult> CircleDetector::detectByHough(const cv::Mat& bgr, const cv::Mat& hsv,
                                                       std::vector<std::string>& trace) {
    std::vector<CircleResult> found;

    double rmin = 0, rmax = 0;
    radiusRange(bgr.size(), rmin, rmax);
    if (rmax <= rmin + 2) return found;

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray, gray, 5);

    std::vector<cv::Vec3f> hc;
    cv::HoughCircles(gray, hc, cv::HOUGH_GRADIENT, p_.hough_dp,
                     p_.hough_min_dist_frac * std::min(bgr.cols, bgr.rows),
                     p_.hough_param1, p_.hough_param2,
                     (int)std::floor(rmin), (int)std::ceil(rmax));

    char b[256];
    for (size_t i = 0; i < hc.size(); ++i) {
        CircleResult cand;
        std::string why;
        const cv::Matx22d M(hc[i][2], 0, 0, hc[i][2]);
        const bool ok = validateShape(bgr, cv::Point2f(hc[i][0], hc[i][1]), M, cand,
                                      p_.trace ? &why : nullptr);
        if (p_.trace) {
            std::snprintf(b, sizeof(b), "hough#%zu cx=%.1f cy=%.1f r=%.1f -> %s %s",
                          i, (double)hc[i][0], (double)hc[i][1], (double)hc[i][2],
                          ok ? "ACCEPT" : "reject:", why.c_str());
            trace.push_back(b);
        }
        if (!ok) continue;
        cand.cx = hc[i][0]; cand.cy = hc[i][1];
        cand.r = cand.a = cand.b = hc[i][2];
        cand.axis_ratio = 1.0;
        cand.method = "hough";
        found.push_back(cand);
    }
    return found;
}

// ---------------------------------------------------------------------------
// 汇总：跑一条或两条路径 -> 打分 -> 去重 -> 排序 -> 分左右
// ---------------------------------------------------------------------------
DetectResult CircleDetector::detect(const cv::Mat& bgr_in) {
    const double t0 = nowMs();
    DetectResult res;
    if (bgr_in.empty()) return res;
    if (!p_.trace) res.trace.clear();

    cv::Mat bgr = bgr_in;
    double scale = 1.0;
    if (p_.work_width > 0 && bgr.cols > p_.work_width) {
        scale = double(p_.work_width) / bgr.cols;
        cv::resize(bgr, bgr, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    const bool want_mask  = (p_.method == "mask"  || p_.method == "auto");
    const bool want_hough = (p_.method == "hough" || p_.method == "auto");

    std::vector<CircleResult> mask_res, hough_res;
    cv::Mat mask_img;
    if (want_mask) mask_res = detectByMask(bgr, hsv, scale, mask_img, res.trace);

    const bool mask_enough = !mask_res.empty() &&
                             (p_.expected <= 0 || (int)mask_res.size() >= p_.expected);

    // ---- Hough 兜底降频 --------------------------------------------------
    // 问题：mask 少找到一个盘时（遮挡/光照漂移），旧逻辑每帧都跑 Hough，
    //       而 Hough 要 28~75ms —— 帧率被拖到 12fps 上下，画面却可能只是"少一个盘"。
    //
    // 但**不能一律降频**：Hough 有时确实能补上第二个盘，压掉它就是降召回。
    // 所以策略是"只在证明无用之后才退让"：
    //   ① Hough 补上了盘       -> 立刻清零，下一帧继续跑（它在挣自己的开销）
    //   ② 连续 N 次没补上      -> 进入冷却，隔几帧再试
    //   ③ mask 检出数变了（场景变）-> 立刻清零，马上重试
    // 这样既不盲降召回，又能把"确实白烧"的那部分开销拿掉。
    const int mask_count = (int)mask_res.size();
    if (mask_count != prev_mask_count_) {     // ③ 场景变了 -> 立即允许重试
        hough_cooldown_left_  = 0;
        hough_useless_streak_ = 0;
        prev_mask_count_      = mask_count;
    }

    last_used_hough_ = false;
    last_hough_helpful_ = false;
    if (want_hough && !mask_enough) {
        if (p_.hough_cooldown_frames <= 0 || hough_cooldown_left_ <= 0) {
            hough_res = detectByHough(bgr, hsv, res.trace);
            last_used_hough_ = true;
        } else {
            --hough_cooldown_left_;
            if (p_.trace) {
                char b[176];
                std::snprintf(b, sizeof(b),
                              "hough: 跳过（冷却中，还剩 %d 帧；mask=%d，期望 %d）",
                              hough_cooldown_left_, mask_count,
                              p_.expected > 0 ? p_.expected : -1);
                res.trace.push_back(b);
            }
        }
    } else {
        hough_cooldown_left_  = 0;   // mask 达标 -> 立刻恢复兜底能力
        hough_useless_streak_ = 0;
    }

    std::vector<CircleResult> all = mask_res;
    all.insert(all.end(), hough_res.begin(), hough_res.end());

    for (auto& c : all) {
        double s;
        if (c.method == "mask") {
            s = 0.35 * clamp01((c.inlier_ratio - 0.5) / 0.45)
              + 0.15 * clamp01((c.fill_ratio   - 0.75) / 0.25)
              + 0.30 * clamp01(c.ring_ratio)
              + 0.20 * clamp01(c.contrast / 80.0);
        } else {
            s = 0.50 * clamp01(c.ring_ratio)
              + 0.30 * clamp01(c.contrast / 80.0)
              + 0.20 * clamp01((c.r / std::min(bgr.cols, bgr.rows)) / p_.max_radius_frac);
        }
        c.score = clamp01(s);
    }

    std::sort(all.begin(), all.end(),
              [](const CircleResult& a, const CircleResult& b) { return a.score > b.score; });

    std::vector<CircleResult> kept;
    for (const auto& c : all) {
        bool dup = false;
        for (const auto& k : kept)
            if (std::hypot(c.cx - k.cx, c.cy - k.cy) < 0.6 * std::max(c.r, k.r)) { dup = true; break; }
        if (!dup) kept.push_back(c);
    }

    int limit = p_.max_circles;
    if (p_.expected > 0 && (int)kept.size() > p_.expected) limit = p_.expected;
    if (limit > 0 && (int)kept.size() > limit) kept.resize(limit);

    // 按 x 排序 + 分左右 + 坐标映射回原图
    std::sort(kept.begin(), kept.end(),
              [](const CircleResult& a, const CircleResult& b) { return a.cx < b.cx; });
    for (auto& c : kept) {
        c.cx /= scale; c.cy /= scale;
        c.a  /= scale; c.b /= scale; c.r /= scale;
        c.side = (c.cx < bgr_in.cols / 2.0) ? 0 : 1;
    }

    res.circles = kept;
    res.method_used = !mask_res.empty() ? "mask" : (!hough_res.empty() ? "hough" : "none");

    // ---- 判定这次 Hough 有没有"挣到自己的开销" ----------------------------
    // 判据：合并去重后，最终数量是否比 mask 单独给出的更多。
    //   补上了 -> 有用：清零无用计数与冷却，下一帧继续允许兜底
    //   没补上 -> 无用：累计；达到上限才开始冷却（隔 hough_cooldown_frames 帧再试）
    if (last_used_hough_) {
        const int mask_kept = (int)std::min<size_t>(mask_res.size(),
                                                    (size_t)(limit > 0 ? limit : mask_res.size()));
        last_hough_helpful_ = (int)kept.size() > mask_kept;
        if (last_hough_helpful_) {
            hough_useless_streak_ = 0;
            hough_cooldown_left_  = 0;
        } else if (++hough_useless_streak_ >= p_.hough_useless_limit) {
            hough_cooldown_left_ = std::max(0, p_.hough_cooldown_frames);
            if (p_.trace) {
                char b[160];
                std::snprintf(b, sizeof(b),
                              "hough: 连续 %d 次未补上盘 -> 进入冷却 %d 帧",
                              hough_useless_streak_, hough_cooldown_left_);
                res.trace.push_back(b);
            }
        }
    } else if (!hough_res.empty()) {
        last_hough_helpful_ = true;
    }

    res.latency_ms = nowMs() - t0;
    if (p_.keep_debug) res.mask = mask_img;
    return res;
}

// ---------------------------------------------------------------------------
// 预览 / 调试图
// ---------------------------------------------------------------------------
namespace {

void text(const cv::Mat& img, const std::string& s, cv::Point org, double scale,
          cv::Scalar color, int thick = 2) {
    cv::putText(img, s, org, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), thick + 3);
    cv::putText(img, s, org, cv::FONT_HERSHEY_SIMPLEX, scale, color, thick);
}

}  // namespace

void drawOverlay(const cv::Mat& bgr, const DetectResult& res, cv::Mat& out,
                 const OverlayOptions& opt) {
    out = bgr.clone();
    if (out.empty()) return;

    const int W = out.cols, H = out.rows;
    const double k = std::max(0.6, W / 1280.0);

    for (size_t i = 0; i < res.circles.size(); ++i) {
        const auto& c = res.circles[i];
        const cv::Point ctr(cvRound(c.cx), cvRound(c.cy));
        const cv::Scalar col = (i == 0) ? cv::Scalar(80, 255, 80)      // 绿
                                        : cv::Scalar(60, 200, 255);    // 橙黄
        const int thick = std::max(2, cvRound(3 * k));

        if (c.is_ellipse) {
            cv::ellipse(out, ctr, cv::Size(cvRound(c.a), cvRound(c.b)), c.angle,
                        0, 360, col, thick, cv::LINE_AA);
            if (opt.axes) {   // 斜视时把长短轴画出来，一眼看出倾斜方向和离心率
                const double ra = c.angle * CV_PI / 180.0;
                const cv::Point2d u(std::cos(ra), std::sin(ra));
                cv::line(out, cv::Point(ctr.x - u.x * c.a, ctr.y - u.y * c.a),
                              cv::Point(ctr.x + u.x * c.a, ctr.y + u.y * c.a),
                         cv::Scalar(255, 140, 60), 1, cv::LINE_AA);
                cv::line(out, cv::Point(ctr.x + u.y * c.b, ctr.y - u.x * c.b),
                              cv::Point(ctr.x - u.y * c.b, ctr.y + u.x * c.b),
                         cv::Scalar(255, 140, 60), 1, cv::LINE_AA);
            }
        } else {
            cv::circle(out, ctr, cvRound(c.r), col, thick, cv::LINE_AA);
        }

        if (opt.crosshair) {
            const int a = std::max(8, cvRound(14 * k));
            cv::line(out, {ctr.x - a, ctr.y}, {ctr.x + a, ctr.y}, cv::Scalar(60, 60, 255), 2, cv::LINE_AA);
            cv::line(out, {ctr.x, ctr.y - a}, {ctr.x, ctr.y + a}, cv::Scalar(60, 60, 255), 2, cv::LINE_AA);
            cv::circle(out, ctr, std::max(2, cvRound(3 * k)), cv::Scalar(60, 60, 255), -1, cv::LINE_AA);
        }

        char txt[224];
        if (c.is_ellipse)
            std::snprintf(txt, sizeof(txt), "#%zu %s  a=%.0f b=%.0f (%.2f)  %.0fdeg  s=%.2f",
                          i, c.side == 0 ? "LEFT" : (c.side == 1 ? "RIGHT" : "?"),
                          c.a, c.b, c.axis_ratio, c.angle, c.score);
        else
            std::snprintf(txt, sizeof(txt), "#%zu %s  r=%.0f  score=%.2f  %s", i,
                          c.side == 0 ? "LEFT" : (c.side == 1 ? "RIGHT" : "?"),
                          c.r, c.score, c.method.c_str());

        const double ts_ = 0.66 * k;
        int bl = 0;
        const cv::Size ext = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, ts_, 2, &bl);
        cv::Point org(ctr.x - cvRound(c.a), std::max(cvRound(24 * k), ctr.y - cvRound(c.b) - cvRound(10 * k)));
        org.x = std::min(org.x, W - ext.width - cvRound(8 * k));
        org.x = std::max(org.x, cvRound(4 * k));
        text(out, txt, org, ts_, col, std::max(1, cvRound(2 * k)));
    }

    if (opt.mask_inset && !res.mask.empty()) {
        const int mw = std::max(80, std::min(W / 6, res.mask.cols));
        const int mh = std::max(1, cvRound(mw * (double)res.mask.rows / res.mask.cols));
        const int pad = cvRound(12 * k);
        const cv::Rect box(std::max(0, W - mw - pad), std::max(0, H - mh - pad), mw, mh);

        cv::Mat small, bgr_small;
        cv::resize(res.mask, small, cv::Size(mw, mh), 0, 0, cv::INTER_AREA);
        cv::cvtColor(small, bgr_small, cv::COLOR_GRAY2BGR);
        cv::Mat roi = out(box);
        cv::addWeighted(roi, 0.30, bgr_small, 0.70, 0, roi);
        cv::rectangle(out, box, cv::Scalar(150, 150, 150), 1);
        text(out, "mask", {box.x + 5, box.y + cvRound(16 * k)}, 0.45 * k,
             cv::Scalar(230, 230, 230), 1);
    }

    if (opt.hud) {
        char head[320];
        int n = std::snprintf(head, sizeof(head), "circles=%zu  method=%s  %.1f ms",
                              res.circles.size(), res.method_used.c_str(), res.latency_ms);
        if (opt.fps > 0)
            n += std::snprintf(head + n, sizeof(head) - n, "  %.1f fps", opt.fps);
        if (opt.frame_index >= 0)
            n += std::snprintf(head + n, sizeof(head) - n, "  #%ld", opt.frame_index);

        const int bar_h = cvRound(38 * k);
        cv::Mat bar = out(cv::Rect(0, 0, W, bar_h));
        cv::addWeighted(bar, 0.35, cv::Mat(bar.size(), bar.type(), cv::Scalar(0, 0, 0)), 0.65, 0, bar);

        const cv::Scalar hud_col = res.circles.empty() ? cv::Scalar(80, 80, 255)
                                                       : cv::Scalar(120, 255, 120);
        text(out, head, {cvRound(14 * k), cvRound(26 * k)}, 0.75 * k, hud_col,
             std::max(1, cvRound(2 * k)));
        if (!opt.source.empty())
            text(out, opt.source, {cvRound(14 * k), bar_h + cvRound(20 * k)}, 0.5 * k,
                 cv::Scalar(210, 210, 210), 1);
    }

    if (res.circles.empty()) {
        const std::string msg = "NO CIRCLE DETECTED";
        int base = 0;
        const cv::Size ts = cv::getTextSize(msg, cv::FONT_HERSHEY_SIMPLEX, 0.9 * k, 2, &base);
        text(out, msg, {(W - ts.width) / 2, H / 2}, 0.9 * k, cv::Scalar(80, 80, 255), 2);
    }
}

void renderOverlay(OverlayJob& job) {
    if (job.frame.empty()) return;
    cv::Mat out;
    drawOverlay(job.frame, job.det, out, job.opt);
    job.frame = std::move(out);   // 绘制结果接管，省一次拷贝
}

// ---------------------------------------------------------------------------
// 合成测试场景。axis_ratio > 1 时画成椭圆，用来模拟斜视
// ---------------------------------------------------------------------------
cv::Mat makeSyntheticScene(int width, int height, double left_cx, double right_cx,
                           double cy, double r_frac, double axis_ratio, double angle_deg) {
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(200, 200, 200));
    const int split = width / 2;
    img(cv::Rect(0, 0, split, height)).setTo(cv::Scalar(40, 40, 200));             // 左红 (BGR)
    img(cv::Rect(split, 0, width - split, height)).setTo(cv::Scalar(200, 90, 30)); // 右蓝
    cv::rectangle(img, cv::Rect(split - width / 64, 0, width / 32, height),
                  cv::Scalar(60, 60, 60), -1);                                     // 深色分隔带

    const double r = r_frac * std::min(width, height);
    const double a = r * std::sqrt(axis_ratio);     // 保持面积不变
    const double b = r / std::sqrt(axis_ratio);
    const cv::Point centers[2] = {{cvRound(left_cx * width), cvRound(cy * height)},
                                  {cvRound(right_cx * width), cvRound(cy * height)}};
    for (const auto& c : centers) {
        cv::ellipse(img, c, cv::Size(cvRound(a * 1.22), cvRound(b * 1.22)), angle_deg,
                    0, 360, cv::Scalar(55, 55, 55), -1);                        // 深色外环
        cv::ellipse(img, c, cv::Size(cvRound(a), cvRound(b)), angle_deg,
                    0, 360, cv::Scalar(240, 240, 240), -1);                     // 白垫
    }
    return img;
}

}  // namespace yuan
