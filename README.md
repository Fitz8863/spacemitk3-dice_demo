# yuan —— K3 摄像头"圆圈"识别（传统图像算法，不走模型）

在 SpacemiT K3 板子上，用**纯传统 CV**（OpenCV 阈值 + 形态学 + 轮廓几何 + RANSAC
圆拟合）定位俯拍摄像头画面里的两个白色圆垫（骰子盘），**不加载任何 ONNX 模型、
不碰 NPU/EP**。

```
┌─────────────────────────────────────────────────────────────┐
│  1280x720 俯拍：左红垫 + 右蓝垫 + 中间深色分隔带              │
│                                                             │
│   ╭───────╮                                  ╭───────╮      │
│   │ ○白垫 │  ← 深色圆环                        │ ○白垫 │      │
│   ╰───────╯                                  ╰───────╯      │
└─────────────────────────────────────────────────────────────┘
        ↓  circle_detect
   LEFT  cx=247.1 cy=452.1 r=118.0   score=0.82
   RIGHT cx=1057.8 cy=435.7 r=119.6  score=0.83
```

## 1. 结论（先说能不能做）

**能做，而且做得很稳。** 在真实采集帧上的实测结果：

| 指标 | 结果 |
|---|---|
| 命中率 | **26/26 帧**，每帧都恰好检出 2 个圆 |
| 半径稳定性 | 左盘 `r=117.97 ± 0.47 px`、右盘 `r=119.62 ± 0.61 px` |
| 圆心稳定性 | `cx` 标准差 1.5~1.7 px（`cy` 受骰子搭边影响，标准差 ~4 px） |
| 单帧耗时 | **36 ms**（work-width 640）/ **26 ms**（work-width 320） |
| 斜视视角 | **椭圆拟合**（白化 + 圆 RANSAC），实测轴比 1.59 对真值 1.60、倾角 25.0° 对真值 25° |
| 盘上放东西 | **flood-fill 填内部空洞**，盘心压住 45% 仍准确（圆心误差 1.0 px） |
| 推流 | **RTSP / H.264 硬编**（`spacemith264enc` → MediaMTX），另有 MJPEG 预览做调试 |
| 配置 | 参数全部在 `config.json`，**直接 `./build/circle_detect` 就行** |
| 依赖 | 只依赖系统 OpenCV 4.10 + GStreamer/V4L2，**零模型文件** |

**真机摄像头也已实测通过**（2026-09-15，罗技 C920 / `/dev/video1`，1280x720@30fps）：

| 取流路径 | 结果 |
|---|---|
| OpenCV V4L2 后端 | 30/30 帧取流成功，42~44 ms/帧 |
| OpenCV GStreamer 后端 | 15/15 帧取流成功，43.9 ms/帧 |
| auto（先 V4L2 后 GStreamer） | 15/15 帧取流成功 |

真机那一轮画面里只摆了一个圆垫（另一个出画了），检测器**如实报 1 个**而不是硬凑
2 个 —— 这一点比命中率更说明问题：`--expected 2` 只限制上限，不会为了凑数造圆。

对照：现有 `yolov8_objdetect` 的 YOLOv8 模型 11 MB + SpaceMIT EP 推理。
这里的方案是 0 模型、0 NPU 占用，CPU 单帧 30 ms。

## 2. 算法原理

两个物理先验构成全部判据，没有任何学习成分：

1. **白垫是画面里唯一"低饱和 + 高亮度"的大块区域**
   —— 红/蓝地垫饱和度高（S≈200），深色圆环亮度低（V≈55），全都落在阈值外。
2. **白垫一定被一圈"非白"像素包住**
   —— 要么是那圈深色圆环本身，要么是环外的彩色地垫。

流程：

```
BGR 帧
 ├─ resize 到 work-width（默认 640，全流程在此分辨率上算）
 ├─ BGR → HSV
 │
 ├─【路径 1 · mask】──────────────────────────────────────
 │   inRange(H∈[0,179], S≤sat_max, V≥val_min)   → 白垫二值图
 │   morphologyEx CLOSE(9, 矩形核)               → 填掉骰点小空洞
 │   morphologyEx OPEN(5, 矩形核)                → 去细碎噪点
 │   fillInteriorHoles()  ★ 泛洪填盘内大空洞      → 盘上放东西也不破
 │   findContours(RETR_EXTERNAL)
 │   逐轮廓：
 │     ├ 面积 → 等效半径 r=sqrt(A/π) 预筛（对离心率不变）
 │     ├ fitShapeRobust()：★ 白化 + 圆 RANSAC = 椭圆 RANSAC
 │     │   ├ 用填充区域二阶矩求白化矩阵 W（椭圆 → 圆）
 │     │   ├ 归一化空间里跑三点定圆 RANSAC
 │     │   ├ 用内点凸包矩重估 W，迭代至收敛（最多 3 轮）
 │     │   └ 映射回原图 -> 椭圆中心 + 长短轴 + 倾角
 │     ├ 归一化空间圆度 4πA/P² ≥ 0.55   （★ 与离心率无关）
 │     ├ 归一化空间填充率 ≥ 0.75
 │     └ validateShape()：盘面够白 + 环带角度覆盖 + 环带对比度
 │
 ├─【路径 2 · hough】只有路径 1 数量不足时才跑 ────────────
 │   灰度 → medianBlur(5) → HoughCircles(HOUGH_GRADIENT)
 │   候选同样过 validateShape()  ← 不依赖地垫颜色，换场景可兜底
 │
 ├─ 两条路径合并 → 打分 → 按圆心去重 → 取分数最高的 N 个
 └─ 按 x 排序 → 标 LEFT/RIGHT → 坐标映射回原图
```

### 三个关键设计（都是踩坑踩出来的）

**① 填充率必须用拟合半径算，不能用 minEnclosingCircle 半径。**
`minEnclosingCircle` 会被任何凸起撑大。实测 `img_20260915_095109`：一颗骰子搭在
左盘边缘，MEC 半径被撑到 71.6（真实 59.6），填充率算出来只有 0.685 < 0.75 阈值
→ 好圆被误杀。改用 RANSAC 拟合半径后填充率 0.99，该帧恢复检出。
这类误杀很隐蔽：**26 帧里只有 1 帧触发**，不看 trace 根本找不到。

**② RANSAC 定圆是抗"骰子搭边"的核心。**
骰子搭在盘外时轮廓会长出一个凸包，最小二乘（Kasa）会被拉偏。RANSAC 用随机三点
定圆 + 数内点，凸起部分的点天然变成外点被丢掉。`--self-test` 第 4 项专门测这个：
盘边搭一颗骰子，圆心误差只有 3.62 px（容差 14.9 px）。

**③ 形态学用矩形核而不是椭圆核。**
OpenCV 对矩形核走可分离的行/列两趟实现，比椭圆核快 3~4 倍。同一个 26 帧数据集：
椭圆核 58.8 ms/帧 → 矩形核 30.5 ms/帧，**结果完全一致**。

## 2.1 盘上放着东西怎么办（骰子 / 骰盅 / 手）

形态学闭运算只能补"骰点"那么小的洞；盘心压一个骰盅，掩码就被啃掉一大块，
圆度、填充率、内点比例全线崩。

**解法：`fillInteriorHoles()` —— 泛洪填内部空洞。**

```
1. 掩码上下左右各补 1 像素黑边
2. 从 (0,0) 泛洪背景（只填与画面边界连通的背景）
3. bitwise_not → 剩下的就是"被白垫整圈包住、泛洪够不到"的区域
4. OR 回掩码
```

无论洞多大、什么颜色，只要被盘面整圈包住就一律补上，掩码恒等于**整个盘子的轮廓**。
代价只有一次 O(N) 泛洪，实测 ~1 ms。`--self-test` 第 6 项：盘心压一个占盘面 45%
的深色圆盖，圆心误差 **1.01 px**。

配套还改了盘面判据：原来要求"盘心采样 ≥60% 是白的"，改成看**近边缘带**
（0.62 / 0.78 / 0.92 半径）—— 盘心被压住时靠边那一圈仍然露着白；而"整个盘都是黑的"
依然会被拒（`core_white < 0.20` 兜底）。

> **边界**：只有**整圈被包住**的洞会被填。物体如果**搭出盘边**（比如一颗骰子一半悬空），
> 掩码会长出凸起 —— 这一条交给 RANSAC 处理（关键设计②）。

## 2.2 斜视视角怎么办（圆在画面里变成椭圆）

俯视变斜视后，圆盘在画面里是**椭圆**。继续用圆去拟合，长短轴各错一半：实测当前
实拍帧（轴比 1.11）圆拟合给 r=114，而真实是长轴 123 / 短轴 111；斜视再大一点
（1.6:1）会直接把候选判成"太扁"整个丢掉 —— 实测就是**两个盘全丢，只剩 Hough
兜底给出两个 r=33 的错误小圆**。

**解法：白化（whitening）+ 圆 RANSAC = 椭圆 RANSAC。**

```
p' = W·(p - mu),   W = diag(1/√l1, 1/√l2)·Vᵀ
```

`l1,l2` 是整体轮廓二阶矩的特征值、`V` 是特征向量。白化把椭圆"搓"成圆
（半长轴都压到 √2 附近），于是：

- **椭圆拟合复用圆 RANSAC 的代码**，不用另写五点定二次曲线那套；
- **圆度 / 填充率 / 内点比例全部在归一化空间判定**，天然与离心率无关 ——
  不然一个 2:1 椭圆的圆度只有 0.84，跟方块（0.785）几乎分不开；
- 映射回原图 `M = r·W⁻¹`，对 `M` 做 SVD 就得到长短轴和倾角，绘制直接 `cv::ellipse`。

**★ 必须记住的坑：二阶矩要用"填充区域"的，不能用"轮廓点"的。**

实心椭圆 `E[x²]=a²/4、E[y²]=b²/4` → `√(l1/l2)` **恰好**等于 a/b，白化一次到位。
而轮廓点（弧长均匀）的比值只有 a/b 的约 0.89 倍（a/b=1.6 → 1.428，数值积分验证过），
白化不足会让归一化后的形状仍是个椭圆，RANSAC 只圈得住中间一条带；**用这条带再估
白化会越估越扁** —— 实测迭代三轮轴比从 1.6 一路跑到 **3.50**，然后被判"太扁"丢弃。

所以用 `cv::moments(contour)`（算的是多边形**填充区域**的矩），并且：
- 第 0 轮用整条轮廓的矩，之后用**内点凸包**的矩（凸包填平缺口，遮挡不带偏轴向）；
- 加收敛早停（轴比变化 <0.5% 就停），避免白迭代。

另一个隐蔽的坑：**`cv::eigen` 的特征值顺序在文档里没写死**，顺序反了会把长短轴
调换、结果完全错。代码里按大小自行排序，不依赖实现细节。

实测（`--self-test` 第 5 项，合成 1.6:1、倾斜 25° 的盘）：

| | 检出 | 真值 |
|---|---|---|
| 轴比 a/b | **1.59** | 1.60 |
| 倾角 | **25.0°** | 25° |
| 等效半径 | 93.1 | 93.6 |
| 圆心误差 | < 0.12 r | — |

真机斜视帧上同样生效：`a=122.8 b=111.0 angle=1.7°`、`a=119.4 b=113.3 angle=10.6°`，
JSON 里带 `"ellipse": true`，预览画面上会同时画出长短轴（橙色细线）。

> 透视投影下圆的像严格来说是**圆锥曲线**，但只要圆没被地平线切开它就是椭圆，
> 所以椭圆拟合对透视也是对的，**不需要相机标定**。环带验证的采样按"相对形状边界的
> 缩放比"走，斜视下环带变宽变窄也能跟上。
>
> 如果相机位姿固定、还需要**真正的俯视正视图**（比如做米制测量、或后续按像素
> 统计骰子点数），那要一次性单应标定 + `warpPerspective` —— 这版没做，可以后面
> 加个 `--rectify` 选项。

## 3. 编译与运行

板子上已装好系统 OpenCV 4.10 和 GStreamer（含 `rtspclientsink` / `spacemith264enc`），直接编：

```bash
cd ~/projects/dice-game/yuan
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

### 配置：config.json（推荐用法）

**所有参数都写在当前目录的 `config.json` 里，直接运行即可，不需要带任何命令行参数：**

```bash
./build/circle_detect
```

优先级：**命令行 > config.json > 代码内默认值**。
日常改配置就编辑 `config.json`，临时试一下才用命令行覆盖。

```bash
./build/circle_detect --dump-config > config.json   # 导出一份默认配置
./build/circle_detect --config /path/to/other.json  # 用别的配置文件
./build/circle_detect --device /dev/video2 --no-rtsp # 临时覆盖并关掉推流
```

配置读取刻意做得**严格**：键的类型写错、枚举值写错、数值越界都会明确报错并指出是哪一项，
不会静默回退到默认值（那种 bug 最难查）：

```
$ ./build/circle_detect --config /tmp/bad.json
[err] 读取配置失败: 配置值非法: method 只能是 auto / mask / hough
$ printf '{"work_width": "abc"}' > b.json && ./build/circle_detect --config b.json
[err] 读取配置失败: config work_width must be a number
```

主要配置项（完整键见 `config.json`，说明见 `--help`）：

| 段 | 键 | 说明 |
|---|---|---|
| 输入 | `camera` / `device` | 摄像头设备路径或数字索引（`device` 优先） |
| | `width` / `height` / `fps` / `backend` | 采集参数；`backend` = auto/v4l2/gst |
| | `focus` / `zoom` | v4l2 控制，**负数 = 不设置** |
| | `image` / `video` | 改用图片目录 / 视频文件（留空则用摄像头） |
| | `max_frames` / `loop` | 跑多少帧后停；图片/视频循环 |
| 算法 | `work_width` | 工作分辨率宽度（默认 640，**别用 480**） |
| | `threads` | 识别线程数，**实测 1 最省 CPU 且帧率不变** |
| | `method` / `expected` / `max_circles` | 检测路径 / 期望个数（触发 Hough 兜底）/ 上限 |
| | `sat_max` / `val_min` | 白垫 HSV 阈值 |
| | `fill_holes` | 泛洪填盘内空洞（抗遮挡） |
| | `ellipse_fit` / `max_axis_ratio` | 椭圆拟合（抗斜视）/ 轴比上限 |
| | `require_ring` / `ring_*` | 环带验证 |
| 推流 | `rtsp.enabled/host/port/path` | **RTSP 推流（生产路径）** |
| 预览 | `preview.enabled/port/bind/width/jpeg_quality` | MJPEG 预览（调试路径） |
| 叠加 | `overlay.hud/mask_inset/crosshair/axes` | 画面上画什么 |
| 输出 | `debug_dir` / `save_video` / `out_json` / `show` / `summary` / `quiet` / `verbose` | |

> `--no-rtsp` / `--no-preview` / `--no-hud` / `--no-ellipse` / `--no-fill-holes`
> 这几个是**纯命令行开关**，专门用来临时关掉配置里打开的东西（`tools/eval.sh`
> 就靠 `--no-rtsp --no-preview` 保证回归测试不会顺手拉起推流）。

### RTSP 推流（生产路径）

走 H.264 硬编 + MediaMTX，和 dice-game 的 `yolov8_segdetect` 是同一条路子：

```
appsrc → queue(leaky) → videoconvert → NV12
       → spacemith264enc(VPU 硬编) → h264parse → rtspclientsink → MediaMTX
```

`config.json` 里：

```json
"rtsp": { "enabled": true, "host": "127.0.0.1", "port": 8554, "path": "/dice/circles" }
```

起来之后有两条路可看：

| 方式 | 地址 |
|---|---|
| RTSP | `rtsp://10.0.90.160:8554/dice/circles` |
| WebRTC | `http://10.0.90.160:8889/dice/circles/` |

```bash
ffprobe -v error -rtsp_transport tcp -show_entries stream=codec_name,width,height \
        -of default=nw=1 rtsp://127.0.0.1:8554/dice/circles
# codec_name=h264   width=1280   height=720
```

**不需要改 `mediamtx.yml`** —— 它里面已经有 `"~^dice/": source: publisher`
这条正则，任何 `dice/*` 子路径都是发布者自建的（和现有 `dice/det`、`dice/seg` 同一机制）。

四个实现要点（照抄参考工程的做法，逐条都验证过）：

1. **编码线程异步 + latest-only**：网络或编码变慢时只丢旧帧，绝不阻塞采集/识别主循环。
2. **零拷贝包装帧**：`gst_buffer_new_wrapped_full` + `shared_ptr` 持有者，不 clone
   （BGR 720p 一帧 2.7 MB）。
3. **`rtspclientsink` 自己建 RTP payloader**，喂它 `rtph264pay` 的输出反而链接失败。
4. **`queue leaky=downstream`**，队列满丢旧帧。

> **坑：`spacemith264enc` 会把 `[MPP-DEBUG]` 日志直接写到 stdout**
> （实测一次编码器初始化 34 行），足以把 JSON Lines 冲烂。库是闭源的、也没找到
> 日志级别开关，所以在 fd 层面隔离：启动时 `dup(STDOUT_FILENO)` 留一份给 JSON，
> 再把 fd 1 指向 stderr。这样第三方库的噪音全落到 stderr，
> `circle_detect --quiet > out.jsonl` 照常干净（实测 112 行输出、0 行非 JSON）。

> **流的帧率 = 识别帧率**（约 11–18 fps），不是配置里的 `fps`（那是采集帧率）。
> 每帧带真实 PTS，播放器按时间戳走，看起来正常。

### MJPEG 预览（调试路径，画面上直接画出圆）

RTSP 是给生产/观看用的；调算法时 MJPEG 更方便（浏览器直接开、能抓单帧）。
两者可以**同时开**（`preview.enabled` 和 `rtsp.enabled` 都设 true）。

板子走 SSH 时没有 `DISPLAY`，`cv::imshow` 开不了窗口，所以预览走浏览器：

```bash
# config.json 里把 preview.enabled 设成 true，或命令行临时开：
./build/circle_detect --preview
```

启动时会打印可访问的网址（自动列出所有网卡地址）：

```
[preview] MJPEG 预览已启动，浏览器打开：
          http://127.0.0.1:8099/
          http://10.0.90.160:8099/        <- 有线网段
          http://100.118.229.28:8099/     <- tailnet
          纯流地址（VLC/ffplay）：http://<ip>:8099/stream.mjpg
```

用笔记本打开 `http://100.118.229.28:8099/` 就能看到**带圆圈标注的实时画面**：

- 检测到的圆用**绿/橙圆环**画出来，圆心打红十字
- 每个圆左上角标 `#序号 左右 半径 置信度 方法`
- 顶部状态条：`circles=2 method=mask 30.1 ms 24.3 fps #114`
- 右下角半透明小窗是**算法眼里的白垫掩码**（白=判为白垫），
  一眼就能看出算法是不是被光照/反光骗了
- 一个都没检出时画面正中会打红字 `NO CIRCLE DETECTED`

四个 HTTP 路由：

| 路由 | 用途 |
|---|---|
| `/` | 预览页（实时统计 + 流） |
| `/stream.mjpg` | 纯 MJPEG 流，可直接喂 VLC / ffplay / `<img>` |
| `/snapshot.jpg` | 抓当前**带标注**的一帧（脚本里很好用） |
| `/raw.jpg` | 抓当前**未标注**的原始帧 —— 调算法/对比时必须用这个，别用带 overlay 的 |
| `/status` | 最近一帧的 JSON（fps / 耗时 / 圆坐标 / 长短轴），给上层程序消费 |

```bash
# 命令行抓一张带标注的图（不需要图形界面）
curl -s http://127.0.0.1:8099/snapshot.jpg -o annotated.jpg
# 实时看检测结果
watch -n0.5 'curl -s http://127.0.0.1:8099/status | python3 -m json.tool | head -20'
```

其他可视化出口（可任意组合）：

```bash
--show                     # 板子本地有图形会话时用 OpenCV 窗口
--debug-dir /tmp/dbg       # 落盘 overlay_XXXX.jpg + mask_XXXX.png
--save-video out.avi       # 录制带标注的视频（Ctrl-C 会优雅收尾，索引写完整）
--preview-width 640        # 预览推流降到 640 宽
--no-hud --no-mask-inset   # 关掉状态条 / 掩码缩略图
--loop                     # 图片/视频循环播放（配合 --preview 做常驻演示）
```

### 不开摄像头先自检

```bash
./build/circle_detect --self-test
```

6 项合成图测试，共 15 条断言（含"白色方块不能误判成圆"、"盘边搭骰子"、
"斜视 1.6:1 椭圆"、"盘心压 45% 深色圆盖"四个难例），无需摄像头。
失败时会自动打印候选诊断。
### 跑真实帧

```bash
# 单张图 + 存调试图
./build/circle_detect --image samples/sample_clean.jpg --debug-dir /tmp/dbg

# 整个目录批量 + 汇总
./build/circle_detect --image samples --expected 2 --summary

# 一键回归（自检 + 批量 + 分辨率耗时表）
./tools/eval.sh
./tools/eval.sh ~/projects/dice-game/yolov8_objdetect/data
```

### 开摄像头

```bash
# 默认 /dev/video1、1280x720，跑 30 帧后退出
./build/circle_detect --device /dev/video1 --frames 30 --expected 2 --summary

# 对齐生产视角（zoom 160 = 现有 dice-game 的 config.json 取值）+ 实时预览
./build/circle_detect --device /dev/video1 --zoom 160 --show

# 原图精度（慢，105 ms/帧）；或更快（18 ms/帧）
./build/circle_detect --device /dev/video1 --work-width 1280 --frames 10
./build/circle_detect --device /dev/video1 --work-width 320  --frames 30
```

> `--zoom` / `--focus` 走 `v4l2-ctl`，语义与现有 C++ 生产链路一致：
> **负数 = 跳过不设置**（默认 -1）。不传就不动摄像头控制。

> **`--expected N` 会触发 Hough 兜底路径**：auto 模式下如果 mask 路径检出的数量
> 已经 ≥ N，就完全跳过 Hough（省 CPU）；小于 N 才会跑 Hough 试图补足。
> 所以 `--expected 2` 时真机耗时约 43 ms，不传时只需 22~27 ms。
> 已知场景恒为 2 个圆垫的话，传 `--expected 2` 更稳（多一层兜底）。

### 输出格式

stdout 是 JSON Lines，一帧一行，方便被上层服务消费：

```json
{"type":"frame","index":0,"name":"sample_clean.jpg","width":1280,"height":720,
 "method":"mask","latency_ms":30.4,"count":2,
 "circles":[
   {"id":0,"cx":247.0,"cy":449.9,"r":118.1,"side":"LEFT",
    "score":0.761,"circularity":0.790,"fill":0.875,"inlier":0.864,
    "ring":1.000,"contrast":41.4,"method":"mask"},
   {"id":1,"cx":1062.2,"cy":433.6,"r":119.2,"side":"RIGHT","score":0.794,...}]}
{"type":"summary","frames":26,"frames_with_circles":26,"total_circles":52,
 "avg_circles":2.000,"avg_latency_ms":30.47,"detect_hit_rate":1.000,"wall_ms":1208.3}
```

字段含义：
- `a` / `b` / `angle` / `axis_ratio` / `ellipse` —— 椭圆长短轴、倾角、轴比；
  正圆时 `a≈b`、`r≈a`、`ellipse: false`（`r` 始终是等效半径 `√(a·b)`，旧字段兼容）
- `circularity` 4πA/P²、`fill` 填充率 —— **都是在白化后的归一化空间算的，与离心率无关**
- `inlier` RANSAC 内点比例
- `ring` 环带角度覆盖率（1.0 = 整圈都被非白像素包住）
- `contrast` 内盘亮度 − 环带亮度（0~255）
- `score` 上述几项的加权综合，0~1

## 4. 性能

单帧检测耗时（X100 @2.0GHz，真实数据平均）：

| work-width | 耗时 | 相当于 | 每帧检出数 |
|---|---|---|---|
| 320 | 20.2 ms | 50 fps | 1.67（3 帧里有 1 帧只检出 1 个，偏小会丢精度） |
| 480 | 31.3 ms | 32 fps | 2.00 |
| **640（默认）** | **36.1 ms** | **28 fps** | **2.00** |
| 960 | 60.1 ms | 17 fps | 2.00 |
| 1280 | ~106 ms | 9 fps | 2.00 |

**默认 640 起才稳定检出 2 个**，320 虽然快但会掉一个盘 —— 别为了帧率往下调。
现有 dice-game 推理预算约 41 ms/帧（24fps），640 够用；只在"稳定帧"上做一次
高精度测量可以临时用 1280。

> 加了 flood-fill + 椭圆拟合后，640 从 33.5 ms 涨到 36.4 ms（+3 ms），
> 换来的是"盘上压东西"和"斜视"两个能力。中途一度涨到 **64 ms**，
> 原因是拟合循环对每个候选跑了 3 遍 RANSAC（轮廓有 ~600 个点）。
> 两个优化压回来：**轮廓抽稀到 ~220 点**（拟合不需要几百个点）+ **收敛早停**
> （轴比不变就不再迭代），另外把 `minEnclosingCircle` 预筛换成面积等效半径预筛
> （O(n) 且对离心率不变）。RANSAC 迭代数也从 600 降到 300
> （白化后内点比例 >0.7，三次全中概率 ~0.34，300 次足够）。

耗时构成（work-width 640）：形态学 ~0 ms（换矩形核后）、resize+色彩空间转换+
阈值+轮廓+泛洪填洞 ~26 ms、椭圆拟合+RANSAC ~10 ms。

## 4.1 CPU 占用（纯识别，不含编码/推流）

用 bash `time` 取 user+sys CPU 再除以帧数（不是看 `top` 的瞬时百分比）。
真机摄像头路径、work-width 640：

| | CPU/帧 | 墙钟/帧 | fps |
|---|---|---|---|
| 默认（OpenCV 多核） | 61.6 ms | 54.4 ms | 18.4 |
| **`--threads 1`** | **47.3 ms** | 54.3 ms | **18.4** |
| `--threads 2` | 49.5 ms | 54.1 ms | 18.5 |
| `taskset -c 0`（参照） | 47.8 ms | 54.5 ms | 18.4 |

**★ 多核是负优化：帧率一模一样，却多烧 14 ms/帧的 CPU。**
拆开看 user/sys 就很清楚 —— user 时间几乎不变（7.85s vs 7.03s / 150 帧），
差的全是 **sys 时间：9.3 ms/帧 → 0.5 ms/帧**，也就是 OpenCV `parallel_for`
跨核唤醒/等待的开销。识别本身在 640×360 上只有 23 万像素，粒度太小，
并行划不来。**建议加 `--threads 1`。**

成本分解（都是 `--threads 1`）：

| 部分 | CPU/帧 | 说明 |
|---|---|---|
| 纯算法（图片输入） | 39.8 ms | 阈值+形态学+轮廓+泛洪+椭圆拟合 |
| V4L2 取流 + MJPEG 解码 | 7.5 ms | 相机送的是 MJPEG，解码算在取流里 |
| **合计** | **47.3 ms** | |

**换算成占用率**（`--threads 1`，真机 18.4 fps）：

- 每帧 47.3 ms → 47.3 × 18.4 ≈ 870 ms CPU/秒 = **0.87 个核**
- 相对本会话可见的 8 个 X100 核（2.0 GHz）→ **约 10.9%**
- 若按生产 24 fps 满跑 → 1.14 个核 → 约 14%
- 相对整块 K3 的 16 个核 → 约 5.4%

### 算法分支的 CPU 占比（图片路径，`--threads 1`）

| 配置 | CPU/帧 | 差值 |
|---|---|---|
| 基线（椭圆拟合 + 填洞） | 39.8 ms | — |
| 关掉椭圆拟合 | ~31 ms | 椭圆拟合 ≈ **9 ms** |
| 关掉填内部空洞 | ~37 ms | flood-fill ≈ **1 ms** |

也就是说**斜视支持（椭圆拟合）花掉约 9 ms/帧，遮挡支持（填洞）只花 1 ms** ——
如果你确定永远用俯视视角，`--no-ellipse` 能省掉这 9 ms。

### 分辨率与 CPU

| work-width | CPU/帧 | fps（真机） |
|---|---|---|
| 320 | 33.9 ms | 25.7 |
| 640（默认） | 47.3 ms | 18.4 |

> ⚠️ **别用 480**：实测 CPU/帧反而比 640 还高（59 ms vs 55 ms，图片路径），
> 是个反常的分辨率坑，怀疑撞上了某条非优化的内部路径。

> 顺带修掉一个隐性开销：以前只要开了 `--preview`，**每帧都会画叠加图**
> （clone 720p + 画圈写字 ≈ +10 ms/帧），哪怕根本没人在看。
> 现在叠加图按需绘制（`MjpegServer::wantsFrame()`），实测
> "开预览但无客户端" 与 "不开预览" 都是 47 ms/帧 —— **零开销才真正成立**。

**可视化开销**（真机 60 帧实测，`wall_ms/60` 折算）：

| 配置 | 有效帧率 | 相对基线 |
|---|---|---|
| 只检测（基线） | 17.3 fps | — |
| `--preview`（720p，**无客户端**） | 17.4 fps | **0 开销** |
| `--preview`（720p，浏览器在看） | 16.7 fps | ≈ −0.7 fps |
| `--preview-width 640`（浏览器在看） | 17.1 fps | ≈ −0.2 fps |
| `--save-video out.avi`（720p MJPG） | 10.0 fps | **−7 fps** |

关键设计：**没有客户端连接时完全不编码 JPEG**，所以常驻挂着预览服务对检测
几乎零成本；真正贵的是 `--save-video`（MJPG 编码 + 落盘，约 +43 ms/帧）。
基线本身也不是检测瓶颈 —— 22.8 ms 检测 + 33 ms 相机取帧 ≈ 57 ms，**是采集
节拍在限制帧率**。

## 5. 调参

大部分场景不用动。真出问题时按这个顺序查：

```bash
# 1. 看每个候选为什么被拒（最有用的一个开关）
./build/circle_detect --image f.jpg --verbose

# 2. 画面偏暗导致白垫没进掩码 → 调低亮度下限
--val-min 90

# 3. 地垫反光/偏色混进掩码 → 调低饱和度上限，或加大开运算
--sat-max 55 --open-ksize 7

# 4. 圆盘尺寸不在默认区间（默认 0.05~0.35 × min(w,h)）
--min-radius-frac 0.08 --max-radius-frac 0.25

# 5. 完全换场景（不是红蓝地垫）→ 走几何路径
--method hough

# 6. 关掉环带验证看原始候选（调试用，会引入误检）
--no-ring-check

# 7. 对比实验：关掉椭圆拟合（斜视下就会掉检，用来确认椭圆这条路有没有在起作用）
--no-ellipse

# 8. 对比实验：关掉填内部空洞（盘上放东西就会掉检）
--no-fill-holes

# 9. 场景里出现特别扁的东西被误判成盘 → 收紧轴比上限
--max-axis-ratio 1.6
```

完整参数表：`./build/circle_detect --help`

**排查斜视/遮挡问题的固定套路**：先用 `/raw.jpg` 抓一张干净原图，
再对这张图跑 `--verbose` 看每个候选的 `rho`（点集轴比估计）、`a/b`（拟合轴比）、
`wr`（归一化空间半径）—— `wr` 应该稳定在 ~2，如果跑到 4 以上说明白化没收敛。
`circularity` / `fill` 现在都是归一化空间的量，斜视下也应该维持在 0.7 / 0.95 以上。

## 6. 已知限制

- **依赖"白垫 vs 彩色地垫"的对比。** 如果换成白色桌面 + 白色圆盘，mask 路径会失效
  （`--method hough` 仍可兜底，但它只找圆、不找椭圆，斜视下精度会掉）。
- **flood-fill 只填"整圈被包住"的洞**。物体搭出盘边时掩码会长凸起，靠 RANSAC 扛；
  如果物体大到把盘面遮掉一半以上，内点比例会跌破阈值而丢检。
- **Hough 兜底路径不会拟合椭圆**（`cv::HoughCircles` 只能找圆），所以斜视 + mask 路径
  失败时会退化成"用圆近似椭圆"，精度下降。这是刻意的取舍：椭圆 Hough 太贵。
- **中线附近的圆**：左右划分目前按图像中线 `cx < W/2`。生产 dice-game 用的是
  `vision.divider.position`（实测真实分界在 0.505），要接入时应改成传分界线位置。
- **强反光/全黑画面**下会退化成 0 检出（这是预期行为，不是崩溃），
  JSON 里 `count: 0` + `method: "none"` 可直接判定。
- **只做了单帧检测**，时序平滑要显式开 `--smooth 0.4`（默认关）。
- **RTSP 流是"识别帧率"而不是采集帧率**（约 11–18 fps）：推的是带标注的识别结果帧，
  所以被识别速度限制。要更高帧率就得降 `work_width`。
- **`[MPP-DEBUG]` 噪音只能丢到 stderr**，屏蔽不掉（闭源 VPU 库、无日志开关）。
- **未做**：单应标定 + 俯视矫正（`--rectify`）、骰子点数识别、圆内骰子分割、
  与 server.py 的 HTTP 对接。

## 7. 文件

```
yuan/
├── config.json                  ★ 全部参数（直接跑就用它）
├── CMakeLists.txt
├── src/
│   ├── config.h/.cpp            config.json 读取 + 校验（cv::FileStorage，无第三方依赖）
│   ├── circle_detector.h/.cpp   核心算法
│   │      掩码 + 泛洪填洞 + 白化椭圆 RANSAC + 环带验证 + Hough 兜底 + 叠加图绘制
│   ├── frame_source.h/.cpp      取帧：图片 / 视频 / V4L2 / GStreamer
│   ├── rtsp_streamer.h/.cpp     RTSP 推流（appsrc → VPU H.264 → rtspclientsink）
│   ├── mjpeg_server.h/.cpp      MJPEG over HTTP 预览（POSIX socket，零外部依赖）
│   └── main.cpp                 CLI/配置合并、JSON 输出、6 项自检、推流与预览接线
├── tools/eval.sh                一键回归（自检 + 批量 + 耗时表）
└── samples/                     3 张真实采集帧（含"骰子搭边"这种难例）
```

`--self-test` 的 6 项（共 15 条断言，不需要摄像头）：

| # | 场景 | 验证什么 |
|---|---|---|
| 1 | 合成红蓝地垫 + 两个白圆盘 | 基本检出 / 圆心 / 半径 / 左右标注 |
| 2 | 纯地垫，没有盘 | 不误检 |
| 3 | 地垫上放一个白色方块 | 几何判据能拒掉非圆（填充率 0.64 < 0.75） |
| 4 | 白盘边缘搭一颗骰子 | RANSAC 抗凸起 |
| 5 | 白盘画成 1.6:1、倾斜 25° 的椭圆 | **斜视：轴比 / 倾角 / 等效半径 / 椭圆标记** |
| 6 | 盘心压一个占盘面 45% 的深色圆盖 | **遮挡：填内部空洞后仍准确** |

任一项失败时会自动打印该帧的候选诊断（哪个轮廓、area、轴比、被哪条判据拒的）。

## 8. 和现有 dice-game 的关系

本目录是**独立验证工程**，没有改动 `yolov8_objdetect` / `main/` 下任何文件。

如果要接进现有链路，最省事的用法是把两个圆心 + 半径当作**骰子区域锚点**：
现有 `vision.divider.position` 只给了一条分界线，而圆垫给出了完整的"骰子应该在哪"
的二值区域，可以直接把 YOLO 的检测框按"是否落在某个圆内"做归属过滤
—— 这正是档案里记的那个遗留隐患（"门控只校验每侧恰好 N 个框，无法判断框是否
真的是骰子"）可以顺手补上的地方。
