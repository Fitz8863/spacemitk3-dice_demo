# SpaceMIT K3 YOLOv10 手势检测

本工程在 SpaceMIT K3 板端实现 YOLOv10 手势模型的实时摄像头推理：摄像头读取（MJPEG 硬解码）、OpenCL GPU 前处理、SpaceMIT EP 推理、检测框绘制，以及 RTSP 推流到 MediaMTX。管线骨架与 `yolov8_objdetect` 一致；区别只在检测器的输出解析（YOLOv10 end-to-end）以及删除了黑线分界检测与 LLM 复核。

## 数据流

```text
摄像头线程 -> 采集队列 -> OpenCL 前处理线程 -> 推理队列 -> YOLOv10 推理线程
                                                    \-> 主线程显示/绘制
                                                    \-> RTSP 最新帧队列 -> GStreamer spacemith264enc(VPU) -> MediaMTX
```

```text
USB 摄像头 V4L2 MJPEG 1280x720@25
  -> GStreamer v4l2src
  -> 优先 spacemitdec code-type=9（K3 VPU 硬件解码）
     无可用 V4L2 M2M 解码节点时自动回退 jpegdec + videoconvert
  -> appsink NV12（只保留最新帧）
  -> OpenCL GPU：Y/UV 上传 + NV12->RGB + resize + letterbox + CHW + FP32/255
  -> SpaceMIT ONNX Runtime EP（可 --no-ep 回退 CPU）
  -> YOLOv10 output0 [1, 300, 6]
  -> conf 过滤（+ no_gesture 过滤）+ RPS 折叠 + letterbox 反算，免 NMS
  -> OpenCV 显示 / RTSP 推流
```

## 模型

`model/yolov10n_gestures.q.onnx`：Ultralytics YOLOv10n 基座（hagrid_v2_gestures 训练、PPQ INT8 量化、opset 24、`end2end=True`）。

- 输入：`images [1,3,640,640]` float32，`/255` 归一化（OpenCL 前处理已按此实现）。
- 输出：`output0 [1,300,6]` float32，每行 `(x1, y1, x2, y2, conf, class_id)`：
  - 坐标是 640×640 letterbox 域的像素值（非 cx,cy,w,h，未乘 stride 之前已解码）；
  - `conf` 是图内已 sigmoid 的最大类别分数，范围 [0,1]；
  - 行已按 conf 降序排列（TopK=300），**不需要再做 NMS**。
- 34 类 HaGRID 手势，类别 id 与标签见 `config.json` 的 `classes`（与模型 metadata `names` 一致）。

### 与 YOLOv8 输出的差异

YOLOv8 导出是 `[1, 4+nc, 8400]`（cx,cy,w,h + 每类分数），需要外部解码 + NMS；YOLOv10 end-to-end 导出是 `[1,300,6]` 已解码的 xyxy + conf + class_id，后处理只剩阈值过滤和 letterbox 反算。

## 编译（必须在 K3 板端）

```bash
cd ~/projects/dice-game/yolov10_objdetect
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

板端如果 OpenCV 装在非标准路径，按 `yolov8_objdetect` 同样加 `-DOpenCV_DIR=/opt/opencv-spacemit/lib/cmake/opencv4`。

## 运行

程序默认从当前工作目录的 `config.json` 读取初始化参数：

```bash
cd ~/projects/dice-game/yolov10_objdetect
./build/yolov10_camera
```

自测（不占用摄像头，验证 OpenCL 前处理 + 模型加载 + 一次推理）：

```bash
./build/yolov10_camera --self-test --no-display --no-rtsp
```

真机短跑（60 帧后自动退出）：

```bash
./build/yolov10_camera --no-display --max-frames 60 --no-rtsp
```

RTSP 推流（MediaMTX 已在板上运行）：

```bash
ffprobe -rtsp_transport tcp rtsp://127.0.0.1:8554/rps/det
ffplay -rtsp_transport tcp rtsp://<K3板端IP>:8554/rps/det
```

`rtspclientsink` 与 `spacemith264enc` 插件现在都在系统 GStreamer 插件目录（`/usr/lib/riscv64-linux-gnu/gstreamer-1.0/`，2026-09-20 实测），**无需再设置 `GST_PLUGIN_PATH` / `LD_LIBRARY_PATH`**，直接 `./build/yolov10_camera` 即可。

## config.json 参数

| 参数 | 作用 |
| --- | --- |
| `model` | ONNX 模型路径；相对路径以程序启动目录为基准。 |
| `classes` | 类别名数组，下标即模型输出 `class_id`，用于画面标签和 `filter_no_gesture` 定位；须与模型内嵌 `names` 顺序一致。缺省时使用内置的 34 类列表。 |
| `filter_no_gesture` | `true`（默认）时丢弃 `no_gesture` 类的检测框（HaGRID 的兜底类，画出来全是噪声）。`--show-no-gesture` 临时关闭。 |
| `rps_mode` | `true`（默认）启用石头剪刀布折叠：把 34 类手势映射到 Rock/Paper/Scissors，映射外类别丢弃；`false` 或 `--no-rps` 保留原始 34 类标签。 |
| `rps_map` | RPS 折射表：游戏标签 → 源手势类名数组（见下节）。写错（未知类名/空标签/源类重复映射）启动即报错。 |
| `camera` / `device` | 摄像头设备路径或编号；`device` 非空时优先。 |
| `width` / `height` / `fps` | 摄像头请求规格；25fps 请求失败会自动回退到设备可协商帧率。 |
| `intra_threads` / `ep_affinity` | SpaceMIT EP 线程数与绑核（`14;15`），数量必须一致。 |
| `conf` | 置信度阈值（模型输出已是概率域，直接比较）。 |
| `focus` / `zoom` | 手动对焦/变焦；`-1` 表示不修改。 |
| `display_enabled` | 是否开 HighGUI 窗口；关闭后仍可 RTSP 推流。 |
| `yolov10_enabled` | `false` 只采集显示画面，跳过 OpenCL/推理。 |
| `max_frames` | 最多处理帧数，`0` 持续运行。 |
| `dump_input` | 首帧 640×640 FP32 CHW 输入保存路径；空为不保存。 |
| `self_test` | 启动后执行一次 OpenCL + 推理自测。 |
| `rtsp.enabled/host/port/path` | RTSP 推流开关与目标 MediaMTX；默认 `rtsp://127.0.0.1:8554/rps/det`。 |

命令行参数在 JSON 加载后覆盖同名配置（`--model/--conf/--classes/--no-ep/--no-display/--max-frames/--rtsp-path` 等，`--help` 查看全部）。

## 石头剪刀布模式（rps_mode）

模型本身是 34 类 HaGRID 手势；RPS 模式（默认开启）把它折叠成三个游戏类别，映射表在 `config.json` 的 `rps_map`（游戏标签 → 源手势类名数组）：

```json
"rps_map": {
  "Rock":     ["fist"],
  "Paper":    ["palm", "stop", "stop_inverted"],
  "Scissors": ["peace", "peace_inverted", "three2"]
}
```

- 多个源手势可共享同一游戏标签（张开手掌/停下都算布）；启动日志会打印解析结果 `RPS mode: ...`。
- 不在映射表里的类别（grabbing、like、ok……）直接丢弃，画面只出现 Rock/Paper/Scissors。
- 画框颜色按**游戏标签**分配（三个标签各一色，不随源类变），标签文本也是游戏标签。
- 映射写错（未知类名、空标签、一个源类映射到两个标签）在启动时报错退出，不会静默用错映射。
- `rps_mode: false` 或 `--no-rps` 恢复原始 34 类行为（画面显示源类名、按类配色）。

## 命令行速查

```text
--config PATH      JSON 配置文件，默认当前目录 config.json
--model PATH       ONNX 模型
--camera VALUE     摄像头编号或设备路径，例如 /dev/video1
--device PATH      显式 V4L2 节点，覆盖 --camera
--width/--height/--fps
--conf FLOAT       置信度阈值
--classes LIST     逗号分隔类别名（空项报错）
--show-no-gesture  不过滤 no_gesture 类
--queue-depth N    每级队列深度
--focus/--zoom     对焦/变焦；-1 不改动
--intra-threads N  SpaceMIT EP 线程数
--ep-affinity LIST EP 线程绑核，数量必须匹配线程数
--no-ep            不注册 SpaceMIT EP，纯 CPU 推理（排查 EP 精度/兼容性时对照）
--no-display       不创建窗口
--max-frames N     处理 N 帧后退出
--dump-input PATH  保存首帧预处理张量
--no-yolov10       只显示摄像头画面
--self-test        OpenCL + 模型一次推理自测
--rtsp/--no-rtsp   开关 RTSP 推流
--rtsp-host/--rtsp-port/--rtsp-path
```

## 实现要点与边界

- 采集/前处理/RTSP 三个模块与 `yolov8_objdetect` 完全同源（零拷贝 NV12、spacemitdec 硬解优先 + 软解回退、latest-only 队列、退出时先 join 采集线程再销毁 GStreamer pipeline）。
- 本模型不含黑线分界检测、分侧计数、LLM 复核——本工程只做「识别目标 + 推流」。
- **ORT 图优化级别固定从 `ORT_ENABLE_BASIC` 起步**：该 PPQ 量化图混用 INT8/UINT8 zero point，`ORT_ENABLE_EXTENDED/ALL` 会在图优化阶段报 `QuantizeLinear ... output_dtype INT8 does not match y_zero_point type UINT8`（开发机 ORT 1.30 实测）；BASIC 失败时自动降级 `ORT_DISABLE_ALL` 重试一次。
- `--no-ep` 走纯 CPU：SpaceMIT EP 对该图的精度/兼容性如有异常（参照 yolov8_posehand 的 EP cls 分支 bug 教训），用它做对照。
- TCM 冲突排查与 `yolov8_objdetect` 相同：`spacemit-tcm-smi -i` 查看、无进程时 `-c` 清理。
- 类别过滤：`filter_no_gesture=true` 按 `classes` 里 `no_gesture` 的下标（模型 id 33）整帧过滤，统计行的 `detections` 已不含该类。

## 板端验证记录（2026-09-20）

- 编译：`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/usr/lib/riscv64-linux-gnu/cmake/opencv4 && cmake --build build -j4` 通过，产物 `build/yolov10_camera`。
- `--self-test --no-display --no-rtsp` 通过：OpenCL PowerVR 前处理 21ms，SpaceMIT EP（BASIC 优化级、affinity 14;15）加载 `[1,3,640,640] → [1,300,6]` 正常。
- 真机短跑（C920 `/dev/video1`，720p MJPEG 硬解，90/120 帧）：`fps_infer` 约 21~23，`infer_ms` 约 35（EP 2 线程），空场景 0 误检。
- `--no-ep` CPU 对照：单次推理约 2.6s（纯 CPU 慢属预期），输出与 EP 一致 → EP 对该 PPQ 量化图行为正常（对照了真实照片的 top5 候选与 conf 序列）。
- RTSP 推流 `rtsp://127.0.0.1:8554/rps/det`：MediaMTX 收流正常，`ffprobe` 得 h264 Main 1280x720，抓帧核对检测框/标签/HUD 完整；SIGTERM 优雅退出、摄像头释放。
- ★ 推理报 `tcm buffer acquire failed for core id N` 时：先 `spacemit-tcm-smi -i` 看占用块，若 PID 已死（`ps -p <PID>` 查无此进程）就是僵尸 TCM，`spacemit-tcm-smi -c` 清理后重启即可（2026-09-20 实测：hand_track 异常退出留下 2 个僵尸块导致本工程起不来，清理后恢复 24fps/35ms）。
- ★ 板子整体当前被钉在 0-7 核（PID1 起就是，cmdline 无 isolcpus，systemd 无 CPUAffinity/AllowedCPUs 配置——来源待查，2026-09-20 记录）。config 里的 `ep_affinity: "14;15"` 在此状态下实际落在受限核上执行；EP 仍能跑满 24fps/35ms，但若想真正用 A100 核（8-15）需先解除系统级限制。
- 结论：检测+推流链路全部验证通过；等有真实手势画面后可进一步调 `conf` 阈值。

